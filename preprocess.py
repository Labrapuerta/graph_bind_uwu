"""
preprocess.py
=============
Run this ONCE locally before uploading to Lightning cloud.

What it does:
    1. Reads pdb_splits_CL.csv
    2. Downloads each full PDB → extracts the single chain → deletes full PDB
    3. Builds a ProteinDataset (parse + ESM2 embeddings + binding labels)
    4. Saves one compressed .pt file per protein into:

        preprocessed/
        ├── train/
        │   ├── batch_1/   (~1,408 files, ~112 MB)
        │   ├── batch_2/
        │   ├── batch_3/
        │   ├── batch_4/
        │   └── batch_5/
        └── test/          (~1,760 files, ~140 MB)

    5. Saves a manifest CSV so you can track what succeeded/failed

Upload `preprocessed/` to Lightning cloud — ~700 MB total.
No PDB files and no ESM2 model ever touch the cloud.

Usage:
    python preprocess.py \
        --csv       pdb_splits_CL.csv \
        --raw_dir   data/raw_pdbs \
        --out_dir   preprocessed \
        --esm_cache .esm_cache \
        --esm_chunk 25 \
        --device    cuda
"""

import argparse
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser, PDBIO, Select

# ── project imports ────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.GraphBuilder import (
    ProteinGraphBuilder,
    ESMProcessor,
    parse_binding_residues,
    get_binding_indices,
)


# ══════════════════════════════════════════════════════════════════════════════
# Chain extraction helpers
# ══════════════════════════════════════════════════════════════════════════════
"""
## Keep ligand too
class ChainSelect(Select):
    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

"""

class ChainSelect(Select):
    """Keeps only ATOM records for the requested chain — no HETATM, no ligands."""

    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        # het field is empty for standard ATOM records, non-empty for HETATM
        het, resseq, icode = residue.get_id()
        return het.strip() == ""


def download_pdb(pdb_id: str, raw_dir: Path) -> Path | None:
    """Downloads a PDB directly from RCSB. Returns saved path or None."""
    dest = raw_dir / f"{pdb_id.lower()}.pdb"
    if dest.exists():
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
    try:
        urllib.request.urlretrieve(url, dest)
        if dest.stat().st_size < 100:
            dest.unlink(missing_ok=True)
            print(f"  [download error] {pdb_id}: empty file received")
            return None
        return dest
    except Exception as e:
        print(f"  [download error] {pdb_id}: {e}")
        dest.unlink(missing_ok=True)
        return None


def download_and_extract_chain(
    pdb_id: str,
    chain: str,
    raw_dir: Path,
    out_dir: Path,
) -> Path | None:
    """
    Downloads the full PDB from RCSB, extracts ATOM-only chain records,
    saves to out_dir, then deletes the full PDB. Safe to re-run.
    """
    out_path = out_dir / f"{pdb_id}_{chain}.pdb"
    if out_path.exists():
        return out_path

    full_path = download_pdb(pdb_id, raw_dir)
    if full_path is None:
        return None

    try:
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(full_path))
        io        = PDBIO()
        io.set_structure(structure)
        io.save(str(out_path), ChainSelect(chain))
        full_path.unlink(missing_ok=True)
        return out_path
    except Exception as e:
        print(f"  [extract error] {pdb_id}_{chain}: {e}")
        full_path.unlink(missing_ok=True)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ProteinDataset — with chunked ESM2
# ══════════════════════════════════════════════════════════════════════════════

class ProteinDataset(Dataset):
    """
    Builds one torch_geometric Data graph per protein.

    esm_chunk: processes ESM2 in chunks of N sequences to avoid RAM OOM.
    Set to 25 for 14 GB RAM machines, higher for servers with more RAM.
    """

    def __init__(
        self,
        pdb_paths: list[str],
        binding_residues: list[str],
        esm_processor: ESMProcessor,
        esm_chunk: int = 25,
    ):
        self.processor  = esm_processor
        self.valid_mask = []
        self.builders   = []
        self.bind_idxs  = []

        # ── Parse all PDB structures ───────────────────────────────────────
        print("Parsing PDB structures...")
        for pdb_path, binding_str in zip(pdb_paths, binding_residues):
            try:
                builder = ProteinGraphBuilder(pdb_path)
                parsed  = parse_binding_residues(binding_str)
                idxs    = get_binding_indices(builder, parsed, validate_aa=True)
                self.builders.append(builder)
                self.bind_idxs.append(idxs)
                self.valid_mask.append(True)
            except Exception as e:
                print(f"  [parse fail] {pdb_path}: {e}")
                self.valid_mask.append(False)

        print(f"  Parsed: {sum(self.valid_mask)}  |  "
              f"Failed: {self.valid_mask.count(False)}")

        # ── Run ESM2 in chunks to avoid OOM ───────────────────────────────
        print(f"\nRunning ESM2 (chunk size={esm_chunk})...")
        sequences    = [b.sequence for b in self.builders]
        self.esm_outputs = []
        total_chunks = (len(sequences) - 1) // esm_chunk + 1 if sequences else 0

        for i in range(0, len(sequences), esm_chunk):
            chunk = sequences[i : i + esm_chunk]
            print(f"  chunk {i//esm_chunk + 1}/{total_chunks} "
                  f"({i}-{min(i+esm_chunk, len(sequences))}/{len(sequences)})...")
            self.esm_outputs.extend(esm_processor.process_batch(chunk))
            torch.cuda.empty_cache()

        print(f"  ESM2 done for {len(self.esm_outputs)} sequences\n")

    def __len__(self) -> int:
        return len(self.builders)

    def __getitem__(self, idx):
        builder  = self.builders[idx]
        esm_out  = self.esm_outputs[idx]
        bind_idx = self.bind_idxs[idx]

        y = torch.zeros(len(builder.residues), dtype=torch.float)
        if bind_idx:
            y[bind_idx] = 1.0

        return builder.build(
            node_features=esm_out.embeddings,   # (N, 1280)
            contacts=esm_out.contacts,           # (N, N)
            y=y,                                 # (N,)
        )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def resolve_output_dir(row: pd.Series, out_dir: Path) -> Path:
    """Returns the correct subfolder for a given CSV row."""
    if row["Split"] == "training":
        folder = out_dir / "train" / f"batch_{int(row['CV_Batch'])}"
    else:
        folder = out_dir / "test"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(
    csv_path:  str,
    raw_dir:   str,
    out_dir:   str,
    esm_cache: str = ".esm_cache",
    esm_chunk: int = 25,
    device:    str = "cuda",
):
    csv_path = Path(csv_path)
    raw_dir  = Path(raw_dir);  raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir  = Path(out_dir);  out_dir.mkdir(parents=True, exist_ok=True)

    chain_dir = raw_dir / "_chains"
    chain_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    print(f"  training : {(df['Split']=='training').sum()}")
    print(f"  testing  : {(df['Split']=='testing').sum()}\n")

    # ── Step 1: download + extract chains (parallel) ──────────────────────────
    print("=" * 60)
    print("Step 1 — Downloading and extracting chains")
    print("=" * 60)

    rows   = list(df.iterrows())
    paths  = {}
    n_done = 0
    total  = len(rows)

    def _download_row(args):
        idx, row = args
        path = download_and_extract_chain(
            row["PDB_ID"], row["Chains_To_Keep"], raw_dir, chain_dir
        )
        return idx, row["PDB_ID"], row["Chains_To_Keep"], path

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {pool.submit(_download_row, r): r for r in rows}
        for future in as_completed(futures):
            idx, pdb_id, chain, path = future.result()
            paths[idx] = str(path) if path is not None else None
            n_done += 1
            status = "ok" if path is not None else "fail"
            print(f"  [{status}] {pdb_id}_{chain}.pdb  ({n_done}/{total})")

    df["_pdb_path"] = df.index.map(paths)
    n_ok   = df["_pdb_path"].notna().sum()
    n_fail = df["_pdb_path"].isna().sum()
    print(f"\n  Downloaded: {n_ok}  |  Failed: {n_fail}")

    df = df[df["_pdb_path"].notna()].reset_index(drop=True)

    # ── Step 2+3: build ProteinDataset (parse PDBs + run ESM2) ───────────────
    print("\n" + "=" * 60)
    print("Step 2+3 — Building ProteinDataset (parse + ESM2)")
    print("=" * 60)

    dataset = ProteinDataset(
        pdb_paths        = df["_pdb_path"].tolist(),
        binding_residues = df["Binding_Residues"].tolist(),
        esm_processor    = ESMProcessor(cache_dir=esm_cache, device=device),
        esm_chunk        = esm_chunk,
    )

    # Align df rows to only the proteins that parsed successfully
    valid_rows = df[dataset.valid_mask].reset_index(drop=True)

    # ── Step 4: save one .pt file per protein ────────────────────────────────
    print("=" * 60)
    print("Step 4 — Saving .pt files")
    print("=" * 60)

    manifest_rows = []
    graph_fail    = []

    for dataset_idx, (_, row) in enumerate(valid_rows.iterrows()):
        pdb_id = row["PDB_ID"]
        chain  = row["Chains_To_Keep"]
        label  = f"{pdb_id}_{chain}"

        target_dir  = resolve_output_dir(row, out_dir)
        target_path = target_dir / f"{label}.pt"

        if target_path.exists():
            builder = dataset.builders[dataset_idx]
            print(f"  [skip]  {label}.pt  (already exists)")
            manifest_rows.append({
                "pdb_id":     pdb_id,
                "chain":      chain,
                "split":      row["Split"],
                "cv_batch":   row["CV_Batch"],
                "pt_path":    str(target_path),
                "n_residues": len(builder.residues),
                "status":     "ok_cached",
            })
            continue

        try:
            data    = dataset[dataset_idx]
            builder = dataset.builders[dataset_idx]

            torch.save(data, target_path)

            n_binding = int(data.y.sum())
            n_edges   = data.edge_index.shape[1]
            print(f"  [saved] {label}.pt  "
                  f"({len(builder.residues)} res, "
                  f"{n_binding} binding, "
                  f"{n_edges} edges)")

            manifest_rows.append({
                "pdb_id":     pdb_id,
                "chain":      chain,
                "split":      row["Split"],
                "cv_batch":   row["CV_Batch"],
                "pt_path":    str(target_path),
                "n_residues": len(builder.residues),
                "n_binding":  n_binding,
                "n_edges":    n_edges,
                "status":     "ok",
            })

        except Exception as e:
            print(f"  [graph fail] {label}: {e}")
            traceback.print_exc()
            graph_fail.append(label)
            manifest_rows.append({
                "pdb_id":   pdb_id,
                "chain":    chain,
                "split":    row["Split"],
                "cv_batch": row["CV_Batch"],
                "pt_path":  None,
                "status":   f"fail: {e}",
            })

    # ── Step 5: save manifest ─────────────────────────────────────────────────
    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    ok_rows = [r for r in manifest_rows if r["status"].startswith("ok")]
    print(f"  Total in CSV       : {len(df)}")
    print(f"  Download failures  : {n_fail}")
    print(f"  Parse failures     : {dataset.valid_mask.count(False)}")
    print(f"  Graph failures     : {len(graph_fail)}")
    print(f"  .pt files saved    : {len(ok_rows)}")
    print(f"  Manifest           : {manifest_path}")

    print("\nOutput folder structure:")
    for folder in sorted(out_dir.glob("**/")):
        pt_files = list(folder.glob("*.pt"))
        if pt_files:
            size_mb = sum(f.stat().st_size for f in pt_files) / 1e6
            print(f"  {folder.relative_to(out_dir)}/"
                  f"  ->  {len(pt_files)} files, {size_mb:.1f} MB")

    print(f"\nUpload `{out_dir}/` to Lightning cloud storage.")
    print("Do NOT upload raw_dir — it is intermediate only.\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess PDB structures into graph .pt files for Lightning training."
    )
    parser.add_argument("--csv",       default="pdb_splits_CL.csv")
    parser.add_argument("--raw_dir",   default="data/raw_pdbs")
    parser.add_argument("--out_dir",   default="preprocessed")
    parser.add_argument("--esm_cache", default=".esm_cache")
    parser.add_argument("--esm_chunk", type=int, default=25,
                        help="Sequences per ESM2 chunk (lower = less RAM)")
    parser.add_argument("--device",    default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    preprocess(
        csv_path  = args.csv,
        raw_dir   = args.raw_dir,
        out_dir   = args.out_dir,
        esm_cache = args.esm_cache,
        esm_chunk = args.esm_chunk,
        device    = args.device,
    )
