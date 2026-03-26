import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, DSSP, Selection
from Bio.Data.PDBData import protein_letters_3to1 as three_to_one
import torch
from torch_geometric.data import Data
import esm
import hashlib
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset


VDW_CUTOFF = 8.0
HBOND_ENERGY_THRESHOLD = -0.5
PEPTIDE_BOND_CUTOFF = 1.5  

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}

# Edge type → integer label (for concatenation later)
EDGE_TYPE_MAP = {"peptide": 0, "vdw": 1, "hbond": 2}


class ProteinGraphBuilder:
    def __init__(self, pdb_path: str):
        self.pdb_path = pdb_path

        # --- Parse structure ---
        parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure("protein", pdb_path)
        self.model = next(self.structure.get_models())

        self._raw_residues = [
            r for r in Selection.unfold_entities(self.model, "R")
            if "CA" in r
        ]

        # --- Sequence extraction + index mapping ---
        self.sequence, self.valid_indices, self.skipped = self._extract_sequence()

        # Final residue list — only standard AA residues, aligned to sequence
        self.residues = [self._raw_residues[i] for i in self.valid_indices]
        self.res_to_idx = {r: i for i, r in enumerate(self.residues)}

        # Precompute once — used by vdw_edges
        self._cb_coords = self._get_cb_coords()

        # Build DSSP → node index map once — used by hbond_edges
        self._dssp_to_node_idx = self._build_dssp_map()

        print(f"[ProteinGraphBuilder] {len(self.residues)} residues | "
              f"sequence length: {len(self.sequence)}")
        if self.skipped:
            print(f"  Skipped: {[(i, r.get_resname(), reason) for i, r, reason in self.skipped]}")

    # ------------------------------------------------------------------
    # Sequence extraction
    # ------------------------------------------------------------------

    def _extract_sequence(self) -> tuple[str, list[int], list]:
        sequence, valid_indices, skipped = [], [], []

        for i, res in enumerate(self._raw_residues):
            resname = res.get_resname().strip()
            het, resseq, icode = res.get_id()

            if het.strip():
                skipped.append((i, res, "HETATM"))
                continue
            if resname not in STANDARD_AA:
                skipped.append((i, res, f"non-standard: {resname}"))
                continue
            try:
                one_letter = three_to_one[resname]
            except KeyError:
                skipped.append((i, res, "no one-letter code"))
                continue
            if icode.strip():
                print(f"  [insertion] resseq={resseq} icode={icode} ({resname}) at index {i}")

            sequence.append(one_letter)
            valid_indices.append(i)

        return "".join(sequence), valid_indices, skipped

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _get_cb_coords(self) -> np.ndarray:
        """
        Cβ for all standard residues, Cα fallback for Glycine.
        Precomputed once at init — do not call this repeatedly.
        """
        coords = []
        for res in self.residues:
            atom = res["CB"] if "CB" in res else res["CA"]
            coords.append(atom.get_vector().get_array())
        return np.array(coords)  # (N, 3)

    # ------------------------------------------------------------------
    # DSSP → node index mapping
    # ------------------------------------------------------------------

    def _build_dssp_map(self) -> dict:
        """
        Maps each DSSP key (chain_id, res_id) → node index in self.residues.
        DSSP may drop terminal residues or residues it can't assign, so this
        mapping will be partial — missing entries are silently skipped in hbond_edges.
        """
        dssp_map = {}
        for res in self.residues:
            chain_id = res.get_parent().id
            res_id = res.get_id()          # (het, resseq, icode)
            dssp_key = (chain_id, res_id)
            node_idx = self.res_to_idx[res]
            dssp_map[dssp_key] = node_idx
        return dssp_map

    # ------------------------------------------------------------------
    # Edge builders
    # ------------------------------------------------------------------

    def peptide_edges(self) -> list[tuple]:
        """
        Consecutive residues in the same chain connected by a real C→N bond.
        Checks atomic distance to exclude chain breaks (missing residues in PDB).
        Weight is fixed at 1.0 — covalent bonds are binary.
        """
        edges = []
        for i in range(len(self.residues) - 1):
            r1, r2 = self.residues[i], self.residues[i + 1]

            # Must be same chain
            if r1.get_parent().id != r2.get_parent().id:
                continue

            # Check actual C(i) → N(i+1) bond distance to detect chain breaks
            if "C" not in r1 or "N" not in r2:
                continue
            c_coord = r1["C"].get_vector().get_array()
            n_coord = r2["N"].get_vector().get_array()
            bond_dist = float(np.linalg.norm(c_coord - n_coord))

            if bond_dist <= PEPTIDE_BOND_CUTOFF:
                edges.append((i, i + 1, 1.0, "peptide"))

        return edges

    def vdw_edges(self) -> list[tuple]:
        """
        Residue pairs within VDW_CUTOFF Å (Cβ–Cβ distance).
        Weight is a Gaussian decay — closer residues get higher weight.
        query_pairs() already excludes self-loops and duplicate pairs.
        """
        tree = cKDTree(self._cb_coords)
        edges = []
        for i, j in tree.query_pairs(r=VDW_CUTOFF):
            dist = float(np.linalg.norm(self._cb_coords[i] - self._cb_coords[j]))
            weight = float(np.exp(-dist / VDW_CUTOFF))
            edges.append((i, j, weight, "vdw"))
        return edges

    def hbond_edges(self) -> list[tuple]:
        """
        Hydrogen bonds from DSSP (Kabsch-Sander energy < HBOND_ENERGY_THRESHOLD).
        DSSP keys are mapped back to node indices via _dssp_to_node_idx to avoid
        index misalignment between DSSP's internal ordering and self.residues.
        Weight is the absolute Kabsch-Sander energy — stronger H-bonds get higher weight.
        """
        dssp = DSSP(self.model, self.pdb_path, dssp="mkdssp", file_type="PDB")
        edges = []

        for key in dssp.keys():
            data = dssp[key]

            # Donor to acceptor and its reverse pair
            for offset_field, energy_field in [(6, 7), (8, 9)]:
                offset = data[offset_field]
                energy = data[energy_field]

                if energy >= HBOND_ENERGY_THRESHOLD or offset == 0:
                    continue

                # Resolve partner DSSP key
                dssp_keys_list = list(dssp.keys())
                src_dssp_idx = dssp_keys_list.index(key)
                partner_dssp_idx = src_dssp_idx + offset

                if not (0 <= partner_dssp_idx < len(dssp_keys_list)):
                    continue

                partner_key = dssp_keys_list[partner_dssp_idx]

                # Map both keys to node indices — skip if either is absent
                src_node = self._dssp_to_node_idx.get(key)
                dst_node = self._dssp_to_node_idx.get(partner_key)

                if src_node is None or dst_node is None:
                    continue

                edges.append((src_node, dst_node, float(abs(energy)), "hbond"))

        return edges

    # ------------------------------------------------------------------
    # Graph builder
    # ------------------------------------------------------------------

    def build(self, node_features: torch.Tensor | None = None, 
              contacts: torch.Tensor | None = None) -> Data:
        """
        Assembles the PyG Data object.

        Args:
            node_features: (N, F) tensor of precomputed node features.
                           If None, uses zeros as placeholder.

        Returns:
            Data with:
                x          — node features (N, F)
                edge_index — (2, E) undirected
                edge_attr  — (E, 2) → [weight, edge_type_int] // Implementing the contact map as an additional edge type is possible, but it requires careful handling

        """
        print(contacts)
        all_edges = self.peptide_edges() + self.vdw_edges() + self.hbond_edges()

        src = torch.tensor([e[0] for e in all_edges], dtype=torch.long)
        dst = torch.tensor([e[1] for e in all_edges], dtype=torch.long)

        # Undirected: duplicate both directions
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src])
        ])

        weights = torch.tensor([e[2] for e in all_edges], dtype=torch.float)
        types = torch.tensor(
            [EDGE_TYPE_MAP[e[3]] for e in all_edges], dtype=torch.float
        )

        # Stack weight + type — both directions get same attributes
        edge_attr = torch.stack([weights, types], dim=1)           # (E, 2)
        edge_attr = edge_attr.repeat(2, 1)                         # (2E, 2)

        if node_features is None:
            print("[ProteinGraphBuilder] No node features provided — using zeros.")
            node_features = torch.zeros((len(self.residues), 1), dtype=torch.float)

        assert node_features.shape[0] == len(self.residues), (
            f"node_features rows {node_features.shape[0]} != "
            f"residues {len(self.residues)}"
        )

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(self.residues)
        )
    

@dataclass
class ESMOutput:
    embeddings: torch.Tensor   # (L, 1280) — per-residue, last layer
    contacts:   torch.Tensor   # (L, L)    — predicted from attention heads
    sequence:   str


class ESMProcessor:
    def __init__(
        self,
        cache_dir: str = ".esm_cache",
        device: str = "cuda",
        repr_layer: int = 33,          # last layer of esm2_t33_650M_UR50D
        batch_token_limit: int = 1024,
    ):
        self.device = device
        self.repr_layer = repr_layer
        self.batch_token_limit = batch_token_limit
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._model = None
        self._alphabet = None
        self._batch_converter = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            print("[ESMProcessor] Loading ESM2...")
            self._model, self._alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self._batch_converter = self._alphabet.get_batch_converter()
            self._model = self._model.eval().to(self.device)
            print("[ESMProcessor] Ready.")

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_path(self, sequence: str) -> Path:
        key = hashlib.md5(sequence.encode()).hexdigest()
        return self.cache_dir / f"{key}.pt"

    def _load_cache(self, sequence: str) -> ESMOutput | None:
        path = self._cache_path(sequence)
        if path.exists():
            data = torch.load(path, map_location="cpu")
            return ESMOutput(**data)
        return None

    def _save_cache(self, sequence: str, output: ESMOutput):
        torch.save({
            "embeddings": output.embeddings,
            "contacts":   output.contacts,
            "sequence":   output.sequence,
        }, self._cache_path(sequence))

    # ------------------------------------------------------------------
    # Core inference — takes (label, seq) tuples, returns raw results
    # ------------------------------------------------------------------

    def _run_inference(self, labeled_seqs: list[tuple[str, str]]) -> dict:
        """
        labeled_seqs: list of ("label", "SEQUENCE") tuples
        Returns raw ESM2 results dict.
        """
        _, _, batch_tokens = self._batch_converter(labeled_seqs)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self._model(
                batch_tokens,
                repr_layers=[self.repr_layer],
                return_contacts=True,
            )
        return results, batch_tokens

    # ------------------------------------------------------------------
    # Single sequence
    # ------------------------------------------------------------------

    def process(self, sequence: str) -> ESMOutput:
        cached = self._load_cache(sequence)
        if cached is not None:
            return cached

        self._load_model()

        results, _ = self._run_inference([("protein", sequence)])

        L = len(sequence)
        output = ESMOutput(
            # Slice [1 : L+1] to remove BOS token (index 0) and EOS token (index L+1)
            embeddings=results["representations"][self.repr_layer][0, 1:L+1].cpu(),  # (L, 1280)
            contacts=results["contacts"][0, :L, :L].cpu(),                           # (L, L)
            sequence=sequence,
        )
        self._save_cache(sequence, output)
        return output

    # ------------------------------------------------------------------
    # Batch — length sorted to minimize padding waste
    # ------------------------------------------------------------------

    def process_batch(self, sequences: list[str]) -> list[ESMOutput]:
        results = {}

        # Cache hits
        uncached = []
        for i, seq in enumerate(sequences):
            cached = self._load_cache(seq)
            if cached is not None:
                results[i] = cached
            else:
                uncached.append((i, seq))

        if not uncached:
            return [results[i] for i in range(len(sequences))]

        self._load_model()

        # Sort by length — minimizes padding waste per bucket
        uncached.sort(key=lambda x: len(x[1]))

        # Build length-aware buckets within token limit
        batches = []
        current_batch = []
        current_max_len = 0

        for i, seq in uncached:
            new_max = max(current_max_len, len(seq))
            projected_tokens = new_max * (len(current_batch) + 1)

            if current_batch and projected_tokens > self.batch_token_limit:
                batches.append(current_batch)
                current_batch = [(i, seq)]
                current_max_len = len(seq)
            else:
                current_batch.append((i, seq))
                current_max_len = new_max

        if current_batch:
            batches.append(current_batch)

        # Run inference per bucket
        for batch in batches:
            indices, seqs = zip(*batch)
            labeled = [(str(idx), seq) for idx, seq in zip(indices, seqs)]

            raw, _ = self._run_inference(labeled)

            for k, (orig_idx, seq) in enumerate(zip(indices, seqs)):
                L = len(seq)
                output = ESMOutput(
                    embeddings=raw["representations"][self.repr_layer][k, 1:L+1].cpu(),
                    contacts=raw["contacts"][k, :L, :L].cpu(),
                    sequence=seq,
                )
                self._save_cache(seq, output)
                results[orig_idx] = output

        return [results[i] for i in range(len(sequences))]
    


class ProteinDataset(Dataset):
    def __init__(self, pdb_paths: list[str], esm_processor: ESMProcessor):
        self.pdb_paths = pdb_paths
        self.processor = esm_processor

        # Build all graph builders first (fast, CPU only)
        print("Parsing PDB files...")
        self.builders = [ProteinGraphBuilder(p) for p in pdb_paths]

        # Batch all sequences through ESMFold in one pass
        print("Running ESMFold...")
        sequences = [b.sequence for b in self.builders]
        self.esm_outputs = self.processor.process_batch(sequences)

    def __len__(self):
        return len(self.builders)

    def __getitem__(self, idx) -> Data:
        builder = self.builders[idx]
        esm_out = self.esm_outputs[idx]

        # ESMFold also gives you a contact map — you can add it as edge weights
        # or use it to add/filter edges here before building
        node_features = esm_out.embeddings   # (L, 1024) — projection happens in GNN
        edge_attention = esm_out.contacts
        return builder.build(node_features=node_features, contacts = edge_attention)



