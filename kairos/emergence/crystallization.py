"""
Pattern Crystallization
=======================
Converts emergent oscillatory patterns into actionable knowledge structures.

Takes the raw output of the Oscillatory Synthesis Core — trajectories,
attractor information, and emergent patterns — and distils them into
discrete, interpretable knowledge representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from kairos.core.state import Trajectory


@dataclass
class KnowledgeStructure:
    """
    A crystallized knowledge structure extracted from emergent dynamics.

    Attributes
    ----------
    id : str
        Unique identifier.
    label : str
        Human-readable label.
    embedding : np.ndarray
        Vector representation in cognitive space.
    confidence : float
        Crystallization confidence (0–1).
    coherence : float
        Internal coherence of the pattern (0–1).
    components : list[np.ndarray]
        Constituent pattern vectors.
    relations : dict
        Relationships to other knowledge structures.
    metadata : dict
        Arbitrary metadata.
    """

    id: str = ""
    label: str = ""
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence: float = 0.0
    coherence: float = 0.0
    components: List[np.ndarray] = field(default_factory=list)
    relations: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class PatternCrystallizer:
    """
    Engine for extracting discrete knowledge structures from continuous
    dynamical patterns.

    Pipeline:
        1. Segment trajectory into quasi-stable epochs
        2. Extract principal modes from each epoch (PCA / NMF)
        3. Cluster similar modes across epochs
        4. Assign confidence & coherence scores
        5. Build relational graph between structures
    """

    min_epoch_length: int = 100
    variance_threshold: float = 0.05
    cluster_eps: float = 1.0
    min_cluster_size: int = 2
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)

    # ── main crystallization pipeline ─────────────────────────────
    def crystallize(
        self,
        trajectory: Trajectory,
        emergent_patterns: List[np.ndarray] | None = None,
    ) -> List[KnowledgeStructure]:
        """
        Full crystallization pipeline.

        Parameters
        ----------
        trajectory : Trajectory
        emergent_patterns : optional pre-extracted patterns from ROS

        Returns
        -------
        List of KnowledgeStructure objects.
        """
        # Step 1: Segment into epochs
        epochs = self._segment_epochs(trajectory)

        # Step 2: Extract principal modes
        modes: List[np.ndarray] = []
        epoch_meta: List[dict] = []
        for start, end in epochs:
            sub = trajectory.positions_array()[start:end]
            pcs, explained = self._principal_modes(sub)
            for pc, exp in zip(pcs, explained):
                modes.append(pc)
                epoch_meta.append({"epoch_start": start, "epoch_end": end, "explained_var": exp})

        if emergent_patterns:
            for p in emergent_patterns:
                if np.linalg.norm(p) > 1e-8:
                    modes.append(p / np.linalg.norm(p))
                    epoch_meta.append({"source": "ros_emergent"})

        if not modes:
            return []

        # Step 3: Cluster modes
        clusters = self._cluster_modes(modes)

        # Step 4: Build knowledge structures
        structures: List[KnowledgeStructure] = []
        for cid, indices in clusters.items():
            cluster_modes = [modes[i] for i in indices]
            embedding = np.mean(cluster_modes, axis=0)
            coherence = self._coherence(cluster_modes)
            confidence = min(1.0, len(indices) / 5.0) * coherence

            ks = KnowledgeStructure(
                id=f"ks_{cid:04d}",
                label=f"pattern_{cid}",
                embedding=embedding,
                confidence=confidence,
                coherence=coherence,
                components=cluster_modes,
                metadata={"member_epochs": [epoch_meta[i] for i in indices]},
            )
            structures.append(ks)

        # Step 5: Build relations
        self._build_relations(structures)

        return structures

    # ── segmentation ──────────────────────────────────────────────
    def _segment_epochs(self, trajectory: Trajectory) -> List[Tuple[int, int]]:
        """
        Segment trajectory into quasi-stable epochs by detecting variance
        change points.
        """
        pos = trajectory.positions_array()
        n = len(pos)
        if n < 2 * self.min_epoch_length:
            return [(0, n)]

        # Sliding variance
        var_trace = []
        for i in range(0, n - self.min_epoch_length, self.min_epoch_length // 2):
            window = pos[i : i + self.min_epoch_length]
            var_trace.append(np.var(window))

        if not var_trace:
            return [(0, n)]

        var_trace = np.array(var_trace)
        median_var = np.median(var_trace)

        # Change points where variance deviates significantly
        change_indices = [0]
        step = self.min_epoch_length // 2
        for i, v in enumerate(var_trace):
            if abs(v - median_var) > self.variance_threshold * median_var:
                idx = i * step
                if idx - change_indices[-1] >= self.min_epoch_length:
                    change_indices.append(idx)
        change_indices.append(n)

        epochs = []
        for i in range(len(change_indices) - 1):
            start, end = change_indices[i], change_indices[i + 1]
            if end - start >= self.min_epoch_length:
                epochs.append((start, end))

        return epochs if epochs else [(0, n)]

    # ── mode extraction ───────────────────────────────────────────
    @staticmethod
    def _principal_modes(data: np.ndarray, n_modes: int = 3, min_var: float = 0.1) -> Tuple[List[np.ndarray], List[float]]:
        """Extract principal modes via SVD."""
        centered = data - data.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return [], []

        total_var = (S ** 2).sum()
        modes, explained = [], []
        for i in range(min(n_modes, len(S))):
            ev = (S[i] ** 2) / total_var if total_var > 0 else 0
            if ev >= min_var:
                modes.append(Vt[i])
                explained.append(ev)
        return modes, explained

    # ── clustering ────────────────────────────────────────────────
    def _cluster_modes(self, modes: List[np.ndarray]) -> Dict[int, List[int]]:
        """Simple agglomerative clustering of mode vectors."""
        n = len(modes)
        if n == 0:
            return {}

        # Compute distance matrix (cosine)
        vecs = np.array(modes)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normed = vecs / norms

        sim = normed @ normed.T
        dist = 1 - np.abs(sim)  # use abs because anti-parallel = same mode

        # Greedy clustering
        assigned = np.full(n, -1, dtype=int)
        cluster_id = 0
        for i in range(n):
            if assigned[i] >= 0:
                continue
            assigned[i] = cluster_id
            for j in range(i + 1, n):
                if assigned[j] < 0 and dist[i, j] < self.cluster_eps:
                    assigned[j] = cluster_id
            cluster_id += 1

        clusters: Dict[int, List[int]] = {}
        for idx, cid in enumerate(assigned):
            clusters.setdefault(cid, []).append(idx)

        return clusters

    # ── coherence / relations ─────────────────────────────────────
    @staticmethod
    def _coherence(modes: List[np.ndarray]) -> float:
        """Mean pairwise cosine similarity as coherence measure."""
        if len(modes) < 2:
            return 1.0
        vecs = np.array(modes)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normed = vecs / norms
        sim = np.abs(normed @ normed.T)
        n = len(modes)
        mask = ~np.eye(n, dtype=bool)
        return float(sim[mask].mean())

    @staticmethod
    def _build_relations(structures: List[KnowledgeStructure]) -> None:
        """Compute pairwise similarity and store in each structure's relations."""
        for i, si in enumerate(structures):
            for j, sj in enumerate(structures):
                if i == j:
                    continue
                ni = np.linalg.norm(si.embedding)
                nj = np.linalg.norm(sj.embedding)
                if ni < 1e-12 or nj < 1e-12:
                    sim = 0.0
                else:
                    sim = float(np.abs(np.dot(si.embedding, sj.embedding) / (ni * nj)))
                si.relations[sj.id] = sim
