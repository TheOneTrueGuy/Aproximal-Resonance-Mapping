"""
Topology mapping for ARM - persistent homology analysis of activation manifolds.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import numpy.typing as npt

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False

try:
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import kneighbors_graph
    from sklearn.manifold import spectral_embedding
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..utils.config import ARMConfig


class TopologyMapper:
    """Maps topological structure of activation manifolds using persistent homology."""

    def __init__(self, config: ARMConfig):
        self.config = config

        if not RIPSER_AVAILABLE:
            raise ImportError("ripser package required for topology analysis. Install with: pip install ripser")

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for topology analysis.")

    def local_persistence(
        self,
        activation_matrix: npt.NDArray[np.float32],
        maxdim: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute persistent homology for local activation neighborhood.

        Args:
            activation_matrix: Activation points, shape (n_points, n_features)
            maxdim: Maximum homology dimension (default: config.max_homology_dim)

        Returns:
            Dictionary containing persistence diagrams and features
        """
        maxdim = maxdim or self.config.max_homology_dim

        # Compute pairwise distances
        D = pairwise_distances(activation_matrix, metric='euclidean')

        # Compute persistent homology
        result = ripser(D, distance_matrix=True, maxdim=maxdim)

        # Extract persistence diagrams
        diagrams = result['dgms']

        # Compute persistence features
        persistence_features = {}
        for dim in range(len(diagrams)):
            if dim < len(diagrams):
                diagram = diagrams[dim]
                if len(diagram) > 0:
                    # Remove infinite persistence points for finite features
                    finite_points = diagram[np.isfinite(diagram[:, 1])]
                    if len(finite_points) > 0:
                        persistences = finite_points[:, 1] - finite_points[:, 0]
                        persistence_features[f'h{dim}_features'] = {
                            'birth_death_pairs': finite_points.tolist(),
                            'persistences': persistences.tolist(),
                            'max_persistence': float(np.max(persistences)),
                            'mean_persistence': float(np.mean(persistences)),
                            'n_features': len(finite_points),
                        }

        return {
            'diagrams': [diag.tolist() for diag in diagrams],
            'persistence_features': persistence_features,
            'n_points': len(activation_matrix),
            'n_features': activation_matrix.shape[1],
        }

    def build_resonance_graph(
        self,
        resonance_signatures: List[Dict[str, Any]],
        n_neighbors: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Build k-nearest neighbor graph based on resonance signature similarity.

        Args:
            resonance_signatures: List of resonance signatures
            n_neighbors: Number of neighbors for kNN graph

        Returns:
            Dictionary containing graph structure and embeddings
        """
        n_neighbors = n_neighbors or self.config.topology_neighbors
        n_samples = len(resonance_signatures)

        # Adjust n_neighbors if we don't have enough samples
        effective_neighbors = min(n_neighbors, n_samples - 1)
        if effective_neighbors < 1:
            effective_neighbors = 1

        # Extract feature vectors from resonance signatures
        feature_vectors = []
        for sig in resonance_signatures:
            # Combine normalized singular values and entropy
            features = np.concatenate([
                sig['s_norm'],
                [sig['entropy'], sig['participation_ratio_normalized']]
            ])
            feature_vectors.append(features)

        X = np.array(feature_vectors)

        # Build kNN graph with adjusted neighbors
        W = kneighbors_graph(X, n_neighbors=effective_neighbors, mode='distance', include_self=False)
        W = W.toarray()  # Convert to dense matrix

        # Compute spectral embedding for global coordinates
        try:
            embedding = spectral_embedding(W, n_components=3, random_state=self.config.random_seed)
        except:
            # Fallback to random embedding if spectral embedding fails
            embedding = np.random.randn(len(X), 3)

        return {
            'adjacency_matrix': W,
            'feature_vectors': X,
            'spectral_embedding': embedding,
            'n_nodes': len(X),
            'effective_neighbors': effective_neighbors,
        }

    def detect_attractor_basins(
        self,
        resonance_signatures: List[Dict[str, Any]],
        graph_data: Dict[str, Any],
        n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Detect attractor basins using community detection on resonance graph.

        Args:
            resonance_signatures: List of resonance signatures
            graph_data: Graph structure from build_resonance_graph
            n_clusters: Number of clusters to find (optional)

        Returns:
            Dictionary with clustering results
        """
        try:
            from sklearn.cluster import SpectralClustering
        except ImportError:
            raise ImportError("scikit-learn required for clustering analysis.")

        X = graph_data['feature_vectors']
        n_samples = len(X)

        # Handle small datasets
        if n_samples < 2:
            return {
                'cluster_labels': [0] * n_samples,
                'n_clusters': 1,
                'centroids': [X[0].tolist()] if n_samples > 0 else [],
                'cluster_sizes': [n_samples],
                'note': 'Single sample dataset'
            }

        # Determine number of clusters
        if n_clusters is None:
            # For small datasets, limit cluster count
            max_clusters = min(5, n_samples)
            n_clusters = min(3, max_clusters)  # Start conservative

            # Try to use silhouette score for small datasets
            if n_samples >= 3:
                from sklearn.metrics import silhouette_score
                best_score = -1
                best_n_clusters = n_clusters

                for n in range(2, min(max_clusters + 1, n_samples)):
                    try:
                        clustering = SpectralClustering(
                            n_clusters=n,
                            affinity='nearest_neighbors',
                            n_neighbors=min(5, n_samples - 1),  # Limit neighbors
                            random_state=self.config.random_seed
                        )
                        labels = clustering.fit_predict(X)
                        if len(np.unique(labels)) > 1:
                            score = silhouette_score(X, labels)
                            if score > best_score:
                                best_score = score
                                best_n_clusters = n
                    except:
                        continue

                n_clusters = best_n_clusters

        # Ensure n_clusters doesn't exceed sample count
        n_clusters = min(n_clusters, n_samples)

        # Final clustering with robust parameters
        try:
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='nearest_neighbors',
                n_neighbors=min(5, n_samples - 1),
                random_state=self.config.random_seed
            )
            labels = clustering.fit_predict(X)
        except:
            # Fallback to simple clustering for very small datasets
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_seed, n_init=10)
            labels = kmeans.fit_predict(X)

        # Compute cluster centroids
        centroids = []
        for i in range(n_clusters):
            mask = labels == i
            if np.any(mask):
                centroids.append(np.mean(X[mask], axis=0))

        return {
            'cluster_labels': labels.tolist(),
            'n_clusters': n_clusters,
            'centroids': np.array(centroids).tolist() if centroids else [],
            'cluster_sizes': [np.sum(labels == i) for i in range(n_clusters)],
        }

    def compute_topological_descriptor(
        self,
        resonance_signature: Dict[str, Any],
        persistence_data: Dict[str, Any]
    ) -> npt.NDArray[np.float32]:
        """
        Combine resonance and topological features into a unified descriptor.

        Args:
            resonance_signature: Resonance analysis results
            persistence_data: Persistence homology results

        Returns:
            Combined descriptor vector
        """
        # Extract resonance features
        resonance_features = np.concatenate([
            resonance_signature['s_norm'],
            [resonance_signature['entropy'], resonance_signature['participation_ratio_normalized']]
        ])

        # Extract topological features
        topo_features = []
        for dim in range(self.config.max_homology_dim + 1):
            key = f'h{dim}_features'
            if key in persistence_data['persistence_features']:
                features = persistence_data['persistence_features'][key]
                topo_features.extend([
                    features['max_persistence'],
                    features['mean_persistence'],
                    features['n_features']
                ])
            else:
                # Pad with zeros if no features in this dimension
                topo_features.extend([0.0, 0.0, 0.0])

        # Combine features
        descriptor = np.concatenate([resonance_features, topo_features])

        return descriptor.astype(np.float32)
