"""
Dataset loading utilities for various graph datasets.

This module provides utilities to load and preprocess graph datasets including:
- Citation networks: Cora, CiteSeer, PubMed
- Social networks: KarateClub, IMDB, Reddit
- Academic networks: WikiCS
- Open Graph Benchmark (OGB) node classification datasets:
  * Small scale: ogbn-arxiv (~170K nodes)
  * Medium scale: ogbn-products (~2.4M nodes), ogbn-proteins (~133K nodes)
  * Large scale: ogbn-mag (~1.9M nodes), ogbn-papers100M (~111M nodes)
- Custom SO_relation datasets for metabolism pathway analysis

All datasets are processed for graph summarization experiments.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, KarateClub, IMDB, Reddit, WikiCS, WebKB
from torch_geometric.datasets.tu_dataset import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from typing import Dict, Tuple, Optional, List, Union
import os
import numpy as np
import pandas as pd
import networkx as nx

# OGB imports
try:
    from ogb.nodeproppred import PygNodePropPredDataset
    OGB_AVAILABLE = True
except ImportError:
    OGB_AVAILABLE = False
    print("Warning: OGB not available. Install with: pip install ogb")


class DatasetLoader:
    """
    Utility class for loading various graph datasets with multiple label tasks.

    Supports multiple dataset types:
    - Citation networks: Cora, CiteSeer, PubMed
    - Social networks: KarateClub, IMDB, Reddit
    - Academic networks: WikiCS
    - Custom datasets: SO_relation_ME, SO_relation_MT
    - OGB node classification: ogbn-arxiv, ogbn-products, ogbn-proteins, ogbn-mag, ogbn-papers100M

    Each dataset supports multiple label tasks:
    1. Original node labels (if available)
    2. Degree-based labels (high/medium/low degree)
    3. Degree Centrality-based labels (high/medium/low degree centrality)
    4. PageRank-based labels (high/medium/low PageRank)
    5. Closeness Centrality-based labels (high/medium/low closeness centrality)

    All graphs are represented in sparse format as required.
    """

    # Dataset categories
    CITATION_DATASETS = ['Cora', 'CiteSeer', 'PubMed']
    SOCIAL_DATASETS = ['KarateClub', 'IMDB', 'Reddit']
    ACADEMIC_DATASETS = ['WikiCS']
    WEBKB_DATASETS = ['Cornell', 'Texas', 'Wisconsin']
    SO_RELATION_DATASETS = ['SO_relation_ME', 'SO_relation_MT']
    LAKE_DATASET_DIRS = ['50lake_networks', '230lake_networks']

    # OGB Node Classification datasets by scale
    OGB_SMALL_DATASETS = ['ogbn-arxiv']  # ~170K nodes
    OGB_MEDIUM_DATASETS = ['ogbn-products', 'ogbn-proteins']  # ~2.4M, ~133K nodes
    OGB_LARGE_DATASETS = ['ogbn-mag', 'ogbn-papers100M']  # ~1.9M, ~111M nodes

    OGB_DATASETS = OGB_SMALL_DATASETS + OGB_MEDIUM_DATASETS + OGB_LARGE_DATASETS

    SUPPORTED_DATASETS = CITATION_DATASETS + SOCIAL_DATASETS + ACADEMIC_DATASETS + WEBKB_DATASETS + SO_RELATION_DATASETS

    # Add OGB datasets only if OGB is available
    if OGB_AVAILABLE:
        SUPPORTED_DATASETS += OGB_DATASETS
    
    def __init__(self, root_dir: str = './data'):
        """
        Initialize dataset loader.
        
        Args:
            root_dir: Root directory to store dataset files
        """
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)
    
    def load_dataset(self,
                     dataset_name: str,
                     task_type: str = 'original',
                     normalize_features: bool = True,
) -> Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load a graph dataset with specified label task.

        Args:
            dataset_name: Name of dataset
            task_type: Type of labeling task ('original', 'degree', 'degree_centrality', 'pagerank', 'closeness_centrality')
            normalize_features: Whether to normalize node features

        Returns:
            Tuple containing:
            - Data: PyTorch Geometric Data object (with labels for specified task)
            - train_mask: Boolean tensor for training nodes
            - val_mask: Boolean tensor for validation nodes
            - test_mask: Boolean tensor for test nodes
        """
        if task_type not in ['original', 'degree', 'degree_centrality', 'pagerank', 'closeness_centrality']:
            raise ValueError(f"Task type {task_type} not supported. Choose from ['original', 'degree', 'degree_centrality', 'pagerank', 'closeness_centrality']")

        transform = NormalizeFeatures() if normalize_features else None
        
        # Load different dataset types
        if dataset_name in self.CITATION_DATASETS:
            dataset = Planetoid(root=self.root_dir, name=dataset_name, transform=transform)
            data = dataset[0]
            train_mask = data.train_mask
            val_mask = data.val_mask  
            test_mask = data.test_mask
            
        elif dataset_name == 'KarateClub':
            dataset = KarateClub(transform=transform)
            data = dataset[0]
            # Create masks for Karate Club (no predefined splits)
            train_mask, val_mask, test_mask = self._create_node_masks(data.num_nodes)
            
        elif dataset_name == 'IMDB':
            dataset = IMDB(root=self.root_dir, transform=transform)
            data = dataset[0]
            # IMDB has predefined masks
            if hasattr(data, 'train_mask'):
                train_mask = data.train_mask
                val_mask = data.val_mask if hasattr(data, 'val_mask') else None
                test_mask = data.test_mask if hasattr(data, 'test_mask') else None
                if val_mask is None or test_mask is None:
                    train_mask, val_mask, test_mask = self._create_node_masks(data.num_nodes)
            else:
                train_mask, val_mask, test_mask = self._create_node_masks(data.num_nodes)
                
        elif dataset_name == 'Reddit':
            # Reddit is too large, so we'll skip it for now
            raise ValueError("Reddit dataset is too large for this benchmark. Try smaller datasets.")

        elif dataset_name == 'WikiCS':
            dataset = WikiCS(root=self.root_dir, transform=transform)
            data = dataset[0]
            # WikiCS has predefined train/val/test splits
            if hasattr(data, 'train_mask') and data.train_mask is not None:
                # WikiCS has multiple training splits, we use the first one
                if data.train_mask.dim() > 1:
                    train_mask = data.train_mask[:, 0]  # Use first split
                    val_mask = data.val_mask
                    test_mask = data.test_mask
                else:
                    train_mask = data.train_mask
                    val_mask = data.val_mask
                    test_mask = data.test_mask
            else:
                # Create masks if not available
                train_mask, val_mask, test_mask = self._create_node_masks(data.num_nodes)

        elif dataset_name in self.WEBKB_DATASETS:
            dataset = WebKB(root=self.root_dir, name=dataset_name, transform=transform)
            data = dataset[0]
            train_mask = data.train_mask[:, 0] if data.train_mask.dim() > 1 else data.train_mask
            val_mask = data.val_mask[:, 0] if data.val_mask.dim() > 1 else data.val_mask
            test_mask = data.test_mask[:, 0] if data.test_mask.dim() > 1 else data.test_mask

        elif dataset_name in self.SO_RELATION_DATASETS:
            data = self._load_so_relation_dataset(dataset_name, normalize_features)
            train_mask, val_mask, test_mask = self._create_node_masks(data.num_nodes)

        elif dataset_name in self.OGB_DATASETS:
            if not OGB_AVAILABLE:
                raise ImportError("OGB package is required for OGB datasets. Install with: pip install ogb")
            data, train_mask, val_mask, test_mask = self._load_ogb_dataset(dataset_name, normalize_features)

        elif self._resolve_lake_network_path(dataset_name) is not None:
            data = self._load_lake_dataset(dataset_name, normalize_features)
            train_mask, val_mask, test_mask = self._create_node_masks(data.num_nodes)

        else:
            supported = self.SUPPORTED_DATASETS + self.list_available_lake_datasets()
            raise ValueError(f"Dataset {dataset_name} loading not implemented yet")
        
        # Ensure all required attributes exist
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            raise ValueError(f"Dataset {dataset_name} does not have edge_index")
        
        if not hasattr(data, 'x') or data.x is None:
            # Create dummy features if none exist
            data.x = torch.eye(data.num_nodes)
            print(f"Created identity features for {dataset_name}")
        
        if not hasattr(data, 'y') or data.y is None:
            # Create dummy labels if none exist  
            data.y = torch.zeros(data.num_nodes, dtype=torch.long)
            print(f"Created dummy labels for {dataset_name}")
        
        # Process labels based on task type
        original_labels = data.y.clone() if hasattr(data, 'y') and data.y is not None else None
        data, train_mask, val_mask, test_mask = self._setup_labels_for_task(
            data, task_type, original_labels, train_mask, val_mask, test_mask)

        # Print dataset info
        print(f"Loaded {dataset_name} dataset (task: {task_type}):")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.edge_index.size(1)}")
        print(f"  Features: {data.x.size(1)}")
        print(f"  Classes: {int(data.y.max()) + 1}")
        print(f"  Training nodes: {train_mask.sum()}")
        print(f"  Validation nodes: {val_mask.sum()}")
        print(f"  Test nodes: {test_mask.sum()}")

        return data, train_mask, val_mask, test_mask
    
    def _create_node_masks(self, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create train/val/test masks for datasets without predefined splits."""
        return DatasetSplitter.create_random_split(
            num_nodes, 
            train_ratio=0.6, 
            val_ratio=0.2, 
            test_ratio=0.2,
            seed=42
        )

    def _setup_labels_for_task(self,
                               data: Data,
                               task_type: str,
                               original_labels: Optional[torch.Tensor],
                               train_mask: torch.Tensor,
                               val_mask: torch.Tensor,
                               test_mask: torch.Tensor) -> Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Setup labels for the specified task type.

        Args:
            data: Graph data object
            task_type: Type of task ('original', 'degree', 'degree_centrality', 'pagerank', 'closeness_centrality')
            original_labels: Original node labels if available
            train_mask, val_mask, test_mask: Data split masks

        Returns:
            Tuple of (Data object with appropriate labels, train_mask, val_mask, test_mask)
        """
        if task_type == 'original':
            # Check if this is SO_relation dataset (use pathway labels as original)
            if hasattr(data, '_dataset_name') and data._dataset_name.startswith('SO_relation'):
                data.y = self._create_pathway_labels(data)
                # Create new random splits for SO_relation original (pathway) task
                new_train_mask, new_val_mask, new_test_mask = self._create_node_masks(data.num_nodes)
                return data, new_train_mask, new_val_mask, new_test_mask

            if original_labels is not None:
                data.y = original_labels
                # Use original splits for original labels
                return data, train_mask, val_mask, test_mask
            else:
                # No original labels available, create dummy labels
                data.y = torch.zeros(data.num_nodes, dtype=torch.long)
                print(f"Warning: No original labels available, created dummy labels")
                return data, train_mask, val_mask, test_mask

        elif task_type == 'degree':
            # Generate degree-based labels
            data.y = self._create_degree_labels(data.edge_index, data.num_nodes)
            # Create new random splits for degree task
            new_train_mask, new_val_mask, new_test_mask = self._create_node_masks(data.num_nodes)
            return data, new_train_mask, new_val_mask, new_test_mask

        elif task_type == 'degree_centrality':
            # Generate degree centrality-based labels
            data.y = self._create_degree_centrality_labels(data.edge_index, data.num_nodes)
            # Create new random splits for degree centrality task
            new_train_mask, new_val_mask, new_test_mask = self._create_node_masks(data.num_nodes)
            return data, new_train_mask, new_val_mask, new_test_mask

        elif task_type == 'pagerank':
            # Generate PageRank-based labels
            data.y = self._create_pagerank_labels(data.edge_index, data.num_nodes)
            # Create new random splits for PageRank task
            new_train_mask, new_val_mask, new_test_mask = self._create_node_masks(data.num_nodes)
            return data, new_train_mask, new_val_mask, new_test_mask

        elif task_type == 'closeness_centrality':
            # Generate closeness centrality-based labels
            data.y = self._create_closeness_centrality_labels(data.edge_index, data.num_nodes)
            # Create new random splits for closeness centrality task
            new_train_mask, new_val_mask, new_test_mask = self._create_node_masks(data.num_nodes)
            return data, new_train_mask, new_val_mask, new_test_mask

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _create_degree_labels(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Create degree-based labels (high/medium/low degree).

        Args:
            edge_index: Graph edge indices
            num_nodes: Number of nodes

        Returns:
            Tensor with degree-based labels (0: low, 1: medium, 2: high)
        """
        from torch_geometric.utils import degree

        # Calculate node degrees
        node_degrees = degree(edge_index[0], num_nodes).float()

        # Use rank-based assignment instead of thresholding so tied degrees do not
        # collapse an entire class, which happens frequently on lake networks.
        sorted_indices = torch.argsort(node_degrees)
        labels = torch.zeros(num_nodes, dtype=torch.long)

        n_low = max(num_nodes // 3, 1)
        n_medium = max(num_nodes // 3, 1)
        low_end = min(n_low, num_nodes)
        medium_end = min(n_low + n_medium, num_nodes)

        labels[sorted_indices[low_end:medium_end]] = 1
        labels[sorted_indices[medium_end:]] = 2

        print(f"Degree-based labels: Low={torch.sum(labels == 0)}, "
              f"Medium={torch.sum(labels == 1)}, High={torch.sum(labels == 2)}")

        return labels

    def _create_degree_centrality_labels(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Create degree centrality-based labels (high/medium/low degree centrality).

        Args:
            edge_index: Graph edge indices
            num_nodes: Number of nodes

        Returns:
            Tensor with degree centrality-based labels (0: low, 1: medium, 2: high)
        """
        import networkx as nx

        # Convert to NetworkX for degree centrality calculation
        edge_list = edge_index.t().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_list)

        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(G)

        # Convert to tensor
        centrality_values = torch.tensor([degree_centrality.get(i, 0.0) for i in range(num_nodes)], dtype=torch.float)

        # Sort centrality to find thresholds
        sorted_centrality, sorted_indices = torch.sort(centrality_values)

        # Split into three roughly equal groups
        n_low = num_nodes // 3
        n_medium = num_nodes // 3

        low_threshold = sorted_centrality[n_low - 1] if n_low > 0 else 0
        medium_threshold = sorted_centrality[n_low + n_medium - 1] if n_low + n_medium > 0 else 0

        # Create labels: 0=low, 1=medium, 2=high
        labels = torch.zeros(num_nodes, dtype=torch.long)
        labels[centrality_values > low_threshold] = 1  # medium
        labels[centrality_values > medium_threshold] = 2  # high

        print(f"Degree centrality-based labels: Low={torch.sum(labels == 0)}, "
              f"Medium={torch.sum(labels == 1)}, High={torch.sum(labels == 2)}")

        return labels

    def _create_pagerank_labels(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Create PageRank-based labels (high/medium/low PageRank).

        Args:
            edge_index: Graph edge indices
            num_nodes: Number of nodes

        Returns:
            Tensor with PageRank-based labels (0: low, 1: medium, 2: high)
        """
        import networkx as nx

        # Convert to NetworkX for PageRank calculation
        edge_list = edge_index.t().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_list)

        # Calculate PageRank
        try:
            pagerank = nx.pagerank(G, max_iter=1000)
        except:
            # If convergence fails, use shorter iteration
            pagerank = nx.pagerank(G, max_iter=100)

        # Convert to tensor
        pagerank_values = torch.tensor([pagerank.get(i, 1.0/num_nodes) for i in range(num_nodes)], dtype=torch.float)

        # Sort PageRank to find thresholds
        sorted_pagerank, sorted_indices = torch.sort(pagerank_values)

        # Split into three roughly equal groups
        n_low = num_nodes // 3
        n_medium = num_nodes // 3

        low_threshold = sorted_pagerank[n_low - 1] if n_low > 0 else 0
        medium_threshold = sorted_pagerank[n_low + n_medium - 1] if n_low + n_medium > 0 else 0

        # Create labels: 0=low, 1=medium, 2=high
        labels = torch.zeros(num_nodes, dtype=torch.long)
        labels[pagerank_values > low_threshold] = 1  # medium
        labels[pagerank_values > medium_threshold] = 2  # high

        print(f"PageRank-based labels: Low={torch.sum(labels == 0)}, "
              f"Medium={torch.sum(labels == 1)}, High={torch.sum(labels == 2)}")

        return labels

    def _create_closeness_centrality_labels(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Create closeness centrality-based labels (high/medium/low closeness centrality).

        Closeness centrality is calculated as the reciprocal of the average shortest-path
        distance from a node to ALL other reachable nodes in the graph. For disconnected
        graphs, the Wasserman-Faust improved formula is used, which normalizes by the
        fraction of reachable nodes.

        This examines whether pruning retains essential shortest-path structures by
        measuring how close each node is to all other nodes in the entire network.

        Args:
            edge_index: Graph edge indices
            num_nodes: Number of nodes

        Returns:
            Tensor with closeness centrality-based labels (0: low, 1: medium, 2: high)
        """
        import networkx as nx

        # Convert to NetworkX for closeness centrality calculation
        edge_list = edge_index.t().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_list)

        # Calculate closeness centrality using Wasserman-Faust improved formula
        # This computes closeness based on ALL reachable nodes in the graph
        # wf_improved=True handles disconnected graphs properly
        try:
            closeness = nx.closeness_centrality(G, wf_improved=True)
        except:
            # If calculation fails, use degree centrality as fallback
            print("Warning: Closeness centrality calculation failed, using degree centrality as fallback")
            closeness = nx.degree_centrality(G)

        # Convert to tensor
        closeness_values = torch.tensor([closeness.get(i, 0.0) for i in range(num_nodes)], dtype=torch.float)

        # Sort closeness to find thresholds
        sorted_closeness, sorted_indices = torch.sort(closeness_values)

        # Split into three roughly equal groups
        n_low = num_nodes // 3
        n_medium = num_nodes // 3

        low_threshold = sorted_closeness[n_low - 1] if n_low > 0 else 0
        medium_threshold = sorted_closeness[n_low + n_medium - 1] if n_low + n_medium > 0 else 0

        # Create labels: 0=low, 1=medium, 2=high
        labels = torch.zeros(num_nodes, dtype=torch.long)
        labels[closeness_values > low_threshold] = 1  # medium
        labels[closeness_values > medium_threshold] = 2  # high

        print(f"Closeness centrality-based labels: Low={torch.sum(labels == 0)}, "
              f"Medium={torch.sum(labels == 1)}, High={torch.sum(labels == 2)}")

        return labels

    def _load_so_relation_dataset(self, dataset_name: str, normalize_features: bool = True) -> Data:
        """
        Load SO_relation dataset (ME or MT).

        Args:
            dataset_name: 'SO_relation_ME' or 'SO_relation_MT'
            normalize_features: Whether to normalize features

        Returns:
            Data object for the SO_relation dataset
        """
        if dataset_name == 'SO_relation_ME':
            network_file = 'ko_relation_min0_network_ME.tsv'
        elif dataset_name == 'SO_relation_MT':
            network_file = 'ko_relation_min0_network_MT.tsv'
        else:
            raise ValueError(f"Unknown SO_relation dataset: {dataset_name}")

        # Load network data
        network_path = os.path.join(self.root_dir, 'SO_relation', 'raw', network_file)
        if not os.path.exists(network_path):
            raise FileNotFoundError(f"SO_relation network file not found: {network_path}")

        # Read network data
        network_df = pd.read_csv(network_path, sep='\t')

        # Get unique KO nodes
        all_kos = set(network_df['KO1'].unique()) | set(network_df['KO2'].unique())
        ko_to_idx = {ko: idx for idx, ko in enumerate(sorted(all_kos))}
        num_nodes = len(all_kos)

        # Create edge index with weight threshold filtering
        edge_list = []
        edge_weights = []
        weight_threshold = 0.05  # Filter edges with weight >= 0.05 to convert weighted to unweighted graph

        for _, row in network_df.iterrows():
            ko1_idx = ko_to_idx[row['KO1']]
            ko2_idx = ko_to_idx[row['KO2']]
            weight = float(row['Weight'])

            # Only include edges with weight >= threshold
            if weight >= weight_threshold:
                # Add both directions for undirected graph
                edge_list.extend([(ko1_idx, ko2_idx), (ko2_idx, ko1_idx)])
                edge_weights.extend([weight, weight])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)

        # Create node features (identity matrix or random features)
        if normalize_features:
            # Use random features that can be normalized
            x = torch.randn(num_nodes, min(100, num_nodes))
            x = torch.nn.functional.normalize(x, dim=1)
        else:
            # Use identity matrix as features
            x = torch.eye(num_nodes)

        # Create data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )

        # Store dataset info for later use
        data._dataset_name = dataset_name
        data._ko_to_idx = ko_to_idx
        data._idx_to_ko = {idx: ko for ko, idx in ko_to_idx.items()}

        return data

    def _resolve_lake_network_path(self, dataset_name: str) -> Optional[str]:
        """
        Resolve a lake network TSV file for names like HongL or XYH.

        Matching is case-insensitive and prefers filenames ending with
        `_<dataset_name>_network.tsv` when multiple matches exist.
        """
        matches = []
        target = dataset_name.lower()

        for lake_dir in self.LAKE_DATASET_DIRS:
            search_dir = os.path.join(self.root_dir, lake_dir)
            if not os.path.isdir(search_dir):
                continue

            for filename in sorted(os.listdir(search_dir)):
                if not filename.endswith('.tsv'):
                    continue
                stem = os.path.splitext(filename)[0].lower()
                if target in stem:
                    matches.append(os.path.join(search_dir, filename))

        if not matches:
            return None

        exact_suffix = f"_{target}_network.tsv"
        exact_matches = [path for path in matches if path.lower().endswith(exact_suffix)]
        if len(exact_matches) == 1:
            return exact_matches[0]

        return matches[0]

    def list_available_lake_datasets(self) -> List[str]:
        """List lake dataset aliases inferred from local TSV filenames."""
        aliases = set()

        for lake_dir in self.LAKE_DATASET_DIRS:
            search_dir = os.path.join(self.root_dir, lake_dir)
            if not os.path.isdir(search_dir):
                continue

            for filename in os.listdir(search_dir):
                if not filename.endswith('.tsv'):
                    continue
                stem = os.path.splitext(filename)[0]
                if stem.endswith('_network'):
                    aliases.add(stem[:-len('_network')].split('_')[-1])

        return sorted(aliases)

    def _load_lake_dataset(self, dataset_name: str, normalize_features: bool = True) -> Data:
        """
        Load a directed lake network TSV as a benchmark dataset.

        The input TSV is expected to contain columns: consumer, resource, weight.
        We keep the original directed edges here; the benchmark preprocessor later
        converts graphs to undirected form for fair model comparison.
        """
        network_path = self._resolve_lake_network_path(dataset_name)
        if network_path is None:
            raise FileNotFoundError(f"Lake network file not found for dataset: {dataset_name}")

        network_df = pd.read_csv(network_path, sep='\t')
        required_columns = {'consumer', 'resource'}
        if not required_columns.issubset(network_df.columns):
            raise ValueError(
                f"Lake network file missing required columns {required_columns}: {network_path}"
            )

        sources = network_df['consumer'].astype(str).tolist()
        targets = network_df['resource'].astype(str).tolist()

        all_nodes = sorted(set(sources) | set(targets))
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

        edge_list = [
            (node_to_idx[src], node_to_idx[tgt])
            for src, tgt in zip(sources, targets)
        ]

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        num_nodes = len(all_nodes)
        if normalize_features:
            x = torch.randn(num_nodes, min(64, max(num_nodes, 1)))
            x = torch.nn.functional.normalize(x, dim=1)
        else:
            x = torch.eye(num_nodes, dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.zeros(num_nodes, dtype=torch.long),
            num_nodes=num_nodes
        )

        data._dataset_name = dataset_name
        data._lake_network_path = network_path
        data._node_to_idx = node_to_idx
        data._idx_to_node = {idx: node for node, idx in node_to_idx.items()}

        return data

    def _create_pathway_labels(self, data: Data) -> torch.Tensor:
        """
        Create pathway-based labels for SO_relation datasets.

        Args:
            data: SO_relation data object with KO mapping

        Returns:
            Tensor with pathway-based labels
        """
        # Load KO label information
        ko_label_path = os.path.join(self.root_dir, 'SO_relation', 'raw', 'KO_label.tsv')
        if not os.path.exists(ko_label_path):
            raise FileNotFoundError(f"KO label file not found: {ko_label_path}")

        ko_labels_df = pd.read_csv(ko_label_path, sep='\t')

        # Create KO to pathway mapping
        ko_to_pathway = {}
        for _, row in ko_labels_df.iterrows():
            ko_to_pathway[row['KO']] = row['Pathway']

        # Get unique pathways and create mapping
        pathways = set(ko_to_pathway.values())
        pathway_to_idx = {pathway: idx for idx, pathway in enumerate(sorted(pathways))}

        # Create labels for all nodes
        labels = torch.zeros(data.num_nodes, dtype=torch.long)
        unknown_pathway_idx = len(pathway_to_idx)  # Use a separate index for unknown pathways

        for node_idx in range(data.num_nodes):
            ko = data._idx_to_ko[node_idx]
            if ko in ko_to_pathway:
                pathway = ko_to_pathway[ko]
                labels[node_idx] = pathway_to_idx[pathway]
            else:
                labels[node_idx] = unknown_pathway_idx

        print(f"Pathway-based labels: {len(pathway_to_idx)} pathways + 1 unknown")
        for pathway, idx in sorted(pathway_to_idx.items(), key=lambda x: x[1]):
            count = (labels == idx).sum()
            print(f"  {pathway}: {count} nodes")
        unknown_count = (labels == unknown_pathway_idx).sum()
        if unknown_count > 0:
            print(f"  Unknown: {unknown_count} nodes")

        return labels

    def _load_ogb_dataset(self, dataset_name: str, normalize_features: bool = True) -> Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load OGB node classification dataset.

        Args:
            dataset_name: Name of OGB dataset (e.g., 'ogbn-arxiv', 'ogbn-products')
            normalize_features: Whether to normalize features

        Returns:
            Tuple containing:
            - Data: PyTorch Geometric Data object
            - train_mask: Boolean tensor for training nodes
            - val_mask: Boolean tensor for validation nodes
            - test_mask: Boolean tensor for test nodes
        """
        try:
            # Handle OGB's interactive update prompt by mocking input to return 'y'
            import builtins
            import torch
            original_input = builtins.input
            original_torch_load = torch.load

            # Mock input to return 'y' and torch.load to use weights_only=False
            builtins.input = lambda x: 'y'
            torch.load = lambda *args, **kwargs: original_torch_load(*args, **{**kwargs, 'weights_only': False})

            try:
                dataset = PygNodePropPredDataset(name=dataset_name, root=self.root_dir)
                data = dataset[0]
            finally:
                # Restore original functions
                builtins.input = original_input
                torch.load = original_torch_load

            # Get the split indices
            split_idx = dataset.get_idx_split()
            train_idx = split_idx['train']
            val_idx = split_idx['valid']
            test_idx = split_idx['test']

            # Create boolean masks
            num_nodes = data.num_nodes
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

            # Handle node features
            if data.x is None:
                # Create identity features if none exist
                data.x = torch.eye(num_nodes)
                print(f"Created identity features for {dataset_name}")
            elif normalize_features and dataset_name != 'ogbn-arxiv':
                # OGBN-Arxiv ships signed 128-dimensional paper embeddings.
                # PyG NormalizeFeatures shifts negative values before row
                # normalization, which destroys the semantics of these features.
                from torch_geometric.transforms import NormalizeFeatures
                transform = NormalizeFeatures()
                data = transform(data)

            # Handle labels - OGB datasets use y for labels
            if data.y is None:
                # Create dummy labels if none exist
                data.y = torch.zeros(num_nodes, dtype=torch.long)
                print(f"Created dummy labels for {dataset_name}")
            else:
                # OGB labels might be multi-dimensional, flatten if needed
                if data.y.dim() > 1:
                    data.y = data.y.squeeze()

            # Ensure edge_index is in the correct format
            if data.edge_index is None:
                raise ValueError(f"Dataset {dataset_name} does not have edge_index")

            # Convert to undirected if needed (most graph summarization methods assume undirected graphs)
            data.edge_index = to_undirected(data.edge_index)

            # Store dataset metadata
            data._dataset_name = dataset_name
            data._is_ogb = True

            print(f"Loaded OGB dataset {dataset_name}:")
            print(f"  Scale: {'Small' if dataset_name in self.OGB_SMALL_DATASETS else 'Medium' if dataset_name in self.OGB_MEDIUM_DATASETS else 'Large'}")

            return data, train_mask, val_mask, test_mask

        except Exception as e:
            raise RuntimeError(f"Failed to load OGB dataset {dataset_name}: {str(e)}")

    def load_all_datasets(self,
                          task_type: str = 'original',
                          normalize_features: bool = True) -> Dict[str, Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Load all supported datasets with specified task type.

        Args:
            task_type: Type of labeling task ('original' or 'degree')
            normalize_features: Whether to normalize node features

        Returns:
            Dict mapping dataset names to (data, train_mask, val_mask, test_mask) tuples
        """
        datasets = {}
        
        for dataset_name in self.SUPPORTED_DATASETS:
            try:
                datasets[dataset_name] = self.load_dataset(dataset_name, task_type, normalize_features)
                print(f"Successfully loaded {dataset_name} with {task_type} task")
            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")
                
        return datasets

    def load_all_tasks_for_dataset(self,
                                   dataset_name: str,
                                   normalize_features: bool = True) -> Dict[str, Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Load all available tasks for a single dataset.

        Args:
            dataset_name: Name of the dataset
            normalize_features: Whether to normalize node features

        Returns:
            Dict mapping task names to (data, train_mask, val_mask, test_mask) tuples
        """
        tasks = {}
        task_types = ['original', 'degree', 'degree_centrality', 'pagerank', 'closeness_centrality']

        for task_type in task_types:
            try:
                task_data = self.load_dataset(dataset_name, task_type, normalize_features)
                tasks[f"{dataset_name}_{task_type}"] = task_data
                print(f"Successfully loaded {dataset_name} with {task_type} task")
            except Exception as e:
                print(f"Failed to load {dataset_name} with {task_type} task: {e}")

        return tasks

    def get_dataset_scale(self, dataset_name: str) -> str:
        """
        Get the scale category of a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Scale category: 'Small', 'Medium', 'Large', or 'Unknown'
        """
        if dataset_name in self.OGB_SMALL_DATASETS:
            return 'Small'
        elif dataset_name in self.OGB_MEDIUM_DATASETS:
            return 'Medium'
        elif dataset_name in self.OGB_LARGE_DATASETS:
            return 'Large'
        elif dataset_name in self.CITATION_DATASETS + self.SOCIAL_DATASETS + self.BIO_DATASETS + self.SO_RELATION_DATASETS:
            return 'Small'  # Most traditional datasets are relatively small
        else:
            return 'Unknown'

    def list_datasets_by_scale(self) -> Dict[str, List[str]]:
        """
        List all available datasets organized by scale.

        Returns:
            Dict mapping scale categories to lists of dataset names
        """
        datasets_by_scale = {
            'Small': self.CITATION_DATASETS + self.SOCIAL_DATASETS + self.BIO_DATASETS + self.SO_RELATION_DATASETS,
            'Medium': [],
            'Large': []
        }

        if OGB_AVAILABLE:
            datasets_by_scale['Small'].extend(self.OGB_SMALL_DATASETS)
            datasets_by_scale['Medium'].extend(self.OGB_MEDIUM_DATASETS)
            datasets_by_scale['Large'].extend(self.OGB_LARGE_DATASETS)

        # Remove duplicates and sort
        for scale in datasets_by_scale:
            datasets_by_scale[scale] = sorted(list(set(datasets_by_scale[scale])))

        return datasets_by_scale

    def load_all_datasets_all_tasks(self,
                                    normalize_features: bool = True) -> Dict[str, Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Load all datasets with all available tasks.

        Args:
            normalize_features: Whether to normalize node features

        Returns:
            Dict mapping "dataset_task" names to (data, train_mask, val_mask, test_mask) tuples
        """
        all_tasks = {}

        for dataset_name in self.SUPPORTED_DATASETS:
            try:
                dataset_tasks = self.load_all_tasks_for_dataset(dataset_name, normalize_features)
                all_tasks.update(dataset_tasks)
            except Exception as e:
                print(f"Failed to load any tasks for {dataset_name}: {e}")

        return all_tasks

    def get_dataset_info(self, dataset_name: str, task_type: str = 'original') -> Dict[str, int]:
        """
        Get basic information about a dataset with specified task.

        Args:
            dataset_name: Name of the dataset
            task_type: Type of labeling task ('original' or 'degree')

        Returns:
            Dict with dataset statistics
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported")
            
        # Load dataset to get info
        data, train_mask, val_mask, test_mask = self.load_dataset(dataset_name, task_type)
        
        return {
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.size(1) // 2,  # Undirected edges
            'num_features': data.x.size(1),
            'num_classes': int(data.y.max()) + 1,
            'train_nodes': int(train_mask.sum()),
            'val_nodes': int(val_mask.sum()),
            'test_nodes': int(test_mask.sum())
        }
    
    @staticmethod
    def verify_sparse_format(data: Data) -> bool:
        """
        Verify that the graph is in sparse format.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            bool: True if graph is in sparse format
        """
        # Check if edge_index exists and is 2D tensor
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            return False
            
        if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
            return False
            
        # Check if edge_index contains valid indices
        if data.edge_index.min() < 0 or data.edge_index.max() >= data.num_nodes:
            return False
            
        return True
    
    def preprocess_for_summarization(self, 
                                     data: Data, 
                                     remove_self_loops: bool = True,
                                     to_undirected_graph: bool = True) -> Data:
        """
        Preprocess graph data for graph summarization experiments.
        
        Args:
            data: Input graph data
            remove_self_loops: Whether to remove self-loops
            to_undirected_graph: Whether to convert the graph to undirected
            
        Returns:
            Data: Preprocessed graph data
        """
        from torch_geometric.utils import remove_self_loops as remove_loops, add_self_loops
        
        processed_data = data.clone()
        
        # Convert features to float32 for compatibility
        if processed_data.x.dtype == torch.float64:
            processed_data.x = processed_data.x.float()
        
        if remove_self_loops:
            # Remove self-loops for cleaner summarization
            processed_data.edge_index, _ = remove_loops(processed_data.edge_index)
        
        # Ensure the graph is undirected when requested.
        if to_undirected_graph:
            from torch_geometric.utils import to_undirected
            processed_data.edge_index = to_undirected(processed_data.edge_index)
        
        # Verify sparse format
        if not self.verify_sparse_format(processed_data):
            raise ValueError("Processed data is not in valid sparse format")
            
        return processed_data


class DatasetSplitter:
    """
    Utility for creating custom train/validation/test splits for experiments.
    """
    
    @staticmethod
    def create_random_split(num_nodes: int, 
                           train_ratio: float = 0.6,
                           val_ratio: float = 0.2,
                           test_ratio: float = 0.2,
                           seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create random train/validation/test split.
        
        Args:
            num_nodes: Total number of nodes
            train_ratio: Fraction of nodes for training
            val_ratio: Fraction of nodes for validation
            test_ratio: Fraction of nodes for testing
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of boolean masks (train_mask, val_mask, test_mask)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
            
        torch.manual_seed(seed)
        
        # Create random permutation
        perm = torch.randperm(num_nodes)
        
        # Calculate split sizes
        train_size = int(num_nodes * train_ratio)
        val_size = int(num_nodes * val_ratio)
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[perm[:train_size]] = True
        val_mask[perm[train_size:train_size + val_size]] = True
        test_mask[perm[train_size + val_size:]] = True
        
        return train_mask, val_mask, test_mask
    
    @staticmethod
    def create_stratified_split(labels: torch.Tensor,
                               train_ratio: float = 0.6,
                               val_ratio: float = 0.2, 
                               test_ratio: float = 0.2,
                               seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create stratified split maintaining class distribution.
        
        Args:
            labels: Node labels
            train_ratio: Fraction of nodes for training
            val_ratio: Fraction of nodes for validation
            test_ratio: Fraction of nodes for testing
            seed: Random seed
            
        Returns:
            Tuple of boolean masks (train_mask, val_mask, test_mask)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
            
        torch.manual_seed(seed)
        num_nodes = len(labels)
        num_classes = int(labels.max()) + 1
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Split each class separately
        for class_idx in range(num_classes):
            class_indices = torch.where(labels == class_idx)[0]
            class_size = len(class_indices)
            
            if class_size == 0:
                continue
                
            # Random permutation within class
            class_perm = class_indices[torch.randperm(class_size)]
            
            # Calculate split sizes for this class
            class_train_size = max(1, int(class_size * train_ratio))
            class_val_size = max(1, int(class_size * val_ratio))
            
            # Assign nodes to splits
            train_mask[class_perm[:class_train_size]] = True
            val_mask[class_perm[class_train_size:class_train_size + class_val_size]] = True
            test_mask[class_perm[class_train_size + class_val_size:]] = True
            
        return train_mask, val_mask, test_mask
