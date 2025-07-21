import networkx as nx
import uuid
import numpy as np
import torch
from collections import OrderedDict

class DAGNode:
    def __init__(self, nn_parameters, peer, epoch, malicious=False, corrupted=False):
        self.id = str(uuid.uuid4())
        self.peer = peer
        self.epoch = epoch
        self.data = nn_parameters  # Now stores neural network parameters (OrderedDict)
        self.approved_by = set()
        self.cw = 1  # weight of this node
        self.malicious = malicious  # Indicates if the node is malicious
        self.corrupted = corrupted  # Indicates if the node is corrupted

    def get_flattened_parameters(self):
        """
        Flatten neural network parameters into a single vector for similarity calculations.
        
        Returns:
            numpy.ndarray: Flattened parameter vector
        """
        if self.data is None:
            return np.array([])
            
        flattened_params = []
        if isinstance(self.data, (dict, OrderedDict)):
            for param_name, param_tensor in self.data.items():
                # Convert tensor to numpy if needed and flatten
                if torch.is_tensor(param_tensor):
                    flattened_params.append(param_tensor.detach().cpu().numpy().flatten())
                elif isinstance(param_tensor, np.ndarray):
                    flattened_params.append(param_tensor.flatten())
                else:
                    # Handle other data types if necessary
                    flattened_params.append(np.array([param_tensor]).flatten())
        else:
            # Fallback for backward compatibility with simple matrices
            if torch.is_tensor(self.data):
                flattened_params.append(self.data.detach().cpu().numpy().flatten())
            elif isinstance(self.data, np.ndarray):
                flattened_params.append(self.data.flatten())
            else:
                flattened_params.append(np.array([self.data]).flatten())
        
        return np.concatenate(flattened_params) if flattened_params else np.array([])

    def get_parameter_statistics(self):
        """
        Get statistical information about the parameters for visualization.
        
        Returns:
            dict: Dictionary containing mean, std, and norm of parameters
        """
        flattened = self.get_flattened_parameters()
        if len(flattened) == 0:
            return {'mean': 0.0, 'std': 0.0, 'norm': 0.0, 'num_params': 0}
        
        return {
            'mean': np.mean(flattened),
            'std': np.std(flattened),
            'norm': np.linalg.norm(flattened),
            'num_params': len(flattened)
        }

    def __repr__(self):
        return f"DAGNode({self.id[:8]}, malicious={self.malicious})"