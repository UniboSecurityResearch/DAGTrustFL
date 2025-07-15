import numpy as np
import torch
from collections import OrderedDict

def flatten_parameters(parameters):
    """
    Flatten neural network parameters into a single vector.
    
    Args:
        parameters: Neural network parameters (OrderedDict, dict, or simple matrix)
        
    Returns:
        numpy.ndarray: Flattened parameter vector
    """
    if parameters is None:
        return np.array([])
        
    flattened_params = []
    if isinstance(parameters, (dict, OrderedDict)):
        for param_name, param_tensor in parameters.items():
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
        if torch.is_tensor(parameters):
            flattened_params.append(parameters.detach().cpu().numpy().flatten())
        elif isinstance(parameters, np.ndarray):
            flattened_params.append(parameters.flatten())
        else:
            flattened_params.append(np.array([parameters]).flatten())
    
    return np.concatenate(flattened_params) if flattened_params else np.array([])

def cosine_similarity_parameters(params1, params2):
    """
    Calculate cosine similarity between two sets of neural network parameters.
    
    Args:
        params1: First set of parameters
        params2: Second set of parameters
        
    Returns:
        float: Cosine similarity score with magnitude penalty
    """
    v1 = flatten_parameters(params1)
    v2 = flatten_parameters(params2)
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Check if either norm is zero to avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # Zero similarity if one of the vectors is null
    
    # Standard cosine similarity
    cosine_sim = np.dot(v1, v2) / (norm_v1 * norm_v2)
    
    # Penalize magnitude differences
    magnitude_penalty = 1 - abs(norm_v1 - norm_v2) / max(norm_v1, norm_v2)
    
    # Return a combination of cosine similarity and magnitude penalty
    return cosine_sim * magnitude_penalty

def get_parameter_statistics(parameters):
    """
    Get statistical information about the parameters for visualization.
    
    Args:
        parameters: Neural network parameters
        
    Returns:
        dict: Dictionary containing mean, std, and norm of parameters
    """
    flattened = flatten_parameters(parameters)
    if len(flattened) == 0:
        return {'mean': 0.0, 'std': 0.0, 'norm': 0.0, 'num_params': 0}
    
    return {
        'mean': np.mean(flattened),
        'std': np.std(flattened),
        'norm': np.linalg.norm(flattened),
        'num_params': len(flattened)
    }

def generate_random_parameters_like(template_params, malicious=False, corrupted=False):
    """
    Generate random parameters with the same structure as template parameters.
    This is useful for testing and generating dummy data.
    
    Args:
        template_params: Template parameters to match structure
        malicious: Whether to generate malicious parameters (larger values)
        corrupted: Whether to generate corrupted parameters
        
    Returns:
        dict/OrderedDict: Parameters with same structure as template
    """
    if not isinstance(template_params, (dict, OrderedDict)):
        # Fallback to simple matrix generation
        size = template_params.shape if hasattr(template_params, 'shape') else (3, 3)
        if malicious:
            return np.random.rand(*size) * 100
        elif corrupted:
            return np.random.rand(*size)
        else:
            return np.random.rand(*size)
    
    generated_params = OrderedDict() if isinstance(template_params, OrderedDict) else {}
    
    for param_name, param_tensor in template_params.items():
        if torch.is_tensor(param_tensor):
            shape = param_tensor.shape
            dtype = param_tensor.dtype
            device = param_tensor.device
            
            # Handle different data types appropriately
            try:
                if dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.long]:
                    # For integer types, use randint
                    if malicious:
                        # Generate larger integer values for malicious parameters
                        new_tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
                    elif corrupted:
                        # Generate different integer values for corrupted
                        new_tensor = torch.randint(0, 10, shape, dtype=dtype, device=device)
                    else:
                        # Generate normal integer values
                        new_tensor = torch.randint(0, 2, shape, dtype=dtype, device=device)
                elif dtype == torch.bool:
                    # For boolean types
                    new_tensor = torch.randint(0, 2, shape, device=device).bool()
                else:
                    # For floating point types (float32, float64, etc.)
                    if malicious:
                        # Generate larger values for malicious parameters
                        new_tensor = torch.rand(shape, device=device).to(dtype) * 100
                    elif corrupted:
                        # Generate slightly different parameters for corrupted
                        new_tensor = torch.rand(shape, device=device).to(dtype)
                    else:
                        # Generate normal parameters
                        new_tensor = torch.rand(shape, device=device).to(dtype)
            except Exception as e:
                # Fallback: create tensor with same shape but safe dtype
                print(f"Warning: Could not generate random tensor for {param_name} with dtype {dtype}. Using fallback.")
                if dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.long]:
                    new_tensor = torch.zeros(shape, dtype=dtype, device=device)
                elif dtype == torch.bool:
                    new_tensor = torch.zeros(shape, dtype=torch.bool, device=device)
                else:
                    new_tensor = torch.zeros(shape, dtype=dtype, device=device)
                
            generated_params[param_name] = new_tensor
        elif isinstance(param_tensor, np.ndarray):
            shape = param_tensor.shape
            if malicious:
                generated_params[param_name] = np.random.rand(*shape) * 100
            elif corrupted:
                generated_params[param_name] = np.random.rand(*shape)
            else:
                generated_params[param_name] = np.random.rand(*shape)
        else:
            # Handle scalar values
            if malicious:
                generated_params[param_name] = float(np.random.rand() * 100)
            else:
                generated_params[param_name] = float(np.random.rand())
    
    return generated_params

def copy_parameter_structure(template_params):
    """
    Create a copy of parameter structure with zeros.
    This is safer than generating random values and useful for initialization.
    
    Args:
        template_params: Template parameters to match structure
        
    Returns:
        dict/OrderedDict: Parameters with same structure as template, filled with zeros
    """
    if not isinstance(template_params, (dict, OrderedDict)):
        # Fallback to simple matrix
        if hasattr(template_params, 'shape'):
            return np.zeros_like(template_params)
        else:
            return np.zeros((3, 3))
    
    generated_params = OrderedDict() if isinstance(template_params, OrderedDict) else {}
    
    for param_name, param_tensor in template_params.items():
        if torch.is_tensor(param_tensor):
            # Create zeros with same shape, dtype, and device
            generated_params[param_name] = torch.zeros_like(param_tensor)
        elif isinstance(param_tensor, np.ndarray):
            generated_params[param_name] = np.zeros_like(param_tensor)
        else:
            # Handle scalar values
            generated_params[param_name] = 0.0
    
    return generated_params
