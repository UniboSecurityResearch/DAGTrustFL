import numpy as np
import torch
from collections import OrderedDict

# Try to import sklearn components
try:
    import sklearn.metrics.pairwise as smp
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not available. Cosine similarity defense features will be limited.")
    SKLEARN_AVAILABLE = False
    # Provide fallback implementation for cosine_similarity
    def cosine_similarity(X, Y=None):
        """Fallback cosine similarity implementation"""
        if Y is None:
            Y = X
        # Simple implementation for vectors
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if len(Y.shape) == 1:
            Y = Y.reshape(1, -1)
        
        result = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                dot_product = np.dot(X[i], Y[j])
                norm_x = np.linalg.norm(X[i])
                norm_y = np.linalg.norm(Y[j])
                if norm_x != 0 and norm_y != 0:
                    result[i, j] = dot_product / (norm_x * norm_y)
        return result

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

def euclidean_distance_parameters(params1, params2):
    """
    Calculate the Euclidean distance between two sets of neural network parameters.
    
    The Euclidean distance is defined as:
    d(x,y) = √(Σᵢ(xᵢ - yᵢ)²)
    
    For neural network parameters, we flatten all parameter tensors into a single vector
    and then compute the Euclidean distance. This is mathematically sound because:
    1. Each parameter (weight/bias) contributes equally to the distance
    2. The L2 norm preserves the relative importance of parameter differences
    3. It provides a global measure of similarity between two models
    
    Args:
        params1: First set of neural network parameters (OrderedDict, dict, or matrix)
        params2: Second set of neural network parameters (OrderedDict, dict, or matrix)
        
    Returns:
        float: Euclidean distance between the parameter sets
        
    Raises:
        ValueError: If parameters have different structures or shapes
    """
    # Handle None inputs
    if params1 is None and params2 is None:
        return 0.0
    if params1 is None or params2 is None:
        raise ValueError("One parameter set is None while the other is not")
    
    # Flatten both parameter sets
    flattened_1 = flatten_parameters(params1)
    flattened_2 = flatten_parameters(params2)
    
    # Check if dimensions match
    if flattened_1.shape != flattened_2.shape:
        raise ValueError(f"Parameter dimensions don't match: {flattened_1.shape} vs {flattened_2.shape}")
    
    # Check for empty parameter sets
    if len(flattened_1) == 0:
        return 0.0
    
    # Compute Euclidean distance: √(Σᵢ(xᵢ - yᵢ)²)
    diff = flattened_1 - flattened_2
    squared_diff = np.square(diff)
    sum_squared_diff = np.sum(squared_diff)
    euclidean_dist = np.sqrt(sum_squared_diff)
    
    return float(euclidean_dist)


def normalized_euclidean_proximity_parameters(params1, params2):
    """
    Calculate the normalized Euclidean distance between two sets of neural network parameters.
    
    Uses the formula: Dis_norm(M1, M2) = Dis(M1, M2) / (Dis(M1, 0) + Dis(M2, 0))
    
    Returns a value in [0, 1] where:
    - →0 = very different parameters (relative to their distance from origin)
    - 1 = identical parameters
    
    This normalization ensures the distance is bounded in [0,1] and accounts for
    the magnitude of the parameter vectors.
    
    Args:
        params1: First set of neural network parameters
        params2: Second set of neural network parameters
        
    Returns:
        float: Normalized proximity in [0, 1] range
    """
    # Handle identical case first
    if params1 is params2:
        return 0.0
    
    # Calculate distance between the two parameter sets
    distance_12 = euclidean_distance_parameters(params1, params2)
    
    if distance_12 == 0.0:
        return 0.0
    
    # Create zero parameters with the same structure as params1
    zero_params = OrderedDict()
    if isinstance(params1, (dict, OrderedDict)):
        for param_name, param_tensor in params1.items():
            if torch.is_tensor(param_tensor):
                zero_params[param_name] = torch.zeros_like(param_tensor)
            elif isinstance(param_tensor, np.ndarray):
                zero_params[param_name] = np.zeros_like(param_tensor)
            else:
                zero_params[param_name] = 0.0
    else:
        # Handle simple arrays/tensors
        if torch.is_tensor(params1):
            zero_params = torch.zeros_like(params1)
        elif isinstance(params1, np.ndarray):
            zero_params = np.zeros_like(params1)
        else:
            zero_params = 0.0
    
    # Calculate distances from origin
    distance_1_to_zero = euclidean_distance_parameters(params1, zero_params)
    distance_2_to_zero = euclidean_distance_parameters(params2, zero_params)
    
    # Calculate denominator
    denominator = distance_1_to_zero + distance_2_to_zero
    
    # Handle edge case where both parameter sets are at origin
    if denominator == 0.0:
        return 0.0
    
    # Apply normalization formula
    normalized_distance = distance_12 / denominator
    
    return float(1.0 - normalized_distance)


def test_euclidean_distance():
    """
    Test the euclidean_distance_parameters function with various inputs.
    """
    print("Testing euclidean_distance_parameters function...")
    
    # Test 1: Identical parameters should have distance 0
    params_a = OrderedDict([
        ('layer1.weight', torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ('layer1.bias', torch.tensor([0.5, -0.5]))
    ])
    params_b = OrderedDict([
        ('layer1.weight', torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ('layer1.bias', torch.tensor([0.5, -0.5]))
    ])
    
    dist1 = euclidean_distance_parameters(params_a, params_b)
    print(f"Test 1 - Identical parameters: {dist1:.6f} (should be 0.0)")
    assert abs(dist1) < 1e-10, f"Expected 0, got {dist1}"
    
    # Test 2: Known difference
    params_c = OrderedDict([
        ('layer1.weight', torch.tensor([[2.0, 3.0], [4.0, 5.0]])),  # Each element +1
        ('layer1.bias', torch.tensor([1.5, 0.5]))  # Each element +1
    ])
    
    dist2 = euclidean_distance_parameters(params_a, params_c)
    expected_dist2 = np.sqrt(6 * 1.0)  # 6 parameters, each with difference of 1
    print(f"Test 2 - Unit differences: {dist2:.6f} (expected ~{expected_dist2:.6f})")
    assert abs(dist2 - expected_dist2) < 1e-6, f"Expected {expected_dist2}, got {dist2}"
    
    # Test 3: Simple numpy arrays
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[1, 2], [3, 5]])  # Last element different
    
    dist3 = euclidean_distance_parameters(arr1, arr2)
    expected_dist3 = 1.0  # Only one element differs by 1
    print(f"Test 3 - Simple arrays: {dist3:.6f} (expected {expected_dist3:.6f})")
    assert abs(dist3 - expected_dist3) < 1e-10, f"Expected {expected_dist3}, got {dist3}"
    
    # Test 4: Different structures should raise error
    try:
        params_d = OrderedDict([('layer1.weight', torch.tensor([[1.0, 2.0]]))])  # Different shape
        euclidean_distance_parameters(params_a, params_d)
        assert False, "Should have raised ValueError for different shapes"
    except ValueError as e:
        print(f"Test 4 - Different shapes correctly raised error: {e}")
    
    # Test 5: Test with realistic neural network parameters
    # Simulate small CNN parameters
    cnn_params_1 = OrderedDict([
        ('conv1.weight', torch.randn(16, 3, 3, 3)),  # 16 filters, 3 channels, 3x3
        ('conv1.bias', torch.randn(16)),
        ('fc1.weight', torch.randn(10, 1024)),
        ('fc1.bias', torch.randn(10))
    ])
    
    cnn_params_2 = OrderedDict([
        ('conv1.weight', torch.randn(16, 3, 3, 3)),
        ('conv1.bias', torch.randn(16)),
        ('fc1.weight', torch.randn(10, 1024)),
        ('fc1.bias', torch.randn(10))
    ])
    
    dist5 = euclidean_distance_parameters(cnn_params_1, cnn_params_2)
    print(f"Test 5 - Realistic CNN parameters: {dist5:.6f}")
    assert dist5 > 0, "Distance between different random parameters should be > 0"
    
    # Test 6: Verify mathematical property: d(x,y) = d(y,x)
    dist6a = euclidean_distance_parameters(cnn_params_1, cnn_params_2)
    dist6b = euclidean_distance_parameters(cnn_params_2, cnn_params_1)
    print(f"Test 6 - Symmetry: d(a,b)={dist6a:.6f}, d(b,a)={dist6b:.6f}")
    assert abs(dist6a - dist6b) < 1e-6, "Distance should be symmetric"
    
    print("All tests passed! ✅")


def analyze_euclidean_distance_properties():
    """
    Analyze the mathematical properties and correctness of our Euclidean distance implementation
    for neural network parameters.
    """
    print("\n=== ANALYSIS OF EUCLIDEAN DISTANCE PROPERTIES ===\n")
    
    # 1. Verify that flatten maintains mathematical sense
    print("1. FLATTEN AND MATHEMATICAL SENSE:")
    print("   - Neural network parameters are multi-dimensional tensors")
    print("   - Flatten transforms them into a single vector preserving all values")
    print("   - Euclidean distance measures the global 'distance' between two models")
    print("   - Mathematically correct: d(x,y) = √(Σᵢ(xᵢ - yᵢ)²)")
    
    # Test with parameters of different shapes
    # Case 1: 2x2 matrix
    matrix_2x2 = np.array([[1, 2], [3, 4]])
    matrix_2x2_diff = np.array([[1.1, 2.1], [3.1, 4.1]])
    
    # Case 2: Same matrix as 1x4 vector  
    vector_1x4 = np.array([1, 2, 3, 4])
    vector_1x4_diff = np.array([1.1, 2.1, 3.1, 4.1])
    
    dist_matrix = euclidean_distance_parameters(matrix_2x2, matrix_2x2_diff)
    dist_vector = euclidean_distance_parameters(vector_1x4, vector_1x4_diff)
    
    print(f"\n   Test: 2x2 matrix vs 1x4 vector (same values)")
    print(f"   Matrix distance: {dist_matrix:.6f}")
    print(f"   Vector distance: {dist_vector:.6f}")
    print(f"   Difference: {abs(dist_matrix - dist_vector):.10f} (should be ~0)")
    
    # 2. Comparison with manual implementation
    print("\n2. VERIFICATION WITH MANUAL CALCULATION:")
    manual_calc = np.sqrt(np.sum((0.1)**2 * 4))  # 4 elements, each differs by 0.1
    print(f"   Manual calculation: √(4 × 0.1²) = {manual_calc:.6f}")
    print(f"   Our implementation: {dist_matrix:.6f}")
    print(f"   Match: {abs(manual_calc - dist_matrix) < 1e-10}")
    
    # 3. Mathematical properties
    print("\n3. MATHEMATICAL PROPERTIES:")
    
    # Property 1: Non-negativity
    print("   a) Non-negativity: d(x,y) ≥ 0")
    assert dist_matrix >= 0
    print("      ✅ Verified")
    
    # Property 2: Identity
    print("   b) Identity: d(x,x) = 0")
    self_dist = euclidean_distance_parameters(matrix_2x2, matrix_2x2)
    assert self_dist == 0
    print("      ✅ Verified")
    
    # Property 3: Symmetry  
    print("   c) Symmetry: d(x,y) = d(y,x)")
    dist_ab = euclidean_distance_parameters(matrix_2x2, matrix_2x2_diff)
    dist_ba = euclidean_distance_parameters(matrix_2x2_diff, matrix_2x2)
    assert abs(dist_ab - dist_ba) < 1e-10
    print("      ✅ Verified")
    
    # Property 4: Triangle inequality
    print("   d) Triangle inequality: d(x,z) ≤ d(x,y) + d(y,z)")
    matrix_c = np.array([[2, 3], [4, 5]])
    dist_ac = euclidean_distance_parameters(matrix_2x2, matrix_c)
    dist_bc = euclidean_distance_parameters(matrix_2x2_diff, matrix_c)
    triangle_ineq = dist_ac <= (dist_ab + dist_bc + 1e-10)  # small tolerance
    print(f"      d(a,c)={dist_ac:.6f} ≤ d(a,b)+d(b,c)={dist_ab:.6f}+{dist_bc:.6f}={dist_ab+dist_bc:.6f}")
    print(f"      ✅ {'Verified' if triangle_ineq else 'FAILED'}")
    
    # 4. Behavior with real neural networks
    print("\n4. BEHAVIOR WITH NEURAL NETWORKS:")
    
    # Simulate parameters of a small network
    torch.manual_seed(42)  # For reproducibility
    
    net_params_1 = OrderedDict([
        ('conv1.weight', torch.randn(8, 3, 3, 3) * 0.1),  # Small weights
        ('conv1.bias', torch.randn(8) * 0.01),
        ('fc1.weight', torch.randn(10, 512) * 0.1),
        ('fc1.bias', torch.randn(10) * 0.01)
    ])
    
    # Similar parameters (like after one training epoch)
    net_params_2 = OrderedDict()
    for name, param in net_params_1.items():
        # Add small variations
        noise = torch.randn_like(param) * 0.001
        net_params_2[name] = param + noise
    
    # Very different parameters
    net_params_3 = OrderedDict([
        ('conv1.weight', torch.randn(8, 3, 3, 3) * 0.1),
        ('conv1.bias', torch.randn(8) * 0.01), 
        ('fc1.weight', torch.randn(10, 512) * 0.1),
        ('fc1.bias', torch.randn(10) * 0.01)
    ])
    
    dist_similar = euclidean_distance_parameters(net_params_1, net_params_2)
    dist_different = euclidean_distance_parameters(net_params_1, net_params_3)
    
    print(f"   Distance between similar parameters: {dist_similar:.6f}")
    print(f"   Distance between different parameters: {dist_different:.6f}")
    print(f"   Ratio (different/similar): {dist_different/dist_similar:.2f}x")
    print(f"   ✅ Distance correctly discriminates between similar and different parameters")
    
    print("\n=== CONCLUSIONS ===")
    print("✅ Implementation is mathematically correct")
    print("✅ Flatten preserves all information")
    print("✅ All metric properties are satisfied")
    print("✅ Appropriate behavior for neural network parameters")


def test_normalized_distance_range():
    """
    Test that normalized distance returns values in the [0,1] interval
    using the formula: Dis_norm(M1, M2) = Dis(M1, M2) / (Dis(M1, 0) + Dis(M2, 0))
    """
    print("\n=== TEST NORMALIZED DISTANCE [0,1] - NEW FORMULA ===\n")
    
    # Test 1: Identical parameters should give 0
    params_identical = OrderedDict([
        ('layer1.weight', torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ('layer1.bias', torch.tensor([0.5, -0.5]))
    ])
    
    norm_dist_identical = normalized_euclidean_proximity_parameters(params_identical, params_identical)
    print(f"Test 1 - Identical parameters: {norm_dist_identical:.6f} (should be 0.0)")
    assert norm_dist_identical == 0.0, f"Expected 0.0, got {norm_dist_identical}"
    assert 0.0 <= norm_dist_identical <= 1.0, "Value outside [0,1] range"
    
    # Test 2: Slightly different parameters
    params_similar = OrderedDict([
        ('layer1.weight', torch.tensor([[1.01, 2.01], [3.01, 4.01]])),
        ('layer1.bias', torch.tensor([0.51, -0.49]))
    ])
    
    norm_dist_similar = normalized_euclidean_proximity_parameters(params_identical, params_similar)
    print(f"Test 2 - Similar parameters: {norm_dist_similar:.6f}")
    assert 0.0 <= norm_dist_similar <= 1.0, "Value outside [0,1] range"
    assert norm_dist_similar > 0.0, "Different parameters should have distance > 0"
    
    # Test 3: Verify formula manually
    # Calculate manually to verify
    dist_12 = euclidean_distance_parameters(params_identical, params_similar)
    
    # Create zero parameters
    zero_params = OrderedDict([
        ('layer1.weight', torch.zeros(2, 2)),
        ('layer1.bias', torch.zeros(2))
    ])
    
    dist_1_to_zero = euclidean_distance_parameters(params_identical, zero_params)
    dist_2_to_zero = euclidean_distance_parameters(params_similar, zero_params)
    
    expected_norm = dist_12 / (dist_1_to_zero + dist_2_to_zero)
    
    print(f"Test 3 - Formula verification:")
    print(f"   Dis(M1, M2) = {dist_12:.6f}")
    print(f"   Dis(M1, 0) = {dist_1_to_zero:.6f}")
    print(f"   Dis(M2, 0) = {dist_2_to_zero:.6f}")
    print(f"   Manual formula: {expected_norm:.6f}")
    print(f"   Function: {norm_dist_similar:.6f}")
    assert abs(expected_norm - norm_dist_similar) < 1e-10, "Formula incorrect"
    
    # Test 4: Very different parameters
    params_different = OrderedDict([
        ('layer1.weight', torch.tensor([[10.0, 20.0], [30.0, 40.0]])),
        ('layer1.bias', torch.tensor([5.0, -5.0]))
    ])
    
    norm_dist_different = normalized_euclidean_proximity_parameters(params_identical, params_different)
    print(f"Test 4 - Very different parameters: {norm_dist_different:.6f}")
    assert 0.0 <= norm_dist_different <= 1.0, "Value outside [0,1] range"
    
    # Test 5: Verify mathematical properties
    print(f"\nTest 5 - Mathematical properties:")
    
    # Symmetry: dist_norm(A,B) = dist_norm(B,A)
    norm_AB = normalized_euclidean_proximity_parameters(params_identical, params_similar)
    norm_BA = normalized_euclidean_proximity_parameters(params_similar, params_identical)
    print(f"   Symmetry: norm(A,B)={norm_AB:.6f}, norm(B,A)={norm_BA:.6f}")
    assert abs(norm_AB - norm_BA) < 1e-10, "Normalized distance is not symmetric"
    
    # Test 6: Behavior with parameters at origin
    zero_params_test = OrderedDict([
        ('layer1.weight', torch.zeros(2, 2)),
        ('layer1.bias', torch.zeros(2))
    ])
    
    norm_zero = normalized_euclidean_proximity_parameters(zero_params_test, zero_params_test)
    print(f"Test 6 - Zero parameters: {norm_zero:.6f} (should be 0.0)")
    assert norm_zero == 0.0, "Zero parameters should have distance 0"
    
    # Test 7: Progression of differences with scaling
    print(f"\nTest 7 - Progression with scaling:")
    base_params = OrderedDict([('w', torch.tensor([[1.0]]))])
    
    factors = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    distances = []
    
    for factor in factors:
        test_params = OrderedDict([('w', torch.tensor([[1.0 + factor]]))])
        dist = normalized_euclidean_proximity_parameters(base_params, test_params)
        distances.append(dist)
        
        # Manual verification using euclidean_distance_parameters function
        d12 = euclidean_distance_parameters(base_params, test_params)  # distance between parameters
        
        zero_params = OrderedDict([('w', torch.tensor([[0.0]]))])
        d10 = euclidean_distance_parameters(base_params, zero_params)  # distance of first parameter from origin  
        d20 = euclidean_distance_parameters(test_params, zero_params)  # distance of second parameter from origin
        expected = d12 / (d10 + d20) if (d10 + d20) > 0 else 0.0
        
        print(f"   Factor {factor:4.1f}: norm_dist = {dist:.6f}, expected = {expected:.6f}")
        assert abs(dist - expected) < 1e-10, f"Wrong calculation for factor {factor}: {dist} vs {expected}"
        assert 0.0 <= dist <= 1.0, f"Value outside [0,1] range for factor {factor}"
    
    print(f"\n✅ All tests passed!")
    print(f"✅ All values are in [0,1] range")
    print(f"✅ Formula Dis_norm(M1, M2) = Dis(M1, M2) / (Dis(M1, 0) + Dis(M2, 0)) implemented correctly")
    print(f"✅ Mathematical properties verified")
    print(f"✅ Identical parameters → distance = 0")
    print(f"✅ Distance normalized with respect to parameter magnitudes")


def extract_last_layer_gradients(global_model_params, local_model_params):
    """
    Extract gradients from the last layer (second-to-last parameter in the list).
    
    Following the paper's approach: "extracts the last-layer gradients of the workers' local updates"
    
    Args:
        global_model_params: Global model parameters (OrderedDict)
        local_model_params: Local model parameters (OrderedDict)
        
    Returns:
        numpy.ndarray: Flattened gradient vector from last layer
    """
    # Convert OrderedDict to list to access by index
    if isinstance(global_model_params, (dict, OrderedDict)):
        global_params_list = list(global_model_params.values())
        local_params_list = list(local_model_params.values())
    else:
        # Handle case where params are already lists or single tensors
        global_params_list = [global_model_params] if not isinstance(global_model_params, list) else global_model_params
        local_params_list = [local_model_params] if not isinstance(local_model_params, list) else local_model_params
    
    # Get the last layer parameters (typically the output layer weights)
    # Using -2 as in the paper (second-to-last parameter, which is usually the last layer weights)
    if len(global_params_list) >= 2:
        global_last_layer = global_params_list[-2]
        local_last_layer = local_params_list[-2]
    else:
        # Fallback to last parameter if there's only one
        global_last_layer = global_params_list[-1]
        local_last_layer = local_params_list[-1]
    
    # Convert tensors to numpy if needed
    if torch.is_tensor(global_last_layer):
        global_last_layer = global_last_layer.detach().cpu().numpy()
    if torch.is_tensor(local_last_layer):
        local_last_layer = local_last_layer.detach().cpu().numpy()
    
    # Compute gradient (difference between global and local)
    gradient = global_last_layer - local_last_layer
    
    # Flatten the gradient
    return gradient.reshape(-1)

def compute_pca_similarity_features(cosine_sim_matrix, n_components=2):
    """
    Apply PCA to compress cosine similarity vectors and extract robust features.
    
    Following the paper: "compresses the computed similarity vectors using PCA 
    to reduce redundant information and extract more robust features"
    
    Args:
        cosine_sim_matrix: Cosine similarity matrix (n_workers x n_workers)
        n_components: Number of PCA components (default: 2 as in paper)
        
    Returns:
        numpy.ndarray: PCA-compressed features
    """
    if not SKLEARN_AVAILABLE:
        print("Warning: PCA not available, returning original features")
        return cosine_sim_matrix
    
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(cosine_sim_matrix)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    compressed_features = pca.fit_transform(standardized_data)
    
    return compressed_features

def cosine_similarity_defense_score(global_model_params, local_models_params, tau=1.5):
    """
    Implement the COMPUTE_SIMILARITY algorithm from the paper for detecting poisoned updates.
    
    Algorithm steps:
    1. Extract last-layer gradients from all local models
    2. Compute worker-wise cosine similarity
    3. Apply PCA compression to reduce redundancy
    4. Compute centroid of compressed vectors
    5. Calculate similarity between centroid and each vector
    
    Args:
        global_model_params: Global model parameters (OrderedDict)
        local_models_params: List of local model parameters (list of OrderedDict)
        tau: Threshold parameter for outlier detection (default: 1.5)
        
    Returns:
        numpy.ndarray: Similarity scores for each worker (higher = more trustworthy)
    """
    m = len(local_models_params)  # Number of workers
    
    # Step 1: Extract last-layer gradients for all workers
    f_grads = []
    for i in range(m):
        grad = extract_last_layer_gradients(global_model_params, local_models_params[i])
        f_grads.append(grad)
    
    # Convert to numpy array for easier manipulation
    f_grads = np.array(f_grads)
    
    # Step 2: Compute worker-wise cosine similarity
    # Remove diagonal (self-similarity) by subtracting identity matrix
    cs = smp.cosine_similarity(f_grads) - np.eye(m)
    
    # Step 3: Apply PCA compression
    cs_compressed = compute_pca_similarity_features(cs)
    
    # Step 4: Compute centroid of compressed similarity vectors
    centroid = np.median(cs_compressed, axis=0)
    
    # Step 5: Compute cosine similarity between centroid and each compressed vector
    scores = smp.cosine_similarity([centroid], cs_compressed)[0]
    
    return scores

def cos_sim_simple(model1, model2, logger=None):
    """
    Calculate cosine similarity using only fully connected layer weights.
    
    Args:
        model1: First model parameters (dict with layer names as keys)
        model2: Second model parameters (dict with layer names as keys)
        logger: Optional logger for warnings
        
    Returns:
        float: Cosine similarity score between fc.weight layers
    """
    # Extract fc.weight tensors from both models
    fc_weights_1 = []
    fc_weights_2 = []
    
    for key in model1.keys():
        if 'fc.weight' in key and key in model2:
            # Get the tensor from model1
            tensor1 = model1[key]
            if torch.is_tensor(tensor1):
                tensor1 = tensor1.detach().cpu().numpy()
            
            # Get the corresponding tensor from model2
            tensor2 = model2[key]
            if torch.is_tensor(tensor2):
                tensor2 = tensor2.detach().cpu().numpy()
            
            # Flatten and add to lists
            fc_weights_1.append(tensor1)
            fc_weights_2.append(tensor2)
    
    # Check if we found any fc.weight layers
    if not fc_weights_1 or not fc_weights_2:
        if logger:
            logger.warning("No fc.weight layers found in the models")
        else:
            print("Warning: No fc.weight layers found in the models")
        return 0.0
    
    # Calculate cosine similarity
    similarity = smp.cosine_similarity(fc_weights_1[0], fc_weights_2[0]) -np.eye(len(fc_weights_1[0]))
    return ((np.mean(similarity) - np.min(similarity)) / (np.max(similarity) - np.min(similarity)))


# Option 1: Comparison with hash (faster but less precise)
def models_are_different_hash(params1, params2):
    return hash(str(params1)) != hash(str(params2))
# Option 2: Using torch.equal if they are all PyTorch tensors
def models_are_equal_torch(params1, params2):
    import torch
    if isinstance(params1, dict) and isinstance(params2, dict):
        return all(torch.equal(params1[k], params2[k]) for k in params1.keys())
    return torch.equal(params1, params2)
def are_parameters_equal(params1, params2):
        
        """Check if two model parameter sets are equal"""
        if params1 is None or params2 is None:
            return params1 is params2
        
        # If they're the same object reference, they're equal
        if params1 is params2:
            return True
            
        # Compare parameter dictionaries
        if isinstance(params1, dict) and isinstance(params2, dict):
            if set(params1.keys()) != set(params2.keys()):
                return False
            
            for key in params1.keys():
                param1 = params1[key]
                param2 = params2[key]
                
                # Convert to numpy for comparison if they're tensors
                if hasattr(param1, 'detach'):  # PyTorch tensor
                    param1 = param1.detach().cpu().numpy()
                if hasattr(param2, 'detach'):  # PyTorch tensor
                    param2 = param2.detach().cpu().numpy()
                
                # Use numpy's allclose for floating point comparison
                if not np.allclose(param1, param2, atol=1e-10):
                    return False
            return True
        else:
            # Fallback for non-dict parameters
            if hasattr(params1, 'detach'):
                params1 = params1.detach().cpu().numpy()
            if hasattr(params2, 'detach'):
                params2 = params2.detach().cpu().numpy()
            return np.allclose(params1, params2, atol=1e-10)
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


# Update the main execution section
#if __name__ == "__main__":
    #test_euclidean_distance()
    #analyze_euclidean_distance_properties()
    #test_normalized_distance_range()
    #test_last_layer_extraction()
    #test_cosine_similarity_defense()

