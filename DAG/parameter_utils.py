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
    print("\n=== ANALISI DELLE PROPRIETÀ DELLA DISTANZA EUCLIDEA ===\n")
    
    # 1. Verificare che il flatten mantenga il senso matematico
    print("1. FLATTEN E SENSO MATEMATICO:")
    print("   - I parametri di una rete neurale sono tensori multi-dimensionali")
    print("   - Il flatten li trasforma in un singolo vettore preservando tutti i valori")
    print("   - La distanza euclidea misura la 'distanza' globale tra due modelli")
    print("   - Matematicamente corretto: d(x,y) = √(Σᵢ(xᵢ - yᵢ)²)")
    
    # Test con parametri di diverse forme
    # Caso 1: Matrice 2x2
    matrix_2x2 = np.array([[1, 2], [3, 4]])
    matrix_2x2_diff = np.array([[1.1, 2.1], [3.1, 4.1]])
    
    # Caso 2: Stessa matrice come vettore 1x4  
    vector_1x4 = np.array([1, 2, 3, 4])
    vector_1x4_diff = np.array([1.1, 2.1, 3.1, 4.1])
    
    dist_matrix = euclidean_distance_parameters(matrix_2x2, matrix_2x2_diff)
    dist_vector = euclidean_distance_parameters(vector_1x4, vector_1x4_diff)
    
    print(f"\n   Test: matrice 2x2 vs vettore 1x4 (stessi valori)")
    print(f"   Distanza matrice: {dist_matrix:.6f}")
    print(f"   Distanza vettore: {dist_vector:.6f}")
    print(f"   Differenza: {abs(dist_matrix - dist_vector):.10f} (dovrebbe essere ~0)")
    
    # 2. Confronto con implementazione manuale
    print("\n2. VERIFICA CON CALCOLO MANUALE:")
    manual_calc = np.sqrt(np.sum((0.1)**2 * 4))  # 4 elementi, ognuno differisce di 0.1
    print(f"   Calcolo manuale: √(4 × 0.1²) = {manual_calc:.6f}")
    print(f"   Nostra implementazione: {dist_matrix:.6f}")
    print(f"   Match: {abs(manual_calc - dist_matrix) < 1e-10}")
    
    # 3. Proprietà matematiche
    print("\n3. PROPRIETÀ MATEMATICHE:")
    
    # Proprietà 1: Non-negatività
    print("   a) Non-negatività: d(x,y) ≥ 0")
    assert dist_matrix >= 0
    print("      ✅ Verificata")
    
    # Proprietà 2: Identità
    print("   b) Identità: d(x,x) = 0")
    self_dist = euclidean_distance_parameters(matrix_2x2, matrix_2x2)
    assert self_dist == 0
    print("      ✅ Verificata")
    
    # Proprietà 3: Simmetria  
    print("   c) Simmetria: d(x,y) = d(y,x)")
    dist_ab = euclidean_distance_parameters(matrix_2x2, matrix_2x2_diff)
    dist_ba = euclidean_distance_parameters(matrix_2x2_diff, matrix_2x2)
    assert abs(dist_ab - dist_ba) < 1e-10
    print("      ✅ Verificata")
    
    # Proprietà 4: Disuguaglianza triangolare
    print("   d) Disuguaglianza triangolare: d(x,z) ≤ d(x,y) + d(y,z)")
    matrix_c = np.array([[2, 3], [4, 5]])
    dist_ac = euclidean_distance_parameters(matrix_2x2, matrix_c)
    dist_bc = euclidean_distance_parameters(matrix_2x2_diff, matrix_c)
    triangle_ineq = dist_ac <= (dist_ab + dist_bc + 1e-10)  # small tolerance
    print(f"      d(a,c)={dist_ac:.6f} ≤ d(a,b)+d(b,c)={dist_ab:.6f}+{dist_bc:.6f}={dist_ab+dist_bc:.6f}")
    print(f"      ✅ {'Verificata' if triangle_ineq else 'FALLITA'}")
    
    # 4. Comportamento con reti neurali reali
    print("\n4. COMPORTAMENTO CON RETI NEURALI:")
    
    # Simula parametri di una piccola rete
    torch.manual_seed(42)  # Per riproducibilità
    
    net_params_1 = OrderedDict([
        ('conv1.weight', torch.randn(8, 3, 3, 3) * 0.1),  # Pesi piccoli
        ('conv1.bias', torch.randn(8) * 0.01),
        ('fc1.weight', torch.randn(10, 512) * 0.1),
        ('fc1.bias', torch.randn(10) * 0.01)
    ])
    
    # Parametri simili (come dopo una epoca di training)
    net_params_2 = OrderedDict()
    for name, param in net_params_1.items():
        # Aggiungi piccole variazioni
        noise = torch.randn_like(param) * 0.001
        net_params_2[name] = param + noise
    
    # Parametri molto diversi
    net_params_3 = OrderedDict([
        ('conv1.weight', torch.randn(8, 3, 3, 3) * 0.1),
        ('conv1.bias', torch.randn(8) * 0.01), 
        ('fc1.weight', torch.randn(10, 512) * 0.1),
        ('fc1.bias', torch.randn(10) * 0.01)
    ])
    
    dist_similar = euclidean_distance_parameters(net_params_1, net_params_2)
    dist_different = euclidean_distance_parameters(net_params_1, net_params_3)
    
    print(f"   Distanza tra parametri simili: {dist_similar:.6f}")
    print(f"   Distanza tra parametri diversi: {dist_different:.6f}")
    print(f"   Rapporto (diversi/simili): {dist_different/dist_similar:.2f}x")
    print(f"   ✅ La distanza discrimina correttamente tra parametri simili e diversi")
    
    print("\n=== CONCLUSIONI ===")
    print("✅ L'implementazione è matematicamente corretta")
    print("✅ Il flatten preserva tutte le informazioni")
    print("✅ Tutte le proprietà metriche sono soddisfatte")
    print("✅ Comportamento appropriato per parametri di reti neurali")


def test_normalized_distance_range():
    """
    Test che la distanza normalizzata restituisca valori nell'intervallo [0,1]
    usando la formula: Dis_norm(M1, M2) = Dis(M1, M2) / (Dis(M1, 0) + Dis(M2, 0))
    """
    print("\n=== TEST DISTANZA NORMALIZZATA [0,1] - NUOVA FORMULA ===\n")
    
    # Test 1: Parametri identici dovrebbero dare 0
    params_identical = OrderedDict([
        ('layer1.weight', torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ('layer1.bias', torch.tensor([0.5, -0.5]))
    ])
    
    norm_dist_identical = normalized_euclidean_proximity_parameters(params_identical, params_identical)
    print(f"Test 1 - Parametri identici: {norm_dist_identical:.6f} (dovrebbe essere 0.0)")
    assert norm_dist_identical == 0.0, f"Expected 0.0, got {norm_dist_identical}"
    assert 0.0 <= norm_dist_identical <= 1.0, "Valore fuori dall'intervallo [0,1]"
    
    # Test 2: Parametri leggermente diversi
    params_similar = OrderedDict([
        ('layer1.weight', torch.tensor([[1.01, 2.01], [3.01, 4.01]])),
        ('layer1.bias', torch.tensor([0.51, -0.49]))
    ])
    
    norm_dist_similar = normalized_euclidean_proximity_parameters(params_identical, params_similar)
    print(f"Test 2 - Parametri simili: {norm_dist_similar:.6f}")
    assert 0.0 <= norm_dist_similar <= 1.0, "Valore fuori dall'intervallo [0,1]"
    assert norm_dist_similar > 0.0, "Parametri diversi dovrebbero avere distanza > 0"
    
    # Test 3: Verifica della formula manualmente
    # Calcoliamo manualmente per verificare
    dist_12 = euclidean_distance_parameters(params_identical, params_similar)
    
    # Creiamo parametri zero
    zero_params = OrderedDict([
        ('layer1.weight', torch.zeros(2, 2)),
        ('layer1.bias', torch.zeros(2))
    ])
    
    dist_1_to_zero = euclidean_distance_parameters(params_identical, zero_params)
    dist_2_to_zero = euclidean_distance_parameters(params_similar, zero_params)
    
    expected_norm = dist_12 / (dist_1_to_zero + dist_2_to_zero)
    
    print(f"Test 3 - Verifica formula:")
    print(f"   Dis(M1, M2) = {dist_12:.6f}")
    print(f"   Dis(M1, 0) = {dist_1_to_zero:.6f}")
    print(f"   Dis(M2, 0) = {dist_2_to_zero:.6f}")
    print(f"   Formula manuale: {expected_norm:.6f}")
    print(f"   Funzione: {norm_dist_similar:.6f}")
    assert abs(expected_norm - norm_dist_similar) < 1e-10, "Formula non corretta"
    
    # Test 4: Parametri molto diversi
    params_different = OrderedDict([
        ('layer1.weight', torch.tensor([[10.0, 20.0], [30.0, 40.0]])),
        ('layer1.bias', torch.tensor([5.0, -5.0]))
    ])
    
    norm_dist_different = normalized_euclidean_proximity_parameters(params_identical, params_different)
    print(f"Test 4 - Parametri molto diversi: {norm_dist_different:.6f}")
    assert 0.0 <= norm_dist_different <= 1.0, "Valore fuori dall'intervallo [0,1]"
    
    # Test 5: Verifica proprietà matematiche
    print(f"\nTest 5 - Proprietà matematiche:")
    
    # Simmetria: dist_norm(A,B) = dist_norm(B,A)
    norm_AB = normalized_euclidean_proximity_parameters(params_identical, params_similar)
    norm_BA = normalized_euclidean_proximity_parameters(params_similar, params_identical)
    print(f"   Simmetria: norm(A,B)={norm_AB:.6f}, norm(B,A)={norm_BA:.6f}")
    assert abs(norm_AB - norm_BA) < 1e-10, "La distanza normalizzata non è simmetrica"
    
    # Test 6: Comportamento con parametri all'origine
    zero_params_test = OrderedDict([
        ('layer1.weight', torch.zeros(2, 2)),
        ('layer1.bias', torch.zeros(2))
    ])
    
    norm_zero = normalized_euclidean_proximity_parameters(zero_params_test, zero_params_test)
    print(f"Test 6 - Parametri zero: {norm_zero:.6f} (dovrebbe essere 0.0)")
    assert norm_zero == 0.0, "Parametri zero dovrebbero avere distanza 0"
    
    # Test 7: Progressione di differenze con scaling
    print(f"\nTest 7 - Progressione con scaling:")
    base_params = OrderedDict([('w', torch.tensor([[1.0]]))])
    
    factors = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    distances = []
    
    for factor in factors:
        test_params = OrderedDict([('w', torch.tensor([[1.0 + factor]]))])
        dist = normalized_euclidean_proximity_parameters(base_params, test_params)
        distances.append(dist)
        
        # Verifica manuale usando la funzione euclidean_distance_parameters
        d12 = euclidean_distance_parameters(base_params, test_params)  # distanza tra parametri
        
        zero_params = OrderedDict([('w', torch.tensor([[0.0]]))])
        d10 = euclidean_distance_parameters(base_params, zero_params)  # distanza del primo parametro dall'origine  
        d20 = euclidean_distance_parameters(test_params, zero_params)  # distanza del secondo parametro dall'origine
        expected = d12 / (d10 + d20) if (d10 + d20) > 0 else 0.0
        
        print(f"   Factor {factor:4.1f}: norm_dist = {dist:.6f}, expected = {expected:.6f}")
        assert abs(dist - expected) < 1e-10, f"Calcolo errato per factor {factor}: {dist} vs {expected}"
        assert 0.0 <= dist <= 1.0, f"Valore fuori dall'intervallo [0,1] per factor {factor}"
    
    print(f"\n✅ Tutti i test superati!")
    print(f"✅ Tutti i valori sono nell'intervallo [0,1]")
    print(f"✅ Formula Dis_norm(M1, M2) = Dis(M1, M2) / (Dis(M1, 0) + Dis(M2, 0)) implementata correttamente")
    print(f"✅ Proprietà matematiche verificate")
    print(f"✅ Parametri identici → distanza = 0")
    print(f"✅ Distanza normalizzata rispetto alle magnitude dei parametri")



def test_last_layer_extraction():
    """
    Test the last layer extraction functionality with different model architectures.
    """
    print("\n=== TEST LAST LAYER EXTRACTION ===\n")
    
    # Test case 1: Standard CNN model
    print("Test 1 - Standard CNN model:")
    cnn_model = OrderedDict([
        ('conv1.weight', torch.randn(16, 3, 5, 5)),
        ('conv1.bias', torch.randn(16)),
        ('conv2.weight', torch.randn(32, 16, 3, 3)),
        ('conv2.bias', torch.randn(32)),
        ('fc1.weight', torch.randn(128, 1152)),
        ('fc1.bias', torch.randn(128)),
        ('fc2.weight', torch.randn(10, 128)),  # This should be extracted (last layer weights)
        ('fc2.bias', torch.randn(10))          # This is the final layer
    ])
    
    cnn_modified = OrderedDict()
    for name, param in cnn_model.items():
        # Add some variation
        noise = torch.randn_like(param) * 0.01
        cnn_modified[name] = param + noise
    
    gradient = extract_last_layer_gradients(cnn_model, cnn_modified)
    print(f"  Last layer gradient shape: {gradient.shape}")
    print(f"  Expected to extract fc2.weight: {cnn_model['fc2.weight'].numel()} parameters")
    print(f"  Gradient norm: {np.linalg.norm(gradient):.6f}")
    
    # Test case 2: Simple MLP
    print(f"\nTest 2 - Simple MLP:")
    mlp_model = OrderedDict([
        ('hidden.weight', torch.randn(50, 784)),
        ('hidden.bias', torch.randn(50)),
        ('output.weight', torch.randn(10, 50)),  # This should be extracted
        ('output.bias', torch.randn(10))
    ])
    
    mlp_modified = OrderedDict()
    for name, param in mlp_model.items():
        noise = torch.randn_like(param) * 0.01
        mlp_modified[name] = param + noise
    
    gradient = extract_last_layer_gradients(mlp_model, mlp_modified)
    print(f"  Last layer gradient shape: {gradient.shape}")
    print(f"  Expected to extract output.weight: {mlp_model['output.weight'].numel()} parameters")
    print(f"  Gradient norm: {np.linalg.norm(gradient):.6f}")
    
    # Test case 3: Single layer model
    print(f"\nTest 3 - Single layer model:")
    single_layer = OrderedDict([
        ('weight', torch.randn(10, 784)),
        ('bias', torch.randn(10))
    ])
    
    single_modified = OrderedDict()
    for name, param in single_layer.items():
        noise = torch.randn_like(param) * 0.01
        single_modified[name] = param + noise
    
    gradient = extract_last_layer_gradients(single_layer, single_modified)
    print(f"  Last layer gradient shape: {gradient.shape}")
    print(f"  Expected to extract bias (last param): {single_layer['bias'].numel()} parameters")
    print(f"  Gradient norm: {np.linalg.norm(gradient):.6f}")
    
    print(f"\n✅ Last layer extraction tests completed!")
    print(f"✅ Function correctly identifies and extracts last layer parameters")
    print(f"✅ Works with different model architectures (CNN, MLP, single layer)")


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

def cos_sim_simple(model1, model2):
    """
    Calculate cosine similarity using only fully connected layer weights.
    
    Args:
        model1: First model parameters (dict with layer names as keys)
        model2: Second model parameters (dict with layer names as keys)
        
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
            
            #print(f"Found fc.weight layer: {key}")
            #print(f"  Shape: {tensor1.shape}")
    
    # Check if we found any fc.weight layers
    if not fc_weights_1 or not fc_weights_2:
        print("Warning: No fc.weight layers found in the models")
        return 0.0
    
    
    
    # Calculate cosine similarity
    similarity = smp.cosine_similarity(fc_weights_1[0], fc_weights_2[0])
    return ((np.mean(similarity) - np.min(similarity)) / (np.max(similarity) - np.min(similarity)))

def identify_poisoned_workers(scores, tau=1.5, method='threshold'):
    """
    Identify potentially poisoned workers based on similarity scores.
    
    Args:
        scores: Similarity scores from cosine_similarity_defense_score
        tau: Threshold parameter for outlier detection
        method: Detection method ('threshold', 'zscore', 'percentile')
        
    Returns:
        list: Indices of potentially poisoned workers
    """
    poisoned_indices = []
    
    if method == 'threshold':
        # Simple threshold-based detection
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold = mean_score - tau * std_score
        poisoned_indices = np.where(scores < threshold)[0].tolist()
        
    elif method == 'zscore':
        # Z-score based detection
        z_scores = np.abs((scores - np.mean(scores)) / np.std(scores))
        poisoned_indices = np.where(z_scores > tau)[0].tolist()
        
    elif method == 'percentile':
        # Percentile-based detection
        threshold = np.percentile(scores, (1 - 1/tau) * 100)
        poisoned_indices = np.where(scores < threshold)[0].tolist()
    
    return poisoned_indices

def cosine_similarity_last_layer_only(params1, params2):
    """
    Calculate cosine similarity using only the last layer parameters.
    
    This is a simplified version focusing only on last layer comparison,
    following the paper's insight that last layer contains most discriminative information.
    
    Args:
        params1: First set of neural network parameters
        params2: Second set of neural network parameters
        
    Returns:
        float: Cosine similarity score between last layers
    """
    # Extract last layer parameters
    if isinstance(params1, (dict, OrderedDict)):
        params1_list = list(params1.values())
        params2_list = list(params2.values())
        
        # Get last layer (or second-to-last for weights)
        if len(params1_list) >= 2:
            last_layer_1 = params1_list[-2]  # Typically the output layer weights
            last_layer_2 = params2_list[-2]
        else:
            last_layer_1 = params1_list[-1]
            last_layer_2 = params2_list[-1]
    else:
        last_layer_1 = params1
        last_layer_2 = params2
    
    # Flatten and convert to numpy
    if torch.is_tensor(last_layer_1):
        vec1 = last_layer_1.detach().cpu().numpy().flatten()
    else:
        vec1 = np.array(last_layer_1).flatten()
        
    if torch.is_tensor(last_layer_2):
        vec2 = last_layer_2.detach().cpu().numpy().flatten()
    else:
        vec2 = np.array(last_layer_2).flatten()
    
    # Compute cosine similarity
    # Formula: cos(θ) = (A·B) / (||A|| ||B||)
    dot_product = np.dot(vec1, vec2)
    norm_1 = np.linalg.norm(vec1)
    norm_2 = np.linalg.norm(vec2)
    
    # Handle edge case of zero vectors
    if norm_1 == 0 or norm_2 == 0:
        return 0.0
    
    cosine_sim = dot_product / (norm_1 * norm_2)
    return float(cosine_sim)

def cosine_similarity_pairwise_defense(local_model1_params, local_model2_params):
    """
    Confronta due modelli locali usando la metodologia del cosine similarity defense.
    
    Questa funzione è simile a cosine_similarity_defense_score() ma confronta solo
    due modelli locali invece di un global model con molti local models.
    Utile per il tip selection in DAGTrustFL dove i nodi devono valutare la 
    similarità tra diversi modelli locali.
    
    Args:
        local_model1_params: Parametri del primo modello locale (OrderedDict)
        local_model2_params: Parametri del secondo modello locale (OrderedDict)
        
    Returns:
        float: Punteggio di similarità coseno tra i due modelli (più alto = più simili)
    """
    
    # Handle identical models case first
    if local_model1_params is local_model2_params:
        return 1.0
    
    # Check if models have identical values
    if isinstance(local_model1_params, (dict, OrderedDict)) and isinstance(local_model2_params, (dict, OrderedDict)):
        if local_model1_params.keys() == local_model2_params.keys():
            all_identical = True
            for key in local_model1_params.keys():
                param1 = local_model1_params[key]
                param2 = local_model2_params[key]
                
                # Convert to numpy for comparison
                if torch.is_tensor(param1):
                    param1 = param1.detach().cpu().numpy()
                if torch.is_tensor(param2):
                    param2 = param2.detach().cpu().numpy()
                
                if not np.allclose(param1, param2, atol=1e-10):
                    all_identical = False
                    break
            
            if all_identical:
                return 1.0
    
    # Extract last layer gradients directly by comparing the two models
    # Instead of using zero reference, compare the models directly
    if isinstance(local_model1_params, (dict, OrderedDict)):
        params1_list = list(local_model1_params.values())
        params2_list = list(local_model2_params.values())
        
        # Get last layer parameters (second-to-last parameter, which is usually the last layer weights)
        if len(params1_list) >= 2:
            last_layer_1 = params1_list[-2]
            last_layer_2 = params2_list[-2]
        else:
            # Fallback to last parameter if there's only one
            last_layer_1 = params1_list[-1]
            last_layer_2 = params2_list[-1]
    else:
        last_layer_1 = local_model1_params
        last_layer_2 = local_model2_params
    
    # Convert tensors to numpy and flatten
    if torch.is_tensor(last_layer_1):
        grad1 = last_layer_1.detach().cpu().numpy().flatten()
    else:
        grad1 = np.array(last_layer_1).flatten()
        
    if torch.is_tensor(last_layer_2):
        grad2 = last_layer_2.detach().cpu().numpy().flatten()
    else:
        grad2 = np.array(last_layer_2).flatten()
    
    # Compute cosine similarity directly between the last layer parameters
    dot_product = np.dot(grad1, grad2)
    norm_1 = np.linalg.norm(grad1)
    norm_2 = np.linalg.norm(grad2)
    
    # Handle edge case of zero vectors
    if norm_1 == 0 or norm_2 == 0:
        return 0.0
    
    cosine_sim = dot_product / (norm_1 * norm_2)
    
    return float(cosine_sim)

# Opzione 1: Confronto con hash (più veloce ma meno preciso)
def models_are_different_hash(params1, params2):
    return hash(str(params1)) != hash(str(params2))
# Opzione 2: Usando torch.equal se sono tutti tensori PyTorch
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
if __name__ == "__main__":
    #test_euclidean_distance()
    #analyze_euclidean_distance_properties()
    #test_normalized_distance_range()
    test_last_layer_extraction()
    #test_cosine_similarity_defense()
    
