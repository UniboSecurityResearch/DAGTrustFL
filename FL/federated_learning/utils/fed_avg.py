def average_nn_parameters(parameters, trust_vector):
    """
    Averages passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    #GIACOMO: MODIFIED TO BE A WEIGHTED AVERAGE BASED ON TRUST VECTOR
    new_params = {}
    for name in parameters[0].keys():
        new_params[name] = sum([param[name].data * trust_vector[i] for i, param in enumerate(parameters)]) / sum(trust_vector)

    return new_params
