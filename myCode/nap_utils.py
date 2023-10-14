

def prepare_NAP_extraction(model):
    nodes_to_extract = get_nodes_to_extract(model=model, activation_name='relu')

    if cfg['architecture'] == 'MNIST_net':
        nodes_to_extract = [nodes_to_extract[2]]
    else:
        raise ValueError(f"specify nodes to extract for {cfg['architecture']}")

    model_feature_extractor = create_feature_extractor(model=model, return_nodes=nodes_to_extract)
    return model_feature_extractor


def get_nodes_to_extract(model, activation_name):
    nodes = get_graph_node_names(model)[1] # 1 mean model.eval layers

    nodes_to_extract = []

    for node in nodes:
        if activation_name in node:
            nodes_to_extract.append(node)

    return nodes_to_extract


def extract_activations(x , model_feature_extractor):
    extracted_features = model_feature_extractor(x)

    layers = list(extracted_features.keys())

    activations = extracted_features[layers[0]].flatten(1).cpu().detach().numpy()
    n_activation = activations.shape[0]
    for layer in layers[1:]:
        activations = np.concatenate((activations, extracted_features[layer].flatten(1).cpu().detach().numpy()), axis=1)

    return activations.sum(0), n_activation