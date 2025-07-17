# unlearning.py
import copy
import torch
from fed_train import federated_train

def get_model_weights(model):
    return {k: v.clone().detach() for k, v in model.state_dict().items()}

def set_model_weights(model, weights):
    model.load_state_dict(weights)
    return model

def average_weights(weights1, weights2, alpha=1.0):
    # Weighted average: W_new = W1 - alpha * (W1 - W2)
    return {k: weights1[k] - alpha * (weights1[k] - weights2[k]) for k in weights1}

def unlearn_by_agnostic_diff(original_model, client_loaders, client_to_remove, global_rounds=5, alpha=1.0):
    print(f"\n[Unlearning] Creating a shadow model without client {client_to_remove}...")

    # Copy model for shadow training
    shadow_model = copy.deepcopy(original_model)

    # Remove client
    filtered_clients = [loader for i, loader in enumerate(client_loaders) if i != client_to_remove]

    # Train the shadow model without the client to be unlearned
    shadow_model = federated_train(shadow_model, filtered_clients, global_rounds=global_rounds)

    # Extract weights
    original_weights = get_model_weights(original_model)
    shadow_weights = get_model_weights(shadow_model)

    # Perform agnostic update
    agnostic_weights = average_weights(original_weights, shadow_weights, alpha)

    # Load new weights into original model
    updated_model = set_model_weights(original_model, agnostic_weights)
    print("[Unlearning] Agnostic unlearning completed using weight difference.")

    return updated_model
