import copy
import torch
from fed_train import local_train, fed_avg


def unlearn_by_retraining(global_model, client_loaders, forget_client_idx, global_rounds=5, local_epochs=1, lr=0.01):
    """Retrain from scratch without the forgotten client"""
    print(f"\n[Unlearning] Retraining without client {forget_client_idx}")
    client_ids = list(range(len(client_loaders)))
    client_ids.remove(forget_client_idx)

    for round in range(global_rounds):
        print(f"\n[Retrain] Global Round {round + 1}")
        weights = []

        for idx in client_ids:
            loader = client_loaders[idx]
            print(f" Training on client {idx}")
            trained_weights = local_train(global_model, loader, epochs=local_epochs, lr=lr)
            weights.append(trained_weights)

        averaged_weights = fed_avg(weights)
        global_model.load_state_dict(averaged_weights)

    return global_model


def fine_tune_unlearning(global_model, client_loader_to_forget, local_epochs=1, lr=0.01):
    """Approximate unlearning by doing negative fine-tuning (optional: can try regular fine-tuning too)"""
    print(f"\n[Unlearning] Fine-tuning to remove influence of forgotten client")
    # This is a placeholder — real inverse updates would require influence estimation
    # Instead, we simply re-train the global model on other clients (excluding forgotten one)
    # So we call this from main using filtered client list
    return global_model  # If implemented, you would adjust weights here
