import copy
import torch
import torch.nn as nn
import torch.optim as optim


def local_train(model, train_loader, epochs=1, lr=0.01):
    model = copy.deepcopy(model)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def fed_avg(models_state_dicts):
    """Averaging model weights (FedAvg)"""
    avg_weights = copy.deepcopy(models_state_dicts[0])

    for key in avg_weights.keys():
        for i in range(1, len(models_state_dicts)):
            avg_weights[key] += models_state_dicts[i][key]
        avg_weights[key] = avg_weights[key] / len(models_state_dicts)

    return avg_weights


def federated_train(global_model, client_loaders, global_rounds=5, local_epochs=1, lr=0.01):
    """Performs federated training using FedAvg"""
    for round in range(global_rounds):
        print(f"\n--- Global Round {round + 1} ---")
        client_weights = []

        for idx, loader in enumerate(client_loaders):
            print(f" Training on client {idx}")
            client_model_weights = local_train(global_model, loader, epochs=local_epochs, lr=lr)
            client_weights.append(client_model_weights)

        # Aggregate weights
        averaged_weights = fed_avg(client_weights)
        global_model.load_state_dict(averaged_weights)

    return global_model
