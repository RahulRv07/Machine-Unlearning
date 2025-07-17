import torch
from model import MLP
from data_preprocess import load_breast_cancer_dataset
from fed_train import federated_train
from unlearning import unlearn_by_agnostic_diff
from mia_attack import run_mia_attack
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy

# -------- CONFIGURATION --------
NUM_CLIENTS = 5
GLOBAL_ROUNDS = 5
RETRAIN_ROUNDS = 5
INPUT_SIZE = 30  # Breast cancer dataset has 30 features
HIDDEN_SIZE = 64
NUM_CLASSES = 2

# -------- EVALUATION FUNCTIONS --------
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f"🔍 Accuracy: {acc:.4f}")
    print("📋 Classification Report:")
    print(classification_report(all_labels, all_preds))
    return acc

def get_softmax_outputs(model, data_loader):
    model.eval()
    outputs_list = []
    with torch.no_grad():
        for X_batch, _ in data_loader:
            logits = model(X_batch)
            probs = F.softmax(logits, dim=1)
            outputs_list.append(probs.cpu())
    return torch.cat(outputs_list, dim=0)

def forgetting_score(probs_before, probs_after):
    kl = F.kl_div(probs_after.log(), probs_before, reduction='batchmean')
    mean_drop = (probs_before - probs_after).abs().mean().item()
    return kl.item(), mean_drop

# -------- MAIN PIPELINE --------
if __name__ == "__main__":
    print("🧬 Loading and splitting dataset...")
    client_loaders, test_loader = load_breast_cancer_dataset(num_clients=NUM_CLIENTS)

    print("🧠 Initializing model...")
    global_model = MLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

    # Set which clients to remove (can be a list)
    clients_to_remove = [2, 3]

    # Federated training
    print("\n🤝 Federated training (all clients)...")
    global_model = federated_train(global_model, client_loaders, global_rounds=GLOBAL_ROUNDS)

    # Evaluate BEFORE unlearning
    print("\n🎯 Evaluating model accuracy BEFORE unlearning...")
    acc_before = evaluate_model(global_model, test_loader)

    # MIA before unlearning (average over all removed clients)
    print("\n🔍 Evaluating MIA before unlearning...")
    mia_scores_before = []
    probs_before_list = []
    for client in clients_to_remove:
        mia_score = run_mia_attack(global_model, client_loaders[client], test_loader)
        mia_scores_before.append(mia_score)
        probs_before_list.append(get_softmax_outputs(global_model, client_loaders[client]))

    # Loop over multiple alpha values and collect metrics
    alphas = [0.5, 1.0, 1.5]
    kl_list, mean_drop_list, acc_list, mia_list = [], [], [], []

    for alpha in alphas:
        print(f"\n🧹 Unlearning clients {clients_to_remove} using agnostic method (alpha={alpha})...")
        model_to_unlearn = copy.deepcopy(global_model)
        # Remove all specified clients
        filtered_loaders = [loader for i, loader in enumerate(client_loaders) if i not in clients_to_remove]
        # Use agnostic diff with all removed clients
        model_to_unlearn = unlearn_by_agnostic_diff(model_to_unlearn, client_loaders, clients_to_remove[0], global_rounds=RETRAIN_ROUNDS, alpha=alpha) \
            if len(clients_to_remove) == 1 else \
            unlearn_by_agnostic_diff(model_to_unlearn, client_loaders, clients_to_remove, global_rounds=RETRAIN_ROUNDS, alpha=alpha)

        # Evaluate AFTER unlearning
        print("\n🎯 Evaluating model accuracy AFTER unlearning...")
        acc = evaluate_model(model_to_unlearn, test_loader)
        acc_list.append(acc)

        # MIA after unlearning (average over all removed clients)
        print("\n🔍 Evaluating MIA after unlearning...")
        mia_scores = []
        kl_scores = []
        mean_drops = []
        for idx, client in enumerate(clients_to_remove):
            mia_score = run_mia_attack(model_to_unlearn, client_loaders[client], test_loader)
            mia_scores.append(mia_score)
            probs_after = get_softmax_outputs(model_to_unlearn, client_loaders[client])
            probs_before = probs_before_list[idx]
            if probs_before.shape == probs_after.shape:
                kl, mean_drop = forgetting_score(probs_before, probs_after)
                kl_scores.append(kl)
                mean_drops.append(mean_drop)
                print(f"🧾 Forgetting Score (KL divergence): {kl:.4f}")
                print(f"🧾 Mean Probability Drop: {mean_drop:.4f}")
            else:
                print("⚠️ Could not compute forgetting score: shapes do not match.")
        mia_list.append(sum(mia_scores) / len(mia_scores))
        kl_list.append(sum(kl_scores) / len(kl_scores))
        mean_drop_list.append(sum(mean_drops) / len(mean_drops))

    # -------- VISUALIZATION --------
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(alphas, kl_list, marker='o')
    plt.title('Alpha vs KL Divergence')
    plt.xlabel('Alpha')
    plt.ylabel('KL Divergence')

    plt.subplot(1, 3, 2)
    plt.plot(alphas, acc_list, marker='o')
    plt.title('Alpha vs Accuracy')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')

    plt.subplot(1, 3, 3)
    plt.plot(alphas, mia_list, marker='o')
    plt.title('Alpha vs MIA Score')
    plt.xlabel('Alpha')
    plt.ylabel('MIA Score')

    plt.tight_layout()
    plt.savefig("unlearning_forgetting_analysis.png")
    print("Plot saved as unlearning_forgetting_analysis.png")
    # plt.show()  # Optionally keep this for local runs with GUI
