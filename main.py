import torch
from model import BreastCancerNet
from data_preprocess import load_breast_cancer_dataset
from fed_train import federated_train
from unlearning import unlearn_by_retraining
from mia_attack import run_mia_attack
from torch.utils.data import DataLoader, Subset

# Setup
NUM_CLIENTS = 5
FORGET_CLIENT = 2
GLOBAL_ROUNDS = 5
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.01

# Step 1: Load and prepare data
print("🧬 Loading and splitting dataset...")
client_loaders, test_loader = load_breast_cancer_dataset(num_clients=NUM_CLIENTS)

# Step 2: Initialize global model
print("🧠 Initializing model...")
global_model = BreastCancerNet()
global_model = federated_train(global_model, client_loaders, global_rounds=GLOBAL_ROUNDS,
                                local_epochs=LOCAL_EPOCHS, lr=LEARNING_RATE)

# Step 3: Evaluate MIA before unlearning
print("\n🔍 Evaluating MIA before unlearning...")
member_loader = client_loaders[FORGET_CLIENT]
non_member_loader = test_loader
run_mia_attack(global_model, member_loader, non_member_loader)

# Step 4: Unlearn client
print(f"\n🧹 Unlearning client {FORGET_CLIENT} by retraining...")
new_model = BreastCancerNet()
new_model = unlearn_by_retraining(new_model, client_loaders, forget_client_idx=FORGET_CLIENT,
                                   global_rounds=GLOBAL_ROUNDS, local_epochs=LOCAL_EPOCHS, lr=LEARNING_RATE)

# Step 5: Evaluate MIA after unlearning
print("\n🔍 Evaluating MIA after unlearning...")
run_mia_attack(new_model, member_loader, non_member_loader)
