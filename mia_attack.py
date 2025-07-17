import torch
import torch.nn.functional as F


def run_mia_attack(model, member_loader, non_member_loader):
    model.eval()

    member_confidences = []
    non_member_confidences = []

    # Collect confidence scores for member data
    with torch.no_grad():
        for X, y in member_loader:
            outputs = model(X)
            probs = F.softmax(outputs, dim=1)
            confidences = probs[range(len(y)), y]  # True class confidence
            member_confidences.extend(confidences.tolist())

    # Collect confidence scores for non-member data
    with torch.no_grad():
        for X, y in non_member_loader:
            outputs = model(X)
            probs = F.softmax(outputs, dim=1)
            confidences = probs[range(len(y)), y]
            non_member_confidences.extend(confidences.tolist())

    # Simple threshold-based attack: guess 'member' if confidence > threshold
    threshold = 0.5
    member_preds = [1 if c > threshold else 0 for c in member_confidences]
    non_member_preds = [1 if c > threshold else 0 for c in non_member_confidences]

    member_labels = [1] * len(member_preds)
    non_member_labels = [0] * len(non_member_preds)

    all_preds = member_preds + non_member_preds
    all_labels = member_labels + non_member_labels

    # Calculate attack accuracy
    correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
    accuracy = correct / len(all_labels)

    print(f"\n🔐 MIA Attack Accuracy: {accuracy:.4f}")
    return accuracy
