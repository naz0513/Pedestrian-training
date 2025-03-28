from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

def evaluate_classifier(model, test_data):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for sample in test_data:
            traj = sample['trajectory'].unsqueeze(1)          # [15, 1, 3]
            neighbors = sample['neighbors']                   # [15, N, 3]
            neighbor_mask = sample['neighbor_mask']           # [15, N]
            label = int(sample['cluster']) - 1

            # Forward pass with mask
            logits = model(traj, neighbors, neighbor_mask)    # Output: [1, 2]
            pred = logits.argmax(dim=1).item()

            y_true.append(label)
            y_pred.append(pred)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
