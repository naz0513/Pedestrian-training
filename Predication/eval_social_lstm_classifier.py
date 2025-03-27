from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

def evaluate_classifier(model, test_data):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for sample in test_data:
            traj = sample['trajectory'].unsqueeze(0)  # [1, 15, D]
            neighbors = torch.stack([
                sample['up'], sample['right'],
                sample['down'], sample['left']
            ], dim=-1).unsqueeze(0)  # [1, 15, 4]

            label = sample['cluster']
            logits = model(traj, neighbors)
            pred = logits.argmax(dim=1).item()

            y_true.append(label)
            y_pred.append(pred)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=3))
    print(" Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
