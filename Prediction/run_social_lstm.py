from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from social_lstm_classifier import SocialLSTMClassifier

trajectory_features = ['Positionx', 'Positiony', 'Distance', 'Speed', 'Speed Change', 'Direction', 'Direction Change']

feature_combos = {
    'xy_2': ['Positionx', 'Positiony'],
    'xyd_3': ['Positionx', 'Positiony', 'Distance'],
    'xydsdc_7': ['Positionx', 'Positiony', 'Distance', 'Speed', 'Direction', 'Speed Change', 'Direction Change'],
    'xysd_4': ['Positionx', 'Positiony', 'Speed', 'Direction'],
    'xyds_4': ['Positionx', 'Positiony', 'Distance', 'Speed'],
    'xydd_4': ['Positionx', 'Positiony', 'Distance', 'Direction'],
    'xydsc_5': ['Positionx', 'Positiony', 'Distance', 'Speed', 'Speed Change'],
    'xyddc_5': ['Positionx', 'Positiony', 'Distance', 'Direction', 'Direction Change'],
    'xys_3': ['Positionx', 'Positiony', 'Speed'],
    'xydi_3': ['Positionx', 'Positiony', 'Direction'],
    'xysc_4': ['Positionx', 'Positiony', 'Speed', 'Speed Change'],
    'xydc_4': ['Positionx', 'Positiony', 'Direction', 'Direction Change']
}

train_data = torch.load("/Users/anzhunie/Desktop/Pedestrian_Training/Prediction/train_social_lstm_full.pt")
test_data = torch.load("/Users/anzhunie/Desktop/Pedestrian_Training/Prediction/test_social_lstm_full.pt")

results = []

def run_experiment(feature_set, name):
    input_size = len(feature_set)
    index_map = [trajectory_features.index(f) for f in feature_set]

    model = SocialLSTMClassifier(input_size=input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 50

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for sample in tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}"):
            traj = sample['trajectory'][:, index_map].unsqueeze(1)
            neighbors = sample['neighbors'][:, :, index_map]
            mask = sample['neighbor_mask']
            label = torch.tensor([int(sample['cluster']) - 1], dtype=torch.long)

            optimizer.zero_grad()
            logits = model(traj, neighbors, mask)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        print(f"\n[Feature Set: {name}] Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f}")


    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
       for sample in tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}"):
            traj = sample['trajectory'][:, index_map].unsqueeze(1)
            neighbors = sample['neighbors'][:, :, index_map]
            mask = sample['neighbor_mask']
            label = int(sample['cluster']) - 1
            logits = model(traj, neighbors, mask)
            pred = logits.argmax(dim=1).item()
            y_true.append(label)
            y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, output_dict=True)
    results.append({
        "Feature_Set": name,
        "Accuracy": acc,
        "Macro_F1": clf_report['macro avg']['f1-score'],
        "Weighted_F1": clf_report['weighted avg']['f1-score']
    })

for name, feats in feature_combos.items():
    run_experiment(feats, name)

df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv("/Users/anzhunie/Desktop/Pedestrian_Training/Prediction/feature_combination_results.csv", index=False)
