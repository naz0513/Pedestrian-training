import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from social_lstm_classifier import SocialLSTMClassifier  

# ==== Hyperparameters ====
input_size = 3           
hidden_size = 64
grid_size = (4, 4)
neighborhood_size = 4.0
dropout = 0.1
observed = 15             
epochs = 10
learning_rate = 0.001

# ==== Load Data ====
train_data = torch.load("/Users/anzhunie/Desktop/Pedestrian_Training/Prediction/train_social_lstm_full.pt")
test_data = torch.load("/Users/anzhunie/Desktop/Pedestrian_Training/Prediction/test_social_lstm_full.pt")

# ==== Initialize Model ====
model = SocialLSTMClassifier(
    input_size=input_size,
    hidden_size=hidden_size,
    grid_size=grid_size,
    neighborhood_size=neighborhood_size,
    dropout=dropout,
    observed=observed
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ==== Training Loop ====
model.train()
for epoch in range(epochs):
    total_loss = 0
    for sample in train_data:
        traj = sample['trajectory'].unsqueeze(1)             # [15, 1, 3]
        neighbors = sample['neighbors']                      # [15, N, 3]
        neighbor_mask = sample['neighbor_mask']              # [15, N]

        label = torch.tensor([int(sample['cluster']) - 1], dtype=torch.long)

        optimizer.zero_grad()
        logits = model(traj, neighbors, neighbor_mask)       # [1, 2]
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_data)
    print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f}")

# ==== Evaluation ====
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for sample in test_data:
        traj = sample['trajectory'].unsqueeze(1)
        neighbors = sample['neighbors']
        neighbor_mask = sample['neighbor_mask']

        label = int(sample['cluster']) - 1
        logits = model(traj, neighbors, neighbor_mask)
        pred = logits.argmax(dim=1).item()

        y_true.append(label)
        y_pred.append(pred)

# ==== Metrics ====
acc = accuracy_score(y_true, y_pred)
print(f" Test Accuracy: {acc:.4f}")
print(" Classification Report:")
print(classification_report(y_true, y_pred, digits=3))

# ==== Save Model ====
torch.save(model, "/Users/anzhunie/Desktop/Pedestrian_Training/Prediction/social_lstm_classifier_full.pth")
print("Model saved as social_lstm_classifier_full.pth")
