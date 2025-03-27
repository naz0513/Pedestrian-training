import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from social_lstm_classifier import SocialLSTMClassifier  

# ==== Hyperparameters ====
input_size = 7           
hidden_size = 64
grid_size = (4, 4)
neighborhood_size = 4.0
dropout = 0.1
observed = 15             
epochs = 10
learning_rate = 0.001

# ==== Load Data ====
train_data = torch.load("train_social_lstm.pt")
test_data = torch.load("test_social_lstm.pt")

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
        traj = sample['trajectory'].unsqueeze(1)         # [15, 1, 7]
        neighbors = torch.stack([
            sample['up'], sample['right'],
            sample['down'], sample['left']
        ], dim=-1).unsqueeze(1)                          # [15, 1, 4]

        label = torch.tensor([sample['cluster']], dtype=torch.long)  # shape [1]

        optimizer.zero_grad()
        logits = model(traj, neighbors)                  # [1, 2]
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
        neighbors = torch.stack([
            sample['up'], sample['right'],
            sample['down'], sample['left']
        ], dim=-1).unsqueeze(1)

        label = sample['cluster']
        logits = model(traj, neighbors)
        pred = logits.argmax(dim=1).item()

        y_true.append(label)
        y_pred.append(pred)

# ==== Metrics ====
acc = accuracy_score(y_true, y_pred)
print(f" Test Accuracy: {acc:.4f}")
print(" Classification Report:")
print(classification_report(y_true, y_pred, digits=3))

# ==== Save Model ====
torch.save(model.state_dict(), "social_lstm_classifier.pth")
print("Model saved as social_lstm_classifier.pth")
