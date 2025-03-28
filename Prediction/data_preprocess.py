import pandas as pd
import torch
from sklearn.model_selection import train_test_split

df = pd.read_csv("./entire_dataset_with_cluster_masked.csv")
df = df.dropna(subset=['ID'])  

traj_length = 15
grouped = df.groupby(['ID', 'Trajectory'])
trajectory_features = ['Positionx', 'Positiony', 'Distance', 
                       'Speed', 'Speed Change', 'Direction', 'Direction Change']

data = []
total_segments = 0

for (pid, traj_id), group in grouped:
    group = group.sort_values(by='Time')

    if group.shape[0] < traj_length:
        continue
    

    num_segments = len(group) - traj_length + 1
    total_segments += num_segments
    print(f"ID {pid}, Trajectory {traj_id}: {num_segments} segments")

print(f"TOTAL SEGMENTS GENERATED: {total_segments}")


data = []
total_segments = 0

for (pid, traj_id), group in grouped:
    group = group.sort_values(by='Time')

    if group.shape[0] < traj_length:
        continue

    cluster = int(group['Cluster'].iloc[0])

    traj_np = group[trajectory_features].values
    up_np    = group['Up'].values
    right_np = group['Right'].values
    down_np  = group['Down'].values
    left_np  = group['Left'].values

    for start in range(len(traj_np) - traj_length + 1):
        segment = traj_np[start:start + traj_length]
        up_seg    = up_np[start:start + traj_length]
        right_seg = right_np[start:start + traj_length]
        down_seg  = down_np[start:start + traj_length]
        left_seg  = left_np[start:start + traj_length]

        sample = {
            'trajectory': torch.tensor(segment, dtype=torch.float32),  
            'up':    torch.tensor(up_seg, dtype=torch.float32),        
            'right': torch.tensor(right_seg, dtype=torch.float32),
            'down':  torch.tensor(down_seg, dtype=torch.float32),
            'left':  torch.tensor(left_seg, dtype=torch.float32),
            'cluster': cluster,
            'id': int(pid)
        }

        data.append(sample)


torch.save(data, "./social_lstm_structured.pt")
print(f"Saved {len(data)} trajectory segments.")

# train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

torch.save(train_data, "./train_social_lstm.pt")
torch.save(test_data, "./test_social_lstm.pt")

print(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples.")

from collections import Counter
labels = [d['cluster'] for d in data]
print(Counter(labels))
