import torch
import torch.nn as nn
import torch.nn.functional as F

class SocialLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, grid_size=(4, 4), neighborhood_size=4.0, dropout=0.1, observed=15):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        self.obs_len = observed

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.social_pool_mlp = nn.Sequential(
            nn.Linear(hidden_size * grid_size[0] * grid_size[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.classifier = nn.Linear(hidden_size, 2)

    def get_social_grid(self, hidden_states, positions, batch_size, reference_pos, mask=None):
        grid_cells = self.grid_size[0] * self.grid_size[1]
        social_tensor = torch.zeros(1, grid_cells, self.hidden_size, device=hidden_states.device)

        cell_width = self.neighborhood_size / self.grid_size[0]
        cell_height = self.neighborhood_size / self.grid_size[1]

        half_grid_x = self.grid_size[0] // 2
        half_grid_y = self.grid_size[1] // 2

        ref_pos = reference_pos.squeeze(0)
        rel_positions = positions - ref_pos

        for j in range(batch_size):
            if mask is not None and mask[j] == 0:
                continue

            rel_x, rel_y = rel_positions[j]

            if (abs(rel_x) > self.neighborhood_size / 2) or (abs(rel_y) > self.neighborhood_size / 2):
                continue

            cell_x = (rel_x / cell_width).long() + half_grid_x
            cell_y = (rel_y / cell_height).long() + half_grid_y

            if 0 <= cell_x < self.grid_size[0] and 0 <= cell_y < self.grid_size[1]:
                idx = cell_y * self.grid_size[0] + cell_x
                social_tensor[0, idx] += hidden_states[j]

        return social_tensor.view(1, -1)

    def forward(self, observed_trajectory_target, observed_trajectory_others, neighbor_mask=None):
        obs_len = len(observed_trajectory_others)

        hidden_target = torch.zeros(1, self.hidden_size, device=observed_trajectory_target[0].device)
        cell_target = torch.zeros(1, self.hidden_size, device=observed_trajectory_target[0].device)

        for t in range(obs_len):
            target_input = observed_trajectory_target[t].unsqueeze(0)  # shape: [1, F]
            target_pos = target_input[:, :2]  # shape: [1, 2]

            hidden_target, cell_target = self.lstm_cell(target_input, (hidden_target, cell_target))
            hidden_target = self.dropout(hidden_target)

            others_input = observed_trajectory_others[t]  # shape: [N_i, F]
            if others_input.shape[0] > 0:
                others_pos = others_input[:, :2]
                hidden_others = torch.zeros(others_input.shape[0], self.hidden_size, device=others_input.device)
                cell_others = torch.zeros(others_input.shape[0], self.hidden_size, device=others_input.device)

                hidden_others, cell_others = self.lstm_cell(others_input, (hidden_others, cell_others))
                hidden_others = self.dropout(hidden_others)

                mask_t = neighbor_mask[t] if neighbor_mask is not None else None
                social_tensor = self.get_social_grid(hidden_others, others_pos, others_input.shape[0], target_pos, mask=mask_t)
                social_context = self.social_pool_mlp(social_tensor)
            else:
                social_context = torch.zeros_like(hidden_target)

            combined = hidden_target + social_context

        return self.classifier(combined)
