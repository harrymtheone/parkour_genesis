import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader

from rsl_rl.modules.odometer.recurrent import OdomRecurrentTransformer


# Define a simple dataset for a single data file loaded into memory
class SingleFileDataset(Dataset):
    """A PyTorch Dataset for a single dictionary of tensors."""

    def __init__(self, data_dict: dict):
        self.data = data_dict
        self.keys = list(data_dict.keys())
        # Assuming all tensors in the dictionary have the same length
        self._len = self.data[self.keys[0]].shape[0]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        return {key: self.data[key][idx] for key in self.keys}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    dataset_dir = '/home/harry/projects/parkour_genesis/odometry/dataset/'
    pt_files = sorted(glob.glob(os.path.join(dataset_dir, '*.pt')))

    batch_size = 64
    epochs_per_file = 3
    num_workers = 4

    transformer = OdomRecurrentTransformer(
        n_proprio=50,
        embed_dim=64,
        hidden_size=128,
        estimator_out_dim=3
    ).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=(device == 'cuda'))
    mse = torch.nn.MSELoss()
    l1 = torch.nn.L1Loss()

    print(f"Found {len(pt_files)} dataset files. Starting training loop...")

    for file_idx, file_path in enumerate(pt_files):
        print(f"\n--- Loading file {file_idx + 1}/{len(pt_files)}: {os.path.basename(file_path)} ---")

        data_dict = torch.load(file_path, weights_only=True)
        dataset = SingleFileDataset(data_dict)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        for epoch in range(epochs_per_file):
            print(f"  Epoch {epoch + 1}/{epochs_per_file}")

            for i, batch in enumerate(dataloader):
                # Move batch data to the correct device
                prop = batch['prop'].to(device).transpose(0, 1)
                depth = batch['depth'].to(device).transpose(0, 1)
                recon_target = batch['recon'].to(device).transpose(0, 1)
                priv_target = batch['priv'].to(device).transpose(0, 1)

                optimizer.zero_grad()

                # Use autocast for the forward pass
                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device == 'cuda')):
                    recon, est = transformer(prop, depth)
                    loss_recon = mse(recon, recon_target)
                    loss_priv = mse(est, priv_target)

                # Scale loss and call backward to create scaled gradients
                scaler.scale(loss_recon + loss_priv).backward()
                # Unscale gradients and call optimizer.step()
                scaler.step(optimizer)
                # Update the scale for the next iteration
                scaler.update()

                if i % 50 == 0:
                    print(f"    Batch {i}/{len(dataloader)}, Loss recon: {loss_recon.item():.3f}, Loss priv: {loss_priv:.3f}")


if __name__ == '__main__':
    main()
