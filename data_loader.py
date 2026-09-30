import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import os
import scipy
from scipy.signal import chirp


class LoRaDataset(Dataset):
    """
    A PyTorch Dataset for loading signal data and corresponding truth signals.
    """

    def __init__(self, data_list, args, transform=None):
        """
        Args:
            data_list (list of dict): Each dict contains:
                - 'signal': Path to the signal file.
                - 'truth_signal': Path to the corresponding truth signal file.
                - 'label_int': Integer representation of the label.
            transform (callable, optional): Transform to be applied on the signal.
        """
        self.data_list = data_list
        self.transform = transform
        self.args = args

    def __len__(self):
        """Return total number of samples."""
        return len(self.data_list)

    def __getitem__(self, idx):

        signal_dict = self.data_list[idx]

        label = signal_dict["label_int"]
        snr = signal_dict["snr_int"]

        signal = sio.loadmat(signal_dict["signal_path"])["chirp"]
        signal = np.squeeze(signal)

        truth_signal = sio.loadmat(signal_dict["truth_signal_path"])["chirp"]
        truth_signal = np.squeeze(truth_signal)

        if self.args.normalization:
            signal = signal / np.mean(np.abs(signal))
            truth_signal = truth_signal / np.mean(np.abs(truth_signal))
            #signal = signal / (np.mean(np.abs(signal)) + 1e-8)
        

        # Convert to torch tensors
        #signal_tensor = torch.tensor(signal, dtype=torch.cfloat)
        signal_tensor = torch.from_numpy(signal).cfloat()
        #truth_signal_tensor = torch.tensor(truth_signal, dtype=torch.cfloat)
        truth_signal_tensor = torch.from_numpy(truth_signal).cfloat()
        label_tensor = torch.tensor(label, dtype=torch.long)
        snr_tensor = torch.tensor(snr, dtype=torch.long)

        # torch.tensor(chirp_raw, dtype=torch.cfloat)

        # Apply any transforms if provided
        if self.transform:
            signal_tensor = self.transform(signal_tensor)

        return signal_tensor, truth_signal_tensor, label_tensor, snr_tensor


def create_dataloader(data_list, args, batch_size=32, shuffle=True, num_workers=0, transform=None):
    """
    Create a PyTorch DataLoader from a data list.

    Args:
        data_list (list of dict): Each dict contains:
            - 'signal': Path to the signal file.
            - 'truth_signal': Path to the corresponding truth signal file.
            - 'label_int': Integer representation of the label.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses for data loading.
        transform (callable, optional): Transform to be applied on the signal.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    dataset = LoRaDataset(data_list, args, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader
 


class LoRaDatasetLite(Dataset):
    """
    A PyTorch Dataset for loading signal data and corresponding truth signals.
    """

    def __init__(self, data_list, args, transform=None):
        """
        Args:
            data_list (list of dict): Each dict contains:
                - 'signal': Path to the signal file.
                - 'truth_signal': Path to the corresponding truth signal file.
                - 'label_int': Integer representation of the label.
            transform (callable, optional): Transform to be applied on the signal.
        """
        self.data_list = data_list
        self.transform = transform
        self.args = args

    def __len__(self):
        """Return total number of samples."""
        return len(self.data_list)

    def __getitem__(self, idx):

        signal_dict = self.data_list[idx]

        label = signal_dict["label_int"]
        snr = signal_dict["snr_int"]

        signal = sio.loadmat(signal_dict["signal_path"])["chirp"]
        signal = np.squeeze(signal)

        file_name = os.path.basename(signal_dict["signal_path"])
        truth_signal_name = f"{str(label)}_35_{'_'.join(file_name.split('_')[2:])}"
        truth_signal_path = os.path.join(self.args.data_dir, truth_signal_name)

        truth_signal = sio.loadmat(truth_signal_path)["chirp"]
        truth_signal = np.squeeze(truth_signal)


        if self.args.normalization:
            signal = signal / np.mean(np.abs(signal))
            truth_signal = truth_signal / np.mean(np.abs(truth_signal))
            #signal = signal / (np.mean(np.abs(signal)) + 1e-8)
        

        # Convert to torch tensors
        #signal_tensor = torch.tensor(signal, dtype=torch.cfloat)
        signal_tensor = torch.from_numpy(signal).cfloat()
        #truth_signal_tensor = torch.tensor(truth_signal, dtype=torch.cfloat)
        truth_signal_tensor = torch.from_numpy(truth_signal).cfloat()
        label_tensor = torch.tensor(label, dtype=torch.long)
        snr_tensor = torch.tensor(snr, dtype=torch.long)

        # torch.tensor(chirp_raw, dtype=torch.cfloat)

        # Apply any transforms if provided
        if self.transform:
            signal_tensor = self.transform(signal_tensor)

        return signal_tensor, truth_signal_tensor, label_tensor, snr_tensor


def create_dataloader_lite(data_list, args, batch_size=32, shuffle=True, num_workers=0, transform=None):
    """
    Create a PyTorch DataLoader from a data list.

    Args:
        data_list (list of dict): Each dict contains:
            - 'signal': Path to the signal file.
            - 'truth_signal': Path to the corresponding truth signal file.
            - 'label_int': Integer representation of the label.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of subprocesses for data loading.
        transform (callable, optional): Transform to be applied on the signal.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    dataset = LoRaDatasetLite(data_list, args, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader

def add_noise(data, snr):
    amp_sig = np.mean(np.abs(data)) 
    amp_noise = amp_sig / 10**(snr/20)
    noise = (np.random.randn(len(data)) + 1j * np.random.randn(len(data))) * amp_noise / np.sqrt(2)
    dataX = data + noise  
    return dataX
