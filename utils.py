import os
import re
import random
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from scipy.signal import chirp as np_chirp

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def set_gpu(gpu_id):
    torch.cuda.set_device(gpu_id)

def save_without_classifier(model, save_path):
    state_dict = model.state_dict()
    # Remove classifier layers from state_dict
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
    torch.save(state_dict, save_path)

def signal_amplitude(x):
    # x: [B, 2, 128, 33] (real and imaginary parts) --> [B, 1, 128, 33]
    abs_signal = torch.sqrt(x[:, 0, :, :]**2 + x[:, 1, :, :]**2).unsqueeze(1)
    return abs_signal

def add_noise(data, snr):
    amp_sig = np.mean(np.abs(data)) 
    amp_noise = amp_sig / 10**(snr/20)
    noise = (np.random.randn(len(data)) + 1j * np.random.randn(len(data))) * amp_noise / np.sqrt(2)
    dataX = data + noise  
    return dataX 

def data_aggregation(arg):

    data_list = []

    for filename in os.listdir(arg.data_dir):
        # filename: {label}_{snr}_{sf}_{bw}_{batch}_{label}_{packet}_{symbol}.mat
        label_str = str( int(round(float(filename.split('_')[0]))) )
        #print(label_str)
        snr_str = filename.split('_')[1]

        if int(snr_str) not in arg.snr_list:
            continue

        signal_path = os.path.join(arg.data_dir, filename)

        # replace snr to 35
        truth_signal_name = f"{label_str}_35_{'_'.join(filename.split('_')[2:])}"
        truth_signal_path = os.path.join(arg.data_dir, truth_signal_name)

        data_list.append({
                "signal_path": signal_path,
                "truth_signal_path": truth_signal_path,
                "label_int": int(label_str),
                "snr_int": int(snr_str),
            })

    return data_list

def data_aggregation_lite(arg):

    data_list = []

    for filename in os.listdir(arg.data_dir):
        # filename: {label}_{snr}_{sf}_{bw}_{batch}_{label}_{packet}_{symbol}.mat
        label_str = filename.split('_')[0]
        
        snr_str = filename.split('_')[1]

        if int(snr_str) not in arg.snr_list:
            continue

        signal_path = os.path.join(arg.data_dir, filename)

        # replace snr to 35
        truth_signal_name = f"{label_str}_35_{'_'.join(filename.split('_')[2:])}"
        truth_signal_path = os.path.join(arg.data_dir, truth_signal_name)

        data_list.append({
                "signal_path": signal_path,
                "label_int": int(label_str),
                "snr_int": int(snr_str),
            })

    return data_list


def split_data(data_list, args):

    if args.use_validation:
        val_ratio = 0.1
    else:
        val_ratio = 0.0

    test_ratio = 1.0 - args.train_ratio - val_ratio

    if test_ratio < 0:
        raise ValueError("Sum of train_ratio and val_ratio exceeds 1.0.")

    total = len(data_list)
    train_end = int(args.train_ratio * total)
    val_end = int((args.train_ratio + val_ratio) * total)

    train_data_list = data_list[:train_end]
    val_data_list   = data_list[train_end:val_end] if args.use_validation else []
    test_data_list  = data_list[val_end:]

    return train_data_list, test_data_list, val_data_list


def perform_stft(data_in, args):
    hann_window = torch.hann_window(2**args.sf // 2, device=data_in.device)

    stft_full_img = torch.stft(input=data_in, n_fft=int(2**args.sf * args.fs / args.bw),
                               hop_length=2**args.sf // 4, win_length=2**args.sf  // 2, 
                               window=hann_window, pad_mode='constant',
                               return_complex=True)

    stft_img = torch.concat((stft_full_img[:, -2**args.sf  // 2:, :], stft_full_img[:, 0:2**args.sf  // 2, :]), axis=1)
    
    return torch.stack((stft_img.real, stft_img.imag), 1)  


def invert_transform_with_zeros(y, args):
    """
    Reconstruct x

    Args:
    y: Tensor, transformed tensor of shape [batch, freq_size, 33].
    original_shape: Tuple, the original shape of x, e.g., [batch, 1024, 33].
    freq_size: int, the reduced frequency size (e.g., 128).
    
    Returns:
    Reconstructed x with the same shape as the original input.
    """
    trim_size = 2**args.sf // 2  # E.g., 64 for freq_size=128, freq_size=2**args.sf
    freq_dim = int(2**args.sf * args.fs / args.bw)
    time_dim = 33

    real_part = y[:, 0, :, :]  # (batch_size, freq_dim, time_dim)
    imag_part = y[:, 1, :, :]  # (batch_size, freq_dim, time_dim)

    stft_img = torch.complex(real_part, imag_part)

    # init output tensor with zeros
    signal = torch.zeros((stft_img.shape[0], freq_dim, time_dim), dtype=torch.complex64, device=y.device)

    # Step 1: Fill the appropriate sections with the values from y
    # fill the first trim_size with the second half of y
    signal[:, :trim_size, :] = stft_img[:, trim_size:, :]  # y[:, trim_size:, :] -> x[:, :trim_size, :]

    # fill the last trim_size with the first half of y
    signal[:, -trim_size:, :] = stft_img[:, :trim_size, :]  # y[:, :trim_size, :] -> x[:, -trim_size:, :]
    data_out = torch.istft(
        signal,
        n_fft=freq_dim,
        hop_length=2**args.sf // 4,
        win_length=2**args.sf // 2,
        window=torch.hamming_window(2**args.sf // 2).to(y.device),
        return_complex=True)

    return data_out



def gen_constants(args):
    num_classes = 2 ** args.sf 
    num_samples = int(num_classes * args.fs / args.bw)  # number of samples per symbol

    
    t = torch.linspace(0, num_samples / args.fs, num_samples, device=args.device)
    chirpI1 = torch.tensor(np_chirp(t.cpu().numpy(), f0=args.bw / 2, f1=-args.bw / 2, t1=2 ** args.sf / args.bw, method='linear', phi=90), device=args.device)
    chirpQ1 = torch.tensor(np_chirp(t.cpu().numpy(), f0=args.bw / 2, f1=-args.bw / 2, t1=2 ** args.sf / args.bw, method='linear', phi=0), device=args.device)
    downchirp = chirpI1 + 1j * chirpQ1

    dataE1 = torch.zeros((num_classes, num_samples), dtype=torch.complex64, device=args.device)
    dataE2 = torch.zeros((num_classes, num_samples), dtype=torch.complex64, device=args.device)

    for symbol_index in range(num_classes):
        time_shift = int(symbol_index / num_classes * num_samples)
        time_split = num_samples - time_shift
        dataE1[symbol_index, :time_split] = downchirp[time_shift:]
        if symbol_index != 0:
            dataE2[symbol_index, time_split:] = downchirp[:time_shift]

    return downchirp, dataE1, dataE2


def loraphy(data_in, downchirp, args):
    # We doing batch processing
    num_classes = 2**args.sf
    upsampling = args.upsampling  

    # Dechirp
    chirp_data = data_in * downchirp.unsqueeze(0)  

    # FFT
    chirp_data_padded = torch.fft.fft(chirp_data, n=chirp_data.shape[1] * upsampling, dim=1)

    # Cut the FFT results to two parts
    target_nfft = num_classes * upsampling
    cut1 = chirp_data_padded[:, :target_nfft]
    cut2 = chirp_data_padded[:, -target_nfft:]

    # Add absolute values of cut1 and cut2 to merge two peaks into one
    merged_peaks = torch.abs(cut1) + torch.abs(cut2)

    # Find the symbol index for each batch
    decoded_symbols = torch.argmax(merged_peaks, dim=1) / upsampling

    #return (decoded_symbols % num_classes).round().long()
    return (decoded_symbols.round() % num_classes).long()

def print_opts(opts):
    print('=' * 80)
    print('Args'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            try:
                print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
            except:
                pass
    print('=' * 80)


if __name__ == "__main__": 
    print('ok')
