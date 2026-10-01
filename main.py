import os
import sys
import torch
import random
import numpy as np
import argparse
import time 
import logging

# model files import their siblings (transformer, localvit) as top-level modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'))
import utils
import data_loader
import run
import model_sf7

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__": 
    arg = argparse.ArgumentParser()

    arg.add_argument('--root_path', type=str, 
        default='./', help='The root path')
    arg.add_argument('--data_dir', type=str, 
        default='path/to/data', help='The data directory')
    arg.add_argument('--ckpt_path', type=str, 
        default='path/to/checkpoint', help='The checkpoint path')
    arg.add_argument('--model_name', type=str, 
        default='LoRaSeekSF7Large', help='model_name')
    arg.add_argument('--sf', type=int, 
        default = 7, help='The spreading factor.') 
    arg.add_argument('--use_checkpoint', action='store_true', 
        default = True, help='Use checkpoint.') 
    arg.add_argument('--bw', type=int, 
        default = 125000, help='The bandwidth') 
    arg.add_argument('--fs', type=int, 
        default = 1000000, help='The sampling rate.') 
    arg.add_argument('--upsampling', type=int, 
        default = 100, help='The upsampling rate.') 
    arg.add_argument("--snr_list", nargs='+',
        default=list(range(-35, -10)), type=int)  
    arg.add_argument('--normalization', action='store_true', 
        default = True, help='Normalizing.') 
    arg.add_argument('--use_validation', action='store_true', 
        default = False, help='Normalizing.') 
    arg.add_argument('--device', type=str, 
        default='gpu', help='Device: gpu | cpu')
    arg.add_argument('--epochs',type=int, 
        default=0, help= 'Train iterations (0 = test)')
    arg.add_argument('--batch_size',type=int, 
        default=64, help= 'Batch size')
    arg.add_argument('--train_ratio', type=float, 
        default = 0.8, help='The train ratio.') 
    arg.add_argument('--alpha', type=float, 
        default = 2048, help='') 
    arg.add_argument('--beta', type=float, 
        default = 1.5, help='') 
    
    args = arg.parse_args()

    args.output_dir = os.path.join(args.root_path, 'output')
    
    # Add time to dir 
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir = os.path.join(args.output_dir, f"SF{args.sf}-" + timestamp)
    utils.create_dir(args.output_dir)

    args.sample_dir = os.path.join(args.output_dir, 'sample')
    utils.create_dir(args.sample_dir)

    args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    utils.print_opts(args)

    log_file = os.path.join(args.output_dir, "log_file.txt")
    logging.basicConfig(
        filename=log_file,
        filemode="w",  
        format="%(asctime)s - %(message)s",
        level=logging.INFO
    )

    print(f"Device: {args.device}")

    set_seed(1)

    data_list = utils.data_aggregation(args)

    random.shuffle(data_list)

    train_data_list, test_data_list, val_data_list = utils.split_data(data_list, args)
    print('Training data:', len(train_data_list))
    print('Testing data:', len(test_data_list))

    train_loader = data_loader.create_dataloader(train_data_list, args, batch_size=args.batch_size, shuffle=False)
    test_loader = data_loader.create_dataloader(test_data_list, args, batch_size=args.batch_size, shuffle=False)
    
    
    # Model Loading
    ##################################################################################
    model = model_sf7.LoRaSeekSF7Large()
    ###################################################################################

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
        #model = torch.nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", total_params)

    if args.use_checkpoint:
        print(f"Loading checkpoint: {args.ckpt_path}")
        run.load_checkpoint(model, args)

    if args.epochs > 0:
        run.training(model, train_loader, args, test_loader)

    run.testing(model, test_loader, args)
    ##################################################################################
