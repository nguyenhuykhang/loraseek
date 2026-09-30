import torch
import torch.nn.functional as F
from tqdm import tqdm
import utils
import numpy as np
import os
import cv2
import logging
import time
import torch 

def save_checkpoint(model, epoch, args):
    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(model_state_dict, os.path.join(args.output_dir, f"epoch{str(epoch) }_model_{args.model_name}.pth"))


def load_checkpoint(model, args):
    try:
        state_dict = torch.load(args.ckpt_path, map_location=args.device)
    except RuntimeError:
        print("Checkpoint was saved on GPU. Loading on CPU instead.")
        state_dict = torch.load(args.ckpt_path, map_location=torch.device('cpu')) 
        
    if list(state_dict.keys())[0].startswith("module."):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # Remove "module." prefix
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)



def DNN_save_checkpoint(dnn, epoch, args):

    torch.save(dnn.state_dict(), os.path.join(args.output_dir, f"epoch{str(epoch) }_DNN.pth"))

def DNN_load_checkpoint(dnn, args):
    dnn.load_state_dict(torch.load(args.dnn_path))

def save_sample(args, signals, labels, total_samples, truth_signals_stft, denoise_stft, signals_stft):
    for batch_index in range(signals.size(0)):
        if batch_index < len(labels):
            # Prepare file path
            path_src = os.path.join(args.sample_dir, f"sample_{total_samples}_batch_{batch_index}")

            # Ground truth
            groundtruth_image = np.squeeze(truth_signals_stft[batch_index].detach().cpu().numpy().transpose(1, 2, 0))
            groundtruth_image = np.abs(groundtruth_image[:, :, 0] + 1j * groundtruth_image[:, :, 1])
            groundtruth_image = (groundtruth_image - np.amin(groundtruth_image)) / (np.amax(groundtruth_image) - np.amin(groundtruth_image)) * 255
            cv2.imwrite(path_src + '_groundtruth.png', groundtruth_image)

            # Denoised
            denoised_image = np.squeeze(denoise_stft[batch_index].detach().cpu().numpy().transpose(1, 2, 0))
            denoised_image = np.abs(denoised_image[:, :, 0] + 1j * denoised_image[:, :, 1])
            denoised_image = (denoised_image - np.amin(denoised_image)) / (np.amax(denoised_image) - np.amin(denoised_image)) * 255
            cv2.imwrite(path_src + '_denoised.png', denoised_image)

            # Noisy input
            raw_image = np.squeeze(signals_stft[batch_index].detach().cpu().numpy().transpose(1, 2, 0))
            raw_image = np.abs(raw_image[:, :, 0] + 1j * raw_image[:, :, 1])
            raw_image = (raw_image - np.amin(raw_image)) / (np.amax(raw_image) - np.amin(raw_image)) * 255
            cv2.imwrite(path_src + '_raw.png', raw_image)

def training(model, train_loader, args, test_loader= None):
    ####################################################################################################
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.1)
    loss_spec = torch.nn.MSELoss(reduction='mean')
    loss_class = torch.nn.CrossEntropyLoss()

    model.train()
    ####################################################################################################

    downchirp, _, _ = utils.gen_constants(args)
    for epoch in range(1, args.epochs + 1):
        # changhe th evariable name
        running_loss = 0.0
        running_mse_loss = 0.0
        running_rec_mse_loss = 0.0
        running_ce_loss = 0.0
        total_samples = 0
        total_equal_dnn = 0
        total_equal_loraphy = 0

        for signals, truth_signals, labels, snr_batch in tqdm(train_loader, desc=f"Training epoch {epoch}/{args.epochs}", leave=False):
            signals = signals.to(args.device)
            truth_signals = truth_signals.to(args.device)
            labels = labels.to(args.device)

            signals_stft = utils.perform_stft(signals, args)
            
            truth_signals_stft = utils.perform_stft(truth_signals, args)

            # print('signal_reconstruct',utils.loraphy(signal_reconstruct, downchirp, args))
            # print('signals',utils.loraphy(signals, downchirp, args))

            # TODO: more loss function 
            denoise_stft, dnn_logits = model(signals_stft)


            denoise_reconstruct = utils.invert_transform_with_zeros(denoise_stft, args)
            truth_signals_reconstruct = utils.invert_transform_with_zeros(truth_signals_stft, args)

            decoded_loraphy= utils.loraphy(denoise_reconstruct, downchirp, args)
            comparison_loraphy = torch.eq(decoded_loraphy, labels)  # Boolean tensor indicating element-wise equality
            total_equal_loraphy += torch.sum(comparison_loraphy).item()


            # Compute losses
            #mse_signal_loss = F.mse_loss(denoise_reconstruct, truth_signals_reconstruct, reduction='mean')
            magnitude_loss = F.mse_loss(torch.abs(denoise_reconstruct), torch.abs(truth_signals_reconstruct))
            phase_input = torch.angle(denoise_reconstruct)
            phase_target = torch.angle(truth_signals_reconstruct)
            phase_loss = F.mse_loss(phase_input, phase_target)

            reconstruct_mse_loss = magnitude_loss + phase_loss
            mse_loss = args.alpha * loss_spec(denoise_stft, truth_signals_stft)
            ce_loss = args.beta * loss_class(dnn_logits, labels)
            loss =  mse_loss + ce_loss + reconstruct_mse_loss #+ mse_signal_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running totals
            running_loss += loss.item()
            running_mse_loss += mse_loss.item()
            running_rec_mse_loss += reconstruct_mse_loss.item()
            running_ce_loss += ce_loss.item()

            # Compute accuracy
            decoded_dnn = dnn_logits.argmax(dim=1)
            comparison_dnn = torch.eq(decoded_dnn, labels)
            total_equal_dnn += torch.sum(comparison_dnn).item()
            total_samples += labels.size(0)


            if total_samples % 2560 == 0:
                batch_accuracy_dnn = (torch.sum(comparison_dnn).item() / decoded_dnn.numel()) * 100
                batch_accuracy_loraphy = (torch.sum(comparison_loraphy).item() / decoded_loraphy.numel()) * 100

                baseline_decoded_loraphy= utils.loraphy(signals, downchirp, args)
                baseline_comparison_loraphy = torch.eq(baseline_decoded_loraphy, labels)  
                baseline_batch_accuracy_loraphy = (torch.sum(baseline_comparison_loraphy).item() / baseline_decoded_loraphy.numel()) * 100

                log_batch = (
                    f"Processed: {total_samples}, for this batch: "
                    f"Loss: {loss:.4f} | "
                    f"MSE Loss: {mse_loss:.4f} | "
                    f"CE Loss: {ce_loss:.4f} | "
                    f"reconstruct MSE Loss: {reconstruct_mse_loss:.4f} | "
                    f"percentage_equal_dnn: {batch_accuracy_dnn} | "
                    f"percentage_equal_loraphy: {batch_accuracy_loraphy}/{baseline_batch_accuracy_loraphy} |"
                )
                print(log_batch)
                logging.info(log_batch)
                
                save_sample(args, signals, labels, total_samples, truth_signals_stft, denoise_stft, signals_stft)


        ####################################################################################################
        # Compute averages
        avg_loss = running_loss / len(train_loader)
        avg_mse = running_mse_loss / len(train_loader)
        avg_rec_mse = running_rec_mse_loss / len(train_loader)
        avg_ce = running_ce_loss / len(train_loader)
        acc_dnn = (total_equal_dnn / total_samples) * 100
        acc_loraphy = (total_equal_loraphy / total_samples) * 100

        log_epoch = (
        f"Epoch {epoch}/{args.epochs + 1} completed, "
          f"avg_loss: {avg_loss}, "
          f"avg_mse: {avg_mse}, "
          f"avg_rec_mse: {avg_rec_mse}, "
          f"avg_ce: {avg_ce}, "
          f"acc_dnn: {acc_dnn}, "
          f"acc_loraphy: {acc_loraphy} "
        )
        logging.info(log_epoch)
        print(log_epoch)

        # print(f"Epoch {epoch}/{args.epochs + 1} completed, "
        #   f"avg_loss: {avg_loss}, "
        #   f"avg_mse: {avg_mse}, "
        #   f"avg_rec_mse: {avg_rec_mse}, "
        #   f"avg_ce: {avg_ce}"
        #   f"acc_dnn: {acc_dnn}"
        #   f"acc_loraphy: {acc_loraphy}"
        #   )
        scheduler.step()
        save_checkpoint(model, epoch, args)
        if test_loader:
            pass
        ####################################################################################################

def testing(model, test_loader, args):
    """
    Evaluate the model on test data (no gradient updates).
    
    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Test or validation data loader.
        device (str): 'cpu' or 'cuda'.
        alpha (float): Weight for MSE loss.
        beta (float): Weight for CE loss.
    
    Returns:
        dict: A dictionary containing:
            - 'loss': Average total loss over the dataset.
            - 'mse_loss': Average MSE loss.
            - 'ce_loss': Average CrossEntropy loss.
            - 'accuracy': Classification accuracy (%).
    """
    print('Testing Phase\n')
    model.to(args.device)
    model.eval()
    # running_loss = 0.0
    # running_mse_loss = 0.0
    # running_ce_loss = 0.0
    total_samples = 0

    baseline_error_matrix_loraphy = np.zeros([len(args.snr_list), 1], dtype=float)

    error_matrix_dnn = np.zeros([len(args.snr_list), 1], dtype=float)
    error_matrix_loraphy = np.zeros([len(args.snr_list), 1], dtype=float)
    error_matrix_count = np.zeros([len(args.snr_list), 1], dtype=int)

    downchirp, _, _ = utils.gen_constants(args)

    with torch.no_grad():
        for signals, truth_signals, labels, snr_batch  in tqdm(test_loader, desc=f"Testing", leave=False):
            signals = signals.to(args.device)
            truth_signals = truth_signals.to(args.device)
            labels = labels.to(args.device)

            signals_stft = utils.perform_stft(signals, args)
            denoise_stft, dnn_logits = model(signals_stft)

            denoise_reconstruct = utils.invert_transform_with_zeros(denoise_stft, args)

            total_samples += labels.size(0)

            decoded_dnn = dnn_logits.argmax(dim=1)
            comparison_dnn = torch.eq(decoded_dnn, labels)

            decoded_loraphy= utils.loraphy(denoise_reconstruct, downchirp, args)
            comparison_loraphy = torch.eq(decoded_loraphy, labels)  # Boolean tensor indicating element-wise equality

            #### BASELINE##########################
            baseline_decoded_loraphy= utils.loraphy(signals, downchirp, args)
            baseline_comparison_loraphy = torch.eq(baseline_decoded_loraphy, labels)  # Boolean tensor indicating element-wise equality
            ########################################

            snr_batch = snr_batch.cpu().numpy()

            if total_samples % 3200 == 0:
                batch_accuracy_dnn = (torch.sum(comparison_dnn).item() / decoded_dnn.numel()) * 100
                batch_accuracy_loraphy = (torch.sum(comparison_loraphy).item() / decoded_loraphy.numel()) * 100

                baseline_batch_accuracy_loraphy = (torch.sum(baseline_comparison_loraphy).item() / baseline_decoded_loraphy.numel()) * 100

                
                log_testing = (
                    f"Processed: {total_samples} | "
                    f"percentage_dnn: {batch_accuracy_dnn} | "
                    f"percentage_equal_loraphy: {batch_accuracy_loraphy}/{baseline_batch_accuracy_loraphy} |"
                )
                print(log_testing)
                logging.info(log_testing)

            for i, snr in enumerate(snr_batch):
                if snr in args.snr_list:
                    snr_index = args.snr_list.index(snr)  # Get index of SNR in the list
                    error_matrix_count[snr_index] += 1  # Increment count for this SNR

                    if comparison_dnn[i]:
                        error_matrix_dnn[snr_index] += 1

                    if comparison_loraphy[i]:
                        error_matrix_loraphy[snr_index] += 1

                    if baseline_comparison_loraphy[i]:
                        baseline_error_matrix_loraphy[snr_index] += 1



    error_matrix_dnn = np.divide(error_matrix_dnn, error_matrix_count)

    error_matrix_loraphy = np.divide(error_matrix_loraphy, error_matrix_count)

    baseline_error_matrix_loraphy = np.divide(baseline_error_matrix_loraphy, error_matrix_count)
    
    print("error_matrix_dnn")
    print(error_matrix_dnn)
    print("\nerror_matrix_loraphy")
    print(error_matrix_loraphy)
    print("\nbaseline_error_matrix_loraphy")
    print(baseline_error_matrix_loraphy)

    np.save(os.path.join(args.output_dir, 'error_matrix_dnn.npy'), error_matrix_dnn)
    np.save(os.path.join(args.output_dir, 'error_matrix_loraphy.npy'), error_matrix_loraphy)
    np.save(os.path.join(args.output_dir, 'baseline_error_matrix_loraphy.npy'), baseline_error_matrix_loraphy)