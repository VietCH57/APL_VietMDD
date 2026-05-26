import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import gc

from model import AcousticPhoneticLinguistic
from dataset import APLSupervisedDataset, make_apl_collate_fn
from metric import calculate_all_metrics

def greedy_decode(log_probs, input_lengths, id_to_vocab, pad_idx, empty_idx):
    arg_maxes = torch.argmax(log_probs, dim=-1).transpose(0, 1) # (Batch, Time)
    decodes = []
    for i in range(arg_maxes.size(0)):
        seq = arg_maxes[i][:input_lengths[i]].tolist()
        prev = -1
        hyp = []
        for idx in seq:
            if idx != prev and idx != pad_idx and idx != empty_idx: 
                phone = id_to_vocab.get(idx, "").strip()
                if phone: 
                    hyp.append(phone)
            prev = idx
        decodes.append(hyp)
    return decodes

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    with open(args.vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    id_to_vocab = {v: k for k, v in vocab.items()}
    
    pad_idx = vocab.get("[PAD]", 69)
    empty_idx = vocab.get("", 68)
    print(f"Dynamic Mapping Found -> [PAD] ID: {pad_idx} | Empty String ID: {empty_idx}")
    
    train_dataset = APLSupervisedDataset(args.train_csv, args.wav_dir, args.vocab_path)
    dev_dataset = APLSupervisedDataset(args.dev_csv, args.wav_dir, args.vocab_path)
    
    collate_fn = make_apl_collate_fn(pad_idx=pad_idx, error_pad_idx=2)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    model = AcousticPhoneticLinguistic(
        num_classes=len(vocab), freq_bins=81, phon_feat_bins=1024, lstm_hidden=256, proj_dim=1024
    ).to(device)
    
    criterion = nn.CTCLoss(blank=pad_idx, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_f1 = -1.0
    
    for epoch in range(args.epochs):
        model.train()
        model.wav2vec2.eval() 
        
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            waveforms = batch['waveforms'].to(device)
            linguistics = batch['linguistics'].to(device)
            transcripts = batch['transcripts'].to(device)
            target_lengths = batch['target_lengths']
            
            optimizer.zero_grad()
            logits, log_probs, min_time = model(waveforms, linguistics)
            
            input_lengths = torch.full((waveforms.size(0),), fill_value=min_time, dtype=torch.long)
            loss = criterion(log_probs, transcripts, input_lengths, target_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            del waveforms, linguistics, transcripts, logits, log_probs, loss
            gc.collect()
            torch.cuda.empty_cache()
            
        print(f"Epoch {epoch+1} Complete. Avg Loss: {running_loss / len(train_loader):.4f}")
        
        if (epoch + 1) % args.eval_every_epochs == 0 or (epoch + 1) == args.epochs:
            model.eval()
            all_hyps, all_trans, all_canons = [], [], []
            
            with torch.no_grad():
                for batch in tqdm(dev_loader, desc="Evaluating"):
                    waveforms = batch['waveforms'].to(device)
                    linguistics = batch['linguistics'].to(device)
                    transcripts = batch['transcripts'].to(device)
                    target_lengths = batch['target_lengths']
                    
                    _, log_probs, min_time = model(waveforms, linguistics)
                    input_lengths = torch.full((waveforms.size(0),), fill_value=min_time, dtype=torch.long)
                    
                    hyps = greedy_decode(log_probs, input_lengths, id_to_vocab, pad_idx, empty_idx)
                    all_hyps.extend(hyps)
                    
                    for i in range(transcripts.size(0)):
                        t_seq = transcripts[i][:target_lengths[i]].tolist()
                        all_trans.append([id_to_vocab[idx] for idx in t_seq if idx != pad_idx])
                        
                    for i in range(linguistics.size(0)):
                        l_len = (batch['linguistics'][i] != pad_idx).sum().item()
                        l_seq = batch['linguistics'][i][:l_len].tolist()
                        all_canons.append([id_to_vocab[idx] for idx in l_seq if idx != pad_idx])
                        
            metrics = calculate_all_metrics(all_hyps, all_trans, all_canons)
            print(f"\n--- Validation Report (Epoch {epoch+1}) ---")
            print(f"PR Correctness: {metrics['PR_Correctness']:.4f} | Accuracy: {metrics['PR_Accuracy']:.4f}")
            print(f"MDD F-measure: {metrics['MDD_F_measure']:.4f} | Precision: {metrics['MDD_Precision']:.4f} | Recall/DR: {metrics['MDD_Recall']:.4f}")
            print(f"MDD FAR: {metrics['MDD_FAR']:.4f} | FRR: {metrics['MDD_FRR']:.4f} | DER: {metrics['MDD_DER']:.4f}")
            
            if metrics['MDD_F_measure'] > best_f1:
                best_f1 = metrics['MDD_F_measure']
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth"))
                print("=> Saved new best checkpoint!")
        else:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "latest_model.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="/kaggle/input/train.csv")
    parser.add_argument("--dev_csv", type=str, default="/kaggle/input/dev.csv")
    parser.add_argument("--wav_dir", type=str, default="/kaggle/input/VietMDD")
    parser.add_argument("--vocab_path", type=str, default="./vocab.json")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--eval_every_epochs", type=int, default=5)
    args = parser.parse_args()
    main(args)