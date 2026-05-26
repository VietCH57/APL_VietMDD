import os
import json
import ast
import pandas as pd
import torch
import torchaudio 
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

SAMPLING_RATE = 16000

class APLSupervisedDataset(Dataset):
    def __init__(self, csv_path, extracted_dir, vocab_json_path, pad_token="[PAD]"):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        # extracted_dir đóng vai trò là thư mục gốc "VietMDD" trên Kaggle
        self.wav_dir = extracted_dir 
        
        with open(vocab_json_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
            
        self.pad_idx = self.vocab.get(pad_token, 69)

    def _text_to_ids(self, text_string):
        if pd.isna(text_string):
            return []
        return [self.vocab[phone] for phone in text_string.split(" ") if phone in self.vocab]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path_str = str(row['Path']).strip()
        
        # Tiền tố đặc biệt thuộc nhóm trường mầm non (kindergarten)
        kinder_prefixes = ("đông", "thành", "tú", "tuyến")
        
        if path_str.startswith(kinder_prefixes):
            # Ví dụ: path_str là "tuyến_1-1"
            # Tách chuỗi tại dấu gạch dưới đầu tiên: prefix="tuyến", rest="1-1"
            if "_" in path_str:
                prefix, rest = path_str.split("_", 1)
            else:
                prefix, rest = path_str, ""
                
            # Tạo đường dẫn: VietMDD/kindergarten/tuyến/1-1.wav
            relative_path = os.path.join("kindergarten", prefix, f"{rest}.wav")
        else:
            # Thuộc nhóm trường tiểu học (primaryschool)
            # Tạo đường dẫn: VietMDD/primaryschool/THA_Nu_6_S00042_201.wav
            relative_path = os.path.join("primaryschool", f"{path_str}.wav")
            
        wav_path = os.path.join(self.wav_dir, relative_path)
        
        # 1. Tải trực tiếp tín hiệu âm thanh thô .wav
        waveform, sr = torchaudio.load(wav_path)
        if sr != SAMPLING_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLING_RATE)
        waveform = waveform.squeeze(0) # (num_samples,)
        
        linguistic_ids = self._text_to_ids(row['Canonical'])
        linguistic_tensor = torch.tensor(linguistic_ids, dtype=torch.long)
        
        transcript_ids = self._text_to_ids(row['Transcript'])
        transcript_tensor = torch.tensor(transcript_ids, dtype=torch.long)
        
        try:
            error_list = ast.literal_eval(row['Error'])
        except:
            error_list = []
        error_tensor = torch.tensor(error_list, dtype=torch.long)

        return (
            waveform, 
            linguistic_tensor, 
            transcript_tensor, 
            error_tensor
        )


def make_apl_collate_fn(pad_idx=69, error_pad_idx=2):
    def collate_fn(batch):
        waveforms, linguistics, transcripts, errors = zip(*batch)
        
        wav_padded = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
        linguistics_padded = pad_sequence(linguistics, batch_first=True, padding_value=pad_idx)
        transcripts_padded = pad_sequence(transcripts, batch_first=True, padding_value=pad_idx)
        errors_padded = pad_sequence(errors, batch_first=True, padding_value=error_pad_idx)
        
        target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
        
        return {
            'waveforms': wav_padded,
            'linguistics': linguistics_padded,
            'transcripts': transcripts_padded,
            'errors': errors_padded,
            'target_lengths': target_lengths
        }
    return collate_fn