import os
import sys
import json
import torch
import h5py
import pandas as pd
import argparse
import random
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import math
import re
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

H5_FEATURE_DIM = 1024
LLM_EMBEDDING_DIM = 1024

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0,1).to(x.device)
        return self.dropout(x)

class CustomImageTextDecoder(nn.Module):
    def __init__(self, vocab_size, llm_embedding_dim, nhead, num_decoder_layers, dim_feedforward, max_seq_len, h5_feature_dim, tokenizer, dropout=0.1):
        super().__init__()
        self.llm_embedding_dim = llm_embedding_dim
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.token_embedding = nn.Embedding(vocab_size, llm_embedding_dim)
        self.image_feature_projector = nn.Linear(h5_feature_dim, llm_embedding_dim)
        self.positional_encoding = PositionalEncoding(llm_embedding_dim, dropout, max_seq_len)
        decoder_layer = TransformerDecoderLayer(d_model=llm_embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_layer = nn.Linear(llm_embedding_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, image_features, tgt_mask=None, memory_key_padding_mask=None, labels=None):
        memory_input = self.image_feature_projector(image_features)
        token_embeddings = self.token_embedding(input_ids)
        text_embeddings_with_pos = self.positional_encoding(token_embeddings)
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(input_ids.size(1)).to(input_ids.device)
        output = self.transformer_decoder(tgt=text_embeddings_with_pos, memory=memory_input, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        logits = self.output_layer(output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
        return {"logits": logits, "loss": loss}
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class FineTuneDataset(Dataset):
    def __init__(self, dataframe, tokenizer, h5_feature_dirs):
        self.manifest = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = 64
        self.h5_feature_dirs = h5_feature_dirs
        self.feature_file_map = {}
        print("H5 dosyaları taranıyor ve haritası oluşturuluyor...")
        for features_dir in self.h5_feature_dirs:
            try:
                for f in os.listdir(features_dir):
                    if f.endswith('.h5'):
                        self.feature_file_map[f] = os.path.join(features_dir, f)
            except FileNotFoundError:
                print(f"UYARI: Özellik klasörü bulunamadı, atlanıyor: {features_dir}")
        print(f"{len(self.feature_file_map)} adet benzersiz .h5 dosyası bulundu ve haritalandı.")
        if self.tokenizer.pad_token is None: self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if self.tokenizer.eos_token is None: self.tokenizer.add_special_tokens({'eos_token': '[EOS]'})
        if self.tokenizer.bos_token is None: self.tokenizer.add_special_tokens({'bos_token': '[BOS]'})

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        feature_filename = row['feature_path']
        correct_feature_path = self.feature_file_map.get(feature_filename)
        if not correct_feature_path:
            return None
        try:
            with h5py.File(correct_feature_path, 'r') as f:
                features = torch.tensor(f['feats'][:], dtype=torch.float)
        except Exception:
            return None
        report_text = row.get('report', '')
        full_report_text = f"{self.tokenizer.bos_token}{report_text}{self.tokenizer.eos_token}"
        encoded_report = self.tokenizer(
            full_report_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        labels = encoded_report.input_ids.squeeze()
        return features, labels, feature_filename.replace('.h5', '.tiff')

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None, None, None
    features, labels, ids = zip(*batch)
    labels = torch.stack(labels)
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    lengths = [len(f) for f in features]
    mask = torch.arange(padded_features.size(1)).expand(len(lengths), -1) >= torch.tensor(lengths).unsqueeze(1)
    return padded_features, labels, list(ids), mask

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a custom decoder model on Azure ML.")
    parser.add_argument('--h5_features_dirs', type=str, nargs='+', required=True, help='.h5 dosyalarını içeren klasörlerin yolları.')
    parser.add_argument('--manifest_csv', type=str, required=True, help='Yerel olarak yüklenen finetune_data.csv dosyasının adı.')
    parser.add_argument('--test_list_path', type=str, required=True, help='Test ID\'lerini içeren .txt dosyasının yolu.')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Warmup sonrası LR (default 5e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='İlk yüksek LR kaç epoch sürecek (default 10)')
    parser.add_argument('--warmup_high_lr', type=float, default=5e-5, help='İlk warmup LR (default 5e-5)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılacak cihaz: {device}")
    
    manifest_df = pd.read_csv(args.manifest_csv)
    
    try:
        with open(args.test_list_path, 'r') as f:
            test_ids = {line.strip().replace('.tiff', '.h5') for line in f if line.strip()}
        print(f"'{args.test_list_path}' dosyasından {len(test_ids)} adet test ID'si okundu.")
    except FileNotFoundError:
        print(f"HATA: Test listesi dosyası bulunamadı: {args.test_list_path}")
        sys.exit(1)

    test_df = manifest_df[manifest_df['feature_path'].isin(test_ids)].copy()
    train_val_df = manifest_df[~manifest_df['feature_path'].isin(test_ids)].copy()
    train_df, val_df = train_test_split(train_val_df, test_size=0.15, random_state=42)
    
    print(f"Toplam: {len(manifest_df)} | Eğitim: {len(train_df)} | Doğrulama: {len(val_df)} | Test (Belirtilen): {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.eos_token is None: tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    if tokenizer.bos_token is None: tokenizer.add_special_tokens({'bos_token': '[BOS]'})
    
    model = CustomImageTextDecoder(
        vocab_size=len(tokenizer),
        llm_embedding_dim=LLM_EMBEDDING_DIM,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        max_seq_len=64,
        h5_feature_dim=H5_FEATURE_DIM,
        tokenizer=tokenizer
    ).to(device)
    
    train_dataset = FineTuneDataset(train_df, tokenizer, args.h5_features_dirs)
    val_dataset = FineTuneDataset(val_df, tokenizer, args.h5_features_dirs)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    
    print("\n--- Model Eğitimi Başlıyor! ---")
    for epoch in range(args.epochs):
        # ---- LR'ı epoch başında ayarla (warmup yüksek LR -> sonra düşük LR) ----
        target_lr = args.warmup_high_lr if epoch < args.warmup_epochs else args.learning_rate
        for pg in optimizer.param_groups:
            pg['lr'] = target_lr

        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_loader, desc=f"Eğitim Epoch {epoch + 1}/{args.epochs}")
        
        for batch_data in train_progress_bar:
            if batch_data[0] is None: continue
            features, labels, _, mask = batch_data
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=labels, image_features=features, labels=labels, memory_key_padding_mask=mask)
            loss = outputs["loss"]
            if loss is not None:
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                train_progress_bar.set_postfix({'train_loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                if batch_data[0] is None: continue
                features, labels, _, mask = batch_data
                features, labels, mask = features.to(device), labels.to(device), mask.to(device)
                outputs = model(input_ids=labels, image_features=features, labels=labels, memory_key_padding_mask=mask)
                if outputs["loss"] is not None:
                    total_val_loss += outputs["loss"].item()
        
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch {epoch + 1} Tamamlandı | LR: {target_lr:.6g} | Ortalama Eğitim Loss: {avg_train_loss:.4f} | Doğrulama Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Yeni en iyi model kaydedildi. Doğrulama Loss: {best_val_loss:.4f}")
    
    print("\n--- Eğitim Tamamlandı! ---")

if __name__ == "__main__":
    main()
