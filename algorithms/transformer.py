"""
Transformer-based Recommendation Systems - PyTorch
BERT4Rec ve SASRec benzeri sequential recommendation
TikTok, YouTube gibi sıralı öneri sistemleri için
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import warnings
warnings.filterwarnings('ignore')


class TransformerDataset(Dataset):
    """PyTorch Dataset for Transformer"""
    def __init__(self, sequences, targets):
        self.sequences = torch.LongTensor(sequences)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) for batch_first format
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model) when batch_first=True
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerRecommenderModel(nn.Module):
    """Transformer Model for Sequential Recommendation"""
    def __init__(self, n_items, d_model=128, n_heads=4, n_layers=2,
                 max_seq_length=50, dropout_rate=0.1):
        super(TransformerRecommenderModel, self).__init__()
        
        self.n_items = n_items
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Item embedding
        self.item_embedding = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, n_items)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x shape: (batch, seq_len)
        # Create mask for padding
        mask = (x == 0)
        
        # Store original input for sequence length calculation
        original_input = x
        
        # Embedding
        x = self.item_embedding(x) * math.sqrt(self.d_model)  # (batch, seq_len, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Transformer encoder
        # Note: PyTorch TransformerEncoder expects (seq_len, batch, d_model) or (batch, seq_len, d_model) with batch_first=True
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Use last non-padded position
        # Get the last non-zero position for each sequence from original input
        # Find last non-padding position (non-zero) in original input
        seq_lengths = (original_input != 0).sum(dim=1) - 1  # -1 because 0-indexed
        seq_lengths = torch.clamp(seq_lengths, min=0)
        
        # Extract last position
        batch_size = x.size(0)
        last_hidden = x[torch.arange(batch_size), seq_lengths]  # (batch, d_model)
        
        # Output
        output = self.output_layer(last_hidden)  # (batch, n_items)
        
        return output


class TransformerRecommender:
    """
    Transformer tabanlı sequential öneri sistemi
    Zaman ve sıra bilgisini kullanır (SVD bunu yapamaz)
    """
    
    def __init__(self, n_items=100, d_model=128, n_heads=4, n_layers=2,
                 max_seq_length=50, dropout_rate=0.1, learning_rate=0.001, random_state=42):
        """
        Args:
            n_items: Toplam ürün sayısı
            d_model: Model boyutu (embedding dimension)
            n_heads: Multi-head attention head sayısı
            n_layers: Transformer layer sayısı
            max_seq_length: Maksimum sequence uzunluğu
            dropout_rate: Dropout oranı
            learning_rate: Öğrenme hızı
            random_state: Rastgelelik için seed
        """
        self.n_items = n_items
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
    
    def _create_sequences(self, user_sequences):
        """
        Kullanıcı sequence'lerini model için hazırlar
        
        Args:
            user_sequences: Her kullanıcı için item sequence listesi
            
        Returns:
            (sequences, targets) tuple
        """
        sequences = []
        targets = []
        
        for seq in user_sequences:
            # Sequence'i max_seq_length'e pad et veya kırp
            if len(seq) > self.max_seq_length:
                seq = seq[-self.max_seq_length:]
            else:
                seq = [0] * (self.max_seq_length - len(seq)) + seq
            
            # Her pozisyon için target oluştur (next item prediction)
            for i in range(1, len(seq)):
                if seq[i] != 0:  # Padding değilse
                    # Input: seq[:i] padded, Target: seq[i]
                    input_seq = seq[:i] + [0] * (self.max_seq_length - i)
                    sequences.append(input_seq)
                    targets.append(seq[i])
        
        return np.array(sequences), np.array(targets)
    
    def fit(self, user_sequences, epochs=10, batch_size=64, validation_split=0.2, verbose=1):
        """
        Modeli eğitir
        
        Args:
            user_sequences: Her kullanıcı için item sequence listesi
            epochs: Eğitim iterasyon sayısı
            batch_size: Batch boyutu
            validation_split: Validation set oranı
            verbose: Verbose modu
        """
        # Sequences oluştur
        X, y = self._create_sequences(user_sequences)
        
        # Train-validation split
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        split_idx = int(n_samples * (1 - validation_split))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices] if len(val_indices) > 0 else None
        y_val = y[val_indices] if len(val_indices) > 0 else None
        
        # Model oluştur
        self.model = TransformerRecommenderModel(
            self.n_items, self.d_model, self.n_heads, self.n_layers,
            self.max_seq_length, self.dropout_rate
        ).to(self.device)
        
        # Optimizer ve loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Dataset ve DataLoader
        train_dataset = TransformerDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = TransformerDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Training loop
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for sequences_batch, targets_batch in train_loader:
                sequences_batch = sequences_batch.to(self.device)
                targets_batch = targets_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(sequences_batch)
                loss = criterion(predictions, targets_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Accuracy
                _, predicted = torch.max(predictions.data, 1)
                train_total += targets_batch.size(0)
                train_correct += (predicted == targets_batch).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for sequences_batch, targets_batch in val_loader:
                        sequences_batch = sequences_batch.to(self.device)
                        targets_batch = targets_batch.to(self.device)
                        
                        predictions = self.model(sequences_batch)
                        loss = criterion(predictions, targets_batch)
                        val_loss += loss.item()
                        
                        # Accuracy
                        _, predicted = torch.max(predictions.data, 1)
                        val_total += targets_batch.size(0)
                        val_correct += (predicted == targets_batch).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = val_correct / val_total if val_total > 0 else 0
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                if verbose >= 1:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                if verbose >= 1:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        
        return history
    
    def predict_next(self, sequence):
        """
        Bir sequence için sonraki item'ı tahmin eder
        
        Args:
            sequence: Item sequence (list)
            
        Returns:
            (item_indices, probabilities) tuple
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        # Sequence'i hazırla
        if len(sequence) > self.max_seq_length:
            sequence = sequence[-self.max_seq_length:]
        else:
            sequence = [0] * (self.max_seq_length - len(sequence)) + sequence
        
        # Tahmin
        self.model.eval()
        with torch.no_grad():
            sequence_tensor = torch.LongTensor([sequence]).to(self.device)
            predictions = self.model(sequence_tensor)
            probs = torch.softmax(predictions, dim=1).cpu().numpy()[0]
        
        # En yüksek olasılıklı item'ları bul
        top_indices = np.argsort(probs)[::-1][:10]
        top_probs = probs[top_indices]
        
        return top_indices, top_probs
    
    def recommend(self, user_sequence, n_recommendations=10):
        """
        Bir kullanıcının sequence'ine göre öneriler üretir
        
        Args:
            user_sequence: Kullanıcının geçmiş item sequence'i
            n_recommendations: Önerilecek item sayısı
            
        Returns:
            (item_indices, probabilities) tuple
        """
        return self.predict_next(user_sequence)
