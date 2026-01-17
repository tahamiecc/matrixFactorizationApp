"""
Factorization Machines (FM) & DeepFM Implementation - PyTorch
Context-aware öneri sistemi - yan bilgileri (context) kullanır
Reklam tıklama tahmini (CTR) için optimize edilmiş
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class FMDataset(Dataset):
    """PyTorch Dataset for FM"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FMModel(nn.Module):
    """Factorization Machine Model"""
    def __init__(self, n_features, n_factors=10):
        super(FMModel, self).__init__()
        self.n_features = n_features
        self.n_factors = n_factors
        
        # Linear term
        self.linear = nn.Linear(n_features, 1, bias=True)
        
        # Factorization term (embedding)
        self.embedding = nn.Linear(n_features, n_factors, bias=False)
    
    def forward(self, x):
        # Linear term
        linear_term = self.linear(x)
        
        # FM term: 0.5 * (sum_square - square_sum)
        embedding = self.embedding(x)  # (batch, n_factors)
        embedding_squared = embedding.pow(2)  # (batch, n_factors)
        
        x_squared = x.pow(2)  # (batch, n_features)
        embedding_from_squared = self.embedding(x_squared)  # (batch, n_factors)
        
        # Sum of squares - Square of sum
        sum_square = embedding_squared.sum(dim=1, keepdim=True)  # (batch, 1)
        square_sum = embedding_from_squared.sum(dim=1, keepdim=True)  # (batch, 1)
        
        fm_term = 0.5 * (sum_square - square_sum)
        
        # Output
        output = linear_term + fm_term
        output = torch.sigmoid(output)
        
        return output.squeeze()


class FactorizationMachine:
    """
    Factorization Machine - Context-aware öneri sistemi
    Sadece kullanıcı-ürün ID'sine değil, yan bilgilere de bakar
    """
    
    def __init__(self, n_factors=10, learning_rate=0.001, random_state=42):
        """
        Args:
            n_factors: Latent faktör sayısı
            learning_rate: Öğrenme hızı
            random_state: Rastgelelik için seed
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.n_features = None
        self.max_user_id = None
        self.max_item_id = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
    
    def _create_features(self, user_ids, item_ids, context_features=None):
        """
        Özellik vektörleri oluşturur
        
        Args:
            user_ids: Kullanıcı ID'leri
            item_ids: Ürün ID'leri
            context_features: Yan bilgiler (saat, cihaz, vb.)
            
        Returns:
            Özellik matrisi
        """
        n_samples = len(user_ids)
        features = []
        
        for i in range(n_samples):
            feature_vec = np.zeros(self.n_features)
            
            # Kullanıcı one-hot
            feature_vec[user_ids[i]] = 1
            
            # Ürün one-hot
            feature_vec[self.max_user_id + item_ids[i]] = 1
            
            # Context features (yan bilgiler)
            if context_features is not None:
                start_idx = self.max_user_id + self.max_item_id
                feature_vec[start_idx:start_idx+len(context_features[i])] = context_features[i]
            
            features.append(feature_vec)
        
        return np.array(features)
    
    def fit(self, user_ids, item_ids, ratings, context_features=None,
            epochs=10, batch_size=256, validation_split=0.2, verbose=1):
        """
        Modeli eğitir
        
        Args:
            user_ids: Kullanıcı ID'leri
            item_ids: Ürün ID'leri
            ratings: Rating'ler (1-5 veya 0-1)
            context_features: Yan bilgiler (opsiyonel)
            epochs: Eğitim iterasyon sayısı
            batch_size: Batch boyutu
            validation_split: Validation set oranı
            verbose: Verbose modu
        """
        # Özellik sayısını belirle
        self.max_user_id = np.max(user_ids) + 1
        self.max_item_id = np.max(item_ids) + 1
        context_dim = context_features.shape[1] if context_features is not None else 0
        self.n_features = self.max_user_id + self.max_item_id + context_dim
        
        # Özellik vektörleri oluştur
        X = self._create_features(user_ids, item_ids, context_features)
        
        # Rating'leri normalize et (0-1)
        if ratings.max() > 1:
            y = (ratings - 1) / 4.0  # 1-5 -> 0-1
        else:
            y = ratings
        
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
        self.model = FMModel(self.n_features, self.n_factors).to(self.device)
        
        # Optimizer ve loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Dataset ve DataLoader
        train_dataset = FMDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = FMDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Training loop
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        predictions = self.model(X_batch)
                        loss = criterion(predictions, y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                if verbose >= 1:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                if verbose >= 1:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}')
        
        return history
    
    def predict(self, user_ids, item_ids, context_features=None):
        """
        Tahmin yapar
        
        Args:
            user_ids: Kullanıcı ID'leri
            item_ids: Ürün ID'leri
            context_features: Yan bilgiler (opsiyonel)
            
        Returns:
            Tahmin edilen rating'ler (0-1 aralığında)
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        X = self._create_features(user_ids, item_ids, context_features)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions


class DeepFMModel(nn.Module):
    """Deep Factorization Machine Model"""
    def __init__(self, n_features, n_factors=10, hidden_layers=[64, 32], dropout_rate=0.2):
        super(DeepFMModel, self).__init__()
        self.n_features = n_features
        self.n_factors = n_factors
        
        # FM Component
        self.linear = nn.Linear(n_features, 1, bias=True)
        self.embedding = nn.Linear(n_features, n_factors, bias=False)
        
        # Deep Component
        layers_list = []
        input_dim = n_features
        for units in hidden_layers:
            layers_list.append(nn.Linear(input_dim, units))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(dropout_rate))
            layers_list.append(nn.BatchNorm1d(units))
            input_dim = units
        
        layers_list.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*layers_list)
    
    def forward(self, x):
        # Linear term
        linear_term = self.linear(x)
        
        # FM term
        embedding = self.embedding(x)
        embedding_squared = embedding.pow(2)
        x_squared = x.pow(2)
        embedding_from_squared = self.embedding(x_squared)
        
        sum_square = embedding_squared.sum(dim=1, keepdim=True)
        square_sum = embedding_from_squared.sum(dim=1, keepdim=True)
        fm_term = 0.5 * (sum_square - square_sum)
        
        # Deep term
        deep_term = self.deep(x)
        
        # Combine
        output = linear_term + fm_term + deep_term
        output = torch.sigmoid(output)
        
        return output.squeeze()


class DeepFM:
    """
    Deep Factorization Machine - FM + Deep Learning
    Hem doğrusal (FM) hem de doğrusal olmayan (Deep) özellikleri öğrenir
    """
    
    def __init__(self, n_factors=10, hidden_layers=[64, 32], 
                 dropout_rate=0.2, learning_rate=0.001, random_state=42):
        """
        Args:
            n_factors: FM için latent faktör sayısı
            hidden_layers: Deep network gizli katmanları
            dropout_rate: Dropout oranı
            learning_rate: Öğrenme hızı
            random_state: Rastgelelik için seed
        """
        self.n_factors = n_factors
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.n_features = None
        self.max_user_id = None
        self.max_item_id = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
    
    def _create_features(self, user_ids, item_ids, context_features=None):
        """Özellik vektörleri oluşturur (FM ile aynı)"""
        n_samples = len(user_ids)
        features = []
        
        for i in range(n_samples):
            feature_vec = np.zeros(self.n_features)
            feature_vec[user_ids[i]] = 1
            feature_vec[self.max_user_id + item_ids[i]] = 1
            
            if context_features is not None:
                start_idx = self.max_user_id + self.max_item_id
                feature_vec[start_idx:start_idx+len(context_features[i])] = context_features[i]
            
            features.append(feature_vec)
        
        return np.array(features)
    
    def fit(self, user_ids, item_ids, ratings, context_features=None,
            epochs=10, batch_size=256, validation_split=0.2, verbose=1):
        """Modeli eğitir"""
        self.max_user_id = np.max(user_ids) + 1
        self.max_item_id = np.max(item_ids) + 1
        context_dim = context_features.shape[1] if context_features is not None else 0
        self.n_features = self.max_user_id + self.max_item_id + context_dim
        
        X = self._create_features(user_ids, item_ids, context_features)
        
        if ratings.max() > 1:
            y = (ratings - 1) / 4.0
        else:
            y = ratings
        
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
        
        self.model = DeepFMModel(
            self.n_features, self.n_factors, self.hidden_layers, self.dropout_rate
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        train_dataset = FMDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = FMDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['loss'].append(train_loss)
            
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        predictions = self.model(X_batch)
                        loss = criterion(predictions, y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                if verbose >= 1:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                if verbose >= 1:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}')
        
        return history
    
    def predict(self, user_ids, item_ids, context_features=None):
        """Tahmin yapar"""
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        X = self._create_features(user_ids, item_ids, context_features)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
