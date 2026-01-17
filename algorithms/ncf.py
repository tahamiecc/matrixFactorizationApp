"""
Neural Collaborative Filtering (NCF) Implementation - PyTorch
Modern Deep Learning tabanlı öneri sistemi
SVD'nin doğrusal olmayan (non-linear) versiyonu
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class NCFDataset(Dataset):
    """PyTorch Dataset for NCF"""
    def __init__(self, users, items, labels):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


class NCFModel(nn.Module):
    """NCF Neural Network Model"""
    def __init__(self, n_users, n_items, n_factors=50, hidden_layers=[64, 32, 16], dropout_rate=0.2):
        super(NCFModel, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        
        # MLP layers
        layers_list = []
        input_dim = n_factors * 2  # user + item embeddings
        
        for units in hidden_layers:
            layers_list.append(nn.Linear(input_dim, units))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(dropout_rate))
            layers_list.append(nn.BatchNorm1d(units))
            input_dim = units
        
        # Output layer
        layers_list.append(nn.Linear(input_dim, 1))
        layers_list.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers_list)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate
        concat = torch.cat([user_emb, item_emb], dim=1)
        
        # MLP
        output = self.mlp(concat)
        return output.squeeze()


class NCFRecommender:
    """
    Neural Collaborative Filtering - Deep Learning tabanlı öneri sistemi
    SVD'nin modern, doğrusal olmayan versiyonu
    """
    
    def __init__(self, n_factors=50, hidden_layers=[64, 32, 16], 
                 dropout_rate=0.2, learning_rate=0.001, random_state=42):
        """
        Args:
            n_factors: Latent faktör sayısı (embedding boyutu)
            hidden_layers: Gizli katmanların nöron sayıları
            dropout_rate: Dropout oranı (overfitting önleme)
            learning_rate: Öğrenme hızı
            random_state: Rastgelelik için seed
        """
        self.n_factors = n_factors
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = None
        self.n_users = None
        self.n_items = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Random seed
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
    
    def fit(self, rating_matrix, epochs=10, batch_size=256, validation_split=0.2, 
            implicit=True, verbose=1):
        """
        Modeli eğitir
        
        Args:
            rating_matrix: Rating matrisi (n_users x n_items)
            epochs: Eğitim iterasyon sayısı
            batch_size: Batch boyutu
            validation_split: Validation set oranı
            implicit: Implicit feedback (True) veya explicit (False)
            verbose: Verbose modu
        """
        self.n_users, self.n_items = rating_matrix.shape
        
        # Model oluştur
        self.model = NCFModel(
            self.n_users, self.n_items, 
            self.n_factors, self.hidden_layers, self.dropout_rate
        ).to(self.device)
        
        # Optimizer ve loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Pozitif ve negatif örnekler oluştur
        if implicit:
            # Implicit feedback: Rating varsa pozitif (1), yoksa negatif (0)
            positive_pairs = []
            negative_pairs = []
            
            for u in range(self.n_users):
                for i in range(self.n_items):
                    if not np.isnan(rating_matrix[u, i]) and rating_matrix[u, i] > 0:
                        positive_pairs.append((u, i, 1))
                    else:
                        negative_pairs.append((u, i, 0))
            
            # Negatif örnekleri örnekle (pozitif sayısı kadar)
            np.random.seed(self.random_state)
            if len(negative_pairs) > len(positive_pairs):
                negative_indices = np.random.choice(
                    len(negative_pairs), 
                    size=len(positive_pairs), 
                    replace=False
                )
                negative_pairs = [negative_pairs[i] for i in negative_indices]
            
            # Birleştir
            all_pairs = positive_pairs + negative_pairs
            np.random.shuffle(all_pairs)
            
            users = np.array([p[0] for p in all_pairs])
            items = np.array([p[1] for p in all_pairs])
            labels = np.array([p[2] for p in all_pairs])
        else:
            # Explicit feedback: Rating değerlerini normalize et (0-1)
            pairs = []
            for u in range(self.n_users):
                for i in range(self.n_items):
                    if not np.isnan(rating_matrix[u, i]):
                        normalized_rating = (rating_matrix[u, i] - 1) / 4.0  # 1-5 -> 0-1
                        pairs.append((u, i, normalized_rating))
            
            users = np.array([p[0] for p in pairs])
            items = np.array([p[1] for p in pairs])
            labels = np.array([p[2] for p in pairs])
        
        # Train-validation split
        if validation_split > 0:
            users_train, users_val, items_train, items_val, labels_train, labels_val = \
                train_test_split(users, items, labels, test_size=validation_split, 
                               random_state=self.random_state)
        else:
            users_train, items_train, labels_train = users, items, labels
            users_val, items_val, labels_val = None, None, None
        
        # Dataset ve DataLoader
        train_dataset = NCFDataset(users_train, items_train, labels_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if users_val is not None:
            val_dataset = NCFDataset(users_val, items_val, labels_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Training loop
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_users, batch_items, batch_labels in train_loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_users, batch_items)
                loss = criterion(predictions, batch_labels)
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
                    for batch_users, batch_items, batch_labels in val_loader:
                        batch_users = batch_users.to(self.device)
                        batch_items = batch_items.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        predictions = self.model(batch_users, batch_items)
                        loss = criterion(predictions, batch_labels)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                if verbose >= 1:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                if verbose >= 1:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}')
        
        return history
    
    def predict(self, user_idx, item_idx):
        """
        Belirli bir kullanıcı-ürün çifti için rating tahmin eder
        
        Args:
            user_idx: Kullanıcı indeksi
            item_idx: Ürün indeksi
            
        Returns:
            Tahmin edilen rating (0-1 arası, explicit için 1-5'e çevrilmeli)
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            item_tensor = torch.LongTensor([item_idx]).to(self.device)
            prediction = self.model(user_tensor, item_tensor).cpu().item()
        
        return prediction
    
    def predict_all(self, explicit_scale=True):
        """
        Tüm kullanıcı-ürün çiftleri için rating tahmin eder
        
        Args:
            explicit_scale: True ise 1-5 aralığına ölçekle
            
        Returns:
            Tahmin matrisi (n_users x n_items)
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        self.model.eval()
        predictions = np.zeros((self.n_users, self.n_items))
        batch_size = 1000
        
        with torch.no_grad():
            for u in range(self.n_users):
                users_batch = torch.LongTensor([u] * self.n_items).to(self.device)
                items_batch = torch.LongTensor(range(self.n_items)).to(self.device)
                
                batch_preds = self.model(users_batch, items_batch).cpu().numpy()
                
                if explicit_scale:
                    # 0-1 aralığını 1-5 aralığına ölçekle
                    predictions[u] = 1 + batch_preds * 4
                else:
                    predictions[u] = batch_preds
        
        return predictions
    
    def recommend(self, user_idx, n_recommendations=10, exclude_rated=True, 
                  rating_matrix=None, explicit_scale=True):
        """
        Bir kullanıcı için öneriler üretir
        
        Args:
            user_idx: Kullanıcı indeksi
            n_recommendations: Önerilecek item sayısı
            exclude_rated: Zaten rating verilen item'ları hariç tut
            rating_matrix: Orijinal rating matrisi
            explicit_scale: True ise 1-5 aralığına ölçekle
            
        Returns:
            (item_indices, predicted_ratings) tuple
        """
        predictions = self.predict_all(explicit_scale=explicit_scale)[user_idx]
        
        if exclude_rated and rating_matrix is not None:
            rated_items = ~np.isnan(rating_matrix[user_idx])
            predictions[rated_items] = -np.inf
        
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        top_ratings = predictions[top_indices]
        
        return top_indices, top_ratings
    
    def evaluate(self, test_matrix, explicit_scale=True):
        """
        Test seti üzerinde model performansını değerlendirir
        
        Args:
            test_matrix: Test rating matrisi
            explicit_scale: True ise 1-5 aralığına ölçekle
            
        Returns:
            RMSE değeri
        """
        predictions = self.predict_all(explicit_scale=explicit_scale)
        mask = ~np.isnan(test_matrix)
        
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(
            test_matrix[mask],
            predictions[mask]
        ))
        return rmse
