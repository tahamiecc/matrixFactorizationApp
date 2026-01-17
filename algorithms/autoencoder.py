"""
Autoencoder Implementation - PyTorch
SVD ve PCA'in Deep Learning karşılığı
Denoising Autoencoder ve Variational Autoencoder (VAE) için öneri sistemi
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class AutoencoderDataset(Dataset):
    """PyTorch Dataset for Autoencoder"""
    def __init__(self, X, X_noisy):
        self.X = torch.FloatTensor(X)
        self.X_noisy = torch.FloatTensor(X_noisy)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X_noisy[idx], self.X[idx]


class DenoisingAutoencoderModel(nn.Module):
    """Denoising Autoencoder Model"""
    def __init__(self, input_dim, encoding_dim=50):
        super(DenoisingAutoencoderModel, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DenoisingAutoencoder:
    """
    Denoising Autoencoder - Gürültü temizleme için
    SVD gürültü temizlemenin Deep Learning versiyonu
    """
    
    def __init__(self, encoding_dim=50, noise_factor=0.2, random_state=42):
        """
        Args:
            encoding_dim: Kodlama boyutu (latent space boyutu)
            noise_factor: Gürültü faktörü
            random_state: Rastgelelik için seed
        """
        self.encoding_dim = encoding_dim
        self.noise_factor = noise_factor
        self.random_state = random_state
        self.model = None
        self.input_dim = None
        self.X_min = None
        self.X_max = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
    
    def fit(self, X, epochs=50, batch_size=32, validation_split=0.2, verbose=1):
        """
        Modeli eğitir
        
        Args:
            X: Veri matrisi (n_samples x n_features)
            epochs: Eğitim iterasyon sayısı
            batch_size: Batch boyutu
            validation_split: Validation set oranı
            verbose: Verbose modu
        """
        X = np.array(X)
        self.input_dim = X.shape[1]
        
        # Veriyi normalize et (0-1)
        self.X_min = X.min()
        self.X_max = X.max()
        X_normalized = (X - self.X_min) / (self.X_max - self.X_min + 1e-8)
        
        # Gürültülü veri oluştur
        np.random.seed(self.random_state)
        X_noisy = X_normalized + self.noise_factor * np.random.normal(
            loc=0.0, scale=1.0, size=X_normalized.shape
        )
        X_noisy = np.clip(X_noisy, 0, 1)
        
        # Train-validation split
        n_samples = len(X_normalized)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        split_idx = int(n_samples * (1 - validation_split))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train = X_normalized[train_indices]
        X_train_noisy = X_noisy[train_indices]
        X_val = X_normalized[val_indices] if len(val_indices) > 0 else None
        X_val_noisy = X_noisy[val_indices] if len(val_indices) > 0 else None
        
        # Model oluştur
        self.model = DenoisingAutoencoderModel(self.input_dim, self.encoding_dim).to(self.device)
        
        # Optimizer ve loss
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()
        
        # Dataset ve DataLoader
        train_dataset = AutoencoderDataset(X_train, X_train_noisy)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = AutoencoderDataset(X_val, X_val_noisy)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Training loop
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_noisy_batch, X_batch in train_loader:
                X_noisy_batch = X_noisy_batch.to(self.device)
                X_batch = X_batch.to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.model(X_noisy_batch)
                loss = criterion(reconstructed, X_batch)
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
                    for X_noisy_batch, X_batch in val_loader:
                        X_noisy_batch = X_noisy_batch.to(self.device)
                        X_batch = X_batch.to(self.device)
                        
                        reconstructed = self.model(X_noisy_batch)
                        loss = criterion(reconstructed, X_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                if verbose >= 1:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                if verbose >= 1:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}')
        
        return history
    
    def denoise(self, X):
        """
        Gürültülü veriyi temizler
        
        Args:
            X: Gürültülü veri matrisi
            
        Returns:
            Temizlenmiş veri matrisi
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        X = np.array(X)
        X_normalized = (X - self.X_min) / (self.X_max - self.X_min + 1e-8)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normalized).to(self.device)
            X_denoised = self.model(X_tensor).cpu().numpy()
        
        # Orijinal aralığa geri dönüştür
        X_denoised = X_denoised * (self.X_max - self.X_min) + self.X_min
        
        return X_denoised


class VAEModel(nn.Module):
    """Variational Autoencoder Model"""
    def __init__(self, input_dim, latent_dim=50, intermediate_dim=200):
        super(VAEModel, self).__init__()
        
        # Encoder
        self.encoder_h = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.ReLU()
        )
        
        self.z_mean = nn.Linear(intermediate_dim, latent_dim)
        self.z_log_var = nn.Linear(intermediate_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder_h(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_decoded = self.decode(z)
        return x_decoded, z_mean, z_log_var


class VAERecommender:
    """
    Variational Autoencoder (VAE) - Öneri sistemi için
    SVD'nin probabilistic, Deep Learning versiyonu
    """
    
    def __init__(self, latent_dim=50, intermediate_dim=200, random_state=42):
        """
        Args:
            latent_dim: Latent space boyutu
            intermediate_dim: Ara katman boyutu
            random_state: Rastgelelik için seed
        """
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.random_state = random_state
        self.model = None
        self.n_users = None
        self.n_items = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
    
    def _vae_loss(self, x, x_decoded, z_mean, z_log_var):
        """VAE loss function (reconstruction + KL divergence)"""
        reconstruction_loss = nn.functional.binary_cross_entropy(x_decoded, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return reconstruction_loss + kl_loss
    
    def fit(self, rating_matrix, epochs=50, batch_size=128, validation_split=0.2, verbose=1):
        """
        Modeli eğitir
        
        Args:
            rating_matrix: Rating matrisi (n_users x n_items)
            epochs: Eğitim iterasyon sayısı
            batch_size: Batch boyutu
            validation_split: Validation set oranı
            verbose: Verbose modu
        """
        self.n_users, self.n_items = rating_matrix.shape
        
        # Rating'leri 0-1 aralığına normalize et
        rating_matrix_normalized = rating_matrix.copy()
        rating_matrix_normalized = np.nan_to_num(rating_matrix_normalized, nan=0.0)
        rating_matrix_normalized = (rating_matrix_normalized - 1) / 4.0  # 1-5 -> 0-1
        rating_matrix_normalized = np.clip(rating_matrix_normalized, 0, 1)
        
        # Train-validation split
        n_samples = len(rating_matrix_normalized)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        split_idx = int(n_samples * (1 - validation_split))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train = rating_matrix_normalized[train_indices]
        X_val = rating_matrix_normalized[val_indices] if len(val_indices) > 0 else None
        
        # Model oluştur
        self.model = VAEModel(self.n_items, self.latent_dim, self.intermediate_dim).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters())
        
        # Dataset ve DataLoader
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(X_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(X_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Training loop
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, _ in train_loader:
                X_batch = X_batch.to(self.device)
                
                optimizer.zero_grad()
                X_decoded, z_mean, z_log_var = self.model(X_batch)
                loss = self._vae_loss(X_batch, X_decoded, z_mean, z_log_var)
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
                    for X_batch, _ in val_loader:
                        X_batch = X_batch.to(self.device)
                        X_decoded, z_mean, z_log_var = self.model(X_batch)
                        loss = self._vae_loss(X_batch, X_decoded, z_mean, z_log_var)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                if verbose >= 1:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                if verbose >= 1:
                    print(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}')
        
        return history
    
    def predict(self, user_idx):
        """
        Bir kullanıcı için tüm ürünlerin rating'lerini tahmin eder
        
        Args:
            user_idx: Kullanıcı indeksi
            
        Returns:
            Tahmin edilen rating'ler (1-5 aralığında)
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        # Kullanıcının mevcut rating'lerini al (sıfır vektör)
        user_ratings = np.zeros(self.n_items)
        user_input = torch.FloatTensor(user_ratings.reshape(1, -1)).to(self.device)
        
        # VAE ile tahmin
        self.model.eval()
        with torch.no_grad():
            predictions, _, _ = self.model(user_input)
            predictions = predictions.cpu().numpy()[0]
        
        # 0-1'den 1-5'e ölçekle
        predictions = 1 + predictions * 4
        
        return predictions
    
    def recommend(self, user_idx, n_recommendations=10, exclude_rated=True, 
                  rating_matrix=None):
        """
        Bir kullanıcı için öneriler üretir
        
        Args:
            user_idx: Kullanıcı indeksi
            n_recommendations: Önerilecek item sayısı
            exclude_rated: Zaten rating verilen item'ları hariç tut
            rating_matrix: Orijinal rating matrisi
            
        Returns:
            (item_indices, predicted_ratings) tuple
        """
        predictions = self.predict(user_idx)
        
        if exclude_rated and rating_matrix is not None:
            rated_items = ~np.isnan(rating_matrix[user_idx])
            predictions[rated_items] = -np.inf
        
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        top_ratings = predictions[top_indices]
        
        return top_indices, top_ratings
