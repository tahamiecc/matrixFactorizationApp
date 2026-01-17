"""
Graph Neural Networks (GNN) for Recommendation
Modern öneri sistemlerinin zirvesi - Pinterest, Uber Eats kullanıyor
Veriyi tablo değil, ağ (graph) olarak görür
"""

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch ve PyTorch Geometric yüklü değil. GNN modülü kullanılamaz.")

import warnings
warnings.filterwarnings('ignore')


class GNNRecommender:
    """
    Graph Neural Network tabanlı öneri sistemi
    Kullanıcı-ürün ilişkilerini graph olarak modeler
    """
    
    def __init__(self, embedding_dim=64, hidden_dim=128, n_layers=2, 
                 learning_rate=0.001, random_state=42):
        """
        Args:
            embedding_dim: Embedding boyutu
            hidden_dim: Gizli katman boyutu
            n_layers: GNN layer sayısı
            learning_rate: Öğrenme hızı
            random_state: Rastgelelik için seed
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch ve PyTorch Geometric gerekli. pip install torch torch-geometric")
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = None
        self.graph_data = None
        self.n_users = None
        self.n_items = None
        
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    def _build_graph(self, rating_matrix):
        """
        Rating matrisinden graph oluşturur
        
        Args:
            rating_matrix: Rating matrisi (n_users x n_items)
            
        Returns:
            PyTorch Geometric Data objesi
        """
        self.n_users, self.n_items = rating_matrix.shape
        n_nodes = self.n_users + self.n_items
        
        # Edge'leri oluştur (rating varsa edge var)
        edge_index = []
        edge_attr = []
        
        for u in range(self.n_users):
            for i in range(self.n_items):
                if not np.isnan(rating_matrix[u, i]) and rating_matrix[u, i] > 0:
                    # User -> Item edge
                    edge_index.append([u, self.n_users + i])
                    edge_attr.append(rating_matrix[u, i])
                    
                    # Item -> User edge (undirected graph)
                    edge_index.append([self.n_users + i, u])
                    edge_attr.append(rating_matrix[u, i])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Node features (basit one-hot veya öğrenilebilir embedding)
        x = torch.eye(n_nodes, dtype=torch.float)
        
        # Graph data
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return graph_data
    
    def _build_model(self):
        """GNN modelini oluşturur"""
        class GNNModel(nn.Module):
            def __init__(self, n_nodes, embedding_dim, hidden_dim, n_layers):
                super(GNNModel, self).__init__()
                self.embedding = nn.Embedding(n_nodes, embedding_dim)
                self.convs = nn.ModuleList()
                
                # GCN layers
                self.convs.append(GCNConv(embedding_dim, hidden_dim))
                for _ in range(n_layers - 1):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                
                self.fc = nn.Linear(hidden_dim, 1)
            
            def forward(self, x, edge_index):
                x = self.embedding(torch.arange(x.size(0), device=x.device))
                
                for conv in self.convs:
                    x = conv(x, edge_index)
                    x = F.relu(x)
                    x = F.dropout(x, training=self.training)
                
                return x
        
        n_nodes = self.n_users + self.n_items
        model = GNNModel(n_nodes, self.embedding_dim, self.hidden_dim, self.n_layers)
        return model
    
    def fit(self, rating_matrix, epochs=50, lr=0.001, verbose=True):
        """
        Modeli eğitir
        
        Args:
            rating_matrix: Rating matrisi
            epochs: Eğitim iterasyon sayısı
            lr: Öğrenme hızı
            verbose: Verbose modu
        """
        # Graph oluştur
        self.graph_data = self._build_graph(rating_matrix)
        
        # Model oluştur
        self.model = self._build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Basit eğitim (link prediction)
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Node embeddings
            node_embeddings = self.model(
                self.graph_data.x,
                self.graph_data.edge_index
            )
            
            # User ve item embeddings'i ayır
            user_embeddings = node_embeddings[:self.n_users]
            item_embeddings = node_embeddings[self.n_users:]
            
            # Rating tahminleri (basit dot product)
            predictions = torch.mm(user_embeddings, item_embeddings.t())
            
            # Loss (sadece mevcut rating'ler için)
            mask = ~torch.isnan(torch.tensor(rating_matrix, dtype=torch.float))
            target = torch.tensor(rating_matrix, dtype=torch.float)
            target[~mask] = 0
            
            loss = criterion(predictions[mask], target[mask])
            loss.backward()
            optimizer.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    def predict(self, user_idx, item_idx):
        """
        Belirli bir kullanıcı-ürün çifti için rating tahmin eder
        
        Args:
            user_idx: Kullanıcı indeksi
            item_idx: Ürün indeksi
            
        Returns:
            Tahmin edilen rating
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model(
                self.graph_data.x,
                self.graph_data.edge_index
            )
            
            user_emb = node_embeddings[user_idx]
            item_emb = node_embeddings[self.n_users + item_idx]
            
            prediction = torch.dot(user_emb, item_emb).item()
        
        return prediction
    
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
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş.")
        
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model(
                self.graph_data.x,
                self.graph_data.edge_index
            )
            
            user_emb = node_embeddings[user_idx]
            item_embeddings = node_embeddings[self.n_users:]
            
            predictions = torch.mm(user_emb.unsqueeze(0), item_embeddings.t()).squeeze().numpy()
        
        if exclude_rated and rating_matrix is not None:
            rated_items = ~np.isnan(rating_matrix[user_idx])
            predictions[rated_items] = -np.inf
        
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        top_ratings = predictions[top_indices]
        
        return top_indices, top_ratings

