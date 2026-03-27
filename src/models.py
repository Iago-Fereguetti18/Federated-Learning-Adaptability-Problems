import torch
import torch.nn as nn

# --- MODELO 1: MLP (O "3NN" do artigo - Leve) ---
class ModeloMLP_HAR(nn.Module):
    def __init__(self, input_dim=561, num_classes=6):
        super(ModeloMLP_HAR, self).__init__()
        
        # Base: Extração de características
        self.base = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Head: Classificação (Personalizável no FTL)
        self.head = nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() > 2: x = x.squeeze(1)
        x = self.base(x)
        x = self.head(x)
        return x

# --- MODELO 2: CNN (O Modelo Complexo do artigo) ---
class ModeloCNN_HAR(nn.Module):
    def __init__(self, input_dim=561, num_classes=6):
        super(ModeloCNN_HAR, self).__init__()
        
        # Base: Camadas Convolucionais (Busca padrões na sequência)
        # Tratamos as 561 features como uma sequência temporal de tamanho 561
        self.base = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        
        # O tamanho da saída da base depende da conta de convolução.
        # Para input 561, a saída achatada é aproximadamente 8768
        self.dim_flatten = 8768 
        
        # Head: Classificador
        self.head = nn.Sequential(
            nn.Linear(self.dim_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Garante que entrada seja [Batch, 1, 561] para Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.base(x)
        x = self.head(x)
        return x