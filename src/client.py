import torch
import torch.nn as nn
import torch.optim as optim

class ClienteFederado:
    def __init__(self, id_cliente, data_loader):
        self.id = id_cliente
        self.loader = data_loader
        self.modelo = None 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- MOTOR DO CUSUM (DETECÇÃO DE DRIFT DO CASADO) ---
        self.cusum_g = 0.0          
        self.cusum_h = 15.0         
        self.cusum_v = 2.0          
        self.erro_medio_hist = 0.0  
        self.rodadas_treino = 0     
        
        self.em_adaptacao = False
        self.rodadas_restantes_adaptacao = 0

    def avaliar(self, loader_teste=None):
        if loader_teste is None: loader_teste = self.loader
            
        self.modelo.eval()
        corretos, total = 0, 0
        with torch.no_grad():
            for x, y in loader_teste:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.modelo(x)
                _, preditos = torch.max(outputs.data, 1)
                total += y.size(0)
                corretos += (preditos == y).sum().item()
        
        acuracia = 100 * corretos / total
        erro_atual = 100.0 - acuracia 
        
        # Retorna exatas 2 coisas!
        return acuracia, erro_atual

    def atualizar_cusum(self, erro_atual):
        self.rodadas_treino += 1
        
        if self.rodadas_treino <= 5:
            self.erro_medio_hist = (self.erro_medio_hist * (self.rodadas_treino - 1) + erro_atual) / self.rodadas_treino
        else:
            desvio = erro_atual - self.erro_medio_hist - self.cusum_v
            self.cusum_g = max(0.0, self.cusum_g + desvio)
            
            if self.cusum_g > self.cusum_h:
                print(f"🚨 [Cliente {self.id}] CUSUM DISPAROU! Drift detectado (Erro acumulado: {self.cusum_g:.1f})")
                self.em_adaptacao = True
                self.rodadas_restantes_adaptacao = 3 
                self.cusum_g = 0.0 
                self.erro_medio_hist = erro_atual 

    def treinar_personalizado(self, modelo_global_state, epocas=5, congelar_base=False):
        self.modelo.load_state_dict(modelo_global_state)
        self.modelo.to(self.device)
        self.modelo.train()
        
        _, erro_atual = self.avaliar()
        self.atualizar_cusum(erro_atual)

        if self.em_adaptacao:
            congelar_base = False 
            self.rodadas_restantes_adaptacao -= 1
            if self.rodadas_restantes_adaptacao <= 0:
                self.em_adaptacao = False 

        if congelar_base:
            for param in self.modelo.base.parameters(): param.requires_grad = False
            for param in self.modelo.head.parameters(): param.requires_grad = True
            lr_atual = 0.05
        else:
            for param in self.modelo.parameters(): param.requires_grad = True
            lr_atual = 0.01

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.modelo.parameters()), lr=lr_atual, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        for epoca in range(epocas):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.modelo(x), y)
                loss.backward()
                optimizer.step()
                
        # O segredo tá aqui: devolvendo SÓ os pesos e o status de adaptação (2 coisas!)
        return self.modelo.state_dict(), self.em_adaptacao