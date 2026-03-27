import torch
from src.data_loader import carregar_uci_har, criar_clientes_federados
from src.models import ModeloMLP_HAR
from src.client import ClienteFederado
from src.server import ServidorFederado

def main():
    print("--- 1. Carregando Dados (UCI HAR) ---")
    X, y, subjects = carregar_uci_har()
    
    # Cria 5 clientes com dados Non-IID (baseados em usuários reais)
    loaders = criar_clientes_federados(X, y, subjects, n_clientes=5)
    
    print("\n--- 2. Inicializando Rede Federada ---")
    modelo_inicial = ModeloMLP_HAR(input_dim=561, num_classes=6)
    servidor = ServidorFederado(modelo_inicial)
    
    # Criando agentes clientes
    agentes_clientes = []
    for id_cliente, loader in loaders.items():
        agente = ClienteFederado(id_cliente, loader)
        # Importante: dar uma instância do modelo pro cliente
        agente.modelo = ModeloMLP_HAR(input_dim=561, num_classes=6) 
        agentes_clientes.append(agente)
        
    # --- 3. Loop de Simulação (Rodadas) ---
    num_rodadas = 20 
    
    for rodada in range(num_rodadas):
        print(f"\n>>> RODADA {rodada + 1} <<<")
        
        # Nas primeiras 10 rodadas: Treina TUDO (Global tenta entender a bagunça)
        # Da 11 em diante: Congela (FTL entra pra salvar a Ovelha Negra)
        congelar_agora = False
        if rodada >= 10: 
             congelar_agora = True
             print("--- [FTL ATIVADO] Congelando Bases para Personalização ---")
        
        pesos_para_agregar = []
        
        # Cada cliente baixa o global e treina
        for agente in agentes_clientes:
            # Pega o estado atual do servidor
            pesos_globais = servidor.modelo_global.state_dict()
            
            # Treina localmente (FTL - Base Congelada)
            novos_pesos = agente.treinar_personalizado(pesos_globais)
            pesos_para_agregar.append(novos_pesos)
            
        # Servidor agrega
        servidor.agregar_pesos(pesos_para_agregar)
        
    print("\n--- Simulação Concluída! ---")
    # Aqui você poderia adicionar um código para testar a acurácia final

if __name__ == "__main__":
    main()