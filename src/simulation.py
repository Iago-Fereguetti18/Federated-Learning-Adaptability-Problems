import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from src.data_loader import carregar_uci_har, criar_clientes_federados
from src.models import ModeloMLP_HAR, ModeloCNN_HAR
from src.client import ClienteFederado
from src.server import ServidorFederado

def rodar_simulacao_interativa(
    n_rodadas, rodada_ftl, tipo_modelo="MLP", n_clientes=5, 
    tipo_teste="Espacial", rodada_drift=15, usar_casado=False
):
    X, y, subjects = carregar_uci_har()
    
    usar_het_inicial = (tipo_teste == "Espacial")
    loaders = criar_clientes_federados(X, y, subjects, n_clientes=n_clientes, heterogeneidade_real=usar_het_inicial)
    
    def get_modelo():
        if tipo_modelo == "CNN (Complexo)":
            return ModeloCNN_HAR(input_dim=561, num_classes=6)
        else:
            return ModeloMLP_HAR(input_dim=561, num_classes=6)

    servidor = ServidorFederado(get_modelo())
    agentes_clientes = [ClienteFederado(id_c, load) for id_c, load in loaders.items()]
    for agente in agentes_clientes: agente.modelo = get_modelo()
        
    for rodada in range(n_rodadas):
        congelar_agora = False
        status_ftl_base = "Aquecimento (Global)"
        
        if rodada >= rodada_ftl:
            congelar_agora = True
            status_ftl_base = "PerFit (Base Congelada)"
            
        # O Terremoto Temporal (Acontece nos dados, sem o modelo saber o momento exato)
        if tipo_teste == "Temporal" and rodada == (rodada_drift - 1):
            for i, agente in enumerate(agentes_clientes):
                if i % 2 == 0: 
                    X_tens, y_tens = agente.loader.dataset.tensors
                    
                    # --- A MÁGICA NOVA: EMBARALHAR AS COLUNAS ---
                    # Isso destrói a lógica da Base congelada, obrigando ela a ser descongelada!
                    indices_baguncados = torch.randperm(X_tens.shape[2])
                    novo_X = X_tens[:, :, indices_baguncados]
                    
                    novo_dataset = TensorDataset(novo_X, y_tens)
                    agente.loader = DataLoader(novo_dataset, batch_size=32, shuffle=True)
        
        pesos_para_agregar = []
        metricas_rodada = []
        
        for agente in agentes_clientes:
            pesos_globais = servidor.modelo_global.state_dict()
            agente.modelo.load_state_dict(pesos_globais)
            agente.modelo.to(agente.device)
            
            # --- ATUALIZAÇÃO 1: Desempacotar Acurácia e Erro ---
            acc_global, erro_global = agente.avaliar()
            
            epocas_treino = 3 
            if congelar_agora: epocas_treino = 5
            
            # Trava de segurança: Se você não ativou a defesa no painel,
            # a gente "amordaça" o CUSUM pra ele não disparar e a CNN quebrar.
            if not usar_casado:
                agente.cusum_g = 0.0
                agente.em_adaptacao = False
            
            # --- ATUALIZAÇÃO 2: Receber o status do Alarme do Cliente ---
            novos_pesos, em_adaptacao = agente.treinar_personalizado(
                pesos_globais, epocas=epocas_treino, congelar_base=congelar_agora
            )
            
            acc_local, erro_local = agente.avaliar()
            pesos_para_agregar.append(novos_pesos)
            
            # Define a mensagem bonita pro painel
            status_final = "🚨 CUSUM: Descongelado p/ Adaptação" if em_adaptacao else status_ftl_base
            
            metricas_rodada.append({
                "Rodada": rodada + 1,
                "Cliente ID": agente.id,
                "Global": acc_global,
                "Local": acc_local,
                "Modo": status_final
            })
            
        servidor.agregar_pesos(pesos_para_agregar)
        yield pd.DataFrame(metricas_rodada)