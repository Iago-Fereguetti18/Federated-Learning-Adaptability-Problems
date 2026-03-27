import torch
import copy

class ServidorFederado:
    def __init__(self, modelo_global):
        self.modelo_global = modelo_global
        
    def agregar_pesos(self, lista_pesos):
        """
        Recebe apenas uma lista de pesos (state_dicts).
        Faz a média simples (FedAvg).
        """
        if not lista_pesos:
            return

        # Pega o primeiro como base
        pesos_media = copy.deepcopy(lista_pesos[0])
        
        # Soma o resto
        for key in pesos_media.keys():
            for i in range(1, len(lista_pesos)):
                pesos_media[key] += lista_pesos[i][key]
            # Divide pelo total
            pesos_media[key] = torch.div(pesos_media[key], len(lista_pesos))
            
        self.modelo_global.load_state_dict(pesos_media)