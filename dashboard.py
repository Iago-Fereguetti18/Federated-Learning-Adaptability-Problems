import streamlit as st
import pandas as pd
import altair as alt
import time
from src.simulation import rodar_simulacao_interativa

# --- Configuração Visual ---
st.set_page_config(page_title="PFL Simulator - IC", layout="wide", page_icon="🧠")

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color); 
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00c0f2;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 PerFit Framework & Defesa Casado")
st.markdown("Simulador de Aprendizado Federado - Testes de Heterogeneidade 🚀")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("🔬 Tipo de Experimento")
    tipo_teste = st.radio("Selecione a Dimensão:", ["Espacial", "Temporal"])
    
    st.divider()
    tipo_modelo = st.selectbox("Arquitetura", ["MLP (Simples - 3NN)", "CNN (Complexo)"])
    n_clientes = st.slider("Quantidade de Clientes", 2, 6, 4)
    n_rodadas = st.slider("Total de Rodadas", 10, 50, 25)
    
    usar_solucao_casado = False # Variável de segurança pra não dar erro
    
    if tipo_teste == "Espacial":
        st.success("🌍 Cenário Espacial: Metade da rede é heterogênea desde o Início.")
        rodada_ftl = st.slider("Ativar PerFit na Rodada:", 1, n_rodadas, 10)
        rodada_drift = 0 
    else:
        st.warning("⏱️ Cenário Temporal: Mudança brusca de hábito (Concept Drift).")
        rodada_drift = st.slider("Rodada do Terremoto (Drift):", 5, n_rodadas-5, 15)
        rodada_ftl = st.slider("Ativar PerFit na Rodada:", 1, rodada_drift, 5)
        
        st.markdown("### 🛠️ Defesa contra Drift (Casado)")
        usar_solucao_casado = st.checkbox("Ativar Solução Casado (Descongelamento Dinâmico)")

    st.divider()
    if st.button("🚀 INICIAR SIMULAÇÃO", type="primary"):
        st.session_state['rodando'] = True

# --- ÁREA PRINCIPAL ---
containers_clientes = []
for i in range(n_clientes):
    titulo = f"👤 Cliente {i}"
    if tipo_teste == "Espacial" and i % 2 == 0: 
        titulo += " (Heterogêneo desde o Início 🌍)"
    elif tipo_teste == "Temporal" and i % 2 == 0:
        titulo += " (Vai sofrer Mutação no Tempo ⏱️)"
        
    with st.container():
        st.subheader(titulo)
        col_graf, col_metrica = st.columns([3, 1])
        containers_clientes.append({"grafico": col_graf.empty(), "metricas": col_metrica.empty()})
        st.markdown("---")

# --- EXECUÇÃO ---
if st.session_state.get('rodando'):
    historico_df = pd.DataFrame() # <--- O SALVADOR DA PÁTRIA TÁ AQUI
    
    simulacao = rodar_simulacao_interativa(
        n_rodadas, rodada_ftl, tipo_modelo, n_clientes, 
        tipo_teste=tipo_teste, rodada_drift=rodada_drift, 
        usar_casado=usar_solucao_casado
    )
    
    bar_progresso = st.progress(0)
    
    for i, dados_rodada in enumerate(simulacao):
        bar_progresso.progress((i + 1) / n_rodadas)
        historico_df = pd.concat([historico_df, dados_rodada])
        
        for cliente_id in range(n_clientes):
            container = containers_clientes[cliente_id]
            df_cliente = historico_df[historico_df["Cliente ID"] == cliente_id]
            if df_cliente.empty: continue

            df_long = df_cliente.melt(id_vars=["Rodada"], value_vars=["Global", "Local"], var_name="Tipo", value_name="Acurácia")
            
            base = alt.Chart(df_long).encode(x=alt.X('Rodada:O', axis=alt.Axis(labels=False), title=None))
            
            linha = base.mark_line(point=True, strokeWidth=3).encode(
                y=alt.Y('Acurácia:Q', scale=alt.Scale(domain=[0, 100]), title="Acurácia (%)"),
                color=alt.Color('Tipo', legend=alt.Legend(title="Modelo"), scale=alt.Scale(domain=['Global', 'Local'], range=['#ff4b4b', '#00c0f2'])),
                tooltip=['Rodada', 'Tipo', 'Acurácia']
            )
            
            grafico_final = linha
            
            # Desenha a linha amarela se for temporal
            if tipo_teste == "Temporal" and cliente_id % 2 == 0:
                regra = alt.Chart(pd.DataFrame({'x': [rodada_drift]})).mark_rule(color='orange', strokeDash=[5, 5], strokeWidth=2).encode(x='x:O')
                grafico_final = linha + regra
            
            container["grafico"].altair_chart(grafico_final.properties(height=150), use_container_width=True)
            
            dado_atual = df_cliente.iloc[-1]
            diff = dado_atual["Local"] - dado_atual["Global"]
            with container["metricas"]:
                st.metric("Global", f"{dado_atual['Global']:.1f}%")
                st.metric("PerFit / Casado", f"{dado_atual['Local']:.1f}%", delta=f"{diff:.1f}%")
        time.sleep(0.05)
        
    st.session_state['rodando'] = False