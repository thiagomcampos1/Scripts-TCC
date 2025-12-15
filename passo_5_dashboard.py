import pandas as pd
import numpy as np
import sys
import warnings
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

LIMIAR_R2_CONFIANCA = 0.05
CONTAMINACAO_ISO_FOREST = 0.01 
LIMIAR_ERRO_STD = 2.5 

def encontrar_unidades_dinamicamente(df_colunas):
    unidades_encontradas = {} 
    
    colunas_kw_all = [
        col for col in df_colunas 
        if (
            'entalpico' in col.lower()       
            or 'kw termico' in col.lower()   
            or 'kw térmico' in col.lower()   
        )
        and 'fornecido' not in col.lower()   
        and 'carga' not in col.lower()       
        and 'total' not in col.lower()       
    ]
    
    colunas_valvula_all = [
        col for col in df_colunas 
        if 'abert. valvula' in col.lower() 
        or '% valvula' in col.lower()
    ]
    
    print("\n" + "="*60)
    print("DIAGNÓSTICO DE COLUNAS:")
    print(f"1. Todas as colunas encontradas no arquivo: {list(df_colunas)}")
    print(f"2. Colunas de Valvula identificadas: {colunas_valvula_all}")
    print(f"3. Colunas ALVO (Y) identificadas: {colunas_kw_all}")
    print("="*60 + "\n")

    for col_y in colunas_kw_all:

        unidade_id = col_y.split(' ')[0].strip() 
        if 'kw' in unidade_id.lower():
            unidade_id = unidade_id.lower().split('kw')[0].upper()
            
        col_x_valvula_encontrada = None
        
        for col_v in colunas_valvula_all:
            if col_v.startswith(unidade_id):
                col_x_valvula_encontrada = col_v
                break 

        if col_x_valvula_encontrada:
            if unidade_id not in unidades_encontradas:
                unidades_encontradas[unidade_id] = {
                    'col_y': col_y,
                    'col_x_valvula': col_x_valvula_encontrada
                }
            
    return unidades_encontradas

@st.cache_data 
def carregar_dados_iniciais(lote_id_str):
    file_entrada = f"resultadosmetricas{lote_id_str.upper()}.csv"
    file_metricas = f"resultados_metricas{lote_id_str.upper()}.csv"
    
    try:
        df_completo = pd.read_csv(file_entrada)
        df_completo['DateTime'] = pd.to_datetime(df_completo['DateTime'])
    except Exception as e:
        st.error(f"ERRO ao carregar '{file_entrada}': {e}"); return None, None, None
    try:
        df_metricas = pd.read_csv(file_metricas)
    except Exception as e:
        st.error(f"ERRO ao carregar '{file_metricas}'. Execute o Passo 2. {e}"); return None, None, None
        
    mapeamento_unidades = encontrar_unidades_dinamicamente(df_completo.columns)
    if not mapeamento_unidades:
        st.error("ERRO FATAL: Nenhum FCPB (kw/válvula) encontrado."); return None, None, None
        
    return df_completo, df_metricas, mapeamento_unidades

@st.cache_resource
def carregar_modelos_hibridos_e_preparar_dados(unidade_id, lote_id_str):

    df_completo, _, mapeamento_unidades = carregar_dados_iniciais(lote_id_str)
    if df_completo is None:
        st.error(f"Erro de cache: Não foi possível obter dados para {lote_id_str} dentro do cache de recursos.")
        return None

    nome_modelo_reg = f"modelo_{unidade_id.lower()}_{lote_id_str}.joblib"
    try:
        modelo_lr = joblib.load(nome_modelo_reg)
        nomes_corretos_features = modelo_lr.feature_names_in_
    except Exception as e:
        st.error(f"ERRO ao carregar '{nome_modelo_reg}': {e}. Execute o Passo 3."); return None
    
    nome_modelo_gb = f"modelo_gb_{unidade_id.lower()}_{lote_id_str}.joblib"
    try:
        modelo_gb = joblib.load(nome_modelo_gb)
        if not np.array_equal(nomes_corretos_features, modelo_gb.feature_names_in_):
            st.error("ERRO: As features do LR e do GB não são idênticas. Re-treine o Passo 3.")
            return None
    except Exception as e:
        st.error(f"ERRO ao carregar '{nome_modelo_gb}': {e}. Execute o Passo 3 (Híbrido)."); return None

    try:
        info = mapeamento_unidades[unidade_id]
        col_y = info['col_y']; col_x_valvula = info['col_x_valvula']
        
        df_unidade = df_completo[list(nomes_corretos_features) + [col_y, 'DateTime']].dropna()
        df_unidade = df_unidade.set_index('DateTime')
        
        X = df_unidade[nomes_corretos_features]
        Y_real = df_unidade[col_y]
    except Exception as e:
        st.error(f"ERRO ao preparar dados para {unidade_id}: {e}"); return None
    if X.empty:
        st.warning(f"Dados insuficientes para {unidade_id} após limpeza."); return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_iso = IsolationForest(contamination=CONTAMINACAO_ISO_FOREST, random_state=42) 
    model_iso.fit(X_scaled)

    Y_pred_lr = modelo_lr.predict(X); erros_lr = Y_real - Y_pred_lr
    limiar_erro_lr = erros_lr.std() * LIMIAR_ERRO_STD
    
    Y_pred_gb = modelo_gb.predict(X); erros_gb = Y_real - Y_pred_gb
    limiar_erro_gb = erros_gb.std() * LIMIAR_ERRO_STD
    
    df_analise = df_unidade.copy()
    df_analise['Consumo_Previsto_LR'] = Y_pred_lr; df_analise['Erro_kW_LR'] = erros_lr
    df_analise['Alerta_Perf_LR'] = erros_lr.abs() > limiar_erro_lr
    df_analise['Consumo_Previsto_GB'] = Y_pred_gb; df_analise['Erro_kW_GB'] = erros_gb
    df_analise['Alerta_Perf_GB'] = erros_gb.abs() > limiar_erro_gb 
    df_analise['Alerta_Op'] = (model_iso.predict(X_scaled) == -1)
    df_analise['Score_Operacional'] = model_iso.decision_function(X_scaled)
    limiar_score_op = df_analise['Score_Operacional'].quantile(CONTAMINACAO_ISO_FOREST) 
    
    return df_analise, limiar_erro_gb, limiar_score_op, modelo_lr, modelo_gb, model_iso, scaler, col_y, col_x_valvula

st.sidebar.title("Configuração da Análise")

lote_selecionado_radio = st.sidebar.radio(
    "Selecione o Lote para Análise:",
    ("A1", "A2"), 
    horizontal=True
)
lote_id_str_global = lote_selecionado_radio.lower()

st.title(f"Painel Híbrido de Detecção e Recomendação (Lote {lote_selecionado_radio})")

dados_iniciais = carregar_dados_iniciais(lote_id_str_global)
if dados_iniciais[0] is None:
    st.error(f"Falha ao carregar dados do Lote {lote_selecionado_radio}.")
    st.info("Por favor, execute os Passos 1, 2 e 3 primeiro.")
    st.stop()

df_completo_global, df_metricas_global, mapeamento_unidades_dinamico = dados_iniciais

if 'R² (Previsibilidade)' not in df_metricas_global.columns:
    st.sidebar.error("Coluna 'R² (Previsibilidade)' não encontrada.")
    unidades_confiaveis_ids = list(mapeamento_unidades_dinamico.keys()) 
else:
    unidades_confiaveis_ids = df_metricas_global[
        df_metricas_global['R² (Previsibilidade)'] >= LIMIAR_R2_CONFIANCA
    ]['Unidade'].tolist()
    
if not unidades_confiaveis_ids:
    st.sidebar.warning(f"Nenhuma unidade FCPB no Lote {lote_selecionado_radio} atingiu o limiar de R² para análise.")
    st.stop()
else:
    unidade_selecionada = st.sidebar.selectbox(
        f"Selecione a Unidade FCPB Confiável (Lote {lote_selecionado_radio}):",
        unidades_confiaveis_ids
    )
    
    if unidade_selecionada:
        
        dados_processados = carregar_modelos_hibridos_e_preparar_dados(
            unidade_selecionada, 
            lote_id_str_global
        )
        
        if dados_processados is not None:
            df_analise, limiar_erro_gb, limiar_score, modelo_lr, modelo_gb, model_iso, scaler, col_y_selecionada, col_x_valvula_selecionada = dados_processados
            
            st.header(f"Painel de Análise Híbrido: {unidade_selecionada} (Lote {lote_selecionado_radio})")
            total_alertas_perf = df_analise['Alerta_Perf_GB'].sum() 
            total_alertas_op = df_analise['Alerta_Op'].sum()
            kpi1, kpi2 = st.columns(2)
            kpi1.metric("Total de Alertas de Performance (GB)", f"{total_alertas_perf}")
            kpi2.metric("Total de Alertas Operacionais (ISO)", f"{total_alertas_op}")
            st.divider()

            st.header("1. Diagnóstico Comparativo: Fatores Críticos (LR vs. GB)")
            col_diag1, col_diag2 = st.columns(2)
            
            with col_diag1:
                st.subheader("LinearRegression (LR)")
                st.write("Mede o Impacto em kW (positivo ou negativo). Usado para as recomendações.")
                coeficientes = pd.Series(modelo_lr.coef_, index=modelo_lr.feature_names_in_)
                coefs_df = pd.DataFrame({'Sensor/Válvula': coeficientes.index, 'Impacto_kW': coeficientes.values})
                coefs_df['Impacto_Absoluto'] = coefs_df['Impacto_kW'].abs()
                coefs_df = coefs_df.sort_values('Impacto_Absoluto', ascending=False).drop(columns=['Impacto_Absoluto'])
                st.dataframe(coefs_df.style.format({'Impacto_kW': '{:+.4f}'}))
            
            with col_diag2:
                st.subheader("GradientBoosting (GB)")
                st.write("Mede a Importância da Feature (0 a 1). Usado para o alerta.")
                importancias = pd.Series(modelo_gb.feature_importances_, index=modelo_gb.feature_names_in_)
                import_df = pd.DataFrame({'Sensor/Válvula': importancias.index, 'Importancia': importancias.values})
                import_df = import_df.sort_values('Importancia', ascending=False)
                st.dataframe(import_df.style.format({'Importancia': '{:.4f}'}))
            
            st.divider()
            
            st.header("2. Detecção: Visualização dos Alertas")
            
            st.subheader(f"Análise de Performance (Gatilho: GradientBoosting, Erro > {limiar_erro_gb:.2f} kW)")
            fig_gb = go.Figure()
            fig_gb.add_trace(go.Scatter(x=df_analise.index, y=df_analise[col_y_selecionada], name="Consumo Real", line=dict(color='blue')))
            fig_gb.add_trace(go.Scatter(x=df_analise.index, y=df_analise['Consumo_Previsto_GB'], name="Previsto (GB)", line=dict(color='green')))
            df_alertas_gb = df_analise[df_analise['Alerta_Perf_GB']]
            fig_gb.add_trace(go.Scatter(x=df_alertas_gb.index, y=df_alertas_gb[col_y_selecionada], name="Alerta (GB)", mode='markers', marker=dict(color='red', size=8, symbol='x')))
            st.plotly_chart(fig_gb, use_container_width=True)

            st.subheader(f"Análise Operacional (Gatilho: Isolation Forest, Score < {limiar_score:.4f})")
            fig_op = px.line(df_analise.reset_index(), x='DateTime', y='Score_Operacional', 
                             title=f"Score de Anomalia Operacional - {unidade_selecionada}")
            fig_op.add_hline(y=limiar_score, line_dash="dash", line_color="red", 
                             annotation_text="Limiar de Anomalia", annotation_position="bottom right")
            st.plotly_chart(fig_op, use_container_width=True)
            
            st.subheader("Tabela de Alertas Detectados (Visão Geral)")
            df_alertas_tabela = df_analise[
                (df_analise['Alerta_Perf_GB']) | (df_analise['Alerta_Op'])
            ].reset_index()

            if df_alertas_tabela.empty:
                st.success("Nenhum alerta (Performance ou Operacional) foi detectado.")
            else:
                st.write(f"Encontrados **{len(df_alertas_tabela)}** minutos de alerta.")
                colunas_display_geral = ['DateTime', 'Alerta_Perf_GB', 'Alerta_Op', col_y_selecionada, 'Consumo_Previsto_GB', 'Erro_kW_GB', 'Score_Operacional']
                df_display_geral = df_alertas_tabela[colunas_display_geral].rename(columns={
                    col_y_selecionada: 'Consumo_Real (kW)'
                })
                st.dataframe(df_display_geral.style.format({
                    'Erro_kW_GB': '{:+.2f}', 'Score_Operacional': '{:.4f}',
                    'Consumo_Previsto_GB': '{:.2f}', 'Consumo_Real (kW)': '{:.2f}',
                    'DateTime': '{:%Y-%m-%d %H:%M:%S}'
                }))
            st.divider()
            
            st.header(f"3. Prescrição: Sistema de Recomendação Híbrido ({unidade_selecionada})")

            if not df_alertas_tabela.empty:
                st.write("Clique no botão abaixo para gerar recomendações detalhadas para os alertas acima.")
                st.warning("Atenção: A simulação de recomendação operacional pode demorar alguns minutos.")
                
                button_key = f"calc_button_{lote_id_str_global}_{unidade_selecionada}"
                
                if st.button(f"Calcular {len(df_alertas_tabela)} Recomendações Híbridas", key=button_key):
                    
                    with st.spinner(f"Rodando simulação de recomendação..."):
                        
                        nomes_corretos_features = modelo_lr.feature_names_in_
                        col_x_valvula = col_x_valvula_selecionada
                        coeficientes = pd.Series(modelo_lr.coef_, index=nomes_corretos_features).sort_values()
                        sensor_mais_positivo = coeficientes.idxmax() if not coeficientes.empty else 'N/A'
                        sensor_mais_negativo = coeficientes.idxmin() if not coeficientes.empty else 'N/A'
                        rec_txt_ineficiente = f"Ineficiente. Ações (LR): Reduzir {sensor_mais_positivo}; Aumentar {sensor_mais_negativo}."
                        rec_txt_subperf = f"Sub-performance. Ações (LR): Aumentar {sensor_mais_positivo}; Reduzir {sensor_mais_negativo}."

                        abertura_min = df_analise[col_x_valvula].min()
                        abertura_max = df_analise[col_x_valvula].max()
                        aberturas_simuladas = np.linspace(abertura_min, abertura_max, 200) 

                        resultados_perf = []
                        resultados_op = []
                        
                        progress_bar = st.progress(0, text="Processando alertas...")
                        
                        for i, row in df_alertas_tabela.iterrows():
                            dados_alerta = df_analise.loc[row['DateTime']]
                            is_alerta_perf = dados_alerta['Alerta_Perf_GB']
                            is_alerta_op = dados_alerta['Alerta_Op']
                            
                            consumo_real_atual = dados_alerta[col_y_selecionada]
                            consumo_previsto_gb_atual = dados_alerta['Consumo_Previsto_GB']
                            erro_kw_gb = dados_alerta['Erro_kW_GB']
                            score_op_atual = dados_alerta['Score_Operacional']
                            abertura_atual = dados_alerta[col_x_valvula]

                            if is_alerta_perf:
                                recomendacao_perf_txt = rec_txt_ineficiente if erro_kw_gb > 0 else rec_txt_subperf
                                
                                dados_atuais_sim_perf = dados_alerta[nomes_corretos_features]
                                leituras_sensores_perf = dados_atuais_sim_perf.copy()
                                
                                if erro_kw_gb > 0: 
                                    if sensor_mais_positivo != 'N/A' and sensor_mais_positivo in leituras_sensores_perf: leituras_sensores_perf[sensor_mais_positivo] -= 1.0
                                    if sensor_mais_negativo != 'N/A' and sensor_mais_negativo in leituras_sensores_perf: leituras_sensores_perf[sensor_mais_negativo] += 1.0
                                else: 
                                    if sensor_mais_positivo != 'N/A' and sensor_mais_positivo in leituras_sensores_perf: leituras_sensores_perf[sensor_mais_positivo] += 1.0
                                    if sensor_mais_negativo != 'N/A' and sensor_mais_negativo in leituras_sensores_perf: leituras_sensores_perf[sensor_mais_negativo] -= 1.0
                                
                                df_final_perf = pd.DataFrame([leituras_sensores_perf])[nomes_corretos_features]
                                consumo_previsto_perf_kW = modelo_gb.predict(df_final_perf)[0] 

                                resultados_perf.append({
                                    'DateTime': row['DateTime'], 
                                    'Consumo_Real (kW)': consumo_real_atual,
                                    'Consumo_Esperado (GB)': consumo_previsto_gb_atual, 
                                    'Erro_GB (kW)': erro_kw_gb, 
                                    'Recomendacao_Perf (LR)': recomendacao_perf_txt,
                                    'Consumo_Previsto_Perf (kW)': consumo_previsto_perf_kW
                                })

                            if is_alerta_op:
                                dados_atuais_sim = dados_alerta[nomes_corretos_features]
                                leituras_sensores_sem_valvula = dados_atuais_sim.drop(col_x_valvula)
                                
                                melhor_score_simulado = -np.inf 
                                melhor_abertura_simulada = abertura_atual 
                                
                                for abertura in aberturas_simuladas:
                                    dados_hipoteticos = leituras_sensores_sem_valvula.copy()
                                    dados_hipoteticos[col_x_valvula] = abertura
                                    df_hipotetico = pd.DataFrame([dados_hipoteticos])[nomes_corretos_features] 
                                    df_hipotetico_scaled = scaler.transform(df_hipotetico)
                                    
                                    score_simulado = model_iso.decision_function(df_hipotetico_scaled)[0]
                                    
                                    if score_simulado > melhor_score_simulado:
                                        melhor_score_simulado = score_simulado
                                        melhor_abertura_simulada = abertura
                                
                                abertura_recomendada_op = melhor_abertura_simulada
                                score_previsto_op = melhor_score_simulado
                                melhoria_score = score_previsto_op - score_op_atual
                                
                                dados_finais_op = leituras_sensores_sem_valvula.copy()
                                dados_finais_op[col_x_valvula] = abertura_recomendada_op
                                df_final_op = pd.DataFrame([dados_finais_op])[nomes_corretos_features]
                                consumo_previsto_op_kW = modelo_gb.predict(df_final_op)[0]
                                
                                if melhoria_score > 0.0001:
                                    status_rec = "Efetiva"
                                elif abs(melhoria_score) < 0.0001:
                                    status_rec = "Ineficaz (Impacto Nulo)"
                                else:
                                    status_rec = "Instável (Score Piorou)"

                                resultados_op.append({
                                    'DateTime': row['DateTime'], 
                                    'Consumo_Real (kW)': consumo_real_atual,
                                    'Abertura_Original_%': abertura_atual, 
                                    'Score_Original': score_op_atual,
                                    'Abertura_Recomendada_%': abertura_recomendada_op,
                                    'Score_Previsto': score_previsto_op,
                                    'Melhoria_Score': melhoria_score,
                                    'Status_Diagnostico': status_rec,
                                    'Consumo_Previsto_Op (kW)': consumo_previsto_op_kW 
                                })
                            
                            progress_bar.progress((i + 1) / len(df_alertas_tabela), text=f"Processando alerta {i+1}/{len(df_alertas_tabela)}...")

                        progress_bar.empty()
                        st.success(f"Simulação concluída!")
                        
                        st.subheader("Tabela de Recomendações de Performance (Ineficiência)")
                        df_resultados_perf = pd.DataFrame(resultados_perf)
                        if not df_resultados_perf.empty:
                            df_resultados_perf = df_resultados_perf.sort_values(by='Erro_GB (kW)', key=abs, ascending=False)
                            df_print_perf = df_resultados_perf.copy()
                            df_print_perf['DateTime'] = df_print_perf['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                            st.dataframe(df_print_perf.style.format({
                                'Consumo_Real (kW)': '{:.2f}', 
                                'Consumo_Esperado (GB)': '{:.2f}',
                                'Erro_GB (kW)': '{:+.2f}', 
                                'Consumo_Previsto_Perf (kW)': '{:.2f}', 
                            }))
                        else:
                            st.info("Nenhum Alerta de Performance foi encontrado.")

                        st.subheader("Tabela de Recomendações Operacionais (Diagnóstico de Anomalia)")
                        df_resultados_op = pd.DataFrame(resultados_op)
                        if not df_resultados_op.empty:
                            df_resultados_op = df_resultados_op.sort_values(by='Score_Original', ascending=True)
                            df_print_op = df_resultados_op.copy()
                            df_print_op['DateTime'] = df_print_op['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                            st.dataframe(df_print_op.style.format({
                                'Consumo_Real (kW)': '{:.2f}', 
                                'Abertura_Original_%': '{:.2f}',
                                'Score_Original': '{:.4f}', 
                                'Abertura_Recomendada_%': '{:.2f}',
                                'Score_Previsto': '{:.4f}', 
                                'Melhoria_Score': '{:+.6f}', 
                                'Consumo_Previsto_Op (kW)': '{:.2f}'
                            }))
                        else:
                            st.info("Nenhum Alerta Operacional foi encontrado.")
                
        else:
            st.error(f"Falha ao processar dados para {unidade_selecionada}.")