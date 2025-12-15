import pandas as pd
import numpy as np
import sys
import warnings
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

STATSMODELS_DISPONIVEL = False
PROPHET_DISPONIVEL = False
try:
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_DISPONIVEL = True
    print("Biblioteca 'statsmodels' (SARIMAX) carregada.")
except ImportError:
    print("AVISO: Biblioteca 'statsmodels' não encontrada. SARIMAX será pulado.")
try:
    from prophet import Prophet
    PROPHET_DISPONIVEL = True
    print("Biblioteca 'prophet' carregada.")
except ImportError:
    print("AVISO: Biblioteca 'prophet' não encontrada. Prophet será pulado.")

warnings.filterwarnings("ignore")

TEST_SIZE_PERCENT_SERIES = 0.2

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

def calcular_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.where(y_true == 0, 1e-6, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rodar_analise_prophet(df_data, col_y, regressores_externos):
    if not PROPHET_DISPONIVEL: return []
    df_prophet = df_data.reset_index().rename(columns={'DateTime': 'ds', col_y: 'y'})
    split_point = int(len(df_prophet) * (1 - TEST_SIZE_PERCENT_SERIES))
    df_train, df_test = df_prophet.iloc[:split_point], df_prophet.iloc[split_point:]
    resultados = []
    y_true = df_test['y']

    try:
        m_baseline = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False).fit(df_train[['ds', 'y']])
        fc_base = m_baseline.predict(df_test[['ds']])
        y_pred_base = fc_base['yhat']
        resultados.append({'Modelo': 'Prophet (Baseline)', 'MAE': mean_absolute_error(y_true, y_pred_base), 'RMSE': sqrt(mean_squared_error(y_true, y_pred_base)), 'MAPE (%)': calcular_mape(y_true, y_pred_base)})
    except Exception as e: pass 

    try:
        m_agregado = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
        regressores_adicionados = []
        for regressor in regressores_externos:
            if regressor in df_train.columns: 
                m_agregado.add_regressor(regressor)
                regressores_adicionados.append(regressor)
        
        m_agregado.fit(df_train[['ds', 'y'] + regressores_adicionados])
        fc_agg = m_agregado.predict(df_test[['ds'] + regressores_adicionados])
        y_pred_agg = fc_agg['yhat']
        resultados.append({'Modelo': 'Prophet (Agregado)', 'MAE': mean_absolute_error(y_true, y_pred_agg), 'RMSE': sqrt(mean_squared_error(y_true, y_pred_agg)), 'MAPE (%)': calcular_mape(y_true, y_pred_agg)})
    except Exception as e: pass 
    
    return resultados

def rodar_analise_sarimax(df_data, col_y, regressores_externos):
    if not STATSMODELS_DISPONIVEL: return []

    endog = df_data[col_y]
    exog = df_data[regressores_externos]
    scaler = StandardScaler()
    exog_scaled = pd.DataFrame(scaler.fit_transform(exog), index=exog.index, columns=exog.columns)
    split_point = int(len(df_data) * (1 - TEST_SIZE_PERCENT_SERIES))
    endog_train, endog_test = endog.iloc[:split_point], endog.iloc[split_point:]
    exog_train, exog_test = exog_scaled.iloc[:split_point], exog_scaled.iloc[split_point:]
    resultados = []
    y_true = endog_test
    order_params = (1, 1, 1); seasonal_params = (1, 0, 1, 60)
    
    try:
        m_base_s = SARIMAX(endog_train, order=order_params, seasonal_order=seasonal_params, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=50)
        pred_base_s = m_base_s.get_prediction(start=endog_test.index[0], end=endog_test.index[-1])
        y_pred_base_s = pred_base_s.predicted_mean
        resultados.append({'Modelo': 'SARIMA (Baseline)', 'MAE': mean_absolute_error(y_true, y_pred_base_s), 'RMSE': sqrt(mean_squared_error(y_true, y_pred_base_s)), 'MAPE (%)': calcular_mape(y_true, y_pred_base_s)})
    except Exception as e: pass 
    
    try:
        m_agg_s = SARIMAX(endog_train, exog=exog_train, order=order_params, seasonal_order=seasonal_params, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=50)
        pred_agg_s = m_agg_s.get_prediction(start=endog_test.index[0], end=endog_test.index[-1], exog=exog_test)
        y_pred_agg_s = pred_agg_s.predicted_mean
        resultados.append({'Modelo': 'SARIMAX (Agregado)', 'MAE': mean_absolute_error(y_true, y_pred_agg_s), 'RMSE': sqrt(mean_squared_error(y_true, y_pred_agg_s)), 'MAPE (%)': calcular_mape(y_true, y_pred_agg_s)})
    except Exception as e: pass 
    
    return resultados

def comparar_modelos_preditivos_ts(df_completo_com_dt_col, unidade_id, lista_sensores, mapeamento_unidades):
    print(f"\n--- FASE 4 ({unidade_id}): Comparação de Modelos Preditivos (Séries Temporais) ---")
    try:
        mapa_unidade = mapeamento_unidades[unidade_id]; col_y = mapa_unidade['col_y']; col_x_valvula = mapa_unidade['col_x_valvula']
    except KeyError:
        print(f"  ERRO: {unidade_id} não encontrado no mapeamento dinâmico. Pulando TS.")
        return None
    
    features = [col_x_valvula] + lista_sensores; features_existentes = [f for f in features if f in df_completo_com_dt_col.columns]
    colunas_necessarias = [col_y] + features_existentes
    
    df_com_indice = df_completo_com_dt_col.set_index('DateTime')
    df_fcpb = df_com_indice.loc[:, colunas_necessarias].dropna()
    
    df_fcpb = df_fcpb.asfreq('min') 
    if df_fcpb.isnull().values.any():
        df_fcpb = df_fcpb.interpolate(method='time')

    df_fcpb = df_fcpb.dropna() 
    
    if df_fcpb.empty or len(df_fcpb) < (1 / TEST_SIZE_PERCENT_SERIES * 2): 
        print(f"  AVISO: Dados insuficientes para Série Temporal de {unidade_id} ({len(df_fcpb)} linhas). Pulando.")
        return None

    resultados_prophet = rodar_analise_prophet(df_fcpb, col_y, features_existentes)
    resultados_sarimax = rodar_analise_sarimax(df_fcpb, col_y, features_existentes)
    
    resultados_finais_unidade = resultados_prophet + resultados_sarimax
    
    if not resultados_finais_unidade:
        print(f"  AVISO: Nenhum modelo de TS pôde ser executado para {unidade_id}.")
        return None
        
    df_resultados = pd.DataFrame(resultados_finais_unidade).sort_values(by='RMSE')

    print("\n" + "="*60)
    print(f"--- Tabela 4: Comparação Preditiva ({unidade_id}) ---")
    print(df_resultados.to_string(index=False))
    print("="*60)
    
    if len(df_resultados) > 1:
        melhor_modelo_row = df_resultados.iloc[0]; melhor_modelo = melhor_modelo_row['Modelo']
        baseline_rmse = np.nan
        if "Prophet" in melhor_modelo: 
            baseline_row = df_resultados[df_resultados['Modelo'] == 'Prophet (Baseline)']
        else: 
            baseline_row = df_resultados[df_resultados['Modelo'] == 'SARIMA (Baseline)']
            
        if not baseline_row.empty: baseline_rmse = baseline_row['RMSE'].values[0]
        erro_melhor_rmse = melhor_modelo_row['RMSE']
        
        print(f"\nConclusão ({unidade_id}): Melhor modelo foi '{melhor_modelo}'.")
        if "Agregado" in melhor_modelo and not pd.isna(baseline_rmse) and baseline_rmse > 0 and erro_melhor_rmse < baseline_rmse:
            reducao_erro = ((baseline_rmse - erro_melhor_rmse) / baseline_rmse) * 100
            print(f" -> Agregar Sensores/Válvula reduziu o erro (RMSE) em {reducao_erro:.1f}%.")
        else: 
            print(" -> Modelo Baseline foi melhor ou não houve melhora significativa.")
            
    return df_resultados

def executar_passo_4_para_lote(file_entrada_dados, file_entrada_metricas, lote_id):
    
    print("="*70); print(f"INICIANDO PASSO 4: Previsão de Séries Temporais (Lote {lote_id.upper()})"); print("="*70)

    try:
        df_completo = pd.read_csv(file_entrada_dados)
        try:
            df_metricas = pd.read_csv(file_entrada_metricas)
        except:
             df_metricas = pd.read_csv(f"resultados_metricas{lote_id.upper()}.csv")
             
        lista_sensores = [col for col in df_completo.columns if col.isnumeric()]
        if 'DateTime' in df_completo.columns:
            df_completo['DateTime'] = pd.to_datetime(df_completo['DateTime'])
    except Exception as e:
        print(f"ERRO (Lote {lote_id.upper()}): Não foi possível carregar arquivos.\n{e}", file=sys.stderr)
        return

    mapeamento_unidades_dinamico = encontrar_unidades_dinamicamente(df_completo.columns)
    if not mapeamento_unidades_dinamico:
        print(f"ERRO FATAL (Lote {lote_id.upper()}): Nenhum FCPB encontrado nos dados.", file=sys.stderr)
        return
    print(f"Mapeamento dinâmico de FCPBs (Lote {lote_id.upper()}): {list(mapeamento_unidades_dinamico.keys())}")
       
    if 'Unidade' in df_metricas.columns:
        unidades_confiaveis_ids = df_metricas['Unidade'].unique().tolist()
    else:
        print("AVISO: Coluna 'Unidade' não encontrada no arquivo de métricas. Usando todas as encontradas.")
        unidades_confiaveis_ids = list(mapeamento_unidades_dinamico.keys())

    if not unidades_confiaveis_ids:
        print(f"Nenhuma unidade confiável encontrada para prever (Lote {lote_id.upper()})."); return
        
    print(f"Unidades confiáveis (Baseado no Passo 2): {unidades_confiaveis_ids}")
    
    resultados_preditivos_gerais = pd.DataFrame()
    for unidade_id in unidades_confiaveis_ids:
        print("\n" + "#"*70); print(f"# PROCESSANDO UNIDADE: {unidade_id} (Lote {lote_id.upper()})"); print("#"*70)
        
        df_predicao_unidade = comparar_modelos_preditivos_ts(
             df_completo, 
             unidade_id, 
             lista_sensores,
             mapeamento_unidades_dinamico 
        )
        if df_predicao_unidade is not None:
             df_predicao_unidade['Unidade'] = unidade_id
             resultados_preditivos_gerais = pd.concat([resultados_preditivos_gerais, df_predicao_unidade])
             
    print("\n" + "="*70); print(f"--- RESUMO FINAL DA COMPARAÇÃO PREDITIVA (Lote {lote_id.upper()}) ---")
    if not resultados_preditivos_gerais.empty: print(resultados_preditivos_gerais.sort_values(by=['Unidade', 'RMSE']).to_string(index=False))
    else: print("Nenhum resultado preditivo foi gerado.")
    print("="*70); print(f"PASSO 4 (Lote {lote_id.upper()}) Concluído."); print("="*70)


def main():
    areas = ["A1", "A2"]

    for area in areas:
        print("\n" + "#"*80)
        print(f">>> Iniciando Passo 4: Previsão de Séries Temporais (Lote {area}) <<<")
        print("#"*80 + "\n")
        
        executar_passo_4_para_lote(
            file_entrada_dados=f"resultadosmetricas{area}.csv",
            file_entrada_metricas=f"metricas_auditoria_{area}.csv", 
            lote_id=area
        )
    
    print("\n--- Processamento de TODAS as Áreas (Séries Temporais) Concluído ---")

if __name__ == "__main__":
    main()