import pandas as pd
import numpy as np
import sys
import warnings
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")

LIMIAR_R2_CONFIANCA = 0.5 

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

def identificar_treinar_salvar_modelos_finais(file_entrada, file_metricas, lote_id, limiar):
    print(f"\n--- FASE 3: Salvando Modelos Finais (Area: {lote_id.upper()}) ---")
    
    try:
        df_completo = pd.read_csv(file_entrada)
        if 'DateTime' in df_completo.columns:
            df_completo['DateTime'] = pd.to_datetime(df_completo['DateTime'])
    except Exception as e:
        print(f"ERRO ao carregar '{file_entrada}': {e}", file=sys.stderr); return
        
    try:
        df_metricas = pd.read_csv(file_metricas)
    except Exception as e:
        print(f"ERRO ao carregar '{file_metricas}'. Execute o Passo 2 primeiro.", file=sys.stderr); return

    lista_sensores = [col for col in df_completo.columns if col.isnumeric()]
    print(f"Sensores encontrados: {lista_sensores}")
    
    mapeamento_unidades_dinamico = encontrar_unidades_dinamicamente(df_completo.columns)
    if not mapeamento_unidades_dinamico:
        print("ERRO FATAL: Nenhum FCPB encontrado nos dados.", file=sys.stderr); return
    print(f"Mapeamento dinâmico de FCPBs encontrado: {list(mapeamento_unidades_dinamico.keys())}")
    
    if 'R² (Previsibilidade)' not in df_metricas.columns:
        print(f"ERRO: Coluna 'R² (Previsibilidade)' não encontrada em '{file_metricas}'", file=sys.stderr)
        return
        

    unidades_confiaveis_ids = df_metricas['Unidade'].tolist()

    if not unidades_confiaveis_ids:
        print(f"Nenhuma unidade aprovada encontrada no arquivo de métricas."); return
        
    print(f"Unidades confiáveis para treinamento final: {unidades_confiaveis_ids}")

    for unidade_id in unidades_confiaveis_ids:
        print(f"\n--- Processando Unidade: {unidade_id} ---")
        try:
            info = mapeamento_unidades_dinamico[unidade_id]
            col_y = info['col_y']; col_x_valvula = info['col_x_valvula']
            features = [col_x_valvula] + lista_sensores
            features_existentes = [f for f in features if f in df_completo.columns]
            
            df_modelo_final = df_completo[features_existentes + [col_y]].dropna()
            X_final = df_modelo_final[features_existentes]; Y_final = df_modelo_final[col_y]

            if X_final.empty:
                print(f"  AVISO: Dados insuficientes para treinar {unidade_id}. Pulando.")
                continue

            print(f"  Treinando e salvando modelo LinearRegression para {unidade_id}...")
            modelo_final_lr = LinearRegression().fit(X_final, Y_final)
            nome_arquivo_lr = f"modelo_{unidade_id.lower()}_{lote_id}.joblib" 
            joblib.dump(modelo_final_lr, nome_arquivo_lr)
            
            print(f"  Treinando e salvando modelo GradientBoosting para {unidade_id}...")
            modelo_final_gb = GradientBoostingRegressor(
                random_state=42, n_estimators=100, max_depth=5
            ).fit(X_final, Y_final)
            nome_arquivo_gb = f"modelo_gb_{unidade_id.lower()}_{lote_id}.joblib" 
            joblib.dump(modelo_final_gb, nome_arquivo_gb)

            print(f"    -> Modelos salvos: LR e GB.")
            
        except KeyError:
            print(f"  ERRO: {unidade_id} (das métricas) não foi encontrado no mapa dinâmico. Pulando.")
        except Exception as e:
            print(f"  ERRO ao treinar/salvar modelos para {unidade_id}: {e}", file=sys.stderr)
            
    print(f"\n--- FASE 3 (Lote {lote_id.upper()}) Concluída ---")

def main():
    areas = ["A1", "A2"] 

    for area in areas:
        print(f"\n" + "="*80)
        print(f" Iniciando Treinamento de Modelos Finais (AREA {area})")
        print("="*80 + "\n")
        
        identificar_treinar_salvar_modelos_finais(
            file_entrada=f"resultadosmetricas{area}.csv", 
            file_metricas=f"resultados_metricas{area}.csv", 
            lote_id=area.lower(),
            limiar=LIMIAR_R2_CONFIANCA
        )
    
    print("\n--- Treinamento de TODAS as áreas Concluído ---")

if __name__ == "__main__":
    main()