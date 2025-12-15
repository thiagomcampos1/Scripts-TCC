import pandas as pd
import numpy as np
import sys
import warnings
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

TEST_SIZE_PERCENT_AUDIT = 0.3
LIMIAR_CORTE_AUDITORIA = 0.5 

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

def auditoria_multi_modelo_universal(file_entrada, file_saida_lr, file_saida_completa):
    print(f"\n--- FASE 2: Auditoria de Confiabilidade ({file_entrada}) ---")
    try:
        df_completo = pd.read_csv(file_entrada)
        if 'DateTime' in df_completo.columns:
            df_completo['DateTime'] = pd.to_datetime(df_completo['DateTime'])
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{file_entrada}' não encontrado. Execute o Passo 1 primeiro.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERRO ao carregar dados: {e}", file=sys.stderr)
        return None
        
    lista_sensores = [col for col in df_completo.columns if col.isnumeric()]
    print(f"Sensores encontrados: {lista_sensores}")
    
    mapeamento_unidades_dinamico = encontrar_unidades_dinamicamente(df_completo.columns)
    if not mapeamento_unidades_dinamico:
        print("ERRO FATAL: Nenhuma coluna 'Entalpico' encontrada. Verifique o nome no CSV.", file=sys.stderr)
        return None
    print(f"Unidades encontradas: {list(mapeamento_unidades_dinamico.keys())}")
    
    modelos_audit = {
        "LinearRegression": LinearRegression(), "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42, max_iter=3000),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=50, n_jobs=-1, max_depth=10),
        "GradientBoosting": GradientBoostingRegressor(random_state=42, n_estimators=50, max_depth=5)
    }
    scaler = StandardScaler(); resultados_auditoria = []

    for unidade_id, info in mapeamento_unidades_dinamico.items():
        col_y = info['col_y']; col_x_valvula = info['col_x_valvula']
        print(f"\n--- Auditando Unidade: {unidade_id} (Alvo: {col_y}) ---")
            
        features = [col_x_valvula] + lista_sensores
        features_existentes = [f for f in features if f in df_completo.columns]
        
        df_modelo = df_completo[features_existentes + [col_y]].dropna()
        X = df_modelo[features_existentes]; Y = df_modelo[col_y]

        if X.empty or len(X) < 20:
            print(f"  AVISO: Dados insuficientes para {unidade_id}. Pulando.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE_PERCENT_AUDIT, random_state=42)
        X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)

        for nome_modelo, modelo_base in modelos_audit.items():
            try:
                modelo = modelo_base.__class__(**modelo_base.get_params())
                if nome_modelo in ["Ridge", "Lasso", "LinearRegression"]: 
                    modelo.fit(X_train_scaled, y_train); y_pred = modelo.predict(X_test_scaled)
                else:
                    modelo.fit(X_train, y_train); y_pred = modelo.predict(X_test)
                    
                resultados_auditoria.append({"Unidade": unidade_id, "Modelo": nome_modelo,
                                             "R2": r2_score(y_test, y_pred),
                                             "MAE": mean_absolute_error(y_test, y_pred),
                                             "RMSE": sqrt(mean_squared_error(y_test, y_pred))})
            except Exception as e: print(f"    ERRO ao treinar {nome_modelo}: {e}")

    if not resultados_auditoria: print("\nNenhuma unidade auditada."); return None
    
    df_resultados_audit = pd.DataFrame(resultados_auditoria).sort_values(by=["Unidade", "R2"], ascending=[True, False])
    
    print("\n" + "="*80)
    print("TABELA FINAL DE AUDITORIA (Ordenada por Unidade > R2)")
    print("="*80)
    
    header = f"{'Unidade':<10} | {'Modelo':<20} | {'R2':<10} | {'MAE':<10} | {'RMSE':<10}"
    print(header)
    print("-" * 80)

    unidade_atual = None
    
    for _, row in df_resultados_audit.iterrows():
        if unidade_atual is not None and row['Unidade'] != unidade_atual:
             print("-" * 80)

        print(f"{row['Unidade']:<10} | {row['Modelo']:<20} | {row['R2']:<10.4f} | {row['MAE']:<10.4f} | {row['RMSE']:<10.4f}")
        
        unidade_atual = row['Unidade']
        
    print("="*80)

    melhores_r2_por_unidade = df_resultados_audit.groupby('Unidade')['R2'].max()
    unidades_aprovadas = melhores_r2_por_unidade[melhores_r2_por_unidade >= LIMIAR_CORTE_AUDITORIA].index.tolist()
    
    print(f"\nUnidades Aprovadas (R2 Máximo >= {LIMIAR_CORTE_AUDITORIA}): {unidades_aprovadas}")
    
    try:
        df_resultados_audit.to_csv(file_saida_completa, index=False)
        df_lr_results = df_resultados_audit[df_resultados_audit['Modelo'] == 'LinearRegression'].copy()
        df_lr_results = df_lr_results[df_lr_results['Unidade'].isin(unidades_aprovadas)]
        df_lr_results.rename(columns={'R2': 'R² (Previsibilidade)'}, inplace=True) 
        
        if not df_lr_results.empty:
            df_lr_results.to_csv(file_saida_lr, index=False)
            print(f"Arquivo de métricas salvo: {file_saida_lr}")
        else:
            print("AVISO: Nenhuma unidade aprovada para salvar.")
            
    except Exception as e:
        print(f"ERRO ao salvar arquivos: {e}", file=sys.stderr)
        
    print("--- FASE 2 Concluída ---")

def main():
    areas = ["A1", "A2"]
    for area in areas:
        auditoria_multi_modelo_universal(
            file_entrada=f"resultadosmetricas{area}.csv", 
            file_saida_lr=f"resultados_metricas{area}.csv", 
            file_saida_completa=f"auditoria_completa{area}.csv"
        )

if __name__ == "__main__":
    main()