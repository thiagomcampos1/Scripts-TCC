import pandas as pd
import sys
import warnings

warnings.filterwarnings("ignore")

def carregar_e_limpar_dados_universal(file_sensores, file_resumo, file_saida):
    try:
        print(f"\n Iniciando Processamento...")
        
        df_sensores = pd.read_csv(file_sensores, encoding='latin-1', delimiter=';')
        df_resumo = pd.read_csv(file_resumo, header=1, encoding='latin-1', delimiter=';')

        df_resumo.columns = df_resumo.columns.str.replace('\n', ' ').str.replace(r'\s+', ' ', regex=True).str.strip()
        df_sensores.columns = df_sensores.columns.str.replace(r'\s+', ' ', regex=True).str.strip()

        mapa_renomeacao = {
            'Data/ Hora': 'DateTime',  
            'Data/Hora': 'DateTime',   
            'Date/Time': 'DateTime'    
        }
        
        df_sensores.rename(columns=mapa_renomeacao, inplace=True)
        df_resumo.rename(columns=mapa_renomeacao, inplace=True)

        try:
            df_sensores['DateTime'] = pd.to_datetime(df_sensores['DateTime'], dayfirst=True)
            df_resumo['DateTime'] = pd.to_datetime(df_resumo['DateTime'], dayfirst=True)
        except KeyError as e:
            print(f"ERRO CRÍTICO: Coluna de data não encontrada. Colunas disponíveis: {list(df_sensores.columns)}")
            return False

        df_completo = pd.merge(df_sensores, df_resumo, on='DateTime', how='inner')
        df_completo.columns = df_completo.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
        

        colunas_para_limpar = []
        for col in df_completo.columns:
            col_lower = col.lower() 
            if col.isnumeric():
                colunas_para_limpar.append(col)
            elif 'kw' in col_lower:
                colunas_para_limpar.append(col)
            elif 'valvula' in col_lower:
                colunas_para_limpar.append(col)

        if not colunas_para_limpar:
            print("AVISO: Nenhuma coluna de sensor ou FCPB foi encontrada para limpar.")
        else:
            print(f"-> {len(colunas_para_limpar)} colunas numéricas identificadas.")

        for col in colunas_para_limpar:
            if col in df_completo.columns:
                if df_completo[col].dtype == 'object' or pd.api.types.is_string_dtype(df_completo[col]):
                       df_completo[col] = df_completo[col].astype(str).str.replace(',', '.', regex=False)
                df_completo[col] = pd.to_numeric(df_completo[col], errors='coerce')
        
        print(f"   [RESULTADO] Total de linhas sincronizadas: {len(df_completo)}")
        
        df_completo.to_csv(file_saida, index=False)
        print(f"[SUCESSO] Arquivo salvo: '{file_saida}'")
        return True

    except FileNotFoundError:
        print(f"[ERRO] Arquivos '{file_sensores}' ou '{file_resumo}' não encontrados na pasta.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ERRO] Falha na preparação dos dados: {e}", file=sys.stderr)
        return False

def main():
    areas = ["A1", "A2"]

    for area in areas:
        print(f"\n Processando {area}...")
        
        sucesso = carregar_e_limpar_dados_universal(
            file_sensores=f"sensores{area}.csv", 
            file_resumo=f"resumo{area}.csv", 
            file_saida=f"resultadosmetricas{area}.csv"
        )
        
        if sucesso:
            print(f" -> {area} finalizada com sucesso.")
        else:
            print(f" -> FALHA ao processar {area}. Verifique se os arquivos existem.")
        
        print("="*30)

if __name__ == "__main__":
    main()