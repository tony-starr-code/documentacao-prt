import pandas as pd
import s3fs
import numpy as np
import re

def main():
    # ==========================================
    # 1. LEITURA DE DADOS
    # ==========================================
    fs = s3fs.S3FileSystem()
    s3_path = 's3://t2rpt/'
    
    df_contratos = pd.read_csv(f'{s3_path}contratos_apolices.csv', sep=";")
    
    # ==========================================
    # 2. PADRONIZAÇÃO E NORMALIZAÇÃO DE VARIÁVEIS
    # ==========================================
    print("Iniciando a padronização das colunas...")
    
    # ID
    df_contratos['cod_individuo'] = df_contratos['cod_individuo'].str.replace('IND-', '', regex=False).str.strip()
    df_contratos.rename(columns={"cod_individuo": "id_cliente"}, inplace=True)
    

    # Tipo Cobertura
    def normalizar_categoria(valor):
        if pd.isna(valor):
            return np.nan
        v = str(valor).strip().lower()
        v = v.replace('.', '')  
        return v

    mapa_cobertura = {
        'premium': 'Premium', 'prem': 'Premium', 
        'básica': 'Básica', 'basica': 'Básica', 'basic': 'Básica', 
        'padrão': 'Padrão', 'padrao': 'Padrão', 'std': 'Padrão', 
        'plus': 'Plus'
    }
    df_contratos['tipo_cobertura'] = df_contratos['tipo_cobertura'].apply(normalizar_categoria).map(mapa_cobertura)

    # Canal Aquisição
    def normalizar_canal(valor):
        if pd.isna(valor):
            return np.nan
        v = str(valor).strip()
        if v.lower() in ['', '-', '?', '#n/d', 'nan']:
            return np.nan
        return v.title() # Correção aplicada

    df_contratos['canal_aquisicao'] = df_contratos['canal_aquisicao'].apply(normalizar_canal)

    # Método de Pagamento
    def normalizar_metodo(valor):
        if pd.isna(valor):
            return np.nan
        v = str(valor).strip().lower()
        v = v.replace('_', ' ').replace('-', ' ')
        v = re.sub(r'\s+', ' ', v)  
        return v

    mapa_metodo = {
        'boleto': 'Boleto', 'bol': 'Boleto', 'boleto bancario': 'Boleto', 
        'cartao': 'Cartao', 'cartão': 'Cartao', 'cc': 'Cartao', 'cartao credito': 'Cartao', 
        'debito auto': 'Debito', 'debito automatico': 'Debito', 'debito_auto': 'Debito', 
        'debito': 'Debito', 'deb auto': 'Debito', 'pix': 'Pix'
    }
    df_contratos['metodo_pagamento'] = df_contratos['metodo_pagamento'].apply(normalizar_metodo).map(mapa_metodo)

    # Pagamento em Dia
    mapa_pagamento = {
        'em dia': 1, 'ok': 1, 'sim': 1, 's': 1, '1': 1,
        'nao': 0, 'não': 0, 'n': 0, '0': 0, 'atrasado': 0,
    }

    def normalizar_pagamento(valor):
        if pd.isna(valor):
            return np.nan
        v = str(valor).strip().lower()
        return mapa_pagamento.get(v, np.nan)

    df_contratos['pagamento_em_dia'] = df_contratos['pagamento_em_dia'].apply(normalizar_pagamento).astype('float64')

    # Valores Monetários
    def limpar_valor_monetario(valor):
        if pd.isna(valor):
            return np.nan
        v = str(valor).strip()
        v = v.replace('R$', '').strip()
        v = v.replace(' ', '')

        tem_ponto = '.' in v
        tem_virgula = ',' in v

        if tem_ponto and tem_virgula:
            if v.rfind(',') > v.rfind('.'):
                v = v.replace('.', '').replace(',', '.')
            else:
                v = v.replace(',', '')
        elif tem_virgula:
            v = v.replace('.', '').replace(',', '.')
        elif tem_ponto:
            partes = v.split('.')
            if len(partes[-1]) == 2:
                pass
            else:
                v = v.replace('.', '')
        try:
            return float(v)
        except ValueError:
            return np.nan

    colunas_monetarias = ['valor_premio_anual', 'valor_cobertura_total', 'franquia_media']
    for col in colunas_monetarias:
        df_contratos[col] = df_contratos[col].apply(limpar_valor_monetario)

    # Datas e Numéricos
    df_contratos['data_primeira_apolice'] = pd.to_datetime(
        df_contratos['data_primeira_apolice'], 
        format='mixed', 
        errors='coerce'
    ).astype('datetime64[us]') # Correção aplicada

    colunas_numericas_restantes = [
        'num_apolices_ativas', 'tempo_cliente_dias', 
        'num_produtos_contratados', 'desconto_aplicado_pct'
    ]
    for col in colunas_numericas_restantes:
        df_contratos[col] = pd.to_numeric(df_contratos[col], errors='coerce')

    # Ajuste decimal do desconto
    df_contratos["desconto_aplicado_pct"] = df_contratos["desconto_aplicado_pct"] * 100

    # ==========================================
    # 3. LIMPEZA DE ERROS E NULOS
    # ==========================================
    print("Iniciando a limpeza e imputação de nulos...")
    
    # Valores Impossíveis -> NaN
    LIM_REALISTAS = {
        'valor_premio_anual':    (0,      500_000),
        'valor_cobertura_total': (0,    2_000_000),
        'tempo_cliente_dias':    (0,       10_950),
    }
    for col, (min_val, max_val) in LIM_REALISTAS.items():
        mask = (df_contratos[col] < min_val) | (df_contratos[col] > max_val)
        df_contratos.loc[mask, col] = np.nan

    # Imputações Básicas (Mediana e Moda)
    df_contratos["num_apolices_ativas"] = df_contratos["num_apolices_ativas"].fillna(df_contratos["num_apolices_ativas"].median())
    df_contratos["num_produtos_contratados"] = df_contratos["num_produtos_contratados"].fillna(df_contratos["num_produtos_contratados"].median())
    df_contratos["desconto_aplicado_pct"] = df_contratos["desconto_aplicado_pct"].fillna(df_contratos["desconto_aplicado_pct"].median())

    # Correção Aplicada: Seleção do primeiro elemento da moda [0]
    df_contratos["tipo_cobertura"] = df_contratos["tipo_cobertura"].fillna(df_contratos["tipo_cobertura"].mode()[0])
    df_contratos["canal_aquisicao"] = df_contratos["canal_aquisicao"].fillna(df_contratos["canal_aquisicao"].mode()[0])
    df_contratos["metodo_pagamento"] = df_contratos["metodo_pagamento"].fillna(df_contratos["metodo_pagamento"].mode()[0])

    moda_pagamento = df_contratos["pagamento_em_dia"].mode()[0]
    df_contratos["pagamento_em_dia"] = df_contratos["pagamento_em_dia"].fillna(moda_pagamento)

    # Imputações Complexas (Agrupamento e Datas)
    for col in colunas_monetarias:
        df_contratos[col] = df_contratos[col].fillna(
            df_contratos.groupby("tipo_cobertura")[col].transform("median")
        )

    # Define uma data de referência fixa para o cálculo de tempo para evitar bugs
    DATA_REFERENCIA = pd.Timestamp("2026-06-01")

    days = (DATA_REFERENCIA - df_contratos["data_primeira_apolice"]).dt.days
    df_contratos["tempo_cliente_dias"] = df_contratos["tempo_cliente_dias"].fillna(days)
    df_contratos = df_contratos.dropna(subset=["tempo_cliente_dias"])

    dates = (DATA_REFERENCIA - pd.to_timedelta(df_contratos["tempo_cliente_dias"], unit="D"))
    df_contratos["data_primeira_apolice"] = df_contratos["data_primeira_apolice"].fillna(dates) # Correção aplicada

    # ==========================================
    # 4. REMOÇÃO DE DUPLICATAS E OUTLIERS (WINSORIZAÇÃO)
    # ==========================================
    print("Tratando duplicatas e aplicando winsorização nos outliers...")
    
    df_contratos = df_contratos.drop_duplicates()

    CONFIG_OUTLIERS = {
        'valor_premio_anual':      {'inf_min': 0, 'sup_max': None, 'fator': 2.5},
        'valor_cobertura_total':   {'inf_min': 0, 'sup_max': None, 'fator': 2.5},
        'franquia_media':          {'inf_min': 0, 'sup_max': None, 'fator': 1.5},
        'tempo_cliente_dias':      {'inf_min': 0, 'sup_max': None, 'fator': 1.5},
        'desconto_aplicado_pct':   {'inf_min': 0, 'sup_max': 100,  'fator': 1.5},
    }

    for col, cfg in CONFIG_OUTLIERS.items():
        Q1 = df_contratos[col].quantile(0.25)
        Q3 = df_contratos[col].quantile(0.75)
        IQR = Q3 - Q1

        limite_inf = max(cfg['inf_min'], Q1 - cfg['fator'] * IQR)
        limite_sup = Q3 + cfg['fator'] * IQR
        
        if cfg['sup_max']:
            limite_sup = min(cfg['sup_max'], limite_sup)

        df_contratos[col] = df_contratos[col].clip(lower=limite_inf, upper=limite_sup)

    # ==========================================
    # 5. SALVANDO DADOS TRATADOS
    # ==========================================
    print("Tratamento concluído! Salvando base limpa...")
    caminho_saida = f'../bases_tratadas/contratos_apolices_tratado.csv'
    df_contratos.to_csv(caminho_saida, index=False)
    
    return df_contratos

if __name__ == "__main__":
    df_tratado = main()