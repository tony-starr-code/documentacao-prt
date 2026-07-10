import pandas as pd
import numpy as np
import re
import s3fs
from dateutil import parser
pd.set_option('display.max_columns', None)  # mostra todas as colunas
pd.set_option('display.max_colwidth', None) # não trunca o conteúdo
pd.set_option('display.width', None) 
fs = s3fs.S3FileSystem()

bucket_name = 't2rpt'

s3_path = f's3://{bucket_name}/'

df_sinistros = pd.read_csv(f'{s3_path}atendimento_sinistros.csv', sep=",")
###FUNÇÕES#######################################################################
NULOS_DISFARÇADOS = ['#n/d', '-', '', '?', 'n/a', 'na', 'null', 'none', '-']


def limpar_nulos(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()
        df[col] = df[col].replace(NULOS_DISFARÇADOS, np.nan)
    return df


def imputar_amostra(serie):
    nulos = serie.isnull()
    serie = serie.copy()
    serie.loc[nulos] = serie.dropna().sample(nulos.sum(), replace=True).values
    return serie


def imputar_categorica(serie):
    nulos = serie.isnull()
    serie = serie.copy()
    frequencias = serie.value_counts(normalize=True)
    serie.loc[nulos] = np.random.choice(frequencias.index, size=nulos.sum(), p=frequencias.values)
    return serie
# ============================================================
# TRATAMENTO — atendimento_sinistros.csv
# ============================================================
df = df_sinistros.copy()
df = limpar_nulos(df)

# 1. Renomear identificador e converter notação científica
df.rename(columns={'customer_key': 'id_cliente'}, inplace=True)
df['id_cliente'] = df['id_cliente'].astype(float).astype(int)

# 2. Canal preferencial
df['canal_preferencial_contato'] = imputar_categorica(df['canal_preferencial_contato'])

# 3. Colunas numéricas
cols_num = ['num_reclamacoes_12m', 'num_sinistros_historico', 'dias_ultimo_contato',
            'tempo_medio_resposta_dias', 'num_ligacoes_suporte_12m',
            'num_acessos_app_mes', 'satisfacao_nps']
for col in cols_num:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = imputar_amostra(df[col])

# 4. Tempo resolução último sinistro
df['tempo_resolucao_ultimo_sinistro'] = pd.to_numeric(df['tempo_resolucao_ultimo_sinistro'], errors='coerce')

sem_sinistro = df['tempo_resolucao_ultimo_sinistro'].isnull() & df['data_ultimo_sinistro'].isnull()
df.loc[sem_sinistro, 'tempo_resolucao_ultimo_sinistro'] = 0

# imputar apenas quem teve sinistro (evita contaminar com os zeros de "sem sinistro")
com_sinistro = ~sem_sinistro
df.loc[com_sinistro, 'tempo_resolucao_ultimo_sinistro'] = imputar_amostra(
    df.loc[com_sinistro, 'tempo_resolucao_ultimo_sinistro']
)
# 5. Data último sinistro
df['data_ultimo_sinistro'] = pd.to_datetime(df['data_ultimo_sinistro'], errors='coerce', format='mixed', dayfirst=True)
df.loc[df['data_ultimo_sinistro'] > pd.Timestamp.today(), 'data_ultimo_sinistro'] = pd.NaT
mask_data = df['data_ultimo_sinistro'].isna() & (df['tempo_resolucao_ultimo_sinistro'] > 0)
datas_existentes = df['data_ultimo_sinistro'].dropna()
df.loc[mask_data, 'data_ultimo_sinistro'] = datas_existentes.sample(
    mask_data.sum(), replace=True
).values

# 6. ── Missingness as a Feature ──────────────────────────────
# Passo 1: flag binária (antes de qualquer transformação na data)
df['teve_sinistro'] = df['data_ultimo_sinistro'].notna().astype(int)

# Passo 2: conversão para dias usando a última data válida como referência
data_referencia = df['data_ultimo_sinistro'].max()
print(f"Data de referência: {data_referencia.date()}")
df['dias_desde_ultimo_sinistro'] = (
    data_referencia - df['data_ultimo_sinistro']
).dt.days

# Passo 3: imputação -1 para quem nunca teve sinistro
df['dias_desde_ultimo_sinistro'] = (
    df['dias_desde_ultimo_sinistro'].fillna(-1).astype(int)
)

# Remove a data bruta — não é mais necessária
df.drop(columns=['data_ultimo_sinistro'], inplace=True)

# Validação
print(df.groupby('teve_sinistro')['dias_desde_ultimo_sinistro'].describe())
# ─────────────────────────────────────────────────────────────

# 7. Outliers — satisfacao_nps (válido: 0 a 10)
mask_nps = ~df['satisfacao_nps'].between(0, 10)
df.loc[mask_nps, 'satisfacao_nps'] = np.nan
df['satisfacao_nps'] = imputar_amostra(df['satisfacao_nps'])

# 8. Outliers — dias_ultimo_contato (winsorização)
Q1 = df['dias_ultimo_contato'].quantile(0.25)
Q3 = df['dias_ultimo_contato'].quantile(0.75)
IQR = Q3 - Q1
df['dias_ultimo_contato'] = df['dias_ultimo_contato'].clip(
    lower=max(0, Q1 - 1.5 * IQR), upper=Q3 + 1.5 * IQR
)

# 9. Converter tipos
cols_int = ['num_reclamacoes_12m', 'num_sinistros_historico', 'num_ligacoes_suporte_12m', 'num_acessos_app_mes']
for col in cols_int:
    df[col] = df[col].astype(int)
df['satisfacao_nps'] = df['satisfacao_nps'].astype(int)
df['dias_ultimo_contato'] = df['dias_ultimo_contato'].astype(int)
df['tempo_resolucao_ultimo_sinistro'] = df['tempo_resolucao_ultimo_sinistro'].astype(int)
df['tempo_medio_resposta_dias'] = df['tempo_medio_resposta_dias'].astype(int)
df['dias_ultimo_contato'] = df['dias_ultimo_contato'].astype(int)
# 10. Remover duplicatas
df.drop_duplicates(subset='id_cliente', keep='first', inplace=True)

df_sinistros_tratado = df