import pandas as pd
import numpy as np
import re
import s3fs
pd.set_option('display.max_columns', None)  # mostra todas as colunas
pd.set_option('display.max_colwidth', None) # não trunca o conteúdo
pd.set_option('display.width', None) 
fs = s3fs.S3FileSystem()

bucket_name = 't2rpt'

s3_path = f's3://{bucket_name}/'

df_sinistros = pd.read_csv(f'{s3_path}atendimento_sinistros.csv', sep=",")
df_cadastro = pd.read_csv(f'{s3_path}cadastro_clientes.csv', sep=",")

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
##################################################################################
# ============================================================
# TRATAMENTO — cadastro_clientes.csv
# ============================================================
from dateutil import parser

df = df_cadastro.copy()
df = limpar_nulos(df)

# 1. Renomear identificador
df.rename(columns={'Id_cliente': 'id_cliente'}, inplace=True)

# 2. Idade
df['idade'] = pd.to_numeric(df['idade'], errors='coerce').astype('Int64')

# 3. Data de nascimento — derivar idade onde possível e descartar
def parse_data(val):
    try:
        return parser.parse(str(val), dayfirst=True)
    except:
        return pd.NaT

df['data_nascimento'] = df['data_nascimento'].apply(parse_data)
hoje = pd.Timestamp.today()
mask_data = df['idade'].isnull() & df['data_nascimento'].notnull()
df.loc[mask_data, 'idade'] = df.loc[mask_data, 'data_nascimento'].apply(
    lambda x: int((hoje - x).days / 365.25)
)
df.drop(columns='data_nascimento', inplace=True)

# 4. Gênero
mapa_genero = {
    'masc': 'M', 'm': 'M', 'masculino': 'M',
    'f': 'F', 'fem': 'F', 'feminino': 'F'
}
df['genero'] = df['genero'].str.strip().str.lower().map(mapa_genero)

# 5. Estado civil
mapa_ec = {
    'c': 'casado', 'casado': 'casado', 'married': 'casado', 'casado(a)': 'casado',
    's': 'solteiro', 'solt': 'solteiro', 'single': 'solteiro', 'solteiro(a)': 'solteiro', 'solteiro': 'solteiro',
}
df['estado_civil'] = df['estado_civil'].str.strip().str.lower().map(mapa_ec)

# 6. Tem filhos
mapa_filhos = {
    'sim': 1, 'true': 1, 's': 1, '1': 1,
    'nao': 0, 'não': 0, 'n': 0, 'false': 0, '0': 0,
}
df['tem_filhos'] = df['tem_filhos'].str.strip().str.lower().map(mapa_filhos)

# 7. Qtd dependentes
df['qtd_dependentes'] = pd.to_numeric(df['qtd_dependentes'], errors='coerce').astype('Int64')

# 8. Escolaridade
df['escolaridade'] = df['escolaridade'].str.strip().str.lower().str.capitalize()

# 9. Renda anual e valor imóvel
for col in ['renda_anual', 'valor_imovel']:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(r'r\$', '', regex=True)
        .str.replace(r'\s', '', regex=True)
        .str.replace(r'\.(?=\d{3})', '', regex=True)
        .str.replace(',', '.', regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 10. Possui imóvel
df['possui_imovel'] = df['possui_imovel'].astype(str).str.strip().str.lower()
df['possui_imovel'] = df['possui_imovel'].replace(NULOS_DISFARÇADOS + ['nan'], np.nan)
df['possui_imovel'] = pd.to_numeric(df['possui_imovel'], errors='coerce').astype('Int64')

# 11. Tempo residência
df['tempo_residencia_anos'] = pd.to_numeric(df['tempo_residencia_anos'], errors='coerce')
mediana_residencia = df['tempo_residencia_anos'].median()
df['tempo_residencia_anos'] = df['tempo_residencia_anos'].fillna(mediana_residencia).astype(int)

# 12. Imputação — categóricas
for col in ['genero', 'estado_civil', 'tem_filhos', 'escolaridade']:
    df[col] = imputar_categorica(df[col])

# 13. Imputação — numéricas
for col in ['idade', 'renda_anual', 'valor_imovel', 'qtd_dependentes', 'possui_imovel']:
    df[col] = imputar_amostra(df[col])

# 14. Remover duplicatas
df.drop_duplicates(subset='id_cliente', keep='first', inplace=True)

# 15. Outliers
for col in ['renda_anual', 'valor_imovel']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=max(0, Q1 - 1.5 * IQR), upper=Q3 + 1.5 * IQR)

df['idade'] = df['idade'].astype(float)
df['idade'] = df['idade'].where(df['idade'].between(18, 100), other=np.nan)
df['idade'] = imputar_amostra(df['idade'])

# 16. Correção de tipos
df['tem_filhos'] = df['tem_filhos'].astype(int)
df['possui_imovel'] = df['possui_imovel'].astype(int)
df['idade'] = df['idade'].astype(int)

# 17. Correção de inconsistências
mask = (df['tem_filhos'] == 0) & (df['qtd_dependentes'] > 0)
df.loc[mask, 'tem_filhos'] = 1

df_cadastro_tratado = df