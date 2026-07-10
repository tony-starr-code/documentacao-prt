import pandas as pd
import numpy as np

def main():
    s3_path = 's3://t2rpt/'
    df_mkt = pd.read_csv(f'{s3_path}engajamento_marketing.csv')

    # PADRONIZAÇÃO

    # ID
    # Garante que seja string e sem espaços, igualando à base de contratos
    df_mkt['ID'] = df_mkt['ID'].astype(str).str.strip()
    df_mkt.rename(columns={"ID" : "id_cliente"}, inplace=True)
    
    
    # Colunas numéricas
    # errors='coerce' converte tudo que for erro/texto para NaN
    numeric_cols = [
        'score_engajamento_digital', 
        'indicou_clientes', 
        'renovacoes_consecutivas', 
        'indice_relacionamento', 
        'ano_veiculo', 
        'km_anual_estimado', 
        'ultimo_login_portal_dias', 
        'score_propensao_churn', 
        'cluster_sugerido_crm'
    ]
    
    for col in numeric_cols:
        df_mkt[col] = pd.to_numeric(df_mkt[col], errors='coerce')
    
    # Colunas categóricas
    # Tirar espaços, normalizar maiuscula/minuscula, nulos disfarçados
    def normalizar_texto(valor):
        if pd.isna(valor):
            return np.nan
        
        # Remove espaços excedentes do começo e fim
        v = str(valor).strip()
        
        # Capturar os nulos disfarçados que sobraram
        if v.upper() in ['', '-', '?', '#N/D', 'NAN']:
            return np.nan
            
        # Padroniza para primeira letra maiúscula (ex: '  Moto' -> 'Moto')
        return v.title()
    
    cat_cols = ['tipo_veiculo', 'segmento_marketing', 'regiao_vendas']
    
    for col in cat_cols:
        df_mkt[col] = df_mkt[col].apply(normalizar_texto)
    
    # Regiões do brasil - dimensionality reduction
    # Padronizando nomenclaturas diferentes que significam a mesma região
    # Premissa/base: regiões oficiais do IBGE
    mapa_regiao = {
        'Oeste': 'Centro-Oeste',
        'Regiao Oeste': 'Centro-Oeste',
        'Centro': 'Centro-Oeste'
    }
    
    # Aplica o mapa apenas para as chaves correspondentes, o resto mantém o valor atual
    df_mkt['regiao_vendas'] = df_mkt['regiao_vendas'].replace(mapa_regiao)


    # LIMPEZA/TRATAMENTO

    # Considerei que, se o login é nulo, ent o cliente nunca logou. Criei coluna nova pra isso
    # Se o login é nulo, nunca_logou recebe 1. Caso contrário, 0.
    df_mkt['nunca_logou'] = df_mkt['ultimo_login_portal_dias'].isna().astype(int)
    
    # Imputo com mediana pra EDA
    mediana_login = df_mkt['ultimo_login_portal_dias'].median()
    df_mkt['ultimo_login_portal_dias'] = df_mkt['ultimo_login_portal_dias'].fillna(mediana_login)
    
    # Indicações e Renovações (Não preencheu = Zero)
    df_mkt['indicou_clientes'] = df_mkt['indicou_clientes'].fillna(0)
    df_mkt['renovacoes_consecutivas'] = df_mkt['renovacoes_consecutivas'].fillna(0)
    
    
    # Pras variáveis categórias, preencho com a moda
    # ~5% nulos
    moda_regiao = df_mkt['regiao_vendas'].mode()[0]
    df_mkt['regiao_vendas'] = df_mkt['regiao_vendas'].fillna(moda_regiao)
    moda_segmento = df_mkt['segmento_marketing'].mode()[0]
    df_mkt['segmento_marketing'] = df_mkt['segmento_marketing'].fillna(moda_segmento)
    
    # Pra tipo de veículo, imputação de mediana por grupo de segmento mkt
    def moda_segura(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else np.nan
    
    # Tentativa 1: Moda do tipo_veiculo agrupada pelo segmento_marketing
    df_mkt['tipo_veiculo'] = df_mkt['tipo_veiculo'].fillna(
        df_mkt.groupby('segmento_marketing')['tipo_veiculo'].transform(moda_segura)
    )
    
    # Pra quem segmento tbm era nulo, uso moda
    moda_veiculo_global = df_mkt['tipo_veiculo'].mode()[0]
    df_mkt['tipo_veiculo'] = df_mkt['tipo_veiculo'].fillna(moda_veiculo_global)
    
    
    # Numéricas: imputo por grupo tbm
    colunas_veiculo = ['ano_veiculo', 'km_anual_estimado']
    
    for col in colunas_veiculo:
        # Tentativa 1: Mediana agrupada pelo tipo_veiculo
        df_mkt[col] = df_mkt[col].fillna(
            df_mkt.groupby('tipo_veiculo')[col].transform('median')
        )
        # Tentativa 2 (Fallback): Mediana global
        df_mkt[col] = df_mkt[col].fillna(df_mkt[col].median())
    
    colunas_score = ['score_engajamento_digital', 'indice_relacionamento']
    
    for col in colunas_score:
        # Zera o score APENAS para os clientes que nunca logaram (flag == 1)
        mask_nunca_logou = df_mkt['nunca_logou'] == 1
        df_mkt.loc[mask_nunca_logou, col] = df_mkt.loc[mask_nunca_logou, col].fillna(0)
        
        # Para os demais clientes (que logaram, mas deu bug e perdeu o score): Mediana Global
        df_mkt[col] = df_mkt[col].fillna(df_mkt[col].median())
    
    # B. Colunas Restritas para a EDA (Não irão para o modelo final)
    df_mkt['score_propensao_churn'] = df_mkt['score_propensao_churn'].fillna(df_mkt['score_propensao_churn'].median())
    
    moda_cluster = df_mkt['cluster_sugerido_crm'].mode()[0]
    df_mkt['cluster_sugerido_crm'] = df_mkt['cluster_sugerido_crm'].fillna(moda_cluster)
    
    # Duplicatas
    df_mkt = df_mkt.drop_duplicates()

    caminho_saida = f'../bases_tratadas/engajamento_marketing_tratado.csv'
    df_mkt.to_csv(caminho_saida, index=False)
    
    return df_mkt

if __name__ == "__main__":
    df_mkt_tratado = main()