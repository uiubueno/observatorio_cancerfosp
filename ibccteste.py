import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
import plotly.express as px
import plotly.graph_objects as go
import requests
import io
import time

# ==========================================
# CONFIGURAÇÃO DA PÁGINA E UI CUSTOMIZADA
# ==========================================
st.set_page_config(page_title="Observatório Oncológico", page_icon="🧬", layout="wide")

# ==========================================
# HACK DE UX/UI: INJEÇÃO DE CSS ADAPTÁVEL (DARK MODE)
# ==========================================
st.markdown("""
    <style>
    /* 1. Importando Fonte Moderna (Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
    }

    /* 2. Animação Suave de Entrada (Fade In) */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main .block-container {
        animation: fadeIn 0.6s ease-out;
        padding-top: 2rem;
    }

    /* 3. Estilização dos Cartões de KPI Dinâmicos */
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border-radius: 12px;
        padding: 15px 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border-left: 6px solid #b8860b; /* IBCC Dourado */
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: var(--text-color);
        opacity: 0.8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 32px !important;
        color: var(--text-color);
        font-weight: 700;
    }

    /* 4. Estilização das Abas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 2px solid var(--secondary-background-color);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 25px;
        font-size: 15px;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.6;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--secondary-background-color);
        color: var(--text-color) !important;
        opacity: 1;
        border-bottom: 4px solid #b8860b !important; /* IBCC Dourado */
    }

    /* 5. Títulos, Divisores e Citações (Blockquotes) */
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    hr {
        border-color: var(--text-color);
        opacity: 0.1;
    }
    blockquote {
        border-left: 4px solid #b8860b;
        background-color: var(--secondary-background-color);
        padding: 15px 20px;
        border-radius: 0px 8px 8px 0px;
        color: var(--text-color);
    }
    </style>
""", unsafe_allow_html=True)

# DICIONÁRIOS GLOBAIS DE TRADUÇÃO DA FOSP
DIC_SEXO = {1: 'MASCULINO', 2: 'FEMININO'}
DIC_ESCOLARIDADE = {1: 'ANALFABETO', 2: 'ENS. FUNDAMENTAL INCOMPLETO', 3: 'ENS. FUNDAMENTAL COMPLETO', 4: 'ENSINO MÉDIO', 5: 'SUPERIOR', 9: 'IGNORADA'}
DIC_ATENDIMENTO = {1: 'CONVÊNIO', 2: 'SUS', 3: 'PARTICULAR', 9: 'SEM INFORMAÇÃO'}
DIC_DIAGPREV = {1: 'SEM DIAGNÓSTICO/SEM TRATAMENTO', 2: 'COM DIAGNÓSTICO/SEM TRATAMENTO', 3: 'COM DIAGNÓSTICO/COM TRATAMENTO', 4: 'OUTROS'}
DIC_BASEDIAG = {1: 'EXAME CLÍNICO', 2: 'RECURSOS AUXILIARES NÃO MICROSCÓPICOS', 3: 'CONFIRMAÇÃO MICROSCÓPICA', 4: 'SEM INFORMAÇÃO'}
DIC_ULTINFO = {1: 'VIVO, COM CÂNCER', 2: 'VIVO, SOE', 3: 'ÓBITO POR CÂNCER', 4: 'ÓBITO POR OUTRAS CAUSAS, SOE'}
DIC_SIM_NAO = {0: 'NÃO', 1: 'SIM'}
DIC_HORMONIO = {0: 'SEM', 1: 'COM'}
DIC_LATERALIDADE = {1: 'DIREITA', 2: 'ESQUERDA', 3: 'BILATERAL', 8: 'NÃO SE APLICA'}
DIC_TRAT_COMBO = {'A': 'Cirurgia', 'B': 'Radioterapia', 'C': 'Quimioterapia', 'D': 'Cirurgia + Radioterapia', 'E': 'Cirurgia + Quimioterapia', 'F': 'Radioterapia + Quimioterapia', 'G': 'Cirurgia + Radioterapia + Quimioterapia', 'H': 'Cirurgia + Radioterapia + Quimioterapia + Hormonioterapia', 'I': 'Outras combinações', 'J': 'Nenhum tratamento'}
DIC_CLINICA = {1: 'ALERGIA/IMUNOLOGIA', 2: 'CIRURGIA CARDIACA', 3: 'CIRURGIA CABEÇA E PESCOÇO', 4: 'CIRURGIA GERAL', 5: 'CIRURGIA PEDIATRICA', 6: 'CIRURGIA PLASTICA', 7: 'CIRURGIA TORAXICA', 8: 'CIRURGIA VASCULAR', 9: 'CLINICA MEDICA', 10: 'DERMATOLOGIA', 11: 'ENDOCRINOLOGIA', 12: 'GASTROCIRURGIA', 13: 'GASTROENTEROLOGIA', 14: 'GERIATRIA', 15: 'GINECOLOGIA', 16: 'GINECOLOGIA/OBSTETRICIA', 17: 'HEMATOLOGIA', 18: 'INFECTOLOGIA', 19: 'NEFROLOGIA', 20: 'NEUROCIRURGIA', 21: 'NEUROLOGIA', 22: 'OFTALMOLOGIA', 23: 'ONCOLOGIA CIRURGICA', 24: 'ONCOLOGIA CLINICA', 25: 'ONCOLOGIA PEDIATRICA', 26: 'ORTOPEDIA', 27: 'OTORRINOLARINGOLOGIA', 28: 'PEDIATRIA', 29: 'PNEUMOLOGIA', 30: 'PROCTOLOGIA', 31: 'RADIOTERAPIA', 32: 'UROLOGIA', 33: 'MASTOLOGIA', 34: 'ONCOLOGIA CUTANEA', 35: 'CIRURGIA PELVICA', 36: 'CIRURGIA ABDOMINAL', 37: 'ODONTOLOGIA', 38: 'TRANSPLANTE HEPATICO', 99: 'IGNORADO'}

IBGE_RMSP = [
    '3503901', '3505708', '3506607', '3509007', '3509205', '3510609', '3513009', '3513801',
    '3515004', '3515103', '3515707', '3516309', '3516408', '3518305', '3518800', '3522208',
    '3522505', '3523107', '3525003', '3526209', '3528502', '3529401', '3530607', '3534401',
    '3539301', '3539806', '3543303', '3543402', '3545001', '3546801', '3547304', '3547809',
    '3548708', '3548807', '3550001', '3552502', '3552809', '3556404'
]

@st.cache_data
def carregar_malha_sp():
    url = "https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-35-mun.json"
    try:
        return requests.get(url).json()
    except:
        return None

def download_plot(fig, filename):
    buf = io.BytesIO()
    try:
        if hasattr(fig, 'savefig'):
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=300, facecolor='white', edgecolor='white')
        elif hasattr(fig, 'write_image'):
            fig.write_image(buf, format="png", scale=3)
    except Exception as e:
        st.error(f"Erro ao gerar imagem. Rode 'python3 -m pip install kaleido'. Detalhe: {e}")
        return
        
    buf.seek(0)
    st.download_button(
        label="📥 Baixar Gráfico/Mapa em Alta Resolução (PNG)",
        data=buf,
        file_name=filename,
        mime="image/png",
        use_container_width=True
    )

# ==========================================
# BARRA LATERAL: MENU DE NAVEGAÇÃO
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965306.png", width=65)
    st.markdown("## Hub RHC FOSP")
    
    modo_sistema = st.radio("Módulo Ativo:", ["📈 Observatório Oncológico", "🔄 Tradutor de Planilha"], label_visibility="collapsed")
    st.divider()

    if modo_sistema == "🔄 Tradutor de Planilha":
        st.markdown("#### 📂 Importar Base Bruta")
        planilha_codificada = st.file_uploader("Arraste o arquivo .XLSX", type=["xlsx", "xls"], key="tradutor_up")
    else:
        st.markdown("#### 📂 Importar Base Traduzida")
        arquivo_upado = st.file_uploader("Arraste o arquivo .XLSX", type=["xlsx", "xls"], key="observatorio_up")
        
        if arquivo_upado:
            st.divider()
            st.markdown("#### ⚙️ Parâmetros Globais")
            filtro_escopo = st.selectbox("Escopo Assistencial", ['Todos os Casos da Planilha', 'Apenas Analíticos', 'Apenas Não Analíticos'])
            filtro_sexo = st.selectbox("Gênero", ['Ambos', 'MASCULINO', 'FEMININO'])
            filtro_estadio = st.selectbox("Estadio Clínico", ['Todos', 'Todos (exceto 0 in situ)', '0 (in situ)', 'I', 'II', 'III', 'IV', 'Outros'])
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🎨 Identidade Visual")
            with st.expander("Customizar Cores dos Gráficos"):
                cor_primaria = st.color_picker("Cor Primária (Geral)", "#1a2b4c")
                cor_fem = st.color_picker("Cor Feminino", "#85299d")
                cor_masc = st.color_picker("Cor Masculino", "#517cbe")
                cor_secundaria = st.color_picker("Cor Secundária (Destaque)", "#b8860b")
        else:
            cor_primaria = "#1a2b4c"
            cor_fem = "#85299d"
            cor_masc = "#517cbe"
            cor_secundaria = "#b8860b"

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("🔒 Seus dados são processados localmente e apagados ao fechar a janela.")

# ==========================================
# FLUXO 1: TRADUTOR
# ==========================================
if modo_sistema == "🔄 Tradutor de Planilha":
    st.title("🔄 Motor de Conversão de Dados")
    st.markdown("Tradutor automatizado de dicionários FOSP para linguagem natural.")
    
    with st.container():
        tipo_exportacao = st.radio(
            "Configuração de Filtro Raiz:",
            ["Gerar Base Completa (Recomendado)", "Pré-filtrar e Gerar Apenas Casos Analíticos"],
            horizontal=True
        )
    
    if not planilha_codificada:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Como funciona?")
            st.markdown("""
            1. Você envia a planilha bruta do sistema hospitalar.
            2. Nosso algoritmo cruza os números com o dicionário oficial da FOSP.
            3. A base final sai texturizada, pronta para leitura humana e importação no Observatório.
            """)
        with col2:
            st.info("👈 Faça o upload do arquivo no painel esquerdo para iniciar a magia.")
    else:
        with st.spinner("🤖 Processando engenharia de dados e aplicando dicionários..."):
            try:
                xls_t = pd.ExcelFile(planilha_codificada)
                sheet_t = next((s for s in xls_t.sheet_names if 'RHC' in s.upper() and 'ANAL' in s.upper()), xls_t.sheet_names[0])
                df_t = pd.read_excel(xls_t, sheet_name=sheet_t)
                
                def traduzir_coluna(dataframe, coluna, dicionario):
                    if coluna in dataframe.columns:
                        dataframe[coluna] = pd.to_numeric(dataframe[coluna], errors='coerce').map(dicionario).fillna(dataframe[coluna])
                
                def traduzir_texto(dataframe, coluna, dicionario):
                    if coluna in dataframe.columns:
                        dataframe[coluna] = dataframe[coluna].astype(str).str.strip().map(dicionario).fillna(dataframe[coluna])

                traduzir_coluna(df_t, 'SEXO', DIC_SEXO)
                traduzir_coluna(df_t, 'ESCOLARI', DIC_ESCOLARIDADE)
                traduzir_coluna(df_t, 'CATEATEND', DIC_ATENDIMENTO)
                traduzir_coluna(df_t, 'DIAGPREV', DIC_DIAGPREV)
                traduzir_coluna(df_t, 'BASEDIAG', DIC_BASEDIAG)
                traduzir_coluna(df_t, 'ULTINFO', DIC_ULTINFO)
                traduzir_coluna(df_t, 'LATERALI', DIC_LATERALIDADE)
                traduzir_coluna(df_t, 'CLINICA', DIC_CLINICA)

                colunas_sim_nao = ['NENHUM', 'CIRURGIA', 'RADIO', 'QUIMIO', 'TMO', 'IMUNO', 'OUTROS','NENHUMANT', 'CIRURANT', 'RADIOANT', 'QUIMIOANT', 'TMOANT', 'IMUNOANT', 'OUTROANT','NENHUMAPOS', 'CIRURAPOS', 'RADIOAPOS', 'QUIMIOAPOS', 'TMOAPOS', 'IMUNOAPOS', 'OUTROAPOS','RECNENHUM', 'RECLOCAL', 'RECREGIO', 'RECDIST', 'PERDASEG']
                for col in colunas_sim_nao: traduzir_coluna(df_t, col, DIC_SIM_NAO)

                for col in ['HORMONIO', 'HORMOANT', 'HORMOAPOS']: traduzir_coluna(df_t, col, DIC_HORMONIO)
                for col in ['TRATAMENTO', 'TRATHOSP', 'TRATFANTES', 'TRATFAPOS']: traduzir_texto(df_t, col, DIC_TRAT_COMBO)
                
                mensagem_extra = ""
                if "Apenas Casos Analíticos" in tipo_exportacao:
                    if 'DIAGPREV' in df_t.columns:
                        mask_t = df_t['DIAGPREV'].astype(str).str.upper().str.contains(r'SEM TRATAMENTO|^1$|^2$|1\.0|2\.0|NAN|NONE', regex=True, na=True)
                        df_t = df_t[mask_t]
                        mensagem_extra = " (Filtro Exclusivo de Casos Analíticos Aplicado na raiz do arquivo)"
                
                st.toast('Engenharia de dados concluída!', icon='✅')
                time.sleep(0.5)
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_t.to_excel(writer, index=False, sheet_name=sheet_t)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.success(f"Base convertida com perfeição! Total de {len(df_t):,} registros processados.".replace(",", "."))
                
                st.download_button(
                    label="💾 Download Planilha FOSP Traduzida (Pronta para Análise)",
                    data=buffer.getvalue(),
                    file_name="Base_FOSP_Traduzida.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
            except Exception as e:
                st.error(f"Ocorreu um erro no mapeamento das colunas: {e}")
    st.stop()

# ==========================================
# FLUXO 2: OBSERVATÓRIO
# ==========================================
if not arquivo_upado:
    st.title("📈 Centro de Inteligência Oncológica")
    st.markdown("Bem-vindo ao observatório executivo de dados assistenciais e estatísticos.")
    
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 🧩 Recursos do Motor Analítico")
        st.markdown("""
        - 🛡️ **Auditoria Ativa:** Limpeza automática de anomalias temporais e de cadastro.
        - 🔬 **Governança CID-O3:** Validação rigorosa de topografia e morfologia oncológica.
        - 📊 **Métricas de Sobrevida:** Curvas de Kaplan-Meier com cálculo automático de IC95%.
        - 🗺️ **Geoprocessamento:** Discretização categórica via malha oficial do IBGE.
        """)
    with col2:
        st.info("👈 **Para ativar o painel, arraste a planilha traduzida no menu lateral esquerdo.**")
    st.stop()

# ==========================================
# PROCESSAMENTO DE DADOS E AUDITORIA
# ==========================================
@st.cache_data
def carregar_dados(arquivo):
    xls = pd.ExcelFile(arquivo)
    sheet_name = next((s for s in xls.sheet_names if 'RHC' in s.upper() and 'ANAL' in s.upper()), xls.sheet_names[0])
    
    df_temp = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=15)
    header_idx = 0
    for i, row in df_temp.iterrows():
        row_str = " ".join(row.dropna().astype(str)).upper()
        if 'SEXO' in row_str and ('DIAG' in row_str or 'IDADE' in row_str or 'TOPO' in row_str):
            header_idx = i
            break
            
    df = pd.read_excel(xls, sheet_name=sheet_name, header=header_idx)
    
    df['Linha do Excel (Aprox)'] = df.index + header_idx + 2
    
    def achar_coluna(opcoes):
        for col in df.columns:
            if str(col).strip().upper() in [o.strip().upper() for o in opcoes]: return col
        for col in df.columns:
            for o in opcoes:
                if o.strip().upper() in str(col).strip().upper(): return col
        return opcoes[0] 

    col_diag = achar_coluna(['DTDIAG', 'DATA DO DIAGNÓSTICO', 'DATA DO DIAGNOSTICO', 'DATADIAG'])
    col_fim = achar_coluna(['DTULTINFO', 'DATA DA ÚLTIMA INFORMAÇÃO', 'DATA DA ULTIMA INFORMACAO', 'PACINETE', 'PACIENTE'])
    col_status = achar_coluna(['ULTINFO', 'ÚLTIMA INFORMAÇÃO', 'ULTIMA INFORMACAO'])
    col_topo_cod = achar_coluna(['TOPO', 'CÓDIGO DA TOPOGRAFIA', 'CODIGO DA TOPOGRAFIA'])
    col_morfo = achar_coluna(['DESCMORFO', 'DESCRIÇÃO DA MORFOLOGIA', 'DESCRICAO DA MORFOLOGIA', 'MORFOLOGIA'])
    col_estadio = achar_coluna(['ECGRUP', 'GRUPO DO ESTÁDIO', 'GRUPO DO ESTADIO', 'ESTÁDIO CLÍNICO', 'ESTADIO CLINICO'])
    col_idade = achar_coluna(['IDADE', 'IDADE DO PACIENTE'])
    col_sexo = achar_coluna(['SEXO'])
    col_atendimento = achar_coluna(['CATEATEND', 'CATEGORIA DE ATENDIMENTO', 'ATENDIMENTO'])
    col_cons = achar_coluna(['DTCONSULT', 'DATA DA PRIMEIRA CONSULTA', 'CONSULTA'])
    col_trat = achar_coluna(['DTTRAT', 'DATA DO INÍCIO DO PRIMEIRO TRATAMENTO', 'DATA DO INICIO DO PRIMEIRO TRATAMENTO', 'TRATAMENTO'])
    col_basediag = achar_coluna(['BASEDIAG', 'BASE DE DIAGNÓSTICO', 'BASE DE DIAGNOSTICO'])
    col_uf = achar_coluna(['UFRESID', 'UF DE RESIDÊNCIA', 'ESTADO'])
    col_ibge = achar_coluna(['IBGE', 'CÓDIGO IBGE'])
    col_cidade = achar_coluna(['CIDADE', 'MUNICÍPIO DE RESIDÊNCIA', 'MUNICIPIO'])
    col_diagprev = achar_coluna(['DIAGPREV', 'DIAGNÓSTICO PRÉVIO', 'DIAGNOSTICO PREVIO', 'TRATAMENTO ANTERIOR'])
    col_ufnasc = achar_coluna(['UFNASC', 'UF DE NASCIMENTO', 'NATURALIDADE'])
    col_escolari = achar_coluna(['ESCOLARI', 'ESCOLARIDADE'])
    
    if col_diag not in df.columns:
        raise ValueError(f"O painel não achou a coluna '{col_diag}'. Verifique se o Excel possui cabeçalhos formatados.")
        
    df[col_diag] = pd.to_datetime(df[col_diag], errors='coerce')
    df[col_fim] = pd.to_datetime(df[col_fim], errors='coerce')
    df[col_cons] = pd.to_datetime(df[col_cons], errors='coerce')
    df[col_trat] = pd.to_datetime(df[col_trat], errors='coerce')
    
    # Processamento Extra: Escolaridade
    if col_escolari in df.columns:
        df['Escolaridade'] = df[col_escolari].astype(str).str.upper().str.strip()
        df['Escolaridade'] = df['Escolaridade'].replace({
            'NAN': 'IGNORADA', 'NONE': 'IGNORADA', '': 'IGNORADA', 
            'NÃO INFORMADO': 'IGNORADA', 'NAO INFORMADO': 'IGNORADA'
        })
        mapa_esc_num = {'1': 'ANALFABETO', '2': 'ENS. FUNDAMENTAL INCOMPLETO', '3': 'ENS. FUNDAMENTAL COMPLETO', '4': 'ENSINO MÉDIO', '5': 'SUPERIOR', '9': 'IGNORADA', '1.0': 'ANALFABETO', '2.0': 'ENS. FUNDAMENTAL INCOMPLETO', '3.0': 'ENS. FUNDAMENTAL COMPLETO', '4.0': 'ENSINO MÉDIO', '5.0': 'SUPERIOR', '9.0': 'IGNORADA'}
        df['Escolaridade'] = df['Escolaridade'].map(lambda x: mapa_esc_num.get(x, x))
    else:
        df['Escolaridade'] = 'IGNORADA'
        
    # Processamento Extra: Naturalidade
    if col_ufnasc in df.columns:
        df['UFNASC_raw'] = df[col_ufnasc].astype(str).str.upper().str.strip()
    else:
        df['UFNASC_raw'] = '99'
        
    valid_ufs = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO']
    
    def classificar_naturalidade(uf):
        if uf in valid_ufs: return 'Brasileiros'
        elif uf in ['99', 'NAN', 'NONE', '', '99.0']: return 'Não Informado'
        else: return 'Estrangeiros'
        
    def classificar_naturalidade_uf(uf):
        if uf in valid_ufs: return uf
        elif uf in ['99', 'NAN', 'NONE', '', '99.0']: return 'Não Informado'
        else: return 'Estrangeiros'
        
    df['Naturalidade'] = df['UFNASC_raw'].apply(classificar_naturalidade)
    df['Naturalidade_UF'] = df['UFNASC_raw'].apply(classificar_naturalidade_uf)
    
    # Processamento Extra: Combinações de Tratamento
    col_trathosp_candidates = [c for c in df.columns if str(c).strip().upper() in ['TRATHOSP', 'TRATAMENTO', 'COMBINAÇÃO DE TRATAMENTO', 'COMBINACAO DE TRATAMENTO']]
    if not col_trathosp_candidates:
        col_trathosp_candidates = [c for c in df.columns if 'TRATHOSP' in str(c).upper() or ('TRATAMENTO' in str(c).upper() and 'DATA' not in str(c).upper() and 'DT' not in str(c).upper())]
    
    col_trathosp = col_trathosp_candidates[0] if col_trathosp_candidates else 'TRATHOSP_MOCK'
    if col_trathosp not in df.columns:
        df[col_trathosp] = 'NÃO INFORMADO'
    
    def padronizar_tratamento(x):
        val = str(x).upper().strip()
        if val in ['NAN', 'NONE', '']: return 'NÃO INFORMADO'
        for k, v in DIC_TRAT_COMBO.items():
            if val == str(k).upper(): 
                val = str(v).upper()
                break
        
        # Expansão Científica de Nomenclaturas
        val = val.replace('RADIO ', 'RADIOTERAPIA ').replace('+ RADIO +', '+ RADIOTERAPIA +')
        if val.endswith(' RADIO'): val = val.replace(' RADIO', ' RADIOTERAPIA')
        if val == 'RADIO': val = 'RADIOTERAPIA'
        
        val = val.replace('QUIMIO ', 'QUIMIOTERAPIA ').replace('+ QUIMIO +', '+ QUIMIOTERAPIA +')
        if val.endswith(' QUIMIO'): val = val.replace(' QUIMIO', ' QUIMIOTERAPIA')
        if val == 'QUIMIO': val = 'QUIMIOTERAPIA'
        
        val = val.replace('HORMONIO', 'HORMONIOTERAPIA').replace('HORMÔNIO', 'HORMONIOTERAPIA')
        
        if val == 'OUTRAS COMBINAÇÕES' or val == 'OUTRAS COMBINACOES': return 'OUTRAS COMBINAÇÕES DE TRATAMENTO'
        if val == 'NENHUM TRATAMENTO': return 'NENHUM TRATAMENTO REALIZADO'
        
        return val
        
    df['Tratamento_Consolidado'] = df[col_trathosp].apply(padronizar_tratamento)
    
    # Cálculos e Auditoria
    df['Tempo_Meses'] = (df[col_fim] - df[col_diag]).dt.days / 30.4375
    df['Idade Numérica'] = pd.to_numeric(df[col_idade], errors='coerce')
    
    df_erro_datas = df[df['Tempo_Meses'] < 0][['Linha do Excel (Aprox)', col_diag, col_fim, 'Tempo_Meses']].copy()
    if not df_erro_datas.empty:
        df_erro_datas[col_diag] = df_erro_datas[col_diag].dt.strftime('%d/%m/%Y')
        df_erro_datas[col_fim] = df_erro_datas[col_fim].dt.strftime('%d/%m/%Y')
    
    mask_cid_valido = df[col_topo_cod].astype(str).str.upper().str.match(r'^[CD]\d{2}', na=False)
    df_erro_cids = df[~mask_cid_valido][['Linha do Excel (Aprox)', col_topo_cod, col_morfo]].copy()
    
    df_erro_idades = df[(df['Idade Numérica'] > 120) | (df['Idade Numérica'] < 0)][['Linha do Excel (Aprox)', col_idade]].copy()
    
    mask_mun_invalido = df[col_ibge].isna() | df[col_ibge].astype(str).str.strip().isin(['', '0', '0.0', '9999999', '999999', 'nan', 'NAN'])
    df_erro_muns = df[mask_mun_invalido][['Linha do Excel (Aprox)', col_cidade, col_ibge]].copy()

    df['Status_Evento'] = df[col_status].apply(lambda x: 1 if isinstance(x, str) and 'OBITO' in x.upper().replace('Ó', 'O') else 0)
    
    df = df.dropna(subset=['Tempo_Meses', 'Status_Evento', col_diag])
    df = df[df['Tempo_Meses'] >= 0] 
    
    df['Ano_Diag'] = df[col_diag].dt.year
    df['Sexo'] = df[col_sexo].astype(str).str.upper()
    
    df['Dias_Cons_Diag'] = (df[col_diag] - df[col_cons]).dt.days
    df['Dias_Diag_Trat'] = (df[col_trat] - df[col_diag]).dt.days
    df['Dias_Cons_Trat'] = (df[col_trat] - df[col_cons]).dt.days
    
    if col_diagprev in df.columns: 
        df['DIAGPREV_BUSCA'] = df[col_diagprev].astype(str).str.upper().str.strip()
    else: 
        df['DIAGPREV_BUSCA'] = 'ANALITICO_PADRAO'
    
    if col_uf in df.columns: df['UFRESID'] = df[col_uf].astype(str).str.upper().str.strip()
    else: df['UFRESID'] = 'SP'
    
    if col_ibge in df.columns: df['IBGE'] = df[col_ibge].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    else: df['IBGE'] = '0'
        
    if col_cidade in df.columns: df['CIDADE'] = df[col_cidade].astype(str).str.upper().str.strip()
    else: df['CIDADE'] = 'NÃO INFORMADO'
    
    def limpar_atendimento(x):
        val = str(x).upper().strip()
        if val.endswith('.0'): val = val[:-2]
        if val == '1' or 'SUS' in val or 'SISTEMA UNICO' in val or 'SISTEMA ÚNICO' in val: return 'SUS'
        elif val == '2' or 'PARTICULAR' in val: return 'Particular'
        elif val == '3' or 'CONVENIO' in val or 'CONVÊNIO' in val or 'PLANO' in val or 'SUPLEMENTAR' in val: return 'Convênio (Saúde Suplementar)'
        elif val == '4' or val == '9' or val == 'NAN' or val == '' or val == 'NONE': return 'Não Informado'
        else: return 'Outros'
    df['Categoria_Atendimento'] = df[col_atendimento].apply(limpar_atendimento)
    
    def agrupar_basediag(x):
        val = str(x).upper().strip()
        if 'CONFIRMA' in val and 'MICROSC' in val: return 'Confirmação Microscópica'
        else: return 'Outros exames'
    df['Base_Diagnostico'] = df[col_basediag].apply(agrupar_basediag)
    
    def definir_quinquenio(ano):
        if pd.isna(ano): return 'Outros'
        try:
            inicio = int(ano) - (int(ano) % 5)
            return f"{inicio}-{inicio+4}"
        except: return 'Outros'
    df['Quinquenio'] = df['Ano_Diag'].apply(definir_quinquenio)
    
    def agrupar_macro_topografia(linha):
        cod_topo = str(linha.get(col_topo_cod, '')).upper().strip()
        morfo_desc = str(linha.get(col_morfo, '')).upper()
        if cod_topo.startswith('C50'): return 'Mama'
        elif cod_topo.startswith('C73'): return 'Tireoide'
        elif cod_topo.startswith('C61'): return 'Próstata'
        elif cod_topo.startswith('C53'): return 'Colo do útero'
        elif cod_topo.startswith('C51'): return 'Vulva'
        elif cod_topo.startswith('C54') or cod_topo.startswith('C55'): return 'Corpo do útero'
        elif cod_topo.startswith('C56'): return 'Ovário'
        elif cod_topo.startswith('C18') or cod_topo.startswith('C19') or cod_topo.startswith('C20'): return 'Cólon e reto'
        elif any(cod_topo.startswith(c) for c in ['C01', 'C00', 'C02', 'C03', 'C04', 'C05', 'C06', 'C09', 'C10']): return 'Cavidade oral e orofaringe'
        elif cod_topo.startswith('C44'): return 'Pele - melanoma' if 'MELANOMA' in morfo_desc else 'Pele - não-melanoma'
        else: return 'Outros'
    df['Macro_Topografia'] = df.apply(agrupar_macro_topografia, axis=1)

    def agrupar_macro_completa(linha):
        cod_topo = str(linha.get(col_topo_cod, '')).upper().strip()
        c3 = cod_topo[:3]
        morfo_desc = str(linha.get(col_morfo, '')).upper()
        if c3 == 'C50': return 'Mama'
        elif c3 == 'C73': return 'Tireoide'
        elif c3 == 'C61': return 'Próstata'
        elif c3 == 'C53': return 'Colo do útero'
        elif c3 == 'C51': return 'Vulva'
        elif c3 in ['C54', 'C55']: return 'Corpo do útero'
        elif c3 == 'C56': return 'Ovário'
        elif c3 in ['C18', 'C19', 'C20']: return 'Cólon e reto'
        elif c3 in ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C09', 'C10']: return 'Cavidade oral e orofaringe'
        elif c3 == 'C44': return 'Pele - melanoma' if 'MELANOMA' in morfo_desc else 'Pele - não-melanoma'
        elif c3 == 'C00': return 'Lábio'
        elif c3 in ['C07', 'C08']: return 'Glândulas salivares'
        elif c3 == 'C11': return 'Nasofaringe'
        elif c3 in ['C12', 'C13']: return 'Hipofaringe'
        elif c3 == 'C14': return 'Faringe SOE'
        elif c3 == 'C15': return 'Esôfago'
        elif c3 == 'C16': return 'Estômago'
        elif c3 == 'C17': return 'Intestino delgado'
        elif c3 == 'C21': return 'Canal anal e Ânus'
        elif c3 == 'C22': return 'Fígado e vias biliares intra-hepáticas'
        elif c3 in ['C23', 'C24']: return 'Vesícula e vias biliares extras'
        elif c3 == 'C25': return 'Pâncreas'
        elif c3 == 'C26': return 'Aparelho digestivo SOE'
        elif c3 == 'C30': return 'Cavidade nasal e ouvido médio'
        elif c3 == 'C31': return 'Seios paranasais'
        elif c3 == 'C32': return 'Laringe'
        elif c3 in ['C33', 'C34']: return 'Traqueia, brônquios e pulmões'
        elif c3 in ['C37', 'C38']: return 'Timo, coração e mediastino'
        elif c3 in ['C40', 'C41']: return 'Ossos e cartilagens articulares'
        elif c3 == 'C42': return 'Sistema hematopoético (Leucemias e Mielomas)'
        elif c3 == 'C47': return 'Nervos periféricos e autônomo'
        elif c3 == 'C48': return 'Retroperitônio e peritônio'
        elif c3 == 'C49': return 'Tecido conjuntivo e partes moles'
        elif c3 == 'C52': return 'Vagina'
        elif c3 == 'C57': return 'Outros órgãos genitais femininos'
        elif c3 == 'C58': return 'Placenta'
        elif c3 == 'C60': return 'Pênis'
        elif c3 == 'C62': return 'Testículo'
        elif c3 == 'C63': return 'Outros órgãos genitais masculinos'
        elif c3 == 'C64': return 'Rim'
        elif c3 in ['C65', 'C66']: return 'Pelve renal e ureter'
        elif c3 == 'C67': return 'Bexiga'
        elif c3 == 'C68': return 'Outros órgãos urinários'
        elif c3 == 'C69': return 'Olho e anexos'
        elif c3 == 'C70': return 'Meninges'
        elif c3 == 'C71': return 'Encéfalo (Tumores Cerebrais)'
        elif c3 == 'C72': return 'Medula espinhal e nervos cranianos'
        elif c3 in ['C74', 'C75']: return 'Suprarrenal e endócrinas'
        elif c3 == 'C76': return 'Localizações mal definidas'
        elif c3 == 'C77': return 'Linfonodos (Linfomas)'
        elif c3 == 'C80': return 'Sítio primário desconhecido'
        elif not c3: return 'Não preenchido'
        else: return f'Outros (CID {c3})'
    df['Macro_Topografia_Completa'] = df.apply(agrupar_macro_completa, axis=1)
    
    if col_estadio in df.columns:
        def limpar_estadio(x):
            val = str(x).strip().upper()
            if val == '0': return '0 (in situ)'
            if val in ['I', 'II', 'III', 'IV']: return val
            return 'Outros'
        df['Estadio_Clinico'] = df[col_estadio].apply(limpar_estadio)
    else:
        df['Estadio_Clinico'] = 'Outros'
        
    metricas_auditoria = {
        'validos': len(df),
        'datas': len(df_erro_datas),
        'cids': len(df_erro_cids),
        'idades': len(df_erro_idades),
        'muns': len(df_erro_muns),
        'df_datas': df_erro_datas,
        'df_cids': df_erro_cids,
        'df_idades': df_erro_idades,
        'df_muns': df_erro_muns
    }
    
    return df, metricas_auditoria

with st.spinner('Acessando servidor e compilando data frame...'):
    try:
        df_base, metricas_auditoria = carregar_dados(arquivo_upado)
        st.toast('Observatório carregado e pronto para análise!', icon='🚀')
    except Exception as e:
        st.error(f"Erro na Leitura da Planilha! Verifique se você enviou o arquivo traduzido correto.\n\nDetalhe técnico: {e}")
        st.stop()

# Configurações Dinâmicas de Data
ano_min_df = int(df_base['Ano_Diag'].min())
ano_max_df = int(df_base['Ano_Diag'].max())

# ===== APLICAÇÃO DOS FILTROS GLOBAIS =====
df_filtrado = df_base.copy()

if filtro_escopo == 'Apenas Analíticos':
    mask_analitico = df_filtrado['DIAGPREV_BUSCA'].str.contains(r'SEM TRATAMENTO|^1$|^2$|1\.0|2\.0|ANALITICO_PADRAO|NAN|NONE', regex=True, na=True)
    df_filtrado = df_filtrado[mask_analitico]
elif filtro_escopo == 'Apenas Não Analíticos':
    mask_nao_analitico = df_filtrado['DIAGPREV_BUSCA'].str.contains(r'COM TRATAMENTO|OUTROS|^3$|^4$|3\.0|4\.0|NÃO|NAO', regex=True, na=False)
    df_filtrado = df_filtrado[mask_nao_analitico]

if filtro_sexo != 'Ambos':
    df_filtrado = df_filtrado[df_filtrado['Sexo'] == filtro_sexo]

if filtro_estadio == 'Todos (exceto 0 in situ)':
    df_filtrado = df_filtrado[df_filtrado['Estadio_Clinico'] != '0 (in situ)']
elif filtro_estadio != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['Estadio_Clinico'] == filtro_estadio]
# ==========================================

# Configurações de Cores e Estilos para Gráficos
COR_AZUL_ESCURO = cor_primaria
COR_DOURADO = cor_secundaria

# ==========================================
# PALETA GLOBAL DE GÊNERO ATUALIZADA
# ==========================================
CORES_SEXO = {'FEMININO': cor_fem, 'MASCULINO': cor_masc}

CORES_TOP10 = [cor_primaria, '#3a7a78', cor_secundaria, '#7b2e3a', '#4a70a3', '#7590b1', '#999999', '#c7a13a', '#7b5592', '#366666']

# Dicionário Científico de CIDs para Legendas
DIC_CIDS_MACRO = {
    'Mama': 'C50',
    'Tireoide': 'C73',
    'Próstata': 'C61',
    'Colo do útero': 'C53',
    'Vulva': 'C51',
    'Corpo do útero': 'C54-C55',
    'Ovário': 'C56',
    'Cólon e reto': 'C18-C20',
    'Cavidade oral e orofaringe': 'C00-C06, C09, C10',
    'Pele - melanoma': 'C44',
    'Pele - não-melanoma': 'C44',
}

# Forçar fundo branco para os gráficos do Matplotlib (para Word)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#555555'
plt.rcParams['ytick.color'] = '#555555'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial']

def configurar_eixos_grafico(ax, titulo):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.set_title(titulo, color=COR_AZUL_ESCURO, fontweight='bold', pad=20, fontsize=18)
    ax.set_xlabel('Meses desde o diagnóstico', fontsize=14, color='#333333')
    ax.set_ylabel('Probabilidade de sobrevida', fontsize=14, color='#333333')
    ax.tick_params(axis='both', which='major', labelsize=12, colors='#555555')
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.05)
    ax.axvline(x=60, color='gray', linestyle='--', linewidth=1, alpha=0.5)

def extrair_metrica_60_meses(k):
    try:
        s = k.predict(60) * 100
        ci = k.confidence_interval_
        tempos_validos = ci.index[ci.index <= 60]
        if len(tempos_validos) > 0:
            t_max = tempos_validos[-1]
            l = ci.loc[t_max].iloc[0] * 100
            u = ci.loc[t_max].iloc[1] * 100
        else:
            l, u = s, s
        return s, l, u
    except: return 0.0, 0.0, 0.0

# ==========================================
# MOTOR ESTATÍSTICO LOG-RANK
# ==========================================
def extrair_pvalue_logrank(df_teste, col_grupo):
    grupos_unicos = df_teste[col_grupo].dropna().unique()
    if len(grupos_unicos) > 1:
        try:
            res = multivariate_logrank_test(df_teste['Tempo_Meses'], df_teste[col_grupo], df_teste['Status_Evento'])
            if res.p_value < 0.001:
                return "Log-Rank: p < 0,001"
            else:
                return f"Log-Rank: p = {res.p_value:.3f}".replace('.', ',')
        except:
            return "Log-Rank: p N/A"
    return ""

def achar_coluna_local(df_local, opcoes):
    for col in df_local.columns:
        if str(col).strip().upper() in [o.strip().upper() for o in opcoes]: return col
    for col in df_local.columns:
        for o in opcoes:
            if o.strip().upper() in str(col).strip().upper(): return col
    return df_local.columns[0]

# Abas Superiores
aba_auditoria, aba_perfil, aba_sobrevida, aba_jornada = st.tabs(["🛡️ Auditoria Data Quality", "👥 Perfil Demográfico", "📈 Análise de Sobrevida", "⏱️ Jornada Assistencial"])

# ==========================================
# VISÃO: AUDITORIA DE DADOS
# ==========================================
with aba_auditoria:
    st.markdown("## Validação e Qualidade dos Dados")
    st.markdown("O algoritmo monitora e isola discrepâncias na fonte para assegurar rigor científico e exatidão estatística (Kaplan-Meier e tempos assistenciais).")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_aud1, col_aud2 = st.columns(2)
    with col_aud1:
        st.success(f"**Integridade da Base:**\n\n✓ **{metricas_auditoria['validos']:,}** linhas auditadas e prontas para processamento analítico.".replace(',', '.'))
        st.info("💡 **Ação do Motor:** Inconsistências temporais foram extirpadas do pipeline de sobrevida. Erros demográficos foram remanejados para a categoria 'Não Informado' visando manter o volume total do Censo.")
        
    with col_aud2:
        st.warning(f"""**Rastreador de Anomalias:**
* ⚠ **{metricas_auditoria['datas']:,}** anomalias cronológicas temporais.
* ⚠ **{metricas_auditoria['cids']:,}** CIDs ausentes ou atípicos.
* ⚠ **{metricas_auditoria['muns']:,}** lacunas de georreferenciamento (IBGE).
* ⚠ **{metricas_auditoria['idades']:,}** discrepâncias biológicas de idade.
        """.replace(',', '.'))
        
    st.markdown("---")
    st.markdown("### 🔎 Terminal de Correção (Localizador Excel)")
    
    if metricas_auditoria['datas'] > 0:
        with st.expander(f"📅 Inspecionar {metricas_auditoria['datas']} anomalias cronológicas"):
            st.dataframe(metricas_auditoria['df_datas'], use_container_width=True, hide_index=True)
            
    if metricas_auditoria['cids'] > 0:
        with st.expander(f"🧬 Inspecionar {metricas_auditoria['cids']} CIDs inválidos"):
            st.dataframe(metricas_auditoria['df_cids'], use_container_width=True, hide_index=True)
            
    if metricas_auditoria['idades'] > 0:
        with st.expander(f"🎂 Inspecionar {metricas_auditoria['idades']} idades atípicas"):
            st.dataframe(metricas_auditoria['df_idades'], use_container_width=True, hide_index=True)
            
    if metricas_auditoria['muns'] > 0:
        with st.expander(f"🗺️ Inspecionar {metricas_auditoria['muns']} falhas de localização"):
            st.dataframe(metricas_auditoria['df_muns'], use_container_width=True, hide_index=True)

# ==========================================
# VISÃO: PERFIL EPIDEMIOLÓGICO
# ==========================================
with aba_perfil:
    st.markdown("<br>", unsafe_allow_html=True)
    col_f1, col_f2 = st.columns([1, 2.5])
    with col_f1:
        anos_perfil = st.slider("Recorte Temporal:", min_value=ano_min_df, max_value=ano_max_df, value=(ano_min_df, ano_max_df))
    with col_f2:
        st.markdown("<br>", unsafe_allow_html=True)
        visao_freq = st.selectbox("Eixo de Análise Visual:", ["Evolução Histórica (Casos por Ano)", "Evolução por Ano e Sexo (Linhas)", "Distribuição por Faixa Etária (Linhas)", "Top 10 Grupos Principais", "Top 10 Comparativo (Homens vs Mulheres)", "Todas as Neoplasias (Grupos Anatômicos)", "Categoria de Atendimento (Pizza)", "Base de Diagnóstico (Pizza)", "Perfil de Tratamento (Barras)", "Escolaridade (Barras)", "Escolaridade vs Estadio Clínico", "Naturalidade (Brasil vs Estrangeiros) (Pizza)", "Naturalidade (Por Estado) (Barras)", "Distribuição Geográfica (Rosca)", "Distribuição Geográfica (Mapa)"], label_visibility="collapsed")
    
    df_perfil = df_filtrado[(df_filtrado['Ano_Diag'] >= anos_perfil[0]) & (df_filtrado['Ano_Diag'] <= anos_perfil[1])]
    df_base_ano = df_base[(df_base['Ano_Diag'] >= anos_perfil[0]) & (df_base['Ano_Diag'] <= anos_perfil[1])].copy()
    texto_ano_titulo = f"{anos_perfil[0]}–{anos_perfil[1]}"
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Volume do Filtro Selecionado", f"{len(df_perfil):,}".replace(",", "."))
    col2.metric("Média Etária do Grupo", f"{df_perfil['Idade Numérica'].mean():.1f} anos")
    col3.metric("Neoplasia Predominante", df_perfil['Macro_Topografia'].value_counts().index[0] if len(df_perfil) > 0 else "N/A")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if len(df_perfil) > 0:
        total_casos_perfil = len(df_perfil)
        
        if visao_freq == "Evolução Histórica (Casos por Ano)":
            st.markdown("## Casos novos de câncer por ano de diagnóstico")
            
            total_geral_anos = len(df_base_ano)
            mask_ana_anos = df_base_ano['DIAGPREV_BUSCA'].str.contains(r'SEM TRATAMENTO|^1$|^2$|1\.0|2\.0|ANALITICO_PADRAO|NAN|NONE', regex=True, na=True)
            casos_ana_anos = len(df_base_ano[mask_ana_anos])
            perc_ana = (casos_ana_anos / total_geral_anos) * 100 if total_geral_anos > 0 else 0
            
            if total_geral_anos == casos_ana_anos:
                st.markdown(f"""
                > 📝 **Draft de Documentação:** Entre {anos_perfil[0]} e {anos_perfil[1]}, compondo a base deste Observatório, foram analisados **{casos_ana_anos:,}** pacientes. Como a base submetida ao painel já está filtrada ou padronizada para casos analíticos, ela representa 100% da casuística do período selecionado. O número anual variou ao longo dos anos, refletindo tanto mudanças na demanda assistencial quanto o amadurecimento do processo de registro na instituição.
                """.replace(',', 'X').replace('.', ',').replace('X', '.'))
            else:
                st.markdown(f"""
                > 📝 **Draft de Documentação:** Entre {anos_perfil[0]} e {anos_perfil[1]}, o RHC registrou **{total_geral_anos:,}** casos de câncer, dos quais **{casos_ana_anos:,} ({perc_ana:.1f}%)** foram classificados como analíticos e compuseram a base principal de análise. O número anual variou ao longo do período, refletindo tanto mudanças na demanda assistencial quanto o amadurecimento do processo de registro na instituição.
                """.replace(',', 'X').replace('.', ',').replace('X', '.'))
            
            titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"Casos novos de câncer por ano de diagnóstico — RHC, {texto_ano_titulo}", key="title_hist")
            
            casos_por_ano = df_perfil['Ano_Diag'].value_counts().sort_index()
            
            fig_ano, ax_ano = plt.subplots(figsize=(12, 6))
            bars = ax_ano.bar(casos_por_ano.index, casos_por_ano.values, color=COR_AZUL_ESCURO, edgecolor='white', width=0.75)
            
            ax_ano.set_xticks(casos_por_ano.index)
            ax_ano.set_xticklabels(casos_por_ano.index, rotation=45, fontsize=12, color='#555555')
            
            # Limpeza das bordas para o visual AC Camargo
            ax_ano.spines['top'].set_visible(False)
            ax_ano.spines['right'].set_visible(False)
            ax_ano.spines['left'].set_visible(False) 
            ax_ano.spines['bottom'].set_color('#cccccc')
            
            ax_ano.set_xlabel('Ano de diagnóstico', fontsize=14, labelpad=10, color='#333333')
            ax_ano.set_ylabel('Número de casos novos', fontsize=14, color='#333333')
            
            # Grade contínua (linha sólida) e ticks zerados
            ax_ano.yaxis.grid(True, linestyle='-', color='#e0e0e0', alpha=1.0)
            ax_ano.set_axisbelow(True)
            ax_ano.tick_params(axis='y', length=0, labelsize=12, colors='#555555')
            ax_ano.tick_params(axis='x', length=0)
            
            ax_ano.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=20, fontsize=18)
            
            # Lógica para inserir os números na vertical dentro da base de cada barra
            max_y = max(casos_por_ano.values) if len(casos_por_ano) > 0 else 100
            for bar in bars:
                altura = bar.get_height()
                if altura > 0:
                    # Se a barra for muito curtinha, coloca o texto em cima de preto, senão dentro de branco
                    if altura > max_y * 0.15:
                        ax_ano.text(
                            bar.get_x() + bar.get_width() / 2,
                            max_y * 0.02, # Leve recuo da base
                            f"{int(altura):,}".replace(',', '.'),
                            ha='center', va='bottom', rotation=90, color='white', fontsize=11, fontweight='bold'
                        )
                    else:
                        ax_ano.text(
                            bar.get_x() + bar.get_width() / 2,
                            altura + (max_y * 0.02),
                            f"{int(altura):,}".replace(',', '.'),
                            ha='center', va='bottom', rotation=90, color='#333333', fontsize=11, fontweight='bold'
                        )
            
            st.pyplot(fig_ano)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1: download_plot(fig_ano, "Evolucao_Casos_Ano.png")
            
            df_tabela_ano = casos_por_ano.reset_index()
            df_tabela_ano.columns = ['Ano de Diagnóstico', 'Número de Casos']
            df_tabela_ano['% do total'] = (df_tabela_ano['Número de Casos'] / total_casos_perfil * 100).apply(lambda x: f"{x:.1f}%".replace('.', ','))
            df_tabela_ano['Número de Casos'] = df_tabela_ano['Número de Casos'].apply(lambda x: f"{x:,}".replace(',', '.'))
            
            linha_total = pd.DataFrame([{
                'Ano de Diagnóstico': 'TOTAL (100% da Base)',
                'Número de Casos': f"{total_casos_perfil:,}".replace(',', '.'),
                '% do total': '100,0%'
            }])
            df_tabela_ano = pd.concat([df_tabela_ano, linha_total], ignore_index=True)
            
            with st.expander("📂 Inspecionar Dataframe Bruto"):
                st.dataframe(df_tabela_ano, use_container_width=True, hide_index=True)

        elif visao_freq == "Evolução por Ano e Sexo (Linhas)":
            st.markdown("## Evolução de casos novos por ano e sexo")
            
            st.markdown(f"""
            > 📝 **Draft de Documentação:** A distribuição dos casos por sexo ao longo do período confirma o perfil assistencial predominantemente voltado à saúde da mulher que caracteriza a instituição, com franca predominância de casos femininos em todos os anos da série histórica.
            """)
            
            titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"Casos novos de câncer por ano e sexo — RHC, {texto_ano_titulo}", key="title_linhas_sexo")
            
            df_tendencia = df_perfil.groupby(['Ano_Diag', 'Sexo']).size().unstack(fill_value=0)
            
            fig_ano_sexo, ax_ano_sexo = plt.subplots(figsize=(12, 6))
            
            ultimo_ano = df_tendencia.index[-1]
            total_ultimo_ano = df_tendencia.loc[ultimo_ano].sum()
            
            if 'FEMININO' in df_tendencia.columns:
                ax_ano_sexo.plot(df_tendencia.index, df_tendencia['FEMININO'], marker='o', linewidth=2.5, markersize=5, color=CORES_SEXO['FEMININO'], label='Feminino')
                
                val_fem = df_tendencia['FEMININO'].iloc[-1]
                pct_fem = (val_fem / total_ultimo_ano) * 100 if total_ultimo_ano > 0 else 0
                ax_ano_sexo.text(ultimo_ano + 0.3, val_fem, f"{pct_fem:.1f}%", color=CORES_SEXO['FEMININO'], va='center', fontweight='bold', fontsize=12)

            if 'MASCULINO' in df_tendencia.columns:
                ax_ano_sexo.plot(df_tendencia.index, df_tendencia['MASCULINO'], marker='o', linewidth=2.5, markersize=5, color=CORES_SEXO['MASCULINO'], label='Masculino')
                
                val_masc = df_tendencia['MASCULINO'].iloc[-1]
                pct_masc = (val_masc / total_ultimo_ano) * 100 if total_ultimo_ano > 0 else 0
                ax_ano_sexo.text(ultimo_ano + 0.3, val_masc, f"{pct_masc:.1f}%", color=CORES_SEXO['MASCULINO'], va='center', fontweight='bold', fontsize=12)
                
            ax_ano_sexo.set_xticks(df_tendencia.index)
            ax_ano_sexo.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
            
            ax_ano_sexo.set_xlim(df_tendencia.index[0] - 0.5, ultimo_ano + 2.0)
            
            ax_ano_sexo.spines['top'].set_visible(False)
            ax_ano_sexo.spines['right'].set_visible(False)
            ax_ano_sexo.spines['left'].set_color('#cccccc')
            ax_ano_sexo.spines['bottom'].set_color('#cccccc')
            ax_ano_sexo.set_xlabel('Ano de diagnóstico', fontsize=14, labelpad=10, color='#333333')
            ax_ano_sexo.set_ylabel('Número de casos', fontsize=14, color='#333333')
            ax_ano_sexo.tick_params(axis='both', which='major', labelsize=12, colors='#555555')
            ax_ano_sexo.legend(frameon=False, fontsize=12)
            
            ax_ano_sexo.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=20, fontsize=18)
            
            st.pyplot(fig_ano_sexo)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1: download_plot(fig_ano_sexo, "Evolucao_Ano_Sexo.png")
            
            df_tendencia_display = df_tendencia.copy()
            df_tendencia_display['Total'] = df_tendencia_display.sum(axis=1)
            
            total_fem_geral = df_tendencia_display['FEMININO'].sum() if 'FEMININO' in df_tendencia_display.columns else 0
            total_masc_geral = df_tendencia_display['MASCULINO'].sum() if 'MASCULINO' in df_tendencia_display.columns else 0
            total_casos_tendencia = df_tendencia_display['Total'].sum()
            
            df_tendencia_display['% do total'] = (df_tendencia_display['Total'] / total_casos_tendencia * 100).apply(lambda x: f"{x:.1f}%".replace('.', ','))
            
            if 'FEMININO' in df_tendencia_display.columns:
                df_tendencia_display['% Feminino'] = (df_tendencia_display['FEMININO'] / df_tendencia_display['Total'] * 100).apply(lambda x: f"{x:.1f}%".replace('.', ','))
                df_tendencia_display['FEMININO'] = df_tendencia_display['FEMININO'].apply(lambda x: f"{int(x):,}").str.replace(',', '.')
                
            if 'MASCULINO' in df_tendencia_display.columns:
                df_tendencia_display['% Masculino'] = (df_tendencia_display['MASCULINO'] / df_tendencia_display['Total'] * 100).apply(lambda x: f"{x:.1f}%".replace('.', ','))
                df_tendencia_display['MASCULINO'] = df_tendencia_display['MASCULINO'].apply(lambda x: f"{int(x):,}").str.replace(',', '.')
            
            df_tendencia_display['Total'] = df_tendencia_display['Total'].apply(lambda x: f"{int(x):,}").str.replace(',', '.')
            
            cols_order = []
            if 'FEMININO' in df_tendencia_display.columns: cols_order.extend(['FEMININO', '% Feminino'])
            if 'MASCULINO' in df_tendencia_display.columns: cols_order.extend(['MASCULINO', '% Masculino'])
            cols_order.extend(['Total', '% do total'])
            
            df_tendencia_display = df_tendencia_display[cols_order].reset_index()
            df_tendencia_display.rename(columns={'Ano_Diag': 'Ano de diagnóstico'}, inplace=True)
            
            pct_fem_geral = f"{(total_fem_geral/total_casos_tendencia)*100:.1f}%".replace(".", ",") if total_casos_tendencia > 0 else "0,0%"
            pct_masc_geral = f"{(total_masc_geral/total_casos_tendencia)*100:.1f}%".replace(".", ",") if total_casos_tendencia > 0 else "0,0%"
            
            linha_total_dict = {
                'Ano de diagnóstico': 'TOTAL (100% da Base)',
                'Total': f"{int(total_casos_tendencia):,}".replace(',', '.'),
                '% do total': '100,0%'
            }
            if 'FEMININO' in df_tendencia.columns:
                linha_total_dict['FEMININO'] = f"{int(total_fem_geral):,}".replace(',', '.')
                linha_total_dict['% Feminino'] = pct_fem_geral
            if 'MASCULINO' in df_tendencia.columns:
                linha_total_dict['MASCULINO'] = f"{int(total_masc_geral):,}".replace(',', '.')
                linha_total_dict['% Masculino'] = pct_masc_geral
                
            linha_total_df = pd.DataFrame([linha_total_dict])
            df_tendencia_display = pd.concat([df_tendencia_display, linha_total_df], ignore_index=True)
            
            with st.expander("📂 Inspecionar Dataframe Bruto"):
                st.dataframe(df_tendencia_display, use_container_width=True, hide_index=True)

        elif visao_freq == "Distribuição por Faixa Etária (Linhas)":
            st.markdown("## Distribuição dos casos por faixa etária e sexo")
            
            titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"Distribuição dos casos por faixa etária e sexo — RHC, {texto_ano_titulo}", key="title_idade_sexo")
            
            df_idade = df_perfil.dropna(subset=['Idade Numérica']).copy()
            
            tot_fem = len(df_idade[df_idade['Sexo'] == 'FEMININO'])
            tot_masc = len(df_idade[df_idade['Sexo'] == 'MASCULINO'])
            total_idade = len(df_idade)
            
            perc_fem = (tot_fem / total_idade) * 100 if total_idade > 0 else 0
            perc_masc = (tot_masc / total_idade) * 100 if total_idade > 0 else 0
            
            idade_media = df_idade['Idade Numérica'].mean()
            idade_mediana = df_idade['Idade Numérica'].median()
            
            bins = [-1, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 150]
            labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85+']
            df_idade['Faixa_Etaria'] = pd.cut(df_idade['Idade Numérica'], bins=bins, labels=labels)
            
            tabela_idade_sexo = pd.crosstab(df_idade['Faixa_Etaria'], df_idade['Sexo']).reindex(labels).fillna(0)
            
            faixa_counts = df_idade['Faixa_Etaria'].value_counts().reindex(labels).fillna(0)
            faixa_max = faixa_counts.idxmax()
            
            st.markdown(f"""
            > 📝 **Draft de Documentação:** Do total filtrado, **{tot_fem:,} ({perc_fem:.1f}%)** ocorreram em mulheres e **{tot_masc:,} ({perc_masc:.1f}%)** em homens. A idade média ao diagnóstico foi de **{idade_media:.2f} anos** (mediana de {idade_mediana:.0f} anos), com maior concentração de casos na faixa etária de **{faixa_max} anos**, compatível com o padrão etário esperado para os principais tipos de câncer atendidos na instituição.
            """.replace(',', 'X').replace('.', ',').replace('X', '.'))
            
            fig_idade, ax_idade = plt.subplots(figsize=(14, 8))
            x_coords = np.arange(len(labels))
            
            val_fem = tabela_idade_sexo['FEMININO'].values if 'FEMININO' in tabela_idade_sexo.columns else np.zeros(len(labels))
            val_masc = tabela_idade_sexo['MASCULINO'].values if 'MASCULINO' in tabela_idade_sexo.columns else np.zeros(len(labels))
            
            if tot_masc > 0:
                ax_idade.plot(x_coords, val_masc, color=CORES_SEXO['MASCULINO'], linewidth=3, label='Masculino')
            if tot_fem > 0:
                ax_idade.plot(x_coords, val_fem, color=CORES_SEXO['FEMININO'], linewidth=3, label='Feminino')
            
            ax_idade.set_xticks(x_coords)
            ax_idade.set_xticklabels(labels, rotation=45, ha='right', fontsize=12, color='#555555')
            ax_idade.spines['top'].set_visible(False)
            ax_idade.spines['right'].set_visible(False)
            ax_idade.spines['left'].set_visible(False)
            ax_idade.spines['bottom'].set_color('#cccccc')
            ax_idade.yaxis.grid(True, linestyle='-', alpha=0.5, color='#dddddd')
            ax_idade.set_axisbelow(True)
            
            ax_idade.set_xlabel('Faixa etária', fontsize=14, color='#333333', labelpad=15)
            ax_idade.set_ylabel('Número de casos novos', fontsize=14, color='#333333', labelpad=15)
            ax_idade.tick_params(axis='y', length=0, labelsize=12, colors='#555555')
            ax_idade.tick_params(axis='x', labelsize=12, colors='#555555')
            
            # Linhas Verticais Divisórias
            v_lines = [3.5, 7.5, 11.5]
            for v in v_lines:
                ax_idade.axvline(x=v, color='#333333', linestyle=':', linewidth=1, alpha=0.5)
            
            # Cálculo de Percentuais por Fase da Vida
            def calc_stage_pct(start_idx, end_idx):
                m = val_masc[start_idx:end_idx+1].sum() / tot_masc * 100 if tot_masc > 0 else 0
                f = val_fem[start_idx:end_idx+1].sum() / tot_fem * 100 if tot_fem > 0 else 0
                return m, f
                
            m1, f1 = calc_stage_pct(0, 3) # Crianças/Adolescentes (0-19)
            m2, f2 = calc_stage_pct(4, 7) # Adultos jovens (20-39)
            m3, f3 = calc_stage_pct(8, 11) # Adultos (40-59)
            m4, f4 = calc_stage_pct(12, 17) # Idosos (60+)
            
            max_y = max(val_fem.max() if tot_fem > 0 else 0, val_masc.max() if tot_masc > 0 else 0)
            y_text = max_y * 1.15
            ax_idade.set_ylim(0, max_y * 1.3)
            
            ax_idade.text(1.5, y_text, f"Crianças e adolescentes\n{m1:.0f}% (M) | {f1:.0f}% (F)", ha='center', va='center', fontsize=11, color='#333333')
            ax_idade.text(5.5, y_text, f"Adultos jovens\n{m2:.0f}% (M) | {f2:.0f}% (F)", ha='center', va='center', fontsize=11, color='#333333')
            ax_idade.text(9.5, y_text, f"Adultos\n{m3:.0f}% (M) | {f3:.0f}% (F)", ha='center', va='center', fontsize=11, color='#333333')
            ax_idade.text(14.5, y_text, f"Idosos\n{m4:.0f}% (M) | {f4:.0f}% (F)", ha='center', va='center', fontsize=11, color='#333333')
            
            # Legenda centralizada embaixo
            ax_idade.legend(frameon=False, fontsize=13, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
            
            ax_idade.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=30, fontsize=18)
            plt.subplots_adjust(bottom=0.25)
            
            st.pyplot(fig_idade)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1: download_plot(fig_idade, "Faixa_Etaria_Sexo_Linhas.png")
            
            val_fem_tab = tabela_idade_sexo['FEMININO'].values if 'FEMININO' in tabela_idade_sexo.columns else np.zeros(len(labels))
            val_masc_tab = tabela_idade_sexo['MASCULINO'].values if 'MASCULINO' in tabela_idade_sexo.columns else np.zeros(len(labels))
            
            df_tabela_idade = pd.DataFrame({
                'Faixa Etária': labels,
                'FEMININO': val_fem_tab,
                'MASCULINO': val_masc_tab
            })
            
            df_tabela_idade['Total'] = df_tabela_idade['FEMININO'] + df_tabela_idade['MASCULINO']
            
            df_tabela_idade['% Feminino'] = df_tabela_idade.apply(lambda row: f"{(row['FEMININO']/row['Total']*100):.1f}%".replace('.', ',') if row['Total'] > 0 else "0,0%", axis=1)
            df_tabela_idade['% Masculino'] = df_tabela_idade.apply(lambda row: f"{(row['MASCULINO']/row['Total']*100):.1f}%".replace('.', ',') if row['Total'] > 0 else "0,0%", axis=1)
            
            df_tabela_idade['FEMININO'] = df_tabela_idade['FEMININO'].astype(int).apply(lambda x: f"{x:,}".replace(',', '.'))
            df_tabela_idade['MASCULINO'] = df_tabela_idade['MASCULINO'].astype(int).apply(lambda x: f"{x:,}".replace(',', '.'))
            df_tabela_idade['Total'] = df_tabela_idade['Total'].astype(int).apply(lambda x: f"{x:,}".replace(',', '.'))
            
            cols_order = ['Faixa Etária', 'FEMININO', '% Feminino', 'MASCULINO', '% Masculino', 'Total']
            df_tabela_idade = df_tabela_idade[cols_order]
            
            with st.expander("📂 Inspecionar Dataframe Bruto"):
                st.dataframe(df_tabela_idade, use_container_width=True, hide_index=True)

        elif visao_freq == "Escolaridade (Barras)":
            st.markdown("## Distribuição por Nível de Escolaridade")
            
            titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"Distribuição da escolaridade no momento do diagnóstico — RHC, {texto_ano_titulo}", key="title_escolaridade")
            
            ordem_esc = ['ANALFABETO', 'ENS. FUNDAMENTAL INCOMPLETO', 'ENS. FUNDAMENTAL COMPLETO', 'ENSINO MÉDIO', 'SUPERIOR', 'IGNORADA']
            
            esc_data = df_perfil['Escolaridade'].value_counts()
            
            # Garantir que todas as categorias da ordem existam, mesmo com zero, para não quebrar o gráfico
            for cat in ordem_esc:
                if cat not in esc_data:
                    esc_data[cat] = 0
                    
            # Reordenar mantendo a lógica de progressão educacional
            esc_data = esc_data.reindex(ordem_esc)
            
            df_grafico = pd.DataFrame({
                "Escolaridade": esc_data.index,
                "N_raw": esc_data.values
            })
            
            # Inverter para o gráfico de barras horizontais (o primeiro da lista fica no topo)
            df_grafico = df_grafico.iloc[::-1]
            
            fig_esc, ax_esc = plt.subplots(figsize=(12, 7))
            y_coords = np.arange(len(df_grafico))
            
            cores_barras = [cor_primaria if cat != 'IGNORADA' else '#999999' for cat in df_grafico["Escolaridade"]]
                
            bars = ax_esc.barh(y_coords, df_grafico['N_raw'], color=cores_barras, edgecolor='white')
            ax_esc.set_yticks(y_coords)
            ax_esc.set_yticklabels(df_grafico["Escolaridade"], fontsize=13, color='#555555')
            
            ax_esc.spines['top'].set_visible(False)
            ax_esc.spines['right'].set_visible(False)
            ax_esc.spines['left'].set_color('#cccccc')
            ax_esc.spines['bottom'].set_color('#cccccc')
            ax_esc.set_xlabel('Número de casos', fontsize=14, color='#333333')
            ax_esc.tick_params(axis='x', labelsize=12, colors='#555555')
            ax_esc.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=15, fontsize=18)
            
            for i, bar in enumerate(bars):
                val = df_grafico.iloc[i]['N_raw']
                ax_esc.text(val + (total_casos_perfil * 0.005), bar.get_y() + bar.get_height()/2, f"{val:,}".replace(",", "."), va='center', ha='left', color='#333333', fontsize=12)
                
            st.pyplot(fig_esc)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1: download_plot(fig_esc, "Escolaridade_Barras.png")
            
            # Tabela Matemática
            dados_tabela_esc = []
            
            for esc in ordem_esc:
                count = esc_data[esc]
                pct = (count / total_casos_perfil) * 100 if total_casos_perfil > 0 else 0
                dados_tabela_esc.append({
                    "Nível de Escolaridade": esc,
                    "Nº de casos": f"{count:,}".replace(",", "."),
                    "% do total": f"{pct:.1f}%".replace(".", ",")
                })
                
            dados_tabela_esc.append({
                "Nível de Escolaridade": "TOTAL (100% da Base)",
                "Nº de casos": f"{total_casos_perfil:,}".replace(",", "."),
                "% do total": "100,0%"
            })
            
            df_tab_esc = pd.DataFrame(dados_tabela_esc)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df_tab_esc, use_container_width=True, hide_index=True)
            
            st.markdown(f"""
            > 📝 **Draft de Documentação:** A análise do perfil educacional da coorte evidencia a distribuição da instrução formal dos pacientes atendidos na instituição. Os dados refletem as condições socioeconômicas da população adscrita e são fundamentais para o planejamento das abordagens de comunicação e letramento em saúde durante o fluxo assistencial.
            """)

        elif visao_freq == "Escolaridade vs Estadio Clínico":
            st.markdown("## Impacto da Escolaridade no Estadiamento Clínico (100%)")
            
            titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"Distribuição do Estadio Clínico por Nível de Escolaridade — RHC, {texto_ano_titulo}", key="title_esc_est")
            
            ordem_esc = ['ANALFABETO', 'ENS. FUNDAMENTAL INCOMPLETO', 'ENS. FUNDAMENTAL COMPLETO', 'ENSINO MÉDIO', 'SUPERIOR', 'IGNORADA']
            ordem_est = ['0 (in situ)', 'I', 'II', 'III', 'IV', 'Outros']
            
            cores_estadio = {
                '0 (in situ)': '#3a7a78',
                'I': COR_AZUL_ESCURO,
                'II': COR_DOURADO,
                'III': '#7b2e3a',
                'IV': '#4a4a4a',
                'Outros': '#999999'
            }
            
            df_esc_est = df_perfil.copy()
            
            # Garantir que todos os estadios existam na matriz, mesmo que vazios
            tabela_cruzada = pd.crosstab(df_esc_est['Escolaridade'], df_esc_est['Estadio_Clinico'])
            
            for esc in ordem_esc:
                if esc not in tabela_cruzada.index:
                    tabela_cruzada.loc[esc] = 0
            for est in ordem_est:
                if est not in tabela_cruzada.columns:
                    tabela_cruzada[est] = 0
                    
            tabela_cruzada = tabela_cruzada.reindex(index=ordem_esc, columns=ordem_est).fillna(0)
            tabela_pct = tabela_cruzada.div(tabela_cruzada.sum(axis=1), axis=0).fillna(0) * 100
            
            fig_esc_est, ax_esc_est = plt.subplots(figsize=(14, 8))
            
            # Inverter a ordem para o gráfico (Analfabeto no topo)
            y_coords = np.arange(len(ordem_esc))[::-1]
            bottom = np.zeros(len(ordem_esc))
            
            for est in ordem_est:
                valores_pct = tabela_pct[est].values
                bars = ax_esc_est.barh(y_coords, valores_pct, left=bottom, color=cores_estadio[est], edgecolor='white', height=0.6, label=f"Estadio {est}")
                
                # Injetar os números dentro das barras se a fatia for maior que 5%
                for i, bar in enumerate(bars):
                    if valores_pct[i] > 5:
                        ax_esc_est.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2, f"{valores_pct[i]:.1f}%".replace('.', ','), ha='center', va='center', color='white', fontsize=11, fontweight='bold')
                
                bottom += valores_pct
                
            ax_esc_est.set_yticks(y_coords)
            ax_esc_est.set_yticklabels(ordem_esc, fontsize=13, color='#555555')
            
            ax_esc_est.spines['top'].set_visible(False)
            ax_esc_est.spines['right'].set_visible(False)
            ax_esc_est.spines['left'].set_color('#cccccc')
            ax_esc_est.spines['bottom'].set_color('#cccccc')
            ax_esc_est.set_xlabel('Distribuição de Estadiamento (%)', fontsize=14, color='#333333')
            ax_esc_est.tick_params(axis='x', labelsize=12, colors='#555555')
            ax_esc_est.set_xlim(0, 100)
            
            # Legenda no topo
            ax_esc_est.legend(frameon=False, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6)
            
            ax_esc_est.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=40, fontsize=18)
            
            st.pyplot(fig_esc_est)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1: download_plot(fig_esc_est, "Escolaridade_vs_Estadio.png")
            
            st.markdown(f"""
            > 📝 **Draft de Documentação:** Este gráfico de barras 100% empilhadas ilustra a distribuição do estadio clínico no momento do diagnóstico estratificada pelo nível de escolaridade do paciente. Esta visualização epidemiológica permite correlacionar a instrução formal com possíveis hiatos no rastreamento precoce (refletidos por um aumento de estadios III e IV nas faixas de menor escolaridade).
            """)
            
            # Gerar Tabela Expandida
            dados_tabela_cruzada = []
            
            for esc in ordem_esc:
                linha = {"Nível de Escolaridade": esc}
                total_linha = tabela_cruzada.loc[esc].sum()
                
                for est in ordem_est:
                    val = tabela_cruzada.loc[esc, est]
                    pct = (val / total_linha) * 100 if total_linha > 0 else 0
                    linha[f"Estadio {est}"] = f"{int(val):,}".replace(',', '.')
                    linha[f"% Estadio {est}"] = f"{pct:.1f}%".replace('.', ',')
                    
                linha["Total na Categoria"] = f"{int(total_linha):,}".replace(',', '.')
                dados_tabela_cruzada.append(linha)
                
            df_tab_cruzada = pd.DataFrame(dados_tabela_cruzada)
            
            st.markdown("#### Consolidação Cruzada: Volumes e Proporções")
            st.dataframe(df_tab_cruzada, use_container_width=True, hide_index=True)

        elif visao_freq == "Top 10 Grupos Principais":
            st.markdown("## As 10 Neoplasias Mais Frequentes (Macro)")
            
            titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"10 neoplasias mais frequentes — RHC, {texto_ano_titulo}", key="title_top10")
            
            # 1. Dados para o GRÁFICO (Apenas Top 10 real)
            top_10_reais = df_perfil[df_perfil['Macro_Topografia'] != 'Outros']['Macro_Topografia'].value_counts().head(10)
            
            df_grafico = pd.DataFrame({
                "Grupo": top_10_reais.index,
                "N_raw": top_10_reais.values
            })
            
            fig_top, ax_top = plt.subplots(figsize=(12, 7))
            y_coords = np.arange(len(top_10_reais))[::-1] 
            
            cores_barras = CORES_TOP10[:len(top_10_reais)]
                
            bars = ax_top.barh(y_coords, df_grafico['N_raw'], color=cores_barras, edgecolor='white')
            ax_top.set_yticks(y_coords)
            ax_top.set_yticklabels(df_grafico["Grupo"], fontsize=14, color='#555555')
            ax_top.spines['top'].set_visible(False)
            ax_top.spines['right'].set_visible(False)
            ax_top.spines['left'].set_color('#cccccc')
            ax_top.spines['bottom'].set_color('#cccccc')
            ax_top.set_xlabel('Número de casos', fontsize=14, color='#333333')
            ax_top.tick_params(axis='x', labelsize=12, colors='#555555')
            ax_top.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=15, fontsize=18)
            for i, bar in enumerate(bars):
                ax_top.text(df_grafico.iloc[i]['N_raw'] + (total_casos_perfil * 0.005), bar.get_y() + bar.get_height()/2, f"{df_grafico.iloc[i]['N_raw']:,}".replace(",", "."), va='center', ha='left', color='#333333', fontsize=12)
            st.pyplot(fig_top)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1: download_plot(fig_top, "Top10_Neoplasias.png")
            
            # 2. Dados para a TABELA (Top 10 + Outros + Total)
            casos_outros = total_casos_perfil - top_10_reais.sum()
            
            dados_tabela_top = []
            for grupo, count in top_10_reais.items():
                dados_tabela_top.append({
                    "Grupo de câncer": grupo, 
                    "Nº de casos": f"{count:,}".replace(",", "."), 
                    "% do total": f"{(count / total_casos_perfil) * 100:.1f}%".replace(".", ",")
                })
                
            if casos_outros > 0:
                dados_tabela_top.append({
                    "Grupo de câncer": "Outros (Demais Neoplasias)", 
                    "Nº de casos": f"{casos_outros:,}".replace(",", "."), 
                    "% do total": f"{(casos_outros / total_casos_perfil) * 100:.1f}%".replace(".", ",")
                })
                
            dados_tabela_top.append({
                "Grupo de câncer": "TOTAL (100% da Base)", 
                "Nº de casos": f"{total_casos_perfil:,}".replace(",", "."), 
                "% do total": "100,0%"
            })
            
            df_display = pd.DataFrame(dados_tabela_top)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        elif visao_freq == "Top 10 Comparativo (Homens vs Mulheres)":
            st.markdown("## Top 10 Neoplasias: Comparativo Homens vs Mulheres")
            
            titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"Comparativo por Sexo das 10 Neoplasias Mais Frequentes — RHC, {texto_ano_titulo}", key="title_comp_sexo")
            
            df_comp = df_perfil[df_perfil['Macro_Topografia'] != 'Outros'].copy()
            
            # Puxa o Top 10 Geral (como era originalmente)
            top_10_gerais = df_comp['Macro_Topografia'].value_counts().head(10).index
            tabela_sexo = pd.crosstab(df_comp[df_comp['Macro_Topografia'].isin(top_10_gerais)]['Macro_Topografia'], df_comp['Sexo'])
            if 'MASCULINO' not in tabela_sexo: tabela_sexo['MASCULINO'] = 0
            if 'FEMININO' not in tabela_sexo: tabela_sexo['FEMININO'] = 0
            tabela_sexo['Total'] = tabela_sexo['MASCULINO'] + tabela_sexo['FEMININO']
            
            # Ordena pelo volume total da doença, assim Mama fica no topo se for o maior
            tabela_sexo = tabela_sexo.sort_values(by='Total', ascending=True)
            
            fig_comp, ax_comp = plt.subplots(figsize=(14, 8))
            y_coords = np.arange(len(tabela_sexo))
            
            fem_vals = tabela_sexo['FEMININO'].values
            masc_vals = tabela_sexo['MASCULINO'].values
            labels_cat = tabela_sexo.index
            
            # As barras femininas crescem para a esquerda (negativo) e masculinas para a direita (positivo)
            ax_comp.barh(y_coords, -fem_vals, color=CORES_SEXO['FEMININO'], edgecolor='white', height=0.7)
            ax_comp.barh(y_coords, masc_vals, color=CORES_SEXO['MASCULINO'], edgecolor='white', height=0.7)
            
            ax_comp.set_yticks([]) # Esconde o eixo Y do meio para não sujar o gráfico
            ax_comp.spines['top'].set_visible(False)
            ax_comp.spines['right'].set_visible(False)
            ax_comp.spines['left'].set_visible(False)
            ax_comp.spines['bottom'].set_color('#cccccc')
            ax_comp.axvline(0, color='#cccccc', linewidth=1) # A linha central do tornado
            ax_comp.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{abs(int(x)):,}".replace(",", ".")))
            ax_comp.tick_params(axis='x', labelsize=12, colors='#555555')
            ax_comp.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=30, fontsize=18)
            
            # Calcula o limite máximo do eixo X para dar espaço para o texto
            max_abs = max(max(fem_vals) if len(fem_vals)>0 else 0, max(masc_vals) if len(masc_vals)>0 else 0)
            limit = max_abs * 1.5 if max_abs > 0 else 10
            ax_comp.set_xlim(-limit, limit)
            
            # Injeta os textos de forma espelhada (Doença nas pontas extremas, números na ponta das barras)
            for i, (masc, fem, label) in enumerate(zip(masc_vals, fem_vals, labels_cat)):
                ax_comp.text(-limit * 0.95, i, label, ha='left', va='center', fontsize=13, color='#555555')
                ax_comp.text(limit * 0.95, i, label, ha='right', va='center', fontsize=13, color='#555555')
                
                if fem > 0:
                    ax_comp.text(-fem - (limit * 0.015), i, f"{int(fem):,}".replace(",", "."), va='center', ha='right', fontsize=12, color='#333333', fontweight='bold')
                if masc > 0:
                    ax_comp.text(masc + (limit * 0.015), i, f"{int(masc):,}".replace(",", "."), va='center', ha='left', fontsize=12, color='#333333', fontweight='bold')
                    
            # Títulos flutuantes no topo das colunas
            ax_comp.text(-limit * 0.45, len(labels_cat) - 0.2, "Feminino", ha='center', va='bottom', fontsize=16, fontweight='bold', color=CORES_SEXO['FEMININO'])
            ax_comp.text(limit * 0.45, len(labels_cat) - 0.2, "Masculino", ha='center', va='bottom', fontsize=16, fontweight='bold', color=CORES_SEXO['MASCULINO'])
            
            st.pyplot(fig_comp)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1: download_plot(fig_comp, "Top10_Comparativo_Sexo.png")
            
            # Matemática Perfeita da Tabela
            total_fem_geral = len(df_perfil[df_perfil['Sexo'] == 'FEMININO'])
            total_masc_geral = len(df_perfil[df_perfil['Sexo'] == 'MASCULINO'])
            
            fem_top10 = tabela_sexo['FEMININO'].sum()
            masc_top10 = tabela_sexo['MASCULINO'].sum()
            
            outros_fem = total_fem_geral - fem_top10
            outros_masc = total_masc_geral - masc_top10
            outros_total = outros_fem + outros_masc
            
            dados_tabela_comp = []
            tabela_sexo_sorted = tabela_sexo.sort_values(by='Total', ascending=False)
            
            for grupo, row in tabela_sexo_sorted.iterrows():
                fem_val = int(row['FEMININO'])
                masc_val = int(row['MASCULINO'])
                tot_val = int(row['Total'])
                
                pct_fem = f"{(fem_val/total_casos_perfil)*100:.1f}%".replace(".", ",") if total_casos_perfil > 0 else "0,0%"
                pct_masc = f"{(masc_val/total_casos_perfil)*100:.1f}%".replace(".", ",") if total_casos_perfil > 0 else "0,0%"
                pct_total = f"{(tot_val/total_casos_perfil)*100:.1f}%".replace(".", ",") if total_casos_perfil > 0 else "0,0%"
                
                dados_tabela_comp.append({
                    "Grupo de câncer": grupo,
                    "Feminino": f"{fem_val:,}".replace(",", "."),
                    "% Feminino": pct_fem,
                    "Masculino": f"{masc_val:,}".replace(",", "."),
                    "% Masculino": pct_masc,
                    "Total": f"{tot_val:,}".replace(",", "."),
                    "% do total": pct_total
                })
                
            if outros_total > 0:
                pct_fem_outros = f"{(outros_fem/total_casos_perfil)*100:.1f}%".replace(".", ",") if total_casos_perfil > 0 else "0,0%"
                pct_masc_outros = f"{(outros_masc/total_casos_perfil)*100:.1f}%".replace(".", ",") if total_casos_perfil > 0 else "0,0%"
                pct_total_outros = f"{(outros_total/total_casos_perfil)*100:.1f}%".replace(".", ",") if total_casos_perfil > 0 else "0,0%"
                
                dados_tabela_comp.append({
                    "Grupo de câncer": "Outros (Demais Neoplasias)",
                    "Feminino": f"{outros_fem:,}".replace(",", "."),
                    "% Feminino": pct_fem_outros,
                    "Masculino": f"{outros_masc:,}".replace(",", "."),
                    "% Masculino": pct_masc_outros,
                    "Total": f"{outros_total:,}".replace(",", "."),
                    "% do total": pct_total_outros
                })
                
            pct_fem_geral = f"{(total_fem_geral/total_casos_perfil)*100:.1f}%".replace(".", ",") if total_casos_perfil > 0 else "0,0%"
            pct_masc_geral = f"{(total_masc_geral/total_casos_perfil)*100:.1f}%".replace(".", ",") if total_casos_perfil > 0 else "0,0%"
            
            dados_tabela_comp.append({
                "Grupo de câncer": "TOTAL (100% da Base)",
                "Feminino": f"{total_fem_geral:,}".replace(",", "."),
                "% Feminino": pct_fem_geral,
                "Masculino": f"{total_masc_geral:,}".replace(",", "."),
                "% Masculino": pct_masc_geral,
                "Total": f"{total_casos_perfil:,}".replace(",", "."),
                "% do total": "100,0%"
            })
            
            df_tab_sex_final = pd.DataFrame(dados_tabela_comp)
            
            with st.expander("📂 Inspecionar Dataframe Bruto"):
                st.dataframe(df_tab_sex_final, use_container_width=True, hide_index=True)

        elif visao_freq == "Todas as Neoplasias (Grupos Anatômicos)":
            st.markdown("## Frequência de TODAS as Neoplasias (Agrupamento CID-O3)")
            
            titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"Ranking Completo de Grupos de Câncer — RHC, {texto_ano_titulo}", key="title_todas_neo")
            
            top_data = df_perfil['Macro_Topografia_Completa'].value_counts()
            dados_tabela_top = []
            for grupo, count in top_data.items():
                dados_tabela_top.append({"Grupo Anatômico (CID-O3)": grupo, "Nº de casos": f"{count:,}".replace(",", "."), "N_raw": count, "% do total": f"{(count / total_casos_perfil) * 100:.1f}%".replace(".", ",")})
            df_top_final = pd.DataFrame(dados_tabela_top)
            
            fig_top, ax_top = plt.subplots(figsize=(12, max(8, len(top_data) * 0.35)))
            y_coords = np.arange(len(top_data))[::-1]
            bars = ax_top.barh(y_coords, df_top_final['N_raw'], color=[COR_AZUL_ESCURO] * len(top_data), edgecolor='white')
            ax_top.set_yticks(y_coords, labels=df_top_final["Grupo Anatômico (CID-O3)"], fontsize=11, color='#555555')
            ax_top.spines['top'].set_visible(False)
            ax_top.spines['right'].set_visible(False)
            ax_top.spines['left'].set_color('#cccccc')
            ax_top.spines['bottom'].set_color('#cccccc')
            ax_top.tick_params(axis='x', labelsize=12, colors='#555555')
            ax_top.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=15, fontsize=18)
            for i, bar in enumerate(bars):
                ax_top.text(df_top_final.iloc[i]['N_raw'] + (total_casos_perfil * 0.005), bar.get_y() + bar.get_height()/2, f"{df_top_final.iloc[i]['N_raw']:,}".replace(",", "."), va='center', ha='left', fontsize=11, color='#333333')
            st.pyplot(fig_top)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1: download_plot(fig_top, "Ranking_Completo_CID.png")
            st.dataframe(df_top_final.drop(columns=['N_raw']), use_container_width=True, hide_index=True)

        elif visao_freq == "Categoria de Atendimento (Pizza)":
            st.markdown("## Distribuição por Categoria de Atendimento (Admissão)")
            
            titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"Categoria de Atendimento — RHC, {texto_ano_titulo}", key="title_cat_atend")
            
            cat_data = df_perfil['Categoria_Atendimento'].value_counts()
            df_cat_final = pd.DataFrame([{"Categoria de Atendimento": cat, "Nº de casos": f"{count:,}".replace(",", "."), "% do total": f"{(count / total_casos_perfil) * 100:.1f}%".replace(".", ",")} for cat, count in cat_data.items()])
            
            mapa_cores_cat = {'SUS': COR_AZUL_ESCURO, 'Convênio (Saúde Suplementar)': COR_DOURADO, 'Particular': '#3a7a78', 'Não Informado': '#999999', 'Outros': '#4a4a4a'}
            fig_pizza, ax_pizza = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax_pizza.pie(cat_data.values, labels=cat_data.index, autopct='%1.1f%%', startangle=140, colors=[mapa_cores_cat.get(c, '#999999') for c in cat_data.index], explode=[0.03] * len(cat_data), textprops={'fontsize': 14, 'color': '#333333'})
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
            ax_pizza.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=20, fontsize=18)
            
            col_graf, col_tab = st.columns([1.2, 1])
            with col_graf:
                st.pyplot(fig_pizza)
                download_plot(fig_pizza, "Categoria_Atendimento.png")
            with col_tab:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.dataframe(df_cat_final, use_container_width=True, hide_index=True)

        elif visao_freq == "Base de Diagnóstico (Pizza)":
            st.markdown("## Base de Diagnóstico (Taxa de Confirmação Microscópica)")
            
            titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"Distribuição da Base de Diagnóstico — RHC, {texto_ano_titulo}", key="title_base_diag")
            
            base_data = df_perfil['Base_Diagnostico'].value_counts().sort_values(ascending=False)
            dados_tabela_base = []
            
            for cat, count in base_data.items():
                perc = (count / total_casos_perfil) * 100
                dados_tabela_base.append({"Base de Diagnóstico": cat, "Nº de casos": f"{count:,}".replace(",", "."), "N_raw": count, "% do total": f"{perc:.1f}%".replace(".", ",")})
            df_base_final = pd.DataFrame(dados_tabela_base)
            
            fig_base, ax_base = plt.subplots(figsize=(10, 7))
            cores_base = [COR_AZUL_ESCURO if cat == 'Confirmação Microscópica' else COR_DOURADO for cat in base_data.index]
            
            def autopct_format(pct, allvals):
                absolute = int(np.round(pct/100.*np.sum(allvals)))
                val_str = f"{absolute:,}".replace(',', '.')
                pct_str = f"{pct:.1f}%".replace('.', ',')
                return f"{pct_str}\n(n={val_str})"
            
            wedges, texts, autotexts = ax_base.pie(
                base_data.values,
                autopct=lambda pct: autopct_format(pct, base_data.values),
                startangle=140,
                colors=cores_base,
                explode=[0.05 if i > 0 else 0 for i in range(len(base_data))],
                textprops={'fontsize': 14, 'weight': 'bold'}
            )
            
            for i, autotext in enumerate(autotexts):
                perc = base_data.values[i] / total_casos_perfil
                if perc > 0.05:
                    autotext.set_color('white')
                else:
                    autotext.set_color('#333333')
                    x, y = autotext.get_position()
                    autotext.set_position((x*1.35, y*1.35))
            
            ax_base.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=20, fontsize=18)
            
            leg_labels = [f"{cat}\n{base_data[cat]/total_casos_perfil*100:.1f}% (n={base_data[cat]:,})".replace(',', 'X').replace('.', ',').replace('X', '.') for cat in base_data.index]
            ax_base.legend(wedges, leg_labels, title="", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), frameon=False, fontsize=13, labelcolor='#333333')
                
            col_graf, col_tab = st.columns([1.2, 1])
            with col_graf:
                st.pyplot(fig_base)
                download_plot(fig_base, "Base_Diagnostico.png")
            with col_tab:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.dataframe(df_base_final.drop(columns=['N_raw']), use_container_width=True, hide_index=True)
                
        elif visao_freq == "Perfil de Tratamento (Barras)":
            st.markdown("## Perfil de Tratamento Oncológico")
            
            titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"Combinações de tratamento mais frequentes — RHC, {texto_ano_titulo}", key="title_tratamento")
            
            trat_data = df_perfil['Tratamento_Consolidado'].value_counts()
            
            df_grafico = pd.DataFrame({
                "Tratamento": trat_data.index,
                "N_raw": trat_data.values
            }).sort_values(by="N_raw", ascending=True) 
            
            fig_trat, ax_trat = plt.subplots(figsize=(12, max(6, len(df_grafico) * 0.6)))
            y_coords = np.arange(len(df_grafico))
            
            bars = ax_trat.barh(y_coords, df_grafico['N_raw'], color=cor_primaria, edgecolor='white')
            ax_trat.set_yticks(y_coords)
            ax_trat.set_yticklabels(df_grafico["Tratamento"], fontsize=12, color='#555555')
            
            ax_trat.spines['top'].set_visible(False)
            ax_trat.spines['right'].set_visible(False)
            ax_trat.spines['left'].set_color('#cccccc')
            ax_trat.spines['bottom'].set_color('#cccccc')
            ax_trat.set_xlabel('Número de casos', fontsize=14, color='#333333')
            ax_trat.tick_params(axis='x', labelsize=12, colors='#555555')
            ax_trat.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=15, fontsize=18)
            
            for i, bar in enumerate(bars):
                val = df_grafico.iloc[i]['N_raw']
                ax_trat.text(val + (total_casos_perfil * 0.005), bar.get_y() + bar.get_height()/2, f"{val:,}".replace(",", "."), va='center', ha='left', color='#333333', fontsize=12)
                
            st.pyplot(fig_trat)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1: download_plot(fig_trat, "Perfil_Tratamento.png")
            
            # Tabela Matemática
            dados_tabela_trat = []
            trat_data_desc = trat_data.sort_values(ascending=False)
            
            for trat, count in trat_data_desc.items():
                pct = (count / total_casos_perfil) * 100
                dados_tabela_trat.append({
                    "Combinação de tratamento": trat,
                    "Nº de casos": f"{count:,}".replace(",", "."),
                    "N_raw": count,
                    "% do total": f"{pct:.1f}%".replace(".", ",")
                })
                
            dados_tabela_trat.append({
                "Combinação de tratamento": "TOTAL (100% da Base)",
                "Nº de casos": f"{total_casos_perfil:,}".replace(",", "."),
                "N_raw": total_casos_perfil,
                "% do total": "100,0%"
            })
            
            df_tab_trat = pd.DataFrame(dados_tabela_trat)
            st.dataframe(df_tab_trat.drop(columns=['N_raw']), use_container_width=True, hide_index=True)
            
            if len(trat_data_desc) > 0:
                top_trat = trat_data_desc.index[0]
                top_trat_val = trat_data_desc.iloc[0]
                top_trat_pct = (top_trat_val / total_casos_perfil) * 100
                st.markdown(f"""
                > 📝 **Draft de Documentação:** A análise do perfil terapêutico demonstra que a modalidade **{top_trat.title()}** foi a mais registrada no período, compreendendo **{top_trat_val:,} casos ({top_trat_pct:.1f}%)**. As combinações de tratamento refletem a complexidade e a abordagem multidisciplinar exigida pelos diferentes estadiamentos clínicos atendidos na instituição.
                """.replace(',', 'X').replace('.', ',').replace('X', '.'))

        elif visao_freq == "Naturalidade (Brasil vs Estrangeiros) (Pizza)":
            st.markdown("## Origem de Nascimento (Brasileiros vs Estrangeiros)")
            
            titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"Distribuição por Naturalidade — RHC, {texto_ano_titulo}", key="title_nat_pizza")
            
            nat_data = df_perfil['Naturalidade'].value_counts()
            
            dados_tabela_nat = []
            for nat, count in nat_data.items():
                perc = (count / total_casos_perfil) * 100
                dados_tabela_nat.append({
                    "Naturalidade": nat, 
                    "Nº de casos": f"{count:,}".replace(",", "."), 
                    "% do total": f"{perc:.1f}%".replace(".", ",")
                })
                
            dados_tabela_nat.append({
                "Naturalidade": "TOTAL (100% da Base)",
                "Nº de casos": f"{total_casos_perfil:,}".replace(",", "."),
                "% do total": "100,0%"
            })
            df_nat_final = pd.DataFrame(dados_tabela_nat)
            
            mapa_cores_nat = {'Brasileiros': COR_AZUL_ESCURO, 'Estrangeiros': COR_DOURADO, 'Não Informado': '#999999'}
            fig_pizza, ax_pizza = plt.subplots(figsize=(10, 7))
            
            def autopct_format(pct, allvals):
                absolute = int(np.round(pct/100.*np.sum(allvals)))
                val_str = f"{absolute:,}".replace(',', '.')
                pct_str = f"{pct:.1f}%".replace('.', ',')
                return f"{pct_str}\n(n={val_str})"
            
            wedges, texts, autotexts = ax_pizza.pie(
                nat_data.values,
                autopct=lambda pct: autopct_format(pct, nat_data.values),
                startangle=140,
                colors=[mapa_cores_nat.get(c, '#4a4a4a') for c in nat_data.index],
                explode=[0.05 if i > 0 else 0 for i in range(len(nat_data))],
                textprops={'fontsize': 14, 'weight': 'bold'}
            )
            
            for i, autotext in enumerate(autotexts):
                perc = nat_data.values[i] / total_casos_perfil
                if perc > 0.05:
                    autotext.set_color('white')
                else:
                    autotext.set_color('#333333')
                    x, y = autotext.get_position()
                    autotext.set_position((x*1.35, y*1.35))
            
            ax_pizza.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=20, fontsize=18)
            
            leg_labels = [f"{nat}\n{nat_data[nat]/total_casos_perfil*100:.1f}% (n={nat_data[nat]:,})".replace(',', 'X').replace('.', ',').replace('X', '.') for nat in nat_data.index]
            ax_pizza.legend(wedges, leg_labels, title="", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), frameon=False, fontsize=13, labelcolor='#333333')
            
            col_graf, col_tab = st.columns([1.2, 1])
            with col_graf:
                st.pyplot(fig_pizza)
                download_plot(fig_pizza, "Naturalidade.png")
            with col_tab:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.dataframe(df_nat_final, use_container_width=True, hide_index=True)
                
            st.markdown(f"""
            > 📝 **Draft de Documentação:** A imensa maioria dos pacientes atendidos na instituição no período de {anos_perfil[0]} a {anos_perfil[1]} é composta por nascidos no Brasil. Esta métrica reforça o perfil de atendimento focado na população nacional, com uma presença minoritária de estrangeiros ou pacientes sem informação de naturalidade registrada no sistema.
            """)

        elif visao_freq == "Naturalidade (Por Estado) (Barras)":
            st.markdown("## Origem de Nascimento Detalhada (Por Estado)")
            
            titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"Distribuição da naturalidade por estado — RHC, {texto_ano_titulo}", key="title_nat_barras")
            
            nat_uf_data = df_perfil['Naturalidade_UF'].value_counts()
            
            df_grafico = pd.DataFrame({
                "Origem": nat_uf_data.index,
                "N_raw": nat_uf_data.values
            }).sort_values(by="N_raw", ascending=True) 
            
            fig_nat_uf, ax_nat_uf = plt.subplots(figsize=(12, max(6, len(df_grafico) * 0.4)))
            y_coords = np.arange(len(df_grafico))
            
            def get_color(origem):
                if origem == 'Estrangeiros': return COR_DOURADO
                if origem == 'Não Informado': return '#999999'
                return COR_AZUL_ESCURO
                
            cores_barras = [get_color(x) for x in df_grafico["Origem"]]
            
            bars = ax_nat_uf.barh(y_coords, df_grafico['N_raw'], color=cores_barras, edgecolor='white')
            ax_nat_uf.set_yticks(y_coords)
            ax_nat_uf.set_yticklabels(df_grafico["Origem"], fontsize=12, color='#555555')
            
            ax_nat_uf.spines['top'].set_visible(False)
            ax_nat_uf.spines['right'].set_visible(False)
            ax_nat_uf.spines['left'].set_color('#cccccc')
            ax_nat_uf.spines['bottom'].set_color('#cccccc')
            ax_nat_uf.set_xlabel('Número de casos', fontsize=14, color='#333333')
            ax_nat_uf.tick_params(axis='x', labelsize=12, colors='#555555')
            ax_nat_uf.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=15, fontsize=18)
            
            for i, bar in enumerate(bars):
                val = df_grafico.iloc[i]['N_raw']
                ax_nat_uf.text(val + (total_casos_perfil * 0.005), bar.get_y() + bar.get_height()/2, f"{val:,}".replace(",", "."), va='center', ha='left', color='#333333', fontsize=12)
                
            st.pyplot(fig_nat_uf)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1: download_plot(fig_nat_uf, "Naturalidade_Estados.png")
            
            dados_tabela_nat_uf = []
            nat_uf_desc = nat_uf_data.sort_values(ascending=False)
            
            for origem, count in nat_uf_desc.items():
                pct = (count / total_casos_perfil) * 100
                dados_tabela_nat_uf.append({
                    "Origem (UF / Categoria)": origem,
                    "Nº de casos": f"{count:,}".replace(",", "."),
                    "N_raw": count,
                    "% do total": f"{pct:.1f}%".replace(".", ",")
                })
                
            dados_tabela_nat_uf.append({
                "Origem (UF / Categoria)": "TOTAL (100% da Base)",
                "Nº de casos": f"{total_casos_perfil:,}".replace(",", "."),
                "N_raw": total_casos_perfil,
                "% do total": "100,0%"
            })
            
            df_tab_nat_uf = pd.DataFrame(dados_tabela_nat_uf)
            st.dataframe(df_tab_nat_uf.drop(columns=['N_raw']), use_container_width=True, hide_index=True)
            
            st.markdown(f"""
            > 📝 **Draft de Documentação:** A estratificação por estado de naturalidade permite mapear o perfil migratório e demográfico da população atendida. É possível observar o volume de pacientes provenientes de outras unidades federativas (bem como estrangeiros) que buscam tratamento no estado de São Paulo, evidenciando o papel da instituição como polo de referência oncológica nacional.
            """)

        elif visao_freq == "Distribuição Geográfica (Rosca)":
            st.markdown("## Distribuição dos casos por região de residência")
            
            titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"Distribuição dos casos por região de residência — RHC, {texto_ano_titulo}", key="title_geo_rosca")
            
            df_geo = df_perfil.copy()
            df_geo['IBGE_7'] = df_geo['IBGE'].astype(str).str[:7]
            
            def categorizar_regiao(linha):
                uf = str(linha['UFRESID']).strip().upper()
                ibge = str(linha['IBGE_7'])
                if uf == 'SP':
                    if ibge == '3550308':
                        return 'São Paulo (capital)'
                    elif ibge in IBGE_RMSP:
                        return 'Região Metropolitana (exceto capital)'
                    else:
                        return 'Interior do Estado de SP'
                else:
                    return 'Outros estados'
                    
            df_geo['Regiao_Residencia'] = df_geo.apply(categorizar_regiao, axis=1)
            geo_counts = df_geo['Regiao_Residencia'].value_counts()
            
            ordem_regiao = ['São Paulo (capital)', 'Região Metropolitana (exceto capital)', 'Interior do Estado de SP', 'Outros estados']
            valores_regiao = [geo_counts.get(k, 0) for k in ordem_regiao]
            labels_filtrados = [k for k, v in zip(ordem_regiao, valores_regiao) if v > 0]
            valores_filtrados = [v for v in valores_regiao if v > 0]
            
            cores_regiao_dict = {
                'São Paulo (capital)': COR_AZUL_ESCURO,
                'Região Metropolitana (exceto capital)': '#3a7a78',
                'Interior do Estado de SP': COR_DOURADO,
                'Outros estados': '#999999'
            }
            cores_filtradas = [cores_regiao_dict[k] for k in labels_filtrados]
            
            st.markdown(f"""
            > 📝 **Draft de Documentação:** A quase totalidade dos pacientes atendidos na instituição reside no Estado de São Paulo, com concentração expressiva no município da capital paulista, refletindo a localização e a área de influência histórica do hospital.
            """)
            
            fig_rosca, ax_rosca = plt.subplots(figsize=(10, 6))
            
            wedges, texts, autotexts = ax_rosca.pie(
                valores_filtrados, 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=cores_filtradas,
                wedgeprops=dict(width=0.4, edgecolor='white'),
                textprops=dict(color="white", weight="bold", fontsize=12),
                pctdistance=0.80
            )
            
            ax_rosca.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=20, fontsize=18)
            ax_rosca.legend(wedges, labels_filtrados, title="Região", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), frameon=False, fontsize=12, labelcolor='#333333')
            
            st.pyplot(fig_rosca)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1: download_plot(fig_rosca, "Distribuicao_Geografica_Rosca.png")
            
            df_geo_tabela = pd.DataFrame({
                "Região de Residência": labels_filtrados,
                "Número de Casos": [f"{v:,}".replace(',', '.') for v in valores_filtrados],
                "% do total": [f"{(v/total_casos_perfil)*100:.1f}%".replace('.', ',') for v in valores_filtrados]
            })
            
            with st.expander("📂 Inspecionar Dataframe Bruto"):
                st.dataframe(df_geo_tabela, use_container_width=True, hide_index=True)

        elif visao_freq == "Distribuição Geográfica (Mapa)":
            st.markdown("## Origem dos Pacientes (Estado de São Paulo)")
            
            titulo_grafico = st.text_input("✏️ Customizar título do mapa:", value=f"Distribuição Geográfica — RHC, {texto_ano_titulo}", key="title_geo_mapa")
            
            df_sp = df_perfil[df_perfil['UFRESID'] == 'SP'].copy()
            total_geral = len(df_perfil)
            total_sp = len(df_sp)
            
            perc_sp = 0
            total_cap = 0
            perc_cap = 0
            total_rmsp = 0
            perc_rmsp = 0
            total_int = 0
            perc_int = 0
            
            if total_sp > 0:
                df_sp['IBGE_7'] = df_sp['IBGE'].str[:7]
                
                perc_sp = (total_sp / total_geral) * 100
                df_capital = df_sp[df_sp['IBGE_7'] == '3550308']
                total_cap = len(df_capital)
                perc_cap = (total_cap / total_sp) * 100
                
                df_rmsp = df_sp[df_sp['IBGE_7'].isin(IBGE_RMSP)]
                total_rmsp = len(df_rmsp)
                perc_rmsp = (total_rmsp / total_sp) * 100
                
                total_int = total_sp - total_cap - total_rmsp
                perc_int = (total_int / total_sp) * 100
                
                with st.spinner("Compilando dados na malha cartográfica..."):
                    geojson_sp = carregar_malha_sp()
                    
                    if geojson_sp:
                        todos_municipios = []
                        for feature in geojson_sp['features']:
                            todos_municipios.append({
                                'IBGE_7': feature['properties']['id'],
                                'CIDADE_GEO': feature['properties']['name']
                            })
                        df_todos_ibge = pd.DataFrame(todos_municipios)
                        
                        casos_por_cidade = df_sp.groupby('IBGE_7').size().reset_index(name='Casos')
                        
                        mapa_data = pd.merge(df_todos_ibge, casos_por_cidade, on='IBGE_7', how='left')
                        mapa_data['Casos'] = mapa_data['Casos'].fillna(0).astype(int)
                        
                        def categorizar_casos(n):
                            if n == 0: return "Nenhum caso"
                            elif n <= 10: return "Até 10"
                            elif n <= 100: return "11 - 100"
                            elif n <= 500: return "101 - 500"
                            elif n <= 5000: return "501 - 5.000"
                            else: return "> 5.000"
                            
                        mapa_data['Faixa de Casos'] = mapa_data['Casos'].apply(categorizar_casos)
                        
                        ordem_faixas = ["Nenhum caso", "Até 10", "11 - 100", "101 - 500", "501 - 5.000", "> 5.000"]
                        
                        cores_faixas_ambos = {
                            "Nenhum caso": "#FFFFFF",
                            "Até 10": "#C7E9C0",
                            "11 - 100": "#74C476",
                            "101 - 500": "#41AB5D",
                            "501 - 5.000": "#238B45",
                            "> 5.000": "#005A32"
                        }

                        cores_faixas_fem = {
                            "Nenhum caso": "#FFFFFF",
                            "Até 10": "#E2D4E8",
                            "11 - 100": "#C5A8D1",
                            "101 - 500": "#A87CBA",
                            "501 - 5.000": "#85299D", 
                            "> 5.000": "#4F185E"
                        }

                        cores_faixas_masc = {
                            "Nenhum caso": "#FFFFFF",
                            "Até 10": "#D6E1F0",
                            "11 - 100": "#ADC4E1",
                            "101 - 500": "#84A6D2",
                            "501 - 5.000": "#517CBE", 
                            "> 5.000": "#304A72"
                        }
                        
                        if filtro_sexo == 'FEMININO':
                            mapa_cores_ativa = cores_faixas_fem
                        elif filtro_sexo == 'MASCULINO':
                            mapa_cores_ativa = cores_faixas_masc
                        else:
                            mapa_cores_ativa = cores_faixas_ambos
                        
                        fig_mapa = px.choropleth(
                            mapa_data,
                            geojson=geojson_sp,
                            locations='IBGE_7',
                            featureidkey='properties.id',
                            color='Faixa de Casos',
                            hover_name='CIDADE_GEO',
                            hover_data={'Casos': True, 'Faixa de Casos': False, 'IBGE_7': False},
                            color_discrete_map=mapa_cores_ativa,
                            category_orders={"Faixa de Casos": ordem_faixas},
                            scope="south america"
                        )
                        
                        # Expandir a visualização e desenhar rótulos dinamicamente
                        fig_mapa.add_trace(go.Scattergeo(
                            lon=[-45.5, -43.5, -51.5, -53.5, -50.0],
                            lat=[-20.0, -22.5, -24.5, -21.0, -19.0],
                            text=["<b>MG</b>", "<b>RJ</b>", "<b>PR</b>", "<b>MS</b>", "<b>GO</b>"],
                            mode="text",
                            textfont=dict(size=14, color="#666666", family="Inter"),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

                        fig_mapa.add_trace(go.Scattergeo(
                            lon=[-46.6333],
                            lat=[-23.5505],
                            text=["São Paulo"],
                            mode="markers+text",
                            textposition="middle right",
                            marker=dict(size=8, color="#ffb3b3", line=dict(width=1, color="black")),
                            textfont=dict(size=12, color="black", family="Inter"),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        fig_mapa.update_traces(marker_line_width=0.5, marker_line_color='#666666', selector=dict(type='choropleth'))
                        
                        fig_mapa.update_geos(
                            visible=False,
                            fitbounds="locations"
                        )
                        
                        fig_mapa.update_layout(
                            height=800,
                            autosize=True,
                            paper_bgcolor="white",
                            plot_bgcolor="white",
                            margin={"r":20,"t":90,"l":20,"b":20},
                            title=dict(
                                text=titulo_grafico,
                                x=0.5,
                                y=0.95,
                                xanchor='center',
                                font=dict(family="Inter", size=28, color=COR_AZUL_ESCURO)
                            ),
                            legend=dict(
                                title=dict(text="<b>Número de casos</b>", font=dict(family="Inter", size=22, color="#333333")),
                                yanchor="bottom",
                                y=0.02,
                                xanchor="left",
                                x=0.02,
                                bgcolor="white",
                                bordercolor="black",
                                borderwidth=1,
                                font=dict(family="Inter", size=18, color="#333333")
                            )
                        )
                        
                        st.plotly_chart(fig_mapa, use_container_width=True)
                        
                        col_d1, col_d2 = st.columns([1, 3])
                        with col_d1: download_plot(fig_mapa, "Mapa_SP_Origem.png")
                        
                        st.markdown(f"""
                        > 📝 **Draft de Documentação:** Dos pacientes tratados na instituição no período de {anos_perfil[0]} a {anos_perfil[1]}, observou-se que a imensa maioria, **{perc_sp:.1f}% ({total_sp:,})**, era residente no Estado de São Paulo. Analisando esta coorte estadual, nota-se uma forte centralização da demanda assistencial: **{perc_cap:.1f}% ({total_cap:,})** residem no próprio município de São Paulo (Capital) e **{perc_rmsp:.1f}% ({total_rmsp:,})** nos municípios que compõem a Região Metropolitana. O fluxo de pacientes residentes no interior e litoral do Estado representa uma fatia menor, correspondendo a **{perc_int:.1f}% ({total_int:,})** dos atendimentos. Essa distribuição geográfica evidencia a consolidação da instituição como um polo de referência oncológica primariamente metropolitano."
                        """.replace(',', 'X').replace('.', ',').replace('X', '.'))
                        
                        # Tabela Geográfica Detalhada
                        df_geo_tabela = df_perfil.copy()
                        df_geo_tabela['IBGE_7'] = df_geo_tabela['IBGE'].astype(str).str[:7]
                        
                        def categorizar_regiao_tab(linha):
                            uf = str(linha['UFRESID']).strip().upper()
                            ibge = str(linha['IBGE_7'])
                            if uf == 'SP':
                                if ibge == '3550308': return 'São Paulo (capital)'
                                elif ibge in IBGE_RMSP: return 'Região Metropolitana (exceto capital)'
                                else: return 'Interior do Estado de SP'
                            else: return 'Outros estados'
                                
                        df_geo_tabela['Regiao_Residencia'] = df_geo_tabela.apply(categorizar_regiao_tab, axis=1)
                        
                        tabela_geo_sexo = pd.crosstab(df_geo_tabela['Regiao_Residencia'], df_geo_tabela['Sexo'])
                        if 'MASCULINO' not in tabela_geo_sexo: tabela_geo_sexo['MASCULINO'] = 0
                        if 'FEMININO' not in tabela_geo_sexo: tabela_geo_sexo['FEMININO'] = 0
                        tabela_geo_sexo['Total'] = tabela_geo_sexo['MASCULINO'] + tabela_geo_sexo['FEMININO']
                        
                        ordem_regiao = ['São Paulo (capital)', 'Região Metropolitana (exceto capital)', 'Interior do Estado de SP', 'Outros estados']
                        tabela_geo_sexo = tabela_geo_sexo.reindex(ordem_regiao).fillna(0)
                        
                        total_fem_geral = len(df_perfil[df_perfil['Sexo'] == 'FEMININO'])
                        total_masc_geral = len(df_perfil[df_perfil['Sexo'] == 'MASCULINO'])
                        
                        dados_tabela_geo = []
                        for regiao in ordem_regiao:
                            row = tabela_geo_sexo.loc[regiao]
                            fem_val = int(row['FEMININO'])
                            masc_val = int(row['MASCULINO'])
                            tot_val = int(row['Total'])
                            
                            pct_fem = f"{(fem_val/total_casos_perfil)*100:.1f}%".replace(".", ",") if total_casos_perfil > 0 else "0,0%"
                            pct_masc = f"{(masc_val/total_casos_perfil)*100:.1f}%".replace(".", ",") if total_casos_perfil > 0 else "0,0%"
                            pct_total = f"{(tot_val/total_casos_perfil)*100:.1f}%".replace(".", ",") if total_casos_perfil > 0 else "0,0%"
                            
                            dados_tabela_geo.append({
                                "Região de Residência": regiao,
                                "Feminino": f"{fem_val:,}".replace(",", "."),
                                "% Feminino": pct_fem,
                                "Masculino": f"{masc_val:,}".replace(",", "."),
                                "% Masculino": pct_masc,
                                "Total": f"{tot_val:,}".replace(",", "."),
                                "% do total": pct_total
                            })
                            
                        pct_fem_geral = f"{(total_fem_geral/total_casos_perfil)*100:.1f}%".replace(".", ",") if total_casos_perfil > 0 else "0,0%"
                        pct_masc_geral = f"{(total_masc_geral/total_casos_perfil)*100:.1f}%".replace(".", ",") if total_casos_perfil > 0 else "0,0%"
                        
                        dados_tabela_geo.append({
                            "Região de Residência": "TOTAL (100% da Base)",
                            "Feminino": f"{total_fem_geral:,}".replace(",", "."),
                            "% Feminino": pct_fem_geral,
                            "Masculino": f"{total_masc_geral:,}".replace(",", "."),
                            "% Masculino": pct_masc_geral,
                            "Total": f"{total_casos_perfil:,}".replace(",", "."),
                            "% do total": "100,0%"
                        })
                        
                        df_tab_geo_final = pd.DataFrame(dados_tabela_geo)
                        
                        with st.expander("📂 Inspecionar Dataframe Bruto"):
                            st.dataframe(df_tab_geo_final, use_container_width=True, hide_index=True)
                        
                    else:
                        st.warning("⚠️ Não foi possível baixar a malha geográfica. Verifique a conexão com a internet ou firewall.")
            else:
                st.warning("Não há pacientes residentes em São Paulo nos filtros selecionados.")

# ==========================================
# VISÃO: SOBREVIDA 
# ==========================================
with aba_sobrevida:
    st.markdown("<br>", unsafe_allow_html=True)
    sugestao_max_sobrevida = max(ano_min_df, ano_max_df - 5)
    col_s1, col_s2 = st.columns([1, 2.5])
    with col_s1:
        anos_sobrevida = st.slider("Coorte de Sobrevida (Diagnóstico):", min_value=ano_min_df, max_value=ano_max_df, value=(ano_min_df, sugestao_max_sobrevida))
    with col_s2:
        st.markdown("<br>", unsafe_allow_html=True)
        tipo_grafico = st.selectbox("Eixo de Análise Visual:", ["Sobrevida Global (Todas as Neoplasias)", "Sobrevida por Doença Específica", "Ranking de Sobrevida (5 Anos)", "Quadro Metodológico (CIDs e Morfologias)"], label_visibility="collapsed")
    
    st.info(f"💡 **Recomendação Metodológica:** Para avaliar sobrevivência em 5 anos com confiabilidade, o limite aconselhável de acompanhamento é **{sugestao_max_sobrevida}**. Dica: O motor respeita seus filtros globais de Estadio. Exclua ou inclua casos *in situ* na barra lateral conforme o desenho do seu estudo.")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    df_sobrevida_base = df_filtrado[(df_filtrado['Ano_Diag'] >= anos_sobrevida[0]) & (df_filtrado['Ano_Diag'] <= anos_sobrevida[1])].copy()
    
    # Aplicando a regra Ouro de Sobrevida: Excluir Pele Não-Melanoma (Estadio in situ é controlado pelo filtro lateral)
    df_global = df_sobrevida_base[df_sobrevida_base['Macro_Topografia'] != 'Pele - não-melanoma'].copy()
    
    titulo_coorte = f"coorte {anos_sobrevida[0]}–{anos_sobrevida[1]}"
    
    str_sexo_base = f", sexo {filtro_sexo.capitalize()}" if filtro_sexo != 'Ambos' else ", ambos os sexos"
    str_estadio_base = f", Estadio {filtro_estadio}" if filtro_estadio != 'Todos' else ""
    str_filtros = f"{str_sexo_base}{str_estadio_base}"
    
    if tipo_grafico == "Sobrevida Global (Todas as Neoplasias)":
        
        comparacao_global = st.selectbox("Comparar por:", ["Quinquênio de Diagnóstico", "Estadio Clínico", "Sexo", "Sem divisão (Curva Única)"])
        
        titulo_default = f"Sobrevida global"
        if comparacao_global != "Sem divisão (Curva Única)":
            titulo_default += f" por {comparacao_global.split(' ')[0].lower()}"
        titulo_default += f" — RHC, {titulo_coorte}"
        
        titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=titulo_default, key="title_surv_global")
        
        if len(df_global) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            kmf = KaplanMeierFitter()
            resultados_tabela = []
            
            if comparacao_global == "Sem divisão (Curva Única)":
                kmf.fit(durations=df_global['Tempo_Meses'], event_observed=df_global['Status_Evento'])
                cor_plot = CORES_SEXO.get(filtro_sexo, COR_AZUL_ESCURO)
                kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.15, color=cor_plot, linewidth=2.5, legend=False)
                surv, low, up = extrair_metrica_60_meses(kmf)
                resultados_tabela.append({
                    "Coorte": "Global", 
                    "N_raw": len(df_global), 
                    "Nº de casos": f"{len(df_global):,}".replace(",", "."), 
                    "Sobrevida em 5 anos (IC95%)": f"{surv:.1f}% (IC95% {low:.1f}%–{up:.1f}%)".replace(".", ","),
                    "% do total": "100,0%"
                })
            
            elif comparacao_global == "Quinquênio de Diagnóstico":
                grupos = sorted([q for q in df_global['Quinquenio'].unique() if q != 'Outros'])
                cores_dinamicas = [COR_AZUL_ESCURO, '#3a7a78', COR_DOURADO, '#7b2e3a', '#4a70a3', '#7590b1']
                for i, grp in enumerate(grupos):
                    df_g = df_global[df_global['Quinquenio'] == grp]
                    if len(df_g) > 0:
                        kmf.fit(durations=df_g['Tempo_Meses'], event_observed=df_g['Status_Evento'], label=grp)
                        kmf.plot_survival_function(ax=ax, ci_show=False, color=cores_dinamicas[i % len(cores_dinamicas)], linewidth=2)
                        surv, low, up = extrair_metrica_60_meses(kmf)
                        pct = (len(df_g) / len(df_global)) * 100
                        resultados_tabela.append({
                            "Quinquênio": grp, 
                            "N_raw": len(df_g), 
                            "Nº de casos": f"{len(df_g):,}".replace(",", "."), 
                            "Sobrevida em 5 anos (IC95%)": f"{surv:.1f}% (IC95% {low:.1f}%–{up:.1f}%)".replace(".", ","),
                            "% do total": f"{pct:.1f}%".replace(".", ",")
                        })
                texto_p = extrair_pvalue_logrank(df_global[df_global['Quinquenio'].isin(grupos)], 'Quinquenio')
                if texto_p: ax.text(0.02, 0.05, texto_p, transform=ax.transAxes, fontsize=12, color='#333333', bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9))
                ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)
            
            elif comparacao_global == "Estadio Clínico":
                grupos_possiveis = ['0 (in situ)', 'I', 'II', 'III', 'IV']
                grupos = [g for g in grupos_possiveis if g in df_global['Estadio_Clinico'].unique()]
                cores_est = {'0 (in situ)': '#3a7a78', 'I': COR_AZUL_ESCURO, 'II': COR_DOURADO, 'III': '#7b2e3a', 'IV': '#4a4a4a'}
                for grp in grupos:
                    df_g = df_global[df_global['Estadio_Clinico'] == grp]
                    if len(df_g) > 5:
                        kmf.fit(durations=df_g['Tempo_Meses'], event_observed=df_g['Status_Evento'], label=f"Estadio {grp}")
                        kmf.plot_survival_function(ax=ax, ci_show=False, color=cores_est.get(grp, COR_AZUL_ESCURO), linewidth=2)
                        surv, low, up = extrair_metrica_60_meses(kmf)
                        pct = (len(df_g) / len(df_global)) * 100
                        resultados_tabela.append({
                            "Estadio Clínico": grp, 
                            "N_raw": len(df_g), 
                            "Nº de casos": f"{len(df_g):,}".replace(",", "."), 
                            "Sobrevida em 5 anos (IC95%)": f"{surv:.1f}% (IC95% {low:.1f}%–{up:.1f}%)".replace(".", ","),
                            "% do total": f"{pct:.1f}%".replace(".", ",")
                        })
                texto_p = extrair_pvalue_logrank(df_global[df_global['Estadio_Clinico'].isin(grupos)], 'Estadio_Clinico')
                if texto_p: ax.text(0.02, 0.05, texto_p, transform=ax.transAxes, fontsize=12, color='#333333', bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9))
                ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)
            
            elif comparacao_global == "Sexo":
                grupos = ['Feminino', 'Masculino']
                for grp in grupos:
                    df_g = df_global[df_global['Sexo'] == grp.upper()]
                    if len(df_g) > 0:
                        kmf.fit(durations=df_g['Tempo_Meses'], event_observed=df_g['Status_Evento'], label=grp)
                        kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.15, color=CORES_SEXO.get(grp.upper(), COR_AZUL_ESCURO), linewidth=2)
                        surv, low, up = extrair_metrica_60_meses(kmf)
                        pct = (len(df_g) / len(df_global)) * 100
                        resultados_tabela.append({
                            "Sexo": grp, 
                            "N_raw": len(df_g), 
                            "Nº de casos": f"{len(df_g):,}".replace(",", "."), 
                            "Sobrevida em 5 anos (IC95%)": f"{surv:.1f}% (IC95% {low:.1f}%–{up:.1f}%)".replace(".", ","),
                            "% do total": f"{pct:.1f}%".replace(".", ",")
                        })
                texto_p = extrair_pvalue_logrank(df_global[df_global['Sexo'].isin(['FEMININO', 'MASCULINO'])], 'Sexo')
                if texto_p: ax.text(0.02, 0.05, texto_p, transform=ax.transAxes, fontsize=12, color='#333333', bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9))
                ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)

            configurar_eixos_grafico(ax, titulo_grafico)
            if comparacao_global != "Sem divisão (Curva Única)":
                fig.subplots_adjust(right=0.75)
            else:
                fig.tight_layout()
            st.pyplot(fig)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1: download_plot(fig, "Sobrevida_Global.png")
            
            desc_comp = ""
            if comparacao_global != "Sem divisão (Curva Única)":
                desc_comp = f", comparada por **{comparacao_global}**"
                
            st.markdown(f"> 📝 **Descrição da Curva:** Probabilidade de sobrevida global estimada em 5 anos para **todas as topografias** (C00-C80, exceto pele não-melanoma){str_filtros}{desc_comp}, pacientes analíticos, no período de {anos_sobrevida[0]}–{anos_sobrevida[1]}.")
            
            if resultados_tabela:
                df_res = pd.DataFrame(resultados_tabela)
                if comparacao_global in ["Quinquênio de Diagnóstico", "Estadio Clínico"]:
                    df_res = df_res.drop(columns=["N_raw"])
                else:
                    df_res = df_res.sort_values(by="N_raw", ascending=False).drop(columns=["N_raw"])
                
                if comparacao_global != "Sem divisão (Curva Única)":
                    s_tot, l_tot, u_tot = extrair_metrica_60_meses(KaplanMeierFitter().fit(df_global['Tempo_Meses'], df_global['Status_Evento']))
                    col_chave = df_res.columns[0]
                    linha_total = pd.DataFrame([{
                        col_chave: "TOTAL (100% da Base)",
                        "Nº de casos": f"{len(df_global):,}".replace(",", "."),
                        "Sobrevida em 5 anos (IC95%)": f"{s_tot:.1f}% (IC95% {l_tot:.1f}%–{u_tot:.1f}%)".replace(".", ","),
                        "% do total": "100,0%"
                    }])
                    df_res = pd.concat([df_res, linha_total], ignore_index=True)
                
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                
                # --- TABELA ADICIONAL: PERFIL DE GÊNERO POR QUINQUÊNIO (GLOBAL) ---
                if filtro_sexo == 'Ambos' and comparacao_global == "Quinquênio de Diagnóstico":
                    st.markdown("#### Perfil de Gênero por Quinquênio")
                    df_quinq_sexo = df_global.copy()
                    tabela_q_s = pd.crosstab(df_quinq_sexo['Quinquenio'], df_quinq_sexo['Sexo'])
                    if 'MASCULINO' not in tabela_q_s: tabela_q_s['MASCULINO'] = 0
                    if 'FEMININO' not in tabela_q_s: tabela_q_s['FEMININO'] = 0
                    tabela_q_s['Total'] = tabela_q_s['MASCULINO'] + tabela_q_s['FEMININO']
                    
                    tabela_q_s = tabela_q_s.reindex(grupos).fillna(0)
                    
                    dados_tabela_q_s = []
                    total_casos_sobrevida = len(df_global)
                    
                    for q in grupos:
                        row = tabela_q_s.loc[q]
                        fem_val = int(row['FEMININO'])
                        masc_val = int(row['MASCULINO'])
                        tot_val = int(row['Total'])
                        
                        pct_fem = f"{(fem_val/total_casos_sobrevida)*100:.1f}%".replace(".", ",") if total_casos_sobrevida > 0 else "0,0%"
                        pct_masc = f"{(masc_val/total_casos_sobrevida)*100:.1f}%".replace(".", ",") if total_casos_sobrevida > 0 else "0,0%"
                        pct_total = f"{(tot_val/total_casos_sobrevida)*100:.1f}%".replace(".", ",") if total_casos_sobrevida > 0 else "0,0%"
                        
                        dados_tabela_q_s.append({
                            "Quinquênio": q,
                            "Nº de casos Masculino": f"{masc_val:,}".replace(",", "."),
                            "% de casos Masculinos": pct_masc,
                            "Nº de casos Feminino": f"{fem_val:,}".replace(",", "."),
                            "% de casos Feminino": pct_fem,
                            "Total": f"{tot_val:,}".replace(",", "."),
                            "% do total": pct_total
                        })
                    
                    tot_masc_g = tabela_q_s['MASCULINO'].sum()
                    tot_fem_g = tabela_q_s['FEMININO'].sum()
                    tot_g = tabela_q_s['Total'].sum()
                    
                    dados_tabela_q_s.append({
                        "Quinquênio": "TOTAL (100% da Base)",
                        "Nº de casos Masculino": f"{int(tot_masc_g):,}".replace(",", "."),
                        "% de casos Masculinos": f"{(tot_masc_g/total_casos_sobrevida)*100:.1f}%".replace(".", ",") if total_casos_sobrevida > 0 else "0,0%",
                        "Nº de casos Feminino": f"{int(tot_fem_g):,}".replace(",", "."),
                        "% de casos Feminino": f"{(tot_fem_g/total_casos_sobrevida)*100:.1f}%".replace(".", ",") if total_casos_sobrevida > 0 else "0,0%",
                        "Total": f"{int(tot_g):,}".replace(",", "."),
                        "% do total": "100,0%"
                    })
                    
                    st.dataframe(pd.DataFrame(dados_tabela_q_s), use_container_width=True, hide_index=True)

    elif tipo_grafico == "Ranking de Sobrevida (5 Anos)":
        titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=f"Sobrevida global em 5 anos — RHC, {titulo_coorte}", key="title_surv_ranking")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        kmf = KaplanMeierFitter()
        dados_barras, dados_tabela = [], []
        
        if len(df_global) > 0:
            kmf.fit(durations=df_global['Tempo_Meses'], event_observed=df_global['Status_Evento'])
            s_g, l_g, u_g = extrair_metrica_60_meses(kmf)
            dados_barras.append({'Grupo': 'Global (todos)', 'Surv': s_g, 'Err_L': s_g - l_g, 'Err_U': u_g - s_g})
            
        for grupo in ['Tireoide', 'Próstata', 'Mama', 'Colo do útero', 'Vulva', 'Corpo do útero', 'Pele - melanoma', 'Cólon e reto', 'Ovário', 'Cavidade oral e orofaringe']:
            df_g = df_sobrevida_base[df_sobrevida_base['Macro_Topografia'] == grupo]
            if len(df_g) > 10:
                kmf.fit(durations=df_g['Tempo_Meses'], event_observed=df_g['Status_Evento'])
                surv, low, up = extrair_metrica_60_meses(kmf)
                pct = (len(df_g) / len(df_global)) * 100
                dados_barras.append({'Grupo': grupo, 'Surv': surv, 'Err_L': surv - low, 'Err_U': up - surv})
                dados_tabela.append({
                    "Grupo de câncer": grupo, 
                    "N_raw": len(df_g), 
                    "Nº de casos": f"{len(df_g):,}".replace(",", "."), 
                    "Sobrevida em 5 anos (IC95%)": f"{surv:.1f}% (IC95% {low:.1f}%–{up:.1f}%)".replace(".", ","),
                    "% do total": f"{pct:.1f}%".replace(".", ",")
                })
        
        if dados_barras:
            df_barras = pd.DataFrame(dados_barras)
            
            df_global_bar = df_barras[df_barras['Grupo'] == 'Global (todos)']
            df_rest = df_barras[df_barras['Grupo'] != 'Global (todos)'].sort_values(by='Surv', ascending=True)
            df_barras = pd.concat([df_rest, df_global_bar]).reset_index(drop=True)
            
            y_pos = np.arange(len(df_barras))
            bars = ax.barh(y_pos, df_barras['Surv'], xerr=[df_barras['Err_L'], df_barras['Err_U']], color=[COR_DOURADO if g == 'Global (todos)' else COR_AZUL_ESCURO for g in df_barras['Grupo']], edgecolor='white', error_kw=dict(ecolor='gray', lw=1, capsize=3))
            ax.set_yticks(y_pos, labels=df_barras['Grupo'], fontsize=13, color='#555555')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cccccc')
            ax.spines['bottom'].set_color('#cccccc')
            ax.set_xlabel('Sobrevida global estimada em 5 anos (%)', fontsize=14, color='#333333')
            ax.tick_params(axis='x', labelsize=12, colors='#555555')
            ax.set_xlim(0, 105)
            ax.set_title(titulo_grafico, color=COR_AZUL_ESCURO, fontweight='bold', pad=15, fontsize=18)
            
            for bar, surv in zip(bars, df_barras['Surv']): ax.text(surv + 3, bar.get_y() + bar.get_height()/2, f'{surv:.1f}%'.replace('.', ','), va='center', fontsize=12, color='#333333')
            fig.tight_layout()
            st.pyplot(fig)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1: download_plot(fig, "Ranking_Sobrevida_5Anos.png")
            
            st.markdown(f"> 📝 **Descrição da Curva:** Probabilidade de sobrevida global estimada em 5 anos para as **10 Neoplasias Mais Frequentes** (casos incluídos conforme filtros){str_filtros}, pacientes analíticos, no período de {anos_sobrevida[0]}–{anos_sobrevida[1]}.")
            
            if dados_tabela:
                df_res = pd.DataFrame(dados_tabela).sort_values(by="N_raw", ascending=False).drop(columns=["N_raw"])
                linha_total = pd.DataFrame([{
                    "Grupo de câncer": "TOTAL (100% da Base)",
                    "Nº de casos": f"{len(df_global):,}".replace(",", "."),
                    "Sobrevida em 5 anos (IC95%)": f"{s_g:.1f}% (IC95% {l_g:.1f}%–{u_g:.1f}%)".replace(".", ","),
                    "% do total": "100,0%"
                }])
                df_res = pd.concat([df_res, linha_total], ignore_index=True)
                st.dataframe(df_res, use_container_width=True, hide_index=True)

    elif tipo_grafico == "Sobrevida por Doença Específica":
        doencas_disponiveis = [d for d in df_base['Macro_Topografia'].unique() if d not in ['Outros', 'Pele - não-melanoma']]
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            doenca_escolhida = st.selectbox("Selecione a doença:", sorted(doencas_disponiveis))
        with col_sel2:
            comparacao_doenca = st.selectbox("Comparar por:", ["Quinquênio de Diagnóstico", "Estadio Clínico", "Sexo", "Sem divisão (Curva Única)"])
            
        df_doenca = df_global[df_global['Macro_Topografia'] == doenca_escolhida]
        total_doenca = len(df_doenca)
        
        titulo_default = f"Sobrevida: {doenca_escolhida}"
        if comparacao_doenca != "Sem divisão (Curva Única)":
            titulo_default += f" por {comparacao_doenca.split(' ')[0].lower()}"
        titulo_default += f" — RHC, {titulo_coorte}"
            
        titulo_grafico = st.text_input("✏️ Customizar título do gráfico:", value=titulo_default, key="title_surv_doenca")
        
        if total_doenca > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            kmf = KaplanMeierFitter()
            resultados_tabela = []
            
            if comparacao_doenca == "Sem divisão (Curva Única)":
                kmf.fit(durations=df_doenca['Tempo_Meses'], event_observed=df_doenca['Status_Evento'])
                cor_plot = CORES_SEXO.get(filtro_sexo, COR_AZUL_ESCURO)
                kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.15, color=cor_plot, linewidth=2.5, legend=False)
                surv, low, up = extrair_metrica_60_meses(kmf)
                resultados_tabela.append({
                    "Coorte": "Geral", 
                    "N_raw": total_doenca, 
                    "Nº de casos": f"{total_doenca:,}".replace(",", "."), 
                    "Sobrevida em 5 anos (IC95%)": f"{surv:.1f}% (IC95% {low:.1f}%–{up:.1f}%)".replace(".", ","),
                    "% do total": "100,0%"
                })
            
            elif comparacao_doenca == "Quinquênio de Diagnóstico":
                grupos = sorted([q for q in df_doenca['Quinquenio'].unique() if q != 'Outros'])
                cores_dinamicas = [COR_AZUL_ESCURO, '#3a7a78', COR_DOURADO, '#7b2e3a', '#4a70a3', '#7590b1']
                for i, grp in enumerate(grupos):
                    df_g = df_doenca[df_doenca['Quinquenio'] == grp]
                    if len(df_g) > 0:
                        kmf.fit(durations=df_g['Tempo_Meses'], event_observed=df_g['Status_Evento'], label=grp)
                        kmf.plot_survival_function(ax=ax, ci_show=False, color=cores_dinamicas[i % len(cores_dinamicas)], linewidth=2)
                        surv, low, up = extrair_metrica_60_meses(kmf)
                        pct = (len(df_g) / total_doenca) * 100
                        resultados_tabela.append({
                            "Quinquênio": grp, 
                            "N_raw": len(df_g), 
                            "Nº de casos": f"{len(df_g):,}".replace(",", "."), 
                            "Sobrevida em 5 anos (IC95%)": f"{surv:.1f}% (IC95% {low:.1f}%–{up:.1f}%)".replace(".", ","),
                            "% do total": f"{pct:.1f}%".replace(".", ",")
                        })
                texto_p = extrair_pvalue_logrank(df_doenca[df_doenca['Quinquenio'].isin(grupos)], 'Quinquenio')
                if texto_p: ax.text(0.02, 0.05, texto_p, transform=ax.transAxes, fontsize=12, color='#333333', bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9))
                ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)
            
            elif comparacao_doenca == "Estadio Clínico":
                grupos_possiveis = ['0 (in situ)', 'I', 'II', 'III', 'IV']
                grupos = [g for g in grupos_possiveis if g in df_doenca['Estadio_Clinico'].unique()]
                cores_est = {'0 (in situ)': '#3a7a78', 'I': COR_AZUL_ESCURO, 'II': COR_DOURADO, 'III': '#7b2e3a', 'IV': '#4a4a4a'}
                for grp in grupos:
                    df_g = df_doenca[df_doenca['Estadio_Clinico'] == grp]
                    if len(df_g) > 5:
                        kmf.fit(durations=df_g['Tempo_Meses'], event_observed=df_g['Status_Evento'], label=f"Estadio {grp}")
                        kmf.plot_survival_function(ax=ax, ci_show=False, color=cores_est.get(grp, COR_AZUL_ESCURO), linewidth=2)
                        surv, low, up = extrair_metrica_60_meses(kmf)
                        pct = (len(df_g) / total_doenca) * 100
                        resultados_tabela.append({
                            "Estadio Clínico": grp, 
                            "N_raw": len(df_g), 
                            "Nº de casos": f"{len(df_g):,}".replace(",", "."), 
                            "Sobrevida em 5 anos (IC95%)": f"{surv:.1f}% (IC95% {low:.1f}%–{up:.1f}%)".replace(".", ","),
                            "% do total": f"{pct:.1f}%".replace(".", ",")
                        })
                texto_p = extrair_pvalue_logrank(df_doenca[df_doenca['Estadio_Clinico'].isin(grupos)], 'Estadio_Clinico')
                if texto_p: ax.text(0.02, 0.05, texto_p, transform=ax.transAxes, fontsize=12, color='#333333', bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9))
                ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)
            
            elif comparacao_doenca == "Sexo":
                grupos = ['Feminino', 'Masculino']
                for grp in grupos:
                    df_g = df_doenca[df_doenca['Sexo'] == grp.upper()]
                    if len(df_g) > 0:
                        kmf.fit(durations=df_g['Tempo_Meses'], event_observed=df_g['Status_Evento'], label=grp)
                        kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.15, color=CORES_SEXO.get(grp.upper(), COR_AZUL_ESCURO), linewidth=2)
                        surv, low, up = extrair_metrica_60_meses(kmf)
                        pct = (len(df_g) / total_doenca) * 100
                        resultados_tabela.append({
                            "Sexo": grp, 
                            "N_raw": len(df_g), 
                            "Nº de casos": f"{len(df_g):,}".replace(",", "."), 
                            "Sobrevida em 5 anos (IC95%)": f"{surv:.1f}% (IC95% {low:.1f}%–{up:.1f}%)".replace(".", ","),
                            "% do total": f"{pct:.1f}%".replace(".", ",")
                        })
                texto_p = extrair_pvalue_logrank(df_doenca[df_doenca['Sexo'].isin(['FEMININO', 'MASCULINO'])], 'Sexo')
                if texto_p: ax.text(0.02, 0.05, texto_p, transform=ax.transAxes, fontsize=12, color='#333333', bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9))
                ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)

            configurar_eixos_grafico(ax, titulo_grafico)
            if comparacao_doenca != "Sem divisão (Curva Única)":
                fig.subplots_adjust(right=0.75)
            else:
                fig.tight_layout()
            st.pyplot(fig)
            
            col_d1, col_d2 = st.columns([1, 3])
            with col_d1: download_plot(fig, f"Sobrevida_{doenca_escolhida.replace(' ', '_')}.png")
            
            cid_str = DIC_CIDS_MACRO.get(doenca_escolhida, 'CID-O3')
            desc_comp = ""
            if comparacao_doenca != "Sem divisão (Curva Única)":
                desc_comp = f", comparada por **{comparacao_doenca}**"
                
            st.markdown(f"> 📝 **Descrição da Curva:** Probabilidade de sobrevida global estimada em 5 anos para neoplasias de **{doenca_escolhida}** ({cid_str}){str_filtros}{desc_comp}, pacientes analíticos, no período de {anos_sobrevida[0]}–{anos_sobrevida[1]}.")
            
            if resultados_tabela:
                df_res = pd.DataFrame(resultados_tabela)
                if comparacao_doenca in ["Quinquênio de Diagnóstico", "Estadio Clínico"]:
                    df_res = df_res.drop(columns=["N_raw"])
                else:
                    df_res = df_res.sort_values(by="N_raw", ascending=False).drop(columns=["N_raw"])
                
                if comparacao_doenca != "Sem divisão (Curva Única)":
                    s_tot, l_tot, u_tot = extrair_metrica_60_meses(KaplanMeierFitter().fit(df_doenca['Tempo_Meses'], df_doenca['Status_Evento']))
                    col_chave = df_res.columns[0]
                    linha_total = pd.DataFrame([{
                        col_chave: "TOTAL DA DOENÇA",
                        "Nº de casos": f"{total_doenca:,}".replace(",", "."),
                        "Sobrevida em 5 anos (IC95%)": f"{s_tot:.1f}% (IC95% {l_tot:.1f}%–{u_tot:.1f}%)".replace(".", ","),
                        "% do total": "100,0%"
                    }])
                    df_res = pd.concat([df_res, linha_total], ignore_index=True)
                
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                
                # --- TABELA ADICIONAL: PERFIL DE GÊNERO POR QUINQUÊNIO (DOENÇA ESPECÍFICA) ---
                if filtro_sexo == 'Ambos' and comparacao_doenca == "Quinquênio de Diagnóstico":
                    st.markdown(f"#### Perfil de Gênero por Quinquênio ({doenca_escolhida})")
                    df_quinq_sexo = df_doenca.copy()
                    tabela_q_s = pd.crosstab(df_quinq_sexo['Quinquenio'], df_quinq_sexo['Sexo'])
                    if 'MASCULINO' not in tabela_q_s: tabela_q_s['MASCULINO'] = 0
                    if 'FEMININO' not in tabela_q_s: tabela_q_s['FEMININO'] = 0
                    tabela_q_s['Total'] = tabela_q_s['MASCULINO'] + tabela_q_s['FEMININO']
                    
                    tabela_q_s = tabela_q_s.reindex(grupos).fillna(0)
                    
                    dados_tabela_q_s = []
                    
                    for q in grupos:
                        row = tabela_q_s.loc[q]
                        fem_val = int(row['FEMININO'])
                        masc_val = int(row['MASCULINO'])
                        tot_val = int(row['Total'])
                        
                        pct_fem = f"{(fem_val/total_doenca)*100:.1f}%".replace(".", ",") if total_doenca > 0 else "0,0%"
                        pct_masc = f"{(masc_val/total_doenca)*100:.1f}%".replace(".", ",") if total_doenca > 0 else "0,0%"
                        pct_total = f"{(tot_val/total_doenca)*100:.1f}%".replace(".", ",") if total_doenca > 0 else "0,0%"
                        
                        dados_tabela_q_s.append({
                            "Quinquênio": q,
                            "Nº de casos Masculino": f"{masc_val:,}".replace(",", "."),
                            "% de casos Masculinos": pct_masc,
                            "Nº de casos Feminino": f"{fem_val:,}".replace(",", "."),
                            "% de casos Feminino": pct_fem,
                            "Total": f"{tot_val:,}".replace(",", "."),
                            "% do total": pct_total
                        })
                    
                    tot_masc_g = tabela_q_s['MASCULINO'].sum()
                    tot_fem_g = tabela_q_s['FEMININO'].sum()
                    tot_g = tabela_q_s['Total'].sum()
                    
                    dados_tabela_q_s.append({
                        "Quinquênio": "TOTAL DA DOENÇA",
                        "Nº de casos Masculino": f"{int(tot_masc_g):,}".replace(",", "."),
                        "% de casos Masculinos": f"{(tot_masc_g/total_doenca)*100:.1f}%".replace(".", ",") if total_doenca > 0 else "0,0%",
                        "Nº de casos Feminino": f"{int(tot_fem_g):,}".replace(",", "."),
                        "% de casos Feminino": f"{(tot_fem_g/total_doenca)*100:.1f}%".replace(".", ",") if total_doenca > 0 else "0,0%",
                        "Total": f"{int(tot_g):,}".replace(",", "."),
                        "% do total": "100,0%"
                    })
                    
                    st.dataframe(pd.DataFrame(dados_tabela_q_s), use_container_width=True, hide_index=True)

    elif tipo_grafico == "Quadro Metodológico (CIDs e Morfologias)":
        st.markdown("## Quadro Metodológico: Classificação, Topografia e Morfologia")
        
        st.markdown(f"""
        > 📝 **Draft de Documentação:** Quadro resumo das categorias de câncer incluídas na análise de sobrevida, detalhando a classificação histológica (CID-10), os códigos de topografia e morfologia (CID-O3), a distribuição por sexo e o volume total de casos (coorte {anos_sobrevida[0]}–{anos_sobrevida[1]}). Os filtros aplicados na barra lateral (como a exclusão de casos *in situ*) são refletidos matematicamente neste quadro.
        """)
        
        DIC_CID10_HISTO = {
            'Mama': 'Carcinoma ductal invasivo de mama (C50)',
            'Tireoide': 'Carcinoma papilífero de tireoide (C73)',
            'Próstata': 'Adenocarcinomas de próstata (C61)',
            'Colo do útero': 'Carcinomas de células escamosas de colo do útero (C53)',
            'Vulva': 'Carcinoma escamoso de vulva (C51)',
            'Corpo do útero': 'Adenocarcinoma endometrióide de corpo de útero (C54-C55)',
            'Ovário': 'Tumores malignos de ovário (C56)',
            'Cólon e reto': 'Adenocarcinomas de cólon e reto (C18-C20)',
            'Cavidade oral e orofaringe': 'Carcinomas de células escamosas de cavidade oral e orofaringe (C00-C10)',
            'Pele - melanoma': 'Melanoma de pele (C43)',
            'Pele - não-melanoma': 'Câncer de pele não-melanoma (C44)'
        }
        
        col_topo_local = achar_coluna_local(df_global, ['TOPO', 'CÓDIGO DA TOPOGRAFIA', 'CODIGO DA TOPOGRAFIA'])
        col_morfo_local = achar_coluna_local(df_global, ['MORFO', 'MORFOLOGIA', 'CÓDIGO DA MORFOLOGIA', 'DESCMORFO', 'DESCRIÇÃO DA MORFOLOGIA'])
        
        quadro_dados = []
        grupos_ordenados = df_global['Macro_Topografia'].value_counts().index
        
        for grupo in grupos_ordenados:
            if grupo == 'Outros' or pd.isna(grupo): continue
            
            df_g = df_global[df_global['Macro_Topografia'] == grupo]
            n_casos = len(df_g)
            
            tipo_hist = DIC_CID10_HISTO.get(grupo, f"Neoplasia maligna ({grupo})")
            
            topos = df_g[col_topo_local].dropna().astype(str).str.upper().str.strip()
            topos_clean = sorted(list(set([t for t in topos if t not in ['NAN', 'NONE', '']])))
            topos_str = ", ".join(topos_clean)
            
            morfos_brutos = df_g[col_morfo_local].dropna().astype(str)
            morfos_extraidos = morfos_brutos.str.extract(r'(\d{4,5})')[0].dropna()
            
            if len(morfos_extraidos) > 0:
                morfos_clean = sorted(list(set(morfos_extraidos)))
            else:
                morfos_clean = sorted(list(set([m.strip() for m in morfos_brutos if m.strip() not in ['nan', 'none', '']])))
                
            morfos_str = ", ".join(morfos_clean)
            
            sexos = df_g['Sexo'].dropna().unique()
            sexos_clean = sorted(list(set([s.capitalize() for s in sexos])))
            sexo_str = " / ".join(sexos_clean)
            
            quadro_dados.append({
                "Tipo Histológico (CID-10)": tipo_hist,
                "Topografia (CID-O3)": topos_str,
                "Morfologia (CID-O3)": morfos_str,
                "Sexo": sexo_str,
                "Nº de Casos": n_casos
            })
            
        casos_outros = len(df_global[df_global['Macro_Topografia'] == 'Outros'])
        if casos_outros > 0:
            df_outros = df_global[df_global['Macro_Topografia'] == 'Outros']
            sexos_outros = df_outros['Sexo'].dropna().unique()
            sexos_outros_clean = sorted(list(set([s.capitalize() for s in sexos_outros])))
            sexo_outros_str = " / ".join(sexos_outros_clean)
            
            quadro_dados.append({
                "Tipo Histológico (CID-10)": "Outras Neoplasias Malignas",
                "Topografia (CID-O3)": "Diversas",
                "Morfologia (CID-O3)": "Diversas",
                "Sexo": sexo_outros_str,
                "Nº de Casos": casos_outros
            })
            
        df_quadro = pd.DataFrame(quadro_dados)
        
        total_geral = df_quadro['Nº de Casos'].sum()
        linha_total = pd.DataFrame([{
            "Tipo Histológico (CID-10)": "TOTAL DA COORTE DE SOBREVIDA",
            "Topografia (CID-O3)": "-",
            "Morfologia (CID-O3)": "-",
            "Sexo": "-",
            "Nº de Casos": total_geral
        }])
        df_quadro = pd.concat([df_quadro, linha_total], ignore_index=True)
        
        df_quadro['Nº de Casos'] = df_quadro['Nº de Casos'].apply(lambda x: f"{x:,}".replace(',', '.'))
        
        st.table(df_quadro)
        
        csv_quadro = df_quadro.to_csv(index=False, sep=';', encoding='utf-8-sig')
        st.download_button(
            label="📥 Baixar Quadro em CSV (Para colar no Excel/Word)",
            data=csv_quadro,
            file_name="Quadro_Metodologico_CID.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==========================================
# VISÃO: JORNADA DO PACIENTE
# ==========================================
with aba_jornada:
    st.markdown("<br>", unsafe_allow_html=True)
    anos_jornada = st.slider("Período da Jornada Assistencial:", min_value=ano_min_df, max_value=ano_max_df, value=(ano_min_df, ano_max_df))
    st.markdown("""
    > 📝 **Monitoramento de Agilidade:** Abaixo estão os tempos medianos auditados da trajetória assistencial do paciente oncológico. O motor detecta e remove anomalias cronológicas de RH para garantir métricas clinicamente reais.
    """)
    df_jornada = df_filtrado[(df_filtrado['Ano_Diag'] >= anos_jornada[0]) & (df_filtrado['Ano_Diag'] <= anos_jornada[1])].copy()
    
    def calcular_metricas(serie_dias):
        dados_limpos = serie_dias.dropna()
        dados_limpos = dados_limpos[dados_limpos >= 0]
        return {
            "Mediana (dias)": int(dados_limpos.median()) if not dados_limpos.empty else 0,
            "Média (dias)": f"{dados_limpos.mean():.1f}".replace(".", ",") if not dados_limpos.empty else "0",
            "P25 - P75 (dias)": f"{int(dados_limpos.quantile(0.25))} – {int(dados_limpos.quantile(0.75))}" if not dados_limpos.empty else "0 - 0",
            "n": f"{len(dados_limpos):,}".replace(",", ".")
        }

    m1 = calcular_metricas(df_jornada['Dias_Cons_Diag'])
    m2 = calcular_metricas(df_jornada['Dias_Diag_Trat'])
    m3 = calcular_metricas(df_jornada['Dias_Cons_Trat'])
    
    tabela_jornada = pd.DataFrame([
        {"Intervalo Assistencial": "1ª consulta → diagnóstico", **m1},
        {"Intervalo Assistencial": "Diagnóstico → início do tratamento", **m2},
        {"Intervalo Assistencial": "1ª consulta → início do tratamento", **m3}
    ])
    st.dataframe(tabela_jornada, use_container_width=True, hide_index=True)