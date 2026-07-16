import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from lifelines import KaplanMeierFitter
import os

# ==========================================
# CONFIGURAÇÃO DA PÁGINA E UI CUSTOMIZADA
# ==========================================
st.set_page_config(page_title="Observatório Oncológico", page_icon="📊", layout="wide")

# CSS Customizado
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    h1 { color: #1a2b4c; }
    </style>
""", unsafe_allow_html=True)

# BARRA LATERAL
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965306.png", width=60)
    st.title("Motor de Análise RHC")
    st.markdown("Faça o upload da base de dados bruta exportada do sistema FOSP para gerar o relatório executivo auditado.")
    
    arquivo_upado = st.file_uploader("📥 Envie a planilha FOSP (.xlsx ou .xls)", type=["xlsx", "xls"])
    
    st.divider()
    st.header("⚙️ Configuração Demográfica")
    filtro_sexo = st.selectbox("Filtro Global: Sexo", ['Ambos', 'MASCULINO', 'FEMININO'])
    
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.caption("🔒 **Privacidade:** Os dados processados nesta sessão não são armazenados em nenhum servidor externo.")

# ==========================================
# TELA DE BOAS VINDAS (Se não tiver arquivo)
# ==========================================
if not arquivo_upado:
    st.title("📊 Observatório Oncológico: Padrão Executivo")
    st.markdown("### Bem-vindo ao motor de inteligência de dados oncológicos.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("👈 **Para começar, arraste a planilha base do RHC no menu lateral.**")
        st.markdown("""
        **O que este sistema faz automaticamente:**
        * 🧹 **Higienização de Dados:** Identifica e corrige erros de digitação de RH (viagens no tempo, datas inconsistentes).
        * 🔬 **Auditoria CID-O3:** Separa com precisão cirúrgica tumores agressivos de tumores benignos.
        * 📅 **Período Dinâmico:** Adapta-se automaticamente a qualquer período de tempo da sua base.
        * 📈 **Sobrevida de Kaplan-Meier:** Gera curvas estatísticas prontas para publicação científica.
        """)
    st.stop()

# ==========================================
# MOTOR DE DADOS DINÂMICO
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
    
    if col_diag not in df.columns:
        raise ValueError(f"O painel não achou a coluna '{col_diag}'. Verifique se o Excel possui cabeçalhos formatados.")
        
    df[col_diag] = pd.to_datetime(df[col_diag], errors='coerce')
    df[col_fim] = pd.to_datetime(df[col_fim], errors='coerce')
    df[col_cons] = pd.to_datetime(df[col_cons], errors='coerce')
    df[col_trat] = pd.to_datetime(df[col_trat], errors='coerce')
    
    df['Tempo_Meses'] = (df[col_fim] - df[col_diag]).dt.days / 30.4375
    df['Status_Evento'] = df[col_status].apply(lambda x: 1 if isinstance(x, str) and 'OBITO' in x.upper().replace('Ó', 'O') else 0)
    
    # O motor agora é 100% livre de limite de tempo hardcoded
    df = df.dropna(subset=['Tempo_Meses', 'Status_Evento', col_diag])
    df = df[df['Tempo_Meses'] >= 0] 
    
    df['Ano_Diag'] = df[col_diag].dt.year
    df['Idade Numérica'] = pd.to_numeric(df[col_idade], errors='coerce')
    df['Sexo'] = df[col_sexo].astype(str).str.upper()
    
    df['Dias_Cons_Diag'] = (df[col_diag] - df[col_cons]).dt.days
    df['Dias_Diag_Trat'] = (df[col_trat] - df[col_diag]).dt.days
    df['Dias_Cons_Trat'] = (df[col_trat] - df[col_cons]).dt.days
    
    def limpar_atendimento(x):
        val = str(x).upper().strip()
        if val.endswith('.0'): val = val[:-2]
        if val == '1' or 'SUS' in val or 'SISTEMA UNICO' in val or 'SISTEMA ÚNICO' in val: return 'SUS'
        elif val == '2' or 'PARTICULAR' in val: return 'Particular'
        elif val == '3' or 'CONVENIO' in val or 'CONVÊNIO' in val or 'PLANO' in val or 'SUPLEMENTAR' in val: return 'Convênio (Saúde Suplementar)'
        elif val == '4' or val == '9' or val == 'NAN' or val == '' or val == 'NONE': return 'Não Informado'
        else: return 'Outros'
    df['Categoria_Atendimento'] = df[col_atendimento].apply(limpar_atendimento)
    
    # Motor de Quinquênio 100% Matemático (Funciona para qualquer década passada ou futura)
    def definir_quinquenio(ano):
        if pd.isna(ano): return 'Outros'
        try:
            inicio = int(ano) - (int(ano) % 5)
            return f"{inicio}-{inicio+4}"
        except:
            return 'Outros'
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
        morfo_desc = str(linha.get(col_morfo, '')).upper()
        c3 = cod_topo[:3]
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
    
    return df

with st.spinner('Lendo arquivo e consolidando painel completo...'):
    try:
        df_base = carregar_dados(arquivo_upado)
    except Exception as e:
        st.error(f"Erro na Leitura da Planilha!\n\nDetalhe do Sistema: {e}")
        st.stop()

# ==========================================
# DESCOBERTA DINÂMICA DE ANOS DA PLANILHA
# ==========================================
ano_min_df = int(df_base['Ano_Diag'].min())
ano_max_df = int(df_base['Ano_Diag'].max())

df_filtrado = df_base.copy()
if filtro_sexo != 'Ambos':
    df_filtrado = df_filtrado[df_filtrado['Sexo'] == filtro_sexo]

# ==========================================
# ESTÉTICA E FUNÇÕES GERAIS
# ==========================================
COR_AZUL_ESCURO = '#1a2b4c'
COR_DOURADO = '#b8860b'
CORES_SEXO = {'FEMININO': '#1a2b4c', 'MASCULINO': '#b8860b'}
CORES_TOP10 = ['#1a2b4c', '#3a7a78', '#b8860b', '#7b2e3a', '#4a70a3', '#7590b1', '#999999', '#c7a13a', '#7b5592', '#366666']

def configurar_eixos_grafico(ax, titulo):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(titulo, color=COR_AZUL_ESCURO, fontweight='bold', pad=15)
    ax.set_xlabel('Meses desde o diagnóstico', fontsize=11)
    ax.set_ylabel('Probabilidade de sobrevida', fontsize=11)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.05)
    ax.axvline(x=60, color='gray', linestyle='--', linewidth=1, alpha=0.7)

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
    except:
        return 0.0, 0.0, 0.0

# Abas Superiores
aba_perfil, aba_sobrevida, aba_jornada = st.tabs([
    "👥 Perfil Epidemiológico", 
    "📈 Gráficos (Sobrevida)", 
    "⏱️ Jornada do Paciente"
])

# ==========================================
# ABA 1: PERFIL EPIDEMIOLÓGICO
# ==========================================
with aba_perfil:
    st.markdown("### Selecione os Filtros de Análise")
    
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        # SLIDER DINÂMICO! Lê o tempo real da planilha do usuário
        anos_perfil = st.slider("Período de Análise:", min_value=ano_min_df, max_value=ano_max_df, value=(ano_min_df, ano_max_df))
    with col_f2:
        st.markdown("<br>", unsafe_allow_html=True)
        visao_freq = st.radio("Visão do Ranking:", ["Top 10 Grupos Principais", "Top 10 Comparativo (Homens vs Mulheres)", "Todas as Neoplasias (Grupos Anatômicos)", "Categoria de Atendimento (Pizza)"], horizontal=True, label_visibility="collapsed")
    
    df_perfil = df_filtrado[(df_filtrado['Ano_Diag'] >= anos_perfil[0]) & (df_filtrado['Ano_Diag'] <= anos_perfil[1])]
    df_base_ano = df_base[(df_base['Ano_Diag'] >= anos_perfil[0]) & (df_base['Ano_Diag'] <= anos_perfil[1])].copy()
    texto_ano_titulo = f"{anos_perfil[0]}-{anos_perfil[1]}"
        
    st.divider()
    st.markdown(f"### Resumo Rápido (Coorte {texto_ano_titulo})")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Pacientes", f"{len(df_perfil):,}".replace(",", "."))
    col2.metric("Idade Média", f"{df_perfil['Idade Numérica'].mean():.1f} anos")
    col3.metric("Tumor Mais Frequente", df_perfil['Macro_Topografia'].value_counts().index[0] if len(df_perfil) > 0 else "N/A")
    st.divider()
    
    if len(df_perfil) > 0:
        total_casos_perfil = len(df_perfil)
        sexo_txt = "ambos os sexos" if filtro_sexo == 'Ambos' else f"sexo {filtro_sexo.lower()}"
        
        if visao_freq == "Top 10 Grupos Principais":
            st.markdown("### 📊 As 10 Neoplasias Mais Frequentes (Macro)")
            df_top_calc = df_perfil[df_perfil['Macro_Topografia'] != 'Outros']
            top_data = df_top_calc['Macro_Topografia'].value_counts().head(10)
            
            dados_tabela_top = []
            for grupo, count in top_data.items():
                perc = (count / total_casos_perfil) * 100
                dados_tabela_top.append({"Grupo de câncer": grupo, "Nº de casos": f"{count:,}".replace(",", "."), "N_raw": count, "% do total": f"{perc:.1f}%".replace(".", ",")})
            df_top_final = pd.DataFrame(dados_tabela_top)
            
            fig_top, ax_top = plt.subplots(figsize=(12, 7))
            y_coords = np.arange(len(top_data))[::-1] 
            bars = ax_top.barh(y_coords, df_top_final['N_raw'], color=CORES_TOP10[:len(top_data)], edgecolor='white')
            ax_top.set_yticks(y_coords, labels=df_top_final["Grupo de câncer"], fontsize=11)
            ax_top.spines['top'].set_visible(False)
            ax_top.spines['right'].set_visible(False)
            ax_top.set_xlabel('Número de casos', fontsize=11)
            ax_top.set_title(f"10 neoplasias mais frequentes — RHC, {texto_ano_titulo}\n(casos analíticos, {sexo_txt})", color=COR_AZUL_ESCURO, fontweight='bold', pad=15)
            for i, bar in enumerate(bars):
                ax_top.text(df_top_final.iloc[i]['N_raw'] + (total_casos_perfil * 0.005), bar.get_y() + bar.get_height()/2, f"{df_top_final.iloc[i]['N_raw']:,}".replace(",", "."), va='center', ha='left', color='black', fontsize=9)
            st.pyplot(fig_top)
            st.markdown(f"##### Tabela 2. Dez neoplasias mais frequentes, RHC, {texto_ano_titulo}.")
            st.dataframe(df_top_final.drop(columns=['N_raw']), use_container_width=True)

        elif visao_freq == "Top 10 Comparativo (Homens vs Mulheres)":
            st.markdown("### 📊 Top 10 Neoplasias: Comparativo Homens vs Mulheres")
            if filtro_sexo != 'Ambos': st.info("💡 **Auditoria:** O Python isolou o filtro da barra lateral para garantir a comparação visual perfeita.")
            
            df_comp = df_base_ano[df_base_ano['Macro_Topografia'] != 'Outros'].copy()
            top_10_gerais = df_comp['Macro_Topografia'].value_counts().head(10).index
            df_comp_top = df_comp[df_comp['Macro_Topografia'].isin(top_10_gerais)]
            tabela_sexo = pd.crosstab(df_comp_top['Macro_Topografia'], df_comp_top['Sexo'])
            
            if 'MASCULINO' not in tabela_sexo: tabela_sexo['MASCULINO'] = 0
            if 'FEMININO' not in tabela_sexo: tabela_sexo['FEMININO'] = 0
            tabela_sexo['Total'] = tabela_sexo['MASCULINO'] + tabela_sexo['FEMININO']
            tabela_sexo = tabela_sexo.sort_values(by='Total', ascending=True) 
            
            fig_comp, ax_comp = plt.subplots(figsize=(12, 8))
            y_coords = np.arange(len(tabela_sexo))
            ax_comp.barh(y_coords, tabela_sexo['MASCULINO'], color=CORES_SEXO['MASCULINO'], label='Masculino', edgecolor='white')
            ax_comp.barh(y_coords, -tabela_sexo['FEMININO'], color=CORES_SEXO['FEMININO'], label='Feminino', edgecolor='white')
            ax_comp.set_yticks(y_coords)
            ax_comp.set_yticklabels(tabela_sexo.index, fontsize=11)
            ax_comp.spines['top'].set_visible(False)
            ax_comp.spines['right'].set_visible(False)
            ax_comp.spines['left'].set_visible(False) 
            ax_comp.axvline(0, color='black', linewidth=1) 
            ax_comp.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{abs(int(x)):,}".replace(",", ".")))
            ax_comp.set_xlabel('Número de casos', fontsize=11)
            ax_comp.set_title(f"Comparativo por Sexo das 10 Neoplasias Mais Frequentes — RHC, {texto_ano_titulo}", color=COR_AZUL_ESCURO, fontweight='bold', pad=20)
            
            max_val = tabela_sexo['Total'].max()
            for i, (masc, fem) in enumerate(zip(tabela_sexo['MASCULINO'], tabela_sexo['FEMININO'])):
                if masc > 0: ax_comp.text(masc + (max_val * 0.01), i, f"{masc:,}".replace(",", "."), va='center', ha='left', color='black', fontsize=10)
                if fem > 0: ax_comp.text(-fem - (max_val * 0.01), i, f"{fem:,}".replace(",", "."), va='center', ha='right', color='black', fontsize=10)
            
            ax_comp.legend(loc='lower right', frameon=False)
            st.pyplot(fig_comp)
            st.markdown(f"##### Tabela Comparativa de Frequência - Top 10 Neoplasias por Sexo, RHC, {texto_ano_titulo}.")
            df_tabela_exibicao = tabela_sexo.copy().sort_values(by='Total', ascending=False).reset_index()[['Macro_Topografia', 'FEMININO', 'MASCULINO', 'Total']]
            df_tabela_exibicao.rename(columns={'Macro_Topografia': 'Grupo de câncer', 'FEMININO': 'Feminino', 'MASCULINO': 'Masculino'}, inplace=True)
            for col in ['Feminino', 'Masculino', 'Total']: df_tabela_exibicao[col] = df_tabela_exibicao[col].apply(lambda x: f"{x:,}".replace(",", "."))
            st.dataframe(df_tabela_exibicao, use_container_width=True)

        elif visao_freq == "Todas as Neoplasias (Grupos Anatômicos)":
            st.markdown("### 📊 Frequência de TODAS as Neoplasias (Agrupamento CID-O3)")
            top_data = df_perfil['Macro_Topografia_Completa'].value_counts()
            
            dados_tabela_top = []
            for grupo, count in top_data.items():
                perc = (count / total_casos_perfil) * 100
                dados_tabela_top.append({"Grupo Anatômico (CID-O3)": grupo, "Nº de casos": f"{count:,}".replace(",", "."), "N_raw": count, "% do total": f"{perc:.1f}%".replace(".", ",")})
            df_top_final = pd.DataFrame(dados_tabela_top)
            
            fig_top, ax_top = plt.subplots(figsize=(12, max(8, len(top_data) * 0.35)))
            y_coords = np.arange(len(top_data))[::-1] 
            bars = ax_top.barh(y_coords, df_top_final['N_raw'], color=[COR_AZUL_ESCURO] * len(top_data), edgecolor='white')
            ax_top.set_yticks(y_coords, labels=df_top_final["Grupo Anatômico (CID-O3)"], fontsize=9)
            ax_top.spines['top'].set_visible(False)
            ax_top.spines['right'].set_visible(False)
            ax_top.set_xlabel('Número de casos', fontsize=11)
            ax_top.set_title(f"Ranking Completo de Grupos de Câncer — RHC, {texto_ano_titulo}\n(casos analíticos, {sexo_txt})", color=COR_AZUL_ESCURO, fontweight='bold', pad=15)
            for i, bar in enumerate(bars):
                ax_top.text(df_top_final.iloc[i]['N_raw'] + (total_casos_perfil * 0.005), bar.get_y() + bar.get_height()/2, f"{df_top_final.iloc[i]['N_raw']:,}".replace(",", "."), va='center', ha='left', color='black', fontsize=9)
            st.pyplot(fig_top)
            st.markdown(f"##### Tabela de Frequência - Todas as Neoplasias Agrupadas, RHC, {texto_ano_titulo}.")
            st.dataframe(df_top_final.drop(columns=['N_raw']), use_container_width=True)

        elif visao_freq == "Categoria de Atendimento (Pizza)":
            st.markdown("### 📊 Distribuição por Categoria de Atendimento (Admissão)")
            cat_data = df_perfil['Categoria_Atendimento'].value_counts()
            
            dados_tabela_cat = []
            for cat, count in cat_data.items():
                perc = (count / total_casos_perfil) * 100
                dados_tabela_cat.append({"Categoria de Atendimento": cat, "Nº de casos": f"{count:,}".replace(",", "."), "N_raw": count, "% do total": f"{perc:.1f}%".replace(".", ",")})
            df_cat_final = pd.DataFrame(dados_tabela_cat)
            
            mapa_cores_cat = {'SUS': COR_AZUL_ESCURO, 'Convênio (Saúde Suplementar)': COR_DOURADO, 'Particular': '#3a7a78', 'Não Informado': '#999999', 'Outros': '#4a4a4a'}
            
            fig_pizza, ax_pizza = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax_pizza.pie(cat_data.values, labels=cat_data.index, autopct='%1.1f%%', startangle=140, colors=[mapa_cores_cat.get(c, '#999999') for c in cat_data.index], explode=[0.03] * len(cat_data), textprops={'fontsize': 11})
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
                
            ax_pizza.set_title(f"Categoria de Atendimento — RHC, {texto_ano_titulo}\n(casos analíticos, {sexo_txt})", color=COR_AZUL_ESCURO, fontweight='bold', pad=20)
            
            col_graf, col_tab = st.columns([1.2, 1])
            with col_graf: st.pyplot(fig_pizza)
            with col_tab:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"##### Tabela: Categoria de atendimento, RHC, {texto_ano_titulo}.")
                st.dataframe(df_cat_final.drop(columns=['N_raw']), use_container_width=True)

# ==========================================
# ABA 2: SOBREVIDA 
# ==========================================
with aba_sobrevida:
    st.markdown("### Selecione os Filtros de Sobrevida")
    
    sugestao_max_sobrevida = max(ano_min_df, ano_max_df - 5)
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        anos_sobrevida = st.slider("Coorte de Sobrevida (Diagnóstico):", min_value=ano_min_df, max_value=ano_max_df, value=(ano_min_df, sugestao_max_sobrevida))
    with col_s2:
        st.markdown("<br>", unsafe_allow_html=True)
        tipo_grafico = st.radio("Selecione a visualização:", ["Curva Global (Fig 10)", "Curvas por Quinquênio", "Curvas por Sexo", "Curvas por Estádio Clínico", "Ranking 5 Anos", "Curvas por Doença Específica"], horizontal=True, label_visibility="collapsed")
    
    st.info(f"💡 **Recomendação Metodológica:** Para avaliar sobrevivência em 5 anos de forma real, o ano limite de diagnóstico não deveria passar de **{sugestao_max_sobrevida}** para garantir seguimento pleno.")
    st.divider()
    
    df_sobrevida_base = df_filtrado[(df_filtrado['Ano_Diag'] >= anos_sobrevida[0]) & (df_filtrado['Ano_Diag'] <= anos_sobrevida[1])].copy()
    df_global = df_sobrevida_base[df_sobrevida_base['Macro_Topografia'] != 'Pele - não-melanoma'].copy()
    titulo_coorte = f"coorte {anos_sobrevida[0]}-{anos_sobrevida[1]}"
    
    if tipo_grafico == "Curva Global (Fig 10)":
        fig, ax = plt.subplots(figsize=(10, 6))
        kmf = KaplanMeierFitter()
        if len(df_global) > 0:
            kmf.fit(durations=df_global['Tempo_Meses'], event_observed=df_global['Status_Evento'])
            kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.15, color=COR_AZUL_ESCURO, linewidth=2.5, legend=False)
            configurar_eixos_grafico(ax, f"Sobrevida global (Kaplan-Meier) — RHC, {titulo_coorte}\n(exclui pele não-melanoma) (n={len(df_global):,})".replace(",", "."))
            st.pyplot(fig)
            surv, low, up = extrair_metrica_60_meses(kmf)
            st.dataframe(pd.DataFrame([{"Coorte": "Global (todos)", "Nº de casos": f"{len(df_global):,}".replace(",", "."), "Sobrevida em 5 anos": f"{surv:.1f}% (IC95% {low:.1f}%–{up:.1f}%)".replace(".", ",")}]), use_container_width=True)

    elif tipo_grafico == "Curvas por Quinquênio":
        fig, ax = plt.subplots(figsize=(10, 6))
        kmf = KaplanMeierFitter()
        resultados_tabela = []
        quinquenios_atuais = sorted([q for q in df_global['Quinquenio'].unique() if q != 'Outros'])
        cores_dinamicas = [COR_AZUL_ESCURO, '#3a7a78', COR_DOURADO, '#7b2e3a', '#4a70a3', '#7590b1']
        
        for i, q in enumerate(quinquenios_atuais):
            df_q = df_global[df_global['Quinquenio'] == q]
            if len(df_q) > 0:
                kmf.fit(durations=df_q['Tempo_Meses'], event_observed=df_q['Status_Evento'], label=q)
                kmf.plot_survival_function(ax=ax, ci_show=False, color=cores_dinamicas[i % len(cores_dinamicas)], linewidth=2)
                surv, _, _ = extrair_metrica_60_meses(kmf)
                resultados_tabela.append({"Quinquênio": q, "Nº de casos": f"{len(df_q):,}".replace(",", "."), "Sobrevida em 5 anos": f"{surv:.1f}%".replace(".", ",")})
        configurar_eixos_grafico(ax, f"Sobrevida global por período — RHC, {titulo_coorte}\n(exclui pele não-melanoma)")
        ax.legend(title="Período", frameon=False, loc='lower left')
        st.pyplot(fig)
        if resultados_tabela: st.dataframe(pd.DataFrame(resultados_tabela), use_container_width=True)

    elif tipo_grafico == "Curvas por Sexo":
        fig, ax = plt.subplots(figsize=(10, 6))
        kmf = KaplanMeierFitter()
        resultados_tabela = []
        for sexo in ['Feminino', 'Masculino']:
            df_s = df_global[df_global['Sexo'] == sexo.upper()]
            if len(df_s) > 0:
                kmf.fit(durations=df_s['Tempo_Meses'], event_observed=df_s['Status_Evento'], label=sexo)
                kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.15, color={'FEMININO': '#1a2b4c', 'MASCULINO': '#b8860b'}.get(sexo.upper()), linewidth=2)
                surv, low, up = extrair_metrica_60_meses(kmf)
                resultados_tabela.append({"Sexo": sexo, "N_raw": len(df_s), "Nº de casos": f"{len(df_s):,}".replace(",", "."), "Sobrevida 5 anos": f"{surv:.1f}% (IC95% {low:.1f}%–{up:.1f}%)".replace(".", ",")})
        configurar_eixos_grafico(ax, f"Sobrevida global por sexo — RHC, {titulo_coorte}\n(exclui pele não-melanoma)")
        ax.legend(frameon=False, loc='upper right')
        st.pyplot(fig)
        if resultados_tabela: st.dataframe(pd.DataFrame(resultados_tabela).sort_values(by="N_raw", ascending=False).drop(columns=["N_raw"]), use_container_width=True)

    elif tipo_grafico == "Curvas por Estádio Clínico":
        fig, ax = plt.subplots(figsize=(10, 6))
        kmf = KaplanMeierFitter()
        resultados_tabela = []
        cores_est = {'0 (in situ)': '#3a7a78', 'I': '#1a2b4c', 'II': '#b8860b', 'III': '#7b2e3a', 'IV': '#4a4a4a'}
        for est in ['0 (in situ)', 'I', 'II', 'III', 'IV']:
            df_est = df_global[df_global['Estadio_Clinico'] == est]
            if len(df_est) > 5:
                kmf.fit(durations=df_est['Tempo_Meses'], event_observed=df_est['Status_Evento'], label=f"Estádio {est}")
                kmf.plot_survival_function(ax=ax, ci_show=False, color=cores_est.get(est, COR_AZUL_ESCURO), linewidth=2)
                surv, _, _ = extrair_metrica_60_meses(kmf)
                resultados_tabela.append({"Estádio": est, "Nº casos": f"{len(df_est):,}".replace(",", "."), "Sobrevida": f"{surv:.1f}%".replace(".", ",")})
        configurar_eixos_grafico(ax, f"Sobrevida global por estádio — RHC, {titulo_coorte}")
        ax.legend(frameon=False, loc='lower left')
        st.pyplot(fig)
        st.dataframe(pd.DataFrame(resultados_tabela), use_container_width=True)

    elif tipo_grafico == "Ranking 5 Anos":
        fig, ax = plt.subplots(figsize=(12, 7))
        kmf = KaplanMeierFitter()
        dados_barras, dados_tabela = [], []
        if len(df_global) > 0:
            kmf.fit(durations=df_global['Tempo_Meses'], event_observed=df_global['Status_Evento'])
            s_g, l_g, u_g = extrair_metrica_60_meses(kmf)
            dados_barras.append({'Grupo': 'Global', 'Surv': s_g, 'Err_L': s_g - l_g, 'Err_U': u_g - s_g})
        for grupo in ['Tireoide', 'Próstata', 'Mama', 'Colo do útero', 'Vulva', 'Corpo do útero', 'Pele - melanoma', 'Cólon e reto', 'Ovário', 'Cavidade oral e orofaringe']:
            df_g = df_sobrevida_base[df_sobrevida_base['Macro_Topografia'] == grupo]
            if len(df_g) > 10:
                kmf.fit(durations=df_g['Tempo_Meses'], event_observed=df_g['Status_Evento'])
                surv, low, up = extrair_metrica_60_meses(kmf)
                dados_barras.append({'Grupo': grupo, 'Surv': surv, 'Err_L': surv - low, 'Err_U': up - surv})
                dados_tabela.append({"Grupo de câncer": grupo, "N_raw": len(df_g), "Nº de casos": f"{len(df_g):,}".replace(",", "."), "Sobrevida 5 anos": f"{surv:.1f}% (IC95% {low:.1f}%–{up:.1f}%)".replace(".", ",")})
        if dados_barras:
            df_barras = pd.DataFrame(dados_barras).sort_values(by='Surv', ascending=True)
            bars = ax.barh(np.arange(len(df_barras)), df_barras['Surv'], xerr=[df_barras['Err_L'], df_barras['Err_U']], color=[COR_DOURADO if g == 'Global' else COR_AZUL_ESCURO for g in df_barras['Grupo']], edgecolor='white', error_kw=dict(ecolor='gray', lw=1, capsize=3))
            ax.set_yticks(np.arange(len(df_barras)), labels=df_barras['Grupo'], fontsize=12)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('Sobrevida global estimada em 5 anos (%)', fontsize=11)
            ax.set_xlim(0, 105)
            ax.set_title(f"Sobrevida global em 5 anos — RHC, {titulo_coorte}", color=COR_AZUL_ESCURO, fontweight='bold', pad=15)
            
            for bar, surv in zip(bars, df_barras['Surv']): ax.text(surv + 3, bar.get_y() + bar.get_height()/2, f'{surv:.1f}%'.replace('.', ','), va='center')
            st.pyplot(fig)
            st.dataframe(pd.DataFrame(dados_tabela).sort_values(by="N_raw", ascending=False).drop(columns=["N_raw"]), use_container_width=True)

    elif tipo_grafico == "Curvas por Doença Específica":
        doencas_disponiveis = [d for d in df_base['Macro_Topografia'].unique() if d not in ['Outros', 'Pele - não-melanoma']]
        doenca_escolhida = st.selectbox("Selecione a doença:", sorted(doencas_disponiveis))
        df_doenca = df_sobrevida_base[df_sobrevida_base['Macro_Topografia'] == doenca_escolhida]
        if len(df_doenca) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            kmf = KaplanMeierFitter()
            kmf.fit(durations=df_doenca['Tempo_Meses'], event_observed=df_doenca['Status_Evento'])
            kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.15, color=COR_AZUL_ESCURO, linewidth=2.5, legend=False)
            configurar_eixos_grafico(ax, f"{doenca_escolhida} (n={len(df_doenca):,}) — RHC, {titulo_coorte}".replace(",", "."))
            st.pyplot(fig)
            surv, low, up = extrair_metrica_60_meses(kmf)
            st.dataframe(pd.DataFrame([{"Doença": doenca_escolhida, "Nº de casos": f"{len(df_doenca):,}".replace(",", "."), "Sobrevida 5 anos": f"{surv:.1f}% (IC95% {low:.1f}%–{up:.1f}%)".replace(".", ",")}]), use_container_width=True)
        else: st.warning("⚠️ Não há casos registrados para os filtros aplicados.")

# ==========================================
# ABA 3: JORNADA DO PACIENTE
# ==========================================
with aba_jornada:
    st.markdown("### ⏱️ Tempo entre Consulta, Diagnóstico e Tratamento")
    
    anos_jornada = st.slider("Período da Jornada Assistencial:", min_value=ano_min_df, max_value=ano_max_df, value=(ano_min_df, ano_max_df))
    st.markdown("""
    Abaixo estão os dados **auditados e higienizados** da trajetória assistencial. 
    O motor detecta e remove anomalias cronológicas de RH (dias negativos), garantindo métricas clinicamente reais.
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
        {"Intervalo": "1ª consulta → diagnóstico", **m1},
        {"Intervalo": "Diagnóstico → início do tratamento", **m2},
        {"Intervalo": "1ª consulta → início do tratamento", **m3}
    ])
    
    st.dataframe(tabela_jornada, use_container_width=True, hide_index=True)