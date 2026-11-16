import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="SOS Indicadores de Saúde Mental Estudantil — CIESA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CONSTANTES E MODELO MATEMÁTICO (ÁLGEBRA LINEAR II)
# ==============================================================================
DIMENSIONS = [
    "Ansiedade",
    "Privação de Sono",
    "Pressão Acadêmica",
    "Isolamento Social",
    "Clima Escolar",
    "Apoio Familiar (Falta)",
]

# Matriz da Transformação Linear T: R^6 -> R^4 (y = A * v)
MATRIX_A = np.array(
    [
        [0.45, 0.20, 0.25, 0.05, 0.00, 0.05],  # y1: Sobrecarga Emocional
        [0.15, 0.45, 0.35, 0.00, 0.05, 0.00],  # y2: Desgaste Físico-Acadêmico
        [0.05, 0.00, 0.05, 0.40, 0.25, 0.25],  # y3: Vulnerabilidade Relacional
        [0.22, 0.20, 0.23, 0.14, 0.10, 0.11],  # y4: Índice Global Ponderado (SOS)
    ],
    dtype=float,
)

INDEX_LABELS = [
    "1. Sobrecarga Emocional",
    "2. Desgaste Físico-Acadêmico",
    "3. Vulnerabilidade Relacional",
    "4. Índice Global Ponderado (SOS)",
]

W_ALERT = np.array([4.5, 4.5, 4.8, 3.8, 3.5, 3.8], dtype=float)
W_MAX = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=float)


def get_initial_dataframe() -> pd.DataFrame:
    """Retorna uma base realista inicial (60 estudantes) para demonstração imediata."""
    records = [
        # CIESA (15 estudantes)
        ("ALU-101", "CIESA", [2.5, 2.8, 2.4, 3.2, 3.5, 2.2]),
        ("ALU-102", "CIESA", [3.0, 2.5, 2.8, 3.6, 3.8, 2.9]),
        ("ALU-103", "CIESA", [1.8, 2.0, 2.1, 2.0, 2.2, 1.7]),
        ("ALU-104", "CIESA", [2.2, 2.4, 2.0, 2.9, 3.1, 2.5]),
        ("ALU-105", "CIESA", [3.5, 3.1, 2.9, 4.1, 4.2, 3.4]),
        ("ALU-106", "CIESA", [2.0, 1.9, 2.2, 2.4, 2.5, 1.9]),
        ("ALU-107", "CIESA", [2.7, 2.9, 2.6, 3.0, 3.3, 2.8]),
        ("ALU-108", "CIESA", [1.5, 1.8, 1.7, 1.6, 1.9, 1.5]),
        ("ALU-109", "CIESA", [3.8, 3.4, 3.1, 4.3, 4.5, 3.7]),
        ("ALU-110", "CIESA", [2.4, 2.2, 2.3, 2.7, 2.9, 2.1]),
        ("ALU-111", "CIESA", [2.9, 2.7, 2.5, 3.4, 3.6, 2.6]),
        ("ALU-112", "CIESA", [2.1, 2.3, 2.0, 2.2, 2.4, 2.0]),
        ("ALU-113", "CIESA", [3.2, 3.0, 2.8, 3.7, 3.9, 3.1]),
        ("ALU-114", "CIESA", [1.9, 2.1, 1.9, 2.1, 2.3, 1.8]),
        ("ALU-115", "CIESA", [2.6, 2.5, 2.4, 2.8, 3.0, 2.4]),
        # UEA (15 estudantes)
        ("ALU-201", "UEA", [3.4, 3.6, 3.5, 2.5, 2.3, 2.6]),
        ("ALU-202", "UEA", [2.8, 3.1, 3.2, 2.2, 2.0, 2.1]),
        ("ALU-203", "UEA", [4.1, 4.2, 4.0, 3.1, 2.8, 3.3]),
        ("ALU-204", "UEA", [2.3, 2.6, 2.7, 1.9, 1.8, 2.0]),
        ("ALU-205", "UEA", [3.7, 3.9, 3.8, 2.9, 2.5, 3.0]),
        ("ALU-206", "UEA", [1.9, 2.2, 2.4, 1.7, 1.6, 1.8]),
        ("ALU-207", "UEA", [3.2, 3.5, 3.4, 2.4, 2.2, 2.5]),
        ("ALU-208", "UEA", [4.4, 4.5, 4.3, 3.5, 3.0, 3.6]),
        ("ALU-209", "UEA", [2.6, 2.9, 3.0, 2.1, 1.9, 2.2]),
        ("ALU-210", "UEA", [3.0, 3.3, 3.1, 2.3, 2.1, 2.4]),
        ("ALU-211", "UEA", [2.1, 2.4, 2.5, 1.8, 1.7, 1.9]),
        ("ALU-212", "UEA", [3.9, 4.0, 3.9, 2.8, 2.6, 3.1]),
        ("ALU-213", "UEA", [2.5, 2.8, 2.9, 2.0, 1.9, 2.1]),
        ("ALU-214", "UEA", [3.3, 3.4, 3.6, 2.6, 2.3, 2.7]),
        ("ALU-215", "UEA", [2.7, 3.0, 2.8, 2.2, 2.0, 2.3]),
        # UFAM (15 estudantes)
        ("ALU-301", "UFAM", [3.1, 3.3, 3.0, 3.4, 2.7, 3.8]),
        ("ALU-302", "UFAM", [2.4, 2.5, 2.3, 2.6, 2.1, 2.9]),
        ("ALU-303", "UFAM", [3.8, 3.7, 3.5, 3.9, 3.2, 4.3]),
        ("ALU-304", "UFAM", [1.7, 1.9, 1.8, 1.9, 1.5, 2.0]),
        ("ALU-305", "UFAM", [2.9, 3.0, 2.8, 3.1, 2.5, 3.5]),
        ("ALU-306", "UFAM", [4.2, 4.0, 3.8, 4.2, 3.5, 4.6]),
        ("ALU-307", "UFAM", [2.2, 2.3, 2.1, 2.4, 1.9, 2.6]),
        ("ALU-308", "UFAM", [3.4, 3.5, 3.2, 3.6, 2.9, 4.0]),
        ("ALU-309", "UFAM", [2.6, 2.7, 2.5, 2.8, 2.2, 3.1]),
        ("ALU-310", "UFAM", [1.9, 2.1, 2.0, 2.1, 1.7, 2.3]),
        ("ALU-311", "UFAM", [3.6, 3.4, 3.3, 3.7, 3.0, 4.1]),
        ("ALU-312", "UFAM", [2.8, 2.9, 2.7, 3.0, 2.4, 3.3]),
        ("ALU-313", "UFAM", [2.0, 2.2, 1.9, 2.3, 1.8, 2.5]),
        ("ALU-314", "UFAM", [3.2, 3.1, 2.9, 3.3, 2.6, 3.7]),
        ("ALU-315", "UFAM", [2.5, 2.6, 2.4, 2.7, 2.1, 3.0]),
        # FAMETRO (15 estudantes)
        ("ALU-401", "FAMETRO", [4.6, 4.7, 4.9, 2.8, 2.2, 3.1]),
        ("ALU-402", "FAMETRO", [4.2, 4.4, 4.6, 2.5, 2.0, 2.8]),
        ("ALU-403", "FAMETRO", [3.8, 4.0, 4.3, 2.2, 1.9, 2.5]),
        ("ALU-404", "FAMETRO", [4.8, 4.9, 5.0, 3.4, 2.6, 3.7]),
        ("ALU-405", "FAMETRO", [3.1, 3.3, 3.6, 1.9, 1.7, 2.1]),
        ("ALU-406", "FAMETRO", [4.4, 4.5, 4.8, 2.9, 2.3, 3.2]),
        ("ALU-407", "FAMETRO", [2.9, 3.1, 3.4, 1.8, 1.6, 2.0]),
        ("ALU-408", "FAMETRO", [4.0, 4.2, 4.5, 2.6, 2.1, 2.9]),
        ("ALU-409", "FAMETRO", [4.7, 4.8, 4.9, 3.2, 2.5, 3.5]),
        ("ALU-410", "FAMETRO", [3.5, 3.7, 4.0, 2.1, 1.8, 2.4]),
        ("ALU-411", "FAMETRO", [4.3, 4.6, 4.7, 2.7, 2.2, 3.0]),
        ("ALU-412", "FAMETRO", [3.3, 3.5, 3.8, 2.0, 1.7, 2.2]),
        ("ALU-413", "FAMETRO", [4.5, 4.7, 4.8, 3.0, 2.4, 3.3]),
        ("ALU-414", "FAMETRO", [2.7, 2.9, 3.2, 1.7, 1.5, 1.9]),
        ("ALU-415", "FAMETRO", [3.9, 4.1, 4.4, 2.4, 2.0, 2.7]),
    ]
    rows = []
    for sid, grupo, vec in records:
        rows.append(
            {
                "id": sid,
                "grupo": grupo,
                "v1": vec[0],
                "v2": vec[1],
                "v3": vec[2],
                "v4": vec[3],
                "v5": vec[4],
                "v6": vec[5],
            }
        )
    return pd.DataFrame(rows)


def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Lê um CSV exportado do Google Forms (6 perguntas) e padroniza como matriz X (n x 6)."""
    df_raw = pd.read_csv(uploaded_file)

    # Remove coluna de carimbo de data/hora do Google Forms se existir
    cols_to_drop = [
        c for c in df_raw.columns if "carimbo" in c.lower() or "timestamp" in c.lower()
    ]
    df_clean = df_raw.drop(columns=cols_to_drop, errors="ignore").copy()

    # Identifica colunas numéricas (as 6 perguntas em escala 1 a 5)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 6:
        raise ValueError(
            "O CSV precisa conter pelo menos 6 colunas numéricas (escala 1 a 5) correspondentes às perguntas."
        )

    q_cols = numeric_cols[:6]

    # Verifica se há coluna categórica (ex: Turma ou Instituição)
    non_num_cols = [c for c in df_clean.columns if c not in q_cols]
    if non_num_cols:
        grupos = df_clean[non_num_cols[0]].astype(str).fillna("Geral")
    else:
        grupos = pd.Series(["Amostra Coletada"] * len(df_clean))

    out = pd.DataFrame(
        {
            "id": [f"RESP-{i+1:03d}" for i in range(len(df_clean))],
            "grupo": grupos.values,
            "v1": df_clean[q_cols[0]].astype(float).clip(1, 5).values,
            "v2": df_clean[q_cols[1]].astype(float).clip(1, 5).values,
            "v3": df_clean[q_cols[2]].astype(float).clip(1, 5).values,
            "v4": df_clean[q_cols[3]].astype(float).clip(1, 5).values,
            "v5": df_clean[q_cols[4]].astype(float).clip(1, 5).values,
            "v6": df_clean[q_cols[5]].astype(float).clip(1, 5).values,
        }
    )
    return out.dropna().reset_index(drop=True)


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def apply_intervention(
    X: np.ndarray, s_sleep: float, s_acad: float, s_social: float, s_family: float
) -> np.ndarray:
    """Aplica matriz diagonal/atenuadora de intervenção preventiva sobre os vetores v in R^6."""
    X_sim = X.copy()
    X_sim[:, 0] = np.maximum(1.0, X[:, 0] * (1 - 0.35 * s_sleep - 0.45 * s_acad - 0.20 * s_social))
    X_sim[:, 1] = np.maximum(1.0, X[:, 1] * (1 - 0.80 * s_sleep - 0.15 * s_acad))
    X_sim[:, 2] = np.maximum(1.0, X[:, 2] * (1 - 0.85 * s_acad - 0.15 * s_sleep))
    X_sim[:, 3] = np.maximum(1.0, X[:, 3] * (1 - 0.75 * s_social - 0.25 * s_family))
    X_sim[:, 4] = np.maximum(1.0, X[:, 4] * (1 - 0.85 * s_social))
    X_sim[:, 5] = np.maximum(1.0, X[:, 5] * (1 - 0.85 * s_family - 0.15 * s_social))
    return X_sim


def compute_eigen_pca(X: np.ndarray):
    """Calcula matriz de covariância C e sua diagonalização ortogonal C = P D P^T."""
    if len(X) < 2:
        eigvals = np.zeros(6)
        eigvecs = np.eye(6)
        return np.zeros((6, 6)), eigvals, eigvecs

    C = np.cov(X, rowvar=False, ddof=1)
    eigvals, eigvecs = np.linalg.eigh(C)
    # Ordenar em ordem decrescente de autovalor
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(0.0, eigvals[idx])
    eigvecs = eigvecs[:, idx]

    # Garantir sinal determinístico nos autovetores
    for j in range(eigvecs.shape[1]):
        if np.sum(eigvecs[:, j]) < 0:
            eigvecs[:, j] *= -1
    return C, eigvals, eigvecs


# ==============================================================================
# ESTADO DA SESSÃO
# ==============================================================================
if "df_students" not in st.session_state:
    st.session_state.df_students = get_initial_dataframe()

# ==============================================================================
# CABEÇALHO
# ==============================================================================
st.title("🧠 SOS Indicadores de Saúde Mental Estudantil")
st.caption(
    "Projeto 04 • Edital nº 02/2026 CIESA (Ciência da Computação — Álgebra Linear II) | "
    "Processamento por Transformações Lineares, Autovalores/Autovetores e Produto Interno"
)

# ==============================================================================
# BARRA LATERAL (FILTROS, UPLOAD CSV E SIMULADOR DE INTERVENÇÃO)
# ==============================================================================
with st.sidebar:
    st.header("📂 Fonte de Dados (CSV)")
    uploaded = st.file_uploader(
        "Importar respostas do Google Forms (.csv)",
        type=["csv"],
        help="Envie o CSV baixado do Google Forms com as 6 perguntas em escala 1 a 5.",
    )
    if uploaded is not None:
        try:
            st.session_state.df_students = parse_uploaded_csv(uploaded)
            st.success(f"CSV carregado com {len(st.session_state.df_students)} respostas!")
        except Exception as err:
            st.error(f"Erro ao ler CSV: {err}")

    if st.button("🔄 Restaurar Base Simulada (60 alunos)", use_container_width=True):
        st.session_state.df_students = get_initial_dataframe()
        st.rerun()

    st.divider()
    st.header("🏫 Filtro de Grupo / Turma")
    grupos_disponiveis = ["Todos os Grupos"] + sorted(
        st.session_state.df_students["grupo"].unique().tolist()
    )
    grupo_selecionado = st.selectbox("Selecione o grupo analisado:", grupos_disponiveis)

    st.divider()
    st.header("🎛️ Simulador de Intervenção")
    st.caption("Aplica atenuação matricial T(v) = A·v sobre os vetores de risco:")
    s_sleep = st.slider("Higiene do Sono & Rotina (%)", 0, 50, 0, 5) / 100.0
    s_acad = st.slider("Apoio Pedagógico & Provas (%)", 0, 50, 0, 5) / 100.0
    s_social = st.slider("Rodas de Convivência / Paz (%)", 0, 50, 0, 5) / 100.0
    s_family = st.slider("Integração Escola-Família (%)", 0, 50, 0, 5) / 100.0

# Filtrar dados
df_base = st.session_state.df_students
if grupo_selecionado != "Todos os Grupos":
    df_filtered = df_base[df_base["grupo"] == grupo_selecionado].copy()
else:
    df_filtered = df_base.copy()

X_raw = df_filtered[["v1", "v2", "v3", "v4", "v5", "v6"]].to_numpy(dtype=float)
X_active = apply_intervention(X_raw, s_sleep, s_acad, s_social, s_family)

# Cálculos Centrais de Álgebra Linear II
v_mean = np.mean(X_active, axis=0)
t_mean = MATRIX_A @ v_mean  # Vetor imagem em R^4
C_cov, eigvals, eigvecs = compute_eigen_pca(X_active)
total_var = np.sum(eigvals) if np.sum(eigvals) > 0 else 1.0
u1, u2 = eigvecs[:, 0], eigvecs[:, 1]

# Risco individual e projeção ortogonal 2D (PCA)
Y_all = (MATRIX_A @ X_active.T).T  # n x 4
global_risks = Y_all[:, 3]
X_centered = X_active - v_mean
pca_coords = X_centered @ eigvecs[:, :2]  # n x 2

# Fator crítico dominante combinando média e peso no autovetor principal u1
scores_impacto = v_mean * (0.6 + 0.4 * np.abs(u1))
top_idx = int(np.argmax(scores_impacto))
top_factor = DIMENSIONS[top_idx]

alert_mask = global_risks >= 3.5
alert_count = int(np.sum(alert_mask))
alert_pct = (alert_count / len(X_active)) * 100.0
wellbeing_score = float(np.clip(((5.0 - t_mean[3]) / 4.0) * 100.0, 0.0, 100.0))
cos_alert = cosine_similarity(v_mean, W_ALERT)

# ==============================================================================
# ABAS PRINCIPAIS (IGUAIS AO PROTÓTIPO)
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Painel Geral",
        "🧮 Motor de Álgebra Linear II",
        "📝 Triagem de Estudante",
        "📘 Cartilha Digital",
    ]
)

# ------------------------------------------------------------------------------
# ABA 1: PAINEL GERAL
# ------------------------------------------------------------------------------
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Índice de Bem-Estar",
        f"{wellbeing_score:.1f} / 100",
        "Saudável" if wellbeing_score >= 70 else "Atenção preventiva",
    )
    k2.metric(
        "Em Zona de Alerta (≥ 3.5)",
        f"{alert_pct:.1f}%",
        f"{alert_count} de {len(X_active)} estudantes",
        delta_color="inverse",
    )
    k3.metric(
        "Fator Crítico Dominante",
        top_factor,
        f"λ₁ explica {(eigvals[0] / total_var) * 100:.1f}% da variância",
    )
    k4.metric(
        "Similaridade c/ Alerta (cos θ)",
        f"{cos_alert:.3f}",
        "Produto Interno em ℝ⁶",
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Perfil Multidimensional vs. Meta Saudável")
        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=np.append(v_mean, v_mean[0]),
                theta=DIMENSIONS + [DIMENSIONS[0]],
                fill="toself",
                name="Vetor Médio (v)",
                line_color="#0d9488",
            )
        )
        meta = [2.2, 2.2, 2.2, 2.0, 1.8, 2.0]
        fig_radar.add_trace(
            go.Scatterpolar(
                r=meta + [meta[0]],
                theta=DIMENSIONS + [DIMENSIONS[0]],
                fill="toself",
                name="Meta de Proteção (≤ 2.2)",
                line=dict(color="#64748b", dash="dash"),
                fillcolor="rgba(100,116,139,0.1)",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
            height=380,
            margin=dict(t=30, b=30, l=40, r=40),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_right:
        st.subheader("Projeção Ortogonal 2D (PCA via Autovetores)")
        df_pca = pd.DataFrame(
            {
                "PC1": pca_coords[:, 0],
                "PC2": pca_coords[:, 1],
                "ID": df_filtered["id"].values,
                "Grupo": df_filtered["grupo"].values,
                "Risco Global": np.round(global_risks, 2),
                "Status": [
                    "Alerta Crítico (≥ 3.5)"
                    if r >= 3.5
                    else ("Atenção Moderada (2.6–3.4)" if r >= 2.6 else "Zona Segura (< 2.6)")
                    for r in global_risks
                ],
            }
        )
        fig_pca = px.scatter(
            df_pca,
            x="PC1",
            y="PC2",
            color="Status",
            hover_data=["ID", "Grupo", "Risco Global"],
            color_discrete_map={
                "Zona Segura (< 2.6)": "#10b981",
                "Atenção Moderada (2.6–3.4)": "#f59e0b",
                "Alerta Crítico (≥ 3.5)": "#e11d48",
            },
            labels={
                "PC1": "Projeção no Autovetor u₁ (PC1)",
                "PC2": "Projeção no Autovetor u₂ (PC2)",
            },
            height=380,
        )
        fig_pca.update_layout(margin=dict(t=30, b=30, l=30, r=30), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_pca, use_container_width=True)

    c_bar, c_idx = st.columns([1.2, 1])
    with c_bar:
        st.subheader("Principais Problemas Identificados (Intensidade Média)")
        df_bar = pd.DataFrame({"Dimensão": DIMENSIONS, "Média": np.round(v_mean, 2)})
        fig_bar = px.bar(
            df_bar.sort_values("Média"),
            x="Média",
            y="Dimensão",
            orientation="h",
            range_x=[0, 5],
            color="Média",
            color_continuous_scale="Tealrose",
            height=300,
        )
        fig_bar.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_idx:
        st.subheader("Índices Gerados por Transformação Linear (y = A·v)")
        for idx_label, val in zip(INDEX_LABELS, t_mean):
            nivel = "🔴 Crítico" if val >= 3.5 else ("🟡 Atenção" if val >= 2.6 else "🟢 Seguro")
            st.write(f"**{idx_label}:** `{val:.2f} / 5.00` — {nivel}")
            st.progress(float(np.clip((val - 1.0) / 4.0, 0.0, 1.0)))

    st.divider()
    st.subheader("Plano Inteligente de Prevenção e Recomendações")
    action_map = {
        "Pressão Acadêmica": (
            "Redistribuir o calendário de avaliações e aplicar simulados diagnósticos sem peso punitivo.",
            "Oficinas práticas de organização de rotina, respiração pré-provas e mentorias colaborativas.",
            "Diversificar instrumentos avaliativos e alinhar expectativas realistas com as famílias.",
        ),
        "Privação de Sono": (
            "Campanha institucional de higiene do sono e adequação do volume de tarefas noturnas.",
            "Monitoramento da disposição matinal e espaços de descompressão nos intervalos.",
            "Sensibilizar responsáveis sobre a relação entre privação de sono e ansiedade juvenil.",
        ),
        "Ansiedade": (
            "Rodas semanais de escuta ativa com a psicologia escolar e protocolo de acolhimento.",
            "Práticas breves de atenção plena (mindfulness) e fortalecimento de projetos culturais/esportivos.",
            "Capacitar docentes para identificar sinais precoces de tensão e manter diálogo sem julgamentos.",
        ),
        "Isolamento Social": (
            "Dinâmicas cooperativas em sala e criação de clubes de interesse estudantil.",
            "Programas de apadrinhamento entre veteranos e calouros para fortalecer pertencimento.",
            "Estimular atividades coletivas presenciais e acompanhar mudanças de sociabilidade.",
        ),
        "Clima Escolar": (
            "Círculos restaurativos de convivência e campanhas permanentes antibullying.",
            "Mediação de conflitos com protagonismo estudantil e cultura de paz.",
            "Envolver pais e docentes em pactos comunitários de convivência pacífica.",
        ),
        "Apoio Familiar (Falta)": (
            "Encontros da Escola de Pais e atendimento individualizado a perfis vulneráveis.",
            "Fortalecimento da rede interna de proteção e escuta psicopedagógica na instituição.",
            "Fornecer guia prático de comunicação afetiva e escuta ativa aos familiares.",
        ),
    }
    rec_prev, rec_bem, rec_fam = action_map.get(top_factor, action_map["Pressão Acadêmica"])
    rc1, rc2, rc3 = st.columns(3)
    rc1.info(f"**🛡️ Ações Preventivas ({top_factor}):**\n\n{rec_prev}")
    rc2.success(f"**🌱 Bem-Estar Estudantil:**\n\n{rec_bem}")
    rc3.warning(f"**🤝 Equipe Pedagógica & Família:**\n\n{rec_fam}")

# ------------------------------------------------------------------------------
# ABA 2: MOTOR DE ÁLGEBRA LINEAR II
# ------------------------------------------------------------------------------
with tab2:
    st.header("Motor Matemático de Álgebra Linear II em Tempo Real")
    st.markdown(
        "Demonstração direta dos cálculos matriciais executados sobre a matriz de dados $X_{n \\times 6}$."
    )

    st.subheader("1. Transformação Linear T: ℝ⁶ → ℝ⁴ (T(v) = A·v)")
    c_m1, c_m2, c_m3 = st.columns([2, 1, 1])
    with c_m1:
        st.caption("Matriz da Transformação Linear A (4×6)")
        df_A = pd.DataFrame(
            MATRIX_A,
            index=["y₁ (Emoc.)", "y₂ (Acad.)", "y₃ (Relac.)", "y₄ (Global)"],
            columns=["Ansied.", "Sono", "Pressão", "Isolam.", "Clima", "Família"],
        )
        st.dataframe(df_A, use_container_width=True)
    with c_m2:
        st.caption("Vetor Médio v ∈ ℝ⁶")
        st.dataframe(
            pd.DataFrame({"v": np.round(v_mean, 3)}, index=[f"v{i+1}" for i in range(6)]),
            use_container_width=True,
        )
    with c_m3:
        st.caption("Imagem T(v) ∈ ℝ⁴")
        st.dataframe(
            pd.DataFrame({"T(v)": np.round(t_mean, 3)}, index=[f"y{i+1}" for i in range(4)]),
            use_container_width=True,
        )

    st.caption(
        "Pelo Teorema do Núcleo e da Imagem: dim(ℝ⁶) = dim(ker T) + dim(Im T) ⇒ 6 = 2 + 4."
    )

    st.divider()
    st.subheader("2. Autovalores, Autovetores e Diagonalização (C = P·D·Pᵀ)")
    ce1, ce2 = st.columns(2)
    with ce1:
        df_eig = pd.DataFrame(
            {
                "Componente": [f"PC{i+1}" for i in range(6)],
                "Autovalor (λᵢ)": np.round(eigvals, 4),
                "% Variância": np.round((eigvals / total_var) * 100, 2),
            }
        )
        st.dataframe(df_eig, use_container_width=True, hide_index=True)
    with ce2:
        df_vec = pd.DataFrame(
            {
                "Variável": DIMENSIONS,
                "Peso em u₁": np.round(u1, 4),
                "Peso em u₂": np.round(u2, 4),
            }
        )
        st.dataframe(df_vec, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("3. Espaços com Produto Interno: Similaridade de Cosseno e Gram-Schmidt")
    grupos_unicos = sorted(df_base["grupo"].unique().tolist())
    if len(grupos_unicos) >= 2:
        means_by_group = []
        for g in grupos_unicos:
            Xg = df_base[df_base["grupo"] == g][["v1", "v2", "v3", "v4", "v5", "v6"]].to_numpy(dtype=float)
            Xg_sim = apply_intervention(Xg, s_sleep, s_acad, s_social, s_family)
            means_by_group.append(np.mean(Xg_sim, axis=0))

        cos_mat = np.zeros((len(grupos_unicos), len(grupos_unicos)))
        for i in range(len(grupos_unicos)):
            for j in range(len(grupos_unicos)):
                cos_mat[i, j] = cosine_similarity(means_by_group[i], means_by_group[j])

        fig_heat = px.imshow(
            np.round(cos_mat, 4),
            x=grupos_unicos,
            y=grupos_unicos,
            text_auto=True,
            color_continuous_scale="Teal",
            title="Matriz de Similaridade de Cosseno entre Grupos/Instituições",
            height=360,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    inner_vw = float(np.dot(v_mean, W_MAX))
    inner_ww = float(np.dot(W_MAX, W_MAX))
    scalar_proj = inner_vw / inner_ww
    ortho_check = abs(float(np.dot(u1, u2)))
    st.info(
        f"**Projeção Ortogonal sobre o Vetor de Alerta Máximo w = [5,5,5,5,5,5]ᵀ:** "
        f"`⟨v, w⟩ = {inner_vw:.2f}` | `⟨w, w⟩ = {inner_ww:.0f}` | "
        f"`proj_w(v) = {scalar_proj:.3f}·w` ({scalar_proj*100:.1f}% do alerta máximo)\n\n"
        f"**Verificação de Ortogonalidade (Gram-Schmidt):** `⟨u₁, u₂⟩ = {ortho_check:.6f} ≈ 0`"
    )

# ------------------------------------------------------------------------------
# ABA 3: TRIAGEM DE ESTUDANTE
# ------------------------------------------------------------------------------
with tab3:
    st.header("Simulador / Cadastro Individual de Triagem (v ∈ ℝ⁶)")
    col_form, col_res = st.columns([1.2, 1])

    with col_form:
        with st.form("form_triagem"):
            sid_new = st.text_input("Identificador Anônimo", value=f"ALU-2026-{len(df_base)+1}")
            grupo_opts = sorted(df_base["grupo"].unique().tolist())
            grupo_new = st.selectbox("Grupo / Instituição", grupo_opts)
            q1 = st.slider("1. Ansiedade e Tensão Diária (v₁)", 1, 5, 3)
            q2 = st.slider("2. Privação de Sono / Cansaço (v₂)", 1, 5, 4)
            q3 = st.slider("3. Pressão Acadêmica / Provas (v₃)", 1, 5, 4)
            q4 = st.slider("4. Isolamento Social / Solidão (v₄)", 1, 5, 2)
            q5 = st.slider("5. Clima Escolar / Conflitos (v₅)", 1, 5, 2)
            q6 = st.slider("6. Falta de Apoio Familiar (v₆)", 1, 5, 3)
            submitted = st.form_submit_button("➕ Inserir no Painel e Recalcular")

        if submitted:
            new_row = pd.DataFrame(
                [
                    {
                        "id": sid_new,
                        "grupo": grupo_new,
                        "v1": float(q1),
                        "v2": float(q2),
                        "v3": float(q3),
                        "v4": float(q4),
                        "v5": float(q5),
                        "v6": float(q6),
                    }
                ]
            )
            st.session_state.df_students = pd.concat(
                [st.session_state.df_students, new_row], ignore_index=True
            )
            st.success("Estudante inserido na base ativa! Atualizando gráficos...")
            st.rerun()

    with col_res:
        v_ind = np.array([q1, q2, q3, q4, q5, q6], dtype=float)
        y_ind = MATRIX_A @ v_ind
        sim_ind = cosine_similarity(v_ind, W_ALERT)
        st.subheader("Processamento Algébrico Instantâneo")
        st.code(
            f"v = {v_ind.astype(int).tolist()}ᵀ ∈ ℝ⁶\n"
            f"T(v) = A·v = {[round(float(x), 2) for x in y_ind]} ∈ ℝ⁴\n"
            f"Similaridade c/ Alerta cos(θ) = {sim_ind:.4f}",
            language="text",
        )
        if y_ind[3] >= 3.5:
            st.error("🔴 **Classificação: Alerta Crítico** — Priorizar escuta psicopedagógica individual.")
        elif y_ind[3] >= 2.6:
            st.warning("🟡 **Classificação: Atenção Moderada** — Incluir em oficinas de rotina e acolhimento.")
        else:
            st.success("🟢 **Classificação: Zona Segura** — Manter acompanhamento preventivo regular.")

# ------------------------------------------------------------------------------
# ABA 4: CARTILHA DIGITAL
# ------------------------------------------------------------------------------
with tab4:
    st.header("📘 Cartilha Educativa: SOS Saúde Mental na Escola")
    st.markdown(
        f"**Resumo Estatístico Atualizado:** Amostra de **{len(X_active)} estudantes** | "
        f"Bem-Estar: **{wellbeing_score:.1f}/100** | "
        f"Zona de Alerta: **{alert_pct:.1f}%** | "
        f"Fator Crítico Principal: **{top_factor}**"
    )
    c_car1, c_car2 = st.columns(2)
    with c_car1:
        st.info(
            "**1. O que é Saúde Mental?**\n\n"
            "Estado de bem-estar em que o estudante reconhece suas capacidades, lida com os estresses cotidianos e aprende de forma produtiva."
        )
        st.warning(
            "**2. Sinais de Alerta**\n\n"
            "- Queda brusca de rendimento ou faltas frequentes\n"
            "- Sonolência excessiva e fadiga\n"
            "- Isolamento repentino dos colegas\n"
            "- Crises de ansiedade antes de avaliações"
        )
    with c_car2:
        st.success(
            "**3. Fatores de Proteção**\n\n"
            "- Rotina equilibrada de sono (7–8h)\n"
            "- Vínculos saudáveis de amizade e acolhimento docente\n"
            "- Diálogo aberto e escuta sem julgamentos em casa"
        )
        st.error(
            "**4. Como Buscar Ajuda (Gratuito e Sigiloso)**\n\n"
            "- **CVV (Centro de Valorização da Vida):** Ligue **188** (24h)\n"
            "- **Rede CAPS / UBS (SUS):** Atendimento psicossocial gratuito\n"
            "- **Coordenação Pedagógica / Psicologia Escolar**"
        )
