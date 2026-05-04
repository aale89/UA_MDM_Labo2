import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
import matplotlib.colors as mcolors

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="PetFinder EDA",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    section[data-testid="stSidebar"] {
        background: #1a1a2e;
        color: #e0e0e0;
    }

    h2 { border-bottom: 2px solid #2ecc71; padding-bottom: 6px; }

    .section-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white !important;
        padding: 16px 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        font-size: 22px;
        font-weight: 700;
    }

    .insight-box {
        background: #f0fdf4;
        border-left: 4px solid #2ecc71;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 14px;
        color: #1a1a1a !important;
    }

    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #f39c12;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 14px;
        color: #1a1a1a !important;
    }

    /* Texto general */
    .main {
        color: #1a1a1a;
    }

    /* Métricas (números grandes) */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* Labels de métricas */
    [data-testid="stMetricLabel"] {
        color: #cccccc !important;
    }

    /* Delta métricas */
    [data-testid="stMetricDelta"] {
        color: #2ecc71 !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] * {
        color: #1a1a1a !important;
    }

/* Insight boxes */
.insight-box {
    color: #1a1a1a !important;
}

</style>
""", unsafe_allow_html=True)


# ── Rutas Relativas Reproducibles ─────────────────────────────────────────────
import os
import streamlit as st
import pandas as pd
import numpy as np

# Al usar rutas relativas, el ZIP funcionará en cualquier computadora
BASE_PATH = os.path.join("input", "train")

# ── Carga y preparación de datos ──────────────────────────────────────────────
@st.cache_data
def load_data():
    # 1. Carga de archivos físicos
    df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
    breed_labels = pd.read_csv(os.path.join(BASE_PATH, 'PetFinder-BreedLabels.csv'))
    color_labels = pd.read_csv(os.path.join(BASE_PATH, 'PetFinder-ColorLabels.csv'))
    state_labels  = pd.read_csv(os.path.join(BASE_PATH, 'PetFinder-StateLabels.csv'))

    # 2. Creación de mapas de referencia
    breed_map = dict(zip(breed_labels['BreedID'], breed_labels['BreedName']))
    color_map = dict(zip(color_labels['ColorID'], color_labels['ColorName']))
    state_map = dict(zip(state_labels['StateID'],  state_labels['StateName']))
    
    # 3. Transformaciones y creación de etiquetas (TODO esto debe estar indentado aquí)
    df['Type_label']          = df['Type'].map({1: 'Perro', 2: 'Gato'})
    df['Gender_label']        = df['Gender'].map({1: 'Macho', 2: 'Hembra', 3: 'Mixto'})
    df['MaturitySize_label']  = df['MaturitySize'].map({1: 'Pequeño', 2: 'Mediano', 3: 'Grande', 4: 'Extra Grande'})
    df['FurLength_label']     = df['FurLength'].map({1: 'Corto', 2: 'Medio', 3: 'Largo'})
    df['Health_label']        = df['Health'].map({1: 'Sano', 2: 'Menor lesión', 3: 'Lesión grave'})
    df['Vaccinated_label']    = df['Vaccinated'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    df['Dewormed_label']      = df['Dewormed'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    df['Sterilized_label']    = df['Sterilized'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    df['Breed1_label']        = df['Breed1'].map(breed_map)
    df['Breed2_label']        = df['Breed2'].map(breed_map)
    df['Color1_label']        = df['Color1'].map(color_map)
    df['Color2_label']        = df['Color2'].map(color_map)
    df['Color3_label']        = df['Color3'].map(color_map)
    df['State_label']         = df['State'].map(state_map)
    df['AdoptionSpeed_label'] = df['AdoptionSpeed'].map({
        0: '0 - Mismo día', 1: '1 - 1ª semana',
        2: '2 - 1er mes',   3: '3 - 2do/3er mes', 4: '4 - No adoptado'
    })
    
    df['HasPhoto']  = (df['PhotoAmt'].fillna(0) > 0).map({True: 'Con foto', False: 'Sin foto'})
    df['HasVideo']  = (df['VideoAmt'].fillna(0) > 0).map({True: 'Con video', False: 'Sin video'})
    df['HasFee']    = (df['Fee'] > 0).map({True: 'Con costo', False: 'Gratuito'})
    df['HasDesc']   = (df['Description'].fillna('').str.len() > 0).map({True: 'Con descripción', False: 'Sin descripción'})
    df['DescLen']   = df['Description'].fillna('').apply(len)
    df['IsGroup']   = (df['Quantity'] > 1).map({True: 'Grupo (>1)', False: 'Individual'})

    # Clasificación de Razas
    razas_gen = ['Mixed Breed', 'Domestic Short Hair', 'Domestic Medium Hair', 'Domestic Long Hair']
    df['TipoRaza'] = np.where(
        (df['Breed1_label'].notna() & df['Breed2_label'].notna()) | df['Breed1_label'].isin(razas_gen),
        'Raza Mixta',
        np.where(df['Breed1_label'].notna() & df['Breed2_label'].isna(), 'Raza Pura', None)
    )

    # Conteo de Colores
    def contar_colores(row):
        return len({row[c] for c in ['Color1_label', 'Color2_label', 'Color3_label'] if pd.notna(row[c])})
    df['n_colores'] = df.apply(contar_colores, axis=1)
    df['ColoresCategoria'] = df['n_colores'].map({1: '1 color', 2: '2 colores', 3: '3 colores'})

    # Perfil de Rescatista
    rs = df.groupby('RescuerID').agg(n_pub=('PetID', 'count')).reset_index()
    rs['Perfil'] = pd.cut(rs['n_pub'], bins=[0, 1, 5, 20, rs['n_pub'].max()],
                          labels=['Ocasional (1)', 'Pequeño (2-5)', 'Mediano (6-20)', 'Alto volumen (>20)'])
    df = df.merge(rs[['RescuerID', 'Perfil']], on='RescuerID', how='left')

    # Retornar el dataframe procesado final
    return df

# Ejecución de la función fuera de la definición
df = load_data()

# ── Paleta ────────────────────────────────────────────────────────────────────
cmap = mcolors.LinearSegmentedColormap.from_list('gr', ['#2ecc71', '#f39c12', '#e74c3c'])
PALETTE_GRAD   = [f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},0.9)' for c in [cmap(i/4) for i in range(5)]]
ADOPTION_ORDER = ['0 - Mismo día', '1 - 1ª semana', '2 - 1er mes', '3 - 2do/3er mes', '4 - No adoptado']
COLOR_SPEED    = dict(zip(ADOPTION_ORDER, PALETTE_GRAD))
PERFIL_ORDER   = ['Ocasional (1)', 'Pequeño (2-5)', 'Mediano (6-20)', 'Alto volumen (>20)']
PERFIL_COLORS  = dict(zip(PERFIL_ORDER, ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']))

# ── HELPER PLOTLY (AGREGDO)─────────────────────────────────────────────────
def fix_plotly_layout(fig):
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1a1a1a')
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐾 PetFinder EDA")
    st.markdown("*UA MCD — Laboratorio II 2026*")
    st.markdown("---")

    seccion = st.radio("📂 **Sección**", options=[
        "🏠 Resumen general",
        "🎯 Variable objetivo",
        "🐕 Tipo de mascota",
        "📊 Variables numéricas",
        "🏷️ Variables categóricas",
        "🐾 Razas",
        "🎨 Colores",
        "💊 Salud",
        "📸 Fotos, videos y costo",
        "🗺️ Geografía",
        "👤 Rescatistas",
        "🔗 Correlaciones",
    ])

    st.markdown("---")
    st.markdown("**⚙️ Filtros globales**")
    tipo_sel  = st.multiselect("Tipo", ['Perro', 'Gato'], default=['Perro', 'Gato'])
    speed_sel = st.multiselect("AdoptionSpeed", ADOPTION_ORDER, default=ADOPTION_ORDER)
    age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
    age_range = st.slider("Edad (meses)", age_min, age_max, (age_min, age_max))
    st.markdown(f"*Registros filtrados: **{len(df[df['Type_label'].isin(tipo_sel) & df['AdoptionSpeed_label'].isin(speed_sel) & df['Age'].between(age_range[0], age_range[1])]):,}***")

# Filtro
mask = (df['Type_label'].isin(tipo_sel) & df['AdoptionSpeed_label'].isin(speed_sel) & df['Age'].between(age_range[0], age_range[1]))
dff  = df[mask].copy()

# ── Helpers ───────────────────────────────────────────────────────────────────
def sh(text): st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)
def insight(t): st.markdown(f'<div class="insight-box">💡 {t}</div>', unsafe_allow_html=True)
def warning(t): st.markdown(f'<div class="warning-box">⚠️ {t}</div>', unsafe_allow_html=True)

def freq_bar_line(data, col, titulo, order=None):
    cross = data.groupby([col, 'AdoptionSpeed_label']).size().reset_index(name='n')
    cross['pct'] = cross.groupby(col)['n'].transform(lambda x: x / x.sum() * 100)
    freq  = cross.groupby(col)['n'].sum().reset_index()
    x_ord = order if order else freq[col].tolist()

    fig = go.Figure()
    for i, spd in enumerate(ADOPTION_ORDER):
        d = cross[cross['AdoptionSpeed_label'] == spd]
        y_vals = []
        t_vals = []
        for x in x_ord:
            row = d[d[col] == x]
            v = float(row['pct'].values[0]) if len(row) > 0 else 0
            y_vals.append(v)
            t_vals.append(f"{int(round(v))}%" if v > 0 else "")
        fig.add_trace(go.Bar(name=spd, x=x_ord, y=y_vals, marker_color=PALETTE_GRAD[i],
                             text=t_vals, textposition='inside', textfont_size=10))

    y_freq = [float(freq[freq[col]==x]['n'].values[0]) if x in freq[col].values else 0 for x in x_ord]
    fig.add_trace(go.Scatter(x=x_ord, y=y_freq, mode='lines+markers', name='Frecuencia',
                             yaxis='y2', line=dict(color='#2c3e50', dash='dash', width=2),
                             marker=dict(size=6, color='#2c3e50')))

    fig.update_layout(title=titulo, barmode='stack', height=400, plot_bgcolor='white',
                      yaxis=dict(title='%', range=[0, 105]),
                      yaxis2=dict(title='Frecuencia', overlaying='y', side='right', showgrid=False),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, font_size=10))
    return fig

# ══════════════════════════════════════════════════════════════════════════════

if seccion == "🏠 Resumen general":
    sh("🏠 Resumen general del dataset")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Registros",      f"{len(dff):,}")
    c2.metric("Perros",         f"{(dff['Type']==1).sum():,}")
    c3.metric("Gatos",          f"{(dff['Type']==2).sum():,}")
    c4.metric("Edad media",     f"{dff['Age'].mean():.1f} m")
    c5.metric("% Adoptados",    f"{(dff['AdoptionSpeed']<4).mean()*100:.1f}%")
    c6.metric("% No adoptados", f"{(dff['AdoptionSpeed']==4).mean()*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Variables del dataset")
        vars_df = pd.DataFrame({
            'Variable':   ['Type','Name','Age','Breed1','Breed2','Gender','Color1','Color2','Color3',
                           'MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health',
                           'Quantity','Fee','State','RescuerID','VideoAmt','Description','PetID','PhotoAmt','AdoptionSpeed'],
            'Tipo':       ['Categ.','Texto','Num.','Categ.','Categ.','Categ.','Categ.','Categ.','Categ.',
                           'Categ.','Categ.','Categ.','Categ.','Categ.','Categ.',
                           'Num.','Num.','Categ.','ID','Num.','Texto','ID','Num.','Target'],
            'Nulos':      [df[c].isnull().sum() for c in ['Type','Name','Age','Breed1','Breed2','Gender',
                           'Color1','Color2','Color3','MaturitySize','FurLength','Vaccinated','Dewormed',
                           'Sterilized','Health','Quantity','Fee','State','RescuerID','VideoAmt',
                           'Description','PetID','PhotoAmt','AdoptionSpeed']]
        })
        st.dataframe(vars_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 🔑 Hallazgos clave")
        for tema, texto in [
            ("Variable objetivo", "Clases relativamente balanceadas. Clase 0 (1er día) la menos frecuente (2.7%) y Clase 2 (1er mes) la más frecuente (26.9%)"),
            ("Salud", "Variable con mayor impacto: lesión grave → ~45% no adoptados"),
            ("Edad", "Las mascotas más jóvenes se adoptan más rápido."),
            ("Fotos", "Las mascotas con fotos tienen mayor velocidad de adopción"),
            ("Costo", "Las mascotas gratuitas se adoptan más rápido"),
            ("Cantidad", "Avisos con >1 mascota tienen mayor tasa de no adopción (clase 4)"),
            ("Raza", "Los perros raza pura se adoptan más rápido que los raza mixta. Para gatos no hay diferencias marcadas"),
            ("Estado", "84% de las mascotas concentradas en 2 Estados"),
            ("Correlación", "Correlaciones numéricas bajas (Spearman). Sin multicolinealidad relevante"),
            ("Asociación categ.", "V de Cramér: State y Breed1 muestran mayor asociación entre categóricas"),
        ]:
            st.markdown(f'<div class="insight-box"><b>{tema}:</b> {texto}</div>', unsafe_allow_html=True)

        st.markdown("### 🛠️ Features candidatas")
        for f in ['HasName','HasPhoto','HasFee','TipoRaza','n_colores','PerfilRescatista','DescLen','IsGroup']:
            st.markdown(f"✦ `{f}`")

elif seccion == "🎯 Variable objetivo":
    sh("🎯 Variable objetivo: AdoptionSpeed")

    counts = dff['AdoptionSpeed_label'].value_counts().reindex(ADOPTION_ORDER).reset_index()
    counts.columns = ['Clase','Cantidad']
    counts['pct'] = (counts['Cantidad'] / counts['Cantidad'].sum() * 100).round(1)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(counts, x='Clase', y='Cantidad', color='Clase',
                     color_discrete_map=COLOR_SPEED,
                     text=counts.apply(lambda r: f"{r['Cantidad']:,} ({r['pct']}%)", axis=1),
                     title='Distribución de AdoptionSpeed')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=420, plot_bgcolor='white')
        st.plotly_chart(fix_plotly_layout(fig),use_container_width=True)
    with col2:
        fig2 = px.pie(counts, names='Clase', values='Cantidad', color='Clase',
                      color_discrete_map=COLOR_SPEED, hole=0.4, title='Proporción de clases')
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

    insight("La clase 2 (1er mes) es la más frecuente con 26.9%, seguida de la 4 (No adoptado) con 28%. Juntas representan más del 50% del dataset.")
    warning("Clase 0 con solo 2.7% está fuertemente subrepresentada. Considerar SMOTE, undersampling o class_weight en el modelo.")

elif seccion == "🐕 Tipo de mascota":
    sh("🐕 Análisis por tipo de mascota")

    col1, col2 = st.columns(2)
    with col1:
        tc = dff['Type_label'].value_counts().reset_index()
        tc.columns = ['Tipo','n']
        tc['pct'] = (tc['n']/tc['n'].sum()*100).round(1)
        fig = px.bar(tc, x='Tipo', y='n', color='Tipo',
                     color_discrete_map={'Perro':'#5BC8F5','Gato':'#4C72B0'},
                     text=tc.apply(lambda r: f"{r['n']:,}\n({r['pct']}%)", axis=1),
                     title='Cantidad por tipo')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=380, plot_bgcolor='white', yaxis_range=[0,9000])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.plotly_chart(freq_bar_line(dff, 'Type_label', 'AdoptionSpeed por tipo', ['Perro','Gato']),
                        use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_a = px.histogram(dff, x='Age', color='Type_label', nbins=40, barmode='overlay', opacity=0.7,
                             color_discrete_map={'Perro':'#5BC8F5','Gato':'#4C72B0'},
                             labels={'Age':'Edad (meses)','Type_label':'Tipo'},
                             title='Distribución de edad por tipo')
        fig_a.update_layout(height=350, plot_bgcolor='white')
        st.plotly_chart(fig_a, use_container_width=True)
    with col4:
        age_s = dff.groupby(['Type_label','AdoptionSpeed_label'])['Age'].median().reset_index()
        fig_b = px.bar(age_s, x='AdoptionSpeed_label', y='Age', color='Type_label', barmode='group',
                       color_discrete_map={'Perro':'#5BC8F5','Gato':'#4C72B0'},
                       labels={'Age':'Edad mediana (m)','AdoptionSpeed_label':'Clase'},
                       title='Edad mediana por clase',
                       category_orders={'AdoptionSpeed_label': ADOPTION_ORDER})
        fig_b.update_layout(height=350, plot_bgcolor='white')
        st.plotly_chart(fig_b, use_container_width=True)

    insight("La distribución de AdoptionSpeed es similar para perros y gatos. No hay diferencia marcada entre especies.")

elif seccion == "📊 Variables numéricas":
    sh("📊 Variables numéricas")

    num_units = {'Age':'meses','Quantity':'unidades','Fee':'MYR','PhotoAmt':'fotos','VideoAmt':'videos'}
    col_sel = st.selectbox("Variable:", list(num_units.keys()))
    unidad  = num_units[col_sel]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Media",   f"{dff[col_sel].mean():.1f} {unidad}")
    c2.metric("Mediana", f"{dff[col_sel].median():.1f} {unidad}")
    c3.metric("Mínimo",  f"{dff[col_sel].min():.0f} {unidad}")
    c4.metric("Máximo",  f"{dff[col_sel].max():.0f} {unidad}")

    col5, col6 = st.columns(2)
    with col5:
        fig_h = px.histogram(dff, x=col_sel, nbins=30, color_discrete_sequence=['#3498db'],
                             title=f'Distribución de {col_sel}',
                             labels={col_sel: f'{col_sel} ({unidad})'})
        fig_h.update_layout(height=380, plot_bgcolor='white')
        st.plotly_chart(fig_h, use_container_width=True)
    with col6:
        fig_bx = px.box(dff, x='AdoptionSpeed_label', y=col_sel, color='AdoptionSpeed_label',
                        color_discrete_map=COLOR_SPEED,
                        category_orders={'AdoptionSpeed_label': ADOPTION_ORDER},
                        title=f'{col_sel} por AdoptionSpeed',
                        labels={'AdoptionSpeed_label':'Clase', col_sel:f'{col_sel} ({unidad})'})
        fig_bx.update_layout(showlegend=False, height=380, plot_bgcolor='white')
        st.plotly_chart(fig_bx, use_container_width=True)

    st.markdown("### Tabla de medianas por clase")
    med = dff.groupby('AdoptionSpeed_label')[list(num_units.keys())].median().reindex(ADOPTION_ORDER)
    fig_hm = px.imshow(med.T, text_auto='.1f', color_continuous_scale='RdYlGn_r',
                       title='Mediana por clase de AdoptionSpeed')
    fig_hm.update_layout(height=280)
    st.plotly_chart(fig_hm, use_container_width=True)

elif seccion == "🏷️ Variables categóricas":
    sh("🏷️ Variables categóricas")

    cat_opts = {
        'Género':              ('Gender_label',        ['Macho','Hembra','Mixto']),
        'Tamaño adulto':       ('MaturitySize_label',  ['Pequeño','Mediano','Grande','Extra Grande']),
        'Largo de pelaje':     ('FurLength_label',     ['Corto','Medio','Largo']),
        'Individual vs Grupo': ('IsGroup',             ['Individual','Grupo (>1)']),
        'Con/Sin descripción': ('HasDesc',             ['Con descripción','Sin descripción']),
    }
    tipo_m  = st.radio("Tipo:", ['Ambos','Perro','Gato'], horizontal=True)
    cat_sel = st.selectbox("Variable:", list(cat_opts.keys()))
    col_c, orden = cat_opts[cat_sel]
    sub = dff if tipo_m == 'Ambos' else dff[dff['Type_label'] == tipo_m]

    col1, col2 = st.columns(2)
    with col1:
        freq = sub[col_c].value_counts().reindex(orden, fill_value=0).reset_index()
        freq.columns = ['Cat','n']
        freq['pct'] = (freq['n']/freq['n'].sum()*100).round(1)
        fig_f = px.bar(freq, x='Cat', y='n', color_discrete_sequence=['#3498db'],
                       text=freq.apply(lambda r: f"{r['n']:,}\n({r['pct']}%)", axis=1),
                       title=f'Frecuencia — {cat_sel}')
        fig_f.update_traces(textposition='outside')
        fig_f.update_layout(height=400, plot_bgcolor='white', showlegend=False, xaxis_title='')
        st.plotly_chart(fig_f, use_container_width=True)
    with col2:
        st.plotly_chart(freq_bar_line(sub, col_c, f'AdoptionSpeed — {cat_sel}', orden),
                        use_container_width=True)

elif seccion == "🐾 Razas":
    sh("🐾 Análisis de razas")

    tab1, tab2, tab3 = st.tabs(["Top razas", "Raza pura vs mixta", "Combinaciones Breed1/2"])

    with tab1:
        tipo_r = st.radio("Tipo:", ['Perro','Gato'], horizontal=True, key='raza_tipo')
        n_top  = st.slider("Cantidad:", 5, 30, 15)
        top_b  = dff[dff['Type'] == (1 if tipo_r=='Perro' else 2)]['Breed1_label'].value_counts().head(n_top).reset_index()
        top_b.columns = ['Raza','n']
        fig = px.bar(top_b.sort_values('n'), x='n', y='Raza', orientation='h',
                     color='n', color_continuous_scale='Blues', text='n',
                     title=f'Top {n_top} razas — {tipo_r}')
        fig.update_layout(height=max(400, n_top*30), plot_bgcolor='white', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        razas_gen = ['Mixed Breed','Domestic Short Hair','Domestic Medium Hair','Domestic Long Hair']
        col1, col2 = st.columns(2)
        for col, tipo, label in zip([col1,col2],[1,2],['Perros','Gatos']):
            with col:
                sub = dff[dff['Type']==tipo]
                cats = {
                    'Raza Pura':  (sub['Breed1_label'].notna() & sub['Breed2_label'].isna() & ~sub['Breed1_label'].isin(razas_gen)).sum(),
                    'Raza Mixta': ((sub['Breed1_label'].notna() & sub['Breed2_label'].notna()) | sub['Breed1_label'].isin(razas_gen)).sum(),
                }
                df_c = pd.DataFrame({'Tipo': list(cats.keys()), 'n': list(cats.values())})
                df_c['pct'] = (df_c['n']/df_c['n'].sum()*100).round(1)
                fig = px.bar(df_c, x='Tipo', y='n', color='Tipo',
                             color_discrete_map={'Raza Pura':'#3498db','Raza Mixta':'#e67e22'},
                             text=df_c.apply(lambda r: f"{r['n']:,}\n({r['pct']}%)", axis=1),
                             title=f'Tipo de raza — {label}')
                fig.update_traces(textposition='outside')
                fig.update_layout(showlegend=False, height=380, plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)

        sub_r = dff[dff['TipoRaza'].notna()]
        st.plotly_chart(freq_bar_line(sub_r, 'TipoRaza', 'AdoptionSpeed por tipo de raza',
                                      ['Raza Pura','Raza Mixta']), use_container_width=True)

    with tab3:
        combos = {
            'Solo Breed1': (dff['Breed1_label'].notna() & dff['Breed2_label'].isna()).sum(),
            'Breed1+Breed2': (dff['Breed1_label'].notna() & dff['Breed2_label'].notna()).sum(),
            'Ninguna': (dff['Breed1_label'].isna() & dff['Breed2_label'].isna()).sum(),
        }
        df_cb = pd.DataFrame({'Combinación': list(combos.keys()), 'n': list(combos.values())})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Razas únicas Breed1", dff['Breed1_label'].nunique())
            st.metric("Razas únicas Breed2", dff['Breed2_label'].nunique())
        with col2:
            fig = px.pie(df_cb, names='Combinación', values='n', hole=0.4,
                         title='Distribución de combinaciones de raza')
            st.plotly_chart(fig, use_container_width=True)

elif seccion == "🎨 Colores":
    sh("🎨 Análisis de colores")

    tab1, tab2 = st.tabs(["Distribución por color", "Cantidad de colores"])

    with tab1:
        color_var = st.selectbox("Variable:", ['Color1_label','Color2_label','Color3_label'])
        tipo_c = st.radio("Tipo:", ['Ambos','Perro','Gato'], horizontal=True, key='col_tipo')
        sub_c  = dff if tipo_c == 'Ambos' else dff[dff['Type_label']==tipo_c]
        orden_c = ['Black','Brown','Golden','Yellow','Cream','Gray','White']
        orden_c_filt = [c for c in orden_c if c in sub_c[color_var].values]

        col1, col2 = st.columns(2)
        with col1:
            freq = sub_c[color_var].value_counts().reindex(orden_c_filt, fill_value=0).reset_index()
            freq.columns = ['Color','n']
            cmap_c = {'Black':'#2c3e50','Brown':'#8B4513','Golden':'#DAA520','Yellow':'#F4D03F',
                      'Cream':'#FDEBD0','Gray':'#95A5A6','White':'#BDC3C7'}
            fig = px.bar(freq, x='Color', y='n', color='Color',
                         color_discrete_map=cmap_c, text='n', title=f'Frecuencia — {color_var}')
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=380, plot_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.plotly_chart(freq_bar_line(sub_c.dropna(subset=[color_var]),
                                          color_var, f'AdoptionSpeed — {color_var}', orden_c_filt),
                            use_container_width=True)

    with tab2:
        tipo_nc = st.radio("Tipo:", ['Ambos','Perro','Gato'], horizontal=True, key='nc_tipo')
        sub_nc  = dff if tipo_nc=='Ambos' else dff[dff['Type_label']==tipo_nc]
        sub_nc  = sub_nc[sub_nc['ColoresCategoria'].notna()]
        orden_nc = ['1 color','2 colores','3 colores']
        verde    = ['#2ecc71','#27ae60','#1a5e37']

        col1, col2 = st.columns(2)
        with col1:
            freq = sub_nc['ColoresCategoria'].value_counts().reindex(orden_nc, fill_value=0).reset_index()
            freq.columns = ['Cat','n']
            freq['pct'] = (freq['n']/freq['n'].sum()*100).round(1)
            fig = px.bar(freq, x='Cat', y='n', color='Cat',
                         color_discrete_map=dict(zip(orden_nc, verde)),
                         text=freq.apply(lambda r: f"{r['n']:,}\n({r['pct']}%)", axis=1),
                         title='Distribución por cantidad de colores')
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=380, plot_bgcolor='white', xaxis_title='')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.plotly_chart(freq_bar_line(sub_nc, 'ColoresCategoria',
                                          'AdoptionSpeed por cantidad de colores', orden_nc),
                            use_container_width=True)

elif seccion == "💊 Salud":
    sh("💊 Variables de salud")

    tipo_s = st.radio("Tipo de mascota:", ['Ambos','Perro','Gato'], horizontal=True)
    sub_s  = dff if tipo_s=='Ambos' else dff[dff['Type_label']==tipo_s]

    salud = {
        'Vacunado':     ('Vaccinated_label',  ['Sí','No','No sabe']),
        'Desparasitado':('Dewormed_label',    ['Sí','No','No sabe']),
        'Esterilizado': ('Sterilized_label',  ['Sí','No','No sabe']),
        'Salud general':('Health_label',      ['Sano','Menor lesión','Lesión grave']),
    }
    cols = st.columns(2)
    for i, (titulo, (col_name, orden)) in enumerate(salud.items()):
        with cols[i%2]:
            st.plotly_chart(freq_bar_line(sub_s, col_name, titulo, orden),
                            use_container_width=True)

    insight("La salud general es la variable con mayor impacto en AdoptionSpeed. Lesión grave → ~45% no adoptados.")
    warning("La esterilización puede estar confundida con la edad — mascotas mayores suelen estar esterilizadas y son menos adoptadas.")

elif seccion == "📸 Fotos, videos y costo":
    sh("📸 Fotos, videos y costo de adopción")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(freq_bar_line(dff,'HasPhoto','AdoptionSpeed según fotos',['Con foto','Sin foto']),
                        use_container_width=True)
    with col2:
        st.plotly_chart(freq_bar_line(dff,'HasVideo','AdoptionSpeed según video',['Con video','Sin video']),
                        use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(freq_bar_line(dff,'HasFee','AdoptionSpeed según costo',['Gratuito','Con costo']),
                        use_container_width=True)
    with col4:
        fig_bx = px.box(dff, x='AdoptionSpeed_label', y='PhotoAmt', color='AdoptionSpeed_label',
                        color_discrete_map=COLOR_SPEED,
                        category_orders={'AdoptionSpeed_label': ADOPTION_ORDER},
                        title='Cantidad de fotos por clase',
                        labels={'AdoptionSpeed_label':'Clase','PhotoAmt':'Fotos'})
        fig_bx.update_layout(showlegend=False, height=400, plot_bgcolor='white')
        st.plotly_chart(fig_bx, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    sf = (dff['PhotoAmt'].fillna(0)==0).sum()
    sv = (dff['VideoAmt'].fillna(0)==0).sum()
    gr = (dff['Fee']==0).sum()
    c1.metric("Sin foto",   f"{sf:,} ({sf/len(dff)*100:.1f}%)")
    c2.metric("Sin video",  f"{sv:,} ({sv/len(dff)*100:.1f}%)")
    c3.metric("Gratuitos",  f"{gr:,} ({gr/len(dff)*100:.1f}%)")

    insight("Las mascotas con fotos y las gratuitas tienen mayor velocidad de adopción.")

elif seccion == "🗺️ Geografía":
    sh("🗺️ Análisis geográfico")

    STATE_COORDS = {
    'Selangor': (3.0738, 101.5183),
    'Kuala Lumpur': (3.1390, 101.6869),
    'Johor': (1.4927, 103.7414),
    'Pulau Pinang': (5.4164, 100.3327),
    'Perak': (4.5921, 101.0901),
    'Negeri Sembilan': (2.7258, 101.9424),
    'Melaka': (2.1896, 102.2501),
    'Pahang': (3.8126, 103.3256),
    'Kedah': (6.1184, 100.3685),
    'Kelantan': (6.1254, 102.2386),
    'Terengganu': (5.3117, 103.1324),
    'Sabah': (5.9804, 116.0735),
    'Sarawak': (1.5533, 110.3592)
    }

    st.markdown("### 🌍 Mapa interactivo")

    geo_df = dff.groupby('State_label').agg(
        n=('PetID','count')
    ).reset_index()

    # distribución por clase
    dist = dff.groupby(['State_label','AdoptionSpeed_label']).size().unstack(fill_value=0)

    # pasar a %
    dist_pct = dist.div(dist.sum(axis=1), axis=0) * 100

    # merge
    geo_df = geo_df.merge(dist_pct, on='State_label', how='left')

    # % no adoptados (por compatibilidad)
    geo_df['pct_no_adopted'] = geo_df['4 - No adoptado']

    geo_df[['lat','lon']] = geo_df['State_label'].map(STATE_COORDS).apply(pd.Series)
    geo_df = geo_df.dropna()

    fig_map = px.scatter_mapbox(
        geo_df,
        lat="lat",
        lon="lon",
        size="n",
        color="pct_no_adopted",
        hover_name="State_label",
        hover_data={
            "n": True,
            "0 - Mismo día": ':.1f',
            "1 - 1ª semana": ':.1f',
            "2 - 1er mes": ':.1f',
            "3 - 2do/3er mes": ':.1f',
            "4 - No adoptado": ':.1f',
            "pct_no_adopted": False,
            "lat": False,
            "lon": False
        },
        
        labels={
            "n": "Cantidad"
        },
        color_continuous_scale="Turbo",
        range_color=(0, 100),
        size_max=50,
        zoom=5
    )


    fig_map.update_layout(
        mapbox_style="carto-positron",
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(
            title=dict(
                text="% No adoptado",
                font=dict(color="#1a1a1a")
            ),
            tickfont=dict(color="#1a1a1a")
        )
    )

    st.plotly_chart(
        fix_plotly_layout(fig_map),
        use_container_width=True,
        config={"scrollZoom": True}
    )



    n_est = st.slider("Estados a mostrar:", 5, dff['State_label'].nunique(), 10)

    col1, col2 = st.columns(2)
    with col1:
        sc = dff['State_label'].value_counts().head(n_est).reset_index()
        sc.columns = ['Estado','n']
        fig = px.bar(sc.sort_values('n'), x='n', y='Estado', orientation='h',
                     color='n', color_continuous_scale='Blues', text='n',
                     title=f'Top {n_est} estados por frecuencia')
        fig.update_layout(height=max(380, n_est*35), plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        sna = (dff.groupby('State_label')['AdoptionSpeed']
               .apply(lambda x: (x==4).mean()*100).reset_index())
        sna.columns = ['Estado','% No adoptados']
        sna = sna.sort_values('% No adoptados', ascending=False).head(n_est)
        fig2 = px.bar(sna, x='% No adoptados', y='Estado', orientation='h',
                      color='% No adoptados', color_continuous_scale='Reds',
                      title=f'% No adoptados por estado (top {n_est})')
        fig2.update_layout(height=max(380, n_est*35), plot_bgcolor='white')
        st.plotly_chart(fig2, use_container_width=True)

    top2 = dff['State_label'].value_counts().head(2).index.tolist()
    dff_g = dff.copy()
    dff_g['State_group'] = dff_g['State_label'].apply(lambda x: x if x in top2 else 'Others')
    st.plotly_chart(freq_bar_line(dff_g, 'State_group',
                                  'AdoptionSpeed — Top 2 estados vs. Others', top2+['Others']),
                    use_container_width=True)

elif seccion == "👤 Rescatistas":
    sh("👤 Perfil del rescatista")

    rs = dff.groupby('RescuerID').agg(
        n_pub=('PetID','count'),
        speed_media=('AdoptionSpeed','mean'),
        pct_adoptados=('AdoptionSpeed', lambda x: (x<4).mean()*100),
        pct_rapidos=('AdoptionSpeed',   lambda x: (x<=1).mean()*100),
    ).reset_index()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total rescatistas",    f"{len(rs):,}")
    c2.metric("Publicaciones (med.)", f"{rs['n_pub'].median():.0f}")
    c3.metric("Publicaciones (máx.)", f"{rs['n_pub'].max()}")
    c4.metric("Con >20 publicaciones",f"{(rs['n_pub']>20).sum():,}")

    col1, col2 = st.columns(2)
    with col1:
        fig_h = px.histogram(rs, x='n_pub', nbins=40, color_discrete_sequence=['steelblue'],
                             title='Publicaciones por rescatista',
                             labels={'n_pub':'Publicaciones'})
        fig_h.update_layout(height=380, plot_bgcolor='white')
        st.plotly_chart(fig_h, use_container_width=True)
    with col2:
        pc = dff['Perfil'].value_counts().reindex(PERFIL_ORDER, fill_value=0).reset_index()
        pc.columns = ['Perfil','n']
        pc['pct'] = (pc['n']/pc['n'].sum()*100).round(1)
        fig_p = px.bar(pc, x='Perfil', y='n', color='Perfil',
                       color_discrete_map=PERFIL_COLORS,
                       text=pc.apply(lambda r: f"{r['n']:,}\n({r['pct']}%)", axis=1),
                       title='Rescatistas por perfil')
        fig_p.update_traces(textposition='outside')
        fig_p.update_layout(showlegend=False, height=380, plot_bgcolor='white')
        st.plotly_chart(fig_p, use_container_width=True)

    sub_perf = dff[dff['Perfil'].notna()]
    st.plotly_chart(freq_bar_line(sub_perf, 'Perfil',
                                  'AdoptionSpeed por perfil de rescatista', PERFIL_ORDER),
                    use_container_width=True)

    st.markdown("### 🏆 Rescatistas con más de 150 publicaciones")
    top_r = rs[rs['n_pub']>150].sort_values('n_pub', ascending=False).copy()
    top_r['RescuerID'] = top_r['RescuerID'].str[:12]+'...'
    top_r.columns = ['RescuerID','Publicaciones','Speed media','% Adoptados','% Rápidos']
    st.dataframe(top_r.style.format({'Speed media':'{:.2f}','% Adoptados':'{:.1f}%','% Rápidos':'{:.1f}%'}),
                 use_container_width=True, hide_index=True)

elif seccion == "🔗 Correlaciones":
    sh("🔗 Análisis de correlaciones")

    tab1, tab2 = st.tabs(["Spearman — numéricas", "V de Cramér — categóricas"])

    with tab1:
        num_cols = ['Age','Quantity','Fee','PhotoAmt','VideoAmt','AdoptionSpeed']
        corr = dff[num_cols].corr(method='spearman').round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1, aspect='auto',
                        title='Correlación de Spearman — Numéricas + AdoptionSpeed')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        insight("Correlaciones bajas entre todas las variables numéricas. Sin multicolinealidad relevante.")

    with tab2:
        @st.cache_data
        def calc_cramers(data):
            def cv(x, y):
                cm = pd.crosstab(x, y)
                chi2 = chi2_contingency(cm)[0]
                n = cm.sum().sum()
                phi2 = chi2/n
                r, k = cm.shape
                phi2c = max(0, phi2-((k-1)*(r-1))/(n-1))
                rc = r-((r-1)**2)/(n-1)
                kc = k-((k-1)**2)/(n-1)
                return np.sqrt(phi2c/min((kc-1),(rc-1)))

            cols = ['Type','Gender','Color1','Color2','Color3','MaturitySize','FurLength',
                    'Vaccinated','Dewormed','Sterilized','Health','State','AdoptionSpeed']
            m = pd.DataFrame(index=cols, columns=cols, dtype=float)
            for c1 in cols:
                for c2 in cols:
                    m.loc[c1,c2] = 1.0 if c1==c2 else cv(data[[c1,c2]].dropna()[c1], data[[c1,c2]].dropna()[c2])
            return m

        with st.spinner('Calculando V de Cramér... (puede tardar unos segundos)'):
            cm = calc_cramers(dff)

        fig = px.imshow(cm.astype(float).round(2), text_auto='.2f',
                        color_continuous_scale='RdYlGn', zmin=0, zmax=1, aspect='auto',
                        title='Matriz de asociación — V de Cramér (categóricas)')
        fig.update_layout(height=560)
        st.plotly_chart(fig, use_container_width=True)
        insight("V de Cramér de 0 a 1. State y Breed muestran mayor asociación con AdoptionSpeed.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🐾 PetFinder Adoption Prediction — Dashboard EDA | UA MCD Laboratorio II 2026")
