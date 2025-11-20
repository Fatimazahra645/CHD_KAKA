import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Configuration de la page
# -------------------------------
st.set_page_config(
    page_title="Prédiction du risque de CHD",
    page_icon="🫀",
    layout="wide"
)

# -------------------------------
# Header avec image et texte
# -------------------------------
col1, col2 = st.columns([1,2])
with col1:
    st.image("coeur.jpg", use_column_width=True)
with col2:
    st.title("🩺 Prédiction du risque de maladie cardiaque (CHD)")
    st.markdown("""
    Cette application web a été **développée avec VS Code** et déployée avec **Streamlit**.  
    Elle utilise un modèle de Machine Learning sauvegardé dans `Model1.pkl`  
    (pipeline : prétraitement + ACP + régression logistique) à partir du dataset *CHD.csv*.
    """)

st.markdown("---")

# -------------------------------
# Chargement du modèle
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("Model1.pkl")
    return model

model = load_model()

# -------------------------------
# Sidebar pour les entrées utilisateur
# -------------------------------
st.sidebar.header("🧾 Informations du patient")

age = st.sidebar.number_input("Âge", min_value=10, max_value=100, value=50)
sbp = st.sidebar.number_input("Pression systolique (sbp)", min_value=80.0, max_value=250.0, value=140.0)
ldl = st.sidebar.number_input("LDL (mauvais cholestérol)", min_value=0.0, max_value=10.0, value=4.0)
adiposity = st.sidebar.number_input("Adiposity", min_value=0.0, max_value=60.0, value=25.0)
obesity = st.sidebar.number_input("Obesity", min_value=0.0, max_value=60.0, value=30.0)
famhist = st.sidebar.selectbox("Antécédents familiaux (famhist)", ["present", "absent"])

submitted = st.sidebar.button("🚀 Prédire le risque")

# -------------------------------
# Prédiction
# -------------------------------
if submitted:
    input_data = pd.DataFrame([{
        "sbp": sbp,
        "ldl": ldl,
        "adiposity": adiposity,
        "obesity": obesity,
        "age": age,
        "famhist": famhist
    }])
    
    st.subheader("📊 Données saisies")
    st.dataframe(input_data)

    proba_chd = model.predict_proba(input_data)[0,1]
    pred_chd = model.predict(input_data)[0]

    st.subheader("💡 Résultat de la prédiction")
    
    if pred_chd == 1:
        st.markdown(f"""
        <div style='padding:20px;background-color:#ffcccc;border-radius:10px'>
            ⚠️ Risque élevé de CHD 
            Probabilité estimée : {proba_chd:.2f}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='padding:20px;background-color:#ccffcc;border-radius:10px'>
            ✔️ Risque faible de CHD  
            Probabilité estimée : {proba_chd:.2f}
        </div>
        """, unsafe_allow_html=True)
    
    # -------------------------------
    # Graphique des probabilités
    # -------------------------------
    fig, ax = plt.subplots()
    ax.bar(["CHD=0","CHD=1"], [1-proba_chd, proba_chd], color=["#2a9d8f","#e63946"])
    ax.set_ylim(0,1)
    ax.set_ylabel("Probabilité")
    ax.set_title("Distribution des probabilités")
    st.pyplot(fig)

    st.info("⚠️ Cette application est à but pédagogique et ne remplace pas un avis médical.")
