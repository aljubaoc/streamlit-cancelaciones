import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Predicción de Cancelación Hotelera", layout="wide")

st.title("🛎️ Predicción de Cancelación de Reservas de Hotel")
st.markdown("Este modelo predice si una reserva será cancelada con base en sus características.")

# Cargar modelo entrenado
@st.cache_resource
def cargar_modelo():
    with open("modelo_final_rf.pkl", "rb") as f:
        return pickle.load(f)

modelo = cargar_modelo()

# Formulario de entrada de datos
with st.form("formulario_prediccion"):
    st.subheader("✍️ Ingrese los datos de la reserva:")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_adultos = st.number_input("Número de adultos", min_value=1, max_value=5, value=2)
        num_niños = st.number_input("Número de niños", min_value=0, max_value=5, value=0)
        num_noches_fin_de_semana = st.slider("Noches en fin de semana", 0, 5, 1)
        num_noches_semana = st.slider("Noches entre semana", 0, 10, 2)

    with col2:
        tipo_plan_comidas = st.selectbox("Tipo de plan de comidas", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
        requiere_parqueadero_texto = st.selectbox("¿Requiere parqueadero?", ["No", "Sí"])
        tipo_habitación_reservada = st.selectbox("Tipo de habitación", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
        antelación_reserva = st.slider("Antelación de la reserva (días)", 0, 500, 50)

    with col3:
        mes_llegada = st.slider("Mes de llegada", 1, 12, 6)
        día_llegada = st.slider("Día de llegada", 1, 31, 15)
        tipo_segmento_mercado = st.selectbox("Segmento de mercado", ["Online", "Offline", "Corporate", "Complementary", "Aviation"])
        huésped_recurrente_texto = st.selectbox("¿Huésped recurrente?", ["No", "Sí"])
        num_cancelaciones_previas = st.slider("Cancelaciones previas", 0, 10, 0)
        num_reservas_previas_no_canceladas = st.slider("Reservas previas no canceladas", 0, 10, 0)
        precio_promedio_por_habitación = st.number_input("Precio promedio por habitación", 0.0, 500.0, 100.0)
        num_solicitudes_especiales = st.slider("Solicitudes especiales", 0, 5, 0)

    enviar = st.form_submit_button("📊 Predecir cancelación")

if enviar:
    # Conversión de campos texto a valores numéricos esperados por el modelo
    requiere_parqueadero = 1 if requiere_parqueadero_texto == "Sí" else 0
    huésped_recurrente = 1 if huésped_recurrente_texto == "Sí" else 0
    año_llegada = 2017  # Valor quemado

    # Crear DataFrame de una sola fila
    datos = pd.DataFrame({
        "num_adultos": [num_adultos],
        "num_niños": [num_niños],
        "num_noches_fin_de_semana": [num_noches_fin_de_semana],
        "num_noches_semana": [num_noches_semana],
        "tipo_plan_comidas": [tipo_plan_comidas],
        "requiere_parqueadero": [requiere_parqueadero],
        "tipo_habitación_reservada": [tipo_habitación_reservada],
        "antelación_reserva": [antelación_reserva],
        "año_llegada": [año_llegada],
        "mes_llegada": [mes_llegada],
        "día_llegada": [día_llegada],
        "tipo_segmento_mercado": [tipo_segmento_mercado],
        "huésped_recurrente": [huésped_recurrente],
        "num_cancelaciones_previas": [num_cancelaciones_previas],
        "num_reservas_previas_no_canceladas": [num_reservas_previas_no_canceladas],
        "precio_promedio_por_habitación": [precio_promedio_por_habitación],
        "num_solicitudes_especiales": [num_solicitudes_especiales]
    })

    # Predicción
    pred = modelo.predict(datos)[0]
    proba = modelo.predict_proba(datos)[0][1]

    st.subheader("🔍 Resultado de la predicción:")

    # Generar texto con los valores de entrada
    texto_entrada = "Para la combinación de estos datos de entrada:\n\n"
    for col, val in datos.iloc[0].items():
        texto_entrada += f"- **{col}**: {val}\n"
    st.markdown(texto_entrada)

    # Mostrar predicción
    if pred == 1:
        st.error(f"⚠️ Esta reserva probablemente será **CANCELADA** ({proba*100:.2f}% de probabilidad).")
    else:
        st.success(f"✅ Esta reserva probablemente **NO será cancelada** ({(1 - proba)*100:.2f}% de probabilidad).")
