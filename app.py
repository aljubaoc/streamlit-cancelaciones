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
