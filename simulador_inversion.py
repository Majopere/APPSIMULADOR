import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from googletrans import Translator

# Configuración de la página de Streamlit
st.set_page_config(page_title="Simulador Allianz OptiMaxx", layout="wide")

# Lista de ETFs disponibles con sus símbolos de Yahoo Finance
etfs = {
    "AZ China": "ASHR",
    "AZ MSCI TAIWAN INDEX FD": "EWT",
    "AZ RUSSELL 2000": "IWM",
    "AZ Brasil": "EWZ",
    "AZ MSCI UNITED KINGDOM": "EWU",
    "AZ DJ US FINANCIAL SECT": "IYF",
    "AZ BRIC": "BKF",
    "AZ MSCI SOUTH KOREA IND": "EWY",
    "AZ BARCLAYS AGGREGATE": "AGG",
    "AZ Mercados Emergentes": "EEM",
    "AZ MSCI EMU": "EZU",
    "AZ FTSE/XINHUA CHINA 25": "FXI",
    "AZ Oro": "GLD",
    "AZ QQQ NASDAQ 100": "QQQ",
    "AZ MSCI ASIA EX-JAPAN": "AAXJ",
    "AZ BARCLAYS 1-3 YEAR TR": "SHY",
    "AZ MSCI ACWI INDEX FUND": "ACWI",
    "AZ SILVER TRUST": "SLV",
    "AZ MSCI HONG KONG INDEX": "EWH",
    "AZ SPDR S&P 500 ETF TRUST": "SPY",
    "AZ MSCI JAPAN INDEX FD": "EWJ",
    "AZ BG EUR GOVT BOND 1-3": "IBGE.MI",
    "AZ SPDR DJIA TRUST": "DIA",
    "AZ MSCI FRANCE INDEX FD": "EWQ",
    "AZ DJ US OIL & GAS EXPL": "IEO",
    "AZ VANGUARD EMERGING MARKET ETF": "VWO",
    "AZ MSCI AUSTRALIA INDEX": "EWA",
    "AZ IPC LARGE CAP T R TR": "LCT.MX",
    "AZ FINANCIAL SELECT SECTOR SPDR": "XLF",
    "AZ MSCI CANADA": "EWC",
    "AZ S&P LATIN AMERICA 40": "ILF",
    "AZ HEALTH CARE SELECT SECTOR": "XLV",
    "AZ MSCI GERMANY INDEX": "EWG",
    "AZ DJ US HOME CONSTRUCT": "ITB"
    
}

# Título de la aplicación
st.title("Simulador Allianz OptiMaxx 🚀")

# Selección de múltiples ETFs
st.subheader("Selecciona uno o más ETFs para la inversión 📈")
etfs_seleccionados = st.multiselect("Selecciona los ETFs", list(etfs.keys()))

# Configuración de inversión
st.subheader("Detalles de la Inversión 💼")
aportacion_inicial = st.number_input("Aportación Inicial (MXN)", min_value=250000, step=1000, value=250000)

# Configuración de pesos específicos para cada ETF seleccionado
st.subheader("Asignación de Pesos para los ETFs Seleccionados ⚖️")
pesos = []
total_peso = 0

# Si se selecciona solo un ETF, asignarle automáticamente el 100% del peso
if len(etfs_seleccionados) == 1:
    st.write("Asignando automáticamente 100% al único ETF seleccionado.")
    pesos = [100]
    total_peso = 100
else:
    for etf in etfs_seleccionados:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(etf)
        with col2:
            peso = st.number_input(f"Ponderación de {etf} (%)", min_value=0, max_value=100, step=1, value=0, key=etf)
            pesos.append(peso)
            total_peso += peso

# Mostrar la suma de los pesos asignados y el faltante/excedente para llegar a 100%
faltante = 100 - total_peso
if faltante > 0:
    st.warning(f"Falta asignar un {faltante}% para alcanzar el 100% de ponderación.")
elif faltante < 0:
    st.error(f"Has excedido el 100% de ponderación en {abs(faltante)}%.")
else:
    st.success("La suma de las ponderaciones es exactamente 100%.")

# Validar que la suma de los pesos sea 100% para permitir la simulación
if faltante == 0:
    # Configuración del horizonte de inversión
    horizonte_inversion = st.slider("Horizonte de Inversión (años)", min_value=1, max_value=30, value=10)
    
    # Configuración para las 3 aportaciones adicionales (opcionales)
    st.subheader("Configuración de Aportaciones Adicionales (Opcionales)")
    anios_aportacion = []
    montos_aportacion = []
    max_anio_aporte = horizonte_inversion - 1

    for i in range(1, 4):
        col1, col2 = st.columns(2)
        with col1:
            anio = col1.number_input(f"Año de la Aportación {i}", min_value=1, max_value=max_anio_aporte, step=1, value=min(i * 5, max_anio_aporte))
        with col2:
            monto = col2.number_input(f"Monto de la Aportación {i} (MXN)", min_value=0, step=1000, value=0)
        if monto > 0:
            anios_aportacion.append(anio)
            montos_aportacion.append(monto)

    # Botón para iniciar la simulación
    if st.button("Generar Simulador"):
        pesos_normalizados = [peso / 100 for peso in pesos]

        # Cálculo del rendimiento y volatilidad ponderada
        rendimiento_portafolio_ponderado = 0.07  # Tasa de rendimiento anual promedio de ejemplo
        rendimiento_mensual_ponderado = (1 + rendimiento_portafolio_ponderado) ** (1/12) - 1

        # Simulación de la evolución del capital acumulado
        saldo_final = aportacion_inicial
        capital_acumulado = [saldo_final]
        fechas = pd.date_range(start=pd.Timestamp.today(), periods=horizonte_inversion * 12, freq='M')

        for mes in range(horizonte_inversion * 12):
            anio_actual = mes // 12 + 1
            if anio_actual in anios_aportacion:
                indice = anios_aportacion.index(anio_actual)
                saldo_final += montos_aportacion[indice]
            saldo_final *= (1 + rendimiento_mensual_ponderado)
            capital_acumulado.append(saldo_final)

        # Iniciar las pestañas
        tabs = st.tabs(["Información de ETFs", "Tabla de Rentabilidad y Volatilidad", "Resultados de la Inversión"])

        # Pestaña de Información de ETFs
        with tabs[0]:
            st.header("Información de los ETFs Seleccionados")
            translator = Translator()
            for etf_nombre in etfs_seleccionados:
                ticker = etfs[etf_nombre]
                etf = yf.Ticker(ticker)
                descripcion_original = etf.info.get("longBusinessSummary", "Descripción no disponible")
                descripcion = translator.translate(descripcion_original, src='en', dest='es').text
                st.subheader(f"{etf_nombre} ({ticker})")
                st.write(descripcion)

        # Pestaña de Tabla de Rentabilidad y Volatilidad
        with tabs[1]:
            st.header("Tabla de Rentabilidad y Volatilidad")
            resultados = []
            rendimiento_portafolio_ponderado = 0

            # Crear la figura para la gráfica interactiva
            fig = go.Figure()

            for idx, etf_nombre in enumerate(etfs_seleccionados):
                ticker = etfs[etf_nombre]
                etf = yf.Ticker(ticker)
                historial = etf.history(period="10y").resample('M').last()  # Resamplear por mes
                historial['Rendimiento Mensual'] = historial['Close'].pct_change()
                rendimiento_mensual = historial['Rendimiento Mensual'].mean()
                riesgo_mensual = historial['Rendimiento Mensual'].std()
                rendimiento_anualizado = (1 + rendimiento_mensual) ** 12 - 1
                riesgo_anualizado = riesgo_mensual * np.sqrt(12)

                # Agregar resultados a la tabla
                resultados.append([etf_nombre, f"{rendimiento_anualizado * 100:.2f}%", f"{riesgo_anualizado * 100:.2f}%"])

                # Calcular rendimiento ponderado del portafolio
                rendimiento_portafolio_ponderado += rendimiento_anualizado * pesos_normalizados[idx]

                # Agregar serie de capital acumulado al gráfico
                capital_acumulado = [aportacion_inicial]
                for mes in range(1, len(historial)):
                    capital_acumulado.append(capital_acumulado[-1] * (1 + historial['Rendimiento Mensual'].iloc[mes]))
                fig.add_trace(go.Scatter(
                    x=historial.index,
                    y=capital_acumulado,
                    mode='lines',
                    name=etf_nombre
                ))

            # Crear DataFrame de resultados
            df_resultados = pd.DataFrame(resultados, columns=["ETF", "Rentabilidad Anualizada", "Volatilidad Anualizada"])
            st.table(df_resultados)

            # Mostrar la gráfica interactiva
            fig.update_layout(
                title="Evolución del Capital Acumulado de Cada ETF",
                xaxis_title="Fecha",
                yaxis_title="Capital Acumulado (MXN)",
                template="plotly_white"
            )
            st.plotly_chart(fig)

            # Tabla de rentabilidad por plazos
            st.header("Rentabilidad por Plazos para Cada ETF")
            rentabilidad_por_plazos = []
            plazos_rentabilidad = ["1 Mes", "3 Meses", "6 Meses", "1 Año", "YTD", "3 Años", "5 Años", "10 Años"]

            for etf_nombre in etfs_seleccionados:
                ticker = etfs[etf_nombre]
                etf = yf.Ticker(ticker)
                historial = etf.history(period="10y").resample('M').last()  # Resamplear por mes
                rentabilidades = []
                for plazo in [1, 3, 6, 12, "YTD", 36, 60, 120]:
                    if plazo == "YTD":
                        ytd_start = historial[historial.index.year == pd.Timestamp.now().year].iloc[0]['Close']
                        ytd_end = historial['Close'].iloc[-1]
                        rent_ytd = (ytd_end - ytd_start) / ytd_start * 100
                        rentabilidades.append(f"{rent_ytd:.2f}%")
                    else:
                        start_price = historial['Close'].iloc[-plazo] if plazo < len(historial) else np.nan
                        end_price = historial['Close'].iloc[-1]
                        if not np.isnan(start_price):
                            rent = (end_price - start_price) / start_price * 100
                            rentabilidades.append(f"{rent:.2f}%")
                        else:
                            rentabilidades.append("N/A")
                rentabilidad_por_plazos.append([etf_nombre] + rentabilidades)

            # Mostrar la tabla de rentabilidad por plazos
            df_rentabilidad_plazos = pd.DataFrame(rentabilidad_por_plazos, columns=["ETF"] + plazos_rentabilidad)
            st.table(df_rentabilidad_plazos)

        # Pestaña de Resultados de la Inversión
        with tabs[2]:
            st.header("Resumen de la Inversión 🧾")

             # Simulación de la inversión combinada con aportaciones adicionales
            saldo_final = aportacion_inicial
            aportaciones_totales = aportacion_inicial
            rendimiento_mensual_ponderado = (1 + rendimiento_portafolio_ponderado) ** (1/12) - 1

            
            # Mostrar la tasa de rentabilidad proyectada
            st.subheader("Tasa de Rentabilidad Proyectada del Portafolio")
            st.write(f"La tasa de rentabilidad anualizada proyectada del portafolio es: **{rendimiento_portafolio_ponderado * 100:.2f}%**")

            # Advertencia de proyección
            st.warning("⚠️ Nota: El capital estimado es solo una proyección basada en datos históricos y no garantiza rendimientos futuros. El mercado puede ser volátil, y las inversiones están sujetas a riesgos.")

            # Resumen de aportaciones incluyendo el capital inicial
            st.subheader("Resumen de Aportaciones")
            st.write(f"Capital Inicial: **${aportacion_inicial:,.2f} MXN**")
            
            # Inicializar total de aportaciones solo con el capital inicial
            total_aportado = aportacion_inicial
            for anio, monto in zip(anios_aportacion, montos_aportacion):
                st.write(f"Año {anio}: ${monto:,.2f} MXN")
                total_aportado += monto

            st.write(f"Total aportado: **${total_aportado:,.2f} MXN**")
            

            # Composición del portafolio
            st.subheader("Composición del Portafolio")
            for etf_nombre, peso in zip(etfs_seleccionados, pesos_normalizados):
                st.write(f"- {etf_nombre}: {peso * 100:.2f}%")


            #PRUEBAAAA
            # Configuración de la gráfica de Evolución del Capital Acumulado con anotaciones
            fig = go.Figure()

            # Iniciar el saldo con la aportación inicial
            saldo_final = aportacion_inicial
            capital_acumulado = [saldo_final]  # Lista para almacenar el capital acumulado mes a mes
            fechas = pd.date_range(start=pd.Timestamp.today(), periods=horizonte_inversion * 12 + 1, freq='M')

            # Cálculo del crecimiento simple
            for mes in range(1, horizonte_inversion * 12 + 1):
                # Convertir el número de meses al año actual
                anio_actual = (mes - 1) // 12 + 1
                
                # Verificar si este mes corresponde a un año en el que hay una aportación adicional
                if anio_actual in anios_aportacion:
                    indice = anios_aportacion.index(anio_actual)
                    # Sumar la aportación adicional en el año especificado
                    saldo_final += montos_aportacion[indice]
                
                # Calcular crecimiento simple (sin interés compuesto)
                saldo_final += aportacion_inicial * rendimiento_mensual_ponderado

                # Guardar el saldo actual en la lista de capital acumulado
                capital_acumulado.append(saldo_final)

            # Línea de evolución del capital acumulado en la gráfica
            fig.add_trace(go.Scatter(
                x=fechas,
                y=capital_acumulado,
                mode='lines',
                line=dict(color='royalblue', width=2),
                name="Capital Acumulado",
                hovertemplate="Fecha: %{x}<br>Capital: %{y:,.2f} MXN"
            ))

            # Añadir marcadores y anotaciones para cada aportación adicional
            for i, anio in enumerate(anios_aportacion):
                fecha_aportacion = fechas[anio * 12]  # Mes correspondiente al año de la aportación adicional
                monto = montos_aportacion[i]
                
                # Añadir marcador de aportación en la gráfica
                fig.add_trace(go.Scatter(
                    x=[fecha_aportacion],
                    y=[capital_acumulado[anio * 12]],
                    mode='markers+text',
                    marker=dict(size=10, color='red', symbol='circle'),
                    text=[f"+${monto:,.2f}"],
                    textposition="top center",
                    name=f"Aportación Año {anio}"
                ))

            # Anotación del capital final al final de la gráfica
            fig.add_annotation(
                x=fechas[-1],
                y=capital_acumulado[-1],
                text=f"Capital Final: ${capital_acumulado[-1]:,.2f} MXN",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
                font=dict(size=12, color="green"),
                bordercolor="green",
                borderwidth=2,
                borderpad=4,
                bgcolor="white",
                opacity=0.8
            )

            # Configuración de la gráfica
            fig.update_layout(
                title="Evolución del Capital Acumulado con Aportaciones Adicionales",
                xaxis_title="Fecha",
                yaxis_title="Capital Acumulado (MXN)",
                template="plotly_white",
                hovermode="x unified"
            )

            # Mostrar la gráfica en Streamlit
            st.plotly_chart(fig)

            # También mostrar el capital final estimado en Streamlit
            st.subheader("Capital Final Estimado")
            st.write(f"Al final del horizonte de inversión de {horizonte_inversion} años, el capital estimado es de: **${capital_acumulado[-1]:,.2f} MXN**.")

                        
