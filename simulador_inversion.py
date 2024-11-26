import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from googletrans import Translator
import sqlite3
import bcrypt
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Configuración de la página de Streamlit
st.set_page_config(page_title="Simulador Allianz OptiMaxx", layout="wide")

#botones y fondo
st.markdown("""
    <style>
    .stButton>button {
        background-color: #b3b7b8; /* Color de fondo azul */
        color: white; /* Color del texto */
        font-size: 16px; /* Tamaño del texto */
        border-radius: 10px; /* Bordes redondeados */
        padding: 10px 20px; /* Espaciado interno */
        border: none; /* Sin bordes */
    }
    .stButton>button:hover {
        background-color: #686564; /* Color de fondo al pasar el mouse */
        color: white;
    }

    .stButton>button:active {
        background-color: #b3b7b8; /* Color del fondo al presionar */
        color: #FFFFFF; /* Mismo color para evitar que cambie */
        box-shadow: inset 0px 3px 5px rgba(0,0,0,0.2); /* Efecto de presión */
    }
    </style>
    """, unsafe_allow_html=True)
#tablas
st.markdown(
    """
    <style>
    table {
        font-size: 16px;
        color: white;
        border-collapse: collapse;
        width: 100%;
    }
    th, td {
        border: 1px solid #dddddd;
        text-align: left;
        padding: 8px;
    }
    th {
        background-color: #b3b7b8;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)
#letra 
st.markdown(
    """
    <style>
    h1 {
        color: #1a188a;
        text-align: center;
        font-family: 'Arial Black', sans-serif;
        font-size: 40px;
    }
    h2 {
        color: #4260ba;
        text-align: left;
        font-family: 'Arial', sans-serif;
        font-size: 30px;
    }
    h3 {
        color: #4a4a4b;
        text-align: left;
        font-family: 'Verdana', sans-serif;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)



# Configuración de la base de datos
def create_user_table():
    conn = sqlite3.connect('usuarios.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nombre TEXT,
                    correo TEXT UNIQUE,
                    telefono TEXT,
                    password TEXT
                 )''')
    conn.commit()
    conn.close()

def add_user(nombre, correo, telefono, password):
    conn = sqlite3.connect('usuarios.db')
    c = conn.cursor()
    c.execute('INSERT INTO users (nombre, correo, telefono, password) VALUES (?, ?, ?, ?)',
              (nombre, correo, telefono, password))
    conn.commit()
    conn.close()

def authenticate_user(correo, password):
    conn = sqlite3.connect('usuarios.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE correo = ?', (correo,))
    data = c.fetchone()
    conn.close()
    if data:
        return bcrypt.checkpw(password.encode(), data[0].encode())
    return False

def get_user_by_email(correo):
    conn = sqlite3.connect('usuarios.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE correo = ?', (correo,))
    user = c.fetchone()
    conn.close()
    return user

# Crear tabla de usuarios al inicio
create_user_table()

HUGGINGFACE_API_KEY= st.secrets["HUGGINGFACE_API_KEY"]
API_URL = "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
def query_huggingface(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

@st.cache_resource
def cargar_modelo():
    modelo = "microsoft/DialoGPT-medium"  # Puedes cambiar a "small" o "large" según prefieras
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    model = AutoModelForCausalLM.from_pretrained(modelo)
    return tokenizer, model

# Función para generar una respuesta del chatbot
def responder_chatbot(input_text, chat_history_ids, tokenizer, model):
    # Tokenizar la entrada del usuario
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")

    # Concatenar el historial de chat con la nueva entrada del usuario
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Generar respuesta del modelo
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decodificar la respuesta generada
    respuesta = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return respuesta, chat_history_ids

def interfaz_chatbot():
    st.title("🤖 Chatbot - Pregúntame sobre el simulador o inversiones")
    
    tokenizer, model = cargar_modelo()
     
    
    # Variables de sesión para guardar el historial del chat
    if "chat_history_ids" not in st.session_state:
        st.session_state.chat_history_ids = None
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    # Entrada del usuario
    input_text = st.text_input("Escribe tu pregunta aquí:")
    if st.button("Enviar") and input_text:
        with st.spinner("Pensando..."):
            respuesta, chat_history_ids = responder_chatbot(input_text, st.session_state.chat_history_ids, tokenizer, model)
            st.session_state.chat_history_ids = chat_history_ids
            st.session_state.chat_log.append((input_text, respuesta))

    # Mostrar historial del chat
    st.subheader("Historial de conversación:")
    for pregunta, respuesta in st.session_state.chat_log:
        st.markdown(f"**Tú:** {pregunta}")
        st.markdown(f"**Chatbot:** {respuesta}")


def create_simulations_table():
    conn = sqlite3.connect('usuarios.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS simulaciones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    nombre_simulacion TEXT,
                    etfs TEXT,
                    aportacion_inicial REAL,
                    rendimiento_proyectado REAL,
                    capital_final REAL,
                    fecha_simulacion TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                 )''')
    conn.commit()
    conn.close()
create_simulations_table()
def guardar_simulacion(user_id, nombre_simulacion, etfs, aportacion_inicial, rendimiento_proyectado, capital_final):
    conn = sqlite3.connect('usuarios.db')
    c = conn.cursor()
    c.execute('INSERT INTO simulaciones (user_id, nombre_simulacion, etfs, aportacion_inicial, rendimiento_proyectado, capital_final, fecha_simulacion) VALUES (?, ?, ?, ?, ?, ?, ?)',
              (user_id, nombre_simulacion, ', '.join(etfs), aportacion_inicial, rendimiento_proyectado, capital_final, pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

def get_simulation_name (): st.text_input("Nombre de la simulación", placeholder="Ingresa un nombre para esta simulación")
    
def simulador():
    st.title("Simulador Allianz OptiMaxx 🚀")
    primer_nombre = st.session_state.user[1].split()[0]
    st.subheader(f"{primer_nombre}, comencemos a invertir!")

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
        "AZ LATIXX MEX CETETRAC": "MXX",
        "AZ QQQ NASDAQ 100": "QQQ",
        "AZ MSCI ASIA EX-JAPAN": "AAXJ",
        "AZ LATIXX MEX M10TRAC": "M10.MX",
        "AZ BARCLAYS 1-3 YEAR TR": "SHY",
        "AZ MSCI ACWI INDEX FUND": "ACWI",
        "AZ LATIXX MEXICO M5TRAC": "M5TRAC.MX",
        "AZ SILVER TRUST": "SLV",
        "AZ MSCI HONG KONG INDEX": "EWH",
        "AZ LATIXX MEX UDITRAC": "UDITRAC.MX",
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
        horizonte_inversion = st.slider("Horizonte de Inversión (años)", min_value=5, max_value=30, value=10)
        
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
                # Advertencia de proyección
                st.warning("⚠️ Nota: El capital estimado es solo una proyección basada en datos históricos y no garantiza rendimientos futuros. El mercado puede ser volátil, y las inversiones están sujetas a riesgos.")


def mostrar_simulaciones(user_id):
    conn = sqlite3.connect('usuarios.db')
    c = conn.cursor()
    c.execute('SELECT nombre_simulacion, etfs, aportacion_inicial, rendimiento_proyectado, capital_final, fecha_simulacion FROM simulaciones WHERE user_id = ?', (user_id,))
    simulaciones = c.fetchall()
    conn.close()

    if simulaciones:
        st.subheader("Tus Simulaciones Guardadas")
        for simulacion in simulaciones:
            st.markdown(f"""
            **Nombre:** {simulacion[0]}  
            **ETFs:** {simulacion[1]}  
            **Aportación Inicial:** ${simulacion[2]:,.2f}  
            **Rendimiento Proyectado:** {simulacion[3]:.2f}%  
            **Capital Final:** ${simulacion[4]:,.2f}  
            **Fecha:** {simulacion[5]}  
            ---  
            """)
    else:
        st.info("No tienes simulaciones guardadas.")



# Pantallas de la aplicación
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.page = "Inicia sesión"

    if st.session_state.logged_in:
        st.session_state.page = "Simulador"

    if not st.session_state.logged_in:
        menu = ["Inicio de Sesión", "Registro"]
        choice = st.sidebar.selectbox("Menú", menu)
    else:
        menu = ["Simulador", "Asistente Virtual",]
        choice = st.sidebar.selectbox("Menú", menu)


    if st.session_state.logged_in:
        with st.sidebar:
            st.write(f"👋 **Bienvenido, {st.session_state.user[1]}**")  # Mostrar nombre del usuario

            if st.button("Cerrar Sesión"):
                st.session_state.logged_in = False
                st.session_state.user = None
                st.experimental_rerun()

    if choice == "Inicio de Sesión":
        st.subheader("Inicia Sesión")
        correo = st.text_input("Correo Electrónico")
        password = st.text_input("Contraseña", type="password")
        if st.button("Iniciar Sesión"):
            if authenticate_user(correo, password):
                st.success("Inicio de sesión exitoso")
                st.session_state.logged_in = True
                st.session_state.user = get_user_by_email(correo)
            else:
                st.error("Correo o contraseña incorrectos")

    elif choice == "Registro":
        st.subheader("Regístrate")
        nombre = st.text_input("Nombre Completo")
        correo = st.text_input("Correo Electrónico")
        telefono = st.text_input("Número de Teléfono")
        password = st.text_input("Contraseña", type="password")
        confirmar_password = st.text_input("Confirmar Contraseña", type="password")

        if st.button("Registrar"):
            if password != confirmar_password:
                st.error("Las contraseñas no coinciden")
            else:
                hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                try:
                    add_user(nombre, correo, telefono, hashed_password)
                    st.success("¡Registro exitoso! Ahora puedes iniciar sesión.")
                except sqlite3.IntegrityError:
                    st.error("El correo electrónico ya está registrado.")

    elif choice == "Simulador": 
        if st.session_state.logged_in:
              # Mostrar nombre del usuario
            # Aquí puedes llamar la función de tu simulador
            simulador()
        else:
            st.warning("Por favor, inicia sesión para acceder al simulador.")
    
    elif choice == "Asistente Virtual":
        st.subheader("Asistente Virtual 🤖")
        st.write("Haz preguntas sobre finanzas, ETFs o cualquier duda que tengas sobre el simulador.")
        user_question = st.text_input("Escribe tu pregunta aquí:")

        if st.button("Enviar"):
            if user_question:
                with st.spinner("Pensando..."):
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",  # Modelo GPT
                            messages=[
                                {"role": "system", "content": "Eres un asistente experto en finanzas y simuladores."},
                                {"role": "user", "content": user_question}
                            ]
                        )
                        respuesta = response['choices'][0]['message']['content']
                        st.write("### Respuesta:")
                        st.write(respuesta)
                    except Exception as e:
                        st.error(f"Error al conectar con el asistente virtual: {str(e)}")
            else:
                st.warning("Por favor, escribe una pregunta antes de enviar.")



# Ejecutar la aplicación
if __name__ == "__main__":
    main()
