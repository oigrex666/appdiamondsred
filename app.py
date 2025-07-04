import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar dataset (global)
@st.cache_data
def load_data():
    return sns.load_dataset('diamonds')

df = load_data()

# Variables y preparación global
features = ['carat', 'depth', 'table', 'x', 'y', 'z']
X = df[features]
y = df[['price']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Menú de navegación
st.sidebar.title("Menú de Navegación")
page = st.sidebar.radio("Selecciona la página:", ("Exploración", "Entrenamiento", "Predicción", "Arquitectura del Modelo"))

# Página 1: Exploración
if page == "Exploración":
    st.title("🔍 Exploración del Dataset de Diamantes")

    with st.expander("Vista previa del Dataset"):
        st.dataframe(df.head())
        st.write(f"Forma del dataset: {df.shape}")
        st.write(df.describe())

    st.subheader("Dispersión Carat vs Price por Claridad")
    clarity_ranking = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'IF']
    fig, ax = plt.subplots()
    sns.scatterplot(x='carat', y='price',
                    hue='clarity', size='depth',
                    palette='ch:r=-.2,d=.3_r',
                    hue_order=clarity_ranking,
                    sizes=(1, 8), linewidth=0,
                    data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Distribuciones de Variables Numéricas")
    numeric_vars = df.select_dtypes(include=['float64', 'int64'])
    for col in numeric_vars.columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, bins=30, color='blue', ax=ax)
        ax.axvline(df[col].mean(), color='red', linestyle='--', label='Media')
        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frecuencia')
        ax.legend()
        st.pyplot(fig)
        st.markdown(f"""
        **Estadísticas de `{col}`**  
        - Media: {df[col].mean():.2f}  
        - Desviación estándar: {df[col].std():.2f}  
        - Mínimo: {df[col].min():.2f}  
        - Máximo: {df[col].max():.2f}  
        """)

# Página 2: Entrenamiento
elif page == "Entrenamiento":
    st.title("🤖 Entrenamiento de la Red Neuronal")

    epochs = st.slider("Selecciona el número de épocas para entrenar:", min_value=1, max_value=20, value=5, step=1)

    model = create_model()

    with st.spinner(f"Entrenando por {epochs} épocas..."):
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=0, validation_split=0.1)

    loss, mae = model.evaluate(x_test, y_test, verbose=0)
    pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    st.success("✅ Modelo entrenado correctamente.")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")

    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(history.history['loss'], label='Pérdida entrenamiento')
    ax_loss.plot(history.history['val_loss'], label='Pérdida validación')
    ax_loss.set_xlabel('Época')
    ax_loss.set_ylabel('Loss (MSE)')
    ax_loss.legend()
    ax_loss.set_title("Curva de pérdida durante el entrenamiento")
    st.pyplot(fig_loss)

    st.subheader("Precio Real vs Predicho")
    fig_pred, ax_pred = plt.subplots()
    ax_pred.scatter(y_test, pred, alpha=0.3)
    ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax_pred.set_xlabel('Precio Real')
    ax_pred.set_ylabel('Precio Predicho')
    st.pyplot(fig_pred)

# Página 3: Predicción
elif page == "Predicción":
    st.title("💡 Predicción de Precio Personalizada")

    st.info("Entrenando modelo rápido para predicción (5 epochs)...")
    model = create_model()
    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)

    with st.form("form_prediccion"):
        st.markdown("Introduce las características del diamante:")
        carat = st.number_input("Carat", min_value=0.1, max_value=5.0, value=0.9, step=0.01)
        depth = st.number_input("Depth", min_value=50.0, max_value=70.0, value=61.0, step=0.1)
        table = st.number_input("Table", min_value=50.0, max_value=80.0, value=57.0, step=0.1)
        x = st.number_input("X (mm)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
        y = st.number_input("Y (mm)", min_value=0.0, max_value=15.0, value=5.1, step=0.1)
        z = st.number_input("Z (mm)", min_value=0.0, max_value=10.0, value=3.2, step=0.1)
        submitted = st.form_submit_button("Predecir Precio")

        if submitted:
            input_data = np.array([[carat, depth, table, x, y, z]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0][0]
            st.success(f"💰 Precio estimado del diamante: **${prediction:,.2f}**")

# Página 4: Arquitectura del Modelo
elif page == "Arquitectura del Modelo":
    st.title("🧠 Arquitectura del Perceptrón")

    from graphviz import Digraph

    def plot_perceptron():
        dot = Digraph(format='png')
        dot.attr(rankdir='LR', size='8,5')

        # Capa de entrada
        with dot.subgraph(name='cluster_input') as c:
            c.attr(style='filled', color='lightgrey')
            for i in range(6):
                c.node(f'X{i+1}', label=f'X{i+1}', shape='circle')
            c.attr(label='Capa de Entrada')

        # Capa oculta 1
        with dot.subgraph(name='cluster_hidden1') as c:
            c.attr(style='filled', color='lightblue')
            for i in range(5):
                c.node(f'H1_{i+1}', label=f'H1-{i+1}', shape='circle')
            c.attr(label='Capa Oculta 1')

        # Capa oculta 2
        with dot.subgraph(name='cluster_hidden2') as c:
            c.attr(style='filled', color='lightblue')
            for i in range(5):
                c.node(f'H2_{i+1}', label=f'H2-{i+1}', shape='circle')
            c.attr(label='Capa Oculta 2')

        # Capa de salida
        dot.node('Y', label='Precio', shape='doublecircle', color='orange')

        for i in range(6):
            for j in range(5):
                dot.edge(f'X{i+1}', f'H1_{j+1}')
        for i in range(5):
            for j in range(5):
                dot.edge(f'H1_{i+1}', f'H2_{j+1}')
        for i in range(5):
            dot.edge(f'H2_{i+1}', 'Y')

        return dot

    st.markdown("Esta es una representación visual simplificada de la red neuronal usada para predecir el precio:")
    st.graphviz_chart(plot_perceptron())
    st.markdown("""
    - **Entrada (X1 a X6):** Variables numéricas del diamante (`carat`, `depth`, `table`, `x`, `y`, `z`).
    - **Capas ocultas:** Dos capas densas con activación ReLU (simplificadas para visualización).
    - **Salida:** Predicción del precio (`Y`).
    """)
