from graphviz import Digraph

def plot_perceptron():
    dot = Digraph(format='png')
    dot.attr(rankdir='LR', size='8,5')

    # Capa de entrada (6 neuronas)
    with dot.subgraph(name='cluster_input') as c:
        c.attr(style='filled', color='lightgrey')
        for i in range(6):
            c.node(f'X{i+1}', label=f'X{i+1}', shape='circle')
        c.attr(label='Capa de Entrada')

    # Capa oculta 1 (64 neuronas - simplificada a 5 para visualización)
    with dot.subgraph(name='cluster_hidden1') as c:
        c.attr(style='filled', color='lightblue')
        for i in range(5):
            c.node(f'H1_{i+1}', label=f'H1-{i+1}', shape='circle')
        c.attr(label='Capa Oculta 1')

    # Capa oculta 2 (64 neuronas - simplificada a 5 para visualización)
    with dot.subgraph(name='cluster_hidden2') as c:
        c.attr(style='filled', color='lightblue')
        for i in range(5):
            c.node(f'H2_{i+1}', label=f'H2-{i+1}', shape='circle')
        c.attr(label='Capa Oculta 2')

    # Capa de salida
    dot.node('Y', label='Precio', shape='doublecircle', color='orange')

    # Conexiones (simplificadas)
    for i in range(6):
        for j in range(5):
            dot.edge(f'X{i+1}', f'H1_{j+1}')
    for i in range(5):
        for j in range(5):
            dot.edge(f'H1_{i+1}', f'H2_{j+1}')
    for i in range(5):
        dot.edge(f'H2_{i+1}', 'Y')

    return dot

# En la sección de Entrenamiento
st.subheader("🧠 Estructura del Perceptrón (Red Neuronal)")
st.graphviz_chart(plot_perceptron())
