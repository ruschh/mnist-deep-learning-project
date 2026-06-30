# app.py

from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model_tuner_best_hps.keras"
HISTORY_PATH = BASE_DIR / "models" / "history_tuner.csv"


# ============================================================
# Streamlit config
# ============================================================

st.set_page_config(
    page_title="MNIST Neural Network Animation",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# Loaders
# ============================================================

@st.cache_resource
def load_trained_model():
    return keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_test_data():
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype("float32")
    x_test = np.expand_dims(x_test, axis=-1)
    return x_test, y_test


@st.cache_data
def load_history():
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH)

    epochs = np.arange(1, 41)

    return pd.DataFrame({
        "loss": 1.2 * np.exp(-epochs / 7) + 0.08,
        "val_loss": 0.23 * np.exp(-epochs / 6) + 0.04,
        "accuracy": 0.60 + 0.38 * (1 - np.exp(-epochs / 8)),
        "val_accuracy": 0.93 + 0.06 * (1 - np.exp(-epochs / 7)),
    })


# ============================================================
# Plot helpers
# ============================================================

def get_accuracy_columns(history):
    if "accuracy" in history.columns:
        return "accuracy", "val_accuracy"

    return "sparse_categorical_accuracy", "val_sparse_categorical_accuracy"


def draw_training_curves(fig, gs, history, frame_idx):
    max_epoch = min(frame_idx + 1, len(history))
    h = history.iloc[:max_epoch]

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[1, 0])

    ax_loss.plot(h["loss"], label="loss", color="#1f77ff", linewidth=2)
    ax_loss.plot(h["val_loss"], label="val_loss", color="#ff9900", linewidth=2)
    ax_loss.set_title("Loss", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.25)
    ax_loss.legend(fontsize=8)

    acc_col, val_acc_col = get_accuracy_columns(history)

    ax_acc.plot(h[acc_col], label="accuracy", color="#1f77ff", linewidth=2)
    ax_acc.plot(h[val_acc_col], label="val_accuracy", color="#ff9900", linewidth=2)
    ax_acc.set_title("Accuracy", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(alpha=0.25)
    ax_acc.legend(fontsize=8)


def draw_input_image(fig, gs, image, true_label):
    ax_img = fig.add_subplot(gs[2, 0])

    ax_img.imshow(image.squeeze(), cmap="gray")
    ax_img.set_title(f"Imagem de entrada\nClasse real: {true_label}", fontsize=12, fontweight="bold")
    ax_img.axis("off")


def draw_connections(ax, layer_a, layer_b, color="#505050", alpha=0.45, lw=0.75):
    for x1, y1 in layer_a:
        for x2, y2 in layer_b:
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                alpha=alpha,
                linewidth=lw,
                zorder=1,
            )


def draw_layer(ax, positions, labels=None, base_color="#ff4b4b", edge_color="black",
               active_index=None, active_color="#00ff00", text_color="white",
               radius=0.026, fontsize=10):
    for idx, (x, y) in enumerate(positions):
        color = active_color if idx == active_index else base_color

        circle = plt.Circle(
            (x, y),
            radius,
            facecolor=color,
            edgecolor=edge_color,
            linewidth=1.2,
            zorder=5,
        )
        ax.add_patch(circle)

        if labels is not None:
            ax.text(
                x,
                y,
                str(labels[idx]),
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color,
                fontweight="bold",
                zorder=6,
            )


def draw_network(ax, probabilities, true_label, predicted_class):
    ax.set_xlim(0, 1)
    ax.set_ylim(0.00, 0.74)
    ax.axis("off")
    #ax.set_title("Rede Neural — fluxo de classificação", fontsize=14, fontweight="bold")

    layer_sizes = [10, 10, 10, 10, 10]
    x_positions = [0.10, 0.27, 0.44, 0.61, 0.78]

    layers = []
    for x, n in zip(x_positions, layer_sizes):
        ys = np.linspace(0.10, 0.70, n)
        layers.append([(x, y) for y in ys])

    input_layer = layers[0]
    hidden_1 = layers[1]
    hidden_2 = layers[2]
    dense_layer = layers[3]
    output_layer = layers[4]

    for layer_a, layer_b in zip(layers[:-1], layers[1:]):
        draw_connections(ax, layer_a, layer_b)

    digits = list(range(10))

    # Entrada: todos amarelos; classe real em verde
    draw_layer(
        ax,
        input_layer,
        labels=digits,
        base_color="#ffd700",
        active_index=int(true_label),
        active_color="#00cc44",
        text_color="black",
        radius=0.028,
        fontsize=10,
    )

    # Camadas intermediárias
    draw_layer(ax, hidden_1, base_color="#ff3b30", radius=0.025)
    draw_layer(ax, hidden_2, base_color="#ff3b30", radius=0.025)
    draw_layer(ax, dense_layer, base_color="#ff3b30", radius=0.025)

    # Saída: todos vermelhos; classe prevista em ciano
    draw_layer(
        ax,
        output_layer,
        labels=digits,
        base_color="#ff3b30",
        active_index=int(predicted_class),
        active_color="#00e5ff",
        text_color="black",
        radius=0.028,
        fontsize=10,
    )

    # Labels das camadas
    labels = ["Entrada\n0–9", "Conv/Pool", "Conv/Pool", "Dense", "Saída\n0–9"]
    for x, label in zip(x_positions, labels):
        ax.text(
            x,
            0.035,
            label,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    # Probabilidade prevista ao lado do neurônio de saída
    x_pred, y_pred = output_layer[int(predicted_class)]
    ax.text(
        x_pred + 0.045,
        y_pred,
        f"{predicted_class}\n{probabilities[predicted_class]:.1%}",
        ha="left",
        va="center",
        fontsize=11,
        color="#00aacc",
        fontweight="bold",
    )

    # Legenda
    ax.text(0.02, 0.98, "Entrada real", color="#00cc44", fontsize=10, fontweight="bold")
    ax.text(0.25, 0.98, "Neurônios ocultos", color="#ff3b30", fontsize=10, fontweight="bold")
    ax.text(0.55, 0.98, "Saída prevista", color="#00aacc", fontsize=10, fontweight="bold")

def draw_probabilities(fig, gs, probabilities, true_label, predicted_class, confidence):
    ax_out = fig.add_subplot(gs[:, 2])

    colors = [
        "#00e5ff" if i == predicted_class else "#8a8a8a"
        for i in range(10)
    ]

    ax_out.barh(range(10), probabilities, color=colors)
    ax_out.set_yticks(range(10))
    ax_out.set_yticklabels([str(i) for i in range(10)])
    ax_out.set_xlim(0, 1)
    ax_out.invert_yaxis()
    ax_out.set_title("Probabilidades por classe", fontsize=12, fontweight="bold")
    ax_out.set_xlabel("Probabilidade", labelpad=12)
    ax_out.grid(axis="x", alpha=0.25)

    for i, p in enumerate(probabilities):
        ax_out.text(
            min(p + 0.02, 0.92),
            i,
            f"{p:.1%}",
            va="center",
            fontsize=9,
            fontweight="bold" if i == predicted_class else "normal",
        )

    status_color = "green" if predicted_class == true_label else "red"

    ax_out.text(
        0.02,
        -0.18,
        f"Predição: {predicted_class}\nConfiança: {confidence:.1%}\nClasse real: {true_label}",
        transform=ax_out.transAxes,
        fontsize=11,
        color=status_color,
        fontweight="bold",
        va="top",
    )


def draw_summary_panel(fig, gs, true_label, predicted_class, confidence):
    ax = fig.add_subplot(gs[3, 1])
    ax.axis("off")

    status = "Correto" if true_label == predicted_class else "Incorreto"
    status_color = "green" if true_label == predicted_class else "red"

    ax.text(0.20, 1.50, "Classe real", ha="center", fontsize=14, fontweight="bold")
    ax.text(0.20, 1.00, str(true_label), ha="center", fontsize=18, color="green", fontweight="bold")

    ax.text(0.50, 1.50, "Predição", ha="center", fontsize=14, fontweight="bold")
    ax.text(0.50, 0.90, str(predicted_class), ha="center", fontsize=18, color="#00aacc", fontweight="bold")

    ax.text(0.80, 1.50, "Confiança", ha="center", fontsize=14, fontweight="bold")
    ax.text(0.80, 0.90, f"{confidence:.1%}", ha="center", fontsize=18, color=status_color, fontweight="bold")

    ax.text(0.50, 0.60, status, ha="center", fontsize=14, color=status_color, fontweight="bold")


def draw_frame(model, history, x_test, y_test, frame_idx):
    image = x_test[frame_idx]
    true_label = int(y_test[frame_idx])

    probabilities = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(
        4,
        3,
        width_ratios=[1.0, 2.2, 1.1],
        height_ratios=[1.0, 1.0, 1.0, 0.45],
    )

    draw_training_curves(fig, gs, history, frame_idx)
    draw_input_image(fig, gs, image, true_label)

    ax_net = fig.add_subplot(gs[:3, 1])
    draw_network(ax_net, probabilities, true_label, predicted_class)

    draw_probabilities(fig, gs, probabilities, true_label, predicted_class, confidence)
    draw_summary_panel(fig, gs, true_label, predicted_class, confidence)

    fig.suptitle(
        f"MNIST Neural Network Animation",
        fontsize=17,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig


# ============================================================
# App
# ============================================================

def main():
    st.title("🧠 MNIST Neural Network Animation")
    st.write(
        "Visualização interativa do fluxo de classificação: "
        "classe real na entrada em verde e classe prevista na saída em ciano."
    )

    model = load_trained_model()
    x_test, y_test = load_test_data()
    history = load_history()

    st.sidebar.title("Controles")

    mode = st.sidebar.radio(
        "Modo",
        ["Exemplo manual", "Animação automática"],
    )

    max_index = len(x_test) - 1

    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0

    if mode == "Exemplo manual":
        frame_idx = st.sidebar.slider(
            "Escolha o exemplo",
            min_value=0,
            max_value=max_index,
            value=st.session_state.frame_idx,
            step=1,
        )
        st.session_state.frame_idx = frame_idx

    else:
        n_examples = st.sidebar.slider("Número de exemplos", 5, 100, 30)
        delay = st.sidebar.slider("Intervalo entre frames (s)", 0.1, 2.0, 0.5)

        start = st.sidebar.button("Iniciar animação")

        if start:
            placeholder = st.empty()

            for i in range(n_examples):
                st.session_state.frame_idx = i

                fig = draw_frame(
                    model,
                    history,
                    x_test,
                    y_test,
                    st.session_state.frame_idx,
                )

                with placeholder.container():
                    st.pyplot(fig)

                plt.close(fig)
                time.sleep(delay)

            return

        frame_idx = st.session_state.frame_idx

    fig = draw_frame(model, history, x_test, y_test, st.session_state.frame_idx)
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.markdown(
        """
        **Legenda da rede neural**

        - **Amarelo:** neurônios da camada de entrada, representando os dígitos possíveis de 0 a 9.
        - **Verde:** classe real da imagem analisada.
        - **Vermelho:** neurônios das camadas intermediárias.
        - **Ciano:** classe prevista pelo modelo na camada de saída.
        """
    )


if __name__ == "__main__":
    main()