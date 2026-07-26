"""
Generates high-resolution PNG diagrams for:
1. Tiered Fallback Architecture Overview
2. Stateful Sequence Rolling Pipeline
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
artifact_dir = r"C:\Users\Dell\.gemini\antigravity-ide\brain\143a790d-3565-4607-8509-bc79e619c7b3"
has_artifact_dir = os.path.exists(artifact_dir)


def draw_fallback_architecture():
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")
    ax.axis("off")

    # Title
    ax.text(
        0.5, 0.94, "AEGIS.AI — Tiered Fallback Architecture Overview",
        color="#f8fafc", fontsize=16, fontweight="bold", ha="center"
    )
    ax.text(
        0.5, 0.89, "Fail-Safe Decoupling, Cold-Start Peer Grouping, Load Shedding & Circuit-Breaker Controls",
        color="#94a3b8", fontsize=11, ha="center"
    )

    # Boxes
    boxes = [
        {"rect": (0.35, 0.74, 0.30, 0.08), "text": "1. Inbound Telemetry Stream Log", "color": "#1e293b", "border": "#38bdf8"},
        {"rect": (0.05, 0.52, 0.26, 0.12), "text": "Level 2: Peer-Group Router\n(Cold-Start Resolution)", "color": "#312e81", "border": "#818cf8"},
        {"rect": (0.37, 0.52, 0.26, 0.12), "text": "Neural Inference Engine\n(Bi-LSTM + PyTorch GNN)", "color": "#064e3b", "border": "#34d399"},
        {"rect": (0.69, 0.52, 0.26, 0.12), "text": "Level 1: Rule Engine\n(Fail-Safe Decoupling)", "color": "#701a75", "border": "#f472b6"},
        {"rect": (0.21, 0.26, 0.26, 0.12), "text": "Level 3: Load Shedding\n(Queue > 5000: Drop SHAP)", "color": "#7c2d12", "border": "#fb923c"},
        {"rect": (0.53, 0.26, 0.26, 0.12), "text": "Level 4: Circuit Breaker\n(ADWIN Retraining Cooldown)", "color": "#831843", "border": "#f43f5e"},
        {"rect": (0.37, 0.04, 0.26, 0.10), "text": "Analyst Dashboard / WebSocket Stream", "color": "#0f172a", "border": "#38bdf8"}
    ]

    for b in boxes:
        x, y, w, h = b["rect"]
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02",
            facecolor=b["color"], edgecolor=b["border"], linewidth=2.0
        )
        ax.add_patch(rect)
        ax.text(
            x + w/2, y + h/2, b["text"],
            color="#ffffff", fontsize=10, fontweight="bold", ha="center", va="center"
        )

    # Arrows
    arrows = [
        # Inbound to Cold Start / Neural
        ((0.40, 0.74), (0.18, 0.64)),
        ((0.50, 0.74), (0.50, 0.64)),
        # Neural to Fail safe
        ((0.63, 0.58), (0.69, 0.58)),
        # Neural to Load Shedding & Circuit breaker
        ((0.45, 0.52), (0.34, 0.38)),
        ((0.55, 0.52), (0.66, 0.38)),
        # Output arrows to Dashboard
        ((0.34, 0.26), (0.45, 0.14)),
        ((0.66, 0.26), (0.55, 0.14))
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.0, mutation_scale=15)
        )

    plt.tight_layout()
    out1 = os.path.join(ASSETS_DIR, "tiered_fallback_architecture.png")
    fig.savefig(out1, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    if has_artifact_dir:
        fig.savefig(os.path.join(artifact_dir, "tiered_fallback_architecture.png"), dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f" -> Saved {out1}")


def draw_stateful_pipeline():
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")
    ax.axis("off")

    ax.text(
        0.5, 0.93, "AEGIS.AI — Stateful Sequence Rolling Pipeline",
        color="#f8fafc", fontsize=16, fontweight="bold", ha="center"
    )
    ax.text(
        0.5, 0.87, "Real-Time Log Ingestion, Haversine Velocity Tracking & Thread-Safe Ring Buffer Rolling",
        color="#94a3b8", fontsize=11, ha="center"
    )

    boxes = [
        {"rect": (0.04, 0.45, 0.16, 0.22), "text": "Incoming\nTelemetry Log", "color": "#1e293b", "border": "#38bdf8"},
        {"rect": (0.23, 0.45, 0.16, 0.22), "text": "Category Hash\n& Mapping\n(Auth, Type)", "color": "#312e81", "border": "#818cf8"},
        {"rect": (0.42, 0.45, 0.16, 0.22), "text": "Haversine Geo\nVelocity Engine\n(Impossible Travel)", "color": "#064e3b", "border": "#34d399"},
        {"rect": (0.61, 0.45, 0.16, 0.22), "text": "Thread-Safe\nRing Buffer\n(deque maxlen=10)", "color": "#701a75", "border": "#f472b6"},
        {"rect": (0.80, 0.45, 0.16, 0.22), "text": "Matrix Frame\n[10, 6] Tensor\n(Zero-Padded)", "color": "#7c2d12", "border": "#fb923c"}
    ]

    for b in boxes:
        x, y, w, h = b["rect"]
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02",
            facecolor=b["color"], edgecolor=b["border"], linewidth=2.0
        )
        ax.add_patch(rect)
        ax.text(
            x + w/2, y + h/2, b["text"],
            color="#ffffff", fontsize=10, fontweight="bold", ha="center", va="center"
        )

    for i in range(len(boxes) - 1):
        x1 = boxes[i]["rect"][0] + boxes[i]["rect"][2]
        y1 = boxes[i]["rect"][1] + boxes[i]["rect"][3] / 2
        x2 = boxes[i+1]["rect"][0]
        y2 = y1
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.5, mutation_scale=18)
        )

    plt.tight_layout()
    out2 = os.path.join(ASSETS_DIR, "stateful_sequence_pipeline.png")
    fig.savefig(out2, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    if has_artifact_dir:
        fig.savefig(os.path.join(artifact_dir, "stateful_sequence_pipeline.png"), dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f" -> Saved {out2}")


if __name__ == "__main__":
    draw_fallback_architecture()
    draw_stateful_pipeline()
