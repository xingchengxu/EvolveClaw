"""Evolution visualization: ASCII score curve and optional matplotlib plot."""
from __future__ import annotations
import json
from pathlib import Path


def _load_scores(run_dir: str) -> list[dict]:
    """Load generation records from log.jsonl, skipping error entries."""
    log_path = Path(run_dir) / "log.jsonl"
    records = []
    with open(log_path) as f:
        for line in f:
            record = json.loads(line)
            if "error" not in record:
                records.append(record)
    return records


def ascii_plot(run_dir: str, width: int = 60, height: int = 15) -> str:
    """Render an ASCII plot of score progression over generations."""
    records = _load_scores(run_dir)
    if not records:
        return "No data to plot."

    scores = [r["score"] for r in records]
    gens = [r["generation"] for r in records]

    # Track best-so-far
    best_so_far = []
    current_best = float("-inf")
    for s in scores:
        current_best = max(current_best, s)
        best_so_far.append(current_best)

    min_score = min(min(scores), min(best_so_far))
    max_score = max(max(scores), max(best_so_far))
    score_range = max_score - min_score if max_score != min_score else 1.0

    lines = []
    lines.append(f"Score Evolution (generations {gens[0]}-{gens[-1]})")
    lines.append(f"{'=' * (width + 8)}")

    # Build grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    def _map_x(idx):
        return int(idx / max(1, len(scores) - 1) * (width - 1)) if len(scores) > 1 else 0

    def _map_y(val):
        return height - 1 - int((val - min_score) / score_range * (height - 1))

    # Plot per-generation scores as dots
    for i, s in enumerate(scores):
        x, y = _map_x(i), _map_y(s)
        if 0 <= y < height and 0 <= x < width:
            grid[y][x] = "."

    # Plot best-so-far as line
    for i, b in enumerate(best_so_far):
        x, y = _map_x(i), _map_y(b)
        if 0 <= y < height and 0 <= x < width:
            grid[y][x] = "#"

    # Render with y-axis labels
    for row_idx, row in enumerate(grid):
        val = max_score - (row_idx / max(1, height - 1)) * score_range
        label = f"{val:6.1f} |"
        lines.append(label + "".join(row))

    lines.append(f"       +{'─' * width}")
    gen_start = str(gens[0])
    gen_end = str(gens[-1])
    lines.append(f"        {gen_start}{' ' * (width - len(gen_start) - len(gen_end))}{gen_end}")
    lines.append(f"        . = per-gen score   # = best-so-far")

    return "\n".join(lines)


def matplotlib_plot(run_dir: str, output_path: str | None = None) -> str | None:
    """Generate a matplotlib plot of score history. Returns path to saved image, or None if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    records = _load_scores(run_dir)
    if not records:
        return None

    gens = [r["generation"] for r in records]
    scores = [r["score"] for r in records]

    best_so_far = []
    current_best = float("-inf")
    for s in scores:
        current_best = max(current_best, s)
        best_so_far.append(current_best)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(gens, scores, s=8, alpha=0.4, label="Per-generation", color="steelblue")
    ax.plot(gens, best_so_far, linewidth=2, label="Best so far", color="orangered")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Score")
    ax.set_title("EvolveClaw-Ramsey: Score Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path is None:
        output_path = str(Path(run_dir) / "score_history.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
