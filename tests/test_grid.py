from harness.config import load_grid, load_models, load_tasks
from harness.run import expand


def test_configs_load():
    assert set(load_models()) == {"qwen", "llama", "gemma"}
    tasks = load_tasks()
    assert tasks["offensive"].metric == "macro_f1"
    assert tasks["sentiment"].metric == "macro_recall"


def test_main_grid_expansion():
    grid = load_grid("configs/grid_main.yaml")
    sent = [c.name for c in grid.conditions_for("sentiment")]
    off = [c.name for c in grid.conditions_for("offensive")]
    assert "code_switch_hinglish" in sent and not any(c.startswith("code_switch") for c in off)
    assert len(off) == 1 + 3 + 3 + 3
    runs = expand(grid)
    assert len(runs) == 3 * 5 * (len(sent) + len(off))
    assert len({r.out_path("outputs") for r in runs}) == len(runs)


def test_smoke_grid():
    runs = expand(load_grid("configs/grid_smoke.yaml"))
    assert [(r.regime.slug, r.condition.name) for r in runs] == [
        ("zeroshot", "none"), ("zeroshot", "typo_0.10"), ("fewshot_k4", "none"), ("fewshot_k4", "typo_0.10")]
