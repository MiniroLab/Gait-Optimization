import pandas as pd
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────
base_dir = Path(
    r"C:\Users\chike\Box\TurtleRobotExperiments\Sea_Turtle_Robot_AI_Powered_Simulations_Project"
    r"\NnamdiFiles\mujocotest1\assets\Gait-Optimization\data\figs"
)
algos = ["LogEI", "PI", "UCB"]

# We only need these four CSVs per algorithm; they all live under bo_results_set_{i}
file_patterns = {
    "best_params":      "{algo}_best_params.csv",
    "actual_speed":     "{algo}_actual_speed.csv",
    "best_speed":      "{algo}_best_speed.csv",
    "diagnostics":      "{algo}_diagnostics.csv",
}
# ────────────────────────────────────────────────────────────────────────────

for set_dir in sorted(base_dir.glob("optplots_set_*")):
    idx   = set_dir.name.rsplit("_", 1)[-1]
    bo_dir = set_dir / f"bo_results_set_{idx}"
    if not bo_dir.is_dir():
        print(f"⚠️  Skipping {bo_dir} (not found)")
        continue

    print(f"\n→ Processing {bo_dir.relative_to(base_dir)}")
    for algo in algos:
        dfs = []

        # 1) READ + CLEAN + ENSURE 'iteration'
        for key, pattern in file_patterns.items():
            path = bo_dir / pattern.format(algo=algo)
            df = pd.read_csv(path)

            # drop any leftover index column
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])

            # unify iteration
            if "iter" in df.columns:
                df = df.rename(columns={"iter": "iteration"})
            elif "iteration" in df.columns:
                pass
            else:
                # push existing row-index into a column named 'iteration'
                df = df.reset_index().rename(columns={"index": "iteration"})

            # for the two _params tables, prefix all param columns
            if key in ("best_params"):
                prefix = "bestp"
                rename_map = {
                    col: f"{prefix}_{col}"
                    for col in df.columns
                    if col not in ("iteration",)
                }
                df = df.rename(columns=rename_map)

            dfs.append(df)

        # 2) MERGE them one by one on iteration
        merged = dfs[0]
        for df in dfs[1:]:
            merged = pd.merge(
                merged,
                df,
                on="iteration",
                how="outer"
            )

        # 3) sort & write out
        merged = merged.sort_values("iteration")
        out_path = bo_dir / f"{algo}_combined_set_{idx}.csv"
        merged.to_csv(out_path, index=False)
        print(f"   • Wrote {out_path.name} ({len(merged)} rows)")

print("\n✅ All sets processed.")
