"""
view_sweep.py — View and plot hyperparameter sweep results.
 
Usage:
    python view_sweep.py                        # print full summary table
    python view_sweep.py --param window_size    # show one param's results + plot
    python view_sweep.py --best                 # print best value per param
"""
 
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
 
RESULTS_CSV = "results/sweep_results.csv"
 
 
def load():
    if not os.path.exists(RESULTS_CSV):
        print("No sweep results found. Run sweep.py first.")
        exit()
    return pd.read_csv(RESULTS_CSV)
 
 
def print_full(df):
    print("\n=== All sweep results ===")
    cols = ["param","value","sharpe_mean","sharpe_std","nw_mean","max_drawdown","num_trades","win_rate"]
    print(df[cols].to_string(index=False))
 
 
def print_best(df):
    print("\n=== Best value per parameter (by Sharpe mean) ===\n")
    for param, group in df.groupby("param"):
        best = group.loc[group["sharpe_mean"].idxmax()]
        print(f"  {param:<25} best = {best['value']:<10}  "
              f"Sharpe = {best['sharpe_mean']:.4f} ± {best['sharpe_std']:.4f}  "
              f"NW = ${best['nw_mean']:.0f}  "
              f"MDD = {best['max_drawdown']:.3f}")
 
 
def plot_param(df, param):
    subset = df[df["param"] == param].copy()
    subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
    subset = subset.sort_values("value")
 
    if subset.empty:
        print(f"No results for param: {param}")
        return
 
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Hyperparameter sweep: {param}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
 
    metrics = [
        ("sharpe_mean",   "sharpe_std",   "Sharpe ratio (mean ± std)", "green"),
        ("nw_mean",       "nw_std",       "Final net worth ($)",        "steelblue"),
        ("max_drawdown",  None,           "Max drawdown",               "red"),
        ("num_trades",    None,           "Number of trades",           "orange"),
        ("win_rate",      None,           "Win rate",                   "purple"),
        ("annual_return", None,           "Annual return",              "teal"),
    ]
 
    for i, (metric, err_col, title, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        x  = subset["value"].values
        y  = pd.to_numeric(subset[metric], errors="coerce").values
 
        if err_col and err_col in subset.columns:
            e = pd.to_numeric(subset[err_col], errors="coerce").values
            ax.errorbar(x, y, yerr=e, fmt="o-", color=color,
                        capsize=5, linewidth=2, markersize=6)
        else:
            ax.plot(x, y, "o-", color=color, linewidth=2, markersize=6)
 
        # highlight best Sharpe value
        best_idx = pd.to_numeric(subset["sharpe_mean"], errors="coerce").idxmax()
        best_x   = subset.loc[best_idx, "value"]
        best_y   = subset.loc[best_idx, metric] if metric in subset.columns else None
        if best_y is not None and not pd.isna(best_y):
            ax.axvline(best_x, linestyle="--", color="gray", alpha=0.5, linewidth=1)
 
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(param, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
 
    os.makedirs("results", exist_ok=True)
    out_path = f"results/sweep_{param}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")
    plt.show()
 
 
def plot_all_best(df):
    """Bar chart of best Sharpe per parameter for quick comparison."""
    rows = []
    for param, group in df.groupby("param"):
        best = group.loc[group["sharpe_mean"].idxmax()]
        rows.append({
            "param": param,
            "best_value": best["value"],
            "sharpe": best["sharpe_mean"],
            "sharpe_std": best["sharpe_std"],
        })
    summary = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
 
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.RdYlGn(
        (summary["sharpe"] - summary["sharpe"].min()) /
        (summary["sharpe"].max() - summary["sharpe"].min() + 1e-8)
    )
    bars = ax.bar(summary["param"], summary["sharpe"], color=colors,
                  yerr=summary["sharpe_std"], capsize=4)
 
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"val={row.best_value}",
                ha="center", va="bottom", fontsize=7, rotation=45)
 
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Best Sharpe ratio per hyperparameter (at optimal value)", fontsize=12)
    ax.set_ylabel("Sharpe ratio (mean across MC runs)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
 
    out_path = "results/sweep_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved summary plot: {out_path}")
    plt.show()
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", default=None, help="Show results for one param")
    parser.add_argument("--best",  action="store_true", help="Print best value per param")
    parser.add_argument("--plot-all", action="store_true", help="Plot summary across all params")
    args = parser.parse_args()
 
    df = load()
 
    if args.param:
        subset = df[df["param"] == args.param]
        print(subset[["value","sharpe_mean","sharpe_std","nw_mean",
                       "nw_ci_low","nw_ci_high","max_drawdown",
                       "num_trades","win_rate","calmar"]].to_string(index=False))
        plot_param(df, args.param)
    elif args.best:
        print_best(df)
    elif getattr(args, "plot_all"):
        plot_all_best(df)
    else:
        print_full(df)
        print_best(df)
 
 
if __name__ == "__main__":
    main()
 