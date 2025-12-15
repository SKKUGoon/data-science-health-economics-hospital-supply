import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_separate_graphs_better(
    comparison_df: pd.DataFrame,
    target_cols: list,
    model_name: str,
    save_path: str,
    is_weekly: bool = False
):
    """
    Draws separate, easy-to-read graphs for each target column.
    X-axis = Date Index
    Y-axis = predicted vs actual
    """

    # Distinct, highly contrasting colors
    pred_color = "#1f77b4"   # blue
    true_color = "#d62728"   # red

    for col in target_cols:

        # Handle MultiIndex columns: comparison_df[col]["pred"] / ["true"]
        # Or fallback if flat columns provided (though updated compare_pred_with_actual returns MultiIndex)
        if isinstance(comparison_df.columns, pd.MultiIndex):
            y_true = comparison_df[col]["true"]
            y_pred = comparison_df[col]["pred"]
        else:
            y_true = comparison_df[f"{col}_true"]
            y_pred = comparison_df[f"{col}_pred"]

        plt.figure(figsize=(14, 4))

        # Plot actual values
        plt.plot(
            comparison_df.index,
            y_true,
            label="Actual",
            linewidth=2.6,
            color=true_color,
            marker="o",
            markersize=6
        )

        # Plot predicted values
        plt.plot(
            comparison_df.index,
            y_pred,
            label="Predicted",
            linewidth=2.6,
            color=pred_color,
            marker="s",
            markersize=6
        )

        filename = f"{col.replace(" ", "_")}__{model_name.replace(" ", "_")}.png"
        plt.title(f"Predicted vs Actual({model_name}) – {col}", weight="bold", fontsize=16)
        plt.xlabel("Date")
        plt.ylabel(f"Usage({'Weekly' if is_weekly else 'Daily'})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename))
        plt.show()
        plt.close()