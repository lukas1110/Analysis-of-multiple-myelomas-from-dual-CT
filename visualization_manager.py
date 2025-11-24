import numpy as np
import pandas as pd
import seaborn as sns
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, mean_absolute_error


class VisualizationManager:
    # TODO: Visualization of LASSO
    # TODO: Visualization of MRMR

    @staticmethod
    def plot_stat_selection(knee: KneeLocator, importance: np.ndarray, threshold: float, name: str) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(importance) + 1), importance, marker='o', label='Feature importance')

        if threshold is not None:
            plt.axhline(y=threshold, color='red', linestyle='--', label=f'Elbow threshold: {threshold:.4f}')
            plt.axvline(x=knee.knee, color='orange', linestyle=':', label=f'Elbow at feature: {knee.knee}')

        plt.title(f"Selected Features for {name}")
        plt.xlabel("Feature rank")
        plt.ylabel("Importance")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_filtered_features(merged_df: pd.DataFrame, significant_features: list[str],
                               remaining_features: list[str], name: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f"Filtration of Significant features for {name}", fontsize=16, fontweight='bold')

        corr_matrix = (merged_df[significant_features].corr(method='spearman'))
        im1 = axes[0].imshow(corr_matrix, cmap='gray', vmin=-1, vmax=1)
        axes[0].set_title(f"All significant features (n={len(significant_features)})")

        final_corr_matrix = merged_df[remaining_features].corr(method='spearman')
        axes[1].imshow(final_corr_matrix, cmap='gray', vmin=-1, vmax=1)
        axes[1].set_title(f"Filtered features (n={len(remaining_features)})")

        fig.colorbar(im1, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05, label='Spearman correlation')
        plt.show()

    @staticmethod
    def plot_rf_classification(labels, predictions, knee, importance, threshold, name: str):
        cm = confusion_matrix(labels, predictions)
        class_accuracies = [(cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0) for i in range(len(cm))]
        accuracy_text = " | ".join([f"Stage {i + 1}: {class_accuracies[i] * 100:.1f}%" for i in range(len(cm))])

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Random Forest Classification for {name}: {round(cm.diagonal().sum() / cm.sum(), 2)}",
                     fontsize=16, fontweight='bold')

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[str(i) for i in [1, 2, 3]],
                    yticklabels=[str(i) for i in [1, 2, 3]], ax=axes[0], cbar=False)
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        axes[0].set_title(f"Confusion Matrix\n{accuracy_text}")

        axes[1].plot(np.arange(1, len(importance) + 1), importance, marker='o',
                     color='tab:blue', label='Feature importance')
        axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Elbow threshold = {threshold:.4f}')
        if knee.knee is not None:
            axes[1].axvline(x=knee.knee, color='orange', linestyle=':', label=f'Elbow at feature {knee.knee}')
        axes[1].set_xlabel("Feature rank")
        axes[1].set_ylabel("Importance")
        axes[1].set_title("Feature Importance Curve")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_rf_regression(labels, predictions, knee, importance, threshold, name: str):
        r2 = r2_score(labels, predictions)
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Random Forest Regression for {name} R²: {r2:.3f} || MSE: {mse:.3f} || MAE: {mae:.3f}",
                     fontsize=16, fontweight='bold')

        axes[0].scatter(labels, predictions, alpha=0.7)
        axes[0].plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'r--', lw=2)
        axes[0].set_xlabel("True values")
        axes[0].set_ylabel("Predicted values")
        axes[0].set_title("True vs Predicted")

        axes[1].plot(np.arange(1, len(importance) + 1), importance, marker='o',
                     color='tab:blue', label='Feature importance')
        axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Elbow threshold = {threshold:.4f}')
        if knee.knee is not None:
            axes[1].axvline(x=knee.knee, color='orange', linestyle=':', label=f'Elbow at feature {knee.knee}')
        axes[1].set_xlabel("Feature rank")
        axes[1].set_ylabel("Importance")
        axes[1].set_title("Feature Importance Curve")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mutual_information(knee: KneeLocator, importance: np.ndarray, threshold: float, name: str) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, len(importance) + 1), importance, marker='o', label='Feature MI')

        if threshold is not None:
            plt.axhline(y=threshold, color='red', linestyle='--', label=f'Elbow threshold = {threshold:.4f}')
            plt.axvline(x=knee.knee, color='orange', linestyle=':', label=f'Elbow at feature {knee.knee}')

        plt.title(f"Mutual information curve for {name}")
        plt.xlabel("Feature rank")
        plt.ylabel("MI")
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.legend()
        plt.tight_layout()
        plt.show()