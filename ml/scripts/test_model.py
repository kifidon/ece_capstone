"""
ModelTester & ModelAnalyzer: test predictions and generate analysis charts.
Run:
    python scripts/test_model.py predict Lying04.mp4
    python scripts/test_model.py plot
    python scripts/test_model.py plot Lying
    python scripts/test_model.py roc
    python scripts/test_model.py confusion
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from tensorflow import keras

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.movenet import KEYPOINT_NAMES, MoveNetProcessor
from src.preprocessor import KEYPOINT_COLUMNS, get_features_and_labels, load_pose_csv

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class ModelAnalyzer:
    """Generates analysis charts from cross-validation predictions."""

    def __init__(self, checkpoint_dir: str, output_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _load_predictions(self):
        pred_path = os.path.join(self.checkpoint_dir, "cv_predictions.npz")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(
                f"No predictions found at {pred_path}. Run training first: python scripts/train_model.py"
            )
        data = np.load(pred_path, allow_pickle=True)
        return data["true_labels"], data["pred_probs"], data["class_names"]

    def plot_roc(self) -> None:
        """Plot per-class ROC curves from cross-validation predictions."""
        true_labels, pred_probs, class_names = self._load_predictions()
        num_classes = len(class_names)
        y_true_bin = label_binarize(true_labels, classes=range(num_classes))

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                    label=f"{cls} (AUC = {roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.set_title("ROC Curves (Cross-Validation)", fontsize=15)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "roc_curves.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    def plot_confusion_matrix(self) -> None:
        """Plot confusion matrix from cross-validation predictions."""
        true_labels, pred_probs, class_names = self._load_predictions()
        pred_labels = np.argmax(pred_probs, axis=1)

        cm = confusion_matrix(true_labels, pred_labels)
        fig, ax = plt.subplots(figsize=(8, 7))
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title("Confusion Matrix (Cross-Validation)", fontsize=15)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

        print(f"\n{classification_report(true_labels, pred_labels, target_names=class_names)}")

    def plot_keypoints(self, df, class_name: str | None = None) -> None:
        """
        Plot mean keypoints for each video, grouped by class.
        If class_name is given, only plot that class. Otherwise plot all.
        """
        video_ids = df["video_id"].unique()
        classes_to_plot = [class_name] if class_name else sorted(df["class_name"].unique())

        for cls in classes_to_plot:
            cls_videos = sorted(v for v in video_ids if df[df["video_id"] == v]["class_name"].iloc[0] == cls)
            if not cls_videos:
                print(f"No videos found for class '{cls}'")
                continue

            n_videos = len(cls_videos)
            cols = min(4, n_videos)
            rows = (n_videos + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            if n_videos == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            for ax in axes[n_videos:]:
                ax.set_visible(False)

            all_xs, all_ys = [], []
            for video in cls_videos:
                frames = df[df["video_id"] == video]
                xs, ys = self._get_mean_xy(frames)
                all_xs.extend(xs)
                all_ys.extend(ys)
            x_margin = (max(all_xs) - min(all_xs)) * 0.15
            y_margin = (max(all_ys) - min(all_ys)) * 0.15
            xlim = (min(all_xs) - x_margin, max(all_xs) + x_margin)
            ylim = (max(all_ys) + y_margin, min(all_ys) - y_margin)

            for idx, video in enumerate(cls_videos):
                ax = axes[idx]
                frames = df[df["video_id"] == video]
                xs, ys = self._get_mean_xy(frames)

                for i, j in SKELETON_CONNECTIONS:
                    ax.plot([xs[i], xs[j]], [ys[i], ys[j]], "b-", alpha=0.4, linewidth=2)
                ax.scatter(xs, ys, c="red", s=60, zorder=5)
                for k, name in enumerate(KEYPOINT_NAMES):
                    ax.annotate(name, (xs[k], ys[k]), fontsize=5, ha="center", va="bottom",
                                textcoords="offset points", xytext=(0, 4))

                label = video.replace(".mp4", "")
                ax.set_title(f"{label} ({len(frames)} frames)", fontsize=12, fontweight="bold")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_aspect("equal")
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.grid(True, alpha=0.3)

            fig.suptitle(f"{cls} — Mean Keypoints Per Video", fontsize=16, fontweight="bold")
            plt.tight_layout()
            out_path = os.path.join(self.output_dir, f"keypoints_{cls.lower()}.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_path}")

    def plot_single_skeleton(self, keypoints_array, title: str, save_dir: str) -> None:
        """
        Plot skeleton with per-frame keypoint clusters, mean skeleton lines,
        and confidence-based coloring (darker = higher confidence).

        Args:
            keypoints_array: ndarray of shape (n_frames, 51) — flattened (y, x, conf) per keypoint.
            title: used for the plot title and filename.
            save_dir: directory to write the png into.
        """
        os.makedirs(save_dir, exist_ok=True)
        reshaped = keypoints_array.reshape(-1, 17, 3)  # (frames, 17, 3)
        n_frames = reshaped.shape[0]

        mean_kp = reshaped.mean(axis=0)  # (17, 3)
        mean_xs = mean_kp[:, 1]
        mean_ys = mean_kp[:, 0]

        all_xs = reshaped[:, :, 1].flatten()
        all_ys = reshaped[:, :, 0].flatten()
        x_margin = (all_xs.max() - all_xs.min()) * 0.15
        y_margin = (all_ys.max() - all_ys.min()) * 0.15

        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = plt.cm.Greys

        cluster_xs = reshaped[:, :, 1].flatten()
        cluster_ys = reshaped[:, :, 0].flatten()
        cluster_conf = reshaped[:, :, 2].flatten()

        scatter = ax.scatter(
            cluster_xs, cluster_ys,
            c=cluster_conf, cmap=cmap, vmin=0, vmax=1,
            s=8, alpha=0.25, zorder=2, edgecolors="none",
        )

        for i, j in SKELETON_CONNECTIONS:
            ax.plot(
                [mean_xs[i], mean_xs[j]], [mean_ys[i], mean_ys[j]],
                "b-", alpha=0.6, linewidth=2.5, zorder=3,
            )

        mean_conf = mean_kp[:, 2]
        mean_scatter = ax.scatter(
            mean_xs, mean_ys,
            c=mean_conf, cmap=cmap, vmin=0, vmax=1,
            s=100, zorder=5, edgecolors="blue", linewidths=1.2,
        )

        for k, name in enumerate(KEYPOINT_NAMES):
            ax.annotate(
                name, (mean_xs[k], mean_ys[k]),
                fontsize=6, ha="center", va="bottom",
                textcoords="offset points", xytext=(0, 6),
                color="blue", fontweight="bold",
            )

        cbar = fig.colorbar(mean_scatter, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("Confidence", fontsize=12)

        ax.set_xlim(all_xs.min() - x_margin, all_xs.max() + x_margin)
        ax.set_ylim(all_ys.max() + y_margin, all_ys.min() - y_margin)
        ax.set_aspect("equal")
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_title(f"{title} — Keypoints ({n_frames} frames)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_name = title.replace(" ", "_").replace("/", "_").replace(".mp4", "")
        out_path = os.path.join(save_dir, f"keypoints_{safe_name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    def _get_mean_xy(self, video_df):
        xs, ys = [], []
        for name in KEYPOINT_NAMES:
            xs.append(video_df[f"{name}_x"].mean())
            ys.append(video_df[f"{name}_y"].mean())
        return np.array(xs), np.array(ys)


class ModelTester:
    """Load a saved model, run predictions, and generate analysis charts."""

    def __init__(
        self,
        checkpoint_dir: str = os.path.join(PROJECT_ROOT, "checkpoints"),
        csv_dir: str = os.path.join(PROJECT_ROOT, "data", "csvs"),
        output_dir: str = os.path.join(PROJECT_ROOT, "outputs"),
    ):
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir

        model_path = os.path.join(checkpoint_dir, "best_model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")
        self.model = keras.models.load_model(model_path)

        labels_path = os.path.join(checkpoint_dir, "label_encoder.json")
        with open(labels_path) as f:
            classes = json.load(f)["classes"]
        self.le = LabelEncoder()
        self.le.classes_ = np.array(classes)

        csv_path = self._find_latest_csv(csv_dir)
        self.df = load_pose_csv(csv_path)

        self.analyzer = ModelAnalyzer(checkpoint_dir, output_dir)

    def _find_latest_csv(self, csv_dir: str) -> str:
        csvs = sorted(f for f in os.listdir(csv_dir) if f.startswith("poses_data") and f.endswith(".csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {csv_dir}")
        return os.path.join(csv_dir, csvs[-1])

    def predict_video(self, video_name: str) -> None:
        """Run the saved model on a video already in the CSV by name."""
        video_ids = self.df["video_id"].values
        if video_name not in set(video_ids):
            print(f"Video '{video_name}' not found. Available videos:")
            for v in sorted(set(video_ids)):
                print(f"  {v}")
            return

        held_out = self.df[self.df["video_id"] == video_name]
        X_df, y = get_features_and_labels(held_out)
        X = X_df.values.astype(np.float32)
        true_class = y.iloc[0]

        preds = self.model.predict(X, verbose=0)
        pred_classes = self.le.inverse_transform(np.argmax(preds, axis=1))

        unique, counts = np.unique(pred_classes, return_counts=True)
        print(f"\n{video_name} (true class: {true_class}, {len(pred_classes)} frames)")
        print("-" * 40)
        for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
            marker = " <-- correct" if cls == true_class else ""
            print(f"  {cls:12s}: {cnt:4d} frames ({cnt / len(pred_classes) * 100:5.1f}%){marker}")

    def predict_file(self, video_path: str) -> None:
        """Run MoveNet + model on an arbitrary video file path."""
        import pandas as pd

        video_path = os.path.abspath(video_path)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        filename = os.path.basename(video_path)
        print(f"\nProcessing {filename} with MoveNet...")
        processor = MoveNetProcessor()

        keypoints_list = []
        for _, kp in processor._process_video(video_path):
            keypoints_list.append(kp.flatten())

        X = np.array(keypoints_list, dtype=np.float32)
        print(f"Extracted {len(X)} frames")

        preds = self.model.predict(X, verbose=0)
        pred_classes = self.le.inverse_transform(np.argmax(preds, axis=1))

        unique, counts = np.unique(pred_classes, return_counts=True)
        print(f"\n{filename} ({len(pred_classes)} frames)")
        print("-" * 40)
        for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
            print(f"  {cls:12s}: {cnt:4d} frames ({cnt / len(pred_classes) * 100:5.1f}%)")

        temp_dir = os.path.join(PROJECT_ROOT, "temp")
        label = filename.replace(".mp4", "")
        self.analyzer.plot_single_skeleton(X, title=label, save_dir=temp_dir)

    def plot_keypoints(self, class_name: str | None = None) -> None:
        self.analyzer.plot_keypoints(self.df, class_name)

    def plot_roc(self) -> None:
        self.analyzer.plot_roc()

    def plot_confusion_matrix(self) -> None:
        self.analyzer.plot_confusion_matrix()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/test_model.py predict <video_name_or_path> [...]")
        print("  python scripts/test_model.py plot [class_name]")
        print("  python scripts/test_model.py roc")
        print("  python scripts/test_model.py confusion")
        sys.exit(1)

    command = sys.argv[1]
    tester = ModelTester()

    if command == "predict":
        if len(sys.argv) < 3:
            print("Specify at least one video name or file path.")
            sys.exit(1)
        for arg in sys.argv[2:]:
            if os.path.isfile(arg) or os.sep in arg:
                tester.predict_file(arg)
            else:
                tester.predict_video(arg)

    elif command == "plot":
        class_name = sys.argv[2] if len(sys.argv) > 2 else None
        tester.plot_keypoints(class_name)

    elif command == "roc":
        tester.plot_roc()

    elif command == "confusion":
        tester.plot_confusion_matrix()

    else:
        print(f"Unknown command: {command}")
        print("Use 'predict', 'plot', 'roc', or 'confusion'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
