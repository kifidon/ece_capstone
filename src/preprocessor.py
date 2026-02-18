"""
Preprocessor: load pose CSV and split into train/test by video (stratified by class).
"""

import logging
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from .movenet import KEYPOINT_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

KEYPOINT_COLUMNS = [f"{name}_{axis}" for name in KEYPOINT_NAMES for axis in ("y", "x")]


def _extract_video_id(frame_id: str) -> str:
    """Extract video id from frame_id. e.g. squat_01.mp4_frame5 -> squat_01.mp4."""
    if not frame_id or "_frame" not in frame_id:
        return frame_id
    return frame_id.rsplit("_frame", 1)[0]


def load_pose_csv(csv_path: str) -> pd.DataFrame:
    """Load pose CSV. Expects columns: frame_id, keypoint columns, class_name."""
    df = pd.read_csv(csv_path)
    df["video_id"] = df["frame_id"].astype(str).apply(_extract_video_id)
    return df


def split_train_test(
    csv_path: str,
    train_ratio: float = 0.8,
    random_state: int = 42,
    train_path: str | None = None,
    test_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split pose CSV into train and test by video, stratified by class.

    Frames from the same video stay together (all in train or all in test)
    to avoid data leakage. Videos are split 80/20 (configurable) per class.

    Args:
        csv_path: Path to the combined pose CSV.
        train_ratio: Fraction of videos per class for training (default 0.8).
        random_state: Random seed for reproducibility.
        train_path: Optional path to save train CSV.
        test_path: Optional path to save test CSV.

    Returns:
        (train_df, test_df) with columns frame_id, video_id, keypoints, class_name.
    """
    df = load_pose_csv(csv_path)

    if df.empty:
        logger.warning("CSV is empty.")
        return pd.DataFrame(), pd.DataFrame()

    videos_per_class = df.groupby("class_name")["video_id"].apply(
        lambda x: x.drop_duplicates().tolist()
    ).to_dict()

    train_video_ids = set()
    test_video_ids = set()

    for class_name, video_ids in videos_per_class.items():
        if len(video_ids) < 2:
            logger.warning("Class %s has only %d video(s); all go to train.", class_name, len(video_ids))
            train_video_ids.update(video_ids)
            continue
        train_ids, test_ids = train_test_split(
            video_ids,
            train_size=train_ratio,
            random_state=random_state,
            stratify=None,
        )
        train_video_ids.update(train_ids)
        test_video_ids.update(test_ids)

    train_df = df[df["video_id"].isin(train_video_ids)].copy()
    test_df = df[df["video_id"].isin(test_video_ids)].copy()

    logger.info("Split: %d train rows (%d videos), %d test rows (%d videos)",
        len(train_df), len(train_video_ids), len(test_df), len(test_video_ids))

    if train_path:
        train_df.to_csv(train_path, index=False)
        logger.info("Saved train to %s", train_path)
    if test_path:
        test_df.to_csv(test_path, index=False)
        logger.info("Saved test to %s", test_path)

    return train_df, test_df


def get_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract keypoint features and labels from a preprocessed DataFrame.

    Returns:
        (X, y) where X has shape (n_samples, 34) and y has shape (n_samples,).
    """
    X = df[KEYPOINT_COLUMNS].astype(float)
    y = df["class_name"]
    return X, y
