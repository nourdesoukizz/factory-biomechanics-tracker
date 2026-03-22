"""Learned task classifier — 1D Temporal CNN trained on extracted pose features."""

import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.classification import TaskClassifier, TaskState
from src.models import PersonFrame, TaskEvent

logger = logging.getLogger(__name__)

# Feature keys extracted per frame
FEATURE_KEYS = [
    "elbow_left", "elbow_right", "shoulder_left", "shoulder_right",
    "trunk_flex", "wrist_height_left", "wrist_height_right",
    "velocity", "centroid_x", "centroid_y", "rula_score"
]

# Task label mapping
TASK_TO_IDX = {"idle": 0, "pick_and_place": 1, "lift_and_place": 2, "move_rack": 3}
IDX_TO_TASK = {v: k for k, v in TASK_TO_IDX.items()}
NUM_CLASSES = len(TASK_TO_IDX)
NUM_FEATURES = len(FEATURE_KEYS)


class TemporalTaskNet(nn.Module):
    """1D temporal CNN for task classification from pose feature sequences."""

    def __init__(self, num_features: int = NUM_FEATURES, num_classes: int = NUM_CLASSES,
                 window_size: int = 30) -> None:
        """Initialize the temporal CNN.

        Args:
            num_features: Number of input features per frame.
            num_classes: Number of output task classes.
            window_size: Temporal window size.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, num_features, window_size).

        Returns:
            Logits of shape (batch, num_classes).
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


def _load_features(features_path: str) -> List[dict]:
    """Load feature records from JSON.

    Args:
        features_path: Path to features JSON file.

    Returns:
        List of feature record dicts.
    """
    with open(features_path, "r") as f:
        records = json.load(f)
    logger.info("Loaded %d feature records from %s", len(records), features_path)
    return records


def _prepare_windows(records: List[dict], window_size: int = 30,
                     stride: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows from feature records.

    Args:
        records: Feature record list.
        window_size: Number of frames per window.
        stride: Sliding stride.

    Returns:
        Tuple of (features array, labels array, confidences array).
    """
    # Group by track_id
    tracks: Dict[int, List[dict]] = defaultdict(list)
    for r in records:
        tracks[r["track_id"]].append(r)

    # Sort each track by frame_index
    for tid in tracks:
        tracks[tid].sort(key=lambda x: x["frame_index"])

    windows = []
    labels = []
    confidences = []

    for tid, frames in tracks.items():
        if len(frames) < window_size:
            continue

        for start in range(0, len(frames) - window_size + 1, stride):
            window_frames = frames[start:start + window_size]

            # Extract features
            feat_window = []
            task_votes = defaultdict(float)

            for fr in window_frames:
                feat = [fr.get(k, 0.0) for k in FEATURE_KEYS]
                feat_window.append(feat)
                task = fr.get("rule_task", "idle")
                conf = fr.get("rule_confidence", 0.5)
                task_votes[task] += conf

            windows.append(feat_window)

            # Label = weighted mode of rule_task in window
            best_task = max(task_votes, key=task_votes.get)
            labels.append(TASK_TO_IDX.get(best_task, 0))
            confidences.append(task_votes[best_task] / window_size)

    return (
        np.array(windows, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(confidences, dtype=np.float32),
    )


def _refine_labels(features: np.ndarray, labels: np.ndarray,
                   confidences: np.ndarray, n_clusters: int = 8) -> np.ndarray:
    """Use K-means clustering to refine pseudo-labels.

    Args:
        features: Feature windows array (N, window_size, num_features).
        labels: Pseudo-labels array.
        confidences: Confidence array.
        n_clusters: Number of clusters.

    Returns:
        Refined labels array.
    """
    if len(features) < n_clusters:
        return labels

    # Flatten windows for clustering
    flat = features.reshape(len(features), -1)
    # Normalize for clustering
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    flat_norm = (flat - mean) / std

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(flat_norm)

    refined = labels.copy()
    for c in range(n_clusters):
        mask = clusters == c
        if mask.sum() == 0:
            continue

        # Find majority label in this cluster
        cluster_labels = labels[mask]
        majority = int(np.bincount(cluster_labels).argmax())

        # Relabel low-confidence disagreements
        for i in np.where(mask)[0]:
            if labels[i] != majority and confidences[i] < 0.7:
                refined[i] = majority

    changed = (refined != labels).sum()
    logger.info("Label refinement: %d/%d labels changed", changed, len(labels))
    return refined


def train_model(features_path: str, model_save_path: str = "data/task_model.pt",
                stats_save_path: str = "data/task_model_stats.json",
                window_size: int = 30, epochs: int = 50,
                batch_size: int = 64, learning_rate: float = 0.001,
                n_clusters: int = 8) -> dict:
    """Train the temporal CNN on extracted features.

    Args:
        features_path: Path to features JSON.
        model_save_path: Path to save model weights.
        stats_save_path: Path to save normalization stats.
        window_size: Temporal window size.
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        n_clusters: Number of clusters for label refinement.

    Returns:
        Training stats dict.
    """
    records = _load_features(features_path)
    if not records:
        raise ValueError("No feature records found")

    features, labels, confidences = _prepare_windows(records, window_size)
    logger.info("Created %d windows (window_size=%d)", len(features), window_size)

    if len(features) < 10:
        raise ValueError(f"Too few windows ({len(features)}) for training")

    # Refine labels using clustering
    labels = _refine_labels(features, labels, confidences, n_clusters)

    # Compute normalization stats
    mean = features.mean(axis=(0, 1))  # (num_features,)
    std = features.std(axis=(0, 1)) + 1e-8

    # Normalize features
    features = (features - mean) / std

    # Save stats
    Path(stats_save_path).parent.mkdir(parents=True, exist_ok=True)
    stats = {"mean": mean.tolist(), "std": std.tolist(), "window_size": window_size}
    with open(stats_save_path, "w") as f:
        json.dump(stats, f)
    logger.info("Saved normalization stats to %s", stats_save_path)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Transpose for Conv1d: (batch, num_features, window_size)
    X_train_t = torch.FloatTensor(X_train).permute(0, 2, 1)
    X_val_t = torch.FloatTensor(X_val).permute(0, 2, 1)
    y_train_t = torch.LongTensor(y_train)
    y_val_t = torch.LongTensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    weight_tensor = torch.FloatTensor(class_weights)

    # Model, optimizer, loss
    model = TemporalTaskNet(window_size=window_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            # Data augmentation: temporal jitter + noise
            if np.random.random() > 0.5:
                shift = np.random.randint(-2, 3)
                X_batch = torch.roll(X_batch, shifts=shift, dims=2)
            if np.random.random() > 0.5:
                noise = torch.randn_like(X_batch) * 0.05
                X_batch = X_batch + noise

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        avg_train_loss = train_loss / max(train_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("Epoch %d/%d — Train loss: %.4f acc: %.3f | Val loss: %.4f acc: %.3f",
                        epoch + 1, epochs, avg_train_loss, train_acc, avg_val_loss, val_acc)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    logger.info("Saved model to %s", model_save_path)

    # Label distribution
    label_dist = {IDX_TO_TASK[i]: int(c) for i, c in enumerate(np.bincount(labels, minlength=NUM_CLASSES))}
    return {
        "windows": len(features),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "best_val_loss": round(best_val_loss, 4),
        "label_distribution": label_dist,
    }


class LearnedTaskClassifier:
    """Learned task classifier wrapping the trained TemporalTaskNet."""

    def __init__(self, model_path: str = "data/task_model.pt",
                 stats_path: str = "data/task_model_stats.json",
                 window_size: int = 30,
                 confidence_threshold: float = 0.6,
                 min_task_duration_frames: int = 8,
                 task_cooldown_frames: int = 15) -> None:
        """Load trained model and normalization stats.

        Args:
            model_path: Path to saved model weights.
            stats_path: Path to normalization stats JSON.
            window_size: Temporal window size.
            confidence_threshold: Minimum confidence to trigger a task.
            min_task_duration_frames: Minimum frames before task event logged.
            task_cooldown_frames: Frames before same task re-triggers.
        """
        self._window_size = window_size
        self._conf_threshold = confidence_threshold
        self._min_duration = min_task_duration_frames
        self._cooldown = task_cooldown_frames

        # Load model
        self._model = TemporalTaskNet(window_size=window_size)
        self._model.load_state_dict(torch.load(model_path, weights_only=True))
        self._model.eval()
        logger.info("Loaded learned classifier from %s", model_path)

        # Load normalization stats
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self._mean = np.array(stats["mean"], dtype=np.float32)
        self._std = np.array(stats["std"], dtype=np.float32)

        # Per-track feature buffers
        self._buffers: Dict[int, deque] = {}

        # Fallback rule-based classifier for warmup period
        self._fallback = TaskClassifier(
            confidence_threshold=confidence_threshold,
            min_task_duration_frames=min_task_duration_frames,
            task_cooldown_frames=task_cooldown_frames,
        )

        # State machine per person per task (same as TaskClassifier)
        self._person_states: Dict[int, Dict[str, Dict]] = defaultdict(
            lambda: defaultdict(lambda: {
                "state": TaskState.IDLE,
                "start_frame": 0,
                "consecutive_frames": 0,
                "last_end_frame": -9999,
                "confidence_sum": 0.0,
            })
        )

    def _extract_features(self, joint_angles: Dict[str, float], velocity: float,
                          centroid: Tuple[float, float], rula_score: float) -> np.ndarray:
        """Extract feature vector from current frame data.

        Args:
            joint_angles: Computed joint angles.
            velocity: Centroid velocity.
            centroid: (x, y) position.
            rula_score: RULA score.

        Returns:
            Feature vector as numpy array.
        """
        return np.array([
            joint_angles.get("elbow_left", 0.0),
            joint_angles.get("elbow_right", 0.0),
            joint_angles.get("shoulder_left", 0.0),
            joint_angles.get("shoulder_right", 0.0),
            joint_angles.get("trunk_flex", 0.0),
            joint_angles.get("wrist_height_left", 0.0),
            joint_angles.get("wrist_height_right", 0.0),
            velocity,
            centroid[0],
            centroid[1],
            rula_score,
        ], dtype=np.float32)

    def _predict(self, track_id: int) -> Tuple[str, float]:
        """Run CNN inference on the feature buffer.

        Args:
            track_id: Person track ID.

        Returns:
            Tuple of (task_name, confidence).
        """
        buf = self._buffers[track_id]
        if len(buf) < self._window_size:
            return "idle", 0.0

        # Build input tensor
        window = np.array(list(buf), dtype=np.float32)
        # Normalize
        window = (window - self._mean) / self._std
        # Shape: (1, num_features, window_size)
        x = torch.FloatTensor(window).unsqueeze(0).permute(0, 2, 1)

        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

        best_idx = int(probs.argmax())
        best_conf = float(probs[best_idx])
        task_name = IDX_TO_TASK[best_idx]

        return task_name, best_conf

    def classify_frame(self, person: PersonFrame, joint_angles: Dict[str, float],
                       velocity: float, rula_score: float, rula_label: str,
                       fps: float) -> Tuple[str, float, List[TaskEvent]]:
        """Classify task for a person using the learned model.

        Same interface as TaskClassifier.classify_frame().

        Args:
            person: PersonFrame data.
            joint_angles: Computed joint angles.
            velocity: Centroid velocity.
            rula_score: RULA score.
            rula_label: RULA label string.
            fps: Video FPS.

        Returns:
            Tuple of (task_name, confidence, list of completed TaskEvents).
        """
        tid = person.track_id

        # Initialize buffer if needed
        if tid not in self._buffers:
            self._buffers[tid] = deque(maxlen=self._window_size)

        # Add current features to buffer
        features = self._extract_features(joint_angles, velocity, person.centroid, rula_score)
        self._buffers[tid].append(features)

        # Use fallback if buffer not full yet
        if len(self._buffers[tid]) < self._window_size:
            return self._fallback.classify_frame(
                person, joint_angles, velocity, rula_score, rula_label, fps
            )

        # Run CNN inference
        task_name, task_conf = self._predict(tid)

        # Apply confidence threshold
        if task_conf < self._conf_threshold:
            task_name = "idle"
            task_conf = 1.0 - task_conf

        # Update state machine
        completed = self._update_state_machines(
            tid, person.frame_index,
            {task_name: task_conf} if task_name != "idle" else {},
            fps
        )

        return task_name, task_conf, completed

    def _update_state_machines(self, track_id: int, frame_index: int,
                               task_confidences: Dict[str, float],
                               fps: float) -> List[TaskEvent]:
        """Update state machines — same logic as TaskClassifier."""
        completed: List[TaskEvent] = []
        states = self._person_states[track_id]

        for task_name in ["pick_and_place", "lift_and_place", "move_rack"]:
            conf = task_confidences.get(task_name, 0.0)
            state_info = states[task_name]

            if frame_index - state_info["last_end_frame"] < self._cooldown:
                continue

            if conf >= self._conf_threshold:
                if state_info["state"] == TaskState.IDLE:
                    state_info["state"] = TaskState.CANDIDATE
                    state_info["start_frame"] = frame_index
                    state_info["consecutive_frames"] = 1
                    state_info["confidence_sum"] = conf
                elif state_info["state"] == TaskState.CANDIDATE:
                    state_info["consecutive_frames"] += 1
                    state_info["confidence_sum"] += conf
                    if state_info["consecutive_frames"] >= self._min_duration:
                        state_info["state"] = TaskState.ACTIVE
                elif state_info["state"] == TaskState.ACTIVE:
                    state_info["consecutive_frames"] += 1
                    state_info["confidence_sum"] += conf
            else:
                if state_info["state"] == TaskState.ACTIVE:
                    avg_conf = state_info["confidence_sum"] / max(state_info["consecutive_frames"], 1)
                    duration = state_info["consecutive_frames"] / fps if fps > 0 else 0.0
                    completed.append(TaskEvent(
                        track_id=track_id,
                        task=task_name,
                        start_frame=state_info["start_frame"],
                        end_frame=frame_index,
                        confidence=round(avg_conf, 3),
                        duration_sec=round(duration, 2),
                    ))
                    state_info["last_end_frame"] = frame_index

                state_info["state"] = TaskState.IDLE
                state_info["consecutive_frames"] = 0
                state_info["confidence_sum"] = 0.0

        return completed

    def finalize(self, track_id: int, final_frame: int, fps: float) -> List[TaskEvent]:
        """Close any open task events at end of video.

        Args:
            track_id: Person track ID.
            final_frame: Last frame index.
            fps: Video FPS.

        Returns:
            List of remaining active TaskEvents.
        """
        completed: List[TaskEvent] = []
        if track_id not in self._person_states:
            return completed

        for task_name, state_info in self._person_states[track_id].items():
            if state_info["state"] == TaskState.ACTIVE:
                avg_conf = state_info["confidence_sum"] / max(state_info["consecutive_frames"], 1)
                duration = state_info["consecutive_frames"] / fps if fps > 0 else 0.0
                completed.append(TaskEvent(
                    track_id=track_id,
                    task=task_name,
                    start_frame=state_info["start_frame"],
                    end_frame=final_frame,
                    confidence=round(avg_conf, 3),
                    duration_sec=round(duration, 2),
                ))

        return completed
