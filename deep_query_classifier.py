import copy
import logging
from typing import List, Iterable, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier  # NEW

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DeepQueryClassifierConfig:
    embed_model_name: str = "all-MiniLM-L6-v2"
    hidden_dim: int = 256
    num_classes: int = 5
    lr: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    val_split: float = 0.2
    patience: int = 3
    random_state: int = 42

    # NEW: choose "mlp" or "xgb"
    head_type: str = "mlp"

    # NEW: XGBoost parameters
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_reg_lambda: float = 1.0
    xgb_reg_alpha: float = 0.0
    xgb_early_stopping_rounds: int = 20  # Early stopping for XGBoost
    xgb_verbose: int = 10  # Print eval metrics every N iterations (0 = silent)


class _EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class DeepQueryClassifier:
    """
    Embeddings + MLP (default) OR XGBoost boosted classifier.
    """

    def __init__(self, config: DeepQueryClassifierConfig = DeepQueryClassifierConfig()):
        self.config = config
        logger.info(f"Initializing DeepQueryClassifier with config: {config}")
        logger.info(f"Using device: {config.device}")

        self.embedder = SentenceTransformer(config.embed_model_name)
        logger.info(f"Loaded embedding model: {config.embed_model_name}")

        self.label_encoder = LabelEncoder()
        self.model = None
        self._fitted = False

    def _embed(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = np.asarray(
            self.embedder.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
            )
        )
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def fit(self, texts: List[str], labels: List[Union[str, int]]) -> None:
        logger.info(f"Starting training with {len(texts)} samples using head_type={self.config.head_type}")

        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        logger.info(f"Encoded {len(self.label_encoder.classes_)} classes: {self.label_encoder.classes_}")

        # Compute embeddings
        X = self._embed(texts)
        input_dim = X.shape[1]
        logger.info(f"Embedding dimension: {input_dim}")

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.val_split,
            random_state=self.config.random_state,
            stratify=y
        )
        logger.info(f"Split into train={len(X_train)}, val={len(X_val)}")

        # ============================================================
        # OPTION 1: MLP HEAD  (original)
        # ============================================================
        if self.config.head_type.lower() == "mlp":
            logger.info("Using MLP head")

            self.model = MLPClassifier(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                num_classes=len(self.label_encoder.classes_),
            ).to(self.config.device)

            train_dataset = _EmbeddingDataset(X_train, y_train)
            val_dataset = _EmbeddingDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

            best_val_loss = float("inf")
            patience_counter = 0
            best_model_state = None

            for epoch in range(self.config.num_epochs):
                # Train
                self.model.train()
                total_loss = 0
                for bx, by in train_loader:
                    bx, by = bx.to(self.config.device), by.to(self.config.device)
                    optimizer.zero_grad()
                    logits = self.model(bx)
                    loss = criterion(logits, by)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * bx.size(0)

                train_loss = total_loss / len(train_dataset)

                # Validate
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(self.config.device), by.to(self.config.device)
                        logits = self.model(bx)
                        loss = criterion(logits, by)
                        val_loss += loss.item() * bx.size(0)
                val_loss /= len(val_dataset)

                logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

            self._fitted = True
            logger.info("MLP training complete")
            return

        # ============================================================
        # OPTION 2: XGBoost HEAD
        # ============================================================
        elif self.config.head_type.lower() == "xgb":
            logger.info("Using XGBoost boosted tree head")

            self.model = XGBClassifier(
                objective="multi:softprob",
                num_class=len(self.label_encoder.classes_),
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                reg_lambda=self.config.xgb_reg_lambda,
                reg_alpha=self.config.xgb_reg_alpha,
                tree_method="hist",
                eval_metric="mlogloss",
                random_state=self.config.random_state,
                early_stopping_rounds=self.config.xgb_early_stopping_rounds,
            )
            logger.info(f"XGBoost configured with early_stopping_rounds={self.config.xgb_early_stopping_rounds}")

            fit_kwargs = {
                "X": X_train,
                "y": y_train,
                "eval_set": [(X_val, y_val)],
                "verbose": self.config.xgb_verbose,
            }

            logger.info(f"Training XGBoost with validation monitoring (verbose={self.config.xgb_verbose})...")
            self.model.fit(**fit_kwargs)

            # Log best_iteration and evaluation results
            try:
                best_iter = self.model.best_iteration
                logger.info(f"\nXGBoost training completed - best_iteration={best_iter}")

                # Get evaluation results
                evals_result = self.model.evals_result()
                if evals_result:
                    logger.info("\nValidation set evaluation results:")
                    for eval_name, metrics in evals_result.items():
                        for metric_name, values in metrics.items():
                            best_score = values[best_iter] if best_iter < len(values) else values[-1]
                            logger.info(f"  {eval_name} - {metric_name}: {best_score:.4f}")
            except AttributeError:
                logger.info("XGBoost training completed (early stopping not triggered)")

            self._fitted = True
            return

        else:
            raise ValueError("head_type must be 'mlp' or 'xgb'")

    def predict(self, texts: List[str]) -> List[str]:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        X = self._embed(texts)

        if self.config.head_type.lower() == "mlp":
            self.model.eval()
            with torch.no_grad():
                X_t = torch.from_numpy(X.astype(np.float32)).to(self.config.device)
                logits = self.model(X_t)
                preds = logits.argmax(dim=-1).cpu().numpy()
        else:
            preds = self.model.predict(X)

        return self.label_encoder.inverse_transform(preds).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba()")

        X = self._embed(texts)

        if self.config.head_type.lower() == "mlp":
            self.model.eval()
            with torch.no_grad():
                X_t = torch.from_numpy(X.astype(np.float32)).to(self.config.device)
                logits = self.model(X_t)
                return torch.softmax(logits, dim=-1).cpu().numpy()
        else:
            return self.model.predict_proba(X)
