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
    val_split: float = 0.2  # Validation split ratio
    patience: int = 3  # Early stopping patience
    random_state: int = 42  # Random seed for reproducibility


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
    Embeddings + small deep network for 5-class query classification.
    """

    def __init__(self, config: DeepQueryClassifierConfig = DeepQueryClassifierConfig()):
        self.config = config
        logger.info(f"Initializing DeepQueryClassifier with config: {config}")
        logger.info(f"Using device: {config.device}")
        
        self.embedder = SentenceTransformer(config.embed_model_name)
        logger.info(f"Loaded embedding model: {config.embed_model_name}")
        
        self.label_encoder = LabelEncoder()
        self.model: Union[MLPClassifier, None] = None
        self._fitted = False

    def _embed(self, texts: Iterable[str]) -> np.ndarray:
        logger.info(f"Generating embeddings for {len(list(texts))} texts")
        embeddings = np.asarray(
            self.embedder.encode(
                list(texts),
                batch_size=32,
                show_progress_bar=False,
            )
        )
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def fit(self, texts: List[str], labels: List[Union[str, int]]) -> None:
        logger.info(f"Starting training with {len(texts)} samples")

        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        logger.info(f"Encoded {len(self.label_encoder.classes_)} classes: {self.label_encoder.classes_}")

        # Precompute embeddings once
        X = self._embed(texts)
        input_dim = X.shape[1]
        logger.info(f"Input dimension: {input_dim}")

        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.val_split,
            random_state=self.config.random_state,
            stratify=y
        )
        logger.info(f"Split data into train ({len(X_train)} samples) and validation ({len(X_val)} samples)")

        # Build model
        self.model = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_classes=len(self.label_encoder.classes_),
        ).to(self.config.device)
        logger.info(f"Created MLP model with hidden_dim={self.config.hidden_dim}, num_classes={len(self.label_encoder.classes_)}")

        # Create datasets and loaders
        train_dataset = _EmbeddingDataset(X_train, y_train)
        val_dataset = _EmbeddingDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        logger.info(f"Created DataLoaders with batch_size={self.config.batch_size}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        logger.info(f"Using Adam optimizer with lr={self.config.lr}")

        # Early stopping setup
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        logger.info(f"Starting training for up to {self.config.num_epochs} epochs with early stopping (patience={self.config.patience})")

        for epoch in range(self.config.num_epochs):
            # ===== TRAIN =====
            total_loss = 0.0
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.config.device)
                batch_y = batch_y.to(self.config.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_x.size(0)

            train_loss = total_loss / len(train_dataset)

            # ===== VALIDATION =====
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.config.device)
                    batch_y = batch_y.to(self.config.device)
                    logits = self.model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item() * batch_x.size(0)

            val_loss /= len(val_dataset)

            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} "
                        f"- train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # ===== EARLY STOPPING CHECK =====
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                logger.info(f"  → New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                logger.info(f"  → No improvement (patience: {patience_counter}/{self.config.patience})")
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}!")
                    break

        # Restore the best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model with validation loss: {best_val_loss:.4f}")

        self._fitted = True
        logger.info("Training completed successfully with early stopping")

    def predict(self, texts: List[str]) -> List[str]:
        if not self._fitted or self.model is None:
            raise RuntimeError("Call .fit() before .predict().")

        logger.info(f"Predicting labels for {len(texts)} texts")
        self.model.eval()
        with torch.no_grad():
            X = self._embed(texts)
            X_t = torch.from_numpy(X.astype(np.float32)).to(self.config.device)
            logits = self.model(X_t)
            preds = logits.argmax(dim=-1).cpu().numpy()

        predictions = self.label_encoder.inverse_transform(preds).tolist()
        logger.info(f"Predictions completed: {predictions}")
        return predictions

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self._fitted or self.model is None:
            raise RuntimeError("Call .fit() before .predict_proba().")

        logger.info(f"Predicting probabilities for {len(texts)} texts")
        self.model.eval()
        with torch.no_grad():
            X = self._embed(texts)
            X_t = torch.from_numpy(X.astype(np.float32)).to(self.config.device)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        logger.info(f"Probability predictions completed with shape: {probs.shape}")
        return probs

