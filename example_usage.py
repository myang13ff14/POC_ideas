"""
Example usage of DeepQueryClassifier with multiple JSON dataset files.
"""

import os
import glob
import json
from deep_query_classifier import DeepQueryClassifier, DeepQueryClassifierConfig


# Option A: point to a directory of JSON files
DATA_DIR = "/home/yang8/POC_ideas/data"   # e.g. ./data_categories/*.json

# Option B: explicitly list JSON files (uncomment if you prefer this)
# JSON_FILES = [
#     "data/psref_100.json",
#     "data/highspot_100.json",
#     "data/customer_success_100.json",
#     "data/account_data_100.json",
# ]


def load_multiple_json_datasets(
    json_paths_or_dir
):
    """
    Load and merge data from multiple JSON files.

    Each file is expected to be a list of objects:
        { "category": "...", "question": "..." }
    """
    texts = []
    labels = []

    if os.path.isdir(json_paths_or_dir):
        # Load all .json files under the directory
        file_paths = glob.glob(os.path.join(json_paths_or_dir, "*.json"))
    else:
        # Assume it's already an iterable of file paths
        file_paths = list(json_paths_or_dir)

    print(f"Found {len(file_paths)} JSON files:")
    for p in file_paths:
        print(f"  - {p}")

    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            # adjust keys here if your schema is slightly different
            texts.append(item["question"])
            labels.append(item["category"])

    print(f"\nTotal samples loaded: {len(texts)}")
    print(f"Classes: {sorted(set(labels))}")
    return texts, labels


if __name__ == "__main__":
    # -------- Load data from multiple JSON files --------
    # Option A: directory
    train_texts, train_labels = load_multiple_json_datasets(DATA_DIR)

    # Option B: explicit list of files
    # train_texts, train_labels = load_multiple_json_datasets(JSON_FILES)

    # -------- Configure classifier --------
    config = DeepQueryClassifierConfig(
        embed_model_name="all-MiniLM-L6-v2",
        hidden_dim=128,
        num_classes=len(set(train_labels)),
        lr=1e-3,
        batch_size=8,
        num_epochs=20,
        val_split=0.2,
        patience=3,
        random_state=42,
        # Choose head:
        head_type="xgb",   # "mlp" or "xgb"
    )

    classifier = DeepQueryClassifier(config)

    print("\n" + "=" * 60)
    print("Training classifier on merged JSON datasets...")
    print("=" * 60)
    classifier.fit(train_texts, train_labels)

    # -------- Test queries --------
    test_texts = [
        "Does the ThinkPad X1 Carbon Gen 12 support dual 4K monitors?",
        "Give me customer success stories for AI in healthcare.",
        "Where can I find product compatibility for ThinkStation P7?",
        "Show me PSREF-style specs for a T14 Gen 6 Intel.",
    ]

    print("\n" + "=" * 60)
    print("Predictions")
    print("=" * 60)
    preds = classifier.predict(test_texts)
    for q, p in zip(test_texts, preds):
        print(f"'{q}'  ->  {p}")

    print("\n" + "=" * 60)
    print("Prediction probabilities")
    print("=" * 60)
    probas = classifier.predict_proba(test_texts)
    for q, probs in zip(test_texts, probas):
        print(f"'{q}':")
        print(f"  {probs}")

    # -------- Interactive Testing --------
    print("\n" + "=" * 60)
    print("Interactive Testing Mode")
    print("=" * 60)
    print("Type your queries below (or 'quit'/'exit' to stop)")
    print("-" * 60)

    # Get class names for display
    class_names = list(classifier.label_encoder.classes_)

    while True:
        try:
            user_query = input("\nEnter query: ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting interactive mode. Goodbye!")
                break

            if not user_query:
                print("Please enter a valid query.")
                continue

            # Predict
            pred = classifier.predict([user_query])[0]
            proba = classifier.predict_proba([user_query])[0]

            # Display results
            print(f"\n  Predicted Category: {pred}")
            print(f"\n  Probabilities:")

            # Get class names and sort by probability
            class_probs = list(zip(class_names, proba))
            class_probs.sort(key=lambda x: x[1], reverse=True)

            for cls, prob in class_probs:
                bar_length = int(prob * 40)  # Scale to 40 chars
                bar = "█" * bar_length
                print(f"    {cls:30s}: {prob:.4f} {bar}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting interactive mode.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue
