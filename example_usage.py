"""
Example usage of DeepQueryClassifier
"""
from deep_query_classifier import DeepQueryClassifier, DeepQueryClassifierConfig

# Sample training data
train_texts = [
    "What is the weather today?",
    "Tell me about machine learning",
    "How do I cook pasta?",
    "What's the capital of France?",
    "Show me the latest news",
    "Explain quantum physics",
    "How to make a cake?",
    "What time is it?",
    "Define artificial intelligence",
    "Recipe for chocolate chip cookies",
    "Current temperature in New York",
    "What is deep learning?",
    "How to bake bread?",
    "What's happening in the world?",
    "Explain neural networks",
]

train_labels = [
    "weather",
    "education",
    "cooking",
    "facts",
    "news",
    "education",
    "cooking",
    "weather",
    "education",
    "cooking",
    "weather",
    "education",
    "cooking",
    "news",
    "education",
]

# Test data
test_texts = [
    "Is it going to rain tomorrow?",
    "How does backpropagation work?",
    "Recipe for pizza",
]

if __name__ == "__main__":
    # Create classifier with custom config
    config = DeepQueryClassifierConfig(
        embed_model_name="all-MiniLM-L6-v2",
        hidden_dim=128,
        num_classes=5,
        lr=1e-3,
        batch_size=8,
        num_epochs=20,  # Set high, early stopping will kick in
        val_split=0.2,  # 20% validation split
        patience=3,  # Stop if no improvement for 3 epochs
        random_state=42,  # For reproducibility
    )

    classifier = DeepQueryClassifier(config)

    # Train the classifier (with validation and early stopping)
    print("\n" + "="*50)
    print("Training the classifier with validation and early stopping...")
    print("="*50)
    classifier.fit(train_texts, train_labels)
    
    # Make predictions
    print("\n" + "="*50)
    print("Making predictions...")
    print("="*50)
    predictions = classifier.predict(test_texts)
    
    print("\nPredictions:")
    for text, pred in zip(test_texts, predictions):
        print(f"  '{text}' -> {pred}")
    
    # Get probability predictions
    print("\n" + "="*50)
    print("Getting probability predictions...")
    print("="*50)
    probabilities = classifier.predict_proba(test_texts)
    
    print("\nProbabilities:")
    for i, text in enumerate(test_texts):
        print(f"  '{text}':")
        print(f"    {probabilities[i]}")

