"""
Quick test to verify the classifier works correctly
"""
from deep_query_classifier import DeepQueryClassifier, DeepQueryClassifierConfig

# Minimal test data
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
]

test_texts = [
    "Is it going to rain tomorrow?",
    "How does backpropagation work?",
]

if __name__ == "__main__":
    print("Testing DeepQueryClassifier with validation and early stopping...\n")
    
    # Create classifier with minimal config for quick testing
    config = DeepQueryClassifierConfig(
        hidden_dim=64,
        batch_size=4,
        num_epochs=10,
        val_split=0.2,
        patience=2,
    )
    
    classifier = DeepQueryClassifier(config)
    
    # Train
    print("="*60)
    classifier.fit(train_texts, train_labels)
    
    # Predict
    print("\n" + "="*60)
    predictions = classifier.predict(test_texts)
    
    print("\nTest Results:")
    for text, pred in zip(test_texts, predictions):
        print(f"  '{text}' -> {pred}")
    
    print("\n✓ Test completed successfully!")

