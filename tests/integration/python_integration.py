"""
Example: XOR Problem with PyNN3 Python Bindings

This demonstrates the high-level Python API for neural network training.
All matrix operations are handled internally - users work only with NumPy arrays.
"""

import numpy as np
import pynn3 as nn

# 1. Prepare Data (Standard NumPy)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 2. Create Network with Optimizer and Loss
model = nn.NeuralNets(
    optimizer=nn.SGD(learning_rate=0.5),  # or nn.Adam(0.01)
    loss=nn.MSE()                          # or nn.BinaryCrossEntropy()
)

# 3. Build Architecture
model.add_layer(input_size=2, output_size=4, activation=nn.ReLU())     # Hidden Layer
model.add_layer(input_size=4, output_size=1, activation=nn.Sigmoid())  # Output Layer

# 4. Train (NumPy arrays in, nothing out - prints progress)
print(f"Training network with {model.num_layers} layers...")
model.train(X, y, epochs=1000, verbose=True)

# 5. Predict (NumPy array in, NumPy array out)
predictions = model.predict(X)
print("\nPredictions:")
print(predictions)

# 6. Evaluate results
print("\nExpected vs Predicted:")
for i in range(len(X)):
    print(f"  {X[i]} -> Expected: {y[i][0]:.0f}, Predicted: {predictions[i][0]:.4f}")