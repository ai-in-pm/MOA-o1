# Mechanistic Interoperability in LLMs: A Step-by-Step Demonstration

This project provides a hands-on demonstration of mechanistic interoperability in Large Language Models (LLMs) by implementing a simplified Transformer architecture from scratch using PyTorch.

## Project Structure

- `transformer.py`: Core implementation of the Transformer architecture components
- `demo.py`: Demonstration script showing the model in action
- `requirements.txt`: Project dependencies

## Components Demonstrated

1. **Token Embeddings**: Convert discrete tokens to continuous vectors
2. **Positional Encodings**: Add sequence position information using sinusoidal functions
3. **Multi-Head Self-Attention**: Allow tokens to attend to different parts of the sequence
4. **Position-wise Feed-Forward Network**: Process each position independently
5. **Transformer Block**: Combine attention and feed-forward with residual connections and layer normalization

## Running the Demo

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the demonstration:
   ```bash
   python demo.py
   ```

   ## Output
   ```bash
         Model Architecture:
      - Embedding dimension: 16
      - Number of heads: 4
      - Feed-forward dimension: 64
      - Number of layers: 2

      Input shape: torch.Size([1, 5])
      Output shape: torch.Size([1, 5, 16])

      Visualizing attention patterns...
   ```

The demo will create a small Transformer model and visualize attention patterns across different heads and layers.

## Key Features

- Clean, modular implementation of Transformer components
- Visualization of attention weights
- Detailed comments explaining each mechanism
- Example of forward and backward passes

## Understanding the Code

- Each component is implemented as a separate PyTorch module
- The code includes extensive comments explaining the purpose of each component
- The demonstration shows how data flows through the model and how different components interact

## References

- "Attention Is All You Need" (Vaswani et al., 2017)
- Neel Nanda's work on mechanistic interpretability
- Various research papers on Transformer interpretability
