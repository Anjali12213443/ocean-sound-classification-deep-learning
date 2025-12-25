# Ocean Sound Classification Using Deep Learning

## Project Overview
This project investigates multiple deep learning approaches to classify marine species using underwater acoustic recordings. Different input representations, learning paradigms, and data augmentation strategies were explored to evaluate their impact on classification performance.


## Dataset
- **Type:** Ocean acoustic recordings
- **Input Representations:**
  - Spectrograms
  - Raw audio waveforms
  - Learned contrastive embeddings
  - Synthetic spectrograms (VAE, DDPM)


## Models Implemented
- Deep Neural Network (DNN)
- Convolutional Neural Network (CNN)
- Bidirectional RNN (BiRNN)
- LSTM
- Transformer
- 1D CNN on raw audio
- Contrastive Learning Classifier
- CNN with VAE-based data augmentation
- CNN with Diffusion (DDPM) augmentation


## Performance Summary
| Model / Method | Input Type | Test Accuracy |
|---------------|----------|---------------|
| DNN | Spectrograms | ~0.43 |
| CNN | Spectrograms | ~0.67 (Best supervised model) |
| BiRNN | Spectrograms | ~0.54 |
| LSTM | Spectrograms | ~0.50 |
| Transformer | Spectrograms | ~0.50 |
| 1D CNN | Raw Audio | ~0.15 |
| Contrastive Learning | Embeddings | ~0.56 |
| CNN + VAE | Spectrograms | ~0.63 |
| CNN + DDPM | Spectrograms | No improvement |


## Key Findings
- Spectrogram-based CNNs consistently outperformed other supervised models
- Contrastive learning improved class separation and generalization
- VAE-generated spectrograms improved class balance and accuracy
- Diffusion models struggled due to limited data and lack of structure
- Raw audio models underperformed without explicit frequency features


## Techniques Used
- Audio preprocessing and spectrogram generation
- Feature learning via contrastive loss
- Synthetic data generation using VAEs and DDPMs
- Model evaluation using accuracy, precision, recall, and confusion matrices
- Visualization with t-SNE and training curves


## Tech Stack
- Python
- PyTorch / TensorFlow
- Librosa
- NumPy, Pandas
- Scikit-learn
- Matplotlib


## Files in This Repository
- Jupyter notebooks with full experimentation
- Final analytical report (PDF)
- Presentation slides


## Future Improvements
- Class-conditional diffusion models
- Hybrid CNN–Transformer architectures
- Larger and more balanced datasets
- Feature-space regularization techniques


## Author
**Anjali Velu Ramalingam**  
Graduate Student – Data Science / Analytics
