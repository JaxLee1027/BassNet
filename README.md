# BassNet
BassNet is an exploratory deep learning project designed to tackle a unique acoustic challenge: Can we reconstruct the missing low-frequency portion of an audio signal using only its high-frequency components?

Experiment 0: Conduct initial experience using Real-Acoustic-Field dataset. We used U-Net as the backbone, with 31 million parameters. The first experience involved only RIR .wav files as dataset, we expected to introduce more dimensions to reinforce the prediction.

Result of Experiment 0:
![alt text](sample_0_comparison.png)

The Average MSE on test dataset is 24.5, which means on average, our model's prediction at any (time, frequency) point will deviate from the true value by ~5 dB. This is a good start as using RIR alone to train the model.