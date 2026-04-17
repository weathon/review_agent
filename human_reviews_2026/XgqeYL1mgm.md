# The Radio-Frequency Transformer for Signal Separation

- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
We study a problem of signal separation: estimating a signal of interest (SOI) contaminated by an unknown non-Gaussian background/interference. Given the training data consisting of examples of SOI and interference, we show how to build a fully data-driven signal separator. To that end we learn a good discrete tokenizer for SOI and then train an end-to-end transformer on a cross-entropy loss. Training with a cross-entropy shows substantial improvements over the conventional mean-squared error (MSE). Our tokenizer is a modification of Google's SoundStream, which incorporates additional transformer layers and switches from VQVAE to finite-scalar quantization (FSQ). Across real and synthetic mixtures from the MIT RF Challenge dataset, our method achieves competitive performance, including a 122x reduction in bit-error rate (BER) over prior state-of-the-art techniques for separating a QPSK signal from 5G interference. The learned representation adapts to the interference type without side information and shows zero-shot generalization to unseen mixtures at inference time, underscoring its potential beyond RF. Although we instantiate our approach on radio-frequency mixtures, we expect the same architecture to apply to gravitational-wave data (e.g., LIGO strain) and other scientific sensing problems that require data-driven modeling of background and noise.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a novel model architecture for signal of interest separation in congested radio frequency spectra. The approach consists of a VQ-VAE-based tokenizer that produces discrete representations of input waveforms and a transformer as source separation module estimating the discrete waveform representations. It uses finite scalar quantization instead of vector quantization for the VQ-VAE model and employs transformer blocks throughout both proposed network architectures. Extensive experiments are conducted, indicating a high performance and excellent generalization capabilities.

### Strengths
- The use of discrete representations for source signal representation appears to be original.
- The proposed method shows state-of-the-art performance in the signal of interest separation task and excellent generalization capabilities.

### Weaknesses
- I identified the use of a tokenizer to train a source separation model in the discrete domain as the main contribution of this paper. However, this design choice is not theoretically motivated nor evaluated. Furthermore, the authors claim that this model design improves training efficiency without providing any analysis. 
- A large part of the paper focuses on the use of scalar quantization in the VQ-VAE and the employment of transformer blocks. However, these architectural improvements appear to be a minor contribution from a scientific POV.  
- The ablation study only shows training loss curves which are hard to interpret. Instead, a comparison using the MSE and BER would should be presented.
- The performance gains for the test-sets QPSK+CommSignal2 and QPSK+EMI appear to be incremental

### Questions
Problems with the Presentation:
- Table 1 is not formatted correctly
- Inconsistent coloring in the result plots (makes it hard to follow)
- Figure 6 (left) what is the grey line representing?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper the authors propose the radio-frequency transformer, a method for performing source separation of (complex) scalar signals of interests (SOI) linearly perturbed by an interference, specifically targeting QPSK (a digital encoding scheme) signals in wireless communications. To this end they propose a conditional generative framework employing (i) a finite-scalar quantization [1] autoencoder based on Soundstream [2] which acts as a tokeniser on the continuous signal (ii) an autoregressive transformer conditioned on the mixed (perturbed) signal, modelling the tokens of the SOI. At inference time they auto-regressively decode the SOI given the conditional mixture, and then the discrete bit sequence can be obtained via matched filtering. The authors showcase state-of-the-art results on common datasets in the domain (CommSignal2,  CommSignal3, CommSignal5G, EMISignal) training a different model for each one, and additionally experiment with a multi-type single model on the four type of interferences combined (showing good results except on CommSignal5G).

### References
- [1] Mentzer, F., Minnen, D., Agustsson, E., & Tschannen, M. Finite Scalar Quantization: VQ-VAE Made Simple. In The Twelfth International Conference on Learning Representations.
- [2] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. Soundstream: An end-to-end neural audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30:495–507, 2021.

### Strengths
The method showcases very good empirical results on common benchmarks in the wireless communications domain, surpassing current state-of-the-art models of the latest ICASSP challenge on wireless signal source separation. I also appreciate the effort put into making the model real-time, since this is the typical real-world use-case. Finally, the experiments on additive Gaussian intereference and the ablations are well received, showcasing improvement over classic baselines such as matched filtering and LMMSE.

The organisation and writing is smooth and the paper is not difficult to follow (maybe except for the details on QPSK bit encoding; I appreciate the appendix detailing those details for someone familiar with signal processing and deep learning but not familiar with the sub-field).

### Weaknesses
My thought reading this paper is that it would rank low on novelty since variants of the proposed method (conditional generation via autoregressive transformers for signal processing) is becoming more ubiquitous, for example in audio source separation [1] and accompaniment generation [2]. Also [3] should be mentioned as being the first method to perform source separation in a quantised autoencoder domain via autoregressive transformers. Nevertheless in the application domain of wireless comunications the method is novel as the use of finite-scalar quantization in this type of task. 

I think the best thing for the work would be to be published in an IEEE journal given its engineering application focus, since ICLR is more oriented to novel methodologies in deep learning. Still, I am not against either for acceptance, given the overall good empirical results and writing, thus my recommendation score is 6.

### References

- [1] Yin, X., Peng, X., Jiang, X., Xiong, Z., & Lu, Y. (2025). Text-Queried Audio Source Separation via Hierarchical Modeling. arXiv preprint arXiv:2505.21025.
- [2] Rouard, S., San Roman, R., Adi, Y., & Roebel, A. (2025, April). MusicGen-Stem: Multi-stem music generation and edition through autoregressive modeling. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE.
- [3] Postolache, E., Mariani, G., Mancusi, M., Santilli, A., Cosmo, L., & Rodolà, E. (2023, June). Latent autoregressive source separation. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 8, pp. 9444-9452).

### Questions
- Line 041 to line 045: This phrase is too long containing many logical connectors (ands and ors that) over which is difficult to put
precedence.
- Line 054: It is not clear what is the problem with classic convolutional designs: I don't understand where they rely on fixed-sized inputs, during training or inference time? This should be the case for Transformers as well. Also convolutions are shift invariant so one could generalize to different sizes at inference time.  Is the problem with long receptive fields the fact that it would increases the latency at inference time? Also jumping to the real-time section of the paper, one does not store key-value cache over past windows to make it not scale quadratically in memory?
- Line 101: the cyclic prefix in OFDM is not explained
- Line 212:  What is the additional context $c_L$ ($c_R$)?
- Line 236: Maybe use we trained instead of evaluate, because evaluation is a synonym of testing.
- Line 237: Please send the reader to the appendix for the definition of QPSK, since it is not common knowledge except of the wireless communications field
- Line 244: I cannot understand what is the unsynchronized setting? Shouldn’t the SOI be encoded via QPSK? Why they are two different things?
- Line 256: "Predicted waveform"
- Figure 3: The authors should use the same colors for same models
- Line 327: In the real-time section one should report RTF for the models involved
- Line 358: "we do not expect specialized models to perform well on unknown structured interference.": This should be tested empirically

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a Transformer-based architecture that recasts the problem of RF signal separation from a continuous regression task to a discrete sequence-to-sequence problem. The key idea is to first train a specialized tokenizer, leveraging finite scalar quantization, to learn a compact, discrete representation of the clean signal of interest (SOI). Then, an encoder-decoder Transformer is trained to predict this tokenized sequence from a noisy, mixed input. This token-based approach allows the model to be trained directly with a cross-entropy objective, which is far better aligned with the ultimate goal of minimizing bit-error rate (BER) than the conventional MSE loss used in prior work. The results are pretty compelling, showing a massive reduction in BER of over 100-fold in the challenging 5G interference case, as well as demonstrating zero-shot generalization to Gaussian noise, suggesting the model learns a robust underlying representation of the signal structure rather than just overfitting to specific interference types.

### Strengths
1. The paper steps outside the well-trodden domain of audio source separation and applies modern sequence-to-sequence modeling to the more constrained problem of RF signal separation. The domain is very intriguing and the success is measured by the unforgiving metric of Bit Error Rate (BER), not perceptual audio quality. By successfully adapting these advanced architectures to this field, the authors bridge a critical gap between mainstream deep learning and a specialized engineering domain with direct, tangible implications for wireless communications.
2. Perhaps the most important empirical result out of this paper is the model's really good zero-shot generalization to Additive White Gaussian Noise (AWGN). The Transformer was only ever trained on highly structured, real-world interference, yet when the input is disturbed with the unstructured Gaussian noise, it performed nearly as well as the theoretically optimal matched filter. This is a powerful demonstration that the model didn't just learn to "invert" a few specific interference types but it learned a deep and robust representation of the underlying structure of the QPSK signal itself.

### Weaknesses
The paper has some weaknesses and I will try to write them down with decreasing order of significance.

1. The authors should consider placing their contributions with proper citations of the previous methods in the field. The first known published method to perform separation of any signal in some latent continuous domain for neural networks was proposed in [D] and more similar to the contribution of this paper, the first method to formulate the separation problem to a classification-like problem where the targets are discrete latent vectors is in [C]. The authors use soundstream and it on their data which is simply a convolutional encoder that performs residual vector quantization identical to [C]. 
2. The authors should consider to compare against more modern state-of-the-art models in the source separation domain like Tiger [A] and TF-LocoFormer [B] and see if any of those models would perform better in terms of the separation of the signals. 
3. Towards the point above an analysis that should be included a more holistic computational complexity analysis of the proposed model. Specifically, a potential issue of the proposed architecture is the missing evaluation in terms of the actual memory and computational requirements of the proposed architecture. I think the paper would be significantly benefited if the authors analyze the execution time of their model under some hardware, the actual memory footprint as well as the number of FLOPs. A complex architecture like the one proposed in here might severely enlarge the actual memory footprint of a forward pass of the network (or backward) mainly because of the intermediate activations tensors (for instance, this is generally true for recurrent and attention-based models that they seem to require less computation because of a smaller number of parameters but in real device deployment those models could be even 10 times slower to train than others). The authors should also report several other benchmark factors like actual memory requirements, time consuming on inference on GPU / CPU as the ones widely used in measuring the computational complexity of audio source separation [D] which is similar to the task solved in here. 

I would gladly increase my score if the authors work properly to address the above issues to a proper degree.



[A] Xu, M., Li, K., Chen, G. and Hu, X., TIGER: Time-frequency Interleaved Gain Extraction and Reconstruction for Efficient Speech Separation. In The Thirteenth International Conference on Learning Representations.

[B] Saijo, K., Wichern, G., Germain, F.G., Pan, Z. and Le Roux, J., 2024, September. TF-Locoformer: Transformer with local modeling by convolution for speech separation and enhancement. In 2024 18th International Workshop on Acoustic Signal Enhancement (IWAENC) (pp. 205-209). IEEE.

[C] Jiang, Xilin. "Vector-quantized speech separation." (2021).

[D] Tzinis, E., Venkataramani, S., Wang, Z., Subakan, C. and Smaragdis, P., 2020, May. Two-step sound source separation: Training on learned latent targets. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 31-35). IEEE.

[E] Tzinis, E., Wang, Z., Jiang, X. and Smaragdis, P., 2022. Compute and memory efficient universal sound source separation. Journal of Signal Processing Systems, 94(2), pp.245-259.

### Questions
Is the second decimal point in the MSE metrics significant to report in these Tables? I doubt that any human being could tell the difference between 28.71 and 28.72 dB (actually I would argue that 27 and 28 would not even be differentiated properly). Are there any more useful statistics to report here?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a transformer-based model for radio-frequency (RF) signal separation, utilizing a SoundStream-style neural tokenizer to discretize input waveforms and training the transformer with cross-entropy loss over the quantized tokens. Evaluated on the MIT RF Challenge dataset, the model achieves improvements in bit error rate compared to prior architectures like WaveNet, but without introducing substantive architectural or theoretical novelty.

### Strengths
The paper tackles a practically relevant and challenging problem -- signal separation under RF interference. The use of cross-entropy loss on quantized token sequences, rather than waveform-level MSE, is technically appropriate for a discrete latent representation and improves compatibility with communication metrics such as BER. The writing and experimental presentation are clear and organized, helping reproducibility.

### Weaknesses
The model is a relatively straightforward adaptation of existing architectures (SoundStream tokenizer + transformer) to an RF dataset. Architectural, theoretical, or algorithmic innovations are limited.

Evaluation is restricted to the MIT RF Challenge dataset.

The paper provides no quantitative evidence on why each design choice (tokenization depth, transformer depth, etc.) matters, making the results difficult to interpret.

Assertions of robustness to unseen interference are based on synthetic, closely related test conditions, offering little evidence of real-world generalization.

Causality is discussed only in the appendix. Given the real-time nature of the application, it should instead be front and center.

The empirical results, though positive, are not compelling enough. The work appears better suited to an applied signal-processing or communications venue (e.g., ICASSP or SPAWC), where empirical improvements on established RF tasks are of primary interest.

### Questions
What happens when the signal experiences unknown channel fading? Can the model separate signals reliably under time-varying or unmodeled propagation effects?

Can the tokenizer and transformer be trained jointly, or are they frozen independently?

How sensitive is the system to quantization depth, token vocabulary size, or transformer depth?

Does the model generalize to modulation formats other than QPSK or to real-world RF mixtures beyond synthetic datasets?

### Soundness
2

### Presentation
3

### Contribution
2
