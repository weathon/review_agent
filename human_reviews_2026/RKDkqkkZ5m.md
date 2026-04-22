# TS-DDAE: A Novel Temporal-Spectral Denoising Diffusion AutoEncoder for Wireless Signal Recognition Model Pre-training

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Wireless Signal Recognition (WSR) aims to identify the property of received signals using Artificial Intelligence (AI) without any prior knowledge, which has been widely used in civil and military radios. The current AI trend of pre-training and fine-tuning has shown great performance, and the existing pre-trained WSR models also achieve impressive results. However, they either apply the ``mask-reconstruction" pre-training strategy, which may disrupt intricate local dependencies of signals, or overlook latent spectral characteristics. Therefore, in this paper, we follow the diffusion models and propose a pre-training framework for WSR, named the Temporal-Spectral Denoising Diffusion AutoEncoder (TS-DDAE), which learns signal representations by corrupting signals with temporal and spectral noise, and then reconstructing the original data with a learned neural network. Moreover, we design a novel neural architecture, named TS-Net, which couples self-attention for temporal encoder with channel attention for spectral encoder. Extensive experiments on several datasets and WSR tasks show that TS-DDAE achieves superior performance compared to state-of-the-art (SOTA) baselines, which demonstrate the potential to be a foundation model for WSR. Code is available at https://github.com/BUPT-GAMMA/FoundWSR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new framework for Wireless Signal Recognition (WSR). The authors introduce a pre-training and fine-tuning paradigm that enables the model to adapt to multiple downstream WSR tasks. Experiments further demonstrate the few-shot learning capability of the proposed method.

### Strengths
- The paper is easy to follow, and each section is well-written. The authors provide detailed texts to describe their methods.
- The authors provide experiments on several benchmarks to validate the performance of the proposed method.

### Weaknesses
- The authors claim that this method is a "foundation model" in the abstract. In this case, did the authors provide details about the total amount of data used in the pre-training stage?
- I am also curious about the scalability of the method. Can it be extended to larger-scale datasets?
- It is recommended to conduct out-of-domain (OOD) experiments to evaluate the generalization ability of the method. The OOD tests are important for any foundation model.

### Questions
Based on the Weaknesses, I have totally three questions:
- Did the authors provide details about the total amount of data used in the pre-training stage?
- Can the authors further test the algorithm on the larger-scale datasets?
- The authors can test the out-of-domain experiments to further verify the generalization ability of the method.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a novel self-supervised representation learning approach for wireless signals based on the paradigm of denoising diffusion. A key contribution is the use of both, the time domain and frequency domain representation of the signals, in the diffusion process. In addition, they propose a model architecture that performs the diffusion process for both data views in an interconnected manner. They use attention to detect important contributions along time-steps and along convolutional channels according to the properties of the respective view.

### Strengths
- The use of denoising diffusion for wireless signals is original. 
- The authors consider the properties of each view of the signals in the design of the neural networks architecture.
Using attention to extract features along time and convolutional channels appears to be thoughtful.
- A comprehensive evaluation on two downstream tasks is performed. The proposed method shows significant gains over the state-of-the-art across multiple datasets.

### Weaknesses
The theoretical model in Section 3.2 may contain a substantive problem. The authors appears to suggest that temporal–spectral diffusion reduces to adding two noise processes with different intensities to the time-domain representation $x_{t-1}$ (Eq. 1-5). Since the FFT is a unitary transform, adding noise in the frequency domain is equivalent to adding noise in the time domain. The manuscript does not currently explain why adding noise twice, rather than once, would improve the diffusion process. In addition, the role of the hyperparameter λ—intended to balance spectral and temporal noise power—remains unclear if these quantities are effectively equivalent.


- Minor Problems: 
	- Unintroduced formula symbols (mu, tau) in section 3.2. 
	- The claim "Masking corrupts fine-grained information in the input data" is not backed up by citations/theory
	- In the ablation study in 4.4 a simple case should be investigated that considers both data views simultaneously without using TS-Net, to ensure the reported gains are achieved by the network design and not by using richer input data. 
	- Figure 2. is not explained sufficiently

### Questions
- The Metric used in section 4 is not explained properly. Is it the Top-1 accuracy? 
- How are the baseline models trained?
- Can you please explain the rationale behind adding noise in the frequency *and* in the time domain?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors propose a Temporal-Spectral Dual Denoising Autoencoder designed for wireless signal classification . The proposed model includes two contributions: (1) a dual-branch encoder. One for temporal (IQ) and one for spectral (FFT of IQ) data; and (2) a dual-diffusion denoising pre-training scheme, where both temporal and spectral branches learn to reconstruct noisy inputs via a diffusion-based objective. The authors claim this design captures complementary information (time and frequency structure) and improves robustness under low SNR or unseen modulation types.

### Strengths
(1) The paper is well-motivated and well-presented. The paper identifies an underexplored limitation in AMC where the existing methods often rely solely on IQ or spectral representations. The temporal-spectral dual design is conceptually well-grounded. 

(2) Introducing a denoising diffusion to selectively mask or corrupt input signals is an elegant and well-motivated regularization strategy for representation learning, particularly in the context of wireless signals. The formulation aligns well with the established diffusion pre-training principles and has clear potential to inspire future research directions in this domain.

(3) The empirical validation is comprehensive. The proposed method achieves strong performance on large-scale datasets and maintains robustness at low SNRs. Results demonstrate that the fusion of temporal and spectral cues contributes meaningfully.

### Weaknesses
(1) While the paper briefly mentions the denoising objective, some important implementation details are missing (see Questions 1 to 3). Without these details, it is difficult to validate the claim that the improvement arises from the proposed pre-training rather than other training heuristics.

(2) In Lines 267 to 269, the authors claim that the Feed-Forward Network (FFN) injects non-linearity. However, Figure 2 and the surrounding explanation show only linear components. There is no mention of a nonlinear activation. If the FFN is purely linear, it cannot increase representational capacity in a nonlinear way. Please clarify whether an activation function is omitted in the figure or truly absent. If absent, revise the wording to avoid implying nonlinearity.

(3) My main concern is the input redundancy across encoders. Both the temporal encoder and spectral encoder are said to take both IQ data and spectral data as input. Conceptually, this does not follow the motivation (temporal and spectral disentanglement) and could lead to redundant or entangled features. Please conduct an ablation study where each encoder only uses its corresponding modality to confirm its necessity.

(4) The reported results show that TS-DDAE underperforms IQFormer on small-scale datasets but outperforms on large ones. This raises a concern that improvements may be primarily from larger model capacity or heavier training rather than architectural novelty. No evidence (e.g., parameter-matched comparison or FLOPs table) is provided to rule this out. Please add an ablation where model parameters are matched with the baseline, or provide FLOPs/parameter counts for each model in a table to isolate architecture-driven performance improvements.

### Questions
(1) What architecture or initialization is used during pre-training (same as fine-tuning or smaller encoder)?

(2) How long is the pre-training and on which dataset?

(3) What self-supervised schedule or learning rate strategy is used in pre-training?

### Soundness
3

### Presentation
3

### Contribution
3
