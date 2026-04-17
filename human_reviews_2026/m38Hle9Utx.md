# NeuroRVQ: Multi-Scale EEG Tokenization for Generative Large Brainwave Models

- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Electroencephalography (EEG) captures neural activity across multiple temporal and spectral scales, yielding signals that are rich but complex for representation learning.  Recently, EEG foundation models trained to predict masked signal-tokens have shown promise
for learning generalizable representations. However, their performance is hindered by their signal tokenization modules. Existing neural tokenizers fail to preserve high-frequency dynamics, limiting their ability to reconstruct EEG signals with high fidelity. We introduce NeuroRVQ, a scalable Large Brainwave Model (LBM)  centered  on a codebook-based tokenizer. Our tokenizer integrates: (i) multi-scale feature extraction modules that capture the full frequency neural spectrum; (ii) hierarchical residual vector quantization (RVQ) codebooks for high-resolution encoding; and, (iii) an EEG signal phase- and amplitude-aware loss function for efficient training. This design enables efficient EEG compression while supporting accurate reconstruction across all frequency bands, leading to robust generative masked modeling. Our empirical results demonstrate that NeuroRVQ achieves lower reconstruction error and outperforms existing LBMs on a variety of downstream tasks. More broadly, NeuroRVQ tokenizer establishes a strong prior for codebook-based general-purpose brainwave models, enabling advances in neural decoding, generative modeling and multimodal biosignal integration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces NeuroRVQ, a scalable Large Brainwave Model (LBM) for EEG signal modeling that addresses limitations in existing signal tokenization methods. It employs a codebook-based tokenizer integrating (i) multi-scale feature extraction, (ii) hierarchical residual vector quantization, and (iii) a phase- and amplitude-aware loss for efficient training. This design enables accurate reconstruction across all frequency bands and efficient EEG compression, outperforming existing approaches in reconstruction quality and downstream tasks.

### Strengths
- Although using frequency-domain information across EEG bands is common in existing brainwave foundation models, this paper’s approach effectively leverages such domain knowledge grounded in neuroscience principles, leading to more biologically informed EEG modeling.

- The paper offers clear and insightful analysis on the role of the EEG tokenizer, and proposes a well-designed hierarchical improvement strategy that enhances existing tokenization methods. The methodology is technically sound and clearly presented.

- The case analysis in Figure 4 vividly and intuitively demonstrates the performance advantages of the proposed codebook-based tokenizer, helping readers grasp the model’s effectiveness.

### Weaknesses
- The authors report training the tokenizer for 100 epochs and the foundation model for 50 epochs. For large models, this is unusually high — typical large-scale training often involves vast data with fewer epochs (e.g., 1–2). Such a high number may suggest insufficient training data or diversity, or risk overfitting.

- Although the paper lists the datasets used, the number of subjects in the training and evaluation sets is not clearly specified. This metric is crucial for assessing generalization, as limited subject diversity can restrict the universality of conclusions.

- The baselines in the four classification tasks are not comprehensive enough. For example, Table 4 includes models up to 79.5M parameters, but omits larger baselines such as Brant (≈500M), which could weaken the strength of the empirical comparisons.

- The absence of released code is a reproducibility concern, especially for a work proposing a new large-scale EEG foundation model.

### Questions
1. The authors mention training the tokenizer for 100 epochs and the foundation model for 50 epochs. Was this choice due to limited data quantity or diversity, or could it risk overfitting? This point should be more thoroughly discussed.

2. How were the hyperparameters (e.g., S = 4, λ = 0.4) selected? Were hyperparameter sensitivity analyses conducted to justify these values?

3. Please report the number of subjects in the training and evaluation datasets, as this is essential for assessing cross-subject generalization.

4. In the in-distribution and out-of-distribution experiments, the tokenizer is compared only with LaBraM. Could the limited number of baselines undermine the strength of the conclusions?

5. The authors are encouraged to include missing baselines (such as the series work of Brant [1] and Brant-2 [2]), or at least explain why these works are not suitable for comparison, and discuss them in the related work section to reflect completeness and fairness.

6. As this is a large-scale EEG model, have the authors considered testing the model and baselines under a zero-shot setting? This would better demonstrate the generalization capabilities of the proposed approach.

7. It is recommended that the authors release their model code to promote reproducibility and community engagement.

> [1] Brant: foundation model for intracranial neural signal. (https://papers.neurips.cc/paper_files/paper/2023/file/535915d26859036410b0533804cee788-Paper-Conference.pdf)
> 
> [2] Brant-2: Foundation Model for Brain Signals. (https://www.arxiv.org/abs/2402.10251v4)

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces NeuroRVQ, a scalable Large Brainwave Model (LBM) architecture for EEG data, which features multi-scale temporal convolutions, hierarchical residual vector quantization (RVQ) codebooks, and a training loss that jointly reconstructs phase and amplitude information using a sine-cosine representation. Extensive experiments demonstrate superior reconstruction error and downstream classification performance compared to prior codebook and transformer-based LBMs.

### Strengths
The manuscript is clearly written and well-structured. The extensive use of tables and figures effectively presents detailed results, ablation studies, and interpretability analyses, which greatly clarify the authors' model choices. The supplemental materials are thorough and provide valuable additional support.

### Weaknesses
1. Applying RVQ and phase-aware loss to EEG modeling is not novel; see more details in [1,2]. For example, BrainOmni replaces the original VQ with RVQ to separate different source components, and combines both time and frequency domain loss to guide pretraining.

2. The datasets evaluated in this work (Table 4) appear to have little overlap with the datasets evaluated by the baseline models [3-7], which poses a significant challenge to head-to-head model comparisons.

3. The multi-scale feature extraction module does not appear to integrate information from different scales into a single token (Line 740). While this operation does not affect the model size, it significantly increases computation time. Furthermore, this operation constitutes an implicit model ensemble, leading to unfair comparisons between models.

**References**:

[1] Xiao Q, Cui Z, Zhang C, et al. BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals[J]. arXiv preprint arXiv:2505.18185, 2025.

[2] Carzaniga F S, Hoppeler G T, Hersche M, et al. The Case for Cleaner Biosignals: High-fidelity Neural Compressor Enables Transfer from Cleaner iEEG to Noisier EEG[J]. arXiv preprint arXiv:2502.17462, 2025.

[3] Wang J, Zhao S, Luo Z, et al. Cbramod: A criss-cross brain foundation model for eeg decoding[J]. arXiv preprint arXiv:2412.07236, 2024.

[4] Jiang W B, Zhao L M, Lu B L. Large brain model for learning generic representations with tremendous EEG data in BCI[J]. arXiv preprint arXiv:2405.18765, 2024.

[5] Cui W, Jeong W, Thölke P, et al. Neuro-gpt: Towards a foundation model for eeg[C]//2024 IEEE International Symposium on Biomedical Imaging (ISBI). IEEE, 2024: 1-5.

[6] Yang C, Westover M, Sun J. Biot: Biosignal transformer for cross-data learning in the wild[J]. Advances in Neural Information Processing Systems, 2023, 36: 78240-78260.

[7] Wang G, Liu W, He Y, et al. Eegpt: Pretrained transformer for universal and reliable representation of eeg signals[J]. Advances in Neural Information Processing Systems, 2024, 37: 39249-39280.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents NeuroRVQ, a large brainwave model (LBM) for EEG signals that focuses on a multi-scale tokenizer using temporal convolutions with varying kernel sizes for frequency band extraction, hierarchical residual vector quantization (RVQ) codebooks (one per scale), and a phase- and amplitude-aware loss for training. The tokenizer aims to enable high-fidelity reconstruction across frequency bands (delta to gamma) for generative masked modeling. Evaluated on reconstruction error and four downstream BCI tasks (e.g., emotion recognition, seizure detection), it claims up to 15% accuracy gains over existing LBMs like LaBraM, BIOT, and NeuroGPT, positioning it as a foundation for neural decoding and multimodal integration.

### Strengths
The paper addresses a timely challenge in EEG foundation models: improving tokenization to capture multi-scale frequency dynamics, which is crucial for noisy, complex brain signals. The multi-scale convolution and RVQ design incorporate domain knowledge (e.g., EEG frequency bands), potentially aiding reconstruction fidelity. The phase/amplitude loss is a thoughtful addition grounded in signal processing. Experiments cover reconstruction metrics and downstream tasks, with claims of superior performance, and the work highlights scalability for LBMs.

### Weaknesses
Novelty is marginal: The tokenizer builds directly on RVQ from VQ-VAE (Esser et al., 2020) and multi-scale convolutions common in EEG models (e.g., EEGNet's depthwise convolutions for bands; Lawhern et al., 2018), without introducing new architectural or theoretical elements—e.g., the hierarchical codebooks per scale are a straightforward extension, and the loss is similar to phase-aware objectives in audio (e.g., EnCodec).

### Questions
1. The RVQ uses one codebook per frequency scale—how was the number of scales (e.g., delta-gamma) chosen, and what happens if bands overlap or vary by task? 

2. Reconstruction error is lower than baselines, but does this translate to better perceptual quality? 

3. Downstream gains are up to 15%—are these consistent across subjects/datasets, or driven by specific ones? Details on variance and p-values would help assess robustness.

4. The model is for generative masked modeling, but how does it compare to non-codebook tokenizers (e.g., BIOT's transformer) in masking ratios or pretraining efficiency? 

5. For broader impact, test on non-EEG biosignals (e.g., ECG)—does the multi-scale approach generalize, or is it EEG-specific? 

I will consider raising my score if all questions and concerns are solved.

### Soundness
2

### Presentation
3

### Contribution
2
