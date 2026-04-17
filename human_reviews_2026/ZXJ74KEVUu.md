# FM-SIREN & FM-FINER: Nyquist-Informed Frequency Multiplier for Implicit Neural Representation with Periodic Activation

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Existing periodic activation-based implicit neural representation (INR) networks, such as SIREN and FINER, suffer from hidden feature redundancy, where neurons within a layer capture overlapping frequency components due to the use of a fixed frequency multiplier. This redundancy limits the expressive capacity of multilayer perceptrons (MLPs). Drawing inspiration from classical signal processing methods such as the Discrete Sine Transform (DST), we propose FM-SIREN and FM-FINER, which assign Nyquist-informed, neuron-specific frequency multipliers to periodic activations. Unlike existing approaches, our design introduces frequency diversity without requiring hyperparameter tuning or additional network depth. This simple yet principled modification reduces the redundancy of features by nearly 50% and consistently improves signal reconstruction across diverse INR tasks, including 1D audio, 2D image regression, 3D shape fitting, and neural radiance field (NeRF) synthesis, outperforming their baseline counterparts while maintaining efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Existing periodic activation-based implicit neural representations (INRs) like SIREN and FINER suffer from hidden feature redundancy, as all neurons in a layer share a fixed frequency multiplier. Drawing inspiration from classical signal processing, this paper introduces FM-SIREN and FM-FINER, which assign a unique, Nyquist-informed, neuron-specific frequency multiplier to each activation. This simple modification introduces frequency diversity without requiring hyperparameter tuning and reduces feature redundancy. As a result, the proposed models consistently improve signal reconstruction quality across diverse tasks, including 1D audio, 2D images, and 3D shapes.

### Strengths
1. Interesting Idea

The proposed solution, assigning neuron-specific frequencies based on the Nyquist theorem, is simple, elegant, and directly validated.

2. Strong Empirical Results

The method is evaluated across four different data modalities: 1D audio, 2D images, 3D shapes, and NeRF.

### Weaknesses
1. The general idea is not well formulated. The introduction of the model should be rewritten to focus on the problem and the proposed solution.
2. Figure 7b shows that the performance of both FM-SIREN and FM-FINER degrades when the network depth is increased beyond three layers.
3. Lack of competition with:
- SPDER: Semiperiodic Damping-Enabled Object Representation
- FreSh: Frequency Shifting for Accelerated Neural Representation Learning
4. The motivation of the paper and Figure 1 is weak. Authors should present results on more complicated images, with a larger network.
5. Authors do not use video data for evaluations. 
6. The lack of a direct visualization of the proposed method.

### Questions
1. Why does the method's performance degrade in networks deeper than three layers? 
2. How does assigning diverse frequencies compare to other methods, like FreSh, that find a single optimal frequency? 
3. Why was the method not evaluated on video data, and how would the Nyquist multipliers be defined for video? 
4. Can the authors provide a diagram visualizing how different frequencies are assigned to each neuron in a layer?
5. Will the implementation code be made publicly available to ensure reproducibility?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces FM-SIREN and FM-FINER to address hidden feature redundancy. These networks improve frequency diversity by applying Nyquist-informed, neuron-specific frequency multipliers to periodic activations, all without requiring extra network depth or hyperparameter tuning. Experiments across 1D audio, 2D images, 3D volumes, and NeRF show consistent gains over SIREN/FINER, with ~50% reduction in hidden-feature redundancy.

### Strengths
1. The proposed method, the frequency multiplier, is simple yet effective for addressing the problem of hidden feature redundancy.

2. Experimental results across various INR tasks demonstrate consistent improvements over SIREN/FINER.

### Weaknesses
1. The epoch count seems too low for typical INR experiments, raising concerns that the network may not have fully converged.

2. The proposed Nyquist-informed, neuron-specific frequency multipliers may not be well-suited for low-frequency signals. For example, when fitting a very smooth $512\times512$ image, the resulting Nyquist frequency ($f_{Nyquist}=256$) could cause the network to introduce high-frequency artifacts.

3. In the Neural Radiance Field experiments, the improvement in reconstruction quality from FM-SIREN/FINER appears marginal. Furthermore, the experimental setup is confusing. For instance, the justification for discretizing the volume into a $100\times100\times100$ resolution is unclear. It is also not explained why FM-SIREN/FINER would reduce computation time compared to SIREN/FINER, given that, theoretically, the time per epoch should be identical.

4. The proposed method may be useful for signal representations where the Nyquist frequency is known a priori. However, for tasks where this prior is unknown, such as NeRF or solving PDEs, empirical hyperparameter tuning may still be necessary. How to choose a proper $f_{Nyquist}$?

### Questions
1. What are the performance metrics after the networks have fully converged?

2. Do FM-SIREN/FINER still provide performance gains when fitting a very smooth image?

3. Given that the hyperparameters ($\omega_k$) are determined by the Nyquist-informed, neuron-specific frequency multipliers, why is $\omega_0$ still listed as a hyperparameter in Appendix Table 6? Then, what exactly are the hyperparameter settings for FM-SIREN/FINER across the various tasks?

4. Are the proposed frequency multipliers applied only to the first layer, or to all hidden layers? This is not clearly specified in the paper.
See Weaknesses for other questions.

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
The paper addresses the subotimality of modern implicit neural representations (INR) with periodical activations (SIREN, FINER) related to suboptimal representational capacity. More specifically, it is observed that the architectures in question suffer from feature redundancy caused by a fixed frequency multiplier in the activation layer. The improvement proposed implies using adaptive frequency multiplier in the SIREN layers (Nyquist sampling rate determined by the layer width), which allow a more expressive representation learning. The paper then comprehensively compares the proposed change to the other periodical INRs, reporting mostly minor improvements on the image and surface overfitting and neural rendering experiments. 

The overall proposal is simple and elegant in spirit, and introducing such altertantive easy-to-integrate initialisation to achieve minor yet consistent improvements across tasks is an attractive feature. However, the quantitative results achieved are rather minor and might not qualify for a top-tier conference contribution, especially since the experimental protocol for the experiments where the method brings most improvement is related to pure overfitting rather than true test-time performance. On the neural rendering task, the performance difference is negligible. This, combined with also some missing discussion of related work, contribute to the overall review leaning towards a reject. Fixing the aforementioned issues (improving quantitative results on unseen image pixels, stronger results in neural rendering and surface reconstruction experiments, extending the discussion to include most recent papers on INR optimisation) will contribute to a stronger paper and more positive outlook.

### Strengths
**S1. Periodic activation analysis and conceptual contribution.** The paper introduces a simple yet well-grounded idea for improving the efficiency and expressivity of implicit neural representations (INRs) through a principled initialization of periodic activation layers. The approach is conceptually clear and supported by solid motivation from sampling theory, offering an elegant bridge between signal processing intuition and neural representation learning.

**S2. Manuscript quality.** The paper is clearly written and well-organised. The exposition of the method, intuition, and experimental results is coherent and easy to follow, making the technical contribution accessible to a broad audience.

### Weaknesses
**W1. Weak quantitative results.** This is the primary weakness of the paper. To the best of my understanding, the main results are obtained on __overfitting__ experiments, where the model is trained and evaluated on the same signals, and its generalisation ability (for example, to unseen pixel locations) is not demonstrated. A fairer evaluation would follow the protocol of [1], where models are assessed on held-out subsets of pixel coordinates. More importantly, the results on a **practically relevant task** (NeRF) show only marginal improvements over the baseline and remain far from the state of the art. One way to strengthen the paper would be to demonstrate improved performance on a more challenging benchmark, such as the MipNeRF-360 [2] dataset and its corresponding baseline.

**W2. Missing related work.** The paper omits discussion of several closely related studies, notably BACON [3], FreSH [1] and TUNER [4]. FreSH addresses frequency allocation in implicit neural representations and introduces an improved strategy for selecting frequency ranges, supported by experiments that overlap in scope with the present work. More broadly, BACON [3] introduces coordinate networks with explicit frequency control enabling multi-scale scene representation. Finally, TUNER [4] focuses on stabilising the training of sinusoidal networks through adaptive frequency modulation to achieve better training stability. Including a comparison and discussion of these approaches would further clarify how the proposed method fits within and advances the recent literature on frequency-aware implicit representations.

**W3. Discussion of positional encoding and layer width could be expanded.**
The connection between the proposed method and positional encoding (PE) could be elaborated further, as FM-SIREN behaves similarly to PE in shallow networks. The authors note that PE requires selecting an appropriate encoding length L (lines 144–146), while in FM-SIREN the hidden-layer width implicitly defines the number of frequency multipliers and thus the spectral span. Although the paper includes an ablation on network width, extending this analysis to discuss its effect on expressivity and stability—and clarifying the analogy to PE—would strengthen the exposition and help readers better appreciate the design choices.


References:

[1] Kania, A., Mihajlovic, M., Prokudin, S., Tabor, J., & Spurek, P. (2024). FreSh: Frequency Shifting for Accelerated Neural Representation Learning. ICLR 2025.

[2] Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P., & Hedman, P. (2022). Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR 2022.

[3] Lindell, David B., Dave Van Veen, Jeong Joon Park, and Gordon Wetzstein. "Bacon: Band-limited coordinate networks for multiscale scene representation." CVPR 2022.

[4] Novello, Tiago, Diana Aldana, Andre Araujo, and Luiz Velho. "Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks." CVPR 2025.

### Questions
The main questions are:

**Q1**. The results on NeRF and other tasks suggest limited practical improvement. Could the authors clarify how the proposed method enhances **real-world*** applicability or generalisation (e.g., performance on unseen pixels or multi-view settings) beyond overfitting experiments? If faster convergence is one of the main advantages, could the authors also demonstrate similar benefits when training on real-world datasets, such as MipNeRF-360 [2], and its corresponding baseline model?

**Q2**. What are the key reasons for the **marginal gains** observed on the NeRF benchmark – do they stem from architectural constraints, limited spectral benefit, or experimental setup? 

The authors are welcome to elaborate further on other points raised in the weaknesses section, such as a more direct comparison to FreSh (both conceptually and practically), and an expanded discussion of the relationship between positional encoding, layer width, and spectral span.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses feature redundancy in implicit neural representations (INRs) that use periodic activations. Many INR uses the same activation function for all the layers, which limits the frequency diversity.This paper introduces FM-SIREN and FM-FINER, two new Nyquist-informed Implicit Neural Representation (INR) architectures designed to overcome redundancy in periodic activation networks such as SIREN and FINER.  Experiment result on 1D audio, 2D image, 3D shape, and NeRF synthesis, showing consistent performance gains and efficiency improvements.

### Strengths
This introduces frequency diversity without requiring hyperparameter tuning or additional network depth. It achieves a 49.92% and 50.43% improvement in the Frobenius norm of the covariance matrix compared to their baselines. It also shows a strong performance in reconstruction on the Kodak and BSDS500 datasets.

### Weaknesses
[1] The Discrete Sine Transform (DST) assumes the odd symmetry at the boundaries: How do you confirm that these datasets satisfy these assumptions?

[2] The Nyquist Sampling Theorem requires uniform sampling. However, INRs don’t use uniformly sampled signals in the classical sense. Instead, they approximate a continuous mapping. Even though the training data might be uniformly spaced in a dataset (e.g., pixels or audio samples), the INR’s internal representation and evaluation grid can be nonuniform or adaptive depending on optimization dynamics, loss weighting, or coordinate encodings. How do you make sure that Nyquist Sampling Theorem can apply well to INR?

[3] The method required to apply in the frequency domain, can it be applied to the non-frequency domain?

[4] Even though the dataset covers from 1D to 3D. How is its performance in the PDE dataset?

[5] More recent baselines should be included. For example, VI^3NR [r1], FINER++ [r2]

[r1]Koneputugodage, Chamin Hewa, et al. "VI^ 3NR: Variance Informed Initialization for Implicit Neural Representations." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025
[r2] Zhu, Hao, et al. "FINER++: Building a Family of Variable-periodic Functions for Activating Implicit Neural Representation." arXiv preprint arXiv:2407.19434 (2024).

### Questions
See the Weakness

### Soundness
2

### Presentation
2

### Contribution
2
