# Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Zero-Shot Super-Resolution Spatiotemporal Forecasting requires a deep learning model to be trained on low-resolution data and deployed for inference on high-resolution. Existing studies consider **maintaining** similar error across different resolutions as indicative of successful multi-resolution generalization performance. However, deep learning models serving as alternatives to numerical solvers should **reduce** error as resolution increases. The fundamental limitation is, the upper bound of physical law frequencies that low-resolution data can represent is constrained by its Nyquist frequency, making it difficult for models to process signals containing unseen frequency components during high-resolution inference. *This results in errors being anchored at low resolution, incorrectly interpreted as successful generalization.* We define this fundamental phenomenon as a new problem distinct from existing issues: **Scale Anchoring**. Therefore, we propose architecture-agnostic Frequency Representation Learning. It alleviates Scale Anchoring through resolution-aligned frequency representations and spectral consistency training: on grids with higher Nyquist frequencies, the frequency response in high-frequency bands of FRL-enhanced variants is more stable. This allows errors to decrease with resolution and significantly outperform baselines within our task and resolution range, while incurring only modest computational overhead.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper involves downsampling the data to different resolutions, then using a special Nyquist-normalized frequency position encoding, along with a frequency consistency loss, to achieve super-resolution. Results on two tasks demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper introduces a new approach based on the frequency domain to address the limitations in current Zero-Shot Super-Resolution Spatiotemporal Forecasting methods. It explains the Nyquist frequency limitation of low-resolution data and offers a new perspective on why current methods struggle with high-quality high-resolution predictions.
2. The paper highlights the flaws in traditional multi-resolution techniques, especially the problems of scale anchoring and spectral bias. It proposes Nyquist-normalized frequency representation and frequency-aware loss to solve these issues, which improves the model's performance.
3.The proposed method demonstrates strong performance in weather forecasting and fluid simulation tasks, highlighting the effectiveness of the approach and its potential for practical applications.
4. The proposed method is validated on multiple network architectures, including CNN, Transformer, Mamba, and GNN, demonstrating its versatility and effectiveness across different networks.

### Weaknesses
1. The method shares similarities with ZSSR and other Deep Internal Learning approaches, making it unclear what is fundamentally new beyond the frequency-domain reinterpretation.
2. The experiments are limited to spatiotemporal forecasting; applying the method to standard image super-resolution datasets (like Set5, Set14, BSD100) would provide more comprehensive comparisons.
3. The frequency consistency loss seems similar to existing frequency-domain losses, and its contribution may be limited. Additional ablation studies comparing it with other methods could clarify its impact.

### Questions
In the experiments, the method is tested at only a few discrete scales. Could you test the method over a wider range of scales by adjusting the scale in fixed steps (e.g., 2x, 3x, ... 64x) and report the results? Plotting a performance curve with these varying scales would help demonstrate the method's effectiveness and robustness across a broader range of resolutions.

### Soundness
3

### Presentation
3

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
The authors identify a problem when performing zero-shot super resolution, when a model is trained on low‐resolution data and then deployed on higher‐resolution data, the error does not decrease despite more fine‐grained input. They argue this is because low‐resolution training data limits the maximum representable frequency (via Nyquist), so the model never “learns” higher‐frequency components. They propose a method called Frequency Representation Learning (FRL) that (a) uses resolution‐aligned frequency representations (normalized to account for the Nyquist at each resolution) and (b) incorporates a spectral‐consistency training objective across multiple resolution levels. They show experiments on spatio‐temporal forecasting tasks (zero-shot super‐resolution forecasting) where their FRL approach results in error decreasing with increasing resolution (unlike baselines) and improved frequency response stability in higher‐frequency bands.

### Strengths
1. The identification of the “scale anchoring” problem is interesting and relevant in several fields, such as neural operators. 

2. The proposed method (FRL) is architecture‐agnostic in principle.

3. The experiments show that the inference error can be even lower for zero-shot super resolution. Existing works usually consider the same error as a success.

### Weaknesses
Some of these weaknesses below fall somewhere between questions and weaknesses, so I’ve included them here.

1. A central conceptual concern is that the zero-shot high-resolution inference task itself appears ill-posed under the stated assumptions. If the model is trained purely on low-resolution data, its training distribution is inherently band-limited by the Nyquist frequency of that grid. Consequently, the high-frequency components present in the high-resolution domain are unobserved and unidentifiable during training.
From a signal-theoretic standpoint, there exist infinitely many high-frequency realizations consistent with the same low-frequency field, so the inverse mapping from coarse to fine scales has no unique solution. Therefore, it is unclear in what precise sense the model can achieve lower error at higher resolution, may be to test model generalization in some sense? 

2. It seems that the testing datasets are fairly smooth, and the low-frequency modes dominate the dynamics. Experiments on data with high-frequency or multi-scale content can provide more insights. 

3. FRL seems to rely implicitly on the assumption that the underlying system exhibits scale-consistent spectral structure (as in turbulence or diffusion). Can the authors elaborate more on this?

4. Zero-shot super resolution suffers from aliasing/discretization errors as noted in [1,2,3], there is no discussion on these works and there is no discussion on if and how FRL mitigates these issues. 

[1] Representation Equivalent Neural Operators: a Framework for Alias-free Operator Learning
[2] Discretization-invariance? On the Discretization Mismatch Errors in Neural Operators
[3] Discretization Error of Fourier Neural Operators

### Questions
1. Given that training data are band-limited by the Nyquist frequency, what information allows FRL to improve predictions beyond that spectral limit?

2. Does FRL rely on the implicit assumption that the underlying physical process is scale-consistent or spectrally smooth?

3. Since multiple high-resolution fields can correspond to the same low-resolution sample, how does FRL regularize or constrain the mapping to select a physically meaningful one?

4. What is the theoretical justification for expecting a neural operator trained on low-resolution data to exhibit convergence behavior analogous to that of a numerical solver?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel method called Frequency Representation Learning (FRL) to address the "Scale Anchoring" problem in Zero-Shot Super-Resolution Spatiotemporal Forecasting (ZS-SR STF). The authors point out that existing methods, when trained on low-resolution data, struggle to process high-frequency components beyond the Nyquist frequency of the training data during high-resolution inference, resulting in errors being "anchored" at the low-resolution level and mistakenly interpreted as successful generalization. FRL achieves resolution-invariant frequency embeddings through multi-resolution data construction, normalized frequency representation, and spectral consistency training, significantly reducing high-resolution inference errors. Its effectiveness is validated across multiple architectures and tasks (fluid simulation and weather forecasting).

### Strengths
1、Novel: The paper is the first to clearly identify the "Scale Anchoring" phenomenon and provides a theoretical analysis of its mechanism (frequency blindness and high-frequency error dominance), addressing an under-recognized limitation in existing zero-shot super-resolution research.
2、Strong Generalizability: FRL is architecture-agnostic and can be seamlessly integrated into various mainstream models such as GNNs, Transformers, and CNNs. Experiments show that it significantly improves high-resolution inference accuracy across all tested architectures.
3、Comprehensive Experimental Validation: Extensive tests on multiple tasks (2D/3D fluid simulation and weather forecasting) combined with error metrics and frequency response analysis thoroughly demonstrate the method’s effectiveness and generalization capability.

### Weaknesses
1、Insufficient Computational Overhead Analysis: Although a training complexity increase of approximately 1.1–1.4× is mentioned, the paper lacks detailed discussions on actual training time and memory usage comparisons across different architectures, as well as the storage requirements for multi-resolution data construction.
2、Limited Generalization in Extreme Physical Scenarios: The authors note in the appendix that FRL fails in high-Reynolds-number turbulence (e.g., Re=10^5), but they do not deeply analyze the reasons for failure or propose improvement directions, which limits the method’s applicability in complex physical systems.

### Questions
1、Regarding computational overhead: FRL requires storing and processing multi-resolution data during training. How can the storage cost be balanced against performance gains in practical deployment? Are there compression or dynamic sampling strategies to mitigate storage pressure?
2、Regarding generalization capability: The performance degradation of FRL in high-Reynolds-number turbulence suggests that the method relies on consistent frequency response patterns in physical systems. Is there a plan to incorporate adaptive mechanisms or physical constraints to enhance adaptability to highly nonlinear and multi-scale coupled systems?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper defines and analyzes an overlooked phenomenon, called scale anchoring, in cross-resolution generalization for zero-shot super-resolution. To mitigate this generalization issue, this paper proposes the frequency representation learning method through three steps from data and optimization perspectives. The results have shown the superiority of the proposed method compared to existing methods on fluid simulation and weather forecasting datasets. 

Contributions:

This observation is new and provides an analysis of why models trained at low resolution fail to improve accuracy at higher resolutions.

### Strengths
- This paper looks into spectral bias-related stuff, which is important in scientific machine learning.

- This paper proposes a new concept, called scale anchoring, in zero-shot super-resolution. 

- This paper provided a theoretical and empirical analysis of this scale anchoring phenomenon. 

- This paper is generally well-written and well-presented.

### Weaknesses
I have several major concerns listed below. 

- First, what is the difference between the defined scale anchoring and the prior concepts, such as spectral bias / discretization-invariance? To me, scale anchoring refers to a failure mode in which a model trained on low-resolution data fails to get good accuracy at finer resolutions, due to missing high-frequency information beyond the coarse data’s Nyquist limit. Spectral bias also refers to the same things that a NN model cannot learn high-frequency features from data. 

- Also, there is a pretty similar paper that discussed resolution generalization and discretization mismatch errors. What is the difference between this paper and the previous paper [1]? 

- The proposed three-step method covers some tricks from the data and optimization sides. It would be good to have an ablation study to test how much each step contributes to the performance improvement. Also, it seems the third step (frequency loss) has already been seen in many prior work [2,3] that uses spectral loss to mitigate the spectral bias issues. 

- I feel like it would strengthen the paper by providing a comparison of computational cost versus accuracy improvement. The computational cost of using low-resolution grids should be small, and with the proposed method, one can get better performance on finer grids. Then, people can have a better sense of the tradeoffs. 

 ---

**Refs:**

[1] Gao, Wenhan, et al. "Discretization-invariance? on the discretization mismatch errors in neural operators." The Thirteenth International Conference on Learning Representations. 2025.

[2] Chattopadhyay, Ashesh, Y. Qiang Sun, and Pedram Hassanzadeh. "Challenges of learning multi-scale dynamics with AI weather models: Implications for stability and one solution." arXiv e-prints (2023): arXiv-2304.

[3] Saccardi, Carlo, et al. "Assessing the Geographic Generalization and Physical Consistency of Generative Models for Climate Downscaling." arXiv preprint arXiv:2510.13722 (2025).

### Questions
- Is there any stability issue when you extrapolate beyond the training Nyquist range?

- On page 6, line 321, do you need bold for “Notably”?

- In Table 3, why do you have some uncommon grids like 43^3?

### Soundness
3

### Presentation
2

### Contribution
3
