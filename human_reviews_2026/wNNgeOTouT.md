# One-for-All Model Initialization with Frequency-Domain Knowledge

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Transferring knowledge by fine-tuning large-scale pre-trained networks has become a standard paradigm, yet the knowledge of pre-trained model is tightly coupled with monolithic architecture, which restricts flexible reuse across models of varying scales.
In response to this challenge, recent approaches typically resort to either parameter selection, which fails to capture the interdependent structure of this knowledge, or parameter prediction using generative models that depend on impractical access to large network collections. In this paper, we empirically demonstrate that a model's foundational, task-agnostic knowledge -- its "learngene" -- is encoded within the low-frequency components of its weights, and can be inherited efficiently by downstream models.
Based on this insight, we propose FRONT (FRequency dOmain kNowledge Transfer), a novel framework that uses the Discrete Cosine Transform (DCT) to isolate the low-frequency "learngene". This learngene can be seamlessly adapted to initialize models of arbitrary size via simple truncation or padding, a process that is entirely training-free. For enhanced performance, we propose an optional low-cost refinement process that introduces a spectral regularizer to further improve the learngene's transferability. Extensive experiments show that FRONT achieves the state-of-the-art performance, accelerates convergence by up to 15✖ in vision tasks, and reduces training FLOPs by an average of 40.5\% in language tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes FRONT, a frequency-domain method for initializing neural networks by extracting low-frequency components (via Discrete Cosine Transform, DCT) from pretrained weights. The authors argue that these components—termed “learngenes”—capture task- and architecture-agnostic knowledge, allowing models of various sizes (e.g., different depths or widths) to inherit such knowledge through a training-free process. They also propose FRONT+, which introduces a spectral regularization term to refine these learngenes through a brief fine-tuning process. The authors report experimental performance across: vision models (DeiT, ResNet) and language models (BERT, RoBERTa, GPT2) via various downstream datasets (classification, detection, segmentation, GLUE benchmark). They claim up to 15× faster convergence and 40% less FLOPs compared to training from scratch.

### Strengths
1. The use of DCT to extract transferable low-frequency components for cross-architecture initialization is interesting. It provides a fresh perspective on model reuse and transfer learning.
2. The paper offers experiments on both vision and language domains, demonstrating broad applicability and consistent improvements. 
3. The proposed method operationalizes the abstract “learngene” concept in a concrete, reproducible way—turning a theoretical notion into a working initialization strategy.

### Weaknesses
1. It lacks rigorous theoretical analysis to support why low-frequency components encode general knowledge. The claim that low-frequency weights correspond to “learngenes” remains speculative.
2. Unlike images, there is no inherent spatial ordering of weight indices. Applying DCT assumes a kind of smoothness across indices that is not theoretically justified.
3. The paper doesn't show that low-frequency weights correspond to smoother or more general representations in the activation space.
4. Although the authors test across multiple datasets, the analysis lacks examination of negative cases—when and why the method fails. There's also little discussion on transfer to fundamentally different architectures.

### Questions
1. Why DCT rather than Fourier, Wavelet, PCA, or SVD? The authors only cite DCT’s “energy compaction” property from image compression.
2. What is the definition of high/low frequency in weight space?

### Soundness
2

### Presentation
2

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
This paper proposes FRONT, a framework that extracts task-agnostic knowledge from pre-trained models by decomposing weights into the frequency domain via DCT and isolating low-frequency components as "learngenes" for initializing models of different sizes. The key empirical observation (Figure 1) shows that low-frequency components remain stable across different model scales and downstream tasks, while high-frequency components are volatile and task-specific. Two variants are proposed: FRONT for direct zero-cost extraction, and FRONT+ with frequency regularization for refinement. Extensive experiments on vision and language tasks demonstrate substantial improvements.

### Strengths
1. The concrete instantiation of learngene as low-frequency components is intuitive and creative, with convincing evidence in Figure 1 demonstrating stability of low-frequency components across models and tasks.

2.  FRONT's zero-cost extraction and flexible padding/truncation mechanism make it substantially more practical than training-based methods like GHN-3 and WAVE.

3. The evaluation spans ViT/ResNet/MLP/CNN architectures, multiple datasets, both vision and language domains, and systematic ablations (Figure 5, Table 11) that strengthen the empirical claims.

### Weaknesses
1. The frequency ratio r varies by model size (2.2M/3.2M/13.0M for Ti/S/B in Table 1) without principled justification, suggesting $r$ is model-size dependent. This systematic issue is not explored, and hyperparameters like decay rates $γ_d$ in Eq. 6 lack principled selection guidelines.

2. When comparing with training-based methods (WAVE/TLEG), FRONT+ also requires 150 epochs of training, so these should be evaluated separately from FRONT's direct extraction.

3. In Table 3, FRONT occasionally underperforms LiGO (e.g., Flowers: 92.9 vs 94.2), indicating instability; Tables 4-5 show large improvements on detection/segmentation but lack direct comparison with other initialization methods beyond random initialization.

4. Applying 3D-DCT across layer/input/output dimensions (mixing different semantic meanings) without per-layer processing warrants explanation—why not apply DCT separately to each layer? 

5. Evaluation only covers homomorphic scaling (BERT-B→BERT-S) without heteromorphic transfer (e.g., BERT→GPT)

### Questions
1. Why do low-frequency components specifically encode task-agnostic knowledge, and why DCT over other transforms like Fourier or wavelets? Figure 1 provides only empirical observation, not principled justification.

2. Why does LiGO fail (Table 1, "/") without explanation, and why is knowledge distillation missing as a baseline in vision tasks despite being used for language tasks?

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
3

### Summary
The paper proposes FRONT, a training-free initializer that treats a pretrained network’s weights in the frequency domain. It applies a 3D-DCT to each weight tensor, keeps only the low-frequency coefficients as the compact “learngene,” and reconstructs target-size weights for new models by simple zero-padding/truncation and IDCT. An optional FRONT+ step lightly fine-tunes a source model with a spectral regularizer to make those low-frequency components even more transferable. Experiments show faster convergence and substantial compute savings on different tasks and models.

### Strengths
- The proposed method extracts a low-frequency learngene and uses padding or truncating to initialize a variety of models across ViT and CNN. It generalizes well across different depths and width, with minimal computation needed.

- The proposed method speeds up convergence and cuts compute versus scratch or learned-transform baselines.

### Weaknesses
- The motivation behind the design is unclear. Why stacking weights across layers and then conduct 3D DCT, what if do this process on 2D weights and then use some selective process to get the learngene?

- The presentation of the experimental results is not that clear, and the experimental settings are concernable. For instance, in table 1, it’s unclear to see what’s the base model in each block is used for initialization? And the results reported in the way of 10-epoch accuracy is not optimal. It should report the final accuracy with the number of epochs of convergence. I would expect a much faster convergence rate of the proposed method versus trivial initialization.

- Results not much improvement over WAVE in Table 1,2,3. Also, why the convergence rate of the proposed initialization method that uses pre-trained knowledge does not show notable advantages over traditional methods?

- Lack ablation studies on deciding the ratio $r$.

### Questions
- What if the architecture is different? For example, the transferring standard attention block to the parallel attention block in [1]?

- There are tons of pre-trained models in the model zoo, any principles to select one as the learngene to initialize future trianing?

- What’s the design choice of using DCT, what about DFT, DWT and other basis?



1.	Dehghani, Mostafa, et al. "Scaling vision transformers to 22 billion parameters." International conference on machine learning. PMLR, 2023.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a training-free learngene paradigm and demonstrates that a model’s fundamental, task-agnostic knowledge is encoded in the low-frequency components of its weights and can be effectively inherited by downstream models. Building on this, it introduces FRONT (Frequency domain Knowledge Transfer), a framework that accelerates model convergence.

### Strengths
1.The motivation of the paper is clear, and the writing is generally well-structured.
2.The paper provides evidence that task-agnostic knowledge resides in a model’s low-frequency components—an intuitively plausible and insightful finding. It also instantiates the learngene concept as low-frequency representations that can be readily extracted from the model.
3.The experiments are generally thorough and demonstrate the effectiveness of the proposed method.

### Weaknesses
Please refer to the Questions section below.

### Questions
1.The paper only provides empirical evidence for the knowledge-carrying role of low-frequency components; it appears to lack theoretical support for the claim that “low-frequency components encode task-agnostic knowledge.”
2.FRONT+ enhances low-frequency knowledge by suppressing high-frequency components. But are high-frequency components entirely without transfer value? For example, between similar tasks (e.g., image classification and fine-grained classification), might high-frequency components carry reusable fine-detail information? It would be helpful to further analyze the potential role of high-frequency components.
3.The adaptation of learngene is implemented solely via “truncation / zero padding,” without considering how architectural differences between the source and target models (e.g., differing numbers of Transformer layers or CNN convolutional kernels) affect knowledge mapping. For instance, when the target model has many more layers than the source model, could zero-padded high-frequency regions introduce invalid information and adversely impact model initialization?

### Soundness
3

### Presentation
3

### Contribution
3
