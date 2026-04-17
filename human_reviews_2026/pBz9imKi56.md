# A Gain for Reconstruction, A Pain for Generation: Exploiting Representation in Visual Tokenization

- Decision: Reject
- Scores: 6, 6, 2

## Abstract
Discrete visual tokenization is a cornerstone of modern auto-regressive (AR) image generation, yet current methods are fundamentally constrained by a trade-off between reconstruction fidelity and semantic expressivity. In this work, we first propose a principled framework for token representation learning based on three pillars: feature alignment with foundation models, structural diversification of the codebook into specialized subspaces, and explicit disentanglement to enforce semantic independence. We materialize these principles in a novel tokenizer, Semantic Subspace Quantization (SSQ), which achieves state-of-the-art image reconstruction. However, this success reveals a critical and previously overlooked paradox: the semantically rich, structured representations that excel at reconstruction cause a significant performance collapse in standard AR generative models. To resolve this Reconstruction-Generation Discrepancy, we introduce a novel tokenizer-generator co-design methodology, systematically adapting the AR model's architecture and training curriculum to harness the multi-faceted nature of SSQ's tokens. Our final, synergistic system effectively alleviates this discrepancy, achieving state-of-the-art performance on high-fidelity reconstruction and generation, demonstrating a new path forward for discrete visual modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new visual tokenizer, Semantic Subspace Quantization (SSQ), built on a principled framework of three pillars: 1) Structural Diversification, which uses multiple factorized codebooks (subspaces) instead of a monolithic one; 2) Explicit Disentanglement, which enforces orthogonality between these subspaces to ensure they learn complementary features; and 3) Feature Alignment, which guides subspaces to learn semantic information by aligning them with features from foundation models like DINOv2 and CLIP.

### Strengths
- The proposed SSQ tokenizer is built on a well-motivated and principled framework. The three pillars of diversification, disentanglement, and alignment provide a systematic way to create a semantically rich and structured token representation, and the results convincingly demonstrate its state-of-the-art reconstruction capabilities.
- This paper throughly investigated the impact of different components in the design, making insightful observations to the community.

### Weaknesses
- In Table 3, the structure divergence seems to lack of a fair baseline with 32768 setting regarding gFID?
- Ablation about hyperparam in noisy training?

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the often-overlooked relationship between visual tokenization quality and autoregressive (AR) image generation performance.  
The authors introduce **Semantic Subspace Quantization (SSQ)**, a tokenizer built on three principles: **feature alignment**, **structural diversification**, and **explicit disentanglement**.  
While SSQ achieves state-of-the-art image reconstruction, it surprisingly worsens AR generation — a paradox termed the **Reconstruction–Generation Discrepancy**.  
To address this, the paper proposes a **tokenizer–generator co-design** strategy, including a **factorized AR head**, **two-stage training**, and **noisy sub-token regularization**, which together restore generation quality and surpass prior AR baselines such as LlamaGen.

### Strengths
- **Novel and well-articulated problem framing.**  
  Identifying the Reconstruction–Generation Discrepancy is a meaningful conceptual contribution that clarifies why better reconstruction can hurt AR generation.

- **Principled and interpretable tokenizer design.**  
  The three-pillar SSQ framework (alignment, diversification, disentanglement) provides a clear structure for improving representation quality.

- **Effective AR co-design.**  
  The proposed factorized AR head and staged training curriculum directly address architectural mismatch and optimization instability, leading to strong empirical gains.

- **Solid empirical performance.**  
  On ImageNet 256×256, SSQ-LlamaGen achieves FID 2.61 vs 3.80 for LlamaGen-L and Inception Score 313.9 vs 248.3, showing both quantitative and qualitative improvements.

- **Comprehensive ablations and honest discussion.**  
  The paper presents negative results (e.g., naïve multi-head classifiers fail) and openly discusses remaining gaps and scalability limits.

- **Readable and well-motivated.**  
  The writing is clear, with good intuition on why semantic alignment helps reconstruction and how disentanglement works.

### Weaknesses
- **Incremental tokenizer innovation.**  
  SSQ combines known techniques (multi-codebook, VFM alignment, orthogonality regularization) into one framework; the novelty lies more in framing than in algorithmic breakthrough.

- **Limited scope of experiments.**  
  Evaluations are confined to ImageNet 256×256 class-conditional AR. The generality to other datasets, text-to-image generation, or diffusion-based methods remains untested.

- **Disentanglement is simplistic.**  
  The squared dot-product loss enforces orthogonality but not true independence between subspaces.

- **Efficiency and scalability not measured.**  
  The factorized AR head increases computation per patch, yet runtime and throughput are not reported.

- **Residual gap persists.**  
  Even after co-design, SSQ-Triple still reconstructs better but generates worse than SSQ-Dual, indicating that the discrepancy is only partially resolved.

### Questions
1. How does SSQ perform when integrated into non-autoregressive or diffusion-based generators?  
   Would the same reconstruction–generation discrepancy appear?
2. What is the computational overhead (training time, inference speed) of the factorized AR head compared to a standard linear head?
3. Have the authors tested robustness or generalization to out-of-domain datasets, given the heavy reliance on DINOv2/CLIP alignment?
4. Could stronger disentanglement measures (e.g., mutual-information-based losses) further improve subspace independence and generation stability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses the long-standing trade-off between reconstruction fidelity and semantic expressivity in discrete visual tokenizers, a crucial component of auto-regressive (AR) image generation. The authors introduce Semantic Subspace Quantization (SSQ), a novel tokenizer built on three principles: Feature Alignment with Foundation Models (e.g., DINOv2, CLIP), Structural Diversification via factorized quantization, and Explicit Disentanglement using an orthogonality loss. SSQ achieves state-of-the-art reconstruction fidelity.

### Strengths
The proposed framework is highly systematic and clearly articulated. The paper provides detailed ablation studies (Tables 3, 4, 5) to validate each component of the SSQ tokenizer and the co-design strategies. The in-depth analysis of the SSQ feature space in the Appendix (Figures 5, 6, 7), demonstrating specialization, affinity, and orthogonality, is well-executed and adds compelling evidence to the paper's claims about representation learning. Quality (Reconstruction Performance): The SSQ tokenizer indisputably sets a new state-of-the-art in reconstruction fidelity (Table 1), beating strong multi-codebook baselines like ImageFolder and TokenFlow.

### Weaknesses
While the paper presents the SSQ framework as a "principled framework," the individual technical components are incremental and have been extensively explored in prior work, diminishing the originality claim:
* Factorized/multi-codebooks are a well-established concept (e.g., RQ-VAE (Lee et al., 2022), ImageFolder (Li et al., 2024b), TokenFlow (Qu et al., 2024)). The paper claims systematization but the concept is not new.
* Aligning VQ codes with features from foundation models (CLIP, DINO) is directly implemented in concurrent works like VA-VAE (Yao et al., 2025), MAETok (Chen et al., 2025), and VQGAN-LC (Zhu et al., 2024b) to enhance semantic content.

The core thesis is that the co-design methodology resolves the Reconstruction-Generation Discrepancy. However, the experimental results contradict this claim:
* Table 6 shows that the SSQ-Triple model, which achieves the lowest reconstruction rFID (best "Gain"), still results in a worse generation FID (3.31) than the SSQ-Dual model (3.11). 
* The gap persists even after applying all co-design strategies (Appendix A.3, L754-758). If the model with the best token representation still yields inferior generation results, the paper has only mitigated the pain, not resolved the fundamental tension. This incomplete resolution fundamentally weakens the central claim and contribution.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2
