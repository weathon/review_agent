# STEP-VQ: Sequence-model Agnostic Frame-level Inference with VQ-VAE for Model-based Reinforcement Learning

- Avg Score: 2.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2

## Abstract
Model-based reinforcement learning (MBRL) from pixels often encodes frames into discrete latent variables that form tokens for sequence model backbones to learn world model dynamics. Previous work adopts two main approaches, each facing distinct limitations. Categorical bottlenecks enable fast frame-level prediction by flattening spatial features into categorical distributions, but suffer from explosive parameter growing with resolution and code dimension. Conversely, vector-quantised variational autoencoder (VQ-VAE) methods achieve parameter efficiency through codebook quantisation but require slow token-level autoregressive prediction within frames, shifting computational complexity to the dynamics model and producing longer sequences that slow training and inference.
  
  We propose STEP-VQ, a novel frame-level VQ-VAE-based world model that enables prediction of entire frames through single forward passes. STEP-VQ follows the latent-imagination paradigm with two components: a world model (VQ-VAE + sequence model) and a behaviour policy. The approach is sequence-model agnostic, working with both Mamba-2 and Transformer architectures without modifications. Our key insight is that fine-grained spatial structure preservation may be unnecessary for effective behaviour learning in latent space, as temporal dynamics can implicitly capture spatial patterns through frame-level prediction. We provide rigorous theoretical analysis grounded in variational inference, showing how our training objective emerges from evidence lower bound (ELBO) optimisation and why Kullback--Leibler (KL) divergence formulations enable superior performance through bidirectional optimisation.
  
  On Atari-100k, STEP-VQ achieves competitive performance whilst dramatically improving efficiency: 11.2× faster training than a strong VQ-VAE based baseline, 4× parameter reduction compared to categorical bottlenecks, and growing advantages at higher resolutions (+27.4\% mean improvement at 96×96). STEP-VQ reaches superhuman performance on 9 games versus 8 for categorical methods, with KL divergence providing 24.5\% improvement over cross-entropy baselines. These results demonstrate that frame-level discrete quantisation offers a practical path to efficient, scalable MBRL using modern sequence architectures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a method for efficient world modeling that combines a VQ-VAE autoencoder with a sequence model to enable frame-level prediction of discrete latent representations, avoiding token-level autoregression. The approach models temporal dynamics by predicting the full grid of VQ code distributions for the next frame in a single forward pass and aligns these predictions with encoder posteriors through a KL-divergence-based dynamics loss derived from variational principles. Evaluated on the Atari 100K benchmark, STEP-VQ achieves performance comparable to categorical and autoregressive VQ-based baselines while providing 11× faster training, 4× fewer parameters, and improved scalability at higher resolutions, maintaining efficiency across both Transformer and Mamba-2 sequence model architectures.

### Strengths
1. The paper demonstrates a substantial training speedup compared to IRIS-like autoregressive VQ-VAE world models, effectively addressing a known computational bottleneck in VQ-based MBRL.

2. The authors provide a well-structured categorization of existing approaches (categorical bottlenecks vs. VQ-VAE) and clearly identify the trade-offs motivating the proposed method.

3. The inclusion of both Transformer and Mamba-2 sequence models in experiments enhances the robustness of the evaluation and shows the method’s compatibility with different sequence modeling paradigms.

### Weaknesses
1. **Unclear methodological exposition:** The description of the proposed approach is mathematically dense and lacks intuitive implementation detail. The role of distributions and losses is not directly linked to concrete architecture design, and the core pipeline (as shown in Figure 6) should appear in the main text to aid understanding.

2. **Limited performance gain over categorical bottlenecks:** While efficiency improves compared with IRIS-like methods, the method does not show clear performance advantages compared to strong CB-based baselines such as DreamerV3 or STORM. Additionally, computational cost comparisons with these CB models are missing, leaving STEP-VQ's relative benefit uncertain.

3. **Insufficient detail in high-resolution comparisons:** The implementation setup of categorical baselines in higher-resolution experiments is under-specified, which weakens the claim that STEP-VQ consistently outperforms CB methods at larger resolutions.

4. **Restricted scalability analysis:** Although the paper highlights resolution scalability, the experiments stop at 96x96 inputs, far below the resolutions addressed by recent models like Dreamer4 [1] (up to 640x360), limiting the evidence for true scalability across visual domains.

[1] Hafner, Danijar, Wilson Yan, and Timothy Lillicrap. "Training agents inside of scalable world models." arXiv preprint arXiv:2509.24527 (2025).

### Questions
See the weaknesses

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes STEP-VQ, a method combining VQ-VAE with frame-level prediction for model-based RL. While STEP-VQ achieves 11.2× speedup over IRIS and maintains comparable performance on the Atari-100k benchmark with fewer parameters. However, given the mediocre performance, limited baseline comparisons, narrow experimental scope, unvalidated theoretical assumptions, and limited novelty, **I recommend rejection**.

### Strengths
The paper proposes replacing cross-entropy with KL divergence for VQ-based methods and demonstrates a 24.5% performance improvement with theoretical justification.

### Weaknesses
* The experimental performance is underwhelming. There already exist many model-based RL methods that achieve both faster overall training time and superior performance. 
* The paper compares against very limited baselines, I suggest comparing with state-of-the-art VQ-based methods such as Simulus [1], REM [2], and $\Delta$-IRIS [3], as well as other categorical-based methods like TWISTER [4], DyMoDreamer [5], and STORM [6].
* The evaluation is restricted to Atari-100k only. The paper lacks validation on other standard MBRL benchmarks such as Crafter, DeepMind Control Vision.
* The core theoretical assumption $I(z_t[i, j]; z_t[i', j'] | h_t) ≈ 0$ is never empirically measured or validated.
* **Limited Novelty**: Independent prediction of each token in discrete latent world models has already been explored by many works (e.g., REM, Simulus, Transformer World Model [7]), and most categorical-based methods also independently predict each token.

**References**

[1] Uncovering Untapped Potential in Sample-Efficient World Model Agents.

[2] Improving Token-Based World Models with Parallel Observation Prediction.

[3] Efficient World Models with Context-Aware Tokenization. 

[4] Learning Transformer-based World Models with Contrastive Predictive Coding

[5] DyMoDreamer: World Modeling with Dynamic Modulation

[6] STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning

[7] Improving Transformer World Models for Data-Efficient RL

### Questions
* **Absolute training time**: What is the actual wall-clock training time (in hours) for a single Atari-100k environment under the experimental settings presented in this paper?
* **Unfair capacity comparison**: STEP-VQ uses a codebook size of 64 while the CB baseline uses categorical classes of 32. Could the performance improvement simply be attributed to this larger representation capacity rather than the method itself?
* **High-resolution scalability insights**: The improved performance at 96×96 resolution seems intuitive given the parameter scaling properties. What unique insights or theoretical understanding do the authors provide beyond the obvious architectural consequence?
* **Vector-based observations**: Would STEP-VQ outperform CB in environments with vector-based observations (e.g., DeepMind Proprio Control) where spatial quantization may not be necessary?
* **Comparison with SOTA**: What advantages does STEP-VQ offer compared to the current state-of-the-art baselines on the Atari-100k benchmark (e.g., TWISTER, EfficientZero V2 [1], EDELINE [2])?

**References**

[1] EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data

[2] EDELINE: Enhancing Memory in Diffusion-based World Models via Linear-Time Sequence Modeling

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces STEP-VQ, a model-based reinforcement learning approach combining vector-quantized VAEs with sequence models to enable frame-level prediction instead of slower token-level autoregression. The key idea is that temporal redundancy between frames can compensate for the loss of fine-grained spatial modelling, thus achieving both efficiency and scalability. The authors claim large speedups over prior VQ-based methods (e.g., IRIS), improved parameter efficiency over categorical bottlenecks, and competitive performance on Atari 100K using both Mamba-2 and Transformer architectures.

### Strengths
- This paper replaces token-level autoregression with frame-level prediction, yielding large training speedups (up to ~11x) while keeping VQ-VAE parameter efficiency.
- The method works with both Mamba-2 and Transformers, showing comparable performance without architecture-specific changes.

### Weaknesses
- The dynamics model remains unclear. In Section 2.3.1, the same function $f_\psi$ appears to serve multiple roles: in Eq. (3), it computes a hidden vector from latent codes $z_t$, but in Eq. (4), it seems to output a 3D tensor of logits indexed spatially. Even Figure 6(b) does not fully clarify the mapping between these representations. Two possible interpretations are: (i) $f_\psi$ maps between vector and 3D tensor representations through convolutional and transposed-convolutional layers (less likely), or (ii) each latent code is processed independently by the same model (more likely, but the notation in Section 2.3.1 would then be inconsistent). This ambiguity should be clarified.
- Line 93: The statement about "explosive parameter scaling" may be overstated. The scaling depends on whether encoder output dimensions $H', W'$ grow with input size $H, W$, since additional downsampling could mitigate the effect. Clarifying this assumption would improve accuracy.
- The KL balancing loss resembles the one in DreamerV2 (Hafner et al., 2020, Algorithm 2). This work should be cited explicitly, and the differences between the two formulations should be discussed. Also, the equation in line 253 lacks a label.
- Evaluation setup:
  * The abstract states: "STEP-VQ reaches superhuman performance on 9 games versus 8 for categorical methods, with KL divergence providing 24.5% improvement over cross-entropy baselines." While this highlights per-game counts and a within-method loss comparison, Tables 1-2 show only small mean gains alongside notably lower medians (Mamba: mean +5.4%, median -35.6%; Transformer: mean +3.5%, median -21.7%), suggesting heavier-tailed outcomes and wins concentrated in fewer games. I recommend rephrasing the claim to reflect these results, and explicitly clarifying that the "24.5% improvement" refers to KL vs. cross-entropy **within** STEP-VQ, not versus categorical baselines.
  * Only three seeds per game are used, which is insufficient given the known high variance in Atari 100K. At least five seeds are typically considered a minimum.
  * Section 3 does not clearly state which architectures (Mamba or Transformer) are used in the results being compared. This makes it impossible to compare to the other results at lower resolution.
  * Figure 3 would be more informative if it included the lower-resolution baselines, as done in Tables 6-7. This would better illustrate scalability trends across resolutions.
- A relevant related work is missing: Robine et al. "Smaller World Models for Reinforcement Learning." Neural Process Lett 55, 11397–11427 (2023).
- Minor issues:
  * Line 156: The encoder is referred to as $Enc_\phi$ here but later as the probabilistic model $q_\phi$. The relationship between these should be clarified or unified in notation.
  * Line 191: The additional variable $z^\star_t$ seems unnecessary. Reusing $z^q_t$ would simplify the presentation.
  * Line 196: The indexing notation (e.g., [:, 1:L]) looks like Python slicing, which implies exclusive upper bounds. Using $z^\star_{2:L}$ and $\hat{z}_{1:L-1}$, consistent with 1-based time indexing elsewhere, would improve clarity.
  * Figure 2(a): The legend is confusing.

Overall, the paper makes a promising step toward more efficient world models, but the technical description of the dynamics model and evaluation methodology need clarification before publication. Further experiments (e.g., with more seeds or RNN variants) would strengthen the empirical case.

### Questions
- Have the authors considered evaluating a recurrent (RNN-based) dynamics model?
- Please clarify the dynamics model as described in the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
3
