# Understanding Post-Training Structural Changes in Large Language Models

- Avg Score: 6.00
- Decision: Reject
- Scores: 8, 4, 4, 6, 8

## Abstract
Post-training fundamentally alters the behavior of large language models (LLMs), yet its impact on the internal parameter space remains poorly understood. In this work, we conduct a systematic singular value decomposition (SVD) analysis of principal linear layers in pretrained LLMs, focusing on two widely adopted post-training methods: *instruction tuning* and *long-chain-of-thought (Long-CoT) distillation*. Our analysis reveals two consistent and unexpected structural changes:**(1) a near-uniform geometric scaling of singular values across layers**, which theoretically modulates attention scores; and **(2) highly consistent orthogonal transformations are applied to the left and right singular vectors of each matrix.** Disrupting this orthogonal consistency leads to catastrophic performance degradation. Based on these findings, we propose a simple yet effective framework that interprets post-training as a reparameterization of fixed subspaces in the pretrained parameter space. Further experiments reveal that singular value scaling behaves as a secondary effect, analogous to a temperature adjustment, whereas the core functional transformation lies in the coordinated rotation of singular vectors. These results challenge the prevailing view of the parameter space in large models as a black box, uncovering the first clear regularities in how parameters evolve during training, and providing a new perspective for deeper investigation into model parameter changes.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates how the weights of large language models change under instruction fine-tuning and Long-CoT distillation. The authors propose that the post-finetuning weight matrix can be approximately expressed as 
$$ W_{\text{post}} = U_{\text{post}} \Sigma_{\text{post}} V_{\text{post}}^{T} \approx (U_{\text{base}} Q)   \alpha \Sigma_{\text{base}} (V_{\text{base}} Q)^{T} $$
suggesting that fine-tuning primarily induces a structured transformation on the singular vectors of the base model, characterized by a rotation $Q$ and a rescaling $\alpha \Sigma_{\text{base}}$.

### Strengths
1) The decomposition provides a clear and interpretable picture of how instruction fine-tuning modifies model representations.

2) The scaling relation $\alpha \Sigma_{\text{base}}$ is convincingly shown in the combination of Figure 2 and Table 1.

3) The claim that a single rotation matrix $Q$ is sufficient to describe the transition is less visually convincing in Figure 3b, but Table 2 provides substantial additional support.

### Weaknesses
I am not convinced by Figure 3b, as, to my understanding, the simple assumption that fine-tuning leads to only small changes in the weight may produce this figure. 

I would suggest adding the zero values to Table 2 to enhance readability.

### Questions
No Questions.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper analyzes how post-training methods (instruction tuning and long chain-of-thought distillation) change the weight matrices inside large language models. Using SVD on attention and FFN layers, the authors report two regularities: (1) singular values are scaled by nearly uniform factors across layers, and (2) the left and right singular vector bases undergo highly consistent, shared rotations.

### Strengths
**Clear conceptual framing**: The paper offers a simple and intuitive way to describe post-training dynamics in LLMs. This conceptual framing is easy to understand and potentially useful for future theoretical or empirical work.

**Extensive and systematic experimentation**: The authors support their claims with a wide range of experiments across multiple model families and sizes.

### Weaknesses
**Potential over-interpretation**: Because post-training typically induces only small parameter drift compared to pre-training, the observed “near-identity rotation” and “near-uniform scaling” could be calibration artifacts rather than evidence of a deep mechanism. The evidence also leans on visual flatness and relative comparisons (SVSM, NF). 

**Mechanistic gap**: The work characterizes what changes (rotation + scaling) but says little about why.

**Lack of actionable insight**: The paper does not concretely demonstrate how these observations can be operationalized. Although it hints at possible implications, the ideas remain at the suggestion level.

### Questions
Minor Errors:

Line 91: strongly suggests that -> strongly suggesting that, the subspaces structure -> the subspace structure

Line 81: DeepSeek R1-Distill-Qwen-1.5B -> DeepSeek-R1-Distill-Qwen-1.5B

Line 278: show -> shows

Line 281: are closed to -> are close to

Lines 336, 340, 399, 1220, and 1461: accuracy(%) -> accuracy (%)

Line 620: Svdqunat -> SVDQuant

Line 167: The notation seems sloppy. Here the symbols refer to single model instances, but they are used as if they were sets. This could be clarified to avoid confusion.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a systematic SVD analysis of LLMs before and after post-training (instruction tuning, Long CoT training). The authors identify two recurring structural changes: 1) a near-uniform geometric scaling of singular values, and 2) highly consistent orthogonal transformations applied to the left and right singular vectors. They argue that the orthogonal transformation is the core mechanism, while singular value scaling is a secondary, temperature-like effect.

### Strengths
1. The empirical evaluation is extensive, including multiple model families (Qwen, LLaMA), scales (1.5B to 14B). The scale of the validation is a strength.

2. The discovery of coordinated orthogonal transformations (U_post = U_base Q, V_post = V_base Q) is novel and interesting. The ablation experiments showing that disrupting this coordination causes catastrophic failure are convincing evidence of its importance.

3. The paper is generally well-written and the proposed mathematical framework is clear and intuitive.

### Weaknesses
1. The paper only describes what happens but fails to explain why it happens. The work establishes a strong correlation between the structural patterns and post-training, but does not prove that these patterns are the cause of improved performance.  The deepth of the analysis is constrained, thus making is below the bar of acceptance. 

2. Besides, the findings don't motivate any methods which also limits its practical value. "To perform post-training, one must learn a coordinated orthogonal transformation."  The paper does not provide a new, more efficient algorithm, a better initialization scheme, or a method to control this process. The "potential applications" in Appendix F are speculative and lack empirical validation.

3. The Singular Value Scaling Argument is Weak: Dismissing singular value scaling as a "secondary temperature effect" undermines the work's own findings. The data in Table 1 and Table 5 shows that for reasoning models (M_reasoning), replacing singular values often improves performance. This suggests the scaling is not just a passive effect but is actively detrimental in some cases, and its "correction" is beneficial.

### Questions
NA

### Soundness
3

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
4

### Summary
The paper aims to characterize how base LLMs change their weights during post-training (instruction tuning and long chain-of-thought). For that, SVD decomposition of weight matrices is used. The paper shows that base weights and post-trained weights differ only up to an orthogonal transformation of the left and right eigenvectors and a near-uniform scaling of the singular values. Furthermore, the near-uniform singular value scaling is proposed to act as a temperature parameter for attention matrices, where a higher scaling induces lower attention entropy. Then, the effect of singular value scaling and orthogonal transformation is assessed by reconstructing post-trained models from base models, showing comparable performance. Finally the representational similarity of base and reconstructed LLMs is assessed using Centered Kernel Alignment (CKA).

### Strengths
* The approach is simple and models very accurately the parameter changes in LLMs during post-training.
* The claims are largely supported by the experiments, showing how near-uniform scaling and orthogonal rotations underlie post-training.
* The findings have a big impact on understanding post-training and its learning dynamics, potentially allowing a simplification of current post-training techniques.

### Weaknesses
* The results in Table 1 are a bit circular, first a scaling factor is estimated, then adding this scaling factor restores the benchmark scores. 
Isn’t this trivial since this parameter was estimated to transform a set of weights into the other? I get that the point of this analysis is to show that near-uniform is enough, but could this be shown by efficiently grid-searching/optimizing the scaling factor?

* A crucial control which seems unexplored is whether any fine-tuning (fine-tune a LLM to better know capital cities) would result in such structural changes, or whether this is specific to instruction/reasoning.

### Questions
* Is there any distinguishable feature in the orthogonal transformations that can distinguish instruct vs reasoning models?

* Is there any regularity in the orthogonal transformations across layers and parameter types? Is the space equally rotated, when does it rotate more with respect to the base model? Is it possible to recreate something like Figure 2?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
- The paper investigates how post-training methods, specifically instruction tuning and Long-CoT distillation, affect the internal parameter structure of large language models (LLMs).
- Uses SVD to analyze principal linear layers in the attention and feedforward components to compare pretrained and post-trained models.
- The authors identify two consistent structural effects of post-training: Near-uniform geometric scaling of singular values across layers, and consistent orthogonal transformations on the left and right singular vectors, reflecting coordinated rotations of subspaces.
- They also demonstrate that disrupting these orthogonal transformations can cause severe performance degradation, indicating their functional importance.
- Their experiments mainly cover models based on Qwen2.5-Math-1.5B, as well as additional models such as Qwen2.5-Math-7B, Llama-3.1-8B, Qwen2.5-14B, etc in the appendix, while evaluating on GSM8K, MAtH-500, MMLU, and GPQA

### Strengths
- Presents an interesting and original perspective by analyzing post-training effects through the structural properties of model weights directly.
- Provides clear and consistent empirical findings: the identification of two structural phenomena, singular value scaling and orthogonal vector rotation, across diverse models and post-training methods highlights strong regularity and robustness.
- The investigation is thorough and rigorous, supported by extensive experiments spanning multiple model families and post-training scenarios.
- Includes a discussion on potential applications and implications in the appendix, adding useful context for future exploration.

### Weaknesses
- While the paper identifies consistent orthogonal transformations as a key structural component of post-training, it would benefit from a deeper discussion or empirical analysis of how these transformations influence model outputs or distributional behavior. Understanding this connection could strengthen the interpretability and practical relevance of the findings.

### Questions
- Is there any insight on how the other parameters of the model (eg. normalization layers and output projection head) change as a result of post-training?
- Is there evidence that these structural patterns persist throughout training, or do they emerge only after convergence of post-training?

### Soundness
4

### Presentation
3

### Contribution
3
