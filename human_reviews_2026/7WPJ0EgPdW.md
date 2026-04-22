# AMiD: Knowledge Distillation for LLMs with $\alpha$-mixture Assistant Distribution

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Autoregressive large language models (LLMs) have achieved remarkable improvement across many tasks but incur high computational and memory costs. Knowledge distillation (KD) mitigates this issue by transferring knowledge from a large teacher to a smaller student through distributional alignment. Previous studies have proposed various discrepancy metrics, but the capacity gap and training instability caused by near-zero probabilities, stemming from the high-dimensional output of LLMs, remain fundamental limitations. To overcome these challenges, several approaches implicitly or explicitly incorporating assistant distribution have recently been proposed. However, the past proposals of assistant distributions have been a fragmented approach without a systematic investigation of the interpolation path and the divergence. This paper proposes $\alpha$-mixture assistant distribution, a novel generalized family of assistant distributions, and $\alpha$-mixture distillation, coined AMiD, a unified framework for KD using the assistant distribution. The $\alpha$-mixture assistant distribution provides a continuous extension of the assistant distribution by introducing a new distribution design variable $\alpha$, which has been fixed in all previous approaches. Furthermore, AMiD generalizes the family of divergences used with the assistant distributions based on optimality, which has also been restricted in previous works. Through extensive experiments, we demonstrate that AMiD offers superior performance and training stability by leveraging a broader and theoretically grounded assistant distribution space.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an α-mixture assistant distribution and a corresponding knowledge distillation framework named AMiD for large language models (LLMs). It unifies existing methods (e.g., m-mixture and e-mixture) as special cases within a broader, theoretically grounded family parameterized by α. The framework is supported by theoretical analysis (optimality, gradient behavior, support properties) and extensive experiments showing consistent improvements over prior approaches across various tasks, model scales, divergences, and data generation strategies.

### Strengths
Introduces a principled and generalized family of assistant distributions via α-mixture, offering a unified view of prior fragmented methods.  
Provides solid theoretical grounding, including optimality guarantees, support analysis, continuity, and gradient-based interpretation of mode-covering vs. mode-seeking behavior.  
Comprehensive experiments validate the effectiveness of AMiD across instruction-following, task-specific distillation, different student sizes, divergences, and SGO strategies.  
The new design variable α offers practical control over the quality-diversity trade-off, independent of the interpolation weight λ.

### Weaknesses
The writing is not very clear and significantly hampers readability.** Despite strong technical content, the exposition is often dense and poorly structured, especially in Section 3. Key concepts (α-mixture, f-mean, α-divergence) are introduced rapidly without sufficient intuition or gradual buildup, making it difficult for readers to follow.  
Figures (e.g., Figure 1 and 2) lack detailed captions and fail to fully clarify the geometric impact of α on interpolation paths.  
Limited discussion on why α ≠ ±1 performs better in practice beyond empirical results; the gap between theoretical optimality and practical instability (e.g., DRKL with α=1) is noted but not deeply analyzed.  
Some notation is ambiguous or inconsistently used (e.g., r vs. r̃, θ vs. θ′), adding unnecessary cognitive load.

### Questions
1. How should one choose the optimal α in practice? Are there adaptive or task-aware strategies for tuning α during training?  
2. The paper fixes the divergence (e.g., DAB) while varying α. Have the authors explored joint optimization or tuning of both α and divergence parameters (e.g., α_AB, β_AB)?  
3. Theorem 3.4 claims optimality for any divergence and α, yet Table 3 shows catastrophic failure for DRKL with α=1. Does this indicate that the “perfect optimization” assumption is too strong, and how should practitioners navigate the theory-practice gap?

### Soundness
4

### Presentation
2

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
This paper presents  a unified framework for knowledge distillation with an assistant distribution, where the teacher and student distributions are mixed to bridge the capacity gap between them. Previous distillation mixing strategies (e.g., using arithmetic and geometric mean) become special cases under this framework.

### Strengths
1. The authors presents an interesting point of view on existing distillation methods.
2. The experiments are generally comprehensive, showing consistent improvements.

### Weaknesses
1. The motivation of adjusting alpha is still a bit unclear to me. The authors mention alpha adjusts the mode-seeking and mode-covering properties. However, this can also be addressed in the divergence function, as mentioned in 2.1. I am not too sure about the intuition behind doing this again in the mixing stage.

2. The paper mainly focuses on instruction following, evaluated by rouge. This is slightly concerning, because the approach may be hacking the ROUGE score rather than making the model better at instruction following. The paper showed very limited results on reasoning (GSM8K, omitting ABKD for some reason, and no standard deviations). 

3. The paper is not very clear on its hyper-parameters. For example, the author mentions that their approach work with any divergence functions, but it's not clear which one is used for Table 1. Also, the values of alpha/lambda is not shown on the table. The author mentions D_AB being used for Table 2, but how about Table 1. Even in Table 2, it says alpha is not -1/+1, but it doesn't state what alpha is. My main worry is that the authors may have spent lots of efforts on tuning these individually, which gave the approach an unfair advantage.

### Questions
See weaknesses about hyper-parameters.

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
5

### Summary
This paper addresses the capacity gap between teacher and student models and the training instability caused by high-dimensional output in LLM knowledge distillation (KD). It proposes α-mixture assistant distribution and a unified distillation framework AMiD (α-mixture distillation). By systematically expanding the auxiliary distribution and divergence selection, it achieves better performance and training stability.

### Strengths
1. Generalization and Unification of Fragmented Methods.

Proposes a novel α-mixture assistant distribution by extending generalized \(f_\alpha\)-mean to KD, introducing the tunable parameter α. This generalizes prior isolated assistant distributions (m-mixture for α=-1, e-mixture for α=1) into a continuous, flexible family, covering new distributions (e.g., harmonic mean for α=3) not explored in LLM KD before.


2. Rigorous Theory and Comprehensive Experiments. Provides formal proofs for key properties (continuity of α-mixture distribution, optimality of AMiD, gradient analysis of f-divergence), establishing a solid mathematical foundation for the framework. Validates AMiD across diverse settings, task-agnostic (5 instruction-following datasets) and task-specific (translation, summarization, reasoning) distillation; multiple model scales (GPT-2 series, OpenLLaMA); various divergences (KL, RKL, AB) and SGO strategies. Results are consistent and statistically robust (5 random seeds), confirming reliability.

3. Addressing Core Limitations and Advancing Practical KD. Directly mitigates the capacity gap between teacher and student models and training instability from near-zero probabilities, two fundamental limitations of prior KD methods. AMiD’s flexibility (compatible with arbitrary divergences and datasets) and robustness (stable performance across λ values) make it adaptable to real-world LLM compression scenarios. The α parameter offers a simple control knob to balance quality and diversity, addressing a longstanding trade-off in generation tasks.

### Weaknesses
1. The lack of a systematic strategy for α parameter tuning limits its practicality. The paper demonstrates that the α parameter can control the balance between "pattern coverage" and "pattern finding" in the student model, but it fails to provide the basis for α selection and efficient tuning methods. The experiments only demonstrate the effects of fixing α (e.g., α=-5, -3, etc.), without explaining the optimal range and tuning logic of α under different tasks (translation/summarization/inference), different model capacity differences (e.g., 10B→0.1B), and different divergences (D), leading to extensive trial and error for users in practical applications.

2. The paper theoretically proves the optimality of AMiD under "perfect optimization," but two unresolved contradictions exist in practice: a) Conflict between divergence and α: Experiments show that the performance of D_RKL combined with α=1 is extremely poor (Avg. only 3.94 in Table 3), attributed to the narrow intersection of support sets, but no specific criteria are given on how to avoid this conflict;

b) Adaptation of optimizer and hyperparameters: The impact of different optimizers (such as AdamW, Lion) and learning rate scheduling on AMiD is not discussed, while in practice, optimizer selection significantly changes the gradient propagation effect of α-mixture.

3. The ablation experiments do not fully decompose the core contributions of AMiD: The interaction between α and λ is not verified: α is only tested with λ=0.1, without analyzing whether the optimal value of α changes under different λ (such as 0.3/0.7), and the collaborative tuning strategy between the two; the contributions of α-mixture and divergence are not isolated: Is the performance improvement of AMiD due to the auxiliary distribution of α expansion or the flexibility of divergence? Verification needs to be conducted by comparing "fixed α = ±1 (i.e., baseline auxiliary distribution) + arbitrary divergence" with "fixed divergence + α ≠ ±1".

4. The paper's baseline does not include cutting-edge LLM distillation methods since 2025, resulting in insufficient proof of advancement. For example, it does not compare distillation methods based on contrastive learning or reinforcement learning-based distillation (RL-KD), which may have already surpassed traditional divergence-based baselines in some scenarios.

### Questions
1. Adaptive Selection of α Parameter.  The paper demonstrates that α controls the trade-off between mode-seeking and mode-covering, but it does not provide a systematic method for selecting α in different scenarios (e.g., different task types, model size gaps, or data characteristics). Is there an adaptive strategy to determine α (e.g., curriculum learning-based scheduling, data-driven tuning) instead of manual adjustment? For small student models (e.g., 0.1B) vs. large gaps (e.g., 7B→0.5B), does the optimal α range differ significantly?

2. Extensibility to Larger Model Scales and Complex Tasks. The experiments focus on GPT-2 (up to 1.5B teacher) and OpenLLaMA-7B→3B. Have the authors tested AMiD on larger teacher models (e.g., 10B+ LLMs like LLaMA 3 70B) or smaller student models (e.g., <0.1B)? Additionally, do the performance gains hold for complex tasks such as code generation, mathematical reasoning with multi-step logic, or cross-lingual understanding?

3. Mitigation of Instability in Specific Divergence-α Combinations. The paper notes that the combination of \(D_{RKL}\) and α=1 leads to extremely poor performance due to support intersection issues. Have the authors explored mitigation strategies (e.g., adjusting λ, adding regularization terms, or modifying the assistant distribution’s normalization) instead of simply avoiding this combination? Are there general principles to avoid such conflicting pairs?

### Soundness
2

### Presentation
3

### Contribution
2
