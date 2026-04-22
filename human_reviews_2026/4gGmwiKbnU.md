# Latent-DPO: Direction-Aware Preference Optimization for Reasoning Alignment

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Although Large Language Models (LLMs) show impressive performance across diverse tasks, how to construct and effectively leverage high-quality supervision data remains an open challenge. While reverse question–answer pairs offer a means of data augmentation, models trained exclusively on forward–reverse mixtures through distillation still struggle to capture directional consistency. Standard Direct Preference Optimization (DPO) enforces uniform separation, often at the expense of shared reasoning structures. To address these limitations, we construct reverse examples and introduce Latent-DPO, an extension of preference optimization built upon reverse-augmented data. Latent-DPO incorporates a binary latent variable to model the consistency of reasoning paths and to modulate the DPO margin. This mechanism adaptively adjusts alignment strength, relaxing separation for pairs with subtle differences while maintaining a strong distinction for clearly divergent pairs. Empirical results demonstrate that our carefully constructed set of only 817 reverse examples produces a 4.5% average improvement across five benchmarks. Moreover, Latent-DPO yields consistent improvements across multiple datasets and base models, achieving average accuracy gains of up to 3.2%. Our code and data are available at the anonymous repository: \url{https://anonymous.4open.science/r/submission_429}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Latent-DPO, a variant of Direct Preference Optimization that introduces a binary latent variable to indicate whether a response follows the intended reasoning direction. The method is evaluated using forward-and-reverse pairs automatically built from the 817-problem LIMO dataset.

### Strengths
1. The writing is clear and easy to follow
2. They provided the code, improving the reproducibility of the work

### Weaknesses
1. The paper seems to lack novelty. Reverse reasoning and DPO are not new concepts and the combination does not seem to yield a particularly notable gain.
2. The experiments are conducted on Qwen3-1.7B-Base and Qwen3-1.7B alone. The paper might benefit from more experiments on 7B, 8B, and 32B models.
3. The improvements seem slight compared to other methods and even the base model, especially considering that some of the benchmarks have very limited samples.

### Questions
See weaknesses.

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
This paper introduces Latent-DPO, an extension of Direct Preference Optimization that incorporates a binary latent variable to model reasoning-path consistency, aiming to preserve shared knowledge while adaptively enforcing separation between forward and reverse reasoning directions in language models. The key contributions are: (1) Reverse Data Augmentation: A method to construct high-quality reverse reasoning questions from a small seed dataset (LIMO), demonstrating that even 817 reverse examples can improve model performance. (2) Latent Direction Alignment: A new algorithm that introduces a binary latent variable to model whether a response follows the intended reasoning path. This allows the model to adaptively modulate the DPO loss, aiming to preserve shared reasoning knowledge while enforcing separation only on direction-specific operations.

### Strengths
(1) Problem Motivation: The paper identifies a potentially interesting and nuanced problem: that standard DPO might be too blunt an instrument for reasoning tasks where "dispreferred" responses may still contain valuable, shared knowledge.

(2) Theoretical Formulation: The latent-variable framework is derived with mathematical care, and the connection to variational inference is a positive aspect of the work.

(3) Empirical Effort: The authors have undertaken a non-trivial empirical study, constructing a reverse dataset and running multiple experiments across different model initializations.

### Weaknesses
(1) Misalignment Between Motivation and Method: The core motivation is to preserve shared knowledge from "partially correct" responses. However, the construction of the preference pairs (x, y+, y-) is problematic. The y- for a forward problem can be a correct solution to the reverse problem, not a partially correct one. The latent variable q_l is trained to detect a different reasoning direction, not a flawed one. This fundamentally undermines the proposed mechanism's ability to achieve its stated goal of preserving shared knowledge from incorrect but structurally similar responses. The y- examples are a mix of directionally incorrect but otherwise valid solutions and potentially fully wrong answers, creating a noisy and inconsistent learning signal for the latent classifier.

(2) Unjustified Complexity: The method introduces significant complexity: a multi-stage pipeline for reverse data generation and an auxiliary neural network (the posterior head) that requires careful training with a novel regularizer. 

(3) Limited Applicability to Non-Invertible Reasoning Tasks: The current approach relies on the construction of well-defined "reverse" problems, however they may not exist for some open-ended reasoning tasks .

### Questions
(1) Mechanism Verification: The latent variable z is intended to model "reasoning-path consistency." Can you provide direct evidence that it learns this? For example, please show examples from the validation set where a y- that shares the correct formula but applies it to the wrong direction has a significantly higher q_l than a y- that is complete nonsense. 

(2) y- Composition and Ablation: Your y- is a mixture of direction-swapped solutions and potentially incorrect answers. Did you ablate this? What is the performance if y- is only the direction-swapped (but otherwise correct) solution versus only a truly incorrect solution? This would help disentangle what signal the latent variable is actually capturing.

(3) The method is evaluated solely on mathematical reasoning dataset. How would you envision applying this method to more open-ended reasoning tasks (e.g., commonsense QA, code generation) ?

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
3

### Summary
The paper studies direction-aware alignment for reasoning in LLMs, arguing that standard DPO over-separates responses and erodes shared reasoning. It proposes Latent-DPO, which introduces a binary latent variable z to model whether a response is direction-consistent with the prompt and modulates the DPO margin accordingly.

### Strengths
- Relative to DPO variants, the key novelty is a latent, learned posterior that adaptively gates margin contributions of win/lose responses based on estimated directional alignment. This is distinct from fixed margins or step-level weighting.
- The reverse data pipeline is not conceptually new, but the careful pairing to form preference tuples and the demonstration that mixed SFT hurts while Latent-DPO helps is a useful empirical insight.
- Overall, contribution is incremental but meaningful: a simple, implementable extension to DPO addressing a concrete failure mode.

### Weaknesses
- Limited baseline coverage
- Statistical reporting is thin in main tables. Small gains may be within variance on some benchmarks.
- Potential confound: reverse vs forward stylistic artifacts from different teachers could be learned by the posterior rather than true directional consistency.

### Questions
- How does Latent-DPO compare to KTO/SimPO and to step-level/process feedback DPO on the same data?
- Can you report human verification of reverse data quality on a subset, beyond model-based judges?

### Soundness
2

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
This paper proposes Latent-DPO, a direction-aware extension of Direct Preference Optimization (DPO) that introduces a latent variable zto model reasoning-path consistency. Building upon the curated LIMO dataset, the authors generate reverse reasoning counterparts and construct bidirectional preference pairs. The method adaptively modulates preference margins to preserve shared reasoning knowledge while separating direction-specific logic. Experiments on multiple reasoning benchmarks (AIME-25, Math500, GPQA, etc.) demonstrate consistent gains of 0.9–3.2% with only 1.6k preference pairs. Overall, the work provides a compelling and data-efficient approach to enhance reasoning alignment and offers valuable insights into direction-aware model training.

### Strengths
1.	The motivation is clear: the paper aims to address a core shortcoming in LLM alignment tasks — namely relying only on forward supervision while neglecting reasoning‐direction consistency. The root of this issue lies in the fact that alignment tasks typically provide only binary rewards. The paper considers introducing the concept of reasoning direction, and proposes to model DPO using a latent variable. This method is novel and can effectively address this problem.
2.	The method design is sound: by introducing a latent variable z, the model can adaptively distinguish between samples whose reasoning directions are consistent versus inconsistent, thereby aligning more robustly while preserving shared knowledge. The design is intuitive, lightweight, and has good theoretical interpretability.
3.	The experiments cover multiple model scales and five reasoning benchmarks, demonstrating stable improvements of 0.9%–3.2%, and include ablation and scaling analyses to verify the method’s generality and robustness. The authors construct only 817 high-quality reverse reasoning samples via reverse question generation (DeepSeek V3) and automatic verification (Qwen series), showing very high data efficiency and reproducibility.

### Weaknesses
1.	Experiments are conducted only on a self-constructed dataset of 1,634 preference pairs, which is highly specialized and idealized. This makes it difficult to assess the scalability and robustness of the method on larger and noisier real-world preference datasets.
2.	The paper presents the idea of combining “reverse reasoning” with DPO and proves that rewarding valuable y− responses improves generalization. However, the experiments do not clearly isolate whether the observed gains come from the reverse-thinking data, the latent-variable mechanism, or their interaction.
3.	The evaluation mainly relies on pass@1 accuracy, without more fine-grained analyses such as response length, direction-consistency score, or human preference evaluation. As a result, the source and nature of the improvements remain unclear.
4.	The study only compares with vanilla DPO, omitting recent robust or enhanced DPO methods (e.g., β-DPO, Dr.DPO, γ-PO) that also address overfitting from binary preferences. Such comparisons would better highlight the distinct advantages of Latent-DPO.
5.	The latent variable z is introduced to model directional consistency, there is no empirical examination of its interpretability or correlation with reasoning-path semantics, leaving its actual role insufficiently verified.

### Questions
1.	Your code link appears broken — most files cannot be accessed, which need to be fixed.
2.	Could you quantify the training overhead of the posterior head (e.g., additional parameters, GPU memory usage, or training time percentage)?
3.	In Figure 3(c), the win rate converges around 0.6. Is this behavior consistent across different seeds and models, or just a coincidence of training dynamics? Could you provide a theoretical or optimization-based explanation for this?
4.	Please tested the method’s stability on larger or noisier preference datasets beyond the curated 1.6k pairs?
5.	Can you clarify whether the observed performance gains mainly come from the reverse-reasoning data, the latent-variable mechanism, or their interaction?
6.	Is the learned latent variable z semantically interpretable? Could you show examples or analyze how q(z) evolves during training?
7.	Compared with other robust DPO variants, what are the distinct advantages of Latent-DPO?
8.	Maybe you can provide case studies or qualitative examples showing that Latent-DPO improves the model’s thinking ability or reasoning-chain quality (e.g., better reverse inference)?

### Soundness
2

### Presentation
3

### Contribution
3
