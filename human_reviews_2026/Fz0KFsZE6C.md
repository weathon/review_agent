# OpenSIR: Open-Ended Self-Improving Reasoner

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Recent advances in large language model (LLM) reasoning through reinforcement learning rely on annotated datasets for verifiable rewards, potentially limiting models' ability to exceed human-level performance. While self-play offers a promising alternative, existing approaches depend on external verifiers or cannot learn open-endedly. We present Open-Ended Self-Improving Reasoner (OpenSIR), a self-play framework where an LLM learns to generate and solve novel problems by alternating teacher and student roles without external supervision. To generate novel problems, OpenSIR optimises for both difficulty and diversity, rewarding problems that challenge appropriately while exploring distinct concepts, enabling open-ended mathematical discovery. Starting from a single trivial seed problem, OpenSIR substantially improves instruction models: Llama-3.2-3B-Instruct advances from 73.9 to 78.3 on GSM8K, and from 28.8 to 34.4 on College Math, while Gemma-2-2B-Instruct rises from 38.5 to 58.7 on GSM8K. Our analyses reveal that OpenSIR achieves open-ended learning through co-evolving teacher-student roles that adaptively calibrate difficulty and drive diverse exploration, progressing autonomously from basic to advanced mathematics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces OpenSIR (Open-Ended Self-Improving Reasoner), a self-play framework designed to enhance LLM reasoning without external supervision by using a single policy that alternates between "teacher" and "student" roles. Starting from a single trivial seed problem , the teacher is trained via reinforcement learning to generate novel problems, optimizing a "novelty" reward that combines both difficulty (calibrated by solve rates and solution length) and diversity (using embedding-based distance to encourage exploration) . The student, in turn, is trained to solve these problems, with correctness determined by majority voting across multiple solution attempts. This self-improving loop allows the model to autonomously bootstrap its capabilities. While the paper has merit, it requires major revision to be ready for publication.

### Strengths
The paper is well motivated.

The method is tested on three model families: Llama-3.2-3B-Instruct, Gemma-2-2B-Instruct, Qwen-2.5-3B-Instruct.

Ablation and analysis is appreciated.

### Weaknesses
Why is R-Zero not in the baseline? Any other potentially missing baselines?

Pseudocode should be available in the paper. If there is no space in the main body, at least in the appendix. It will dramatically help with clarity and reproducibility.

The work should mention how it compares and contrasts with “AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners” (NeurIPS 2025) as their method also directly tackled the same problem: (1) difficulty (2) diversity. This seems like a key related work.

> To generate novel problems, OpenSIR optimises for both difficulty and diversity, rewarding problems that challenge appropriately while exploring distinct concepts, enabling open-ended mathematical discovery.

There seems to be numerous hyperparameters; e.g. the two s values, alpha, avg@16. What are all the hyperparameters (in the method and also in the empirical study)? How are they set? Heuristically? Empirically? Any sensitivity tests? Large newly introduced hyperparameter space can be a pain for practitioners, especially if the method is sensitive to them.

It seems like this method incurs numerous additional costs. Is the additional cost overhead clearly communicated in the paper? E.g. the embeddings’ cosine similarity will incur costs. This should especially be clearly provided in Tab. 1 where it is very possible that GRPO_gsm8k and GRPO_math has used less training compute than the proposed method.

Does this method only work on the math domain? If so, the scope seems slightly limited.

To my understanding, it is normal practice to keep GRPO going until it hits peak performance. It is concerning that an arbitrary fixed compute budget has been provided.
> To compare models trained on the same number of problem-solution pairs, we train the GRPO baselines with 100 steps, and OpenSIR for 200 steps since OpenSIR allocates half of its training budget to problem generation.

It would be ideal if there was at least one model with larger model size. These models are very small.

Among the three models tested Qwen 2.5 is the strongest one. Naturally, post-training is done on the strongest available base model. Considering this, the Qwen results are most important. However, this method’s gain over baselines in the Qwen results is negligably small; even possibly due to noise. Accuracy gain over best baseline is 0.29% which is virtually no real-world gain.

While it does make sense that the proposed method does not use any labels. There is no need to not use existing labels from e.g. gsm8k, math. There should be experiments with train on existing available labeled train set + opensir to show that this is meaningful in the real-world.
> OpenSIR consistently outperforms GRPO baselines across model architectures despite generating training data through self-play from a single seed problem, while GRPO baselines use over 7,000 human-annotated examples.

### Questions
Is there a reason why there is such little use of colors in the text? The clarity of the paper may improve with some coloring, e.g. sections, references.

Does this method work on thinking models, not just instruct models?

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
4

### Summary
This paper introduces the Open-Ended Self-Improving Reasoner (OpenSIR), a novel framework that enables a Large Language Model (LLM) to autonomously improve its mathematical reasoning capabilities. The core of OpenSIR is a self-play mechanism where a single LLM policy alternates between two roles: a "teacher" that generates new mathematical problems and a "student" that solves them. Starting from a single trivial seed problem (e.g., "What is 1+1?"), the system bootstraps its own learning curriculum without any external human-annotated data. The teacher is rewarded for creating problems that are both appropriately difficult (calibrated via the student's solve rate) and conceptually diverse (measured by embedding distance to previously seen problems). The authors demonstrate that this approach significantly improves the performance of smaller LLMs (Llama-3.2-3B and Gemma-2-2B) on several math reasoning benchmarks, outperforming baselines trained on thousands of human-labeled examples.

### Strengths
1.The paper's primary strength lies in its contribution to open-ended, autonomous learning. By successfully demonstrating that an LLM can bootstrap complex reasoning skills from a single trivial example without human supervision, OpenSIR presents a compelling alternative to data-intensive RLHF methods. This addresses a major bottleneck in scaling LLM capabilities and is a significant step towards more autonomous AI systems.

2.The design of the reward function for the teacher role is very effective. Decomposing "novelty" into two intuitive dimensions, difficulty (via scoresol and scorelen) and diversity (scorediv), provides a robust mechanism for generating a dynamic and adaptive curriculum. This allows the model to avoid getting stuck on trivial problems or generating impossibly hard ones, guiding it from basic arithmetic to advanced topics like calculus and trigonometry

### Weaknesses
1.The experiments are confined to smaller models (2B-3B parameters). While the results are impressive, the paper shows minimal gains for the stronger Qwen-2.5-3B model. The authors suggest this may be due to benchmark contamination, but it could also indicate that the self-improvement process yields diminishing returns for models that are already highly capable. A discussion on the scalability of this approach to state-of-the-art models (e.g., 8B+) is a notable omission.

2.The self-play loop requires multiple forward passes for each problem generated (G solution attempts per problem) before a single policy update. This process seems computationally expensive compared to standard supervised fine-tuning. The paper does not provide a clear analysis of the computational overhead, making it difficult to assess the practical feasibility and cost-effectiveness of OpenSIR versus simply training on a large, existing dataset.

3.The analysis in Section 4.2 reveals a critical weakness: problems with very low solve rates are often invalid rather than genuinely difficult. The framework's reliance on solve rate as a proxy for difficulty struggles to distinguish between these two cases. While the chosen thresholds (e.g., s_min = 0.5) seem to work, this suggests the curriculum generation might be sensitive to these hyperparameters and could inadvertently filter out challenging but valid new problem domains where the model initially has a very low success rate.

### Questions
1. The concept of using self-play for generation and reasoning has been explored in prior work, such as R-Zero. Could the authors further elaborate on the core mechanistic novelty of OpenSIR, particularly its key distinctions from existing approaches? Furthermore, the performance improvement attributed to reinforcement learning appears relatively modest. Does this suggest a potential performance ceiling for this method?

2. The paper's evaluation is primarily conducted on established benchmarks like GSM8K and MATH, where current models already demonstrate strong performance. Does the proposed framework have the potential to generate and solve more complex, competition-level problems (e.g., from AIME)? How robust is the framework's effectiveness in these more challenging scenarios?

3. The experiments are conducted mainly on small-scale models (3B-parameter range). Could the authors comment on the anticipated efficacy of this approach on larger-scale models (e.g., 8B, 14B)? Is the performance upper-bound of the method constrained by the capability ceiling of the initial base model? In other words, is the framework primarily eliciting latent abilities rather than imparting genuinely new skills?

4. The current comparisons are primarily against base models and a general-purpose RL method (GRPO). The paper lacks a direct comparison with contemporary state-of-the-art models in mathematical reasoning  that also leverage synthetic data

### Soundness
3

### Presentation
3

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
This paper introduces OpenSIR (Open-Ended Self-Improving Reasoner), a self-play framework for large language models (LLMs) that autonomously improves reasoning abilities without external supervision. OpenSIR alternates teacher and student roles, generating and solving novel problems optimized for difficulty and diversity, enabling open-ended mathematical discovery. Starting from a single trivial seed problem, OpenSIR drives autonomous progression from basic to advanced concepts. Experiments show significant performance improvements on benchmarks like GSM8K and College Math, with models achieving substantial gains. The framework's adaptive teacher-student co-evolution fosters diverse exploration and calibrated learning, advancing LLM reasoning capabilities effectively.

### Strengths
1. The paper tests its framework, OpenSIR, across multiple benchmarks (e.g., GSM8K, College Math) using various backbone LLMs, such as Llama-3.2B-Instruct and Gemma-2-2B-Instruct. This demonstrates some generality and effectiveness of the approach across different models and tasks.

2. The paper tackles an important topic—using reinforcement learning (RL) to improve LLM reasoning capabilities. RL is a compelling approach for driving autonomous learning, making this work relevant and interesting for advancing LLMs.

### Weaknesses
1. The core idea of OpenSIR lacks novelty, appearing more like a combination of popular concepts (self-play, RL, curriculum learning) rather than introducing a new approach. The paper could benefit from showcasing deeper insights or unique contributions that distinguish it from existing methods.

2.  The authors do not provide code or other necessary materials, making it difficult for researchers to replicate the results or experiment with the framework. Including well-documented code and resources would significantly enhance the paper's impact and accessibility.

3. The font used in Figure 1 appears informal and less readable, which detracts from the professional presentation of the paper. Using a more formal and easily readable font would improve the clarity and visual impact of the figure, making it more suitable for academic audiences.

4. The experiments are conducted on relatively small models. While the results are promising, the robustness and effectiveness of the proposed method on larger-scale models remain unverified. Expanding the experiments to larger models would strengthen the paper's claims and demonstrate broader applicability.

### Questions
Refer to Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces OpenSIR, a self-play reinforcement learning framework where a single policy jointly optimizes two roles: a teacher that generates novel and diverse math problems, and a student that solves them accurately. Both roles are jointly trained to form an open-ended self-improvement loop that enables the model to continuously enhance its problem generation and reasoning abilities. This dual-role optimization enables the model to bootstrap from a single seed problem and progressively enhance both problem generation and reasoning capability without human supervision. Experiments show that OpenSIR consistently outperforms GRPO baselines and base instruction models, achieving comparable or superior accuracy without any human-annotated data.

### Strengths
- The proposed approach significantly outperforms supervised RL approaches (GRPO) and instruction-tuned baselines across a number of models and benchmarks.

- The proposed approach requires no human-annotated data, reducing cost and reliance on manual labeling.

- Joint optimization of teacher and student creates a self-calibrating cycle, enabling continuous self-generated training at optimal difficulty.

### Weaknesses
- In 4.1 Figure 2, the observed V-shaped difficulty trend is interesting, but the authors should provide evidence of the student model’s performance over training (e.g., accuracy or solve rate) to substantiate the claim that this pattern reflects true self-calibration.

- In 2.1, the author states, “We initialise the problem pool P_0 with a single trivial problem (“What is 1+1?”)”  Given the simplicity of this seed, it is worth discussing whether and how this choice constrains the initial diversity or attainable difficulty of the generated problems, and whether the model can robustly escape such a limited starting point.

- While the paper conducts ablations on diversity and length rewards and dual-role training, it does not provide individual analyses for the solvability reward components. It remains unclear how the component contributes to the overall accuracy performance.

### Questions
- In Table 8, equal weights (α = λ = γ = 1.0, δ = 0.1) are assigned to most teacher rewards and 1.0 to the student accuracy reward. Could the authors clarify how these weights were chosen? Were they empirically tuned, or are the results robust to moderate changes in these hyperparameters?

### Soundness
3

### Presentation
3

### Contribution
3
