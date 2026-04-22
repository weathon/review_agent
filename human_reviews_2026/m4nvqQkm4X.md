# Reference-guided Policy Optimization for Molecular Optimization via LLM Reasoning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Large language models (LLMs) benefit substantially from supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR) in reasoning tasks. However, these recipes perform poorly in instruction-based molecular optimization, where each data point typically provides only a single optimized reference molecule and no step-by-step optimization trajectory. We reveal that answer-only SFT on the reference molecules collapses reasoning, and RLVR provides sparse feedback under similarity constraints due to the model's lack of effective exploration, which slows learning and limits optimization. To encourage the exploration of new molecules while balancing the exploitation of the reference molecules, we introduce **Re**ference-guided **P**olicy **O**ptimization (RePO), an optimization approach that learns from reference molecules without requiring trajectory data. At each update, RePO samples candidate molecules with their intermediate reasoning trajectories from the model and trains the model using verifiable rewards that measure property satisfaction under similarity constraints in an RL manner. Meanwhile, it applies reference guidance by keeping the policy’s intermediate reasoning trajectory as context and training only the answer in a supervised manner. Together, the RL term promotes exploration, while the guidance term mitigates reward sparsity and stabilizes training by grounding outputs to references when many valid molecular edits exist. Across molecular optimization benchmarks, RePO consistently outperforms SFT and RLVR baselines (e.g., GRPO), achieving improvements on the optimization metric (Success Rate $\times$ Similarity), improving balance across competing objectives, and generalizing better to unseen instruction styles. Our code is publicly available at https://github.com/tmlr-group/RePO.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a new method called DePO which was modified from standard RLVR procedure via using a demonstration-guided term in the objective function. It provides a novel framework to LLM-driven molecular optimization and through experimental results, demonstrates that the method can enhance reasoning and task performance.

### Strengths
The paper is well written and the experimental design is robust. It also provides a detailed analysis of current drawbacks on RLVR and how to address them via modification on the architecture, algorithm and reward function. The results demonstrate good improvement compared to the baseline and support the paper's aim of providing better reasoning in molecular optimisation.

### Weaknesses
The paper uses only Qwen-2.5-3B Instruct as a single backbone for the baseline and base model (and to demonstrate scaling also use Qwen-2.5-7B Instruct). The results and findings could be more robust if it has had demonstrated results using other models since the base reasoning and optimisation capacity can be different. This would be useful to determine whether DePO still offer improvement to the other methods (while using the same base model).

### Questions
Is there any disadvantage with constraining the exploration space in DePO?
Table 1 and 2 show the overall results on on TOMG-Bench. If that is an average performance over the dataset, what is the range of errors in addition to the mean and do they vary  or stay similar across the methods?

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
2

### Summary
A solid and well-executed paper proposing Demonstration-guided Policy Optimization (DePO) for LLM-based molecular optimization. The idea of integrating demonstration-guided supervision into RLVR is clear, novel enough for ICLR, and empirically convincing. Methodologically clean, though somewhat incremental from GRPO + SFT.

### Strengths
1. Novel combination of demonstration and RL: Clear motivation and principled integration of supervised guidance into RL objective.

2. Strong empirical gains: Up to 13% improvement over SFT and GRPO on TOMG-Bench and MuMOInstruct, with convincing generalization and ablations.

3. Well-written and interpretable: The framework (gradient masking, demonstration term) is intuitive and illustrated clearly with chemical reasoning examples.

### Weaknesses
1. Incremental contribution: Conceptually close to existing RLHF/RLVR + imitation learning hybrids; novelty may feel limited.

2. Limited scope: Only tested on molecular optimization—no evidence DePO generalizes to other scientific or reasoning domains.

3. Weak theoretical insight: Lacks formal analysis of convergence or policy improvement guarantees under demonstration bias.

### Questions
1. How sensitive is DePO to low-quality or noisy demonstrations (e.g., suboptimal molecules)?

2. Could DePO degenerate into simple imitation when β is high? How is β chosen in practice?

3. Does DePO still help when using stronger base models (e.g., >7B) trained with rich CoT reasoning data?

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
The paper introduces DePO a framework that enhances large language models’ reasoning ability for molecular optimization by combining reinforcement learning with expert demonstrations. Unlike prior approaches such as SFT, which suppress reasoning, or RL with verifiable rewards, which suffers from unguided exploration, DePO integrates reference molecules as supervised signals to guide the search toward chemically valid regions while preserving intermediate reasoning steps through gradient masking. Evaluated on TOMG-Bench and MuMOInstruct, DePO achieves up to 13% performance gains over strong baselines, demonstrating improved balance between property optimization and structural similarity, better generalization to unseen instructions, and chemically coherent reasoning, establishing a new paradigm for LLM-driven scientific reasoning and molecular design.

### Strengths
The paper introduces a novel demonstration-guided reinforcement learning paradigm (DePO) that effectively bridges the gap between language-model reasoning and domain-constrained scientific optimization. It presents comprehensive experiments across multiple molecular optimization benchmarks, providing strong empirical evidence for the framework’s effectiveness. The significance of this work lies in extending LLM reasoning beyond text and mathematics to molecular design, demonstrating that demonstration-guided optimization can deliver substantial improvements in both performance and generalization within complex chemistry-related tasks.

### Weaknesses
1. The discussion of related work is limited. Several recent GPT-based molecular optimization studies are neither cited nor compared as baselines, making it difficult to contextualize DePO’s contributions within existing literature.

2. The novelty appears somewhat incremental, as the method primarily adds guiding exploration and a regularization term to the GRPO objective. The introduced regularization is heuristic in nature, and the paper does not provide sufficient discussion on how the hyperparameters (β and γ) are selected or tuned, which may significantly influence final performance.

3. Although the paper motivates DePO’s gradient-masking mechanism, it lacks theoretical justification or a detailed ablation study to assess how different masking strategies impact reasoning fidelity, convergence, and optimization stability.

4. While DePO presents qualitative improvements in reasoning traces, the evaluation of reasoning quality remains largely subjective. Incorporating quantitative reasoning metrics or expert chemical validation would make the claims more convincing.

5. how to create the demonstration dataset is not discussed.

### Questions
Additional questions regarding weaknesses:

1. The design and formulation of the regularization component are not clearly explained.

2. Why is the gradient flow disabled during the reasoning (“thinking”) phase? Could the authors provide an ablation study to justify this choice?

3. In Figure 7, there is a large discrepancy between DePO and GRPO. The result appears somewhat unreliable, as GRPO seems to lack a valid expression. Were both methods fine-tuned on the same base model? 
Without providing the source code, it's hard to believe such difference considering on overall similarity in objective function.

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
The paper addresses a key limitation of LLMs in scientific reasoning tasks, especially molecular optimization, designing molecules with better drug-like properties while maintaining structural similarity. Existing approaches like: SFT need reasoning traces, which are absent in molecule datasets, RLVR (Reinforcement Learning with Verifiable Rewards, e.g., GRPO) performs poorly due to sparse rewards and unguided exploration. Thus, the paper introduces DePO (Demonstration-guided Policy Optimization), a method that combines the benefits of RL and supervision using reference molecules as demonstrations to guide the optimization process.

The core idea is to guide the policy optimization process with demonstration molecules that serve as structural anchors, balancing exploration (via RL) and exploitation (via supervision). The task evaluated is molecular optimization i.e., given an input molecule (in SMILES form) and instruction (e.g., “make it more drug-like”), an optimized molecule needs to be generated. The training objective (Eq. 4) combines three components: RL term (Encouraging Exploration): Similar to GRPO, updates policy toward higher rewards, supervised term (Guiding Exploration): Encourages the model to generate solutions similar to demonstration molecules, regularization term: KL divergence to keep policy updates stable. Gradient masking ensures that only final answers (not reasoning tokens) are supervised, preserving reasoning ability. Two main rewards are proposed (a) structural similarity reward (r_struct): Tanimoto coefficient between input and generated molecule; (b) property reward (r_prop): Binary reward indicating whether target property (QED, LogP, MR) improves.

The paper evaluates DePO on two benchmarks: TOMG-Bench and MuMOInstruct. The metric used is the product of Success Rate (achieving property objective) and Structural Similarity (Tanimoto score).

### Strengths
- The paper is very well-written and easy to understand. I like how the problem is first formulated and clearly defined in Section 2, followed by a discussion of specific limitations in Section 3. The proposed method is then presented clearly and succinctly in Section 4, making it easy for readers to follow and grasp the details.

- DePO is an elegant fusion of RL and supervised signals, guiding exploration with domain demonstrations while maintaining reasoning depth. It demonstrates constraint search to chemically meaningful regions, overcoming limitations of GRPO.

- The paper presents strong empirical results. DePO outperforms SFT and GRPO on multiple benchmarks (TOMG-Bench, MuMOInstruct) by up to 13%.

### Weaknesses
- In the DePO scheme, the LLM output is first parsed into reasoning tokens and the generated final answer. Then, the generated final answer is replaced with the gold-standard final answer. This way, the reasoning tokens are preserved, and gradient masking for the intermediate reasoning steps excludes these tokens from parameter updates during optimization. The authors claim that this approach prevents the LLM from learning potentially erroneous reasoning patterns, but it is not clear why that is the case. There is no constraint imposed on the reasoning tokens that prevents the model from generating erroneous reasoning patterns. In fact, training only occurs on tokens excluding the reasoning tokens. The authors further state that DePO "constrains its exploration to chemically valid and promising regions of the solution space," but it is not evident why this is true. Moreover, what is the rationale behind replacing the generated answer with the gold-standard answer? The reward is anyway calculated based on the generated answer. The overall design seems to offer only incremental improvements over GRPO.

- For the reward assessing the target property, a binary reward value of 1 is assigned if the generated molecule achieves a favorable change in the target property relative to the original molecule. This means that even a minuscule positive change results in a reward of 1. Such a design is highly susceptible to reward hacking, where the model may settle after making only a small improvement and fail to pursue further optimization, since it has already received the maximum reward of 1. 

- Why is the product of Success Rate (achieving the property objective) and Structural Similarity (Tanimoto score) considered a meaningful metric? Wouldn’t this metric be biased if one component is significantly high while the other is only moderate? From Section 2.1, it seems more reasonable to report these as separate metrics: (a) drug-likeness, conditioned on structural similarity being within a specified threshold (0 otherwise), and (b) structural similarity itself.

- The paper only compares against different fine-tuning paradigms, namely, Baseline, SFT, GRPO, and GRPO (SFT init). However, it does not include comparisons with existing state-of-the-art methods for molecular optimization, nor with any foundation models.

- Figure 2 is difficult to interpret, and the corresponding observations are not clearly conveyed. For instance, Observation 3.1 ("Models trained with GRPO exhibit a conservative bias, generating molecules nearly identical to the input, as shown in Fig. 2 (Left)") is not directly supported or easily interpretable from the figure. I suggest modifying it to a different type of plot that more clearly illustrates the stated observations and providing a more descriptive caption to enhance interpretability.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
2
