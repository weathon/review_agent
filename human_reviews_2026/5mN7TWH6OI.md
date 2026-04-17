# COFormer: Towards a Foundation Model for Solving Combinatorial Optimization Problems

- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
Combinatorial Optimization Problems (COP) encompasses a wide range of real-world scenarios. While learning-based methods have achieved notable success on specialized COPs, the development of a unified architecture capable of solving diverse COPs with a single set of parameters remains an open challenge. In this work, we present COFormer, a novel framework that offers significant gains in both efficiency and practicality. Drawing inspiration from the success of next-token prediction in sequence modeling, we formulate the solution process of each COP as a Markov Decision Process (MDP), convert the resulting sequential trajectories into tokenized sequences, and train a transformer-based model on this data. To mitigate the long sequence lengths inherent in trajectory representations, we introduce a CO-prefix design that compactly encodes static problem features. Furthermore, to handle the heterogeneity between state and action tokens within the MDP, we adopt a three-stage learning strategy: first, a dynamic prediction model is pretrained via imitation learning; this model then serves as the foundation for policy generation and is subsequently fine-tuned using reinforcement learning. Extensive experiments across eight distinct COPs and various scales demonstrate COFormer’s remarkable versatility, emphasizing its ability to generalize to new, unseen problems with minimal fine-tuning, achieving even few-shot or zero-shot performance. Our approach provides a valuable complement to existing neural methods for COPs that focus on optimizing performance for individual problems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces COFormer, a Transformer-based framework designed to solve diverse combinatorial optimization problems (COPs) within a single unified architecture. Drawing inspiration from next-token prediction paradigms (as in large language models), COFormer reformulates COPs as sequential decision processes, representing their trajectories as token sequences. The framework introduces two key ideas:
- A CO-prefix mechanism that encodes static problem information to reduce sequence length.
- A three-stage learning scheme combining imitation learning (forward dynamics and policy generation) with reinforcement learning fine-tuning.

The authors evaluate COFormer on eight distinct COPs (e.g., TSP, VRP, Knapsack, FFSP, 3D Bin Packing), showing strong multi-problem generalization, few-shot transfer, and even some zero-shot capabilities.

### Strengths
**Originality**

- The paper addresses a key open challenge in neural combinatorial optimization by developing a unified, cross-problem model. This aligns well with the current research trajectory aiming for foundation models for decision-making.
- The use of a CO-prefix to separate static and dynamic features is elegant and practically motivated, reducing token length and improving efficiency.

**Significance**

- The empirical evaluation spans eight distinct CO problems, including both graph-based (TSP, VRP) and non-graph-based (FFSP, 3DBP) tasks. Considerably broader than most prior “generalist” NCO works.
- The reported results show strong generalization and competitive performance against specialized baselines. 

**Clarity & Quality**
- The paper communicates its motivation, contributions, and architecture clearly, with well-structured exposition and helpful figures, though technical descriptions, especially regarding tokenization, could be improved.
- Quality (Strength): The learning process is well motivated and comprehensive

### Weaknesses
**Lacking Clarity**

The paper lacks a clear and rigorous description of how heterogeneous problem features are represented in the token space. The paper’s description of the tokenization process (Sec. 3.1.2, App. B.1) lacks sufficient clarity to reproduce or fully understand how heterogeneous problem states are unified.

While it states that both continuous and discrete values are flattened and mapped to integer token IDs, it is unclear how structural or semantic distinctions (e.g., between coordinates, demands, processing times) are preserved.

The brief mention of “arbitrary key–value dictionaries” (App. B.1) implies flexible data interfaces but does not specify whether keys are tokenized, whether per-problem schemas are used, or how feature ordering is standardized.

As a result, it is difficult to assess whether the model genuinely learns a shared latent space across problem types, or whether it relies on per-problem positional regularities hidden in the data pipeline.

This opacity is problematic for reproducibility and for understanding how COFormer achieves semantic generalization.


**Lack of clarity and analysis on semantic representation learning**

The paper does not clearly explain how COFormer achieves true semantic unification across heterogeneous COPs and would benefit from a deeper analysis of why COFormer can generalize across problems. For instance, are representations for routing and scheduling problems aligned in the same latent space? How sensitive is performance to feature ordering or scaling in the CO-prefix?

If features are distinguished mainly by positional or problem-specific ordering rather than shared meaning, the model’s generality may be superficial. 

No ablations or visualization of learned embeddings are provided to demonstrate that the learned embeddings capture transferable structure.


**Ablation analysis missing for representation choices** 

No experiments isolate the effects of specific design choices (e.g., with/without µ-law scaling, or alternative encoding of continuous features). It’s unclear how sensitive performance is to these details

**Baseline selection in experimental results**

In the experimental section, results of more recent NCO models (e.g. [1], [2], [3], [4], just to name a few) should be included. Also, comparisons with specialist models seem incomplete, as COFormer often uses sampling while specialists use greedy generation mode. For FFSP for example, MatNet achieves an average makespan of 25.4 on (probably the same) FFSP test instances with sampling enabled. [4] even brings this down to 24.9, giving a performance gap of 14%, which appears very large for such small instances. 

**Missing training times**

The paper claims to improve training efficiency, but does not report wall-clock training times.

----- 
[1] Grinsztajn, N., Furelos-Blanco, D., Surana, S., Bonnet, C., & Barrett, T. (2023). Winner takes it all: Training performant rl populations for combinatorial optimization. Advances in Neural Information Processing Systems, 36, 48485-48509.

[2] Liao, Z., Chen, J., Wang, D., Zhang, Z. &amp; Wang, J.. (2025). BOPO: Neural Combinatorial Optimization via Best-anchored and Objective-guided Preference Optimization. Proceedings of the 42nd International Conference on Machine Learning

[3] Pirnay, J., & Grimm, D. G. (2024). Self-improvement for neural combinatorial optimization: Sample without replacement, but improvement. arXiv preprint arXiv:2403.15180.

[4] Hottung, A., Mahajan, M., & Tierney, K. (2024). PolyNet: Learning diverse solution strategies for neural combinatorial optimization. arXiv preprint arXiv:2402.14048.

### Questions
- How exactly is the feature ordering determined when flattening MDP states across problem types? Are keys (feature names) tokenized or embedded, or are they implicit via positional order? Can you provide a concrete example of a serialized sequence (prefix + trajectory) for two distinct problems (e.g., VRP and FFSP)?

- How does the model distinguish between, e.g., a coordinate token and a demand token, if both occupy the same token ID range? How does the model, which problem it is currently solving?

- Did you observe shared latent structure between different problems (e.g., via attention visualization or representation similarity)?

- Can the model generalize to problems with different constraint types not seen during training?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes **COFormer**, a unified transformer-based framework designed to solve diverse combinatorial optimization problems (COPs) using a single model and shared parameters. Inspired by next-token prediction in large language models, the authors formulate COP solving as a sequential decision process, tokenizing both states and actions into unified trajectories. A **CO-prefix** mechanism is introduced to compactly encode static problem information, and a **three-stage learning scheme** (dynamics imitation, policy imitation, and reinforcement fine-tuning) is proposed to improve training efficiency and generalization. Evaluated on eight distinct COPs (e.g., TSP, VRP, Knapsack, FFSP, 3DBP), COFormer matches or surpasses problem-specific NCO methods and prior generalist baselines, while demonstrating strong few-shot and even zero-shot generalization to unseen problem types.

### Strengths
- **\[S1]** The idea of building a _foundation model_ for combinatorial optimization via tokenization is interesting and, to the best of my knowledge, novel.

- **\[S2]** The **CO-prefix** mechanism is a clever and effective way to handle static problem information efficiently.

- **\[S3]** The proposed **multi-stage training pipeline** is clearly structured and modular, improving interpretability and training stability.

- **\[S4]** The **cross-problem evaluation** is extensive; the benchmark setup itself constitutes a valuable contribution to the community.

- **\[S5]** The model demonstrates **promising generalization** abilities, including few-shot and even zero-shot adaptation.

- **\[S6]** The approach exhibits **good scalability**, effectively handling large instances such as TSP1000, supporting its claim as a potential foundation model for COPs.

### Weaknesses
See “Questions” below.

### Questions
- **\[Q1]** In COFormer, both states and actions are tokenized and embedded into the same latent space, with their roles distinguished only by positional and separator tokens. While this enables a unified architecture across heterogeneous COPs, it also “blurs” the structural distinction between state and action representations that is central to the underlying MDP formulation. Could the authors elaborate on how this design choice affects learning efficiency and generalization? Specifically, have they compared this unified tokenization scheme to architectures where states and actions are modeled in separate embedding spaces (e.g., distinct encoders or dual-stream attention), which might preserve causal structure and reduce representational ambiguity?

- **\[Q2]** Many combinatorial optimization problems possess strong structural priors—for example, graphs in routing tasks or bipartite relations in scheduling—that can be compactly and efficiently represented in their native forms. In COFormer, however, these structured instances are flattened into token sequences, which may obscure their relational topology and lead to substantial increases in sequence length. Could the authors discuss how this loss of structural inductive bias impacts model scalability and sample efficiency? Have they considered incorporating lightweight structure-aware modules (e.g., graph or set encoders) that could preserve structural efficiency while maintaining the unified token-based framework?

- **\[Q3]** The three-stage training pipeline (dynamics imitation → policy imitation → RL fine-tuning) is a key design choice. Could the authors elaborate on how critical each stage is to final performance? For example, what happens if the dynamics stage is skipped, or if RL fine-tuning is applied directly after policy imitation? An ablation or sensitivity analysis would help clarify the necessity of this modular training design.

- **\[Q4]** Continuous features are discretized via μ-law transformation into 1,800 bins. Could the authors discuss the trade-off between discretization resolution and training stability? In particular, how sensitive is performance to the bin count or to the quantization noise introduced by this process?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes COFormer, a unified framework based on the Transformer architecture for solving various combinatorial optimization problems. It introduces two core techniques:
CO-prefix, which compactly encodes static problem information through prefix token blocks;
and a three-stage training strategy, which first performs dynamic and policy pretraining via imitation learning, followed by fine-tuning through reinforcement learning.

### Strengths
1.	The authors introduce the concept of next-token prediction to handle different combinatorial optimization problems and propose two methods to improve training efficiency. The effectiveness of these methods is validated across 8 combinatorial optimization problems.

### Weaknesses
1.	Although the paper claims that its idea is inspired by next-token prediction, it remains unclear how the proposed model fundamentally differs from existing Neural Combinatorial Optimization (NCO) approaches. NCO models also treat nodes as tokens and perform autoregressive next-token prediction using Transformer-like architectures. From the perspective of policy network and training algorithm, the method appears to simply adopt a different Transformer-like model and RL-like algorithm.
2.	Based on the above point, the main difference between COFormer and prior NCO methods appears to lie primarily in how problem inputs are processed and tokenized rather than in the core modeling framework. The paper proposes a unified tokenization strategy that converts heterogeneous problem inputs into sequences of tokens, claiming this enables a single model to handle various CO problems. However, this approach seems to be more of a technical workaround than a conceptual breakthrough. In essence, the heterogeneity issue in existing NCO methods arises because different problems have node features with varying dimensions when each node is treated as a token. A straightforward alternative is to flatten all node features into a one-dimensional sequence and regard each scalar feature as a token, which would already standardize the input dimensionality across different problems. From this perspective, the proposed “tokenization of each scalar” resembles such a flattening operation, and the paper does not clearly articulate what essential advantages COFormer introduces beyond NCO methods with this simple re-encoding trick.
3.	Although COFormer claims to handle diverse combinatorial optimization problems, it remains unclear whether the approach can effectively process problems with edge-level features, such as asymmetric TSP, where the token sequence could become extremely long and computationally expensive.
4.	In the experimental results, COFormer does not show a clear advantage over GOAL [1] in performance. Moreover, several problem settings evaluated in GOAL, such as asymmetric TSP, are not considered in the experiments of this paper, making the claimed generalization ability less convincing.
5.	Existing encoder-decoder NCO models have already achieved the effect of encoding static information only once [2], so the core CO-prefix method is a relatively weak innovation.
6.	There is a lack of ablation experiments separating the CO-prefix and each learning stage.
7.	In imitation learning, the input content for the two stages differs for the same network, but the method explanation, including the definition of the state and action, is unclear.
8.	The reinforcement learning training approach has already been extensively validated in the NCO field. Therefore, the conclusion in Section 4.5 — “The ability to improve performance without external supervision highlights the potential of COFormer as a general-purpose solver for a wide range of COPs.” — is not sufficiently supported.
9.	The experimental results for the COFormer RL sampling method are missing.
10.	The few-shot learning capability is only demonstrated on a few routing problems of the same type and does not include the 8 CO problems in the main results.

[1] GOAL: A generalist combinatorial optimization agent learning. ICLR 2025.

[2] MVMoE: Multi-task vehicle routing solver with mixture-of-experts. ICML 2024.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces COFormer, a Transformer-based framework designed to tackle diverse combinatorial optimization problems (COPs) within a unified sequence modeling paradigm. The method incorporates three core innovations:

(1) a CO-Prefix representation that encodes static, problem-specific features (e.g., city coordinates or item attributes) into reusable embeddings shared across tasks;

(2) a Hybrid Non-Causal Transformer that applies bidirectional attention to static tokens and causal attention to dynamic state–action sequences, improving inductive bias and efficiency; and

(3) a Three-Stage Training Paradigm comprising a dynamics pretraining stage (supervised learning of environment transitions), a policy generation stage (imitation learning from expert trajectories), and an RL finetuning stage (policy improvement beyond imitation).

Empirical evaluations across multiple benchmark problems demonstrate the potential of COFormer as a general-purpose neural solver capable of handling various COPs within a single architecture, highlighting its promise toward a unified foundation model for combinatorial optimization.

### Strengths
1) Unified and Efficient Framework – The proposed CO-Prefix abstraction elegantly decouples static problem representations from dynamic sequences, enabling a single model to generalize across diverse COPs. This is a step toward truly general-purpose neural optimizers.

2) Three-stage training paradigm – The proposed Dynamics Forward → Policy Generation → RL Finetuning pipeline is conceptually well-motivated. It mirrors the classical model-based RL structure — learning environment dynamics first, then imitation, and finally reinforcement fine-tuning.

3) Promising empirical direction – The reported results show that COFormer can outperform prior methods on several small-scale COPs, suggesting potential for broader applicability.

### Weaknesses
1) Lack of empirical validation for key components
The Dynamics Forward Stage—one of the paper’s most central contributions—is not ablated or analyzed in isolation.
Without direct comparisons (e.g., COFormer without dynamics vs. full COFormer), it remains unclear whether this stage truly improves optimization performance or merely stabilizes training. This omission significantly weakens the empirical credibility of the claimed contributions.

2) Limited analysis of generalization in the Few-Shot Ability section
It remains unclear how the model’s few-shot capability extends beyond similar routing domains—such as to packing (3DBP) or scheduling (FFSP)—especially given that the proposed Dynamics Forward Stage should, in theory, facilitate more transferable representations across different COP families.

3) Limited experimental coverage and baseline clarity
Table 1 does not include RL fine-tuning results for 3DBP and FFSP, and the reason for this omission is not clearly explained. Additionally, it would be helpful to clarify why MatNet and PCT were not trained with RL to ensure fair and consistent comparison across methods.

### Questions
1. Could the authors provide a quantitative ablation isolating the contribution of the Dynamics Forward Stage? What specific improvements (if any) does this stage offer in convergence speed or final solution quality?

2. Has the model demonstrated cross-task transfer, e.g., training on routing and testing on scheduling or packing problems?

3. Why were RL fine-tuning experiments for 3DBP and FFSP omitted, and why are some baselines trained differently? Were there technical limitations or theoretical reasons?

4. From Table 5, COFormer-direct-greedy and COFormer-RL-greedy exhibit only marginal differences, which raises doubts about whether the current training paradigm scales effectively to larger instances. Could the authors clarify whether the proposed framework remains computationally and performance-wise effective as problem size grows, and also provide more evidence on how it performs across different tasks such as CVRP and FFSP?

### Soundness
3

### Presentation
3

### Contribution
2
