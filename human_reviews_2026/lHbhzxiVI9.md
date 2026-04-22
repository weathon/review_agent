# Incentivizing LLM Reasoning via Reinforcement Learning with Functional Monte Carlo Tree Search

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
In this work, we propose ***R**einforced **F**unctional **T**oken **T**uning* (RFTT), a novel reinforced fine-tuning framework that empowers Large Language Models (LLMs) with learn-to-reason capabilities. Unlike prior prompt-driven reasoning efforts, RFTT embeds a rich set of learnable functional tokens (*e.g.*, \<analyze\>, \<verify\>, \<refine\>) directly into the model vocabulary, enabling chain-of-thought construction with diverse human-like reasoning behaviors. Specifically, RFTT comprises two phases: (1) supervised fine-tuning performs prompt-driven tree search to obtain self-generated training data annotated with functional tokens, which warms up the model to learn these tokens for initial reasoning capability; and (2) online reinforcement learning further allows the model to explore diverse reasoning pathways through functional token sampling without relying on prompts, thereby facilitating effective self-improvement for functional reasoning. Extensive experiments demonstrate the superiority of the proposed RFTT on mathematical benchmarks and highlight its strong generalization capability to other general domains. Moreover, the performance of RFTT exhibits consistent gains with increased test-time computation through additional search rollouts. Our code and dataset are available at https://github.com/sastpg/RFTT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Reinforced Functional Token Tuning (RFTT), a novel reinforcement-based fine-tuning framework that equips large language models with learn-to-reason capabilities. Unlike prior prompt-driven reasoning methods, RFTT introduces learnable functional tokens (e.g., <analyze>, <verify>, <refine>) directly into the model’s vocabulary to enable structured and human-like reasoning behaviors. The framework consists of two stages: supervised fine-tuning with functional tree search to teach token usage, and online reinforcement learning to encourage autonomous exploration of reasoning pathways. Experimental results on mathematical and general reasoning benchmarks demonstrate that RFTT significantly outperforms baselines and continues to improve with increased test-time computation.

### Strengths
1. The introduction of learnable functional tokens provides an interpretable and flexible mechanism for modeling reasoning behaviors within the LLM’s vocabulary.
2. The two-phase framework combining supervised functional tree search with reinforcement learning.
3. Experiments demonstrate consistent and significant improvements on mathematical and general reasoning benchmarks

### Weaknesses
1. The training framework may computationally expensive due to the integration of Monte Carlo Tree Search, which limits scalability to larger models.
2. The semantics of functional tokens are predefined by design, which may constrain flexibility and introduce human bias into the reasoning process.
3. Rewarding reasoning chains may cause reward hacking, where the model learns to produce text that merely appears logically sound rather than genuinely correct.

### Questions
1. Please provide a analysis of the potential **reward hacking** problem that may arise when rewarding reasoning chains, where the LLM could exploit the reward model by generating answers that align with its biases or merely appear correct to maximize the reward.
2. How sensitive is RFTT to the choice and number of functional tokens? Would the performance drop if some tokens (e.g., <verify>, <refine>) were removed or replaced?
3. During functional tree search, what criteria are used to select the optimal reasoning branch?
4. How to get the dependencies between actions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Reinforced Functional Token Tuning (RFTT), a two-phase framework to enhance LLM reasoning via functional token–guided Monte Carlo Tree Search. During the SFT phase, the model learns reasoning behaviors, such as <analyze>, <verify>, <refine>) from self-generated trajectories. In the RL phase, it explores diverse reasoning paths using these tokens as actions, optimized by process and outcome rewards. Experiments on math and general reasoning benchmarks show improved performance over GRPO and ReFT, suggesting functional tokens enable structured, self-improving reasoning in smaller LLMs.

### Strengths
1. The paper is clearly written and well organized, with a logical flow from motivation to methodology and experiments. Figures and tables effectively illustrate the framework and results.

2. The method is sound, with detailed descriptions of the two-phase (SFT + RL) pipeline, reward design, and MCTS procedure. All experiment settings are reasonable and detailed across different domain and model scale size.

### Weaknesses
1. The overall framework is somewhat similar to Satori (Feb 2025) and TreeRL (Jun 2025), as well as other recent RL-based reasoning frameworks that employ token-level actions or tree-structured exploration. It would be helpful to clarify how your method differs from these tree-RL-like approaches and to provide side-by-side results with these baselines [1, 2].

2. The current comparisons include only ReFT and GRPO, omitting strong recent baselines such as TreeRL and DAPO. Adding at least one of these baselines—or discussing the expected difference under a comparable compute budget—would make the evaluation more convincing [2, 3].

3. The paper attributes its improvement to high-entropy branching, but it remains unclear whether this approach provides a clear advantage over existing entropy-based RL methods, which often achieve similar goals without the added cost of increased training time. Since several recent studies have explicitly examined entropy mechanisms to enhance RL for reasoning LMs [4], a comparison with those methods would make the claim more solid and help clarify whether the added training cost is justified.


[1] Satori: Functional Token Guided Reinforcement Learning for Structured Reasoning, Feb 2025.
[2] TreeRL: Tree-Structured Reinforcement Learning for Multi-Step Reasoning, Jun 2025.
[3] DAPO: Diversity-Aware Policy Optimization for LLM Reasoning, May 2025.
[4] The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models, May 2025.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes **Reinforced Functional Token Tuning (RFTT)**, a two-phase framework to endow LLMs with learn-to-reason abilities by internalizing *functional tokens*. Phase I performs SFT on training data generated by a functional-prompt-guided MCTS that stitches together correct and incorrect branches via cross verification and branch merging, yielding structured, token-annotated reasoning paths. Phase II replaces prompt guidance with token-guided exploration and optimizes with Reinforce++ under a KL penalty to a reference SFT model, optionally using a PRM for step-level rewards. On math benchmarks (MATH-500, GSM8K, SVAMP, OlympiadBench, AMC), RFTT improves small models (≤10B) markedly over CoT, self-consistency, ReFT, and GRPO, and shows monotonic gains with more inference-time rollouts. Ablations indicate benefits from SFT warmup, MCTS, and PRM, while token masking highlights the importance of `<verify>` and `<refine>`. The authors also report training dynamics, cost trade-offs, and limitations. Overall, the contribution is a principled way to compress a vast vocabulary-level action space into a compact, semantically meaningful policy over functional tokens, enabling more effective exploration and reinforcement of reasoning behaviors.

### Strengths
The work is original in elevating “functional prompts” (e.g., from rStar-style search) into learnable tokens that become part of the model’s action space and are optimized end-to-end. This shift from external, prompt-level constraints to internalized, token-level policies is a clean conceptual advance that reduces search branching at the word level while preserving diversity (§3.1–3.3, pp.3–5). 

The qualityof the methodology is supported by a careful algorithmic design—cross-verification and branch-merging produce structured SFT targets (§3.2, p.4), token-guided MCTS with UCT balances exploration/exploitation (Eqns. (1),(5)–(8), pp.4,23), and PRM-based process rewards address long-chain credit assignment (§4.4, Table 3 on p.8). Empirically, the paper demonstrates clear, consistent gains across multiple backbones (Qwen-3-4B, Qwen-2.5-7B, LLaMA-3.1-8B) and datasets, with strong improvements versus ReFT and GRPO and favorable scaling with additional rollouts (Table 1, Figure 2, pp.6–7). 

The exposition is clear, with a unifying diagram (Figure 1, p.3), explicit token definitions and prompts (Appendix C, pp.18–19), and transparent training/eval details (§4.2, p.6). RFTT offers a general recipe to couple MCTS and RL via a compact, behavior-level action space, pushing small-model reasoning toward larger-model performance, and the ablations convincingly attribute where the gains come from (Table 3, p.8; Table 4 and Figure 3, p.8).

### Weaknesses
1. The OOD benchmarks are still somewhat in-distribution, as they are essentially math reasoning problems. It would be interesting to test models on other reasoning benchmarks, like FOLIO (logical reasoning), CruxEval (code reasoning), TableBench (table reasoning), etc.

2. Baseline methods are not comprehensive. Performance comparisons with more recent tuning based methods (e.g. rStar-math, Satori, MCTS-DPO) and non-tuning based methods (e.g. rStar, RAP) would bring more insights into the strength of this work.

3. Scalability test of the method is missing. Experiements on smaller/larger models would also be interesting.

4. Showing these heuristic designs of reasoning actions also work on other reasoning domains, like code generation, for synthesizing training data, would also strengthen the work.

### Questions
1. Could you elaborate the motivation of using PRM? How does it compare with ORM?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes to train LLMs to reason by introducing new special tokens (i.e., functional tokens in this paper) and combining it with tree search. The proposed method warm-up the model with SFT data that teach the model how to follow the functional tokens. The second stage trains the model to generate diverse reasoning paths with MCTS. The results show that the proposed method (RFTT) improves the performance over RL fine-tuning on math reasoning tasks.

### Strengths
The idea of combining tree search and RL during training is a promising idea since it may improve data efficiency significantly.

### Weaknesses
- Writing is not clear. See my questions.
- Training model with Equation 4 with the data collected by MCTS sampling (Equation 1) is biased. The PPO's objective needs to sample states and actions from the policy pi_old, but MCTS sampling draws actions based on UCT (Equation 1), which won't yield an accurate expectation on the clipped objective L_RL
- The ablation study (Table 3) shows the setting of ablating SFT-warmup, MCTS, and PRM, but it's unclear that if MCTS and PRM are required. One possibility is that RFT + SFT-warmup would yield similar performance.

### Questions
- Line 64: What does human-like reasoning tree mean?
e
- Line 68: What does "prompt-guided" mean?
- Line 69: Why the exploration is diverse needs to be explained.
- Line 172: How do you tell the intersection?
- Line 166-Line 180: The notations are very hard to parse.
- Line 210: Unclear what ANS predicts.
- Line 348: It's not rigourous to use "siginifcantly improve" when there are no confidence interval on the results.
- Line 352: What is self-consistency sampling?
- Line 363: I'm confused. I thought you need to add functional tokens to the dictionary also in RL/SFT training, but it reads like you only add them to the dictionary while doing tree search?
- Line 365: Without reward model or any other reward signal, how do you perform MCTS?
- Line 204: UCT score's definition is missing. Though it was a classic notion in MCTS, it's better to clarify the definition in the context of this paper.
- Line 262: What is "MATH dataset"? Needs reference.

### Soundness
2

### Presentation
2

### Contribution
2
