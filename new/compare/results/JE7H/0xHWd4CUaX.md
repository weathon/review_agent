---
job_id: 50d9f070-598b-4d9d-8345-b052b4f44078
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 0xHWd4CUaX.pdf
paper: Contrastive Code Graph Embeddings for Reinforcement Learning-Based Automated Code Refactoring
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work addresses contrastive representation learning on code graphs combined with reinforcement learning for automated code refactoring, which clearly falls within ICLR topics (representation learning, RL, graph learning).

## Minimum Quality
Pass ✅.  
The paper contains all required sections (Abstract, Introduction, Related Work, Method/Approach, Experiments/Results, Discussion, Conclusion). While there are issues with novelty, clarity, and technical completeness, they do not rise to the level of an automatic desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
The only meta-text is the short statement on LLM usage in Section 8, which is transparent and not manipulative. No hidden instructions or prompt-injection attempts are present.

---

# Expected Review Outcome:

## Summary

The paper proposes an RL framework for automated code refactoring that uses a contrastively pre-trained graph encoder over code graphs to obtain “refactoring-aware” embeddings. These embeddings are incorporated into a composite reward (combining static code metrics, embedding movement, and a semantic-preservation signal from differential testing) and also feed a graph-attention policy network, with an additional embedding-guided exploration mechanism. Experiments on several code datasets report improvements over rule-based tools, learning-based baselines, and prior RL methods, along with ablations and a small cross-language evaluation.

## Strengths

1. **Conceptual combination of contrastive GNN pretraining and RL for refactoring.**  
   The idea of first learning code graph embeddings through syntax-preserving augmentations (Section 4.1, Eq. (4)) and then using those embeddings both as policy input and as part of the reward is a coherent and potentially impactful design. Using contrastive learning to bias RL toward structure-aware and semantics-aware regions of the state space is, at least at the level of application to automated refactoring, interesting.

2. **Metric fusion reward is thoughtfully constructed, at least conceptually.**  
   The composite reward in Eq. (5) that includes normalized static code quality metrics, embedding dynamics \(\Delta \mathbf{h}_t\), and a semantic-preservation term is a reasonable attempt to balance syntactic cleanups, representation-level changes, and behavioral safety. The use of a saturating \(\tanh(\beta \Delta\mathbf{h}_t)\) term suggests some attention to stabilizing gradients and avoiding unbounded rewards.

3. **Empirical results show consistent gains (if taken at face value).**  
   Table 1 shows that the proposed method improves over all listed baselines across all five metrics (SI, SP, ED, MG, GS). For instance, SI increases from 79.4 (NeuroRefactor) and 77.8 (GraphRL) to 83.7, while SP rises to 93.8 and ED drops to 0.36. These multi-dimensional gains, if reproducible, indicate that the approach can simultaneously improve syntactic quality, maintain semantics, and generalize better to unseen projects.

4. **Ablation study provides some evidence that each component helps.**  
   Table 2 on the Refactory dataset shows non-trivial drops when removing contrastive pretraining (SI: 83.7 → 76.2), embedding-based reward terms (SI: 83.7 → 79.5), or semantic tests (SP: 93.8 → 85.2), and when replacing embedding-guided exploration with random exploration. This supports the claim that the three main ingredients (encoder, embedding reward, semantic tests / exploration) all contribute.

5. **Figures support some of the story.**  
   - **Figure 1** shows learning curves comparing the proposed method to GraphRL, where the proposed method reaches ~0.9 normalized reward by around 15k episodes while GraphRL only gets there close to 25k episodes. This is compelling evidence that the pre-trained embeddings accelerate RL convergence and improve final performance.  
   - **Figure 2** shows a reasonably strong positive correlation (reported Pearson \(r=0.72\)) between embedding movement \(\Delta h\) and syntactic improvement (SI). This is at least suggestive that the embedding dynamics term in the reward is not just noise and has some relationship to quality changes.  
   - **Figure 3** visualizes the changing dominance of reward components across refactoring stages, showing code quality metrics dominating early, then embedding dynamics increasing in relative contribution later. This aligns with the intended design and provides some interpretability of how the agent uses different signals over time.

6. **Problem setting is relevant and practically motivated.**  
   Automated refactoring is an important problem in software engineering; integrating semantic checks (Section 4.5) and working on real-world datasets like Refactory, CodeRef, and BigCloneBench increases potential practical value. The cross-language generalization experiment (Table 3) hints at the promise of language-agnostic graph embeddings.

## Weaknesses

I see several significant issues around novelty/positioning, technical completeness, and experimental rigor. I list them in some detail because they directly affect whether this is ready for ICLR.

1. **Weak novelty and incomplete positioning vs. prior contrastive code and RL work.**  
   The core recipe of “contrastive representation learning + RL” is well explored in other domains and increasingly in code. The paper cites some code representation works (Syncobert, GraphCodeBERT), but it omits and does not compare to multiple directly relevant efforts: contrastive pretraining specifically for code (e.g., ContraCode, TransformCode, GCC-style graph contrastive coding), and prior combinations of contrastive learning with RL (e.g., CURL for visual RL, graph contrastive RL variants). The proposed method uses largely standard ingredients:
   - InfoNCE contrastive loss (Eq. (4)) on augmented graphs,
   - a GNN encoder (here a GAT),
   - PPO for RL,
   - plus a handcrafted combination of static metrics, latent movement, and a test-based semantic term.
   
   What is really new beyond, say, “apply off-the-shelf graph contrastive pretraining to code graphs, then plug them into an RL policy and reward”? There is no clear, formal definition of what makes the embeddings “refactoring-aware” beyond the choice of augmentations, and that idea very closely resembles existing code-oriented contrastive frameworks. Without a more principled distinction and careful related work comparison, the contribution risks being seen as an incremental adaptation of standard techniques to another application.

2. **Very incomplete and sometimes inconsistent description of the RL environment and action space.**  
   The RL formulation in Section 3.1 and Section 4 is high-level and omits key details:
   - What exactly constitutes a state \(s_t\)? Is it the entire method’s code graph, a project-level graph, or something else?  
   - What are the possible refactoring actions \(A\)? Are they standard refactoring operators (extract method, rename variable, inline function, move field), or are they lower-level AST edits?  
   - How are episodes defined and terminated (max number of edits, until no further improvement, etc.)?  
   - Is the environment offline (using fixed datasets) or online (applying edits and re-running tools/tests)?  

   Without a clear description of the action space and environment dynamics, it is very difficult to assess whether the proposed method is actually practical, or to reproduce the work. This is not a small omission; it is central to RL-based refactoring.

3. **Reward formulation has conceptual and mathematical inconsistencies.**  
   - In Section 4.2, \(\delta_t\) is defined as \(\mathbb{I}[\text{test}(G_t) = \text{test}(G_{t-1})]\), a binary indicator, and used in Eq. (5) as a semantic-preservation factor through a penalty \(-\gamma (1 - \delta_t)\).  
   - In Section 4.5, however, \(\delta_t\) is redefined via Eq. (8) as  
     \[
       \delta_t = 1 - \frac{1}{L} \sum_{k=1}^L \mathbb{I}[\text{trace}_k(G_{t-1}) \neq \text{trace}_k(G_t)] ,
     \]
     which is a *continuous* similarity score in \([0,1]\) based on normalized Hamming distance of traces, not the earlier discrete indicator. The paper never reconciles these two definitions or explains which one is actually used in Eq. (5) during training. This is a non-trivial discrepancy because the magnitude and gradients of the semantic term depend heavily on whether it is binary or fractional. At present, Eq. (5) is under-specified and mathematically inconsistent with the later definition of \(\delta_t\).

   - Similarly, the embedding dynamics term \(\Delta \mathbf{h}_t = \|\mathbf{h}_t - \mathbf{h}_{t-1}\|_2\) is rewarded via \(\alpha \tanh(\beta \Delta\mathbf{h}_t)\). This appears to reward *larger* changes in embedding space, which conflicts with the goal of learning refactoring steps that make targeted, not arbitrary, changes. There is no justification for why larger \(\|\mathbf{h}_t - \mathbf{h}_{t-1}\|\) should be beneficial beyond correlation plots (Figure 2), nor any regularization to avoid encouraging massive but semantically meaningless edits early in training.

   - The specification of \(\mathbf{w}_q\) in the reward (Eq. (5)) is given as \([0.4, 0.3, 0.3]\) in Section 5, but it is never clearly tied to specific metrics in \(\mathbf{q}_t\) (cyclomatic complexity, coupling, style violations). How these are normalized and aggregated into a single scalar reward is only partially described via \(\phi(\cdot)\), but critical details (e.g., how multiple code metrics are combined, whether lower complexity is always rewarded, etc.) are absent.

4. **Exploration distribution (Eq. (6)) is underspecified and arguably ill-posed.**  
   Equation (6) defines
   \[
     \pi_{\text{explore}}(a|s) \propto \exp\left(-\tfrac{1}{2} (\mathbf{h}_s - \mathbf{h}^*)^\top \Sigma^{-1} (\mathbf{h}_s - \mathbf{h}^*) \right),
   \]
   which depends only on the *state* embedding \(\mathbf{h}_s\), not on the action \(a\). As written, the right-hand side does not vary with \(a\), so it cannot define a proper distribution over actions. It is unclear whether this term is meant to modulate the overall probability of exploring at state \(s\) or to reweight action logits; the paper merely says it “incorporates Mahalanobis distance to prototype states” and biases exploration toward regions with “effective refactorings”. There is also no description of how \(\pi_{\text{explore}}\) is combined with the PPO policy \(\pi_\phi\) (e.g., mixture, additional bonus in the advantage function, etc.). This is a significant gap in the technical description that raises questions about what exactly was implemented.

5. **Contrastive learning setup is not sufficiently detailed, and Eq. (2) even has notation errors.**  
   - The generic InfoNCE loss (Eq. (2)) uses \(\mathbb{P}_{k \neq i}\) in the denominator, which appears to be a typo for an indicator or masking operator (\(\mathbb{1}_{k \neq i}\)); as written it is mathematically meaningless.  
   - The main pretraining objective for graphs (Eq. (4)) claims to use “mean pooling of node representations”, but there is no discussion of whether multi-head attention is used, what edge types are included, how different graph types (AST, CFG, dataflow) are combined, and how the temperature \(\tau\) and batch sampling interact with large code graphs.  
   - The syntax-preserving augmentations (subtree masking, edge rewiring, identifier shuffling) are described at a high level, but there is no concrete operational description that would guarantee semantic preservation. For example, “edge rewiring” of non-critical control flow edges is underdefined: which edges are considered non-critical and how is program validity ensured?

   For a paper whose central claim is about contrastive pretraining of code graphs, these missing details undermine the technical clarity and reproducibility.

6. **Experimental methodology is under-specified and lacks robustness checks.**  
   Several issues stand out when examining Table 1, Table 2, Table 3, and the text in Section 5:

   - **Evaluation setup is unclear.** It is not stated how the three datasets (Refactory, CodeRef, BigCloneBench) are used jointly. Are the metrics in Table 1 aggregated across datasets, computed only on Refactory, or averaged? How are training, validation, and test splits done, especially for RL-based methods?  
   - **No variance, no statistical tests.** All tables present single-point metrics with no standard deviations, confidence intervals, or multiple random seeds. For RL in particular, variance can be large; without sensitivity analysis, it is hard to trust that the differences (e.g., SI 83.7 vs. 79.4) are statistically meaningful.  
   - **Baselines and their configurations are opaque.** For example, GraphRL is cited as a survey paper, not necessarily a concrete refactoring system with the exact metrics used here; it is not clear what specific implementation was used. For NeuroRefactor and RLRefactor, hyperparameters, reward configurations, and training budgets are not described. This raises concerns about fair comparison.  
   - **Metric semantics are not always consistent.** Table 1 states “higher is better” for all metrics, yet Edit Distance (ED) is usually a *lower-is-better* metric measuring difference from the original code. The table shows ED decreasing (0.41 → 0.36) which aligns with “lower is better”, but the text labels ED as “higher is better”. This inconsistency suggests that either the ED definition or the table header is incorrect, which directly affects interpretability of Table 1’s results.

7. **Risk of metric leakage between reward and evaluation.**  
   The reward in Eq. (5) explicitly uses traditional metrics \(\mathbf{q}_t\) like code smells and style violations, which are measured with tools such as PMD and Checkstyle. Syntactic Improvement (SI), defined as reduction in the *same* violations, is then used as a primary evaluation metric in Table 1. This likely means the RL agent is optimized directly on the test metric, which inflates reported gains and does not reflect generalization to unseen notions of quality. At minimum, the paper should discuss this coupling explicitly and either:
   - use separate metrics or held-out rule sets for evaluation, or  
   - provide additional human or independent assessments of refactoring quality.  

   As it stands, SI improvements may largely reflect “overfitting” to the static analysis tools used in the reward.

8. **Semantic testing via symbolic execution is not convincingly justified as “lightweight”.**  
   Section 4.5 proposes computing \(\delta_t\) by generating test cases through symbolic execution and comparing execution traces. While this is theoretically appealing, there are several unaddressed practical issues:
   - Symbolic execution is notoriously expensive and does not scale well to large or complex code; however, Section 6.1 calls the equivalence checker “lightweight” without providing timing measurements, limits on code size, or mitigation strategies.  
   - No runtime or overhead analysis is presented, and the experimental section contains no ablation isolating the cost of semantic testing or discussing time per episode.  
   - It is not clear how many test cases \(L\) are generated per method, nor how timeouts or path explosion are handled.  

   Given this component is claimed to be central for semantic preservation and is heavily used in Eq. (5), the lack of analysis is a serious gap.

9. **Cross-language generalization evaluation is too shallow to support the claims.**  
   Table 3 compares the proposed method to PyLint (Python) and Cppcheck (C++) on SI and SP, showing modest improvements in SI and slight drops in SP. However:
   - There is no description of how refactoring actions are defined for Python and C++ when the model is trained only on Java. Are the same graph augmentations and action sets used?  
   - The table reports only two metrics on unspecified datasets; no details on which Python / C++ corpora, how large they are, or how the agent interacts with them are provided.  
   - The claim that the model “outperforms language-specific rule-based tools” is overstated given the narrow metrics and missing context.

10. **Writing quality and precision are not at ICLR standard.**  
    The paper contains numerous grammatical errors and awkward phrasing (“recent lemon deep learning technologies”, “has been a study of note to translate the defect of static approaches”, “that most often do last year”, etc.). There are also several inconsistent notations (e.g., mixing \(\delta_t\) indicator vs. continuous value, use of \(\mathbb{P}\) in Eq. (2)), and some informal or incorrect technical claims (e.g., claiming the graph attention mechanism “increases in a linear fashion with the number of edges” without discussing attention heads or constant factors). While not fatal alone, this level of sloppiness complicates interpretation of the technical content and suggests that important details may have been glossed over.

11. **Reproducibility concerns.**  
    There is no mention of releasing code or pre-trained models, and given the missing specifications for the RL environment, augmentations, and symbolic execution setup, reproducing the experiments from the main text would be very challenging. For an applied RL + representation learning paper, this is a nontrivial concern.

Overall, while the high-level idea is appealing, these weaknesses, especially the technical inconsistencies in the reward, incomplete description of exploration and environment, and experimental under-specification, prevent the paper from reaching ICLR quality in its current form.

## Potentially Missing Related Work

The following works are, in my view, directly related and should be cited and discussed, especially in Sections 2 and 4:

1. **Jain et al., “Contrastive Code Representation Learning”, 2020 (ContraCode).**  
   This paper introduces a contrastive pretraining framework specifically for learning code representations that are invariant to certain code transformations, which closely matches the stated goal of learning refactoring-aware representations via structural augmentations. It should be discussed in Section 2.2 and Section 4.1, and possibly compared experimentally or at least contrasted conceptually.

2. **Srinivas et al., “CURL: Contrastive Unsupervised Representations for Reinforcement Learning”, 2020.**  
   CURL explicitly combines contrastive representation learning with RL to improve sample efficiency, directly paralleling the proposed combination of contrastive code embeddings with PPO. It should be cited in Sections 3.2 and 4.6, with a discussion of similarities and differences in how the representation is used in the policy and reward.

3. **Qiu et al., “GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training”, 2020.**  
   GCC proposes self-supervised graph contrastive pretraining, which is very relevant to the use of graph-based contrastive learning on code graphs here. Section 4.1 should reference GCC when motivating augmentations and the choice of InfoNCE on graphs.

4. **Xian et al., “TransformCode: A Contrastive Learning Framework for Code Embedding via Subtree Transformation”, 2023.**  
   TransformCode uses subtree transformations as augmentations for code contrastive learning, which is highly related to the “subtree masking” and “identifier shuffling” augmentations in Section 4.1. This should be cited as prior art and used to better position the novelty of the proposed encoder.

5. **Li et al., “CoMatch: Semi-supervised Learning with Contrastive Graph Regularization”, 2020.**  
   While focused on semi-supervised learning, CoMatch uses contrastive regularization on graphs, which is conceptually relevant to the way the paper leverages graph embeddings within RL. It could be mentioned in the broader context of contrastive graph learning in Section 2.2 or 3.2.

6. **Liu & Wang, “Graph Contrastive Learning with Reinforced Augmentation”, 2023.**  
   This work ties reinforcement learning and graph contrastive learning via reinforced augmentation policies. It is particularly relevant to the claimed “embedding-guided exploration” in Section 4.3 and should be referenced when discussing the design of augmentations and exploration strategies.

7. **Bi & Xuan, “Graph Adversarial Refinement for Robust Code Fixes: Enhancing Policy Networks via Structure-Aware Contrastive Learning”, 2025.**  
   This paper looks at reinforcement learning for code fixes with structure-aware contrastive learning, which appears quite close in spirit to this work’s code refactoring setting. It should be discussed in Section 2.3 as a closely related RL + contrastive approach for code, to better delineate the contribution.

8. **Zhang et al., “CLG-Trans: Contrastive Learning for Code Summarization via Graph Attention-Based Transformer”, 2023.**  
   CLG-Trans combines contrastive learning with graph attention architectures for code, which is directly relevant to the GAT-based contrastive encoder in Section 4.1. It should be cited when discussing the choice of attention-based encoders and contrastive objectives.

Including and properly discussing these works would significantly strengthen the related work section and clarify how the proposed method differs from existing contrastive code and RL frameworks.

## Questions

Author responses that substantially clarify or address the following points could change my assessment:

1. **Precise RL environment and action space.**  
   - How exactly are states, actions, and episodes defined?  
   - What are the refactoring primitives available to the agent (e.g., extract method, rename, move, inline, AST-level edits)?  
   - How is the environment implemented on datasets like Refactory and CodeRef? Is it a simulated environment replaying known refactorings, or does the agent operate directly on code and re-run static analyzers / tests?

2. **Clarification of \(\delta_t\) and the semantic reward component.**  
   - Which definition of \(\delta_t\) is actually used in Eq. (5): the binary indicator from Section 4.2 or the continuous similarity from Eq. (8)?  
   - If Eq. (8) is used, how is the penalty term \(-\gamma(1 - \delta_t)\) scaled relative to the other terms, and how many test cases \(L\) are generated per code fragment?  
   - Can you provide empirical results on the computational overhead of symbolic execution and trace comparison, and discuss how this scales with codebase size?

3. **Exploration mechanism in Eq. (6).**  
   - How is \(\pi_{\text{explore}}(a|s)\) combined with the base PPO policy \(\pi_\phi(a|s)\)? Is it a separate distribution used to sample a fraction of steps, or is it an additive bonus in the advantage computation?  
   - As written, Eq. (6) is independent of \(a\); can you provide the correct formulation and describe the implementation details?

4. **Reward design and potential metric leakage.**  
   - Given that SI and SP (Table 1) are tightly coupled to the metrics and tests used in the reward function (Eq. (5)), how do you ensure that the agent is not simply overfitting to the evaluation metric?  
   - Have you tried evaluating the model with *different* rule sets or additional human-judged code quality measures that are not part of the reward?

5. **Experimental robustness and baselines.**  
   - How many random seeds were used for each method in Tables 1–3? Could you provide variances or confidence intervals, at least in the rebuttal?  
   - For RL baselines (RLRefactor, GraphRL, NeuroRefactor), what training budgets and hyperparameters were used, and how did you ensure fair comparison (e.g., equal number of environment steps, same reward metrics)?

6. **Details of graph construction and augmentations.**  
   - What exact graph representation is used (AST only, AST+CFG, AST+dataflow)? How are edge types encoded?  
   - For each augmentation (subtree masking, edge rewiring, identifier shuffling), can you describe specific rules to guarantee syntactic and semantic validity? Do any augmentations ever produce invalid code graphs, and if so, how are they filtered?

Concrete, detailed answers to these questions, along with any updated experiments or corrections to equations, would significantly improve my confidence in the work.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A. The paper focuses on automated refactoring of source code without direct human subjects, sensitive data, or obvious dual-use concerns beyond standard software engineering impacts.

## Soundness Rating

2: fair.  
The high-level idea is reasonable and some empirical results look promising, but there are important technical inconsistencies (especially around the reward and exploration formulations), missing details in the RL setup, and insufficiently robust experimental methodology.

## Presentation Rating

2: fair.  
The structure is standard and figures/tables are informative, but the writing has many grammatical issues, several equations have notation errors or inconsistencies, and key methodological details are missing or vaguely described, which hampers comprehension.

## Contribution Rating

2: fair.  
The application of contrastive graph embeddings to RL-based refactoring is relevant and may have some practical value, but novelty over existing contrastive code representation and RL work is not clearly articulated, and the incomplete technical exposition limits the contribution’s clarity and impact.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper presents an appealing high-level idea and some encouraging results (as in Table 1 and Figure 1), but the combination of weakly articulated novelty, technical inconsistencies (notably around Eq. (5) and Eq. (6)), under-specified experimental setup, and missing key details makes it fall short of ICLR standards in its current form. With substantial clarification, stronger positioning relative to existing contrastive code/RL literature, and more rigorous experiments, it could become a solid contribution.

## Reviewer Confidence

4: confident.  
I am familiar with contrastive representation learning, graph-based code modeling, and RL for program transformation, and I examined the equations and experimental sections carefully. Some implementation details are missing, so there is room for clarification in rebuttal, but it is unlikely that I have fundamentally misunderstood the core claims.