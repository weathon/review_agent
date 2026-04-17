Now I have enough calibration data. Let me write the final review.

## Summary

The paper proposes a framework that combines contrastive pre-trained code graph embeddings with reinforcement learning for automated code refactoring. The approach consists of three components: (1) a syntax-guided contrastive encoder that learns invariant representations of code graphs through structure-preserving augmentations, (2) a composite reward function fusing traditional code quality metrics with embedding dynamics and differential test verification, and (3) a graph attention policy network with embedding-guided exploration. Experiments on three datasets claim improvements over rule-based, learning-based, and RL-based baselines across syntactic improvement, semantic preservation, and generalization metrics.

## Strengths

- **Well-motivated problem and promising direction**: The tension between syntactic improvement and semantic preservation in code refactoring is a real challenge. Leveraging contrastive pre-training to produce refactoring-aware representations that reduce reliance on handcrafted features is a sensible and timely idea.
- **Reasonable architectural composition**: The combination of contrastive pre-training (InfoNCE with graph augmentations), composite reward design, and embedding-guided exploration as a package is novel for this task domain, even though individual components are standard.
- **Ablation study design**: The systematic removal of key components (Table 2) provides some evidence for the utility of each design choice, particularly showing contrastive pre-training accounts for a significant SI drop (−7.5%) and semantic tests for SP preservation (−8.6%).
- **Cross-language transfer experiments**: Table 3 evaluates zero-shot transfer from Java to Python/C++, demonstrating that learned representations capture some transferable patterns—an attractive property for practical refactoring tools.

## Weaknesses

### Major:

- **The composite reward function contradicts the paper's motivation and has conceptual issues.** The paper explicitly motivates the approach as overcoming "the limitations of the traditional heuristic-based reward functions" and avoiding "handcrafted metrics," yet the reward (Eq. 5) is a hand-designed, manually weighted mixture of three terms with fixed hyperparameters (w_q = [0.4, 0.3, 0.3], α = 0.2, β = 1.0, γ = 0.5). No sensitivity analysis on these weights is provided. More critically, the Δh term α·tanh(β·Δh_t) directly rewards *magnitude of embedding change*, which encourages large latent-space movements per step. Figure 2 only shows correlation between Δh and SI on *completed* refactorings, not per step—there is no justification that per-step embedding displacement correlates with improvement. This risks incentivizing oscillatory or gratuitous changes. Since the reward is the core signal for RL training, this conceptual gap undermines the central claim.

- **The RL environment (states, actions, transitions) is insufficiently defined.** The paper formulates refactoring as an MDP (Sec. 3.1) but never concretely specifies: (a) what constitutes a state (method? file? repository?), (b) what the action space is (which refactoring operations? parameterized or discrete templates?), or (c) when an episode terminates. The qualitative examples (Sec. 5.5) describe non-trivial transformations like strategy pattern introduction, which are far more complex than local AST edits, yet no mechanism for such higher-level refactorings is described. Without these definitions, the experimental results (1M environment steps, Table 1 metrics) cannot be properly interpreted or reproduced.

- **Semantic preservation mechanism is likely infeasible at stated scale.** The differential testing component δ_t (Sec. 4.5) requires extracting method signatures, generating test cases via symbolic execution, and comparing execution traces at *every* RL environment step. With 1M training steps, this is computationally prohibitive. No discussion of budgets, caching, timeout strategies, or how methods with side effects/I/O are handled is provided. Additionally, the same SP metric used in the reward is also the evaluation metric—"test case pass rate after refactoring"—creating a potential evaluation circularity if training-time and evaluation-time test suites overlap.

- **No variance, confidence intervals, or significance tests reported.** All results in Tables 1–3 are single point estimates. RL training is notoriously sensitive to random seeds and hyperparameters; without multiple runs, it is impossible to assess whether differences like 83.7% vs. 79.4% SI are statistically meaningful. Per the Exploiting Code Symmetries review and similar RL papers, this is a serious methodological gap.

- **Evaluation metric circularity.** SI is defined as "percentage reduction in code smells (PMD/Checkstyle violations)," yet PMD and Checkstyle are also baselines in Table 1. The composite reward's q_t component includes these same traditional metrics. If the RL agent is trained to optimize PMD/Checkstyle scores, then comparing against PMD/Checkstyle as baselines while also using those metrics in the reward is partially circular. The paper does not address whether the traditional metrics in the reward are the same as those used to evaluate SI.

### Minor:

- **Contrastive augmentations may not correspond to refactoring.** The augmentations used (subtree masking, edge rewiring, identifier shuffling) are generic syntax-preserving perturbations. Real refactoring operations (extract method, introduce guard clause, strategy pattern) heavily alter structure while preserving behavior—the opposite direction. The paper claims the encoder learns "refactoring-aware" representations, but the augmentations are not designed to mimic refactoring patterns. The only empirical link is the r=0.72 correlation in Fig. 2, which shows correlation at final states only.

- **Cross-language generalization evaluation is incomplete.** Table 3 compares against only rule-based tools (PyLint, Cppcheck) for Python/C++, not against any learning-based methods in a transfer setting. This leaves open whether simpler transfer approaches would also outperform rule-based tools.

- **Ablation study is incomplete.** Table 2 omits ED and GS metrics without explanation. It is also unclear whether ablation variants use the same number of training steps and hyperparameters.

- **Writing quality issues throughout.** The paper contains numerous errors: "Recent lemon deep learning technologies" (typo/artifact), "Remark 1: The second fundamental domain is a fundamental constant" (spurious text next to Eq. 2), duplicate equation numbering (two Eqs. 6 and 7), a Section 8 ("The Use of LLM") that merely states LLM was used to polish writing, and generally unpolished prose. These errors are distracting and reduce confidence in the technical content.

- **Several references have unusual sourcing** (e.g., "Marvellous et al., 2025" cited from researchgate.net; "Polu, 2025" from academia.edu). While I do not question their existence, the sourcing pattern is atypical for an archival venue and raises questions about the rigor of the related work section.

### Trivial:

- The transition from GCN-style message passing (Sec. 3.3) to GAT (Sec. 4.1) is abrupt and without clear justification.
- The exploration distribution π_explore (Eq. 6) is defined but never explicitly stated how it integrates with PPO during training.

## Nice-to-Haves

- Comparison with LLM-based refactoring approaches (e.g., prompting code LLMs), given the 2026 publication context.
- Sensitivity analysis on reward weights (α, β, γ, w_q) to demonstrate the framework is not fragile to these choices.
- Actual code examples (before/after) for the qualitative patterns described in Sec. 5.5.
- Failure analysis of the ~6% of refactorings that break tests.
- Learning curves (reward/SI/SP vs. training steps) to demonstrate convergence properties.
- Computational cost analysis (pre-training time, RL fine-tuning time, inference latency).

## Removed Points

These points were flagged by reviewers but removed or weakened for the following reasons:

- **"Comparison unfairness in favor of the author's method"** — The harsh critic raises this, but per the rules, I should not criticize unfairness where the asymmetry favors baselines. However, the circularity concern (SI evaluated via the same metrics used in the reward) IS kept as valid, as it's a methodological concern, not a baseline fairness concern.

- **"Missing LLM-based baselines"** — Demanding comparison with LLM-based refactoring tools is partially scope creep; this paper's scope is RL + contrastive learning for refactoring, not LLM-based refactoring. Moved to Nice-to-Have.

- **"Missing related works"** — Per rules, I do not flag missing related works since I cannot verify their existence.

- **"Ablation doesn't compare against simpler policies"** — This is a reasonable suggestion but the ablation already removes components systematically. Moved to Nice-to-Have as a suggestion.

- **"Reproducibility concerns about hyperparameters or implementation details"** — Per rules, minor implementation details not included in a submission are not a weakness. However, the *absence* of environment/action-space definitions is a major methodological gap and is kept.

- **"References may not exist or be unverifiable"** — Per rules, I must assume cited works exist. However, I note the atypical sourcing pattern as a minor comment.

## Novel Insights

The most insightful observation across reviews is the fundamental tension between the paper's stated motivation ("avoiding handcrafted reward functions") and its actual method (a hand-designed, manually weighted composite reward). The contrastive encoder's role reduces to providing a *single component* of this handcrafted reward (the Δh term), rather than replacing handcrafting as claimed. This means the paper's core contribution is less about "learning refactoring-aware representations that replace handcrafted metrics" and more about "adding a learned embedding signal to an otherwise handcrafted reward"—a more modest but more honest claim. Additionally, the contrastive augmentations used are generic syntax-preserving perturbations rather than refactoring-mimicking transformations, so calling these "refactoring-aware" representations is an overstatement; they are more accurately "syntax-invariant" representations.

## Suggestions

1. **Redefine the reward's Δh term** to measure embedding *alignment* with high-quality states rather than raw displacement magnitude, or at minimum provide justification for why larger Δh correlates with better per-step refactoring.
2. **Clearly define the MDP**: specify states (code graph representations), actions (catalogue of refactoring operations with parameters), and termination conditions. This is essential for reproducibility.
3. **Report variance**: run with at least 3-5 random seeds and report means with standard deviations.
4. **Separate training and evaluation test suites** for semantic preservation, and discuss the computational feasibility of the differential testing component.
5. **Reframe the contribution**: rather than claiming to "replace handcrafted rewards," honestly characterize the contribution as augmenting handcrafted metrics with a learned contrastive signal.
6. **Thorough proofreading**: remove the spurious "Remark 1," fix "lemon" and other artifacts, and resolve equation numbering conflicts.

## Score and Decision

**Calibration comparison:**
- RLCF (similar: code + RL, weak evaluation, no variance): scores 3, 5, 6, 5 → **Rejected**
- RLEF (similar: code RL, limited novelty, missing ablations): scores 5, 3, 5, 5 → **Rejected**
- EA2N (similar: compositional novelty but questionable overall, poor writing): scores 3, 3, 5, 5 → **Rejected**
- RefactorBench (refactoring benchmark, better-executed): scores 6, 6, 8, 6 → **Accepted (poster)**

This paper has an interesting high-level idea but suffers from fundamental methodological gaps (undefined MDP, conceptually problematic reward term, potentially infeasible semantic preservation mechanism), inadequate evaluation (no variance, potential metric circularity), and poor writing quality (multiple artifacts, typos, spurious text). It is below the quality threshold of RLCF/RLEF (which were also rejected) because those papers at least had well-defined setups, even if their evaluation was limited.

MY FINAL SCORE: <pineapple>3.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>