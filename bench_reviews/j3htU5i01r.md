## Summary

The paper proposes a compositional meta-learning framework where tasks are represented as structured combinations of reusable modules. A gating RNN learns the "grammar" of module transitions while module RNNs learn the "syllables" (within-module dynamics). Training maximizes marginal likelihood via particle filtering; test-time adaptation requires no parameter updates, instead inferring module sequences through constrained probabilistic hypothesis testing. Experiments on synthetic rule-learning and motor-learning tasks demonstrate ground-truth component recovery, one-shot task inference, and robustness to sparse feedback.

## Strengths

- **Inference-based adaptation without parameter updates is a genuine conceptual departure from gradient-based meta-learning.** The paper shows (Figure 3e) that MAML, MLDG, and standard fine-tuning require hundreds of episodes, while the proposed method infers the solution from a single episode. The ablation in Figure 3 systematically establishes that both the gating network and modular structure are necessary, particularly under sparse feedback where unconstrained inference fails (Figure 3c vs. 3d).

- **Ground-truth recovery provides interpretable evidence that the model learns what it claims.** Figure 2a-c shows that both learned modules and learned transition statistics converge to the ground truth, going beyond typical meta-learning papers that only report task performance. This verifiability is a meaningful advantage of the controlled synthetic setting.

- **Sparse feedback handling is well-demonstrated and non-trivial.** Figures 2e, 2f, and 4e show that the model maintains hypothesis branching during feedback gaps and collapses appropriately when observations return. This leverages the learned grammar in a way that standard meta-learning methods fundamentally cannot, and the extended-sequence result (4× training length) shows generalization of the learned statistics rather than memorization of specific sequence patterns.

## Weaknesses

### Major:

- **Computational cost of inference is unanalyzed, making the efficiency comparison against gradient-based methods incomplete.** The paper claims "rapid acquisition" and compares against MAML and others on an episodes-based axis (Figure 3e/f), but each inference episode requires running K=250 particles through the module RNNs for T timesteps. A single inference episode may cost orders of magnitude more in FLOPs than a gradient step on a monolithic network. The sample-efficiency advantage is clear, but whether it translates to a wall-clock or compute advantage is unaddressed and potentially significant. This is not a fatal flaw—250× one episode may still be less than 1× 100 episodes—but the paper should quantify this rather than leave it implicit.

- **Evaluation is limited to low-dimensional synthetic tasks with known compositional structure.** The rule-learning task (6D vector shifts) and motor task (2D trajectories) are deliberately simple to enable ground-truth verification, which is valuable. However, the paper makes broad claims about compositional meta-learning that currently lack demonstration on more complex or higher-dimensional domains. Tasks with ambiguous or hierarchical compositional structure, noisy observations, or larger state spaces would better stress-test whether the approach scales beyond settings where the ground truth is cleanly factorizable. The authors acknowledge this as proof-of-principle, but the gap between the claimed framework and the demonstrated scope is notable.

### Minor:

- **Fixed, pre-specified number of modules limits practical applicability.** The number N of modules is set at initialization. While Figure A1a-d explores mismatches between module count and ground-truth operation count, there is no mechanism for dynamic module addition during training or inference. The Discussion identifies this as future work for continual learning, but it remains a constraint on the current framework.

- **Train-test distribution shift between soft (Gumbel-softmax) and hard (argmax) module selection is not analyzed.** Training uses soft relaxation for gradient flow while inference uses hard argmax. This is standard practice, but no analysis of gating entropy during training is provided to verify that the learned distributions are sufficiently peaked to support hard selection.

- **Non-Markovian capacity of the gating network is claimed but not quantitatively evaluated.** The paper states the gating RNN learns "strongly non-Markovian statistics" (Section 2.2), with Figure 2c showing history-dependent transition matrices visually. However, no experiment isolates the benefit of non-Markovian gating over, e.g., a higher-order HMM baseline, leaving the architectural choice partially unmotivated beyond visual evidence.

### Trivial:

- **Sparse feedback handling is described narratively in Section 2.3 but not given explicit mathematical formulation.** The mechanism (skipping likelihood-based resampling during missing-observation timesteps) is standard in particle filtering but could be stated as a formal equation for completeness.

## Nice-to-Haves

- **Ablation on particle count K versus inference accuracy and feedback gap length** would clarify how many particles are actually needed and whether the 250-particle budget is overkill for these tasks or necessary for degeneracy avoidance.

- **Wall-clock time or FLOP-normalized comparison against gradient-based baselines** would substantiate or qualify the efficiency claims and is straightforward to report.

- **Empirical comparison to Hummos et al. (2024)**, which the Discussion identifies as the closest related approach (compositional inference via gradient-based latent embedding optimization), would sharpen the contribution. Even a conceptual runtime analysis contrasting sequential module search vs. embedding optimization would help.

- **Evaluation on at least one established multi-task or meta-learning benchmark** (e.g., a structured prediction task with compositional generalization splits) would significantly broaden the impact.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"The abstract should hint at computational trade-offs"** — This is a formatting/style suggestion about what the abstract should contain; the abstract accurately describes the method, and the concern about computational cost is already captured in the Major Weaknesses above.

- **"The introduction should clarify why explicit modularity is necessary"** — The paper motivates this through the analogy to dynamical motifs (Yang et al., 2019; Driscoll et al., 2024) and by showing that a monolithic RNN without task identity fails (Figure 3a). The ablation in Figure 3c,d further supports the architectural choices.

- **"Missing comparison to Alet et al. (2019) and other modular meta-learning methods"** — Demanding comparisons to every related method is scope creep. The paper discusses Alet et al. conceptually and notes it uses simulated annealing rather than probabilistic inference, which is the key distinction.

- **"Quantitative module disentanglement metrics"** — The paper already provides quantitative ground-truth recovery metrics (module and gating accuracy in Figure 2a) alongside visual verification. Additional disentanglement metrics would add limited value for these controlled tasks where ground truth is directly measurable.

- **"Reproducibility concerns about hyperparameters or training instability"** — The paper provides full implementation details in Appendix A.1–A.2, including initialization schemes, learning rates, and batch sizes. The "chicken-and-egg" instability is acknowledged and mitigated through careful initialization (Section A.1).

- **"OoD detection threshold is undefined"** — Figure A1e demonstrates qualitative separation of likelihood between in-distribution and OoD tasks. Quantifying threshold policies is more relevant to a continual-learning extension than to the current proof-of-principle.

- **"Section 2.4 architectural changes for motor learning make results not directly comparable"** — The paper clearly states these modifications and motivates them (autonomous trajectories vs. input-driven outputs). The core framework remains the same.

## Novel Insights

The key insight emerging from this work is the clean separation between learning compositional *structure* (via gradient descent on parameters during training) and deploying that structure (via probabilistic inference at test time). This decoupling means that test-time adaptation requires no weight updates—only hypothesis testing over learned module sequences—making the approach inherently immune to catastrophic forgetting. The sparse-feedback results reveal an underappreciated advantage of this inference-based approach: the learned transition grammar acts as a strong prior that prunes the hypothesis space during observation gaps, something gradient-based adaptation cannot exploit because it lacks an explicit generative model of task structure.

## Suggestions

- Report wall-clock time or FLOP counts for inference vs. gradient-based baselines to clarify whether sample efficiency translates to compute efficiency, which is the metric practitioners care about most.

- Run a parameter sweep on K (e.g., K ∈ {10, 50, 100, 250, 500}) and plot inference accuracy vs. K, particularly under sparse feedback conditions. This would establish whether the method's success depends on generous particle budgets or is robust to limited compute.

- Test the framework on at least one task with higher-dimensional observations (e.g., 2D pixel inputs with compositional structure) to probe scalability beyond the current proof-of-principle setting.

## Evaluation Axis Summary

- **Novelty**: Moderate-to-high. The combination of modular architecture with learned gating grammar and particle-filter-based inference for meta-learning is distinctive, though individual components (HMMs, particle filtering, modular networks) are well-established.
- **Technical soundness**: Sound in its current scope, with clearly defined generative model and appropriate inference procedure. The main gap is the unanalyzed computational cost trade-off.
- **Empirical support**: Convincing for the synthetic, controlled setting. Ground-truth recovery, ablation studies, and sparse-feedback results are thorough within that scope. The gap is in demonstrating scalability and broader applicability.
- **Significance**: Potentially significant if the framework scales to more complex domains. The inference-based paradigm for meta-learning is conceptually important, but the current proof-of-principle limits immediate practical impact.
- **Clarity**: Well-organized and clearly written, with consistent notation and useful appendices. The paper's scope and limitations are honestly stated.