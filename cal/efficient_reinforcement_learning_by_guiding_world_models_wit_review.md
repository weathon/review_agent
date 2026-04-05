=== CALIBRATION EXAMPLE 90 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title Accuracy:** The title accurately reflects the core contribution: using non-curated offline data to guide a world model for improved RL sample efficiency.
- **Abstract Clarity:** Clearly states the problem (leveraging reward-free, mixed-quality, multi-embodiment data), the method (experience rehearsal + execution guidance during fine-tuning), and the key quantitative result (~2x aggregate score across 72 tasks).
- **Claim Verification:** The claim of "nearly twice the aggregate score" is supported by the mean scores in Tables 3-5 (e.g., Meta-World mean success: 0.748 vs 0.430/0.360 for baselines; DMControl mean return: 617.7 vs 372.2/320.9). However, the abstract implies broad methodological novelty without noting that execution guidance closely resembles prior policy-mixing strategies (e.g., JSRL). While differences are discussed later, the abstract could more precisely frame the contribution as *adapting* retrieval and guidance mechanisms to the non-curated, world model fine-tuning setting rather than proposing entirely new paradigms.

### Introduction & Motivation
- **Problem Motivation & Gap:** Strongly motivated. The paper correctly identifies a practical bottleneck: curated, reward-labeled offline datasets are expensive, whereas abundant non-curated data is underutilized. The gap is clearly identified: naive fine-tuning of pre-trained world models fails due to distributional shift between pre-training and fine-tuning data distributions (Fig. 2).
- **Contributions Stated:** Contributions C1–C4 are explicit and accurately map to the paper's structure.
- **Over/Under-claiming:** The introduction slightly under-contextualizes the "distributional shift" problem relative to the broader offline RL literature. Methods like Cal-QL (Nakamoto et al., 2024) and CQL already tackle distribution mismatch in offline-to-online transitions. The paper would benefit from explicitly stating why these Q-value based corrections don't directly translate to stabilizing *world model* dynamics during fine-tuning, which would sharpen the motivation for the proposed experience rehearsal mechanism. Otherwise, the motivation is well-grounded and aligns with current ICLR trends in representation and dynamics pre-training.

### Method / Approach
- **Clarity & Reproducibility:** The pipeline is clearly described. Pre-training uses a standard RSSM objective (Eq. 1), adapted for multi-embodiment via action zero-padding. Fine-tuning integrates standard DreamerV3 actor-critic updates (Eq. 3, 4). Retrieval (Eq. 5) and execution guidance schedules are well-specified. Algorithm 1 provides a complete procedural view.
- **Assumptions & Justifications:** 
  - The method assumes offline data is *in-domain* (Sec 3.1 & Limitations D). This is a critical constraint that should be emphasized earlier. 
  - Retrieval relies on L2 distance in the visual feature space $e_\theta(o_{on}, o_{off})$. A key unstated assumption is that the pre-trained encoder's representation remains stable and discriminative enough during fine-tuning to retrieve relevant trajectories. As the encoder is continuously fine-tuned on online data, the retrieval metric could drift, potentially destabilizing training over long horizons. This dynamic is not discussed or analyzed.
- **Logical Gaps / Edge Cases:** 
  - Retrieval uses only the *initial* observation $o_{on}$ of a trajectory (Eq. 5). For long-horizon or multi-stage tasks (e.g., Meta-World's "Coffee Pull" or "Shelf Place"), the initial frame may be insufficiently discriminative, leading to noisy retrieval. The paper acknowledges occasional task overlap (Sec A.3) but does not analyze how retrieval degrades if initial states are visually similar across tasks.
  - Execution Guidance alternates between $\pi_{bc}$ and $\pi_{RL}$ mid-episode. This introduces non-stationarity in the rollout distribution. The theoretical justification (Prop. 2, Sec B.2) simply adapts Kakade & Langford (2002) and assumes $\pi_{bc}$ outperforms $\pi_{RL}$ early on. However, if the non-curated data is heavily skewed toward low-quality or failed trajectories for a target task, $\pi_{bc}$ may be detrimental. The method lacks a safeguard for poor-quality BC priors.

### Experiments & Results
- **Claims vs. Tests:** The experiments thoroughly address Q1–Q3. The 72-task evaluation across DMControl and Meta-World is commendably comprehensive and directly supports the claim of broad sample efficiency.
- **Baselines & Fair Comparison:** Baselines (R3M, UDS-RLPD, ExPLORe, JSRL-BC, DreamerV3, DrQ-v2, iVideoGPT) are appropriate and adapted where necessary. The aligned iVideoGPT comparison (Sec A.2) is a strength.
- **Missed Ablations:** 
  - *Retrieval vs. Random Replay:* Fig. 14 tests robustness to injected irrelevant data, but does not compare against a simpler baseline: uniformly sampling from the offline dataset during fine-tuning (given a large replay buffer). Without this, it's unclear whether the improvement stems from *intelligent retrieval* or simply from *continuing to replay any offline data* during fine-tuning. This ablation is crucial to validate Eq. 5's necessity.
  - *Seed Matching:* The authors report 5 seeds for NCRL but only 3 for DreamerV3/DrQ-v2 (Tables 3-5 caption). This asymmetry can artificially narrow/confound confidence intervals. For ICLR standards, results should be reported with matched seed counts to ensure statistical fairness.
- **Statistical Significance:** 95% CIs are plotted, but no hypothesis testing is provided. Given the tight margins on some tasks (e.g., Quadruped Run), formal significance testing would strengthen the claims.
- **Compute vs. Sample Efficiency:** The title claims "Efficient RL," but only sample efficiency is measured. Pre-training requires ~48 GPU hours (Sec F), plus fine-tuning. While sample efficiency is clearly improved, the wall-clock/compute trade-off relative to training-from-scratch is not discussed. ICLR reviewers typically expect compute parity analysis when claiming "efficiency."

### Writing & Clarity
- **Clarity of Explanations:** The paper is generally well-organized and readable. The rationale for why naive fine-tuning fails (Sec 3.3, Fig. 2) is intuitively explained and visually supported.
- **Figures/Tables:** Tables 3-5 are comprehensive. Fig. 2 effectively illustrates the distribution shift and the effect of rehearsal. However, Fig. 6/13 (ablation) curves lack task-specific breakdowns in the main text, forcing readers to cross-reference with Tables 3/5. The theoretical appendix (Sec B) contains standard derivations that don't yield new insights specific to the proposed method; it could be condensed to a lemma or moved to supplementary material to improve flow.
- **No Major Clarity Issues:** The core methodology and experimental setup are understandable without excessive back-tracking.

### Limitations & Broader Impact
- **Acknowledged Limitations:** The authors correctly note the RNN scaling bottleneck, in-domain data assumption, lack of real-world validation, and limited generalization to novel embodiments (Sec D).
- **Missed Limitations:** 
  - As noted in Method, the stability of the retrieval metric during continuous encoder fine-tuning is unaddressed.
  - The reliance on a linear annealing schedule for execution guidance (Sec A.4) lacks an adaptive mechanism. If a task requires rapid exploration or if the offline data lacks relevant skills, fixed scheduling may bottleneck performance.
  - Retrieval overhead at scale (Faiss indexing time) is not quantified. For datasets with millions of trajectories, retrieval latency could impact synchronous RL training loops.
- **Broader Impact:** Standard and acceptable. The paper focuses on algorithmic efficiency; ethical statements appropriately note autonomous systems risks without overclaiming.

### Overall Assessment
This is a strong, empirically well-grounded paper that addresses a practically important gap in offline-to-online RL: how to effectively reuse non-curated, multi-embodiment data during world model fine-tuning. The identification of distributional shift as the primary failure mode is insightful, and the proposed combination of retrieval-based experience rehearsal and execution guidance yields consistent, substantial sample efficiency gains across 72 diverse tasks. For ICLR, the extensive evaluation and clear problem framing are significant strengths. However, the contribution would be considerably more compelling if the authors addressed a few key methodological gaps: (1) include a retrieval-vs-random-replay ablation to isolate the necessity of Eq. 5, (2) report results with matched random seed counts for fair statistical comparison, and (3) discuss the compute/wall-clock trade-off relative to the claimed "efficiency." The theoretical section is perfunctory and adds limited value. With these clarifications and additional ablations, the paper represents a solid, reproducible contribution to sample-efficient RL and world model adaptation, well-suited for ICLR acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces NCRL, a framework that leverages non-curated, reward-free, and mixed-quality offline data to improve the sample efficiency of online reinforcement learning. By identifying that naive fine-tuning of pre-trained world models suffers from severe distributional shift, the authors propose experience rehearsal (retrieving task-relevant offline trajectories) and execution guidance (interleaving a behavior-cloned prior with the RL policy) to stabilize fine-tuning and steer exploration. Across 72 visuomotor tasks in Meta-World and DMControl, NCRL significantly outperforms both training-from-scratch baselines and existing offline-to-online methods under strict sample budgets.

### Strengths
1. **Well-Motivated & Practically Relevant Problem Formulation:** The paper tackles a highly relevant bottleneck in sample-efficient RL: leveraging large-scale, unlabeled, multi-embodiment datasets without costly reward annotation. The identification of distributional shift during world model fine-tuning (Sec. 3.3, Fig. 2) is a clear, empirically grounded insight that explains prior failures.
2. **Comprehensive & Statistically Rigorous Evaluation:** The experimental scope is substantial, covering 72 tasks with multiple random seeds and reporting 95% confidence intervals (Fig. 3, Tables 3-5). The baseline selection is strong, including model-free (DrQ-v2), model-based (DreamerV3, iVideoGPT), and offline-to-online methods (JSRL-BC, ExPLORe, UDS-RLPD), with careful implementation adjustments to ensure fair comparisons (Sec. 4.1, Sec. G).
3. **Clear & Effective Methodological Integration:** The proposed techniques are computationally lightweight and consistently improve performance. The ablation studies cleanly isolate contributions (Fig. 6, 13), and the retrieval mechanism (L2 distance on initial observations via FAISS) is practically scalable, as evidenced by the precision metrics (Table 2) and robustness to injected irrelevant trajectories (Fig. 14).
4. **High Reproducibility & Transparency:** The paper provides extensive implementation details, full hyperparameter tables (Sec. J), explicit compute budgets (Sec. F), and a complete algorithm (Sec. G/Alg. 1). The public release of code and datasets further strengthens reproducibility, meeting ICLR's transparency expectations.

### Weaknesses
1. **Moderate Algorithmic Novelty:** The core mechanisms (replaying offline data, interleaving a BC prior for exploration) build directly on established offline-to-online RL techniques (e.g., RLPD, JSRL, AWAC). The primary novelty lies in system-level adaptation to world model fine-tuning rather than introducing a fundamentally new optimization objective or architectural component.
2. **Oversimplified Retrieval Metric & Limited Theoretical Depth:** Trajectory retrieval relies solely on L2 distance between initial observation embeddings (Eq. 5), which may fail for tasks where early frames do not encode task-relevant dynamics or object interactions. Additionally, the theoretical analysis in Appendix B largely reproduces standard policy improvement bounds (Kakade & Langford, 2002) and does not derive new insights specific to latent world model error or distribution shift during imagination.
3. **Dataset Construction Limits Real-World Generalization Claims:** The offline dataset is synthetically constructed from simulation trajectories (unsupervised curiosity agents + noisy TDMPCv2 policies), not true in-the-wild or real-world teleoperated data. While this controls experimental variables, the paper's framing of "non-curated" data may overstate readiness for cross-embodiment or domain-out-of-distribution settings, a limitation only briefly acknowledged in Sec. D.
4. **Baseline Comparisons Lack Architectural Isolation:** Comparisons with large pre-trained models like iVideoGPT disable their reward shaping and demonstration pre-filling to create a fair control, but this obscures whether NCRL's gains stem from the fine-tuning pipeline or from architectural differences. A direct application of the rehearsal/guidance components to a standard DreamerV3 or TDMPC2 backbone is missing, making it harder to attribute improvements purely to the proposed methods.

### Novelty & Significance
**Novelty:** Moderate. The individual components are well-known in the literature, but their targeted integration to solve world model fine-tuning instability with non-curated data is a novel and practically valuable contribution. The framing shifts the focus from architecture scaling to fine-tuning stability, which is a timely perspective.
**Clarity:** High. The paper is well-structured, with clear problem definitions, intuitive diagrams (Fig. 1-2), and logical progression from failure analysis to solution and validation. Mathematical formulations and algorithmic steps are precise and easy to follow.
**Reproducibility:** Excellent. Detailed hyperparameters, compute costs, full result tables, pseudocode, and public code/data releases meet or exceed ICLR's reproducibility standards.
**Significance:** High. Sample efficiency remains a critical barrier to deploying RL in robotics. By demonstrating that large, unannotated, multi-task datasets can be effectively repurposed via simple fine-tuning interventions, the paper offers a scalable pathway toward data-efficient policy learning. The empirical gains across 72 tasks are convincing and will likely influence future work on pre-training/fine-tuning pipelines for model-based RL.

### Suggestions for Improvement
1. **Strengthen the Retrieval Mechanism & Provide Failure Analysis:** Evaluate alternative similarity metrics (e.g., trajectory-level dynamic time warping, attention-over-features, or goal-conditioned encoders) and report a brief qualitative analysis of cases where initial-observation retrieval fails (e.g., tasks with identical starting states but divergent objectives).
2. **Deepen the Theoretical Analysis:** Replace or augment Appendix B with a derivation specifically bounding the error in imagined rollouts or policy gradients caused by distributional shift in latent dynamics. This would directly connect the theoretical support to the world model fine-tuning setting rather than relying on generic RL bounds.
3. **Decouple Methodology from Architecture:** Apply the experience rehearsal and execution guidance components as plug-ins to different base models (e.g., standard DreamerV3, TDMPC2, or a smaller RSSM) to empirically verify that the fine-tuning strategy, not just the 280M parameter backbone, drives the observed gains.
4. **Clarify Dataset Scope & Cross-Domain Transfer Potential:** Explicitly distinguish between "in-domain non-curated" data (current setting) and "in-the-wild/cross-embodiment" data in the introduction and limitations. A brief pilot experiment or discussion on how retrieval/guidance would need to adapt for zero-shot transfer to unseen robots would better ground the method's scalability claims.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a DreamerV3/DrQ-v2 baseline that simply concatenates retrieved offline trajectories with online interactions; without isolating execution guidance, the performance gains cannot be definitively attributed to your proposed techniques rather than naive data mixing.
2. Include a uniform offline sampling ablation during experience rehearsal to prove neural feature retrieval is strictly necessary; if random sampling from the multi-embodiment pool yields similar gains, the core retrieval mechanism is unmotivated.
3. Test cross-embodiment retrieval explicitly by feeding offline data from *different* robot bodies into the fine-tuning loop; without this, the claim of leveraging "multi-embodiment" data for downstream tasks is unverified and may hide detrimental distributional interference.
4. Compare NCRL against compute-matched baselines that scale DrQ-v2/DreamerV3 to equivalent wall-time and FLOPs; claiming sample efficiency is invalid if the massive 280M parameter pre-training cost is omitted from the efficiency metric.

### Deeper Analysis Needed
1. Quantify latent rollout horizon error growth with and without experience rehearsal; without empirical error bounds, the assertion that rehearsal stabilizes world model fine-tuning against catastrophic forgetting remains a hypothesis rather than a demonstrated mechanism.
2. Analyze gradient interference between reward-free offline updates and sparse online reward model training; mixing these distributions likely misaligns the reward predictor with the latent dynamics, which must be characterized to trust the model-based updates.
3. Empirically verify the theoretical assumption that the BC execution policy $\pi_{bc}$ outperforms the early RL policy $\pi_{\phi}$ on mixed-quality data, and document failure regimes where $\pi_{bc}$ actively misguides exploration.

### Visualizations & Case Studies
1. Overlay state-space trajectory traces for NCRL versus training-from-scratch baselines; visual proof that execution guidance actually navigates agents out of local optima or dead-ends is required to validate the exploration claim.
2. Provide side-by-side visualizations of successful versus failed neural retrievals alongside their immediate impact on policy value updates; this exposes whether Eq. 5 reliably finds actionable priors or frequently injects misleading behaviors during fine-tuning.
3. Plot the latent manifold evolution (e.g., dynamic PCA) across fine-tuning steps with and without rehearsal; showing whether the method genuinely preserves a coherent latent space or merely masks model degradation through behavioral regularization is critical.

### Obvious Next Steps
1. Evaluate on a genuinely messy, real-world multimodal dataset (e.g., OpenX-Embodiment or DROID subsets) containing missing states, asynchronous controls, and non-uniform timestamps; simulated "mixed quality" via Gaussian noise does not validate robustness to true non-curated data.
2. Ablate the 280M parameter world model against smaller architectures (e.g., 30M, 100M) to establish performance scaling laws; the community cannot adopt this pipeline if the compute overhead is not proven to yield proportionally superior sample efficiency.

# Final Consolidated Review
## Summary
This paper proposes NCRL, a framework that improves the sample efficiency of online reinforcement learning by leveraging reward-free, mixed-quality, offline data. The authors identify distributional shift between offline pre-training and online fine-tuning data as the primary cause for the failure of naive world model adaptation. To resolve this, they introduce experience rehearsal (retrieving task-relevant offline trajectories to anchor fine-tuning) and execution guidance (interleaving a behavior-cloned prior with the RL policy). Evaluated across 72 pixel-based locomotion and manipulation tasks, NCRL demonstrates consistent, substantial gains in sample efficiency over both training-from-scratch baselines and existing offline-to-online methods.

## Strengths
- **Rigorous, broadly scoped empirical validation:** The evaluation spans 72 diverse visuomotor tasks across two standard benchmarks (Meta-World, DMControl) under a strict 150k sample budget. The consistent ~2x aggregate score improvement over strong baselines (DrQ-v2, DreamerV3, JSRL-BC, ExPLORe) with clear 95% confidence intervals provides compelling evidence of the method's practical utility.
- **Clear diagnostic insight driving methodological design:** The paper correctly identifies that naive fine-tuning of pre-trained world models fails not due to architectural limits, but because of a distributional mismatch between pre-training and early online data. The proposed rehearsal and guidance mechanisms directly target this diagnosed failure mode, offering a principled rather than heuristic solution.
- **High transparency and reproducibility:** The authors release code and datasets, provide exact compute budgets (~48 GPU hours for pre-training, ~8 per fine-tuning run), detailed hyperparameter tables, and a complete procedural algorithm. This level of transparency meets and exceeds standard ICLR expectations for empirical reproducibility.

## Weaknesses
- **Retrieval mechanism lacks a strict necessity ablation:** The core methodological claim rests on Eq. 5's neural feature retrieval (L2 distance on initial observation embeddings). While Table 2 reports high retrieval precision and Fig. A.14 demonstrates robustness to injected irrelevant trajectories, the paper does not compare against a baseline that uniformly samples offline trajectories into the replay buffer at the same mixing ratio. Given the dataset size (~60k trajectories, 10M transitions), uniform sampling may yield comparable regularization benefits without FAISS indexing overhead. Without this control, the marginal necessity of the retrieval component remains ambiguous.
- **Execution guidance relies on an unverified performance assumption:** Proposition 2 assumes that the behavior-cloned prior $\pi_{bc}$ outperforms the early exploration policy $\pi_{\phi}$ to guarantee performance improvement. However, the offline dataset is explicitly mixed-quality and reward-free. In regimes where retrieved trajectories are heavily biased toward failed attempts or suboptimal behaviors, $\pi_{bc}$ could actively misguide early exploration rather than accelerate it. The paper does not empirically verify that $\pi_{bc}$ consistently dominates early $\pi_{\phi}$ across the evaluated task distribution.

## Nice-to-Haves
- Provide a compute/wall-clock parity analysis. While sample efficiency is the explicit focus, quantifying the FLOP and time overhead of pre-training a 280M RSSM and running retrieval against extended-training baselines would help practitioners assess real-world deployment trade-offs.
- Extend the theoretical analysis beyond standard Kakade & Langford policy improvement bounds. Deriving bounds specific to latent dynamics prediction error or distributional drift during imagined rollouts would more directly ground the theory in the world model fine-tuning context.
- Conduct a brief empirical check on latent rollout horizon error growth with vs. without rehearsal to explicitly verify the claim that experience rehearsal mitigates catastrophic forgetting in the dynamics model.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add a direct ablation comparing neural feature retrieval during experience rehearsal against uniform random offline sampling (matched by replay ratio). If retrieval yields significantly higher sample efficiency or stability, report the magnitude of the gap.
- Empirically track and report the early-episode return of $\pi_{bc}$ vs. $\pi_{\phi}$ across a representative task subset to validate the assumption underpinning execution guidance. If $\pi_{bc}$ occasionally underperforms, discuss whether the linear annealing schedule adequately compensates.
- In the introduction or limitations, explicitly clarify that "non-curated" in this setting refers to in-domain, simulation-generated trajectories with controlled noise, distinguishing this from cross-embodiment or messy real-world teleoperation data to accurately scope future work.

# Actual Human Scores
Individual reviewer scores: [6.0, 10.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
