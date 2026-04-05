# Direct-Scoring Baseline Results

Model: qwen/qwen3.6-plus:free

## Me0n0iESJY

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (4.7/10)
- Match: N/A

### Review

## Summary
This paper introduces a comprehensive benchmark for merging Multimodal LLMs (MLLMs) across five distinct capability domains and three modalities, alongside a new merging method, OptMerge. OptMerge denoises task vectors via low-rank SVD approximation, optimizes them using SGD with mean initialization, and demonstrates consistent gains over ten baselines. The authors also provide a theoretical bound linking fine-tuning steps and learning rates to merging interference.

## Strengths
- **Valuable Benchmark & Open Release:** The paper fills a clear gap in the MLLM merging literature by providing a fine-grained capability benchmark (VQA, Geometry, Chart, OCR, Grounding) with publicly released expert checkpoints, training data, and evaluation scripts, facilitating reproducible future research.
- **Comprehensive Empirical Validation:** Experiments span multiple architectures (InternVL2.5, Qwen2-VL), scales (1B to 32B), and fine-tuning paradigms (full fine-tuning vs. LoRA). Tables 2, 3, and 9 show consistent average improvements, and the ablation study clearly isolates the contribution of each component in OptMerge.
- **Practical Insights on Fine-Tuning Dynamics:** The empirical observation that excessive fine-tuning harms merging performance, coupled with the analysis of how LoRA's low-rank structure changes task vector distributions compared to full fine-tuning, offers actionable guidance for practitioners selecting or training models intended for merging.

## Weaknesses
- **Unfair Mixture Training Baseline:** The paper claims merging "potentially surpasses mixture training," but for Qwen2-VL, it uses the official `Qwen2-VL-Instruct` model as the upper bound. This model was trained on orders of magnitude more diverse data, not the simple union of the 5 benchmark datasets. A fair comparison requires training a true mixture baseline on the exact same combined data, making this central claim empirically unsupported.
- **Incremental Methodological Novelty:** OptMerge combines established techniques: truncated SVD for denoising, SGD over Adam for null-space stability, and mean initialization. While effective, these choices are largely heuristic and lack rigorous theoretical justification for why this specific combination is optimal for task vector optimization in MLLMs.
- **Standard Theoretical Contribution:** Theorem 3.1 derives a merging error bound using standard Polyak-Łojasiewicz and L-smoothness assumptions. The resulting terms ($O(\gamma^T)$, $O(\delta \eta T)$, $O(\eta^2 T^2)$) are well-known in optimization literature and do not directly inform or justify the low-rank truncation heuristic proposed in the method.

## Nice-to-Haves
- A deeper empirical or theoretical analysis of the rank truncation ratio (set to `1/num_tasks`). Plotting the singular value decay across layers for different tasks could explain why ~20% retention works best and whether layer-adaptive rank selection would yield further gains.

## Novel Insights
The paper provides a nuanced characterization of the "task vector landscape" for MLLMs that moves beyond treating all fine-tunes as interchangeable. By empirically distinguishing the multi-modal, clustered magnitude distribution of LoRA task vectors from the right-skewed distribution of fully fine-tuned vectors, the authors reveal that optimization dynamics must be tailored to the fine-tuning paradigm. Specifically, the identification of Adam's instability in LoRA's sparse gradient/null-space directions, and the demonstration that mean initialization prevents magnitude explosion during data-free optimization, highlights a critical but often overlooked practical reality: successful merging is as much about controlling optimization trajectory and parameter magnitude as it is about resolving directional interference.

## Potentially Missed Related Work
- *Model Merging by Uncertainty-based Gradient Matching* (Daheim et al., ICLR 2024) — highly relevant for discussing how task uncertainty and gradient alignment interact during merging, which complements the paper's interference analysis.
- *Activation-based merging / Online Composing methods* (e.g., Chen et al., 2024a DAMC is cited, but recent work on dynamic routing or activation stitching like Twin-Merging or MoE-style post-hoc merging could be better contrasted against static weight merging for modalities).

## Suggestions
- **Fix the Mixture Training Baseline:** Train a true multi-task model on the exact concatenation of your 5 benchmark datasets (with standard data mixing/sampling ratios) and report those results. This will either strengthen your claim that weight merging can rival joint training or clarify the trade-offs.
- **Justify Rank Selection Heuristic:** Add an analysis of the singular value spectrum of the task vectors across layers. Showing that the top ~20% captures >90% of the task-specific energy across layers would ground your `k` ratio choice in empirical spectral properties rather than an arbitrary heuristic.
- **Clarify Theorem-Method Connection:** Either tighten the theoretical analysis to explicitly bound the error introduced by SVD truncation, or clearly state in the text that Theorem 3.1 motivates controlling fine-tuning drift ($\eta T$), while OptMerge addresses the resulting vector noise through a practical, empirically validated denoising pipeline.

## Overall Score
4.7

---

## USyGD0eUod

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (7.6/10)
- Match: N/A

### Review

## Summary
This paper conducts a systematic empirical investigation into whether standard sparse autoencoder (SAE) evaluation metrics, particularly automated interpretability scores, can distinguish between trained and randomly initialized transformers. The authors find that aggregate auto-interpretability metrics (e.g., fuzzing AUROC) are often surprisingly similar across trained and randomized models, demonstrating that high scores alone do not guarantee recovery of learned, computationally meaningful features. Through extensive scaling experiments, robustness checks, and a toy model analysis of superposition preservation, the paper advocates for mandatory randomized baselines and proposes token distribution entropy as a supplementary diagnostic for feature complexity.

## Strengths
- **Rigorous and comprehensive experimental design:** The study spans five Pythia model scales (70M–6.9B), multiple randomization schemes (Step-0, re-randomized with/without embeddings, and a negative control), and includes thorough ablations over training data size, SAE hyperparameters, and random seeds. This breadth ensures the findings are highly robust and generalizable across the Pythia family.
- **Clear, actionable critique with a practical diagnostic:** The paper effectively demonstrates that current auto-interpretability pipelines are vulnerable to superficial statistical patterns. The introduction of token distribution entropy as a lightweight, quantitative proxy for feature specificity provides practitioners with an immediately usable tool to complement aggregate metrics, as evidenced by the clear separation of control, random, and trained variants in Appendix H.

## Weaknesses
- **Limited mechanistic dissection of the auto-interpretability pipeline's failure mode:** The paper shows *that* random models score well but does not deeply analyze *why* LLM-based auto-interp is so effective at recognizing these superficial patterns (e.g., is it sensitivity to token morphology, syntactic regularities, or prompt-induced pattern matching?). This matters because understanding the failure mechanism is necessary to design下一代 auto-interpretability metrics that are genuinely grounded in computational relevance rather than surface-level description.
- **Token distribution entropy is an imperfect proxy for "abstractness":** The proposed metric measures activation diversity across token IDs, but high entropy could simply indicate noisy, polysemantic, or poorly converged features rather than semantically abstract representations. This matters because relying on it as a validation signal could inadvertently reward features that are broad and non-specific, potentially misleading researchers who lack qualitative review capacity.

## Nice-to-Haves
- Include causal steering or activation patching experiments to quantitatively compare the behavioral impact of SAE features from trained vs. random models, which would provide direct evidence of their computational utility beyond descriptive metrics.

## Novel Insights
The paper's most compelling contribution is empirically grounding the hypothesis that apparent "interpretability" in SAEs can emerge from the confluence of data structure, architectural inductive biases, and sparse coding, independent of learned weights. By demonstrating that random linear transformations (plus nonlinearities) preserve or even amplify the superposition inherent in natural language, the authors reveal that auto-interpretability scores often reflect the SAE's ability to efficiently sparsify input statistics rather than its success at reverse-engineering trained computation. This reframes high auto-interpretability scores not as proof of discovered mechanisms, but as a measure of how well a decomposition aligns with the statistical regularities of the data distribution, fundamentally challenging how the field validates feature discovery.

## Potentially Missed Related Work
- None identified (the literature review is thorough and well-aligned with recent SAE benchmarks, auto-interpretability pipelines, and studies on random transformers).

## Suggestions
- When introducing token distribution entropy, explicitly correlate it with human-judged abstraction levels or intervention-based feature importance metrics to validate its use as a proxy for computational relevance.
- Add a controlled ablation on the auto-interp LLM's prompting strategy and temperature to isolate whether high scores stem from genuine pattern recognition or LLM over-confidence/prompt compliance, which would strengthen the methodological critique.
- Provide a concrete, recommended checklist for future SAE papers (e.g., mandatory Step-0 baseline reporting, multi-metric dashboards including entropy, and qualitative sampling guidelines) to maximize the practical impact of these findings on community standards.

7.6

---

## KsWRLyIAKP

- GT: Withdrawn (treated as Reject) (avg 3.2)
- Predicted: N/A (4.3/10)
- Match: N/A

### Review

## Summary
This paper frames financial lead-lag detection as a temporal link prediction task on dynamic graphs and systematically evaluates eight temporal GNN architectures against a sequential LSTM baseline. Using a custom dataset of 37 financial assets, the authors demonstrate that graph-based modeling, particularly the lightweight GraphMixer architecture, significantly outperforms sequence-only approaches across multiple ranking metrics and statistical tests.

## Strengths
- **Clear task formalization and rigorous empirical protocol:** The mapping of lead-lag relationships to directed temporal edges is well-motivated, and the experimental setup includes multiple seeds, statistical significance testing (Friedman + Conover post-hoc), and dual-scenario evaluation (bullish-only vs. dual-direction), which strengthens result credibility.
- **Strong empirical validation of architectural simplicity:** The consistent outperformance of GraphMixer over memory- and attention-heavy TGNNs (TGN, APAN, TGAT) provides valuable, data-driven evidence that financial temporal dependencies may be effectively captured by lightweight token-mixing strategies rather than complex recurrent or attention mechanisms.
- **Practical financial framing:** Explicitly separating positive-only and dual-direction scenarios aligns the evaluation with realistic trading objectives (e.g., long-only momentum strategies), and the demonstration that graph structure captures cross-asset dependencies missed by LSTMs highlights the value of relational inductive biases.

## Weaknesses
- **Limited methodological novelty:** The core contribution is primarily empirical/benchmark-oriented. No new architecture, training objective, or theoretical insight is introduced, making the paper incremental for a top-tier ML venue like ICLR where algorithmic or conceptual advances are typically expected.
- **Unrealistic threshold and dataset scale:** A fixed $\epsilon=5\%$ daily return threshold is unusually high for daily financial data, artificially sparsifying the graph and likely biasing it toward high-volatility tech/EV stocks. Combined with only 37 nodes, this severely limits generalization and raises overfitting concerns, especially given the observed performance degradation when adding financial indicators.
- **Insufficient baselines and financial realism:** Dismissing traditional statistical/ML methods as "outside the scope" weakens the comparative analysis. Without comparisons to Granger causality, VAR, or strong pairwise/tabular models (e.g., LightGBM), it is unclear if TGNNs provide genuine advantages over simpler, highly interpretable alternatives. Additionally, the edge construction inherently uses future returns ($t+\tau$) to label edges at time $t$, and the paper neglects look-ahead bias, transaction costs, or slippage, which are critical for assessing real-world trading viability.
- **Clarity and reproducibility gaps:** Several typos and formatting artifacts (e.g., `num ~~n~~ eighbors`, `structure ~~t~~ ime ~~g~~ ap`) remain from drafting. The ablation explanation for why added features degrade performance is hand-wavy, and the exact temporal alignment of node vs. edge features during training/inference is insufficiently detailed.

## Nice-to-Haves
- A sensitivity analysis on the $\epsilon$ threshold (e.g., volatility-adjusted or percentile-based thresholds) would clarify whether the results are robust or heavily dependent on the arbitrary 5% cutoff.
- Visualizing which asset sectors or specific pairs drive the highest prediction scores would greatly enhance financial interpretability.

## Novel Insights
The paper's most compelling finding is the robust dominance of GraphMixer—a purely MLP-based temporal mixer—over sophisticated memory/attention TGNNs on financial dynamic graphs. This suggests that short-horizon lead-lag dynamics in markets are driven more by localized structural co-movements than by long-term temporal dependencies or global attention, challenging the recent trend of increasingly complex TGNN designs. Furthermore, the ablation reveals that static semantic embeddings (LLM-derived descriptions) outperform explicit price/technical features when combined with dynamic edges, implying that sector/identity priors provide stronger relational signals than raw time-series indicators once the graph topology encodes co-movement. This points toward a paradigm shift in quantitative modeling: inductive graph structure + simple feature mixing may be more effective than heavy feature engineering or complex architectures for certain financial prediction tasks.

## Potentially Missed Related Work
- *Recent financial dynamic GNN benchmarks (e.g., FinGNN, StockFormer, or Sawhney et al.'s dynamic financial graph works)* — relevant for comparing feature integration strategies, handling of market microstructure noise, and standard evaluation protocols in financial graph learning.
- *Pairwise ML baselines for lead-lag (e.g., cross-correlation forests, gradient-boosted pairwise regressors)* — relevant to establish whether the graph relational bias provides advantages over simpler, non-graph ML approaches commonly used in systematic trading literature.

## Suggestions
- Expand the baseline suite to include classical econometric models (Granger causality, VAR) and strong pairwise/tabular ML models to rigorously quantify the added value of the TGNN framework.
- Replace or supplement the fixed 5% threshold with a volatility-normalized or rolling-percentile threshold, and report how graph density/sparsity changes affect model performance.
- Provide a clear, step-by-step diagram of the temporal split and feature alignment to rule out look-ahead bias, and include a simple backtesting simulation (even heuristic) accounting for transaction costs to ground the findings in practical trading contexts.
- Thoroughly proofread to remove LaTeX compilation artifacts and clarify the GM-TNF definition, particularly regarding when node vs. edge temporal features are applied.

## Score: 4.3

---

## M14YpuTejd

- GT: Withdrawn (treated as Reject) (avg 4.0)
- Predicted: N/A (6.7/10)
- Match: N/A

### Review

## Summary
This paper identifies critical flaws in the two-stage training and evaluation protocol for online map-based motion prediction, specifically highlighting dataset split-induced train-validation gaps, perception range mismatches, and non-discriminative metrics. It introduces OMMP-Bench, featuring a spatially disjoint data partition, refined evaluation metrics focusing on moving non-ego agents (stratified by distance), and a baseline that supplements missing long-range map context with raw image features via deformable attention.

## Strengths
- **Rigorous protocol analysis with empirical validation:** The authors clearly demonstrate how the standard nuScenes split creates a severe train-val gap in two-stage pipelines, where motion prediction models train on highly accurate (cheating) maps but evaluate on degraded predictions. Table 1 provides concrete evidence that their spatially disjoint partition yields significant, consistent metric improvements.
- **Alignment of evaluation with real-world safety requirements:** By shifting focus from the ego vehicle to moving non-ego agents, excluding static trajectories, and stratifying performance by proximity to the map perception boundary, the proposed metrics directly address the core objective of collision avoidance. Table 6 empirically confirms that ego-centric metrics mask severe degradation in long-range agent prediction.
- **Comprehensive analysis of map element utility:** The systematic ablation of different HD map components (Table 5) reveals that centerlines provide the strongest predictive signal, offering actionable guidance for the online mapping community on feature prioritization and output design.

## Weaknesses
- **Methodological simplicity of the proposed baseline:** The "boundary-free" solution relies on a standard deformable attention module to pull image features for out-of-range agents. While effective, it matters because it functions as an architectural patch rather than a principled solution to long-range spatial reasoning or map uncertainty propagation, limiting its theoretical contribution to the field.
- **Dataset restriction and lack of a true test split:** Relying exclusively on nuScenes and proposing only a train/val partition without a held-out test set or cross-dataset validation limits the benchmark's generalizability. This matters because autonomous driving evaluation increasingly demands robustness across diverse geographic environments, sensor configurations, and domain shifts, which nuScenes alone cannot fully represent.

## Nice-to-Haves
- Including performance variance across multiple training seeds, or reporting confidence intervals for the minADE/minFDE metrics, would strengthen the statistical reliability of the claimed improvements, especially for the close/far agent stratification.

## Novel Insights
The paper exposes a fundamental disconnect in modular autonomous driving pipelines: downstream motion prediction demands long-range, high-fidelity environmental context for all agents, while upstream online mapping models are practically constrained to short-range, noisy predictions. By explicitly isolating this mismatch through spatial data partitioning and distance-stratified evaluation, the work demonstrates that many apparent gains in recent online map-based motion prediction literature are artifacts of data leakage and ego-centric evaluation bias rather than genuine algorithmic progress. This reframes the field's priority from chasing incremental model tweaks to enforcing rigorous pipeline alignment, honest evaluation design, and explicit handling of perception uncertainty.

## Potentially Missed Related Work
- **Uncertainty-aware modular driving pipelines (e.g., works on probabilistic BEV representations or Monte Carlo feature distillation)** — Highly relevant, as propagating map uncertainty into the motion prediction loss offers a more principled alternative to simply substituting missing maps with raw image features, potentially improving safety guarantees for faraway agents.

## Suggestions
- Implement a standardized uncertainty quantification protocol for the upstream map/image features and integrate this uncertainty into the motion prediction training objective (e.g., via heteroscedastic loss or ensemble weighting). Additionally, release a held-out test split with diverse scene conditions to track true generalization and prevent overfitting to the proposed motion val set.

## Overall Score: 6.7

---

## bm3rbtEMFj

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (7.4/10)
- Match: N/A

### Review

## Summary
ELMUR introduces a transformer architecture augmented with per-layer external memory, managed via a simple Least Recently Used (LRU) replacement and convex blending rule, to address long-horizon decision-making under partial observability. The method is rigorously evaluated across synthetic benchmarks, POPGym, and robotic manipulation tasks, demonstrating robust retention far beyond native attention windows and consistent improvements over strong baselines.

## Strengths
- Exceptional empirical validation of extreme-horizon retention and generalization, notably achieving 100% success on T-Maze corridors up to 1 million steps (100,000× the context window) and consistently outperforming baselines across 48 POPGym tasks. Evidence: Figures 3, 4, and Tables 1, 2, 5 show clear, statistically significant gains over DT, RATE, and BC-LSTM.
- Rigorous mechanistic analysis conclusively proves that performance gains stem from functional memory usage rather than increased parameter capacity. Evidence: Section A.9's probing experiments, PCA projections, and cross-attention maps (Figures 9–11, 15) demonstrate targeted one-shot writes, stable slot retention without drift, and head specialization.

## Weaknesses
- The theoretical analysis is limited to elementary properties of convex combinations and exponential decay, lacking formal connections to RL/POMDP theory (e.g., error bounds on value functions, sample complexity under partial observability). — This matters because it leaves the conditions under which ELMUR improves policy optimization theoretically ungrounded, making it harder to generalize the findings beyond the tested imitation learning settings.
- The evaluation is strictly confined to Imitation Learning and offline datasets, despite the title explicitly claiming applicability to "Long-Horizon RL Problems." — This matters because real-world long-horizon RL fundamentally requires active exploration and online credit assignment, where memory mechanisms face compounding errors and distributional shift not addressed here.
- Many MIKASA-Robo tasks yield 0% success across all methods, and the claimed "~70% aggregate improvement" is mathematically driven by relative changes on near-zero baselines. — This matters because it inflates the perceived practical impact on complex visual manipulation tasks and obscures the actual ceiling of current offline IL under sparse rewards.

## Nice-to-Haves
- A detailed wall-clock/FLOPs and VRAM comparison against baselines to quantify the overhead of per-layer cross-attention and MoE routing, alongside a variant with an adaptive/context-dependent LRU blending factor instead of a fixed hyperparameter.

## Novel Insights
The paper’s most compelling contribution lies in its empirical demonstration that a deliberately simplistic, non-differentiable memory management policy (LRU with fixed blending) can naturally induce hierarchical, slot-specific memory allocation without explicit meta-learning or complex addressing gradients. The mechanistic probing reveals that this simple rule forces shallow layers to rapidly encode salient cues into dedicated slots, which deeper layers then refine into structured, task-specific manifolds that remain stable over million-step horizons. This suggests that in long-horizon imitation learning, enforcing strict capacity limits and sparse, temporally localized writes is more effective for preventing catastrophic interference than fully learned, continuous memory updates, offering a practical and highly interpretable blueprint for scalable memory-augmented policies.

## Potentially Missed Related Work
- xLSTM (Beck et al., NeurIPS 2024) — Highly relevant as a modern sequential architecture featuring structured external memory matrices with exponential gating; comparing against it would contextualize ELMUR’s fixed LRU rule within state-of-the-art learned memory mechanisms.
- Amago-2 (Grigsby et al., NeurIPS 2024) — Focuses on scalable, multi-task long-horizon RL using transformers; relevant as it directly tackles online long-horizon RL where ELMUR currently lacks evaluation.
- Titans / Learning to Memorize at Test Time (Behrouz et al., 2024) — Explores dynamic memory creation/adaptation during inference; relevant for contrasting static LRU allocation vs. adaptive, context-driven memory expansion.

## Suggestions
- Expand the experimental scope to include at least one online RL benchmark with active exploration (e.g., MiniGrid, Crafter, or a simplified continuous control environment) to validate whether the memory mechanism facilitates online credit assignment and exploration, justifying the "RL Problems" claim in the title.
- Replace or augment the fixed LRU blend factor $\lambda$ with a lightweight, learnable gating mechanism conditioned on temporal distance or prediction uncertainty to improve adaptability across diverse task horizons without sacrificing the simplicity that enables theoretical analysis.
- Clarify computational trade-offs by reporting exact training/inference FLOPs, parameter counts per component, and memory footprint relative to standard DT and RATE baselines, ensuring the architectural additions are transparently evaluated for scalability.

## Overall Score
7.4

---

## hQZQVLJrH9

- GT: Withdrawn (treated as Reject) (avg 4.5)
- Predicted: N/A (4.6/10)
- Match: N/A

### Review

## Summary
This paper establishes a first-order theoretical equivalence between activation steering and training-data influence functions, framing both as projections of a shared sensitivity tensor onto different operational subspaces. It introduces the Influence-Aligned Steering (IAS) vector, a geometric alignment diagnostic $\omega$ to predict steering feasibility, and derives spectral optimality conditions alongside low-rank generalization bounds. Empirical validation is provided on GPT-2 Medium and ResNet-50, showing strong first-order logit correlation and depth-dependent alignment trends.

## Strengths
- **Unified Geometric Framework:** The primal-dual convex optimization formulation cleanly maps steering vectors to data influence measures, providing a rigorous mathematical bridge between two historically separate interpretability toolkits. The derivation of IAS via Jacobian pseudoinverses and the Fisher-metric dual certificate $\|\epsilon\|$ offers genuine theoretical clarity.
- **Actionable Feasibility Diagnostic:** The alignment metric $\omega$ (cosine of the smallest principal angle between activation and parameter Jacobian subspaces) directly addresses a key practitioner pain point. Theorem 5.1 and the associated pre-check rule give a cheap, theoretically grounded way to decide steering vs. weight-level editing before running expensive experiments.

## Weaknesses
- **Computational Tractability Claim is Overstated:** The paper asserts that all quantities reduce to "two backward passes," yet exact computation of $\mathbf{J}_{h \to y}^\dagger \mathbf{J}_{\omega \to y}$ requires handling an $m \times d$ matrix where $m$ is vocabulary size (typically 30k–150k for LLMs). Explicit pseudoinversion or SVD at this scale is computationally impractical without iterative solvers (e.g., LSQR, Nystrom) or low-rank approximations, which are neither detailed nor benchmarked. This gap severely limits the claimed "billion-parameter" scalability.
- **First-Order Regime vs. Practical Steering Magnitudes:** The equivalence strictly holds for infinitesimal perturbations, yet real-world steering (detoxification, knowledge editing) requires finite, often large-magnitude vectors where higher-order saturation dominates. Figure 1 reports a slope of 1.50 between predicted and actual shifts, indicating systematic underestimation even in the claimed "small-edit" regime. Without second-order corrections or trust-region analysis, the theory's applicability to practical, non-infinitesimal interventions remains narrow.

## Nice-to-Haves
- Incorporating a tractable second-order correction or adaptive step-size scheduler would significantly extend the framework's utility beyond the infinitesimal regime, bridging the gap between the proven linear theory and the non-linear behavior observed in practice.

## Novel Insights
The paper’s most valuable conceptual shift is reframing activation steering not as an empirical search over heuristic vectors, but as a constrained projection of the data-influence manifold onto the activation-reachable subspace. By quantifying the "geometric effort" required to cover unreachable logit directions via the dual certificate $\|\epsilon\|$, the work provides a diagnostic lens to understand why steering succeeds for certain behaviors (e.g., toxicity in late layers where $\omega \to 0.94$) and fundamentally fails for others. This moves the field from trial-and-error prompt/vector tuning toward a geometry-driven engineering workflow where intervention feasibility is mathematically pre-certified before any compute is spent.

## Potentially Missed Related Work
- **Burns et al. (2023), "Discovering Latent Knowledge in Language Models Without Supervision" & Marks et al. (2024), "The Geometry of Representation Steering in Language Models"** — These works already explore the linear structure of conceptual representations, steering vector construction via linear probes, and subspace alignment diagnostics. Positioning IAS against these geometric steering analyses would clarify its distinct contribution versus existing representation engineering pipelines.

## Suggestions
- Provide a concrete, scalable approximation recipe for the pseudoinverse and Hessian-inverse computations at LLM scale (e.g., randomized SVD for the logit-output map, or EK-FAC/LiSSA for Hessian approximation). Benchmark the approximation error vs. compute cost across model sizes to demonstrate that the theoretical workflow actually scales to modern architectures without resorting to intractable exact matrix operations.

## Overall Score
4.6

---

## NfO2Lt2WY7

- GT: Reject (avg 2.0)
- Predicted: N/A (4.2/10)
- Match: N/A

### Review

## Summary
This paper systematically ablates the components of Group Relative Policy Optimization (GRPO) for LLM reasoning post-training, finding that PPO-style clipping is unnecessary while negative feedback (via group-relative advantage estimation) remains essential for stability. The authors propose RGRA, a simplified objective that removes clipping and policy ratios but retains KL regularization, demonstrating that it matches or marginally outperforms GRPO across several mathematical and STEM benchmarks on sub-2B models.

## Strengths
- **Well-motivated and structured ablation**: The paper cleanly isolates GRPO components (positive-only advantages, direct REINFORCE rewards, and clipping removal), providing direct empirical evidence for which mechanisms drive learning stability and reasoning emergence.
- **Comprehensive benchmark evaluation**: Testing across 9 diverse benchmarks (multi-lingual math, STEM) with multiple base models (Qwen2.5 0.5B/1.5B, Llama3.2 1B) offers a broad view of performance, and the consistent trend of RGRA ≥ GRPO strengthens the practical claim.
- **Clear qualitative analysis of failure modes**: The observation that methods discarding negative feedback (GRPO-pos, RAFT) collapse into short, degenerate responses and fail to produce reasoning traces is well-documented and underscores the necessity of relative advantage estimation in discrete policy optimization.

## Weaknesses
- **Severe scale limitation**: Experiments are restricted to ≤1.5B parameter models. RLVR dynamics, policy collapse behavior, and the necessity of trust-region constraints are known to shift significantly at 7B+ scales, making the paper's conclusions about LLM reasoning post-training highly speculative for models where reasoning emergence is typically studied.
- **Marginal gains without statistical rigor**: Reported improvements over GRPO are frequently ≤2% absolute accuracy, yet no variance across random seeds, confidence intervals, or statistical significance tests are provided. Without this, it is impossible to determine if RGRA's wins are consistent or noise.
- **Incomplete isolation of the KL penalty**: The paper frames RGRA as a move toward "simpler REINFORCE-based approaches," but the KL divergence term remains active and is widely recognized as the primary stabilizer in pure policy gradient LLM training. Failing to ablate or analyze the KL coefficient obscures whether stability comes from GRPO's group normalization or the reference model constraint.

## Nice-to-Haves
- Reporting training dynamics beyond reward/response length, such as gradient norm trajectories, policy shift (KL) curves, or entropy metrics, would clarify why clipping is redundant in this regime.
- A hyperparameter sensitivity sweep (learning rate, KL coefficient, group size) would strengthen claims about robustness and practical usability.
- Extending at least one main experiment to a 7B+ model would substantially increase confidence in the scalability of the findings.

## Novel Insights
The paper effectively demonstrates that the perceived necessity of PPO-style clipping in LLM post-training stems largely from an overgeneralization of continuous-control RL heuristics to discrete, high-capacity language models initialized from strong supervised checkpoints. However, the deeper insight is that within-group reward normalization (GRPO's advantage estimator) naturally acts as a variance-reduction and implicit update-bound mechanism: by centering rewards around the group mean and scaling by group variance, high-probability tokens that yield below-average rewards receive negative gradients, while low-probability but correct tokens are upweighted. This self-calibrating gradient shape mimics the effect of clipping without explicitly truncating the likelihood ratio, provided a KL penalty prevents catastrophic policy drift. This reframes GRPO's efficacy not as a triumph of complex surrogate objectives, but as an elegant application of relative feedback that aligns well with the high-confidence, low-variance nature of pre-trained LLM action spaces.

## Potentially Missed Related Work
- Ahmadian et al. (2024) "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" — directly investigates and theoretically/empirically questions the need for PPO clipping in LLM alignment; substantial overlap with this paper's core premise and should be discussed more critically.
- RLOO (REINFORCE Leave-One-Out) literature (e.g., Kool et al., 2019; recent LLM adaptations) — uses identical leave-one-out/group-normalized advantage estimation for variance reduction without clipping; positioning RGRA against RLOO would clarify novelty.

## Suggestions
- Explicitly scope claims to "small-scale models" in the abstract and introduction, and acknowledge that scaling to 7B+ may reintroduce the need for explicit update constraints.
- Run each main experiment with ≥3 random seeds and report mean ± std; add statistical significance tests for benchmark improvements.
- Add an ablation on the KL coefficient (e.g., β ∈ {0, 0.001, 0.005, 0.01}) to quantify how much stability relies on reference model anchoring vs. group-relative advantages.
- Discuss how RGRA relates to RLOO and recent REINFORCE-style LLM optimizers (e.g., REFT, Dr. GRPO) to better situate the contribution within the evolving RLVR landscape.

## Overall Score
4.2

---

## Vit5M0G5Gb

- GT: Accept (Poster) (avg 6.0)
- Predicted: N/A (8.2/10)
- Match: N/A

### Review

## Summary
This paper introduces a unified theoretical framework that explains dynamical simplicity bias across multiple neural architectures (fully-connected, convolutional, ReLU, and self-attention) through saddle-to-saddle learning dynamics. By formalizing embedded fixed points, invariant manifolds, and two distinct timescale separation mechanisms (data-induced vs. initialization-induced), the authors show how gradient flow progressively recruits "effective units" during training, with clear predictions verified by controlled experiments.

## Strengths
- **Architectural unification via general parameterization:** Theorems 1 and 3 establish embedded fixed points and invariant manifolds for a broad class of networks defined in Equation (1), successfully generalizing prior architecture-specific analyses (e.g., Fukumizu & Amari, 2000) to convolutional and attention-based models without ad-hoc assumptions.
- **Clear mechanistic disentanglement:** Section 5 rigorously distinguishes between data-induced timescale separation (driving low-rank, distributed weight growth in linear networks) and initialization-induced separation (driving sparse, localized growth in quadratic/attention models). This resolves apparent contradictions in the literature regarding whether simplicity manifests as rank minimization or sparsity.
- **Predictive empirical validation:** Section 6 moves beyond descriptive simulations to test novel, theory-derived hypotheses (e.g., singular value spectrum effects, initialization scale/structure, and width scaling laws). The experiments are carefully designed to isolate each theoretical mechanism, and results closely match predictions.

## Weaknesses
- **Dynamics proofs are limited to linear/quadratic two-layer cases:** The rigorous dynamical analysis (Theorem 4, Proposition 5) applies only to specific low-degree homogeneous cases near zero initialization. Extensions to deep or general nonlinear activations rely on Taylor expansion heuristics and conjectures rather than formal trajectory bounds, which limits theoretical completeness and leaves open questions about when the heuristic breaks down.
- **"Simplicity" is defined narrowly and architecturally:** Equating simplicity with the count of effective units captures dynamical progression but does not connect to statistical complexity, loss landscape flatness, or generalization guarantees. Without this link, it remains unclear whether this dynamical bias consistently improves or harms out-of-distribution generalization, limiting the practical significance of the framework.

## Nice-to-Haves
- Deriving a quantitative proximity bound or Lipschitz condition specifying how close weights must be to an invariant manifold for the escape trajectory to remain guided by it would strengthen the heuristic arguments and provide concrete initialization guidelines.

## Novel Insights
The paper’s most compelling contribution is reframing stage-wise learning as a geometric inevitability driven by architectural symmetries rather than an optimization artifact. By demonstrating that gradient flow naturally tracks invariant manifolds where the network behaves as a narrower sub-network, the authors provide a clean structural explanation for why overparameterized models learn "piece-by-piece." The disentanglement of data-driven versus initialization-driven pathways to simplicity elegantly reconciles the low-rank bias in linear models with the sparse feature learning in attention/polynomial networks, revealing that the operative definition of "simple" is dictated by which symmetry-breaking force dominates first. This perspective bridges dynamical systems theory, architectural inductive biases, and implicit regularization into a single coherent picture.

## Potentially Missed Related Work
- None identified. The literature review is thorough, spanning foundational singularity analysis, recent saddle-to-saddle proofs, spectral bias, symmetry-based learning theories, and empirical simplicity bias studies.

## Suggestions
- Validate the framework on a moderate-scale, real-world dataset (e.g., CIFAR-10/100 with shallow CNNs or a small ViT) to demonstrate that predicted plateau structures, effective unit recruitment, and timescale separation effects persist beyond synthetic/toy settings and synthetic ICL tasks.
- Explicitly map the theoretical "effective width" to practical, computable diagnostics (e.g., activation covariance rank, attention head entropy, or weight sparsity thresholds) so practitioners can monitor or deliberately manipulate simplicity bias during training.

## Overall Score
8.2

---

## b6qQmQ2F13

- GT: Accept (Poster) (avg 5.0)
- Predicted: N/A (7.8/10)
- Match: N/A

### Review

## Summary
The paper presents a large-scale empirical study on memory-accuracy trade-offs for reasoning LLMs, demonstrating that optimal memory allocation between model weights, token budgets, and KV cache compression is fundamentally scale- and task-dependent. It identifies a clear inflection point (~8-bit 4B effective size) where deployment strategies should flip: smaller models benefit from larger/higher-precision weights and KV eviction, while larger models benefit from extended generations, parallel scaling, and KV quantization.

## Strengths
- **Comprehensive and systematic empirical design:** The evaluation spans >1,700 configurations across multiple model families (Qwen3, R1-Distill, Nemotron), four challenging reasoning benchmarks, and covers weight precision, serial/parallel scaling, and KV cache compression, providing high statistical confidence in the observed trends.
- **Actionable, counter-heuristic findings:** The paper successfully challenges prevalent industry defaults (e.g., "4-bit is universally optimal" or "maximize test-time compute at all costs"), providing clear, memory-bounded deployment guidelines that are immediately useful for practitioners deploying reasoning models.
- **Clear communication and visualization:** Pareto frontier plots effectively communicate complex multi-dimensional trade-offs, and the scale-dependent threshold is intuitively presented with concrete latency/throughput backing in the appendices.

## Weaknesses
- **Lack of theoretical or mechanistic grounding:** The paper is purely empirical and offers no analytical model or scaling law derivation to explain *why* the ~8-bit 4B threshold exists or how it might shift with architectural changes (e.g., MoE, different attention mechanisms). This limits extrapolation to future model generations.
- **Narrow KV cache compression baselines:** The eviction and quantization experiments rely on relatively basic methods (StreamingLLM, R-KV, HQQ) while omitting recent state-of-the-art approaches (e.g., SnapKV, H2O, KIVI, GEAR, or Lexico). Modern methods may shift the Pareto frontiers and alter the claimed superiority of eviction for smaller models.
- **Deployment realism gap in latency analysis:** While throughput/latency is analyzed in the appendix, it uses simplified theoretical batch sizes rather than end-to-end serving measurements with dynamic batching, PagedAttention, and realistic request distributions. This weakens claims about real-world deployment efficiency.

## Nice-to-Haves
- Provide a lightweight decision matrix or interactive lookup tool that maps memory budget + task type → recommended weight precision, KV strategy, and token/group size, turning the empirical findings into a practitioner-ready artifact.

## Novel Insights
The paper’s most valuable contribution is reframing test-time scaling from a pure FLOPs/token budget problem to a constrained memory allocation problem, revealing that the widely adopted "aggressive 4-bit quantization + maximal context" paradigm is systematically suboptimal for reasoning workloads. The discovery of a scale-dependent inflection point where the memory-accuracy Pareto frontier pivots from weight-centric to KV/cache-centric strategies offers a principled alternative to heuristic compression rules. Moreover, demonstrating that mathematical/code reasoning is highly sensitive to weight precision (favoring 8/16-bit even at higher memory costs) while knowledge-intensive tasks tolerate 4-bit highlights that reasoning modalities impose fundamentally different hardware bottlenecks, necessitating task-aware, scale-conditioned deployment policies rather than one-size-fits-all compression presets.

## Potentially Missed Related Work
- **Dynamic/Mixed-Precision Decoding & KV Cache Methods** (e.g., works on adaptive precision during attention, or recent KV cache scaling analyses) — While the paper cites core quantization and scaling law literature, incorporating recent mixed-precision KV cache or test-time compute allocation frameworks could strengthen the methodological positioning, though none are strictly required to validate the core findings.
- None critically identified that would change the conclusion.

## Suggestions
- Conduct a brief mechanistic ablation (e.g., layer-wise weight sensitivity, attention entropy, or gradient norm analysis under 4-bit vs 8-bit quantization) to explain why math/code reasoning degrades sharply at low precision, which would elevate the paper beyond pure empiricism.
- Update the KV cache compression comparison to include at least one modern baseline (e.g., SnapKV or KIVI) to verify that the eviction vs quantization trade-offs remain robust, and explicitly discuss how architectural changes (e.g., MLA, grouped-query attention) might affect the memory equations.
- Move the latency/throughput vs accuracy analysis into the main text and frame it under a realistic serving setup (vLLM with PagedAttention + dynamic batching) to directly address production deployment concerns.

## Overall Score
7.8

---

## xFo13SaHQm

- GT: Accept (Poster) (avg 6.5)
- Predicted: N/A (8.2/10)
- Match: N/A

### Review

## Summary
The paper introduces WithAnyone, a diffusion-based framework for controllable, identity-consistent image generation that directly addresses the prevalent "copy-paste" artifact in personalized models. It contributes MultiID-2M, a large-scale paired multi-person dataset; MultiID-Bench, a standardized evaluation protocol featuring a novel copy-paste metric; and a four-phase training strategy combining GT-aligned identity supervision with extended-negative contrastive learning. Extensive experiments demonstrate that the method successfully decouples identity fidelity from rigid copying, achieving state-of-the-art performance while maintaining strong controllability.

## Strengths
- **Clear problem formalization and metric design:** The paper precisely identifies the "copy-paste" failure mode and introduces a mathematically grounded Copy-Paste metric (Eq. 2) that penalizes over-reliance on reference embeddings relative to ground-truth variation. This directly corrects the bias in prior Sim(Ref)-only evaluations and provides a more reliable optimization signal.
- **High-quality, reproducible resources:** MultiID-2M fills a critical gap with 500k paired group photos and diverse single-ID references, while MultiID-Bench offers a zero-overlap test set with 435 carefully curated cases. Both are released openly, which will significantly lower the barrier for future research in multi-ID customization.
- **Effective curriculum training pipeline:** The four-phase strategy (reconstruction → caption alignment → paired tuning → quality/style) combined with extended-negative InfoNCE successfully breaks the fidelity-diversity trade-off. Ablations (Table 3, Table 6) and correlation with human rankings provide strong empirical backing for each component's contribution.

## Weaknesses
- **GT-aligned loss dependency lacks inference-time justification:** The GT-aligned ID loss relies on ground-truth facial landmarks during training to compute ArcFace embeddings, which stabilizes gradients but leaves a gap in explaining how the model internalizes spatial alignment for inference. If the model implicitly expects landmark-aware features, it may struggle with extreme poses, heavy occlusions, or non-standard face crops not well-represented in the GT pairs.
- **Dataset demographic and copyright biases:** Despite using CC-filtered sources, the dataset is heavily skewed toward publicly known figures from specific regions (USA, China) and may underrepresent certain ethnicities, ages, and non-celebrity demographics. This limits zero-shot generalization to everyday users and introduces latent biases that could compound in downstream deployments.
- **Copy-Paste metric sensitivity in low-variation cases:** The denominator $\max(\theta_{tr}, \epsilon)$ normalizes against the angular distance between reference and target. When $\theta_{tr}$ is naturally very small (e.g., reference and GT share nearly identical pose/expression), the metric becomes highly sensitive to minor embedding fluctuations, potentially inflating or deflating scores without reflecting true copying behavior.

## Nice-to-Haves
- Include a failure-case gallery with quantitative analysis of edge scenarios (e.g., extreme profile views, partial occlusions, low-light references) to clearly delineate the practical boundaries of WithAnyone's controllability.

## Novel Insights
The paper's most profound contribution is conceptual: reframing identity consistency not as absolute feature matching, but as controlled deviation within a natural identity manifold. By recognizing that real-world photographs of the same person inherently occupy a dispersed region in embedding space (due to pose, lighting, expression), the authors shift the optimization target from reconstruction to relative geometric consistency. This perspective elegantly resolves the longstanding fidelity-controllability tension, suggesting that future personalization models should prioritize embedding-space dispersion and contrastive discrimination rather than minimizing distance to a single reference. The training pipeline operationalizes this insight by forcing the model to learn invariant identity signals while discarding instance-specific copies, a paradigm that could extend to other attribute-preserved generation tasks.

## Potentially Missed Related Work
- None identified. The related work section comprehensively covers single-ID preservation, multi-ID control, datasets, and benchmarks, including very recent DiT/FLUX-compatible methods.

## Suggestions
- Clarify the inference-time mechanism for facial alignment: explicitly state whether the model learns an implicit landmark prior during GT-aligned training, or if an external face detector/aligner is still required at test time, and discuss how this affects deployment latency/robustness.
- Analyze the Copy-Paste metric's behavior in low-$\theta_{tr}$ regimes and propose a stabilized variant (e.g., adaptive thresholding or confidence-weighted normalization) to prevent score volatility when reference and target are naturally similar.
- Add demographic/attribute diversity reporting (age, gender, ethnicity distribution) for MultiID-Bench to ensure the evaluation set does not inadvertently privilege majority groups and to help users gauge model fairness.
- Provide training efficiency metrics (GPU hours, FLOPs, negative sampling overhead) compared to baselines, as scaling to 4096 extended negatives may impact scalability for researchers with limited compute.

8.2

---

