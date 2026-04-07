=== CALIBRATION EXAMPLE 85 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "Embodied Navigation Foundation Model" is broadly accurate, though it positions the work more ambitiously than its scope fully justifies—the model still requires task-specific scaling factors and waypoint converters for different embodiments, and there is no true zero-shot transfer *between* embodiment types. The abstract's claim of "state-of-the-art or highly competitive performance … without requiring task-specific fine-tuning" deserves scrutiny: on NAVSIM (Table 4), NavFoM (84.3 PDMS) is actually *below* the camera-only baseline LAW (84.6), and LiDAR-based methods reach 88.1. The "SOTA" claim is thus true only in the VLM-with-camera-only subcategory on some benchmarks.

---

### Introduction & Motivation

The motivation for a cross-task, cross-embodiment navigation model is well-articulated and timely. The authors correctly identify the gap between VLM generality and the embodiment-specific nature of most prior navigation work.

However, contributions are scattered across several paragraphs rather than being enumerated cleanly. The statement that NavFoM handles "diverse tasks … without requiring task-specific fine-tuning" (p.2) slightly misleads: task-specific scaling factors *α_task* (Eq. 6, Table 5) and task-specific trajectory normalization are still required. This is an important detail that should be clarified upfront.

Additionally, Figure 1 (radar chart) is referenced for benchmark comparisons but the axes and absolute numbers are not legible from the description, making it difficult to assess claims in the introduction.

---

### Method

**TVI Tokens (Sec. 2.1.1):** The design is principled. Using sinusoidal encodings of cos(φ) and sin(φ) to preserve circular continuity of azimuthal angles is technically sound. The UMAP visualization in Figure 3 is a nice qualitative verification of separability. The three-way formulation (navigation / video QA / image QA) using different combinations of base + time + angle embeddings is elegant and covers the intended use cases.

*Concern:* The ablation (Table 8) only tests TVI tokens on the VLN-CE RxR four-camera setting. There is no evaluation of how TVI tokens behave when camera count varies from 1 to 8 within the same model, which is claimed as a key flexibility advantage. The clustering visualization samples a fixed 60-angle, 30-timestep grid—it is not clear whether this regularity persists during actual inference with irregular timestep distributions.

**BATS (Sec. 2.1.2):** The forgetting-curve inspiration is intuitive. However, several issues arise:

- The sampling is *stochastic* at inference time. The authors do not discuss variance across inference runs or whether results in Tables 1–4 were averaged over multiple seeds. For real-world deployment, stochastic token selection could cause non-reproducible behavior.
- Eq. 5 and its derivation contain a typographical inconsistency: the displayed formula reads E_frames ≈ T + εT/k but the integral (1-ε)(1-e^{-k})/k · T + εT does not simplify to that expression for general k; the approximation is only valid for small k. No justification is given for why k is always small.
- The bound "Equation 5 may become unsolvable for very large T" is acknowledged but dismissed as rare (p.5). No fallback ablation is provided—only a qualitative note about removing oldest frames.

**Trajectory Prediction (Sec. 2.1.3):** The three-layer MLP action head is lightweight, but the loss term β = 10 is described as "important when training scale is small" and heuristically set. No ablation of β sensitivity is provided, even though the authors acknowledge its importance.

**Training Configurations:** A single training epoch over 12.7M samples on 56 H100 GPUs for 72 hours. Training only one epoch raises questions about convergence—is this a deliberate stopping criterion, or a compute constraint? No training loss curves or validation loss trends are shown.

---

### Experiments & Results

**VLN-CE (Table 1):** The improvement on RxR Val-Unseen (single-view SR: 51.8% → 57.4%) versus StreamVLN-RGB-only is the clearest apples-to-apples comparison and represents a genuine gain. The multi-view result (56.3% → 64.4%) is impressive but less directly comparable because HNR, the prior SOTA, uses RGB-D depth *and* odometry information in addition to RGB. NavFoM achieves competitive results without depth/odometry, but it uses four cameras, so the absolute comparison needs qualification. The paper's notation in Table 1 shows NavFoM under "M.RGB ✓" while the baselines under the multi-view block also use "Depth ✓" and "Odo. ✓"—this difference should be highlighted and discussed more prominently rather than buried.

**Object Navigation (Table 2 / Table 7):** The paper reports a single zero-shot result of 45.2% SR (VAL UNSEEN) vs 43.6% for Uni-NaVid* (fine-tuned), an improvement of 1.6%. This is a relatively small margin given that NavFoM uses roughly 8× more training data. The gain may be largely attributable to data scale rather than architecture. A data-controlled ablation (e.g., training NavFoM on the same data as Uni-NaVid) is absent.

**Autonomous Driving (Table 9):** LiDAR-based methods (DiffusionDrive: 88.1 PDMS) substantially outperform NavFoM (84.3 PDMS). Among camera-only methods, LAW (84.6) is marginally better. The authors do not analyze why NavFoM—despite a much larger VLM backbone—underperforms these domain-specialized models; there is no discussion of whether the multi-task co-training hurts driving performance compared to driving-only training.

**Active Tracking (Table 3 / Table 8):** The main paper (Table 3) omits NavFoM's own results and only shows them in Table 8 of the appendix. This is confusing and makes the main-text contribution appear incomplete. The gain from single-view to four-view tracking is modest (0.6% in SR), contrary to the VLN improvements. The authors attribute this to target-spawn distribution in EVT-Bench (targets mostly in front), which is reasonable, but it also raises a concern about generalization to real multi-view tracking.

**Ablation Study (Sec. 3.3):** The ablation of BATS (Table 8) is solid and shows consistent gains. However:
- All ablations are run only on RxR four-camera VLN, not across other tasks or embodiments.
- There is no ablation isolating the contribution of web-video navigation data (Sekai, 2.03M samples with "imperfect instructions and trajectories"). Given its large proportion (25% of navigation data), its effect on and potential degradation of benchmark performance should be assessed.
- The cross-task synergy plot (Figure 7) is described but results are only partially reported in text—the improvement in Tracking from 12.6% to 62.0% is a dramatic gain that deserves more analysis (is it primarily from tracking data itself, or from cross-task transfer?).

**Real-World Experiments:** Figure 6 shows qualitative visualization for multiple platforms (humanoid, quadruped, drone, wheeled). No quantitative metrics are reported (success rate, tracking quality, etc.) for real-world experiments. This weakens the "strong generalizability and practical applicability" claim in the abstract. At minimum, a small-scale quantitative study (e.g., N trials per robot type, SR) would substantially strengthen the real-world section.

**Statistical Significance:** No confidence intervals, standard deviations, or repeated experiment statistics are reported anywhere. Given that BATS involves stochastic sampling, result variance is non-trivial.

---

### Data

The dataset assembly is the paper's most impressive engineering contribution, but several issues deserve attention:

1. **Data leakage risk**: NavFoM is trained on data from VLN-CE R2R, RxR, and HM3D ObjectNav and then evaluated on those same benchmarks' val-unseen splits. The paper clarifies that episodes are disjoint (different environments), but the training scenes and evaluation scenes come from the same underlying 3D scene datasets (HM3D, MP3D). The val-unseen splits are specifically designed to test generalization to unseen environments—it would be useful to confirm that no evaluation-environment visual data appears in the training set.

2. **Trajectory quality of web-video data**: Sekai trajectories are generated via SLAM systems on web videos. These pseudo-trajectories may encode ego-motion from arbitrary cameras (moving people, handheld cameras) that don't correspond to robotic navigation kinematics. No analysis of trajectory quality or filtering criteria is given.

3. **Data imbalance**: VLN data is 3.37M samples vs Autonomous Driving 681K. The authors use no mention of per-task sampling ratios or curriculum strategies. This could lead to VLN domination of the gradient signal.

---

### Related Work

The related work section is brief (one paragraph) and adequate but does not engage with concurrent work on foundation models for robotics at a conceptual level (e.g., comparisons with RT-X, GNM, ViNT in terms of how NavFoM differs architecturally from cross-embodiment navigation policies). The distinction between NavFoM and prior generalist VLN methods (GR-1, Uni-NaVid) is not clearly articulated in the related work.

---

### Limitations & Broader Impact

The ethics statement is formulaic. Key technical limitations are acknowledged in passing across the paper but not synthesized:

- The system requires a remote server with RTX 4090/5090 + Internet communication for real-world use (latency/reliability concerns for navigation are non-trivial).
- Task-specific scaling factors mean that deploying NavFoM on a new embodiment still requires domain knowledge.
- The model cannot handle new camera configurations at test time without re-solving BATS parameter k offline.
- No discussion of failure modes: what happens when the language instruction is ambiguous or when the agent faces situations outside training distribution?

---

### Writing & Clarity

There are numerous typographical errors throughout (e.g., "Comprasion," "Detials," "performence," "correspondgding," "Toekns," "Sever" for Server) that impede professional presentation. More importantly:

- Table 2 in the main paper is inlined without NavFoM's own numbers, which appear only in Table 7 of the appendix.
- The "Basic Architecture" paragraph at the start of Section 2 precedes the formal definition of variables (e.g., N, T), creating readability issues.
- Figure 7 (cross-task ablation) is described in the text but hard to interpret from the raw numbers provided, with no supplementary table.

---

## Overall Assessment

NavFoM is an ambitious and technically sound paper that makes a genuine contribution to multi-task, multi-embodiment navigation. The TVI token design is well-motivated and the empirical results on VLN-CE benchmarks (particularly the multi-view setting) represent meaningful improvements over the state of the art. The large-scale data collection effort is impressive. However, the paper has several notable weaknesses that give pause for an ICLR acceptance decision at its current state: (1) the data-scale advantage over baselines is not controlled for, making it hard to isolate architectural contributions; (2) real-world experiments are purely qualitative; (3) the autonomous driving results are not SOTA even within camera-only methods; (4) BATS introduces stochastic inference behavior with no variance analysis; (5) ablations are confined to a single benchmark configuration; and (6) the "cross-embodiment generalization" framing somewhat overstates what is demonstrated—each benchmark evaluates a specific embodiment, not transfer between them. With these issues, the contribution is meaningful but not yet fully substantiated in a way that meets ICLR's bar for rigor. A significant revision addressing controlled baselines, real-world quantitative results, and broader ablations would substantially strengthen the paper.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents NavFoM, a cross-embodiment and cross-task foundation model for embodied navigation trained on 8.02 million diverse navigation samples including quadrupeds, drones, wheeled robots, and vehicles. The model introduces Temporal-Viewpoint Indicator (TVI) tokens and a Budget-Aware Temporal Sampling (BATS) strategy to handle varying camera configurations and token budgets within a unified VLM-based architecture. Extensive evaluations across seven public benchmarks and real-world experiments demonstrate state-of-the-art or competitive zero-shot performance without task-specific fine-tuning.

### Strengths
1.  **Comprehensive Scale and Diversity:** The collection and curation of 12.7 million total training samples (8M navigation + 4.76M QA) across 4 distinct embodiment types (quad, drone, wheeled, car) and multiple tasks (VLN, Tracking, Driving, ObjectNav) is a significant contribution to the field. This scale directly supports the foundational model claim better than prior narrower works.
2.  **Robust Generalization:** The model achieves zero-shot performance that is competitive or SOTA on multiple benchmarks (e.g., RxR VLN-CE, HM3D-OVON, EVT-Bench) without requiring fine-tuning for specific tasks or camera counts. Notably, it outperforms SOTA baselines using RGB-only inputs in multi-view VLN settings where baselines often rely on depth or odometry (Table 1).
3.  **Effective Architectural Solutions:** The Temporal-Viewpoint Indicator (TVI) tokens address the critical challenge of aligning temporal and multi-view inputs in a unified LLM context. Ablation studies (Section 3.3, Table 8) convincingly demonstrate that TVI tokens significantly outperform standard positional embeddings or handcrafted tokens. Similarly, the BATS sampling strategy effectively balances inference speed and context retention (Figure 4).
4.  **Real-World Validation:** Unlike many works limited to simulation, this paper includes real-world deployment experiments on a diverse set of platforms (humanoid, quadruped, drone, wheeled) with 110 reproducible test cases (Section E), confirming the model's practical utility beyond synthetic environments.

### Weaknesses
1.  **Performance Gap in Autonomous Driving:** While competitive, NavFoM shows a performance gap compared to specialized autonomous driving models on certain metrics. For instance, on NAVSIM (Table 9), NavFoM achieves a PDMS of 84.6 compared to DiffusionDrive's 88.1. The paper attributes this to a lack of explicit driving-related information modeling (lanes, vehicles), but a deeper analysis of where and why this gap occurs in complex scenarios would strengthen the contribution.
2.  **Task-Specific Normalization:** Although the architecture is unified, the trajectory prediction step requires task-specific scaling factors ($\alpha_{task}$) for different embodiments (indoor, UAV, car) (Eq. 6 and Appendix A.1). While this is a practical necessity, the need for manual configuration of these scaling factors slightly undermines the "foundation" claim compared to a fully end-to-end normalizable solution.
3.  **Limited Analysis of Real-World Uncertainty:** The real-world experiments (Section E) report success on 110 cases but lack detailed statistical analysis of failure modes or confidence intervals compared to the rigorous benchmark reporting. Failure analysis (Section F) focuses heavily on simulation or general limitations rather than specific deployment challenges (e.g., network latency, lighting changes) encountered in the real-world tests.
4.  **Inference Latency:** The reported inference speed (218 ms per prediction, ~5 Hz on RTX 4090) may be a bottleneck for high-speed dynamic embodiments (quadrotors or fast wheels). While quantization is discussed, the impact of this latency on closed-loop control stability in real-time scenarios without aggressive prediction horizons warrants more discussion.

### Novelty & Significance
**Novelty:** Moderate to High. The core novelty lies in the scale of the cross-embodiment dataset and the specific integration of TVI tokens for multi-view temporal alignment within a VLM. While the VLM backbone is standard, the application to navigation across such varied physical dynamics and the specific token engineering for camera horizons represent a meaningful technical advancement.
**Significance:** High. The embodied navigation community faces fragmentation across tasks and embodiments. NavFoM successfully demonstrates that a unified model can generalize across these domains effectively. The results set a strong benchmark for future "embodied foundation models" and provide a valuable resource (data + code).
**Reproducibility:** High. The authors provide detailed training configurations, data sources, and explicitly state that code and weights will be made publicly available. The appendix provides sufficient implementation details (e.g., scaling factors, token budgets) to replicate the results.

### Suggestions for Improvement
1.  **Deepen Analysis of Driving Performance Gap:** Investigate the ~4-5% PDMS gap in driving tasks further. Specifically, analyze if incorporating scene structure prompts or specific driving tokens (as suggested in the discussion) could bridge this gap without compromising the cross-task generality.
2.  **Refine Normalization Strategy:** Discuss the trade-offs of the $\alpha_{task}$ scaling factors. Could a learnable normalization scale or domain-adaptive layer replace manual tuning in the next iteration to further unify the architecture?
3.  **Expand Real-World Evaluation Metrics:** Include more statistical rigor in the real-world section, such as success rate confidence intervals and failure analysis specifically tied to sensor noise or latency issues encountered during deployment, rather than just listing test cases.
4.  **Clarify "Zero-Shot" Claims:** Ensure all "zero-shot" comparisons are fair regarding input modalities. While the paper highlights RGB-only strength, explicitly clarifying if any baseline used auxiliary information (like semantic maps or odometry) that NavFoM deliberately excluded, and why that design choice was made, would strengthen the validity of the comparative claims.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Strict Cross-Embodiment Hold-Out:** Train exclusively on one embodiment class (e.g., wheeled) and evaluate on a held-out class (e.g., quadruped/drone) without any target-embodiment data in the training mix to validate the "cross-embodiment" claim.
2. **Generalist VLA Baselines:** Compare against recent generalist Vision-Language-Action models (e.g., OpenVLA, RT-2) adapted for navigation to contextualize whether NavFoM offers advances over existing foundation policies.
3. **Real-World Safety Metrics:** Report quantitative safety statistics (collision rates, near-miss counts) for the 110 real-world test cases, as success rate alone is insufficient to validate safe deployment claims.
4. **QA Data Ablation:** Remove the 4.76M QA samples from training to verify if co-tuning genuinely enhances navigation capabilities or merely acts as a regularizer.

### Deeper Analysis Needed (top 3-5 only)
1. **Long-Horizon Stability:** Analyze success rate degradation as trajectory length increases to determine if the model suffers from compounding errors over extended navigation horizons.
2. **TVI Token Utilization:** Provide attention weight analysis on Temporal-Viewpoint Indicator tokens to confirm the LLM actively uses these embeddings for reasoning rather than ignoring them.
3. **Token Budget Sensitivity:** Evaluate performance at budgets significantly below 1024 tokens to establish the minimum compute threshold required for functional edge deployment.
4. **Training Data Contribution:** Quantify the performance gain attributable to the web-video navigation data (Sekai) versus curated simulator data to assess the value of noisy web-scale data.

### Visualizations & Case Studies
1. **Failure Case Trajectories:** Visualize specific instances where the model collided or deviated significantly to expose systematic failure modes (e.g., viewpoint confusion, obstacle Ignorance).
2. **Cross-Embodiment Execution:** Show side-by-side trajectories of distinct robots (e.g., drone vs. quadruped) executing the *same* instruction to visually demonstrate true policy transfer.
3. **TVI Embedding Generalization:** Expand Figure 3 to include unseen viewpoint angles in the clustering visualization to verify the indicator tokens generalize beyond training configurations.

### Obvious Next Steps
1. **Closed-Loop Long-Horizon Real-World:** Extend real-world evaluation beyond short test cases to assess recovery from errors in unstructured, dynamic environments over longer durations.
2. **Inference Speed Optimization:** Address the 5Hz inference bottleneck via model distillation or smaller backbones to meet the latency requirements of high-speed embodiments (drones/cars).
3. **Explicit Obstacle Avoidance Metrics:** Include metrics like Time-to-Collision (TTC) or minimum clearance distance in driving/tracking benchmarks to rigorously evaluate safety beyond path completion.

# Final Consolidated Review
## Summary

NavFoM presents a cross-embodiment and cross-task navigation foundation model trained on 8 million navigation samples spanning quadrupeds, drones, wheeled robots, and vehicles. The model introduces Temporal-Viewpoint Indicator (TVI) tokens to encode camera configuration and temporal context within a unified VLM architecture, along with Budget-Aware Temporal Sampling (BATS) for efficient inference. Evaluations across seven benchmarks demonstrate competitive or state-of-the-art performance without task-specific fine-tuning, with real-world validation on multiple robotic platforms.

## Strengths

- **Comprehensive Scale and Diversity:** The curated dataset of 12.7 million samples (8M navigation + 4.76M QA) across four distinct embodiment types and multiple navigation tasks (VLN, ObjectNav, Tracking, Driving) represents a substantial engineering contribution that directly supports the foundation model ambition.

- **Strong VLN-CE Results:** On VLN-CE RxR Val-Unseen, NavFoM achieves meaningful improvements: single-view SR improves from 51.8% to 57.4% over StreamVLN-RGB-only, and multi-view SR reaches 64.4% compared to prior SOTA (HNR: 61.0% with RGB-D + odometry). The multi-view gains using only RGB cameras are particularly notable.

- **Principled Multi-View Token Design:** The TVI token mechanism (Eq. 3) elegantly encodes temporal and viewpoint information using sinusoidal encodings of cos(φ) and sin(φ) for azimuthal angles, preserving circular continuity. The UMAP visualization (Figure 3) provides qualitative verification that learned embeddings separate by viewpoint and timestep.

- **Effective Token Management:** The BATS strategy (Eq. 4-5) provides a principled approach to temporal sampling under token budgets, with ablations (Table 8) demonstrating consistent improvements over uniform and linear sampling across different budget settings (1024 and 2048 tokens).

- **Real-World Validation:** Unlike many embodied AI papers limited to simulation, this work includes real-world deployment experiments across humanoid, quadruped, drone, and wheeled platforms with 110 reproducible test cases (Section E), providing practical validation beyond synthetic benchmarks.

## Weaknesses

- **Task-Specific Scaling Factors Undermine Foundation Claim:** Despite the unified architecture, trajectory prediction requires task-specific scaling factors α_task (Eq. 6, Table 5) for different embodiments (indoor: 1.0m, UAV: 7.93m, car: 50.8m). While practical, this manual configuration partially contradicts the "foundation model" framing that suggests zero-shot applicability to new embodiments. Deploying on a new platform still requires domain knowledge to set these parameters.

- **Autonomous Driving Performance Not Competitive:** On NAVSIM (Table 9), NavFoM achieves 84.3 PDMS, which is below LAW (84.6) among camera-only methods and substantially below LiDAR-based methods (DiffusionDrive: 88.1). The paper acknowledges this gap but does not deeply analyze whether the multi-task co-training dilutes driving performance compared to driving-only training.

- **Cross-Embodiment Generalization Not Rigorously Validated:** The paper evaluates on different benchmarks for different embodiments but does not test strict cross-embodiment transfer (e.g., training exclusively on indoor robots and testing zero-shot on drones). The current evaluation shows task generalization within each embodiment domain, not transfer across embodiment classes.

- **Stochastic Inference Behavior:** BATS sampling is stochastic at inference time (Eq. 4), but no variance analysis across multiple seeds is reported. For real-world deployment where reproducibility matters, understanding the variance in trajectory predictions would be valuable.

- **Qualitative Real-World Reporting:** While the real-world section includes 110 test cases, the primary reporting is qualitative (Figure 6, 16). A summary table with success rates, failure modes, and comparison to baselines for each robot type would significantly strengthen practical claims.

## Nice-to-Haves

- **Attention Analysis on TVI Tokens:** Visualizing attention weights on TVI tokens would confirm whether the LLM actively uses these embeddings for reasoning or treats them as auxiliary.

- **QA Data Contribution Ablation:** Training without the 4.76M QA samples would clarify whether co-tuning genuinely enhances navigation or merely regularizes the model.

- **Learnable Scaling Factor Exploration:** Discussing whether α_task could be replaced with learnable embodiment-specific adapters in future work would strengthen the foundation model roadmap.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **TVI Token Ablation Across Camera Counts:** The harsh critic claimed "no evaluation of how TVI tokens behave when camera count varies from 1-8." This is incorrect—Table 14 in the appendix provides exactly this analysis, comparing 1/2/3/4/6 camera configurations on VLN-CE RxR.

- **Typographical Errors as Major Criticisms:** Minor typos ("Comprasion," "Detials") are noted but do not constitute substantive technical weaknesses.

- **Single-Run Evaluation Standards:** Demanding confidence intervals for large-scale benchmark evaluation is not standard practice in embodied AI; most VLN papers report single-run results.

- **Data Leakage Concern:** The critic raised concerns about training on R2R/RxR/HM3D and evaluating on their val-unseen splits. This is standard practice—val-unseen splits are specifically designed to use disjoint environments from training.

- **Comparisons to Manipulation VLA Models:** The spark finder suggested comparing to OpenVLA/RT-2, but these are manipulation-focused models. This is scope creep beyond the navigation focus of this paper.

## Novel Insights

The cross-task synergy analysis (Figure 7) reveals an interesting pattern: Tracking and Searching tasks show dramatic improvements (12.6% → 62.0% and 10.3% → 45.2%) when co-trained with other navigation data. The authors attribute this to the mismatch between training conditions (single-view, closed-set categories) and evaluation settings (multi-view, open-vocabulary). This suggests that the primary benefit of multi-task training may be improving robustness to distribution shift rather than transferring core navigation skills. This observation warrants deeper investigation into how multi-task training affects generalization boundaries.

## Suggestions

1. **Add Cross-Embodiment Hold-Out Experiment:** Train on three embodiment classes and evaluate zero-shot on the fourth to substantiate the cross-embodiment claim.

2. **Quantify Real-World Results:** Include a summary table with success rates for each robot type in real-world experiments, with failure mode categorization.

3. **Analyze Driving Performance Gap:** Investigate whether incorporating scene structure prompts (as suggested in the discussion) can bridge the performance gap without compromising cross-task generality.

4. **Report Inference Variance:** Given stochastic BATS sampling, report standard deviation across multiple seeds for key benchmarks.

5. **Clarify Zero-Shot Comparisons:** Explicitly note in tables which baselines use auxiliary inputs (depth, odometry) to ensure fair comparisons are transparent.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 10.0]
Average score: 8.0
Binary outcome: Accept
