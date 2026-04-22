# MoFA: Dual-Task Motion Factorization for Human Motion Synthesis with Imperfect and Limited Data

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Human motion synthesis has recently benefited from diffusion models, achieving unprecedented realism and diversity. Yet precise and controllable generation remains challenging: text, audio, and 2D cues are often ambiguous, while existing trajectory-keyframe approaches suffer from limited generalization, naive feature fusion, and poor robustness to unpaired control signals. We identify this bottleneck as the entanglement between keyframe and trajectory signals, which are inherently coupled in training but frequently mismatched at inference. To address this, we propose MoFA, a diffusion-based Motion Factorization framework that decomposes synthesis into two complementary sub-tasks: (i) Local Motion Completion, focusing on keyframe dynamics, and (ii) Trajectory Adaptation, ensuring global spatial consistency. MoFA integrates the Local Motion Refinement Stack (LMRS) and the Trajectory-Aware Motion Integration (TAMI) to jointly refine local poses and adapt them to trajectories. In addition, we introduce a Quality-Aware Dual Training (QADT) strategy that leverages imperfect or low-quality data as auxiliary supervision, substantially expanding the effective training set and improving generalization. Extensive experiments demonstrate that MoFA achieves more stable, controllable, and robust motion synthesis than advanced baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
In this paper, it proposed to utilize both high-quality and low-quality data for training. It decompose motion synthesis into local motion completion and trajectory adaption. The proposed method has been compared with many prevailing method and shows better performance than competitors.

### Strengths
1. The proposed method shows better performance than many prevailing methods.
2. It has conducted many ablation studies.

### Weaknesses
1. Local motion synthesis and global trajectory alignment has widely been utilized in different motion synthesis framework.
2. There is no explicit description of test dataset. It is hard to judge the performance without explicit explanation of test dataset.
3. Some figures are quite small. For instance, it is hard to read the Figure 4. In addition, Figure 1 is not illustrative, making it hard to understand the main contribution.
4. According to the visualization, ProMoGen has better performance than the proposed method. The improvement over ProMoGen is not obvious.

### Questions
NA

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
This paper proposes MoFA, a diffusion-based Motion Factorization framework for controllable human motion synthesis with potentially imperfect and limited data. The core idea is to decompose generation into two conditional sub-tasks:

Local Motion Completion (driven mainly by keyframes), and Trajectory Adaptation (aligning the local motion to a global path).

The system integrates a Local Motion Refinement Stack (LMRS) and Trajectory-Aware Motion Integration (TAMI), plus a Quality‑Aware Dual Training (QADT) scheme that leverages both high‑quality (reliable trajectories) and low‑quality (unreliable trajectories) data by replacing trajectories with a learned embedding when needed. Architecture and training flow are illustrated in Figure 1 (p. 3), losses in Eq. (12), and datasets/metrics in §4. Experiments show consistent improvements over recent baselines across multiple keyframe counts (K=1,3,5,7,10), with the strongest results reported at K=5 in Table 1 and additional extensive comparisons in Tables 4–8. The paper also provides qualitative examples (e.g., backflips, falls) in Figure 3, and a feature-level analysis supporting the factorization in Figure 5

### Strengths
The paper is well written with some nice figures. It addresses a practical problem about motion synthesis training with imperfect data.

### Weaknesses
Weaknesses

Novelty about imperfect data: The claim of training motion synthesis models with imperfect data captured from video has already been demonstrated in Actor (ICCV21) - could the authors please clarify how their strategy differs from Actor - in the abstract sense of using imperfect data for training.

Novelty vs prior “local-to-global” formulations
While the paper’s decomposition is well argued, it resembles earlier hierarchical or local-to-global pipelines (for example, local refinement followed by trajectory or scene alignment). The related work is cited, but the specific modeling advance over hierarchical diffusion or curriculum-based anchor methods is not fully disentangled. Clearer positioning about what MoFA can represent that prior pipelines cannot would strengthen the originality claim (see Section 2).

Independence assumption is strong and only qualitatively probed
The central claim is that the final global motion does not depend on keyframes once local motion is inferred. The support offered (Figure 5 with PCA and embedding statistics) is suggestive rather than conclusive. A quantitative test—such as measuring performance changes when keyframe features are also provided to the global stage via a residual cross-attention or gating—would better validate that keyframes add negligible signal after the local stage.

Fairness of comparisons when using more data
QADT uses additional low-quality data. It is unclear whether top baselines in Tables 1 and 4–7 were trained with the same expanded training set or at least the same number of sequences. Table 3 contrasts “Base” (high-quality only) with MoFA (with QADT), but this does not address whether external baselines benefited from the same data budget. This leaves ambiguity about how much of the gain comes from method design versus data scale.

Ambiguity or inconsistency in dimensionality notation
Section 3.3 says low-quality inputs have f1 features and high-quality inputs have f2 features, with high-quality exceeding low-quality by six channels. Appendix A.4, however, lists high-quality as 138 features and low-quality as 132 features. The two statements contradict each other and read like a notation swap or typo that should be corrected.

Choice of diffusion head and prediction target
Appendix A.2 states that the diffusion heads output motion representations instead of noise. This deviates from common DDPM practice and could affect stability or sampling. There is no ablation comparing noise prediction, velocity prediction, or direct data prediction to justify the chosen target.

Contact handling remains light
Although foot sliding and joint smoothness are reported (Tables 2–3), there is no explicit contact loss or contact consistency metric beyond foot sliding. Adding contact precision/recall or foot velocity thresholds during stance would strengthen claims about physical plausibility.

Efficiency and sampling details
The paper mentions a 1,000-step cosine schedule and some training details (Appendix A.2–A.3), but it omits inference sampling steps, wall-clock throughput, and parameter counts. Given practical importance, these should be included.

Mixed signals on Motion Fluency
In Table 2, Motion Fluency is sometimes lower than a baseline even though higher is claimed to be better. The text suggests average improvements, but this is not consistently visible across different numbers of keyframes. The definition of the metric and its correlation with perceived quality should be clarified.

Reproducibility gaps
Loss weights in the composite objective are not enumerated. FID computation details (feature backbone, whether the metrics network is pretrained or trained) and Diversity computation specifics are not fully described. Releasing code and an evaluation harness would meaningfully improve reproducibility.

### Questions
Validate the independence claim
Please add an ablation where keyframe features are also provided to the global module (for example, via a small residual cross-attention or gating). Report the change in MPJPE, FID, and trajectory error. If performance improves, it would suggest the final motion still depends on keyframes after the local stage. A complementary conditional-dependence probe (for example, a contrastive probing setup) would help quantify any remaining dependence.

Fair data budget comparison
Clarify whether external baselines in Table 1 and Tables 4–7 were retrained on exactly the same union of datasets used by MoFA with QADT. If not, please add results where a strong baseline (for example, StableMoFusion or PMG) is trained with the same additional low-quality data so the architectural gains can be separated from data-scale gains.

Definition and selection of “low-quality” data
How are sequences labeled as low-quality versus high-quality? Is there an automated thresholding scheme (for example, trajectory smoothness, jerk, missing markers) or manual curation? Please provide counts, sequence-length distributions, and an ablation showing performance as a function of the amount of low-quality data used.

Diffusion prediction target
You state that the generators output motion representations rather than noise. Are you training with a standard noise-prediction loss, a data-prediction loss, a velocity-style target, or a hybrid? Please specify and add an ablation comparing these choices and their impact on convergence and sample quality.

Notation consistency for feature dimensions
Please fix the mismatch between Section 3.3 and Appendix A.4 regarding feature dimensions and confirm the six additional trajectory channels. Even if minor, this inconsistency is confusing for reproduction.

Robustness under extreme mismatch
Beyond the qualitative examples in Figure 3, please include a stress test where keyframes and trajectories are deliberately inconsistent (for example, facing backward while the trajectory turns sharply). Quantify failure rates for foot sliding, trajectory deviation, and pose drift relative to baselines.

Contact metrics and constraints
Would adding a contact-aware loss (for example, low foot velocity during stance, ground-penetration penalties) further reduce foot sliding without hurting diversity? A small ablation would strengthen claims about physical realism.

Interpreting Motion Fluency
Please define Motion Fluency precisely and explain why higher values are desirable. In Table 2, MoFA’s Motion Fluency is sometimes lower than a baseline; does this correlate with human preference or perceived smoothness?

Compute and efficiency details
What are parameter counts, FLOPs, and inference latency (for example, number of sampling steps and seconds per four-second clip on the reported GPU)? Is there a fast sampler or distilled variant compatible with the two-stage design?

Curriculum and loss ablations
The progressive keyframe curriculum in Appendix A.5 appears helpful. Please add an ablation removing it, and another varying the loss weights, to quantify their influence.

Where QADT helps most
In Table 3, gains appear across different numbers of keyframes. Breaking down improvements by action class or motion type (for example, locomotion versus acrobatics) would show which behaviors benefit most from low-quality local supervision.

Failure cases and limitations
Please add a short discussion and figure of typical failure cases, such as self-intersections, drift on very long trajectories, or instability with very sparse keyframes. This would make the paper more balanced and actionable.

Overall recommendation (tentative)

Lean reject. The paper tackles a practical problem—entangled controls and imperfect data—with a clean factorization and a useful training strategy. The two most important items for rebuttal are: (1) a fair data-budget comparison where a strong baseline is trained on the same expanded data, and (2) a quantitative probe of the independence claim by allowing keyframes into the global stage. 

I also think the paper is being mis-sold: the idea of using imperfect data for training has been shown to work before; the hierarchal decomposition of motion has also been shown before. I am not quite sure I see what this paper adds to the current literature.

### Soundness
2

### Presentation
2

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
This paper proposes MoFA, a diffusion-based Motion Factorization framework for controllable human motion synthesis when trajectory and keyframe controls are imperfect or unpaired. The core idea is to decompose generation into (i) Local Motion Completion from keyframes (and optionally trajectory) using a Local Motion Refinement Stack (LMRS), and (ii) Trajectory Adaptation that aligns the local motion to a path via Trajectory-Aware Motion Integration (TAMI). A Quality-Aware Dual Training (QADT) regime exploits high-quality datasets for both stages and low-quality datasets (with unreliable trajectories) for the local stage via a learnable “trajectory” embedding. On HumanML3D/CombatMotion (global) plus AIST++/MotionX++ subset (local), MoFA reports consistent gains over strong baselines across MPJPE, FID, Trajectory Error, K-MPJPE, foot sliding, and smoothness for K∈{1,3,5,7,10} keyframes, with detailed architecture, loss, and training schedules provided.

### Strengths
- Clean formulation that disentangles keyframe-driven local completion from global trajectory adaptation; QADT is a pragmatic way to mine low-quality motion data without harming the global branch.  
- Consistent quantitative gains across MPJPE/FID/Traj.err/K-MPJPE under multiple K, plus perceptual metrics (foot sliding, smoothness, autocorrelation). Comparisons include recent diffusion baselines and a controlled “Base” variant.  
- Architecture blocks (KPE/GTE, CIM, LMRS, TAMI, LMG/GMG), losses (L_{\text{rot}},L_{\text{traj}},L_{\text{vel}},L_{\text{acc}},L_{\text{joint}}), data representation (22 joints × 6D + 6 trajectory dims), and training schedule are explicitly stated.  
- Addresses a real deployment pain-point—keyframe/trajectory entanglement and data quality scarcity—showing improved stability without sacrificing diversity.

### Weaknesses
- Evidence for “unpaired control” robustness is limited. While the motivation centers on mismatched keyframes/trajectories at inference, experiments mainly use paired ground truth (Qh) for evaluation; explicit mismatch tests (e.g., swap trajectories between clips) are missing.  
- QADT relies on a binary Qh/Ql partition but criteria and label noise sensitivity are under-specified. An ablation varying the fraction/corruption of Ql or mislabels would strengthen the claim.  
- Several compared methods are not designed for joint keyframe+trajectory control; adding targeted control baselines (e.g., TLControl or adapting OmniControl with keyframes) would better isolate MoFA’s benefits. Ensure identical keyframe/trajectory protocols and report wall-clock/sample counts.  
- FID for motion features, Diversity, JS, and foot-sliding need precise definitions (feature extractor, windowing, units). Report exact training/seed counts and CIs; some tables show small SDs without detailing runs.  
- Beyond Base vs. MoFA, a finer ablation (LMRS only / TAMI only / QADT off) and swapping the 6-channel trajectory encoding would clarify which parts drive which metrics.

### Questions
1. Can you report results when keyframes and trajectories are intentionally mismatched (e.g., random trajectory with fixed keyframes)? Does MoFA degrade gracefully vs. direct-fusion baselines?  
2. How are Qh/Ql labels assigned in practice? What happens if 10–30% of Ql actually contain reliable trajectories (or vice versa)? Please add sensitivity curves.  
3. What exactly are the 6 trajectory channels (XY position + facing + height + velocities?) and at what rate? Any benefit to spline or velocity-only encodings?  
4. Results for (LMRS only), (TAMI only), and (QADT off) under K={1,5,10} would help attribute gains.  
5. Training/inference speed vs. baselines, guidance cost for larger K, and sampling steps? Any trade-offs with DiT depth/heads?  
6. Do results transfer to in-the-wild or longer sequences beyond benchmark stats? Any failure modes (e.g., sharp turns, jumping with sparse keyframes)?

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
4

### Summary
The paper presents a motion factorization framework, MoFA, that decomposes the motion synthesis task into two complementary sub-tasks: (1) Local Motion Completion and (2) Trajectory Adaptation. MoFA comprises Local Motion Refinement Stack (LMRS) and Trajectory-Aware Motion Integration (TAMI) to perform local pose refinement and trajectory alignment. Furthermore, the paper introduces the Quality-Aware Dual Training (QADT) strategy, which utilizes low-quality data as auxiliary supervision, improving generalization. Experimental results show that MoFA can outperform previous methods in terms of frame-wise pose quality and trajectory alignment.

### Strengths
1. Although there is room for improvement in presentation, the paper itself is written well enough to make readers understand the authors' motivation, the proposed method, and the experimental results.
2. The proposed framework is well-designed for motion synthesis conditioned on keyframe pose and trajectory.
3. The experimental results demonstrate that the proposed method outperforms previous methods in terms of frame-wise pose quality and trajectory alignment.

### Weaknesses
1. When I saw the generated samples (supplementary material), I felt that the foot sliding was large. In my impression, the foot sliding was not so large in OmniControl or MotionLab. The proposed method should be compared to the previous methods also in terms of Foot Sliding.
2. L.198 (Section 3.2) says "the multimodal distribution $P(m_2|t, k)$ is simplified into two conditional distributions", but this is not obvious. An objective explanation of why the combination of two decomposed conditional distributions is simpler than the original distribution would be required. In my opinion, this decomposition is one of the reasons why the foot sliding is large, even though the decomposition facilitates frame-wise pose quality and trajectory alignment. So, it is not obvious that this decomposition is effective.
3. This is a minor weakness. Figures 1, 2, and 4 are a little fine. The figures are a little hard to see.

### Questions
I would appreciate it if the authors could address Weaknesses 1 and 2 that I provided above.

### Soundness
3

### Presentation
2

### Contribution
2
