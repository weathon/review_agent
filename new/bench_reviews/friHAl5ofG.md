Now let me search for calibration anchors in the human reviews.Now I have enough information to write the final review. Let me synthesize all findings.## Summary
Generative Value Learning (GVL) is a zero-shot and few-shot framework that repurposes frozen VLMs (specifically Gemini-1.5-Pro) to predict task progress ("temporal value functions") for visual robot trajectories. The core insight is that naively presenting ordered video frames causes VLMs to exploit temporal bias and produce degenerate monotonic predictions; shuffling the input frames forces the model to reason about each frame's actual completion state. The method introduces VOC (Value-Order Correlation) as an evaluation metric and demonstrates value prediction at a remarkable scale — 50 OXE datasets, 250 bimanual ALOHA tasks — with three downstream applications: dataset quality estimation, success detection, and advantage-weighted regression.

---

## Strengths

- **Novel and well-motivated shuffling insight.** The observation that VLMs shortcut temporal ordering and that shuffling forces genuine semantic reasoning is a simple but non-obvious contribution. The ablation (Table 4) demonstrates concretely that the same Gemini backbone with single-frame prediction collapses to VOC = −0.08 vs. 0.74 with GVL, and Fig. 5 (right) shows success/failure trajectories become indistinguishable without shuffling — a compelling, consistent validation across multiple settings.

- **Exceptional evaluation breadth.** Evaluating on 51 datasets, 20 embodiments, and 300+ distinct real-world tasks is substantially beyond prior work in VLM-based reward/value estimation (which typically evaluates on few simulated tasks). This scale genuinely supports the "universal" characterization.

- **Cross-embodiment in-context learning.** Fig. 4 shows that human video demonstrations improve robot value prediction, a concrete and non-trivial finding that demonstrates VLM knowledge transfer across embodiments without any fine-tuning.

- **Practical downstream applications with coherent story.** Dataset quality estimation (Table 1/DROID analysis), success detection (Table 2, Fig. 5–6), and advantage-weighted regression (Table 3) form a coherent narrative about how a universal progress estimator can improve robot learning at three distinct levels of granularity.

- **VOC as a scalable surrogate metric.** Formalizing and scaling up what was previously an informal "eye-test" in prior work (Ma et al., 2022; 2023a) into VOC is a practical contribution, and the DROID quality finding (consistent with Kim et al., 2024) is a compelling external validation.

---

## Weaknesses

### Fatal
None.

### Major

- **Backbone capability gap confounds the headline comparison.** The primary quantitative baseline is LIV — a contrastive CLIP-based model fine-tuned on human videos — while GVL runs on Gemini-1.5-Pro, trained on orders of magnitude more data at internet scale. The large VOC gap (Fig. 2) cannot be attributed to GVL's design choices alone. The ablation in Table 4 validates shuffling vs. single-frame within the same backbone, but *only on RT-1* (one dataset, 20 trajectories). Whether the shuffling mechanism — rather than the backbone's baseline capability — drives results across the full OXE scale remains unconfirmed. A "Gemini without shuffling" baseline on a broader subset would close this gap. This is the most actionable gap in the paper.

- **AWR downstream evaluation is statistically underpowered.** Table 3 uses 10 trials per task. At this sample size, binomial SE is ≈15 pp, meaning differences like 7/10 vs 6/10 (bowl-in-rack) or 7/10 vs 7/10 (fold-dress) are within noise. The two tasks where GVL decisively outperforms are close-laptop (9 vs 6.5/10) and pen-handover (1.5 vs 0/10); GVL loses clearly on open-drawer (4 vs 6/10) and remove-gears (4.67 vs 7/10). The claim that "GVL-DP outperforms DP on a majority of tasks" is directionally reasonable but not statistically substantiated, and the mixed results are material since AWR is the only evaluation that tests GVL as a genuine value-based policy improvement tool.

### Minor

- **Success detection evaluated only in simulation.** Tables 2 and Fig. 5–6 assess GVL-SD entirely on simulated ALOHA tasks. Given that the paper's strongest claims are about real-world generalization and that simulated environments have simpler visual appearances, an equivalent evaluation on real-world mixed-quality data would directly strengthen the paper's applicability claims.

- **Anchoring dependence is undercharacterized.** The paper acknowledges that the first frame must remain unshuffled to resolve temporal direction ambiguity (Section 3, Eq. 3), but does not report variance over multiple shuffle seeds, nor analyze how anchor choice affects prediction stability. This is a practically important detail for deployment.

- **In-context scaling on ALOHA-13 may overstate breadth.** The subset used for the scaling curve (Fig. 3 right) is specifically selected as the 13 tasks with ≥500 demonstrations, which likely correlates with visual distinctiveness or task simplicity. The scaling result is interesting but may not generalize to the full 250-task distribution.

- **VOC metric conceptual limitations insufficiently discussed.** VOC assumes expert trajectories have monotonically increasing progress, but real demonstrations can include pauses, backtracks, or camera occlusions that violate this. The paper acknowledges that LIV's comparison on image-goal datasets may be partially unfair (Section 4.1) but does not systematically characterize how often non-monotonic expert behavior introduces noise in VOC.

### Trivial

- The "Generative Value Learning" name implies model training or generation, but the method is a prompting technique applied to a frozen model. A clarifying sentence in the introduction would help set expectations.

---

## Nice-to-Haves

- Reporting variance of VOC across multiple random shuffles for the same trajectory would characterize prediction stability and is easy to add.
- A failure-mode analysis (qualitative or quantitative) on ALOHA tasks with near-zero or negative VOC — the paper reports that ~40% of zero-shot ALOHA trajectories have non-positive VOC — would sharpen understanding of when GVL struggles semantically vs. when it detects genuinely non-monotonic demonstrations.
- Testing GVL with open-weight VLMs (beyond the appendix mention) would substantially improve reproducibility and community uptake, given heavy reliance on the proprietary Gemini API.
- More AWR trials (≥30 per task) or confidence intervals in Table 3 to substantiate the policy learning claims.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"VOC is circular / GVL is just frame-reordering, not a value function" (Harsh Critic, Issue 1):** Partially valid in spirit but overstated in severity. The paper explicitly defines its "value function" as task progress (V^πE(oi;g) = i/T), which is standard in the robotics progress estimation literature (Serban et al., 2016; 2018; Eysenbach et al., 2020; Lee et al., 2021). The paper never claims Bellman consistency in the traditional RL sense — it explicitly notes "VLMs are not inherently trained with any consistency objective" (Section 3) and describes autoregressive consistency as a practical alternative. Calling this framing "misleading" or "circular" ignores that progress-as-value is a well-established concept, not a novel equivocation. The downstream evaluations (success detection, AWR) also validate GVL beyond VOC alone. This criticism is reduced to a Minor point (VOC conceptual limitations).

- **"Bellman equation appears and is never revisited" (Harsh Critic):** Strawman. The paper's setup uses the Bellman equation to motivate why consistency matters, then explicitly explains why VLMs cannot satisfy it via standard TD training, and proposes autoregressive generation as a practical substitute. This is clear and honest, not misleading.

- **"ALOHA-13 subselection overstates scalability" (Harsh Critic):** The subselection criterion (≥500 demonstrations) is disclosed and the result is correctly scoped to ALOHA-13. This is a reasonable experimental design choice, moved to Minor.

- **"Dataset quality estimation is qualitative and post-hoc" (Harsh Critic):** Removed. The paper is transparent that the DROID case study is a consistency check, not a held-out validation, and the Appendix G shows GVL VOC can be used for effective co-training dataset selection with concrete policy improvements.

- **"Autoregressive consistency is an intuition, not a demonstrated property" (Harsh Critic):** The ablation (Table 4) *is* the empirical demonstration. The paper does not claim a theorem; it claims a practical design principle, validated experimentally. This is appropriate for an empirical systems paper.

---

## Novel Insights

GVL reveals a specific and previously underappreciated failure mode of VLMs as value estimators: temporal bias from video pre-training causes VLMs to use chronological ordering as a shortcut rather than semantic task reasoning, producing uniformly inflated VOC scores that cannot discriminate success from failure. The shuffling intervention is deceptively simple but generalizes across 20 embodiments and 300+ tasks with no fine-tuning. This finding has implications beyond robotics — any domain where VLMs are used to assess sequential progress (e.g., video understanding, process monitoring) may face the same bias. The cross-embodiment transfer (human → robot value estimation) is an independently interesting discovery: VLMs' video pre-training provides a sufficiently abstract representation of "task completion" that generalizes across embodiment boundaries.

---

## Suggestions

1. Add a "Gemini without shuffling" ablation on at least a 5–10 dataset subset of OXE to establish that shuffling — not raw backbone capability — drives the VOC improvements at scale.
2. Increase AWR evaluation to ≥30 trials per task and report 95% confidence intervals; this is feasible in a rebuttal period for at least the highest-stakes comparisons.
3. Add one real-world success detection experiment (e.g., on a DROID subset with known failure rate or on real ALOHA mixed-quality data) to match the real-world claims made in the abstract.
4. Characterize shuffle seed variance and anchor sensitivity with a brief ablation table.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Decision | Scores | Avg |
|---|---|---|---|
| RoboFlamingo (VLM for robot imitation, spotlight) | Accept | 6, 6, 6, 8 | 6.5 |
| STRAP (sub-trajectory retrieval for policy, poster) | Accept | 5, 6, 8, 6 | 6.25 |
| GR-1 (video generative pretraining for robot, poster) | Accept | 8, 6, 5, 3 | 5.5 |
| AutoRT (VLM for large-scale robot data collection) | Reject | 5, 6, 5, 5 | 5.25 |
| DINO-WM (zero-shot world model planning) | Reject | 6, 6, 6, 5 | 5.75 |
| PIDM (scalable predictive inverse dynamics, oral) | Accept | 6, 8, 8, 8 | 7.5 |

**Reasoning:** GVL is clearly stronger than AutoRT and DINO-WM (both rejected), primarily because GVL has a cleaner technical insight (shuffling), a much larger evaluation (300+ tasks vs. single buildings / 3 toy environments), and concrete downstream applications. GVL is comparable to RoboFlamingo (spotlight, avg 6.5) — both apply VLMs to robotics with broad evaluation; GVL has more applications and larger scale but weaker quantitative downstream results (mixed AWR vs. strong CALVIN improvements). GVL is comparable to STRAP (poster, avg 6.25) in terms of scale and practical utility. The major weaknesses (backbone gap, limited AWR power, simulation-only success detection) are real but do not invalidate the core contributions. The paper lands solidly in the Accept range, most naturally as a **poster**.

**Final score: 6.5**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>