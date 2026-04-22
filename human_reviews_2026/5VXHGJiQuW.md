# Learning Personalized Driving Styles via Reinforcement Learning from Human Feedback

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Generating human-like and adaptive trajectories is essential for autonomous driving in dynamic environments. While generative models have shown promise in synthesizing feasible trajectories, they often fail to capture the nuanced variability of personalized driving styles due to dataset biases and distributional shifts. To address this, we introduce TrajHF, a human feedback-driven finetuning framework for generative trajectory models, designed to align motion planning with diverse driving styles. TrajHF incorporates multi-conditional denoiser and reinforcement learning with human feedback to refine multi-modal trajectory generation beyond conventional imitation learning. This enables better alignment with human driving preferences while maintaining safety and feasibility constraints. TrajHF achieves performance comparable to the state-of-the-art on NavSim benchmark. TrajHF sets a new paradigm for personalized and adaptable trajectory generation in autonomous driving.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
* This paper introduces TrajHF, a framework for finetuning generative trajectory models in autonomous driving motion planning to align with personalized driving styles.
* The approach addresses a key limitation of standard imitation learning: Learning a multi-model distribution of human preferences instead of an average.
* The authors first train a DDPM using a Multi-Conditional Denoiser architecture that processes multi-modal sensor inputs (camera and LiDAR).
* This base model is then finetuned with RLHF. For that, a reward model is trained on a specially curated semi-synthetic dataset of human preferences and used to update the diffusion policy via the DPGRPO algorithm (combination of adapted GRPI and BC loss for regularization).
* Experimental results show that the approach works well on navtest. Furthermore, results on a newly introduced BOE score show that human evaluators prefer the TrajHF-proposed trajectories of their preferred driving styles.

### Strengths
* This paper addresses an interesting problem of adjusting driving styles based on user preferences.
* The proposed method is sound. It achieves good (but not SOTA) performance on navtest.
* The evaluation on both navtest and internal data with the newly introduced BOE is thorough and shows good results.
* The approach is well ablated in the appendix, incl. comparisons to PPO and DPO.
* The data collection strategy based on takeovers that correspond to critical moments of preference makes sense and is an interesting idea.

### Weaknesses
* The method does not achieve SOTA on navtest (although the authors claim “comparable to state-of-the-art” in the Abstract).
* The reward model is not independently validated. This could be done on a held out test set of human preference pairs.
* The claim about safe and feasible trajectories is not well supported by experimental results.
* The split on defensive and aggressive seems somewhat simple (albeit easy to understand) and might not translate well to the real-world driving task (aggressive might be more accident-prone).

### Questions
* Can you add an independent validation of the reward model?

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
4

### Summary
TrajHF is a diffusion-based trajectory planner finetuned with human feedback to personalize driving style. It adds a multi-conditional denoiser for images, LiDAR, and action history, then applies preference-based RL alignment (a GRPO-style objective over groups of K sampled trajectories) plus an EM selector to refine multi-modal samples. Empirical performance shows TrajHF improves human-rated style alignment but remains below GoalFlow on NavSim, indicating modest gains in personalization without surpassing public-benchmark baselines.

### Strengths
1. A simple critic-free RL recipe on the driving problem that is compute-friendly.
2. Preference alignment improves style without collapsing feasibility.

### Weaknesses
1. Moderate novelty: diffusion-as-MDP and group-relative advantages are adapted from prior work; EM selection echoes earlier trajectory aggregation ideas.
2. Benchmark performance: Public benchmark underperforms SOTA without the ``selector” upper bound. Key wins rely on internal datasets, semi-synthetic pairs, and human BOE.

### Questions
1. After style tuning, what happens to TTC and comfort?
2. Why is your best deployable PDMS lower than GoalFlow; what ablations explain the gap?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes **TrajHF**, a framework for learning **personalized driving styles** in autonomous driving by combining diffusion-based trajectory generation with RLHF. The method extends a DDPM-based diffusion policy with a Multi-Conditional Denoiser (MCD) Transformer that conditions on multi-modal inputs (camera, LiDAR, and past actions). To align the generated trajectories with diverse human preferences (e.g., “aggressive” vs. “defensive” styles), the authors introduce a DPGRPO algorithm for diffusion finetuning using human feedback. Experiments are conducted on the public NavSim benchmark and internal preference datasets.

Overall, the paper is well-written and presents a well-engineered system addressing preference-aligned autonomous driving. However, results on public benchmarks show **parity rather than improvement** over prior methods, and key findings on personalization rely heavily on private internal datasets without comparison to other baselines, making the evidence for the claimed advantages **less convincing**.

### Strengths
* The paper tackles an interesting and practical problem, **personalization of driving trajectories**, by leveraging RLHF within a diffusion policy framework. 
* The anchor-free trajectory generation and multi-conditional denoiser are technically elegant designs that remove limitations of anchor- or vocabulary-based approaches in prior work.
* The work contributes to a growing line of research connecting generative modeling, imitation learning, and human preference alignment, which is a significant direction for building human-trustworthy robotic systems.
* The construction of an **internal dataset** focused on driving style variation (aggressive v.s. defensive) is valuable and demonstrates substantial engineering effort, especially in developing the human preference evaluation framework.

### Weaknesses
1. **Experimental results are not sufficiently strong to validate the main claims.**
   On the NavSim benchmark, TrajHF (EM) achieves 87.6 PDMS, comparable to Hydra-MDP (86.5) and DiffusionDrive (88.1), but below GoalFlow (90.3). These results indicate that TrajHF performs competitively but is **not comparable to the state-of-the-art** methods, contrary to the paper’s claim.

2. **Personalization results rely entirely on private data.**
   The main contribution, personalized trajectory generation, is verified solely through an internal dataset that is unavailable for public evaluation. This raises questions about the quality of driving data, annotation consistency, and label balance. The authors also acknowledge that standard metrics such as ADE and FDE do not capture behavioral styles, yet these metrics are still heavily used for evaluation, which weakens the argument. The proposed BOE metric for human evaluation is interesting but subjective and lacks statistical rigor (e.g., variance, confidence intervals, significance testing).

3. **Lack of safety and generalization analysis.**
   Personalization could introduce safety-critical behavior (e.g., overly aggressive trajectories), but the paper does not evaluate whether the finetuned policy maintains safety or robustness under distribution shifts. Metrics such as collision rate or rule compliance are not reported.

4. **Minor presentation and mathematical issues.**
   Some typographical and mathematical inconsistencies should be corrected:

   * Line 53: “Multi-Conditioned Denoiser (MDC)” should be “(MCD)”.
   * Line 215: If timestep $l = 1$, the projection $ \hat{x}_1 = s_1 - s_0 $, but Equation (1) defines the state starting from $s_1$, causing an inconsistency in the definition.

### Questions
1. **PDMS selector setup:**
   The setup of the PDMS selector variant is unclear. Please clarify how it differs from the single-sample and EM variants of TrajHF, and what assumptions or oracle information it uses.

2. **Dataset transparency and annotation quality:**
   Given that most results rely on private internal datasets, could the authors consider releasing them for further evaluation?
   How are the “aggressive” and “defensive” ground truths defined and ensured to be feasible and meaningful?
   Are there any statistics on inter-annotator agreement or annotation consistency?

3. **Algorithmic contribution of DPGRPO:**
   How exactly does DPGRPO differ from existing GRPO or DPO implementations?
   Is there measurable improvement in stability, sample efficiency, or alignment quality attributable to this modification?
   A quantitative comparison or ablation isolating DPGRPO’s contribution would strengthen the paper.

4. **Evaluation of personalization on NavSim:**
   Since NavSim primarily evaluates feasibility and comfort rather than driving style, would a comparison of preference-conditioned v.s. non-conditioned models on NavSim help demonstrate alignment improvements?
   Additionally, can the authors provide results of other state-of-the-art methods on the internal preference datasets for a fairer comparison?

5. **Safety considerations:**
   How do the authors ensure that finetuning toward “aggressive” preferences does not violate safety constraints or produce unsafe behavior?
   Are there empirical checks, such as collision rates or safety rule compliance?

6. **Details on PPO and DPO variants:**
   In Appendix C.1, the paper reports results using PPO and DPO variants. Could the authors provide more implementation details on the PPO setup, specifically, the design of the critic network and reward model within that framework?

7. **Demonstration of behavior:**
   Could the authors provide a video demonstration illustrating the “aggressive” and “defensive” driving behaviors? Static visualizations (e.g., Figure 4) are insufficient to assess feasibility or collision risk.

### Soundness
2

### Presentation
3

### Contribution
3
