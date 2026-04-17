# Revisiting Group Relative Policy Optimization: Insights into On-Policy and Off-Policy Training

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
We revisit Group Relative Policy Optimization (GRPO) in both on-policy and off-policy optimization regimes. Our motivation comes from recent work on off-policy Proximal Policy Optimization (PPO), which improves training stability, sampling efficiency, and memory usage. In addition, a recent analysis of GRPO suggests that estimating the advantage function with off-policy samples could be beneficial. Building on these observations, we adapt GRPO to the off-policy setting. We show that both on-policy and off-policy GRPO objectives yield an improvement in the reward. This result motivates the use of clipped surrogate objectives in the off-policy version of GRPO. We then compare the empirical performance of reinforcement learning with verifiable rewards in post-training using both GRPO variants. Our results show that off-policy GRPO either significantly outperforms or performs on par with its on-policy counterpart.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper revisits Group Relative Policy Optimization (GRPO) and develops an off-policy variant alongside a theoretical analysis of policy improvement in both on- and off-policy regimes. The authors (i) prove lower bounds showing that maximizing a GRPO-style advantage (a whitened reward) improves expected reward, for both on-policy and off-policy cases under bounded rewards and distributional closeness; (ii) derive a clipped surrogate for off-policy GRPO with a KL regularizer; and (iii) empirically compare configurations that differ in sample reuse and serving/update frequency, reporting that off-policy GRPO is at least on par and sometimes better than on-policy GRPO, with potential serving-time savings.

### Strengths
1.The paper extends GRPO analysis to an off-policy setting with an explicit policy-improvement lower bound, and derives an off-policy clipped surrogate analogous to PPO while exploiting GRPO’s normalized reward form.

2.For practitioners training LLMs with GRPO, the paper offers a principled path to reduce serving/communication cost in colocated/tensor-parallel settings without hurting accuracy.

### Weaknesses
1.The primary practical claim hinges on setting v>1, and the paper uses v=10 in its experiments. This value seems arbitrary. The theory only vaguely suggests v should not be "too large". The paper would be much stronger with an ablation study analyzing the trade-off between v, wall-clock speedup, and final task performance. 

2.The improvement bound depends on total-variation terms and a variance factor (1−σ)/σ that explodes when reward variance is near zero (e.g., consistently correct/wrong prompts). The paper suggests masking such cases, but more discussion is needed on how often this occurs in practice for common verifiable rewards and what happens when masking is imperfect.

3.Most results are on math datasets with small models (0.5B / 1.5B); evidence that the method scales or competes with strong recent baselines (e.g., DAPO/DR-GRPO, off-policy PPO variants) is thin.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes and analyzes an off-policy variant of GRPO (a recent policy gradient algorithm for training LLMs).
In particular, the authors identify two sources that render the policy gradient estimator off-policy: the optimization loop, which increases the discrepancy between the behavior policy and the target policy, and the use of past data. 

The authors test the algorithm both by training from scratch and by finetuning an existing model (Deep Seek) on math problems, showing that the off-policy version does not lose performance relative to the on-policy version.

### Strengths
The paper analyzes GRPO, a recent, simple, and efficient algorithm for training LLMs.
The authors, inspired by recent off-policy variants of PPO, propose an Off-Policy variant of GRPO. They also propose a policy improvement bound that covers both off-policy and on-policy development.

### Weaknesses
First, let me write here that I am not familiar with the LLM literature. However, I am very familiar with the reinforcement learning literature and understand GRPO and PPO well. I scored my own assessment of the paper with _confidence=3_ for this reason.

Clarity
--------

The paper's presentation, in my opinion, is quite confusing. Let me state why:

* As far as I understand, the central motivation of the paper revolves around the development of a decentralized update scheme of GRPO. This point remains unclear until page 6. 
* It is not clear to me why the authors use three nested loops for their policy update scheme. In particular, I am confused about a) the outer loop and the need for the "reference policy"; and b) the need to run the optimization (lines 15--17 of algorithm 1) several times even though the policy is not used to collect data. In particular, it appears that every optimization overrides the previous one, without having a real effect on the algorithm.
* The authors call off-policy GRPO, GRPO with a number of optimization steps i>1. While I acknowledge that the gradient estimator becomes off-policy  after one policy update, the overall algorithm, according to the literature, is still considered on-policy (e.g., PPO is still considered on-policy, even if it applies several optimization steps on the same data)
* The experimental section introduces acronyms and names like GSM8K, DeepScaleR-Preview-Datasets, Aime24, "extractive matrching", light-eval, and Pass@1 without explaining what they are. Perhaps they are clear for researchers from the LLM community, but they are not from me, and I needed to google them to understand the results. A bit of explanation in the paper wouldn't have hurt.
* It is not clear why the experiment Off-Policy GRPO uses S=1, while all the other experiments use S=3. 

In addition to the points made, the paper is really math-dense, making it hard to follow. 

Results
----------

Given the paper's scope, it is entirely acceptable for the authors to use only three seeds for their LLM experiments. However, it would have been nice to see downscaled experiments with more seeds to see statistical significance, and hyperparameter sensitivity (i and \nu). Furthermore, I am really suspicious about how the authors used the seeds: In Section 5.1, the authors report that the three seeds give such a small standard deviation (0.003--0.03) that it is not visible in the plots. I have never seen such a little deviation, and I can't understand how it is possible, since the learning curves are pretty noisy. It suggests that the policy always selects the same actions. I can't really figure out how this is possible (unless the policy is basically deterministic, which would make it very hard to train).

Most importantly, it is not clear how the off-policy version does not really outperform the on-policy version. The off-policy version reuses data and should be more sample-efficient. GePPO shows a consistent improvement over "on-policy" PPO.

Summary
-------------

The paper does not seem ready for publication at this stage. The authors should consider writing the paper for a broader audience, clarifying details and results. In my view, the empirical section is also weak, given the low variability across results, the absence of small-scale ablation studies and hyperparameter sensitivity, and the fact that the proposed algorithms do not really improve on the on-policy version.

### Questions
I would ask the authors to clarify all the points above. Most importantly about:

- design choices of the algorithms (outer loop and reference policy); multiple optimization steps even though the policy is not updated
- lack of variability among seeds
- why off-policy GRPO does not outperform the on-policy version
- why the experiment Off-Policy GRPO uses S=1, while all the other experiments use S=3

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper revisits **Group Relative Policy Optimization (GRPO)**, a reinforcement learning algorithm widely adopted for post-training large language models (LLMs) with verifiable rewards.
While prior work (Shao et al., 2024) introduced GRPO as an on-policy variant of PPO using standardized rewards (thus avoiding a critic network), this paper extends it to the **off-policy regime**, theoretically analyzes **reward improvement guarantees**, and empirically validates the new formulation.

The authors:

* Prove **policy improvement lower bounds** for both on-policy and off-policy GRPO (Theorem 1, Corollary 1).
* Derive a **clipped surrogate objective** analogous to PPO but specialized for GRPO’s whitened rewards.
* Provide a **masking strategy** for zero-variance samples (grounded theoretically).
* Demonstrate experimentally that **off-policy GRPO** matches or exceeds the on-policy variant while reducing **communication costs** in distributed LLM training setups.
  Experiments cover **GSM8K**, **DeepScaleR**, and scaling tests on **Qwen2.5-7B** models.

### Strengths
1. **Theoretical Rigor and Novelty**

   * Provides the **first formal policy improvement bound for GRPO**, including both on- and off-policy cases.
   * The proofs elegantly extend beyond standard MDP-based analyses, avoiding dependence on state visitation distributions and instead exploiting GRPO’s analytical advantage form.
   * The derivation of **clipped surrogate objectives** for off-policy GRPO generalizes the PPO framework in a principled way.

2. **Practical Impact for LLM Training**

   * Introduces a simple off-policy configuration (controlled by parameters *v* and *i*) that directly reduces communication overhead when serving models via vLLM.
   * Empirically confirms the **stability and efficiency** of this approach, a highly relevant concern for multi-GPU and TP-based deployments.

3. **Bridges Gaps Between RL Theory and LLM Post-Training**

   * Clarifies the relationship between GRPO, PPO, DAPO, and off-policy PPO variants (G-PPO, ToPPO) within a unified analytical framework.
   * Provides theoretical justification for “zero-variance masking” (previously used heuristically in DAPO) by linking it to the reward improvement bound.

4. **Clear Experimental Verification**

   * Ablation studies on GSM8K (Pass@1) demonstrate that both **masking** and **off-policy training** improve stability and accuracy.
   * Finetuning experiments on DeepSeek-R1-Distill-Qwen 1.5B confirm the off-policy variant’s ability to maintain or slightly improve performance on **AIME24** and **Math500** benchmarks.
   * Scaling experiments with **Qwen 2.5 7B** demonstrate ~1.35× speedup without loss in performance.

5. **Reproducibility and Transparency**

   * The codebase is based on public TRL and Open-R1 repositories, ensuring reproducibility.
   * Detailed setup and hardware configurations are included in the appendix.

### Weaknesses
1. **Limited Experimental Diversity**

   * Experiments focus primarily on math datasets (GSM8K, DeepScaleR). Broader testing on reasoning, code generation, or instruction-following datasets would strengthen generality.
   * The number of random seeds (three) is relatively small, and there’s little reporting of variance beyond Pass@1 standard deviation.

2. **Theory–Practice Gap**

   * While the proofs are strong, some constants in the policy improvement bound (e.g., the variance-dependent term (1-σ_{α,r,ε}/σ_{α,r,ε})) lack empirical interpretation or sensitivity analysis.
   * More intuition or visualization for these terms would help practitioners understand stability conditions.

3. **Presentation and Readability**

   * Dense notation (π, α, v, µ, ε, β, σ, Δ) makes the theoretical sections heavy.
   * Some equations (especially Theorem 1 and derivations in Appendix B) could benefit from concise summaries or diagrams.
   * Figures could include clearer captions and consistent scaling.

4. **Minor Empirical Limitations**

   * The speedup results (1.35×) are modest; while communication cost reductions are meaningful, scaling analysis beyond a single node would be more convincing.
   * Comparison to other RLHF or off-policy RL methods (e.g., OPPO, ToPPO) is missing in quantitative form.

### Questions
1. How does the choice of *v* (off-policy lag) affect both convergence and stability? Are there theoretical or empirical bounds on how large *v* can be before degradation occurs?
2. Could the policy improvement bounds be extended to **partially observable or multi-step MDPs** (where GRPO is applied iteratively on dialogue tasks)?
3. Does masking zero-variance samples introduce bias in reward estimation over time?
4. How does off-policy GRPO behave when the reward distribution is highly skewed or continuous rather than Bernoulli?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper revisits GRPO and develops a unified view of on-policy and off-policy variants by deriving a lower bound on reward improvement. From a KL-regularized objective, the authors obtain a clipped surrogate objective for GRPO and provide a theoretical rationale for masking zero-variance samples, which would otherwise worsen the bound constants. Building on this, they propose off-policy GRPO with two practical knobs: $v$ (server/model update frequency) and $i$ (SGD steps per batch). This configuration aims to reduce communication overhead in multi-GPU/tensor-parallel setups without sacrificing accuracy. Empirically, the paper reports that on-policy can be unstable on GSM8K, masking stabilizes training, and off-policy (e.g., $v=10, i=1$) attains similar or slightly better accuracy. On larger models (7B with TP), the off-policy setting yields a reported $\sim$1.35× speedup per iteration.

### Strengths
- An explicit derivation of a reward-improvement lower bound for GRPO (on/off-policy) and a principled route from KL-constraint to a clipped surrogate objective.  
- The $(i,v)$ design is simple and addresses a real bottleneck (communication/model updates) in multi-GPU/TP training.  
- The analysis explains why zero-variance reward samples degrade constants and how masking can stabilize training in verifiable-reward tasks.  
- Results on reasoning benchmarks show that the off-policy variant is competitive with the on-policy baseline while improving throughput.

### Weaknesses
- Most comparisons are within the GRPO family (on vs off; with/without masking). The paper should include controlled head-to-head comparisons with **DAPO / DR-GRPO / OPPO / ToPPO / DPO / IPO** under matched rewarders, sampling budgets, and compute. And the 7B setting reports system speed but lacks full accuracy curves and significance tests. Sensitivity to temperature, context length, $\beta$, group size $G$, and $(i,v)$ needs more systematic coverage.  
-  Evidence is concentrated on verifiable rewards. It remains unclear how the bound and masking behave with non-binary or hard-to-verify rewards.  
-  The paper states that $v$ should not be too large, but lacks an empirical guideline (e.g., KL/TV drift vs. $v$ curves) or a recommended operating range.
- Minor writing/formatting fixes for camera-ready: unify the epsilon symbol (use ε or define $\epsilon$ and use it consistently), normalize dataset names across text/tables/figures (e.g., AIME24/Math500, no spaces around the slash), standardize the tool name to a single spelling (e.g., lighteval), correct a few misspellings/grammar (libraries, standardized, laid; learning rate $5×10^6$), and tighten math displays so TV is typeset as $TV^{2}$ and long fractions/clipping expressions aren’t broken across lines.

### Questions
1. Can the authors add **DAPO/DR-GRPO/OPPO/ToPPO/DPO/IPO** results with identical rewarders, sampling temperature, context length, and compute budgets (ideally including a 7B+ model)?  
2. Please report KL/TV drift curves across training for different $v$ and provide a recommended range for stable/efficient operation.  
3. How is the zero-variance masking threshold chosen in practice? Can the authors provide a sample utilization vs. accuracy/variance?  
4. Do the conclusions hold for code generation, safety alignment, or open-ended QA with graded or noisy reward signals?

### Soundness
3

### Presentation
3

### Contribution
2
