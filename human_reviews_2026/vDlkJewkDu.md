# Escaping Policy Contraction: Contraction-Aware PPO (CaPPO) for Stable Language Model Fine-Tuning

- Decision: Accept (Poster)
- Scores: 6, 10, 4

## Abstract
Reinforcement learning from human feedback (RLHF) with proximal policy optimization (PPO) is widely used but often yields less diverse outputs than supervised fine-tuning, suggesting an effect in which the policy’s support contracts during on-policy optimization. We formalize this “policy contraction” with the Support Retention Ratio (SRR)—the share of SFT completions that retain non-negligible probability under the RL policy—and additionally track token-entropy, Kullback–Leibler (KL) divergence to the reference, and repetition. We propose Contraction-Aware PPO (CaPPO), a minimum-norm multi-gradient update that co-optimizes reward, entropy, and KL, paired with a controller that steers exploration toward a target token entropy. On HH-RLHF, Summarize-from-Feedback, and UltraFeedback with Qwen2-7B, Qwen2.5-14B, Mistral-7B-Instruct, and Llama-3-8B-Instruct, CaPPO increases win rate by 2 to 4 points over PPO and improves diversity, gaining 0.2 to 0.3 higher SRR. The gains persist under decoding sweeps and are robust to reward scaling and critic variance. Treating reward, diversity, and stability as first-class objectives, CaPPO mitigates contraction without sacrificing alignment performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents Contraction-Aware PPO (CaPPO), a practical improvement to PPO for RLHF. The authors point out that standard PPO tends to collapse a model’s output diversity as training progresses—they call it "policy contraction". CaPPO solves this by balancing reward, entropy, and KL-regularization as equal objectives, using a simple multi-gradient update instead of manual weighting. It also adds an adaptive entropy controller that keeps exploration at a healthy level. The idea is easy to follow, the motivation makes sense, and the experiments show clear benefits in both stability and diversity without hurting alignment quality.

### Strengths
1. The paper does a great job of turning the vague observation that PPO reduces diversity into a formal, testable concept. The Support Retention Ratio feels like something people will actually adopt for diagnosing policy collapse.

2. The multi-objective reformulation of PPO is both theoretically justified and surprisingly lightweight — it’s clear the authors thought about practical integration rather than proposing a totally new framework.

3. The entropy controller is well-motivated by the observed entropy trajectories; it’s an elegant way to stabilize exploration without resorting to arbitrary tuning.

### Weaknesses
1. The paper admits that SRR depends on a chosen threshold but never tests how sensitive results are to that choice. A short analysis or plot of SRR versus threshold would make the metric feel more reliable.

2. The core multi-gradient idea is interesting, but the paper doesn’t show how the mixing weights change or whether gradients conflict during training. Some visualization or discussion would make the method easier to understand.

3. The claim that contraction is unique to PPO or RL methods isn’t fully demonstrated. Off-policy baselines like DPO and KTO are only compared at the endpoint, not throughout training. Showing their entropy or SRR over time would make the argument more solid.

4. The paper calls CaPPO’s overhead “a small constant factor” but provides no runtime or memory measurements. Since the method adds multiple gradients, a quick comparison of compute cost with PPO would be important for judging practicality.

5. It’s not clear what higher SRR actually means in practice beyond diversity metrics. The paper assumes preserving SFT support is good but doesn’t show that it improves model helpfulness, safety, or calibration.

6. The contraction analysis feels surface-level. Figure 1 shows entropy dropping, but there’s no look into *when* or *why* that happens, or which prompts are most affected. More insight into these dynamics would strengthen the explanation. (Maybe a figure to show this?)

### Questions
Please read the weaknesses section and answer the questions. Thank you!

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper tackles the problem of diversity collapse due to RL training algorithms in LLMs. Authors call this phenomenon *policy contraction*, implying the distribution of responses shrinks after RL training towards fewer high rewarding sequences. The authors systematically analyze the effects of different RL training methods by measuring policy contraction using measures like Self-BLEU, Distinct-2 (distinct bigrams) and Support Retention Ratio (SRR). SRR is a newly introduced metric which measures the fraction of SFT model competions whose length-normalized log probabilty under the RL tuned policy is greater than some fixed threshold. Using all three measures, authors find that preference-based off-policy methods retain SRR levels of SFT, however take a hit in Self-BLEU and Distinct-2 metrics. On the other hand on-policy policy gradient methods like PPO, Vine-PPO, PPO + entropy, etc. deteriorate in all three diversity measuring metrics.

To mitigate this policy contraction, they propose Contraction-Aware PPO (CaPPO) which implements reward maximization, entorpy and KL regularization as multi-objectives instead of additive. At the core, their method maintains 3 gradient vectors, one for each objective, and calculates a minimum-norm convex combination of gradients. They first precondition the gradients (to account for different scales) via dialgonal metric (second moment of Adam optimizer) and then solve the quadratic problem in this preconditioned space. Then the multiplers for this optimization method are found via projected-gradient or Frank-Wolfe steps. 

For more robust comparison with PPO, they also introduced an adaptive entropy scheduling which tries to maintain the exponential moving average entropy of the policy close to the target entropy.

In the experiments they compare CaPPO against a variety of other basline preference based methods and PPO counterparts, including PPO enhanced w/ their adaptive entropy sechduler. Overall, they found their method consistently matches or execeeds the performance of the baseline methods while retaining relatively higher diversity and SRR compared to PPO counterparts.

### Strengths
- Very clear motivation and empirical grounding to justify the problem
- Interesting and theoretically sound multi-objective solution to the problem of policy contraction in RL training
- Detailed experimental analysis, baseline comparison and ablation studies to demonstrate the efficacy of their method over the other RL baselines.

### Weaknesses
- I would have preferred to see a detailed comparison of the throughput hit when using a more complex optimization method such as CaPPO compared to PPO. How much more memory and time overhead does it incur? Will the additional overhead become a blocker for large LMs in long sequences tasks?

### Questions
- Interestingly, in Table 4, GRPO shows much less degradation in diversity metrics compared to PPO. Do the authors have any intuition for this obersvation? Does the group sampling in-turn enhances the sampling diversity of the RL training? 
  - I am also curious, how would GRPO mixed with entropy or entropy scheduling behave in comparison to CaPPO.
- Typo in the intro: KTO refers to "Kahneman-Tversky Optimization"

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
PPO-based RLHF often narrows the model’s output distribution (“policy contraction”): entropy drops, repetition rises, and many SFT-feasible completions get near-zero probability.
To mitigate this, the authors propose Contraction-Aware PPO (CaPPO), which:
1. Treats reward, entropy, and KL-to-reference as peer objectives in a multi-objective optimization framework.
2. Uses a minimum-norm multi-gradient update to ensure Pareto-improving steps without brittle scalarization.
In addition, the authors introduce Support Retention Ratio (SRR) to quantify how much of the SFT support survives RL training, complementing entropy and KL.

For experiments, the authors compare off-policy methods (DPO/IPO/ORPO/KTO/RRHF) and on-policy baselines (PPO/VinePPO/GRPO) on HH-RLHF, Summarize-from-Feedback, and UltraFeedback with Qwen2-7B/2.5-14B, Mistral-7B-Instruct, and Llama-3-8B-Instruct.
CaPPO lifts win rate by ~2–4 points over PPO and raises SRR by ~0.20–0.30, while improving Distinct-2 and lowering Self-BLEU.

### Strengths
- CaPPO outperforms other on-policy algorithms in both diversity and performance, as measured by win rate.
- SRR is a reasonable metric for measuring diversity.

### Weaknesses
- How to evaluate win-rate is not clear.
- How to implement CaPPO in practice & its wall-clock time is not clear.

### Questions
- While the paper reports win rate as the primary performance metric, it is unclear how a “win” is determined in the experiments. Which reward model is used to decide the winner? If GPT-4 is used, which prompt is employed?
- How to evaluate $\lambda$ values in line 5 of Algorithm 1?
- Intuitively, it isn’t obvious that CaPPO should outperform other on-policy methods while also improving diversity (in Table 4), since one might expect a trade-off between performance and diversity. Could you provide an intuitive explanation for this result?
- It is unclear how the entropy-scheduling controller is applied in CaPPO, as described in lines 79–80, because Algorithm 1 does not include a $\beta$ term.
- As I understand it, CaPPO employs Equation (3) as the main training objective. Then, could you clarify why you chose the multi-objective update in (3) over a simpler linear scalarization (i.e., PPO + $\alpha$·KL + $\beta$·Entropy) with scheduled coefficients $\alpha$ and $\beta$ as in §4.3?
  1. Does (3) reduce sensitivity to $\alpha$, $\beta$ tuning and reward scale or provide more reliable Pareto behavior?
  2. Without explicit $\epsilon$-bounds as in (4), how do you ensure KL and entropy remain within desired ranges (e.g., guards, line search, or dual-style updates)?
  3. What are the wall-clock and compute trade-offs compared to scheduled scalarization?
  4. Could you include ablations that sweep  $\alpha$, $\beta$ schedules to match similar KL/H targets and report win-rate, SRR, and stability differences?

### Soundness
2

### Presentation
3

### Contribution
2
