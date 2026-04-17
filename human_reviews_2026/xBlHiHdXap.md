# Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 2

## Abstract
Reinforcement learning (RL) has become central to enhancing reasoning in large language models (LLMs). Yet on-policy algorithms such as Group Relative Policy Optimization (GRPO) often suffer in early training: noisy gradients from low-quality rollouts lead to unstable updates and inefficient exploration. We introduce Slow-Fast Policy Optimization (SFPO), a simple yet efficient mechanism to address the above limitations via decomposing each iteration into three stages: a short fast trajectory of inner steps on the same batch, a reposition step to control off-policy drift, and a final slow correction. This reposition-before-update design preserves the objective and rollout process unchanged, making SFPO plug-compatible with existing policy-gradient pipelines. Extensive experiments demonstrate that SFPO consistently improves stability, reduces number of rollouts, and accelerates convergence of reasoning RL training. Specifically, it outperforms GRPO by up to 2.80 points on math reasoning benchmarks. It also achieves up to 4.93$\times$ fewer rollouts and a 4.19$\times$ reduction in wall-clock time to match GRPO’s best accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Slow-Fast Policy Optimization (SFPO), a plug-and-play method to stablize and accelerate the learning of on-policy RL algorithms such as GRPO. Specifically, SFPO has three stages for each training step: Fast Trajectory, Reposition, and Slow Correction, which mitigate the noisy single-step rollout updates and ensures a more stable optimization process. Empirical studies in mathmatical benchmarks demonstrates the consistent improvement of SFPO compared with GRPO.

### Strengths
1. The method is well-structured and thoughtfully designed. The Fast Trajectory stage stabilizes the optimization process, while the Reposition and Slow Correction stages help to better align the optimization with the on-policy objective. The motivation behind each stage is clearly articulated, making the overall approach sound and convincing.

2. The empirical results are thorough and well-presented. The paper evaluates SFPO across 5 models, 3 methods, and 6 benchmarks, clearly demonstrating substantial improvements in both efficacy and efficiency. Furthermore, comprehensive analyses and ablation studies are provided, strengthening the validity of SFPO’s performance claims.

3. The paper is well-organized, featuring clear and informative figures, tables, and algorithm illustrations. Additionally, Section 3 provides intuitive explanations that greatly aid in understanding the underlying mechanism of the proposed SFPO.

### Weaknesses
1. The empirical evaluation is somewhat limited in scope, as all experiments are conducted only on mathematical benchmarks, without including domains such as code or logic reasoning. Additionally, GRPO is the sole baseline used; incorporating additional baselines like DAPO and GSPO would further strengthen the evidence for SFPO’s effectiveness.

2. The ablation studies indicate that SFPO is sensitive to the hyperparameters $\alpha$ and $K$, which may hinder its direct applicability to domains beyond mathematical reasoning. Users may need to carefully tune these hyperparameters to achieve optimal performance.

### Questions
1. What does "Slow" refer to in stage 3? Does this stage require significantly more time to update the gradient?

2. Could you elaborate on the time consumption in Stage 1? Since gradient updates are computationally intensive, this stage might become slower as K increases. Additionally, in agentic scenarios involving multi-turn interactive trajectories, performing multiple policy gradient updates may significantly reduce optimization efficiency.

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
5

### Summary
This paper introduces Slow-Fast Policy Optimization (SFPO), a three-stage update mechanism designed to improve the stability and sample efficiency of on-policy reinforcement learning for LLM reasoning. SFPO replaces the standard single gradient step with a fast trajectory of multiple inner updates, a reposition step to control off-policy drift, and a final slow correction step. Extensive experiments demonstrate that SFPO significantly outperforms the GRPO baseline in accuracy, convergence speed, and wall-clock time on various math reasoning benchmarks.

### Strengths
1.  The proposed method addresses a significant and well-recognized problem: the instability and sample inefficiency of on-policy algorithms like GRPO in the context of LLM reasoning. The proposed three-stage solution is simple, intuitive, and clearly motivated as a way to reduce update variance and make better use of collected data.
2.  A major strength of SFPO is its "plug-and-play" compatibility with existing policy gradient pipelines. By modifying only the parameter update rule while leaving the loss function, rollout generation, and regularization terms of GRPO intact, the method offers a practical, low-effort path to improving existing training setups.
3.  The paper is supported by a comprehensive set of experiments that demonstrate consistent and substantial gains over a strong GRPO baseline across multiple models and benchmarks. The reported improvements in sample efficiency (up to $4.93\times$ fewer rollouts) and wall-clock time (up to $4.19\times $ reduction) are highly significant. Furthermore, the ablation studies effectively justify the inclusion of each component of SFPO, particularly the adaptive entropy control and the slow correction stage.

### Weaknesses
1.  The reporting of experimental results contains significant numerical errors and omissions. In Section 4.2.1, the claimed performance gains for Qwen2.5-Math-7B (+1.80) and DS-distilled-Qwen-7B (+0.8) do not match the values in Table 1, which show gains of +0.83 and +2.57, respectively. 
2.  The reference list is not properly curated. For example, the citations "Yu et al., 2025a" and "Yu et al., 2025b" both point to the exact same arXiv preprint, which is misleading.

### Questions
1.  Could you clarify the nomenclature for Stage I, the 'Fast Trajectory'? The term 'fast' is somewhat ambiguous. Does it refer to the speed of convergence, the magnitude of the parameter change, or some other property?
2.  A crucial baseline seems to be missing from the ablation studies. To isolate the benefits of Stage II (Reposition) and Stage III (Slow Correction), have you considered a baseline that only uses Stage I? This would be equivalent to applying GRPO with $K$ inner updates on the same batch, which would more clearly demonstrate the contribution of the reposition and slow correction steps.
3.  Stage II is designed to control off-policy drift via interpolation. A more common approach in policy gradient methods is to use a KL divergence penalty to constrain the policy update. Could you comment on the design choice of interpolation over a KL penalty? Have you compared the performance of your reposition step against a KL-constrained update in this multi-step update setting?
4.  The paper claims that SFPO addresses the high-variance gradient issue in GRPO. A significant source of variance in GRPO can come from batches with uniform rewards (e.g., all successes or all failures), which result in zero advantage for all samples. Does SFPO's performance gain stem primarily from better handling these 'zero-advantage' batches? To verify this, it might be necessary to conduct experiments with an algorithm like DAPO, which is explicitly designed to use *dynamic sampling* to mitigate this issue, to see if SFPO provides similar benefits in a setting where this specific source of variance is already addressed.
5.  Could you provide a derivation or at least a reference for the descent guarantee in Equation (10)? A more formal explanation of the assumptions and the term $F(K, \alpha) $ would strengthen the paper's theoretical grounding.
6.  Would you be able to correct the numerical errors in the text describing the results in Table 1 (lines 322-323)? The discrepancies between the reported gains and the values in the table are substantial.
7. The interpretation of lower entropy in SFPO as "more efficient exploration" (lines 374-375) is intriguing. Could you elaborate on this? Is it possible that the method simply encourages faster convergence to a more deterministic policy, which might be beneficial for exploitation but could also be interpreted as a sign of reduced exploration?

I look forward to the author's response. I am willing to reconsider my score if the questions above, particularly those concerning the experimental details, numerical correctness, and key design choices, are adequately addressed.

### Soundness
3

### Presentation
3

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
This paper argues that the direction of token-level updates, measured by the signed log-probability difference between base and RLVR models, is more informative than magnitude-based metrics for understanding RLVR's effect on LLM reasoning. The authors propose and validate two methods, test-time extrapolation and training-time advantage reweighting, that exploit the directional insight to improve reasoning performance.

### Strengths
The paper introduces a novel and intuitive directional metric that effectively captures sparse, reasoning-critical updates, supported by rigorous token-replacement experiments and gradient analysis.

 The proposed methods are simple yet effective, demonstrating consistent gains across multiple models and benchmarks without requiring additional training data.

### Weaknesses
The test-time extrapolation method requires access to both the base and RLVR models, which may limit its practicality in settings where only the fine-tuned model is available.

    The paper focuses primarily on mathematical reasoning benchmarks (e.g., AIME, AMC); it remains unclear whether the findings generalize to other reasoning domains or more diverse tasks.

    The theoretical justification (Theorem 4.1) relies on a simplified tabular softmax bandit setting, which may not fully reflect the complexity of modern LLM training dynamics.

### Questions
How does the proposed direction-based extrapolation perform in non-mathematical reasoning tasks?

The token-replacement experiment convincingly shows that \Delta\log p identifies critical tokens. However, does this intervention sometimes harm performance? Are there cases were replacing a base model token with the RLVR model's choice leads to a wrong answer, and what characterizes those tokens?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Slow–Fast Policy Optimization (SFPO), it's a three-stage modification to on-policy GRPO-style training for reasoning LLMs: (i) a fast trajectory of $K$ inner gradient steps on the same rollout batch, (ii) a reposition interpolation back toward the start point controlled by $\alpha$, and (iii) a slow correction extra step. See Alg. 1 (p. 3). Authors also introduce an entropy-triggered schedule that sets $\alpha \leftarrow 0$ after a $z$-score threshold on policy entropy (Eq. 11), which effectively reverts to GRPO near convergence.

### Strengths
- Simple, drop-in recipe that practitioners can try with minimal code churn (Alg. 1).
- Large rollout/time reductions to reach GRPO’s best accuracy (Fig. 4), aligning with the observation that rollouts dominate wall-clock.

### Weaknesses
1. GRPO-K: $K$ inner updates on the same batch without reposition or slow correction (a direct control for “just do more steps per batch”).  

2. Lookahead-GRPO applies only the Stage II interpolation around GRPO.  

3. Extra-grad-GRPO applies only the Stage III predictor–corrector step.  

4. Clarify whether per-iteration compute (backprop) increases due to $K$; if so, compare at matched FLOPs (or matched wall-clock) and at fixed accuracy targets, not “GRPO’s best accuracy”.

5. The theoretical section is insufficient. Make Eq. (10) precise (define $c$ and $F(K,\alpha)$; assumptions on clipping/KL; conditions handling negative curvature), or clearly mark it as heuristic.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
