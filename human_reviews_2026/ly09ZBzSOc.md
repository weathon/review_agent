# ADORA: Training Reasoning Models with Dynamic Advantage Estimation on Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Reinforcement learning has become a cornerstone technique for developing reasoning models in complex tasks, ranging from mathematical problem-solving to imaginary reasoning. The optimization of these models typically relies on policy gradient methods, whose efficacy hinges on the accurate estimation of an advantage function. However, prevailing methods typically employ static advantage estimation, a practice that leads to inefficient credit assignment by neglecting the dynamic utility of training samples over time. This limitation results in suboptimal policy updates, which in turn manifest as slower convergence rates and increased learning instability, as models fail to adapt to evolving sample utilities effectively. To address this problem, we introduce ADORA (Advantage Dynamics via Online Rollout Adaptation), a novel framework for policy optimization. ADORA dynamically adjusts the advantage function's weighting by adaptively categorizing training data into temporarily advantageous and disadvantageous samples, based on their evolving utility during online model rollouts. This tailored data differentiation strategy allows ADORA to be seamlessly integrated into existing policy optimization algorithms without significant architectural modifications, enabling the policy to prioritize learning from more informative experiences and thereby achieve more efficient policy updates. Extensive evaluations on various tasks demonstrate that ADORA significantly enhances long reasoning in both geometric and mathematical tasks across large vision–language models and large language models, achieving notable performance gains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes ADORA, a reinforcement learning framework that dynamically updates the advantage estimation weights during training. 
The basic idea is to categorize samples into temporarily advantageous and temporarily disadvantageous based on their evolving utility, and ADORA enables more efficient policy optimization. Experiments across both large language models and vision–language models shows consistent gains over baselines like GRPO and DAPO, especially in reasoning-intensive tasks such as mathematical and geometric problem solving.

### Strengths
-  Feasible approach to dynamic weighting: The paper introduces a conceptually clean yet empirical effective way to dynamically calibrate the advantage function during reinforcement learning. It targets solving a key problem with static estimations.
-  Strong empirical validation: The experiments are relative extensive, covering both LLMs and VLMs. It shows clear and consistent performance improvements even with limited data. This gives confidence in the robustness of ADORA’s approach. However, in terms of ablation, there remain several concerns, which will be detailed next. (See Weakness)

### Weaknesses
- Generalization not deeply explored: Although ADORA performs well on tested benchmarks, the discussion on transferability and performance on out-of-distribution or unseen domains feels somewhat limited. Also, the weighting is based on heuristic rule and there lacks some theoretical insights.
- Dependence on rollout quality: The authors themselves note that ADORA’s success depends on the quality of generated rollouts (Appendix D). However, the paper does not clearly propose methods to mitigate low-quality samples or noise in the rollout process.
- Missing some prompt difficulty related works: There are related works that can be discussed in the manuscript. For example, MoPPS [1] actively infers the prompt difficulty and selects the subset to improve both efficiency and performance. SPO does a similar thing from policy optimization perspective.
- Besides the above, I have raised some questions below, which should be well addressed in either added discussions or experiments.

References: 

[1] Qu, Yun, et al. "Can prompt difficulty be online predicted for accelerating rl finetuning of reasoning models?." arXiv preprint arXiv:2507.04632 (2025).

[2] Xu, Zhongwen, and Zihan Ding. "Single-stream policy optimization." arXiv preprint arXiv:2509.13232 (2025).

### Questions
- How sensitive is ADORA to the hyperparameters like τ and ws? Would small changes significantly affect convergence?
- How does ADORA behave with other family of base models such as Qwen and other sizes such as 1.5B base models?
- It seems reweighting mechanism adjusts the learning rate, so is the VLA result’s advantage from the learning rate adjustment.

Other suggestions: 

- (1) It would be better to include an illustration Figure to show the detailed reweighting implementation steps in rollout and policy optimization. 
- (2) Learning curves for all results should be reported if necessary.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces ADORA (Advantage Dynamics via Online Rollout Adaptation), a framework to improve the sample efficiency of policy gradient methods for training VLMs and LLMs. It works by dynamically calibrate advantage estimation based on two heuristics that assess the model's current capacity for the given samples. The paper demonstrates the proposed framework in mathematical and geometric tasks improving over the baseline on several benchmarks for the Qwen2.5-VL-7B and Qwen2.5-7B

### Strengths
The paper investigate the important topic of sample efficiency for LLMs for hard reasoning tasks.

The proposed heuristics show promising results to improve the sample efficiency of GRPO without any additiona significant computational cost.

The benchmark results are supported with more qualitative analysis.

### Weaknesses
The method introduces several key hyperparameters ($\tau=0.5$, $w_s=0.1$, $w_s=2.0$) that are presented without any justification or sensitivity analysis. These values are likely to heavily influence performance and would almost certainly require re-tuning for new models or tasks thus undermining the paper's central claim of being a general and lightweight approach. This weakness is compounded by the fact that all experiments are limited to a single model family (Qwen).

The heuristics are calculated using a very small number of rollouts ($G=5$ for LLMs, $G=8$ for VLMs). Basing the TAS/TDS classification on such a small sample size means the signal is statistically noisy and high-variance. A sample's classification could easily "flicker" between advantageous and disadvantageous from one step to the next due to simple sampling luck, not a true change in the model's capability.

The length advantage criterion actively punishes correct, concise reasoning and creates a strong incentive for verbosity. While this may not have compromised performance on the chosen benchmarks, it raises serious questions about the method's generalizability. The paper attempts to investigate this overthinking issue in Section 5.3, but its analysis relies on an "Overthinking Score" that is never properly introduced, making the results in Table 3 difficult to interpret.


The "Difficulty Advantage" heuristic ($R_{succ}^s \le 0.5$) [cite: 340] uses a sharp, binary threshold that encourages proficiency rather than mastery. As soon as the model achieves >50% success on a sample, its learning signal is halved (from $w_s=2.0$ to $w_s=1.0$). This may prematurely de-prioritize the sample and *prevent* the model from achieving true robustness (e.g., 90-100% success). Furthermore, this rule conflates "instructive" ($R^s=50\%$) with "impossibly hard" ($R^s=0\%$), treating them both as equally advantageous.

### Questions
The paper's claim of a "general" framework is a significant one, but it's evaluated exclusively on the Qwen model family. Can you provide any results on ADORA's performance when applied to other model families (e.g., Llama, Mistral)? Relatedly, how were the key hyperparameters ($\tau=0.5$, $w_s=0.1$, $w_s=2.0$) selected? A sensitivity analysis would be critical to understand how much these "magic numbers" must be re-tuned for new models.

Regarding the heuristics themselves, the TAS/TDS classification relies on a very small number of rollouts ($G=5$ or $G=8$). How do you ensure this signal is statistically stable and not just high-variance noise? A sample's classification could flicker from step to step based on sampling luck alone.

The reflection frequency analysis in Section 5.2 is also a concern. Since ADORA produces longer responses (Figure 7), how did you disentangle *true reflection* from *mere verbosity*? A more verbose model will naturally use more reflective words. Could you provide a length-normalized analysis (e.g., "reflection keywords per 100 tokens")? On that topic, your investigation in Section 5.3 relies on an "Overthinking Score" that is never introduced. Could you please provide a clear definition of this score and how it is calculated?

Finally, on the heuristic design: Why was a sharp $\tau=0.5$ threshold chosen for the "Difficulty Advantage"? This rule seems to punish mastery by de-prioritizing samples once proficiency exceeds 50%. Have you considered a "softer" weighting? And critically, why does the VLM strategy only use the "Length Advantage"? We are missing the ablation study, similar to Figure 2, that justifies this specific design choice for VLMs.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper argues that static advantage estimation, as traditionally used, leads to inefficient credit assignment due to ignoring the dynamic utility of training samples over time. To address this, they propose ADORA (Advantage Dynamics via Online Rollout Adaptation) that dynamically tune the importance of the advantage function based on the utility of the samples. They perform considerable number of experiments with LLMs and VLMs.

### Strengths
### Strengths:

1. ADORA categorizes the training data into temporarily advantageous and disadvantageous samples and adaptively assigns sample-wise weights based on predefined criteria to estimate the ultimate advantage.

2. Validation and ablations are conducted across different domains and datasets, especially on both LLMs and VLMs. Further, ADORA achieves a consistent performance gain to an extent.

### Weaknesses
Weaknesses:

1. The paper only consider length and success rate to measure the utility. It does not consider more complex evaluations such as step consistency. I believe there is opportunity to define the criteria more comprehensively. 

2. The authors mention "How to assign a corresponding weight $w_s$ that reflects its training utility?". However, I don't see an appropriate answer to this question. The rationale behind choosing the specific values is not discussed. I would consider them as hyperparameters, and the sensitivity to those hyperparameters are yet to explore.

3. Same concern goes for $\tau$ in Eqn. 6 which is set to 0.5. How the model reacts with the changes to $\tau$?

### Questions
### Questions:

1. In Eqn. 5, why do you need the **longest** successful rollout? Shouldn't every successful rollout which has a length $> L_{fail}$ be considered?

2. The paper mentions that a direct indicator of explicit reasoning is the frequency of reflective vocabulary usage. How you come up with that vocabulary? Can you provide any reference in favor of this?

3. Can you simply describe the interpretation of Figure 4.? 

Also, look at the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ADORA (Advantage Dynamics via Online Rollout Adaptation), a framework that dynamically reweights advantages in reinforcement learning for reasoning models. 

The core idea is to classify training samples into Temporarily Advantageous Samples (TAS) and Temporarily Disadvantageous Samples (TDS) based on rollout statistics, then amplify or attenuate their learning signals accordingly. The method uses Length Advantage and Difficulty Advantage as criteria. 

Experiments on Qwen2.5-VL and Qwen2.5 show improvements over GRPO baseline across mathematical and geometric reasoning tasks.

### Strengths
Originality: The dynamic reweighting of advantages based on online rollout statistics is a practical contribution. 

Quality: Strong experimental validation across both VLMs (MathVista: 73.5%) and LLMs (3.5% average improvement over GRPO).

Clarity: Well-structured presentation with clear motivation. And figures effectively illustrate training dynamics and sample evolution.

Significance: Addresses a real problem in RL-based reasoning training.

### Weaknesses
1.  Limited experiment scope:
* Only tested on Qwen family models. Generalization to other model family (Gemma, Llama, Phi, etc.) is unknown.
* VLM experiments use only 2K samples—unclear if benefits persist at larger scales.

2. Incomplete ablations:
* No ablation on weight values (see Question#1 and #2).
* Figure 2 shows ablations on advantage criteria but only for LLMs—missing VLM ablations.
* The threshold τ=0.5 for difficulty appears arbitrary—no ablation on this critical hyperparameter.
* No formal analysis of why ws=0.1 (VLM) or ws=2.0 (LLM) are optimal choices.

3. Questionable claims:
* "No cold-start" (Table 6) is misleading—they start from Qwen2.5-VL-7B-Instruct, which is already instruction-tuned, while Vision-R1 starts from base model.
* Section 5.3: Lower overthinking on AIME24 (40.1 vs 44.8) is attributed to ADORA, but ADORA also achieves higher accuracy—is this confounded?
* Length Advantage conflates token count with reasoning quality:
```
Equation 5 assumes longer responses indicate deeper reasoning, but consider:

Solution A: 15×4 = 15×2×2 = 60 (20 tokens, uses factorization)
Solution B: 15+15=30, 30+15=45, 45+15=60, verify: 60÷4=15✓ (100 tokens, brute-force)

ADORA prefers B over A despite A showing better insight. The paper provides no evidence that:

- Short correct solutions (<50 tokens) only solve trivial problems
- Length correlates with reasoning depth rather than redundant verification
```
4. Statistical rigor:
* While 3 runs are reported, no significance tests or confidence intervals are provided.
* Table 1 shows ADORA (73.5%) matches Vision-R1 (73.5%) on MathVista—is this difference statistically significant vs GRPO (70.2%)?



### Minor Issues

* Equation 4: ws is sample-level but notation doesn't clearly distinguish from trajectory-level weights.
* Figure 4 visualization is cluttered—consider simplifying or providing clearer legends.

### Questions
1. Threshold sensitivity: How sensitive is performance to τ? Have you tried τ ∈ {0.3, 0.5, 0.7}? What happens at extremes (τ=0.1 or τ=0.9)?
2. Weight justification: Why ws=2 for LLM amplification and ws=0.1 for VLM attenuation? Have you ablated ws ∈ {0.05, 0.1, 0.2} and ws ∈ {1.5, 2, 3}?
3. Can you analyze solution length vs. quality (weakness#3)? Please provide evidence that Length Advantage captures reasoning depth rather than just token count.
4. Cold-start claim: Table 6 compares against Vision-R1 with 200K cold-start data, but your base model is already instruction-tuned. Can you clarify whether this is a fair comparison?
5. Overthinking confound: Lower overthinking scores on AIME24 (Table 3) correlate with higher accuracy. Is the reduction in overthinking due to ADORA, or simply a byproduct of better performance?
6. Cross-architecture validation: Have you tested ADORA on non-Qwen models (Gemma, Llama, Phi, etc.) ? This is critical for establishing generalizability.

### Soundness
3

### Presentation
3

### Contribution
3
