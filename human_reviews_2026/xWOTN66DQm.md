# ScaleMoE: Mixture-of-Experts for Scalable Continuous Control in Actor–Critic Reinforcement Learning

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Scaling model size has been a key driver of progress in supervised learning, but remains a challenge in deep reinforcement learning (RL), where naively increasing the parameters of actor-critic networks often leads to instability and performance degradation. While recent architectures like SimBa and BRC have shown that careful inductive biases can enable positive scaling in continuous control, they remain monolithic, activating all parameters for every input. In this work, we introduce \textbf{ScaleMoE}, an architecture that integrates Mixture-of-Experts (MoE) modules into both the actor and critic of state-of-the-art continuous control algorithms. This approach effectively turns parameter growth into consistent performance gains. We propose two integration strategies: (i) \emph{output-level gating}, where a learned gating network selects the top-$K$ expert actors and critics per state and merges their outputs (policy means, variances, and $Q$-values) via gating weights; and (ii) \emph{feature-level gating}, where experts produce penultimate features that are combined by top-$K$ gating and passed through a shared output layer for both policy and value predictions. We implement ScaleMoE on a single-task actor–critic baseline (SimBa) and a multi-task baseline (BRC), two representative monolithic scaling RL methods. Experiments on the DeepMind Control Suite and HumanoidBench demonstrate improved returns as the number of experts increases. In multi-task settings, ScaleMoE with smaller experts matches or outperforms a larger monolithic network with substantially less model parameters. Our findings indicate that MoE offers an effective and compute-efficient scaling axis for deep RL in continuous control, narrowing the gap with supervised learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a Soft Mixture of Experts (Soft-MoE) model for single & multi-task continuous control environments in reinforcement learning. The authors propose two gating mechanisms to aid in the aggregation of features or outputs of the Soft-MoE models. The main claim of this work is SOTA performance across single and multi-task settings.

### Strengths
The idea of gating for features or outputs is an interesting idea that could have promise in the aggregation of multiple model outputs (at the feature or output level), across pre-trained, online, or a mixture of pre-trained and online models.

### Weaknesses
1. It is widely noted that RL experiments are incredibly noisy and often require a large number of seeds. It is likely that 8 seeds is not enough to gain a full understanding of the performance distribution [1]. In order to overcome this, there have been attempts at developing measures that overcome the (relatively) small sample size such as IQM [2]. The authors need to update their results to use a measure such as IQM to ensure that their results are statistically significant. From the current manuscript there are many different results that seemingly overlap with the baseline (Simba or BRO) results and therefore we cannot determine if the results are statistical significant or not. 

In a similar vein to point 1, I am wondering if the authors of this manuscript ran the baselines of Simba and BRC themselves or not. I am a bit concerned about the use of dotted lines with no standard error (even though we can't use this to compare performance) compared to the box plots of the Scale-MoE methods. 

2. There is a gap in the manuscripts related works & experimental results. Recent work has found that architectural improvements are seemingly not needed in the MTRL setting [3]. The authors should attempt to incorporate simpler models with various network depths & widths in order to fully (a) justify the need for the Soft-MoE method, and (b) fully scope out the scaling performance profiles.

3. Another gap in the submitted work is the lack of use of a MTRL benchmark, such as Meta-World [4]. From [3], 10-20 tasks isn't a lot of tasks, which could be an explanation for any plasticity loss. Using the MT50 set of tasks from [4] would help to better understand the effects of a more difficult setting.

4. There is a lack of general polish in the writing. It seems like there are quite a few changes in tense (for example using "passes" instead of "passed") in the text, and there are also a number of small errors (i.e. line 362, "with 8 xexperts"). I suggest that the authors do a pass (without the use of a LLM) to catch all of these issues. There also seems to be implementation details mixed in with details about the proposed methods (i.e. Section 3.1 starts to describe the method but then mentions Simba and BRC). 


Citations
1. Patterson, A., Neumann, S., White, M., White, A. (2024). Empirical Design in Reinforcement Learning. Journal of Machine Learning Research.
2. Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare (2021). Deep RL at the Edge of the Statistical Precipice. NeurIPS
3. McLean, R., Chatzaroulas, E., Terry, J., Woungang, I., Farsad, N., Castro, PS. (2025). Multi-task Reinforcement Learning Enables Parameter Scaling. https://arxiv.org/abs/2503.05126v3
4. McLean, R., Chatzaroulas, E., McCutcheon, L., Röder, F., Yu, T., He, Z., Zentner, KR., Julian, R., Terry, J., Woungang, I., Farsad, N., Castro, PS. (2025) Meta-World+: An Improved, Standardized, RL Benchmark. https://arxiv.org/abs/2505.11289

### Questions
How does using the Soft-MoE model affect gradient flow? Does choosing a higher top K value affect this? I would think that using a soft MoE model means that there is some parameter change per gradient update for the Top K experts, but not the other N - K experts which allows for the specialization of the experts. 


Can you show the optimization profile over the course of training to acknowledge this comment? "In the multi-task setting, we attribute the advantage of activating more experts primarily to mitigating under-training: a larger K encourages broader gradient coverage per batch and leads to more balanced expert utilization, reinforced by our load-balancing and importance regularizers."

### Soundness
1

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
The paper focuses on approaches that enable parameter scaling in the domain of off-policy actor-critic algorithms. In this context, the authors propose to leverage the mixture of experts (MoE) architecture. The authors test various MoE implementation details and propose best practices for utilizing MoE in online continuous control via actor-critic. The authors test their approach in single-task and multi-task RL, where they compare using their proposed MoE method with previously used SimBa and BRO architectures on DeepMind Control and HumanoidBench benchmarks.

### Strengths
- Studying architectures that allow for scaling actor-critic is timely and important
- The proposed method is simple and seems to offer a more computationally efficient alternative to dense scaling via BRO or SimBa

### Weaknesses
- MoE was previously shown to work in the context of RL; as such, the method has limited novelty. I appreciate the ablation studies and the proposed gating mechanisms

### Questions
1. How does MoE+gating impact the expressibility of the policy class? Is the final policy still constrained to the TanhNormal class?
2. Previous work (eg. BRO, SimBa) found that scaling the critic yields better improvements than scaling the actor, and as such recommend scaling only the critic networks. To the best of my understanding, the authors find that scaling the actor via MoE leads to significant improvements (Fig. 6). I find this result interesting. Can authors elaborate why would MoE+gating scaling lead to substantially different results when it comes to actor scaling than dense methods like BRO or SimBa?
3. How would Figure 4 look for multi-task learning? Do authors think that there is a correlation between expert activation and task ID (ie. experts specializing in certain tasks)?
4. Could the authors summarize the differences in MoE implementation between their proposed approach other MoE in RL papers (eg. [1])?


[1] Obando-Ceron et al. Mixtures of Experts Unlock Parameter Scaling for Deep RL

### Soundness
3

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
3

### Summary
This paper proposes ScaleMoE, a Mixture-of-Experts (MoE) framework for continuous-control actor–critic RL.
It integrates top-K gated experts into both the actor and critic to enable conditional computation — i.e., only a subset of experts is active per state.
Two integration variants are tested: output-level gating (merging Gaussian policy and Q-values) and feature-level gating (combining penultimate features).
Experiments on DeepMind Control Suite and HumanoidBench claim that increasing the number of experts (N) improves performance and efficiency compared to monolithic baselines (SimBa and BRC).

### Strengths
- The motivation is clear: current RL architectures fail to scale reliably with network size.
- The method is technically clean and general, requiring no change to the underlying algorithms.
- Empirical comparisons against SimBa and BRC are systematic, and analysis (expert activation, dormant neuron ratio) is thoughtful.
- The results do show stable training with conditional computation and improved efficiency per parameter (but see weakness on this point as well).

### Weaknesses
### **Overall**

The paper is well written and the experiments are run carefully, but I don’t see convincing theoretical or empirical evidence for the main “scaling” claim.
The results mostly show modest **parameter-efficiency improvements**, not actual **scaling behaviour** or **scaling laws** in RL.
At this point, my score leans closer to a **2–3 range rather than a 4**, since the evidence doesn’t really support the core narrative about scaling.


### **Scaling evidence is insufficient**

The main claim that ScaleMoE “unlocks scaling in RL” is not well substantiated.
Experiments explore only **N ≤ 8 experts**, with total parameters increasing linearly with N.
No scaling-law analysis, extrapolation, or asymptotic regime is presented.
The curves in Figure 2 show *positive but shallow* trends within a narrow capacity range — this indicates *improved stability*, not a scaling law.

### **Baseline comparison undercuts the claim**

The strongest baseline, **BRC (critic width = 4096)**, already exceeds the total parameter count of the largest MoE model (**8 experts × 512 width**).
Thus, the reported gains mainly reflect **parameter efficiency**, not the ability to handle truly larger networks.
A convincing scaling study would start from the 4096-width baseline and scale *beyond* it while maintaining stability — which is not attempted.
Indeed, Appendix F confirms that the standard BRC is substantially larger in parameters.

### **Theoretical section is descriptive, not analytical**

Appendix J provides qualitative bias–variance and optimization intuitions but lacks formal derivations or predictive quantitative relations akin to supervised-learning scaling laws.
While conceptually reasonable, the analysis does not meaningfully connect to or explain the empirical results, limiting its interpretive value.

### **Experimental scope is narrow**

All experiments use **low-dimensional proprioceptive tasks**; no pixel-based or high-dimensional domains are tested.
Moreover, the explored ranges of **N** and **K (1–8)** are too limited to draw robust conclusions about scaling behaviour or compute–performance trade-offs.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces ScaleMoE, a Mixture-of-Experts (MoE) architecture inserted into both the actor and critic of continuous control RL. Two plug-in variants are proposed: output-level gating (late fusion of policy means/variances and Q-values) and feature-level gating (early fusion of penultimate features with a shared head). Using top-K routing and simple load-balancing/importance regularizers, the method is dropped into SimBa (single-task) and BRC (multi-task) without changing their learning rules. On DMC hard tasks and HumanoidBench, returns often scale up with the number of experts. Analyses show task-specific expert specialization and a lower dormant-neuron ratio.

### Strengths
- Scaling actor and critic networks in continuous control is a crucial problem to study, and this paper takes a mixture-of-experts perspective on it.
- Figure 4 about expert utilization is interesting as it shows that the gating mechanism does not collapse to a single dominant expert, and also that the same expert is utilized across various skills.

### Weaknesses
- The selection of the Q-value’s top-k experts is dependent on the output of the actor network, which is an unusual dependence. Is there a guarantee that this actor-critic architectural modification is a sound modification to the underlying RL algorithm, such that its convergence properties extend easily?
- What is the motivation and justification of the load-balancing loss? Why this specific loss, is it based on some prior work?
- Multi-task HumanoidBench benefits of ScaleMOE are not statistically significant.
- In Figure 6, the overall best setting is fully expert activation, then why does the paper propose top-k gating at all? Doesn't this invalidate one of the key contributions of this work w.r.t. gating?
- Ablation about load-balancing regularization is missing.
- From the full reward curves for the results in Figure 2, it seems that the benefits of ScaleMOE often happen in one of the several environments, while the performance is same in most environments as the baselines. Is there any convincing analysis that demonstrates *why* and *how* ScaleMOE results in better performance than the baseline?

### Questions
- Was the UTD ratio tuned specifically for the baselines like SimBA, or was it taken from the tuning of ScaleMOE?
- How much is the actual empirical compute benefit of using top-K actor-critics with K < N?

### Soundness
2

### Presentation
3

### Contribution
2
