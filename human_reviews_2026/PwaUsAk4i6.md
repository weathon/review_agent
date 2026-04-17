# Diffusion Guidance Is a Controllable Policy Improvement Operator

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
At the core of reinforcement learning is the idea of learning beyond the performance in the data. However, scaling such systems has proven notoriously tricky. In contrast, techniques from generative modeling have shown to be remarkably scalable and are simple to train. 
In this work, we combine these strengths, by deriving a direct relation between policy improvement and guidance of diffusion models. The resulting framework, CFGRL, is a policy improvement operator that is trained with the simplicity of supervised learning, yet is more effective than typically-used weighted policy extraction strategies. On offline RL tasks, we observe a reliable trend---increased guidance weighting leads to increased performance. Additionally, the CFGRL framework can be adapted to "directly'' extract policies from offline data *without* running a full end-to-end RL algorithm, allowing us to generalize simple supervised methods (e.g. goal-conditioned behavior cloning) to further prioritize optimality, gaining performance across the board without additional cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper reframes classifier-free guidance [1] from flow models as a controllable policy-improvement operator in RL, sampling from a prior policy tilted by an optimality term. Empirically, the proposed CFGRL improves over imitation/weighted-extraction baselines such as Goal-conditioned BC and AWR [2].

[1] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance." arXiv preprint arXiv:2207.12598 (2022).

[2] Peng, Xue Bin, et al. "Advantage-weighted regression: Simple and scalable off-policy reinforcement learning." arXiv preprint arXiv:1910.00177 (2019).

### Strengths
1. Theory & clarity.

The theoretical development is solid. The paper cleanly formalizes the link between generative-model guidance and policy improvement, and presents it in a way that is both accessible and precise.

2. Guidance weight $w$: evidence matches the claim.

Figure 4 empirically supports Remark 2: increasing the classifier-free guidance weight $w$ consistently steers the policy toward better directions. The additional results in the Appendix (Fig. 6) further corroborate this trend across settings, strengthening the central guidance-as-control narrative.

3. CFGRL for GCBC: clear setup, convincing gains.

Section 6 is well-specified and persuasive. The experiments substantiate the claim that CFGRL provides a principled, tunable alternative to standard GCBC: whereas prior GCBC effectively fixes $w=1$, the proposed framework exposes $w$ as a controllable knob and yields improved performance accordingly. The alignment between the theoretical framing and the observed gains is compelling.

### Weaknesses
1. The paper positions CFGRL as a scalable policy-extraction tool inspired by classifier-free guidance, and shows consistent gains over GCBC/flow-GCBC on OGBench. I think the paper's central thesis is to scale up RL by leveraging the scalability of “modern generative learning,” and, by implication, generative-model-based techniques like CFGRL should match or exceed the performance of RL-based methods such as FQL [3], which are harder to scale under the author's claim (as noted at line 28). To substantiate the broader “scale up RL” thesis, it would be essential to include strong baselines. Including FQL results would clarify where the advantage of CFGRL lies, in competing with modern value-gradient methods rather than with an unexpressive policy-extraction scheme like AWR [4].

[3] Park, Seohong, Qiyang Li, and Sergey Levine. "Flow q-learning." arXiv preprint arXiv:2502.02538 (2025).

[4] Park, Seohong, et al. "Is value learning really the main bottleneck in offline RL?." Advances in Neural Information Processing Systems 37 (2024): 79029-79056.

### Questions
1. Positioning vs. SOTA offline RL: strengths and weaknesses.

Even if CFGRL is not intended to replace SOTA offline RL, a clear trade-off analysis is needed. Please add representative value-/policy-gradient baselines (e.g., FQL) under matched protocols, and discuss where CFGRL is advantageous (simplicity, stability, test-time controllability $w$) and where it is weaker (asymptotic performance, OOD robustness, sample efficiency). A brief ablation on wall-clock, hyper-sensitivity, and compute/memory would help practitioners decide when to choose CFGRL as a “designer’s toolbox” component.


2. Task selection in Table 2.

OGBench contains many tasks, but Table 2 reports nine. What criteria determined this subset? Please provide a justification for the selection or expand the evaluation to a broader, representative slice of OGBench to reduce selection bias.


3. Baselines for the sampling/extraction objective (Eq. 8).

Since many policy-extraction schemes target the same tilted distribution, AWR alone is insufficient as a comparator. Please include Relative Trajectory Balance [5] as an additional baseline: it optimizes an equivalent target distribution in theory and achieves stronger performance than AWR on several D4RL tasks [6]. 

4. Scope beyond $o\in\{0,1,\emptyset\}$: return-conditioned guidance.

Section 4.1 instantiates $o\in\{0,1,\emptyset\}$, but under the Control-as-an-Inference view, the optimality variable naturally extends to learned Q-values or discounted returns. This suggests a return-conditioned variant—analogous to RvS/Decision Transformer/Decision Diffuser [7][8][9]—where classifier-free guidance tilts actions by target return. Please clarify how the $o\in\{0,1,\emptyset\}$ setting in CFGRL compares to return-conditioned BC with classifier-free guidance (both conceptually and empirically). 

[5] Venkatraman, Siddarth, et al. "Amortizing intractable inference in diffusion models for vision, language, and control." Advances in neural information processing systems 37 (2024): 76080-76114.

[6] Fu, Justin, et al. "D4RL: Datasets for Deep Data-Driven Reinforcement Learning."

[7] Emmons, Scott, et al. "RvS: What is Essential for Offline RL via Supervised Learning?." International Conference on Learning Representations.

[8] Chen, Lili, et al. "Decision transformer: Reinforcement learning via sequence modeling." Advances in neural information processing systems 34 (2021): 15084-15097.

[9] Ajay, Anurag, et al. "Is Conditional Generative Modeling all you need for Decision Making?." The Eleventh International Conference on Learning Representations.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the CFGRL framework, which establishes a theoretical connection between classifier-Free diffusion guidance and the policy improvement operator in classical reinforcement learning. It formulates the improved policy as a "product policy", defined as a reference policy multiplied by an advantage-based optimality function. The authors highlight that the guidance weight $w$ enables flexible control over the extent of policy improvement at test time, without requiring retraining as in methods such as AWR. The proposed CFGRL framework is validated on both offline RL and goal-conditioned behavior cloning (GCBC) tasks, demonstrating superior performance over AWR and GCBC methods that can be interpreted as using a fixed guidance weight of 1. These findings provide evidence that guidance weights greater than one can effectively cooperate with diffusion models to achieve more efficient policy improvement.

### Strengths
1. The paper is well written, and the presentation of results is clear and easy for readers to follow.

2. To the best of my knowledge, this paper is the first to theoretically establish and prove the connection between classifier-free guided diffusion policy sampling and the policy improvement operator in RL.

3. The authors’ analysis of AWR’s weakness in Section 5, together with the experimental observation that CFGRL can sustain larger guidance weights than AWR, constitutes an interesting result.

### Weaknesses
1. The main limitation of this paper lies in that most of its ideas have already appeared independently in prior works. For example, the relationship between classifier-free guidance and weighted regression has been discussed in [1], while the use of classifier-free guidance for policy improvement and the adjustment of different guidance weights was explored in [2]. Although the authors argue that [2] focuses on generating future state sequences whereas CFGRL generates single-step actions, I consider this distinction in output space relatively trivial. Apart from these aspects, the most notable contribution is the explicit formulation of the connection between the **policy improvement operator** and **classifier-free guided diffusion policy sampling**. However, this result is not particularly surprising, and the theoretical derivation itself is fairly straightforward. Considering this as the paper's primary contribution, it remains questionable whether the work alone is sufficient to support publication at ICLR.

2. Moreover, the baseline methods in the experiments are relatively weak, and the proposed algorithm does not obtain a significant performance advantage under these settings, which limits the overall algorithmic contribution. Although the authors emphasize that "By itself, CFGRL does not represent a state-of-the-art RL algorithm, but rather an additional tool in the algorithm designer's toolbox," the paper would be more compelling if the authors could further demonstrate how innovative algorithms can be derived using the CFGRL framework.

[1] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance." *arXiv preprint arXiv:2207.12598* (2022).

[2] Ajay, Anurag, et al. "Is Conditional Generative Modeling all you need for Decision Making?." *The Eleventh International Conference on Learning Representations*.

### Questions
1. What kinds of more advanced algorithms do the authors expect could be derived from the CFGRL framework, beyond current implementations that largely reproduce methods already proposed in prior work?

2. Beyond GCBC and offline RL, what other problem domains could the CFGRL framework be applied to?

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
This paper proposes Classifier-Free Guided Reinforcement Learning (CFGRL), which interprets the classifier-free guidance (CFG) mechanism of diffusion models as a policy improvement operator.
The authors show that by treating the optimality condition as a discrete binary variable $o\in \{0,1\}$, CFG can be reinterpreted as applying an advantage-weighted transformation on a reference policy. This allows controllable policy improvement via the guidance weight $w$, similar in spirit to temperature or KL coefficients in regularized RL. Experiments demonstrate that CFGRL improves over Advantage-Weighted Regression (AWR) and Goal-Conditioned Behavioral Cloning (GCBC) across offline RL benchmarks.
- An LLM was used to improve writing.

### Strengths
1. Simplicity and practical appeal – The method requires only standard diffusion training and allows tuning the improvement strength $w$ at inference, offering a practical way to control policy quality without retraining.

2. Solid empirical demonstration – Results on offline RL and goal-conditioned control tasks consistently show improvements over strong baselines such as AWR and GCBC.

3. Readable and well-presented – The paper is clearly written, with theoretical and empirical sections well balanced.

### Weaknesses
1. Limited novelty beyond reinterpretation

The core idea—recasting classifier-free guidance as a policy improvement operator—is conceptually elegant but incremental.

The method mainly replaces the continuous classifier (score function) in diffusion guidance with a discrete optimality variable, which is a small modification rather than a fundamentally new algorithmic contribution.

Much of the theoretical framing follows directly from existing formulations of advantage-weighted regression and control-as-inference.

2. Lack of comparative analysis with continuous guidance

The paper would be significantly stronger if it empirically compared continuous value-based guidance (e.g., by Q-values or advantages) versus discrete optimality-based guidance.

Such a comparison could clarify what is actually gained by discretizing optimality in this setting.

Without this, CFGRL appears as a special case of prior decision-diffuser-style methods using advantage-conditioned diffusion.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes CFGRL, a simple, controllable policy-improvement operator that leverages classifier-free guidance from diffusion/flow models. Policies are factorized as a product of a reference policy and an “optimality” term that is a monotone function of the advantage; sampling from this product is achieved by composing unconditional and optimality-conditioned policy scores, with a test-time guidance weight $w$ controlling the strength of improvement. The authors prove that such product policies improve over the reference and that increasing $w$ yields further improvement (with the usual trade-off against distribution shift). They also show that, under certain choices, guided sampling matches the solution to a KL-regularized policy-improvement objective. Practically, they instantiate CFGRL with a single diffusion/flow network and provide simple training/sampling algorithms

### Strengths
The central insight—viewing policy improvement as classifier-free guidance over an advantage-conditioned policy—is elegant. It unifies guided diffusion sampling with KL-regularized policy improvement and control-as-inference via a clean product-policy view, and shows that test-time guidance directly tunes the improvement strength. 

The theory is tidy. The paper also avoids learning an explicit optimality predictor via a Bayes inversion that merges unconditional and optimality-conditioned policies into a single network. Algorithms are minimal and clear. 

Empirically, CFGRL improves over AWR on a strong majority of ExORL and OGBench tasks and shows a more favorable scaling trend than AWR’s temperature sweep. The GCBC “upgrade” is impactful: simply guiding the goal-conditioned policy (no value function) yields sizable gains, including hierarchical variants. Code is released; runs are modestly resource-bound.

### Weaknesses
The paper notes that larger w both improves $ A_{ \hat \pi }$ and deviates more from the dataset policy, possibly hurting performance; the ablation indeed shows performance sometimes declines past a point, but there’s no adaptive or trust-region control of $w$ or measured KL to the prior. 

For the offline RL part, results are averaged over four seeds; gains are consistent but sometimes modest. The GCBC part uses more seeds, but a wider set of domains and stronger end-to-end RL baselines would further cement significance. The paper itself notes it is not a full SOTA RL algorithm. 

Flow steps (16–32) suggest non-trivial inference cost relative to a single-shot policy; wall-clock sampling latency and throughput are not measured, which matters for deployment.

### Questions
Could you add comparisons to CRR/CCR, IDQL with diffusion policy extraction, and Q-score-matching / rejection-sampling approaches, ideally normalizing compute and tuning budgets? This would position CFGRL more clearly among diffusion-policy extractors.

### Soundness
3

### Presentation
3

### Contribution
3
