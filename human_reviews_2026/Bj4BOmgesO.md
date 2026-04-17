# Mutual Information Dynamics Learning: A New Paradigm for Unsupervised Reinforcement Learning

- Decision: Reject
- Scores: 4, 2, 4, 8

## Abstract
Unsupervised reinforcement learning (URL) aims to develop general-purpose agents that can adapt to unseen downstream tasks without relying on task-specific supervision. Existing approaches predominantly focus on learning diverse skills by maximizing mutual information, but they are often limited to simple navigation tasks and fail to scale to more complex domains such as robotic manipulation, where prior knowledge is typically required. In this work, we demonstrate that mutual information-based objectives can be leveraged far beyond skill learning. We propose a novel URL framework that trains exploratory skills to collect diverse transition data with distinct dynamics. This diverse dataset enables the training of a mixture of dynamic models, where each model captures the dynamics of a specific region. Collectively, these models provide comprehensive coverage of the dynamics required for a wide range of downstream tasks. Our straightforward and prior-free learning objective outperforms existing state-of-the-art skill discovery approaches in URL. Our results advocate a paradigm shift in URL, from skill learning toward dynamics learning, to acquire fully generalizable knowledge during pretraining.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a new unsupervised reinforcement learning (URL) objective based on mutual information between the next state predicted by a dynamics model and the real next state, with a skill latent conditioned on the current state and action. The objective encourages a skill-conditioned policy to cover diverse dynamics while being distinct. An ensemble of models is trained from the data collected by the policy, where each model is specialized for one corresponding skill.
At finetuning the pre-trained policy and ensemble dynamics models used for planning, and then the policy is fine-tuned on the data collected by the planning scheme.
The method is evaluated on the URL benchmark (URLB) environment and on the humanoid environment.

### Strengths
- The paper is mostly clear and easy to follow.
- Results are strong and promising.

### Weaknesses
- Experiments are mainly focusing on locomotion environments (only one manipulation task), this is might be sufficient for locomotion tasks, but I think more evaluation is needed on manipulation benchmarks.
- The derivation of the objective is not clear, especially the jump from equations 4 to 5, and the justification does not make sense.

### Questions
1. I find it confusing that, in the pre-training phase, there is an additional policy trained with a pure exploration term. How can we guarantee that the data collected by this policy will be accurate or useful for training one of the specialized models? How does this policy affect the theory introduced in the previous section? Also, since Equation (8) already includes a pure exploration term, in principle, there should be no need for an additional pure exploration policy. Did the authors run their method with a policy trained only using Equation (8)? How does that affect the performance of the method during both pre-training and fine-tuning? Would removing the pure exploration policy make a significant difference?
1. Can you clarify the choice of $z$ in the model during fine-tuning? Do you use the selector network $\phi$? I suggest adding an additional line in the pseudocode in Algorithm 2 to clarify this step.
1. Have you evaluated the exploration performance of the pre-trained policies? For example, by counting the number of covered states or (x,y) positions (when feasible)? Do you think the coverage would be correlate with fine-tuning performance?
1. Can you also include a plot or table showing performance before and after fine-tuning, or a learning curve for the fine-tuning process? I think this is important to assess the efficiency of the fine-tuning procedure.
1. The results on URLB are not conclusive (the confidence bounds overlap).
1. Can you clarify how many runs were performed for the experiments in Section 4.3? Line 464 says “5 independent runs for each downstream task.” Does that mean that pre-training was done once? Is each of the five runs initialized with a different random seed? Also, can you add the confidence bounds to Figure 5b?
1. In Appendix E.1, I do not understand the justification for using $q_{\theta}(s' \mid s, a, z)$ in place of $q(\hat{s}' \mid s, a, s', z)$. Can you explain this argument more clearly?

I am willing to raise my score if the authors addressed most of my concerns.

### Soundness
2

### Presentation
3

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
This paper proposes a new paradigm, MIDL, which shifts from unsupervised skill learning to unsupervised dynamic learning, addressing the limitations of MISL approaches in capturing complex behaviors.

### Strengths
In the MIDL framework, a gating network is trained instead of assuming a uniform prior over skills. This network may better samples appropriate skills for a given state–action pair.

### Weaknesses
It is debatable whether the claim that skill learning should be shifted to dynamics learning under the unsupervised RL paradigm is well supported, as skill learning and model-based learning focus on different aspects of reinforcement learning.

The paper does not clearly explain what $Z$ denotes in the MIDL framework.

In the experiments section, the analysis of the downstream task is insufficient, and some references are missing. Moreover, there is no explanation provided for Fig. 5.

### Questions
Could you conduct an additional experiment using ICM [1]?

[1] Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In International Conference on Machine Learning, 2017.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Presents a mutual-information objective that combines Mutual Information Skill Learning(MISL) with model-based pretraining to learn a specialized ensemble of dynamics models.

maximize information gain + dynamic matching

Reports that this yields more diverse/accurate dynamics and SOTA downstream performance on the standard Unsupervised Reinforcement Learning Benchmark (URLB) and competitive performance on Humanoid, with compatiblility with standard model-based backbones.

### Strengths
Clear motivation citing that mutual information skill learning struggles to transfer beyond simple navigation

Mutual information-based decomposition combines exploration (info gain via prediction variance/error) with model specialization, and the proposed method avoids the uniform-prior approximation used in prior work.

### Weaknesses
Performance gains on URLB are not noticeable over EUCLID: The total scores are very close (8072 vs 8058), and the stronger advantage appears mainly on Humanoid, so more analysis across various settings and initializations, etc are needed.

Training an ensemble + gating would increase memory/computation requirements and the authors mention this, but this should be discussed more, e.g. varying ensemble size vs performance vs computation

Derivations rely on Gaussian/variance proxies. What happens under model missspecification?

No ablation study

### Questions
Ablation of the three core elements will be needed : dynamic-matching term, information-gain term, and gating vs. uniform mixing. which is most critical for the examples presented?

Humanoid is known to be very sensitive to initial conditions; what happens when tested with more wider distribution across seeds and initial conditions ?

Performance gain over EUCLID seems weak. Can you provide benefit for the tasks where each (or any) of the core elements noticeably better performance?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose Mutual Information Dynamics Learning (MIDL), a novel unsupervised reinforcement learning framework in which different skills capture the dynamics of distinct state-space regions. MIDL leverages mutual information objectives to collect data with diverse dynamics, enabling the training of a mixture of specialized dynamics models. These models provide sufficient coverage for solving a wide range of downstream tasks. The framework consists of a pretraining phase that learns the dynamics model ensemble and a finetuning phase that uses model-based planning for task adaptation.

### Strengths
- The paper effectively critiques existing MISL methods using empirical evidence and makes a compelling case for prioritizing approaches focusing on dynamics.
- The approach and objectives are mathematically justified. The mutual information objective (Equation 2) decomposes into information gain and dynamic matching terms, with clear intuition provided.
-  The 2D quadrant wind environment (Figure 2, Table 1) provides an interpretable toy experiment that demonstrates MIDL's ability to learn diverse dynamics where baselines fail.
- Strong results across multiple benchmarks (URLB, Humanoid) with diverse baselines. The model accuracy analysis via a two-player game framework (Table 2) is insightful.

### Weaknesses
- While Table 2 provides valuable insights into model learning quality, this analysis is only conducted on the Swimmer environment. It would strengthen the claims to demonstrate whether the superior accuracy of MIDL generalizes across other environments.
- Minor typos and formatting issues: 
    - "Diagreement" (line 295)
    - "exlploit" (line 388)
    - incorrect citation formatting throughout (using \citet where \citep would be appropriate, e.g. lines 168 and 180)
    - a missing reference at line 434,
    - $\widehat{S}'$ but should be $S'$ line 260
    - $q_\theta (s'|s,a))$ should be $q_{\theta,\psi} (s'|s,a))$ in Equation 8

### Questions
- Did you perform the model accuracy analysis (similar to Table 2) for additional environments beyond Swimmer to demonstrate the generalizability of MIDL's superior dynamics learning? If yes, are observations the same?
- How does the performance evolve as you vary the number of skills $|\mathcal{Z}|$?

### Soundness
4

### Presentation
4

### Contribution
3
