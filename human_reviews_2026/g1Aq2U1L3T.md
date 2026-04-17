# Differential Information Distribution: A Bayesian Perspective on Direct Preference Optimization

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Direct Preference Optimization (DPO) has been widely used for aligning language models with human preferences in a supervised manner. However, several key questions remain unresolved: the rationale behind its log-ratio reward, how the statistical structure of preference datasets shapes its training dynamics, and how those dynamics impact downstream capabilities. We approach these questions from a Bayesian perspective, interpreting the goal of preference optimization as learning the differential information required to update a reference policy into a target policy. To formalize this view, we introduce the Differential Information Distribution (DID), defined as the distribution over samples that carry the Bayesian evidence required to update policies. We introduce three complementary insights by viewing preference optimization through the DID. First, we find that DPO's log-ratio reward is uniquely justified when preferences encode the Differential Information needed to update a reference policy into the target policy. Second, we discuss how commonly observed training dynamics in DPO, including changes in log-likelihood and policy exploration, stem from a power-law DID relationship. Finally, we analyze how training dynamics influence downstream performance using the entropy of DID, a principled measure of uncertainty in the learned information. We observe that learning high-entropy DID improves open-ended instruction-following, while low-entropy DID benefits knowledge-intensive QA. Taken together, our results show that DPO’s reward design, training dynamics, and downstream capabilities all emerge as natural consequences of learning Differential Information, offering both a principled theoretical foundation and practical guidance for preference-based alignment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a Bayesian reinterpretation of Direct Preference Optimization (DPO). It introduces the concept of Differential Information Distribution (DID), which represents the Bayesian evidence required to update a reference policy `π_ref` into a target policy `π*`. The authors show that DPO’s log-ratio reward can be derived as the unique form consistent with a power-law DID structure, and that training dynamics, specifically changes in log-likelihood and exploration, arise naturally from the properties of DID. Furthermore, the paper introduces DID entropy as a principled measure of the uncertainty of alignment information, showing that low-entropy DIDs improve factual QA while high-entropy DIDs improve open-ended generation. Theoretical claims are supported by formal derivations (e.g., the Likelihood Ratio Representation) and experiments on preference-based LLM fine-tuning.

### Strengths
1. Establishes a Bayesian formulation of preference optimization, linking DPO to information-theoretic evidence accumulation.  
2. Explains DPO’s reward structure, dynamics, and task-dependent behaviors (open-ended vs factual) under one consistent lens.  
3. Provides closed-form derivations (e.g., Likelihood Ratio Representation, Entropy of DID) that connect policy updates to Bayesian ratios.  
4. Offers a principled way to reason about why different `β` or entropy configurations produce distinct behaviors in alignment training.

### Weaknesses
1. The conditional independence of `X` from the prior and the power-law DID assumption are not empirically testable or demonstrated to hold in real preference data.  
2. DID “existence” is defined by construction, not derived, which weakens claims of theoretical generality.  
3. DID entropy estimation uses a small sample (`K=32`) with potentially large variance; no confidence intervals or significance testing are reported.  
4. Theorem 3.2’s “unique justification” of the log-ratio reward is contingent on strong assumptions and may not hold under alternative data generation processes.  
5. Experiments lack ablations (e.g., sensitivity to `β`, prompt distribution, or entropy threshold) and multi-seed averages.  
6. The analysis remains conceptual, real-world preference datasets are rarely power-law structured or independently sampled as assumed.

### Questions
1. Can the authors empirically validate that preference data follow a power-law DID or that `P(X|Y)` is conditionally independent of the prior?  
2. How sensitive are the DID entropy and corresponding trends (open-ended vs factual tasks) to sample size and importance weighting variance?  
3. Does Theorem 3.2’s “uniqueness” still hold if the DID deviates from power-law structure or when sampling dependencies between `(y_w, y_l)` exist?  
4. Can the authors clarify whether the entropy results persist under token-level DID instead of sequence-level ratios?  
5. Would alternative estimators (e.g., bootstrapped or token-normalized entropy) yield the same qualitative findings?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Differential Information Distribution (DID), which represents the distribution over samples that carry the Bayesian evidence required to update policies. The paper provides theoretical insights to prove that the log-ratio reward of DPO is the only optimal solution when preferences encode the differential information. Through DID analysis of policy training dynamics, the DID entropy is empirically linked to the performance of the model on open-ended instruction-following and knowledge-intensive QA tasks.

### Strengths
+ The paper introduces the Differential Information Distribution (DID), providing a deep understanding of how Bayesian evidence drives the updating of policies in DPO.

+ The paper demonstrates that the reward parameterization, training dynamics, and learned capabilities in DPO emerge naturally by analyzing DID.

+ By analyzing the Shannon entropy of the DID, this paper demonstrates how DID entropy influences the trade-off between factual accuracy and open-ended task performance via a real LLM experiment in Section 5.

### Weaknesses
+ The controlled Energy-Based Model experiments and empirical tests (Figure 1, Figure 2, Figure 3) in Section 3 and 4 are built around strong assumptions of matched data generating processes and synthetic settings where DIDs align almost perfectly. While some results (Table 1) use real-world LLMs and datasets, they are limited. It's uncertain whether the findings from synthetic setups apply to LLMs on real tasks.

+ There is the assumption that $\pi_{w} = \pi_{\text{ref}}$ in Section 4, along with the assumption that $\pi_{\text{ref}}$ is the unoptimized policy. However, the initial policy is not necessarily fine-tuned on $y_w$, and there is research focusing on applying different $\pi_{\text{ref}}$ strategies, which limits the applicability of the theory.

+ There are too many theorems, and many key proofs are in the appendix, which increases the difficulty of reading and hampers the flow.

### Questions
+ Can the authors explain why SimPO's JS divergence exhibits a distinct behavior in Figure 1 (Right)?

+ How about the performance of other baselines like SimPO in the experiment of Table 1? Can the same phenomenon be observed?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces an idea of differential information which captures the information needed to update a reference policy to a target policy. They utilize this framework to show that when the differential information distribution between preferred and dispreferred is related to the DID between the target and reference policy by a power law, the preference data contains the information needed to learn the target policy. They also demonstrate how DPO learns the optimal policy under this framework and setting. They provide validation on synthetic data and show that high and low DID entropy is correlated with different model behavior.

### Strengths
Framework - The paper provides a thorough analysis under the DID framework and show conditions under which the DPO reward is optimal. The framework utilizes a Bayesian perspective and brings a new approach to analyzing the behavior of DPO.  The analysis leads to potential new insights on the effects of likelihood displacement and the structure of preference data.

### Weaknesses
Justification - The framework while interesting and novel lacks justifications for key assumptions and definitions. For example, in definition 2.2, it is unclear why should conditional independence and a bayesian update model preference data and learning well. The framework relies on this definition, so it is important that justification and empirical support is provided. Furthermore, in Theorem 4.1, it is assumed that either preferred or dispreferred responses are sampled from the reference model which is in many settings not the case. Lastly, there is a lack of empirical evidence on real-world data that these assumptions hold or that the results do closely align with practice. 

Experimental results - Following up on previous comments, the experiments do not provide strong support for the claims. In particular, a demonstration of the power law on real-world data should be provided as well as a comparison across different rewards. Other results that would support claims are verifying Theorem 4.1 which can be directly verified on real-world data. Currently, the primary results on real-world data are a comparison between DPO and DPO-PG which was not a central part of the earlier results. 

Direct verification of results and justification of key assumptions would greatly improve the paper.

### Questions
- Can you provide justification for Definition 2.2?
- How might Theorem 4.1 generalize to other datasets?
- Can you directly verify the power-law relationship, the optimality of DPO, or the likelihood change on real-world data?

### Soundness
2

### Presentation
3

### Contribution
2
