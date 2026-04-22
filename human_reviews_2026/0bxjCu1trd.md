# Teach a Reward Model to Correct Itself: Reward Guided Adversarial Failure Discovery for Robust Reward Modeling

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 8

## Abstract
Reward models (RMs) trained from human preferences are central to aligning large language models, yet they often break under distribution shift or targeted perturbations. Existing failure discovery methods rely on prior knowledge of preference attributes and therefore do not scale to new models or data. We introduce a preference distribution agnostic procedure that uses the reward model itself to \textit{guide} controlled decoding toward mis specified responses while preserving the underlying preference class. Building on this discovery mechanism, we propose REFORM, a self improving RM framework that (i) searches for class consistent but reward inconsistent variants and (ii) fine tunes the RM on a small, targeted augmentation of these failures. On Anthropic Helpful Harmless and PKU Beavertails, REFORM consistently improves robustness without degrading in distribution reward quality  across different models (e.g., Mistral-7B and Qwen-14B), with an average improvement of \textbf{35\%–45\%}.Further, across Best of N sampling, PPO, and DPO, REFORM preserves downstream generation quality and reduces spurious correlations. Our results show that RMs can serve as their own adversary to expose and fix blind spots, yielding robust alignment without manual attribute priors or large scale relabeling.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Reward Model trained with human preferences are important for aligning LLMs but can break with distribution shift. The paper introduces a preference distribution agnostic approach to use the reward model to guide control decoding toward mis-specified responses.  Using this approach, they train on HH-RLHF and Beaver-tails and shows improvements of 35-45% across PPO, DPO and Best of N (rejection sampling).

### Strengths
1.	Targeted correction is reasonable since a small proportion of the samples might be mislabelled. 
2.	The approach of constrained decoding to generate synthetic data is reasonable to train models to avoid tokens that have high likelihood of harm (e.g. vulgar words or words commonly associated with crime).
3.	There is a rich diversity of illustrations in the main paper and appendix to clarify the work. I would recommend putting a few illustrated examples (ideally at the front) to better situate the work.

### Weaknesses
1.	The approach described in Fig. 2 (and Eq. in line 64) suggests that there’s a clean line separating chosen and rejected responses (into 2 distinct classes). However, preferences are only ordinal (and not cardinal) – meaning to say that we can have a great chosen vs good rejected - or - good chosen vs bad rejected. This means that class-conditioning doesn’t necessary work because in some settings where we have 4 responses A>B>C>D, response B will fail in both chosen and rejected baskets. (I recognize that this issue is more applicable to helpfulness than harmlessness,  but might still happen with harmful settings where one is more harmful than another).
2.	The assumption that we can use a reward model as a value function by only giving it a partial completion up to a certain token is poorly supported. The paper cites Khav et al. (2024) – which shows reward of partial responses can be used to guide decoding. However, in my experience training and inferencing with reward models, I think it’s very unlikely for Reward Models to accurately grade partial responses (since they were never trained to do so). Can the authors provide further evidence that this approach does indeed work? (I recognize that this might be less of a problem for harmlessness but would still like some evidence of this).
3.	The paper doesn’t use standard safety benchmarks such as ToxiGen [1] or XSTest [2] to evaluate the performance of their aligned models. Instead, it only measures it based on the predicted rewards of the reward models, which could be overfitted to. This makes it hard to compare this work with other approaches.

[1] Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, Ece Kamar. ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection ACL 2022 

[2] Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. Xstest: A test suite for identifying exaggerated safety behaviours in large language models. NAACL 2024.

### Questions
1.	How does the approach compare to methods such as Rule-based Rewards[1] or DExperts[2]?
2.	Not exactly a question but I recommend that the authors mention that the paper is on safety-based Reward Modelling early in the paper (ideally in the abstract) – this wasn’t clear to me until I got to line 268. Prior to that, my impression was that the paper was applicable to both helpfulness and harmlessness (which inspired weakness 1 and 2). 
3.	Not exactly a question but some of the figures (5 to 9) are less informative (not clear what information they are seeking to communicate) and could be better presented in a table instead.

[1] Tong Mu, Alec Helyar, Johannes Heidecke, Joshua Achiam, Andrea Vallone, Ian Kivlichan, Molly Lin, Alex Beutel, John Schulman, Lilian Weng. Rule Based Rewards for Language Model Safety. NeurIPS 2024.

[2] Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula, Noah A. Smith, Yejin Choi. DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts. ACL 2021.

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
In this papers authors highlight the failures of exisiting Reward Modes (RMs) under distribution shifts or adversarial prompts. They propose a lightweight method called REFORM which lets the RM essentially find its own blind spots, by using a clever decoding approach to find reponses from policy where RM doesn’t flip the preference but the reward is very low. So this guided decoding finds “low-reward preferred or high-reward non-preferred responses”. These adversarial examples are then used to make the model more robust by finetuning on top 5% of the failures.

### Strengths
- Simple light-weight and low cost approach to find adversarial examples for RM by using RM’s own scoring to guide controlled decoding
- Well formulated objective with clear explanation of the components involved
- It is an automatic approach and doesn’t need any human labelling to find adversarial examples
- Thorough evaluation setup and experiments which highlight the benefits of using the robust RM from their approach. Especially experiments on downstream Alignment show no degradation.

### Weaknesses
- Novelty  is limited since [1] has a similar approach of adversarial example generation followed by reward-model augmentation to improve robustness.
- The authors approach is lightweight but potentially limiting.  Work like [1] which uses RL optimization over a discrepancy/loss like R_1 - R_2  can potentially  explore much more diverse attacks leading to much more robust-RM, ofc at the cost of much more computation overhead. Comparisons to this baseline could be helpful.
- Minor: Models used for evaluation are old and not sota open LLMs
- Writing is a bit convoluted and can be improved upon

[1] https://arxiv.org/abs/2504.06141v2 , Adversarial Training of Reward Models

### Questions
- Rather than a token-level decode is it possible to do a high-temperature sampling from the policy to elicit the same behaviour. And possibly those discovered failures reflect the sequence level reward hacking examples encounterd in RLHF training better.
- Its unclear if the RM reward for partial completions will correlated with the final reward for the full completion. Can authors clarify this point.

### Soundness
3

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
2

### Summary
This paper proposes REFORM, a novel framework for improving the robustness of reward models (RMs) used in aligning large language models (LLMs). The key idea is to let the RM act as its own adversary to discover and correct its failure modes. Instead of relying on external attribute priors or large human relabeling efforts, REFORM uses reward-guided controlled decoding to generate class-consistent but reward-inconsistent samples — responses that belong to the same preference class but are mis-scored by the RM. The method involves two stages:  failure discovery and targeted correction. Empirically, REFORM demonstrates consistent improvements in robustness on Anthropic Helpful–Harmless  and PKU Beavertails datasets using Mistral-7B and Qwen-14B reward models, without sacrificing in-distribution performance or downstream alignment quality under Best-of-N, PPO,  and DPO  training.

### Strengths
1. The paper offers a creative approach to reward model self-improvement. Its key innovation lies in using *the reward model itself* as a guide to generate adversarial counterexamples, circumventing the need for external attribute priors or large LLM queries. 
2. The empirical performance shows the potential of the proposed method.

### Weaknesses
1. While the empirical findings are promising, the paper lacks deeper theoretical grounding on *why* self-adversarial discovery effectively generalizes across preference distributions. The surrogate objective (Eq. 6) and proxy use of partial reward evaluation are empirically justified but theoretically heuristic. A discussion on convergence guarantees or connections to robustness theory (e.g., adversarial training or influence functions) would strengthen the framework.

2. Despite that the high-level idea and the methodology proposed by the paper looks quite interesting, the introduction of detailed concepts and corresponding implementations are not satisfactory, please see my questions below. This casts concerns to the actual soundness of the method. Overall, I believe the question studies by the paper is very critical -- how to discover the failure mode or misspecified examples of a reward model without relying on prior attributes or distillating a stronger teacher but only using the reward model itself -- but the quality of presentation needs further improvement. I am willing to increase my rating if the concenrs are well addressed in the response.

### Questions
1. The introduction of the failure mode in Section 3.1 seems a little bit confusing to me: On the one hand, the reward model is assumed to be learned via the Bradley & Terry model (Eq (1)), a model which only specifies a relative order for a given response pair, i.e., given $y_1$ and $y_2$, either $y_1$ is preferred or $y_2$ is preferred. This is by the fact that the BT model only depends on the reward difference. However, later the authors introduce another concept "the preference class is preserved", where an explicit preferred or non-preferred label is assigned to a response. How is that defined? If a variant response pair only satisfies $y_+'$ is preferred than $y_-'$, it's ture that the BT model can not guarantee the relative order of $(y_+', y_-)$ and $(y_+, y_-')$. Is the concept of "class consistency" more like an extra constaint to the BT model?
2. What does "a policy aligned with the preference distribution" mean here? Does it mean a policy that has been fine-tuned on the preference dataset that is used to generate the failure mode?
3. I believe one of the core parts of the entire methodology proposed by the paper (see e.g. Figure 2) is that how to find the adversarially labeled responses that remain in the same "preference class" purely by the reward model itself. That is, how to ensure that a variant response searched is still in the same "preference class". But the algorithm part of the paper (Section 3) discusses nothing on this essential issue. How to guarantee that the searched response, e.g. through Eq (6), is going to be a preferred response? Also, how is the TopK hyperparameter choosen in practice? How is the  partial response $y_{<i}$ constructed?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes REFORM, a self-improving framework for reward models (RMs) trained from human preferences. The main motivation is that RMs, used pervasively in RLHF pipelines and test-time alignment, exhibit brittleness under distributional shift or targeted perturbations. Existing failure-discovery techniques typically require explicit knowledge of preference attributes (e.g., length, tone, toxicity) and thus do not generalize across domains or models.

REFORM introduces two key components:
	1.	Reward-guided failure discovery.  Instead of relying on external attributes, the method uses the reward model itself to generate class-consistent but reward-inconsistent samples—responses that belong to the same preference class (e.g., harmless, helpful) but are scored incorrectly. Controlled decoding guides the generation process by penalizing the reward score while keeping the response fluent and semantically aligned with the class. Formally, at each token, REFORM modifies the decoding objective
\min_{y_i \in \text{Top-K}} \ r_\theta(x,[y_{<i},y_i]) - \alpha \log \pi_D(y_i|x,y_{<i}),
steering toward low-reward variants for “preferred’’ responses (and symmetrically high-reward variants for “non-preferred’’ ones). This produces adversarial but meaningful failures.
	2.	Influence-targeted correction.  After generating these failures, the method fine-tunes the RM on a small augmented dataset built from the top 5% of “influential’’ training pairs—those with the lowest Bradley–Terry loss. The model is retrained (or fine-tuned) on these augmented pairs, thereby correcting the discovered mis-specifications without requiring broad relabeling.

Empirical evaluation uses Anthropic Helpful–Harmless and PKU Beavertails datasets with Mistral-7B and Qwen-14B reward models. REFORM is shown to:
	•	generate coherent and semantically valid failure examples without external attribute priors;
	•	improve robustness under four types of adversarial perturbations (verbosity, capitalization, repetition, misspelling);
	•	maintain in-distribution accuracy and downstream alignment quality (measured via BoN, PPO, and DPO policies).

The results suggest a 35–45% average gain in robustness with minimal degradation in reward quality.

### Strengths
1.	Conceptually elegant and well-motivated.  Turning the reward model into its own adversary is an appealing and general idea. The paper formalizes a clear operational notion of RM “failure’’ and exploits it for targeted data augmentation.
	2.	Technically sound within the RM framework.  The controlled-decoding formulation is simple yet effective. By combining reward minimization with language-model likelihood regularization, the authors achieve fluency while probing the RM’s vulnerabilities. The influence-based sampling for correction is principled and computationally light.
	3.	Empirical rigor.  Evaluation covers two standard safety datasets, two model families, and three alignment pipelines (BoN, PPO, DPO). Metrics include both in-distribution reward accuracy and out-of-distribution robustness. Figures 5–9 provide convincing evidence that REFORM improves robustness while preserving quality.
	4.	Practical impact.  The method requires no new labeling, no access to external attribute priors, and only a small fine-tuning budget. It could realistically be integrated into existing RLHF workflows.

### Weaknesses
1.	Limited theoretical analysis.  The paper lacks a quantitative characterization of the discovered failures or convergence guarantees for the fine-tuning step. One might ask whether the adversarial search reliably covers diverse failure modes or simply re-weights local perturbations around a few high-influence examples.
	2.	Restricted evaluation scope.  While robustness to perturbations is measured, the study focuses on specific lexical attacks. It would be valuable to see stress tests on semantic or reasoning-level shifts (e.g., long-context instructions, domain transfer).
	3.	Potential overfitting to adversarial decoding artifacts.  Because the fine-tuning samples are generated by the same reward model, there is a risk of reinforcing spurious patterns from the model’s own biases rather than true failures. A small discussion of safeguards (e.g., cross-model validation) would strengthen confidence.
	4.	Clarity and presentation.  The exposition is generally clear but occasionally dense; the decoding objectives (Eqs. 5–7) could benefit from a concise mathematical summary, and the distinction between class-consistency and semantic fidelity deserves clearer operationalization.

### Questions
1.	How sensitive is REFORM to the hyperparameter \alpha controlling fluency vs. reward deviation? Does a larger \alpha lead to trivial near-policy samples, while a smaller one yields incoherent outputs?
	2.	Since the failure discovery uses the same RM, how do you ensure diversity in the generated failures—could the method get trapped in a narrow mode of failure?
	3.	Can REFORM be iterated multiple times (self-play style) to progressively improve robustness? If so, does it converge or oscillate?
	4.	How would the framework handle non-text modalities or structured outputs where “class consistency’’ is harder to define?
	5.	In downstream alignment (PPO/DPO), does the observed robustness transfer when policies are further fine-tuned with new data, or is it specific to the evaluation distribution?

### Soundness
3

### Presentation
3

### Contribution
4
