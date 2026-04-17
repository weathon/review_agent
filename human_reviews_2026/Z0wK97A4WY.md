# Context Attribution with Multi-Armed Bandit Optimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
Understanding which parts of the retrieved context contribute to a large language model’s generated answer is essential for building interpretable and trustworthy generative QA systems. We propose a novel framework that formulates context attribution as a combinatorial multi-armed bandit (CMAB) problem. Each context segment is treated as a bandit arm, and we employ Combinatorial Thompson Sampling (CTS) to efficiently explore the exponentially large space of context subsets under a limited query budget. Our method defines a reward function based on normalized token likelihoods, capturing how well a subset of segments supports the original model response. Unlike traditional perturbation-based attribution methods (e.g., SHAP), which sample subsets uniformly and incur high computational costs, our approach adaptively balances exploration and exploitation by leveraging posterior estimates of segment relevance. This leads to substantially improved query efficiency while maintaining high attribution fidelity. Extensive experiments on diverse datasets and LLMs demonstrate that our method achieves competitive attribution quality with fewer model queries.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies the problem of segment-level context attribution. The authors formulate attribution as a multi-armed bandit problem, where each segment is an arm and an action corresponds to selecting a subset of segments. The reward is defined as the change in token log-likelihood of the original answer when conditioning on a subset versus the full context. The objective is to find high-reward subsets under a limited query budget. The proposed method CAMAB uses Combinatorial Thompson Sampling with Gaussian posteriors per segment. At each round, it samples per-segment utilities, selects the top-p subset, queries the LLM to obtain the reward, and updates the posteriors using a linear semi-bandit observation model. The authors compare CAMAB with baselines such as ContextCite, Leave-One-Out, and SHAP. Experiments are conducted on 3 different tasks using 3 different LLMs of varying sizes.

### Strengths
* Context attribution is an important problem for RAG applications.


* The experiments include both token-level and sentence-level attribution.

### Weaknesses
* The main motivation of this work is to improve efficiency compared to perturbation-based methods such as LIME and SHAP. However, there already exist efficient variants such as TreeSHAP that mitigate the extensive sampling problem. The authors should discuss and include comparisons with these variants. Moreover, none of the datasets used in the paper involves long-context scenarios, which somewhat weakens the motivation.
* I am not fully convinced by the MAB framing. What is exploration and exploitation in the context attribution setting?
* There is more recent work on context attribution that the authors neither discuss nor compare against [1, 2, 3].
* The results in Tables 1–3 do not clearly show the advantages of CAMAB as the improvements are marginal. The authors should report standard deviations and/or conduct statistical significance tests.

[1] Cohen-Wang, Benjamin, Yung-Sung Chuang, and Aleksander Madry. "Learning to Attribute with Attention." arXiv preprint arXiv:2504.13752 (2025).

[2] Liu, Fengyuan, Nikhil Kandpal, and Colin Raffel. "AttriBoT: A Bag of Tricks for Efficiently Approximating Leave-One-Out Context Attribution." ICLR 2025.

[3] Xiao, Yingtai, et al. "TokenShapley: Token Level Context Attribution with Shapley Value." arXiv preprint arXiv:2507.05261 (2025).

### Questions
* Could the paper report runtime comparisons between CAMAB and other baselines?
* Since the Leave-One-Out baseline removes one segment at a time and measures the effect on the model's output likelihood, shouldn't it serve as an upper bound performance when $k = 1$? Could the authors clarify why CAMAB performs better than this baseline?

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
The paper proposes CAMAB, a new framework for context attribution in retrieval-augmented generation (RAG) systems with large language models (LLMs). By formulating the attribution task as a combinatorial multi-armed bandit (CMAB) problem and leveraging Combinatorial Thompson Sampling (CTS), the method aims to efficiently identify which context segments most support a model’s generated answer. The approach is evaluated on three datasets (SST2, HotpotQA, CNN/DailyMail) and three LLMs (LLaMA3-8B, Qwen3-8B, SmolLM-1.7B), showing competitive or slightly superior attribution quality with fewer model queries compared to established baselines like SHAP and ContextCite.

### Strengths
Novel Formulation: The CMAB framing for context attribution is original and addresses the inefficiency of traditional perturbation-based methods.
Principled Algorithm: The use of Combinatorial Thompson Sampling for adaptive exploration/exploitation is well-motivated and theoretically grounded.
Empirical Evaluation: The experiments are thorough, spanning multiple datasets and models, and use both log-probability drop and BERTScore metrics.
Practical Relevance: Query efficiency is a real concern for LLM applications, and the method is designed with this in mind.
Clear Baseline Comparisons: The paper benchmarks against strong baselines and explores performance under varying query budgets.

### Weaknesses
Marginal Empirical Gains: The improvements over existing baselines are generally small, and in some cases, CAMAB is only comparable rather than clearly superior—even under limited query budgets. This raises questions about the practical impact and significance of the contribution.

### Questions
Given the marginal improvement over existing prior works, what is key value add to attribution with the proposed approach?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces CAMAB, a framework for context attribution that identifies which retrieved segments most influence an LLM's output. It formulates the attribution task as a CMAB problem, where each context segment is treated as an arm, and uses combinatorial Thompson sampling to efficiently explore the exponentially large space of possible context subsets under limited query budgets. A reward function based on normalized token-likelihood differences measures how well a subset of segments supports the model's original answer. By maintaining Bayesian posteriors over each segment’s latent importance and sampling from these distributions to choose subsets, CAMAB adaptively balances exploration and exploitation, focusing model queries on the most informative context portions. This approach achieves attribution fidelity comparable to or better than traditional perturbation-based methods like SHAP or LIME, while requiring far fewer model evaluations.

### Strengths
* The authors address an important problem of contextual attribution
* The experimental sections design demonstrates generality across different levels of compositionality and context length.
It also supports the claim that CAMAB is versatile across diverse generative settings.

### Weaknesses
* The combinatorial MAB framing is elegant in theory but effectively reduced to an additive approximation that ignores segment interactions. While CTS provides tractable sampling, it does not truly explore the exponentially large action space; it merely assumes away the combinatorial complexity through a linear independence assumption. 
* Although presented as a Bayesian bandit innovation, the algorithm effectively performs an online approximation of SHAP-like marginal attribution. The independence and additivity assumptions make it efficient but at the cost of fidelity: segment dependencies, negations, or redundancy are not properly captured. 
* Results in Tables 1–3 show that CAMAB’s advantage is not consistent. The authors evaluate only 500 samples per dataset, randomly drawn from validation sets.
* A more principled alternative would be to frame attribution as a contextual MAB problem, where each context segment's expected reward depends on features derived from the question, retrieval context, and model generation. Such a formulation would allow attribution to adapt dynamically to query-specific semantics, capture complementarity and redundancy between segments, and leverage shared structure for improved sample efficiency. Such modelling would also help to propose different reward-schemes that do not mimic SHAP. For e.g., reward based on semantic alignment between a segment and the generated answer, pproximate influence using gradients etc.

### Questions
Kindly refer to weaknesses

### Soundness
2

### Presentation
2

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
This paper presents a framework for interpreting generative QA models by identifying context segments that influence a large language model’s responses. The proposed method formulates context attribution as a combinatorial multi-armed bandit problem and applies Combinatorial Thompson Sampling to efficiently explore context subsets under limited queries. A reward function based on normalized token likelihoods quantifies each segment’s contribution. Unlike perturbation-based methods adaptively balances exploration and exploitation through posterior relevance estimates, achieving high attribution fidelity with significantly fewer model queries.

### Strengths
The authors reformulate segment-level context attribution as a combinatorial multi-armed bandit problem.

### Weaknesses
The proposed context attribution method does not fundamentally differ from perturbation- or mask-based approaches; it mainly introduces improvements in sampling or subset partitioning. Moreover, as acknowledged by the authors, traditional methods can achieve better performance when computational resources are sufficient. Hence, the theoretical guarantees and advantages of the proposed method require further clarification.
In the experimental section, the dataset used by the authors appears to be small, which limits the ability to accurately assess the model’s performance and relevance to current benchmarks.
As a reminder, that many formulas are difficult to follow and lack adequate annotations or explanations, making it challenging for readers to fully understand the theoretical derivations.

### Questions
Please refer to the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
