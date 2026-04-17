# Black-Box Privacy Attacks on Shared Representations in Multitask Learning

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
The proliferation of diverse data across users and organizations has driven the development of machine learning methods that enable multiple entities to jointly train models while minimizing data sharing. Among these, *multitask learning* (MTL) is a powerful paradigm that leverages similarities among multiple tasks, each with insufficient samples to train a standalone model, to solve them simultaneously. MTL accomplishes this by learning a *shared representation* that captures common structure between tasks and generalizes well across them all. Despite being designed to be the smallest unit of shared information necessary to effectively learn patterns across multiple tasks, these shared representations can inadvertently leak sensitive information about the particular tasks they were trained~on.

In this work, we investigate privacy leakage in shared representations through the lens of inference attacks. Towards this, we propose a novel, *black-box task-inference* threat model where the adversary, given the embedding vectors produced by querying the shared representation on samples from a particular task, aims to determine whether the task was present in the multitask training dataset. Motivated by analysis of tracing attacks on mean estimation over mixtures of Gaussian distributions, we develop efficient, purely black-box attacks on machine learning models that exploit the dependencies between embeddings from the same task without requiring shadow models or labeled reference data. We evaluate our attacks across vision and language domains when MTL is used for personalization and for solving multiple distinct learning problems, and demonstrate that even with access only to fresh task samples rather than training data, a black-box adversary can successfully infer a task's inclusion in training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies a new class of black-box privacy attacks on federated learning (FL) that operate by inverting shared representations rather than raw gradients. The proposed approach relies solely on the exchanged embeddings or logits accessible under practical FL protocols. The key idea is to train a small decoder network that learns to reconstruct private client data from aggregated or intermediate representations by leveraging auxiliary generative priors and feature alignment loss. The attack is evaluated under various FL architectures and datasets. The authors demonstrate that with limited black-box access, attackers can recover coarse visual features and demographic attributes, revealing new privacy risks.

### Strengths
- **Realistic and underexplored attack setting.** The black-box assumption makes the proposed attack more realistic than previous white-box gradient inversion works. This aligns well with real-world FL deployments where model internals are hidden.

- **Clear motivation and threat model.** The paper clearly defines the attacker’s capabilities and constraints. The scenario of accessing shared embeddings (but not gradients) accurately reflects common FL architectures such as SplitFed or representation-sharing systems.

### Weaknesses
- **No formal privacy quantification.** The paper demonstrates qualitative image reconstructions but lacks rigorous privacy leakage measurement, e.g., mutual information, feature attribution metrics, or membership inference accuracy. Without quantitative metrics, it is difficult to assess the real severity of the attack.

- **Dependence on auxiliary data.** The decoder is trained using an auxiliary dataset sampled from a similar distribution as the target data. This assumption is strong and limits the generality of the attack, particularly for domains like healthcare or finance where public data is unavailable.

- **Weak comparison baseline design.** Some baselines (e.g., DLG, MI-FL) are disadvantaged because they rely on gradient access. The paper could include more relevant black-box baselines such as feature-space inversion or model inversion methods.

- **Limited scalability and realism.** The attack is evaluated only on relatively small models and datasets. In practice, large models (e.g., ViT-L or ResNet-152) and high-dimensional embeddings could make black-box inversion considerably harder, but this is not tested.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper investigates privacy leakage in the shared representations from Multitask learning. To do this, the paper presents a black box attach which tries to infer the inclusion of a task during MTL. This is done by assuming an adversary has access to samples from the target task's distribution (either from the training set, a strong adversary, or independently from the target task distribution; a week adversary). and queries the shared representation of the model. Experiments demonstrate the success of the attack across vision and language settings.

### Strengths
* The paper instroduces a novel task-inference threat model that generalizes the sample-level membership inference to the task level. 

* The attachs presented a black box and is not limited by the requirement of shadow models nor labeled reference data.

* The empirical evaluations span both vision and language areas and use multiple datasets (CelebA, FEMNIST, StackOverflow) and MTL two common MTL cases (personalization and multi-problem learning).

* The attack algorithms are clearly described, and the results highlight that the non-trivial success of the attack.

### Weaknesses
* It is not clear how this novel attack compares to existing relevant baselines in these inference attachks (such as potentially those requiring stronger assumptions for the adversary).

### Questions
* How is the relationship between the attack simplicity and its attack power in the context of existing methods from prior work?

* Can the paper comment on how some existing defenses mitigate the task-inference risk?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a novel privacy threat called task-inference in Multitask Learning (MTL). The authors investigate whether a shared representation, the minimal component required for joint training in MTL, leaks information about the specific tasks used to train it. They propose a purely black-box threat model where an adversary, with only query access to the shared representation, aims to determine if a given task was included in the training set. The paper develops two efficient attacks, a Coordinate-Wise Variance Attack and a Pairwise Inner Product Attack, which exploit the codependencies of embeddings from the same task without requiring shadow models or auxiliary reference data. Experiments across vision (CelebA, FEMNIST) and language (Stack Overflow) datasets demonstrate that these attacks are effective, even for a "weak" adversary who only has access to fresh samples from a task, not the original training data. The paper links this leakage to the model's generalization gap, suggesting that models memorize properties of entire task distributions.

### Strengths
- This paper presents a novel and highly relevant threat model, "task-inference," which generalizes sample-level membership inference to the task level. This is a more practical and realistic threat for collaborative learning paradigms like MTL and federated learning, where the unit of privacy is often an entire user or data silo.
- The proposed attacks are efficient and operate under minimal adversarial assumptions. A significant strength is that they are purely black-box and do not require training costly shadow models or having access to large, labeled auxiliary datasets, which are common barriers for other inference attacks.
- The paper provides strong theoretical motivation for the attacks by analyzing an analogous tracing attack in a simplified Gaussian mean estimation problem. This analysis formally explains the observed performance gap between "strong" and "weak" adversaries.
- The experimental evaluation is comprehensive, validating the attacks across both vision and language domains (ResNet on CelebA/FEMNIST and BERT on Stack Overflow).
- The authors test the attacks in two distinct and practical MTL scenarios: personalization (where a task is a user) and solving multiple learning problems (where a task is a distinct classification objective), demonstrating the broad applicability of the threat.

### Weaknesses
- The paper does not clearly state whether the primary experimental results (e.g., in Figure 1 and 2) are averaged over multiple independent MTL training runs. While ablations mention multiple runs, the stability of the main attack results against different model initializations is not explicitly confirmed.
- The practical utility of the "blind" percentile-based thresholds is questionable. As shown in Table 1, these thresholds can result in very high false positive rates (e.g., 47.6% or 41.7%), which would make the attack unreliable for an adversary who truly has no reference data to set a better threshold.
- The "Pairwise Inner Product Attack" uses a whitening transformation that is computed by pooling all available embeddings. This seems to imply the adversary needs access to a broad set of embeddings from many different tasks, which somewhat weakens the "minimal knowledge" claim. The paper does not ablate the impact of this step.
- The paper’s discussion of defenses is very limited. It mentions user-level differential privacy (DP) but does not evaluate it or any other potential mitigation, such as representation compression or other regularization techniques beyond the gradient clipping that was already used during training.
- There is no comparison against adapted baselines. For instance, it's unclear how these novel attacks would perform against a simpler approach, like running a standard sample-level membership inference attack on all of the adversary's samples and aggregating the results (e.g., via majority vote) to get a task-level prediction.

### Questions
- Could the authors clarify how many independent MTL training runs were used for the main experiments in Figures 1 and 2? Are the reported ROC curves and AUC scores averaged across these runs?
- How critical is the whitening transformation to the "Pairwise Inner Product Attack"? How many tasks must an adversary pool embed to compute a stable covariance matrix, and how does the attack perform without this step?
- Have the authors considered comparing their attacks to a baseline where a standard, sample-level membership inference attack's predictions are aggregated (e.g., by averaging confidence scores or a majority vote) to produce a task-level inference?
- For the "multiple learning problems" case on Stack Overflow, the performance gap between strong and weak adversaries nearly disappears. Does this imply that any high-utility MTL model where tasks are defined by distinct labels will be inherently vulnerable to task inference, even from adversaries with no access to training data?
- L2​ decay or dropout on the shared representation be effective mitigations, or are formal defenses like user-level DP essential?

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
The paper introduces a new privacy attack in the multi-task learning (MTL) framework, where a shared representation is learned from data across multiple tasks. The goal of the attack is to infer whether a given task was included in the model’s training phase. Unlike traditional membership inference attacks at the sample level, this work operates at the task level, making it a higher-level privacy threat. The proposed attack is black-box and notably requires no auxiliary data or shadow models. 
Two adversary settings are considered: 
Strong adversary, which has access to some task data or samples drawn from the task’s distribution.
Weak adversary, which lacks access to such samples.

### Strengths
- The paper introduces a new threat model that examines how shared representations in multi-task learning can lead to privacy leakage.
- It presents a black-box attack formulated under both weak and strong adversary assumptions, which makes the study applicable to real-world conditions.
- A theoretical discussion is provided to explain the connection between representation sharing and privacy risks, highlighting the main factors that affect the effectiveness of the attack.

### Weaknesses
- There is a noticeable difference in performance between the weak and strong adversary scenarios.
- The paper does not explore or evaluate any potential defense strategies against the proposed attack.
- The method used to determine the attack’s decision threshold seems ad hoc, with limited justification or analysis of how it might perform in more realistic environments.

### Questions
- Can applying DP-SGD or other differential privacy methods mitigate the attack’s success rate? 
- Is the proposed attack scalable to larger and more complex models such as transformers or LLM-scale encoders?

### Soundness
3

### Presentation
3

### Contribution
3
