# Descriptive History Representations: Learning Representations by Answering Questions

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Effective decision making in partially observable environments requires compressing long interaction histories into informative representations. We introduce descriptive history representations (DHRs): sufficient statistics characterized by their capacity to answer relevant questions about past interactions and potential future outcomes. DHRs focus on capturing the information necessary to address task-relevant queries, providing a structured way to summarize a history for optimal control. We propose a learning framework, involving representation, decision, and question-asking components, optimized using a joint objective that balances reward maximization with the representation's ability to answer informative questions. This yields representations that capture the salient historical details and predictive structures needed for effective decision making. We validate our approach on user modeling tasks with public movie and shopping datasets, generating interpretable textual user profiles which serve as sufficient statistics for predicting preference-driven behavior of users.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Descriptive History Representations (DHRs), where an encoder maps interaction histories to compact representations characterized by their ability to answer task-relevant questions. Training couples a decision objective with an f-divergence aligning a learned answerer to a generator built from full trajectories including future outcomes. Experiments on MovieLens-25M and Amazon Reviews generate textual user profiles and report improvements in pairwise prediction accuracy and recommendation reward over LLM baselines and two RecSys models.

### Strengths
Clear modular framework with encoder, answerer, decision agent, and discriminator, coupled via a precise dual objective

Strong interpretability: Generates human-readable textual user profiles that can be inspected

Consistent empirical gains over prompt-only LLM baselines and specialized RecSys models across multiple LLM sizes 

Useful ablations

Detailed appendices with complete prompts, rater instruments, implementation details, and qualitative examples help with reproducibility.

### Weaknesses
1. The paper's biggest problem is that the QA-generator sees future user interactions during training but obviously can't at test time. This means baselines only get user history while your method trains with future supervision. That does not seem right. Unless I misunderstood the setup. More concerning, the paper never explains how questions actually get generated at deployment when one does not have that future information. I'd also expect to see a baseline where you simply fine-tune an LLM on the same history-future pairs used to build questions, which would tell us whether the QA framework actually helps or if you're just benefiting from using more data.

The method fundamentally depends on ν*_QA(h,ω), which requires:

Access to future trajectories ω (only feasible offline/in-simulation)
Hand-designed question templates (Sec 5.1: "Rank the following movies...")
Domain-specific engineering for each new task

This creates a circular dependency: to learn sufficient representations, you need sufficient questions; to design sufficient questions, you need to know what's decision-relevant, but that is part of the learning objective. The paper acknowledges this is "challenging" (p.4) but provides no principled solution or failure mode analysis.

2. The recommendation reward metric assumes one can always check the user's true rating for whatever you recommend. That's not how it works. One probably needs inverse propensity scoring, or something similar. Without these, I can't tell if your method is actually better or just recommending popular items. There's no analysis of diversity or coverage either.

3.Where's the LLM that's fine-tuned on exactly the same supervision you use? That's the most obvious comparison. 

4. The definitions are confusing. The paper notes that a set of question-answer pairs is a sufficient statistic. I don't quite follow this. The definition says the representation should answer questions in a set that depends on the history you're trying to encode. That seems like circular reasoning. 

5. Looking at Tables 1-2, those confidence intervals could be quite close, there's no significance testing.

### Questions
How does one actually generate questions at test time when you don't have future outcomes?

Are you restricting actions to items with logged ratings? If so, why not use standard ranking metrics? 

What stops the same items from appearing in both your QA training supervision and your evaluation set?

Please add an LLM baseline that's trained on the exact same supervision you use

The definitions seem circular. How do you determine which questions are sufficient when that depends on the representation you're learning?

Have you tested whether just adding the raw history to your representation improves things? That would tell us if it's actually sufficient.

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
Summary: 

This paper introduces Descriptive History Representations (DHRs), a novel framework for learning informative representations of long interaction histories in partially observable environments. The core idea is to move away from traditional methods like belief states or predicting low-level observations. Instead, DHRs are learned by optimizing a representation's ability to answer a set of high-level, task-relevant questions about the past or future. The authors formalize this by defining a QA-space (Question-Answer-space) and posit that a representation that can successfully answer a set of "sufficient questions" acts as a sufficient statistic for the history. To learn these DHRs, a multi-agent cooperative framework called DHRL (Descriptive History Representation Learning) is proposed.


The key contributions of this paper are:
1. Formalization of DHRs: The paper introduces Descriptive History Representations (DHRs) as a new type of history representation defined by its ability to answer relevant questions. It formally links DHRs to the concept of f-sufficient statistics.

2. Multi-Agent Learning Framework (DHRL): It proposes a novel and flexible learning algorithm (DHRL) that operationalizes the DHR concept. This framework jointly trains an encoder, an answer agent, and a decision agent using a combined objective derived from a variational form of an f-divergence.

3. Interpretable Representations: The framework is shown to be capable of generating interpretable, textual user profiles that serve as effective representations for downstream tasks.

4. Empirical Validation: The paper provides empirical evidence on public recommendation datasets (MovieLens and Amazon) demonstrating that DHRL can outperform baseline methods in both prediction accuracy and recommendation reward.

### Strengths
* Novelty and Abstraction: The paper's main strength is its novel approach to representation learning. By focusing on answering high-level, semantically meaningful questions rather than predicting low-level observations, it shifts the representation burden to a more abstract and potentially more task-relevant level.

* Interpretability: In an age of black-box models, the ability to generate an interpretable textual user profile as the history representation is a significant advantage, particularly for domains like recommender systems.

* Flexible Framework: The DHRL algorithm is flexible, supporting different f-divergences (e.g., TV-distance, KL) and capable of operating in both online and offline training paradigms.

* Thorough Ablation Studies: The paper includes useful ablation studies analyzing the impact of history length, profile (DHR) length, and the number of questions, providing valuable insights into the model's behavior.

### Weaknesses
* Reliance on QA-Generator: The entire framework is critically dependent on the availability of a high-quality QA-generator ($\nu_{QA}^{*}$) to provide "sufficient" questions and ground-truth answers during training. The paper acknowledges that designing this oracle is challenging and relies on a pre-trained LLM for its main experiments. This dependency might limit its applicability in domains where such questions are hard to formulate or where a powerful pre-trained generator is unavailable.

* Training Complexity: The proposed DHRL framework is complex, involving the joint optimization of four components: the DHR encoder ($\theta_{E}$), the answer agent ($\theta_{A}$), the decision agent ($\theta_{D}$), and a discriminator ($g$). This could be computationally expensive and difficult to tune.

* Marginal Gain from Learned Generator: The paper reports that adversarially learning the QA-generator (as opposed to using a fixed one) "led to a marginal improvement in reward (2-3%)". This suggests that the framework for learning questions is not yet a major strength, and the success of the method hinges more on having good questions (e.g., from a pre-trained LLM).

* Limited Domain Exploration: The experiments are confined to recommender systems. While this is a suitable domain, the paper's claims about DHRs being a general approach for POMDPs would be much stronger if validated in other, more dissimilar domains (e.g., robotics, navigation, or game-playing).

### Questions
1. On the Objective Function: The paper proposes the joint objective in (OPT 1):$max_{E,\nu_{A},\pi_{D}}(1-\lambda)V(\pi)-\lambda D_{f}(d^{\nu_{A}^{*}}||d^{\nu_{A}})$. How was the hyperparameter $\lambda$ (which balances reward vs. question-answering) selected? How sensitive is the final performance of the decision agent $\pi_{D}$ to this trade-off?

2. On Attributing Performance: Since the main experiments use a QA-generator bootstrapped with a pre-trained LLM, how much of the performance gain over baselines is attributable to the novel DHRL framework versus the strong prior knowledge already embedded within the LLM used to generate questions and answers?

3. On Training Stability: Given the max-min optimization of the variational objective (OPT 2) involving a discriminator $g$ and three other agents, did you encounter any training instabilities? How does the choice of f-divergence ($D_{f}$) impact not only final performance (as shown) but also training stability?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Descriptive History Representations (DHRs), a novel framework that learns to generate compact and interpretable history representations by answering semantically meaningful questions about user preferences and future behaviour. The authors propose a multi-agent learning algorithm that optimises a representation encoder, a decision policy and a question-answering module simultaneously.

### Strengths
1. The core idea is novel: defining representation sufficiency in terms of the ability to answer task-relevant questions offers a new, principled approach to interpretable representation learning.
2. The experimental evaluation is thorough, covering multiple datasets and metrics. The results show consistent improvements over standard LLM-based and specialised recommendation methods.
3. The practical framework leverages off-the-shelf LLMs as QA generators and supports offline training, enabling deployment in industrial settings.

### Weaknesses
1. The QA generator uses future user behaviour to construct questions and ground-truth answers. However, can the method work in online cold-start settings where new users have no explicit ratings or reviews? Without ground-truth answers to supervise the QA generator, it is unclear how meaningful questions could be constructed or how the DHR would learn.
2. The optimization objective is a complex min-max game involving the joint training of multiple components. Did the authors observe any training instability or mode collapse? How does the convergence speed of DHRL compare to a baseline that simply fine-tunes a single decision-making model?
3. Current experiments are concentrated on recommendation systems with rich textual interactions. How would the DHR framework be applied to partially observable environments where the observation space is non-linguistic? How can a meaningful QA-space be defined for these domains?

### Questions
The same as the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Descriptive History Representations, a framework for learning compressed representations of interaction histories in partially observable environments. It claims that an effective representation should be a sufficient statistic characterized by its capacity to answer questions about the past and future, rather than merely predicting low-level future observations. The authors propose a multi-agent learning algorithm that jointly optimizes a representation encoder, a decision agent, and an answer agent to maximize task reward while ensuring the representation can accurately answer questions. 

Experimental results on public datasets show that the proposed method significantly outperforms base LLMs and state-of-the-art recommender systems in prediction accuracy and recommendation reward. The generated textual profiles are also rated highly by both AI and human evaluators, underscoring the method's strength in both performance and interpretability.

### Strengths
(1) The formulation of representation learning through question-answering is novel. It provides a interpretable alternative to methods that learn representations implicitly through prediction or reconstruction losses.

(2) The framework offers a high degree of interpretability. This is a significant advantage over existing methods.


(3) The paper provides comprehensive experiments on several datasets.  It includes extensive ablation studies (e.g., on history length, profile length, number of questions, choice of f-divergence) that provide valuable insights into the factors influencing performance.

### Weaknesses
(1) Dependence on a powerful QA-generator: The framework's performance is dependent on the quality and relevance of the questions generated by the oracle. While the use of a fixed LLM is practical, it introduces a dependency on the capabilities and potential biases of that specific model. The paper notes that adversarial training of the QA-generator yielded marginal gains, suggesting room for improvement in dynamically learning the optimal question set. 

(2) Computational complexity: The multi-agent training paradigm, requiring the sampling of full trajectories and the training of multiple LLM-based agents, is computationally intensive.

(3) Limited evaluation scope: While the recommendation domain is a compelling and challenging testbed, the generalizability of the proposed method to other tasks and domains remains an open question and is not demonstrated.

(4) The proof of Theorem 1 relies on the assumption that the set of sufficient questions is a function of the representation itself. How to enforce it during learning is not deeply explored.

### Questions
Please find the problems in the Weaknesses part.

### Soundness
3

### Presentation
2

### Contribution
3
