# Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions

- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
Large language models (LLMs) have seen increasing popularity in enterprise applications where AI agents and humans engage in objective-driven interactions. However, these systems are difficult to evaluate: data may be complex and unlabeled; human annotation is often impractical at scale; custom metrics can monitor for specific errors, but not previously-undetected ones; and LLM judges can produce unreliable results. We introduce the first set of weakly supervised metrics for objective-driven interactions, leveraging statistical properties of unlabeled interaction data and reasonable assumptions on the performance of frontier LLMs. We apply small fine-tuned LLMs to adapt to distributional shifts and develop metrics for clustering user goals, classifying goal completion, and quantifying LLM uncertainty without grounding evaluations in human-generated ideal responses. Our approach is validated on open-domain and task-specific interaction data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an unsupervised framework for evaluating large language models (LLMs) in multi-turn, goal-oriented interactions without relying on human labels or LLM judges. The framework has three modules: a) Goal Identification uses LLM summarization and embedding-based clustering to discover latent user intents in unlabeled dialogues; b) Completion Detection fine-tunes a Llama 3 8B model to predict whether a conversation has achieved its goal using a special <|end|> token; c) Response Trees – introduces a novel visualization of model uncertainty based on the branching structure of possible next responses. Experiments across datasets such as LMSYS-Chat-1M, Code-Feedback, Insurance, WebShop, and SQL+OS+KB show that the small completion model can approach or even outperform large LLM judges on binary goal-completion classification (F1 up to 0.99). However, the Response Tree component is conceptual and not empirically validated.

### Strengths
The paper focuses on a highly relevant and underexplored challenge in the current LLM research landscap: how to evaluate multi-turn, goal-oriented agents without relying on costly human annotations or unstable LLM-as-a-judge paradigms. As LLMs increasingly act as autonomous or semi-autonomous agents engaging in extended task-driven interactions, scalable and objective evaluation becomes a bottleneck for both research and deployment.

### Weaknesses
1. The “unsupervised” claim is misleading: all three modules depend on LLM summarization, synthetic labeling, or fine-tuning.
2. The negative sampling strategy (random truncation) does not accurately simulate real incomplete dialogues.
3. The goal-identification module requires embedding and clustering the entire dataset at once, relying on iterative global merging and cross-cluster sampling. This design makes the method inherently batch-based, it cannot process streaming data without full re-clustering (e.g., in online systems).
4. The Response Tree idea is not evaluated quantitatively; no correlation analysis links branching structure to uncertainty or correctness.
5. The overall empirical contribution is confined to the completion detector, leaving the framework’s other parts remain speculative.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces three unsupervised metrics for evaluating multi-turn objective-driven interactions between users and LLMs: (1) goal identification through LLM-guided clustering that combines k-means with LLM-generated labels, (2) goal completion detection using fine-tuned LLMs that predict an "end tag" for completed conversations, and (3) response uncertainty quantification via response trees that enumerate probable LLM outputs. The authors validate their approach on diverse datasets including LMSYS-Chat-1M, code-feedback, insurance underwriting, WebShop, and technical assistance tasks (SQL+OS+KB), demonstrating that 8B fine-tuned models can match or exceed 70B LLM judges on completion labeling.

### Strengths
1. The core ideas—unsupervised goal clustering via LLM+k-means hybridization, completion detection via distributional fine-tuning, and uncertainty quantification via response trees—are highly original.

2.  The methods address a real and pressing need in enterprise AI development. The ability to perform evaluation without labels or human judges is a substantial practical advance.

### Weaknesses
1. The method assumes "a majority of interactions are complete" to train the completion detector, which fundamentally undermines its unsupervised nature and creates a validation problem.

2. The pipeline likely depends on embedding model, initial k, prompt phrasing for merges/labels, and decoding hyperparameters. Robustness to these choices is not fully demonstrated.

### Questions
1. How do you address the fundamental circularity in assuming most interactions are complete to detect completion? Could you validate on datasets with known failure rates?

2. Why does completion detection fail so dramatically on Code-Feedback (F1=0.21)? This seems to invalidate the approach for follow-up conversations.

3. What is the computational cost vs. using LLM judges? Is fine-tuning + inference cheaper?

4. In Algorithm 1, an LLM is prompted to decide if two clusters should be merged. Given the paper's valid critiques of LLM judge instability, what steps were taken to ensure this "merge" decision is stable and reliable? Did you observe any of the instability (e.g., order sensitivity) that you found in the LLM-only baseline?

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
4

### Summary
This paper proposes the first unsupervised metrics to evaluate objective-driven AI–human interactions, addressing the limits of human annotation, bespoke checks, and unreliable LLM judges on complex unlabeled data.
Leveraging statistical signals and fine-tuned LLMs to handle distribution shifts, the metrics label user goals, assess goal completion, and estimate LLM uncertainty, with validation on open-domain and task-specific datasets.

### Strengths
- The paper addresses key shortcomings in current LLM evaluation methods.
- It presents extensive experiments demonstrating the effectiveness of the proposed approaches.

### Weaknesses
- The overall paper structure could be improved. A concise introduction is fine, but it should still highlight the main points to orient the reader.
- The selected attributes require a clear rationale for why they are important for evaluating LLM responses.
- The metrics used to measure them also need clearer explanation and justification: how well do these metrics reflect the degree or quality of the attributes?
- The title is potentially misleading. What exactly is meant by “unsupervised”? Is a fine-tuned model used for completion evaluation?
- How do you clearly justify the claim that a larger number of leaves implies greater uncertainty in LLM responses?

### Questions
Please see the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a critical and timely problem in the deployment of LLMs in enterprise settings: the evaluation of multi-turn, objective-driven interactions (e.g., customer service, task-oriented agents) without relying on human annotations or unreliable LLM-as-a-judge paradigms. The authors propose a novel suite of three unsupervised metrics:
LLM-Guided Clustering: An algorithm that combines k-means on text embeddings with LLM-based labeling and merging to automatically discover and label user goals from unlabeled conversation data.
Interaction Completeness: A method that fine-tunes a small LLM on a dataset of (mostly) completed conversations, tagged with a special "end" token, to predict whether a given interaction is complete. Incomplete interactions are flagged as outliers.
Response Uncertainty: A framework using "response trees" to quantify the LLM's uncertainty during a conversation by exploring the branching factor and log-probabilities of potential responses, serving as a proxy for interaction quality and potential error likelihood.
The methodology is empirically validated on a range of datasets, including open-domain chats and specialized, tool-using interactions. The key result is that their approach, often using a fine-tuned 8B parameter model, can match or exceed the performance of a much larger (70B) LLM judge on the task of completion labeling, demonstrating the potential for efficient, specialized evaluation.

### Strengths
Novelty and Practical Impact: The core idea of moving beyond LLM judges and human references to unsupervised, statistically-grounded metrics is highly novel and addresses a significant pain point in real-world AI system development. The potential for online monitoring and resource-saving interventions is a compelling practical contribution.
Holistic Evaluation Framework: The paper doesn't propose a single metric but a suite of three complementary metrics that address different aspects of an interaction (goal, completion, uncertainty), providing a more comprehensive evaluation toolkit.
Strong Empirical Validation: The paper is thorough in its experimentation, using multiple diverse datasets (LMSYS, Code-Feedback, Insurance, WebShop, etc.) to demonstrate the generality of the approach. The ablation studies (e.g., "No end tag," comparison of fine-tuned vs. base models) effectively justify key design choices.
Efficiency Focus: The successful use of small (8B parameter) models, fine-tuned for specific distributions, is a major strength. It challenges the prevailing trend of using ever-larger models as judges and opens avenues for cost-effective, scalable evaluation.
Clarity of Core Methodology: The underlying concepts for the completeness and uncertainty metrics—modeling the token distribution of completed interactions and constructing response trees—are well-explained and theoretically sound.

### Weaknesses
Reliance on Key Assumptions: The methodology rests on two strong assumptions that may not always hold in practice: a single user goal per interaction and that failures are "rare." The performance degradation on Code-Feedback, where follow-up questions violate the "single well-defined end" assumption, highlights this fragility. The applicability in noisier, real-world environments with frequent failures is not fully established.
Limited Statistical Rigor: While the results are promising, the statistical significance of the improvements is not tested. Table 2 reports accuracy and F1 scores, but without confidence intervals or statistical tests, it's difficult to gauge the robustness of the claims, especially on smaller datasets like Insurance (n=380).
Writing and Presentation Issues:
Inconsistent Metric Naming: The three core metrics are introduced as labeling user goals, measuring goal completion, and quantifying LLM uncertainty. However, in the methodology and experiments, they are referred to as "clustering," "completion labeling," and "response trees." A consistent, clear naming scheme would improve readability.
Undefined "Stability": The clustering algorithm is praised for its "stability," but this term is not quantitatively defined. Figure 4 is visual and compelling, but a quantitative measure of cluster stability (e.g., Adjusted Rand Index between runs) would strengthen the claim.
Supervision in "Unsupervised" Learning: The approach requires a dataset where "the majority of interactions are complete" to fine-tune the completeness model. While not requiring manual labels, this does require a curated set of complete conversations, which is a form of weak supervision. The term "unsupervised" should be nuanced to acknowledge this data curation requirement.

### Questions
Assumptions and Generalizability: How would your completeness metric perform in a production environment where user interactions frequently contain multiple, interleaved goals or where the failure rate is significantly higher than in your training data? Are there strategies to make the approach more robust to such violations of its core assumptions?
Statistical Significance: Given the varying sizes of your datasets, have you performed any statistical significance testing (e.g., bootstrapping) on the results in Table 2? This would help clarify whether the observed performance differences, particularly on smaller datasets like Insurance, are reliable.
Clustering Stability Metric: You state your clustering algorithm produces "highly stable clusters." Beyond the confusion matrices, did you calculate a quantitative stability metric like the Adjusted Rand Index between different runs? If so, what were the results?
Response Tree Practicality: The response tree is a powerful conceptual tool for quantifying uncertainty. However, the computational cost of building these trees for long, complex conversations must be non-trivial. Could you comment on the computational overhead of this method and its feasibility for near-real-time monitoring?
Defining "Completion": The "end" tag is a clever heuristic, but it conflates syntactic and semantic completion. A conversation could end with the user giving up. Does your method have a way to distinguish between a successfully resolved goal and an abandoned one, or is it solely based on the model's learned distribution of "end-of-dialogue" tokens?

### Soundness
3

### Presentation
3

### Contribution
2
