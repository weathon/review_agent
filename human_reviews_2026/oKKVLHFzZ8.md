# RPM: Reasoning-Level Personalization for Black-Box Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
While black-box large language models are widely deployed, they produce generic outputs that overlook individual user preferences.
Current personalization methods are fundamentally limited to response-level personalization; they only match final outputs, failing to model the underlying reasoning that connects user behavior to responses. 
To address this, this work introduces reasoning-level personalization as a new paradigm and proposes RPM, the first systematic framework that automatically discovers user-specific reasoning structures from raw behavioral data to guide the model's personalized inference.
RPM constructs a structured model of user behavior—built from response-influential features and statistical factors—to create personalized reasoning paths and retrieve beneficial examples for guiding inference through a feature-based retrieval mechanism. 
Extensive experiments across four diverse tasks demonstrate that RPM consistently outperforms existing response-level methods while simultaneously enhancing both personalization performance and interpretability, providing a promising direction for black-box LLM personalization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the limitations of the black-box LLM Response-Level Personalization, which leads to superficial pattern learning and a lack of interpretability. Accordingly, it proposes Reasoning-Level Personalization, which extracts features from user histories, clusters them into quantifiable factors, constructs a structured user model, and generates explicit reasoning paths for each historical interaction. Experimental results demonstrate that the proposed RPM consistently outperforms existing response-level methods in terms of accuracy and exhibits improved interpretability.

### Strengths
1. The introduction of "reasoning-level personalization" is reasonable and meaningful.
2. An interpretable personalization framework is proposed.
3. Comparisons with baseline methods and ablation experiments demonstrate the effectiveness of the proposed strategy and the rationality of each module.

### Weaknesses
1. The definition of structured features is confusing and unclear until the appendix. However, the main text provides no further guidance.
2. Explanations should be faithful to the model's decisions and cannot be verified through human evaluation. 
3. Although the inference cost in Table 9 appears negligible (i.e., 0.0037), for a large service provider, this represents a cost increase of several times, or even dozens of times, compared to the baseline method. This requires further discussion.

### Questions
1. Does the preprocessing process refer to the personalized factor construction and personalized inference building process?

### Soundness
3

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
3

### Summary
Black-box LLMs often ignore user-specific preferences, as most methods personalize only the response, not the reasoning behind it. RPM introduces reasoning-level personalization: it extracts response-influential features, clusters them into statistical factors, and builds personalized reasoning paths. Using feature-based retrieval to surface factor-aligned exemplars, RPM consistently beats response-level baselines on four tasks while improving both personalization and interpretability.

### Strengths
- Well-specified method with prompts provided for each step.
- Includes latency and cost analyses (online and offline).
- Explicit feature–factor–reasoning structure facilitates diagnosis and analysis.
- The experimental evaluation is comprehensive.

### Weaknesses
- Extraction/clustering robustness: Reliance on prompt-based LLMs for feature extraction, clustering, and influence/polarity judgments may cause spurious features or misclustered factors—especially in term-heavy domains or under style drift—propagating errors to retrieval and generation.
- “Personalized reasoning paths” are a pragmatic approximation of user cognition; they may appear plausible without being literally true.
- The method is fundamentally prompt-centric; consequently, the technique-level contribution is modest despite solid engineering and interpretability.
- With no trainable components, it may fail to refine query representations or reasoning mechanisms, reducing robustness under terminology ambiguity.

### Questions
- When factor-aware extraction encounters previously unseen factors, do RPM falls back to generic factors or dynamically expands the factor set? What thresholds, validation gates, and rollback procedures govern this?
- Evidence alignment & hallucination control: Can a personalized reasoning path hallucinate features that don’t exist?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces RPM (Reasoning-Level Personalization), a framework for personalizing black-box large language models by aligning the model's reasoning process with individual user behavior patterns, rather than only matching final outputs. They extracts response-influential features from user interactions, clusters them into statistical factors, and constructs personalized reasoning paths that explain the connection between queries and responses. For inference, RPM retrieves similar reasoning examples based on feature matching to guide the model's output generation. The paper evalautes four benchmarks showing improvements over existing personalization methods, and presents human evaluations and claims that generated reasoning paths are interpretable and grounded in user behavior. RPM operates without requiring parameter tuning or model access.

### Strengths
S1. The paper is well written and addresses an important gap in LLM personalization which is the reasoning step.

S2. The evaluation seems to be comprehensive and results show the effectiveness of the technique. 

S3. The framework is prompting compatible and cost efficient.

### Weaknesses
W1: The set of published baseline is rather limitted - it would be good to see more baselines to understand where the technique stands.  

W2. The paper does not evaluate generailizablity across various choices of LLM from small to large and different architectures. 

W3. The human evaluation in Sec 4.4 compares RPM against baselines with CoT prompting, but there is no analysis of whether raters might confuse longer, more detailed explanations with genuinely better reasoning.

 W4. No length-normalized comparisons or controlled studies separating verbosity from interpretability are provided.

W5. The paper does not examine continuously evolving preferences or online adaptation scenarios.

W6. It would be good if the paper provides theoretical justification for why coverage-influence-polarity measures constitute stable behavioral indicators.

### Questions
How can RPM handle model's self-bias ?

Can RPM handle handle out-of-domain tasks? Will HYDRA (Zhuang et al., 2024) out perform in this? 

What is the scalability charecteristics of factor generation via LLM-based clustering ?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This study presents a structured reasoning framework for personalization using black-box LLMs. It intends to align models' generation process with user-specific decision pattern. The experiment results show that the proposed RPM framework outputs some personalization baselines in the four selected benchmarks. Moreover, it provides more interpretability.

### Strengths
(1) This study introduces structured reasoning in LLM generation (which may be quite common nowadays) to the personalization task. This attempt is original.

(2) Extensive experiments show that the proposed structure benefits the personalization task more compared to other simpler prompting, RAG-based baselines and two personalization-focus baselines.

(3) The method description is clear.

### Weaknesses
(1) The novelty seems to lie in the application of a common method to a specific task (i.e., personalization), which can be limited.

(2) More baselines are necessary to support the claim that the proposed structured prompting framework is better. There have been many studies that give LLMs explicit structure for reasoning and are not compared in the current manuscript. For example, least-to-most prompting which decompose a hard problem into ordered subproblems and solve them sequentially. This is linear. Additionally, there are methods that search over reasoning paths such as tree-of-thought, and graph-of-thought. Furthermore, RL-based methods are not compared here. I think a major question that this paper has yet addressed is why the structure has to be the way presented in this work. Maybe this structure is particularly more suitable for the personalization task?

(3) More insights into how each component of the framework elicits the behavior of the LLM would strengthen the contributions of the study.

(4) Applying the framework using another LLM backbone would strengthen the generalizability of the method.

### Questions
(1) How does the framework perform when compared to RL-based reasoning baselines?

### Soundness
2

### Presentation
3

### Contribution
2
