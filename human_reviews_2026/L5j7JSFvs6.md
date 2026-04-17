# PIFE: Progressive Insight driven Feature Engineering via Multimodal Reasoning

- Decision: Reject
- Scores: 4, 8, 6, 4

## Abstract
Despite significant advances in Automated Machine Learning (AutoML), one of its persistent blind spots remains the automation of data-centric tasks such as exploratory data analysis (EDA), contextual insight extraction, and feature engineering. These steps-often more critical than model selection itself-are still largely manual, domain-specific, and reliant on human intuition. Existing automated feature engineering (AutoFE) techniques either rely on rigid transformation sets or complex optimization strategies that struggle with interpretability and fail to leverage the rich, visual cues that guide human decision-making. In this work, we introduce PIFE: Progressive Insight driven Feature Engineering via Multimodal Reasoning; a novel AutoFE framework that employs multimodal language models as collaborative agents in an iterative pipeline. PIFE systematically performs automated EDA, generating statistical summaries and visualizations that are jointly interpreted through text–vision reasoning. These multimodal insights inform the synthesis of candidate transformations, represented as symbolic programs in executable Python code to ensure interpretability and reproducibility. By coupling iterative insight extraction with validation-driven refinement, PIFE produces high-quality, interpretable features that consistently enhance the performance of diverse predictive models, outperforming existing AutoFE baselines. Extensive experiments across diverse tabular datasets demonstrate the effectiveness and adaptability of our approach, paving the way for a new class of human-aligned, insight-aware AutoFE systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes an automated feature engineering framework that involves iteratively performing exploratory feature analysis and feature generation, guided by large language models. The method is evaluated in a number of tabular datasets, and ablation studies are performed.

### Strengths
- The approach is interesting, the idea that LLMs can guide EDA and extract features that are more effective at downstream tasks is explained well.
- The paper is written well and easy to follow.

### Weaknesses
- The use of a VLM is questionable to me, I think the method tries to mimic how a human would go about the process of exploratory data analysis and feature engineering. However, if the goal is to improve performance on downstream tasks, (which most of the evaluation focuses on) the benefit of using large models is their ability to understand complex relationships within the data, where generating plots would be unnecessary.
- Table 2 shows that in the experiments performed, the explanatory data analysis step provides little benefit, despite what I imagine is a large computational cost. Table 4 shows that the benefit of using such features in deep learning models is minimal.
- I feel that human in the loop feedback could benefit the method - allow a human to guide each iteration with insights gained from the automatic EDA. Additionally, insights from EDA are often not with the goal of improving classification, but understanding where the model will make mistakes or visualizing and comparing distributions of certain features.
- The claim that the model is more human-aligned than others is not proven, I believe human experiments would be necessary to say that this model is more human-aligned than others.
- I believe the evaluation focuses on the wrong aspect. In terms of EDA, if the authors could show that the data analysis produced is more human-like or more informative to humans than other methods, this would be a benefit of the proposed method.

### Questions
- What are the additional computational costs of each module in the method? I imagine that the cost is large compared to the very minimal performance gain.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes PIFE (Progressive Insight Driven Feature Engineering), which selects features by iteratively doing EDA (exploratory data analysis), making plots and analyzing the plots with LLM and VLM and code execution. Their method takes advatange of human-like reasoning process where humans look at data analaysis plots to decide about the results. Each iteration, they use feature importance from the random forest to decide which features to keep. They are the first to propose automated feature engineering framework that integrates textual and visual exploratory data insights into a unified, iterative pipeline. They compared with a comprehensive set of baselines and show that their method selects better features. They also show the effectiveness of EDA in an ablation comparing with removing EDA from their pipeline. Although it doesn't help two datasets with EDA, it helps most. They also tried to further use OpenFE on the end outputting features from PIFE, and find there only to be a small improvement, showing the effectiveness of PIFE.

### Strengths
1. The paper proposes a very nice and intuitive method (using insights from data exploratory phrase, and using an VLM to achieve that) to improve auto feature engineering.
2. The results show that PIFE is better than other methods.
3. The paper has comprehensive results to isolate effects of EDA and also how combining their method with OpenFE works.

### Weaknesses
1. The method replies on the model being an inherently interpretable model that exposes feature importance, which would not be true for deep learning models, although we could use some post-hoc methods to approximate the feature importance.

### Questions
1. How would the proposed method work if the underlying predictive model is not random forest but a neural network?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents PIFE, an AutoFE framework that leverages multimodal reasoning (LLMs + VLMs) to automate EDA, insight extraction, and feature generation for tabular datasets. PIFE iteratively generates statistical summaries and visualizations, interprets them, and synthesizes candidate features as executable Python code. Downstream model feedback is used to refine feature generation. Experiments across 22 datasets show PIFE outperforms existing AutoFE baselines in interpretability and context-awareness.

### Strengths
Novelty: First to tightly integrate textual and visual EDA insights in an iterative, feedback-driven AutoFE pipeline.
Interpretability: Features are generated as symbolic programs, ensuring transparency and reproducibility.
Empirical Rigor: Evaluated on diverse datasets, with fair comparisons and ablation studies.
Performance: Consistently outperforms baselines in both classification and regression tasks.
Generalization: Features transfer well to different downstream models and unseen datasets.
Open Science: Code and datasets are released for reproducibility.

### Weaknesses
Scope and Robustness of EDA
- The EDA routines in PIFE are primarily statistical and visual, focusing on distributions, correlations, and categorical/temporal trends. This scope is well-suited for standard tabular data, but may not generalize to domains requiring specialized EDA (e.g., time series, graphs, or highly unstructured data).
- Robustness of the approach depends on the diversity and quality of EDA routines. If the EDA is limited or misses important patterns, the generated features may be suboptimal.
- The paper notes that in some datasets, EDA-driven features do not improve performance, especially when original features are already strong or datasets are small.

Supported Visualizations
- PIFE supports common tabular visualizations: histograms, scatter plots, heatmaps, LOWESS curves, binned means, and categorical plots (bar, boxplots).
- The framework does not appear to support more advanced or domain-specific visualizations (e.g., time series plots, network diagrams).
- The quality and diversity of insights are limited by the types of visualizations produced and interpreted.

LLM Hallucinations and Reliability
- LLMs may generate plausible but incorrect or ungrounded features, especially if the EDA context is noisy or incomplete.
- The framework mitigates hallucinations via downstream validation (feature importance feedback, cross-validation, feature selection), but does not provide explicit mechanisms for detecting or correcting hallucinations at the insight or feature generation stage.
- No formal guarantees are provided against hallucinations; robustness is empirically assessed via multiple seeds and ablation studies.
- There is a risk of overfitting to spurious patterns or memorized solutions from prior competitions.

### Questions
See weaknesses

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces an AutoFE framework that incorporates automatic data analysis like EDA, leveraging LLMs and VLMs. A VLM generates statistical summaries of features from EDA plots that are used as the context in the LLM-based AutoFE process. The paper also explores the effects of different feature selection methods.

### Strengths
This paper suggests a good direction of incorporating the visual information of datasets in AutoFE. The paper presents comprehensive experimental results involving both traditional and deep learning downstream models. The experimental setup is explained clearly.

### Weaknesses
While it is beneficial to enrich the context information of LLM-based AutoFE, I do not think the automatic data analysis process has been well integrated. Algorithm 1 suggests that the process starts from scratch at each iteration, which is quite inefficient. The data analysis process also seems quite random, and there is no feedback mechanism to guide it. From the ablation study, the performance gain is limited. 

The presentation of the paper is not very clear in some parts especially Section 3. More detailed explanations may help. Algorithm 1 is a bit hard to follow and not positioned appropriately.

In the experimental section, the statistical significance of results is not reported. It would be great to also include a cost study of the framework.

### Questions
Does the framework represent feature transformations in code or RPN? Listing 3 seems to suggest that both are adopted. I think this is unnecessary and may create inconsistencies.

What VLM has been used in experiments?

In A.5.2, why are the parameters presented as ranges, different from A.5.1? 

Somehow the repository shows “the requested file is not found”  and the code is inaccessible.

### Soundness
2

### Presentation
1

### Contribution
2
