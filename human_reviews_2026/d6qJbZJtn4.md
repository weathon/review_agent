# AN ITERATIVE PROMPTING FRAMEWORK FOR LLM-BASED DATA PREPROCESSING

- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Data preprocessing plays a crucial role in machine learning, directly impacting model convergence and generalization, especially for simple yet widely used linear models. However, preprocessing methods are diverse, and there are no deterministic rules for selecting the most suitable method for each feature in a dataset. As a result, practitioners often rely on exhaustive manual searches, which are both time-consuming and costly.
In this paper, we propose an LLM-based iterative prompting framework that automates the selection of preprocessing methods. Our approach significantly reduces the number of iterations required to identify effective preprocessing strategies, thereby lowering human effort. We conduct an ablation study to analyze the contribution of each design component and provide extensive empirical evaluations. Results show that our method matches or surpasses baselines while substantially improving efficiency. The discovered preprocessing methods also accelerate training—either by improving convergence speed, enhancing generalization performance, or both.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes LLM-Presto, an iterative prompting framework that automates data preprocessing with large language models. The method iteratively generates and refines preprocessing strategies based on structured feedback from validation results under a fixed training-epoch budget. It grounds the design in theoretical analysis of convergence and generalization for linear models and evaluates performance on six tabular datasets, showing improved efficiency and comparable or better accuracy than several prompting-based baselines.

### Strengths
- The paper is well-motivated and theoretically grounded, offering a clear connection between data preprocessing, conditioning, and generalization behavior, which gives the approach conceptual depth beyond empirical heuristics.
- The experimental evaluation is carefully structured, with ablation studies, baseline comparisons, and reproducibility details that enhance credibility and provide solid evidence for the efficiency gains claimed.

### Weaknesses
- The methodological novelty is limited, as the framework mostly combines existing iterative prompting and feedback techniques without introducing substantially new algorithmic ideas.
- The experiments are restricted to simple linear models and small-scale six tabular datasets, which constrains the generality and practical relevance of the conclusions.
- The analysis of performance variance and stability remains shallow, with oscillations noted but not rigorously explained or quantified.
- Finally, the paper’s presentation could be improved by adding clearer figures or concrete examples of LLM-generated preprocessing strategies to make the iterative process more intuitive and interpretable.

### Questions
See #Weakness

### Soundness
3

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
This paper proposes LLM-Presto, an iterative prompting framework that uses LLMs to design and refine ML data preprocessing pipelines. LLM-Presto integrates LLM-generated preprocessing suggestions with few-epoch downstream validation feedback. Experiments show that LLM-Presto improves downstream performance with lower total training cost.

### Strengths
- **Significance**: The paper demonstrates a practical path toward integrating LLM reasoning into data-centric AutoML, showing potential for cost-efficient and interpretable pipeline design.
- **Clarity**: The framework is clearly structured and easy to follow.

### Weaknesses
1. **Limited methodological novelty**. The framework mainly combines existing elements, such as LLM prompting, few-epoch evaluation, and iterative refinement, without introducing a fundamentally new mechanism. The theoretical discussion on convergence and generalization is motivating but doesn’t serve as an integral part of the method.

2. **Insufficient experiment**. The experiments lack diversity on ML algorithms, so it’s difficult to judge whether the framework generalizes beyond specific settings. Also, the method has not been compared with traditional AutoML methods to show competitive performance.

3. **Presentation issues**. The term *iteration* is used inconsistently (sometimes referring to training epochs, other times to LLM feedback rounds), which causes confusion. The lack of explicit description of downstream models and implementation details, e.g. prompts,  also limits reproducibility.

### Questions
1. What exactly are the “losses during the search epoch” in Figure 1(d/e/f). Can you provide detailed explanations?
2. Efficiency is measured only by training cost. Have you taken LLM inference latency into consideration, since it may affect the practical runtime advantage?
3. How are the textual preprocessing suggestions from the LLM grounded into executable transformations?
4. Other questions are reflected in weaknesses.

### Soundness
2

### Presentation
2

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
This paper proposes LLM-Presto, an iterative prompting framework to automate data preprocessing for (primarily linear) models. The system uses an LLM to generate candidate preprocessing strategies based on dataset statistics. These strategies are then evaluated in a leakage-safe manner using a fixed, low-epoch training budget (e.g., k=5) as a proxy objective. The resulting validation loss is fed back to the LLM, which iteratively refines the strategy. Empirical results on six benchmark datasets show this method finds preprocessing pipelines that achieve superior final model performance and converge faster than baselines like zero-shot, CoT, and self-refine.

### Strengths
- Novel application of LLM on automatic pre-processing selection. The core contribution is the design of an LLM-driven search that is grounded by empirical, low-cost evaluation to find the best data pre-processing strategy. 
- The method is shown to outperform multiple baselines, including CoT, Zero-shot, etc. The approach improves both model performance and sample efficiency.

### Weaknesses
- I think the scope is limited. The most significant limitation, which the authors correctly acknowledge, is the focus on linear models. While this is a fine starting point and allows for clean analysis (e.g., via condition numbers), the framework's utility for more complex non-linear models (e.g., Gradient Boosting, Deep Neural Networks) is unproven.
- Although the application is inspiring, but the methodology is not. The propose framework is very standard, so I consider the scientific contribution is also limited. 
- The method appears to very sensitive to budget $k$. The selection of 5 is supported by empirical evidence but the root cause is not thoroughly explained. How can one know the proper budget when generalizing to a new task?

### Questions
- The finding that a larger budget (k=20) led to a premature focus on PCA is interesting. What is the root cause? Can one prevent it through prompting?

### Soundness
2

### Presentation
3

### Contribution
2
