# GlucoBench: Curated List of Continuous Glucose Monitoring Datasets with Prediction Benchmarks

- Decision: Accept (poster)
- Scores: 5, 6, 8

## Abstract
The rising rates of diabetes necessitate innovative methods for its management. Continuous glucose monitors (CGM) are small medical devices that measure blood glucose levels at regular intervals providing insights into daily patterns of glucose variation. Forecasting of glucose trajectories based on CGM data  holds the potential to substantially improve diabetes management, by both refining artificial pancreas systems and enabling individuals to make adjustments based on  predictions to maintain optimal glycemic range. Despite numerous methods proposed for CGM-based glucose trajectory prediction, these methods are typically evaluated on small, private datasets, impeding reproducibility, further research, and practical adoption. The absence of standardized prediction tasks and systematic comparisons between methods has led to uncoordinated research efforts, obstructing the identification of optimal tools for tackling specific challenges. As a result, only a limited number of prediction methods have been implemented in clinical practice.  

To address these challenges, we present a comprehensive resource that provides (1) a consolidated repository of curated publicly available CGM datasets to foster reproducibility and accessibility; (2) a standardized task list to unify research objectives and facilitate coordinated efforts; (3) a set of benchmark models with established baseline performance, enabling the research community to objectively gauge new methods' efficacy; and (4) a detailed analysis of performance-influencing factors for model development. We anticipate these resources to propel collaborative research endeavors in the critical domain of CGM-based glucose predictions. Our code is available online at github.com/IrinaStatsLab/GlucoBench.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents a curated list of continuous glucose monitoring (CGM) datasets with a list of standard tasks for the problem of CGM both in terms of predictive accuracy and uncertainty quantification. It also provides benchmarking results using classic to modern methods on the datasets. Detailed analysis and discussion of the impacts of different factors on performance are provided.

### Strengths
1. The authors provide systematic benchmark results of different CGM models on multiple CGM datasets. Their evaluation methods seem to be reproducible and fair.
2. They provide diverse analyses of the benchmarking results, including the impact of dataset size, generalization of the models, and impact of time of day.

### Weaknesses
1. The writing of some parts is not clear and easy to understand. For example, when describing Task 2, it is not clear what “uncertainty” actually means here.; the caption of Figure 3 is also not clear enough about what the “green block” refers to with more than one green element in the figure; on Page 8, the authors did not provide an exact definition of in-distribution and out-of-distribution test, making the readers have to guess what it means by the authors.
2. I am concerned about the level of contribution of this work. Many of the conclusions of the paper are not novel. For example, people should already know “Healthy subjects demonstrate markedly smaller errors” and “using simpler shallow models when data is limited”.

### Questions
1. What is the motivation for some analysis you did in the paper? For example, what is the motivation for exploring the impact of time of day on accuracy? Why should researchers care about the analysis you did in Section 5, especially under the context of CGM?
2. How is this work different from other review papers mentioned in related works?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The manuscript analyses continuous glucose monitoring (CGM) data and glucose prediction models for diabetes management. It addresses the challenges associated with accurate glucose prediction, evaluates various models, and offers a curated collection of public CGM datasets. The paper focuses on two main tasks: (1) enhancing predictive accuracy and (2) improving uncertainty quantification in glucose prediction. It discusses the impact of dataset size, patient composition, and covariates on model performance and generalization. The study emphasizes the need for cautious use of glucose predictions in diabetes management and proposes future research directions.

### Strengths
1) The paper addresses the important issue of model generalization to new patient populations, providing evidence of the challenges associated with individual-level variation.
2) The key strength of the manuscript is it presents a valuable resource for the diabetes research community by offering public CGM datasets, standardized tasks, benchmark models, and detailed performance analysis. It also provides valuable insights into the factors affecting glucose prediction, including dataset size, patient populations, and time of day. It highlights the varying impact of covariates on model performance across different datasets.
3) I believe the study follows rigorous principles of reproducibility, ensuring that results are consistent across different data splits and model re-runs. It also focuses on fair comparisons by considering out-of-the-box model performance.

### Weaknesses
Some terms and concepts in the manuscript, such as ARIMA, Latent ODE, and Bayesian optimization, may be challenging for readers without a deep background in machine learning and statistics. Also, the choice of benchmark models might not be exhaustive, and the paper could benefit from a more extensive discussion of the models' suitability for different scenarios.

### Questions
I recommend authors consider below suggestions/limitations:-
1) Are there any potential biases in the selection of the public datasets, and how might they impact the generalizability of the results? Authors should discuss about the quality of the selected CGM datasets assessed, and what criteria were used to determine data quality.
2) The paper discusses the impact of covariates on model performance. Can you provide insights into which specific covariates had the most influence on the predictions and whether their quality was uniform across all datasets? And, how does the decision to omit certain models due to their lack of support for covariates affect the overall model comparison?
3) Perhaps authors should consider including, at least, in the supplemental material about any potential risks or challenges in directly applying the findings from the study to clinical settings especially when translated into practical clinical applications for healthcare providers and individuals with diabetes.
4) The paper mentions not considering additional model-specific tuning for benchmark models. Could such optimizations significantly enhance the models' performance, and why were they excluded from the evaluation?
5)  How diverse were the patient populations in the selected datasets in terms of demographics, disease severity, or other relevant factors, and how might this diversity affect the results and generalization?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper demonstrates the initiative and efforts in creating a public benchmark data repository for research in continuous glucose monitoring (CGM), with a curated collection of diverse CGM datasets, popular data tasks, bench-marking protocol and baseline models performance comparisons from existing literature. Code repository is available as supplementary material.

### Strengths
- Originality: the paper demonstrates an initiative in public data repository of ML application in CMG research
- Significance: The initiative of creating a public data repository of CGM research documented in the paper is of great value to the both clinical and ML research communities for experiment reproduction, bench-marking new methods and potential application adoption.  
- Quality: The paper is well-written with inspiring research questions. Model comparisons are performed on multiple datasets with inspiring research questions and discussion on results in part 5.
- Clarity: The paper is well-organized with problem formulation, related work, dataset and data tasks description, benchmarking protocols and detailed discussion.
- Code repository is provided with the paper submission

### Weaknesses
- Cross validation results will be preferred than simple train/test/validation splitting in benchmark model performance comparison.
- Automated machine learning (AutoML) could also help in benchmarking performance across all datasets with more ML models and pre-processing pipelines.

### Questions
1. How will new datasets be added to the repository?
2. How could other researchers contribute to the repository?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent
