# Improving Active-Learning Evaluation, with Applications to Protein-Property Prediction

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
We highlight that current evaluations of active-learning methods often fail to reflect important aspects of real-world applications, giving an incomplete picture of how methods can behave in practice. Most notably, evaluation problems are commonly constructed from heavily curated datasets, limiting their ability to rigorously stress-test data acquisition: even the worst acquirable data in these datasets is often reasonably useful with respect to the task at hand. To address this we introduce Active Learning on Protein Sequences (ALPS), a set of problems constructed to test key challenges that active-learning methods need to handle in real-world settings. We use ALPS to assess a number of previously successful methods, revealing a number of interesting behaviours and methodological issues. The ALPS codebase serves to support straightforward extensions of our evaluations in future work.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a protein-property-based benchmark aimed at exposing weaknesses in current active learning evaluations that rely on overly curated datasets, offering more realistic yet domain-specific test conditions.

### Strengths
1.The paper raises a highly insightful and underexplored issue, that standard active learning benchmarks are often constructed from carefully curated datasets lacking truly non-informative samples. This leads to evaluations that may not reflect real-world conditions, where many samples are noisy, redundant, or irrelevant. The authors’ recognition of this gap and their effort to design ALPS as a more realistic, less curated benchmark represent a meaningful and original contribution to improving the validity of active learning evaluation.
2.Although the benchmark is built within a specific biological domain, the paper successfully designs several realistic and broadly relevant challenges such as label imbalance, redundant inputs, and acquisition restrictions. These conditions effectively test how active learning methods behave under real-world constraints. The protein-property setting provides a natural environment where such challenges can be observed and controlled, making ALPS a useful complement to existing benchmarks. It may not yet serve as a domain-independent standard, but it is a solid and practical step toward more reliable and diagnostic active learning evaluation.

### Weaknesses
1.While the paper insightfully identifies that existing active learning benchmarks rely on overly curated datasets, it does not quantitatively demonstrate that its proposed ALPS benchmark avoids the same issue. The authors only provide qualitative reasoning such as emphasizing that protein-property prediction data are experimentally derived and that ALPS minimizes filtering, but offer no measurable evidence (e.g., label noise level, data redundancy ratio, or informativeness variance) to substantiate the claim that ALPS is genuinely less curated. As a result, the paper does not fully address, in a quantifiable way, the very evaluation bias it aims to highlight.
2.The chosen experimental domain, protein-property prediction. is overly narrow to justify such a broad claim to me. The authors use this domain as evidence that AL evaluation should move away from curated datasets, yet this reasoning does not generalize: in many application areas, such as medical imaging or safety perception, highly curated test sets are necessary to ensure fairness and reliability rather than bias. The paper therefore conflates a domain-specific insight with a general methodological critique; without cross-domain evidence or domain-agnostic metrics, the claim that ALPS improves general AL evaluation remains under-supported.

### Questions
Could you provide some quantitative evidence or proxy metric (e.g., redundancy ratio, noise estimation, task-relevance measure) to support that the ALPS datasets are indeed less curated than standard benchmarks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors posit that current Active Learning (AL) evaluations are done on unrealistic datasets and therefore skew the reported performance on acquisition functions. The main drawback of current datasets is the highly curated nature of the data samples, where even the most uninformative sample at any point is still well correlated with the target variable.
To remedy this situation, the authors propose ALPS, a protein-property prediction benchmark for AL, comprised of 6 protein-datasets that are evaluated on 3 novel tasks (in addition to standard batch AL).
In addition to describing the construction of each task, the authors mainly provide results for two Bayesian acquisition functions (BALD, EPIG) on their benchmark.

### Strengths
- Valuable collection of new and novel tasks for AL that significantly expand the types of problems AL is tested on
- Large bank of implemented encoder models, which should serve as a role-model for other benchmark papers in and across AL

### Weaknesses
- TypiClust is a bad representative of diversity-based methods for the tested budgets, as it was originally developed for ulta-low budgets and has shown lacking performance in other setups (e.g. Werner et al)
- The "Redundant" Task is fundamentally ill-posed: Constructing a machine learning task with 3 classes in the training set, but only 2 in the test set is scientifically not sound. This is mostly a problem with the description of the task. As an alternative, the authors should consider to stick to the original setup of (Bickford et al) that is defined on the basis of class imbalance. This would result in the following datasets (based on App. C.2.4): D_pool (86.8% neither, 12.4% Class 0, 0.7% Class 1); D_test (86.8% neither, 12.4% Class 0, 0.7% Class 1); D* (0% neither, 50% Class 0, 50% Class 1). All methods not taking advantage of D*, drastically fall behind EPIG in this setup.
- Missing technical details: (i) number of repetitions for each experiment, (ii) training setup (data augmentation, cold vs. warm-starts, seeding, etc.). Since this paper is a benchnark, we argue would that these details are especially important

### Questions
- Where are the results for BADGE, BAIT and ProbCover? The paper states in line 254 that these methods are implemented. All three methods represent strong acquisition functions in the field, so it is highly relevant, whether they exhibit the same shortcomings. Especially ProbCover is a way better fit to the tested budgets than TypiClust (see weakness as well).
- Some of your ablation studies - especially Fig. 3 - seem to indicate a structural problem in your experimental setup, as the behavior of the acquisition does not seem to correspond to the imbalance ratio. What number of repetitions was employed in the computation of your results and have you checked you inter-experiment variance? (see Werner et al for analysis on the number of repetitions needed for stable AL experiments) 

[Brickford et al] Bickford Smith, Kirsch, Farquhar, Gal, Foster, & Rainforth (2023). Prediction-oriented Bayesian active learning. International Conference on Artificial Intelligence and Statistics. \
[Werner et al] Werner, Burchert, Stubbemann, & Schmidt-Thieme (2024). A cross-domain benchmark for active learning. Conference on Neural Information Processing Systems.

### Soundness
2

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
4

### Summary
This paper introduces Active Learning on Protein Sequences (ALPS), a set of active learning benchmarks that are designed to incorporate challenges that are prevalent in real-world settings. These benchmarks can help to address issues that active learning methods may face in practical use cases, issues which might otherwise go unnoticed if the methods are usually tested on better curated datasets. The paper also presents experiment results of testing active learning methods on the ALPS dataset, the results demonstrate that some popular approaches (such as BALD and EPIG) perform suboptimally under certain conditions and exhibit issues such as uncertainty miscalibration, sensitivity to class imbalance etc. Overall, the new benchmark would allow for a more robust evaluation and can help drive the development of improved active learning methods.

### Strengths
- The paper introduces a novel benchmark that addresses the limitation of existing heavily curated benchmarks datasets by including challenges commonly encountered in real-world settings. This allows for more realistic and robust evaluation of AL methods for practical applications, and empirical experiments demonstrate how it can reveal previously unknown failure modes in popular AL methods. 
- The empirical experiments are very thorough, evaluating multiple encoders, prediction heads, acquisition methods etc. across multiple tasks. Components appear to be implemented in a modular setup which allows for ease of testing specific parts of the AL pipeline.

### Weaknesses
- There is a lack of data quality evaluation for this benchmark dataset. There is little discussion and evaluation regarding how reliable the labels are, or the level of ambiguity / noise in the dataset. It is crucial for benchmark datasets to ensure the quality of its data and labels, hence a more thorough evaluation of the reliability of the data would strengthen the benchmark’s credibility and utility.
- While the benchmark supports a large array of encoders / predictors / acquisition methods, the main evaluations are limited (focused mainly on BALD, EPIG, TypiClust and random), resulting in many other popular AL methods not being fully explored and making it harder to draw conclusions about how the current methods behave in this more challenging, real-world-like setting
- Protein sequencing is a highly domain specific task, and it is unclear whether the observed challenges that various AL methods face in this benchmark would transfer to other tasks of modalities. This potentially limits the generalizability of the insights gained from this benchmark.

### Questions
- Some tasks in this benchmark were converted from regression tasks into a binary classification tasks using constant thresholds. I am curious if the authors have any insight into if this binarization affects the quality of the data (might introduce ambiguity around the decision boundary). In particular, in the cases where thresholds were used to create class imbalance, I wonder if the performance of the tested methods were influenced by the imbalance, or the label transformation. Could the authors isolate the two contributing factors here to better understand the consequences of each change?
- Could the authors provide more information / rationale behind their dataset design decisions? For example, why were specific imbalance ratios or batch sizes chosen, and how these might have an impact on the observed behaviours of the tested methods.

### Soundness
3

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
The paper argues that many active learning evaluations use heavily curated datasets and therefore under-stress data acquisition. It proposes a new benchmark suite derived from experimental protein datasets, ALPS to better reflect messy pools, task adaptation, class imbalance, acquisition restrictions, and batching. Using ALPS, the authors benchmark several acquisition strategies with pretrained protein encoders and various heads.

### Strengths
1. The application of active learning to protein-property prediction is both novel and important. This work commendably addresses a challenging and under-explored domain, moving beyond the standard.

2. Clear formalization of evaluation goals via frequentist risk and explicit factors (task, loss, pool, p_train, p_eval, model, budgets).

3. Codebase and datasets are provided for reproducibility. This represents a contribution to the community, facilitates reproducibility, and allows for future extensions.

### Weaknesses
1. The empirical evaluation is not comprehensive. The authors state that the codebase implements 12 acquisition methods, including widely-used gradient-based methods like BADGE and BAIT. However, the core experimental analysis (Section 6) focuses almost exclusively on BALD and EPIG, with TypiClust and random sampling as baselines. Given that BADGE, in particular, is a strong and practical baseline in many deep AL settings, its omission from the main comparisons (Figures 2-8) is a significant flaw and undermines the comprehensiveness of the benchmark.

2. A central motivation of the paper is to improve practical evaluation, highlighting the cost of labeling proteins. However, the analysis of "cost" is limited to the number of acquired labels. It completely omits the computational cost (e.g., wall-clock time, FLOPs, or memory footprint) of the acquisition step itself. Since the search space of protein is prohibitively large, and the prediction of the properties is also expensive, it is necessary to discuss about the computational cost in my view. Methods like EPIG, BALD, and BADGE have vastly different computational overheads. To truly evaluate practical utility, the benchmarks must present performance as a function of both label cost and computational cost.

3. The reliance on binarization as the primary task formulation is a limitation. While the authors investigate varying thresholds in the ALPS-Unbalanced tasks, the binarization in the ALPS-Core tasks (based on a single wild-type reference value) can be arbitrary and may not reflect the true scientific objective (e.g., finding a variant with maximum fluorescence). The paper would be stronger if it either (a) included a sensitivity analysis for this threshold choice in the Core tasks or (b) directly evaluated performance on the native regression task, which the paper acknowledges but does not explore.

4. I think the motivation is questionable. On one hand, many existing AL benchmarks are also derived from real world, and contains useless even harmful data; on the other hand, ALPS is still a retrospective simulation on a specific and static pool, some of its crafted variants, e.g., ALPS-Unbalanced, ALPS-Redundant, may not really reflect the real scenarios. A truly real-world setting, particularly in protein engineering or drug discovery, is often an optimization or search problem (e.g., "find the protein with the highest binding affinity") rather than a pool-based classification task. This objective is better modeled by Bayesian Optimization or an exploration-exploitation framework. Therefore, I think the problem of active learning evaluation has not been sufficiently addressed.

5. The paper's positioning and literature review almost entirely overlook the substantial, existing body of work in active learning for drug discovery and virtual screening. This field has its own established benchmarks, models, and practical settings and has extensively studied the exact problems the authors claim to motivate their work. The paper should not be comparing itself only to generic CV/NLP benchmarks but to the established practices in its own target domain.

### Questions
Please see weakness.

### Soundness
3

### Presentation
3

### Contribution
2
