# CoDBench: A Critical Evaluation of Data-driven Models for Continuous Dynamical Systems

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6, 6

## Abstract
Continuous dynamical systems, characterized by differential equations, are ubiq- uitously used to model several important problems: plasma dynamics, flow through porous media, weather forecasting, and epidemic dynamics. Recently, a wide range of data-driven models has been used successfully to model these systems. However, in contrast to established fields like computer vision, limited studies are available analyzing the strengths and potential applications of different classes of these models that could steer decision-making in scientific machine learning. Here, we introduce CoDBench, an exhaustive benchmarking suite comprising 11 state-of-the-art data-driven models for solving differential equations. Specifically, we comprehensively evaluate 4 distinct categories of models, viz., feed forward neural networks, deep operator regression models, frequency- based neural operators, and transformer architectures against 8 widely applicable benchmark datasets encompassing challenges from fluid and solid mechanics. We conduct extensive experiments, assessing the operators’ capabilities in learning, zero-shot super-resolution, data efficiency, robustness to noise, and computational efficiency. Interestingly, our findings highlight that current operators struggle with the newer mechanics datasets, motivating the need for more robust neural oper- ators. All the datasets and codes are shared in an easy-to-use fashion for the scientific community. We hope this resource will be an impetus for accelerated progress and exploration in modeling dynamical systems. For codes and datasets, see: https://anonymous.4open.science/r/cod-bench-7525.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes an benchmarking suite comprising 11 state-of-the-art data-driven models for solving differential equations. Authors conduct extensive experiments, assessing the operators’ capabilities in learning, zero-shot super-resolution, data efficiency, robustness to noise, and computational efficiency.

### Strengths
1. Building benchmark is important for research community.
2. Authors conduct a lot of experiments.

### Weaknesses
1. I feel like more contributions should be made to meet the ICLR requirement. Authors only compare 11 methods in standard benchmarks. I suggest authors should add more methods or new datasets for a more comprehensive benchmark. 
2. More observations and conclusion should be made throughout the experiments. Now the observations is lack of insights. 
3. A lot of dynamical system modeling methods are missed for comparison, e.g., [1]. More video prediction methods can be also compared. 

[1] Solving High-Dimensional PDEs with Latent Spectral Models ICML 23

### Questions
See above.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents CoDBench, a computational benchmarking suite which contains 11 state-of-the-art data-driven models for solving differential equations. In addition, it provides eight benchmarking datasets for model evaluation. Using the computational suite the authors evaluate the models according to a division into four categories, according to their architecture. The comprehensive evaluation is used to draw key insights regarding the studied frameworks and datasets.

### Strengths
$\underline{\textrm{Originality}}$: instead of suggesting a novel approach this paper aims to analyze the strengths and potential applications of existing data-driven tools to study continuous dynamical systems. In a sense optimizing the usage of existing frameworks. 

$\underline{\textrm{Quality}}$: the paper is well written. It provides a detailed introduction to the field of data driven approaches on PDEs as well as an elaborate description of benchmark results and conclusions. 

$\underline{\textrm{Clarity}}$: all sections, from introduction to the concluding insights, are clearly presented.

$\underline{\textrm{Significance}}$: the significance of the paper is in providing a resource for exploration of dynamical systems as well as insights to optimal usage of assessed frameworks.

### Weaknesses
1. CoDBench package: the code package is presented as a major contribution which shall serve as a resource for studying dynamical systems. However, the code that is currently available (https://anonymous.4open.science/r/cod-bench-7525) is lacking a basic README with minimal guidelines for a user interested in using the package. It will be beneficial to accompany the code with proper documentation as well as detailed examples as common for many scientific code packages (see next point). 
2. Related work: it would be beneficial for the authors to relate to existing code alternatives and or attempts to study data-driven models. For example, DeepXDE (lu et al. 2021) is a code package that contains most of the components of CoDBench and further allows more flexibility in terms of data construction. 
3. Overall contribution: the authors present as major contributions the analysis of Super-resolution as well as out-of-distribution tasks (stress and strain) however both are presented in earlier works for studying data-driven models and benchmarked against baselines (e.g. Fanaskov et al. 2022 and Rashid et al. 2022). Hence while it is beneficial to analyze these at a broader scale over all 11 models the scope of this contribution is limited. Following this, the key insight, limitations, and future work suggested appear minimal in terms of scope and practicality.
4. Minor: Figure 1 is never referenced. It will be beneficial to change the coloring of optimal values in tables and or underline the second best to have better discrimination between second and third.

### Questions
1. Would it be possible to improve the provided package in light of the comment above?
2. Could the authors clarify what they see as the major contributions in light of the comments?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes an exhaustive study of several machine learning algorithms of various kind (neural operators, frequency based decomposition, and more traditional approaches) on several types of dataset involving both fluid dynamics and mechanics. 

The study not only focuses on the prediction capabilities of the tested algorithms but also studies their ability in selected related tasks such as super resolution and out-of-distribution prediction. 

This work cannot be considered with high technical contributions yet it depicts an interesting snapshot of the current literature on several tasks related to dynamical system estimation and is in that sense valuable.

### Strengths
This work is a systematic study providing practitioners with a clear view of the tested algorithms capacities to fit different problems and what limit one should expect when tackling another task. 

The presentation is clear and the paper is easy to follow and the auxiliary tasks are in my opinion interesting and valuable to test the model capacities.

### Weaknesses
The main limitation of the present paper is the lack of technical novelty and limited novel contribution to the field. 

Yet, such a paper can define an interesting milestone of the field depending on the release and quality of the code and dataset provided alongside the paper.

For other concerns see questions.

### Questions
1. What training methodology is used for each algorithm ? I find section 2.3 quite general and it is difficult to understand what physical quantity is estimated / trained on by each algorithm.

2. Can the authors comment on the limitation of their study for the ood evaluation. PDEs solution can exhibit very different behavior with the same equation varying the initial condition, notably for fluid dynamics data. Such a test could strengthen the analysis proposed by the authors.

3. The error in the super resolution settings seems to explode. Can the authors comment on the results they report in the paper ?

4. While it is very difficult to be exhaustive in such a study, testing all the literature is unfeasible, I strongly encourage the authors to include an extensive discussion on why they chose the selected models or at least why they chose not to select other models. Such a discussion would be a valuable addition to the paper.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper provides a benchmark for data-driven dynamic modeling. Evaluating 4 types of the model - feed-forward neural networks, deep operator regression models, frequency-based neural operators, and transformer architectures, on 8 benchmark datasets - evaluate the robustness to noise, computational efficiency, and data efficiency of the model, with opensource data and codebase.

### Strengths
This work performs quite a comprehensive quantitative analysis of different models on a wide range of datasets, testing model capability over data efficiency, run time, training time, prediction accuracy, and super-resolution accuracy. It is nice to have a benchmark of different model on regular girds.

### Weaknesses
This is a benchmark paper on simulation on a regular grid. I am not sure if I missed this, the discussion on performance/adaptability of different models on irregular grids is not included. I found the overall analysis and insights comprehensive but also a little bit simple with missing key discussion on a simulation where irregular grids are required.

### Questions
I am curious about more details on how hyperparameters are set for different model training across dataset.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
