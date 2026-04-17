# Rethinking Knowledge Distillation: A Data Dependent Regulariser With a Negative Asymmetric Payoff

- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Knowledge distillation is often considered a compression mechanism when judged on the resulting student's accuracy and loss, yet its functional impact is poorly understood. In this work, we quantify the compression capacity of knowledge distillation and the resulting knowledge transfer from a functional perspective, decoupling compression from architectural reduction, which provides an improved understanding of knowledge distillation. We employ hypothesis testing, controls, and random control distillation to understand knowledge transfer mechanisms across data modalities. To rigorously test the breadth and limits of our analyses, we explore multiple distillation variants and analyse distillation scaling laws across model sizes. Our findings demonstrate that, while there is statistically significant knowledge transfer in some modalities and architectures, the extent of this transfer is less pronounced than anticipated, even under conditions designed to maximise knowledge sharing. Notably, in cases of significant knowledge transfer, we identify a consistent and severe asymmetric transfer of negative knowledge to the student, raising safety concerns in knowledge distillation applications. Across 18 experimental setups, 9 architectures, and 8 datasets, our findings show that knowledge distillation functions less as a compression mechanism and more as a data-dependent regulariser with a negative asymmetric payoff.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper discusses the negative aspects of knowledge distillation (KD). Through various observations, the authors argue that KD itself is merely a regularizer and might even introduce detrimental tradeoffs. They compare naive KD with Random Control Distillation (RCD), suggesting that simply injecting noise into labels might be sufficient. Furthermore, they highlight KD's drawbacks – such as backdoor transfer or significant transfer of false positives (errors) – asserting that because these phenomena occur consistently, the utility of KD needs reconsideration.

### Strengths
This paper addresses important cautionary points regarding the use of KD.

### Weaknesses
Frankly, I feel that most of the claims are largely exaggerated. For instance, the RCD methodology doesn't differ significantly from the randomized teacher experiments they mentioned. Moreover, the adversarial transfer experiment ultimately boils down to whether backdoors are transferred, a claim already known, particularly in the context of LLMs.

Additionally, I find it difficult to agree with the issue raised about false positives (error transfer). The difference between the SIDDO baseline's accuracy and the teacher's true accuracy essentially represents the ceiling for improvement in correct prediction agreement. Conversely, for incorrect predictions, the teacher has already made errors, and the potential for the student to mimic these errors represents a much less restricted setting (a higher or less relevant ceiling).

Furthermore, learning from incorrect predictions (dark knowledge) is part of KD's original intention. Confidence-based methodologies, which avoid learning from incorrect predictions due to the problems the authors point out, already exist. It's difficult to agree that their claims are truly novel.

Also, their methodology was applied only to very small datasets. They should demonstrate whether their claims hold true on at least ImageNet.

### Questions
see weakness.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the mechanism of Knowledge Distillation (KD), arguing that its benefits stem from as being a data-dependent regularizer. Through controlled experiments, the authors demonstrate that the performance gains from KD can be replicated even with randomized teacher outputs, suggesting that the primary advantage stems from the regularization effect of soft labels. Another finding is a negative asymmetry that students inherit teacher errors more readily than its correct knowledge, meaning flawed teacher predictions are disproportionately amplified. The work argues that auditing teacher quality is essential, as applying KD to an imperfect model can significantly compromise the safety and reliability of the student.

### Strengths
1. The experiments are conducted based multiple modalities, architectures and datasets.
2. The mechanism of knowledge distillation (KD) is of great interest to the community because it involves model compression, representation learning, and transfer of inductive biases between deep models.
3. The negative asymmetric error transfer is an interesting finding.

### Weaknesses
1. Although multiple datasets are used, whether the conclusion can be generalized to huge teacher/student language models needs further verification.
2. The claim that KD functions as a data-dependent regularizer is not novel, as existing studies have already established its connection to label smoothing (Random Control Distillation in this paper).
3. While the transfer of teacher errors is concerning, the paper does not deeply examine why this happens and how to prevent it from happening.

### Questions
1. Most of the models adopted in this paper are pretty small. How are the experiment results on large models?
2. How can we prevent the teacher model from transferring bad knowledge to the student?
3.The temperature parameter has a significant impact on KD performance. How does the choice of temperature affect the validity of the paper’s conclusions?

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
4

### Summary
Knowledge distillation has become a popular paradigm for compressing larger models into smaller models. This paper aims to improve our understanding of knowledge distillation through experiments in controlled settings, challenging the framework of knowledge distillation as a framework of knowledge transfer. The paper employs randomized control trials to study distillation across data modalities, i.e., the teacher replaced by noise that is subsequently fed to the student through the loss function. Then, they compute several alignment metrics between student and teacher, such as activation distance, JS divergence, prediction disagreement, etc., and also accuracy for these two sets of experimental setups. Experiments are performed across multiple experimental setups architectures, and datasets, aiming to show that knowledge distillation functions less as a compression mechanism and more as a data-dependent regulariser with an asymmetric payoff.

### Strengths
Strengths

-- Several experimental setups have been considered in the paper. Experiments span multiple datasets and architectures, using compute worth more than a thousand hours. 

-- Fig. 1 is nice and clarifies the setup well.

-- The paper has also studied multiple modalities. The nuances of knowledge distillation in different modalities are interesting.

-- Adversarial transfer in language models, some mutual information-based measures, and distillation scaling laws have been included.

### Weaknesses
Weaknesses

-- While the extensive experiments are quite appreciated, I have some concerns/confusion about the main claim of the paper and the conclusions being drawn from the experiments. The terms "assymetric payoff" , "knowledge transfer", and "negative knowledge transfer" are quite confusing. Is it sufficient to contend that knowledge distillation does not transfer knowledge since the similarity measures do not increase? It is an optimization with two disagreeing terms anyway, and can be sensitive to the choice of hyperparameters. I do not fully understand the main argument about assymetric payoff and why it is not knowledge transfer.

-- The metrics used to compute "knowledge transfer" simply rely on the empirical alignment between teacher and student. It is unclear to me why similarity/alignment between teacher and student alone is a good measure of knowledge transfer since this doesn't consider anything specific about the task at hand. Separately, accuracy is only task-specific and doesn't consider both teacher and student together. Shouldn't an ideal measure of knowledge transfer capture what information about the task is transferred from teacher to student? Some information-theoretic metrics are introduced, but they are still analogous to either direct similarity/alignment or to accuracy. The measures are not about the task-specific alignment/similarity between teacher and student.

--  In setups where randomized control distillation has higher accuracy, how is the performance of the student without distillation at all (regularizer=0)? It seems the accuracies in these cases are still kind of close, e.g., 0.952, 0.954, and 0.957. Similarly, 0.605, 0.604, and 0.607. Could the small jump in randomized control distillation be from some small overfitting that is avoided? 

-- Some of these settings seem to be cases where distillation itself might not make much difference (unless I am mistaken). What about setups where students with and without distillation show a big gap in accuracy? Please point me to it if already done.

-- How to know if the student has the full capacity as the teacher? What is the measure of student capacity?

### Questions
Q1. I am confused about the main claim. Why is asymmetric payoff wrong/surprising and not knowledge transfer?

Q2. Why is similarity between teacher and student a good measure of knowledge transfer? It will be interesting to consider measures that capture task-specific alignment/similarity rather than just similarity.

Q3. What about setups where students with and without distillation show a big gap in performance? Please point me to it if already done.

Q4: How to know if the student has the full capacity as the teacher? What is the measure of student capacity?

### Soundness
2

### Presentation
2

### Contribution
2
