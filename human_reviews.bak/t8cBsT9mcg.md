# Classification with Conceptual Safeguards

- Decision: Accept (poster)
- Scores: 6, 8, 8, 6

## Abstract
We propose a new approach to promote safety in classification tasks with concept annotations. Our approach – called a *conceptual safeguard* – acts as a verification layer for models that predict a target outcome by first predicting the presence of intermediate concepts. Given this architecture, a safeguard ensures that a model meets a minimal level of accuracy by abstaining from uncertain predictions. In contrast to a standard selective classifier, a safeguard provides an avenue to improve coverage by allowing a human to confirm the presence of uncertain concepts on instances on which it abstains. We develop methods to build safeguards that maximize coverage without compromising safety, namely techniques to propagate the uncertainty in concept predictions and to flag salient concepts for human review. We benchmark our approach on a collection of real-world and synthetic datasets, showing that it can improve performance and coverage in deep learning tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a perspective on selective classification within deep learning using concepts. It suggests a strategy for balancing accuracy and coverage by abstaining from predictions in situations where errors can be costly. The proposed approach involves creating a concept bottleneck model, enabling the front-end model to use soft concepts and improving coverage and performance through concept confirmation. The paper presents techniques for handling uncertainty and pinpointing concepts for confirmation.

### Strengths
The paper exhibits good clarity in its articulation, with ideas clearly presented and structured in an organized manner. Exploring the integration of user feedback into ML models to enhance accuracy and ensure broad coverage is intriguing and holds significance for ML models in real life usage. Furthermore, the paper touches interpretability in machine learning, which is an important aspect for ML models in real life.

### Weaknesses
1. The abstract should be expanded to encompass key concepts that effectively summarize the paper's contributions. In the introduction, the authors emphasize the significance of interpretability and the challenges it poses in achieving high accuracy. By including these vital points in the abstract, the paper can provide a more comprehensive overview of its content and contributions.

2. Regarding the abstention process, it appears to be based on a prediction probability threshold, where if the probability is lower than the threshold, the prediction is abstained? How does it different from a decision threshold used by the models? Can authors clarify that?

3. In the results and discussion section, there's limited exploration and commentary on the impact of the solution on system accuracy, as seen in Table 2. Notably, the confirmation budget appears to have a limited effect on datasets like "noisyconcepts25" and "warbler" compared to others. The paper can delve into the reasons behind this discrepancy.

4. In real-world applications of this solution, questions about the ease of concept approval and handling conflicting user feedback arise. While these aspects may be considered out of scope, addressing them would be beneficial for evaluating the practicality of implementing this approach in real-world scenarios. This is particularly important when considering the potential challenges of user feedback and conflicting inputs in such applications.

Minor things:
Page 4, confirm. we —> replace . with comma
Section 4.2, Table Table 2 —> Table 2
Shouldn’t Table 2 rather be labelled as Figure 2?

### Questions
stated above

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose a classification system that uses the combined approaches of a conceptual bottleneck and abstaining outputs to increase the reliability of models. The conceptual bottleneck approach trains a classification model for each concept identified in the training data. The end-model is a classifier that uses the presence or absence of concepts to make the target classification. The abstain mechanism allows the end-model to abstain from prediction.  When the model is uncertain about the presence of a concept, it may query the user for confirmation, thereby increasing trust and performance. Concept uncertainies are propagated through the end-model by using concept identification model scores as probabilities and sampling over potential concept vectors. This also improves performance.

### Strengths
The three strengths of the proposed approach are a functional abstaining method, requests for confirmation, and uncertainty propagation. Together these methods raise a classification model to something that is more intelligent, capable of some corrective action when faced with unusual inputs.

### Weaknesses
1. The uncertainty propagation methodology doesn't seem computationally efficient.
2. The performance of the default classifier (always predict majority class, uniformly randomly abstain) ought to be included in Table 2. The default performance ought to always be presented when using accuracy as a performance metric.

### Questions
Can a deeper analysis of the consequences of abstaining be provided? Abstaining almost always improves average performance on the remaining predictions. Reporting the average is almost illusory, since those non-abstain predictions would have been correct or incorrect regardless. Rather, there is a real cost associated with refusing to provide an answer. The benefit is that the model reduces risk of error, but the costs are application dependent. How can we think about these costs in a constructive manner?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose the use of a concept bottleneck model as input for selective classification. Moreover, they propose a greedy algorithm to select concepts to be confirmed by human experts, with the objective to increase the coverage of the selective classifier while guaranteeing a minimum accuracy level of the selective classifier. They evaluate their method with competitive baselines using both synthetic and real datasets.

### Strengths
The work appears to be the first to use concept bottleneck models to capture the uncertainty of the entire model for selective classification. Moreover, the idea of getting human feedback to confirm concepts to improve selective classification is quite interesting and adds to the increasing literature of human-in-the-loop algorithms. 

The paper is very well organized, has a clear structure, and is nicely written. The authors clearly state their contributions as well as the assumptions of their method. They also provide a detailed description of the experimental setup and provide the code for reproducibility in an anonymized repo. The experimental evaluation seems comprehensive including experiments with on both synthetic and real datasets, as well as a  robustness analysis under violations of the Assumptions 1 and 2.

### Weaknesses
Even though the meaning of coverage might be clear to experts in selective classification, it might be helpful to include a high level definition of coverage in the introduction, so that it is clear for a broader ML audience.  

In Proposition 4, the authors assume a perfectly calibrated predictor. However, in practice, perfect calibration is impossible. As a results, it would be useful to include theoretical results that complement proposition 4 that account for the calibration error a classifier.     

Style/Typos:
1. Figure 3 has no caption.
2. The style of citations and captions of tables and figures does not follow the ICRL author instructions.

### Questions
1. Assuming that there is (small) calibration error of the predictions of the classifier, how would the results of proposition 4 change? 
2. It seems that abstention happens when $\bar{y}_i = \tau$. Could one also assume that abstention happens when $\bar{y}_i \in( \tau_1, \tau_2)$, that is when the prediction of the classifier is within some range? How could this affect the results of proposition 4, as well as the accuracy guarantees assuming a not perfectly calibrated classifier?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors present an approach to do selective classification in deep learning with concepts, by constructing a concept bottleneck model where the front end model can make predictions given soft concepts and leverage concept confirmations to improve coverage and performance under abstention.

### Strengths
The authors provide a good motivation and introduction. Authors also provide emperical validations on multiple datasets. The problem statement is very relevant to practical problems and provide an insight into how to automate classification tasks by making it safe and interpretable.

### Weaknesses
The writing and flow could be improved better, some of them are raised in questions below. Table 1 is referenced in Section 1, however what the columns means is defined only in Section 2, which makes it harder to read the table meaning. 

It would also be better to provide more details in the evaluation dataset around what each datasets means, and some statistics around it.

In my opinion the paper lacks novelty in terms of the innovation, and answers to the questions raised would help to understand better. Its not very clear about dataset statistics and how it changes and aligns with the interpretations that are presented.

### Questions
In the introduction its mentioned the front model can make predictions given soft concepts, however later in the text its mentioned in Section 2 under: `Propagating Concept Uncertaininty` its mentioned the front-end model requires hard concepts as inputs, which is not very clear?

In Introduction, its not very clear why the two objectives would conflict with other, if there are papers to cite that would help to make the claim stronger?

How does the choice of models to more complex architectures change the performance of the system?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
