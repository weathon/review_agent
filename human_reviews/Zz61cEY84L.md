# Meta-Learning Strategies through Value Maximization in Neural Networks

- Avg Score: 6.25
- Decision: Reject
- Scores: 8, 6, 5, 6

## Abstract
Biological and artificial learning agents face numerous choices about how to learn, ranging from hyperparameter selection to aspects of task distributions like curricula. Understanding how to make these `meta-learning’ choices could improve engineered systems and offer normative accounts of cognitive control functions in biological learners. Yet optimal strategies remain challenging to compute in modern deep networks due to the complexity of optimizing through the entire learning process. Here we theoretically investigate optimal strategies in a tractable setting. We present a learning effort framework capable of efficiently optimizing control signals on a fully normative objective: discounted cumulative performance throughout learning. We obtain computational tractability by using average dynamical equations for gradient descent, available for simple neural network architectures. Our framework accommodates a range of meta-learning and automatic curriculum learning methods in a unified normative setting. We apply this framework to investigate the effect of approximations in common meta-learning algorithms; infer aspects of optimal curricula; and compute optimal neuronal resource allocation in a continual learning setting. Across settings, we find that control effort is most beneficial when applied to easier aspects of a task early in learning; followed by sustained effort on harder aspects. Overall, the learning effort framework provides a tractable theoretical test bed to study normative benefits of interventions in a variety of learning systems, as well as a formal account of optimal cognitive control strategies over learning trajectories posited by established theories in cognitive neuroscience.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper explores various meta-algorithmic choices ("control signals") that influence the learning dynamics of neural networks. The authors define optimal control signals by the condition that they maximize the cumulative learning performance (minus control costs) over the learning process. They study the setting of deep linear networks which allows analytic solutions of the learning dynamics equations, and, consequently, enables gradient-based optimization of the control signal. In the experiments, the authors consider control signals describing parameter initialization, learning-rate choice, learning curriculum, and gain modulation. The authors discuss the relevance of their results in the context of cognitive control in neuroscience.

### Strengths
The writing style of the paper is excellent and the authors supplement their exposition with helpful explanations and figures (e.g., Fig. 1). The proposed framework appears as a nice approach to reason about meta-algorithmic choices in the cognitive sciences and machine learning. The authors derive a general framework and apply it to a range of experiments which seem to be well fleshed out. I appreciate that the authors study an analytically tractable variant (linear networks) and derive results that seem to be consistent with the cognitive control literature. While I am uncertain of the practical relevance of the experimental results (cf. weaknesses), I consider theoretical studies of the proposed kind interesting and relevant in their own right. Thus, I recommend acceptance (with low confidence).

### Weaknesses
While the authors argue that their results are mostly consistent with the cognitive control literature (on which I cannot comment because I'm not an expert in this field), I doubt that the results have any practical relevance for the design of modern machine learning/meta-learning algorithms. I would be interested in whether the authors think that any practical advice for the machine learning practitioner can be derived  from their experiments.

### Questions
cf. weaknesses

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors present a meta-learning framework based on control signals via discounted cumulative performance.
In this framework, the authors describe a model, learning dynamics and a discounted reward based on a measure of the performance.
The authors then present a simplifying example that helps the reader better understand the approach and then continues with a two layer linear network model, evaluating on different perspectives of a classification task.
The authors show how this approach generalizes other existing meta-learning methods and present different instances of meta-learning tasks.

### Strengths
The paper is technically sound and presents a great exposition that helps the reader better understand the ideas in the paper. 
The experiments are backed by an extensive appendix that clarifies details.

### Weaknesses
The main limitation of the paper is as the authors mention, based on the linear models they use. This limits applicability.
Another limitation is the lack of comparison against other meta-learning instances, where the evaluation could compare computational time. However, the linear limitation probably makes this a non-important issue, and lifting it might introduce tractability problems.

### Questions
I found it interesting how the control adapts as expected. However, I'm wondering if for the given linear models that you present, if we were to collapse the two layers into a single one, your meta-learning turns into a form of regularization or instance weighting?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a framework for meta-learning where the learning dynamic can be analytically solved. The framework optimizes the control signals through gradient descent by considering  the total discounted learning performance minus control costs as an objective. Analytical tractability is reached by using a simple two-layer linear neural network that simplifies the dynamic. The authors show that they can model MAML, Bilevel Programming and other meta-learning methods with the introduced framework.

### Strengths
1) By suitable choice of the control signal, one can model MAML, Bilevel Programming, task switch, and other techniques. 
2) The experimental section is vast, well-described, and explained.

### Weaknesses
The authors propose the framework as a test bed for meta-learning. However, I have several concerns regarding its practical applicability:

1) A simple two-layer linear neural network is used, so it may not account for all the effects during training, and the results may not translate to more complex NNs. 
2) It's likely one will need to find a solution for every novel considered intervention.
3) I don’t think all interventions can be modeled within this framework, even when restricted to the context of two-layer linear neural networks. (please see the Question section)

So I believe the framework may have limited usage as a test bed.

### Questions
1) Will the framework be able to find tractable solutions if we consider learning rules as a control signal (e.g. [1])? 
2) To what extent can we generalize control signals in a two-layer linear neural network while still maintaining solution tractability?
3) Do you think it is possible to develop task switching in your framework when one doesn't know what task will be next in advance?

Small incorrectness. 
Images (b) and (c) should be swapped In Fig. 2. 

[1] Andrychowicz, Marcin, et al. "Learning to learn by gradient descent by gradient descent."

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
The works introduces a learning effort framework which is capable of optimizing control signals on objectives with discounted cumulative performance throughout learning. Frameworks and settings being analyzed in the work are meta-learning, curriculum learning, and continual learning and a number of results are provided, showing how and when the control effort might be helpful under linear settings.

### Strengths
1. The authors present a learning effort framework over a number of problem settings to analyze optimal strategies for learning. Having an understanding and intuition about this can help our current deep learning design problems learn better.
2. The document motivates the problem well, emphasizing the importance of the work.
3. The arguments made by the work are linked to cognitive science and neuroscience, which can be used to get inspiration from when designing our current models.
4. The work conducts experiments over multiple paradigms, including meta-learning & continual learning.

### Weaknesses
Some of the areas that the work could be improved upon:
1. As pointed out by the authors themselves, a limitation is the assumption of linear models. Since the motivation behind the current work is to provide ways in improving the current neural networks, more analysis on non-linear systems is needed. Although very large neural networks are hard to analyze, simpler variations could be considered for non-linear settings to make this direction even more interesting.
2. Optimization is a big challenge in neural networks. An analysis focused on the effect of different optimization techniques might be helpful when extending the work on non-linear settings, which is currently being analyzed in only simpler artificial settings.
3. A number of other biases exist in the cognitive literature, eg having modular systems in which different modules do different tasks, or learn them over multiple tasks. Analyzing this explicitly can be an interesting addition to the work.
4. Figure 1 could be made a bit larger for better readability.

### Questions
1. Have the authors come across any unexpected results while analyzing the settings? For example, for curriculum learning where defining the curriculum itself might effect the learning in unexpected ways.
2. Is it possible to include some experiments from non-linear models (without the approximation) to address the concerns from the "weaknesses" section?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
