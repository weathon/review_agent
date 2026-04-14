# WardropNet: Traffic Flow Predictions via Equilibrium-Augmented Learning

- Decision: Accept (Poster)
- Scores: 5, 5, 6, 8

## Abstract
When optimizing transportation systems, anticipating traffic flows is a central element. Yet, computing such traffic equilibria remains computationally expensive. Against this background, we introduce a novel combinatorial optimization augmented neural network pipeline that allows for fast and accurate traffic flow predictions. We propose WardropNet, a neural network that combines classical layers with a subsequent equilibrium layer: the first ones inform the latter by predicting the parameterization of the equilibrium problem's latency functions. Using supervised learning we minimize the difference between the actual traffic flow and the predicted output. We show how to leverage a Bregman divergence fitting the geometry of the equilibria, which allows for end-to-end learning. WardropNet outperforms pure learning-based approaches in predicting traffic equilibria for realistic and stylized traffic scenarios. On realistic scenarios, WardropNet improves on average for time-invariant predictions by up to 72\% and for time-variant predictions by up to 23\% over pure learning-based approaches.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes WardropNet to predict traffic flow by the combinatorial optimization-augmented machine learning (COAML) pipeline. Using supervised learning and Fenchel-Young loss, this method minimizes the difference between predicted and actual traffic flows by leveraging a Bregman divergence, ensuring it fits the geometry of traffic equilibria. WardropNet improves on average for traffic flow predictions compared with pure neural network method.

### Strengths
1. The essay provides a solid theoretical foundation for the approach.
2. The visualization and illustration are well made and help the readers to understand the paper.
3. By incorporating real-world scenarios, the method emphasizes its potential for real-world traffic management and urban planning.

### Weaknesses
1. Lack of comparison with more advanced baselines. While the paper compares WardropNet with basic machine learning models like FNN and GNN, it doesn't compare against other state-of-the-art traffic flow prediction models, such as deep reinforcement learning approaches or physics-informed neural networks.

2. Not well-structed essay presentation. As paper introduces a complex method, the explanation of how the model works is too technical too quickly. Meanwhile, the scenarios of the experiments section feels too long.

3. Sensitivity to Hyperparameters. The paper does not detail the sensitivity of WardropNet's performance to different hyperparameters such as layers number and learning rates. Since the model integrates multiple components, tuning could be more challenging, and it's unclear how stable the model is across different settings.

### Questions
1. In the end of section 1, the paper put up with " backpropagation through general, possibly combinatorial, equilibrium layers". What does it mean, and can the paper give some examples?

2. In the end of section 3, how the Bregman divergence, the non-convex problems, turn to a convex surrogate? Why can the paper do the alternation?

3. In the comparision part, is there any numberical results (listed in the table) can be shown to clearly explain the strengths of the new algorithms? Meanwhile, it can be noticed that, the results of ER perform poorly in mostly scenarios. Please explain the reason and the strengths of this aspect of algorithm. In addition, can the paper give a comparision different pipelines in various scenatios in terms of the mathmatical theroies, and give the differences among them?

4. Also in comparision part, can you analyze and explain the reasons and aspects why the proposed algorithm is better than the baseline algorithm, and how these aspects confirm the strengths of the algorithm you proposed in the conclusion?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper describes a neural network architecture designed to predict traffic flows. 
The main idea is to combine traditional neural network layers with an "equilibrium layer" that models traffic flow equilibria, 
Then, train the network end-to-end given training data pairs of (network, target flow)


The authors provides anonymized code which is commended.

### Strengths
S1: An interesting problem, combining learning and combinatorial optimization. This seems to be novel.
S2: A detailed theoretical foundation for the proposed architecture. 
S3: Extensive experiments on 6 six scenarios, using traffic simulators to generate GT for training.

### Weaknesses
W1: I found the paper very hard to understand. It may have been written by researchers outside the ICLR community. The introduction spends half a page to describe in detail supervised learning and ERM, but does not clearly define the problem or explain current state-of-the-art approaches. Then, it is not made explicitly clear enough which parts are novel and which parts were previously introduced. Terms like paradigm, pipeline, layer, model and architecture, are sometimes used loosely and interchangeably. See Q1 for specific questions. 

W2: The method is compared with the an architecture that has the CO layer removed, and with various variants of the method. Not with other baselines in this field.  I am not a member of this community, but a quick search shows previous approaches do exist. See AA Kashyap 2022, Traffic flow prediction models – A review of deep learning techniques. 
The author should analyze their proposed architecture in the context of previous work.

W3: There is a gap between the general formulation of the problem and the iterative relaxations, and it is not clearly stated how each relaxation limits the application of the COAML pipeline in practice.

W4:  The COAML problem formulation attempts to address a general latency function, but in the end the experiments are done with a simple (possible unrealistic) latency function. 

W5: It is not clear whether WardropNet yields a meaningful improvement (e.g, improving the accuracy to 2% from 1% SOTA ML baseline may sound like a great improvement, but if a non-ML approach can achieve 90% accuracy, then the gap to practicality is still huge).

### Questions
Q1: What is the architecture of the new equilibrium layer? What exactly is its input, output, and tunable parameters? 
(I realize that this information maybe given somewhere in the 20-page supplemental. But a paper that states that its main contribution is a new layer, should describe this layer in the main paper. Or revise their claimed contribution).

Q2: Which results in section 3 were previously known? What new theoretical results are presented in the paper?

Q3: The paper stated it contained 9 training instances samples. Could you clarify how these instances are used? I'm assuming they are used for generating many labeled training samples (x,y). How exactly is this done? If only 9 samples are provided

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper introduces a theoretical framework designed to address the challenges of understanding the convergence and generalization properties of machine learning algorithms. The authors propose a set of mathematical constructs and algorithms aimed at improving the understanding and solution of issues related to algorithm performance in various settings. Key contributions of the paper include the development of new analytical models and convergence proofs, which are presented alongside rigorous theoretical analyses. The authors also discuss the implications of their findings and how they relate to current practices in the field of machine learning.

### Strengths
1. The paper introduces a novel theoretical framework that enhances the understanding of convergence and generalization properties in machine learning algorithms, addressing critical gaps in existing literature.

2. The rigorous mathematical analysis, including new analytical models and convergence proofs, adds credibility and depth to the research.

3. The well-organized structure and effective use of examples make complex theoretical concepts accessible and easy to understand.

4. The findings have the potential to impact future research directions in machine learning and improve algorithm design, offering valuable insights for practical applications.

### Weaknesses
1. The paper lacks a thorough comparison with existing theoretical frameworks or analyses in the field. A comparative analysis highlighting the proposed framework's advantages and limitations relative to established methods would clarify its contributions and significance.

2. Some of the theoretical constructs presented are quite complex and may be challenging for readers who are not deeply familiar with the underlying mathematics. Simplifying certain sections or providing additional explanations and visual aids could enhance understanding and accessibility.

3. The authors do not sufficiently address the limitations of their framework. A more transparent discussion about potential shortcomings, assumptions made, and scenarios where the framework may not apply would provide a more balanced view of the research.

### Questions
I would like to clarify that I am not an expert in the specific field addressed in this paper. While I can appreciate the effort and the theoretical contributions made by the authors, my limited background in this area restricts my ability to fully evaluate the nuances and implications of the proposed framework. Therefore, my feedback may not capture all the intricacies of the work or its potential impact on ongoing research.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Traffic flow on a transportation network is influenced by many contextual factors such as weather conditions, time of day, road capacity, etc. Under mild hypotheses, it can be shown that the traffic flow will converge to an equilibrium known as the Wardrop equilibrium (WE). Predicting how the  Wardrop equilibrium will change as a result of changes to these factors is crucial for the design of better transportation systems. This paper introduces a novel approach to this problem which combines a neural network with a combinatorial solver.

### Strengths
- The paper is well-written and easy to follow. Although the topic of traffic flow prediction might not be familiar to the average ICLR reader, the appendices do a great job of introducing the reader to the fundamentals of this problem.
 - Novelty: I believe this is the first paper to apply Fenchel-Young losses to the problem of predicting WE, which is an important contribution.
 - The numerical experiments are sufficient to convince me of the utility of the proposed method.

### Weaknesses
See questions.

### Questions
- Do you have any thoughts on predicting non-equilibrium traffic flows? Do you think these are important for modeling?
 - I appreciate your generalized notion of latency function. However, it seems to me that requiring generalized latency functions to derive from a potential is quite a strong assumption. Could you give an example of a set of non-decomposable latency functions deriving from a potential? (Maybe the regularization by perturbation model of Section 4 is such an example?)
 - Could you provide some background, for non-experts like myself, on the state of the art of WE solvers? From Appendix D.1, I gather that MATSim uses a genetic algorithm to solve for the WE. Are there not faster approaches that use techniques from convex optimization, e.g. interior point method? Also, it would be helpful if you included a runtime comparison between WarDropNet and MatSIM. 

Minor Questions: 

 - in Line 129 "unilateral deviation would incur in a longer travel time" should be "unilateral deviation would incur a longer travel time" (no "in").
 - In line 185, "Following, the supervised learning setting" should be "Following the supervised learning setting" (no comma after "Following").
 - For consistency with eq (2), the sum in eq. 9 should probably have a $\frac{1}{N}$ in front of it.
 - In line 302 "K represents the amount of parameters" should be "K represents the number of parameters"
 - In line 335, "We note that the arg max is unique on a sampled realization of Z" should probably be "We note that, with probability 1, the arg max is unique on a sampled realization of Z".
 - On line 458 "each roads context." should be "each road's context." 
 - On line 1141 "raods" should be "roads"
 -  Suggest citing _End-to-end learning of user equilibrium with implicit neural networks_ by Liu et al in "Related Works".

### Soundness
3

### Presentation
4

### Contribution
3
