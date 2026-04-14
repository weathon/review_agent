# On Understanding of the Dynamics of Model Capacity in Continual Learning

- Decision: Reject
- Scores: 6, 5, 5, 5

## Abstract
The core issue in continual learning (CL) is balancing catastrophic forgetting of prior knowledge with generalization to new tasks, otherwise, known as the stability-plasticity dilemma. We argue that the dilemma is akin to the capacity~(the networks' ability to represent tasks) of the neural network~(NN) in the CL setting. Within this context, this work introduces ``CL’s effective model capacity (CLEMC)" to understand the dynamical behavior of stability-plasticity balance point in the CL setting. We define CLEMC as a function of the NN, the task data, and the optimization procedure. Leveraging CLEMC, we demonstrate that the capacity is non-stationary and regardless of the NN architecture and optimization method, the network’s ability to represent new tasks diminishes if the incoming tasks’ data distributions differ from previous ones. We  formulate these results using dynamical systems' theory and conduct extensive experiments to complement the findings. Our analysis extends from a small feed-forward~(FNN) and convolutional networks~(CNN) to medium sized graph neural networks~(GNN) to transformer-based large language models~(LLM) with millions of parameters.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Effective Model Capacity for Continual Learning (CLEMC) to quantify and understand how a neural network's capacity evolves in a continual learning setting. It provides theoretical justifications demonstrating that network capacity decreases with even slight changes in task distribution, eventually making the network unusable. The paper includes experiments on diverse network backbones and data modalities to evaluate the claimed theoretical results.

### Strengths
1. The paper presents rigorous theoretical justifications and detailed proofs; however, my current expertise limits my ability to fully validate these derivations.

2. The paper attempts to experiment on across different datasets and different modalities.

### Weaknesses
1. The paper concludes that with slight changes in tasks, the model's effective capacity decreases, eventually rendering it unusable. While the reduction in capacity appears justified, the paper lacks experiments that clarify what "unusable" entails. Establishing a threshold, perhaps based on loss or accuracy, to define when the model becomes unusable would be beneficial.

2. While the paper targets to justify the theoretical results drawn to four different modalities, the experimental setup vary without any rationale. The experiments in graph NN consists of gradients norm changes, while missing out the capacity change experiments in other modalities. And, the paper is hard to follow at times. For instance, a detailed explanation on figure 1 is missing. 

3. The visualizations are hard to analyze. There is no indication of task switching in the figures. A grid line for each task change can help improve the visualizations for both panels in fig 2A, B. The legend for fig 2 panel A conflicts with that of fig 1. So it would be better to have a consistent notation for all results. Also, a marking to indicate 4000 step, mentioned in line 373, would be better to include.

### Questions
1. Is $\| \Delta T(p) \|$ substantial compared to the initial values as the values are relatively small? What would be the percentage change compared to initial amplitude and frequency values?
1. Why does the experimental setup vary across modalities? A standardized set of experiments applied consistently across all modalities would enhance the strength of the paper.
2. How does the experiment with regularization having improved forgetting results connect with the theoretical insights provided? 
3. What is the rationale behind difference in pattern of Fig 4A and fig 2B? 4A has relatively increasing trends for the spikes while the previous ones don't.
4. How is $\gamma^{(k)}$ selected? Also, additional information on the hyper-parameters used would be appreciated.

### Soundness
2

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
3

### Summary
This paper explores the stability-plasticity balance in continual learning (CL) from a dynamical systems perspective. To examine the interplay between model, data, and optimization, the authors introduce CLEMC (CL’s Effective Model Capacity), finding that the network eventually becomes ineffective for representing tasks if each new task varies even slightly from the previous one.

### Strengths
1) Theoretical justification is provided for the proposed perspective.

2) Extensive experiments is conducted across various architectures, from small feed-forward (FNN) and convolutional networks (CNN) to medium-sized graph neural networks (GNN) and large transformer-based language models (LLMs).

3) They investigate the interplay among model, task, and optimization, a trio previously examined only in pairwise combinations—either model and optimization or model and data, which I think is interesting to understand their complex relationships.

### Weaknesses
1) The paper misses a key theoretical study on continual learning that explores catastrophic forgetting and task similarity[1].

2) Could the authors discuss how CLEMC handles task order? I'm particularly interested in understanding CLEMC's sensitivity to the sequence of tasks.

3) Could the authors provide a more detailed and clear explanation of Figure 1? I find it difficult to understand. 

4) In Figure 6, for the books task, the authors note that ER (Experience Replay) results in a lower cost compared to the scenario without ER. They attribute this to initialization bias and task similarity. However, the explanation is unclear to me. Could the authors elaborate on this and clarify the type of task similarity measure they are referring to?

[1] Doan, Thang, et al. "A theoretical analysis of catastrophic forgetting through the ntk overlap matrix." International Conference on Artificial Intelligence and Statistics. PMLR, 2021.

### Questions
Check the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper studies both theoretically and empirically the interaction between a model's capacity and continual learning. The paper first introduces CLEMC, a theoretical concept which describes the ability of a model to learn a new task continually without harming tasks learned in the past. The theory suggests that the forgetting cost grows with more continual tasks and capacity diverges (becomes worse) as more tasks are added. These findings are tested on MLP/CNN/GNN/LMMs.

### Strengths
S1: The topic of study is a very important one: Continual learning. The paper has a solid related works section motivating why this study is needed.

S2: The theory in the study is clear, and its conclusions are well highlighted.

S3: Experiments to confirm the predictions of the theory are done on different model architectures, which is essential to capture the generality of the theory.

### Weaknesses
W1: Some figures are unclear whether they support the claim. In specific, In Figure 4a, the capacity seems to drop back to a low value even though the initial bump gets bigger with more tasks. In Figure 5, there seems to be more changes seen in the weight, not explained by $\partial V^{*} / \partial x$.

W2: The FNN and GNN results could use more complex data to confirm its findings. While the authors mention that the experimental section is mostly meant for the ease of analysis, it is unclear how well these results will generalize to more complex settings. Can the authors show these results on realistic graph classification or MLP classification tasks?

W3: Although having a theoretical framework for CL is important, it isn't clear how surprising the finding from the theory are. It is unclear how surprising it is that 1) model hyperparameters (architecture, size, optimization, loss) affects the continual learning capacity and 2) more distant tasks causes a capacity degradation  and 3) the forgetting cost increases with more tasks. These finding seem intuitive. Are any of these claims surprising? If so, can this be discussed further?

W4: How does Figure 6 support its caption? Is wiki not treated as a task as it is close to the initial pre-training data? Should I interpret that the forgetting cost increase for the subsequent tasks?

W5: Figure 7,8 are hard to interpret. See Q1 below.

### Questions
Q1: It is quite hard to interpret Fig 7 and 8. Could a summary statistic be devised instead?

Q2: It would be nice to have highlighted claims for section 4.1: Theoretical analysis

Q3: This is very minor, but the wording "capacity" seems like a big values should be "good", i.e. more capacity, unlike what it is. Could this be clarified for quick skimming readers?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper provides a theoretical exploration of the relationship between model capacity and forgetting within a continual learning framework. The authors demonstrate that model capacity in CL is non-stationary, and the network's ability to represent or memorize new tasks diminishes as long as there is a distribution shift between tasks. This theoretical finding is supported by experiments conducted on various neural network architectures, including feed-forward, convolutional, graph neural networks, and transformer-based models.

### Strengths
1. The authors validate their theoretical results through extensive experiments using datasets relevant to the CL paradigm.
2. The derived theoretical results apply to various NN architectures, including feed-forward, convolutional, graph neural networks, and transformer-based models.

### Weaknesses
1. I went through the derivation of Lemma 1 and Theorem 1, which forms the basis for Theorems 2 and 3. However, I identified some potential issues that may have led to inaccuracies in the final results.

2. I believe the paper's presentation could be improved. There are several areas where the definitions regarding "capacity" appear to conflict, and important experimental details are missing, preventing the reader from fully understanding the results.

**Disclaimer**: It is possible that I may have misunderstood some of the derivations. If this is the case, I would appreciate a clear explanation from the authors. I am open to discussing and reconsidering my evaluation based on their responses. Please refer to the questions for more details.

### Questions
### Regarding derivation for Lemma 1 and Theorem 1 
Please refer to the supplementary material for these questions unless stated otherwise.

First for Lemma 1:
1. **line 167, equation (6a), undefined term**. $\Delta V_{(j)}$ is not clearly defined. It seems to be higher-order terms, yet the authors defined $\Delta$ as the first-order difference so I am not sure. More importantly, moving from (6a) to (6b), it appears that  $\Delta V_{(j)}$ was assumed negligible by the authors, as it was omitted in (6b). If so, could the authors elaborate on what does $\Delta V_{(j)}$ represent and why is it negligible?
2. **line 167, equation (6a), Appropriateness of Taylor Expansion**. Is using a Taylor expansion at $V_{(j)}$ for approximating $V_{(j+1)}$ appropriate? Based on the definition of the value function $V$ e.g., Equation $(J_F)$ Equation (14), $V_{(j+1)}$ contains different terms in comparison to $V_{(j)}$, in the summations, therefore, they are not simply the same function at two different time steps?
3. **line 214, equation (9). Transition from (7) to (9)**. How was the transition made from (7) to (9) and ultimately to Lemma 1 (see line 196, main text). The minus sign on the left-hand side (LHS) of (7) appears to have been omitted when directly substituting (7) into (9) for the innermost minimization term. No adjustments seem to have been made to the two outer max/min operators or the order of k and k+1 on the LHS in Lemma 1 to account for this omission. Is the result correct? Additionally, was Lemma 3 a misreference, given that there is no Lemma 3 in the paper?

Assume Lemma 1 is still correct, for Theorem 1:

4. **line 252, Lemma. Definition of $\boldsymbol{k}$**. $\forall k\in \boldsymbol{k}$, was $\boldsymbol{k}$ defined somewhere before?
5. **line 263, deduction from Theorem 1**. The statement "then we know from Theorem 1 using the property $<a,b>=||a||||b||$" needs clarification. How was this deduced from Theorem 1 (and perhaps you mean Lemma 1)? The equality $<a,b>=||a||||b||cos(\theta)$, holds true when the two vectors are aligned. Can you elaborate on this deduction?
6. **line 269, missing closing and opening $||$ symbols for the two terms**?
7. **line 294, equation (15a) to (15b), Triangle Inequality**. Should the equality be replaced with a less than or equal to sign, as the norm of sum generally does not equal to the sum of norms based on Triangle inequality? And equation (15b) to (15c), inequality should be equality instead, as the norms without $w_{j}$ are all 0s?
8. **line 307,equation (16), Decay Factor**. The superscript for the decay factor $\gamma^{p}$ does not match the index used in summation (over $m$), is this correct?  
9. **line 323, equation (17b), and line line 296, equation (15c), Implicit Dependency in optimization**. The authors derived these by stating that the norms in the summation without $w_{j}$ (the argument for which the derivate was taken) are all 0s. However, have we ignored the implicit dependency of $w_{j+1}$ on $w_{j}$ and likewise $w_{j+1}^*$ on $w_{j}^*$? The authors mentioned that $w_{j}^*$ is the starting point for optimizing $w_{j+1}^*$ when we consider the optimization trajectory in CL? If so, how is this justified?

These points need to be addressed to ensure the accuracy and clarity of the argument presented in Lemma 1 and Theorem 1, and consequently Theorems 2&3.

>

### Definition of "Effective Capacity" or "Capacity" in the Paper
The term "capacity" is used in multiple contexts, leading to potential confusion. It would be helpful if the authors could clarify the distinctions and connections between these definitions to ensure a consistent understanding throughout the paper.

- line 123, equation (EMC) first defines $\epsilon$ itself as the "effective model capacity"
- line 179: "as capacity at each $k$ is defined by the highest forgetting loss in the interval [0,k]". The description "as capacity at each k is defined by the highest forgetting loss in the interval [0,k]" suggests a different definition of capacity. It appears to refer to the term inside the summation in Equation CLEMC, introducing an extra outer maximization. This seems to deviate from the initial EMC definition in Definition 1.
- line 181:  The text mentions, " If the model learns ten tasks, then we obtain an EMC corresponding to each task, then, the $\epsilon_{(k)}^*$ is the sum of individual task capacities". This is confusing because it implies that individual task capacities (which are still seem to be defined using EMC) are summed to form the overall capacity, but this interpretation doesn't align clearly with the previous definitions.
- in line 243: The statement "Under these assumptions, we show that for both settings (i) and (ii) above, the effective capacity diverges". This "effective capacity" seem to refer to the gap in $\epsilon_{k}^*-\epsilon_{K}^*$ later as stated in line 250: "$\epsilon_{k}^*-\epsilon_{K}^*$ diverges"


>

### Empirical Observations and Their Alignment with Text and Theorems
It is possible that my current conclusions are a result of lacking detailed description on the experiments and setups. Clarification from the authors would be helpful to resolve these issues.

1. Figures. 2, 3, 4, how was $\epsilon_{k}^*$ calculated, was it strictly following Equation CLEMC? If so, based on Theorem 1 and 3, we expect an increasing (diverging) $\epsilon_{k}^*$, indicating that capacity gets poorer, for all $||\Delta T(p)||$, as long as $||\Delta T(p)||$ is non-zero. However, the green and blue lines in Figures 2B (top) and 3B (top) are decreasing over the training steps. This seems to contradict the expected theoretical trend.
2. Figure 2, what is $\Delta x$ exactly? I am aware that it is related to $||\Delta T(p)||$, as stated by the authors, but are these two quantities equivalent? For example, how is $\Delta x$ and $||\Delta T(p)||$ calculated in the sine experiments?
3. Line 333, "blue curve is better than the orange curve", is this referring to Fig 3, A? However, it's evident that the orange curves deteriorates slower then the blue ones, for all $||\Delta T(p)||$.
4. lines 346-353, analysis for Fig 4 A. I find the connection between the stability gap phenomenon (large jumps in losses at the start of each task) and Theorems 3 and 2 a bit unclear. Although the gaps increase over training steps, suggesting a loss of model capacity, the losses (or $\epsilon_{k}^*$  as plotted in A) quickly recovers, to a small value. Does this recovery conflict with the theoretical predictions?

>

### Minor comments on notations and typos
1. line 136-140, Equation $(J_F)$, main text. What is the feasible weight set $\mathcal{W}\_{(k)}$? Specifically, $T\_{(k)}=\\{T(0),T(1),....,T(k)\\}$ means the *collection* of all tasks until $k$, so I wonder whether $\mathcal{W}_{(k)}=\\{W(0),W(1),...,W(k)\\}$ is also a collection of all feasible sets until $k$ ? But based on Equation $(J_F)$ and the text that follow, it seems that $\mathcal{W}\_{(k)}$ represents only *one* specific feasible set at step $k$, with subscript $\_{(k)}$ for indexing. If my understand is correct, perhaps consider using $(k)$ and $k$ to distinguish between collection until $k$ and one specific index $k$ for improved clarity?
2. line 089, Equation $(CL)$, supplementary, an missing open bracket in the aurguments for the minimization problem. More importantly though, line 161, Equation $(CL)$, main text, should $w_{(i)}=\\{W_{(i)},i=k,k+1,...,K\\}$ be changed to $\\{w_{(i)}=W_{(i)},i=k,k+1,...,K\\}$ instead, since minimizing the sequence of $w_{(i)}$ is done with each $w_{(i)}$ being associated with its own feasible set i.e., $w_{(i)}\in W_{(i)}$?
3.  Why are the Lipschitz constants in square brackets, in Theorem 2 and 3? Are they interpreted as an argument of $alpha$, the learning rates, if written in this way?
4. Line 205, main text, maybe "effect" $\rightarrow$ "affect"?
5. Line 239 main text, "due to change in the data at the at the $p^{th}$ task", repeated "at the".
6. Supplementary Section B title: should refer to Lemma 1 instead of Theorem 1 and Supplementary Section C title should refer to Theorem 1 instead of Lemma 1.

### Soundness
2

### Presentation
2

### Contribution
2
