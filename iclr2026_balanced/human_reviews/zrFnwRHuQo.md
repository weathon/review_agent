## Human Reviewer 1

### Summary
This paper looks at how to make communication more efficient in decentralized distributed training, where there is no central node and the network connections are random. The authors study when communication should happen during training, checking if communicating more at certain times helps the final shared model perform better.

In their experiments, they allow heavy communication only during a specific part of training, while at other times, each node talks to just one random neighbor. The results show that increasing communication near the end of training works best.

The paper also includes a theoretical analysis that proves their method converges. It further shows that training can be faster when certain gradient patterns are present.

### Strengths
The paper makes an interesting analysis of mergability of models trained on different (heterogeneous) subsets of data. Theoretical analysis shows that it depends on the sharpness of the merging point, which coincides with the usual intuition of how loss surface behaves. Moreover, proposed approach helps to reduce not needed communication to minimum.

### Weaknesses
Investigation of the communication after convergence is missing - in particular Figure2 demonstrates that continued individual training after full merge leads to reduction of the performance, the question is does it ever stop? Can the model converge to a state that neither merging, no continuing training changes the resulting performance?

### Questions
1 - Can you align your analysis with the dynamic averaging approach in [1], which claims that communication after convergence is not needed, motivating it also through the properties of loss surface; in particular, when models are already in one convex hull their performance will be basically same and aggregation is not needed anymore (I understand that there the setup is federated learning, but I would guess that this is the general understanding of how individually trained models behave).

[1] Kamp, Michael, et al. "Efficient decentralized deep learning by dynamic model averaging." Joint European conference on machine learning and knowledge discovery in databases. Cham: Springer International Publishing, 2018.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper showed that the performance of decentralized SGD can be improved by performing a single global merging at the end of the training. Then, this paper analyzed the convergence rate and provide the intuition of why the single global merging can improve the performance.

### Strengths
* It is an interesting finding that single global merging at the final training phase can significantly improve the accuracy of Decentralized SGD.
* This paper comprehensively evaluated how the effect of a single global merging affects the performance, depending on when we did it, showing that a single global merging is helpful at the final training phase.

### Weaknesses
* The experimental findings are interesting, but the theoretical explanation provided by this paper is difficult to follow.
* This paper claims that the convergence rate of Decentralized SGD with a single global merging matches the rate of Parallel SGD in Remark 2, but it is an overclaim. The convergence rate shown in Theorem 1 is not closed-form, which still depends on $\Xi_t$. Using Assumption 4, this term can disappear, but it is not fair to compare this rate and the convergence rate of Decentralized SGD since these analysis use the different assumptions.
* This paper claims that $A_t$ would be negative under Assumption 4 and the condition shown in Proposition 3 holds, but it is still unclear to the reviewer what Assumption 4 implies. Can the authors provide several functions that satisfy Assumption 4 or an intuition to understand Assumption 4?

Minor Comments:
* The statement of theorems and propositions must explicitly mention which assumptions are used. For instance, Theorem 1 and Propositions 2 and 3 seem to use Assumption 1, but it is not explicitly mentioned in their statements. Additionally, Proposition 2 seems to use Assumption 2, but it is not mentioned in the statement.
* This paper cited the same paper multiple times in the reference. (Kong 2021a and Kong 2021b)

### Questions
See the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper studies how communication should be scheduled over time to improve global generalization. The authors show that a single global merging in decentralized training can achieve performance close to that of federated learning, and that limited but non-zero communication helps preserve the mergeability of local models throughout training.

In addition, the authors provide a theoretical explanation for why limited but non-zero communication can ensure mergeability, and why communication should be concentrated in the later stages of training. The investigated problem is novel, and the theoretical analysis appears sound. The findings may further inspire future research on communication scheduling and allocation.

### Strengths
This paper studies how communication should be scheduled over time to improve global generalization. The authors show that a single global merging in decentralized training can achieve performance close to that of federated learning, and that limited but non-zero communication helps preserve the mergeability of local models throughout training.

In addition, the authors provide a theoretical explanation for why limited but non-zero communication can ensure mergeability, and why communication should be concentrated in the later stages of training. The investigated problem is novel, and the theoretical analysis appears sound. The findings may further inspire future research on communication scheduling and allocation.

### Weaknesses
Comments:

1.	The paper does not discuss the impact of network topology in decentralized learning. In realistic decentralized settings, multiple agents are connected through an underlying topology, which may prevent a single global merging due to communication constraints. Under a general topology, if agents communicate randomly according to the topology and perform full communication within that topology—rather than global merging— in the final stage, would the main conclusions still hold?

2.	Regarding Proposition 3, could the authors provide more intuition on why communication is needed when the gradient norm is small? Intuitively, when the gradient norm is large, more communication should be beneficial to accelerate optimization; when the gradient norm is small—indicating that the model is approaching a stationary point—less communication should suffice. This seems to contradict the paper’s conclusion.

### Questions
N/A

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 4

### Summary
The primary goal of the paper is to investigate optimal strategies for scheduling communication in decentralized learning, specifically focusing on when and how often devices should synchronize during training. The authors' empirical findings reveal that allocating most of the communication budget toward the end of training, particularly by performing a single, fully connected global merging at the final step, can achieve generalization performance comparable to traditional server-based (centralized) training, even under severe communication constraints and high data heterogeneity. On the theoretical side, the paper demonstrates that the globally merged model resulting from decentralized stochastic gradient descent (SGD) can converge as quickly as centralized mini-batch SGD, and provides new insights into why limited but well-timed communication is sufficient for effective model merging and generalization in decentralized settings

### Strengths
1. The paper tackles the core issues of communication bottlenecks and data heterogeneity in decentralized learning.
2. The paper investigates how to optimally schedule communication over time and links this to model merging.
3. The paper provides theoretical guarantees that the merged decentralized model can match or outperform centralized mini-batch SGD.
4. The paper shows that local models remain mergeable even with limited but nonzero communication.
5. Demonstrates findings with robust experiments on Tiny ImageNet, CIFAR-100 and CIFAR-10 datasets with up to 32 agents.

### Weaknesses
The decentralized setup in this paper assumes that all-reduce communication is impractical, so presenting a single global merging as an all-reduce operation may not be appropriate. Instead, a more suitable alternative is to perform simple gossip averaging (without additional SGD updates) for n iterations (where n is the number of nodes) when using a doubly stochastic graph structure. However, this shouldn't impact any of the results presented in the paper.

### Questions
Suggestions:
1. Can the authors add server-side training (centralized SGD) curve to the figure 3c? It would be insightful to add the centralized baseline there.
2. "Surprisingly, we uncover that fully connected communication at the final step. Low communication throughout training preserves the mergeability of local models." -- This phenomenon has already been shown in the literature [1] as Skew-Compensated Sparse Push in [1] but the current paper has no reference to [1]

References:
1. Aketi, S. A., Singh, A., & Rabaey, J. (2021). Sparse-push: Communication-& energy-efficient decentralized distributed learning over directed & time-varying graphs with non-iid datasets. arXiv preprint arXiv:2102.05715.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3