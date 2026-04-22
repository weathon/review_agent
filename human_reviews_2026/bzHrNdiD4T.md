# Centroid Approximation for Byzantine-Tolerant Federated Learning

- Avg Score: 3.20
- Decision: Reject
- Scores: 4, 2, 2, 4, 4

## Abstract
Federated learning allows each client to keep its data locally when training machine learning models in a distributed setting. Significant recent research established the requirements that the input must satisfy in order to guarantee convergence of the training loop. 
This line of work uses averaging as the aggregation rule for the training models. 
In particular, we are interested in whether federated learning is robust to Byzantine behavior, and observe and investigate a tradeoff between the average/centroid and the validity conditions from distributed computing.
We show that the various validity conditions alone do not guarantee a good approximation of the average. 
Furthermore, we show that reaching good approximation does not give good results in experimental settings due to possible Byzantine outliers. 
Our main contribution is the first lower bound of $\min\{\frac{n-t}{t},\sqrt{d}\}$ on the centroid approximation under box validity that is often considered in the literature, where $n$ is the number of clients, $t$ the upper bound on the number of Byzantine faults, and $d$ is the dimension of the machine learning model. We complement this lower bound by an upper bound of $2\min\{n,\sqrt{d}\}$, by providing a new analysis for the case $n<d$. In addition, we present a new algorithm that achieves a $\sqrt{2d}$-approximation under convex validity, which also proves that the existing lower bound in the literature is tight. We show that all presented bounds can also be achieved in the distributed peer-to-peer setting. We complement our analytical results with empirical evaluations in federated stochastic gradient descent and federated averaging settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors provide upper bounds (and some lower bounds) for different validity conditions in Byzantine federated learning, offering useful insights into the theoretical guarantees of aggregation algorithms. However, the experimental evaluation does not sufficiently support the theoretical findings.

### Strengths
1. The paper is well structured and presented in a clear and coherent manner.
2. The theoretical foundations and analyses are thorough and well supported.

### Weaknesses
1. BOX and MDA correspond to the box and centroid validity conditions, respectively. The authors should consider including an aggregation algorithm based on the convex hull of all non-faulty input vectors for comparison. Intuitively, the upper bound under the box condition should be tighter than that under the convex hull, which may imply that the BOX aggregation is more stable.

2. Can the authors provide a theoretical explanation for why BOX converges faster than MDA in the experiments?

3. It appears that the experiments are conducted mainly under the case n < (d + 1)t. The current results are insufficient to align with the theoretical analysis. The authors are encouraged to analyze both cases, n > (d + 1)t and n < (d + 1)t, and provide preliminary empirical verification of the derived upper and lower bounds under different validity conditions (aggregation algorithms).

4. In line 282, when mentioning n < d, there is no such case shown in Table 1. Should it be n < (d + 1)t instead?

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper the authors present analytical results for centroid approximation in federated learning scenarios with the presence of Byzantine clients. In particular, they show that known validity state-of-the-art conditions do not guarantee good approximation of the average. In particular, through a geometric setting they tighten the upper bound and introduce an improved lower bound.

### Strengths
With respect to the majority of FL literature the introduction is fairly original and effectively highlights what are the issues in the state-of-the-art, posing this work under an interesting geometrical perspective.

### Weaknesses
After a thorough reading of the paper I am concerned of several aspects, mostly it results unclear and non-immediate what is the true contribution of the work. Therefore, I encourage the authors to be more assertive in their statements and better highlight the novelty of their results, as, in its current state, the work is a little hard to follow.

W1. The mathematical rigour of the definitions should be improved. On one hand it is important to verbally explain definitions and claims, on the other it is important to use a proper notation with symbols, in order to avoid to fall into ambiguity (Definitions 2.3 -- 2.6). Also I think that introducing Notation 1 after its actual use in Def. 2.6.

W2. The overall exposition should be more formal and clear, particularly statements and lemmas.

W3. In the introduction, the authors remark that their results are valid up to a number of attackers $t < n/3$. This sounds a little limiting, as most of the FL literature assumes that $t < n/2$, and now more recent works are moving towards $t< n$. 

W4. In the 'Centroid Approximation' setting (lines 179-180), the authors assume that the number of participating clients at each round $m$ is between $n-t$ and $n$. The first concern is that, hence in the worst case scenario, when $ t = n/3$ (or something slightly smaller), the server samples more $2n/3$ clients, causing a non negligible bottleneck -- thus violating the low communication overhead principle in FL. Consider that most of the FL literatures considers federations larger than 100 clients, and the sampled clients are around 10%, or less.Furthermore, as far as I have understood, in order to make the method work, the server should know exactly the number of faulty clients $t$, which is not possible.

W5. The result section is a little bit poor. What happens when you increase the size of the federation $n$? At least another dataset should be evaluated, and more baselines considered.

### Questions
See weaknesses.

### Soundness
1

### Presentation
1

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
The paper analyzes Byzantine-tolerant federated learning through the lens of centroid approximation. It defines the set of all (n−t)-subset centroids, the minimum covering ball around those centroids, and measures any aggregator by how close its output is to the true non-faulty centroid relative to the ball’s radius. Using validity conditions drawn from distributed agreement, such as weak, strong, box, and convex validity, the authors derive upper and lower bounds on the achievable approximation ratios, with new and mostly tight results for box validity and clear statements for convex validity. They also argue that the guarantees extend to synchronous peer-to-peer FL by combining deterministic aggregation with Byzantine agreement, and they provide preliminary simulations on MNIST under several parameter-based attacks that suggest a practical trade-off between approximation quality and validity constraints.

### Strengths
1. The definitions of candidate centroids, minimum covering ball, and the centroid approximation ratio are crisp and useful for analyzing Byzantine aggregation beyond worst-case distances.
2. Upper/lower bounds under weak/strong/box/convex validity are tabulated; the new analysis for box validity (including n<d) and the tight 2d bound for convex validity are valuable. 
3. The extension argument to peer-to-peer (interactive consistency → identical inputs → deterministic aggregation) is straightforward and helps position the theory for decentralized FL.

### Weaknesses
1. The setup treats all clients equally (ignores sample-size heterogeneity) “to restrict Byzantine power,” which deviates from standard sample-count weighting in FL. Please justify this modeling choice and discuss implications for deployment, including whether your bounds/algorithms still hold under non-uniform weights or can be adapted with importance weighting.
2. (Eq. 3) is stated for general n,d,t but never checks corner cases like t=0, n≤2t, n<d, or n=d+1. The statement should list validity regimes explicitly and clarify vacuous/impossible cases.
3. (Eq. 5) relies on a denominator that can be zero for certain client selections (e.g., when all trimmed coordinates coincide); the text doesn’t specify the fallback (skip, clip, or add ε).
4. The step from (Eq. 7) to (Eq. 8) uses an ℓ2bound and then applies an ℓ∞ argument without a stated inequality tying them together.
5. The inequality used from (Eq. 9) → (Eq. 10) assumes convexity/linearity in the wrong place; it moves the norm outside an average without justifying it.
6. (Eq. 11) bounds a gradient/model drift term but never states the Lipschitz or smoothness constants required to make that bound legal.
7. (Eq. 12) exchanges 𝐸 with a coordinate-wise trimming/selection; measurability and selection dependence on the sample aren’t addressed.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper studies the theoretical and empirical foundations of aggregation robustness in Byzantine-tolerant federated learning (FL). The authors analyze the centroid approximation problem, i.e., how closely an aggregation algorithm can approximate the average of non-faulty clients when up to t out of n clients are Byzantine. They establish lower and upper bounds for centroid approximation under different validity conditions, provide almost-tight bounds for box validity, and extend the results to peer-to-peer settings. Theoretical analysis is illustrated by experiments under various Byzantine attacks and heterogeneous data distributions.

### Strengths
1. The paper provides the first lower bound of $\min \{n/t - 1, \sqrt{d} \}$ for centroid approximation under box validity and an improved upper bound of $2\min \{n , \sqrt{d} \}$ establishing nearly tight limits.

2. Bridging approximate agreement theory and federated learning offers a good perspective and strengthens theoretical rigor in Byzantine robustness. Extension to a peer-to-peer network setting is provided.

### Weaknesses
The main concern is in the simulation part. The experimental setting (30 clients) is small. It remains unclear how the proposed bounds or algorithms behave on large-scale FL systems with realistic models (e.g., CNNs, Transformers).

### Questions
Could the authors provided more experimental results on more practical settings and other complex tasks?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a novel theoretical framework for Byzantine-tolerant Federated Learning (FL) leveraging the concept of centroid approximation. In contrast to other methods that typically aim to detect or exclude faulty clients, the paper proposes to keep all updates and evaluate how well an aggregation rule can approximate the true average of the non-faulty clients, even when some updates are Byzantine. The paper uses ideas from distributed computing and FL, analyzing how different validity conditions constrain the quality of possible approximations. The theoretical contributions include 1) a lower bound of the centroid approximation under box validity, 2) an upper bound by providing a new analysis for the case when the number of clients is smaller than the number of parameters, and 3) an algorithm that achieves a sqrt(2d) approximation under convex validity. The paper also includes experiments with SGD and federated averaging aggregation schemes.

### Strengths
+ The theoretical analysis is sound and establishes nearly tight upper and lower bounds on centroid approximation under different validity conditions, extending classical results from distributed computing to FL. 
+ The angle of using the centroid approximation metric as a way to analyze Byzantine robustness is interesting. The paper clearly explains the geometric intuition. On the other hand, the idea of not excluding the Byzantine clients (unlike other aggregation methods) and yet achieving robust aggregation is also quite interesting. 
+ The proofs are well structured and grounded and the theoretical contributions includes both lower and upper bounds.
+ The authors show that the results can be easily extended to peer-to-peer FL scenarios. 
+ The experiments help to support the theoretical findings and contributions.

### Weaknesses
+ The safe area assumption (t < n/(d+1)) in Definition 2.8 is quite impractical for most FL scenarios, collapsing to t=0 in realistic settings where d is large. Then, some of the results, like Lemma 3.4, although mathematically valid, has a limited relevance for low-dimensional problems.
+ Some of the assumptions are also restrictive for many practical FL scenarios: the paper assumes synchronous communication, equal local training data sizes, or static participation. It helps to simplify the theoretical analysis but is realistic for many practical FL deployments. 
+ Although the centroid approximation provides an interesting view on robustness, its link to actual learning performance (like convergence or accuracy) remains unquantified in the paper. In other words, how does centroid approximation connect with convergence during training or with accuracy?
+ The experimental evaluation uses simple settings: all experiments use a single dataset (MNIST), a simple MLP model, and basic FedSGD and FedAvg setups. On the other hand, while several attacks are tested (in the appendix), the empirical study lacks diversity in data, model architectures, or comparison to other robust aggregation methods in the state-of-the-art.

### Questions
+ Given that (t < n/(d+1)) becomes infeasible when d is much bigger than n, do you see any realistic settings where convex-validity results (like in Lemma 3.4) could apply? Is it possible to relax this assumption to make this theoretical framework more usable in practice?
+ Although the proposed sqrt(2d) approximation algorithm seems theoretically sound, could you explain how does this algorithm scale for models with millions of parameters or in scenarios with a large number of clients?
+ Related to the previous question on computational complexity: can the algorithm be applied to other architectures beyond an MLP (e.g., CNNs, transformers)?
+ Have you empirically checked whether smaller centroid approximation ratios correlate with faster convergence or higher accuracy?
+ Have you considered adaptive attacks for evaluating the robustness of the algorithm?

### Soundness
3

### Presentation
3

### Contribution
2
