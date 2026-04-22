# The Oversmoothing Fallacy: A Misguided Narrative in GNN Research

- Avg Score: 2.80
- Decision: Reject
- Scores: 2, 2, 2, 4, 4

## Abstract
Oversmoothing has been recognized as a main obstacle to building deep Graph Neural Networks (GNNs), limiting the performance. This paper argues that the influence of oversmoothing has been overstated and advocates for a further exploration of deep GNN architectures. Given the three core operations of GNNs, aggregation, linear transformation, and non-linear activation, we show that prior studies have mistakenly confused oversmoothing with the vanishing gradient, caused by transformation and activation rather than aggregation. Our finding challenges prior beliefs that oversmoothing in GNNs is dominated by the GNN-specific structure of aggregation. Furthermore, we demonstrate that classical solutions such as skip connections and normalization enable the successful stacking of deep GNN layers without performance degradation. Our results clarify misconceptions about oversmoothing and highlight the untapped potential of deep GNNs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper argues that the performance degradation in deep GNNs has been mistakenly attributed to oversmoothing, while the real problem in practice is “zero-collapsing” and vanishing gradients arising from the transformation+activation steps. This paper  (i) separates aggregation, linear transformation, and activation, (ii) measures oversmoothing with several metrics across different scenarios (σ+A+T, σ+T, A+T, A), and (iii) claims aggregation has only a marginal role. This work further shows that residual/skip connections and batch normalization enable very deep GNNs without degradation, and position asymptotic oversmoothing results as complementary but not explanatory for finite-depth models.

### Strengths
- The paper is well written.
- This paper is a good reminder that optimization is also important in GNNs (vanishing gradients).

### Weaknesses
- Even though this paper is a good reminder that optimization is important, it doesn’t bring anything new about the oversmoothing problem in the literature. For example, the following statement is already known and well studied theoretically: “Furthermore, we demonstrate that classical solutions such as skip connections and normalization enable the successful stacking of deep GNN layers without performance degradation”. See [R2]. From a practical perspective, the following claim has been observed empirically since 2019: “Remarkably, we show that by properly integrating these two simple yet effective strategies, it is feasible to train extremely deep GNNs, successfully scaling up to 1,024 layers”. See [R3]. Solving oversmoothing by using residual connections does not mean that oversmoothing is caused by vanishing gradients. This is a logical fallacy; correlation does not imply causation.
- The following claim is not true: “Our finding challenges prior beliefs about oversmoothing being unique to GNNs”. Previous works have reported oversmoothing in other architectures, e.g., transformers [R1].
- Regarding the remark in the introduction, I do not get the point. This remark says this work is not in contradiction with asymptotic analyses and that practical issues are dominated by vanishing gradients; yet, elsewhere, the narrative suggests the community has “misunderstood” oversmoothing and overestimated the aggregation’s role. The take-home message from this paper oscillates between “oversmoothing exists asymptotically but is not the practical bottleneck” and “aggregation’s role is marginal and oversmoothing has been overestimated” (too broad without stronger causal evidence).
- In Figure 3, this paper argues that “oversmoothing is observable without the aggregation step”. I disagree with this statement; again, correlation does not imply causation.
- Typo, I guess: “Note that A+T is equal to the MLP architecture”.
- I don’t understand Figure 4. Is this just a histogram of all values in the embeddings? Oversmoothing should be analyzed column-wise in $X^{(l)}$. The argument in the literature is that each of these columns (also known as graph signals) converges to a stationary state, also characterized by a zero Dirichlet energy. Plotting such a histogram does not make sense.
- While this paper attributes the oversmoothing problem primarily to vanishing gradients and argues that the aggregation operator plays only a marginal role, Roth and Liebig (2023) [R4] theoretically demonstrate that the spectral properties of the aggregation operator are, in fact, the main cause, driving node representations into a low-dimensional subspace irrespective of the feature transformations. From that perspective, the residual connections proposed here do not resolve vanishing gradients per se, but rather correspond to a specific implementation of a Sum of Kronecker Products to alleviate rank collapse. Besides, the following statement in that remark seems to be false: “These results provide valuable insight into the limiting behavior of message-passing networks, but they largely characterize asymptotic convergence rather than explaining the degradation observed at practical depths”. Indeed, [R4] provided empirical evidence for “practical depths” (see figures 2 and 4 in [R4]).

**General comment**: The main claim of this paper goes against much of the existing theoretical and experimental research, but the authors do not provide a new theory to support this different view. The experiments shown are not strong enough to question or replace the current understanding of oversmoothing, and some results do not fit the authors’ own explanations. At the very least, a clear and convincing theoretical framework would be needed before such strong claims can be accepted.

---
[R1] “Mitigating Over-smoothing in Transformers via Regularized Nonlocal Functionals”, NeurIPS 2023.
[R2] “Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs”, ICLR 2025.
[R3] “DeepGCNs: Can GCNs Go as Deep as CNNs?”, ICCV 2019.
[R4] “Rank Collapse Causes Over-Smoothing and Over-Correlation in Graph Neural Networks”, LoG 2023.

### Questions
In Figure 3, results for aggregation only (A) contradict all results about oversmoothing in the literature. This is basically an SGC model where the power of the adjacency matrix (for GCN) is $l$. We have enough theoretical evidence to say that the node embeddings should converge to a stationary distribution when the graph is connected and not bipartite. It seems to me this is a one-layer SGC with multiple linear layers. How is this experiment performed? Could you compute tr(X’LX) (Dirichlet energy)? It should converge to 0 quite fast.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper investigates the impact of oversmoothing on the degradation of performance of graph neural networks. 
The paper empirically investigates which components of the update rule contribute most to the problem in an ablation study.
The authors come to the conclusion that oversmoothing is not the main issue, but that a common problem in conventional neural networks called "dying ReLU" - a form of vanishing gradients - is the main obstacle to overcome in order to train deep GNNs.
The authors therefore propose to use conventional deep learning methods called residual connections and batch normalization to combat the problem of performance degradation and show empirically that performance degradation can be mitigated using these techniques.

### Strengths
- The paper asks an interesting and profound question about the strength of the impact of certain phenomena in the performance of GNNs. 
- The paper displays a thorough review of related literature and identifies problems in previous manuscripts.

### Weaknesses
-  The main hypothesis of the paper that "Oversmoothing is not a problem for the performance of deep GNNs" is not corroborated by the evidence layed out. Theoretical considerations are missing entirely.
    - The premise that researchers believe that oversmoothing is the single most impactful problem for deep GNNs is flawed. It makes little sense to believe that the problems plaguing MLPs would not carry over to GNNs (that use MLPs internally). However, oversmoothing is a *new* phenomenon and problem that is unique to GNNs, which may be the reason it has received the attention that is has. Additionally, oversmoothing has a much more devastating effect than e.g. vanishing gradients: For any sensible weights, the model will exponentially converge to an uniformative representation of the nodes and not be able to recover the initial signal. 
    - Figure 1 and Figure 5 depict the "recovery from oversmoothing" for GCN. However, there is no recovery: Initializing a GCN with all-ones does not start it off in a state of oversmoothing as it would for row-stochastic graph operators like the one used in GAT. Instead oversmoothing manifests in GCN as the exponential decay toward $\sqrt{d}$. So using $\sqrt{d}$ as the columns of $X$ would actually start GCN in a state of oversmoothing. The GCN hence does not "recover" from oversmoothing in Figure 1 - it was never in a state of oversmoothing to begin with. The same idea leads to the phenomenon depicted in Figure 5. The claim that "Based on the empirical evidence, [...] GCNs are capable of escaping the oversmoothing regime." is not a consequence of the empirical evaluation shown.
    - The authors conclude from the experiments shown in Figure 3 and Figure 4 that the aggregation does not have as great an impact as oversmoothing and that the activation function is responsible for the oversmoothing observed. This is a fallacy, the effect the authors observe is in fact the "dying ReLU" phenonmenon [1,2]. As they discuss, the Glorot intitialisation paired with ReLU is responsible and this is a known problem. A rigorous way to inspect the layed out hypothesis would be to ablate the activation function and initialization. Can we see the same effects when using other activation functions such as tanh, LeakyReLU, PReLU or ELU? Can we initialize in a way that mitigates this phenomenon? [7]
    - The proposed mitigation techniques have been known and used for years [3]. Recently, there have also been theoretical advancements as to why normalization and residual connections prevent oversmoothing [4-6]. So indeed, the proposed methods not only work well for the vanishing gradient problem, but also mitigate the oversmoothing caused by aggregation, leading to the absence of performance decline seen here. A rigorous way to approach this would be again to ablate only what is hypothesized to be respoinsible for the decline in performance - i.e. the vanishing gradient, but not oversmoothing. Normalization and residual connections do not work for this, as they prevent both problems.

- The presentation is suboptimal. 
     - Most prominently the experiments in the form of the figures are not presented well. As an example take figure 4: The y-axis is not legible and necessary information is missing, e.g. which dataset was used, how many different random initializations were used, etc. Additionally, figures should be understandable by the figure caption alone. Al lot of details are missing from the caption. There is no reason not to plot the other two architectures from the previous ablation (those without the activation function). 
    - The terms "oversmoothing" and "vanishing gradient" are not defined precisely. As a consequence, some statements can be confusing, e.g.: (Line 291) "Since the initial features are collapsed to zero in MLPs without proper normalization and residual connections, exponential oversmoothing is not a problem unique to GNNs." This statement directly contradicts the usual definition of oversmoothing as the phenomenon where "for GNNs with nondiverging weights, repeated message-passing invariably leads to the collapse of node signals into a one-dimensional subspace, regardless of initial features." [4]. This surely does not happen in conventional MLPs, where setting the weight matrices $W^{(i)} = I_n$ to the identity matrix displays no such degradation. 

To conclude, the paper leads with an interesting proposition but fails to collect enough evidence to support the claims made. The paper does not add to the previous understanding of the vanishing gradient problem or oversmoothing or their magnitudes in the degradation of node features. The proposed mitigation techniques are known to work for both vanishing gradient and oversmoothing. Additionally, the presentation is subpar.

[1] Lu, Lu, et al. "Dying relu and initialization: Theory and numerical examples." arXiv preprint arXiv:1903.06733 (2019).

[2] Arroyo, Álvaro, et al. "On vanishing gradients, over-smoothing, and over-squashing in gnns: Bridging recurrent and graph learning." arXiv preprint arXiv:2502.10818 (2025).

[3] Chen, Ming, et al. "Simple and deep graph convolutional networks." International conference on machine learning. PMLR, 2020.

[4] Scholkemper, Michael, et al. "Residual Connections and Normalization Can Provably Prevent Oversmoothing in GNNs." The Thirteenth International Conference on Learning Representations.

[5] Kelesis, Dimitrios, Dimitris Fotakis, and Georgios Paliouras. "Analyzing the effect of residual connections to oversmoothing in graph neural networks." Machine Learning 114.8 (2025): 184.

[6] Chen, Ziang, et al. "Residual connections provably mitigate oversmoothing in graph neural networks." arXiv preprint arXiv:2501.00762 (2025).

[7] Kelesis, Dimitrios, Dimitris Fotakis, and Georgios Paliouras. "Reducing oversmoothing through informed weight initialization in graph neural networks" Applied Intelligence 55.7 (2025).

[8] Wu, Xinyi, et al. "Demystifying oversmoothing in attention-based graph neural networks." Advances in Neural Information Processing Systems 36 (2023): 35084-35106.

### Questions
- Can we see the same effects when using other activation functions such as tanh, LeakyReLU, PReLU or ELU?

- Can we initialize the weights in a way that mitigates this phenomenon?

- Could you clarify what you mean by "Since the initial features are collapsed to zero in MLPs without proper normalization and residual connections, exponential oversmoothing is not a problem unique to GNNs." (Line 291)?

- Can you provide more evidence that "Dying ReLU" is mistaken for oversmoothing by the broader GNN community other than Rusch et. al.? E.g. the oversmoothing analysis in [8] applies to any 1-Lipshitz function. They empirically evaluate using GeLU and find in their evaluation a similar result as this paper, in that, ReLU intensifies the problem.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors study the problem of over-smoothing in Graph Neural Networks (GNNs). The authors argue that if we consider three of the main components of GNNs, aggregation, linear transformation, and non-linear activation, prior research studies have mistakenly confused over-smoothing with the vanishing gradient and zero-collapsing phenomenon, caused by transformation and activation rather than aggregation. The paper shows two propositions of over-smoothing which are specific to Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs): degree-scaled embedding convergence and uniform embedding convergence, implying that unless all degrees of the graph nodes are equal, or embeddings converge to the zero vector, 

Their experiments show that aggregation does not play an important role in over-smoothing, while non-linear activation and linear transformation steps contribute to zero-collapsing, meaning that node embeddings converge toward zero.

### Strengths
- Interesting study showcasing that over-smoothing is sometimes confused with zero collapsing in related work
- Selecting the three base components of a GNN, and showing that aggregation does not play a significant role in over-smoothing

### Weaknesses
- The main weakness is the following: from my understanding when talking about GNNs it’s not just vanishing gradients. Over-smoothing and over-squashing can still appear with healthy gradients. Fixes like residual/identity mappings, careful init, normalization, JK connections, DropEdge, etc., help optimization and slow over-smoothing, but they don’t fully eliminate over-squashing or the diffusion-limit behavior. So the statement “performance degradation is not a phenomenon specific to GNNs, and it can be resolved by addressing the vanishing gradient problem” in line 417 is too strong. As seen in previous work by Arroyo et al. (2025), which is cited by the authors, over-squashing can be addressed by a combination of graph rewiring and vanishing gradient mitigation.
- The authors state in their contributions: "We show that the celebrated graph convolutional network can effectively overcome oversmoothing, provided that zero-collapsing is addressed separately". Again to my understanding the authors do not show or prove that in Section 4, rather show some results coming by specific datasets.

### Questions
- It would be interesting to connect all these phenomenon and under what properties/assumptions do they co-occur, e.g. vanishing gradients with oversmoothing

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
4

### Summary
This paper challenges the notion that oversmoothing is the main culprit for performance degradation with depth in GNNs, posits vanishing gradients as the root cause, and experimentally shows that skip connections and normalization can help overcome this.

### Strengths
1. The authors address, albiet not the first time, a common misconception regarding performance degradation in GNNs with depth, i.e. oversmoothing is not the sole culprit and vanishing gradients play a major role in practice. This is a pertinent problem to be highlighted.

2. The cause of confusion between the oversmoothing definitions for GCNs and GATs, and how that has affected commonly used metrics to measure oversmoothing is clarified.

### Weaknesses
1. There have been previous studies that identify similar reasons, primarily vanishing gradients and training problems, as a crucial factor in degraded performance with GNN depth rather than oversmoothing. [1,2,3,4]. Missing relevant literature should be discussed. In fact, some literature with the same insights as this paper are also already mentioned in the related work. This also challenges the novelty and contribution of the paper. 

2. In section 3.2, the authors discuss initialization and zero collapsing. It is known that orthogonality can substantially reduce this if not prevent it. [3,4]. Such initializations are missing from the analysis. 

[1] Decoupling the depth and scope of GNNs (NeurIPS 2021)
[2] Revisiting Oversmoothing in Deep GCNs (arXiv:2003.13663)
[3] Old can be Gold: Better Gradient Flow can make Vanilla GCNs Great (NeurIPS 2022)
[4] Are GATs Out of Balance? (NeurIPS 2023)

### Questions
1. Could the authors be more explicit about what is meant by 'when a model is properly trained' online 53? How would we define 'properly trained'?

2. What model/architecture is used for the cross-hatched bar in Fig 5?

3. For the experiment whre all node features are uniform, why are they set to the features of an arbitrary first node $X_{one}$ and not a random vector? How does the performance vary when nodes from different classes (or even different nodes from the same class) is used as $X_{one}$.?

4. Could the authors comment on other normalization techniques proposed for graphs such as nodeNorm, pairNorm GraphNorm etc, as opposed to batch normalization, and whether they are (or aren't) effective.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors argue that aggregation alone has a minimal impact on oversmoothing, whereas the nonlinear activation and linear transformation steps are responsible for oversmoothing and 'zero-collapsing phenomenon'. Some techniques such as residual connections and batch normalization can be used to mitigate zero-collapsing. The main contributions are: 1. the authors clarify the previous misunderstanding of zero-collapsing as oversmoothing 2. the authors clarify that the aggregation step in GNNs has a marginal effect on oversmoothing compared to transformation and activation. 3. the authors empirically show residual connections and batch normalization are efficient solutions to mitigate oversmoothing. The paper conducts a decomposition of the three GNN operations — aggregation, linear transformation, and nonlinear activation — and quantifies their combined contributions to oversmoothing. Empirical results show performance improvement in deep GNN on skip connection and batch normalization.

### Strengths
1.The paper challenges a previous core assumption in GNN research that oversmoothing has been overstated, whereas vanishing gradient is the main issue caused by transformation and activation rather than aggregation.  
2. The author did a systematic component analysis by isolating the effects of aggregation, transformation, and activation, and clearly show where oversmoothing actually arises with different combination of components.  
3. The experimental results demonstrate that deep GNNs with batch normalization and skip connections can perform effectively without oversmoothing.

### Weaknesses
1.Although the author empirically show the effects of aggregation, transformation, and activation on oversmoothing, the paper provides limited theoretical proofs explaining why aggregation has marginal impact.  
2. Skip connections and normalization techniques are wide-known solutions to oversmoothing. While the findings encourage deeper models, the paper provides limited guidance on how to practically design or train them beyond skip connections and normalization.

### Questions
Can the author show individual impacts of transformation and activation on oversmoothing? Besides skip connections and normalization techniques, can the authors provide additional solutions directly target activation and transformation?

### Soundness
3

### Presentation
3

### Contribution
2
