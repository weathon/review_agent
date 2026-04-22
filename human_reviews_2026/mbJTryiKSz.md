# Neural Feature Geometry Evolves as Discrete Ricci Flow

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Deep neural networks learn feature representations via complex geometric transformations of the input data manifold. Despite the models' empirical success across domains, our understanding of neural feature representations is still incomplete. In this work we investigate neural feature geometry through the lens of discrete geometry.   
Since the input data manifold is typically unobserved, we approximate it using geometric graphs that encode local similarity structure. We provide theoretical results on the evolution of these graphs during training, showing that nonlinear activations play a crucial role in shaping feature geometry in feedforward neural networks. Moreover, we discover that the geometric transformations resemble a discrete Ricci flow on these graphs, suggesting that neural feature geometry evolves analogous to Ricci flow. This connection is supported by experiments on over 20,000 feedforward neural networks trained on binary classification tasks across both synthetic and real-world datasets. We observe that the emergence of class separability corresponds to the emergence of community structure in the associated graph representations, which is known to relate to discrete Ricci flow dynamics. Building on these insights, we introduce a novel framework for locally evaluating geometric transformations through comparison with discrete Ricci flow dynamics. Our experimental results further suggest connections between the evolution of feature geometry, and training time and network depth.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies how features of a neural network evolve as a function of the the depth over the training dynamics under the manifold hypothesis. They show that a linear neural network (NN) can only learn an isometric transformation with high probability. They then argue for the importance of non-linearity in learning meaningful feature representations. They then study the changes in feature geoemtry across layers using a *local Ricci evolution coefficient*. Their experiments show efficacy of this framework and also a new phenomena reflected in the curvature gap which relates to data filtering.

### Strengths
The paper is well written with an argument that the authors build up intuitively. I particularly liked the idea of discretizing a continuous curved space to measure the curvature. Their use of the Ollivier-Ricci curvature is apt and clever. The authors build up to Section 3.2 where they present local Ricci coefficient in an intuitive manner. The various graphs and figures (including in the Appendix) help back their theoretical argument. Their measure correlates well with various phenomena such as over-fitting and data filtering.

### Weaknesses
I see two main weaknesses and hope the authors can clarify these in the rebuttal.

1. **Theorem 3.1:** I am unsure how effective this bound is. I see that in Appendix A.4.1 you show the proportion of Linear NNs that preserve graphs (and give you isometry) but I am not sure if this is a result of your toy empirical setting i.e. sampling points on a unit ball or it is in fact indicative of the lower bound in Theorem 3.1. More specifically, if $m \times \epsilon^3$ is too small in magnitude and $|X|$ is too large wouldn't the lower bound be vacuous? I hope the authors can clarify this point. A few more questions: Is the lower bound independent for all $k$? That is a bit counter intuitive to me. Is $k$ (the size of the neighborhood) something you choose? Also, I realize that assuming data sampled from within a unit ball is common practice but your work hints at an underlying geometric structure of data and I wonder how that is captured in the toy setting in A.4.1.

2. The graphs in figures 2, 4, 3 are missing std error indicators. Is this because of the manner in which you measure the curvature gap, modularity etc or is this because these were omitted? What is the choice of $k$ in Section 4? How does your estimate of Ricci curvature vary with how you construct the discretized graph e.g. the parameters $k,r$? If these details are in the appendix or main body please feel free to point me to the same. Some of these details are missing (see questions below) which is keeping me from giving a higher score. 

I am willing to be convinced to increase my score to accept the paper if these issues are adequately addressed given how well the rest of the authors' arguments are made.

### Questions
I see the following minor issues:
1. Line 173:" While this definition is well-founded in Forman’s framework and computationally efficient, it is often too simplistic to capture the geometric complexity required in many applications": why is it too simplistic? 
2. Line 234: what is the meaning of $\overset{\sim}{=}$? Does it mean isomorphism? I would introduce the symbol in text before the theorem. If this was explained in the paper and I missed it please let me know.
3. Figure 1: What neural network architecture are you using here?
4. figure 5: width is fixed at 25: can you remark how this would change for larger widths? I would expect the the Layer-Ricci coeeficient to vary across widths or maybe stabilize at larger widths.
5. Question related to Ansuini et al. 2017: If the intrinsic dimensionality of feature representations is changing across layers (Figure 3 by Ansuini et al) then how do you relate it to your result on Layer-Ricci coefficients? I see that your measure is over discrete graphs but I assume the edge density of the graphs changes as intrinsic data dimensionality changes. 
6. Line 708: "depth increases representational power" In randomly initialized neural networks Hanin et al [1] found that depth doesn't necessarily increase representational power beyond the number of neurons. Moreover, Brahma et al show that depth helps disentangle the manifolds. I am not sure if this statement can be made with high accuracy as of now.

------------------------------

**References:**

[1] Complexity of Linear Regions in Deep Networks, Boris Hanin, David Rolnick

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper analogizes the evolution of features in neural networks to a discrete Ricci flow and, on this basis, provides several interesting proofs and empirical findings.

### Strengths
The paper offers an interesting new perspective on understanding how features evolve in neural networks and provides some mathematical justification based on random matrix theory. The experimental results—especially Section 4.4—are interesting and reveal how features evolve as the number of layers increases.

### Weaknesses
* Although Section 3.1 offers interesting theoretical results, these results rely primarily on randomness and i.i.d. assumptions. It is currently unclear to what extent they can be generalized to real-world settings. In addition, I find that Section 3.1 is almost independent of the rest of the paper.

* Lines 353–354 state: “these findings provide compelling evidence that the evolution of feature geometry in deep neural networks is fundamentally curvature-driven, closely aligned with Ricci flow.” In my view, the authors appear to conflate correlation with causation. Table 1 only shows a negative correlation between $\mathcal O(x)$ and $\eta(x)$; it does not establish that the former causes the latter. The gradual shrinkage of distances among similar samples across layers is likely driven by their sharing the same labels, rather than by mean curvature.

* This paper is empirical. While it reports several interesting phenomena, it remains unclear how these findings could benefit the deep-learning community. Could the authors articulate potential application scenarios for these findings? Doing so would substantially strengthen the paper.

### Questions
+ Could you elaborate on the assumption used in Theorem 3.1 (line 229)? It seems to impose a kind of smoothness in the feature space. How do you justify that this assumption is appropriate?

+ Given the material presented, considering node-level curvature appears more natural than edge-level curvature (e.g., node resistance curvature, combinatorial curvature, or diffusion curvature). Could the authors discuss this further?

+ Based on the results in Figure 5, as the number of layers increases further, is it possible for the Layer-Ricci coefficient to become positive?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies how the geometry of data representations changes across layers of a feed‑forward neural network by constructing k‑nearest‑neighbor graphs on the features and measuring discrete Ricci curvature/ the local Ricci evolution coefficient on those graphs. The authors prove that nonlinearities (ReLU) are essential to alter geometry; They define local Ricci evolution coefficients that correlate curvature with local changes in distances in pairs of points (x,y) across layers; Finally, they deploy their framework on experiments, where these coefficients are largely negative, consistent with Ricci flow‑like dynamics;

### Strengths
1) Exposition is clear.
2) Results are compelling. 
3) Theory looks correct.

### Weaknesses
Overall, I enjoyed reading this paper (although I am not an expert in Ricci curvature or neural network feature geometry).

I have two main reservations for this paper. The first is that the paper is overstating the significance of its results. The introduction states "We propose a novel early-stopping heuristic that leverages the emergence of curvature-driven geometric behavior during training (Sec. 4.3).4. By analyzing layer-wise curvature-driven transformations, we introduce a geometric criterion for network depth selection (Sec. 4.4)." In my opinion, this is overstated. The authors show some empirical evidence that the behavior of the Layer-Ricci coefficients might be correlated to the test accuracy. But this evidence is circumstantial at best here, and while the authors could hypothesize that this could be used for layer selection or early stopping heuristic, they haven't tried it out. What would this bring compared to regular stopping criteria? While I appreciate the paper and the work that the authors have put in, I do think that the significance of the findings is a little overstated.


The second pertains the specifics of the experiments and conclusions of the authors (see questions below). In particular, the authors don't spend too much time discussing the choice of the number of neighbors k. One could argue that Figure 14 shows some signs of sensitivity to $k$ (also, it might be worthwhile to add error bars on these plots). Did you use symmetric‑kNN (edge if either is in the other’s kNN) or mutual‑kNN? How sensitive are results to this choice and to feature normalization between layers?

### Questions
1) What distance is the local change of distances based on? I'm assuming Euclidean? Can you also explain why it makes sense to compare layer to layer using a Euclidean distance? I would have thought that Euclidean distances would not be the best to compare from layer to layer. So agreed, the correlation takes care of scaling --- but what if you used a spearman instead? Would that be more robust?

2) I am not sure I understand the point of the community structure experiment: "While removing such a small fraction of samples should not noticeably alter the graph geometry, it leads to a qualitatively different behavior: the curvature gap increases consistently across layers instead of collapsing." Doesn't this mean that the metric that the authors are using is extremely sensitive to just a few outliers (5 misclassified points out of 1,000)? Wouldn't this be a red flag for using this method to diagnose problems/ early stopping, etc?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies how data representations change across layers of neural networks. It builds k-nearest-neighbor graphs on the samples at each layer and measures how their pairwise distances and discrete Ricci curvature evolve. Across many of trained feed-forward ReLU networks, the authors find that regions of positive curvature contract and negative curvature expand, mirroring the behavior of Ricci flow in geometry. Thus, network depth acts like a discrete time variable of a geometric flow on the data manifold. Experiments on synthetic, MNIST, Fashion-MNIST, and CIFAR-10 binary tasks confirm this trend. The study attempts to highlight that deep learning implicitly performs a curvature-driven transformation of the data manifold. It also suggests geometric heuristics for early stopping and optimal depth based on when curvature evolution saturates.

### Strengths
- Overall - I think this is an interesting observation and a refreshing approach to investigating neural network architectures 
- large empirical base: 20,000 networks tested across multiple datasets and consistent findings: clear negative curvature–distance correlation across layers.

### Weaknesses
- The main weakness, perhaps is that it is quite unclear what the scope of the claimed observation is to a wide plethora of neural network architectures. The experiments are primarily made with only small binary tasks and MLPs. Residual networks, Neural ODEs, equivariant networks, and transformers seem like the most important avenues for investigation, and it is a bit disappointing to not find anything about that in this submission. Additionally, the heuristics have not yet been validated on real large-scale networks.
- too little conceptual exposition on Ricci flow itself. sec 2.3 and appendix A.2.4 seems inadequate. I also miss a larger discussion on the different physical and computational prior works on Ricci flow and how that connects to deep learning connection explored here.
- k-NN graphs may poorly capture manifold geometry in high dimensions.
- Just as a further proof of concept, it seems valuable to observe if the Ricci flow metrics for standard kernel-based methods for feature extraction. Is the observation exclusive only to neural-based feature representations?

### Questions
Overall, this is an interesting attempt to use Ricci flow to explain nerual network architectures and is somewhat convincing. The lack of comprehensive evaluation is a drawback. Taken together, I am inclined to weigh in borderline slightly positively.

### Soundness
3

### Presentation
2

### Contribution
2
