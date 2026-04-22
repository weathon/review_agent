# Generalization Below the Edge of Stability: The Role of Data Geometry

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6, 8

## Abstract
Understanding generalization in overparameterized neural networks hinges on the interplay between the data geometry, neural architecture, and training dynamics. In this paper, we theoretically explore how data geometry controls this implicit bias. This paper presents theoretical results for overparametrized two-layer ReLU networks trained *below the edge of stability*. First, for data distributions supported on a mixture of low-dimensional balls, we derive generalization bounds that provably adapt to the intrinsic dimension. Second, for a family of isotropic distributions that vary in how strongly probability mass concentrates toward the unit sphere, we derive a spectrum of bounds showing that rates deteriorate as the mass concentrates toward the sphere. These results instantiate a unifying principle: When the data is harder to “shatter” with respect to the activation thresholds of the ReLU neurons, gradient descent tends to learn representations that capture shared patterns and thus finds solutions that generalize well. On the other hand, for data that is easily shattered (e.g., data supported on the sphere) gradient descent favors memorization. Our theoretical results consolidate disparate empirical findings that have appeared in the literature.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel prospective on how data geometry influences the implicit bias of overparameterized two-layer ReLU networks, with a number of theoretical results demonstrating that solutions below the edge of stability have various generalizability properties that can be desirable or undesirable depending on the setting. They provably show that below the edge of stability solutions can adapt to intrinsic lower dimension subspaces within an ambient space and that a spectrum of generalizability occurs determined by the implicit regularization induced by the "shatterability" of the data. Further empirical results are given to validate their theoretical claims and proof techniques.

### Strengths
- The paper provides a novel perspective on how the final solutions of the edge of the stability regime will generalize, while avoiding the dynamics of the rich regime. This formulation allows for intriguing new insights into how the structure of the data affects the implicit bias within the training of these models.
- This work has interesting implications for a number of different areas as outlined in appendix B and helps foster further work into the emerging direction of data shatterability.
- The empirical results are presented nicely with subsection 4.2 adding substantive support to practically verifying the effect of theorems on the form of the representations learnt in two-layer ReLU networks.
- The discussion and further work section is well written and suggests good follow-up directions on the topic.

### Weaknesses
- Data shatterability is not concretely defined or explained as a concept in the paper. While it can be roughly gleaned from previous work, a clear and concrete definition or explanation of the concept within the paper would substantially improve it.
- Definition 2.1 claims to be for "Isotropic Beta-radial distributions" but then proceeds to define "Isotropic alpha-powered distributions". It is unclear what isotropic beta-radial distributions are in this work as I do not believe they are defined.
- The theoretical claims are stated, but not much intuition or interpretation is given other than the overall message stated in the abstract and the introduction. A more fleshed out narrative and explanation between the theorems would have helped a deeper understanding of the work presented.
- The empirical results in subsection 4.1 could be improved by giving a more complete explanation of how to interpret them.
- For the right panel in figure 3 it is unclear why the correlation coefficient is given when the magnitude of the coefficient is so small. Additionally, the current figure provides no sense of how many of the points are in the bottom left corner.
- There are some linguistic issues such as: the first sentence in the "Disclaimers and Limitations" does not make sense, "deffered" on page 4.

### Questions
- Is it possible to run experiments on other architectures ideally to hypothesise how one could extend these results beyond the two-layer ReLU networks?
- Could further experiments be conducted to help elucidate the potential benefit of batch normalization through data shatterability?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies how data geometry (especially intrinsic dimension and shatterablity) determines generalization under the edge of stability implicit bias.
The paper shows that (1) for features from mixtures of low-dimensional balls, such bias provably drives two-layer ReLU networks's generalization to be controlled by the intrinsic instead of the ambient dimension; (2) for isotropic data, its concentration toward boundary / shatterablity controls the generalization.
Such theoretical predictions are verified by experiments. 
In a finer-grained level, in both proofs and empirical results, the implicit path norm regularization induced by EoS mainly regularizes the harder-to-shatter samples, improving generalization on them, and ignores the easier-to-shatter samples. As a result, data geometry of less shatterablity improves generalization.

### Strengths
- The paper proves the adaptation of low-dimensionality and the spectrum of generalization wrt to data concentration under EoS implicit bias. The latter is also equipped with a lower-bound to show tightness.
- The results are verified by empirical results.
- The sketch of proof is clearly discussed. Combined with the empirical results, it also clarifies on finer-grained roles of EoS implicit regularization and clarifies why and how data geometry affects controls generalization under EoS. Such results emphasize the data-dependent nature of EoS regularization and provides explanations on the highly data-dependent behaviour of overparameterized neural networks.
- The two main theoretical results and empirical results point to a promising shatterablity principle.

### Weaknesses
- In Sec 4.1, the experiments use label noise instead of I.I.D. sampling to construct training set and measures MSE losses instead of difference between the empirical and population risks. Such choice makes it more difficult to compare with theoretical results. What is the motivation for such choices?
- Minor:
  - Line 289: "g is the population version of the *weighted*."

### Questions
- In Figure 1a, the slope of theoretical prediction is not marked and compared to the actual slope. They seem to be actual≈-0.9 vs theoretical=-1/6, where a gap is still observed. What is the source of this gap? Does it come from looseness of bounds or that the experimented problem is not the worst case to reach the upperbound (eg, the directions of the $J$ lines are not worst-case)? Or is it beyond the (B)EoS bias, similar to Sec 3.3, and is governed by some bias else? Can this question be answered by some experiments, eg, searching the worst BEoS models and comparing their slopes with the actual and the theoretical? Maybe this question demands too much efforts. But I would greatly appreciate it because it may offer a clear view on the limit as well as relation of EoS biases with other biases.
- The discussion in Sec 4.3 seems quite generic. Is it possible to develop more general generalization bounds for BEoS weights assuming shatterablity instead of specific assumptions like low-dimensionality that leads shatterablity.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates the role of data geometry in shaping the generalization behavior of overparameterized two-layer ReLU networks trained below the edge of stability (BEoS). Building on recent studies on the implicit bias of gradient descent, the authors propose a unifying principle termed data shatterability, which measures how easily data geometry allows ReLU thresholds to separate samples. The main theoretical contributions include:
(1) A generalization bound for data supported on a mixture of low-dimensional subspaces, showing adaptation to the intrinsic dimension (Theorem 3.2);
(2) A family of generalization bounds for isotropic distributions parameterized by a concentration parameter α (Theorem 3.5), together with lower bounds (Theorem 3.6) and a constructive example demonstrating perfect interpolation on the sphere (Theorem 3.7).
Empirical experiments on synthetic data and MNIST illustrate how data geometry affects generalization and representation structure.

### Strengths
1. The motivation is clear and well-grounded in contemporary discussions around implicit regularization and the edge-of-stability regime.
2. The notion of data shatterability offers an elegant conceptual synthesis that connects data geometry, implicit bias, and generalization.

### Weaknesses
1. Definition and formalization of “data shatterability.”
While the paper emphasizes shatterability as the central concept, it is not clearly or formally defined in a mathematical sense. The text gives intuitive descriptions (“harder to shatter data generalizes better”), but the precise operational definition is vague. 

  (1) Can the authors introduce a rigorous definition or metric of shatterability, perhaps analogous to VC dimension or some geometric measure of separability?

  (2) Why do the authors choose the beta-radial distributions with a parameter $\alpha$ to characterize this data property?

  (3) There is a gap between the rates in Theorem 3.5 and 3.6. How are they related to the general claim?

  (4) Would a toy mode concretely showing how different data geometries affect generalization make the concept more intuitive?

2. Clarity and logical structure of results.
The theoretical results (Theorems 3.2–3.7) are presented in isolation, and their interconnections are not fully clear. The reader may struggle to see how they jointly establish a unified principle.

(1) How do the results for subspace mixtures and isotropic distributions fit into a single theoretical framework?

(2) Is there an overarching theorem or lemma that ties them together through the concept of shatterability?

(3) A high-level diagram or summary of theoretical dependencies would help to improve readability and logical coherence.

3. Lack of dynamics analysis.
The paper claims to study generalization “below the edge of stability,” yet the analysis focuses entirely on static properties of stable minima rather than on the gradient descent dynamics that give rise to them.

(1) Without examining the time evolution of GD (e.g., curvature oscillations, stability trajectories), the results seem closer to a stability condition rather than a genuine characterization of the EoS regime.

(2) The current framework could be better described as a stability-based generalization bound rather than an analysis of edge-of-stability generalization.

4. Relation to prior work.
Theorems 3.2 and 3.5 resemble results in [Wu & Su, 2023] and related stability-based analyses, with the main difference being the explicit dependence on data geometry. However, the paper does not clearly articulate the essential technical innovation over these works.

(1) What are the key mathematical difficulties introduced by considering non-isotropic or low-dimensional data distributions, and how are they overcome here?

(2) A more explicit comparison or ablation (possibly in the appendix) would strengthen the contribution.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work studies the interaction between data geometry and the implicit bias of edge of stability.
It shows that EoS bias can drive two-layer ReLU networks to adapt low-dimensionality for mixture of low-dim balls data, and that for isotropic data, its shatterablity determines generalization.
Experiments verify the theoretical predictions.
This work then proposes the principle of shatterablity, where the shatterable data points attract specialized neurons, which are less regularized by the implicit weighted path norm in below the edge of stability.

### Strengths
- Based on the algorithmic bias of EoS, this work provides the data dependence aspect of neural network generalization, a valuable problem in modern data- and algorithm-dependent theory for generalization. 
- The work picks two representative examples of interest, where the first one is related to how neural network overcomes curse of dimensionality and the second one novelly reveals the role of shatterablity. The results are verified by empirical results. 
- This work also reveals that under EoS regularization, the network may still overfits, and it is data geometry with low shatterablity that helps resisting overfitting. 
- This work provides principled lens for studying feature learning and data geometry reflected in it, eg, neuron activation rate that impacts regularization strength of EoS bias and affects generalization.

### Weaknesses
- The paper supports the shatterablity principle using two proved cases, followed by intuitive interpolation/extrapolation. However, a formal results is missing, leaving shatterablity relying on intuitive definition and restricting its application to more complicated data. Is it possible to derive formal definition of shatterablity and provide more abstract generalization bounds with shatterablity and BEoS as parameters?      
- In experiments, the training data is constructed by perturbing the label instead of IID sampling. How does this setting fits into the assumption of theories? Under standard setting, what will be low-dimension adaptation like?

### Questions
- Some works have emphasized the surprising importance of (benign) memorization for generalization, especially under long-tailed data distribution. Then is there any connection from lon-tailedness and memorization to shatterablity and neuron specialization? If so, what benign memorization looks like in the framework of shatterablity? At what threshold does memorization becomes harmful?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates how the geometry of the training data fundamentally controls the implicit bias of gradient descent (GD) and the resulting generalization performance of overparameterized two-layer ReLU networks trained in the "Below Edge of Stability" (BEoS) regime.

### Strengths
The author claims that  "The less shatterable the data geometry, the stronger the implicit regularization of EoS becomes." and illustrates this observation via two specific example.

I thought this is a very interesting result and made a serious try to understand the performance neural network comparing with other "not even wrong" work.

### Weaknesses
I have not checked the whole proof. The results sound reasonable to me. However, to broad its impacts, it would be more beneficial if the author could make more implications of their theoretical results. e.g., its connection with some exiting theories?  Moreover, the rates stated in theorems are more less to technical,  could the authors  make it more comparable with some existing results?

### Questions
Same to the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
