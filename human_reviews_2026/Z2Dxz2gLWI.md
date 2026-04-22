# $(MPO)^2$: Multivariate Polynomial Optimization based on Matrix Product Operators

- Avg Score: 2.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 4, 2, 2

## Abstract
Central to machine learning is the ability to perform universal function approximation and learn complex input-output relationships from limited numbers of observations.
Suitable Multivariate Polynomial Optimization can in theory provide universal function approximations. However, the coefficients of the polynomial regression model grows exponentially in the polynomial degree. To reduce exponential growth tensor factorizations of the associated weight tensor have been explored, including the canonical polyadic decomposition (CPD) and tensor train (TT) decompositions.
Whereas CPD has expressive power proportional to rank and current TT formulations are feature order dependent with each input feature associated to a specific factorization block.
Furthermore, these procedures account for redundancies sub-optimally in the weight tensor.
We presently explore multivariate polynomial optimization of matrix product operator (MPO) structures forming the (MPO)$^2$.
Notably, the (MPO)$^2$ defines a flexible framework that naturally combines MPO polynomial weight tensors with MPO feature embeddings.
The (MPO)$^2$ consequently produces an expressive yet compact representation of multivariate polynomials that is feature order independent and explicitly accounts for symmetries in the weight tensor.
On a series of regression and classification problems we observe that the proposed (MPO)$^2$ provides superior performance when compared to existing tensor decomposition based multivariate polynomial regression approaches even outperforming conventional universal function approximation procedures on some datasets.
The (MPO)$^2$ provides an expressive and versatile alternative to deep learning for universal function approximation with simple and efficient inference using second order methods.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
In this paper, the authors propose $(\mathrm{MPO})^2$, a variant of the tensor networks. In this framework, the per-coordinate feature maps are replaced by matrix product operators (MPO). Compared with the usual  matrix product state (MPS) approach, this makes the model feature-order-independent and allows them to encode structured priors into the model more easily. The authors also conduct experiments on some dataset from the UCI-ML datasets.

### Strengths
It is nice for a model to be able to handle structured priors more easily and potentially beneficial to be feature-order-independent.

### Weaknesses
* The paper is poorly written. It is either rushed and/or machine translated, as there are numerous typos and confusing 
  sentences in the paper, for example, the use of "apex" in Sec. 2.1 (maybe the authors mean "superscript") and line 184.
* I do not think the topic of this paper fits the scope of NeurIPS. None of the tensor network-based methods the authors
  compare in the experimental section are published in a machine learning conference/journal such as NeurIPS/ICML/JMLR.
  In addition, the method proposed by the authors cannot even beat more traditional learning methods such as Gaussian 
  processes and XGBoost. 
* Conceptually, I also do not see why we may want to use/study this method, as it neither gives strong empirical results
  nor is theoretically founded or amenable to theoretical analysis.

### Questions
See the Weakness section.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces (MPO)²  for multivariate polynomial optimization based on Matrix Product Operator (MPO) structures. The approach integrates MPO-based polynomial weights with feature embeddings to pbtain a symmetry-preserving representation. Experimental results are reported on against existing tensor-decomposition-based polynomial regression methods on regression and classification tasks.

### Strengths
The paper introduces (MPO)², a new tensor-network-based architecture for modeling higher-order multivariate polynomials, extending conventional polynomial tensor networks.

By leveraging Matrix Product Operator (MPO), the model jointly learns feature and polynomial representations while maintaining feature-order invariance.

The method supports loss-agnostic optimization applicable to both regression and classification through second-order minimization.

### Weaknesses
The paper is not easy to follow, nor for readers to quickly get the main idea. I fel that the paper should carefully explain things woth more visually informative presentations to illustrate your idea. 

The performance of the proposed model, when compared with existing baseliens, seems to possess limited advantages. For example, in regression tasks (Table~1), the proposed method is outperformed by XGBoost, GP. The average performance is very close to another tensor based method  - CPD-A. In classification tasks (table~2), the performance is again worse than XGboost and GP.

### Questions
(1) What is the index dd' in equation (5)?

(2) The number of coefficients  is claimed to be independent of the input features, and can you explicitly state what is the number of parameters?

(3) The authors should consider what is the key intended benefit of the proposed model if you are targeting on solving classification and regression tasks. for example, can you report the time consumption of tensor-based models or traditional ML algorithms to show if there is improvement in efficiency?

(4) If traditional baselines have higher accuracy in fitting the data, does it mean that their way of enforcing nonlinear correlation among input features are more advantageous?

### Soundness
3

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
This paper proposes (MPO)², a tensor network approach for multivariate polynomial optimization using Matrix Product Operators. The method addresses limitations of existing tensor decomposition approaches (CPD and TT) by providing feature-order independence and better handling of polynomial coefficient redundancies. The authors combine MPO polynomial weight tensors with MPO feature embeddings through three specialized operators: linear projection, convolution, and masking. Experiments on 20 small tabular datasets and MNIST/Fashion-MNIST show modest improvements over tensor-based baselines.

### Strengths
The mathematical development is rigorous and the MPO formulation elegantly addresses feature-order independence issues in tensor train methods. The specialized operators (linear, convolution, masking) represent thoughtful adaptations to different structural constraints.

The comparison includes relevant tensor network baselines and proper statistical evaluation with multiple random seeds across diverse small-scale problems.

The feature-order independence and explicit handling of coefficient symmetries address legitimate limitations in existing tensor decomposition approaches for polynomial regression.

The alternating natural gradient optimization and comprehensive ablation studies demonstrate careful experimental design within the chosen scope.

### Weaknesses
The paper provides no evidence that (MPO)² can scale to problems of practical modern relevance. Without parameter count comparisons to CNNs on MNIST/Fashion-MNIST, it's impossible to assess efficiency claims. The restriction to polynomial functions without nonlinear activations severely limits expressiveness compared to deep networks.

Evaluation is limited to toy datasets (mostly <10K samples, <60 features) that don't stress-test the method's claimed advantages. Modern deep learning routinely handles millions of parameters and features, while this work's largest dataset has 58 features. The absence of comparisons with modern neural architectures makes the practical relevance unclear.

No discussion of parameter scaling to larger problems, computational complexity comparisons with CNNs/MLPs, or analysis of why polynomial methods would be preferred over established deep learning approaches. The paper doesn't address whether (MPO)² could ever scale to billion-parameter regimes that are standard in current AI such as those LLMs (a 1B model is considered really small! which is already trained with several T tokens, definitely not do-able in (MPO)²).

The restriction to polynomial functions without nonlinearity severely constrains the method's applicability. Most modern ML problems require the expressive power of deep networks with nonlinear activations, which (MPO)² fundamentally cannot provide.

### Questions
1. Can you provide parameter count comparisons between (MPO)² and CNN/MLP baselines achieving similar accuracy on MNIST/Fashion-MNIST? How do parameter requirements scale with problem size?

2. What evidence suggests (MPO)² could scale to modern problem sizes (millions of features, billions of parameters)? What are the fundamental computational bottlenecks?

3. How does the restriction to polynomial functions limit practical applicability compared to neural networks with nonlinear activations? Can (MPO)² be extended with nonlinearities?

4. Given that billion-parameter models are now routine, what specific use cases justify choosing polynomial methods over established deep learning approaches?

5. How does the alternating natural gradient method's computational cost compare to standard gradient descent on equivalent-capacity neural networks?

6. Why not evaluate on high-dimensional problems where tensor methods should theoretically excel, rather than limiting to toy datasets where simple methods often suffice?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes (MPO)^2, a new method for multivariate polynomial optimization that solves the "curse of dimensionality", i.e., the exponential growth of polynomial coefficients. Unlike prior tensor-based methods like CPD (which is less expressive) or TT/MPS (which are dependent on the order of input features), (MPO)^2 uses Matrix Product Operators to create a compact, highly expressive, and feature-order independent model. On various regression and classification tasks, (MPO)^2 demonstrated superior performance compared to these existing approaches.

### Strengths
1. This paper extends the CP/TT-structure multivariate polynomial optimization framework to a two-layer MPO structure. Specifically, linear projection, convolution, and masking MPO structures are discussed in the paper. The experimental results also demonstrate its superiority.

2. This paper is quite clear and easy to understand.

### Weaknesses
1. The novelty of this paper is limited as it extends the CP-structured multivariate polynomial optimization to the MPO structure with a general linear mapping. Thus, the main advantages and novelty seem to be limited.

2. The main advantages of TN-based multivariate polynomials are also not clear. In the experimental section, only limited baselines like MLP, GP, etc are compared in two experiments. In addition, what kind of mapping did the author use for the two experiments? Linear projection? Convolution MPO? or Masking MPO?

3. The motivation of this paper is also unclear.  In Section 2.2.1, the authors claimed that "a linear transformation is ... suitable latent feature ..." is not a very strong motivation for this method. If so, why not stack the MPO structure into a three-layer or even N-layer? Compared with the one-layer CP/TT structure, the main motivation for proposing this (MPO)^2 has not been clearly illustrated.

4. In the optimization algorithm, the authors adopted the natural gradient descent for optimizing the core tensors of MPO. Why not directly adopt the SGD or Adam-like first-order optimization algorithm? Natural gradient descent requires computing the Fisher information matrix,  which will cause high computation cost when the core tensor A is relatively large.

5. The computational time should be discussed or presented, as the two-layer structure may dramatically increase the computational cost.

### Questions
See the above weakness section.

### Soundness
2

### Presentation
2

### Contribution
2
