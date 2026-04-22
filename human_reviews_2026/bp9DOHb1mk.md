# A Generalized Geometric Theoretical Framework of Centroid Discriminant Analysis for Linear Classification of Multi-dimensional Data

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
With the advent of the neural network era, traditional machine learning methods have increasingly been overshadowed. Nevertheless, continuing research on the role of geometry for learning in data science is crucial to envision and understand new principles behind the design of efficient machine learning. Linear classifiers are favored in certain tasks due to their reduced susceptibility to overfitting and their ability to provide interpretable decision boundaries. However, achieving both scalability and high predictive performance in linear classification remains a persistent challenge. Here, we propose a theoretical framework named geometric discriminant analysis (GDA). GDA includes the family of linear classifiers that can be expressed as a function of a centroid discriminant basis (CDB0) - the connection line between two centroids - adjusted by geometric corrections under different constraints. We demonstrate that linear discriminant analysis (LDA) is a subcase of the GDA theoretical framework, and we show its convergence to CDB0 under certain conditions. Then, based on the GDA framework, we propose an efficient linear classifier named centroid discriminant analysis (CDA) which is defined as a special case of GDA under a 2D plane geometric constraint. CDA training is initialized starting from CDB0 and involves the iterative calculation of new adjusted centroid discriminant lines whose optimal rotations on the associated 2D planes are searched via Bayesian optimization. CDA has good scalability (quadratic time complexity) which is lower than LDA and support vector machine (SVM) (cubic complexity). Results on 27 real datasets across classification tasks of standard images, medical images and chemical properties, offer empirical evidence that CDA outperforms other linear methods such as LDA, SVM and logistic regression (LR) in terms of scalability, performance and stability. Furthermore, we show that linear CDA can be generalized to nonlinear CDA via kernel method, demonstrating improvements on the linear version with tests on three challenging datasets of images and chemical data. GDA's general validity as a new theoretical framework may inspire the design of new classifiers under the definition of different geometric constraints, paving the way towards a deeper understanding of the role of geometry in learning from data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a novel theoretical framework named Geometric Discriminant Analysis (GDA), which unifies a family of linear classifiers expressible as functions of a Centroid Discriminant Basis (CDB0)—the line connecting two class centroids—adjusted by geometric corrections under varying constraints. The authors formally demonstrate that Linear Discriminant Analysis (LDA) is a special case of GDA.

### Strengths
GDA provides a unified geometric foundation for linear classifiers, explicitly connecting centroid-based discriminants (CDB0) to broader geometric constraints.

 GDA achieves higher accuracy, stability, and efficiency than LDA, SVM, and logistic regression (LR) on 27 real-world datasets.

Kernelized CDa extends linear CDA to nonlinear classification via kernel methods.

### Weaknesses
CDA initializes training from CDB0 (the line between centroids), which may be suboptimal if centroids are poorly estimated.

It is crucial to explore the model's extension to multi-class scenarios since traditional LDA is based on multi-class problems.

While kernelized CDA extends to nonlinear problems, its performance depends heavily on kernel choice and bandwidth.

### Questions
In equation 10, how to determine various correction terms since  LDA is based on the first-order and second-order information of data.

What are the theoretical explanations for obtaining the bias?

If each class of data has multiple cluster centers, using a single center may not be appropriate.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a generalized geometric theoretical framework called Geometric Discriminant Analysis (GDA) for linear classification. GDA unifies certain linear classifiers by defining them as geometric adjustments to a foundational Centroid Discriminant Basis (CDB0), the line connecting the centroids of two classes. Building on this framework, the authors propose a novel, efficient, and scalable linear classifier named Centroid Discriminant Analysis (CDA). CDA starts with CDB0 and iteratively refines the discriminant by performing performance-guided rotations on 2D planes, optimized through Bayesian optimization, achieving a lower quadratic time complexity compared to the cubic complexity of methods like LDA and SVM. The key contributions are the GDA framework itself, the high-performance CDA classifier, and a nonlinear extension of CDA via the kernel method.

### Strengths
The paper demonstrates strong originality on two main fronts. Firstly, the introduction of the GDA framework presents a novel conceptual lens for viewing centroid-based linear classifiers. Instead of treating methods like Linear Discriminant Analysis (LDA) and Minimum Distance Classifier (MDC) as disparate algorithms, the paper reframes them as specific instances within a unified system. This is achieved by defining a foundational "Centroid Discriminant Basis" (CDB0) and characterizing different classifiers by the unique geometric corrections they apply to this basis. This unification is a creative and insightful contribution that provides a new language for understanding and designing these models.

The mathematical derivation of the GDA framework is well-executed, particularly in how it deconstructs LDA into the CDB0 and a covariance-based correction term, with clear explanations for various special cases (e.g., equal covariance, no correlation). This provides a solid theoretical foundation for the proposed concepts.

The empirical evaluation is convincing. The authors test CDA against a comprehensive suite of strong baselines (including multiple fast SVM variants, LDA, and LR) across 27 diverse, real-world datasets spanning standard images, medical imaging, and chemical properties. The breadth of this evaluation strongly supports the claim that CDA is a robust and generally applicable classifier. Furthermore, the inclusion of a large-scale experiment on a 1.3-million-cell dataset directly addresses the critical aspect of scalability, providing powerful evidence that CDA's theoretical efficiency advantage (quadratic vs. cubic complexity) translates into significant real-world performance gains. The rigor of the experiments, utilizing multiple metrics and cross-validation, further enhances the quality and credibility of the results.

### Weaknesses
* The framework is primarily used in a descriptive capacity to show that LDA is a special case. The paper proposes that GDA "may inspire the design of new classifiers," but it stops short of demonstrating how. CDA itself is presented as a single, highly effective derivative, but the framework's utility in systematically generating other novel classifiers is not shown.

* Rationale for Bayesian Optimization (BO). The algorithm uses BO to perform a 1D search for the optimal rotation angle on each 2D plane. While BO is a powerful technique, its use for a simple 1D optimization problem is not justified. Given that the performance-score function is relatively inexpensive to evaluate for a single vector, simpler methods might have been sufficient and faster.

* The section on nonlinear kernel CDA feels underdeveloped compared to the thoroughness of the linear experiments. While the results on two datasets are promising, they are not sufficient to make a strong claim about its general effectiveness.

### Questions
* The GDA framework is presented as a unifying concept. However, a key claim is that the framework can "inspire the design of new classifiers." Please elaborate on how to use the GDA framework to design a new classifier systematically. A concrete, hypothetical example would greatly strengthen the claim of GDA's generative utility.

* The use of Bayesian Optimization (BO) for the 1D search of the optimal rotation angle on each 2D plane is a notable design choice. Could you provide a more detailed rationale for this? Given that the performance-score evaluation for a single vector is computationally inexpensive, simpler methods like a fine-grained grid search might seem sufficient and potentially faster in terms of overhead.

* The preliminary results for the nonlinear kernel CDA are promising but are limited to two datasets. Could you please provide more experimental evidence about this claim?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The work proposes a theoretical framework named geometric discriminant analysis (GDA), which includes the family of linear classifiers that can be expressed as function of a centroid discriminant basis (CDB0). A linear classifier named centroid discriminant analysis (CDA) is defined as a special case of GDA under a 2D plane geometric constraint.

### Strengths
1.	A geometric theoretical framework for classifiers is proposed.
2.	A scalable linear geometric classifier is designed.

### Weaknesses
1.	The authors claimed that the proposed GDA is a framework includes the family of linear classifiers, but in fact only on CDA is designed. The claimed way for this is not suitable.
2.	There are many classifier works, but the authors only compared LDA and SVM, which cannot reflect the sota of the propose method.
3.	In the experiments, only LDA, SVM and LR are used as compared methods, which has no persuasiveness. There are many L
4.	The proposed method uses kernel, so why not add kernel LDA for comparison?
5.	As we all known, linear classifier is hard to discuss nonlinear data. In fact, the distribution of the data is always nonlinear, so how about the generation of the proposed method for nonlinear data?
6.	Minor errors: Where in line 147; Eq. equation 6 in line 177 and similar errors in other places.

### Questions
Please see the Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a Geometric Discriminant Analysis (GDA) framework, a unified geometric perspective for linear classifiers based on a Centroid Discriminant Basis (CDB0)—the vector connecting class centroids—adjusted by geometric corrections. Within this framework, the authors propose Centroid Discriminant Analysis (CDA), a linear classifier that iteratively rotates the discriminant in 2D planes using Bayesian optimization to maximize a performance score. CDA achieves quadratic time complexity, outperforming traditional methods like LDA and SVM (which have cubic complexity) in scalability. 

Extensive experiments on 27 real-world datasets (images, medical, chemical) show CDA's superior performance, stability, and efficiency. A kernel-based nonlinear extension (nCDA) is also introduced and shown to improve upon linear CDA and kernel SVM on challenging datasets.

### Strengths
- computational efficiency of CDA, which achieves quadratic time complexity, 
This potentially makes the CDA significantly fastethan traditional cubic-complexity methods like LDA and SVM.
 This is supported by  empirical evidence across 27 diverse real-world datasets, where CDA consistently outperforms established linear classifiers in terms of predictive performance, stability, and training speed, including on a large-scale single-cell dataset

- the method offers  interpretability due to its geometric foundation and performance-dependent learning process, 

- the method can be extended to a kernelized nonlinear version (nCDA) demonstrates flexibility and improved capability on complex data, broadening its applicability.

### Weaknesses
-  The GDA framework is presented as a conceptual model and unifying perspective but does not seem to have rigorous theoretical guarantees?

- The training process relies on several heuristic components, including Bayesian optimization for rotations and a sample reweighting strategy, which are difficult to follow and are maybe not robust?

-  while the core CDA algorithm is efficient, the kernelized version inherits the standard computational bottleneck of kernel matrix construction, and the overall approach remains fundamentally designed for binary classification, relying on external methods  for multiclass problems.

### Questions
- how does the training time of CDA scale for a multiclass problem ?

- The paper uses flattened raw pixels and simple tokenization. Do you believe CDA's performance is currently limited by this pre-processing? Have you tested it on more sophisticated, learned features (e.g., from an autoencoder) to see if it can serve as a powerful final-layer classifier?

- In high-dimensional spaces, the 2D planes for rotation are constructed heuristically. How stable is the Bayesian Optimization process  especially in the initial rotations where the number of samples is low?

- Can you show that the method is not loosing too much performance compared to LDA if the data is (exactly) Gaussian?

- what about the performance on elliptic distributions? Maybe it could be interesting to show how this compare with "robust" methods build for elliptic distributions.

### Soundness
3

### Presentation
3

### Contribution
3
