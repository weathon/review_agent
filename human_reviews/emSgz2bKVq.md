# Score-based pullback Riemannian geometry

- Decision: Reject
- Scores: 5, 5, 6, 5

## Abstract
Data-driven Riemannian geometry has emerged as a powerful tool for interpretable representation learning, offering improved efficiency in downstream tasks. Moving forward, it is crucial to balance cheap manifold mappings with efficient training algorithms. In this work, we integrate concepts from pullback Riemannian geometry and generative models to propose a framework for data-driven Riemannian geometry that is scalable in both geometry and learning: score-based pullback Riemannian geometry. Focusing on unimodal distributions as a first step, we propose a score-based Riemannian structure with closed-form geodesics that pass through the data probability density. With this structure, we construct a Riemannian autoencoder (RAE) with error bounds for discovering the correct data manifold dimension. This framework can naturally be used with anisotropic normalizing flows by adopting isometry regularization during training. Through numerical experiments on various datasets, we demonstrate that our framework not only produces high-quality geodesics through the data support, but also reliably estimates the intrinsic dimension of the data manifold and provides a global chart of the manifold, even in high-dimensional ambient spaces.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors propose a metric between data samples based on the score information of probability density. Many notations are used without clear definitions. I can assume p in line 135 represents a point in the data manifold, and the score is a vector in the tangent space according to Eq. (5). The tangent space similarity determines the metric between sample points. Because the probability density in use adopts the form of energy-based model with a convex energy psi, the equations can be formulated using the gradient of psi. Along with the learning objective function from normalizing flow (NF), a data generating algorithm can be developed. When an NF is employed, the NF neural network is the diffeomorphism mentioned in the text. It is unclear how the metric obtained from D_X\phi_{\theta_2} information makes sense.

### Strengths
The notion of point similarity using Riemanian metric from generative models is interesting. The algorithm works well with synthetic data upon complex manifold.

### Weaknesses
The authors need to provide clear explanations of how the equations are motivated and how the algorithms can be implemented.
Real-world experiment is missing. The authors made a strong manifold assumption, and it is necessary to know how the algorithm works with data having noisy manifold.
What do the contour lines in Figure 2 represent? There is no information what the horizontal and vertical axes represent nor the meaning of the contour lines in the figure.
The paper lacks motivation for why the score information should be used in the metric. Why not simply use \nabla p?

### Questions
Can you provide the motivation why tangent space similarity obtained from density function is related to metric?
Can you provide a detailed procedure for learning?
The figures should be explained better.

### Soundness
2

### Presentation
1

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
This work introduces score-based pullback Riemannian geometry, a scalable framework that combines elements of pullback Riemannian geometry and generative models to enhance interpretable representation learning. Focusing on unimodal distributions, it features a Riemannian autoencoder (RAE) with closed-form geodesics that traverse the data's probability density, allowing for accurate estimation of the data manifold's dimension. Numerical experiments show that the framework effectively produces high-quality geodesics and reliably assesses intrinsic dimensions, even in high-dimensional spaces.

### Strengths
1.	This framework facilitates the learning of interpretable representations by leveraging Riemannian geometry, which can capture complex data structures more effectively than traditional methods.
2.	Under the assumption of the unimodal distribution, the score-based approach facilitates the construction of the pullback geometry with closed-form manifold mappings.
3.	The paper also showed that the resulting geodesics always pass through the support of data probability

### Weaknesses
1.	The proposed approach has a strong assumption on the data distribution (unimodal distributions), which limits its applicability to more complex or multimodal data.
2.	In the experimental results, the proposed method is only compared with some discrete-time normalizing flow methods (NF, Anisotropic NF, Isometric NF). It is too limited.

### Questions
1. What is the computational complexity of the proposed approach? How does it scale with the intrinsic dimension of the data dimension and the ambient dimension? 
2. What is the main difficulty to extend it to multimodal data?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In the article, the authors introduce the scored-base Riemannian structure that utilizes pullback Riemannian geometry in conjunction with generative models. Specifically:
(1) Based on Gaussian-like distribution in the Euclidean space, they define a Riemannian metric, which is linked to the pullback of the score function.
(2) A comprehensive study about this metric is provided, including the geodesics, logarithmic map, exponential map, distance, and barycenter. 
(3) They propose a novel auto-encoder and decoder methods, based on the metric, offering theoretical guarantees on the consistency of estimation (Theorem 1). 
(4) A mechanism for learning probability densities is introduced, utilizing an adapted normalizing flow loss function, with multiple experiments.

### Strengths
In overall, the approach in this article is both innovative and compelling. Specifically:
1. The Riemannian metric structure proposed in the article is notably original, with a comprehensive examination of its fundamental geometric properties. 
2. The proof supporting auto-encoder and decoder mechanism is well-articulated and appears to be mathematically rigorous.
3. The paper is clearly written.

### Weaknesses
1. The data used in the study is somewhat artificial. The work would benefit from employing more realistic datasets.
2. Although the methodology is quite straightforward, it would be preferable if the authors included step-by-step derivations to verify the  geometric properties (mentioned in Proposition 1). 
3. Given that the data is generated from quadratic structures, the model appears to perform well only for data with similar characteristics (such as the banana-shaped data in the paper). For more complex and diverse data structures, the model may require further development to capture such complexities.

### Questions
1. As I mentioned in the weakness section, although the quasi-Gaussian assumption of the data is comprehensible, can we generalize it to another class of density functions?
2. Do we have any computational advantage when using this method comparing with the other ones (for ex, NF, Anisotropic NF, Isometric NF)? 
3.  Can you elaborate on the intuition of using the score function in this article?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a score-based pullback Riemannian geometry—a data-driven geometry in which geodesics follow the data supports. Assuming specific unimodal densities, it derives closed-form expressions for geodesics, distances, the exponential map, and the logarithmic map, and formulates a Riemannian autoencoder with an error bound. Additionally, it proposes a generative model by adapting normalizing flow as a learning algorithm. The model is tested on synthetic datasets, demonstrating its capability to compute geodesics along the data manifold, estimate intrinsic dimensions, and provide global coordinate charts.

### Strengths
- The proposed geometry seems novel, with intriguing closed-form solutions derived from the geometry.
- Most of the derivations seem correct (I could not verify all the details).

### Weaknesses
- The properties of the proposed geometry are insufficiently discussed and lack intuitive explanations. For example, it would be helpful to explain how the geodesics behave and the form of the Riemannian metrics for different distributions. Without such clarification, the current title may overstate the paper’s contributions.
- Practical applications of the proposed geometry, beyond the Riemannian autoencoder, are not clearly identified. For the Riemannian autoencoder, it remains unclear in what contexts it outperforms alternative methods.
- Furthermore, the unimodal density assumption is restrictive; because computational efficiency depends heavily on this assumption, extending the method to multimodal cases while retaining computational advantages seems challenging.
- The experiments are limited to synthetic data, without comparisons to similar Riemannian geometry methods, such as those mentioned in the introduction (e.g., Arvanitidis et al. 2016).

### Questions
In Table 1, how are the starting and goal points for geodesics, as well as the perturbations, selected when evaluating the geodesic and variation errors?

### Soundness
2

### Presentation
2

### Contribution
2
