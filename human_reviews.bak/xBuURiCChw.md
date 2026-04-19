# Isometric Regularization for Manifolds of Functional Data

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6, 6

## Abstract
While conventional data are represented as discrete vectors, Implicit Neural Representations (INRs) utilize neural networks to represent data points as continuous functions. By incorporating a shared network that maps latent vectors to individual functions, one can model the distribution of functional data, which has proven effective in many applications, such as learning 3D shapes, surface reflectance, and operators.
However, the infinite-dimensional nature of these representations makes them prone to overfitting, necessitating sufficient regularization. Naïve regularization methods -- those commonly used with discrete vector representations -- may enforce smoothness to increase robustness but result in a loss of data fidelity due to improper handling of function coordinates. 
To overcome these challenges, we start by interpreting the mapping from latent variables to INRs as a parametrization of a Riemannian manifold. We then recognize that preserving geometric quantities -- such as distances and angles -- between the latent space and the data manifold is crucial. As a result, we obtain a manifold with minimal intrinsic curvature, leading to robust representations while maintaining high-quality data fitting. Our experiments on various data modalities demonstrate that our method effectively discovers a well-structured latent space, leading to robust data representations even for challenging datasets, such as those that are small or noisy.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a novel isometric regularization technique for Implicit Neural Representations (INRs) aimed at regularizing learned functions, a crucial component of optimizing over a theoretically infinite-dimensional space. This regularization is interpreted through the lens of Riemannian geometry over functional spaces, which helps maintain a well-behaved "latent space" while reducing overfitting, even for small or noisy datasets. The proposed method is validated across multiple scenarios, including synthetic, 2D, and 3D visual data.

### Strengths
The proposed method is tested across various types of functional data, including neural SDFs, BRDFs, and neural operators. This cross-domain applicability highlights the generality and versatility of the approach. Experimental results show significant improvements in the quality of latent space, interpolation, and reconstruction tasks. The breath of synthetic to realistic tasks also gives a degree of insight both into the underlying functionality of the method as well as its practical use. The authors also address the computational complexity of infinite-dimensional regularizations and use practical approximations for computational feasibility. Finally, the authors provide an interesting theoretical perspective by interpreting the regularization problem through a Riemannian lens, allowing future researchers to leverage differential geometry principles to derive further regularization algorithms over functional spaces.

In general, this was a concrete, interesting problem with a fresh take that was presented well and demonstrated very well. I enjoyed reading this paper and believe it will make a positive impact on the field.

### Weaknesses
1.  While the paper demonstrates robustness and generalizability across several datasets, it does not provide sufficient exploration into the scalability of the approach for very high-dimensional latent spaces or large datasets. An obvious interesting application is in NeRFs, but in my view it is reasonable to delegate this to future work -- there is enough for the paper itself to stand.
2. The paper lacks comprehensive ablation studies on the choices of parameters and effect of the individual components (e.g. the approximations), which may help future researchers in applying and developing from this method.

### Questions
1. How does the proposed isometric regularization scale with respect to the dimensionality of the latent space and the number of data points? Have you tested its performance on more complex datasets or tasks beyond those mentioned in the paper?
2. How sensitive is the approach to the choice of hyperparameters, such as the weight of the isometric regularization? Is there a systematic way to select these parameters for different types of data?
3. Can you provide more insights into the computational costs relative to other regularization methods, such as Lipschitz regularization or weight decay, especially for larger datasets?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a way to conditional implicit neural models that may have been conditioned on additional information, such as a class label, or shape vector information. The core idea is to create a mapping from latent-codes to generated output in a way that preserves isometry – that is changes in input and changes in output should be related in a way that preserves distances and angles in the input and output spaces, subject to certain scaling parameters encoded by the function Jacobian. The paper shows experimental results that show the favorable properties of this approach.

### Strengths
Geometry-preserving and isometry-preserving loss functions are a novelty, and are a growing area of work in generative models. 

The approach to add an isometry-preserving loss to conditional INRs is novel. Approximate algorithms to compute this isometry measure are proposed which enhance the strengths of the paper.

Multiple types of results for proposed INR learning are shown including on 2D, 3D, and neural operator learning.

### Weaknesses
Notation is unnecessarily complicated and hinders readability. For instance in the definition of the INR itself, \mathcal{X} is more often than just \mathbb{R}^m with m = 1, 2, or 3. Similarly, V is simply  \mathbb{R}. The lean toward generality isn’t helping with comprehensibility, although everything could be described without loss of generation for f: \mathbb{R}^2 -> \mathbb{R}. I would recommend taking such an approach and moving the generalized framework for an appendix instead. We suffer from the same issue once again when defining the structure of the latent space \mathcal{Z} which always happens to be simply a vector space of the type \mathbb{R}^m.

The premise of section 4, which is the core starts of in a confusing way by stating “Without proper regularization, the latent space can become ill-behaved, overfitting to the data instances, which is exacerbated by the infinite dimensionality of the function space” – here it is unclear what is meant by ‘latent space can become ill-behaved’ – because until this point   \mathcal{Z} is referred to as the latent-space, which is more or less given and not learnt end-to-end. E.g as I understood it,  \mathcal{Z} can refer to class categorical labels, or shape-features that are learnt elsewhere. So I am not at all clear what is meant by ‘latent space can become ill-behaved’.

After reading section 4, we are presented with a way to compute a measure of isometry – as I understand it, this measure really applies to the core INR map, but it is not clear how this measure is actually used in a loss function. No loss function is described from what I can tell. It is unclear thus where latent-codes themselves are being optimized for, or is only the INR mapping portion being optimized. On reading the appendix, it is found that network parameters and the latent codes are both being optimized. I would recommend clarifying this upfront, as we do not see any loss function that suggests latent-code optimization, and this makes it hard to understand how this is being done. This also means that the latent-codes cannot be class categorical labels, as I had initially thought they could be. Overall, this leads to multiple types of confusion which I would suggest really clearly describing.

The paper also uses the term auto-decoder multiple times, which is unclear what that means. Do they mean ‘decoder-only’ architecture, or ‘auto-encoder’. I did additional digging among the references cited to see if I am missing something, but I could not find what an ‘auto-decoder’ means anywhere.

Many conditional INRs also include an element of randomness that creates variation in output, which could be easily included in this model, but was less clear if this was already considered. 

Details about training convergence, comparison with other approaches in terms training times, model size would help position the experiments more clearly.

### Questions
Please clarify what is the total loss function that is being optimized for, and what parameters are being learnt, and what are being kept fixed?

How is the latent-space itself being learnt?

Discussion about training convergence and model size comparison (without needing new experiments).

Clarify ‘auto-decoder’ terminology.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a regularization method, Isometric Regularization, for handling manifolds in implicit neural representations (INRs) designed for infinite-dimensional functional data. It suggests that existing regularization methods often over-smooth data, leading to a loss of fidelity. To address this, the authors introduce a Riemannian manifold-based regularization that minimizes curvature and preserves the geometric consistency between the latent space and data manifold. Experiments across multiple data modalities demonstrate that this approach enhances the structure of the latent space, providing better generalization, especially for noisy and small datasets.

### Strengths
1.	The approach is grounded in differential geometry, offering a mathematically rigorous framework for managing functional data representations. 
2.	It demonstrated effectiveness on various data types, including neural signed distance functions (SDFs), bidirectional reflectance distribution functions (BRDFs), and neural operators, showcasing its versatility. 
3.	The method preserves data fidelity while regularizing, resulting in better interpolation, reconstruction, and generalization across different types and qualities of data.

### Weaknesses
1.	This work uses Hutchinson’s stochastic trace estimator to approximate the trace terms, but it is unclear how much additional computational cost this method incurs compared to the baselines. It is better to include a discussion of the computational requirement.
2.	It utilizes offline samples from the "ground truth functional data" to compute the expectation of $J^TJ$. It should provide the justification of accessing the ground truth data.
3.	The regularization process should involve parameter tuning, which might be non-trivial. The paper lacks discussion of how to choose the strength of isometric regularization when applying the method to novel datasets or configurations.

### Questions
1.	In Line 416, it says “The effect of regularization is prominent when the training data N is greatly reduced to 20”. However, it is difficult to object the significant improvement of IsoAD with N=20 from Fig. 6? Could you clarify this argument?
2.	How does the strength of regularization choose for different datasets in the paper?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new regularizer for implicit neural representations (INRs). Motivated by the fact that existing methods such as Lipschitz regularization result in overly smooth neural representations with respect to the spatial coordinates, the authors propose isometric regularization. This regularizer encourages the image of the latent variable model F(x,z) with respect to the latent z variable to form a well-behaved manifold, so that small changes in the latent variable lead to small changes in the resulting INR. As their regularizer requires expensive Jacobian computations at many spatial and latent inputs during training, the authors employ the Hutchinson trace estimator to avoid materializing full Jacobian matrices during training, and pre-sample input points to avoid the need to draw fresh samples during training. They then demonstrate that their method improves on an unregularized baseline and Liu et al’s Lipschitz regularization on surface reconstruction, BRDF learning, and neural operator learning tasks.

### Strengths
- The paper is well-written, and the authors do a good job of introducing the mathematical preliminaries needed to describe their method.
- The experiments persuasively make the case that isometric regularization empirically improves on the Lipschitz regularization baseline.
- The numerical scheme the authors propose in Algorithm 1 to avoid costly Jacobian computations and sampling of spatial coordinates at training time is a reasonable strategy for approximating their regularizer.

### Weaknesses
I’m not sure if I understand the motivation for isometric regularization relative to simpler alternatives. The authors state that existing works such as Liu et al (2022) apply Lipschitz regularization to the latent variable models F to make them smooth across both the input space X and the latent space Z; this is problematic because for any fixed latent z, the function h(z) is overly smooth in X-space. While I agree that this is a notable deficiency in existing methods, why can one not simply enforce the smoothness of F with respect to the latent coordinates alone?

This question is especially salient in light of the high cost of isometric regularization relative to simpler baselines like Lipschitz regularization. I wonder whether the undeniable improvements arising from using isometric regularization outweigh its costs.

### Questions
- How does the computational cost of the proposed method compare to competing methods like LipDeepSDF? I’m particularly interested in understanding how the marginal cost of training the latent variable model with isometric regularization compares to the marginal benefit relative to cheaper baselines.
- Why is it insufficient to enforce the smoothness of F with respect to the latent coordinates alone? Is this especially challenging from a technical standpoint? More challenging than isometric regularization?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an isometric regularization technique for manifolds embedded in an (infinite-dimensional) function space. The goal is to enforce the learned operator $h: Z \to \mathcal{F}$ as an isometric embedding. Here, $Z$ represents a (finite-dimensional) latent space with the standard metric, and $\mathcal{F}$ is the space of functions on $X$, equipped with an appropriately defined metric. A distortion measure that quantifies the lack of isometry (up to rescaling) is introduced, along with an efficient method for its estimation. Numerical experiments validate the effectiveness of the proposed method.

### Strengths
1. The paper is well-written, well-organized, and easy to follow.  
2. The concept of encouraging isometric embedding from $z$—the latent variable—to $h(z) = F(\cdot, z)$, a functional representation, is interesting.  
3. The efficient estimation of a distortion measure that quantifies the lack of isometry is also novel.  
4. Extensive experiments, including applications to neural SDFs and DeepONets, effectively demonstrate the robustness of the proposed method

### Weaknesses
1. Algorithm 1 and its explanation in Appendix B should be expanded. For instance, further elaboration is needed on estimating the numerator in Equation 9. Additionally, in line 14 of Algorithm 1, should $G$ be given as $G = G_2 / G_1$?  
2. Given the significant computation involved in estimating the distortion measure, especially when computing gradients, a comparison of computational time should be included in the paper.  
3. While the method promotes an isometric embedding between the latent variable $z$ and its functional embedding $h(z) = F(\cdot, z)$, it remains unclear if an autoencoder mapping the original input $u \in \mathcal{U}$ to its latent code is isometric. For example, in DeepONet, where $u$ represents the input function, if the encoder from $u$ to $z$ diverges substantially from an isometric mapping, enforcing isometry in $z \mapsto h(z)$ may have limited impact.

### Questions
Please refer to the previous section

### Soundness
3

### Presentation
3

### Contribution
3
