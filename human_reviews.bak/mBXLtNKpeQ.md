# Pullback Flow Matching on Data Manifolds

- Decision: Reject
- Scores: 5, 3, 5, 3

## Abstract
We propose Pullback Flow Matching (PFM), a novel framework for generative modeling on data manifolds. Unlike existing methods that assume or learn restrictive closed-form manifold mappings for training Riemannian Flow Matching (RFM) models, PFM leverages pullback geometry and isometric learning to preserve the underlying manifold’s geometry while enabling efficient generation and precise interpolation in latent space. This approach not only facilitates closed-form mappings on the data manifold but also allows for designable latent spaces, using assumed metrics on both data and latent manifolds. By enhancing isometric learning through Neural ODEs and proposing a scalable training objective, we achieve a latent space more suitable for interpolation, leading to improved manifold learning and generative performance. We demonstrate PFM’s effectiveness through applications in synthetic data, protein dynamics and protein sequence data, generating novel proteins with specific properties. This method shows strong potential for drug discovery and materials science, where generating novel samples with specific properties is of great interest.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes Pullback Flow Matching, a novel framework for generative modeling on data
with manifold structures. Building on the previous study, this work leverages pullback geometry
to define a new metric on the entire ambient space $\mathbb{R}^d$, by learning an isometry that preserves the
geometric structure of the data manifold $\mathcal{D}$ on the latent manifold $\mathcal{M}$. The corresponding metric of
the assumed latent manifold $\mathcal{M}$ is then used to perform Riemannian Flow Matching (RFM) on the
latent manifold. The effectiveness of this method is demonstrated through experiments on synthetic
data, molecular dynamics data, and experimental peptide sequences.

Main contributions: 
1. A novel generative modeling framework that combines isometric learning and Flow Matching. 
2. An objective for isometric learning that only relies on distance measure on data manifold. 
3. Improved parameterization of isometries via neural ODEs.

### Strengths
###  Originality
This paper proposes a new generative modeling approach for datasets with manifold structure. The approach combines flow matching and isometric learning, which preserve the intrinsic geometric structure of the data manifold in the latent space.

### Quality

The research question studied in this paper, i.e. generative modeling for data with manifold structure, is definitely interesting. The proposed learning approach, i.e. isometric learning and flow matching, as well as the training objectives, are reasonable. The experiments are discussed with sufficient details.    

### Clarity

Overall, the presentation of the research question, the learning approach and the numerical experiments of this paper is  clear. 

### Significance

The proposed approach may have certain advantages over existing methods for dataset with manifold structure.

### Weaknesses
1. In the numerical experiments (Section 5), a relatively large part of the text focuses on isometry learning, and less effort is made on the study of the generation part. Results in Table 3 of Section 5.3 show that 1-PFM is better than CFM/PFM, but 1-PFM is solving a modified/simplified generative task instead of generating true data distribution. The same is true for the analogue generation in Section 5.4. Given these facts, the referee is afraid that the presented material may not be sufficient to firmly support some of the statements/claims made in the the abstract and introduction.

1. Section 2 and 3 consider a general setting, but the practical implementation only addresses a special case, which may cause confusion. Concretely, while the latent space is defined as a general manifold $\mathcal{M}$ in Section 2 and 3, it is chosen as the Euclidean space $\mathbb{R}^d$ in practical computations. In particular, if I understand correctly, Eq. (6) and (7) are actually flow matching losses in the latent space $\mathbb{R}^d$ or $\mathbb{R}^{d'}$ with the standard Euclidean metric. Writing them in a general form does not help readers understand the idea.

### Questions
1. What is the dimension $d$ of the data in GRAMPA dataset in Section 5.4? Why is $d'$ chosen as 128? This should be explained.

1. In Eq. (8), how do you choose $\mu$ in $T_{\mu}$? What role does this translation play? How is $\psi$ in Eq. (8) selected, and is it simply the identity mapping when $\mathcal{M}'$ is Euclidean space $\mathbb{R}^{d'}$? Could you provide more related details in the paper?

1. In the submanifold loss (line 279), why is the notation $T_{\mu}$ changed to $T_{z}$? Is $z$ related to $z_i$ in the stability regularization (line 282)?

1. It seems that choosing the latent manifold $\mathcal{M}$ as $\mathbb{R}^d$ with Euclidean metric is  sufficient (to approximate the distance measure $(\cdot,\cdot)^{\mathcal{D}}$ on the data manifold using an isometric mapping). In fact, this is how it was selected in the experiments. Why not present equations in Section 3 in a more specific form? The presentation there in general form makes it harder to understand.

1. In all the experiments except the last one, $d'$ is chosen to be 1. Is this choice made because the datasets are essentially concentrated on one-dimensional manifolds? How to choose $d'$ in more general cases? Can the author consider examples where $d'$ is greater than 1? 

1. The sizes of the datasets in the experiments are quite small, sometimes even lower than the dimension of the data. Can the method apply to large datasets? Training the isometries (using the global isometry loss and graph matching loss) requires iterating over all point pairs, which seems to be computationally intensive when the batch size is large. Can the authors discuss the computational cost of isometric learning, especially when the size of the dataset is large?

 1.   There are several typos throughout the paper.

- Line 070, the sentence doesn't read correctly. 
- Figure 2, the direction of the arrow connecting $x_1$ and $\varphi(x_1)$ seems to be wrong. 
- Line 321:  "as well ass" should be "as well as".
- Line 284: "induces" should be "induced".
- Line 1043: Missing citation.
- Line 279: To align with Eq (8), $I_{d-d'}$ and $0_{d'}$ should be switched in the submanifold loss.
- The initial letter of the names in references, e.g. poincare, ode, riemannian, jacobian, bayes, should be capitalized.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors propose a Riemannian flow-matching scheme by learning a metric that pulls the associated interpolations near the support of the given data. The Riemannian metric is induced in the data space by training a diffeomorphism that isometrically maps the data to a latent manifold, preserving the distances of the original data. They propose an objective function that aims to satisfy the properties of the diffeomorphism. In the experiments, they consider a synthetic example to demonstrate the properties of the proposed model and a real-world problem on protein data to highlight its effectiveness.

### Strengths
- The technical aspects of the paper seem to be solid.
- The approach for learning a Riemannian metric in the data space is interesting.
- The experiments demonstrate the effectiveness of the proposed model.

### Weaknesses
- The paper seems to rely heavily on Diepeveen (2024), with the primary difference being an extension of the diffeomorphism, while the Riemannian flow matching part is a direct application of Chen & Lipman (2024). Although the experiment with protein data is interesting, the technical contribution feels somewhat limited.
- While the writing is technically accurate, the clarity could be improved. For example, adding figures would aid in explaining the modeling approach.
- The objective function is reasonable, but it involves many hyperparameters, which are not straightforward to select (see questions).

### Questions
Q1. If I understood correctly, the goal of the diffeomorphism (in the simplest scenario) is to map the data into a Euclidean space of the same dimensionality, where the straight-line distance between data points corresponds to the geodesic distance in the original space (as measured by the Isomap approach). I believe Fig. 1 and Fig. 2 could be improved to aid understanding, or perhaps, adding a new figure.

Q2. The objective function is rather complex and depends on many hyperparameters. A critical quantity, the geodesics between the original data are computed using a graph-based method. While this should work well for the ARCH dataset, we know that graph-based interpolations are less ideal in high-dimensional spaces. Additionally, how do the parameters $\alpha_i$ affect the learned model? I believe an ablation study would have been useful here.

Q3. The data are mapped near a lower-dimensional submanifold, $\mathcal{M}_{d'}$, where dimensionality $d'$ is an additional hyperparameter. How does the choice of $d'$ influence the results? Also, is this achieved by pushing the $d - d'$ dimensions near zero using the "submanifold loss"? I think the diagonal terms in this part of the objective function should be inverted.

Q4. Minor: I think that definitions for $\psi$ (line 229) and the concept of local isometry should be included.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The manuscript is on learning representations tailored for downstream tasks such as interpolation on a lower dimensional manifold and generation. The proposed method uses a Neural ODE to learn a diffeomorphism, which enable simulation free training of dynamics with flow matching.

### Strengths
The introductions and background read well, especially I find the method to be well motivated. Overall, the manuscript is clear and the presentation is easy to follow. The authors validate their method on toy and real data, and they conduct an ablation study on various loss terms.

### Weaknesses
- The contributions of the paper appear somewhat limited, as it builds upon Diepeveen (2024). I think it would be beneficial if in the main body of the text the authors explained in more details how their work is different from Diepeveen (2024).
- To improve on the current experiments, It would be great to add another toy dataset with known geodesic, e.g. a swiss roll rotated in higher dimensions..
- On line 200, the authors mention that the method requires fewer epochs, but I don't see an experiment backing this claim. Specifically, it could be interesting to compare total training time including learning isometries and dynamics. 
- My main concern with the experiments is that there are no comparison with other methods using geometric regularization. The authors mentioned a few in the related work section, but they are not used for comparisons.

### Questions
Questions and minor comments:
- I find the notation for the metric $(\cdot,\cdot)^M$ a bit unconventional, maybe $g$ is better? Further, the dependence with $p$ is not clear with the current notation, whereas you could use $g_p$.
- The related work section could be improved. For instance [1,2,3,4] are work that explore geometric regularization of latent space, specifically [1] regularize for geodesics. 
- The link with scaling laws in the introduction is a bit confusing. What I understand is that having prior knowledge on the geometry of the data is a great inductive bias, so learning the geometry should be useful. 
- On line 200, the authors mention that the method requires fewer epochs and parameters, could you add a reference to the section backing this claim ? Further, does this claim take into account the training of the isometries ?
- On line 287 "we demonstrate the effectiveness of the new regularization terms", what are these new regularization terms ? Is it the graph matching loss and the stability regularization ?
- In section 5.1, for the isometry computation, my understanding is that you need a ground truth geodesic distance. How do you compute it for I-FABP?
- In section 5.3 and 5.4, how many seeds did you use for the experiments ? Could the authors include standard deviation?
- Missing reference on line 1044.


[1] Huguet, Guillaume, et al. "Manifold interpolating optimal-transport flows for trajectory inference." Advances in neural information processing systems 35 (2022): 29705-29718.

[2] LEE Yonghyeon, Sangwoong Yoon, Minjun Son, and Frank C Park. Regularized autoencoders for isometric representation learning. In International Conference on Learning Representations, 2021.

[3] Philipp Nazari, Sebastian Damrich, and Fred A Hamprecht. Geometric autoencoders–what you see is what you decode. 2023.

[4] Fasina, O., Huguet, G., Tong, A., Zhang, Y., Wolf, G., Nickel, M., ... & Krishnaswamy, S. (2023, July). Neural FIM for learning Fisher information metrics from point cloud data. In International Conference on Machine Learning (pp. 9814-9826). PMLR.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces Pullback Flow Matching (PFM), a novel generative modeling framework designed to operate on data manifolds, which uses pullback geometry and isometric learning to preserve the geometric structure of data manifolds while enabling efficient generation and precise interpolation in latent spaces. PFM leverages Neural Ordinary Differential Equations (Neural ODEs) to improve the expressiveness and efficiency of learning diffeomorphisms. This approach enables more accurate interpolation and generation of data. The paper demonstrates the effectiveness of PFM through experiments on synthetic datasets, molecular dynamics simulations, and protein sequence data. The results show that PFM can generate novel proteins with desired properties, indicating its potential for real-world applications in scientific domains where precise modeling of complex systems is required.

### Strengths
- This paper demonstrates strong mathematical foundations and theoretical rigor, which helps ensure the method’s reliability and provides theoretical guarantees for its performance.
- The neural ODE-based diffeomorphisms and design of product manifold eliminates need for expensive Riemannian metric tensor calculations and requires only geodesic distance approximations.

### Weaknesses
- While PFM is presented as an improvement over RFM, there’s no quantitative evidence showing the relative performance. The paper does not provide direct experimental comparisons between PFM and RFM.
- The paper mainly compare PFM with CFM and VAE, which misses comparasion with more state-of-the-art models, such as Diffusion, GAN, SLERP.
- While the paper claims geometric preservation through pullback geometry and isometric learning, no formal theoretical proof is provided showing that PFM actually preserves geometric structures. The experiments are insufficient to verify geometric preservation claims as well.

### Questions
- The notations can be more clear. For example, the scheduler $κ(t)$ and vector field $v(t)$ are not clearly defined in the main body of the paper. This may lead to misunderstandings of the content of the article.
- The article will be more convincing if experiments that compare PFM with RFM are included.
- How might the properties of different manifolds, particularly those with non-trivial curvature, affect the performance of PFM?

### Soundness
2

### Presentation
3

### Contribution
2
