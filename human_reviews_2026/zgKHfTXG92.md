# Riemannian generative decoder

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
Riemannian representation learning typically relies on an encoder to estimate densities on chosen manifolds. This involves optimizing numerically brittle objectives, potentially harming model training and quality. To completely circumvent this issue, we introduce the Riemannian generative decoder, a unifying approach for finding manifold-valued latents on any Riemannian manifold. Latents are learned with a Riemannian optimizer while jointly training a decoder network. By discarding the encoder, we vastly simplify the manifold constraint compared to current approaches which often only handle few specific manifolds. We validate our approach on three case studies --- a synthetic branching diffusion process, human migrations inferred from mitochondrial DNA, and cells undergoing a cell division cycle --- each showing that learned representations respect the prescribed geometry and capture intrinsic non-Euclidean structure. Our method requires only a decoder, is compatible with existing architectures, and yields interpretable latent spaces aligned with data geometry. A temporarily anonymized codebase is available on: https://anonymous.4open.science/r/rgd-470F.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose an alternative method for Riemannian representation learning that does not rely on an encoder which estimates densities on chosen manifolds. Instead, they use a decoder-only framework that is trained to find manifold-valued latents for arbitrary Riemannian manifolds. The proposed method, termed Riemannian Generative Decoder, is evaluated on three datasets including both synthetic and real-world biological data to demonstrate that it can respect the prescribed geometry and capture the intrinsic non-Euclidean structure.

### Strengths
1. It is an interesting idea to use a decoder-only framework for Riemannian representation learning.
2. It is great that the authors considered various classes of Riemannian representation spaces (spherical, hyperbolic, general geometries) and properly used the regular versus Riemannian optimizer for different parameters.
3. Overall, the aesthetics look great. The colors and styles for most figures and tables are good.
4. Runtime comparison shows significant advantage over most encoder-decoder alternatives.

### Weaknesses
While the idea is interesting, the presentation of the work has several weaknesses.

1. The decoder-only architecture is not well motivated. Why not an encoder-only or encoder-decoder architecture? The authors considered encoder-decoder architectures and mentioned that they optimize numerically brittle objectives, potentially harming model training and quality, but have not supported the claim with results. To back up their claims, I would recommend the authors show "more stable training" and "faster convergence" or something along these lines. On the other hand, the authors do not seem to even consider encoder-only architectures, which can be learned in an unsupervised fashion as well, for example by enforcing similarity of distance measures between the ambient space and latent space (see GAGA [1]).
2. It seems to me that the proposed method depends on a predefined manifold prior which is not learnable, which implies that using the method requires trying out many different types of Riemannian manifolds (and various curvatures) for any new dataset. This seems like a big disadvantage compared to data-driven alternatives that do not assume the manifold geometry.
3. The baselines are a bit limited. The authors mentioned AE, VAE and DGD, but have not included AE methods in baselines. Besides, all the baselines are using predefined manifold priors, lacking data-driven alternatives such as GAGA [1] and geometric AE [2].

[1] Geometry-Aware Generative Autoencoders for Warped Riemannian Metric Learning and Generative Modeling on Data Manifolds. AISTATS 2025.

[2] Geometric Autoencoders – What You See is What You Decode. ICML 2023.

### Questions
See Weaknesses.

Another minor point: For Table 4, I would appreciate if the top rule is added back.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the Riemannian Generative Decoder, a decoder-only framework that learns manifold representations in latent space through combining Riemannian optimization and a standard decoder network. 

Instead of using an encoder, RGD directly parameterizes the latent codes and updates them via Riemannian optimization, jointly maximizing the posterior of the reconstructed points. The paper also introduces geometric regularization that adapts to the manifold's local curvature by perturbing latent variables with noise based on the inverse Riemannian metric, which penalizes large output gradients and aligns the decoder's smoothness with the manifold's geometry. 
By evaluating on three datasets, including a synthetic branching diffusion process, human migrations inferred from mitochondrial DNA, and cells undergoing a cell division cycle, the paper showed that RGD outperformed Riemannian VAE methods in correlation between latent geodesic distance and meaningful biological distance in the data input space.

### Strengths
1. The paper is relatively well organized and structured, with an easy-to-follow background section to properly introduce the decoder-only framework. The method section is well explained and the mathematical derivation for the proposed geometric regularization is clear and well-supported.

2. I appreciate the plots and visualizations for different geometries. The table is well-made.

3. The idea of only using decoder to learn non-Euclidean representations is very interesting. The geometric regularization proposed in this paper is clever and well-motivated. It leverages the inverse Riemannian metric to "undo” the distortion in large curved regions and to ensure that the decoder is penalized evenly across the entire manifold.

4. The training algorithm seems straightforward and relatively easy to implement as it leverages the existing Riemannian optimization packages and analytically known manifolds.

### Weaknesses
1. Lack of motivation and support to validate the need for decoder-only representation learning. The paper claims encoder-only or VAE representation learning is numerically brittle during optimization, but provides no theoretical evidence nor sufficient empirical evidence to back this claim. They empirically compared RGD with only 2 or 1 Riemannian VAE baseline(s) in different datasets. Including more baselines, especially those encoder-only representation learning methods, would strengthen the argument for the advantages of RGD. 

2. The empirical experiments are relatively weak. As mentioned above, comparing with more comprehensive baselines (across encoder-only, AE, and VAE) would lend more support to the decoder-only framework. On the qualitative side, the paper mainly compared with UMAP, include other non-linear/non-euclidean dimension reduction methods such as PHATE, Diffusionmap, etc., would be necessary to provide a fair and comprehensive qualitative comparison.

3. The geometry learned by RGD is limited to known manifolds implemented by the Riemannian optimization package geoopt, and to apply the geometric regularization, the Riemannian metric must be analytically known, contradicting the paper's claim that it “applies to any manifold”.

4. Lack of related work on encoder-only and other geometry representations learning methods, such as Geometry-Aware Autoencoder (Sun, Xingzhi, et al.), Diffusionmap(Coifman, et al.), Poincaré Embeddings(M Nickel, et al.), and PHATE(Moon KR, et al.). For example, Geometry-Aware Autoencoder could learn any latent manifold by matching the local distance in latent space with the meaningful distance in input data space.

### Questions
The concept of using the decoder only for manifold representation learning is intriguing, and the geometric regularization is smart. However, it lacks solid experiments and comparisons with prior methods, which makes the motivation and arguments overall less convincing. In addition, the requirement to select the geometry first and learn it rather than learning any manifold geometry through the data limits the potential scope of this framework.

To further improve the paper, I highly recommend 1) thoroughly discussing existing literature on learning non-Euclidean latent representation methods, 2) conducting more comprehensive experiments, and 3) demonstrating the advantages of RGD with more metrics than distance correlation and reconstruction error. To further prove its downstream performance, I would also recommend testing on more than the clustering task. 

Last but not least, although I appreciate the visualization and aesthetics of the plots, I often need to go back and forth between the plots and texts to understand them. The paper would be clearer if the image annotations were more self-contained.

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
The authors propose a method to train a decoder from a structured latent to a chosen dataset, where the structure can be, in theory, any Riemannian manifold. They propose a simple MAP training objective. Empirically, they validate their method on a variety of datasets, using a host of manifolds for their prior.

### Strengths
- The paper is very well-presented, easy to read. The idea behind it is simple and clear; the derivation of the objective is equally simple and elegant.
- The empirical evidence is extensive, and the set of tasks is varied enough to be convincing.
- Overall, the chosen latents’ structure seems to better represent the structure of the data according to the correlation coefficients on the ranking of the distances.
- The improvements are significant on some of the datasets. In particular, I find the evidence on the downstream tasks most convincing – in terms of the metric itself, and the relative improvements.

### Weaknesses
- Perhaps I have missed out on it in the paper, but I have found that there is little motivation backing the need to choose *a priori* a specific Riemannian structure for the latents of the decoder.
- Since the model is encoder-free, the retrieval of latents is a more complicated task.
- The reconstruction metrics, when available, are typically *slightly* worse than the simplest method – Euclidean.
- I think that the relevance of the correlation scores is to be disputed, despite it being intuitive in some sense: it would have to be shown that respecting the distances of the original space imply better latents. It is only a supposition unless it is a requirement set for some other reason.
- The paper could probably benefit from a listing/pseudocode for inference time.
- The theory is a bit light.

### Questions
- Why is it useful, unless, again, it is an explicit requirement, to impose a specific Riemannian structure on the latents?
- Is it not possible to design similarly a simple (variational) encoder-decoder architecture? Of course, it would be for further work, but what is the fundamental difficulty in designing such a general VAE?
- Why are the correlation coefficients relevant whatsoever? Or could you argue for formally why it could improve anything to have those to be higher? One of the reasons to have latents is dimensionality reduction, and therefore no isometry can be expected in lower dimensions anyway, while “good quality” latents can exist. Moreover, as shown in appendix H, the reconstruction error seems to degrade when correlation seems to augment, at least with respect to the noise regularisation.

Overall, I think it is a good paper, but that just lacks a bit of substance. I would be happy to revise my score if the authors could motivate a bit more why the Riemannian structure is relevant for the latents, especially as they are not available easily without an encoder. It seems to me that letting the optimisation procedure converge to the “best” (well, tautologically, the ones that minimise the loss) latents on its own is the best thing to do. I could be convinced otherwise in some specific settings.

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
2

### Summary
This paper proposes a new approach for learning latent representations directly on Riemannian manifolds without relying on an encoder network. The method involves training a decoder while simultaneously learning manifold-valued latent variables, thereby simplifying manifold constraints and enabling scaling to higher-dimensional and diverse manifolds. Authors introduce a geometry-aware regularization technique based on noise perturbations, which ensures that the smoothness of the decoder aligns with the curvature of the manifold, preserving the intrinsic non-Euclidean geometry of the data. The approach is validated on various datasets, including synthetic branching diffusion processes, evolutionary mitochondrial DNA data, and cyclic gene expression profiles. Overall, this provides a novel framework for representation learning on complex geometric spaces.

### Strengths
- The decoder-only approach for learning latent representations is simple, elegant, and generalizable to any Riemannian manifold. It avoids the complexities of density estimation.
- Empirical validation is comprehensive, including diverse real-world biological datasets that effectively capture hierarchical and cyclic structures.
- The geometry-aware regularization based on noise perturbations is insightful.

### Weaknesses
- The paper builds incrementally on DGD framework by Schuster & Krogh (2023), with somewhat limited novelty in methodology.
- Inference requires optimization at test time to compute latent codes, which could slow down inference.
- The tuning of regularization parameters lacks a thorough discussion and may need adaptive or automated strategies. 
- There may be scalability concerns regarding memory and computation for very large datasets?

### Questions
Please refer to my comments.

### Soundness
3

### Presentation
3

### Contribution
2
