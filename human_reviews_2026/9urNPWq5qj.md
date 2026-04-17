# Style Waltz: Dancing Between Content and Style in Face Stylization

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Achieving precise artistic control while preserving identity remains a central challenge in facial image stylization, with most methods requiring costly training and offering limited flexibility. This paper introduces **StyleBrush**, a training-free stylization framework grounded in Riemannian geometry, which resolves this tension through a principled, dual-control optimization. Our core theoretical contribution is to reframe style transfer as a geodesic path-finding problem on a latent manifold. By leveraging the pullback metric, we establish a local isometry that validates optimizing a path’s energy in the embedding space as a means to approximate true geodesics, providing a rigorous foundation for style interpolation. This geometric framework is uniquely applied at two critical stages of the diffusion process: first, for interpolating content and style latents to ensure a semantically continuous fusion, and second, for modulating query features in self-attention layers to dynamically control stylization intensity. The unification of these two control mechanisms under a single geometric principle constitutes the primary novelty of our approach, enabling fine-grained and theoretically-grounded stylization control without any model training. Empirical validation on standard benchmarks confirms that our method significantly outperforms existing state-of-the-art approaches across a suite of quantitative and qualitative metrics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents StyleBrush, a training-free framework for facial image stylization based on Riemannian geometry. It reformulates style transfer as a geodesic path-finding problem on a latent manifold, enabling mathematically grounded and fine-grained control over style blending while preserving facial identity. The method introduces two complementary mechanisms: geodesic-based style interpolation in latent space for smooth content–style fusion, and adaptive style injection within self-attention layers for dynamic stylization control. StyleBrush achieves state-of-the-art results on benchmarks like CelebA, MetFace, and WikiArt, outperforming previous diffusion-based and CNN-based methods.

### Strengths
This paper shows theoretical foundation that connects style transfer with Riemannian geometry, offering a principled framework for controlling artistic stylization. By formulating style fusion as a geodesic path optimization problem on a latent manifold, the method provides a mathematically rigorous explanation for smooth and semantically consistent interpolation. Additionally, the framework is training-free and generalizable, making it efficient and easily applicable across different diffusion architectures while still outperforming state-of-the-art methods quantitatively.

### Weaknesses
While the method claim one of its main contribution as training-free, it relies on iterative Jacobian-based optimization that can be computationally heavy. The process of estimating the Jacobian and its vector products using power iteration within diffusion U-Net architectures increases inference time and memory, making the method less suitable for general applications. 

Additionally, the framework assumes that the generator’s mapping g is a local diffeomorphism with moderate curvature, which may not hold for complex or high-frequency style distributions, potentially leading to suboptimal geodesic approximations or loss of fine style texture. 

More practically, while the paper claims to “better preserve identity while transferring style compared to baselines,” it is difficult to determine which result is actually better (e.g., compared to StyleID, ArtFlow, and StyTR^2). This becomes even more unclear in Figure 3 for ablation study since the w/o both variant still appears plausible, showing no preference difference, even though both of the main mechanisms are absent.

### Questions
It would be interesting to see how the method performs if the geodesic path is replaced with a simpler or reversible trajectory (e.g., linear). Would such a path significantly affect the smoothness of style transition or identity preservation?

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
5

### Summary
This paper introduces a method called StyleBrush. A new training-free facial stylization framework. The core idea is to view style transformation as a problem of finding geodesic paths on latent manifolds. This method utilizes dual control optimization based on Riemannian geometry to achieve a balance between artistic style and identity preservation. The first control mechanism interpolates the potential representations of content and style along the geodesic path. The second mechanism dynamically controls the intensity of stylization by adjusting the query features in the self attention layer. The author uses the pullback metric to establish local isometric isomorphism, providing a theoretical basis for their method. Experimental results have shown that this method outperforms existing advanced methods in both quantitative and qualitative indicators.

### Strengths
1.	The main advantage of this paper is that it proposes a principled and novel theoretical framework. Build the style transformation into a geodesic path finding problem on a latent manifold. This provides a solid mathematical foundation for style interpolation.
2.	Proposed StyleBrush, which is a significant advantage compared to many existing methods that require expensive training for new styles. This makes the method more flexible and applicable.
3.	Unifying two control mechanisms, geodesic based style interpolation and adaptive style injectio,n under a single geometric principle is an innovated design that allows for precise control over the stylization process.
4.	The paper is well written, with clear concept explanations and accompanying illustrations.

### Weaknesses
1. Even though the author claims that this method can ensure the preservation of facial identity information when applied to facial stylization. However, in its method, there is no additional control or protection of facial identity information.
2. Facial identity information is a more fragile and complex semantic aggregation compared to class information. However, this method still focuses on optimizing in the latent space based on noise, and cannot achieve accurate semantic registration, which makes it possible for this method to style facial images at the semantic level. Resulting in poor performance of the final stylized facial image.
3. The author provided quantitative experiments to demonstrate the superiority of their method. However, based on the qualitative experiments provided, the facial stylization results generated by this method are not ideal, accompanied by a large number of artifacts that make the generated results appear low-quality and unnatural.

### Questions
1.	How to ensure the invariance of facial identity information during the optimization process of geodesic distance on Riemannian manifolds without training?
2.	This paper did not use any prior knowledge injection, but only optimized the fusion of face images and style images in a noise based hidden space. In my opinion, this inevitably leads to the problem of style leakage. How can this be solved?
3.	In the last section of the method, the author suddenly shifts the focus to the attention mechanism. What part of these attentions are applied to? If it is in UNet, is it full UNet attention or specific layer attention?
4.	Regarding the dual control mechanism, the paper mentions that interpolating only potential code may reduce content details, therefore a second control (style injection) is needed. However, the geodesic path should be the 'optimal' path. Why does the optimal path in the latent space still lead to the degradation of content details, thus requiring a second correction mechanism in the feature space? Does this mean that there are limitations to representing ideal stylized manifolds only in the latent space of the generated model?

### Soundness
2

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
4

### Summary
This paper introduces StyleBrush, a facial stylization approach that casts style transfer as a geodesic path-finding problem on the latent manifold. This formulation enhances both the granularity and the controllability of the transferred styles. Qualitative and quantitative experiments consistently show that StyleBrush outperforms existing benchmark methods.

### Strengths
Following the instructions of this section, the strengths in terms of the four aspects are listed below:

**Originality:** The central contribution of this study is to theoretically re-frame the style transfer task as a geodesic path-finding problem on a latent manifold, which is somehow novel to the best knowledge of the reviewer.

**Quality:** The manuscript is of solid quality, and experiments show that the method produces style-interpolated images with high visual fidelity and fine-grained control.

**Clarity:** Although generally clear, the presentation could still be refined; see the forthcoming “Weaknesses” section.

**Significance:** Because the approach has potential applications beyond stylization (for example, in general image editing), it holds appreciable significance for a broad research audience.

### Weaknesses
While the core idea behind StyleBrush is appealing, some steps in the argument are hard to follow. After describing the limitations of existing work, the paper moves straight to the geodesic-path solution without clearly showing how it tackles each limitation. For example, the Introduction says that geodesic paths give 'careful control' over content–style fusion, but it seems that the reason is not explained. Section 3 likewise uses the term 'optimal' without stating what is being optimized or why geodesic distance achieves it. 

Adding a brief, plain-language bridge would help: first list the specific control problems left unsolved by prior methods, then explain, before diving into equations, how the geodesic formulation answers them and how the solution is conceptually derived. This extra context should make the subsequent technical details easier to understand.

Admittedly, these points may indeed be covered in the manuscript, but they are hard to discern amid the extensive mathematical and theoretical detail presented with limited explanatory guidance.

### Questions
Please refer to the 'Weaknesses' Section for my concern regarding the manuscript. I encourage the authors to provide a concise, intuitive narrative that traces how the limitations of prior work lead, step by step, to the proposed solution, without relying heavily on equations or technical detail. Should this clarification make the contribution more compelling, I will gladly raise my rating; conversely, the score may be lowered if further issues are identified.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes StyleBrush, a training-free facial stylization framework based on Riemannian geometry. It formulates style transfer as a geodesic path-finding problem on a latent manifold, enabling smooth and theoretically grounded control between content and style. The method unifies geodesic-based style interpolation and attention-based style injection under one geometric principle, achieving fine-grained stylization control without retraining diffusion models. Experiments demonstrate clear improvements in both visual quality and identity preservation over existing methods.

### Strengths
- This method reconstructs the style transfer problem into a geodesic path optimization problem on a Riemannian manifold, which is novel. This connection between Riemannian geometry and diffusion-based stylization offers an interesting view.

- This method has solid theory, and the derivation of energy minimization, local isometry proofs, and geometric gradients contributes to a transparent and interpretable framework

- Experiments demonstrate statistically significant performance improvements across a comprehensive suite of quantitative metrics and systematic qualitative assessments

### Weaknesses
- The proposed Jacobian-based geodesic computation, even with approximations, may still be computationally heavy. How practical is it for real-time or interactive applications? 

- The theoretical foundation relies on local isometry between the latent and generative manifolds. How to ensure that the local isometry is definitely established?

- The study lacks a detailed sensitivity analysis of hyperparameters (e.g.,  δ) and their interaction.

- This method does not seem to have a specific design for human faces, so how does it perform on other non-face style transfer tasks? The generality of this method will be stronger if it can be on non-facial domains such as object, scene, or fashion stylization.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
