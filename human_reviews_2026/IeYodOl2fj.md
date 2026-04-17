# Probabilistic Decomposable Embeddings: Uncertainty-Aware Composition in Vision-Language Models

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Vision-Language Models (VLMs) organize concepts into shared embedding spaces, enabling compositional reasoning across modalities. Prior works demonstrated that composite concepts can be constructed by combining “ideal words” derived from attribute–object pairs. However, they rely solely on mean representations, neglecting the uncertainty inherent in these embeddings. In this work, we introduce Probabilistic Decomposable Embeddings (PDE), a framework that explicitly models ideal words as distribution. Instead of simply averaging attribute and object vectors, PDE formulates composition as a maximum a posteriori (MAP) estimation problem, producing composite embeddings biased toward concepts with lower variance. This probabilistic treatment yields partner-aware, precision-weighted composites with a simple count-based scale recovery. We first visualize PDE, showing that it reorients composite directions toward higher-precision axes while decoupling direction from scale. On compositional classification, PDE often matches or surpasses linear decomposable embeddings and geodesically decomposable embeddings in both modalities—improving harmonic mean and AUC. These results highlight \emph{compositional pliability} as a useful inductive bias for uncertainty-aware composition in VLM embeddings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The goal of the paper is to decompose CLIP embeddings of composite concepts (e.g., "red apple") into embeddings corresponding to their primitives ("red" and "apple").
The idea is that these primitive embeddings (referred to as *ideal words*) can be recomposed to embed unseen combinations (e.g., "blue apple").
The paper starts from the work of Trager et al. (2023) and Berasi et al. (2025), which propose linear and geodesic decompositions, respectively, and offers a probabilistic extension that models ideal words as Gaussian distributions, instead of single point estimates.
This results in a method that takes into account the covariance of the data distribution;
this encodes what the authors call _compositional pliability_ (primitives with higher variance yield more during composition, while lower-variance primitives dominate).
The paper evaluates this method in the context of compositional zero-shot classification on two standard datasets (UT-Zappos and MIT-States) and shows that it is generally better than the two starting baselines as well as the original CLIP.

Justification for the rating: The paper does a good job at extending the work of Trager et al. (2023) and Berasi et al. (2025), but the contribution is limited to this specific research thread and fails to engage with the very rich literature on compositional zero-shot learning.

### Strengths
- The paper proposes a principled probabilistic extension to existing work on embeddings decomposability.
- The method shows good gains compared to the baselines.

### Weaknesses
- The literature on compositional zero-shot learning (CZSL) is very rich (see below for some references). The authors fail to acknowledge any of these or other related works, and do not compare their approach to any of them neither qualitatively, nor quantitatively.
- The paper doesn't discuss the computational cost of the proposed method. Since it relies on covariances matrices and their inverses, this is certainly slower than the baselines (LDE and GDE). But I assume the computational cost can be amortized by storing the precision matrices for all attributes and objects.

Some references on CZSL:
- Bao, Wentao, et al. "Prompting language-informed distribution for compositional zero-shot learning." _ECCV_, 2024.
- Huang, Siteng, et al. "Troika: Multi-path cross-modal traction for compositional zero-shot learning." _CVPR_, 2024.
- Li, Lin, et al. "Compositional zero-shot learning via progressive language-based observations." arXiv preprint arXiv:2311.14749 (2023).
- Mancini, Massimiliano, et al. "Learning graph embeddings for open world compositional zero-shot learning." _TPAMI_ (2022).
- Qu, Hongyu, et al. "Learning clustering-based prototypes for compositional zero-shot learning." _ICLR_, 2025.
- Saini, Nirat, Khoi Pham, and Abhinav Shrivastava. "Disentangling visual embeddings for attributes and objects." _CVPR_, 2024.
- Zhou, Kaiyang, et al. "Learning to prompt for vision-language models." _IJCV_ (2022).

### Questions
- How does the proposed uncertainty-aware composition compare to other compositional mechanisms (such as those listed in weaknesses section)?
- In Table 1, why do Closed-World+Seen and Open-World+Seen results are (almost) always the same? Shouldn't the Closed-World setting be easier? Are the results correct?
- In Table 1, why are there multiple values underlined in some of the settings (e.g., UT-Zappos, Attr and AUC, text)?
- In Table 1, can the authors elaborate why the image-based composition sometimes works better than the text-based one (UT-Zappos, Closed-World setting)? Is this due to the specifity of the data?
- In the caption of Table 5, what does s = "all" (number of composing primitives) and s = "median" (dataset-wise median of composing set sizes) mean? Isn't the number of composing primitives always two, since the datasets have attributes and objects?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Probabilistic Decomposable Embeddings (PDE). This framework is based on the study of LDE and GDE. They suggest modeling ideal words as probability distributions rather than deterministic mean vectors. This approach addresses the inherent uncertainty and variability within VLM embeddings, resulting in an uncertainty-aware concept representation. T

To formalize the composition of primitives, PDE casts the process as a Maximum a Posteriori (MAP) estimation problem, yielding a precision-weighted fusion that naturally biases the resulting composite embedding toward concepts with lower variance.

### Strengths
The main strength is proposing the probabilistic version of ideal words and modeling it using a Gaussian distribution. They propose using MAP to formulate it, and the proposed approach is sound. The paper also provides a concept of compositional pliability and uncertainty, and stiffness. 

The paper provides experiments comparing the results with GDE and LDE, and the proposed approach performs well in text modality. The paper reports results across multiple datasets (UT-Zappos, MIT-States, CGQA), modalities (text/image), and architectures (CLIP, SigLIP, SigLIP2).

The proposed method unifies the linear, geodesic and probabilistic formulations.

The paper is well-written, with consistent notation.

### Weaknesses
The presentation of the method is good; however, I have questions wrt the difference of the method with Neculai et al. (2022), how novel the method is compared to the idea and how the performance is. In addition, discussing the uncertainty-awareness, the idea is sound. However, I would like to see more analysis over the uncertainty modeling and its role in the model. There is no sufficient theoretical or empirical information provided for the uncertainty aspect of the model.

While the model is performing well, the difference with the baselines are marginal and there is limited impact. Can the paper also provide computational analysis compared to the baselines?

### Questions
Please check weaknesses.

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
This paper proposes Probabilistic Decomposable Embeddings (PDE), a method for compositional modeling in vision–language embedding spaces that explicitly accounts for uncertainty in component concepts. Prior compositional methods like LDE (Linear Decomposable Embeddings) and GDE (Geodesically Decomposable Embeddings) treat ideal words (for attributes or objects) as deterministic points and compose them via vector addition (or tangent‐space addition on the sphere). By contrast, PDE models each ideal word as a Gaussian distribution (mean + covariance), which captures anisotropic uncertainty or “pliability” in various directions of the latent space. Composition is then formulated as a MAP (maximum a posteriori) fusion of those Gaussian distributions, which naturally weights more “stiff” (low-variance) components more strongly and yields a precision-weighted composite. The resulting composite embedding tends to align more with directions of high certainty and decouples directionality from scale.

### Strengths
1. PDE reframes compositional reasoning in vision–language models through a rigorous probabilistic lens. Instead of treating attributes and objects as deterministic vectors, it models them as Gaussian distributions with covariance structures that capture directional uncertainty. This formulation naturally leads to a MAP-based fusion rule grounded in Bayesian inference, offering a sound theoretical foundation for how concepts with varying uncertainty should interact during composition.
2. By incorporating covariance information, PDE captures anisotropic uncertainty—the fact that different semantic directions may vary in confidence. This results in “precision-weighted” composites that are partner-aware, meaning the contribution of each concept adapts depending on the uncertainty of its partner, mirroring how humans modulate conceptual dominance based on context.
3. PDE unifies and generalizes prior deterministic frameworks like LDE and GDE. When variances collapse to zero, PDE recovers these earlier formulations, showing that it extends them rather than replaces them. This hierarchical view clarifies how existing compositional embeddings fit within a broader probabilistic continuum.
4. Experiments demonstrate consistent improvements on compositional classification benchmarks, notably in harmonic mean and AUC, showing that uncertainty modeling leads to better generalization. Moreover, the visualization of covariance-driven reorientations enhances interpretability, as one can observe how PDE shifts composite directions toward regions of higher precision.
5. The paper introduces compositional pliability—a novel inductive bias describing how flexible or dominant a concept is during composition. This notion provides an interpretable link between probabilistic uncertainty and semantic interaction, which could inspire future extensions in multimodal reasoning and uncertainty-aware learning.

### Weaknesses
1. Although PDE shows improvements on compositional classification benchmarks, the experiments are relatively narrow in scope—mostly limited to CLIP-based settings and attribute–object compositions. It remains unclear how well PDE generalizes to more complex or abstract compositional reasoning tasks (e.g., temporal or relational composition) or to other multimodal architectures such as large multimodal transformers.
2. PDE relies on Gaussian assumptions for concept distributions and MAP-based fusion, which may not fully capture the nonlinear and multimodal nature of real-world concept embeddings. Many vision–language representations are highly non-Gaussian, and modeling them with isotropic or approximately elliptical covariances might oversimplify the true uncertainty structure.
3. Estimating reliable covariance matrices in high-dimensional embedding spaces is challenging, especially with limited samples per concept. The paper uses shrinkage and small diagonal regularization to stabilize estimates, but these approximations may blur meaningful anisotropic patterns or require careful hyperparameter tuning.
4. While PDE introduces an elegant notion of compositional pliability, its tangible interpretability for downstream users remains limited. The framework adds theoretical sophistication, but the observed performance improvements are modest, raising questions about the trade-off between added complexity and practical benefit in real-world applications.
5. Because PDE operates post hoc on frozen representations (e.g., CLIP embeddings), it does not jointly optimize the probabilistic structure with the embedding space. This restricts its capacity to reshape underlying semantics and may limit its effectiveness compared to end-to-end learned probabilistic models.

### Questions
1. How sensitive is PDE to the assumption that concept embeddings follow Gaussian distributions? Have you explored non-Gaussian or mixture-based uncertainty models?
2. Could compositional pliability be related to or derived from information-theoretic quantities (e.g., entropy or mutual information) instead of variance?
3. PDE uses a MAP-based fusion—have you considered a full Bayesian marginalization approach, or would that be computationally intractable?
4. In what sense is the probabilistic fusion “partner-aware”? Does it explicitly capture contextual interaction between concepts beyond inverse-variance weighting?
5. How are the covariance matrices estimated in practice given the limited samples per attribute or object? How do you ensure stability in high-dimensional spaces?
6. How does the choice of the isotropic prior and regularization parameter λ affect the results? Is there a principled way to set it rather than tuning empirically?
7.	The paper mentions a “count-based scale recovery.” Could you elaborate on how this mechanism operates and why it’s necessary?
8. Did you explore the impact of using visual versus textual embeddings as the base space for constructing distributions?
9. The evaluation focuses on compositional classification—have you tested PDE on retrieval, captioning, or reasoning benchmarks to assess broader generalization?
10. How significant are the gains when applied to stronger base models (e.g., CLIP variants or large multimodal transformers)?
11.	How robust is PDE to noisy attribute–object pairs, especially in cases where attribute labels are ambiguous or overlapping?
12.	Could you share any intuition or visualization about how PDE changes the geometry of the embedding space globally (not just locally around examples)?
13.	Do you envision PDE being integrated end-to-end into model training (so uncertainty is learned jointly), or do you see it primarily as a post-hoc compositional adapter?
14.	Could compositional pliability be extended to temporal or causal compositions—for instance, actions applied to objects over time?
15.	Do you think uncertainty-aware composition could improve interpretability or trustworthiness in safety-critical applications, such as robotics or autonomous systems?

### Soundness
2

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
3

### Summary
The authors propose Probabilistic Decomposable Embeddings, which take the ideal words interpretability framing and make each word a probability distribution rather than a deterministic point, also modeling conditional effects. This leads to better compositional classification.

### Strengths
This extension to the ideal words methodologies is interesting, intuitive, and captures some insightful intuitions about how concepts should interact, which are interestingly borne out in the theoretical analysis.

### Weaknesses
W1 Though the method is clearly an insightful improvement on LDE, I’m left not quite clear on what it can contribute to interpretability as a whole. As I understand it, the paper does not really provide too many extra insights beyond introducing the methods. Some more qualitative analysis of what including this covariance estimation gives us in terms of concepts, and in terms of understanding the latent space of models, would make the paper more interesting

W2 The presentation is a bit confusing, making it hard to understand the implications of the results. For example, what the evaluation datasets are testing, what broader implications does good AUC of these performance datasets have? Though I can see what table row is doing better, it’s hard to get a good estimation of what any of it means about the latent space decomposition that we are trying to do.

### Questions
Q1: How would you compare the ideal words that we get from PDE to other less supervised methods like dictionary learning? It seems that there is some interpretable and interesting ideas that we get about compositionality, but I’m not sure how to apply them to the overall project of interpretability, and understanding how latent spaces work. 

Q2: What are some examples where the probabilistic approach gave something intuitively more interesting? I think a few qualitative examples would make the paper stronger.

### Soundness
3

### Presentation
2

### Contribution
2
