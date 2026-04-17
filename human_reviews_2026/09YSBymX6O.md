# Spatially Informed Autoencoders for Interpretable Visual Representation Learning

- Decision: Accept (Poster)
- Scores: 6, 4, 8

## Abstract
We introduce spatially informed variational autoencoders (SI-VAE) as self-supervised deep-learning models that use stochastic point processes to predict spatial organization patterns from images.  Existing approaches to learning visual representations based on variational autoencoders (VAE) struggle to capture spatial correlations between objects or events, focusing instead on pixel intensities. We address this limitation by incorporating a point-process likelihood, derived from the Papangelou conditional intensity, as a self-supervision target. This results in a hybrid model that learns statistically interpretable representations of spatial localization patterns and enables zero-shot conditional simulation directly from images. Experiments with synthetic images show that SI-VAE improve the classification accuracy of attractive, repulsive, and uncorrelated point patterns from 48% (VAE) to over 80% in the worst case and 90% in the best case, while generalizing to unseen data. We apply SI-VAE to a real-world microscopy data set, demonstrating its use for studying the spatial organization of proteins in human cells and for using the representations in downstream statistical analysis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose using spatial point processes as a self-supervision prior that explicitly models spatial distributions of objects to address the gap in previous un- and self-supervised methods that miss the spatial correlations.  Thus, the paper proposes spatially informed variational autoencoders (SI-VAE) to predict spatial organization patterns from images. The authors apply SI-VAE to a real world microscopy dataset, OpenCell, and correctly identify the protein localization classes.

### Strengths
- Improving VAEs by fusing them with spatial point processes, thus learning statistically interpretable representations of spatial distributions of objects in images can improve biological data analysis.

- Furthermore, the decomposition into interpretable potentials is also a non-trivial contribution that is based on well-established statistical frameworks.

- The paper introduces a principled way of interpreting the learnt representations within spatial statistics. The results on the biological dataset are promising.

### Weaknesses
- The real world applications were limited in the paper. Given the novelty of the proposed method, I would have wanted to see more results on biological data and also interpretations. While the results are promising, it is unclear to me whether SI-VAE generalizes to more complex localization patterns or proteins with overlapping spatial distributions.

- SI-VAE assumes that given z, the image and point pattern would be independent.
But since X is deterministically obtained from x, wouldn't this assumption be false? Could the authors clarify this?

### Questions
- Biological tissues can have direction dependent spatial structure (e.g.  muscle fibers, neuronal axons) which would lead to directionally correlated spatial patterns. Since SI-VAE assumes pairwise interaction potential to be symmetric, isotropic, from an implementation standpoint, how difficult would it be to extend SI-VAE to include anisotropic or direction aware interactions?

- How sensitive are the SI-VAE representations to errors in spot detection, and could the model be extended to be fully end-to-end (e.g. learning jointly the point locations and spatial interactions)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a spatially informed variational autoencoder - SI-VAE, which consists of a VAE augmented with a spatial point process.  The latent representation z from the VAE is used as input to a neural network that then predicts the Gibbs potentials that define the point process.  Experiments are performed with synthetic data, showing a comparison to standard VAE, generalization to unseen processes, and zero-shot conditional simulation.  Lastly, an application to protein localization patterns is given, where it is demonstrated that the learned potentials agree with domain knowledge (eg proteins in vesicles being homogeneously distributed versus nucleus proteins being inhomogeneously distributed within the nuclei).

### Strengths
Interesting proposed model, sufficient technical contribution/novelty.  Validation on synthetic data.

### Weaknesses
Although the proposed model is interesting, majority of the validation is on synthetic data in some sense tuned to the specifics of the model.  Demonstration of applicability to a real problem is somewhat limited - consisting of only one specific test application where the final evaluation is a check that the learned model potentials agree at a high level with what is expected from domain knowledge.  Within this particular task, there is also no notion of a baseline for comparison.  This is an area where the impact could be much improved, by showing broader applicability to other domains, or giving a more qualitative analysis against baseline methods, or showing some unexpected/novel finding instead of confirming existing domain knowledge.

### Questions
see weaknesses above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper developed a self-supervised deep-learning model  that use stochastic point processes to predict spatial organization patterns from images, coined as Spatially Informed Variational Autoencoders (SI-VAE). The self-supervision mechanism is modeled by the Papangelou conditional intensity. Extensive experiments were presented to illustrate the effectiveness of the SI-VAE model.

### Strengths
The paper provided a comprehensive illustration of the idea of Spatially Informed Variational Autoencoders (SI-VAE), which leverages the Papangelou conditional intensity as the self-supervision target for measuring the spatial information of images. Extensive experiments provided convincing results to showcase the effectiveness of the SI-VAE model in capturing spatial interactions and its generalization to unseen data in terms of zero-shot learning. The impact of the SI-VAE was also demonstrated on a challenging real-world application of protein localization in human cells.

### Weaknesses
Despite the strength as mentioned above, the paper only compares the proposed SI-VAE to the original VAE, while ignoring the existence of similar techniques, such as 

1) Semenova, et al., PriorVAE: encoding spatial priors with variational autoencoders for small-area estimation, J R Soc Interface, 2022
2) Jazbec, et al., Scalable gaussian process variational autoencoders, AISTATS 2021.

Such an incompleteness of refereeing and comparisons weakens the overall quality of the paper.

### Questions
What are the connections between the present work with existing Gaussian process variational autoencoders? What is new in the SI-VAE? Does the current model promote easy implementations?

If Gaussian process variational autoencoders can also be used to capture spatial information in the images, can the SI-VAE model still outperform its Guassian process counterparts?

### Soundness
3

### Presentation
3

### Contribution
3
