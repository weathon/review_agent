# $\infty$-Diff: Infinite Resolution Diffusion with Subsampled Mollified States

- Decision: Accept (poster)
- Scores: 6, 8, 8, 6

## Abstract
This paper introduces $\infty$-Diff, a generative diffusion model defined in an infinite-dimensional Hilbert space, which can model infinite resolution data. By training on randomly sampled subsets of coordinates and denoising content only at those locations, we learn a continuous function for arbitrary resolution sampling. Unlike prior neural field-based infinite-dimensional models, which use point-wise functions requiring latent compression, our method employs non-local integral operators to map between Hilbert spaces, allowing spatial context aggregation. This is achieved with an efficient multi-scale function-space architecture that operates directly on raw sparse coordinates, coupled with a mollified diffusion process that smooths out irregularities. Through experiments on high-resolution datasets, we found that even at an $8\times$ subsampling rate, our model retains high-quality diffusion. This leads to significant run-time and memory savings, delivers samples with lower FID scores, and scales beyond the training resolution while retaining detail.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
**Edit: I have raised my score to 6 to reflect the author's updates to the submission.**

This paper proposes a generalization of diffusion models for data living in infinite dimensional Hilbert spaces. The main motivation for doing so is to enable models which can be trained and sampled at arbitrary resolutions and to enable scaling of diffusion models to high-dimensional data. Section 3 discusses some theoretical concerns for developing such models, Section 4 proposes an architecture well-suited for this task on natural image data, and Section 5 provides an empirical evaluation of the proposed methodology.

### Strengths
- The proposed neural architecture is novel and a significant contribution to the literature on infinite-dimensional models. In particular, the proposed model shows significant performance gains in comparison to several previously proposed infinite-dimensional generative models (Table 1) in terms of FID scores.
- The experiments are generally well-executed and convincing in terms of the conclusions drawn from these throughout the paper.

### Weaknesses
- A major weakness of this paper is how the authors frame their contributions within the existing literature. 
     - For example, in the abstract the authors write "Unlike prior infinite-dimensional models, which use point-wise functions requiring latent compression, our method employs non-local integral operators to map between Hilbert spaces, allowing spatial context aggregation." Similarly, in Section 1, the authors write "We introduce a new Gaussian diffusion model defined in an infinite-dimensional state space that allows infinite resolution data to be generated (see Fig. 2)". However, numerous recent works have posed infinite-dimensional diffusion models which precisely use these integral operators and a similar theoretical framework [1, 2, 3, 4, 5, 6, 7]. 
     - The authors are clearly aware of this work (see Section 6) but do not appropriately frame their contributions, i.e. many of these prior works develop the theory of infinite-dimensional models which is very closely related to the proposed theory, but the authors of this submission do not state this. The authors claim that these works are concurrent, but the earliest of these works appeared in December 2022 [4], February 2023 [1], and March 2023 [2, 3], more than 6 months prior to the ICLR submission deadline. In addition, the authors are missing a citation to Kerrigan et al. [4] who previously develop a theory for discrete-time diffusion models which is closely related to the theory the authors propose in Section 3, as well as references to several other infinite-dimensional diffusion models in continuous time [5, 6, 7].

- The theory in Section 3 is imprecise to the degree of incorrectness. 
     - The authors write "The Radon-Nikodym theorem tells us the density for a measure $\nu$ absolutely continuous with respect to a base measure $\mu$". This is not what the Radon-Nikodym says -- the Radon-Nikodym theorem states the *existence* of a density given the absolute continuity. To actually compute this density, you require stronger results, such as the Cameron-Martin and Feldman-Hajek theorems. See [1, 4] for a discussion of these theorems in the context of diffusion models.
     - The authors miss key regularity assumptions necessary on the Gaussian noise in order to obtain a well-posed model, see again [1, 4] for a discussion of these requirements.


### Minor
- The discussion of the "diffusion autoencoder framework" in Section 5 could use a short description for those unfamiliar with the work, i.e. it was not entirely clear to me how model is trained based on the description provided alone.


### References:
[1] [Lim et al., Score-Based Diffusion Models in Function Space](https://arxiv.org/abs/2302.07400)

[2] [Franzese et al., Continuous-Time Functional Diffusion Processes](https://arxiv.org/abs/2303.00800)

[3] [Hagemann et al., Multilevel Diffusion: Infinite Dimensional Score-Based Diffusion Models for Image Generation](https://arxiv.org/abs/2303.04772)

[4] [Kerrigan et al., Diffusion Generative Models in Infinite Dimensions](https://arxiv.org/abs/2212.00886)

[5] [Lim et al., Score-based Generative Modeling through Stochastic Evolution Equations
](https://neurips.cc/virtual/2023/poster/72191)

[6] [Baldassari et al., Conditional score-based diffusion models for Bayesian inference in infinite dimensions](https://arxiv.org/abs/2305.19147)

[7] [Pidstrigach et al., Infinite-Dimensional Diffusion Models](https://arxiv.org/abs/2302.10130)

### Questions
See weakness section.

Overall, I think this work provides an important and novel practical contribution (namely an empirically sound architecture for infinite-dimensional diffusions), but the contributions regarding the theory are over-stated and the framing of the work with regards to prior work in this area needs to be significantly improved.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper considers the generative modeling problem and the diffusion models in particular. The proposed novel diffusion model is defined in an infinite-dimensional Hilbert space in order to possibly model infinite resolution data. The model is trained only on randomly sampled subsets of coordinates and denoising data only there. It allows for learning a continuous function for arbitrary resolution sampling. Whereas the standard infinite-dimensional models use point-wise functions requiring latent compression, the proposed model employs non-local integral operators to map between Hilbert spaces and, as such, allows spatial context aggregation. The method is compared against the state-of-the-art models across different tasks achieving comparable results.

### Strengths
The paper has a few significant strengths overall, which I will outline below:
1. Proposed model achieve the best or comparable results to the state-of-the-art models.
2. Formulating the generative diffusion model in an infinite-dimensional Hilbert space and allowing to denoise data only on a subset of coordinates is very interesting. I would assume that it might be to complex problem but maybe the mollification is why the model is able learn?
3. Formulating the diffusion model in a Hilbert space in a strict mathematical regime, which seems to be correct.
4. Including additional experiments on not so common tasks like inpainting.
5. The presentation is very clear. Overall, the flow of the manuscript is well-organized.

### Weaknesses
However, despite the strengths, the paper has a few major and minor weaknesses:
1. The works based on similar ideas are mentioned only in the Discussion section, but the proposed model is not compare against them in an experimental way. 
2. The lack of information regarding the computational costs of both training and inference for the proposed model. 
3. I couldn’t find the information which is actual upscaling procedure limit, which will be significant from the practitioner point of view.

### Questions
I would like to see especially the following experiments and improvements regarding specifically to the Weaknesses section:
1. It will be great if the authors might compare the works mentioned in the Discussion section against their model in the same experimental settings.
2. I would like to see a new experiment comparing the computational costs of proposed model on different resolutions (e.g., in FLOPs).  
3. The authors show the results on up to 8x subsampling rate but I’m considering what will be the highest subsampling rate (and resolution) when the model is still doing well. Please, if you could include such comparison.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces an infinite-dimensional diffusion model to handle images at arbitrary resolutions. The authors introduce an architecture based on neural operators that achieves state-of-the-art FID among infinite-dimensional approaches at resolutions up to 256x256.

### Strengths
The network architecture seems to have been well designed, and the results are very convincing. In particular, the use of irregular grids seems to be a good tradeoff between efficiency and retaining fine detail information (particularly if the sampling grid varies from training point to training point, which is not clear).

### Weaknesses
- The paper lists the extension of diffusion models to infinite-dimensional spaces as one of its contributions, but there are already many works on this topic [1-5] (of which only [2-4] are briefly mentioned in the paper). The authors should discuss the relationship of their framework to this related work in greater detail (and they should be mentioned before the discussion). Also, the mathematical treatment of the extension to infinite dimensions in the paper is lacking compared to [1-5] and thus cannot be considered a contribution.
- While the paper aims to work at any resolution, the use of "mollification" (I suggest to simply call this blurring which is a better-known term in the NeurIPS community) on the data distribution effectively limits the maximum resolution of the generated images. A close inspection of Figure 2 also reveals artifacts on the images generated at resolution larger than the training resolution. On a related note, the authors never discuss the mechanisms by which the score network should be expected to generate to higher resolutions while never having observed high-resolution images. Could the authors expand on this point?
- The central contribution of the paper seems to be its architecture. While its components are described rather abstractly in Section 4, I do not understand what is being implemented exactly. The discretization of the integrals is never discussed in detail, as well as the grid on which the various intermediate activations are represented. When using a regular grid, what is the difference between the proposed architecture and a regular CNN such as UNet? As another important point, how discretization invariance is obtained when using images with different resolutions is not discussed. Does the number of points used in the finite-sum approximation of the integrals changes?
- Super-resolution and inpainting are rather unfaithful to the original images: downsampling or cropping the generated image does not yield back the input image.

[1] Kerrigan, Gavin, Justin Ley, and Padhraic Smyth. “Diffusion Generative Models in Infinite Dimensions.” arXiv, February 24, 2023. https://doi.org/10.48550/arXiv.2212.00886.

[2] Lim, Jae Hyun, Nikola B. Kovachki, Ricardo Baptista, Christopher Beckham, Kamyar Azizzadenesheli, Jean Kossaifi, Vikram Voleti, et al. “Score-Based Diffusion Models in Function Space.” arXiv, February 14, 2023. https://doi.org/10.48550/arXiv.2302.07400.

[3] Hagemann, Paul, Sophie Mildenberger, Lars Ruthotto, Gabriele Steidl, and Nicole Tianjiao Yang. “Multilevel Diffusion: Infinite Dimensional Score-Based Diffusion Models for Image Generation.” arXiv, April 29, 2023. https://doi.org/10.48550/arXiv.2303.04772.

[4] Franzese, Giulio, Giulio Corallo, Simone Rossi, Markus Heinonen, Maurizio Filippone, and Pietro Michiardi. “Continuous-Time Functional Diffusion Processes.” arXiv, July 7, 2023. https://doi.org/10.48550/arXiv.2303.00800.

[5] Pidstrigach, Jakiw, Youssef Marzouk, Sebastian Reich, and Sven Wang. “Infinite-Dimensional Diffusion Models.” arXiv, October 3, 2023. http://arxiv.org/abs/2302.10130.

### Questions
- What is the "pointwise" constraint that is mentioned in the abstract and conclusion? Perhaps related, could the authors elaborate what is meant by "coordinates are treated independently" in neural fields?
- In section 4.1, what does the notation $c \in {m \choose D}$ means?
- In equation (13), I am confused by the use of both $x$ and $v_l$. I do not understand if the operator represents a function from $x$ to $s$ or $v_l$ to $v_{l+1}$.
- I did not understand the second paragraph of section 5 on diffusion autoencoder framework. What do the authors mean by "using the first half of our architecture"?
- What do the authors mean by "each model being trained to optimize validation loss" in Appendix A? Is validation data used to train the networks on top of training data?

### Soundness
3 good

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this work, the mollified diffusion models is proposed to process infinite resolution data efficiently and effectively, to overcome the weakness of low sampling speed of existing diffusion models, where the mollification of data, Fourier neural operators, multi-scale architectures and efficient sparse operators are applied. The experiments are conducted accordingly with impressive results.

### Strengths
1. Originality: It is a novel idea to apply mollification for diffusion models  to enable the data to be processed in the Hilbert space and allow diffusion models to generate high resolution or even infinite resolution data with efficacy. Smoothing using mollification can ensure the regularity of data and enhance enable the modelling in the Hilbert space to reduce the sampling speed.

2. Quality: The experimental results seem to be promising and impressive quantitatively and qualitatively. FIDs of the proposed $\infty$-Diff are comparative to those of finite-dimensional methods. 

3. Significance: In this paper, regularity is discussed in diffusion models. It opens the horizon of future theoretical analysis of diffusion models.

### Weaknesses
The writing really needs to be polished. A lot of typos should be corrected, for example, Page 4 Section 3.2 Paragraph 1 Line 11 $\mu_\theta: \mathcal{H}, \mathbb{R} \rightarrow \mathcal{H}$ → $\mu_\theta: \mathcal{H} \times \mathbb{R} \rightarrow \mathcal{H}$; Page 5 Section 4 .1  Paragraph 2 Line 1-2 “respectfully” → “respectively”; etc.

### Questions
Q1. What does $x_t(x_0, \psi)$ mean in Equation (11)? What is the difference between $x_t(x_0, \psi)$ and $x_t$?

Q2. Could the authors provide any theoretical justification and guarantee on why mollification is necessary and suitable for diffusion models to analyze infinite resolution data?

Q3. Would the authors clarify what $v_0$ means above Equation (14)?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
