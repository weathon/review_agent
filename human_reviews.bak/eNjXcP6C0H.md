# Generating Physical Dynamics under Priors

- Decision: Accept (Poster)
- Scores: 6, 5, 8, 6

## Abstract
Generating physically feasible dynamics in a data-driven context is challenging, especially when adhering to physical priors expressed in specific equations or formulas. Existing methodologies often overlook the integration of ''physical priors'', resulting in violation of basic physical laws and suboptimal performance. In this paper, we introduce a novel framework that seamlessly incorporates physical priors into diffusion-based generative models to address this limitation. Our approach leverages two categories of priors: 1) distributional priors, such as roto-translational invariance, and 2) physical feasibility priors, including energy and momentum conservation laws and PDE constraints. By embedding these priors into the generative process, our method can efficiently generate physically realistic dynamics, encompassing trajectories and flows. Empirical evaluations demonstrate that our method produces high-quality dynamics across a diverse array of physical phenomena with remarkable robustness, underscoring its potential to advance data-driven studies in AI4Physics. Our contributions signify a substantial advancement in the field of generative modeling, offering a robust solution to generate accurate and physically consistent dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes two ways to incorporate physical priors into generative models: (1) they inject distributional priors by choosing the proper equivariant models and (2) they incorporate physical feasibility priors by decomposing nonlinear constraints into elementary cases. The key is to identify elementary cases where Jensen's gap can be omitted and decompose complicated constraints into elementary cases. In general, this paper is a good contribution to the AI4Science community, where many scientific priors are available and should be incorporated into learning systems.

### Strengths
* The motivation is clear - this paper is a clear contribution to the AI4Science (esp AI4Physics) community where data-driven models should be combined with scientific inductive biases. Although physics-informed learning has been mainstream in scientific machine learning, physics-informed learning in the context of generative modeling is relatively new. 
* This paper is both theoretically (supported by theorems) and practically sound (supported by experiments). The idea of decomposing a complex constraint into elementary cases is a good strategy.

### Weaknesses
* Empirical improvement is marginal (Table 1, 2, 3).
* This framework is useful when (1) we don't understand the system fully but (2) we know some partial information (existence of energy etc). However, all the examples in the paper are synthetic, so we know both the underlying equation and energy but "pretend" that we don't know about the equation. I understand this is only a proof of concept, but a more realistic example would greatly strengthen the paper.

### Questions
* Why does performance improve only incrementally after adding priors?
* In Equation (4) and line 205, are there typos? \nabla x -> \nabla_x?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work explores ways of incorporating physics-informed priors in diffusion models for dynamical systems. In particular, they investigate **distributional priors**, aiming to incorporate appropriate symmetries (such as roto-translational invariance, permutation invariance, etc.), and **physical feasibility priors** as a way to encourage adherence to physical laws (e.g. momentum and energy conservation, PDE constraints, etc.). The manner in which the latter are imposed depends on the nature of the constraints, and the authors analyse several such cases separately (e.g. linear, multilinear, nonlinear, etc.).

The work provides empirical evidence on i) four PDE datasets, where PDE constraints are used, and ii) two particle dynamics datasets, where the diffusion model leverages the relevant physical laws, and training involves data augmentation to incorporate appropriate symmetries.

### Strengths
1. **Relevant topic.** There is increased interest in the dynamical system modelling community to incorporate relevant priors into models. Not only would this help with more efficient training, but perhaps more importantly, it should lead to better stability and capabilities to generalise out-of-distribution (OOD - when the test data comes from a different distribution than the training data).
2. **Wide range of datasets.** The experiments are performed on a wide range of datasets - four PDE ones, and two particle dynamics datasets. This helps at showcasing how the techniques would be employed in various scenarios, studying both linear and multilinear PDE constraints in the PDE datasets, as well as (reducible and general) nonlinear ones in the particle dynamics datasets.

### Weaknesses
1. **Contribution in distributional prior incorporation.** I believe there is some related work that is not being mentioned in section 3.1, as there are now plenty of works studying the incorporation of relevant priors in diffusion models. For example, Mathieu et al. [1] incorporate a series of geometric priors in infinite-dimensional modelling, also focusing on conditions for the diffusion process to be G-invariant assuming that the score network is G-equivariant.
2. **Lack of empirical evidence from the first methods subsection.** I do not think that the theory behind the first Methods subsection (Incorporating distributional priors) is sufficiently evidenced empirically.
- In the PDE datasets, the only result used empirically is that the data matching objective should be used, but this is rather weakly supported theoretically with the smoothness argument (although evidenced empirically in Table 4).
- In the particle dynamics datasets, I understand that finding the right architecture that incorporates all the relevant inductive biases can be hard, but if in the end data augmentation was used, how is the theory supported empirically in this case? I think you should include a discussion on this, as well as more details on how data augmentation was performed. 
- Also related to the particle dynamics dataset, it seems that for the five-spring dataset you are using the EGNN architecture, (but you do not cite it - I am assuming you are referring to Satorras et al. [2]). I think this should be stressed in the main text with a brief discussion on the choice of architecture (referring to the inductive biases it incorporates). It's an important component given the chosen topic and I do not understand why it wasn't more clearly emphasised. And why are you using it in the five-spring dataset, but not in the three-body one?
- Finally, the section on ECM from Methods is not referenced later on in the paper. From my understanding, its purpose is to justify the data augmentation approach (in which case this connects to the previous point). I think a clearer connection between methods and how they are applied should be made.
3. **Lack of error bars.** The results do not contain any sort of error bars, which actually makes it very difficult to assess the effectiveness of the proposed methods. Especially in the PDE datasets results, the differences between the mean RMSEs are fairly small, and, depending on the errors, the distinction between the method w/o and w/ prior might actually turn out not to be significant. Could you please include them?
4. **Inadequate sample quality.** For the Darcy flow samples, the quality is very poor. I understand that no super-resolution or denoising procedures have been employed, but it seems like there is too much noise left in the samples. Do you take a final denoising step after the end of the diffusion process (by applying Tweedie’s formula)? Do you normalise the data? I can see that the data values for Darcy flow (with poor quality) are much lower (0.025-0.150) vs. Burgers (0.0-1.0), but the last sigma value is probably the same, so the proportional influence of the noise is higher in the Darcy flow dataset, potentially leading to poorer quality. 
5. **Effectiveness of general nonlinear constraints.** The results on general nonlinear constraints are not that promising. In the three-body dataset, the results for noise matching + conservation of energy (general nonlinear) are very much comparable (especially for certain hyperparameters) to plain noise matching (although this is hard to assess because of the lack of error bars) - see Table 8. In the five-spring dataset (Table 9), the improvement is once again marginal for general nonlinear. Therefore, although I liked the idea of utilising the same hidden states as the model, I would argue that it did not prove to be effective in these experiments. If the authors agree, I think this limitation should be highlighted properly in the paper.
6. **No discussion on extra computational cost.** The paper does not discuss anything about the extra computational costs of the proposed method.

**Minor**

7. **Lack of thorough details/references on architecture.** In Table 7, I believe the Karras Unet should be referenced (from the EDM [3] paper). The paper also mentions that in the five-spring dataset the EGNN architecture is used but no reference is given (Satorras et al. [2]) and maybe the acronym should be defined for completeness.
8. **Normalised metrics.** For easier comparison between datasets, it might be better to use normalised metrics rather than absolute ones.
9. **Small legend in graphs and typos.** The font of the legends in the plots is too small (see Figure 2, Figure 8, Figure 9, etc.). There are also several typos throughout the paper (L107 extra “of”, L114 should be $\alpha_T$ and $\sigma_T$?, L271 related to the absolute, etc.). 
10. **Inconsistent notation for score.** Sometimes you specify the score as $\nabla_{\mathbf{x}} \log q_t(\mathbf{x})$, sometimes as $\nabla_{\mathbf{x}} \log q_t(\mathbf{x}_t)$.

Overall, I think that the paper contains some nice ideas, but fails to clearly connect them to the empirical evidence, the experiments are poorly presented (with lack of error bars) and the effectiveness of some cases seems overstated.

[1] Mathieu, E., Dutordoir, V., Hutchinson, M.J., Bortoli, V.D., Teh, Y.W., & Turner, R.E. (2023). Geometric Neural Diffusion Processes. ArXiv, abs/2307.05431.

[2] Satorras, V.G., Hoogeboom, E., & Welling, M. (2021). E(n) Equivariant Graph Neural Networks. ArXiv, abs/2102.09844.

[3] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. ArXiv, abs/2206.00364.

### Questions
1. **Related to W1** - What is the element of novelty in your exposition of incorporating distributional priors? If there is not any, this is not a problem, but then the relevant related works should be cited and the paper should indicate this clearly.
2. **Related to W3** - Can you also provide error bars on all results?
3. **Related to W6** - Could you comment on the extra cost of introducing the physical constraints using finite difference methods?
4. **Alternative objective** - For the distributional prior, have you ever tried a different objective that is a mix between the noise and data objective (for example, v-prediction [Salimans et al. [4]])? 
5. For the PDE datasets, why don’t you provide the RMSE of the PDE constraints on all datasets?
6. From my understanding, you only impose either the momentum conservation constrain or the energy conservation constraint. What if in the particle dynamics datasets you imposed both a conservation of momentum and a conservation of energy penalty? Would this lead to instabilities?
7. When employing finite difference methods to approximate the differential equations, how do you make sure that the spatio-temporal discretisation of the datasets is fine enough to yield significant results? Is the discretisation the reason why the RMSE of the PDE constraints in Table 2 is not closer to 0?
8. Could you please include in the captions of Table 4 and 5 the quantity that you are reporting (I am assuming Traj error)?
9. Isn’t the result in Table 4 for Shallow Water with distributional prior the same as the result in Table 2 Shallow water? If so, shouldn’t it be 8.151 instead of 8.150? And for the three-body dataset, shouldn’t it be 2.5613 as in Table 3 instead of 2.6084, or is there a difference between those settings?

[4] Salimans, T., & Ho, J. (2022). Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a novel framework to generate physically feasible dynamics by incorporating physical priors into diffusion-based generative models. Unlike traditional generative approaches, which often disregard physical constraints, this model integrates two types of priors: distributional priors (such as invariance to roto-translation) and physical feasibility priors (e.g., conservation laws for energy and momentum, and partial differential equation constraints). By embedding these priors into the generative process, the method produces realistic dynamics across a variety of physical systems.

### Strengths
- The work is innovative in its focus on embedding physical feasibility directly into diffusion-based generative models, specifically through the use of both distributional and physical priors. By designing a process that enforces these constraints, the model is capable of generating realistic dynamics, distinguishing it from more traditional generative approaches that may ignore or only partially enforce physical laws.

- Moreover, the paper is well-written and is thorough with both the theoretical part and the empirical results for the proposed model. 

- The paper addresses a challenge in generating accurate physical dynamics with AI methods. It enables more reliable and scientifically sound simulations in areas like environmental, material science, and possibly the fluid simulations.

### Weaknesses
- The experiments focus on synthetic datasets or physics-inspired datasets, and while this is valuable, additional testing on real-world, noisy datasets would definitely help.

- If possible, comparing the model's performance with more baselines will make the empirical results more convincing. Yet it might be difficult to find the methods from the exactly same field, but some methods from, e.g. time-series forecasting, can be taken into consideration.

### Questions
- The model includes various hyperparameters, especially in the weighting of physical feasibility losses. How sensitive is the model to these settings? Could the authors provide guidance on tuning these parameters, or propose default values based on their findings?

- What does "intrinsic structures of pairwise distance" refer to?

- Does the resolution matter for "reducible nonlinear cases"?

- As I noticed, all of the datasets in this work seem to be with multi-dimensional features. Yet in biology, we usually just have single-dimensional features possibly collected at a time-step. Will this affect the performance of the method?

- I think the five springs dataset is probably more like an ODE dataset instead of a PDE. The transition functions are second-order ODEs.

- Following last question, how about reporting the errors of NRI (Kipf et al., 2018) on the experiments in Section 4.2 as well? 

- I have a tiny suggestion on the setup of the paper. If it is hard to fit a whole section of related works in the main body, it would be a good practice to merge a part of the related works to the section of preliminaries. Readers need the clear message in the related works to recall the research gap.

- Please check the score function in line 146, it does not consistent with the one in line 115. Please be careful about the suffices.

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
2

### Summary
This work proposes a framework that incorporates physical priors, including distributional priors and physical feasibility priors, to generate physically realistic dynamics with diffusion-based models.

### Strengths
1. The manuscript is well-written and easy to follow.
2. Thorough derivations are provided in the main content and appendix.
3. Sufficient experiments and ablation studies are conducted to validate the effectiveness of the proposed method.

### Weaknesses
Please see the Question part.

### Questions
- The three contributions listed in Section 1 appear similar and convey the same meaning. Accurately summarizing the manuscript's contributions will be more helpful for the readers.

- Why distribution priors (such as translational rotational invariance) are important for generating physical dynamics? As stated in [1], the invariant sampling and invariant loss functions by restricting architecture designs often sacrifice empirical performances. Are simple data augmentations that approximate probability equivariance also available in physical dynamics generation? Experiments with data augmentations that approximate probability equivariance might help analyze the effectiveness of equivariant models.

- How are the physical priors selected in this work? It appears that the roto-translational invariance and the priors based on energy and momentum conservation laws are chosen arbitrarily. There seems to be no systematic analysis that explores all possible categories of physical priors or how to incorporate each type of prior into diffusion models. I believe there are many more physical priors that generative models should be constrained by. Categorizing them and discussing each category systematically would provide more insight.

- In Line 498-508, the paragraph title is "Data matching vs noise matching". However, the ablation study is conducted for incorporating a distributional prior or not, which is expected to investigate the training objective instead.

---
[1] SwinGNN: Rethinking Permutation Invariance in Diffusion Models for Graph Generation. TMLR 2024.

### Soundness
3

### Presentation
3

### Contribution
2
