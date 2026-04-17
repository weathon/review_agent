# Test-time Uncertainty Estimation for Medical Image Registration via Transformation Equivariant

- Decision: Reject
- Scores: 8, 6, 6, 0

## Abstract
Accurate image registration is essential for downstream applications, yet current deep registration networks provide limited indications of whether and when their predictions are reliable. Existing uncertainty estimation strategies, such as Bayesian methods, ensembles, or MC dropout, require architectural changes or retraining, limiting their applicability to pretrained registration networks. Instead, we propose a test-time uncertainty estimation framework that is compatible with any pretrained networks. Our framework is grounded in the transformation equivariance property of registration, which states that the true mapping between two images should remain consistent under spatial perturbations of the input. By analyzing the variance of network predictions under such perturbations, we derive a theoretical decomposition of perturbation-based uncertainty in registration. This decomposition separates into two terms: (i) an intrinsic spread, reflecting epistemic noise, and (ii) a bias jitter, capturing how systematic error drifts under perturbations. Across four anatomical structures (brain, cardiac, abdominal, and lung) and multiple registration models (uniGradICON, SynthMorph, TransMorph), the uncertainty maps correlate consistently with registration errors and highlight regions requiring caution. Our framework turns any pretrained registration network into a risk-aware tool at test time, placing medical image registration one step closer to safe deployment in clinical and large-scale research settings.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a test-time, model-agnostic framework for estimating uncertainty in deep medical image registration networks. The core idea exploits the transformation equivariance property which means that for a true registration mapping, spatial perturbations of the input should produce consistent transformations. The authors show that deviations from this equivariance can be used to estimate uncertainty without retraining or modifying the model. The authors further provide a theoretical decomposition of the approach and evaluate it across multiple datasets and registration models. Experimental results demonstrate a strong correlation between the proposed uncertainty maps and the true registration errors.

### Strengths
1. Good novelty: The paper provides the first clear theoretical derivation of perturbation-based uncertainty estimation specifically tailored for image registration. The proposed formulation naturally aligns with the equivariance property inherent to the registration task.

2. Strong theoretical grounding: The authors decompose perturbation-induced variance into intrinsic spread and bias jitter. This formalism connects geometric consistency with probabilistic uncertainty and bridge the conceptual gap in registration theory. 

3. Solid experimental validation: The authors perform comprehensive experiments across multiple datasets (brain, cardiac, abdominal, lung) and different pretrained models (uniGradICON, SynthMorph) under various perturbations (translation, affine, deformation). Across all settings, the uncertainty maps show strong correlation with true registration errors, and the method successfully identifies field-of-view (FOV) mismatches and anatomical inconsistencies, demonstrating both robustness and clinical relevance.

### Weaknesses
1. Lack of comparison with existing uncertainty estimation methods: The evaluation primarily focuses on the proposed framework without direct quantitative comparison to established uncertainty estimation approaches (e.g., Bayesian inference, MC dropout, deep ensembles, or variational registration models). While the proposed method demonstrates strong correlation between uncertainty maps and true registration errors, it remains unclear whether it outperforms these existing techniques. Including such baselines would strengthen the empirical evidence for the claimed advantages in both accuracy and practicality.

### Questions
1. Could you please provide a comparison with other uncertainty estimation methods?

2. Could you add color bars in Figure 5, similar to those in Figure 4?

### Soundness
3

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
3

### Summary
This paper presents a test-time uncertainty estimation framework for deep medical image registration based on transformation equivariance. The method leverages spatial perturbations applied to input images and measures the resulting variability in the predicted deformation fields to quantify uncertainty. It introduces a theoretical decomposition of this variability into intrinsic spread and bias jitter, corresponding to different sources of uncertainty. The approach is model-agnostic and can be applied to any pretrained registration network without retraining. Experiments on multiple anatomical datasets and architectures show that the proposed uncertainty maps correlate well with registration errors and highlight anatomically inconsistent regions.

### Strengths
- The framework provides a clear and general way to estimate uncertainty at test time, requiring no modification or retraining of existing registration models.
- The authors evaluate the method across several datasets, anatomical regions, and architectures, showing strong empirical correlation between predicted uncertainty and registration errors.

### Weaknesses
- **Empirical nature of the uncertainty measure**: The proposed uncertainty reflects the variability of the warped output under spatial perturbations (i.e., uncertainty over $g(y)$), rather than directly capturing the model’s epistemic uncertainty. While the authors demonstrate a strong empirical correlation between this variance and registration error, the theoretical connection remains somewhat informal and could be further clarified.

- **Simplifying additive error assumption**: Equation (1) assumes that the model’s prediction error is additive with respect to the deformation field. This assumption may not hold in nonlinear or diffeomorphic registration, where transformation errors are typically compositional rather than additive, raising concerns about the theoretical validity of the variance decomposition.

- **Limited technical novelty**: The idea of using test-time perturbations for uncertainty estimation is not new and has been applied in other domains such as classification and segmentation. This work primarily adapts the concept to registration rather than introducing a fundamentally novel methodology.

### Questions
Please see the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposed to estimate image registration uncertainty, given equivariance under deformations.
In my understanding, registration solutions are perturbed, then uncertainty is estimated in terms of intrinsic spread and bias jitter terms, which correspond to mean and variance. Experiments demonstrate the method on images of brain, heart, thorax, abdominal regions, MRI and CT modalities.

### Strengths
The method can be applied to the output of registration algorithm. 

Uncertainty maps remain relatively consistent across linear perturbations, less so for deformable perturbations. 

Uncertainty appears highest in image regions containing occlusion or structure present in only one image, e.g. brain tumors or variable field of view, this is encouraging and aligns with intuition.

### Weaknesses
Regarding identifying high uncertainty in occlusions, pathology, however it’s not necessarily difficult for a naive image differencing or intensity variance estimation to pick up such patterns.

Regarding graphs in Figure 6, 8 b), I do not have a strong intuition as to how meaningful or useful these are. 

The literature seems to be lacking, the Gaussian Process model is has been used to estimation medical image registration uncertainty in a very similar way.
Wassermann, Demian, et al. "Probabilistic diffeomorphic registration: representing uncertainty." International Workshop on Biomedical Image Registration. Cham: Springer International Publishing, 2014.

### Questions
See previous section

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper deals with the uncertainty of image registration methods that are typically ill posed and have a multitude of possible solutions. Instead of standard Bayesian networks that require architectural changes, the method aims to use equivariances of registration to use a test-time augmentation of the inputs to estimate the variance of the produced deformation fields. The paper estimates the variance of the model under different perturbations of the input which is decomposed into two terms: an intrinsic "spread", and a "bias jitter".

### Strengths
1. Evaluation is performed with two state-of-the-art methods on multiple organs, spatial dimensions, and imaging modalities. The experiment  setup is well motivated for the task that is being tackled , i.e. uncertainty estimation. Applications like registration with limited FOV and tumor registration are good motivating examples of showing the impact of a good uncertainty registration method.

2. The method proposes estimation of a "perturbation-based uncertainty" given a model and . This is a research area that is critical in calibrating predictions by the amount of uncertainty due to the illposedness of the data. Registration is typically highly overparameterized, even with regularization, with potentially infinite compatible displacement fields that satisfy the image similarity constraint. This implies a high amount of aleatoric uncertainty due to the nature of the problem, which can form an excellent basis for imbuing models with the capacity to estimate this uncertainty. This uncertainty can also allow a user to perform guided registration in areas where the aleatoric uncertainty is high (i.e. large number of solutions are possible).

### Weaknesses
1. The interpretation of performing test-time augmentation to perform some kind of  uncertainty estimation deviates from the interpretation in green box in Line 192. The latter interpretation implies that a "true deformation" is equivariant to linear or nonlinear diffeomorphic perturbations but the network may not be - essentially measuring the equivariance error of the model. For a (hypothetical) model that does respect the equivariance of the task and somehow produces transforms consistent with the applied transform $\tau$, does it imply that the model has no estimates of uncertainty? I would argue that it is not true - as the ill-posedness of the problem does not disappear if the model becomes aware of the equivariance of the problem (and by extension the aleatoric uncertainty remains unchanged) . So the formulation is **not** measuring aleatoric uncertainty since the "equivariance error" can be reduced in the model by training it to respect the constraint in Equation (7) but the aleatoric uncertainty is intrinsic to the datum and cannot be reduced. On the other hand, epistemic uncertainty estimates the variance of the output (more below) under a posterior distribution over the model parameters. The proposed idea works with a pretrained network and does not impose any distribution over the model weights - so the proposed method does not compute epistemic uncertainty either. To me, this is a major fundamental issue with the paper - a simple argument showing that the proposed method captures neither epistemic or aleatoric uncertainty - and I would urge the authors to discuss and clarify this in the rebuttal.
2. Math in the paper is not rigorous at all (see concerns below). There is no such thing as "perturbation-based uncertainty". If new concepts are introduced, they must be compared and contrasted with more established concepts like uncertainty like aleatoric, epistemic, and predictive uncertainties. A highly theoretical area of work like uncertainty estimation requires very rigorous treatment of the proposed method. 
3. A lot of terms are used very colloquially - e.g. in Line 39 "they predict a deformation field from an image pair, but not an estimate of how trustworthy that field is" - what is the scope of a "trustworthy field"? Is it a prediction with low uncertainty estimates? 
4. Line 47 - "While effective, these approaches are computationally expensive and do not scale well." - any method that aims to estimate epistemic uncertainty will be computationally expensive for two reasons - (1) the epistemic uncertainty is a measure of the variance of the posterior $Var_{P(W|D)}[E_{P(Y|X;W)}Y]$ where $W$ are the weights of the network, and $D$ is the dataset, $X, Y$ are the inputs and outputs. Since the expectation or variance under the posterior typically does not have a closed form solution, one has to compute the empirical MC estimates [1, 2]. There is no other way to estimate model uncertainty unless a closed form solution can be derived. The other way to compute epistemic uncertainty cheaply is to choose an "restricted prior" $P(W)$ (delta distribution on the backbone layers from a pretrained network and a regular prior on the final layer) - but that is not interesting enough (similar to linear probing if the backbone feature extractor is robust enough) 
5. Line 88: "We present the first theoretical account of perturbation-based uncertainty in image registration, showing how perturbation-induced variance decomposes into intrinsic spread and bias jitter" - this is an incorrect statement. Test-time augmentation has been around in the literature for a long time - the paper happens to perform test-time augmentation with invertible linear (rigid, affine) or nonlinear (spline) deformations. This claim needs to be rewritten to highlight the connections of the perturbation-based uncertainty to test-time augmentation and why this particular test-time augmentation makes sense for the considered task. 
6. Line 116: "However, few works have explored uncertainty estimation in medical image registration." - no citations! Discussion with these other methods and how they are different from the proposed work are key pieces of related work missing from the paper.
7. Line 131 "Test time augmentation fundamentally measures something different compared to the bayesian deep learning frameworks. It is important to distinguish these  concepts instead of clubbing them together" - this is indeed the case because test-time approaches measure something that is indicative of aleatoric uncertainty rather than epistemic. If the input contains intrinsic uncertainty (e.g. registration task can have multiple solutions) regardless of the model, it falls under the purview of aleatoric uncertainty. These concepts must be discussed more holistically in the paper.
8. shouldn't the "error model" be compositive instead of additive - $\epsilon(y)$ captures the possible space of diffeomorphic perturbations that do not decrease the image similarity cost function (e.g. local diffeomorphic flows within subcortical regions). More formally, consider the support set of $\epsilon(y)$, i.e. $S_e(I_f, I_m) = \{\epsilon(y): Sim(I^A\circ (\phi \circ \epsilon), I^B) \ge Sim(I^A\circ (\phi), I^B)\}$ that tells us all the admissible perturbations that can be applied without decreasing the image similarity - these lead to amiguity of the solution. For example, if $S_e = \phi$ (empty set), then only a single minimizer exists, and there should be no uncertainty about which deformation maximizes image similarity. Using a well-defined measure of the support set, i.e. $\int_{S_e} d\mu(\epsilon)$ is an estimate of the amount of perturbations that can be applied. This is not a variance by itself, but is a meaningful proxy that quantifies the "amount of ill-posedness". An additive loss will allow inclusion of non-diffeomorphic updates that satisfy the similarity constraint, but a compositive "error term" will possibly make analysis difficult.
9. Lines 238-244 : There is no epistemic uncertainty in the blue arrow term. Any amount of mathematical connection between this term and how it relates to a standard definition of epistemic uncertainty. A similar statement for the purple arrow. This ties back to my feedback on the rigor of math (Point 2) and the argument that the quantity captures neither epistemic nor aleatoric invariance (Point 1). 
10. "Bias jitter" is a slightly misleading term (although I understand where the naming comes from - a "jitter/variance" of the bias term) but it can be potentially misleading - The second term in Equation (9) is not a bias either, it is also a covariance term. I'd recommend choosing a name that makes more sense. Jitter is a colloquial term and can mean different things depending on context [3]. 
11. Figure 3: In contrary to what the paper says, there are actually no meaningful trends between the PCC / SC and the performance. For each dataset, the mean error is also not high or low for any dataset / perturbation combination. 


Minor nits:
1. Line 36: "Modern learning based approaches can now deliver real-time registration that rivals or surpasses classical methods." - add citations to support this statement. Also see CLAIRE, FireANTs, NiftyReg and ConvexAdam for counter-examples to this statement.  
2. Line 50: "but they typically underestimate uncertainty, fail to capture the spatial structure of deformation fields, and remain tightly coupled to specific architectures" - this is not a limitation of the variational formulation but the chosen priors on the model. The phrase about "failure to capture spatial structure of deformation fields" needs to be removed or supported with related literature. 
3. Literature on classical image registration and their strengths and weaknesses is totally missing from the paper. Since the model only applies test-time augmentations regardless of models, it should be applicable to iterative methods as well. Do these methods show different behavior to the test-time augmentation as well? 
4. Line 117 "Moreover, the diversity in architectures, training schemes, similarity measures, and tasks makes it nontrivial to combine uncertainty estimation methods with existing pretrained registration networks." - why? Aleatoric uncertainty is typically modeled using variance estimates of the output, and epistemic uncertainty imposes some prior on the model weights. Fundamentally, there is no special case in architectures or loss functions that would be incompatible with existing formulations. Consider removing this statement. 
5. Line 143: "In the deep learning era, however, uncertainty estimation remains largely underexplored." - there is a wealth of work applying uncertainty estimation with deep networks (a lot is cited in the paper). Variance of a posterior is fundamentally a Bayesian concept and makes no assumptions about the model (the model can even be non-parameteric - e.g. Gaussian Processes). Related work cited in the paper are different choices of specifying appropriate priors on the model or the output space. 
6. In Equation (1), $\epsilon(y)$ is typically the noise added in the problem formulation itself (to define maximum likelihood-like objectives), not in the model itself. Line 172 can be corrected to reflect this interpretation. 
7. Please number all equations (Line 222, 226). 
8. Line 204, "is a diffeomorphic," -> "is a diffeomorphic transform" or "is diffeomorphic,"
9. Line 322 is unclear and can benefit from further explanation
10. Fonts in all boxplots need to increase - figures are hard to read.
11. In Figure  4(a), the uncertainty maps are very qualitatively very different from each other. Which one should a practitioner interpret in their downstream task? 
12. Figure 4(a), what is the uncertainty map conveying? From the map it looks like the boundaries are hardest to align - but in practice it is the easiest to align due to the high contrast of the brain with background.
13. Line 978: "To be noted, we use" -- "We note that a composition of two elastic deformations in …"
14. In Figure 10, the uncertainty map does not show the insights mentioned in the paper (i.e. if the tumor is harder to register / has more registration error because of a lack of correspondence). The paper mentions in line 459 - "Overlaying the uncertainty map on the target and warped T2 in Fig. 5a confirms that high-uncertainty regions align well with the location of the tumor" but the figure does not highlight the tumor at all. In fact, for a lot of image pairs the regions of high uncertainty do not align with the tumor at all.
15. How does Figure 10 strengthen the narrative of the paper in terms of the usefulness of the "uncertainty maps" in assessing the difficulty / potential lack of quality in registration? How does a user visualize the maps shown in Figure 10 and make a clinical decision for example? 
16. Line 354: this is an inaccurate statement. Synthmorph is trained to do joint affine and deformable registration — use the “joint” parameter. The only correct portion of this statement is the fact that this reflects model’s inductive bias (i.e. lack of equivariance of the output w.r.t. moving image). 
17. Line 356: "When the task matches its capacity (Fig. 3b (c)), the correlations are consistent with uniGradICON." - this meaning of this statement is unclear; consider clarifying.
18. There are several missing implementation details - how is the red line computed? Is the PCC/SC computed between per-pixel uncertainty estimates and pixelwise MSE? 
19. The model performs multiple forward passes through the network to obtain different transforms. How is the computational efficiency any different from MC methods for epistemic uncertainty that also perform multiple forward passes? 
20. What are the methods used in the case studies? 

Overall, the paper makes statements that are either not rigorous enough, are not supported with literature, or are too bold. In my opinion, the paper can provide very valuable insights but will require significant work on improving the narration, calibrating claims, and being more rigorous with the contributions, before the quality is satisfactory for publication at a venue like ICLR. 

[1] https://proceedings.mlr.press/v48/gal16.pdf 

[2] Lakshminarayanan, B., Pritzel, A. and Blundell, C., 2017. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30.

[3] https://en.wikipedia.org/wiki/Jitter

### Questions
Few questions are mentioned in the weaknesses section.

### Soundness
1

### Presentation
2

### Contribution
1
