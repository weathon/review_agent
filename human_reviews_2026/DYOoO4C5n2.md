# Disentangling Primitive Representation Structures for Image Generation

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
This paper explains a neural network for image generation from a new perspective, i.e., explaining representation structures for image generation. We propose a set of desirable properties to define the representation structure of a neural network for image generation, including feature completeness, spatial boundedness and consistency. These properties enable us to propose a method for disentangling primitive feature components from the intermediate-layer features, where each feature component generates a primitive regional pattern covering multiple image patches. In this way, the generation of the entire image can be explained as a superposition of these feature components. We prove that these feature components, which satisfy the feature completeness property and the linear additivity property (derived from the feature completeness, spatial boundedness, and consistency properties), can be computed as OR Harsanyi interaction. Experiments have verified the faithfulness of the disentangled primitive regional patterns.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a method for interpretation of generative models.
Namely, the authors study the outputs of a middle layer of a generative network, and try to decompose it into so-called primitive patterns.
Those patterns should satisfy a set of axioms, also proposed by the authors. 
The experimental section demonstrates that indeed these properties can be reasonably fulfilled, and the resulting image can be thought of as a composition of low-level patterns.

### Strengths
I am not an expert in interpretability of generative models, and for me the presented ideas sound quite novel and interesting.
The proposed set of desirable properties looks well-motivated, and the experimental results indeed demonstrate that some meaningful patters can be found in this way.
Probably, this approach can be interesting to the members of ICLR community that work in the field of explainable AI.

### Weaknesses
1. From my personal perspective, usefulness of such decomposition can be measured by how well it solves the problem of image editing. The prior approaches, e.g. [1] and [2], criticized by the authors, allowed such semantic editing. And, in my opinion, their editing results were more convincing than the ones presented in this manuscript (Figs. 8 and 10).

2. The whole procedure of learning feature components is not described very well in the main text. While they are defined through their action fields (Eq. 6), it seems from lines 900-901, that subsets of patches are selected randomly. It is unclear, how this randomness affects the learned feature components. Isn't it natural to assume that different selection of the patches will lead to different components? And is there any possibility to identify the best  action fields? Overall, I find this part of the paper very unclear.


References.

[1] A. Voynov and A.Babenko. Unsupervised discovery of interpretable directions in the GAN latent space. In ICML, 2020.

[2] Z. Chen et al. Exploring gradient-based multi-directional controls in gans. In ECCV, 2022.

### Questions
I ask the authors to address the weaknesses listed above:

1. Explain more clearly how action fields are selected, and how feature components are learned afterwards.
2. Provide meaningful examples of image editing by means of learned feature components.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper completely breaks down one whole feature in the middle hierarchy of deep neural network DNNs and defines a representative structure that can interpret image generation as a simple sum of multiple primitive feature components. 
The core of this structure is that each primitive feature component has its own action field and is only responsible for generating image patterns in that particular region.
The author first establishes a rigorous foundation by presenting three axioms that a 'Faithful' explanation of a model's internal workings must satisfy: Feature Completeness, Spatial Boundness, and Consistency.
Then, to extract independent components that satisfy these axioms, they introduce a mathematical formula called OR interaction.
The predictive features of components extracted from four different models (SiD Diffusion, NVAE, BigGAN, StyleGAN) using OR interaction successfully demonstrated all three axioms: they showed linear additionality matching the real minimum features, remained spatially bound as patterns did not leak outside their Action Field, and generated consistent patterns even when conditions changed.

### Strengths
Strong points
This paper proves that the proposed methodology is mathematically satisfactory and visually and numerically demonstrates it through systematic experimental verification. 
Sparsity finds that most of the extracted feature components are zero and only a small number of values are activated, and that only a small number of components contribute to image generation.
When model draws different faces from each other, they find that they reuse the eye and nose feature components’ regional patterns they used before.
It uses components to create precise control over only certain areas of the image in the order in which it is desired.

### Weaknesses
Weak points
This paper argued for the integrity of being able to completely decompose the entire image (e.g., 36 patches), but due to the computational cost problem of NP-complete, the actual experiment only takes place in a reduced environment of any 9 patches (1/4). It's okay if it's 9 patches, but I wonder if I can guarantee that it's okay if it's 36 patches (whole).
To verify transferability, the Top 20 transferable regional patterns list in Image A was compared with the Top 100 transferable regional patterns list in Image B. Fair comparison is Top 20 vs. Top 20 or Top 100 vs. Must be Top 100. The author does not provide an academic basis for any number l_a=20 or l_b=100. As a result, the high t value of 76-88% may have been intentionally inflated, and the results of this experiment are unreliable.

Main argument
(1)
The experimental verification presented in this paper is insufficient to support the validity of the proposed methodology. 
3 EXPERIMENT - Narrow Patch Selection
The data processing method used for internal representation structure analysis (the process of dividing and selecting image patches) introduces limitations that seriously hinder the generalizability of the methodology. The author divides the generated image x into a uniform 6 ×6 grid, defining a total of 36 patches. This partitioning is an essential preprocessing step to specifically understand what Action Fields a particular Regional Pattern is associated with; that is, which feature components affect which part of the image. However, the core experiment did not cover all 36 patches. The authors cited Li & Zhang (2023) and analyzed only n=9 patches selecting them randomly from the foreground object. 
While this approach is understandably due to the enormous computational cost of computing interactions, an analysis based solely on 9 foreground patches cannot fully explain the complex representation structure of the entire 36 patch grid. Therefore, the current experimental results can only be considered valid for a biased subset of the model's output (i.e., the foreground). This remains a significant limitation, as it is insufficient to verify the general representation structure claimed by the paper.

3.4 TRANSFERABILITY OF FEATURE COMPONENTS’ REGIONAL PATTERNS
This paper compares the list of feature components in the top $l_a=20$ from image $x^((a) )$ with the list in the top $l_b=100$ from image $x^{(b)}$ to verify the transferability ratio ($t$). Such an asymmetric comparison ($l_a≠l_b$) raises serious questions about the fairness of the experiment. 
A fair comparison of importance between the two sets would require a symmetric approach, such as Top 20 vs. Top 20 or Top 100 vs. Top 100. Furthermore, the authors provide no academic or experimental rationale (or justification) for why they chose these specific asymmetric values of $l_a=20$ and $l_b=100$. The absence of such justification does not rule out the possibility that the reported $t$ values (76-88%) were intentionally overestimated. The current setup means that a feature component that was, for example, the 15th most important in $x^((a) )$ would still be counted as transferred even if it were only the 95th most important (i.e., almost meaningless) in $x^((b) )$. This distorts the answer to the more critical question, "Is the core pattern still core in other images?", and seriously undermines the reliability of the results. Therefore, the authors must provide a clear rationale for this asymmetric setting or, preferably, re-evaluate the t value by conducting a sensitivity analysis using a fair, symmetric comparison (e.g., $l_a=l_b$).

(2) 
Line 358: The pixel-wise differences of the pattern $∆x_k$ was termed the real action field.
-> were
The subject of the sentence is "The pixel-wise differences" (plural). Therefore, the verb should be were (plural), not "was" (singular).

Line 727-732: For $i ∈ N$ and $S ⊆ N \ {i}$, the interaction effect of the pattern $S ∪ {i}$ is equal to the interaction effect of $S$ with the presence of $i$ minus the interaction effect of $S$ with the absence of $i$, i.e., $∀S ⊆ N \ {i}, I(S ∪ {i}) = I(S|iis always present) - I(S).I(S|iis always present)$ denotes the interaction effect when the variable i is always present as a constant context, i.e. $I(S|iis always present) = ∑ S⊆I(S) (-1)|S| · u(L ∪ {i})$. 
-> … $= I(S|i is always present) - I(S). I(S|i is always present)$ …
The period (.') immediately following $I(S)$ unnaturally breaks the sentence, causing the end of the equation to be jumbled with the start of the explanation.
It should be ... $- I(S). I(S|i is always present)$ denotes..., with a space inserted after the period to properly begin a new sentence.

### Questions
I have mentioned the concerns in the Weakness section.

### Soundness
2

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
3

### Summary
The paper presents three axioms; feature completeness, spatial boundedness and consistency, to explain the internal representation of an image. An intermediate feature f can be decomposed as a base f0​ plus a sum of feature components {Δfk}. Each component has an action field Ak​ (the image patches it affects) and contributes a regional pattern Δx​, thus, the image is explained as a superposition of these regional patterns.

### Strengths
- The work formalizes a representation structure via three axioms: feature completeness, spatial boundedness, and consistency. The feature decomposition turns an intermediate feature into a set of localized components with precise action fields.

- The paper provides interesting insights into the intrinsic representation. The analysis indicates high sparsity of the extracted feature representation, meaning that a region can be explained by a small number of feature components while the contribution of the others is negligible.

### Weaknesses
- Human interpretability gap: while the paper “disentangles” the representation into minimal regional features, this does not automatically yield human-interpretable factors. The extracted components remain complex. Minimal for reconstruction ≠ meaningful for humans. A stronger claim of interpretability would require concept aligned minimality, e.g. components that are minimal for foreground vs. background, hair vs. skin, sky vs. building, etc. Extending the framework to be minimal with respect to human-interpretable concepts would better support both interpretability and controllability of generation.
- Missing computational analysis: compute resources are listed, analysis isn’t. Beyond acknowledging cost and describing setup choices, the paper does not quantify end-to-end time per image/model. The high computational cost makes practical applicability questionable. 
- Overlap in action fields: action fields heavily overlap, and the feature components are not orthogonal,making attribution within shared regions ambiguous. The paper offers no weighting/attribution rule or orthogonality control to resolve this.
- Minor weakness; evaluation on diffusion models is limited: The diffusion evaluation is restricted to a single-step model on small resolution images (64x64).

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of interpretability in generative DNNs, arguing that most current methods are "superficial"  because they only analyze the meaning of isolated feature vectors rather than a complete "representation structure". To address this, the authors first propose a rigorous theoretical framework, defining a set of three axiomatic properties that any faithful explanation of a generative model's representation structure must satisfy.
The paper's core technical contribution is to prove that feature components satisfying these axioms can be formally computed. The method is validated on four generative models (SiD Diffusion, NVAE, BigGAN, and StyleGAN). The experiments are specifically designed to verify that the extracted components successfully adhere to the proposed axioms (Linear Additivity , Spatial Boundedness , and Consistency ).

### Strengths
The paper's most significant contribution is conceptual. It is not just another explanation method but a formal definition of what a faithful, complete explanation of a generative model's internal structure is. By establishing the three axioms (Completeness, Boundedness, Consistency) , the authors provide a rigorous mathematical foundation for a field that has largely relied on "superficial"  and incomplete analyses. The experimental design is excellent and serves as a model for theory-driven research. Instead of just showing qualitative results, the authors systematically test each of their theoretical claims.

### Weaknesses
Similar to many interpretability studies, the current analysis provides a useful “window” into understanding the generative model. However, the explanation relies on a relatively narrow theoretical perspective, which may limit its generalizability. For most researchers in generative modeling, there remains a conceptual gap: how can the generative process be understood at a higher, more intuitive level? Developing a simpler, working model that builds on the current interpretations could help a broader audience grasp the key mechanisms.

### Questions
1. Regarding the definition and selection of features, to what extent do the chosen priors constrain the completeness of the model interpretation? Could alternative feature choices reveal additional insights?
2. For researchers aiming to build better generative models, the practical value of the framework is also important. Beyond explaining a trained model, it would be valuable to discuss how these disentangled primitives could be leveraged to improve generative models—for example, enabling finer-grained controllable editing, diagnosing biases, or enhancing generalization.

### Soundness
3

### Presentation
3

### Contribution
4
