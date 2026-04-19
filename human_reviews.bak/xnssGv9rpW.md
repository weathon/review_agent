# SymmCD: Symmetry-Preserving Crystal Generation with Diffusion Models

- Decision: Accept (Poster)
- Scores: 8, 8, 5

## Abstract
Generating novel crystalline materials has potential to lead to advancements in fields such as electronics, energy storage, and catalysis. The defining characteristic of crystals is their symmetry, which plays a central role in determining their physical properties. However, existing crystal generation methods either fail to generate materials that display the symmetries of real-world crystals, or simply replicate the symmetry information from examples in a database.  To address this limitation, we propose SymmCD, a novel diffusion-based generative model that explicitly incorporates crystallographic symmetry into the generative process. We decompose crystals into two components and learn their joint distribution through diffusion: 1) the asymmetric unit, the smallest subset of the crystal  which can generate the whole crystal through symmetry transformations, and; 2) the symmetry transformations needed to be applied to each atom in the asymmetric unit. We also use a novel and interpretable representation for these transformations, enabling generalization across different crystallographic symmetry groups. We showcase the competitive performance of SymmCD on a subset of the Materials Project, obtaining diverse and valid crystals with realistic symmetries and predicted properties.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a novel diffusion-based generative model, SymmCD, for symmetry-preserving crystal generation. The proposed approach explicitly incorporates crystallographic symmetry into the generative process, using a unique representation that decomposes crystals into asymmetric units and symmetry transformations. This design enhances both computational efficiency and the diversity of generated crystal structures, addressing some limitations of existing models in terms of symmetry and structural validity.Overall, the paper presents a strong contribution to the field of crystal generation. By explicitly incorporating symmetry into a generative diffusion framework, SymmCD addresses critical limitations of prior methods and provides a promising tool for materials discovery.

### Strengths
1. Innovation in Representation: The paper introduces a physically motivated representation based on crystallographic symmetry, using a binary matrix to encode symmetry, which addresses data fragmentation and enables generalization across symmetry groups.
2. Computational Efficiency: By focusing on asymmetric units rather than full crystal structures, the model demonstrates significant improvements in memory usage and training speed, an aspect well-supported by experimental evidence.
3. Diversity and Validity of Generated Structures: SymmCD shows impressive results in generating diverse, valid, and symmetry-conforming crystal structures across multiple symmetry groups, even those that are less common in training data.

### Weaknesses
1.Comprehensive Evaluation of Generated Crystal Properties: While the model’s ability to generate symmetric and diverse crystals is demonstrated, additional quantitative evaluations of properties such as thermodynamic and mechanical stability would further solidify the model’s applicability to real-world scenarios. Metrics that reflect physical applicability, such as structural stability under various conditions, could significantly strengthen the evaluation section.
2.Efficiency on Larger Datasets: SymmCD’s efficient crystal representation is highlighted as a key advantage. However, a more comprehensive analysis of its computational efficiency on larger datasets, or under different hardware setups, could provide a more complete understanding of its scalability and practical utility in materials science applications.
3.Clarification of the Binary Symmetry Encoding: The binary matrix representation for symmetry is an intriguing solution to data fragmentation, yet further explanation on why this approach outperforms traditional encodings in practical settings would be beneficial. Additional details in the architecture and experimental sections could clarify how the representation is effectively utilized in training.
4.It may be helpful to provide a clearer explanation of the training algorithm, particularly in how the diffusion and denoising processes maintain symmetry.

### Questions
Since I am not a researcher in this field, I don’t know much about the specific background, so I am not very clear about the process shown in Figure 3. Figure 3 and the training pipeline section could benefit from additional annotations to improve readability for those unfamiliar with diffusion models in this context.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors present the generative model SymmCD, which allows for the generation of datasets of crystalline structures of non-molecular crystals while explicitly considering symmetry. The results obtained exhibit both high symmetry diversity and a significant percentage of thermodynamically stable structures, making SymmCD a solid choice for crystal structure prediction systems or virtual screening of crystalline materials.

### Strengths
- The developed method for vectorizing crystalline structures, which explicitly accounts for both the spatial symmetry of the crystal and the point symmetry of the orbits, is to my knowledge the first of its kind, therefore unique, and holds a great promise for application in crystal structure prediction (CSP) for both inorganic and organic crystals.
- The article is well-structured and clearly conveys information, allowing individuals unfamiliar with this field to understand the crystallographic features of the problem with some investment of time.
- I believe this work could be highlighted at the conference as a fine example of how rational design of vector representation can influence the overall effectiveness of the developed deep learning model.

### Weaknesses
The major weaknesses of the paper lie in the discussion of the obtained empirical results. Addressing these will significantly enhance the presentation of the work accomplished:

1. Please indicate in the introduction that the initial focus is on non-molecular/inorganic crystals.

2. Please mention in the conclusion remarks that your method of structural representation seems to be well-suited for molecular crystals as well. For the latter, the presence of intrinsic point symmetry and its interaction with the point symmetry of orbitals is one of the key factors determining the crystal structure.

3. In your conclusions, when you state "go beyond single crystals, and consider generating multi-component crystals and alloys," please clarify what you mean. "Single crystal" is a broad term contrasting with polycrystalline materials and does not directly relate to crystalline structure. A multi-component crystal refers to a crystal composed of multiple chemical substances; for instance, this includes pharmaceutical co-crystals. Clearly, your approach should be applicable to these systems.

4. Please consider rewriting conclusions to emphasize advantages (applicability to molecular crystals, including co-crystals) rather than deficiencies (inapplicability to non-crystalline systems), but also, to provide a deeper discussion of the limitations of SymmCD along with practical implications for the actual industrial problems.

### Questions
1. In Table 2, why is CDVAE bold instead of SymmCD (10 SGs) for Validity Comp.? I understand an argument could be made that comparing the 10 SGs version to the other methods might not be entirely appropriate, but then I would consider dropping this version from tables 1 and 2, and only discuss it in the context of table 3, where the S.U.N. shines. Please clarify the logic behind inclusion of SymmCD (10 SGs) for Table 1 and Table 2.
2. The evaluation presented in Table 3 involves random subsampling of 10% of the generated crystals, followed by two predictive models to evaluate stability and S.U.N. properties. At the same time, the SymmCD shows only a marginal improvement compared to DiffCSP and DiffCSP++. Are these results statistically significant? Please provide details on the robustness of this evaluation.
3. Which model appears in the Table 4 as Conventional Unit Cell? Please provide a citation and clarify how this model was included in other comparisons as well.
4. Is it possible to provide a link to the anonymized repository reproducing experiments?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The submission, "Symmetry-Preserving Crystal Generation with Diffusion Models," proposes a method for generating single-crystal structures with precise symmetric properties. The authors use asymmetric units and site symmetry representation, followed by a diffusion model for generation. This method explicitly addresses the generation of crystals with respect to their symmetry group.
The method performs on par with existing approaches but has a lower computational footprint.

### Strengths
•	The manuscript's structure and clarity are excellent overall.

•	The manuscript includes a well-written and comprehensive introduction, with a clear and well-developed motivation for the crystal generation problem as an application of diffusion models.

•	The method is well-formalized and understandable even to non-experts in crystal generation.

•	Experimental tasks and evaluation: The authors assess their method and the baselines on relevant additional tasks, such as S.U.N. structure prediction and other proxy metrics, which highlight the proposed method's strengths.

### Weaknesses
•	Introduction: From my perspective, the problem of generating symmetric crystals is closely related to other structure generation tasks in general representation learning. For instance, in biological applications, such as neuron structure generation or vascular structure generation, it would be beneficial if the authors discussed the relation to other domains in structure generation and the types of methods that have been developed. For example, I see certain similarities to diffusion methods in molecule generation [https://ieeexplore.ieee.org/abstract/document/10419041] or graph generation [https://arxiv.org/abs/2209.14734], which are partly mentioned in the methods since they are used; however, a discussion of how these applications relate to the context of representation learning would be valuable.

•	Reproducibility: I did not find a link to an anonymous repository or source code in OpenReview, hindering the evaluation of reproducibility for this submission.


•	Experimentation: There are only minor performance gains (if any) compared to the state of the art. What are the practical uses of crystal symmetry generation in academia or industry? Is the computational gain truly relevant, considering the regular applications and scenarios in which crystal symmetry generation methods are used?

•	Experimentation: "We withhold 20% of the dataset as a validation set, and 20% as a test set" (Line 377). The experimental setup suggests that the authors do not use a form of cross-validation or cross-testing. Is there a specific reason for this choice? Given that the authors describe their computational efficiency as a strength, extensive cross-validation across experiments would seem reasonable.

•	Experimentation: Hyperparameter Selection (Section E.2). The authors briefly describe their final hyperparameters: “These hyperparameters were chosen using a sweep” (Line 919). Without code availability and the validation issues mentioned earlier, this appears to be a limited experimental description. What was the hyperparameter search space/budget? How were the hyperparameters for the four baselines tuned exactly? The results show very small differences in performance, so a fair description of hyperparameter search is crucial for reproducibility.


I am not an expert in crystal generation and potentially some of my questions are atypical in the field, I am curious to hear the authors and other reviewer comments and willing to change my rating accordingly.





## Post rebuttal and discussion period comment: 

I want to thank the authors for replying and interacting in the discussion period. While the authors reply to all comments, I am not fully convinced by their replies. 

I am still convinced that the experimentation is not rigorous and sufficiently validated. Unfortunately, the authors do not reply to the very concrete questions I asked; see details in an extra comment.

While I still like the idea and methodological contribution, I am worried about reproducibility, and I am slightly lowering my score and kindly ask the ACs to discuss the reproducibility aspect when considering the paper for acceptance.

### Questions
Please consider the questions raised in the Weaknesses section.

### Soundness
3

### Presentation
4

### Contribution
2
