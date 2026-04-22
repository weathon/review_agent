# Decomposition of Concept-Level Rules in Visual Scenes

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 2

## Abstract
Human cognition is compositional, and one can parse a visual scene into independent concepts and the corresponding concept-changing rules. By contrast, many vision-language systems process images holistically, with limited support for explicit decomposition. Previous methods of decomposing concepts and rules often rely on hand-crafted inductive biases or human-designed priors. We introduce a Concept-Rule Decomposition (CRD) framework to decompose concept-level rules with Large Vision-Language Models (LVLMs), which explains visual input by leveraging LVLM-extracted concepts and the rules governing their variation. The proposed method operates in two stages: (1) a pretrained LVLM proposes visual concepts and concept values, which are employed to instantiate a space of concept rule functions that model concept changes and spatial distributions; (2) an iterative process to select a concise set of concepts that best account for the input according to the rule function. We evaluate CRD on an abstract visual reasoning benchmark, a spatial reasoning benchmark, and a real-world image caption dataset. Across both settings, our approach outperforms baseline models while improving interpretability by explicitly revealing underlying concepts and compositional rules, advancing explainable and generalizable visual reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CRD, a two-stage framework designed to decompose visual scenes into a compact set of human-interpretable concepts and the rules governing their spatial variation. In Stage 1, a pretrained LVLM is used to propose a Visual Concept Set (VCS) and estimate patch-wise concept values. These values parameterize a Concept Rule Function (CRF), modeled as a deep-kernel Gaussian Process, whose log marginal likelihood serves as a per-concept interpretability score. In Stage 2, the framework samples an optimized size-K VCS using an LVLM-guided Metropolis–Hastings (MH) replacement kernel that evaluates proposals via an LVLM-based acceptance ratio. Experiments on two settings: (i) a curated VSB-Sub meta-attribute benchmark of 100 real images, and (ii) RAVEN/I-RAVEN abstract reasoning datasets, demonstrate consistent improvements over vanilla LVLMs for attribute extraction.

### Strengths
- **Originality**: This paper introduces an integration of Large Vision-Language Models (LVLMs) with Gaussian Process–based rule modeling, forming a bridge between probabilistic reasoning and LVLM-driven concept extraction; 
- **Quality**: Methodologically solid: the two-stage inference process (concept proposal + CRF learning) is well-motivated and mathematically grounded in GP theory; Results improve meta-attribute extraction across diverse LVLMs and sizes (Table 1).
- **Clarity**: The paper is well-organized, with clear definitions for components such as Visual Concept Set (VCS) and Concept Rule Function (CRF). Figures (especially Figure 1) effectively convey the pipeline and improve understanding of the model’s decomposition process.
- **Significance**: Advances explainable visual reasoning; Demonstrates that a lightweight, post-hoc decomposition can boost both perception-level extraction and abstract reasoning without retraining the LVLM.

### Weaknesses
- **Missing baselines**: Authors compare CRD to PrAE, LGPP, and CLAP-NP on the RAVEN datasets, but several important baselines are missing — including LEN, ResNet + DRT, CoPINet, MRNet, and SRAN, which are standard on RAVEN and I-RAVEN. These models specifically target abstract visual reasoning and compositional rule induction, providing a fairer comparison for CRD’s intended goals.
- **Data scale**: The natural-image study uses a 100-image subset (VSB-Sub).  There is no clear indication on how the subset was selected. While authors mention the dataset was curated, this is small; thus generalization to varied scenes/domains remains uncertain.
- **Limited statistical reporting**: Tables omit variance, confidence intervals, or multiple runs, so performance differences might not be statistically significant.
- **Scalability & efficiency**: GP inference is O(N^3) in the number of patches. The paper doesn’t quantify runtime vs. patch count or images, nor compare with scalable kernels/inducing points. A complexity/latency table would help.
- **Reproducibility details**: Some choices (e.g., logit clipping of LVLM proposals, replacement selection policy trade-offs, MH schedule/temperature, number of steps) are described but not specified with hyperparameters or seeds.
- **Human study**: The paper involves human annotators for dataset curation and evaluation but provides no discussion on ethical considerations, recruitment, consent, or reproducibility of the human annotation process.

### Questions
1. How does inference time scale with the number of patches or image size? Could you provide empirical runtime or memory profiles?
2. Have you tried scalable GPs (e.g., SVGP or SKI) to see if performance holds under approximate inference?
3. How was the 100-image VSB-Sub subset selected? Was any stratification or randomization used to avoid bias?
4. How sensitive is CRD to the choice of pretrained LVLM? 
5. In Table 3, the CRD framework shows modest performance cоmpared to LVLM baselines. Could the authors elaborate on this observation? How can the performance improve on these benchmarks using CRD?

### Soundness
3

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
3

### Summary
The paper presents CRD (Concept Rule Decomposition), a two-stage framework that leverages large vision–language models (LVLMs) to automatically discover and decompose concept-level rules in visual scenes. In the first stage, a pre-trained LVLM proposes visual concepts and their corresponding concept values. In the second stage, the method iteratively selects a compact subset of these concepts that best explains the visual input, effectively uncovering interpretable concept–rule relationships within the image.

### Strengths
Decomposing visual reasoning into concept–rule pairs is an interesting idea, and leveraging LVLMs for this purpose is novel. The work clearly builds upon prior research in compositional reasoning while introducing a more data-driven and scalable formulation. Moreover, the experimental results demonstrate that integrating CRD consistently improves the performance of LVLMs.

### Weaknesses
1. Unclear writing and motivation. The motivation behind the work is not clearly articulated. While the authors discuss how human visual perception is compositional, it remains unclear why such a decomposition framework is needed in the context of LVLMs or what specific applications it enables. Additionally, the stated contributions are somewhat confusing. One claim suggests that CRD improves the visual representation capabilities of LVLMs, while another claims that CRD outperforms standard LVLMs. Since CRD itself builds upon LVLMs, it is unclear whether the comparison is against the same backbone or a new model.
2. Unproven interpretability claims. The paper repeatedly emphasizes interpretability, but no clear experiments or metrics are provided to validate this claim. Qualitative evidence (e.g., visualization of learned concepts or rules) or human evaluation would strengthen this aspect significantly.
3. Lack of qualitative and quantitative baselines. The paper provides limited qualitative examples, making it difficult to understand what CRD produces in practice. Moreover, several relevant baselines such as SpatialVLM and VisuoThink, which also address compositional or structured visual reasoning, are discussed but not directly compared experimentally. Including such comparisons would help contextualize the claimed improvements.
4. Minor: The font size in Figure 1 is small and difficult to read; increasing it would improve presentation clarity.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a Concept and Rule Decomposition (CRD) framework that enhances LVLMs with interpretable reasoning. CRD extracts visual concepts from images using an LVLM, models their spatial patterns via Gaussian Process–based Concept Rule Functions (CRFs), and selects the most consistent concept–rule pairs through LVLM-guided sampling. Experiments on real-image and abstract reasoning benchmarks show that CRD improves both performance and interpretability across diverse LVLMs.

### Strengths
1. It introduces a unified, model-agnostic approach that explicitly decomposes visual understanding into concepts and spatial rules, improving interpretability.
2. The proposed method consistently enhances multiple LVLMs across both real-image and abstract reasoning benchmarks without additional supervision.
3. The LVLM-guided sampling and GP-based rule modeling allow CRD to adapt flexibly to different models and datasets.

### Weaknesses
1. The method focuses mainly on spatial rules within static images and does not address temporal or relational reasoning across scenes.
2. The performance and interpretability depend heavily on the LVLM’s initial concept extraction accuracy. It would be better if there is human evaluation of LVLM’s initial concept extraction accuracy.
3. The Gaussian Process–based CRF and sampling procedure introduce extra computational overhead compared to direct LVLM inference.

### Questions
How sensitive is the CRD framework to the quality of concept proposals from different LVLMs, and could errors in these initial concepts propagate to rule learning and sampling outcomes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper describes a procedure for concept extraction that sits on top of generic LVLMs. The method samples a set of "visual concepts" that appear in the image, and the learns a set of "rules" which relate them--where the rules appear to be spatial positions of the identified concepts (this is not entirely clear to me from reading, see questions below). The model is evaluated on two tasks, the first is a visual attribute extraction task (something like high level object recognition) and the second is RAVEN. The model performs better than generic LVLMs in both cases.

I am lukewarm about this paper because I think I am missing some key aspects of the intuition and motivation behind the model. The paper is a bit hard to read -- it is heavy on notation and formalization but lacking in intuition and examples. I thus don't feel I am fully able to appreciate what the contribution is, and how significant it would be. If the authors can provide more clarification in their response, I would feel more comfortable making a stronger recommendation.

### Strengths
* The paper presents a new procedure for making LLMs perform better on some visual reasoning tasks

### Weaknesses
* As written, the exact contribution and significance thereof is not very clear. Please see my questions below. Without better explanation and motivation, its hard to tell if how impactful the work can potentially be

### Questions
* My understanding is that "concepts" you extract are just objects, and the "rules" you extract are just estimates of the possible positions (spatial location) of detected objects. Is this correct? If not, can you give an example of a concept that is extracted that is not a noun/object and a rule that isn't spatial? I ask because this greatly affects the generalizability of the architecture -- many concepts don't readily fit this framework. E.g., the concept of "wearing" itself refers to a spatial relationship between objects, so is higher order than what (I think) your method can handle. Apologies if I misunderstood your method, please correct me if I did.
* Can you explain what role the "rules" play in the meta-attribute extraction task? I don't understand the motivation for this task -- it looks like it is just object recognition (with some higher-level labels). Why is this task a good illustration of the strengths of your method?
* Also on the attribute extraction task -- why did you only select 100 examples? Why could you not just use the whole dataset for evaluation?
* You place emphasis on the generality of your procedure, as it is based on LVLMs which are generic. However, you don't actually demonstrate that a system that has undergone CRD training is still generic -- i.e., does it retain the general purpose abilities that it had before the CRD was applied? The results give the feeling that it is in fact a procedure that is fairly specific to RAVEN, and the results are primarily on RAVEN. It would be good to demonstrate that this is not the case by evaluating on more diverse types of tasks to back of the claim of genericity. E.g., in the intro, you mention several probabilistic programming models such as Lake et al...would your procedure extend to those tasks?
* Can you say how many parameters are involved in fitting this model? Apologies if I missed this, but it wasn't clear to me how much to consider this a new model vs. a wrapper on a pretrained model

Typos: check your citation formats to make sure they are standard. E.g., when the citation is inline with the text, it should be in caps (e.g., "Jane Doe (2015) previously showed this was possible"), and otherwise in parenthesis (e.g. "It has been previously shown that this is possible (Doe, 2025)")

### Soundness
2

### Presentation
1

### Contribution
2
