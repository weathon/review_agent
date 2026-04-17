# Visual symbolic mechanisms: Emergent symbol processing in Vision Language Models

- Decision: Accept (Oral)
- Scores: 8, 6, 6, 6

## Abstract
To accurately process a visual scene, observers must bind features together to represent individual objects. This capacity is necessary, for instance, to distinguish an image containing a red square and a blue circle from an image containing a blue square and a red circle. Recent work has found that language models solve this ‘binding problem’ via a set of symbol-like, content-independent indices, but it is unclear whether similar mechanisms are employed by Vision Language Models (VLM). This question is especially relevant, given the persistent failures of VLMs on tasks that require binding. Here, we identify a previously unknown set of emergent symbolic mechanisms that support binding specifically in VLMs, via a content-independent, spatial indexing scheme. Moreover, we find that binding errors, when they occur, can be traced directly to failures in these mechanisms. Taken together, these results shed light on the mechanisms that support symbol-like processing in VLMs, and suggest possible avenues for reducing the number of binding failures exhibited by these models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates emergent symbolic mechanisms for binding features and objects in vision language models. It presents evidence for a binding process where the model first computes a spatial index for the target object and then uses that index to retrieve its features. Overall, this paper looks like a good paper with solid methodology and evidential conclusions to me. The proposed binding mechanism is justified and intuitive, and more importantly generalisable across different models, which makes the paper relevant and useful to community.

### Strengths
1. The paper is well-written and easy to follow.
2. The analysis methods used including PCA and RSA to localise when position and features dominate, CMA and intervention to find contributing heads are soild methods.
3. Given that the experiments throughout 7 VLMs show a consistent pattern in the feature binding mechanism, the findings and conclusions are generalisable.

### Weaknesses
1. Although 7 VLMs are tested, they are from 2 model families (Qwen and Llava). The mechanism still generalises with the current setting but can be strengthened when tested on more model families. 
2. The experiment is conducted under controlled synthetic settings. While this is required for a grounded analysis, explanaing how the mechanism can be exploit/identified in open-domain can be useful.
3. Some more explanation under Fig1 caption can help me to understand the paper better.

### Questions
The paper says improving binding performance may require either architectural innovations or training strategies. Is there a good benchmark/criterion (i.e. the one used in this paper) to show progress? How generalisable is the conclusion to more complex open-domain?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates whether vision-language models employ symbolic mechanisms similar to those found in language models for solving the binding problem: the challenge of correctly associating features to form coherent object-level representations. Using a scene description task where models must describe remaining objects in multi-object images, the authors employ a thorough suite of methodologies in order to uncover the underlying mechanisms across a range of VLM architectures: PCA, representational similarity analysis (RSA), causal mediation analysis (CMA), and targeted interventions. The core finding is a three-stage binding process: 1) Position ID heads compute spatial indices for objects mentioned in the prompt; 2) ID selection heads select the position ID for the target object; and 3) Feature retrieval heads use the selected position ID to access target object features. The authors then perform a series of analyses to characterize the binding mechanisms in greater detail. They find that models tend to use relative vs. absolute position IDS; that the binding mechanisms generalize to more realistic images; that position IDs are localized to image patches containing the relevant object; and that the same position IDs are used in more complex relational reasoning tasks. They finally perform an error analysis, demonstrating that binding errors correlate with failures in position ID processing. 


The authors then demonstrate that binding errors correlate with failures in position ID processing, particularly in low-entropy conditions where objects share features.

### Strengths
1. **Importance of the research question**. The binding problem represents a fundamental challenge in AI systems, and understanding how VLMs solve it has significant implications for improving multi-object reasoning capabilities. 
2. **Novel mechanistic insight into VLMs**. The paper makes a strong conceptual and empirical contribution by identifying position IDs (content-independent, spatially grounded indices that play a symbolic role in feature binding). This is a novel mechanistic finding in multimodal interpretability.
3. **Methodological rigor**. The paper employs a comprehensive multi-method approach. The progression from RSA to CMA to direct interventions provides converging evidence for the proposed mechanisms.
4. **Cross-model validation & generalization**. The authors test their findings across multiple VLM architectures (Qwen-VL, LLaVa variants). The appendices contain detailed methodological descriptions, extensive additional analyses, and validation experiments.

### Weaknesses
1. **Limited exploration of real-world consequences.** While the authors link the discovered symbolic binding mechanisms to binding failures, they do not connect the mechanisms to the downstream behavioral consequences of binding failures gestured towards in the introduction (e.g. counting, visual search). Bridging this gap would strengthen the work.
2. **No investigation into learning origins.** The paper identifies what the mechanisms are but not how they arise. Do they emerge naturally from pretraining data distributions, or are they artifacts of architectural bias? 
3. **Unclear how to improve binding**. Related to Weakness 2, the authors do not attempt to modify / strengthen the binding mechanisms. This would be practically useful and would also enable direct causal claims about the relationship between these mechanisms and downstream behavior.

### Questions
1. Do the authors have hypotheses or preliminary observations about when / how these position ID mechanisms emerge during training or what factors might influence their emergence (architecture, training data, etc.)?
2. Given the discovery of these three distinct stages (retrieval, selection, feature retrieval), could architectures be explicitly designed to separate or reinforce them, similar to slot-attention or object-centric models?
3. Is there evidence that the position IDs interact systematically with the text tokens (e.g., positional embeddings in the language stream), or are they confined to the vision encoder?
4. In the scene description task, are object attributes (e.g. shape, color) repeated in the scene? (e.g. two or more green objects, two or more squares). If a scene of three objects contains three distinct colors and shapes, then it seems that models could solve the task by just picking whatever attributes are left out of the prompt rather than binding.
5. The term "symbol" is strong and implies total content-independence. To what extent do the authors believe these mechanisms are truly symbolic, e.g. in the way cognitive scientists mean?

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
4

### Summary
The paper studies how vision-language models address the binding problem, which involves linking visual features, such as color and shape, to specific objects. The paper proposes that these models use spatial "position IDs" as symbolic indices to perform this binding. Through representational similarity analysis, causal mediation, and targeted interventions on various VLM families, the authors identify three groups of attention heads responsible for retrieving, selecting, and using these position IDs to recover object features. The authors argue that this mechanism explains how VLMs achieve visual binding and why binding errors occur.

### Strengths
The paper addresses an important question about how VLMs perform visual binding and compositional reasoning.

It evaluates a diverse range of model architectures and scales (Qwen2.5-VL, LLaVA-1.5, and LLaVA-OneVision), differing in backbones, design choices, and training data, which strengthens the generalizability of the findings.

It employs multiple complementary methods, representational analysis (RSA), causal mediation, and targeted interventions, to support its hypotheses from several analytical perspectives.

### Weaknesses
The main claim that VLMs develop symbolic binding mechanisms similar to those in LLMs seems incremental since comparable mechanisms are already well-established in language models. The results show that related processes emerge in multimodal settings, which seems expected given the shared Transformer backbone. Could the authors clarify what is genuinely new about these mechanisms in the visual domain beyond applying known LLM findings to spatial inputs?

The paper aims to demonstrate general mechanisms of visual binding, yet the datasets are highly controlled and simplified. Even the “photorealistic” PUG scenes contain only two clearly separated, uniformly sized objects, avoiding occlusion or clutter. Testing whether the proposed mechanisms persist under more realistic conditions would be essential to support their claim of generality.

The explanation of the relative versus absolute index analysis could be made clearer, as the underlying idea is difficult to follow in its current form.  In addition, could the observed differences between models simply reflect their positional encoding priors (for example, relative RoPE embeddings in Qwen versus absolute sinusoidal embeddings in LLaVA) rather than evidence for an emergent symbolic mechanism? Clarifying this distinction would strengthen the interpretation of the results.

Minor comments:
Line 306: Typo in “wihtin”
Line 347: Missing space.
Line 427: The reference to the Appendix does not lead to the generation details as stated. The appendix structure should be cleaned up.

### Questions
Could the authors clarify what is genuinely new about these mechanisms in the visual domain beyond applying known LLM findings to spatial inputs?

Testing whether the proposed mechanisms persist under more realistic conditions would be essential to support their claim of generality.

Could the observed differences between models simply reflect their positional encoding priors (for example, relative RoPE embeddings in Qwen versus absolute sinusoidal embeddings in LLaVA) rather than evidence for an emergent symbolic mechanism?

### Soundness
3

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
4

### Summary
In this paper the authors shed light on how VLMs work, specifically how they solve the binding problem (associating features together to represent distinct objects). The authors identify the “position IDs” mechanism composed of 3 steps:  first retrieval of the position ID of the image objects corresponding to the input prompt. Then, selecting the position ID for the target object, and finally retrieving the semantic features of the target object. The authors provide evidence for this scheme across 7 models using synthetic and photorealistic datasets.

### Strengths
- The authors identified an interesting internal working mechanism of VLMs, explaining also why VLMs sometimes fail in spatial reasoning. I think understanding how VLMs work is a high-impact problem given recent benchmarks showing that VLMs fail in spatial reasoning tasks.
- I think the authors provide enough evidence to support the “position IDs” mechanism. They identify what layers correlate with each step and perform causal mediation analysis to identify the specific attention heads that are causally involved in the 3 steps. 
- The authors show that binding errors are correlated with failures in the position ID mechanism, potentially informing future improvements to VLMs architectures.

### Weaknesses
- It is not clear what the novelty is in the paper in terms of methodology and techniques compared to the paper of Yang et. al (2025) that identifies similar mechanisms yet for LLMs. I think the authors should discuss it in the paper.
- The correlation between VLMs failures and position ID mechanism failures is just correlation, not causation. It is not clear if mechanism failures really cause binding errors.
- I think the writing can be improved, e.g. provide the specific prompt that is used in each case, specifically in section 3.4 and in 4.1.
- Minor: there are 2 “Figure ??” in the paper.

### Questions
How does the size of the model affect the results ?

### Soundness
3

### Presentation
2

### Contribution
3
