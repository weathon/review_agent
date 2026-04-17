# What Drives Compositional Generalization in Visual Generative Models?

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Compositional generalization, the ability to generate novel combinations of known concepts, is a key ingredient for visual generative models. Yet, not all mechanisms that enable or inhibit it are fully understood. In this work, we conduct a systematic study of how various design choices influence compositional generalization in image and video generation in a positive or negative way. Through controlled experiments, we identify two key factors: (i) whether the training objective operates on a discrete or continuous distribution, and (ii) to what extent conditioning provides information about the constituent concepts. Building on these insights, we show that relaxing the MaskGIT discrete loss with an auxiliary continuous JEPA-based objective can improve compositional performance in discrete models like MaskGIT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a systematic empirical investigation into the factors that enable or hinder compositional generalization in visual generative models. Through controlled experiments across multiple domains, authors demonstrate that achieving robust compositional generalization in visual generative models can be facilitated by continuous distribution modeling and full, non-quantized conditioning signals. Building on these insights, the authors propose a JEPA-based auxiliary loss that improves the compositional capabilities of MaskGIT.

### Strengths
1. This paper is well written.
2. To answer the central question raised by authors (``What are factors that enable or hamper compositional generalization in visual generative models?''), the authors designed extensive experiments. The results are insightful.

### Weaknesses
1. This paper experimentally verifies a previously intuitive factor. A deeper discussion of why discrete objectives hinder compositionality would strengthen the narrative.
2. This paper uses controlled experiments to demonstrate the relationship between continuous loss and compositional generalization. However, mainstream works such as MaskGIT or DiT are founded on large-scale pre-training, which means experiments without scaling may show completely different results. I am curious about, under the premise of large-scale pre-training, how is the generalization ability of different models for unseen combinations?
3. The result of this paper could be significant in training LLMs. However, the discussion of `Do our results extend to language?' only provides a comparison of COCONUT and CoT, which are inference-time continuous. I suggest that the comparison should be conducted in the pre-training stage to illustrate that LDM can have a higher upper limit than LM.

### Questions
See ``Weakness'' section.

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
4

### Summary
This paper investigates compositional generalization in generative models. More specifically, it performs extensive experiments to ablate the recipe of DiT and MaskGIT including tokenizer, masking, objective function, output space (continuous or discrete) to find out what contributes most to compositional generalization in visual generative models. Based on the observation that continuous objective function is a key factor, the paper proposes add an auxiliary continuous JEPA-based objective term to enhance compositional performance of MaskGIT.

### Strengths
**Strength**
1. Understanding compositionality in generative model is an interesting direction. 

2. Extensive experiments are performed to figure out the most important factors that give rise to compositional generation in the training recipes. The study is straitforward and easy to follow.

### Weaknesses
1. The main claim of the paper goes too generic and cannot be supported by enough evidence. Since the paper only studies a class of visual generative models such as DiT and MaskGIT, it can not claim the study of general generative models. For example, it is overstated in the title"WHAT DRIVES COMPOSITIONAL GENERALIZATION IN VISUAL GENERATIVE MODELS?". It can only support "WHAT DRIVES COMPOSITIONAL GENERALIZATION IN DiT?" and "IMPROVED TECHNIQUES for ENHANCING COMPOSITIONALITY of MaskGIT". Likewise, there are many such overstatements in the paper.

2. In section 6.2, it studies how the incorporation of an auxiliary loss enhances compositional generation performance of MaskGIT. However, it is unclear if this extra objective term degrades image generation performance. In other works, are you trading image quality for compositionality? No qualitative or quantitative (FID) results are provided. Actually, FID should be reported for all comparisons because generation quality are as important as compsitionality in generation tasks. 

3. As an experimental work that simply compare models and training factors, more complicated dataset are expected. Regarding this point, opinions from other reviewrs will be taken into account.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors studied the design choices of visual generative models that affect models' compositionality, driven by the performance of existing models. The authors focused on 4 design choices: the tokenizer type, the underlying generative process, and the conditioning quality. The authors concluded that the tokenizer does not matter but continuous generative distributions and good conditioning quality are important. The authors also proposed to augment MaskGIT with the JEPA objective based on their analysis.

### Strengths
1. The paper is easy to follow.
2. The authors did a thorough analysis of DiT and MaskGIT models which helps understand the compositional generation task.
3. Experimental details are provided in the appendix.

### Weaknesses
1. The authors' conclusions are too strong given that the analysis is entirely based on a set of four models. Comparing these models is insufficient to conclude that using discrete distributions is not suitable for compositional generation, simply because MaskGIT underperforms. The other conclusions are also not convincing for the same reason.
2. There is no clear evidence showing that the proposed MaskGIT with JEPA loss outperforms existing models, especially when compared with DiT. Experiments on the real dataset show no visual improvement in Figure 18 and quantitative comparisons are lacking.
3. There are typos here and there. For example, the "replaces" in line 150.

### Questions
1. Many works composed diffusion models or energy-based models directly in the pixel space for compositional generation. How do those methods compare against the models discussed here?
2. For a practical perspective, the generation quality presented in the paper is far behind the SOTA LLM/VLM. How do they perform on these tasks?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper conducts a systematic study of compositional generalization in modern visual generative models.
Through carefully controlled experiments across several architectures (DiT, MaskGIT, MAR, GIVT), the authors identify two key factors influencing compositionality:

1. Whether the model learns a continuous vs. discrete distribution;
2. The degree to which the conditioning signal preserves complete information about underlying factors.

While the choice of tokenizer does not fundamentally alter the compositional abilities.

They further propose an auxiliary JEPA-based loss that augments discrete models like MaskGIT with a continuous predictive objective. This addition improves compositional generalization and yields more disentangled internal representations, as confirmed by mechanistic interpretability analyses (polysemanticity and circuit overlap).

### Strengths
1. Compositional generalization is a fundamental and increasingly important challenge for generative models. Investigating what architectural and objective factors drive it is both interesting and highly relevant to the broader ML community.
2. The authors conducted a series of experiments considering model architecture, training objective, conditioning description, and data modality, making their statements comprehensive and generalizable.
3. By incorporating a JEPA-style continuous predictive loss into MaskGIT, the authors successfully improved the compositionality of MaskGIT, which further proved continuous representation learning outperforms discrete token-based modeling in terms of compositional generalization. Moreover, the authors performed polysemanticity and circuit analysis, showing how continuous learning objective affects the model.

### Weaknesses
1. Though the authors conducted systematic experiments to support their findings, however the conclusions themselves are some what intuitive:
- Continuous training objective favors compositionality: continuous representations enforce the model to learn a smooth structure which permits interpolation between elements, therefore benefiting composition. It is intuitive that continuous latents supremes discrete tokens in this ability.
- Precise conditioning is critical: This one is more appearent. If the conditioning is precise and factor-complete, the model can learn to represent and recombine each factor explicitly. Otherwise, the learned representation to the training data itself could be problematic, not to mention compositionality.

2. The experiments on MaskGIT with JEPA loss showed how the polysemanticity and the neuron overlap are reduced, however, the causality between them and compositionality is unclear.

3. Several sections still lack sufficient detail, and certain descriptions are ambiguous and confusing. For example, how are the signals binarized in Section 5? Line 404~405 said "A head is considered polysemantic if the feature similarity difference exceeds a threshold", however, the head should be considerd polysemantic if the difference is large **across multiple factor pairs**. The "Overlap(%)" in Fig.6(c) also seemed to be a mistake.

4. The paper assumes that GIVT and MAR can serve as clean “interpolations” between DiT (continuous diffusion) and MaskGIT (discrete masked modeling). However, in practice, GIVT and MAR models differ MaskGIT and DiT in more than just the objective function or masking mechanism. The experiments is not strictly rigorous in the fundamental model settings.

### Questions
1. Why the compositional accuracy falls low or even down to zero during the later period of training? Especially in Fig. 4(b), the line "101" achieved almost 100% accuracy in the middle of training and then falls to zero. How were the labels dropped out, and why it caused such performance？

2. How does the reduced polysemanticity and neuron overlap lead to better compositionality? If this is intuitively considered, it could also be intuitively found out that continuous training objective and precise conditioning lead to better compositionality. In other words, could the improvements stem from smoother optimization dynamics rather than semantic disentanglement?

### Soundness
2

### Presentation
3

### Contribution
2
