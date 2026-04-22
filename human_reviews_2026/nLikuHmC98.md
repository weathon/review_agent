# Deconstructing Guidance: A Semantic Hierarchy for Precise Diffusion Model Editing

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Text-guided image editing requires more than prompt following—it demands a principled understanding of what to modify versus what to preserve. We investigate the internal guidance mechanism of diffusion models and reveal that the guidance signal follows a structured semantic hierarchy. We formalize this insight as the Semantic Scale Hypothesis: the magnitude of the guidance difference vector ($\Delta\boldsymbol{\epsilon}$) directly encodes the semantic scale of edits. Crucially, this phenomenon is theoretically grounded in Tweedie’s formula, which links score prediction to the variance of the underlying data distribution. Low-variance regions, such as objects, yield large-magnitude differences corresponding to structural edits, whereas high-variance regions, such as backgrounds, yield small-magnitude differences corresponding to stylistic adjustments. Building on this principle, we introduce Prism-Edit, a training-free, plug-and-play module that decomposes the guidance signal into semantic layers, enabling selective and interpretable control. Extensive experiments—spanning direct visualization of the semantic hierarchy, generalization across foundation models, and integration with state-of-the-art editors—demonstrate that Prism-Edit achieves precise, robust, and controllable editing. Our findings establish semantic scale as a foundational axis for understanding and advancing diffusion-based image editing.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces semantic scale hypothesis that frames guidance magnitude as an information-theoretic signal to reflect a semantic hierarchy and proposes Prism-Edit that decomposes the guidance signal into different semantic layers for selective control of diffusion model editing.

### Strengths
1.	Extensive experiments have been conducted to demonstrate the effectiveness of Prism-Edit, especially for challenging background edits. 
2.	The semantic scale hypothesis provides a theoretical perspective to reveal the semantic hierarchy in diffusion model guidance signal.

### Weaknesses
1.	Since DiffEdit also leverages guidance differences to achieve selective control in diffusion models, this work’s contribution requires a detailed empirical comparison between DiffEdit’s hard spatial mask and Prism-Edit’s soft binary mask. Without such analysis, the novelty and advancements of the proposed method over previous studies are not sufficiently convincing. Moreover, because Algorithm 1 incorporates a hard-mask module, an ablation study on this component would also be necessary.
2.	A substantial portion of the paper is devoted to deriving closed-form bounds for the guidance difference; however, these results do not seem to be connected to the method’s technical design, nor do they provide deeper insights beyond the analyses already discussed in Sections 4.2 and 4.3.
3.	Some parts of the method are unclear. The choice of probe internal and how to calculate the soft binary mask $W$ lack explanation. Algorithm 2 is unclear.
4.	It miss the introduction of baselines and the details about how to apply Prism-Edit to P2P, PnP and LEDITS++.

### Questions
1.	In Fig. 4, Prism-Edit has worse results w.r.t. the CLIP score. Can the authors give some explanation or analysis?
2.	In Fig. 5b, it seems DDIM Inv. obtains better results than Prism-Edits. It would be better to include some analysis for this kind of results.
3.	How does the proposed method utilize negative prompts?
4.	For the different datasets, are the hyperparameter settings for Prism-Edit kept the same?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper suggests the Semantic Scale Hypothesis: the magnitude of the guidance vector correlates with the semantic scale of an edit. Specifically, low-variance regions, where the model is more certain and which typically correspond to foreground objects, exhibit higher guidance magnitudes, while high-variance regions, often background, show lower magnitudes. Building on this, the authors propose Prism-Edit, which adjusts edits by selecting ranges of guidance magnitudes, enabling object-only or background-only manipulation without requiring any object masks. The method is shown to work across various types of models and editing techniques.

### Strengths
- Clear conceptual novelty. To my knowledge, this is the first work to analyze editing behavior through the magnitude of the guidance vector itself. Furthermore, the analysis that higher-magnitude regions align with more certain, typically object-centric areas, while lower magnitudes map to uncertain, often background regions, is both intuitive and seems to have its own advantages

- Mask-free, plug-and-play editing with broad model coverage. By selecting guidance-magnitude ranges, Prism-Edit enables foreground/background-targeted edits without segmentation masks. The procedure is training-free and plug-and-play, making it easy to drop into existing pipelines. Moreover, successful applications across models and editing methods indicate a model-invariant control axis, improving reliability and external validity.

### Weaknesses
- Condition sensitivity. Guidance vectors change with prompts, seeds, schedulers, and editing method choices. It is unclear whether the proposed magnitude, semantics relationship consistently holds under such variations. Is this tendency of the relation between guidance magnitude and the semantic scale of edit preserved regardless of the given condition?  Additional analysis would strengthen the claim.

- Thresholding and disentanglement. Thresholds are manually tuned and depend on the editing method (Table 2, Appendix). Figure 10 also suggests continuous trade-offs as the threshold varies, implying object and background information remain partially entangled. While the paper achieves useful partial disentanglement, the dependence on thresholds should be better characterized.

- Per-image statistics & efficiency. The method appears to compute per-image guidance-magnitude statistics. Even with z-score normalization, distributions may differ across images, raising concerns about the speed and stability of edited results depending on the given image and condition.

### Questions
- Object-scarce scenes. The paper frames “semantic scale” mainly as object vs. background. How does the interpretation extend to images without salient objects (e.g., textures, landscapes, or abstract scenes)? What do high- vs. low-magnitude regions represent in such cases?

- Magnitude variability. How much do guidance-magnitude distributions vary across images, prompts, and inversion methods? Including representative histograms for multiple settings would clarify variance and help practitioners choose thresholds.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the guidance mechanism of the diffusion models, and proposes a hypothesis that the magnitude of the guidance difference vector directly encoded the semantic scale of the edits. Low-variance regions yield large magnitude differences with structural changes, while high-variance regions yield small magnitude differences with stylistic adjustments. By using this observation, it proposes to apply either a dynamic mask or a static mask computed from the guidance magnitude to the original guidance difference, then to perform edit. The results show that the proposed method can do the edit while preserving the layout better than the baselines.

### Strengths
1. It’s interesting to mathematically show that the magnitude of the guidance difference itself contains useful information that can be useful to guide the edits.

2. The proposed method can be applied to different editing methods in a plug-and-play way, which is flexible and universal.

### Weaknesses
While after applying the proposed method the results can be improved over the original editing method, the edited results are still not very satisfactory. I understand that the results are also greatly influenced by the base model / base method. Just to say in Figure 7, where the base model is advanced, the results after applying the proposed method are not good. With SD3, especially for the bear example, the background looks unrealistic. With Flux RF-Inversion, the background improves, but the overall style goes from realistic to more animation style. With Flux Stable-flow, the foreground objects either have a boundary shadow, or deteriorate. 

While I do agree that the proposed method is simple and can improve over the baselines, it seems that the proposed method itself is still not able to achieve very high quality results without any other modifications. This makes me doubt that either there is a better way to utilize the hypothesis rather than simply applying a static/soft mask, or the proposed hypothesis itself alone is not able to achieve a better quality.

### Questions
1. What is $M_{final}$ in Equation 8?

2. What is $W_{sem,t}$ in Equation 9? Is it the absolute value of the guidance difference?

3. Is the high-noise window in line 275 needed to be calculated per editing example, or is there a universal $t$ value that can work across every example?

### Soundness
3

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
4

### Summary
The paper proposes the Semantic Scale Hypothesis which motivates a training-free, plug-and-play module to improve image editing performances across models.

The paper could be a nice contribution on understanding the diffusion process. I'm willing to increase my score if the questions are clarified.

### Strengths
- The paper is well-written
- Strong empirical support and a nice theoretical connection for the Semantic Scale Hypothesis
- The method is model-agnostic

### Weaknesses
- The thresholds used in Table 1 seem different for each baseline method, but "fixed thresholds" are claimed to be stable (286-288). The guidance scales seem to be different as well. How sensitive are these hyperparameters? How should a user pick the good ones?

### Questions
- Why is the noise range $t \in [900,880]$. Does a fixed $t=900$ or a wider range $t \in [900,800]$ not work? How to justify the number $880$?
- What is the motivation behind $W_{sem,t}$ as it's computed from $\Delta_{\epsilon_t}$ itself?

### Soundness
3

### Presentation
4

### Contribution
3
