# Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Visual prompt tuning offers significant advantages for adapting pre-trained visual foundation models to specific tasks. However, current research provides limited insight into the interpretability of this approach, which is essential for enhancing AI reliability and enabling AI-driven knowledge discovery. In this paper, rather than learning abstract prompt embeddings, we propose the first framework, named Interpretable Visual Prompt Tuning (IVPT), to explore interpretability for visual prompts by introducing cross-layer concept prototypes. Specifically, visual prompts are linked to human-understandable semantic concepts, represented as a set of category-agnostic prototypes, each corresponding to a specific region of the image. IVPT then aggregates features from these regions to generate interpretable prompts for multiple network layers, allowing the explanation of visual prompts at different network depths and semantic granularities. Comprehensive qualitative and quantitative evaluations on fine-grained classification benchmarks show its superior interpretability and performance over visual prompt tuning methods and existing interpretable methods. Our code is available at https://github.com/ThomasWangY/IVPT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes a novel, interpretable approach to visual prompt tuning. Here, interpretable tokens are appended to each ViT layer's computation, thereby extracting relevant information while keeping the image encoder frozen. The interpretability comes from the fact that the learnable tokens are not just embeddings, but they can be interpreted as localized prototypical parts, which give insights into which parts of the image are being focused on for each prompt token.

### Strengths
This work manages to nicely integrate interpretability methodology into visual prompt tuning(VPT). Although related to prototype-based literature, it is non-trivial how to integrate it into VPT. The method is predominantly presented in a clear manner. Figures support the textual understanding.
Results look promising in the aspects accuracy as well as interpretability, showcasing that the interpretability framework does not bottleneck the predictive performance on CUB. Also, the results cover various aspects of interpretability, ranging from quantitative metrics to visualizations and human user studies. I appreciate that clean code is provided.

### Weaknesses
* (i) My main concern is that each across datasets, the axes of evaluations are not the same. That is, Accuracy is only shown for CUB, and visualizations are missing for PartImageNet and PASCAL-Part. This gives the impression that for the left-out evaluations, the results were omitted due to being poor. This is concerning for the performance aspect, as CUB is a quite simple dataset and if this method is unable to generalize to other more complex datasets, it is problematic. Note that I would expect a performance degradation due to the interpretability adjustments, so the goal for IVPT should just be to not lose out too much. In the current state of the manuscript, I have to assume that IVPT is unable to reach a performance similar to baselines in all datasets apart from CUB.

* (ii) An additional major concern is the faithful interpretability for more complex datasets. For CUB, Gleason-2019, Stanford Cars, FGVCAircraft datasets, the subparts are similar across samples, thereby making them interpretable. Additionally, the parts are also what is important for classification. However, I could envision that for more complex datasets (e.g. ImageNet and Pascal), the prototypes are becoming less interpretable, as they might be used to process information in ways that are not directly understandable when looking at the activation map. That is, without explicitly enforcing it, the visualization in image-space might not be anymore what is going on in the computations. 

* In line with the previous point, there might be considerable leakage, as the image regions are processed via weighted average and MLP such that the actual attention computations could deviate greatly from what is visualized. 

* I think the separation of $n$ learnable prompts and $m$ concept prototypes is sometimes unclear. E.g. line 128-129 and formula 3 use similar notation but different lengths, which was confusing at first.

### Questions
* Can the authors provide predictive performance compared to baselines on other datasets to counter concern (i)?

* Can the authors provide qualitative results for PartImageNet and PASCAL-Part to counter concern (ii)?

* If I understand correctly, previously in VPT, the prompt tokens were learnable embeddings that were the same for different samples. Now, the prompt tokens are weighted average of the input image, thereby different per-image. How does that differ conceptually from the previous global prompt tokens? What are the benefits and potential downsides?

* The concept regions are computed patch-wise. How did the authors obtain non-rectangular prompt visualizations, such as in Figure 3?

### Soundness
4

### Presentation
3

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
This work aims to propose a fine-tuning paradigm that learns interpretable visual prompt during fine-tuning via cross-layer concept prototypes. A concept region discovery module is proposed to learn prototypes with semantic meaning, and an intro-region feature aggregation module is proposed to group the features belonging to certain regions. Experimental results in CUB demonstrate improved performance compared to visual prompt tuning and show superior interpretability scores for the learned visual prompts compared to part-prototype networks.

### Strengths
- This work investigates the interpretability of visual prompts during fine-tuning, which is a less explored area.
- The method design in making the visual prompts more interpretable is reasonable.
- Experimental results show improved performance and interpretability compared to 2 types of baselines.

### Weaknesses
*Motivation*

Why at all should the “visual prompt” be interpretable is not well justified. Since they are parameters learned to adapt the model from one domain to another domain, if they are possible to be interpretable, I would expect them to explain the domain shift instead of part-prototypes. Part-prototypes could explain a decision making process, but are not that meaningful in a fine-tuning process? Especially when the sum of the contribution of part-prototypes do not fully explain the decision making process in the proposed framework (see next concern).


*Interpretability*

The final classification process in part-prototype networks can be fully interpreted by the contribution of each individual part-prototype. However, the prompts in this work only contribute to part of the classification logits, most tokens from frozen part of the network remain uninterpretable. Do the framework adopt a classification token in the final prediction or the average of all token representations? If using a classification token, the information are aggregated from both interpretable visual prompt tokens and rest tokens via self-attention, making the contribution of the visual prompt unclear. Such a mechanism also makes it less meaningful to make visual prompts interpretable.

*Evaluation*

How are the areas that the evaluated visual prompts correspond to calculated and evaluated against the annotations in CUB?

*Missing details/analysis*

What’s the difference between the interpretability scores of prompts in different layers? How many prompts are used in each layer? What’s the influence of number of prompts on their interpretability?

### Questions
1.	How do you obtain the performances of ProtopNet and its following works based on ViT architectures in Table 1? How are they implemented? These original works are not designed for ViT and do not report relevant results.

2.	Can the learned visual prompts explain anything related to visual prompt **tuning**?

### Soundness
3

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
This paper proposes IVPT, an interpretable visual prompt tuning framework that aligns prompts with category-agnostic concept prototypes across network layers. It introduces concept-region discovery, intra-region feature aggregation, and cross-layer concept fusion to make prompts human-understandable while preserving accuracy. Experiments show IVPT achieves better interpretability and comparable performance to standard VPT.

### Strengths
1. It introduces a clear, concept-grounded approach that links visual prompts to human-understandable concepts across layers.
2. It demonstrates improved interpretability metrics and visualization quality without sacrificing classification accuracy.
3. The method is model-agnostic and works across different ViT backbones and domains, showing good generalization.

### Weaknesses
1. Experiments focus mainly on fine-grained classification. Broader tasks (e.g., detection, segmentation) are not explored.
2. The method relies on well-learned concept prototypes, which may be sensitive to initialization or domain shift.
3. The multi-layer prototype alignment and multiple loss terms increase implementation and computational complexity.

### Questions
How stable are the learned concept prototypes when transferring to new domains or unseen categories?

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
4

### Summary
This paper explores the interpretability of vision–language models (VLMs) through a task-specific concept alignment framework. The authors aim to understand how visual and linguistic concepts align across different tasks, providing both qualitative and quantitative analyses to uncover internal reasoning processes of VLMs.

### Strengths
The topic is highly relevant and well-motivated, addressing the growing need to make large-scale vision–language models more interpretable and transparent.

The paper provides comprehensive experimental analysis, including multiple datasets and evaluation settings, with clear presentation of alignment patterns.

### Weaknesses
While the analysis is useful, the methodological contribution is limited compared to prior interpretability frameworks. The paper primarily extends known alignment techniques rather than introducing a fundamentally new interpretability paradigm. The related work section lists many recent studies but lacks an in-depth synthesis or a critical comparison. A more detailed discussion of existing SOTA interpretability methods and their limitations would help clarify what specific gap this work fills. Because the related work discussion is broad but not deep, the method section does not clearly delineate the paper’s unique conceptual or technical contribution relative to existing approaches.

While the experiments are well executed, the paper would benefit from showing a potential use case, for example, how this concept alignment framework could assist in model debugging, bias detection, or downstream task understanding.

### Questions
Can you expand the related work section to more deeply analyze existing SOTA interpretability methods and highlight the specific gap your work addresses?

Have you considered evaluating your approach in a concrete application setting (e.g., identifying bias, failure analysis, or improving model transparency for users)?

How generalizable is your approach across different VLM architectures? Are there differences in alignment quality depending on the model type?

Could you include an example of how concept alignment results might be used in a downstream interpretability workflow or decision-support scenario to make the method’s impact more tangible?

### Soundness
2

### Presentation
3

### Contribution
2
