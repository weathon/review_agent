# Post-training quantization of vision encoders needs prefixing registers

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 4

## Abstract
Transformer-based vision encoders---such as CLIP---are central to multimodal intelligence, powering applications from autonomous web agents to robotic control. Since these applications often demand real-time processing of massive visual data, reducing the inference cost of vision encoders is critical. Post-training quantization offers a practical path, but remains challenging even at 8-bit precision due to massive-scale activations (i.e., outliers). In this work, we propose \textit{RegCache}, a training-free algorithm to mitigate outliers in vision encoders, enabling quantization with significantly smaller accuracy drops. The proposed RegCache introduces outlier-prone yet semantically meaningless prefix tokens to the target vision encoder, which prevents other tokens from having outliers. Notably, we observe that outliers in vision encoders behave differently from those in language models, motivating two technical innovations: middle-layer prefixing and token deletion. Experiments show that our method consistently improves the accuracy of quantized models across both text-supervised and self-supervised vision encoders.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes RegCache, a method to improve post-training quantization of vision encoders by using prefixing registers to handle activation outliers. It claims to be training-free and shows improved results across various models and quantization techniques.

### Strengths
- Interesting observation about middle-layer outlier emergence in vision encoders

- Clear empirical demonstration of performance improvements

- Comprehensive evaluation across multiple vision encoders (CLIP, DINOv2, SigLIP, etc.) and quantization methods

### Weaknesses
- The method tunes the number of tokens to delete (k̃) based on downstream performance.
This process constitutes a form of task-specific, validation-based tuning that is computationally expensive and requires labeled data from the target domain. This violates the standard premise of PTQ, which is designed to be a lightweight process that does not require access to the final task's labels or metric. 

- The approach inserts external tokens as pre-computed KV at middle layers and removes tokens deemed sinks. This goes beyond calibration as it alters the attention context and risks semantic/architectural drift.

### Questions
- What is the computational cost of the entire RegCache pipeline, including the candidate curation and hyperparameter search? How does this cost compare to simply performing a brief Quantization-Aware Training (QAT) round, which would likely be more effective?

- The paper compares against other PTQ methods. Were these baselines also allowed to perform a similar level of task-specific hyperparameter tuning on the ImageNet validation set? If not, the comparison is not fair.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
RegCache is a training-free method to reduce inference cost in transformer-based vision encoders (e.g., CLIP) by mitigating activation outliers that hinder 8-bit post-training quantization. It injects semantically meaningless, outlier-prone prefix tokens to prevent other tokens from producing outliers and, based on the distinct behavior of vision encoders vs. language models, introduces two key techniques: middle-layer prefixing and token deletion. This enables quantization with significantly smaller accuracy drops and consistently improves performance across both text-supervised and self-supervised encoders.

### Strengths
1. This paper is well organized and easy to follow.
2. The method is plug-and-play.
3. The authors provide code for reproducing.

### Weaknesses
2. There are lots of works that propose to address the issue of outliers for LLM (which can also apply to ViT) and ViT. However, this paper does not compare the proposed method with them. For example, QuaRot (NeurIPS 2024), https://openreview.net/pdf?id=Uh5XN9d2J4 (ICML 2024), etc.
3. This paper does not implement their method on top of the SOTA quantization methods to show the effectiveness of their approaches. For example, FIMA-Q (CVPR 2025) and APHQ-ViT (CVPR 2025).
4. The introduced overhead of this method for real-time inference is not being discussed.
5. The idea that introducing additional tokens to mitigate outliers and sinks is not novel (see https://arxiv.org/abs/2402.17762 and https://arxiv.org/abs/2410.05265).

### Questions
N/A

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
4

### Summary
This paper proposes the RegCache algorithm to address the outlier problem in post-training quantization (PTQ) of vision encoders. Specifically, RegCache introduces outlier-prone yet semantically meaningless prefix tokens to the target vision encoder, which prevents other tokens from having outliers. This design ultimately reduces the quantization error of PTQ.

### Strengths
The analysis of outliers in PTQ quantization is interesting, and its approach to reducing model outliers is innovative, which merits further research.

### Weaknesses
The paper proposes several key hypotheses and observations that underpin its RegCache method, yet these claims undermine the work’s generality and reliability. They are not sufficiently supported by either extensive experimental validation (e.g., across more diverse image domains or model architectures) or in-depth theoretical analysis—leaving their applicability to vision encoder post-training quantization (PTQ) insufficiently verified.

### Questions
1. The paper lacks theoretical analysis for key observations—specifically, it fails to provide a theoretical explanation for the intrinsic reason behind the "cross-image similarity of middle-layer outliers". For instance, an analysis from perspectives such as the attention mechanism or LayerNorm is absent.  
2. The paper has issues with its figures. The left subfigure of Figure 1 does not provide sufficiently detailed information. The right subfigure of Figure 1 lacks adequate descriptions of experimental setup details. Figure 2 is deficient in introductions to key terms.  
3. Although the authors cite numerous references, the paper still needs to explain in the text why sink tokens can mitigate outliers.  
4. Provide a detailed comparison between RegCache and recent outlier quantization methods for vision encoders.  
5. Supplement quantitative experiments to analyze the proportion of quantization error contributed by outliers, so as to further support the urgency of the research motivation.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposed a novel training-free algorithm to mitigate outliers in post-training quantization of vision encoders through pruning semantically meaningless prefix tokens. Experiments reveal the effectiveness of the proposed methods.

### Strengths
1. The discovery of semantically meaningless prefix tokens is enlightening. 
2. The motivations and corresponding methods are clearly revealed and discussed.

### Weaknesses
Major: 
1. In Line 96-98, the prefixed tokens are only applied for middle-to-final layers. Is there any experimental comparison and corresponding discussion about how the layer settings effect the quantization process and model performance?
2. In section 3.1, why only layer-wise sensitivities are discussed, since the sensitivities and outliers are in transformers are about channel level or even token level as discussed in previous arts. 
3. How to deal with vision encoders like NaViT, which process images as any arbitrary resolution. Thus in these vision encoder pipeline, the prefix can be varied and the quantization strategy proposed in this paper may not perform well. 


Minor: 
1. It would be better to add legends in Figure 2 for more clarity. 
2. Maybe section 3.1 and 3.3 can be combined as one section about how layer index affect the outlier and quantization performance, for better writing logic.

### Questions
See weaknesses. My major concern is the third point in the "Major" part.

### Soundness
2

### Presentation
2

### Contribution
2
