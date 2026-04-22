# Trust but Verify: Adaptive Conditioning for Reference-Based Diffusion Super-Resolution via Implicit Reference Correlation Modeling

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Recent works have explored reference-based super-resolution (RefSR) to mitigate hallucinations in diffusion-based image restoration. A key challenge is that real-world degradations make correspondences between low-quality (LQ) inputs and reference (Ref) images unreliable, requiring adaptive control of reference usage. Existing methods either ignore LQ–Ref correlations or rely on brittle explicit matching, leading to over-reliance on misleading references or under-utilization of valuable cues. To address this, we propose Ada-RefSR, a single-step diffusion framework guided by a "Trust but Verify " principle: reference information is leveraged when reliable and suppressed otherwise. Its core component, Adaptive Implicit Correlation Gating (AICG), employs learnable summary tokens to distill dominant reference patterns and capture implicit correlations with LQ features. Integrated into the attention backbone, AICG provides lightweight, adaptive regulation of reference guidance, serving as a built-in safeguard against erroneous fusion. Experiments on multiple datasets demonstrate that Ada-RefSR achieves a strong balance of fidelity, naturalness, and efficiency, while remaining robust under varying reference alignment. Code and models are available at https://github.com/vivoCameraResearch/AdaRefSR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Ada-RefSR, a diffusion-based model for Reference-based image super-resolution. The core idea of this paper lies in two terms: the fidelity and generalization. It proposes AICG to adaptively balance intrinsic SR fidelity with reference-guided enhancement, mitigating hallucinations. AICG enables Ada-RefSR to generalize RefSR beyond narrow domains to diverse scenarios, maintaining robustness under varying reference alignment. Further, built on one-step diffusion model, its inference is efficient compared to other multi-step diffusion model.

### Strengths
1.	The paper proposes AICG designed for RefSR task. Ada-RefSR injects LQ features directly using a residual connection besides the reference feature selection. It allows the model to preserve the prior knowledge. Also, a gating mechanism is applied in the reference feature attention components to adaptively select useful information from the reference image.
2.	The model achieves SOTA performance with efficient one-step diffusion model.
3.	The idea is straightforward and easy to follow.

### Weaknesses
1.	The novelty is limited, since most of the model design and idea like LQ feature residual connection is straightforward and has been proposed in classic non-diffusion methods. The paper should discuss the novelty specifically for Ref-SR task.
2.	Though claiming efficiency as one of the contributions, the paper provides insufficient discussions and experiments on efficiency and model size.
3.	The visualizations are mostly derived from animal images, are there results and visual comparisons on more complex textures like text or some artificial objects to better show the generalization of this model?

### Questions
Refering to the weaknesses. Clarify the contributions specfically for Ref-SR. The visualization of some complex textures are also important to reveal the generalization of the proposed method.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents Ada-RefSR, a single-step diffusion framework for reference-based super-resolution that follows a “Trust but Verify” strategy. The method first injects reference features through a reference-attention module and then uses an Adaptive Implicit Correlation Gating (AICG) mechanism to adaptively filter mismatched reference cues. Experimental results on multiple datasets show improved fidelity, perceptual quality, and efficiency compared with recent diffusion and RefSR methods.

### Strengths
- The paper clearly defines the problem of over- or under-reliance on reference images in diffusion-based SR and provides a coherent conceptual framework to address it. 

- The proposed AICG module is lightweight and easily pluggable into existing backbones, enabling adaptive control of reference guidance without additional supervision. 

- This paper provides clear visual evidence, such as attention maps, gating masks, and token visualizations, that help interpret the mechanism’s behavior.

### Weaknesses
- The main limitation lies in the novelty boundary of AICG. Its design using learnable tokens for implicit correlation modeling is conceptually close to DETR-style query or prototype aggregation, and the paper does not clearly explain how it fundamentally differs from those approaches. 

- The paper also lacks comparisons with strong face-specific RefSR methods, which would help validate the generalization of Ada-RefSR in specialized domains. 

- There is no detailed analysis of different reference scenarios, such as varying reference quality, domain gaps, or alignment errors, which would strengthen the claim of robustness under diverse reference conditions.

### Questions
See weaknesses.

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
5

### Summary
The paper proposes Ada-RefSR, a single-step reference-based diffusion framework for super-resolution. The method addresses the challenge of establishing reliable correspondences between low-quality (LQ) inputs and reference images, which is particularly difficult under severe degradations. To mitigate issues of over- or under-reliance on references seen in existing weighting-based fusion strategies, the authors adopt a “Trust but Verify” paradigm: the model leverages references when they align and suppresses them otherwise. The core component, Adaptive Implicit Correlation Gating (AICG), uses learnable summary tokens to capture dominant reference patterns and implicit correlations with LQ features. Integrated into the attention backbone, AICG adaptively regulates reference guidance and prevents erroneous feature fusion. Experiments on multiple datasets show that Ada-RefSR achieves a good balance of fidelity, naturalness, and efficiency, while maintaining robustness under varying reference alignment.

### Strengths
1. The proposed “Trust but Verify” perspective is an interesting and promising approach. 

2. The authors conduct extensive experiments to validate the effectiveness of their method.

### Weaknesses
1. The key innovation of this paper is the Adaptive Implicit Correlation Gating (AICG) mechanism. Currently, the introduction lacks a critical figure illustrating the authors’ approach and the core differences compared with prior methods, which is highly important.

2. The authors need to explain why their method does not perform well on the WRSR dataset.

### Questions
No

### Soundness
2

### Presentation
3

### Contribution
3
