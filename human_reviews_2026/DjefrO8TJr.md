# Sparse CLIP: Co-Optimizing Interpretability and Performance in Contrastive Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Contrastive Language-Image Pre-training (CLIP) has become a cornerstone in vision-language representation learning, powering diverse downstream tasks and serving as the default vision backbone in multimodal large language models (MLLMs). Despite its success, CLIP's dense and opaque latent representations pose significant interpretability challenges. A common assumption is that interpretability and performance are in tension: enforcing sparsity during training degrades accuracy, motivating recent post-hoc approaches such as Sparse Autoencoders (SAEs). However, these post-hoc approaches often suffer from degraded downstream performance and loss of CLIP's inherent multimodal capabilities, with most learned features remaining unimodal.

We propose a simple yet effective approach that integrates sparsity directly into CLIP training, yielding representations that are both interpretable and performant. Compared to SAEs, our Sparse CLIP representations preserve strong downstream task performance, achieve superior interpretability, and retain multimodal capabilities. We show that multimodal sparse features enable straightforward semantic concept alignment and reveal training dynamics of how cross-modal knowledge emerges. Finally, as a proof of concept, we train a vision-language model on sparse CLIP representations that enables interpretable, vision-based steering capabilities. Our findings challenge conventional wisdom that interpretability requires sacrificing accuracy and demonstrate that interpretability and performance can be co-optimized, offering a promising design principle for future models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper challenges the prevailing assumption that interpretability must come at the cost of performance in representation learning. The authors propose Sparse CLIP, a method that seamlessly co-optimizes for both objectives by integrating sparsity directly into the Contrastive Language-Image Pre-training (CLIP) framework. The approach is notably simple, requiring only two key modifications to the standard architecture: a significant expansion of the final projection layer's dimensionality and the introduction of a ReLU activation function. This combination effectively induces a sparse, high-dimensional representation space without altering the core contrastive learning objective.

Empirical results demonstrate that Sparse CLIP not only matches but in some cases surpasses the performance of its dense counterparts on zero-shot classification benchmarks, while achieving extreme sparsity. Crucially, the learned features are natively multimodal, activating for semantically aligned concepts across both image and text modalities. This enables direct concept-level interpretability, allowing features to be named and analyzed based on their top-activating words and images. The paper further reveals intriguing training dynamics, tracing how multimodal concepts emerge and evolve, and showcases the practical utility of these interpretable features through vision-based steering in a downstream Vision-Language Model (VLM), enabling controlled generation and security filtering.

### Strengths
1. Simplicity and Effectiveness: The method simply increases the dimensionality of the final projection layer and adds an activation of ReLU. The result shows that it achieves interpretability and better performance.
2. Refutation of a prevailing assumption: It provides a good counterexample for "training-time sparsity compromises accuracy".
3. High application potential: The paper demonstrates performance enhancements in downstream applications, as well as new application scenarios such as steered generation and security filtering.

### Weaknesses
1. Theoretical innovation is limited: One of the key techniques (ReLU activation) was proposed by NCL. It lacks a theoretical analysis of dimensionality expansion. The emphasis is more on applications in multimodal setting.
2. GPU memory and model parameters overhead: The model significantly increases the dimensionality of the final projection layer, resulting in substantially more parameters and the additional GPU memory required for training. Due to limitations of the GPU memory, it is impossible to further research the impact of dimensionality.
3. Limited evolution for downstream task: Only four fine-tuning tasks are reported, which is insufficient to robustly support the claim that the method "maintains strong performance on downstream tasks" across a broad range of applications.

### Questions
1. In 2.3, the distinction between the training settings for Sparse and Sparse+ is not clarified. Do they merely employ different logit scale caps? If so, based on the conclusions from small scale training, is the Sparse+ model trained using a logit scale cap of 40?
2. Model Scale: Since the final projection layer is significantly enlarged, could the performance gain be attributed primarily to the increase in parameters rather than the sparsity mechanism? It is necessary to supplement the Sparse CLIP model with a set of ablation experiments—specifically, removing the ReLU activation function and dimension expansion individually.
3. Association between text and features: The paper explains that one should first establish a vocabulary list, then identify the corresponding relationships. This demonstrates dependence on vocabulary quality for concept labeling.
    1. Can the model interpret multi-word phrases, or is it limited to single tokens?
    2. How is it verified that the top-activated word accurately captures the feature's semantic role?
    3. What happens if the underlying concept is not contained in the vocabulary?
4. Steering Sensitivity: In the "dog"→"cat" steering example, the "cat" feature is set to 2.0.
    1. How was this value chosen? Would other values (e.g., 1.0 or 10.0) lead to unpredictable generation?
    2. Is there a systematic method for determining safe and effective intervention magnitudes?

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
4

### Summary
This work addresses the interpretability limitations of CLIP’s dense multimodal representations. The authors propose integrating sparsity directly into CLIP training to create “Sparse CLIP features” that remain highly performant on downstream tasks while becoming significantly more interpretable. Unlike post-hoc sparse autoencoder methods, this approach preserves CLIP’s inherent multimodal alignment and reveals transparent feature evolution and concept emergence during training. Experiments further demonstrate practical benefits by enabling interpretable visual-language control in downstream applications. The results challenge the common belief that interpretability must come at the cost of performance, offering a promising direction for building multimodal models that are both accurate and understandable.

### Strengths
1. This paper proposes the SPARSE CLIP method, which effectively combines the strengths of both CLIP and SAE models, ensuring interpretability and performance in contrastive learning.
2. The paper provides a solid analysis of the interpretability achieved by combining large models with SPARSE CLIP. The cases presented in paper demonstrate the strong interpretability of SPARSE CLIP as well as its powerful application potential.

### Weaknesses
The proposed SPARSE CLIP method exhibits strong interpretability, but its performance drops on many benchmarks compared to the original ViT-based CLIP baseline. I hypothesize that this is due to CLIP’s reliance on maintaining rich and continuous directional information in the embedding space. The discontinuity introduced by ReLU indeed produces highly sparse and interpretable features, but it also collapses many feature dimensions, resulting in sparse vectors that weaken cross-modal alignment and semantic expressiveness. A potential solution would be to decouple the interpretable sparse features from the output by employing an independent sparse decoder.

### Questions
The paper presents many interpretability examples and analyzes the distribution of Sparse CLIP features, yet it lacks more fine-grained analysis that connects specific feature activations to corresponding image regions or pixels. For instance, applying masking-based methods could enable more detailed ablations to identify which parts of the image activate particular features. I hope the authors can include some cases demonstrating this aspect.

If the authors provide reasonable and satisfactory responses to the Weaknesses and Questions, I would consider increasing the score.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes training a vision-language encoder (CLIP) with a drastically expanded embedding dimension (in this case, by a factor of 72). The goal is to learn a sparse representation allowing for interpretability and steering. Experimental results show that Sparse CLIP is able to perform well in zero-shot classification on ImageNet, as well as its representation can be interpreted through concept discovery.

### Strengths
This paper's strengths lie in the novelty and simplicity of the method:
1. The idea to train a CLIP with an inherently sparse representation is very interesting and can lead to significant progress in research on interpretability. It is simple and could be easily translated to other bi-modal encoders trained with the contrastive loss.
2. I appreciate the paper's flow, especially the 'lessons learned' given in Section 2.2, articulating the research process progressing into Section 2.3.

### Weaknesses
Overall, this work is in an early stage, requiring revision and extension to be considered for publication:
1. Regarding soundness, the experiments are limited (see questions below). Contribution 3 (implementing interpretable
vision-based steering in vision-language models) executed in Section 4 is overstated significantly: the two examples shown in Table 2 cannot reliably demonstrate real-world applicability.
2. The paper requires proofreading (see feedback below; I just stopped listing at some point). Several elements in the paper appear to be missing (see questions), and I do not believe that mentioning additional content exists in the Appendix and then incorporating it during the rebuttal would be an acceptable practice.
3. Furthermore, the comparison with the state-of-the-art is missing. Section 3 could follow experimental evaluation from the omitted related work on training SAEs for CLIP [a,b] instead of comparing to a TopK SAE with a single $k$ value (not to mention that the Prisma SAE uses a different baseline model).
4. The paper should include code to reproduce the results. 

[a] Sparse autoencoders reveal selective remapping of visual concepts during adaptation. ICLR 2025

[b] Interpreting CLIP with hierarchical sparse autoencoders. ICML 2025

### Questions
1. The sentence in L200-201 "After conducting comprehensive ablation studies (additional results in Appendix A.3)," is wrong. There are no "comprehensive ablation studies" in Appendix A.3. It only shows Figure 9 with a single experiment on "ViT-L/14 Sparse+".
2. Appendix A.2 mentions an "interactive web visualization to access visualizations" but the demo is empty, i.e. says "Cool Sparse CLIP visualizations".
3. Why did you choose the dimension expansion factor of 72? Comprehensive ablations are essential when proposing a method with such a critical parameter. Why does the Appendix say "32x dimension expansion"?
4. Table 1 should include results for ViT-B/32 Sparse.

**Other feedback**:
- Use `\citep` correctly instead of `\citet` (many such cases e.g. in L153-157).
- L137: add "equation" to "1"
- L149: typo in "architectures"
- Figure 1: 
    - Be more specific in the caption, e.g. add the information about "OpenCLIP (ViT-B/32) trained on MetaCLIP".
    - Improve colors in (c); use an actual gradient, e.g. yellow-orange-red-purple. 
    - Graphs are pixelated; improve their quality, e.g. export them to PDF.
- Figure 9 (Appendix A.3): It is completely unclear what "ViT-L/14 Sparse+" or this result means when linking here from L200.
- L206: "After demonstrating proof of concept on smaller models and datasets," reads like an overstatement, because it was a single model (ViT-B/32) and data subset.
- L212: typo in "(0.66Following"
- Table 1/L232: Again, it is unclear what does "+" in "ViT-L/14 Sparse+" mean.
- L244: Again, "Having achieved satisfactory performance and sparsity" reads like an overstatement, because the model was trained with a single hyperparameter setting and evaluated only on the zero-shot classification task. We know little regarding the generalizability of the approach.
- Table 3 (Appendix A.1.2) should be a part of main text.
- L420: What does "Not on existing model" mean?

### Soundness
2

### Presentation
1

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Sparse CLIP, a experimental modification and integration to the standard CLIP framework that incorporates sparsity during training.

### Strengths
1. The paper incorporates sparsity into training, it achieves both interpretability and high performance, offering a compelling design.

2. Comprehensive experiments demonstrate that Sparse CLIP performs comparably to dense CLIP on zero-shot classification tasks and surpasses post-hoc Sparse Autoencoders (SAEs) in interpretability.

3. Applications in Vision-Language Models: This paper presents a practical demonstration of Sparse CLIP’s utility and relevance for downstream tasks.

In summary, this paper offers intriguing experimental results and new insights into sparse training than providing a clear technical contribution or a robust methodological advancement.

### Weaknesses
1. While the findings are interesting, the proposed method is totally established on the existing works, making it relatively simple and lacks significant novelty.

2. Unclear generalizability: While Sparse CLIP performs well on zero-shot classification benchmarks, it is unclear how well the method generalizes to downstream tasks as the other CLIP models, such as object detection, segmentation, or open-vocabulary retrieval

### Questions
1. Introduce quantitative interpretability metrics beyond classification metrics.

2. Hard case testing and failure case analysis. For example, does Sparse CLIP produce meaningful features although it misclassifies an object? What is the performance of Sparse CLIP on more complex datasets?

3. Clarify contributions.

### Soundness
3

### Presentation
3

### Contribution
2
