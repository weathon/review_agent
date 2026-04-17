# Uni-directional Blending: Learning Robust Representations for Few-shot Action Recognition with Frame-level Ambiguities

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Leveraging vision-language models (VLMs) for few-shot action recognition has shown promising results, yet direct image-text alignment methods, such as CLIP, encounter significant challenges in video domains due to frame-level ambiguities. Videos frequently include irrelevant and redundant frames, leading to intra-class ambiguity from non-essential content within the same action and inter-class ambiguity from visually overlapping elements across classes. These ambiguities hinder the learning of distinctive prototypes and robust semantic representations.

To overcome this, we introduce Uni-FSAR, a novel framework that employs uni-directional blending to selectively integrate relevant frames, preventing contamination of prototypes by irrelevant visual noise. Additionally, a learnable text query (LTQ) bridges the semantic gap between visual features and class labels, enhancing representation alignment. Furthermore, our LTQ-based Semantic Bridging Loss promotes focus on informative frames through similarity-based gradient propagation, mitigating inter-class overlap and fostering more generalizable representations.

Extensive experiments, including cross-dataset evaluations, demonstrate that Uni-FSAR achieves superior robustness in handling frame-level ambiguities compared to prior works. Quantitatively and qualitatively, our method outperforms the state-of-the-art by an average of 2.34% across benchmarks, with a notable 6.5% top-1 accuracy gain on HMDB51, where ambiguities are most pronounced.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Uni-FSAR, which introduces uni-directional blending with a learnable text query to mitigate frame-level noise in few-shot action recognition. A Semantic Bridging Loss selectively optimizes only the most relevant frames, and OTAM-based prototype alignment is used for temporal matching.

### Strengths
- S1: The motivation behind Uni-FSAR is well-grounded. Frame-level ambiguity represents a critical and realistic challenge in video-based few-shot action recognition.
- S2: Consistent and significant performance gains are reported across multiple standard benchmarks, indicating strong robustness.
- S3: The motivation is effectively conveyed through well-designed and intuitive visual illustrations.

### Weaknesses
- W1: The LTQ and uni-directional blending appear to be minor variations of existing cross-attention or masked attention mechanisms in BLIP-2 and text-guided prototype learning.
- W2: The proposed Top-K selection with a contrastive objective is quite similar to hard attention or selective loss used in prior video FSAR works, such as [1].
- W3: SSv2 5way-1shot results in Tab. 2 are weaker or marginal vs SOTA → contradicts "generalizable under ambiguity" claim. 
- W4: While Uni-FSAR aims to mitigate frame-level ambiguity by selectively emphasizing the Top-K frames most aligned with text semantics, it remains unclear whether such single-frame–focused selection sufficiently preserves motion cues that are critical for temporal understanding, especially in datasets like SSv2 where the action is defined by subtle frame-to-frame changes rather than static appearances. By suppressing non-selected frames entirely during optimization, the method may risk discarding essential temporal evidence and oversimplifying the underlying action dynamics that require multi-frame context to recognize.
- W5: Figures lack sufficient explanation (e.g., Figure 2 symbols).

[1] Task-adaptive Spatial-Temporal Video Sampler for Few-shot Action Recognition

### Questions
- Q1: While an ablation is provided for K=3 in the Top-K strategy, a more detailed explanation or empirical reasoning could further support this selection.
- Q2: It is unclear why LTQ is necessary when CLIP text encoder already embeds class semantics; what exact semantic gap is being "bridged"?
- Q3:  Prototype formation relies on OTAM, but the paper does not clearly explain whether and how the temporal alignment interacts with the proposed uni-directional blending strategy.

### Soundness
2

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
4

### Summary
This paper introduces the Uni-FSAR framework. The authors identify two types of ambiguities: (1) intra-class ambiguity from irrelevant frames within the same action class, and (2) inter-class ambiguity from redundant frames shared across classes. To tackle these challenges, three core components are introduced: uni-directional blending strategy, Learnable Text Query, and LTQ-based Semantic Bridging Loss.

### Strengths
1) Problem formulation sharply defines intra- and inter-class frame-level ambiguities with quantitative evidence, providing a compelling, data-driven motivation.
2) Extensive experiments demonstrate SOTA performance across multiple benchmarks, with particularly impressive gains on noisy datasets and strong cross-dataset generalization capabilities.

### Weaknesses
1) The paper compares Uni-FSAR using BLIPv2 ViT-L/14 (508.21M total parameters) against CLIP-FSAR using ViT-B/16 (89.34M parameters). This 5.7× difference in total parameters and the substantially larger vision encoder make it impossible to isolate whether the reported improvements stem from the proposed methodological innovations or simply from using a more powerful backbone. (Tables 1-2)
2) The proposed method directly applies the Q-Former with 32 learnable visual queries originally designed for static images to each video frame. While this reuse simplifies integration, it remains unclear whether these static-image queries are sufficient to capture temporal dynamics or motion-related cues that are crucial for video-based action understanding.
3) The paper's core components—uni-directional attention masking, learnable text queries, and Top-K frame selection—are well-established techniques borrowed from existing work.

### Questions
Please refer to weakness.

### Soundness
2

### Presentation
2

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
The Uni-FSAR framework introduces a Uni-Directional Blending (UDB) mechanism such that visual queries (VQs) can attend to a learnable text query (LTQ), but not vice versa. Here Top-K most relevant frames are aligned with the LTQ to suppress noisy or irrelevant frames and combined with a Learnable Text Query-based Semantic Bridging (LSB) loss,  
The model is further integrated into an OTAM-based pipeline for few-shot action recognition. While the proposed Uni-FSAR framework presents an interesting architecture for few-shot video understanding, there are some conceptual and methodological weaknesses which limit its claim to effectively handle frame-level ambiguity.

### Strengths
1. The use of a learnable text query avoids reliance on handcrafted textual prompts.
2. The LSB loss encourages semantic selectivity and yields better interpretability through attention visualization.

### Weaknesses
1. Even after Top-K selection in the loss function, all frame embeddings are averaged equally to form the video prototype. This contradicts the goal of mitigating frame-level redundancy — the prototype is still influenced by uninformative frames.
2. The choice of Top-K  and the loss weighting factor α  seems to be dataset-specific and tuned manually. This undermines claims of generalization.
3. Despite claiming to address frame-level ambiguity, the paper introduces no mechanism  that measures inter-frame differences (e.g., variance). The Top-K selection in the LSB loss merely filters frames based on similarity to the LTQ, but it does not take into account the embedding variance between frames within a video.

### Questions
I am wondering  how without ever measuring or minimizing frame-level differences in the learning objective it is possible to  “handle frame-level ambiguities”, maybe  I am missing something,  then kindly clarify!

### Soundness
2

### Presentation
2

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
This paper proposes a Uni-FSAR framework that aims to improve prototype construction under frame-level ambiguity by selectively utilizing semantically relevant frame information.
The authors design a uni-directional blending strategy to prevent irrelevant frames from contaminating class prototypes and introduce a Learnable Text Query (LTQ) module to achieve semantic alignment between visual features and class labels.

### Strengths
1.The motivation is clear . 
2.The paper is very clearly written. The methodology is presented in a logical and accessible manner, with well-organized sections, clear mathematical formulations, and informative visualizations that make the technical design easy to follow.

### Weaknesses
1.The main comparison is made against CLIP-FSAR, yet all experiments in this paper adopt a more powerful BLIP backbone, making it difficult to disentangle whether the performance gain stems from the stronger backbone or from the proposed method itself.

2.The paper lacks a clear ablation study on the Uni-directional Blending mechanism.
A comparison among uni-directional, bi-directional, and reverse-directional attention designs would be necessary to verify that the observed improvements truly correspond to the claimed motivation.

### Questions
The motivation of this work is meaningful, and the results indeed demonstrate improved performance.
However, it remains unclear why the uni-directional blending strategy can selectively integrate relevant frames while preventing prototype contamination by irrelevant visual noise.
Please clarify how the proposed mechanism theoretically or empirically achieves this selective filtering, and explicitly connect the motivation to the method’s operational design.

### Soundness
2

### Presentation
3

### Contribution
3
