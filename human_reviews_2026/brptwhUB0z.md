# Background Blurring Matters: Improving Visual Grounding by Merging Text-Irrelevant Tokens

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Visual grounding (VG) aims to precisely localize the object in input images based on its natural language descriptions. Most recently proposed methods deal with this task with transformer-based architectures that can inject the textual information into the visual features. However, due to the image tokenlization procedure, there will be a large amount of image tokens located in text-irrelevant background areas. These tokens can introduce noise into the attention calculation, thus reducing the significance of foreground object tokens and ultimately affecting the effectiveness of these methods. To this end, we propose a novel Token Blurring (ToB) module, which dynamically merges image tokens based on the pair-wise visual similarity between them and their textual relevance with input expressions. By reducing the number of text-irrelevant background tokens and preserving the density of text-referred ones, ToB can improve both model effectiveness and efficiency in solving VG tasks. Experiments on RefCOCO, RefCOCO+, and RefCOCOg show that a transformer-based model equipped with our ToB module yield better results while reducing computational overhead compared to various VG methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Token Blurring (ToB), a novel module for visual grounding (VG) that reduces computational overhead and improves accuracy by merging text-irrelevant background tokens. ToB uses a language-guided merging strategy that considers both visual similarity and textual relevance, making it more effective than prior token merging methods. Integrated into a transformer-based model (DINOv2-B + BERT-B), ToB improves grounding performance across RefCOCO, RefCOCO+, and RefCOCOg datasets, and generalizes well to other VG architectures like CLIP-VG and SimVG.

### Strengths
1. First to jointly use visual similarity and textual relevance for token merging in VG.
2. Unlike ToMe or ToE, ToB explicitly avoids merging foreground tokens, preserving object details.
3. Extensive experiments across 3 datasets and 3 model backbones.
4. Plug-and-play integration into existing models (CLIP-VG, SimVG) shows consistent gains (up to +7.74% on RefCOCO+ testB).

### Weaknesses
1. Token merging (ToMe, ToE) and task-aware pruning (LAPS, MustDrop) already exist. ToB’s core idea is incremental, not radical.
2. Fps improvement is < 1 frame (20.8 → 21.57). Not a major speedup.

### Questions
See Weakness

### Soundness
3

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
5

### Summary
This paper studies visual grounding. The authors point out that current transformer-based VG models are affected by many text-irrelevant background tokens, which not only introduce noise in attention but also increase computational cost. To solve this, they propose Token Blurring (ToB), a module that merges image tokens based on both visual similarity and textual relevance. Different from previous token merging methods like ToMe that only consider visual similarity, ToB assigns text-aware weights and merges visually similar but text-irrelevant tokens while keeping important foreground ones. When added to models such as CLIP-VG and a DINOv2-B/BERT-B baseline, ToB consistently improves accuracy on RefCOCO, RefCOCO+, and RefCOCOg with faster inference.

### Strengths
1. Motivation is clear. The paper clearly identifies an issue with redundant background tokens and proposes a neat and intuitive fix. Making token merging text-aware is a simple but effective idea.

2. It's a plug-and-play module. ToB works across multiple backbones and datasets, showing that it is not tied to a single architecture. I think this is quite a practical contribution. But there's also a concern regarding this perspective. I articulate it in the weakness.

3. The authors test many variants and provide clear visualizations, which help me understand how ToB preserves important foreground regions compared to baselines like ToMe.

### Weaknesses
1. The paper misses a lot of recent works on Visual Grounding. Therefore, the related work section is seriously incomprehensive. Tables should cite and compare SegVG (ECCV'24), AttBalance (ACM-MM), ExpVG, etc. Those are all visual grounding methods without comparison and discussion in this paper.

2. Textual relevance computation seems oversimplified. In Eq. (3), you just take the average correlation between image and language tokens. I wonder if this averaging loses fine-grained alignment. Maybe you can show visualizations of the learned weights or try different aggregation methods (max or attention-weighted pooling) to see if it helps?

3. Performance gain inconsistency is not well explained. For CLIP-VG, the improvement is +7.74% on RefCOCO+ testB but only +0.71% on RefCOCO testA. Also, the gain is much smaller on SimVG, which uses stronger pretrained features. Does ToB have diminishing returns when the base model already encodes spatial context well?

4. Even though the paper claims they are plug-and-play, however, it is not clear and shown how to involve this method to MLLMs for visual grounding. Considering that MLLMs dominate in all the VL tasks, e.g. InternVL series, QwenVL series, Seed1.5VL, etc, it is necessary to clarify the motivation on building this plug-and-play module only compatible to specialist models.

### Questions
Please refer to my questions raised in each weakness.

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
This paper presents Token Blurring (ToB), a plug-and-play module designed to improve visual grounding by dynamically merging image tokens that are visually redundant and text-irrelevant. ToB first computes pair-wise visual similarity and text-based relevance for image tokens, then selects the top-r background-like token pairs to merge via a bipartite matching scheme. he method is simple, model-agnostic, and effective in improving grounding performance, though its claimed efficiency benefits are not empirically validated in the paper.

### Strengths
(1)ToB can be seamlessly inserted into existing visual grounding pipelines without modifying model architectures or requiring additional supervision, making it practical and widely applicable.

(2)Unlike prior token merging approaches that rely solely on visual similarity, ToB incorporates textual relevance into the merging decision. This dual-modality criterion is conceptually novel and well-motivated for visual grounding tasks.

(3)ToB demonstrates consistent performance gains when applied to weak (CLIP-VG), medium-sized (DINOv2-based baseline), and strong models (SimVG with BEiT-3). This cross-model effectiveness indicates strong generalizability.

### Weaknesses
(1)The authors highlight efficiency improvement as a key advantage of ToB, but the paper does not provide any quantitative experiments (e.g., inference speed, FLOPs reduction, or memory savings) to support this claim. Since the merging process itself adds computation (pair-wise similarity, textual relevance, ranking), it is unclear whether the overall pipeline is actually faster.

(2)The method assumes that background tokens are visually similar and text-irrelevant, which does not hold in many realistic visual grounding scenarios involving multiple similar objects or cluttered scenes.

(3)W_i is computed through AvgPool of text–vision correlations, yielding only a scalar per token. This severely limits the ability to distinguish foreground from background, making the merging decision potentially unreliable.

(4)The A/B splitting mechanism is heuristic and lacks theoretical or empirical justification. This operation may disrupt spatial locality and lead to unstable token pairing.

(5)Token averaging destroys spatial and boundary information critical for VG tasks, but the paper does not analyze its effect on bounding box regression or small-object localization.

(6)The merging policy is rule-based rather than learned, raising concerns about adaptability across diverse scenes.

### Questions
(1)Absence of efficiency evaluation despite claimed benefits.

(2)Potential loss of spatial and structural information.

(3)Non-learnable, heuristic merging policy.

### Soundness
3

### Presentation
3

### Contribution
2
