# Enhancing Visual Token Representations for Video Large Language Models via Training-free Spatial-Temporal Pooling and Gridding

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
Recent advances in Multimodal Large Language Models (MLLMs) have significantly advanced video understanding tasks, yet challenges remain in efficiently compressing visual tokens while preserving spatiotemporal interactions. Existing methods, such as LLaVA family, utilize simplistic pooling or interpolation techniques that overlook the intricate dynamics of visual tokens. To bridge this gap, we propose ST-GridPool, a novel training-free visual token enhancement method designed specifically for Video LLMs. Our approach integrates Pyramid Temporal Gridding (PTG), which captures multi-grained spatiotemporal interactions through hierarchical temporal gridding, and Norm-based Spatial Pooling (NSP), which preserves high-information visual regions by leveraging the correlation between token norms and semantic richness. Extensive experiments on various benchmarks demonstrate that ST-GridPool consistently enhances performance of Video LLMs without requiring costly retraining. 
Our method offers an efficient and plug-and-play solution for improving visual token representations.
Our code is available in [https://github.com/bingjunluo/ST-GridPool](https://github.com/bingjunluo/ST-GridPool).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes ST-GridPool, a training-free method to enhance visual token representations for Video LLMs, addressing the challenge of efficient token compression while preserving spatiotemporal interactions. It integrates two core components: Pyramid Temporal Gridding (PTG) for multi-grained temporal feature capture and Norm-based Spatial Pooling (NSP) for preserving high-semantic regions via token norms. Evaluated on 6 benchmarks (e.g., VideoMME, MVBench) with LLaVA-OneVision-7B and LLaVA-Video-7B, ST-GridPool achieves consistent performance gains without retraining, while reducing inference cost.

### Strengths
This paper tackles a critical research problem about visual token enhancement for Video LLMs. The proposed ST-GridPool method is effective. It offers a training-free, plug-and-play framework that improves visual token representations without adding extra computational cost, a very appealing design choice.
The paper is well positioned within the literature and clearly explains how it differs from prior work. The motivation is convincing, the narrative flows nicely, and the writing is clear throughout. It’s an easy and enjoyable read.
Capturing multi-grained spatiotemporal interactions across varying temporal lengths and leveraging the positive correlation between token norms and semantic importance is a really interesting idea.
The results are solid, covering diverse tasks, different backbones, and a range of metrics and hyperparameters. The experimental section is thorough and leaves little doubt about the method’s effectiveness.

### Weaknesses
The paper demonstrates the overall computational advantages (inference time and memory) of ST-GridPool. However, it does not provide a detailed breakdown of the computational cost for the PTG and NSP components individually. Furthermore, the potential interaction between these two modules is not discussed, e.g. how PTG's multi-scale processing affects the norm distribution of the input to NSP.

The Pyramid Temporal Gridding (PTG) module generates a summary token for each temporal segment and then uses it to update the token of the last frame in that segment. This update operation seems potentially destructive, as it replaces the representation of an actual, observed frame with a synthetic, interpolated summary. This design choice is not clearly motivated over less destructive alternatives, e.g., appending the summary token.

The NSP component is explicitly designed to prioritize high-importance regions (i.e., salient objects) and suppress low-importance backgrounds. This strategy could be detrimental in tasks that require holistic scene understanding or reasoning about subtle interactions within the background regions, which are now actively suppressed.

### Questions
1. Given the model's clear overall efficiency gains, could you provide a more detailed computational cost breakdown contributed by the PTG and NSP modules individually?
2. Why do you choose to make the PTG summary token overwrite the last frame's token instead of a less destructive alternative like appending?
3. How does your method balance the trade-off between prioritizing high-importance regions and preserving potentially critical, low-importance backgrounds in practical video understanding tasks?

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
5

### Summary
This paper introduces a well-motivated and effective training-free method, ST-GridPool, for enhancing visual token representations in Video LLMs. The approach is simple yet intuitive, combining PTG and NSP to improve performance without retraining while also reducing computational overhead. The empirical validation is extensive, with strong ablation studies supporting the design choices.

### Strengths
1. Both components are well-justified and technically sound. PTG provides an intuitive hierarchical approach to model events at different time scales. NSP is motivated by both intuitive and quantitative analysis, which is a clever insight for preserving important information during compression.

2. A comprehensive evaluation is conducted, covering both long and general video understanding tasks. The method is tested across multiple backbone models, demonstrating its generalizability and effectiveness.

3. The paper is mostly well-written, clearly structured, and easy to follow, with helpful visualizations that aid in understanding the proposed approach.

### Weaknesses
1. The method for updating the token sequence in Pyramid Temporal Gridding involves overwriting the last frame token of a segment with a generated summary token. This specific design choice should be justified.

2. The computational analysis focuses on the net efficiency gains, but offers limited insight into the additional computational cost (e.g., latency, memory) introduced by the ST-GridPool module itself.

3. A deeper theoretical explanation for why token norms (specifically L_2 norm) reliably correlate with semantic importance in this context would be beneficial.

### Questions
Please see the weakness above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes ST-GridPool, a practical training-free method to enhance visual tokens for Video LLMs. The approach is technically sound, combining a hierarchical temporal gridding strategy (PTG) and a saliency-aware spatial pooling mechanism (NSP). The method demonstrates consistent performance improvements on LLaVA-family models across six benchmarks and compelling gains in computational efficiency. The empirical evaluation is extensive, including strong ablations and comparisons to other token reduction methods.

### Strengths
1.The proposed ST-GridPool method is training-free and plug-and-play, offering an efficient solution to enhance existing Video LLMs without the need for costly retraining. It demonstrates consistent performance improvements across various backbones on all six evaluated video understanding benchmarks, with notable gains in Long Video Understanding.\
2.The motivation behind Norm-based Spatial Pooling is well-founded, supported by a clear analysis that establishes a positive correlation between token norms and semantic saliency, contributing to the method's effectiveness.\
3.The empirical evaluation is comprehensive, featuring extensive ablation studies on individual components, key hyperparameters, and design choices such as pooling shape and gridding strategy.

### Weaknesses
1.The performance depends on several non-trivial hyperparameters, including the number of temporal levels, norm order, and temperature. Balancing these parameters effectively is crucial for their successful application in real-world scenarios.\
2.The paper does not clearly show whether PTG provides complementary benefits or duplicates functionality already present in recent Video LLMs.\
3.The NSP module relies heavily on an empirical correlation between token norm and semantic importance, yet offers limited empirical comparison or grounding.\
4.The experimental section primarily focuses on quantitative results but lacks deeper analytical discussion. The paper does not explore why the proposed ST-GridPool achieves varying degrees of improvement across different benchmarks or tasks.\
5.There are some typos in the paper. E.g., the percentage gains reported for "LLaVA-Video-7B + Ours" are identical to those reported for "LLaVA-OneVision-7B + Ours".

### Questions
1.Is it possible to explore an adaptive mechanism that automatically determine the optimal values of non-trivial hyperparameters in your method?\
2.How does the proposed PTG interact with or differ from the built-in temporal reasoning mechanisms already present in modern Video LLMs such as NVILA?\
3.Could other unsupervised indicators serve as complementary or alternative signals to the token norm?\
4.How to interpret the observed trend of a larger improvement for Long Video Understanding tasks compared to General Video Understanding tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ST-GridPool, a training-free visual token compression method tailored for video understanding with multimodal large language models. ST-GridPool introduces Pyramid Temporal Gridding to construct multi-granularity temporal grids of visual tokens and employs Norm-based Spatial Pooling to further reduce token count while preserving informative regions. The method is evaluated across multiple video understanding benchmarks, demonstrating its effectiveness

### Strengths
* The paper is clearly written and well-structured, making it easy to follow.
* The proposed approach is intuitive and straightforward to implement; its plug-and-play design enhances practical applicability across different MLLMs.
* In terms of performance, ST-GridPool demonstrates strong potential, consistently outperforming or matching existing token reduction methods across multiple benchmarks.

### Weaknesses
* The core idea of ST-GridPool bears strong resemblance to prior training-free methods such as TS-LLaVA (which combines gridding with token sampling) and IG-VLM (which relies solely on image gridding). However, comparisons with these approaches are relegated to the appendix and lack sufficient implementation details. Given that ST-GridPool appears to be a more complex integration of similar components, a more rigorous and fair comparison is warranted, ideally involving hyperparameter tuning for the baselines (e.g., number of grids, token compression ratios, or retained visual token counts) to ensure a level playing field. Without such controlled experiments, it is difficult to assess the true contribution of the proposed design.
* The efficiency analysis currently only compares LLaVA-Video with and without ST-GridPool. While it is unsurprising that adding a token compression module reduces computational cost, this alone does not demonstrate relative superiority. To substantiate its efficiency claims, ST-GridPool should be benchmarked against other token reduction methods under identical settings. 
* As shown in Table 1, the performance gains from integrating ST-GridPool are sometimes marginal. A more granular breakdown of results, e.g., by question type, video duration, or scene complexity on benchmarks like VideoMME and LongVideoBench, would help identify failure modes or scenarios where the method underperforms. Additional ablation studies and evaluation on more diverse datasets would further strengthen the analysis.
* The last row of Table 1 appears to be duplicated and should be corrected in the final version.

### Questions
Please refer to the weaknesses. I'll update my rating accordingly based on the authors' response.

### Soundness
2

### Presentation
3

### Contribution
3
