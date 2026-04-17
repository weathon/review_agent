# Exploring High-Order Self-Similarity for Video Understanding

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Space-time self-similarity (STSS), which captures visual correspondences across frames, provides an effective way to represent temporal dynamics for video understanding. In this work, we propose higher-order STSS and demonstrate how STSS at different orders reveal distinct aspects of these dynamics. We then introduce multi-order self-similarity (MOSS) module, a lightweight neural module designed to learn and integrate multi-order STSS features and readily applied to video classification architectures to enhance motion modeling capabilities while consuming only marginal computation cost and memory usage. Evaluated on Kinetics-400 and Something-Something V1 & V2 benchmarks, our method achieves strong performances, achieving the best memory-accuracy trade-off compared to state-of-the-art approaches. Source code and checkpoints of our model will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses conventional video understanding tasks by exploring high-order self-similarity. It begins by analyzing how high-order spatio-temporal self-similarities (STSS) facilitate learning temporal dynamics in videos, supported by vivid visualizations. Subsequently, it proposes MOSS, a lightweight neural module designed for comprehensive temporal understanding. Experimental results on widely-used benchmarks, including Kinetics-400, Something-Something V2 (SthSth-V2), Diving48, and FineGym, demonstrate the effectiveness and efficiency of the proposed approach.

### Strengths
1. This paper is well-written, presenting substantial qualitative and quantitative results that clearly support the effectiveness of the proposed high-order STSS for video understanding.

2. The chosen entry point, i.e., learning distinct high-order STSS representations particularly for dynamic temporal video understanding, is both novel and interesting. Furthermore, the visualizations of high-order STSS effectively illustrate their operational mechanisms.

3. This work is generally solid, as the proposed approach is thoroughly verified across multiple benchmarks and various video understanding tasks, including action recognition, temporal action detection, and generic event boundary detection.

### Weaknesses
1. My primary concern revolves around the recency of the compared methods. The main baselines, ATM and Side4Video, were both released in 2023, and generally, most works compared in this paper date from 2023 or earlier. This raises questions about the absence of comparisons with more recent advancements in the field. If there are indeed no significant video understanding (e.g., action recognition) works from 2024 or 2025 that are relevant, it would be important for the authors to clarify this. Otherwise, the lack of comparison with contemporary methods might suggest that incremental modifications to conventional and relatively smaller models may not be considered a substantial contribution to the broader video understanding area.

2. Regarding the overall architecture, could the authors explain the rationale behind bridging certain layers with MOSS modules while others are connected solely via Fully Connected (FC) layers? Is there any analysis presented to demonstrate that different layers exhibit varying preferences for motion-augmented visual features?

### Questions
Some additional typos:  
1. Given the paper's analysis that the 2nd order STSS provides motion segment information, and the 3rd order STSS demonstrates the overall layouts of motion segments, it is puzzling why combining the 1st, 2nd, and 3rd order STSS does not lead to better results in Table 4(b). While the authors claim that "redundant temporal dynamics" are captured for the 2nd and 3rd orders, this explanation does not seem to align well with the vivid illustrations provided in the main paper.

2. How do the 1st- to 3rd-order STSSs help to distinguish the "(a) Moving sth closer to sth" from "(b) Moving sth away from sth", in Figure 5?

3.  Can the format for chart references be standardized? (Figure _vs._ Fig. )

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
3

### Summary
This paper introduces the concept of high-order space-time self-similarity (STSS) for video understanding, proposing that while 1st-order STSS captures basic motion flows, 2nd-order STSS identifies coherent motion segments, and 3rd-order STSS captures their layout. To leverage this, the authors created the Multi-Order Self-Similarity (MOSS) module, a lightweight neural component that learns and integrates these multi-order features to enhance motion modeling. When applied to video action recognition, this method achieves new state-of-the-art results on motion-centric datasets like Something-Something V1 & V2, Diving48, and FineGym, demonstrating a superior memory-accuracy trade-off compared to previous approaches.

### Strengths
1. The proposed method achieves new state-of-the-art results on several motion-centric action recognition benchmarks, e.g. Something-Something V1 & V2 , Diving48, and FineGym, outperforming previous methods with substantial margins.
2. The paper introduces the Multi-Order Self-Similarity (MOSS) module, which is a lightweight neural module. This module enhances motion modeling capabilities while adding only marginal computation cost and memory usage.

### Weaknesses
1. The method is explicitly designed to enhance "motion modeling capabilities" and achieves its state-of-the-art results on motion-centric benchmarks like Something-Something V1/V2, Diving48, and FineGym. However, on benchmarks like Kinetics-400, the performance gain is modest. The MOSS-L model achieves only 0.7% improvement over its baseline. The generalization of such method is yet to be clarified.
2. The paper explores Self-Similarity in videos, in the era of large models, what will be the benefit of such method on MLLM video understanding?
3. The paper's main novelty is the exploration of high-order self-similarity. However, the paper's own analysis shows diminishing returns. While 1st-order STSS provides a significant boost, the individual gains from 2nd, 3rd, and 4th-order STSS become progressively smaller.

### Questions
1. Since I am not an expert in this domain, what are the real-world applications of STSS?

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
3

### Summary
This paper presents a method of high-order space-time self-similarities (STSS) for video understanding. It consists of stacks of self-similarity modules that are similar to self-correlation modules where the difference is general similarity vs. vanilla correlation. The authors then stacks multiple layers of STSS in the neural network model. The authors argue that the proposed method achieves SOTA results on several motion-centric video understanding benchmarks.

### Strengths
- The proposed method achieves SOTA on multiple motion-centric benchmarks. 
- Extensive experiments and ablation studies are provided. The analysis and visualizations provide good insights.
- The presentation of this paper is good and easy to understand and follow.
- The toy visualization in Figure 3 is a good way to help readers understand the method.

### Weaknesses
- The novelty is somewhat limited. The proposed STSS superficially looks interesting. However, the core operation is essentially multiple stacks of correlation/distance + encoding. I believe there has been multiple previous works with similar ideas. It would be surprising to me if the proposed method achieve SOTA while previous methods did not.
- In Table 4, it seems that higher order of STSS does not necessarily lead to increased performance. In fact, the authors admits that "STSS beyond 3rd-order do not provide significant benefits for action recognition tasks". This may have shown that high orders of STSS may not be necessary or even correct, which makes the contribution of this paper quite limited.
- The proposed method heavily depends on the pre-trained encoder as shown in Table 9. Not sure if this is OK.
- Not sure how the proposed STSS module can be used in other video related operations such as generative models.

### Questions
- What is the difference between the proposed STSS vs. attention mechanism in transformer? If the similarity function $\phi$ is replaced by a learnable similarity, isn't it be reduced to an attention layer whose attention mask applies to the rectangular region of $L\times U\times V$?
- How can STSS module be used in other video related operations such as diffusion models?

### Soundness
3

### Presentation
3

### Contribution
2
