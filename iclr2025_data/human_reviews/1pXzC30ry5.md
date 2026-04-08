## Human Reviewer 1

### Summary
The paper presents a real-time, versatile segmentation model capable of interactive segmentation, panoptic segmentation, and video instance segmentation. 
While retaining the SAM encoder-decoder structure, the model incorporates an efficient encoder and adapter to enhance performance.
In the decoder, RAP-SAM introduces a three-stage pipeline that leverages novel pooling-based dynamic convolutions to refine mask tokens. Following the decoder, two additional prompt adapters are implemented to improve interaction between visual prompts and segmentation tokens.
RAP-SAM demonstrates efficiency and generalizability across various segmentation benchmarks.

### Strengths
1. The model achieves multi-purpose segmentation through an efficient structure and unified training approach.
2. The paper is well-written and easy to follow.
3. The experiments on panoptic segmentation, interactive segmentation, and video segmentation are solid, comprehensive, and persuasive, effectively demonstrating the model's contribution.

### Weaknesses
1. The paper lacks a detailed comparison with other SAM-like methods. A single COCO instance segmentation comparison in Table 4 is insufficient to substantiate claims of superiority over SAM. The results presented in Table 4 are not particularly outstanding. Additional experiments, such as on the SegAny task, with detailed metrics (AP for small, medium, large objects) on COCO instance segmentation, and evaluations with different object detectors, would strengthen the case.

2. Efficiency benchmarks are insufficiently detailed. For a model promoting efficiency, there should be a more comprehensive evaluation across different GPU platforms, such as the 3090 and V100, testing throughput and latency. Additionally, plotting latency versus performance compared to other SAM-like methods would provide a clearer visualization of the model's efficiency.

### Questions
1. Do you have the results for TopFormer in Table 3? Additionally, please bold the results in all comparison tables for clarity.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This work addresses the need for real-time multi-purpose segmentation by introducing a novel setting that encompasses interactive, panoptic, and video instance segmentation, striving for a single end-to-end model capable of handling all tasks in real-time. The proposed Real-Time Multi-Purpose SAM (RMP-SAM) utilizes an efficient encoder and a decoupled adapter for prompt-driven decoding, along with innovative training strategies and adapter designs, demonstrating effectiveness and strong generalization across benchmarks and specific semantic tasks while achieving an optimal balance between accuracy and speed.

### Strengths
1.Demonstrates impressive performance and inference speed.

2.Filling the gap in real-time multi-purpose segmentation.

3.The whole method is very simple and easy to understand.

4.Code is provided for easy reproduction by the reader.

### Weaknesses
1.Based on existing technology development, the entire pipeline is not novel.

2.Differences with SAMv2 should be further clarified, especially in terms of claimed semantic labels?

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper presents a real-time multi-purpose segmentation model called RMP-SAM. RMP-SAM handles various tasks such as interactive segmentation, panoptic segmentation, and video instance segmentation using a single model. To balance the accuracy and speed, RMP-SAM utilizes a lightweight encoder and a dynamic convolution-based decoder. RMP-SAM achieves fast inference while maintaining satisfactory performance.

### Strengths
- RMP-SAM unifies interactive segmentation, panoptic segmentation, and video instance segmentation within a single model.
- RMP-SAM offers a good trade-off between speed and accuracy.
- Extensive experiments demonstrate the model's effectiveness.

### Weaknesses
- The authors do not provide detailed information for joint training. Joint training for multiple tasks can be complex. How do the authors train RMP-SAM for some potential problems, such as avoiding the model being dominated by a single task and performance degradation by conflicts between different tasks?

- This paper ignores some related methods, making it difficult to assess the model's performance relative to existing SOTA approaches. For example, some universal methods[1,2,3] obtain better results than RMP-SAM using ResNet50. The authors should make a comprehensive comparison with other methods. 

[1] Tube-Link: A flexible cross tube framework for universal video segmentation. CVPR 2023.

[2] Dvis: Decoupled video instance segmentation framework. CVPR 2023.

[3] Univs: Unified and universal video segmentation with prompts as queries. CVPR 2024.

### Questions
Please see above.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The authors explore a novel real-time segmentation setting called real-time multi-purpose segmentation. It contains three fundamental sub-tasks: interactive segmentation, panoptic segmentation, and video instance segmentation. In contrast to previous methods that use a separate design for each task, the authors use only a single end-to-end model to handle all these tasks in real time. To fulfill the real-time requirements and balance multitask learning, a new dynamic convolution-based method, Real-Time Multi-Purpose SAM (RMP-SAM), is introduced. They benchmark several strong baselines by extending existing work to support multi-purpose segmentation.

### Strengths
- Large models can perform many tasks, but are not real-time capable because of the large encoders, while real-time models are often specialized in only one task. The method presented here aims to combine the two things, i.e., "the first real-time multi-purpose segmentation model".
- Precise implementation details are given and the comparisons with the other methods appear to be fair.
- The method achieves good results in the trade-off between performance and speed across the various tasks and datasets.
- The ablation studies are useful and show interesting insights.

### Weaknesses
- Many architectural elements were adopted from other works, it is not clear to me if there are already similar architectures as proposed here, or where exactly is the innovation (except the jointly training).
- In the related work section, many works are cited and also compared at the task level, but I also miss a comparison at the architectural level.
- The tables, especially table 3, are difficult to read because nothing is in bold print and you have to search for the trade-off here. A plot like Fig. 1b would be more useful.
- The references to the appendix could be a little more precise and there is no reference to Table 2.

### Questions
- What does the dot size in Fig. 1b indicate?
- The abstract says "generalization ability of these models across diverse scenarios", a learnable classifier with CLIP text embeddings is also used and “segment anything” is in the title. Is there a connection to open-vocabulary?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
8

### Confidence
3