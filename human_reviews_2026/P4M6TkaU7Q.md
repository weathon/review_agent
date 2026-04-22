# HashPose: Memory-Efficient Human Pose Estimation via Progressive Hash Codes

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Real-time human pose estimation on edge devices demands memory-efficient, high-precision methods. The dominant heatmap approaches, however, scale quadratically with input size, waste computation on background regions, and require slow post-processing. We propose HashPose, a framework replacing heatmaps with progressive hash codes: each keypoint is a binary sequence where successive bits refine localization, cutting memory complexity from $\Theta(HW)$ to $\Theta(log(HW))$. This direct bit prediction avoids dense heatmap-style background computations and removes the need for argmax or non-maximum suppression to decode coordinates. Furthermore, HashPose utilizes image classification backbones without upsampling layers to achieve high accuracy while significantly boosting its speed. We validate HashPose's performance envelope across a wide range of model sizes, ranging from an efficient 3.5M-parameter HashPose-XT model with 0.82 milliseconds frame latency and 85.9\% $AP^{50}$ on the COCO, to a 196.8M Large model that achieves a state-of-the-art 91.9\% $AP^{50}$ with 5.57 milliseconds frame latency using only the COCO training set. Simultaneously, HashPose has 510x lower output memory than heatmap configurations (0.48 MB vs 244.8 MB) for a typical $256 \times 192$ input, enabling high-throughput pose analysis that maintains high practical precision for on-device applications. Furthermore, its discrete representation is inherently suited for integer-only quantization, offering a clear path to further hardware acceleration on edge devices. Code is provided as supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper propose HashPose, a novel decoding method for human keypoints localization. Each keypoint in HashPose is a binary sequence and progressive refine the localization with 2-bits code like a depth-first search tree.  The direct bit prediction avoids dense heatmap-style background computations and removes the need for argmax or non-maximum suppression to decode coordinates, which result in efficieny improvement of memory usage and inference latency. Based on their decoding methods, this paper construct a neural network to predict the human pose and achieve better AP50 than previous counterparts.

### Strengths
1. This paper propose a novel and interesting decoding method for human keypoints localization.
2. HashPose do not rely on 2D heatmaps and NMS, result in better efficiency of memory usage and inference latency.
3. HashPose achieve better AP50 on COCO dataset than pervious conterparts.
4. Good writing and easy to follow.

### Weaknesses
- Major
1. HashPose employ a powerful backbone - ConvNeXt v2 and there are not performance on other backbone like ResNet 50 and HRNet. While their counterparts in experiments do not use such a strong backbone. The comparison under different setting seems meaningless.

2. When compare with other methods on COCO, HashPose only consider AP50, which is a simple metric. According common practice, the AP and AR on COCO are more valuable and meaningfull.

3. According Line 320, when compare the inference efficiency, HashPose use TensorRT and BF16, while other methods do not employ such accelaration technique. And when compare the latency, they use the different GPU. For example, HashPose test the latency on RTX 3090, but ED-Pose (51ms ) are tested on A100.

4. The author claim 'HashPose is Plug-and-Play' in contribution. Actually, the HashPose just decoples the backbone and decoding head,  their decoding head still need be trained. Such overclaim weakens the contribution of the article.

- Minor
1. Some recent works are not compared, such as GroupPose (ICCV 2023).

2. The paper only report the memory usage of decoding head. Due to the model is function together, the memory usage of whole model is truely considered.

3. The paper uses a lot of space to derive the error bounds, which should be put in the appendix. What really needs to be described is the decoding method rather than these formulas. HashPose decoding is essentially a two-dimensional binary search, and their error bounds are easy to understand. These simple formulas do not enhance the contribution but waste valuable space.

### Questions
1. HashPose is an interesting method for HPE, but the authors shoulde re-design their experiments. If the comparisons are fair, I think it will be a good paper.

2. When evaluate on COCO, which bounding box file do you use? There are two different bbox file: gt bbox and AP56 bbox. Due to FlatPose is a top-down method, incorrect usage can result in a 2-4 point difference in AP. So it is better to directly indicate this in implementation details.

3. The HashPose-XT use 3.5M parameters, but the minist backbone, ConvNeXT-a has 3.7M parameters, how you get this value.

4. Due to the author only consider the memory usage of the head in Table 1, I want to know the Params, FLops and Time in same table belong to whole model, or only the head?

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes to encode keypoint positions with quadtree hash code. It removes the cost of upsampling head in conventional heatmap-based methods. Results on COCO and MPII are reported.

### Strengths
HashPose provides a model family that replaces heatmaps with binary hash codes for keypoints, reducing memory and computation, removing post-processing, enabling fast, accurate, quantization-friendly pose estimation ideal for real-time edge deployment. A novel bitwise loss is proposed for effective training.

### Weaknesses
- [Fair comparison] HashPose is based on more advanced model family ConvNext-v2. Since the hash code formulation is flexible as mentioned in the paper, the authors should provide apples-to-apples comparison to heatmap and regression **using the same backbone and post-processing (with or without)** (e.g. Table 3 in RLE[1]).
- [Detailed results] Only $AP^{50}$ and $AR^{50}$ are provided in Table 1 and Table 5. Please provide the full metrics list. (e.g. Table 3 in RLE[1])

[1] Li, Jiefeng, et al. "Human pose regression with residual log-likelihood estimation." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

### Questions
- [Cost profiling] The running time of HashPose in Table 1 is optimized by TensorRT. What's its original inference cost in PyTorch? Are the reported numbers of other models also optimized by TensorRT?
- [Params] Why the parameters cost is significantly higher than other methods?

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
4

### Summary
This paper introduces HashPose, a novel framework for human pose estimation designed to address the Θ(HW) memory complexity of existing heatmap-based methods, which becomes prohibitive at high resolutions. The authors argue that this quadratic memory growth makes heatmap methods unsuitable for resource-constrained edge devices.
As an alternative, the paper proposes a "progressive hash codes" representation with Θ(log(HW)) complexity. Each keypoint is encoded as a binary sequence where successive bits progressively refine its spatial location. The model is designed to predict these bits directly, thereby avoiding the dense computations and complex post-processing (e.g., argmax) associated with heatmaps. The authors also propose a "Progressive Code Learning" framework, which includes progressive bit re-weighting and a push-pull regularization loss.
Experiments claim that the method achieves extremely high memory efficiency (reportedly 510x lower than heatmaps) and can directly leverage classification backbones without upsampling layers, all while maintaining high accuracy (AP⁵⁰).

### Strengths
1. Elegant Keypoint Representation: The core contribution of this paper—replacing dense Θ(HW) heatmaps with Θ(log(HW)) progressive hash codes—represents a fundamental and elegant paradigm shift. The idea itself is highly valuable as it directly addresses a long-standing and practical bottleneck in high-resolution pose estimation.
2. Potential for High Efficiency: The method demonstrates a significant theoretical advantage in memory consumption, especially for high-resolution inputs. Furthermore, the framework's ability to use standard classification backbones without complex decoders or upsampling layers is a major engineering advantage that simplifies the model architecture.
3. Strong Coarse-grained Accuracy: The experimental results show that HashPose (particularly the -L and -H models) can achieve state-of-the-art or near state-of-the-art AP⁵⁰ metrics on the COCO dataset. This confirms that this novel coordinate representation is effective for localization tasks.

### Weaknesses
While the core idea is highly innovative, the paper's argumentation, especially concerning its central contribution of memory efficiency, suffers from a serious lack of rigor and clarity, which significantly undermines the credibility of its claims.
1. Serious Concerns about Memory Saving Claims: There is a critical calculation error in Appendix C, The denominator in Equation (1) of Appendix C is 1024. If the numerator is in Bytes (as indicated by "* 4 expresses 4 bytes"), then dividing by 1024 should yield units of KB (Kilobytes), not MB (Megabytes). This is a glaring error that directly impacts the paper's primary contribution claim, as KB-level memory savings seems trivial compare to MB-level model parameters. Meanwhile, in Appendix C, the authors assume a 32-bit heatmap while using 16-bit hash codes. For a more rigorous comparison, the authors should have at least discussed or evaluated the use of 16-bit heatmaps.
2. Lack of Transparency in Experimental Setup: Appendix B states that the paper follows a two-stage top-down pipeline, which raises crucial questions about the scope of the "Time (ms)" reported in Table 1. It is highly likely that this measurement excludes the latency of the first-stage person detector. Furthermore, it is also unclear whether the reported time includes the feature extraction backbone of the pose estimator itself, or if it only measures the forward pass of the novel prediction head. While excluding the detector is a common practice when comparing top-down methods, true practical performance depends on the entire pipeline. For the sake of transparency and reproducibility, the authors should explicitly state in the table's caption precisely which components are included in the timing (e.g., "pose head only", "estimator only" or "end-to-end inference"). Without this clarification, readers cannot accurately assess the model's true, practical inference speed.
3. Trade-off in Fine-grained Precision: As shown in Table 2, HashPose lags significantly behind state-of-the-art methods like HRNet on the AP⁷⁵ metric. The authors acknowledge this is due to the difficulty in predicting the Least Significant Bits (LSBs). This indicates that the method faces an unresolved trade-off between efficiency and high-precision accuracy.
4. Counter-intuitive Ablation Study: The ablation study in Table 3 shows that Label Smoothing (LS) contributes a massive +2.8% AP⁷⁵ improvement. This is highly counter-intuitive. The purpose of LS is to soften labels to prevent overconfidence, whereas the purpose of the Push-pull (PP) regularization is to sharpen predictions and penalize ambiguity. The two methods seem contradictory in principle. The authors provide no explanation for why they work well together or why LS is so crucial for what is fundamentally a binary classification task.

### Questions
1. [Critical] Could you please clarify the memory calculation in Appendix C? How was the 244.8 MB figure derived? Is the unit resulting from dividing by 1024 KB or MB? Please correct the formula. Given that the inference latency is also significantly reduced, the overall efficiency gains of the proposed method are evident. However, a precise and clear justification of the memory savings is crucial for accurately positioning this important contribution.
2. [Critical] Please clarify in the caption for Table 1 whether the "Time (ms)" includes the time for the first-stage person detector and the pose estimator backbone.
3. Given the performance gap in AP⁷⁵, do you consider HashPose to be competitive for tasks that require high-precision localization rather than just coarse detection?
4. Could you explain the synergy between LS and PP regularization? Why does LS technique lead to such a significant +2.8% AP⁷⁵ improvement in a binary hash code prediction task? This seems counter-intuitive.
I have opted for a "borderline reject" score because, despite the high novelty of the core idea, the paper's central claims about memory efficiency appear to be based on flawed calculations (Appendix C) and are presented in a highly misleading manner (Table 1). However, I want to emphasize that I am open to changing my evaluation. I am fully prepared to raise my score if the authors can provide a clear, rigorous, and convincing rebuttal that satisfactorily addresses all of the [Critical] questions I have raised.

### Soundness
2

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
HashPose proposes a memory-efficient human pose estimation framework that replaces traditional heatmaps with progressive binary hash codes, reducing output memory from Θ(HW) to Θ(log(HW)). Each keypoint is encoded as a sequence of bits refined from coarse to fine spatial levels, enabling direct coordinate prediction without post-processing and achieving state-of-the-art accuracy with up to 510× less output memory.

### Strengths
1. The paper is well-written and easy to follow.
2. The proposed methods are interesting and innovative.
3. Experimental results reflect the effectiveness of the proposed method.

### Weaknesses
1. Figure 5 shows that HashPose exhibits unstable predictions for LSB, which limits its fine-grained localization accuracy (AP75), making it inferior to certain heatmap-based methods.
2. HashPose may be sensitive to training details. I am curious whether the proposed method can maintain strong performance on joints with high degrees of freedom or in complex scenarios involving occlusion.

### Questions
Please refet to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3
