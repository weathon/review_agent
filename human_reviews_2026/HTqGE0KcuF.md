# WAFT: Warping-Alone Field Transforms for Optical Flow

- Avg Score: 6.67
- Decision: Accept (Oral)
- Scores: 6, 8, 6

## Abstract
We introduce Warping-Alone Field Transforms (WAFT), a simple and effective method for optical flow. WAFT is similar to RAFT but replaces cost volume with high-resolution warping, achieving better accuracy with lower memory cost. This design challenges the conventional wisdom that constructing cost volumes is necessary for strong performance. WAFT is a simple and flexible meta-architecture with minimal inductive biases and reliance on custom designs. Compared with existing methods, WAFT ranks 1st on Spring, Sintel, and KITTI benchmarks, achieves the best zero-shot generalization on KITTI, while being 1.3-4.1x faster than existing methods that have competitive accuracy (e.g., 1.3x than Flowformer++, 4.1x than CCMR+). Code and model weights are available at https://github.com/princeton-vl/WAFT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a simplified meta-architecture for optical flow estimation without using cost volumes. The proposed WAFT algorithm consists of an input encoder which leverages existing large-scale pre-trained models for feature extraction, and a recurrent update module based on vision transformers that can iteratively updates optical flow with large displacements. Experiment shows that the proposed WAFT algorithm achieves top rankings on various benchmarks, including Spring and KITTI, furthermore, it does so with significantly lower memory cost and up to 4.1x faster inference times.

### Strengths
1. The design of WAFT without cost volume computation is very simple, flexible and effective, making it a significant contribution for computer vision research community
2. By avoiding cost volumes computation, WAFT can perform warping on original resolution feature maps, which can help achieving sharper boundary predictions in optical flow estimation
3. WAFT has shown best zero-shot cross-dataset generalization on KITTI, which is an important property towards generalization capability on unseen data.
4. The paper is well-structured and clearly written.

### Weaknesses
1. The iterative recurrent update module may restrict the algorithm's potential for parallel optimization to achieve low latency. 
2. WAFT relies on existing pre-trained vision foundation models, which may limits its potential for further computational efficiency improvement on feature extraction.
3. Compared with improved memory and computational efficiency improvement, the improvement on flow accuracy is relatively limited.

### Questions
1. In table 2, while the ratio of MACs reduction is high (CCMR+'s 12653 vs. WAFT-DAv2-a1's 853), the speed up of latency is not at the same scale (CCMR+'s 999 vs. WAFT-DAv2-a1's 240), it would be good to give further explanation on this.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper points out the drawback of constructing cost volume in the optical flow field, and propose to replace the cost volume with warping. To achieve competitive performance, the authors propose to utilize: 1) stronger feature encoder 2) high-resolution warping 3) attention- (and pretrained) based updater.

### Strengths
1. This paper is well written.
2. This paper has a clear and extensive ablation study to show the effectiveness of each design choice.
3. The author introduced an attention-based updater to replace the cost volume for feature similarity computation. This design is resonable and novel.

### Weaknesses
1. Since the author replaces the commonly used CNN updater with attention-based one, it is better to provides more details of the layers.
2. For the models used in Table 2, what is the downsampled ratio? And which line corresponds to the statement in the abstract "while being up to 4.1× faster than methods with similar performance". From my understanding, WAFT-Twins-a2 uses the same feature encoder as FlowFormer++, achieves similar performance but not significant speedup?
3. Can the authors provide more explaination about why context encoder is not useful in the WAFT architecture?

### Questions
Please refer to the Weaknesses section.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes to abandon the cost volume that's a standard component in deep optical flow architectures, and uses the warped target feature vector alone (plus the source feature vector) for flow estimation. It implicitly builds a global context via self-attention in the recurrent update module, which is why it can get rid of the cost volume.

### Strengths
1. Removing the cost volume is a good contribution, which may make estimation of optical flow on high-res images much more feasible.

### Weaknesses
1. No detailed evaluation of how the model performs on large displacements, on which WAFT might be slightly weaker than models using a cost volume. The authors could make artificial displacements to stress-test WAFT to see where its limit lies.
2. No details are given for the Recurrent Update Module. For example, how many layers (esp. self attention layers), what's the total param count?

### Questions
1. Since no cost volume is adopted, I'm worried that the initial errors may accumulate and become larger as the model iterates. The authors could consider such a perturbation test: in the first iteration, perturb the flow prediction with random values, then see how well the model recovers from it.
2. Another challenging scenario is if there are multiple similar objects (e.g. a table with many cups), how well WASP would perform. Chance is the self attention may overly smooth features across these similar objects and make the prediction more random. Of course this would be challenging for methods **with** a cost volume, but I'm curious if it would be more challenging for WASP.

 (Note: I would not lower my rating if the model is not so robust under such perturbations; this is just to better inform readers whre are the "sweet spots" in which the method performs well.)

### Soundness
3

### Presentation
2

### Contribution
2
