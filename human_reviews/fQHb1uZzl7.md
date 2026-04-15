# Unifying Feature and Cost Aggregation with Transformers for Semantic and Visual Correspondence

- Decision: Accept (poster)
- Scores: 5, 5, 6, 6

## Abstract
This paper introduces a Transformer-based integrative feature and cost aggregation network designed for dense matching tasks. In the context of dense matching, many works benefit from one of two forms of aggregation: feature aggregation, which pertains to the alignment of similar features, or cost aggregation, a procedure aimed at instilling coherence in the flow estimates across neighboring pixels. In this work, we first show that feature aggregation and cost aggregation exhibit distinct characteristics and reveal the potential for substantial benefits stemming from the judicious use of both aggregation processes. We then introduce a simple yet effective architecture that harnesses self- and cross-attention mechanisms to show that our approach unifies feature aggregation and cost aggregation and effectively harnesses the strengths of both techniques. Within the proposed attention layers, the features and cost volume both complement each other, and the attention layers are interleaved through a coarse-to-fine design to further promote accurate correspondence estimation. Finally at inference, our network produces multi-scale predictions, computes their confidence scores, and selects the most confident flow for final prediction. Our framework is evaluated on standard benchmarks for semantic matching, and also applied to geometric matching, where we show that our approach achieves significant improvements compared to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a dense matching estimation method by unifying both feature and cost volume aggregation with transformer. The authors first analyze the merits and faults of feature and cost aggregation and then claim that using them interleavely could improve the feature representation. Experiment results show the effectiveness of the proposed method.

### Strengths
1. To the best of my knowledge, this is the first work that discusses the relationship between feature and cost aggregation (learning). The authors carefully discuss their merits and faults in Sec.1 and Sec.3.
2. Interleavely aggregating features between feature and cost volumes is interesting. The visualization results of Fig.2, and Fig.3 also verify the effectiveness.
3. The authors provide complete experimental details in the supplementary.

### Weaknesses
1. The usage of "aggregation" is a little confusing. In my opinion, "aggregation" means combining multiple features into a single one, which should usually be used to describe the process of attention aggregation of ($QK^T$ and $V$). However, I am not sure whether "aggregation" is suitable to be used to indicate the whole learning process of cost volume learning. Because many cost volume learning is not related to attention learning.
2. Although the authors analyze the merits of feature/cost aggregation, some claims have not been clarified. For example, feature matching is "challenged by repetitive patterns and background clutters", while cost volume learning enjoys "robustness to repetitive patterns and background clutter". No evidence is shown in this paper to support this claim.
3. The authors did not formulate the method presentation well in Sec.4 and Fig.4, which makes the proposed method suffer from too complicated designs and difficult to follow. I strongly recommend the authors introduce the shape and reshape of the most important tensors to make the whole pipeline clearer. The concatenation in Eq(3) is operated along which dimension? Why $C'$ appears again in Eq.5 as $QK^T$, while $C'$ should be already defined as the output of the cost volume feature?
4. The experiments are not solid enough. The proposed method needs to be compared with more recent methods. In the geometric matching results from Tab2, most competitors are from 2020 and 2021, which are far from "state-of-the-art". Only one flow estimation method is considered (GMFlow). However, as discussed in the supplementary, the comparison is not fair, because GMFlow is trained on Sintel rather than DPED-CityScape-ADE+MegaDepth fine-tuning. Besides, all these competitors are trained on DPED-CityScape-ADE **or** MegaDepth (the proposed method is trained with DPED-CityScape-ADE **and** MegaDepth finetuning) as said in supplementary B.1. The authors should clarify this.

### Questions
As discussed in the related works, many stereo-matching and optical flow works use Transformer-based cost aggregation networks. 
The idea proposed in this paper should be a general way to improve all feature matching-based tasks, and I think there are no enormous model differences among stereo, flow, and dense matching. So the authors should compare these SOTA stereo and flow methods in a more fair way. For example, re-training the model with the same data setting for dense matching or verifying the effectiveness of the proposed method in stereo and flow estimation benchmarks.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to combine feature and cost aggregation to address the dense feature matching task.

The main idea is to use cost score matrix for both self- and cross-attention feature updating, to learn more discriminative features and compute better cost matrix.

Experiments on some semantic and geometric matching datasets show the effectiveness of the proposed method.

### Strengths
1) This paper is generally presented well;

2) The idea of use cost score matrix for both self- and cross-attention feature updating is simple and effective;

3) Experimental results are good.

### Weaknesses
1)  Section 5.1. It is quite blurry for me whether previous state-of-the-art methods in Table 1 and 2 are trained on the same datasets; For example, the proposed method is trained on the DPED-CityScape-ADE and MegaDepth datasets;

2) The performance of the proposed method on the optical flow (KITTI, Sintel) task is blurry for me;

3) It's good to see the improved matching performance on the HPatches dataset. However, I want to see whether the improved matching would lead to better Rotation and translation estimations. 

4) Using cost score matrix for both self- and cross-attention feature updating is good. However, this contribution may be constrained to large overlapping ratio between images. If pairwise images have small overlapping ratio, the cost score matrix is noisy, and may provide wrong guidance for feature updating. Would you please check whether the proposed method works on some challenging image pairs from the MegaDepth dataset.

5) Please show some failure cases.

### Questions
Please refer to the weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a new vision transformer architecture to conduct feature aggregation and cost aggregation for dense matching tasks. The authors show distinct characteristics of feature aggregation and cost aggregation. and use self- and cross-attention mechanisms to unify the feature and cost aggregation. They validate the effectiveness of the proposed method with semantic matching and geometry matching.

### Strengths
1. The idea to unify feature aggregation and cost aggregation is interesting. It compensates for the lack of semantic information in cost representation and helps to drive the features in each image to become more compatible with others.
2. They conduct extensive experiments on semantic matching and geometry matching to validate the effectiveness of the proposed method UFC. UFC can improve the matching performance. And they provide step-by-step ablations of each component.
3. The paper is well-organized and easy to understand. The authors visualize the changes in feature maps and cost volumes, which helps understand how their method works. I see that feature aggregation can preserve semantic information and geometry structure and the cost aggregation reduces the noise in cost volumes.

### Weaknesses
1. In visualization results Figure 2, features with integrative aggregation methodology preserve the semantic information. However, it seems the proposed method damages the local discriminative ability of features.

### Questions
1. The local discriminative ability of features is also important for dense matching tasks. I would like to see an analysis of whether the proposed method causes damage in this perspective or whether this issue can be avoided in some design of the method.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces an integrative feature and cost aggregation modules in a CNN architecture for a semantic correspondence task. The introduced module cleans up noisy matches in the cost volume and thus improves the matching accuracy. The paper demonstrates better accuracy on semantic matching and geometric matching tasks by using their method.

### Strengths
- Good results

  The paper achieves good accuracy on both semantic and geometric matching tasks (Table 1 and 2). It demonstrates the effectiveness of the proposed aggregation modules.

- Detail analysis

  The paper provides a sufficient amount of analysis. Fig. 3 visualizes the qualitative comparison of the proposed modules (from (f) to (h)). Further, the ablation study (Table 3 and 4) validates the proposed ideas.

### Weaknesses
Despite the good accuracy on both tasks, there are concerns about novelty/contributions.

- Existing ideas in other literature

  Similar ideas on feature and cost volume aggregation have been demonstrated in other literature such as stereo matching [a,b] and optical flow estimation [c]. Actually the related work section (Sec. 2) summarizes those relevant papers very well. Compared to the existing solutions, what would be the new technical design (in self-/cross-attention) of the proposed module, except for applying it to semantic & geometric matching problems? Can the newer technical design from the proposed module also benefit other tasks that use cost volume, eg., stereo matching, optical flow, scene flow, etc. ?

   [a] Attention-Aware Feature Aggregation for Real-time Stereo Matching on Edge Devices, ACCV 2020
   
   [b] Attention Concatenation Volume for Accurate and Efficient Stereo Matching, CVPR 2022
   
   [c] GMFlow: Learning Optical Flow via Global Matching, CVPR 2022

- Limitation

  Discussion on the limitation is missing. What would be the limitation of the method or unsolved problems?


- Can the paper provide more qualitative examples and discuss where the gain mainly originates?

  Table 4 shows the accuracy improvement by adding more components. I am wondering if the paper can also include some qualitative examples and discuss where the gain mainly originates, such as resolving some particular matching ambiguity. It would be great if the paper can provide more insights related to its improvement.

### Questions
- Increase of learnable parameters

  How many number of learnable parameters account for the new module (i.e., integrative feature and cost aggregation module)? How significant are they compared to the number of parameters of the entire network (15.5M)? Probably it's also good to include an extra column in Table 3 and 4 for the number of network parameters.

- Resolution of the cost volume

  What's the resolution of the cost volume (saying the input image resolution is HxW)? At each level the features are upsampled ($D^{l}_s$), but how can the resolution of the cost volume ($C^{l})$ remain the same over different pyramid levels? Is there any reason to fix the resolution of the cost volume?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
