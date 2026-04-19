# SD4Match: Learning to Prompt Stable Diffusion Model for Semantic Matching

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 8

## Abstract
In this paper, we address the challenge of matching semantically similar keypoints across image pairs. Existing research indicates that the intermediate output of the UNet within the Stable Diffusion (SD) framework can serve as robust image feature maps for such a matching task. We demonstrate that by employing a basic prompt tuning technique, the inherent potential of Stable Diffusion can be harnessed, resulting in a significant enhancement in accuracy over previous approaches. We further introduce a novel conditional prompting module that conditions the prompt on the local details of the input image pairs, leading to a further improvement in performance. We designate our approach as SD4Match, short for Stable Diffusion for Semantic Matching. Comprehensive evaluations of SD4Match on the PF-Pascal, PF-Willow, and SPair-71k datasets show that it sets new benchmarks in accuracy across all these datasets. Particularly, SD4Match outperforms the previous state-of-the-art by a margin of 12 percentage points on the challenging SPair-71k dataset.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper is a follow-up work to (Tang et al., 2023) and (Zhang et al., 2023), which uses stable diffusion model's feature maps for semantic matching, due the fact that stable diffusion could provide very well semantic meaningful features. In this work, three prompting options are designed for stable diffusion on semantic matching, which are single, class, and conditional prompting. The results show the effectiveness of the simple prompting.

### Strengths
1) The paper is easy to understand

2) The constructed prompting strategies is easy and effective.

3) The stable diffusion's feature is good for semantic matching.

### Weaknesses
1)  Even though one can design lots of different detailed schemes regarding conditional prompting, the idea is already appeared in CoCoOp [1] and it does not show very big difference. The reviewer expects some novel things in addition to some plain designs.

    [1] Conditional prompt learning for vision-language models @ CVPR'22

2) Why skip VAE and directly refer to UNet's input as image I?

3) All the credits went to stable diffusion and the discovery of its features that is good for semantic matching (Tang et al., 2023) and (Zhang et al., 2023). The reviewer deeply felt that there is no much advance in this work, either comparing to (Tang et al., 2023) and (Zhang et al., 2023) or prompt tuning method CoCoOp [1]

Based on the above concerns, the review suggests rejecting the paper.

### Questions
see above

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a prompt tuning method for Stable Diffusion (SD) feature extraction to solve the semantic matching problem. Recently, semantic correspondence has achieved significant performance improvements by extracting discriminative local features from the U-Net structure of SD. In this paper, the authors propose a conditional prompting module (CPM) using local patch embedding in addition to global descriptors using object categories to construct prompts. Prompt using the CPM module in the proposed SD features significantly improves performance for some cases of SPair-71k, but this performance is not significantly different from SD features using class label prompt.

### Strengths
Good motivation: using the features of U-Net within SD for semantic matching is a good approach. The additional use of langauge information (or prompt embedding) for robust feature extraction is a novel direction.

Proper citation: The citation of existing research on semantic matching in section 2 related work is appropriate and well categorized. In addition, the recent developments in diffusion models and prompt tuning and the related research synthesis are helpful for understanding the research history.

High performance: Even though the study uses weak-supervision (class labels) at inference time for strong-supervised training, PCK@10=75.5 on SPair-71k is quite high performance.  
However, the excessive increase in image resolution needs a fair comparison. (See Weakness for details) 

Ablation study: Fig. 3(a) shows the effect of the proposed CPM module under various conditions.

### Weaknesses
Results of geometric matching: The results on the three standard benchmarks for semantic matching are impressive. However, for a definitive proof of SD's ability to extract discriminative features, it should also perform on general geometric matching. 
For example, HPatches [1] has a name for each sequence, possible to evaluate the proposed method in this benchmark.

Results on other benchmarks in Table 1: It is impressive to see the performance improvement on SPair-71k by only changing the empty string to object category, can you show this result on PF-PASCAL as well?

Missing reference: this paper is missing a citation to a paper that proposes a method [2] to extract discriminative local features for semantic matching. Please cite this paper.

Section 3.3 N_{dino} subscript should not be italic. Italic is for enumerate, please use roman. 

Section 4.1. "we set the timestep t = 50 as empirical tests suggest it provides optimal results, even though our method was trained at t = 261. "
How did you decide on this t=50? Did you get the results from your test set? Figure 3 (b) looks to evaluate  the performance on the test set. I am worried to tune the model on test set and list the highest one.  It would be fair to find the optimal model by searching on the validation set.

Section 4.1. "Images are resized to 768 × 768 for both the training and testing phases. "
Was this image resize done the same for all the other methods in Table 2? I doubt if the performance improvement is from the prompt tuning you propose or from the image resize. 
Please provide information on image resolution, computational cost (GFlops), and performance on PF/SPair.

Concern of the performance gain 1: Table 2. The proposed CPM is not significantly different from the class prompt (a photo of [category]). Does this mean that simply giving a class prompt will give the same result? TThis requires a detailed analysis and explanation.

Concern of the performance gain 2: Table 3. The zero-shot generalization of CPM is actually worse than empty string (single) baseline. 

Figure 5. Visualization effectively decomposes semantic meaning in different category objects. Can this be evaluated across multiple instances of the same class? [3] I don't need quantitative results, I'm just wondering whether the possibility of prompt tuning using SD can discriminate instances of a category.


[1] HPatches: A benchmark and evaluation of handcrafted and learned local descriptors (Balntas et al., CVPR 2017)
[2] Learning to Distill Convolutional Features into Compact Local Descriptors (Lee et al., WACV 2021)
[3] MISC210K: A Large-Scale Dataset for Multi-Instance Semantic Correspondence (Sun et el., CVPR 2023)

### Questions
Please refer to the weakness section.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper employs prompt tuning to the previous UNet + stable diffusion solutions for semantic matching. It also introduces a new conditional prompting module to condition the prompt on the local details. These two elements contribute to the proposed method SD4Match to achieve higher accuracy than existing methods on three benchmarks.

### Strengths
Applying prompt tuning for SD based semantic matching methods is new.

The proposed conditional prompting module is reasonable for the semantic matching task.

The superior results than previous works on the standard benchmarks.

### Weaknesses
The paper lacks comparison to typical semantic matching methods based on deep graph matching, such as 
[1] Joint graph learning and matching for semantic feature correspondence, PR, 2023
From the results in [1], the paper can not beat many existing works. Further explanations and validation are required.

As the paper lacks comparisons to some highly related works, the paper also misses to discuss a majority works on semantic correspondence, for instance,
[1] Deep graph matching via blackbox differentiation of combinatorial solvers, ECCV 2020.
[2] Deep Graph Matching under Quadratic Constraint, CVPR, 2021.
[3] GLMNet: Graph learning-matching convolutional networks for feature matching, PR, 2022.

### Questions
There is query feature extraction for the I^A, how about put this on the I^B side? Does this will affect the final results?

The complexity analysis about the training and run time can be included in the paper.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
