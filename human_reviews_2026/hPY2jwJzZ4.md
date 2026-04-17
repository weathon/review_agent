# Pose Prior Learner: Unsupervised Categorical Prior Learning for Pose Estimation

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 2

## Abstract
A prior represents a set of beliefs or assumptions about a system, aiding inference and decision-making. In this paper, we introduce the challenge of unsupervised categorical prior learning in pose estimation, where AI models learn a general pose prior for an object category from images in a self-supervised manner.
Although priors are effective in estimating pose, acquiring them can be difficult. We propose a novel method, named Pose Prior Learner (PPL), to learn a general pose prior for any object category. PPL uses a hierarchical memory to store compositional parts of prototypical poses, from which we distill a general pose prior. This prior improves pose estimation accuracy through template transformation and image reconstruction. PPL learns meaningful pose priors without any additional human annotations or interventions, outperforming competitive baselines on both human and animal pose estimation datasets. Notably, our experimental results reveal the effectiveness of PPL using learned prototypical poses for pose estimation on occluded images. Through iterative inference, PPL leverages the pose prior to refine estimated poses, regressing them to any prototypical poses stored in memory. Our code, model, and data are publicly available at: [link](https://github.com/ZhangLab-DeepNeuroCogLab/Pose-Prior-Learner).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the task of unsupervised categorical prior learning in pose estimation. The authors propose Pose Prior Learner, which is tasked with learning a general pose for a given category. The key design insight invloves the hierarchical memory that stores prototypical poses, from which the general pose is distilled. PPL does not require any annotated supervision, and its performance is improved after a few iterations of inferences.

### Strengths
1. The paper is written clearly.

2. Outperforms the baselines.

3. Design choices are clearly justified and explained. For example:

a. The hierarchical memory component is innovative and clearly justified in Section 3.3, through the 3 points mentioned. 

b. In line 286: while the computation makes sense as a standard/reasonable regularizer, this interpretation gives it another justification.

4. The experiments conducted seem to convey about the effectiveness of the approach. Table 1 shows that the approach outperforms all other baselines. I appreciated lines 360-361 that cancel the temporal effect. In addition, the ablations in Table 2 are a good experiment showing the importance of finetuning the pose (especially with regards to human-defined priors).

### Weaknesses
1. The distinction between pose estimation and prior learning is nor clear to me. If you learn a pose, you implicitly learn a "pose prior". If there is an actual difference please clarify. If not, please consider stating that your approach estimate poses. This distinction challenged my understanding of what your approach does - it might be inferred that your learned prior will be used for pose estimation (lines 67-68 and lines 146-147).  
Or, for example, in lines 70-76, why your approach (or an approach that predicts pose priors) cancels the risk of failure that you mention?

2. Teaser figure: the arrows over-complicate your presentation, and are more relevant to the actual solution and are less relevant to the challenge itself. Maybe simplifying the teaser figure by focusing on the task introduction would help. I don’t think the distinction between blue and red arrows are helpful, especially when it is not clear what is transformed or refined in the stage of observing this figure.

3. Hierarchical memory distillation is mentioned rather early and it is left underexpained untill section 3.3. First, it is suggested to hint earlier that further details are given in this section. Also, I would consider moving it as the beggining of Section 3, as this is the first component and seems like the key design contribution of the work.

4. Using a 2D CNN is considered out-dated in many use-cases. Did you consider using a transfomer-based frozen (or not frozen) foundation model as an image encoder?

5. In line 278 you mention that I_recon should be identical to I, and your loss do not really measure identity (if you did really want identity, why not use MSE which is lighter compared to perceptual loss). Also, can the boundary loss be also considered as regularization?

### Questions
Is there a semantic consistency between the predicted keypoints? That is, if keypoint i is in the head of the person, are we expected to see i on the heads of all other persons?

If so, maybe you should mention it, and visualize the keypoint numbers on top of at least one figure, so readers could see that the model learns semantic keypoints.

### Soundness
4

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
The paper suggests a new unsupervised pose estimation approach using learned pose priors.
It predicts a shared pose prior using a hierarchical memory module.
During inference, it iteratively predicts an affine transformation that adapts the priors to the input image, overcoming difficult scenarios like occlusions.
Training is done through image reconstruction and optimizing the memory bank.

### Strengths
- The paper suggests a new unsupervised pose estimation approach that can learn pose priors to overcome occlusions.
- The authors are willing to publish their code, model, and data.
- The method achieves SOTA performance on multiple datasets compared to similar-sized methods.

### Weaknesses
- The authors should better explain what is the difference between current unsupervised pose estimation methods and "categorical prior learning for pose estimation".
Unsupervised pose estimation learns pose estimation priors over the trained data distribution. It's not clear how this is different from the suggested framing (as described in contribution 1). If the suggested explicit way of describing the learned priors has an advantage over implicit ones, it should be justified.

### Questions
- I find the introduction a bit confusing - framing the suggested method as a prior learning problem, while the rest of the paper discusses pose estimation. I would like to better understand what the authors meant and how it differs from the latter.
- There is a discrepancy between line 917 and Table A3 regarding the performance using 1 memory bank.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes to learn categorical keypoint pose prior via image reconstruction. It proposes a hierarchical memory mechanism to first learn the key point distribution into several different features space, and compute the keypoint prior by decoding the average features in the learned feature space. Then, the connectivity prior is learned jointly with the image reconstruction decoder while the keypoint prior is fixed. The image reconstruction results are refined iteratively during inference. Experiments are done on multiple datasets to demonstrate the effectiveness of this method.

### Strengths
1. Experiments are done on multiple datasets and the performance is strong, which proves the effectiveness of this method.

2. Ablation studies are done properly, showing how the keypoint prior and conectivity prior influences the performance.

3. Learning categorical prior is an interesting and important topic.

### Weaknesses
1. What is the downstream application of the learned prior? All experiments focus on image reconstruction, and since the prior is learned under the supervision of this task, it seems primarily tailored to it. However, it would be interesting to know whether the learned prior can generalize to or benefit other tasks beyond image reconstruction.

2. The evaluation protocol is not clearly described. Is the occluded image used as the input while the clean image serves as the ground truth? The paper should explicitly clarify this setting, along with the specific evaluation metrics used.

3. In Figure A5, the keypoints from AutoLink appear to have clearer semantic meaning compared to those extracted by the proposed method, especially around the upper body. Do the authors have insights into the semantic interpretation or consistency of the learned keypoint priors?

4. The evaluation of the keypoint prior learning process seems insufficient. Including qualitative visualizations of all learned keypoint templates, as well as the average learned prior, would provide valuable insight into the representational quality of the method.

### Questions
Please refer to the weaknesses. Overall, the method appears technically sound. My main concern lies in the lack of discussion or demonstration of potential downstream applications beyond image reconstruction. However, as I am not an expert in this area, I would defer to other reviewers’ opinions regarding the broader applicability and impact of the proposed approach.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces the challenge of unsupervised categorical prior learning in pose estimation and proposes PPL to learn the general pose prior for any object category. The core novelty is to use a hierarchical memory to store the compositional parts of prototypical poses for distillation, achieving improved pose estimation accuracy through template transformation and image reconstruction. The experiments show their effectiveness for pose estimation from occluded images.

### Strengths
1. The paper introduces the challenge of unsupervised categorical prior learning for human/animal pose estimation.
2. The proposed method presents an iterative strategy to progressively refine the estimated poses to the nearest prototypical poses in memory, showing superior performance even in occluded scenes.

### Weaknesses
1. Overstated novelty: The paper presents unsupervised categorical prior learning for pose estimation as a new challenge, but this novelty is not new and has also been addressed in category-level object pose estimation methods with learnable pose priors in self-supervised manner, like in [1][2]. Also, the related work needs to include these categorical prior learning methods to avoid any misleading.
2. Insufficient quantitative evaluation on non-human data. Although the authors claim that their approach generalizes beyond human datasets, only qualitative visualizations are provided in the appendix. More quantitative evaluations on diverse non-human datasets are necessary to support this generalization claim and should be included in the main manuscript.
3. Lack of ablation studies on method components. The paper heavily builds on AutoLink [3] and introduces several modules and four loss terms for optimization, but no ablation experiments are conducted to evaluate their individual contributions. This is crucial to demonstrate the effectiveness and necessity of each proposed design component. I am also interested in both the training and inference efficiency of the proposed method compared to AutoLink. Will the prior learning affect the computational efficiency compared to the prior-free method?
4. The layout and the writing of the paper need to be improved. For example, Table 1 needs to be placed on page 7, Figure 2 and Figure 3 also need to be rearranged to appear near the related content.
5. Lack of description for the used metrics for pose estimation. It is also recommended to include directional arrows for each metric to indicate whether higher or lower values denote better performance.
6. The comparison set includes relatively old methods. To ensure a fair and up-to-date evaluation, more recent approaches such as [4][5] need to be included in the comparison.

[1] Chen, Xu, et al. "Category-level object pose estimation via neural analysis-by-synthesis." European Conference on Computer Vision. Cham: Springer International Publishing, 2020.

[2] Guo, Jiaxin, et al. "A visual navigation perspective for category-level object pose estimation." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.

[3] He, Xingzhe, Bastian Wandt, and Helge Rhodin. "Autolink: Self-supervised learning of human skeletons and object outlines by linking keypoints." Advances in Neural Information Processing Systems 35 (2022): 36123-36141.

[4] Hedlin, Eric, et al. "Unsupervised keypoints from pretrained diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[5] Wang, Dongkai, et al. "Generalizable Object Keypoint Localization from Generative Priors." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
Please refer to the weaknesses. Overall, I feel that the paper is not ready for publication, due to its overstated contribution, insufficient experiments, and limited writing clarity.

### Soundness
2

### Presentation
2

### Contribution
2
