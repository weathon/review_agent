# PBCAT: Patch-Based Composite Adversarial Training against Physically Realizable Attacks on Object Detection

- Decision: Reject
- Scores: 5, 6, 6, 3

## Abstract
Object detection plays a crucial role in many security-sensitive applications, such as autonomous driving and video surveillance. However, several recent studies have shown that object detectors can be easily fooled by physically realizable attacks, \eg, adversarial patches and recent adversarial textures, which pose realistic and urgent threats. Adversarial Training (AT) has been recognized as the most effective defense against adversarial attacks. 
While AT has been extensively studied in the $l_\infty$-bounded attack settings on classification models, 
AT against physically realizable attacks on object detectors has received limited exploration. 
Early attempts are only performed to defend against adversarial patches, leaving AT against a wider range of physically realizable attacks under-explored.
In this work, we consider defending against various physically realizable attacks with a unified AT method. 
We propose PBCAT, a novel Patch-Based Composite Adversarial Training strategy. PBCAT optimizes the model by incorporating the combination of small-area gradient-guided adversarial patches and imperceptible global adversarial perturbations covering the entire image. With these designs, PBCAT has the potential to defend against not only adversarial patches but also unseen physically realizable attacks such as adversarial textures.
Extensive experiments in multiple settings demonstrated that PBCAT significantly improved robustness against various physically realizable attacks over state-of-the-art defense methods. Notably, it improved the detection accuracy by 29.7\% over previous defense methods under one recent adversarial texture attack.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors introduce a patch-based adversarial training technique designed to improve the robustness of object detection models against both patch-based and more recent texture-based attacks. The method involves two types of perturbations: local perturbations applied to the attacked object and a global perturbation affecting the entire image. The global perturbation is aimed at enhancing the robustness against texture-based attacks. In their evaluation, the authors compare their technique to one adversarial training (AT) approach and several non-AT methods across three patch-based attacks. They also present ablation studies to assess the impact of various hyperparameters. Finally, the evaluation is extended to other object detection models to demonstrate the method's broader applicability.

### Strengths
1. The paper addresses a practical yet underexplored topic: adversarial training for defending object detection models against realizable attacks.
2. The evaluation setup is well-detailed, and the provided code ensures easy reproducibility.
3. The proposed method achieves excellent performance in terms of adversarial robustness.

### Weaknesses
1. Incomplete literature review – while the authors state that there are no previous works that specifically propose patch-based AT for object detection, a more in-depth review of the literature would have revealed that techniques such as Ad-YOLO [1] and PatchZero [2] already exist (and should be compared to). Additionally, including comparisons to more recent non-AT methods (e.g., PatchBreaker [3], NAPGuard [4]) would strengthen the paper's overall contribution.‏

2. Lack of novelty – The proposed method appears relatively simple, primarily combining existing techniques adapted for object detection without introducing substantial new contributions, aside from the patch partitioning and selection strategy.

3. Experiments - While the authors conduct a relatively comprehensive evaluation, several aspects are lacking:

* Models: Since the focus is on person detection, which typically involves real-time scenarios, the evaluation should prioritize low-latency models (e.g., one-stage detectors) rather than slower ones like Faster R-CNN. Including YOLO models, particularly the most recent versions, would have been more relevant, as they are widely used in real-time object detection.
* "Clean" results: While the authors acknowledge the performance drop on clean images as a limitation, the degradation in accuracy is significant, especially when compared to (Li et al. 2023) in Tables A1, 5, and 6. This raises concerns about whether the improved robustness stems from a robustness-accuracy trade-off. A more fair comparison would require matching the AP on clean images across methods before assessing robustness. 
* Results discussion: The results are presented with limited interpretation. The discussion would benefit from addressing edge cases and explaining unintuitive findings (as highlighted in question 4 below).

4. Presentation - the submission is held back by the writing quality, particularly in the method section, mainly focused around the partially existing formulaic descriptions. For instance, the number of selected sub-patches should be parametrized (with an accompanying equation or algorithm) to better align with the presentation of the ablation study in Section 4.3.2.

Minor comments:
- Algorithm 1 – the use of $m$ and $m_p$ is confusing.
- The placement of the tables on Page 9 makes them hard to read.
- Best “Clean” performance should also be marked with bold.

[1] Ji, N., Feng, Y., Xie, H., Xiang, X., & Liu, N. (2021). Adversarial yolo: Defense human detection patch attacks via detecting adversarial patches. arXiv preprint arXiv:2103.08860.

[2] Xu, K., Xiao, Y., Zheng, Z., Cai, K., & Nevatia, R. (2023). Patchzero: Defending against adversarial patch attacks by detecting and zeroing the patch. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 4632-4641).

[3] Huang, S., Ye, F., Huang, Z., Li, W., Huang, T., & Huang, L. (2024). PatchBreaker: defending against adversarial attacks by cutting-inpainting patches and joint adversarial training. Applied Intelligence, 54(21), 10819-10832.

[4] Wu, S., Wang, J., Zhao, J., Wang, Y., & Liu, X. (2024). NAPGuard: Towards Detecting Naturalistic Adversarial Patches. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 24367-24376).

[5] Liu, X., Yang, H., Liu, Z., Song, L., Li, H., & Chen, Y. (2018). Dpatch: An adversarial patch attack on object detectors. arXiv preprint arXiv:1806.02299.

‏

### Questions
1. It is unclear whether the patch selection calculation (i.e., the $l_2$ norm calculations) is performed on the clean image or the adversarial image (the one containing the adversarial patch).
* Could you please clarify this?
* Additionally, what is the rationale behind choosing a square-shaped mask?
* Have you considered experimenting with different norms beyond the $l_2$ norm?

2. In the model training, are the weights initialized to random values or pre-trained weights? If random initialization is used, the object detector may risk overfitting on the Inria dataset, which contains only a few hundred images. This could explain the inconsistencies observed between the results on MS-COCO and Inria.

3. In line 345, the total number of sub-patches is set to $n^2=64$ , and in lines 238-239, you mention that the top half are selected, indicating that 32 patches are chosen. However, in the ablation study regarding the number of sub-patches used during the selection process (Table 3), only a single value (16) is presented as a portion of the sub-patches, since using 64 means utilizing the entire set. This leads me to infer that 16 is deemed the optimal value. Does using 32 sub-patches result in better performance? It would be beneficial to explore additional values in this experiment.

4. Could you provide some insights into the results presented in Table 2, particularly concerning the "Global" component? I find it challenging to understand why the "Global" component enhances robustness against AdvTexture and AdvCat attacks, given the significant differences in perturbation styles between them. Additionally, why does robustness decrease against AdvTexture when the Patch and Partition components are added (Lines 3 and 4)?

5. Following the above question, in line 465 it is stated that “Partition” denotes the patch partition strategy. What is the strategy other than “Gradient”? what does Line 4 in Table 2 mean?

5. While I acknowledge that the paper focuses on patches attached to objects, it would be beneficial to evaluate the proposed approach against attacks that place patches in different locations (e.g., DPatch [5]) and to study the effect of the "Global" component on these attacks. Demonstrating the ability to mitigate the impact of such patches could significantly enhance the paper's contributions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a novel adversarial training method designed to defend against various physically realizable attacks on object detection tasks. The perturbation for generating adversarial examples during training includes a global perturbation, constrained by an 
ℓ-inf norm with a small budget applied across the entire image, and a local patch, randomly positioned within the bounding box. This local patch is composed of sub-patches, with only some selected to inject a larger budget constraint.

### Strengths
- As remarked by different experiments, the proposed method increases the robusteness over different attacks. 

- Overall I think that the results are quite intersting, it provides a quite large gap above other strategies.

### Weaknesses
- The approach may impact accuracy sometime, especially when dealing with large datasets like COCO, as shown in Table 5. However, the effectiveness in terms of improved robustness is noteworthy.

- The authors could have added metrics on training costs in the table to better clarify possible efficiency with respect to other training strategies

### Questions
- The authors mention physically realizable attacks that extend beyond adversarial patches. Why should these represent distinct attacks if they are computed to fool the same model? Adversarial patches could potentially encompass also features of an adversarial t-shirt, as they are capable of generalizing and representing any potential adversarial texture. For instance, at the end of Section 2.3, the authors suggest that real-world adversarial patches may not generalize well to other types of physical attacks, why?

- The adversarial training is applied only for inf-norm bounded attacks. It would be interesting to explore SOTA patch and texture attacks bounded on different norms. What about the robustness against L-2 Attacks? How much it is the model cpaable of etending the robustness against L-2 Attacks?

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
4

### Summary
The authors propose a adversarial training method to defend against physically realizable attacks. Specifically, they propose a new adversarial patch attack and use them to train the model.

### Strengths
- The method is simple and effective.
- The experimental results and ablation studies are convincing.

### Weaknesses
- It is curious that the proposed methods work for naturalistic patch attacks. Experiments on defending naturalistic patch attack will strengthen the paper.
- No black-box experiments are conducted. For example, FastRCNN trained with the proposed method against different datasets and attacks using other surrogate models such as Yolo.
- Hyper-parameter tuning and training time is a concern

### Questions
See the Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
Early efforts have primarily focused on defending against adversarial patches, leaving adversarial training (AT) against a broader range of physically realizable attacks underexplored. In this work, the authors address this gap by proposing a unified AT method to defend against various physically realizable attacks. They introduce PBCAT, a Patch-Based Composite Adversarial Training strategy, which optimizes the model by combining small-area gradient-guided adversarial patches with imperceptible global adversarial perturbations that cover the entire image. This design enables PBCAT to defend not only against adversarial patches but also against unseen physically realizable attacks, such as adversarial textures. Extensive experiments across multiple settings demonstrate that PBCAT significantly enhances robustness against various physically realizable attacks compared to state-of-the-art defense methods.

### Strengths
1.	The topic studied in the paper is practical.
2.	The proposed method demonstrates a degree of generalization, as it does not rely on specific attack algorithms.
3.	The proposed method is effective against common adversarial attack algorithms.
4.	The experiments conducted are relatively comprehensive.

### Weaknesses
1. The paper lacks novelty.
2. The authors should emphasize why standard adversarial training cannot effectively address physically realizable attacks and highlight the advantages of the proposed method presented in this paper. 
3. In lines 251-253, the authors' findings seem meaningless, as unlimited adversarial noise will inevitably lead to a decline in training performance.
4. Although the training cost of PBCAT is comparable to that of standard training, it still demands additional computational resources due to the gradient post-processing steps (partial partitioning and selection).

### Questions
1. What are the differences between square adversarial patches and physically realizable attacks? 
2. Why is it necessary to design defense algorithms specifically for these attacks, and what are the limitations of existing defense methods ?
3. What is the purpose of designing a binary mask? Could you please explain?
4. The location of the mask is randomly selected, and then gradient information is used to determine the final patch. What is the difference between this approach and selecting the mask first followed by a random selection of the patch? Is there any advantage to this method ?
5. Why is the adversarial training method presented in this paper inferior to L_\infty-bounded adversarial training when applied to clean data?

### Soundness
4

### Presentation
4

### Contribution
3
