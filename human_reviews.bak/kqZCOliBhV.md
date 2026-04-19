# Leveraging Side Information with Deep Learning for Linear Inverse Problems: Applications to MR Image Reconstruction

- Decision: Reject
- Scores: 5, 6, 3, 6

## Abstract
Reducing the time it takes to acquire a Magnetic Resonance Imaging (MRI) scan is an important problem in healthcare, as it can improve patient care and reduce costs. One way to achieve this is by acquiring only a fraction of the frequency space data and reconstructing diagnostic-quality images from it. This problem can be formulated as a linear inverse problem (LIP), where the forward operator, which maps the structure of the imaged object to the acquired frequency space data, can become rank-deficient or exhibit many small singular values. This leads to ambiguities in the reconstruction process, where multiple images (most of them non-diagnostic) can map to the same set of acquired data. To resolve these ambiguities, it is essential to leverage domain knowledge and, whenever possible, exploit additional context (a.k.a., relevant side information) when solving the LIP. We present a novel, end-to-end trainable deep learning-based method, called Trust-Guided Variational Network (TGVN), that reliably incorporates side information into LIPs to eliminate undesirable solutions from the ambiguous space of the forward operator, while remaining faithful to the acquired data. We demonstrate its effectiveness through applications in multi-coil, multi-contrast MR image reconstruction, where incomplete or low-quality measurements from one contrast are used as side information to reconstruct a high-quality image of another contrast from heavily under-sampled data. Its robustness is validated by reconstructing images from different contrasts across different anatomies and field strengths. Compared to a set of baselines that also use side information, our method reconstructs high-quality images in the presence of heretofore challenging levels of under-sampling, thereby speeding up the acquisition drastically while providing protection against hallucinations. Our approach is also versatile enough to incorporate many different types of side information into any LIP.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a novel deep learning-based method called Trust-Guided Variational Network. This method addresses the problem of reconstructing high-quality MRI images from undersampled frequency space data. The primary challenge in MRI reconstruction is that undersampling results in ambiguities due to rank-deficient forward operators, leading to multiple non-diagnostic solutions. TGVN integrates relevant side information to constrain the solution space and eliminate undesirable solutions, enhancing image quality.

### Strengths
1. This paper incorporates side information into the MRI reconstruction process. It introduces an "ambiguous space consistency constraint" to tackle the ill-posed nature of the problem.
2. The paper is generally clear, with well-organized sections on background, methodology, and empirical validation. The explanations of the core concepts, such as the linear inverse problem and the ambiguous space consistency constraint, are detailed and easy to follow for readers familiar with the domain.

### Weaknesses
1. limitation of experiments in terms of the methods that have been compared, please consider comparing some other famous methods such as VN, E2E VarNet.

2. Imposing additional constraints is not a new topic. For example, the paper: Bian, Wanyu, et al. "A learnable variational model for joint multimodal MRI reconstruction and synthesis." is also an optimization-based recon model that utilizes the multicontrast information for enhancing performance. Authors should cite some related paper that leveraging additional contrast.

3. Since it is a model based recon paper, authors need to add optimization based deep learning methods such as ADMM-net, ISTA-net, in the related work. End-to-end and generative methods are not too close to the topic.

4. Authors didn't provide a network structure or whole framework of this work.

### Questions
1. From the figure 3 and 12, DMSI provides more details of the tissue and close to the ground truth. The proposed method smooth out some features of the knee joint, fat part. 

2. Please indicate the box of the zoom in region in all of the figures. 

3. Conduct a doctor study for evaluating the performance of these compared results should strengthen  robustness of the study.

4. In equation 3 and 4, author didn't explain what is \phi.

5. H(s, \gamma) is assumed to be learnable, and s is fixed variable, how does author choose s? Why not updating s iteratively with s^t and x^t alternatively?

### Soundness
3

### Presentation
2

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
This paper proposes a Trust-Guided Variational Network (TGVN), an end-to-end deep learning (DL) method for accelerating MRI reconstruction. Existing physics-based unrolled networks face the problem of ambiguities in the solution space, degrading reconstruction image quality. By introducing the concept of ambiguous space consistency, this work proposes to incorporate side information (i.e., diverse modalities here) into unrolled DL frameworks for further enhancing MRI reconstruction. This approach appears technically and theoretically sound. And extensive experiments confirm the effectiveness of the proposed TGVN model.

### Strengths
**Motivation and Novelty**: This work introduces the concept of ambiguous space consistency to address ambiguities in the solution space of existing unrolled methods and proposes using side information to resolve this issue. Although the work largely follows the general framework of unrolled methods, its novelty meets the standards for ICLR conference publication.

**Clarity and organization**: This paper is well-written and easy to follow.

**Experimental evaluation**: This work conducts extensive experiments on two MRI datasets to validate the proposed method, including: 1) comparisons with other unrolled methods without side information and with existing side-information-based methods, and 2) ablation studies on key components of the proposed method. The results confirm the effectiveness of the TGVN model.

### Weaknesses
**Clarity**: In this work, ambiguous space consistency is a key concept. Providing some intuitive explanations would improve readability.

**Type of side information**: In this work, different MR image modalities are used as side information. I wonder if the differences between the main information (i.e., measurements) and side information could significantly affect model performance. More specifically, could complementary main and side information (e.g., T1w and T2w) better improve performance? If so, this use of side information appears similar to multi-modality MRI reconstruction. Please clarify the differences between the proposed approach and existing multi-modality MRI reconstruction methods.

**Compared methods**: The proposed method follows a supervised learning paradigm, requiring extensive high-quality MRI images for pre-training. However, state-of-the-art MRI reconstruction methods are not included, such as supervised Transformer-based methods [1][2][3] and unsupervised diffusion methods [4][5].

**Generalization evaluation on different datasets**: On the two datasets, the training, validation, and test are conducted independently. I wonder if the use of side information can enhance the model's generalization to out-of-domain data.

**Under-sampling mask**: The under-sampling masks used in this work are based on Cartesian sampling. Please evaluate the model performance on other types of sampling, such as radial sampling.

**Experimental results**: 

1.	I observe that the performance of the E2E VarNet in Exp. 3 is significantly decreased compared to its performance in Exp. 1 and 2. However, other baselines and the proposed method do not show this trend. Please explain this result.

2.	In the ablation studies, only the SSIM metric is reported. Please provide other two metrics (PSNR and NRMSE).

> [1] Guo, Pengfei, et al. "ReconFormer: Accelerated MRI reconstruction using recurrent transformer." IEEE transactions on medical imaging (2023).

> [2] Huang, Jiahao, et al. "Swin transformer for fast MRI." Neurocomputing 493 (2022): 281-304.

> [3] Feng, Chun-Mei, et al. "Task transformer network for joint MRI reconstruction and super-resolution." Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part VI 24. Springer International Publishing, 2021.

> [4] Chung, Hyungjin, and Jong Chul Ye. "Score-based diffusion models for accelerated MRI." Medical image analysis 80 (2022): 102479.

> [5] Peng C, Guo P, Zhou S K, et al. Towards performant and reliable undersampled MR reconstruction via diffusion model sampling[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2022: 623-633.

### Questions
See the Weaknesses above, please.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper deals with solving the inverse problem in magnetic resonance imaging by leveraging domain knowledge. Therefore, the authors introduce the so-called Trust-Guided Variational Network (TGVN).  Despite the data consistency where the reconstructed images should align with the measurements the ambiguous space consistency is included which ensures that the projected data aligns with the side information.

### Strengths
* The idea to incorporate side information is good. Reducing acquisition time and therefore gathering undersampled data which leads to artifacts in the reconstruction process will require sophisticated but trustable reconstruction algorithms as MRI is a versatile application.
* “Easy” integration of the ambiguous space consistency constraint.
* The approach is end-to-end trainable.

### Weaknesses
* There exist already many attempts to incorporate side information.
* The paper includes a lot of general information in the paper but a quite short explanation of the novel work.
* In this paper, four research questions are mentioned but only two are discussed whereby also the other ones, especially Q3, would be also considered as important as it is part of the main contribution of this paper. The general stuff should be shortened and a focus of the new method and the results should be 
* The authors advertise that different types of side information can be included but not exactly how. Furthermore, it is mentioned that this is part of future work 
* The generalizability is not shown across different anatomies, as stated. The brain and knee dataset are handled separately.
* A lot of stuff is described generally but the key points are missing.

### Questions
* Why should we use this approach and not another one to incorporate side information? What is the benefit of this method (beyond more feasible solutions which is the goal when incorporating side information)?
* How does the threshold effect the results?
* Which projection was chosen?
* How is the module learned?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper with title: Leveraging Side Information with Deep Learning for Linear Inverse Problems: Applications to MR Image Reconstruction presents a Deep learning-based framework that can leverage the information from other contrast to help the reconstruction, the key framework uses a Null space projection method to disentangle the side information. Results on fastMRI dataset show improved image quality compared to other SOTA methods.

### Strengths
1. Using prior knowledge from other contrast is a very interesting topic in terms of MRI reconstruction since for clinical routine, usually multiple contrasts are acquired. 

2. Using Null-space to disentangle the contrast-information is interesting.

### Weaknesses
1. The presentation and story telling is not clear and there isn't any figure illustration of the problem/method/novelty, which makes the paper very difficult to follow, especially the null-space and theory part. 

2. Using multi-contrast information for joint reconstruction has been a popular topic, but the authors didn't compare with any-of-them.

3. Following the previous weaknes, instead of using Null-space, one way to extract information from side contrast is by learning an representation, you should incorporate this experiments - like for example in E2E experiment, try inputing two channels.

4. Instead of using aliased side information (3x undersampled image from side contrast), it makes sense to use a reconstructed version of other contrast, like using E2E, have you done this experiments?

5. ~20x undersmapling doesn't make sense at all, in practice, or in fastMRI dataset, only a few lines are sampled beside the ARC region, how do you expect recovering any details? especially for pathologies.

### Questions
1. Do you use paired data for training - like the main and side contrasts from the same object/ same anatomy? how do you do the alignment?

2. I need more ablation study to understand how the side information contributes.

### Soundness
3

### Presentation
3

### Contribution
2
