# IPR-NeRF: Ownership Verification Meets Neural Radiance Field

- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
Neural Radiance Field (NeRF) models have gained significant attention in the computer vision community in the recent past with state-of-the-art visual quality and produced impressive demonstrations. Since then, technopreneurs have sought to leverage NeRF models into a profitable business. Therefore, NeRF models make it worth the risk of plagiarizers illegally copying, re-distributing, or misusing those models. This paper proposes a comprehensive intellectual property (IP) protection framework for the NeRF model in both black-box and white-box settings, namely IPR-NeRF. In the black-box setting, a diffusion-based solution is introduced to embed and extract the watermark via a two-stage optimization process. In the white-box setting, a designated digital signature is embedded into the weights of the NeRF model by adopting the sign loss objective. Our extensive experiments demonstrate that not only does our approach maintain the fidelity (i.e., the rendering quality) of IPR-NeRF models, but it is also robust against both ambiguity and removal attacks compared to prior arts.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Since intellectual property rights (IPR) matter for NeRFs, this paper argues that the proposed method has high fidelity, effectiveness, and robustness against adversarial activities and situations. For the black-box protection, they effectively incorporate a diffusion method to generate a watermark from novel-view generations of NeRFs. For the white-box protection, they propose to use the signs of normalization layers in the NeRF model. They experimentally validate their method in aspects of fidelity, ownership verification with black/white-box schema, resilience against ambiguity attacks, and robustness against removal attacks, including model pruning and overwriting.

### Strengths
- The extensive validation confirms various aspects of watermarking and leaving and extracting signature methods. 

- Robustness toward image degradation and forged signature seems to be promising.

### Weaknesses
- W1. Novelty and effectiveness. The Black-box watermark extraction method is a straightforward way to apply diffusion models. However, it relies on a catastrophic performance on unseen samples, only empirically validated on samplings (Sec 4.3). The below question Q2 raises an issue of false positive detection.

- W2. Clarity on the white-box method. The author did not clearly state where are the normalization layers in the NeRF model. As raised in question Q1, the readers cannot pinpoint the normalization layers in the original NeRF model, so reproducibility is unclear.

### Questions
- Q1. Which ones are the normalization layers in the NeRF model? The author states that they used the original implementation in Mildenhall et al. (2020), but their implementation does not have any normalization layers.

- Q2. In Fig 5, a substantial portion of the protected model (red) and random images (blue) are overlapping. How do you say that they are significantly different?

- In Sec 3, Para 2, by initiate -> by initiating

- Appendix A.5, to qualitatively -> to quantitatively

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper suggests an intellectual property protection framework for the NeRF model in both black-box and white-box settings. IPR-NeRF used a diffusion-based model to embed and extract watermark images for black-box ownership protection. In addition, it adopted a technique from DeepIPR method to embed a textual signature into the NeRF for a white-box protection scheme.

### Strengths
1. To the best of my knowledge, It first proposes to embed both watermarking and textual signature simultaneously.
2. Using diffusion models seems to be quite effective compared to the previously suggested simple decoder-based methods.
3. Various experiments to show the robustness of the proposed scheme make the paper more convincing.

### Weaknesses
1. My major concern is that It lacks the original technical contribution. It is interesting to see that the diffusion models (DDIM) are quite effective in this task. However, I think it may not be a sufficient contribution to be a full conference paper. 
2. For textual embedding, the authors adopted the technique from DeepIPR work into neural fields. It is also useful information to the community, but I still think the direct application to the well-defined neural network architecture would not be a sufficient technical contribution.
3. I wonder if the DDIM method can generate high-fidelity images. The comparison to the previous method, e.g., staganerf would be great.

### Questions
Questions are embedded in the weakness section.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents an intellectual property protection framework for NeRF models in black-box (i.e., w/o access to model weights) and white-box (i.e., w/ access to model weights). Specifically, the authors adopt a diffusion-based model for embedding and extracting watermark in rendered images by NeRF, in the black-box setting. A digital signature is embedded into NeRF model weights for the white-box setting. The experiments on LLFF-forward and NeRF-Synthetic datasets verify that the protection framework can identity the protected NeRF without degrading the rendering quality.

### Strengths
1. Well-motivated: Since NeRF-based 3D reconstruction is more and more easy-to-use, people are sharing their NeRF models on web. Thus, it is worth exploring how to alleviate copying, re-distributing, or misusing those models.

2. Impressive Results: The baseline works can protect NeRF models but degrade the rendering quality obviously. This work nearly does not affect the rendering quality.

3. Easy-to-follow draft: the draft is well-written and the figures are easy-to-understand.

### Weaknesses
1. Scalability to explicit NeRF representations: To accelerate NeRF inference and rendering, multiple works [1,2,3] have proposed to use explicit representations (e.g., grid, mesh, and point cloud) instead of MLP as the NeRF representations. Modifying the weights of explicit NeRF representations seems to have a larger effect on the rendering quality as compared to implicit representations because it can be regarded as changing the location/color of the grid/mesh/point cloud. Thus, it is not sure whether the proposed protection framework can still maintain the rendering quality of those explicit NeRF representations.

2. Insufficient details in "False Positive Detection Prevention": In Sec. A.5, the dataset used in this experiment is not mentioned. Does Fig. 5 show the averaged histogram for different scenes in a specific dataset or the histogram for a specific scene? The unprotected standard NeRF is claimed to be trained in the same dataset as a protected one. Is it trained on the same scene or a different scene but in the same dataset?

3. Will the dataset affect the selection of ownership verification? A threshold of 0.75 is set for the black-box ownership verification, and the author claimed that 0.75 is selected "as the visibility is still evident." However, it is uncertain whether different datasets have different optimal thresholds. For example, scenes in Unbound-360 [4] or ARKitScene [5] is more close to the real-world applications [6,7]. Will the optimal threshold still be 0.75 on those datasets?

[1] https://creiser.github.io/merf/

[2] https://mobile-nerf.github.io/

[3] https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

[4] https://jonbarron.info/mipnerf360/

[5] https://github.com/apple/ARKitScenes

[6] https://poly.cam/gaussian-splatting

[7] https://lumalabs.ai/interactive-scenes

### Questions
See weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents IPR-NeRF, an intellectual property protection framework for NeRF models. It offers protection in both black-box and white-box settings. In the black-box approach, a watermark is embedded and extracted using a diffusion-based method. In the white-box scenario, a digital signature is incorporated into the NeRF model's weights using a sign loss objective. The experiments show that IPR-NeRF maintains rendering quality while being robust against ambiguity and removal attacks, providing a solution to safeguard NeRF models from unauthorized use and distribution.

### Strengths
- I think the paper investigates an important and interesting topic in the 3D vision community.
- The paper is well-written and easy to follow.
- The comprehensive experimental results significantly demonstrate the benefit of the proposed method.

### Weaknesses
- The motivation for using the diffusion model to learn black-box protection is unclear. It would be great if the authors could provide more elaboration.

### Questions
Major:
- Is there any simple alternative solution for black-box protection? If so, could the author provide some comparisons?
- Just out of curiosity, is there any reason that StageNeRF is significantly vulnerable to Gaussian noise?

Minor:
- Enlarging the font in the figures could be helpful.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
