# CryoLVM: Self-supervised Learning from Cryo-EM Density Maps with Large Vision Models

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Cryo-electron microscopy (cryo-EM) has revolutionized structural biology by enabling near-atomic-level visualization of biomolecular assemblies. However, the exponential growth in cryo-EM data throughput and complexity, coupled with diverse downstream analytical tasks, necessitates unified computational frameworks that transcend current task-specific deep learning approaches with limited scalability and generalizability. We present CryoLVM, a foundation model that learns rich structural representations from experimental density maps with resolved structures by leveraging the Joint-Embedding Predictive Architecture (JEPA) integrated with a SCUNet-based backbone, enabling rapid adaptation to various downstream tasks. We further introduce a novel histogram-based distribution alignment loss that accelerates convergence and enhances fine-tuning performance. Through comprehensive evaluation across three critical cryo-EM tasks—density map sharpening, density map super-resolution, and missing wedge restoration—CryoLVM consistently outperforms state-of-the-art baselines across multiple density map quality metrics, confirming its potential as a versatile foundation model for a wide spectrum of cryo-EM applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces CryoLVM, a pretrained model leveraging JEPA self-supervised pretraining. The pretrained model can be further finetuned for downstream tasks of map sharpening, super-resolution, and missing wedge restoration. Experiments on these three tasks show that CryoLVM surpasses baseline methods.

### Strengths
- The paper is well-written and easy to follow.
- The usage of distributional alignment loss is novel and appears to be effective in the ablation study.
- The performance of CryoLVM on all downstream tasks is better than baseline methods, showing the effectiveness of the method.

### Weaknesses
- While the paper claims that the adoption of JEPA, which operates in feature space, is to avoid the low-SNR density. However, the CryoLVM model is trained on Cryo2StructData that consists of densities spanning 1-4 Å. With the high resolution, these densities should not contain much noise.
- While the composite loss is better, the improvement to the MSE loss is marginal, as shown in Table 9.

### Questions
- It’s not clear how the loss for downstream tasks is applied to each of the three tasks. Do they use exactly the same loss or any other choices?
- Have you compared the JEPA training with the MAE training? This could validate the claim of low-SNR protein density.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CryoLVM, the first self-supervised foundation model for cryoEM. By adopting JEPA, the model learns to predict masked regions from surrounding context, thereby training a powerful and generalizable encoder. In addition, the authors propose a histogram-based distribution alignment loss to accelerate convergence and enhance fine-tuning performance. Comprehensive experiments across three downstream tasks demonstrate that CryoLVM achieves state-of-the-art performance, outperforming established baselines such as DeepEMhancer, EMReady, EMGAN, and IsoNet.

### Strengths
(1) The application of JEPA to volumetric cryo-EM data is novel and well-justified. The histogram-based loss is an intuitive addition that addresses data distribution mismatches.

(2) Comprehensive evaluations are conducted across three downstream tasks using multiple cryo-EM quality metrics, and the results consistently surpass previous state-of-the-art baselines.

### Weaknesses
(1) While the paper provides strong evidence that JEPA is effective for cryo-EM representation learning, it would be more convincing to include comparisons with other self-supervised or generative pretraining methods (e.g., VAE- or MAE-based encoders) under identical data settings.

(2) Since the study mainly focuses on pre-training a density encoder, the results would be more solid if the authors compare CryoLVM with a model trained entirely from scratch. This will directly quantify the contribution of the pretraining stage and validate its effectiveness.

### Questions
(1) In the JEPA paper, the authors highlight its scalability. Can CryoLVM continues scaling with more data and more parameters? This would be an interesting topic both for cryoEM and JEPA. 

(2) The motivation of utilizing JEPA for cryoEM is not very clear. Can you show some case study to illustrate what JEPA brings to building foundation models for cryoEM.

(3) What is the motivation of the histogram loss? I understand that it would help calibrate the distribution of density values, but I think it is not very clear how it benefits the map quality, or does it hacks the evaluation metric?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents CryoLVM, a foundation model for cryo-electron microscopy (cryo-EM) 3D density maps. Thanks to the leverage of the Joint-Embedding Predictive Architecture (JEPA) for self-supervised representation learning, it eliminates reliance on hand-crafted augmentations while retaining high-level semantic information. The model employs a SCUNet-based backbone to learn semantically rich representations of cryo-EM density maps. CryoLVM is pretrained on ~7K experimental density maps and evaluated across three major downstream applications: density map sharpening, density map super-resolution, and missing wedge restoration. The results demonstrate consistent performance improvements, with qualitative results showing clearer structural detail recovery and improved interpretability in experimental cryo-EM maps.

### Strengths
1. The adoption of JEPA for volumetric cryo-EM data is a forward-looking design choice. It effectively removes the dependency on task-specific or hand-crafted data augmentations, allowing the model to learn semantic, noise-robust representations directly from experimental density maps.

2. CryoLVM is among the first to systematically explore a foundation modeling approach for cryo-EM 3D density maps. This area previously lacked large-scale pretraining paradigms despite the existence of related efforts for cryo-EM images (e.g., DRACO, CryoFastAR).

3. The authors conduct both quantitative and qualitative analyses on three post-processing tasks, density map sharpening, super-resolution, and missing wedge restoration, demonstrating the adaptability and robustness of the pretrained model.

### Weaknesses
1. While CryoLVM focuses on 3D density maps, related models such as DRACO (Shen et al., NeurIPS 2024) and CryoFastAR (Zhang et al., ICCV 2025) have introduced foundation frameworks for 2D cryo-EM images and pose estimation, respectively. Although these works address distinct aspects of cryo-EM analysis, they fall under the broader category of “foundation models for cryo-EM” and should be acknowledged and positioned as complementary.

2. Designing a foundation model for 3D volumetric data poses significant computational challenges compared to 2D image models due to cubic complexity. The current implementation appears to rely on relatively small input volumes (48³ voxels), which may limit representational granularity. It would be valuable for the authors to discuss how CryoLVM scales to larger 3D contexts or whether insights from other 3D foundation model architectures (e.g., medical imaging) can be leveraged to mitigate this issue.

3. The pretraining corpus includes only ~7K experimental density maps, which is orders of magnitude smaller than typical datasets used to train modern foundation models in vision or language domains. Such a limited dataset raises significant concerns about whether CryoLVM truly demonstrates foundation-level generalization versus task-specific overfitting. The manuscript does not discuss dataset diversity (e.g., coverage of molecular weights, symmetry classes, or reconstruction quality), nor any data augmentation or synthetic data generation strategies to potentially compensate for the scale limitation. 

4. Although the reported results are quantitatively promising, these tasks are not convincingly linked to practical biological or structural analysis workflows. It remains unclear how these improvements translate into measurable benefits for end users such as structural biologists. The authors should provide a more explicit discussion or demonstration of how CryoLVM outputs improve real-world cryo-EM analysis, or consider including more impactful downstream tasks.

### Questions
1. In Figure 1, the “context” and “target voxels” are visualized using 2D patches. I am assuming JEPA here actually operates on 3D volumetric patches. Could the authors clarify?

2. The input generation section mentions cropping to 48³ volumes, while Table 4 lists “patch size = 48” and “voxel box size for input/output.” Could the authors clarify the distinction between the input volume resolution and patch granularity used?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presented CryoLVM, a pretrained foundation model trained on a large scale of cryo-EM density maps, using JEPA and a SCUNet-based backbone architecture. The base model was further finetuned on three different training pairs in a supervised manner, and compare to the baselines for different downstream tasks.

### Strengths
- The application of JEPA in cryo-EM is novel and intuitive. Furthermore, although not the first pretrained model on cryo-EM density maps, it makes great sense to leverage the large scale of cryo-EM maps to help various downstream tasks.

- The paper is well written and most of the presentations are easy to understand and interpret.

### Weaknesses
My main critiques of this paper are mostly on the downstream tasks and evaluations.

- DeepEMhancer and EMReady used different training targets. CryoLVM used the EMReady approach, in which the training targets are simulated maps from atomic models. Strictly speaking, this should not be called as "sharpening" anymore, since the training targets are not sharpen maps but simulated maps.

- It seems that the training set used in the sharpening and super-resolution task also differs from that of the baselines. The authors should at least provide a comparison with EMReady, using the exact same training data to finetune the pretrained base model, and compare the results with EMReady.

- Additionally, I do not believe that "super-resolution" should be considered as an distinct job type. The goal of map postprocess, whether sharpening or not, is all to help model building. The "super-resolution" term here and in the EM-GAN paper, is actually to make the "resolution" higher when generating the simulated maps, which is very similar to that of EMReady. Therefore, I do not see the great difference between the two tasks.

- About the missing wedge task: IsoNet was introduced to be applied on the tomograms, but it seems the experiments were performed on a "subtomogram-like" setting. Also in Fig. 5, the example is not very clear and the difference seems subtle. Could the authors share more visual examples, showing the input (with the missing wedge artifact), and the outputs from both methods?

- Since I think both "sharpening" and "super-resolution" do not have much difference and should be both under the same umbrella of map postprocessing, it becomes a bit of overclaim for CryoLVM to be a foundation model.

### Questions
- I believe that the same model architecture can be applied to task-specific training from scratch, instead of fine-tuning from a pretrained base model, using the same training pairs in the finetuning. Could the authors show the ablation study of this?

- In postprocess (Appendix D.2), what does the normalization "to remove biases resulting from uneven sampling" mean (L806-807)? For at least one task (maybe sharpening), could the authors ablate the postprocess (e.g. replace Gaussian-weighted fusion with a simple overlapping average, and if possible, remove the additional normalization step)?

### Soundness
2

### Presentation
3

### Contribution
3
