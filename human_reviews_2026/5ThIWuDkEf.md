# Anatomy-aware Representation Learning for Medical Ultrasound

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Diagnostic accuracy of ultrasound imaging is limited by qualitative variability and its reliance on the expertise of medical professionals. Such challenges increase demand for computer-aided diagnostic systems that enhance diagnostic accuracy and efficiency. However, the unique texture and structural attributes of ultrasound images, and the scarcity of large-scale ultrasound datasets hinder the effective application of conventional machine learning methodologies. To address the challenges, we propose Anatomy-aware Representation Learning (ARL), a novel self-supervised representation learning framework specifically designed for medical ultrasound imaging. ARL incorporates an anatomy-adaptive Vision Transformer (A-ViT). The A-ViT is parameterized, using the proposed large-scale medical ultrasound dataset, to provide anatomy-aware feature representations. Through extensive experiments across various ultrasound-based diagnostic tasks, including breast and thyroid cancer, cardiac view classification, and gallbladder tumor and COVID-19 identification, we demonstrate that ARL significantly outperforms existing self-supervised learning baselines. The experiments demonstrate the potential of ARL in advancing medical ultrasound diagnostics by providing anatomy-specific feature representation

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Anatomy-Aware Representation Learning (ARL), a self-supervised framework for ultrasound (US) imaging that integrates anatomical context into representation learning. The authors propose Anatomy-aware Vision Transformer (A-ViT), which incorporates an Anatomy-Conditioned Deformable Transformer (ACDT) to extract features according to the organ being analyzed. The authors evaluate ARL across multiple downstream tasks, including breast, thyroid, and gallbladder cancer classification, cardiac view classification, cardiac segmentation, and COVID-19 diagnosis. Experiments show that ARL consistently outperforms state-of-the-art (SoTA) self-supervised methods.

### Strengths
The paper introduces a foundation model for ultrasound imaging, trained on 5.2M images, covering 16 anatomical categories.

The model is tested on six downstream tasks across both classification and segmentation. 

Quantitative results demonstrate consistent gains over state-of-the-art baselines.

### Weaknesses
The authors emphasizes speckle noise and low color variation as major challenges unique to ultrasound imaging. However, the proposed A-ViT primarily introduces anatomy-conditioned deformable attention, which addresses anatomical variability rather than these low-level texture or color issues. The authors introduces an adversarial term to preserve high-frequency content. However, the paper lacks direct evidence that speckle-related distortions are mitigated or that low-color variation are effectively handled. 

Anatomy-aware conditioning was previously proposed in MRI/CT segmentation & registration. Deformable attention exists in general vision. MAE + distillation hybrid ideas has also been previously proposed. Therefore the paper doesn’t introduce a new model, but rather a new instantiation tailored to ultrasound images. However, there is limited explicit explanations on how each architectural choice is uniquely designed for ultrasound or directly tied to ultrasound physics. Most of the mechanisms introduced in A-ViT are general and could apply to other modalities. As a result, the authors should better explain the novelty by explicate how A-ViT differs from existing models for other modalities. 

In Tables 2 and 3, the proposed model does not have the highest specificity in some cases and the highest specificity values are not bolded correctly

### Questions
1. Are there evidence that the proposed model addresses speckle noise and low color variation challenges in ultrasound imaging.

2. How is the proposed model differed, comparing to existing models for other modalities? Is there any architectural choice that is uniquely designed for ultrasound or directly tied to ultrasound physics?

### Soundness
2

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
This paper proposes Anatomy-aware Representation Learning (ARL), a self-supervised representation learning framework tailored specifically for medical ultrasound imaging. The authors identify key challenges in ultrasound diagnostics, including qualitative variability, reliance on expert knowledge, the unique texture/structural properties of the images, and the scarcity of large-scale datasets. The ARL framework is intended to address these issues. The work is evaluated by fine-tuning a Vision Transformer encoder and classification head on a diverse set of medical downstream tasks, including breast, thyroid, and gallbladder cancer classification, COVID-19 identification, and cardiac view classification, as well as a dense prediction task (echocardiography left ventricle blood-pool segmentation) using a UPerNet configuration.

### Strengths
1.The integration of anatomical conditioning into deformable attention for US is a novel and well-motivated architectural contribution. Unlike prior SSL methods that treat all images uniformly, ARL explicitly conditions feature extraction on anatomical context.
2.The 5.2M-image dataset is a significant contribution, and the ablation studies (Table 3, Fig. 4) convincingly isolate the impact of each component (ACDT, adversarial loss, etc.).
3.The paper is well-structured and clearly written, with intuitive figures (Fig. 1, 2, 3) and logical flow.
4.Medical US is an underexplored modality in SSL, and ARL provides a strong foundation for future work. The dataset alone will likely become a community resource.

### Weaknesses
1.The central weakness is the lack of a clear description of the Anatomy-aware Representation Learning (ARL) mechanism itself. While the goal is anatomy-aware learning, the specific self-supervised loss function or task design that enforces this awareness is not described, making it impossible to fully assess the work's technical depth or novelty.
2.The submission provides an extensive experimental plan but no quantitative results (e.g., AUC, F1, Dice Score). Without empirical evidence, the claim of technical soundness and significance remains unverified. It is impossible to determine if the proposed ARL method actually advances the state-of-the-art or even works as intended.
3.The paper claims to address the unique texture and structural attributes of ultrasound images. However, without detailing how ARL achieves this (i.e., the mechanism of anatomy-awareness), its true originality over existing self-supervised methods (like SimCLR, MAE, etc.) applied to medical imaging cannot be confirmed.

### Questions
1.Please fully describe the proposed Anatomy-aware Representation Learning (ARL) framework. What are the specific self-supervised tasks or loss functions that compel the model to learn "anatomy-aware" features? How do these tasks specifically leverage or model the unique texture and structural attributes of ultrasound images better than standard methods?
2.Please provide the full set of quantitative results (e.g., AUROC, F1-Score, Dice Similarity Coefficient) for all mentioned downstream tasks (classification and segmentation), and critically, compare your method against strong baselines such as ImageNet pre-trained models and non-anatomy-aware self-supervised learning methods applied to ultrasound data.
3.Did the authors perform an ablation study to justify the anatomical-aware component? Showing results without the 'anatomy-aware' loss/mechanism would be critical evidence for the necessity and effectiveness of the proposed novelty.
4.How are anatomical labels obtained for the 5.2M pretraining images? Are they derived from metadata, manual annotation, or automated prediction? If the latter, could label noise degrade representation quality?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an anatomy-aware representation learning framework (ARL) for medical ultrasound (US) imaging, based on a large-scale US dataset and a novel Anatomy-aware Vision Transformer (A-ViT). The method incorporates anatomical context via a deformable transformer and combines multiple self-supervised objectives to improve feature learning. Extensive experiments across multiple downstream tasks demonstrate improved performance over existing self-supervised learning baselines.

### Strengths
1.The paper introduces a large-scale, multi-source US dataset, which is a valuable contribution to the community.

2.The proposed A-ViT model effectively integrates anatomical information and shows consistent improvements across diverse US tasks.

3.The combination of MIM, adversarial loss, and self-distillation is well-motivated and empirically validated.

4.Comprehensive evaluation on multiple organs and tasks (classification and segmentation) strengthens the claim of generalizability.

### Weaknesses
1.The motivation for choosing certain design elements (e.g., deformable attention, specific loss weighting) could be better justified.

2.The comparison to other anatomy-aware or medical-specific transformers is limited.
[1]Anatomy-Aware Contrastive RepresentationLearning for Fetal Ultrasound
[2]Anatomy-Aware Self-Supervised Learning for Aligned Multi-Modal Medical Data
[3]SELF-SUPERVISED REPRESENTATION LEARNING FOR ULTRASOUND VIDEO

3.The computational cost and inference speed of A-ViT are not discussed, which may limit practical deployment.

4.Some baseline results (e.g., DINO v3) are strong, and the margin of improvement is not always substantial.

5.The main text lacks detailed statistical information about the dataset, such as organ, image size, depth, and classes.

### Questions
1.The authors are advised to check the title of the paper: “ANATOMY-AWARE REPRESENTATION LEARNING FOR MEDICAL ULTRASOUND.” There appears to be an extra symbol (a period at the end).

2.Why was the deformable attention mechanism chosen over other spatial-aware transformers? Were alternatives considered?

3.How does the model perform when the anatomical label is noisy or misassigned?

4.Is the performance gain mainly from the proposed architecture or the large-scale dataset? An ablation on dataset scale would be helpful.

### Soundness
3

### Presentation
2

### Contribution
3
