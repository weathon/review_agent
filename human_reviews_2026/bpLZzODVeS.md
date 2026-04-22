# DiCoH: Rethinking Self-Supervised Pretraining for Semantic Segmentation in Homogenous Medical Domains

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Self-supervised learning (SSL) for pretraining has become critical for improving segmentation performance when labeled data is scarce. However, existing contrastive methods are primarily designed for diverse, object-centric natural images and struggle to generalize to $\textit{homogenous}$ medical datasets that exhibit low semantic variation across both images and pixels. Low semantic variations make aligning positive pixel-to-pixel pairs trivial and make identifying true negative pairs extremely challenging. Additionally, we identify that $\textit{architectural asymmetry}$, demonstrated to stabilize contrastive pretraining, is detrimental when applied to homogeneous data. To tackle these limitations, we present $\textbf{Di}$verse $\textbf{Co}$ntrastive Learning for $\textbf{H}$omogeneous Data (DiCoH), an SSL pretraining framework for homogeneous medical data. DiCoH improves representation learning by diversifying positive pixel-to-pixel alignments and guaranteeing true negative pairs through a novel \textit{hard} pixel-to-image selection strategy. Comprehensive evaluations on five medical segmentation datasets demonstrate that DiCoH significantly and consistently outranks state-of-the-art SSL methods, achieving +2.00\% mIoU gains under extremely low-data conditions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes DiCoH (Diverse Contrastive Learning for Homogeneous Data), a self-supervised pretraining method for semantic segmentation in homogeneous medical images. Existing contrastive learning methods such as SimCLR or MoCo work well for natural images but can perform poorly on medical data where images are visually similar and lack clear boundaries. The paper identifies three main problems in such data: trivial positive pairs, false negatives caused by semantic collisions, and feature variance collapse from asymmetric architectures.

To address these, the proposed method DiCoH introduces (1) diverse positive alignment combining spatial and similarity-based pixel matches, (2) pixel-to-image negative sampling to avoid semantic overlap, and (3) a symmetric siamese architecture that preserves feature variance. The model is trained with a novel joint objective which is specially designed to balance local and global contrastive losses.
On five medical segmentation datasets, DiCoH outperforms prior self-supervised and supervised baselines, achieving up to about two percentage points higher mean IoU under low-data conditions (5–10% labeled samples). The results show that adapting SSL designs to homogeneous domains can significantly improve label-efficient medical image segmentation.

### Strengths
**Originality**:

The paper considers an important problem—SSL on homogeneous datasets—and introduces a new strategy combining diverse pixel alignments, pixel-to-image negatives, and symmetry. The idea of using pixel-to-image negatives seems novel.

**Quality**:
The methodology is rigorous and well-presented, including ablations isolating each of the 3 key components (alignment, negatives, architecture). Experimental design which uses ImageNet-initialized backbones and consistent evaluation across five datasets seems fair.

**Clarity**:
The paper is well-organized with clear figures. Mathematical notation is clear. It is especially good that the motivation for each key design is directly supported by empirical findings (e.g., Fig. 4’s variance collapse evidence).

**Significance**:
The contribution is meaningful for medical imaging and potentially for other homogeneous domains (e.g., agriculture, manufacturing). Improving label efficiency in such areas can have high real-world impact. The work also challenges an established design norm—architectural asymmetry—in contrastive learning.

### Weaknesses
(1) The paper presents a clear motivation and shows consistent improvements. But I feel a major weakness is that its empirical analysis lacks sufficient depth. The experiments demonstrate that DiCoH performs better than prior methods, but they do not fully explain why each of the three components (diverse positive alignment, pixel-to-image negatives, and symmetricity) contributes to the gain or how these components interact. For example, the analysis mostly reports performance differences gained from the three components without probing the underlying behavior of representations or learning dynamics.

(2) Architecture Choice:
The use of a ResNet-50 + DeepLabv3 backbone is fine but somewhat quite old. Evaluating DiCoH on newer architecture such as Vision Transformers might be a better choice? It can also demonstrate the adaptivity of DiCoH to various architectures.

(3) While the authors mention potential generalization beyond medical data, experiments are restricted to medical segmentation. Validation on non-medical homogeneous datasets (e.g., industrial defect detection) would strengthen claims of general applicability.

(4) The reported performance improvement seems to lack a statistical confidence interval (i.e. error bar).

### Questions
(1) How sensitive is DiCoH’s performance to the weighting parameter $\lambda$ between global and local losses? 

(2) How large is the negative image queue, and how does its size affect the performance of the method?

(3) The paper claims that symmetric architectures “preserve feature variance.” But it seems unclear how this translates to better representation geometry? In there any quantitative metric that can demonstrate that?

(4) What is  the computational cost of pixel-to-image negatives and dual alignment strategies? Is it huge?

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
The authors propose a self-supervised learning approach, DICOH for pre-training deep learning models for dense tasks in medical imaging contexts, where data is more homogeneous than in computer vision. DICOH is a contrastive approach with global (instance-level) and local (pixel-level) losses that aims to avoid trivial positive pairs (by defining a positive based on spatial correspondence and another one based on feature cosine similarity), false negative pairs (by adopting a pixel-to-image negative strategy), and that uses a symmetric siamese architecture. The proposed method is compared against various instance-level pre-training strategies, semi-local contrastive strategies, and histopathology SSL methods (ConCL, CA2CL), assuming ImageNet-based pre-retraining, showing generally good performance.

### Strengths
1. The proposed approach shows promising performance in low data settings.
2. The ablation study shows the impact of specific elements introduced by the proposed approach
3. Clarity: The paper is easy to read and generally clear.

### Weaknesses
1. I have reservations regarding some of the main contributions:

Contribution 1. Actually no "analysis" is provided as to "why trivial positives, false negatives, and asymmetric designs degrade contrastive pretraining in homogeneous medical data", only an ablation study that supports this statement.

Contribution 2. "Diversified positives (spatial + similarity)" were already introduced in VICRegL, Bardes et al. 2022. The proposed method is not compared against VICRegL. "Pixel-to-image negatives" were already proposed in DenseCL. The "symmetric siamese design" is the standard in contrastive approaches, to which the proposed method belong, rather than a novelty. (Asymmetric designs are more commonly used in self-distillation or joint embeddings SSL)

Contribution 3. I have reservations here. Indeed, dense SSL benchmarks in the medical domain in annotation scarce regimes are generally not conducted under an ImageNet-pre-pretraining. But it is not clear 1) whether this ImageNet-initialization + SSL improves upon random initialization + SSL across most medical domains, due to the domain gap. 2) whether ImageNet-Initialization should be seen as a positive (fairer) or negative (detrimental) change to the baselines the authors compare against. It would be helpful to see the performance of each method, including the proposed method, without this initialization. This is also a limitation to the scope of the paper, as this type of initialization is very specific to 2D applications, whereas many medical domains are 3D.

2. A "pure" pixel-level SSL baseline defining positive pairs from the natural correspondence maps after augmentations, coming from the medical domain, for example vox2vec [1], is missing. 

3. A dense representation baseline coming from foundation models, such as DINOv2 or DINOv3, would also be a welcome addition.

4. The impact of ImageNet-initialization is not ablated.

5. It is not clear whether baselines are trained using their optimal hyperparameter settings or the same settings as for the proposed method (regarding learning rates, batch size, choice and strengths of augmentations, etc.), which have probably been fine-tuned specifically for the proposed method. Could the authors kindly clarify ?


[1] vox2vec: A Framework for Self-supervised Contrastive Learning of Voxel-level Representations in Medical Images, Goncharov et al. MICCAI 2023

### Questions
The comments and questions raised in the weaknesses section above are the main factors in my initial rating.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents DiCoH (Diverse Contrastive Learning for Homogeneous Data), a novel self-supervised learning (SSL) framework tailored for homogeneous medical datasets. The authors address the limitations of existing SSL methods that struggle with low semantic variation in medical images. DiCoH introduces three key innovations: diversified positive pixel-to-pixel alignments, robust pixel-to-image negative sampling, and a symmetric siamese architecture. Comprehensive experiments on five medical segmentation datasets demonstrate that DiCoH significantly outperforms state-of-the-art SSL methods, achieving up to +2.00% mIoU gains under extremely low-data conditions.

### Strengths
1. Comprehensive Evaluation: The authors provide extensive experiments on multiple medical segmentation datasets with varying amounts of labeled data (5%–10%). The results clearly demonstrate the effectiveness of DiCoH, making a strong case for its superiority over existing methods.

2. Ablation Studies: The paper includes thorough ablation studies that validate the importance of each component of DiCoH. This provides valuable insights into the design choices and helps build confidence in the proposed framework.

### Weaknesses
1. Although the article title focuses on semantic segmentation, I am still concerned about the performance of other downstream tasks. The absence of experiments on classification and visual question answering tasks is notable. To further demonstrate the effectiveness of the proposed method, additional experiments on datasets related to fundus photography, X-ray and dermatology will be help.

2. No comparison has been made with pre-training methods such as masked image modeling.

### Questions
It appears that the framework of the paper could be seamlessly extended to 3D medical imaging. Do you think this method will still be effective on datasets like the LA or Pancreas-NIH datasets? After all, 3D images make up a significant proportion of medical data

For other issues, Please refer to the weaknesses mentioned above, I would consider increasing my score if the related issues are addressed.

### Soundness
2

### Presentation
3

### Contribution
3
