# Do We Need All the Synthetic Data? Targeted Image Augmentation via Diffusion Models

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Synthetically augmenting training datasets with diffusion models has become an effective strategy for improving the generalization of image classifiers. However, existing approaches typically increase dataset size by 10–30× and struggle to ensure generation diversity, leading to substantial computational overhead. In this work, we introduce TADA (**TA**rgeted **D**iffusion **A**ugmentation), a principled framework that selectively augments examples that are not learned early in training using faithful synthetic images that preserve semantic features while varying noise. We show that augmenting only this targeted subset consistently outperforms augmenting the entire dataset. Through theoretical analysis on a two-layer CNN, we prove that TADA improves generalization by promoting homogeneity in feature learning speed without amplifying noise. Extensive experiments demonstrate that by augmenting only 30–40% of the training data, TADA improves generalization by up to 2.8% across diverse architectures including ResNet, ViT, ConvNeXt, and Swin Transformer on CIFAR-10/100, TinyImageNet, and ImageNet, using optimizers such as SGD and SAM. Notably, TADA combined with SGD outperforms the state-of-the-art optimizer SAM on CIFAR-100 and TinyImageNet. Furthermore, TADA shows promising improvements on object detection benchmarks, demonstrating its applicability beyond image classification. Our code is available at https://github.com/BigML-CS-UCLA/TADA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This author studies diffusion-based synthetic data augmentation for image classification. Unlike prior works that augment the entire dataset (often by generating 10×–30× more images), the authors propose targeted augmentation: identify the subset of training samples whose features are “slow-learnable” and generate synthetic variants only for these.

### Strengths
1) Focusing diffusion-based augmentation on slow-learnable examples is a fresh take on synthetic augmentation.
2)  The authors analyze a simplified two-layer CNN to compare SAM vs. SGD (showing SAM learns “noise” more slowly) and prove that generating faithful images accelerates learning of slow features without amplifying noise (Theorems 4.1–4.3).
3) The experiments are extensive and well-controlled. On CIFAR-10/100 and TinyImageNet, across multiple architectures, the proposed method consistently outperforms baselines (random subset, full augmentation, simple upsampling) and yields up to 2.8% test accuracy
4) By augmenting only ~30–40% of data, the method substantially reduces synthetic data generation time (e.g. 3.6h vs 12h on CIFAR-10) compared to full-data diffusion.

### Weaknesses
1) All experiments are on relatively small benchmarks (CIFAR-10/100, TinyImageNet). It is unclear if the approach scales to large datasets (e.g. full ImageNet) or real-world settings.
2) Identifying slow-learnable examples via early clustering of model outputs (or high loss) is somewhat heuristic.
3) The method relies on a text-conditional diffusion model (GLIDE) with class prompts and uses the real image as guidance. It is not fully clear how much the performance depends on prompt engineering or the specific diffusion backbone.
4) The 2-layer CNN analysis, while insightful, assumes a very stylized data distribution (two patches, Gaussian noise, cubic activations) and early-training approximations. It is not guaranteed these results carry over to deep architectures and natural images.

Missing relevant references:

1) GenMix: Effective Data Augmentation with Generative Diffusion Model Image Editing

2) Context-guided Responsible Data Augmentation with Diffusion Models

### Questions
1) How sensitive are the results to the specifics of the clustering step? For instance, does choosing a different number of clusters or a different layer for features change which examples are selected as “slow”?
2) Did you compare clustering versus simply picking top-θ% highest-loss examples (or uncertain examples)? Table 7 suggests clustering works better than high-loss, but can you elaborate on why?
3) How was the denoising step (e.g. 50 steps) chosen? Would an adaptive schedule (tailored per image) improve results?
4) Have you considered whether targeted augmentation helps other tasks (e.g. detection) or robustness measures beyond accuracy?
5) How does performance compare to simply oversampling the slow examples (as in weighted sampling) without generating new images? This would isolate the benefit of synthetic variety.

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
3

### Summary
This paper proposes a targeted augmentation approach using diffusion models, focusing on "slow-learnable" images (identified via early training clustering) by adding noise and denoising to create faithful variations with different noise. A theoretical analysis with a two-layer CNN suggests this promotes uniform feature learning and reduces minibatch variance compared to upsampling. Experiments on CIFAR-10/100 and TinyImageNet show accuracy improvements (up to 2.8%) with ResNet, ViT, and other models, and the method complements optimizers like SGD and SAM.

### Strengths
**Efficiency**: Augmenting only 30–40% of data outperforms full-dataset augmentation, offering a practical, resource-aware solution.

**Empirical Support**: Ablation studies on augmentation factors and initialization provide useful insights.

**Compatibility**: Works well with existing methods (e.g., SAM), boosting performance further.

### Weaknesses
**Missing Prior**: The method overlaps with "Boomerang" [1], which uses similar noise-add-and-denoise techniques for data augmentation for classification, but it’s not cited or compared. Notably, they use all of the dataset for synthetic data generation, and they see gains in accuracy, which contradict experiments in this paper. 

**Theory-Practice Gap**: The claim of mimicking SAM’s feature learning (e.g., sections 4.1–4.2 suggest SAM-like noise suppression and uniform learning) doesn’t fully align with empirical results, where gains add to SAM’s effects (abstract notes up to 2.8% improvement with SAM). This suggests the method might address different aspects of training dynamics than intended, and further analysis could clarify this discrepancy.

**Convergence claim**: The assertion of faster SGD convergence (Theorem 4.3, Corollary 4.4) relies on synthetic noise variance being lower than upsampling variance, but the link to the "small noise" assumption (section 4.4) isn’t fully derived or supported with training curves, leaving uncertainty about its practical impact.

**Idealized and unrealistic theory setting**: The model assumes simplified conditions (e.g., P=2 patches, orthogonal features in section 3), which may not capture the complexity of real image data, potentially limiting the theory’s applicability to broader settings. 

**Scope limitation**: Experiments are confined to small datasets, and the claim of effectiveness across diffusion models (abstract) lacks support from multiple generators, which could restrict the method’s generalizability and leave its robustness untested.

[1] Luzi L, Mayer PM, Casco-Rodriguez J, Siahkoohi A, Baraniuk R. Boomerang: Local sampling on image manifolds using diffusion models. Transactions on Machine Learning Research.

### Questions
- Could you explain why the atypical activation function $\sigma(z) = z^3$ was chosen over ReLU, and does the theory hold if ReLU is used instead?

- Please cite, and compare/discuss results in the 'boomerang' paper, as it seems to contradict results in this paper.

-  Can authors please add training-loss curves to support the convergence claim and explore why stacking with SAM works?

- It would be great (but no necessary) to include a simple metric (e.g., feature similarity) to verify "faithfulness" of synthetic images.

- Could authors test on a larger dataset and with another diffusion model to broaden applicability?

## Overall 

This is a helpful and promising approach for efficient augmentation, with solid small-scale results. Addressing the prior comparison, clarifying theory-practice links, and expanding experiments could make it even more impactful!

### Soundness
3

### Presentation
2

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
This paper proposes a novel data augmentation strategy designed to improve the generalization of image classification models. The core contribution is a method that selectively applies augmentation only to a subset of the training data identified as slow-learning samples. The authors demonstrate that this targeted approach improves classification performance.

### Strengths
- The central idea of targeting slow-learning samples for augmentation is novel and intuitive. The rationale that focusing augmentation efforts on more challenging examples seems a logical approach to improving model robustness and generalization.
- The paper provides extensive empirical validation across three different datasets, showing credibility to the proposed method's effectiveness. The observation regarding the characteristics of slow-learned samples is particularly interesting and further discussion on this would make paper more interesting.

### Weaknesses
- The theoretical analysis relies on a simplified two-layer CNN assumption. This raises questions about the direct applicability and relevance of the derived theorems to the deeper, more complex architectures commonly used in practice. The paper would be strengthened by a discussion bridging this theoretical gap.
- I have concerns regarding the significant computational overhead of the proposed method. Utilizing a diffusion model for data generation, even for a subset of the data, is inherently more expensive than traditional augmentation techniques. Also, the multi-step pipeline may limit its practical adoption.
- The experiments are confined to relatively small-scale datasets. It is unclear how the method would perform on larger, more complex datasets such as ImageNet. An explanation for the choice of datasets and a discussion on the method's potential scalability would be beneficial.

### Questions
- What are the fundamental, identifiable differences between the samples classified as "slow-learning" versus "fast-learning"? If distinct features or patterns characterize these slow-learning samples, could a model be developed to identify them a priori? Such an approach could simplify the overall pipeline by removing the need for an initial training phase solely to identify these samples.
- Could the proposed augmentation strategy, which focuses on difficult examples, be adapted to benefit tasks outside of classification, such as improving sample quality or diversity in image generation?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposed a training strategy that can smartly combine real data and synthetic data for training an improved classifier. To verify the effectiveness, this paper evaluated the method on image data augmentation for classification across backbones and datasets. This paper also provided some analysis on simple MLP layers. Besides, the method is a plug-and-play module and also evaluated plugged into the DiffuseMix.

### Strengths
The dominant strength is that the current data augmentation paper only focuses on how to generate data with high fidelity and diversity for a more robust decision boundary. However, a very small paper focuses on how to balance the real set and the synthetic set during the training process. This paper fills the blank for current generative-based data augmentation research.

### Weaknesses
This method is general, but the evaluations are limited.

1/ The evaluated backbones are too weak, and whether better-pretrained backbones can overlay the benefit of your method.

2/ Since this method is a plug-and-play module, why not evaluate it based on more state-of-the-art methods like [1,2,3,4]? Meanwhile, you should at least discuss them in the related work.

3/ Lack of evaluations on fine-grained datasets.

4/ This method seems like can be applied not only for image classification datasets, how for the augmentations in detection, segmentation even in other modalities like text and videos.


References

[1] Effective Data Augmentation With Diffusion Models

[2] Enhance image classification via inter-class image mixup with diffusion model

[3] Inversion Circle Interpolation: Diffusion-based Image Augmentation for Data-scarce Classification

[4] Advancing Fine-Grained Classification by Structure and Subject Preserving Augmentation

### Questions
If you can solve my concerns, the method can be very general, and then I can raise my score to 8.

### Soundness
3

### Presentation
3

### Contribution
3
