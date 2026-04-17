# Paying More Attention to Images in Large-scale Dataset Compression

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Dataset pruning (DP) and dataset distillation (DD) fundamentally differ in their outputs: DP selects original image subsets, while DD generates synthetic images. Recently, DD's increasing reliance on original images suggests a convergence of the two directions. To investigate this convergence trend, we propose a unified dataset compression (DC) benchmark. This benchmark reveals an interesting trade-off for soft-label-DD: while soft labels provide valuable information, they can make the distillation process less essential, as distilled images may not always outperform random subsets. In addition, the benchmark reveals that in current stages, dataset pruning outperforms dataset distillation at small dataset sizes.

Given these observations, we explore hard-label-DC as a complementary approach that emphasizes image quality while offering substantial storage efficiency. Our PCA (Prune, Combine, and Augment) is the first framework that does not rely on soft labels but instead focuses on image quality. (1) "P'' means selecting easy samples based on dataset pruning metrics, (2) "C'' indicates combining these samples effectively, and (3) "A'' is to apply constrained image augmentation during training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Summary
Dataset Distillation (DD) often requires soft labels such as pre-label outputs from stronger teacher models, which can take up huge storage. With the presence of soft labels and at larger image-per-class, sometimes DD can perform worse than random selection. Under an **all settings unified for different DD / DP experiments**, such as image size, batch size, # epochs/steps, and pruning ratio, the authors find out: **(DD + Soft Label) < (Random + Soft Label) < (Pruning + Soft Label)**, implying that soft labels brings performance gains. They also find out that this holds for hard labels, implying that DP's advantage is due to better image quality.

The authors address 4 design insights for their proposed PCA framework: achieving perfect class balance; using simpler images with lower entropy; and performing pruning on the entire dataset instead of subsets. The second point implies that cropping images across patches is not ideal with theoretical proof because the lower NLL doesn't reduce dataset entropy. Their PCA is thus a combination of :
Prune (balanced classes, simpler image with EL2N, prune entire dataset); Combine (downsample, combined grid); Augment (crop only to single patch).

Experiments are on ImageNet-1K + [ResNet-[18,50,101], MobileNet, EfficientNet, SwintT]. Dataset / training protocols are unified across compression methods. PCA significantly outperforms (by a large margin) all other methods in hard-label setting. Ablation also showed that each component of PCA is crucial to this great performance gain.

### Strengths
Strength
Observations and hypotheses for both DP and DD are insightful and grounded. Method design is straightforward and well supported. Experiments are solid. Unified evaluation is solid, such as resizing each image to a fixed size for pruning methods.
The insight that we can focus on hard labels while achieving better performance is great.

### Weaknesses
Weakness
It seems PCA is mostly dependent on the quality of pruning backbones (table 10), and the whole pipeline is a combination of existing techniques. Performance on e.g., SwinT degrades more than on ResNets with data from PCA (but since its observed by others too I guess it's fine). Table 2 still shows a major performance gap between softlabel and hardlabel settings, suggesting the extra storage spent is worth it if performance is prioritized, especially when smaller labs can afford storage but not compute.

### Questions
How to define "easy / simple image" (L241) v.s. "high-importance" image (L279)? Are these terms used interchangeably in your paper?

### Soundness
4

### Presentation
4

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
This paper formulates a unified, hard-label-only “dataset compression” benchmark that places dataset distillation and dataset pruning on a common footing. Within this setting, the authors propose a simple yet well-engineered PCA (Prune, Combine, Augment) pipeline. The central empirical finding is that carefully pruned, full-image subsets, coupled with constrained augmentation, can consistently outperform both dataset distillation methods and random subsets when evaluation protocols are matched.

### Strengths
1. The DC benchmark explicitly disentangles the influence of labels and data sources, and it shows that soft labels can partly conceal deficiencies in image quality. This is a constructive contribution to how the community evaluates dataset distillation and pruning, because it encourages reporting results under stricter, more comparable conditions.
2. By operating on images only (i.e., without relying on soft-label tensors), the proposed approach avoids substantial storage overheads and dataloader complexity, while still delivering accuracy gains under tight compression budgets. This makes the method attractive for real-world or resource-constrained scenarios.
3. The propositions/theoretical arguments provide a rationale for avoiding cropping, showing that cropping may remove informative content or cancel out the benefits of augmentation. This theoretical support makes the “combine rather than crop” decision appear motivated rather than ad hoc.
4. The paper reports extensive comparisons and ablations, covering optimizers, architectures, and dataset/model scales, that reveal stable improvements and reasonable trends. This breadth of experimentation strengthens the case that the method is not overly tuned to a single configuration.

### Weaknesses
1. Although the unified benchmark is valuable, the PCA pipeline itself can be interpreted as a careful recombination of existing principles—balanced selection, preference for “easy” images, and controlled augmentation.
2. Most results are presented on ImageNet-1K. To support stronger claims of generality, especially for long-tail, domain-shifted, or non-classification settings, the paper should include experiments on more diverse datasets and tasks.
3. The approach appears to rely heavily on EL2N/AUM-style metrics for identifying high-quality examples. Robustness to alternative metrics, as well as to noisy or distribution-shifted data, is only partially examined.
4. The negative view on cropping is derived under particular assumptions (e.g., fixed observer, specific entropy behavior). It remains unclear whether these conclusions continue to hold in tasks that crucially depend on localized evidence.

### Questions
1. How sensitive is PCA to the choice of pruning/quality metric (e.g., EL2N, AUM, Forgetting), and what is the corresponding wall-clock and computational overhead compared with strong dataset distillation baselines under the same compression budget?
2. Is it possible to integrate partial soft labels—such as periodically refreshed soft targets for a small subset—without reintroducing substantial storage or engineering complexity?
3. How does PCA behave under class imbalance, label noise, or distribution shift (e.g., OOD evaluation), given that the method explicitly favors “easy” and balanced examples?
4. Since constrained augmentation is central to the method, what default augmentation ranges are recommended across architectures, and how should these be adapted for higher resolutions or for tasks beyond classification (e.g., detection, segmentation)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies dataset compression (DC) by benchmarking dataset distillation (DD) versus dataset pruning (DP) and then proposes a hard label framework, PCA (Prune, Combine, and Augment). The proposed PCA keeps only easy representative real images combines them in a cropping-free fashion, and trains with a constrained augmentation recipe, reporting gains over prior DD and DP methods under a unified evaluation protocol from CDA.

### Strengths
* The paper carefully surfaces protocol mismatches between DD and DP (batch sizes, epochs vs. iterations, storage accounting), then evaluates them within a single protocol on ImageNet-1K with a fixed backbone, helping clarify several confusions in prior work.
* The paper quantifies the soft-label storage blow-up (~40$\times$ image storage for ImageNet-1K at IPC-10) and discusses the GPU/CPU transfer bottlenecks of label generation, valuable for practitioners who care about wall-clock and memory.

### Weaknesses
Refer to questions.

### Questions
* As I understand it, the experimental results in Figure 3 (a) appear to compare the performance of existing soft-label dataset distillation methods with that of label-only optimization starting from randomly initialized images.
In this case, does a theoretical gap exist between the procedure that jointly optimizes both images and labels and the one that optimizes only labels, such that the latter consistently suffers from performance degradation?
* In Figure 3 (b), the claim that (random selection + hard labels) outperforms the real-image–initialized DD methodologies is difficult to trust. Could the optimization of synthetic images proceed in a direction that reduces performance, as observed in DWA? It would be valuable to include direct comparisons with real-image–initialized, hard-label DD methods such as IDC [1], DDiF [2], and $D^3HR$ [3].
* Beyond data-storage efficiency and the preservation of target-model performance, dataset distillation also has auxiliary benefits in AI security, for instance in protecting sensitive information contained in the original data.
From this perspective, the proposed PCA approach seems to be closer to a coreset-selection strategy than to traditional dataset condensation, and I would be interested to hear the authors’ view on this interpretation.



[1] Kim, Jang-Hyun, et al. "Dataset condensation via efficient synthetic-data parameterization." International Conference on Machine Learning. PMLR, 2022.

[2] Shin, Donghyeok, et al. "Distilling dataset into neural field." arXiv preprint arXiv:2503.04835 (2025).

[3] Zhao, Lin, et al. "Taming Diffusion for Dataset Distillation with High Representativeness." arXiv preprint arXiv:2505.18399 (2025).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a unified framework and benchmark for comparing dataset pruning (DP) and dataset distillation (DD)—two primary approaches to dataset compression. The authors reveal a key paradox: synthetic images produced by DD, even with soft labels, often perform worse than random subsets of original images, while DP consistently outperforms both. They identify that DD’s perceived advantages largely stem from the use of storage-intensive soft labels (up to 40× larger than images) rather than genuine improvements in image quality. To address this, the authors introduce a new hard-label-only framework called PCA (Prune, Combine, and Augment), which focuses on selecting high-quality, class-balanced, and simple images through pruning, combining them without cropping to preserve information, and applying constrained augmentations aligned with data-scaling laws. Extensive experiments on ImageNet-1K show that PCA significantly outperforms prior DD and DP methods.

### Strengths
1. The paper has included enough analysis to illustrate the efficiency of the method under the pre-defined settings. 

2. The paper's presentation of the image classification experiments is clear and easy to follow.

3. The motivation of the paper is clearly stated, and the problem that the paper targets is good.

### Weaknesses
1. The paper only runs the experiments on classification tasks.  It is not clear how the conclusion generalizes to other tasks, such as segmentation, detection, and generative tasks.

2. The claim on the data-scaling law is not very persuvisive to me. To verify if the proposed method follows the data scaling law, it is better to actually derive the formul and measure the scaling coefficient. 

3. As the paper claims to have a benchmark and have drawn some obserbations from the results, it is better to run more baseline methods to have more generalizable conclusion.

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
3

### Contribution
3
