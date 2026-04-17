# KLAS: Using Similarity to Stitch Neural Networks for Improved Accuracy-Efficiency Tradeoffs

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Given the wide range of deployment targets, flexible model selection is essential for optimizing performance within a given compute budget.
Recent work demonstrates that stitching pretrained models within a model family enables cost-effective interpolation of the accuracy-efficiency tradeoff space.
Stitching transforms intermediate activations from one pretrained model into another, producing a new interpolated stitched network.
Such networks provide a pool of deployment options along the accuracy-efficiency spectrum.
However, existing stitching approaches often yield suboptimal tradeoffs and lack generalizability, as they primarily rely on heuristics to select stitch configurations.
We argue that constructing improved accuracy-efficiency tradeoffs requires explicitly capturing and leveraging the similarity between pretrained models being stitched.
To this end, we introduce KLAS, a novel stitch selection framework that automates and generalizes stitch selection across model families by leveraging KL divergence between intermediate representations.
KLAS identifies the most promising binary stitches from the $\mathcal{O}(k^2n^2)$ possibilities for $k$ pretrained models of depth $n$.
Through comprehensive experiments, we demonstrate that KLAS improves the accuracy-efficiency curve of stitched models at the same finetuning cost as baselines.
KLAS achieves up to $1.21\%$ higher ImageNet-1K top-1 accuracy at the same computational cost, or maintains accuracy with a $1.33\times$ reduction in FLOPs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes KL-divergence–based Anchor Stitching (KLAS), a framework for selecting where and how to “stitch” pretrained networks so as to interpolate accuracy–efficiency trade-offs more effectively than heuristic stitching (e.g., SN-Net). KLAS (i) chooses anchor pairs by the last-block KL divergence of their predictive distributions and (ii) ranks block pairs with a stitch score \\(\\Gamma(i,j)\\) that combines cross-anchor activation distance and the target block’s “capacity” (consecutive-block KL). To obtain the distributions, the authors train lightweight linear probes (ProbeNet) on each block; probes converge in ≈4 epochs and add a small one-time cost. Across DeiT, Swin, LeViT, and ResNet (ImageNet-1K, CIFAR-100/10), KLAS improves the AUC of the accuracy–FLOPs trade-off curves over SN-Net; gains include up to +1.21% top-1 accuracy at equal FLOPs or 1.33× FLOPs reduction at equal accuracy.

### Strengths
1. **Principled selection vs. heuristics.** The paper clearly articulates why nearest/paired stitching can be suboptimal and replaces it with a similarity-driven criterion grounded in KL divergence, with explicit formulas for \\(\\Theta\\) and \\(\\Gamma\\).
2. **Dual notion of similarity.** The KL criterion is argued (and operationalized) to reflect both epresentational alignment and functional compatibility, addressing shortcomings of CKA/MSE/CE/DM for choosing stitch points.
3. **Efficient probing.** ProbeNet trains one set of blockwise probes efficiently (≈0.25 GPU-days for Swin-B) and shows fast convergence, keeping selection overhead modest.
4. **Broad empirical coverage.** Results span multiple families (ViTs, CNNs) and even cross-architecture stitches (e.g., ResNet↔Swin), consistently improving trade-off AUC vs. SN-Net.
5. **Anchor selection that adapts by family.** Last-block KL correctly prefers far stitching (Ti→B) for Swin but nearest (Ti↔S, S↔B) for DeiT, demonstrating generality beyond a fixed heuristic.

### Weaknesses
1. **Supervision dependence for “similarity.”** KL is computed on softmax outputs of supervised probes; thus, selection intrinsically depends on labels and probe training. This limits claims for unsupervised/self-supervised representation learning and may bias choices toward the probe’s training distribution.
2. **Asymmetry & calibration sensitivity.** KL’s asymmetry and dependence on calibration/temperature may distort distances across blocks/models; the method normalizes by an intra-anchor term, but robustness to temperature, class imbalance, or label smoothing is underexplored.
3. **Metric/selection design choices.** The final selection uses bucketized FLOPs and a threshold \\(\\tau\\) (5% of the minimum in each bucket). The AUC metric and bucket granularity can influence conclusions; more sensitivity studies would strengthen the case.
4. **Scale of gains in some regimes is small.** Several settings show marginal improvements (e.g., cross-architecture ΔAUC≈+0.002), raising questions about practical significance across all families.
5. **Representation-learning scope.** Experiments are largely supervised classification; there is no evaluation with self-supervised anchors (e.g., DINO/MAE) nor transfer via frozen-backbone linear probes on diverse tasks.
6. **Limited evidence for dense prediction/generalization.** Dense-task adaptation is left as future work with only preliminary results; rigorous detection/segmentation studies are missing.
7. **Compute accounting.** While probe training is “negligible,” it is still an added search cost versus purely heuristic SN-Net; a wall-clock comparison for the full pipeline (probes + stitch fine-tuning) would help.

### Questions
1. **Un/SSL compatibility.** Can KLAS be made label-free, e.g., by using self-supervised probes (DINO-style heads) or pseudo-labels? How does anchor/block selection change when anchors are self-supervised?
2. **Calibration sensitivity.** How sensitive are \\(\\Theta\\) and \\(\\Gamma\\) to softmax temperature, label smoothing, and class imbalance? Could you report ablations and perhaps use temperature-scaled KL?
3. **Search cost accounting.** What is the end-to-end overhead (wall-clock/GPU-days) of ProbeNet + KLAS vs. SN-Net’s heuristics at equal stitch fine-tuning budgets? Please include variability across families.
4. **Ranking validity.** Beyond the Min-KL vs. \(\Gamma\) comparison, can you provide rank correlation between KLAS scores and post-fine-tuning accuracy across candidates (per bucket and overall)?
5. **Cross-family stitches.** For cases with very small ΔAUC, what failure modes arise (capacity mismatch, optimization instability, probe mis-ranking)? Any diagnostics (e.g., KL heatmaps) that predict such cases?
6. **Dense tasks.** Could you include full experiments on detection/segmentation (COCO/ADE20K) where stitching locations affect multi-scale features, and compare to dynamic routing/pruning baselines?

**If the author can address my questions, I am willing to improve my rating.**

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the use of inserted linear probes and KL divergence between them to find layers compatible for stitching between different networks.

### Strengths
- clear language and well written

- the idea of using linear probes + KL divergence for stitching is sound and well supported by experiments

### Weaknesses
- The declared goal of the KLAS approach is to use models from a pre-trained zoo to construct models that provide new accuracy / cost trade-offs. This is presented as an alternative to NAS (a somewhat unfair comparison as NAS enriches to base-model pool). However, the standard method of addressing this problem are model cascades / committees (e.g. "WISDOM OF COMMITTEES: AN OVERLOOKED APPROACH TO FASTER AND MORE ACCURATE MODELS" or "Efficient Inference With Model Cascades"). Model cascades are not discussed at all and not compared against in this paper. A comparison against a strong cascade baseline is essential to be able to claim utility of the proposed method for the suggested purpose. In addition to the direct comparison to a model cascade, please also explain whether there are situations where stitching is preferrable over cascading for structural reasons (maybe average case complexity vs. worst case complexity?). A 1) worst-case and 2) average-case acc/cost tradeoff curve comparing stitching and cascading would be meaningful.

- "KL divergence uniquely satisfies the dual objectives" this claim is not substantiated. Probably the authors mean ~ 'uniquely among the few measures we consider here'; if yes, please rephrase, if not, please provide a proof that the KL divergence is indeed unique in this regard.

- it is unclear whether the method is specific to image classifiers or works in other domains as well. The impact of the paper would be much broader, if there was some indication that it works for regression tasks (--> how to substitute the KL div??) and for non-vision classification (e.g. also token prediction in LLMs). Currently the method is of limited interest as it only applies to a niche task.

### Questions
In what scenarios is stitching preferrable to a model cascade? Why?

Do linear probes work for regression problems? For very large classifiers (think LLMs)? (the discussion mentions this very shallowly, there is no need to show improvement both for prefill and decode, just pick the easier case and show some improvement)

### Soundness
3

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
The paper proposes stitching blocks of neural network by computing similarity between blocks by taking the KL-Divergence of linear probe probability distributions. The paper suggests that "far-stitching" where models with significant difference in complexity or performance may also be worthy of stitching. The paper proposes to automatically identify block pairs to stich opposed to SN-Net where they use defined constraints to choose the pairs. The paper compares the proposed KLAS stitching framework with SN-Net on DeiT and Swin model architectures trained on CIFAR-100 and imagenet-1k datasets. The results show marginal gain in performance compared to SN-Net.

### Strengths
1. The paper explores functional similarity along with representation similarity as a metric while choosing to stitch networks which is logical. If two networks produce similar same functional output, then the models are better suited for selecting stitching pairs, when considered with representation similarity.
2. The paper also shows that far stitching where two networks may not have similar performance still can be stitched together is an important contribution.
3. The experimentation on DeiT and Swin family of models to compare with SN-Net shows similar performance but seemingly less finetuning cost.

### Weaknesses
1. In line 232 the paper states "As a representational similarity metric, KL divergence captures distributional differences between intermediate activations, indicating whether two blocks generate patterns that
can be mapped via lightweight transformations." which is not supported in my opinion as KL divergence compares the probabilistic score distribution not the representation itself. (Think different features can be used to reach same conclusion with similar confidence).

2. There is only marginal improvement when comparing stitched models of similar flops compared to SN-Net, while individual anchor-block stitches give an edge to KLAS, collectively the gain is minimal.

3. The lack of representational similarity is a concern (as mentioned in point 1 KL divergence doesnt directly compare representations), if there is difference in representational similarity there would be more finetuning cost.
 
4. The details on stitched model finetuning could be added for better assessment. This is important to assess the improvement in AUC.

### Questions
1.  For the stitch finetuning step is the number of training steps per stitched model constant or does it vary depending on when the stitch converges?
2. I did not get if the the finetuned stitched model is same for both SN-NET and KLAS if the stitch pairs overlap.

### Soundness
1

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Stitching pretrained models offers a cost-effective way to explore accuracy-efficiency tradeoffs. Current stitching methods rely on heuristics, leading to sub-optimal results. This paper improves this by using KL-divergence to automate stitch selection and generalize across architectures. KLAS outperforms baselines, achieving higher accuracy at the same computational cost or reducing FLOPs while maintaining accuracy.

### Strengths
* Technical contribution: This paper addresses a fundamental challenge in current model stitching methods: these methods rely on heuristic-based stitch selection, fix anchors and blocks, and thus yield suboptimal accuracy-efficiency tradeoffs. Specifically, it proposes a coarse-grained anchor selection strategy that leverages the KL divergence of the last block for anchor identification, and employs block-level KL divergence for fine-grained selection. The metric for candidate set selection is well-justified, as it considers both sampling coverage during fine-tuning and the quality of anchor filtering. 

* Experiments effectively demonstrate the advantages of the proposed KL-divergence-based anchor stitching approach.

### Weaknesses
* Why MSE, CE, CKA, DM can have significantly lower percentage of stitch configurations than KL divergence? Mathematically, it’s not that straightforward.

* The method’s applicability to large language models and multimodal LLMs has yet to be explored—extending it to these models would further enhance the paper’s impact.

* What performance can KLAS achieve on dense prediction tasks, including object detection, semantic segmentation, and depth estimation?

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
