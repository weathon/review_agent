# Bad-OOD: Discovering Harmful Synthetic Diffusion Outliers via Confidence Calibration

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Utilizing synthetic outlier samples has shown great promise in out-of-distribution (OOD) detection. In particular, impressive results have been achieved by employing diffusion models to generate synthetic outliers in the low-density manifold. However, guiding diffusion models to generate meaningful synthetic outliers remains challenging. The synthesized samples often fall either too close to the in-distribution (ID) data (risking overlap and ambiguity) or too far (leading to visually unrealistic results). Both extremes have been shown to degrade OOD detection performance. In this work, we propose a novel OOD synthesis framework that combines a pre-trained Representation Diffusion Model (RDM) with a simple yet effective classifier calibration strategy. RDM enables global semantic embedding generation without requiring auxiliary labels or text, producing diverse yet ID-relevant outliers, thereby facilitating a more compact ID-OOD decision boundary. To ensure the utility of these samples, we calibrate a binary classifier on both ID data and synthesized OODs to assign confidence-based anomaly scores. We find that mid-confidence outliers, i.e., those balancing realism and deviation, are most informative, and using them significantly boosts detection performance. Extensive experimental results validate the superiority of our calibrated OOD sampler over several strong baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores an outlier exposure–based approach to out-of-distribution (OOD) detection. The authors propose a pipeline that synthesizes effective outliers using a Representation Diffusion Model (RDM) guided by a minority score, followed by MAGE for pixel-level reconstruction. To improve boundary separation between in-distribution (ID) and OOD data, they apply an energy-based OOD regularizer and employ beta calibration for confidence refinement and OOD score computation. The proposed confidence-calibrated detector is further evaluated on the Syn-IS benchmark, demonstrating improved sensitivity to both semantic and covariate shifts compared to prior scoring methods.

### Strengths
1. The proposed framework allows for OOD synthesis and detection without requiring class labels or textual conditions, enabling applicability under weak supervision.
2. Evaluation on the Syn-IS benchmark provides a clear analysis of how the method handles different types of distributional shifts (semantic and covariate), showing balanced sensitivity compared to existing baselines.
3. The integration of beta calibration for filtering mid-confidence synthetic outliers is conceptually simple yet empirically effective in improving detection quality.

### Weaknesses
1. Given the rapid progress of vision-language and multimodal foundation models, the relevance of this line of research—focusing on ResNet-based standalone OOD classifiers—appears limited. It is unclear whether such a direction remains practically or scientifically necessary in the current landscape.
2. Although Figure 4(a) reports higher within- and cross-class diversity of synthetic samples compared to Dream-OOD, Table 1 shows that Dream-OOD with calibration achieves similar performance to the proposed method trained with bad-OOD samples. This raises questions about whether the increased sample diversity indeed contributes to the performance improvement, or if it remains largely decorrelated from the final detection effectiveness.
3. The approach relies on several hyperparameters (e.g., minority guidance scale, filtering threshold), which may be dataset-specific and limit reproducibility and generalization.

### Questions
1. The beta calibration parameters (a,b,c) are learned from the full set of synthetic OOD samples, even though the classifier is trained with potentially noisy or low-quality outliers. How can these parameters be reliably optimized in such conditions? If the claim is that relative ranking among samples is meaningful, experimental justification is needed.
2. The definitions of OOD-II and OOD-III (Figure 3, Table 3) require clarification. How is “hardly recognizable” operationalized—through human labeling, CLIP similarity, or a quantitative threshold?
3. Is decoding into the pixel space via MAGE necessary? Could the classifier instead be trained directly in the latent representation space without losing OOD discriminative power?

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
4

### Summary
This paper proposes the BAD-OOD framework to address poor-quality synthetic outlier samples in OOD detection (either too close to ID data causing overlap or too far leading to unreality). It uses RDM to generate ID-semantically aligned OOD samples without auxiliary labels, then filters mid-confidence (0.3–0.5) samples via beta calibration to optimize the ID-OOD boundary. Experiments on ImageNet-100/CIFAR-100 with ResNet backbones show advantages over baselines like Dream-OOD.

### Strengths
1. The label-free OOD generation design fits real-world needs. RDM generates OOD samples based on ID semantic embeddings without auxiliary labels, avoiding high label-acquisition costs, and retains ID-related features to support a compact decision boundary.

2. Confidence calibration filtering effectively solves sample quality issues. Beta calibration aligns confidence with anomaly degree; mid-confidence samples reduce FPR95 from 51.20 to 39.69 and boost AUROC to 91.71, outperforming unfiltered or extreme-confidence samples.

3. Basic experiments cover multi-dimensional scenarios. Comparisons across ID datasets (ImageNet-100/CIFAR-100), generators (MAGE/DiT/LDM), and ResNet backbones (34/101), plus Syn-IS validation for shift capture, show initial adaptability.

### Weaknesses
1. Core innovation is a combination of existing technologies. RDM directly uses Li et al. (2024)’s pre-trained model without modifying denoising logic; beta calibration follows Kull et al. (2017)’s formula without OOD-specific adaptations. The workflow (“RDM→MAGE→beta”) lacks breakthroughs beyond existing paradigms.

2. Model adaptability is extremely limited—only ResNet is proven effective. No validation for other CNNs (e.g., DenseNet), ViT (MoCo-v3 ViT-L is only for ID feature extraction, not as detector backbone), or CLIP (only for OOD type classification, not detector adaptation), leaving their compatibility unknown.

3. Mechanistic explanation for mid-confidence samples is shallow. Only phenomenological descriptions (high-confidence = ID-like, low-confidence = unrealistic) are provided, without quantitative analysis of MoCo feature space distance distribution or cross-category consistency; qualitative analysis covers only 4 categories, lacking causal explanation.

4. Ablation experiments for core module necessity are missing. No comparison of RDM vs. Dream-OOD’s Gaussian sampling, beta calibration vs. temperature/Platt scaling, or analysis of minority guidance scale’s impact on sample quality, weakening experimental rigor.

5. Large-scale dataset/benchmark validation is insufficient. No experiments on ImageNet-1k; OpenOOD v1.5 evaluation is only in Appendix I without in-depth main-experiment analysis of complex scenarios, leaving real-scenario applicability unproven .

### Questions
see Cons

### Soundness
2

### Presentation
2

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
The paper presents a method for outlier synthesis in the latent space of an RDM and uses a calibrated classifier to filter out fake OOD samples that harm the performance of the downstream OOD detector. These OOD samples can either be too easy or too difficult. The method is evaluated on the standard ImageNet-100 and CIFAR100 benchmarks against several baselines, and ablated in various aspects.

### Strengths
- The paper is well-written and easy to follow.
- The observation that synthesized outliers at both extremes degrade the OOD detection performance is interesting.
- The experiments and ablation studies conducted are extensive, clearly place the contribution in the broader field, and show the influence of various components.

### Weaknesses
- More synthesized outliers lead to better performance [1], and previous outlier synthesis methods all use 100k synthetic outliers for training [2,3]. As the proposed method uses 100k samples after filtering, it would be fair to compare it to the baselines with 130k samples as well to see whether the performance gains are due to sampling more outliers. 
- The very inconsistent results on CIFAR100 w.r.t. SONA, where on three datasets SONA is far better, and vice versa on the final two, should be explained. I would expect that this is largely due to the resizing strategy (bilinear vs. nearest) used to downsample synthesized outliers to 32x32, cf. the "pixel interpolation method" section in [1], rather than a measure of quality between methods. 
- The observation that "most removed samples belong to OOD-I and OOD-III" is not surprising, as Tab. 3 shows most outliers do not belong to OOD-II, and relatively more OOD-II samples are removed relative to the other types. It also shows that the proposed method largely generates undesirable outliers, i.e., "near-ID and meaningless far-OOD samples". Overall, these results seem to contradict the rest of the method, as the kept outliers are not disproportionately "mid-OOD". It is also not clear what the percentages in the "Reduce" row refer to. 

[1] Doorenbos, Lars, Raphael Sznitman, and Pablo Márquez-Neila. "Non-Linear Outlier Synthesis for Out-of-Distribution Detection." arXiv preprint arXiv:2411.13619 (2024).
[2] Liao, Qilin, et al. "BOOD: Boundary-based Out-Of-Distribution Data Generation." International Conference on Machine
Learning 2025
[3] Du, Xuefeng, et al. "Dream the impossible: Outlier imagination with diffusion models." Advances in Neural Information Processing Systems 36 (2023): 60878-60901.

### Questions
- Fig. 4 would be cleaner if the images were sorted by their anomaly score.
- Part three of the related work section concerns OOD detection and should probably be labeled as such.
- Typos: "metrics" should be singular on L82, and "resulting in the outlier representations naturally occupy the periphery" is grammatically incorrect.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an outlier synthesis approach for improving the training of out-of-distribution (OOD) detectors. The method leverages a Representation Diffusion Model (RDM) to synthesize OOD representations that lie between in-distribution and far-OOD regions, i.e., neither too close nor too distant from the in-distribution data. Using these synthesized representations, MAGE is then employed to generate corresponding OOD images. After training the detector with the generated outliers, the method further applies confidence calibration to enhance detection performance. Experimental results show that the proposed method outperforms recent outlier synthesis baselines such as Dream-OOD.

### Strengths
- Leveraging self-supervised representations instead of labeled data appears useful.
- The RDM-based outlier synthesis approach seems effective.
- The analysis on semantic and covariate shifts is interesting and provides useful insights.

### Weaknesses
My main concerns lie in the experimental results and writing clarity.

1. Experimental results
    - In Table 1, SONA significantly outperforms Bad-OOD on CIFAR-100 in-distribution data. It would therefore be helpful to compare Bad-OOD with SONA on other datasets as well. For instance, SONA reports results on ImageNet-200 (ID) vs various OOD datasets (e.g., SSB-hard, iNaturalist). The authors could evaluate their method in this setup for a more comprehensive evaluation.
    - Additional near-OOD datasets (e.g., SSB-hard) should also be included for a more comprehensive evaluation.
    - On ImageNet-100, the performance gap between Bad-OOD and Dream-OOD is marginal. Moreover, as the authors mentioned, Dream-OOD with confidence calibration achieves comparable results with the proposed method, suggesting that the proposed components (Section 3.2~3.3) might have limited impact on effective outlier synthesis.
    - In Table 3, confidence-based filtering reduces the number of OOD samples of types I/II/III by roughly 15-25%. The authors note that "most removed samples belong to OOD-I and OOD-III", but this observation may simply reflect the small number of OOD-II samples. Since OOD-I and OOD-III correspond to near-ID and far-OOD cases, respectively, the filtering procedure does not appear to remove these effectively.
    - In Figure 4(a), the proposed method exhibits lower cross-class diversity than Dream-OOD. Lines 412-422 provide no explanation for this observation. What does this result imply?
    - In the computational cost analysis (L444-447), comparisons with other methods should be provided.
2. Writing clarity
    - Several details are missing. For example, what exactly does $\hat{z}_0(z_t)$ represent, and why is $z_t^0$ required instead of directly using $z_0$? In addition, how is the guided gradient applied in practice?
    - This paper introduces many hyperparameters, for example, the uncertainty regularization weight $\beta$, the numbers of timesteps $t$ and $s$, the number of training epochs, and the choice of detector architecture. However, this paper provides no discussion of how they are selected. Some analysis on hyperparameter sensitivity should be provided.
    - When training the OOD detector, are all parameters optimized from scratch? Or, is it fine-tuned from a pretrained model?
    - For calibration, which dataset is used? Is it simply the training dataset containing ID and synthetic OOD images?
    - Were the Table 1 results obtained after additional fine-tuning using mid-confidence OOD samples? The paper mentions that the OOD detector is further tuned using such samples after calibration (L77-78), but this additional procedure is not described in the Method section.

### Questions
See the weaknesses part.

### Soundness
2

### Presentation
2

### Contribution
2
