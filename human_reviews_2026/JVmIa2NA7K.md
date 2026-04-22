# Maximally Useful and Minimally Redundant: The Key to Self Supervised Learning for Imbalanced Data

- Avg Score: 2.80
- Decision: Reject
- Scores: 4, 2, 2, 2, 4

## Abstract
Contrastive self supervised learning(CSSL) usually makes use of the "multiview" assumption which states that all relevant information must be shared between all views. The main objective of CSSL is to maximize the mutual
information(MI) between representations of different views and at the same time  compress irrelevant information in each representation. Recently, as part of future work, Schwartz Ziv \& Yan LeCun pointed out that, when the multi-view assumption is violated, one of the most significant challenges in SSL is in identifying new methods to separate relevant from irrelevant information based on alternative assumptions. Taking a cue from this intuition we make the following contributions in this paper: 1) We develop a CSSL framework wherein multiple images and multiple views(MIMV) are considered as input, which is different from the traditional multiview assumption 2) We adopt a novel augmentation strategy that includes both normalized (invertible) and augmented (non-invertible) views so that complete information of one image can be preserved and hard augmentation can be chosen for the other image 3) An Information bottleneck(IB) principle is outlined for MIMV to produce optimal representations 4) We introduce a loss function that helps to learn better representations by filtering out extreme features 5) The robustness of our proposed framework is established by applying it to the imbalanced dataset problem wherein we achieve a new state-of-the-art accuracy (2\% improvement in Cifar10-LT using Resnet-18, 5\% improvement in Cifar100-LT using Resnet-18 and 3\% improvement in Imagenet-LT (1k) using Resnet-50).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
In contrastive self-supervised learning (CSSL), when the data follows a long-tailed or imbalanced distribution, the traditional approach of generating two augmented views from the same image performs poorly on tail classes (categories with few samples). To enable self-supervised multi-view frameworks to extend to settings involving more than two views (MTTV), this paper proposes a method to filter out extreme or uninformative features, thereby facilitating the learning of better representations. Experiments on imbalanced self-supervised datasets demonstrate that the proposed method achieves strong performance.

### Strengths
S1: The motivation of this work is reasonable. Extending from two-view contrastive learning to cross-image and cross-view fusion is intuitive and practically meaningful.

S2: From an information-theoretic perspective, exploring representations that are maximally useful and minimally redundant is a logical and well-grounded research direction.

S3: The experimental results demonstrate the effectiveness of the proposed method. Across several imbalanced datasets (CIFAR10-LT, CIFAR100-LT, and ImageNet-LT (1K)), the method consistently achieves performance improvements over existing methods.

### Weaknesses
W1: The theoretical analysis relies on a coarse approximation where the information content of different views is simply summed. This makes Section 3.1 appear somewhat superficial. A more rigorous treatment is needed, including a formal derivation or conditions under which this approximation holds.

W2: The writing of the paper is somewhat verbose, which obscures the main contributions. In particular, a substantial portion of the paper (starting around line 063) is devoted to reintroducing methods such as SimCLR, which appears redundant.

W3: As a consequence of (2), the authors do not clearly isolate and evaluate the contribution of each key component. The experimental section therefore lacks clarity and persuasive power, and would benefit from more systematic ablation studies to demonstrate the necessity of each design choice.

### Questions
Given that the Related Work section is not very detailed, could the authors supplement it with a brief summary of recent papers on long-tailed self-supervised and semi-supervised learning, and select appropriate strong baseline algorithms to add to the experimental section?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes MTTV, a self-supervised contrastive framework for long-tailed data that:
(i) forms compact fused representations from more-than-two views (augmented and normalized), and
(ii) masks extreme similarities via a thresholded InfoNCE variant.
Experiments on CIFAR10-LT/100-LT and ImageNet-LT subsamples report linear-evaluation gains over prior SSL baselines such as SimCLR, BYOL, SwAV, SDCLR, and FASSL.

### Strengths
* Shows clear empirical gains on CIFAR-LT and ImageNet-LT over several standard SSL baselines.
* The proposed “extreme-feature elimination” (thresholded similarity filtering) is a simple idea that appears to stabilize training and may reduce representation collapse.

### Weaknesses
* Missing Comparison to Multi-View Self-Supervised Learning:
The manuscript repeatedly claims novelty in using more-than-two views and redundancy suppression, yet fails to compare to the extensive multi-view SSL literature where these principles were introduced and validated, including:
    * An Embedding-Dynamic Approach to Self-Supervised Learning (WACV 2023)
    * VICReg (ICLR 2022)
    * SwAV (NeurIPS 2020)
    * MIL-NCE: End-to-End Learning of Visual Representations from Uncurated Instructional Videos (CVPR 2020)
    * Learning by Sorting: Self-Supervised Learning with Group Ordering Constraints (ICCV 2023)
Moreover, the proposed “feature filtering” closely resembles hard-negative mining; a direct comparison to such methods is missing.
* Incorrect citation: FASSL is listed as an ICCV proceedings paper; it is actually an ICCV 2023 Workshop  paper. Please correct this.
* Missing baselines and discussion: The paper omits “Contrastive Learning with Boosted Memorization” (ICML 2022), which is conceptually close. It should be included in comparisons and discussed in detail.
Additionally, evaluations should consider more recent visual-only backbones such as DINOv3 and SigLIP-v2 to ensure fair benchmarking.
* Motivation vs. Method Mixing: The introduction reads like a methods section, filled with equations and implementation details but little conceptual framing or problem motivation. The broader context of long-tailed SSL is not well developed.
* Questionable novelty framing: The abstract credits a LeCun comment as inspiration for multi-view design, while multi-view SSL has been explored for years (see above). This framing misrepresents the originality of the idea.
* Figures and clarity: Figures are small, crowded with equations, and not self-contained. They do not effectively illustrate the pipeline or contributions.
* Insufficient ablations: No analysis disentangles the impact of the fusion mechanism, thresholding, or other design choices; thus, the source of reported gains remains unclear.

### Questions
see above

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper targets the robustness of self-supervised contrastive learning (CSSL) on long-tailed/class-imbalanced image data. It proposes “More Than Two Views (MTTV)”: unlike conventional CSSL that generates two augmented views of the same image for contrastive learning, the authors advocate using multi-views from two different images, mixing “invertible normalized views” and “non-invertible augmented views,” building more compact representations via “fusion” (sum or concat), and optimizing with a modified NT-Xent (InfoNCE)-style loss. Notably, this appears to be a very rushed submission; the presentation is very poor, making the paper extremely hard to read.

### Strengths
The authors completed a paper.

### Weaknesses
1. Setting aside the method itself, the presentation is very poor and falls far short of ICLR standards.  **I consider such a carelessly prepared submission disrespectful to reviewers.**

a. The abstract is filled with redundant, uninformative content. The core “what the paper does” is reduced to a single short sentence, which does not meet the norms of an abstract.

b. The introduction is excessively long (three pages). For example, there is no need to describe how contrastive learning works in such detail in the introduction; it gives the impression of padding pages.

c. Numerous formatting errors: an obvious extra space at line 59; equations are sometimes referenced as “Equation: 1” and sometimes as “Equation (1)”; “Knn” and “KNN” are mixed at line 432; and many other nonstandard expressions, such as missing punctuation after display equations. The authors need careful proofreading and revision.

d. The notation is very confusing, which makes the paper hard to read.

2. The assumption about the invertibility of the “normalized view” is vague: there is no clear specification of the transformation (pixel normalization or representation L2-normalization?), and DPI/MI are misused/oversimplified; the numerical analysis of “information volume/shared information” is subjectively assigned (e.g., 0.8) and lacks rigorous derivation or estimation methodology.

3. The paper treats fused representations from two different images as positive pairs without any mechanism to ensure semantic sharing, which contradicts the standard SSL assumption that positives are different views of the same instance, and is likely to cause optimization confusion or collapse.

4. Critical implementation details are missing: data augmentation pipeline, the exact definition and implementation of “normalized,” temperature, projection dimension, whether a queue/memory bank is used, composition of negatives, and how thresholding is applied to the InfoNCE denominator.

5. The paper does not explain why the proposed method should work specifically in long-tailed (LT) scenarios.

### Questions
None.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses self-supervised learning (SSL) under imbalanced or long-tailed data distributions.
The authors argue that conventional contrastive SSL methods (e.g., SimCLR, MoCo) are biased toward head classes because they only form intra-image positive pairs.
To mitigate this, the paper introduces a “More-Than-Two-View” (MTTV) framework, inspired (as claimed) by Yann LeCun’s discussions on multi-view SSL.
The key idea is to construct cross-image feature fusion pairs between normalized and augmented views from different images, aiming to learn maximally useful and minimally redundant representations. The approach is evaluated on CIFAR-10/100-LT and ImageNet-LT.

### Strengths
1. Problem Motivation:  The paper focuses on an underexplored but important issue—class imbalance in self-supervised representation learning.


2. High-Level Idea:  Extending contrastive learning beyond single-image pairs to multi-image fusion introduces an interesting direction that may encourage feature diversity.

### Weaknesses
**1. Conceptual Vagueness:**
 The MTTV concept (“More Than Two Views”) is poorly defined, with no formal reference or clear operational meaning.
 The citation to LeCun is not substantiated by a verifiable publication.


**2. Weak Theoretical Grounding:**
 Although framed as information-theoretic, the paper does not compute or approximate mutual information. The Information Bottleneck formulation serves more as a conceptual narrative than a measurable objective.


**3. Lack of Experimental Rigor:**
- Section 4 Results section only reports main benchmark results (Tables 1–3). 
- No ablation studies are provided to isolate the contribution of each component (fusion type, filtering thresholds, etc.).
- No analysis explaining why the method works (e.g., embedding visualization, MI analysis, or per-class behavior).


**4. Unconvincing Baseline Selection:**
 The comparisons are only made against standard SSL baselines (SimCLR, MoCo v2, BYOL, DINO) that do not address imbalance.
 Recent imbalance-aware SSL methods (e.g., Kukleva et al., ICLR, 2023, Bai et al., ICLR, 2023) are omitted.
 Consequently, the reported improvements (~2–5%) may reflect baseline selection bias.


**5. Shallow Related Work:**
 The Related Work section only discusses “imbalanced SSL” papers, ignoring the rich literature on long-tailed recognition [1, 2, 3] and SSL methods (SimCLR, MoCo). Including a broader set of prior works would make the paper more self-contained and better positioned within the field.

For example, some representative methods are:

[1] Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss, NeurIPS, 2019 --> Reweighting method 

[2] Parametric Contrastive Learning, ICCV, 2021 --> Supervised Contrastive learning method

[3] The Majority Can Help The Minority: Context-rich Minority Oversampling for Long-tailed Classification, CVPR, 2022 --> Imbalance-aware augmentation method



**6. Speculative Causal Claims:**
 The link between “redundancy reduction” and “imbalance robustness” is asserted but not validated.

### Questions
Please refer to the weaknesses above.

1  How exactly is “More Than Two Views” defined, and can you provide a verifiable reference for LeCun’s supposed remark?

2. Have you quantitatively measured mutual information to support your “maximally useful” claim?

3. Ablations: What is the sensitivity of performance to the filtering thresholds (λₗ, λₕ)?  Is the fusion operation (sum vs. concat) critical to performance, and what are the trade-offs?

4. How does your approach compare against imbalance-aware SSL methods  (e.g., Kukleva et al., ICLR, 2023, Bai et al., ICLR, 2023) ?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of applying contrastive self-supervised learning (CSSL) to imbalanced datasets, a significant and practical problem given that real-world unlabeled data is often long-tailed.1 The authors propose a novel framework, "More Than Two Views" (MTTV), which departs from the standard paradigm of generating two augmented views from a single image. Instead, MTTV processes two distinct images simultaneously, creating views through a unique combination of an invertible transformation (normalization, denoted $x^n$) and a non-invertible one (standard data augmentation, denoted $x^a$).3 The core idea is to generate a richer set of inter- and intra-image feature comparisons, which is particularly beneficial for learning from the sparse data in tail classes.

### Strengths
- Originality of the framework: The core strength of this paper is the conceptual novelty of the MTTV framework. By processing multiple images at once and, most importantly, by combining normalized (invertible) and augmented (non-invertible) views, the authors propose a new and effective way to generate pairs for contrastive learning. This design is well-suited for the low-data regime characteristic of tail classes, as it generates a richer set of training signals from each sample.

### Weaknesses
- Overstated Novelty: The paper frames "more than two views" as a key novelty. However, the use of multiple views from a single image is well-established, most notably through the multi-crop augmentation strategy in SwAV (Caron et al., 2020). More recent work, such as "Multiple Positive Views in Self-Supervised Learning", has also explicitly analyzed this setting. The authors fail to differentiate their multi-image approach from these existing single-image, multi-view methods.
- Insufficient Context: Works of SSL for long-tailed recognition are missed. In the specific research direction this paper focuses on—modifying the contrastive learning objective—several highly relevant papers were published in 2023-2025. For instance, "Decoupled Contrastive Learning" (AAAI 2024), BaCon (OpenReview 2024), "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments" et al.
- Experimental Problems:
    - Baseline discrepancy: Under-reporting of the SDCLR baseline on CIFAR10-LT (80.49% in Table 1 vs. 82.00% in the original paper). This calls into question the true magnitude of the improvement and should be corrected.
    - Selective comparison on ImageNet-LT: In Table 3, the comparison on ImageNet-LT is limited to a supervised baseline and MoCo v2. This is an incomplete comparison. For instance, the BCL paper (Zhu et al., CVPR 2022) reports a Top-1 accuracy of 56.7% on ImageNet-LT with a ResNet-50 backbone, which is substantially higher than the 52.90% reported here. This suggests the proposed method is not state-of-the-art on this benchmark, contrary to the paper's claims.
    - Comparing baselines are out of date: The latest baseline is FASSL(2023). The rest highly related works we listed above are not even discussed.
    - Missing Ablation Studies: The paper fails to disentangle the contributions of its three main components: (1) the multi-image setup, (2) the normalized+augmented view strategy, and (3) the thresholded loss. Without ablation studies, it is impossible to determine which of these components is responsible for the performance gains. A rigorous evaluation would require experiments that isolate each component—for example, running MTTV with only augmented views, or running the full framework without the loss thresholds. The absence of these experiments is a major flaw.

### Questions
As discussed in the "Weaknesses" section, could you please explain more about the novelty, related work discussion, and experimental problems?

### Soundness
3

### Presentation
2

### Contribution
2
