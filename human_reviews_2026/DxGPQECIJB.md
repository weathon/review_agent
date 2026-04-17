# D-TPT: Dimensional Entropy Maximization for Calibrating Test-Time Prompt Tuning in Vision-Language Models

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Test-time adaptation paradigm provides flexibility towards domain shifts by performing immediate adaptation on unlabeled target data from the source model. Vision-Language Models (VLMs) leverage their generalization capabilities for diverse downstream tasks, and test-time prompt tuning has emerged as a prominent solution for adapting VLMs. In this work, we explore contrastive VLMs and identify the modality gap caused by a single dominant feature dimension across modalities. We observe that the dominant dimensions in both text and image modalities exhibit high predictive sensitivity, and that constraining their influence can improve calibration error. Building on this insight, we propose dimensional entropy maximization that regularizes the distribution of textual features toward uniformity to mitigate the dependency of dominant dimensions. Our method alleviates the degradation of calibration performance in test-time prompt tuning, offering a simple yet effective solution to enhance the reliability of VLMs in real-world deployment scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies calibration issues in test-time prompt tuning for VLMs. The authors observe that feature distributions in both textual and visual modalities are dominated by a single dimension, causing prediction overconfidence and sensitivity. To mitigate this, the authors introduce Dimensional Entropy Maximization regularization, encouraging uniformity across text embedding dimensions. Extensive experiments across fine-grained datasets and natural domain shift datasets demonstrate that D-TPT improves calibration.

### Strengths
1. This paper analyzes the calibration issue from a new perspective, focusing on dominant feature dimension, which is different from prior diversity-based interpretations.
2. The authors provide extensive experiments demonstrating that DTPT shows beneficial effects on calibration under both fine-grained classification and natural distribution changes.
3. The paper is clearly written and well-organized.

### Weaknesses
1. The primary motivation relies heavily on analysis from individual examples shown in Fig. 2 and Fig. 3. Can this hold across more samples, more datasets, and different backbone architectures?
2. Although calibration results appear strong, D-TPT does not consistently perform better than C-TPT in both accuracy and calibration.
3. Eq. (1) incorrectly formulates the TPT objective, while TPT actually minimizes the marginal entropy for the mean predictions of the selected augmented views. Due to this incorrect objective, the analysis in Eq. (7) and the related geometric interpretation become questionable.
4. “dominant dimension > modality gap > overconfidence”, this causal chain still lacks deeper formal proof and intuitive explanations.

### Questions
1. What is the difference between $\bar{t}_c$ in formula 3 and $t_c$ defined in the PRELIMINARY?
2. Can the proposed method be combined with test-time methods other than TPT, such as TTL?

### Soundness
3

### Presentation
4

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
This paper proposes **D-TPT (Dimensional Entropy Maximization for Test-Time Prompt Tuning)** to improve model calibration in vision-language models like CLIP. The key idea is that a few **dominant feature dimensions** cause modality gaps and overconfidence during TPT. By maximizing the entropy of text feature dimensions, D-TPT regularizes feature distribution to reduce reliance on dominant dimensions. Experiments on multiple fine-grained and domain-shift datasets show that D-TPT achieves **better calibration and stable accuracy** than prior TPT variants, though its **novelty and theoretical depth are limited**.

### Strengths
1. **Motivation is Clear**  
   The paper presents a clear motivation and introduces a novel perspective for analyzing model calibration. It reveals a new source of the modality gap — the dominant dimension — and proposes Dimensional Entropy Maximization (DEM) to suppress the excessive influence of these highly sensitive dimensions.

2. **Presentation is Good**  
   The paper is well-written and logically structured. Starting from the modality gap problem, the authors identify the dominant dimension issue and further propose dimensional entropy maximization. The figures are particularly well-designed — for example, Figure 2 illustrates the discovered phenomenon, and Figure 5 clearly compares the proposed method with prior approaches. I like the figures in this paper.

3. **Experiments are Comprehensive**  
   The experiments are extensive and demonstrate the stability and generality of the proposed method. The setup covers:  
   * 11 fine-grained classification datasets and 4 out-of-distribution test sets (ImageNet-A/R/V2/Sketch)  
   * Two backbone architectures (CLIP-ViT-B/16 and CLIP-RN50)  
   * Five evaluation metrics (Accuracy, ECE, AECE, MCE, AURC)  
   * Additional analyses including average and variance reporting, failure-case analysis, Pareto front analysis, and prompt initialization studies.

### Weaknesses
1. **Limited Novelty and Theoretical Depth**  
   Although the authors claim the proposed method is effective, its novelty is limited, and the theoretical depth is weak. The core idea of D-TPT is highly similar to that of C-TPT and O-TPT — all add feature regularization on top of the TPT framework. D-TPT merely shifts from inter-feature to intra-feature regularization, still using a KLD + λ weighted form, without introducing a new learning mechanism or architecture.  
   While the intuition behind Dimensional Entropy Maximization (DEM) is clear — reducing dominant dimension sensitivity by increasing feature entropy — the theoretical analysis remains mostly empirical.  
   * Section 4.4 relies on geometric interpretation without mathematical derivation or quantitative validation.  
   * Equation (3) defines the KLD loss against a uniform distribution, but it is unclear why this is theoretically equivalent to reducing dominant dimension sensitivity.  
   * No theorem, proposition, or proof is provided to establish a causal link between DEM and calibration error.  
   Overall, the main idea in this paper is more like an empirical heuristic regularization trick.

### Questions
1. I am still not sure why this method performs better than TPT. Unsupervised test-time prompt tuning fundamentally relies on confidence estimation of test samples, and TPT improves accuracy by amplifying model confidence. By contrast, removing dominant text or image features seems to intentionally suppress feature saliency, which should negatively affect model performance. How does D-TPT overcome this potential drawback and even better than the baseline methods on the Pareto front analysis?

2. In Figure 6, what are the underlying reasons for some failure cases? In Section 5.3, Why does D-TPT perform worse on the CLIP-RN50 backbone?

3. Some implementation details remain unclear. For instance, in Table 1, are the reported results averaged over 11 datasets? Given the large domain differences among them, is the proposed method effective across all datasets? Is the method dataset-agnostic — can it be applied to any image classification dataset?

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
4

### Summary
This work is driven by the observation that text and video feature vectors have non-aligned dominant features.  The method therefore seeks to reduce the calibration error of TPT by regularizing the feature vectors so that they are moved toward the all-zero vector.  As expected, this proposal reduces the overconfidence effect of TPT, but it also usually reduces accuracy.

The idea feels a little bit trivial, in the sense that there are many regularization algorithms that push many types of weight vectors and feature vectors toward the all-zero vector.  On the other hand, it has not previously been observed that pushing the prompt text embedding toward the all-zero vector improves calibration, or that the effectiveness of this step is well-motivated by the observation that each text vector is strongly dominated by one of its dimensions.  On balance, it feels like this observation needs a little bit more rigorous analysis before publication.

I'm also concerned about the apparent errors in the presentation of TPT in Eq. (1) and Algorithm 1.

### Strengths
* The connection between dominant features and zero-directed regularization has not been previously applied to prompt tuning, as far as I know.
* Zero-directed regularization of test-time prompts is easy to implement, and if the experimental results hold up, could be quite useful.

### Weaknesses
Motivation/Innovation:

This proposal reduces the distance between an image vector and its corresponding text vector, as claimed, but only because it is shifting ALL feature vectors in the direction of the all-zeros vector (the vector whose sigmoid transformation has the lowest KL divergence to a uniform distribution).  It is well known that regularizing feature vectors will reduce overconfidence: the only new contribution of this paper is to point out that regularization of this kind also works for TPT.

Since this method also reduces accuracy relative to unmodified TPT, it's not clear that it is beneficial.

The motivation for this proposal is the observation, in Figure 2 and Figure 3(a), that the text and image feature vectors are each dominated by one dimension, and that the dominant dimension differs by modality.  Figures 2 and 3(a) only demonstrate this effect for two individual cases, however. The text claims that this example is typical, but provides no proof.  Actually other papers have also reported this effect, but I've never seen any quantification of the size of this effect.

Correctness:

Eq. (1) is an incorrect statement of TPT, and the part of Algorithm 1 that is claimed to reimplement TPT does not do so correctly.  The TPT objective is the class entropy of an averaged probability; the averaged probability is computed as the average across high-confidence augmented images.  Eq. (1) and Algorithm 1 perform averaging of the entropy across images, rather than averaging the probability.  This could have strange effects in some cases, e.g., the proposed incorrect formula might incorrectly select a prompt that causes different augmentated images to confidently predict different answers, in preference to a prompt that causes each augmentation to choose the same answer but with lower confidence.

The averaging should only be across augmented views of the same image; the text before Eq. (1) suggests that i=1 to N includes multiple images, not just multiple augmentations, which would be another error in this supposed reimplementation of TPT.

Results:

Table 3(b) shows that average accuracy degrades when the dominant dimension is replaced by its average, contrary to claims in the text.

Results in Tables 1 and 2 have the proposed algorithm highlighted in all columns, even though it is not the best in all columns.  In particular, TPT usually has better accuracy, O-TPT usually has better AURC, and proposed algorithm usually has the best ECE and AECE.

English usage:

There are a number of small English usage errors, e.g., p. 1 par. 1: Based on the observation... --- This sentence lacks a verb.

### Questions
See "Weaknesses."

### Soundness
2

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
4

### Summary
The paper studies why CLIP-based test-time prompt tuning (TPT) tends to improve accuracy but hurt calibration, and it argues that the root cause is not just lack of inter-class feature dispersion (as in C-TPT, O-TPT) but an overreliance on a few dominant dimensions in the text/image embeddings that create a modality gap. To address this, the authors propose D-TPT, which keeps the standard TPT entropy-minimization loss but adds a dimensional entropy maximization term that pushes each text feature to distribute its mass more uniformly across embedding dimensions, thereby reducing the influence of the dominant dimension. Across 11 fine-grained benchmarks and 4 ImageNet-shift datasets, D-TPT largely preserves the accuracy gains of TPT while recovering or improving calibration (ECE, MCE) compared to existing TPT variants.

### Strengths
* Calibration in CLIP test-time prompt tuning (TPT) has been studied quite a bit lately, with methods like C-TPT [1], O-TPT [2], and A-TPT [3]. Most of these approaches build on the same core intuition first introduced in C-TPT, that improving feature dispersion helps calibration. This paper takes a different angle: instead of proposing yet another dispersion-based variant, it digs into why TPT on CLIP becomes miscalibrated in the first place and points to dominant dimensions / modality gap as a causal factor. This seems like a meaningful technical contribution to the field.

[1] https://arxiv.org/abs/2403.14119

[2] https://arxiv.org/abs/2503.12096

[3] https://www.arxiv.org/abs/2510.26441

* The paper reports multiple calibration error metrics other than ECE, which strengthens the empirical evaluation and makes the conclusions about calibration more reliable.

### Weaknesses
* It would be nice to see if such method works on critical domains such as medical domain. 

* The results show that suppressing the dominant dimension can help, and then infer it is the main driver. But they don’t fully rule out alternative explanations (e.g. regularization just reduces logit range in general). So the 'causal factor' framing is a bit stronger than what the experiments actually prove.

* Since the proposed D-TPT regularizes intra-feature dimensional entropy, whereas prior methods such as C-TPT and O-TPT focus on inter-feature dispersion/orthogonality, it would be valuable to examine whether the two types of regularization are complementary. For example, can D-TPT be applied on top of C-TPT’s feature dispersion term or O-TPT’s orthogonality constraint.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
