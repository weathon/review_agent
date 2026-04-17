# EReLiFM: Evidential Reliability-Aware Residual Flow Meta-Learning for Open-Set Domain Generalization under Noisy Labels

- Decision: Reject
- Scores: 4, 4, 4, 4, 6

## Abstract
Open-Set Domain Generalization (OSDG) aims to enable deep learning models to recognize unseen categories in new domains, which is crucial for real-world applications. Label noise hinders open-set domain generalization by corrupting source-domain knowledge, making it harder to recognize known classes and reject unseen ones. While existing methods address OSDG under Noisy Labels (OSDG-NL) using hyperbolic prototype-guided meta-learning, they struggle to bridge domain gaps, especially with limited clean labeled data. In this paper, we propose Evidential Reliability-Aware Residual Flow Meta-Learning (EReLiFM). We first introduce an unsupervised two-stage evidential loss clustering method to promote label reliability awareness. Then, we propose a residual flow matching mechanism that models structured domain- and category-conditioned residuals, enabling diverse and uncertainty-aware transfer paths beyond interpolation-based augmentation. During this meta-learning process, the model is optimized such that the update direction on the clean set maximizes the loss decrease on the noisy set, using pseudo labels derived from the most confident predicted class for supervision. Experimental results show that EReLiFM outperforms existing methods on OSDG-NL, achieving state-of-the-art performance. The source code is available at https://anonymous.4open.science/r/ERELIFM-CBCB/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses Open-Set Domain Generalization under Noisy Labels (OSDG-NL), where models must recognize unseen categories in new domains while training data contains label noise. The authors propose EReLiFM, which consists of three main components: (1) Unsupervised Two-Stage Evidential Loss Clustering (UTS-ELC) to separate clean from noisy samples using evidential loss trajectories, (2) Domain and Category Conditioned Residual Flow Matching (DC-CRFM) to generate diverse transfer paths across domains and categories, and (3) a meta-learning framework that trains on clean/augmented data and cautiously incorporates noisy samples with pseudo-labels. Experiments on PACS, DigitsDG, and TerraINC show improvements over the baseline HyProMeta.

### Strengths
1. **Important problem**: OSDG-NL is a relevant and challenging problem combining multiple realistic constraints.

2. **Comprehensive experiments**: The paper evaluates on multiple datasets, noise types (symmetric/asymmetric), and noise ratios (20%/50%/80%).

3. **Consistent improvements**: Results show improvements across most settings, though statistical significance is unclear.

4. **Thorough ablations**: Table 7 provides ablations of major components, though clarity could be improved.

### Weaknesses
1. **Weak theoretical justification**: Why should evidential loss trajectories be better than standard loss for clean/noisy separation? The paper provides intuition but no theoretical analysis or proof.

2. **Computational overhead**: Adding DiT (129.6M params) during training significantly increases computational cost. The paper doesn't analyze whether simpler augmentation strategies could achieve similar results.

3. **Failure mode analysis**: What happens when UTS-ELC fails (as suggested by 52-56% accuracy in Table 17)? How does error propagate through the pipeline?

4. **Limited domain diversity**: PACS has only 4 domains with similar image statistics. More diverse domains (e.g., natural images → medical images) would better validate the approach.

5. **Hyperparameter sensitivity**: No analysis of sensitivity to key hyperparameters (N_e, number of GMM components, meta-learning rates, etc.).

6. **Comparison fairness**: Some baselines (TCL, NPN, BadLabel) are not designed for OSDG, making comparisons less meaningful. More fair comparisons would be against other meta-learning or domain generalization methods adapted for noisy labels.

### Questions
1. **UTS-ELC failure modes**: Given the low separation accuracy (~55%) under high noise, how does the method remain robust? Can you provide analysis of what happens when clean samples are misclassified as noisy?

2. **Flow matching vs. MixUp**: Can you provide theoretical or empirical evidence that learning residuals via flow matching provides fundamentally different augmentations than MixUp? Visualizations of generated samples would help.

3. **Ablation details**: Please clarify the exact difference between "w/o UTS-ELC in RFM" and "w/ UTS-LC in RFM" in Table 7.

4. **Cross-dataset generalization**: How does a model trained on PACS with DC-CRFM perform when directly tested on DigitsDG without retraining?

5. **Hyperparameter selection**: How were hyperparameters (especially N_e=10) chosen? Is this consistent across datasets and noise levels?

6. **Statistical significance**: Can you provide error bars or significance tests for the main results?

7. **Pseudo-label quality**: What is the accuracy of pseudo-labels y_pseudo in the meta-test stage? How does this correlate with final performance?

### Soundness
3

### Presentation
2

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
This paper studies the problem of open-set domain generalization under noisy labels. The proposed method consists of two modules: denoising (dataset separation) and training. In the denoising stage, the authors use Evidential Learning to estimate the uncertainty of samples, then apply a clustering algorithm to divide them into a clean set (low uncertainty clusters) and a noisy set (high uncertainty clusters). Training is conducted within a meta-learning framework, where the meta-train data come from the original and flow-model-augmented clean set, while the meta-test data are derived from the noisy set with pseudo-labels. Experimental results demonstrate the effectiveness of the proposed approach.

### Strengths
1. The paper is clear and easy to understand.
2. The application of Evidential Learning, though rarely seen in open-set recognition，seems reasonable.
3. Experimental results demonstrate the effectiveness of the proposed method.

### Weaknesses
1. The choice to use the clean set for meta-training and the noisy set for meta-testing appears odd. Performance gains can often be achieved through various tuned meta-learning strategies—whether traditional sampling from the same distribution or MEDIC’s domain-class sampling. Therefore, the authors should provide stronger justification that their approach offers unique advantages. Moreover, the meta-learning experiments should include comparisons across different meta-learning strategies, rather than only ablation studies (e.g., removing the pseudo-labeling module), since it is somewhat obvious that pseudo-labeling would help.
2. The proposed dataset separation strategy always produces a clean set and a noisy set. However, on a completely clean dataset, this could have adverse effects: if the noisy set is small, there would be insufficient meta-test data; if it is large, many correct samples could be incorrectly assigned pseudo-labels.
3. The paper repeatedly emphasizes the advantage of the flow model over Mixup interpolation, yet the evidence is limited to accuracy comparisons. Given the emphasis placed on this claim, more qualitative or intuitive analyses are needed to substantiate it. Furthermore, Mixup is not the only data augmentation approach worth comparing.
4. The studies cited in lines 104–105 are flow matching, and some studies referenced in lines 386–388 are close set domain generalization. They are not open set domain generalization.

### Questions
1. Please see weaknesses.
2. What is the motivation for studying open set domain generalization and noisy labels simultaneously? Do the authors assume that either open set domain generalization or domain generalization under noisy labels has already been well-studied?
3. How is y_a chosen in the algorithm?

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
The paper proposes EReLiFM, a three-stage pipeline for Open-Set Domain Generalization (OSDG) under noisy labels. The approach: (i) separates clean from noisy samples via UTS-ELC (Unsupervised Two-Stage Evidential Loss Clustering), (ii) enriches the training data using DC-CRFM (Domain- and Category-Conditioned Residual Flow Matching), and (iii) optimizes a meta-learning objective that decouples clean and noisy supervision. Experiments demonstrate strong performance on PACS, DigitsDG and TerraINC datasets.

### Strengths
Well-motivated pipeline. The experimental results demonstrate substantial improvements over existing baselines across multiple benchmarks.

### Weaknesses
1. The presentation is kind of poor. Instead of spending a whole page (i.e., page 3) presenting the motivation for all components, I think it is better to present a high-level framework, e.g.:

    - separates clean from noisy samples
    - enriches the training data
    - optimizes a meta-learning objective that decouples clean and noisy supervision

   Then, within each subsequent section for individual components, describe the specific motivation and highlight key differences from previous works. This structure may help readers grasp the overall strategy before exploring component-level details.

2. Meta-learning step relies heavily on plain text description. Provide explicit mathematical formulations showing the inner/outer loop objectives, gradient flows, and how clean vs. noisy samples are weighted.

3. Since the proposed method incorporates many existing techniques (Finch clustering, GMM, residual flow matching), adding a Background/Preliminaries subsection that concisely explains these foundational methods would make the paper accessible to readers less familiar with these techniques.

4. Experiments: Rather than relying solely on percentage improvements (which readers can see in the tables), consider including:

    - t-SNE or UMAP plots demonstrating UTS-ELC's ability to separate clean from noisy samples
    - Sample quality comparisons or interpolation visualizations showing DC-CRFM-generated data quality and diversity

5. Acronym introduction. UTS-ELC and DC-CRFM are used before being introduced (lines 64 and 66).

### Questions
Please see the weakness above.

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
2

### Summary
This paper attempts to tackle the problem of Open-Set Domain Generalization under Noisy Labels (OSDG-NL). The authors proposed an approach based on evidential loss and residual flow, named Evidential Reliability-Aware Residual Flow Meta-Learning (EReLiFM). EReLiFM includes two modules, named UTS-ELC and DC-CRFM, respectively. UTS-ELC promotes better clean/noise separation across domains. DC-CRFM could augment more data with structured residuals. Subsequently, these two modules are integrated within a meta-learning framework. Experiments demonstrates that EReLiFM could enhance the performance of noise diagnosis and data augment for OSDG-NL.

### Strengths
[+] The detail of the method and experiment is well described.

[+] The related work is detailed.

[+] The experiments conducted are extensive.

### Weaknesses
Major weakness:

[-] The authors propose UTS-ELC to better separate clean and noisy samples using evidential loss. Please clarify the mechanism and explain its advantage in achieving more reliable separation compared to state-of-the-art methods such as HyProMeta. In addition, please provide visualizations or other metrics to demonstrate this claimed advantage.

[-] The authors propose DC-CRFM to expand clean data with structured diversity. However, there is a lack of visual results illustrating the advantage of DC-CRFM in enhancing diversity compared to existing augmentation methods (e.g., MixStyle [1] and FACT [2]).

[-] The benchmarks used appear limited. Can the proposed approach consistently achieve better performance on commonly used domain generalization datasets such as OfficeHome and VLCS? 

[1]. Domain Generalization with MixStyle. ICLR, 2020.

[2]. A Fourier-based Framework for Domain Generalization. CVPR, 2021.

Minor weakness:

[-] The manuscript requires further refinement in details to improve readability. For instance, "DirectFM" in Table 7 is not defined.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel reliability-aware meta-learning framework called EReLiFM,  that combines evidential loss–based clean/noisy data separation with domain- and category-conditioned residual flow matching. It aims to improve open-set domain generalization under noisy labels by filtering clean samples, expanding them through structured residual flows, and recycling noisy data with evidential pseudo-labeling. Experiments on DG benchmarks show that it achieves SOTA performance, outperforming prior methods across multiple noise settings.

### Strengths
1. The paper introduces a novel integration of evidential uncertainty modelling and residual flow matching, effectively addressing noisy labels for open-set domain generalization task.

2. It has proposes a novel framework, EReLiFM that demonstrates strong and consistent performance improvements across multiple benchmarks, showing robustness to different noise levels and backbone architectures.

### Weaknesses
1. See questions
2. Below papers are needed to be cited.

[1] Towards Multimodal Open-Set Domain Generalization and Adaptation through Self-supervision, ECCV 2024
[2] OSLoPrompt: Bridging Low-Supervision Challenges and Open-Set Domain Generalization in CLIP, CVPR 2025

### Questions
(1) Why are the evidential loss trajectories preferred over the feature-space embeddings for clean/noisy sample separation?

(2) What is the reason behind separating clean data for meta-train and noisy data for meta-test rather than mixing them?

(3) Is it possible to check the sensitivity of separating clean and noisy samples with other unsupervised clustering methods like K-means and DBSCAN ?

### Soundness
3

### Presentation
3

### Contribution
2
