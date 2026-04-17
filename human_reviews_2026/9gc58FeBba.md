# Provenance-Enabled Multi-View Diabetic Retinopathy Diagnosis Through Interpretable Process Mining

- Decision: Reject
- Scores: 4, 2, 8, 6

## Abstract
Diabetic retinopathy (DR) is a leading cause of blindness among individuals with diabetes. Although the existing deep learning models have demonstrated potential in DR diagnosis, they still lack full-process interpretability. Specifically, these models suffer from three key challenges: reliance on single-source inputs, opaque and untraceable reasoning processes, and the absence of a mechanism for result verification. To meet the requirements of the medical scenario for a trustworthy diagnostic model, we propose a provenance-enabled concept-based framework for multi-view DR diagnostic (ProConMV). This work integrates DR lesion masks, clinical text and multi-view data, utilizing multimodal prompt analysis and visual-text concept interaction to learn the interpretable multi-source input. During the reasoning stage, the proposed framework introduces lesion concepts for causal reasoning chains combining clinical guidelines, and adds doctor intervention for human-machine collaboration. For dynamic fusion decision and verification in multi-view DR diagnosis, we derive via generalization theory that incorporating each view’s lesion concept uncertainty and grading uncertainty reduces the generalization error upper bound. Accordingly, we design a dual uncertainty-aware module to enable provenance-based verification, ultimately enabling verifiable analysis of DR diagnostic results. Extensive experiments conducted on two public multi-view DR datasets demonstrate the effectiveness of our method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a multi-view framework for diabetic retinopathy grading, incorporating concept-based reasoning, RWKV backbone, and an uncertainty-aware mechanism. The topic is highly relevant and the results are remarkable. However, there are several aspects that need further clarification and discussion to strengthen the overall narrative and contribution.

### Strengths
1. The topic discussed in the paper is important for AI-assisted medical analysis.
2. The propsoed method shows a promising performance to enhance the transparency of the diagnosis process.

### Weaknesses
###Major concerns
1. L130-131, 'We observe that existing Transformer-based multi-view methods are less effective at fine-grained local concept perception'. There appear to be no empirical results or evidence provided to support this claim. It is recommended to add relevant experimental comparison or references to prior work
2. The motivation behind introducing the "multi-directional Hilbert attention mechanism" is unclear. Is there any benefit to introducing Hilbert curve?
3. Why was TE3 chosen as the encoder instead of alternatives specifically designed for medical applications? Domain specific encoders are often tailored to extract relevant features, and it would be helpful to discuss why TE3 was preferred over others in this instance.
4. L236-238, "Through this fusion, the model obtains each view’s lesion concept embeddings that are aligned with both visual information and diagnostic knowledge, thereby enhancing the interpretability and predictive accuracy of the concepts." It is unclear why the interpretability could be enhanced by this fusion
5. How is a view determined to have "poor interpretability"? (L342) I don't think the uncertainty is equivalent with interpretability
6. The role and mechanism of the mask mentioned in various parts of the paper are vague. Where exactly is the mask applied, and how does it operate within the framework? Are those compared methods integrated with masks?
7. Is it possible to release the annotated concept labels? It is critical for reproducing the results.
8. The ablation results are substantial, but the impact of each module is not thoroughly discussed. e.g. DU-MVFD in Tab. 3. showed only a slight drop in concept prediction metrics compared to the baseline. Yet, significant performance gains were observed when combined with Hilbert-RWKV and VT-RWKV. It is recommended to have more discussion highlighting the effect of modules and the combination of them to help readers understand the mechanism
9. My main concern centers on interpretability, which appears somewhat contradictory to the paper's core contribution. While the multiview setup improves accuracy and it is intuitive, it actually introduces negative implications for interpretability since decision-making based on multiple views is inherently more complex than single-view reasoning. The uncertainty-aware mechanism further complicates what should be a straightforward c→y process. Complexity typically means harder to interpret. That said, I'm not suggesting the proposed method doesn't make sense, but this seems to conflict with the paper's stated goal of "effectively solving the problem of opaque and untraceable reasoning processes in traditional models." 

### Minor Concerns:
1. Line 141: "fundas" -> "fundus"
2. The generalization theory in Section 2.3 is commendable but could benefit from improved narrative flow. There’s a noticeable gap between the first paragraph of 2.3 and 2.3.1 that affects readability.

### Questions
Please check the weakness part.

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
5

### Summary
This paper introduces ProConMV, a provenance-enabled multimodal and multi-view framework for diabetic retinopathy diagnosis. The approach integrates lesion masks, GPT-generated clinical texts, and fundus images via a Hilbert-RWKV backbone, a Visual-Text RWKV reasoning module, and a dual uncertainty-aware fusion mechanism. While the framework is technically elaborate and theoretically motivated, the motivation and experimental design raise serious concerns. The multimodal data are synthetically constructed (segmentation-derived lesion masks, GPT-generated text, and simulated physician feedback), which undermines fairness, reproducibility, and clinical validity. Moreover, the model’s complexity appears disproportionate to the relatively simple five-class DR grading task, and the performance gains are marginal. The practical applicability of the proposed multi-view design to DR imaging is also questionable.

### Strengths
1. The framework integrates several advanced components, including RWKV attention mechanisms, uncertainty-aware fusion, and concept-level reasoning.
2. The authors provide a formal generalization bound for multi-view concept-based fusion, which adds some theoretical rigor.
3. The architecture is clearly presented, and the empirical results on public datasets are reasonable.

### Weaknesses
1. The multimodal inputs (lesion masks and clinical texts) are synthetically generated, raising major concerns regarding data validity, fairness, and reproducibility.
2. The paper provides no evaluation of segmentation accuracy or potential biases in the GPT-generated texts. Supervision leakage between modalities may possibly occur.
3. The proposed model is excessively complex for a relatively simple five-class DR grading task, yielding only marginal improvements in accuracy.
4. The ‘multi-view’ assumption is not well-grounded for fundus imaging, where multi-view acquisitions are uncommon in clinical settings. The method would be more convincing if validated on genuinely multi-view modalities (e.g., ultrasound, CT), which better reflect the intended design motivation.
5. The interpretability claims are largely theoretical, no user studies or physician-based validation are included to substantiate them.

### Questions
1. How was the accuracy of the generated lesion masks validated, and how sensitive are the results to segmentation errors?
2. Are the GPT-generated clinical texts consistent across samples with identical DR grades but differing lesion distributions?
3. Could the proposed approach be extended to genuinely multi-view medical imaging tasks, such as ultrasound or CT, where the ‘multi-view’ assumption naturally holds?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
ProConMV is a provenance-enabled, concept-bottleneck framework for multi-view DR diagnosis. Contributions: (i) *Hilbert-RWKV*, a linear-time vision backbone that preserves 2D locality via Hilbert-curve tokenization; (ii) *VT-RWKV* to align visual lesion concepts with textual clinical descriptions (fixed, LLM-derived) and enable clinician test-time intervention; (iii) a *multi-view CBM generalization bound* motivating dynamic fusion: with per-view predictions $\hat{y}^{(v)}$ and weights $w_v$ ($\sum_v w_v=1$), the late-fusion $\hat{y}=\sum_v w_v \hat{y}^{(v)}$ admits a tighter upper bound when
$$
\mathrm{Cov}\big(w_v,\ \ell_y(G(c^{(v)}),y)\big)\le 0
\quad\text{and}\quad
\mathrm{Cov}\big(w_v,\ | \hat{c}^{(v)}-c^{(v)}|_1\big)\le 0.
$$
Experiments on MFIDDR (4 views) and DRTiD (2 views) report SOTA grading and concept prediction with fast inference.

### Strengths
* **S1: Principled fusion for multi-view CBMs.** The bound explicitly exposes covariance terms, giving a clean criterion for dynamic, reliability-aware fusion instead of ad hoc averaging.

* **S2: Efficient locality-aware backbone.** Hilbert-RWKV is a clear, linear-time adaptation of RWKV to images with locality preserved by the Hilbert curve; ablations show gains over ResNet/ViT/VMamba.

* **S3: End-to-end interpretability.** Concept-level predictions align with clinical findings and support test-time clinician intervention; uncertainty provides verifiable provenance.

* **S4: Solid empirical results.** Consistent improvements on grading and lesion concepts with competitive latency; ablations attribute gains to each module.

### Weaknesses
* **W1: Inputs at inference are under-specified.** The abstract/figures imply *Seg* masks and "clinical text." Section 2.1.4 suggests the text is *fixed, offline* lesion descriptions (not patient notes). Clarify precisely what modalities are *required at test time*: images only? images+seg masks? any patient text?

* **W2: Theory $\to$ practice bridge is assumed.** The bound requires negative correlation between $w_v$ and *loss/error*, but the implementation weights by *uncertainty*. Please verify (with plots/statistics) that per-view uncertainties are positively correlated with per-view grading loss and concept error, so Eq. (16) instantiates the bound's prescription.

* **W3: Context vs. locality-preserving Transformers.** No Swin/hierarchical-window baseline is reported. A brief comparison (accuracy/latency or discussion) would better situate Hilbert-RWKV among locality-aware designs.

* **W4: Reproducibility of text embeddings.** Provide the LLM prompts, curation steps, and sensitivity to paraphrases; VT-RWKV depends on these fixed descriptions.

* **W5: Minor notation/editorial.** Eq. (17) mixes $\alpha/\lambda$ and inconsistent indices; clarify focal-loss form (multi-class vs. one-vs-rest). Correct "GPT-5 (Achiam et al., 2023)" to GPT-4.

### Questions
1. **Exact test-time inputs:** Are lesion *segmentation masks* provided at inference? Is any *patient-specific* text used, or only fixed lesion descriptions?

2. **Uncertainty vs. error:** Show quantitative correlations (scatter + Spearman/Pearson) between per-view uncertainty and (a) per-view grading loss, (b) per-view concept error; ideally on validation data.

3. **Backbone context:** How does Hilbert-RWKV compare to Swin (accuracy/latency) on your tasks? Any trade-offs you can articulate?

4. **Text prompts:** Release prompts and an ablation on rephrasing quality to establish robustness of VT-RWKV.

5. **Loss details:** Specify the exact focal-loss variant and the tuned weighting between grading and concept losses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper builds a multi-view DR grading system that tries to follow a clinician’s workflow: first detect lesion concepts on each fundus view, then map those concepts through a guideline-style reasoning chain to a grade, and finally fuse views by letting more reliable evidence speak louder. Concretely, it uses a lightweight Hilbert-RWKV backbone for local concept perception, a visual–text interaction module to align images with lesion terms, and a dual-uncertainty scheme (at the concept and grade levels) to drive dynamic fusion; the pipeline is provenance-aware so a reviewer or clinician can correct a concept at test time and see the final grade update accordingly. The authors also argue—via a simple generalization bound—that assigning higher weights to lower-loss/uncertainty views is preferable to static averaging. On MFIDDR and DRTiD, the method reports better grading and concept recognition than standard multi-view baselines, with ablations supporting the contribution of the backbone, the vision–text alignment, and the uncertainty-aware fusion, and with favorable efficiency numbers.

### Strengths
A clear, provenance-aware pipeline that mirrors clinical practice: detect lesion concepts per view, reason via guideline-like rules, then fuse views. The evidence chain is exposed end to end, so test-time corrections to concepts transparently update the final grade—useful for auditing and real clinical use.
The dynamic fusion is principled and easy to grasp: concept- and grade-level uncertainties steer weights toward more reliable views, with a simple bound to justify why this should beat static averaging. Ablations largely support the mechanism, not just the outcome.
Thoughtful engineering with broad empirical coverage. A lightweight backbone and vision–text interaction keep inference efficient, and results on two multi-view datasets—plus intervention and ablation studies—make a convincing case for utility when views are noisy or partially informative.

### Weaknesses
Dynamic fusion relies on uncertainty to allocate weights, but the article does not fully demonstrate that these uncertainties are well calibrated and consistent across perspectives. If the confidence level is too high or too low, or if the confidence scales from different perspectives are inconsistent, the fusion process may actually amplify the noise, ultimately affecting the credibility and reproducibility of the conclusion. It is suggested to supplement temperature calibration here ECE/ACE or risk coverage curve and cross perspective consistency analysis.

The author's theoretical motivation is that "weight should be negatively correlated with loss/uncertainty", but this lacks direct empirical evidence and counterfactual comparison. A more ideal approach would be to provide a statistical analysis of the correlation coefficient between weights and losses (or uncertainties) during the training process over iterations, and to design a comparison of scrambled uncertainties or using uniform weights to demonstrate how performance degrades when the current lift is disrupted, thus truly closing the chain of "why it works".

In terms of practicality, stronger clinical transferability and interactive adaptation evidence may be needed, such as robustness in out of domain settings (different devices, hospitals, imaging conditions); Degradation curve and automatic weight reduction capability when there is a lack of perspective or poor perspective quality; And interactive intervention experiments involving real readers (time constraints, net benefits/costs of one intervention, reader consistency). These will determine whether the method can be smoothly embedded into the actual workflow, rather than just performing well within the dataset.

### Questions
The uncertainty story is not fully convincing. I would like pre-/post–temperature ECE, ACE, and risk–coverage, and a cross-view consistency check: map each view’s confidence to the same calibration curve and compare, to confirm they are on the same scale and won’t bias fusion weights due to scale differences.

The “higher uncertainty → lower weight” mechanism also needs more direct evidence. A formal theorem is not necessary, but please provide a simple diagnostic: correlations between fusion weights and losses/uncertainties during training (or at convergence). Add two controls: shuffle the uncertainties, and enforce uniform weights. If performance drops under these controls, the mechanism assumption is more credible.

Please clarify transferability and robustness. Provide a small out-of-domain result and degradation curves for missing or low-quality views (randomly dropping 1–N views or adding real noise/occlusion), and show whether the model automatically down-weights such views. Also explain the source and order of the clinical text and concept labels, and run a weak/no-text ablation (remove or weaken text; report text-only and image-only ceilings) to rule out template or closed-loop labeling shortcut effects.

### Soundness
2

### Presentation
3

### Contribution
2
