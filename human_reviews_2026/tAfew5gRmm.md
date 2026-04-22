# AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Referring Image Segmentation (RIS) aims to segment the object in an image uniquely referred to by a natural language expression. However, RIS training often contains hard-to-align and instance-specific visual signals; optimizing on such pixels injects misleading gradients and drives the model in the wrong direction. By explicitly estimating pixel-level vision–language alignment, the learner can suppress low-alignment regions, concentrate on reliable cues, and acquire more generalizable alignment features.
In this paper, we propose Alignment-Aware Masked Learning (AML), a simple yet effective training strategy that quantifies region–referent alignment (PMME) and filters out unreliable pixels during optimization (AFM). Specifically, each sample first computes a similarity map between visual and textual features, and then masks out pixels falling below an adaptive similarity threshold, thereby excluding poorly aligned regions from the training process. AML does not require architectural changes and incurs no inference overhead, directing attention to the areas aligned with the textual description. Experiments on the RefCOCO (vanilla/+/g) datasets show that AML achieves state-of-the-art results across all 8 splits, and beyond improving RIS performance, AML also enhances the model’s robustness to diverse descriptions and scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Alignment-Aware Masked Learning (AML) for referring image segmentation (RIS). AML computes a patch–token alignment map via PMME (random projections per modality) and then applies an Alignment-Aware Filtering Mask (AFM) that suppresses low-alignment image regions during training only; inference is unchanged. Experiments show consistent gains on RefCOCO/+/g when AML is added to CARIS, smaller gains on DETRIS, early-stage experiments on ReLA, and robustness to synthetic perturbations.

### Strengths
1.  **Plug-and-Play Method with Zero Inference Overhead** - Masking is applied exclusively during training. Consequently, the method requires no architectural changes and incurs no additional compute or memory costs during inference.

2.  **Strong Performance** - AML consistently boosts CARIS performance across all eight RefCOCO/RefCOCO+/RefCOCOg splits. It also transfers effectively to other baselines, such as DETRIS, and shows promising results with ReLA.

3.  **Robustness and Interpretability Analysis** - The paper shows improved performance under seven different visual perturbations (e.g., haze, low-light, occlusion), supported by clear qualitative maps that explain where AML helps.

4.  **Thorough Ablation Studies** - The paper contains comprehensive ablations examining the threshold, projection dimension, projection strategy (random vs. learnable), and masking level (image vs. feature), which validate the design choices.

### Weaknesses
1. **Gaps in Mathematical Derivations.** The appendix asserts
   $$
   \langle W_i v, W_t u\rangle = \tfrac{1}{2}\langle \tilde W z, \tilde W z\rangle,\quad
   \tilde W = \tfrac{1}{\sqrt{2}}\mathrm{diag}(W_i, W_t),\quad
   z = \begin{bmatrix} v \ u \end{bmatrix}.
   $$
   But
   $$
   \langle \tilde W z, \tilde W z\rangle
   = \left\langle \tfrac{1}{\sqrt{2}}\begin{bmatrix} W_i v \ W_t u \end{bmatrix},
   \tfrac{1}{\sqrt{2}}\begin{bmatrix} W_i v \ W_t u \end{bmatrix}\right\rangle
   = \tfrac{1}{2}\big(|W_i v|^2 + |W_t u|^2\big),
   $$
   which is the *sum of squared norms* (no cross term), so it cannot equal $\langle W_i v, W_t u\rangle$ except in degenerate cases. Moreover, with *independent* Gaussian projections $W_i, W_t$ (entries $\sim \mathcal N(0, 1/D_a)$) and unit $v, u$,
   $$
   \mathbb{E}[\langle W_i v, W_t u\rangle] = 0,\qquad
   \mathrm{Var}[\langle W_i v, W_t u\rangle] = \tfrac{1}{D_a},
   $$
   so the projected cross dot *concentrates at 0* as $D_a$ grows, making preservation of a nonzero cross-modal inner product *impossible* under the stated construction. This invalidates the paper’s stated Theorem 1 (cross-modal inner-product preservation under block-diagonal projection) and any corollaries that rely on it (e.g., PMME’s “geometry-preserving” property).

2. **Evaluation Gaps**:
    1. *Lack of Direct Baseline Comparison:* The evaluation is missing a direct comparison against a key alignment-based baseline, Mask Grounding, within the same experimental setting. Since AML is presented as a plug-and-play, alignment-aware strategy similar to Mask Grounding, its claimed universality should be validated by outperforming this baseline under identical, fully trained conditions. Table 2, however, only reports CARIS/DETRIS add-ons and omits Mask Grounding, leaving this critical parity untested. The table should have directly compared Mask Grounding on CARIS/DETRIS with AML on CARIS/DETRIS.
    2. *Omission of MagNet mIoU Score:* Table 1 omits the mIoU score for MagNet, an important alignment baseline. 

3. **Training Efficiency:** Training and memory costs from implementing AML should be put in the main paper, not the appendix.

4. **Generalization / Real-world robustness:** The paper's visualizations are limited to the heavily optimized RefCOCO datasets, making it difficult to assess performance on unconstrained "in-the-wild" images. Including qualitative results on such images, mirroring the style of Figure 4(a), is essential to demonstrate the model's real-world generalization

5. **Reproducibility:** Since code is not given, adding some crucial pseudo code in the paper will be tremendously helpful for the broader audience to reproduce this work.

6. **Many Writing & Formatting Issues**: 
    1. *References style:* inconsistent reference format; inconsistent capitalization (e.g., an author name in ALL CAPS on line 594); duplicate/inconsistent entries (e.g., CRIS 2022a/2022b). Standardize to ICLR format.
    2. *Typos and formatting errors:* e.g. casing of softmax/Softmax is sometimes small (equation 16) and sometimes large (equation 6); no spacing in Theorem1 and Theorem2; PPME in line 1106.
    3. *Unconventional figure naming:* Figure a, Figure b, Figure c etc.
    4. *Readability of text in figures:* Some texts in Figure 1, Figure a and Figure b are too small.

### Questions
Refer to weaknesses.

### Soundness
2

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
- This work proposes Alignment-Aware Masked Learning (AML), a training strategy that quantifies region–referent alignment (PMME) and filters out unreliable pixels during optimization (AFM), which is validated to improve RIS performance.

### Strengths
- This work proposes Alignment-Aware Masked Learning (AML), a training strategy that quantifies region–referent alignment (PMME) and filters out unreliable pixels during optimization (AFM), which is validated to improve RIS performance.

- The writing overall is good and it is easy for readers to understand the proposed framework.

- The experiments analysis is detailed for readers  to realize the benefits of AML.

### Weaknesses
- Motivation clarification: the motivation is not well clarified from figure-1.  I suppose the author's motivation is that a number of regions (especially background regions) dominate the training loss.

- Method contributions
  - Based on the above motivation, I am more inclined to believe that this work is actually an implementation of curriculum learning in the RIS task. It is also be validated from the efficiency of early-training stage. In view of the originality, it decreases the contributions of this work.
  - The work claims that the stage-I is forward-only. I am curious how the similarity of these raw visual-language features reflects the degree of alignment.
  - The whole performance improvement is weak especially on the more strong models, which makes the work seem less significant at present community.

- Some writings are confusing.
  - The explanation about $B^h$ and  $B^w$ (line-267).
  - For the `early-stage efficiency', to my knowledge, this is not a common term and it deserves a specific explanation.

- Extra experiments for validation
  - The author can verify the results of using different models at different stages (e.g., utilize ReLA for stage-1, CARIS for stage-2), which may bring some new observations.

### Questions
Refer to the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents an alignment-aware masked learning method for referring image segmentation. In particular, each sample is masked out by discarding pixels below an adaptive similarity threshold. The similarity map between visual and textual features is quantified by region-referent alignment. The framework is then trained after the first masking step. The above two steps are conducted in an interleaved fashion. In addition, the region-referent alignment is implemented via a PatchMax Matching Evaluation strategy on randomly projected visual and textual features. Experimental results validated the effectiveness of the proposed method.

### Strengths
(1) The motivation is well presented of using the proposed alignment-aware masked learning approach for referring image segmentation.

(2) The explanations and illustrations are mostly clear and intuitive of the PatchMax Matching Evaluation, the alignment-aware filtering mask and the training strategy.

### Weaknesses
(1) The approach of using a previous-step inference for mask prediction and guide the current learning may face convergence issue. In fact, the initial state of mask is largely incorrect and can result in unexpected learning curves. There is no discussion on this issue.

(2) On the fairness of experimental comparison, since CARIS+AML uses 17.2% more training time than CARIS (according to Appendix G.2), the performance gain in Table 1 is also possibly coming from longer training. There is no ablation study and discussion on this issue.

### Questions
No.

### Soundness
4

### Presentation
4

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
This work proposes a novel framework called AMLRIS, which aims to improve the performance of Referring Image Segmentation (RIS). The framework first introduces a Johnson-Lindenstrauss random projection to measure the similarity between image representations and token features. Then, the image pixels with low similarity are filtered out from the training process to stabilize the training and improve performance. Experimental results demonstrate that AMLRIS achieves superior performance compared to standalone training, and the ablation experiments show the effectiveness of the proposed modules.

### Strengths
• The projection design provides a novel approach for measuring similarity between representations of different modalities. This method can be extended to more tasks.
• Experiments demonstrate the effectiveness of the proposed structure, achieving competitive results across multiple downstream datasets.
• The proposed structure does not significantly increase training overhead while maintaining inference time.
• The proposed idea is interesting and generally well-motivated, and the experimental evaluation is relatively thorough

### Weaknesses
My primary concern is the method’s sensitivity to small or low-contrast objects. The AML
framework relies on PMME to generate alignment-based masks by identifying high-confidence visual patches. This mechanism inherently depends on the relative distribution of features within the image. As a result, small objects or objects with low visual saliency may produce low peak alignment scores and be incorrectly masked out during training. Consequently, the model’s performance may degrade on images where the target occupies a very small region or is visually subtle.
• In Figure 2, the masked pixels appear to be almost exclusively background regions. It is unclear
whether masking such areas truly helps the model focus on the target objects. In my view, it would be more important to mask regions corresponding to potentially confusing objects rather than background. This raises some concerns regarding the effectiveness of PMME in guiding the model’s attention.
• The mechanisms underlying some of the core components of the model remain unclear. I would be willing to consider a higher score if the authors provide clear explanations addressing my concerns.

### Questions
See Weakness above.

### Soundness
3

### Presentation
3

### Contribution
3
