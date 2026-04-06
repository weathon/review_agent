=== CALIBRATION EXAMPLE 34 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title**: Clearly reflects the contribution: a weak-to-strong (W2S) paradigm for generalized VQA.
- **Abstract**: Succinctly states the problem, approach, and key results. However, it omits a critical detail: the “strong” student is a large multimodal model (LLaVA-OneVision-Chat-7B) with ~8B parameters, while the “weak” teachers include much smaller models (e.g., 86M-121M parameters). This architectural and capacity disparity could be a major confounder for the claimed W2S effect. The abstract should note this to set accurate expectations.

### Introduction
- **Motivation**: Well-articulated: annotation costs and poor OOD generalization are real challenges in VQA.
- **Contributions**: Clearly listed. However, the introduction frames the W2S effect as a paradigm shift without sufficiently acknowledging that the student’s superior performance might stem from its vastly larger capacity and pre-training rather than the distillation process itself. The comparison between the student and teacher architectures is deferred to Table 4 (Appendix), but this critical point should be discussed upfront.

### Related Work
- Adequately covers supervised, self-supervised, ranking-based VQA, and W2S generalization from NLP. Could benefit from connecting more explicitly to knowledge distillation and ensemble methods in vision. The discussion of W2S in vision is limited to a single citation (Guo et al., 2024); more grounding in related vision literature would strengthen the context.

### Method (Sections 3 & 4)
- **Problem Setup (3.1)**: Standard.
- **W2S Implementation (3.2)**:
  - **Strong Model Choice**: Using an LMM as the student introduces a massive capacity gap. While this may be a valid design choice, it confounds the interpretation of the “weak-to-strong” effect. An ablation study with students of varying capacities (e.g., same architecture as teacher but larger, or same size but different pre-training) is necessary to isolate the contribution of the W2S paradigm.
  - **Training Dataset**: The 200k videos are selected by matching low-level metrics to LSVQ. This strategy may bias the dataset toward the LSVQ distribution, potentially limiting OOD diversity. The authors should discuss how this selection affects generalization claims.
  - **Weak Teachers**: All five teachers are trained on LSVQ, making them homogeneous in training source. This limits the diversity of the pseudo-labels.
- **Improving W2S (4)**:
  - **Ranking Formulation**: A sensible approach to unify heterogeneous signals.
  - **Ensembling & Heterogeneous Teachers**: Good ideas to improve supervision quality and diversity. However, synthetic distortions may not fully capture real-world degradation patterns.
  - **Iterative Training with Difficulty Sampling**: Innovative use of gMAD for hard sample mining. However, the gains from iteration could be due to increased training data (700k pairs total) rather than the iterative refinement itself. A control experiment with the same total number of pairs but without iterative difficulty sampling would clarify this.
  - **Confidence Loss**: Helps mitigate noise; well-motivated.
- **Overall Methodological Concern**: The core claim—that W2S learning enables a student to surpass teachers without human labels—is not sufficiently disentangled from the effects of model scale and pre-training. The paper needs to demonstrate that the paradigm itself, not merely the use of a much larger foundation model, is responsible for the gains.

### Experiments & Results
- **Benchmarks & Metrics**: Comprehensive evaluation across 10 datasets, standard metrics.
- **Section 3.3 Results**: Figure 4 shows aggregated results; per-dataset numbers are in Appendix Table 5. The main text should highlight key per-dataset findings, especially that the student sometimes surpasses the teacher even on in-domain data.
- **Section 4.4 Results (Table 1)**: Shows progressive improvements from components (I) to (V). The final model achieves SOTA on many OOD benchmarks, which is impressive.
- **Comparison with SOTAs**: The model outperforms recent LMM-based methods (VQA², VQAThinker) that use human labels. This is a strong result, underscoring the potential of W2S to reduce annotation dependence.
- **Critical Missing Comparisons**:
  1. **Supervised Upper Bound**: Appendix Table 5 includes a “supervised student” (same LMM trained on LSVQ human labels). This shows that W2S can approach supervised performance, but LSVQ has only 27k labeled videos, while W2S uses 200k unlabeled videos. A more informative comparison would be to train the same LMM on a large human-labeled dataset (if available) to establish the performance gap. Alternatively, the authors should discuss how much of the gain comes from scale of data vs. the W2S method.
  2. **Ablation on Student Capacity**: No experiments vary the student architecture to separate capacity effects from W2S distillation effects.
- **Statistical Significance**: No significance tests or confidence intervals are reported. Given the variability in VQA benchmarks, such analysis is essential to confirm that improvements are not due to chance.
- **OOD Gains**: While substantial, the authors should analyze whether these gains are consistent across all OOD datasets or driven by specific ones (e.g., LIVE-YT-HFR shows huge improvements). Understanding which distortion types benefit most would provide insight.

### Discussion
- Forward-looking and proposes a pathway toward VQA foundation models. However, it lacks a critical discussion of **limitations**:
  - The reliance on existing weak teachers: if they have systematic biases, the student may inherit them.
  - Computational and environmental costs of training large LMMs.
  - The applicability of synthetic distortions: they may not cover complex real-world artifacts (e.g., encoding artifacts from specific codecs, camera sensor noise).
  - The dataset selection strategy (matching to LSVQ) may limit diversity; the authors should discuss potential biases.

### Conclusion
- Appropriately summarizes contributions. Could briefly mention limitations.

### Reproducibility & Ethics
- **Reproducibility**: Code is promised via an anonymous link. The appendix provides extensive details on data processing, model architecture, and training. However, the 200k-video dataset is not publicly available (collected from social media). To ensure reproducibility, the authors should commit to releasing video IDs or features, or at least a detailed sampling script.
- **Ethics**: States that videos are filtered for permissive licenses. Using user-generated content raises privacy concerns even if publicly available; a more detailed ethics statement on data anonymization and compliance with platform terms would be beneficial.

## Overall Assessment

This paper introduces a novel weak-to-strong learning paradigm for VQA, achieving state-of-the-art results on multiple benchmarks without human annotations. The idea is timely and well-motivated, and the extensive experiments demonstrate strong performance, especially on out-of-distribution data. However, the contribution is currently undermined by several critical issues:

1. **Confounding effect of model capacity**: The “strong” student is an 8B-parameter LMM, while teachers are orders of magnitude smaller. The reported gains may be attributable to the student’s inherent capacity and pre-training rather than the W2S distillation process itself. Controlled experiments varying student capacity are necessary.
2. **Missing supervised upper-bound comparison**: While the paper shows W2S outperforms teachers, the comparison to a fully supervised equivalent (same LMM trained on large-scale human labels) is incomplete. This makes it difficult to assess the true gap and the efficiency of the W2S paradigm.
3. **Lack of statistical significance**: No statistical tests are provided to support the claimed improvements.
4. **Insufficient discussion of limitations**: Key limitations regarding bias inheritance, computational cost, and synthetic distortion coverage are not addressed.

If the authors can adequately address these concerns—particularly by disentangling the W2S effect from model capacity, providing a fair comparison to supervised baselines, and adding statistical analysis—the paper would be a strong candidate for acceptance. In its current form, however, these issues prevent a clear endorsement. **Major revision is required.**

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a novel weak-to-strong (W2S) learning paradigm for no-reference Video Quality Assessment (VQA) to address poor generalization and reliance on costly human annotations. The core idea is to train a strong student model (an LMM) using pseudo-labels from diverse, weaker "teacher" models (existing VQA models and synthetic distortion simulators) via a ranking-based formulation, followed by iterative self-teaching cycles focused on challenging samples. The method achieves state-of-the-art results on both in-domain and, especially, out-of-distribution (OOD) benchmarks without using human-labeled training data.

### Strengths
1.  **Clear Problem & Novel Solution:** The paper clearly identifies the major bottlenecks in VQA: poor OOD generalization and reliance on expensive human labels. The proposed W2S paradigm, particularly its application to the perceptual task of VQA, is a novel and promising direction. The empirical demonstration of the "weak-to-strong effect" in VQA (Sec. 3.3) is a solid foundational contribution.
2.  **Comprehensive & Well-Designed Framework:** The method integrates several thoughtful components: ensemble of homogeneous teachers for robust labels, integration of heterogeneous (synthetic) teachers for distortion diversity, a ranking formulation to unify signals, an iterative training strategy with difficulty-guided sampling (gMAD), and a confidence loss to mitigate noise. The ablation study (Table 1) convincingly shows the incremental value of each component.
3.  **Strong Empirical Results:** The experimental evaluation is extensive, covering 10 benchmarks (5 in-domain, 5 OOD). The final model (V) achieves SOTA or highly competitive performance across the board, with particularly impressive gains on challenging OOD datasets like LIVE-YT-HFR (+0.382 SRCC over the best teacher). The results robustly support the paper's claims.

### Weaknesses
1.  **Limited Analysis of Failure Modes & Limitations:** While OOD performance improves significantly, absolute performance on some OOD sets (e.g., SRCC of 0.698 on Waterloo-IVC-4K) indicates room for improvement. The paper does not deeply analyze cases where the method still fails or the limitations of the synthetic distortion simulators in modeling complex, real-world degradation chains.
2.  **Computational Cost & Efficiency are Under-discussed:** The strong student is a 7B-parameter LLaVA-based model trained iteratively on 200k-700k video pairs. The computational cost (time, energy, GPU memory) is substantial and is only briefly mentioned (2 days on 8xA800). For a method aiming at scalability, a discussion of efficiency trade-offs versus smaller, supervised models is warranted but missing.
3.  **Clarity Gaps in the Iterative Process:** The mechanism for selecting "difficult samples" for the next iteration (Eq. 1, gMAD) is clearly described for the ensemble teacher pairs. However, the process for the synthetic distortion pairs is stated as selecting "only those misclassified by the student," which presumes a ground-truth severity order. The interplay and balance between these two sample streams during iterative training could be explained more clearly.

### Novelty & Significance
**Novelty:** The work is highly novel in its context. While W2S generalization has been explored in NLP and vision classification, its application to the subjective, regression/ranking-based task of VQA is new. The specific innovations—integrating heterogeneous teachers via ranking, and the iterative difficulty-guided W2S training—constitute a significant advance over simple knowledge distillation or existing self-supervised VQA methods.

**Significance:** The significance is potentially high. If successful, this paradigm offers a path to break the annotation bottleneck and build more generalizable VQA models. The strong OOD results are directly impactful for real-world applications where video content and distortions are unpredictable. The work aligns well with ICLR's interest in innovative learning paradigms and foundation models.

### Suggestions for Improvement
1.  **Conduct a deeper failure analysis.** Include a qualitative analysis or case studies showing videos where the model's predictions diverge significantly from human scores (especially on OOD data). This would strengthen the discussion and provide clearer directions for future work.
2.  **Add a dedicated discussion on computational efficiency.** Compare the training cost (FLOPs, time) of the proposed method with standard supervised training of the teacher models and other SOTA methods. Acknowledging this trade-off and perhaps suggesting pathways to more efficient student architectures (e.g., distillation into a smaller model) would improve the paper's balance.
3.  **Improve the exposition of the iterative training pipeline.** A clearer schematic or flowchart detailing the data flow between stages (what pairs are carried forward, how the pools are updated) would be helpful. Additionally, explicitly stating how many video *pairs* (not just videos) are used in each stage would enhance reproducibility.
4.  **Strengthen the supervised baseline comparison.** The paper shows the final W2S model outperforms models trained on LSVQ. To rule out the simple benefit of more data, consider an additional baseline: train the same strong student architecture on LSVQ *plus* the 200K unlabeled videos, using data augmentation or a semi-supervised technique. This would better isolate the benefit of the W2S paradigm from the benefit of larger, more diverse unlabeled data.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on the source of improvement: data scale vs. W2S paradigm.** The "strong" student is trained on 200k videos, while teachers are trained on 27k (LSVQ). A control experiment where the student is trained via direct regression on the same 200k videos, but using human labels from a single teacher (or ensemble), is missing. Without this, gains could be attributed to more data, not the novel W2S ranking/iteration framework.
2. **Comparison to state-of-the-art unsupervised/weakly-supervised VQA methods.** The paper only compares to fully supervised models. To substantiate the claim of "advancing VQA without reliance on large-scale human-labeled datasets," it must be benchmarked against recent self-supervised (e.g., ConvIQT, QPT-v2) or zero-shot (e.g., using VLMs) methods on the same OOD datasets.
3. **Analysis of iterative training with alternative difficulty sampling strategies.** The iterative W2S uses gMAD for sample selection. An ablation comparing gMAD to random sampling, or confidence-based sampling, is needed to prove that focusing on "challenging cases" is the key driver of improvement, not simply more training iterations on more data.
4. **Cross-dataset generalization test.** The current in-domain/OOD split is based on content type, but a more rigorous test is to train on pseudo-labels from one dataset (e.g., KoNViD) and test on completely disjoint datasets (e.g., LIVE-VQC, YouTube-UGC). This would better validate the "generalized" VQA claim.

### Deeper Analysis Needed (top 3-5 only)
1. **Error analysis and failure modes.** The paper lacks analysis of where the model still fails, especially on OOD data. A breakdown of performance by distortion type (e.g., compression vs. noise vs. temporal) on datasets like LIVE-YT-HFR or Waterloo-IVC-4K is needed to understand what specific distortions the method does and does not generalize to.
2. **Analysis of teacher bias and student amplification.** All teacher models are trained on LSVQ (UGC). The student's supervision is thus biased towards LSVQ's quality concepts. An analysis is needed to see if the student's OOD gains are uniform or if it underperforms on distortion types underrepresented in LSVQ, potentially amplifying teacher biases rather than overcoming them.
3. **Quantifying the "quality" of pseudo-labels.** The confidence loss is used to handle noise, but there is no analysis of the noise level in teacher ensembles vs. single teachers, or how the noise correlates with video content/distortion. Measuring pseudo-label accuracy (where possible, on a small human-labeled set) would solidify the claims about ensemble reliability.

### Visualizations & Case Studies
1. **Qualitative examples of "difficult samples" and model progression.** Visualize video pairs selected by gMAD across iterations, showing frames and the disagreement between teacher and student predictions. This would concretely demonstrate the "challenging cases" the method progressively learns.
2. **Case studies on specific distortion types.** For key OOD datasets (e.g., LIVE-YT-HFR for frame rate, Waterloo-IVC-4K for compression), show example videos where the student model's score aligns better/worse with human perception compared to the teacher ensemble scores. This reveals whether the method truly learns new quality dimensions.

### Obvious Next Steps
1. **Semi-supervised baseline.** The most obvious missing step is a "lower-bound" experiment: fine-tuning the strong student on a *small* set of human labels from LSVQ. This would establish the performance gap between the proposed label-free method and a data-efficient supervised approach, contextualizing the contribution.
2. **Varying teacher strength.** The paper uses SOTA models as "weak" teachers. To truly test the W2S paradigm, experiments with deliberately weakened teachers (e.g., older VQA models, or models trained on less data) should show a stronger student emerging, which is more aligned with the original W2S narrative.
3. **Sensitivity analysis on synthetic distortion parameters.** The synthetic teachers use fixed distortion levels. An analysis of how the choice and range of these synthetic distortions impact final OOD performance is needed; the current setup may inadvertently favor the tested OOD benchmarks.

# Final Consolidated Review
## Summary
This paper introduces a weak-to-strong (W2S) learning paradigm for no-reference video quality assessment (VQA), aiming to improve generalization and reduce reliance on costly human annotations. The method trains a strong, large multimodal student model using pseudo-labels from an ensemble of weaker, existing VQA teachers and synthetic distortion simulators, unified via a ranking formulation and refined through iterative, difficulty-aware training. It achieves state-of-the-art performance, with especially strong gains on out-of-distribution benchmarks, demonstrating a clear W2S effect in VQA.

## Strengths
- **Novel and well-motivated paradigm:** The paper clearly identifies the annotation bottleneck and poor OOD generalization in VQA and provides the first empirical demonstration and framework for weak-to-strong generalization in this perceptual task, offering a viable path toward scalable, label-free VQA.
- **Comprehensive and effective framework:** The method thoughtfully integrates multiple components—ensemble of homogeneous teachers, integration of heterogeneous synthetic teachers via ranking, iterative training with difficulty-guided sampling (gMAD), and a confidence loss—which together drive significant performance gains, as shown in a clear ablation study (Table 1).
- **Strong empirical results, particularly on OOD data:** The final model achieves state-of-the-art results on a comprehensive set of 10 benchmarks. The most impressive gains are on challenging out-of-distribution datasets (e.g., LIVE-YT-HFR SRCC improves from 0.301 to 0.683 over the best teacher), substantiating the core claim of improved generalization without human labels.

## Weaknesses
- **Incomplete disentanglement of gains from data scale versus the W2S mechanism:** The student model is trained on 200k unlabeled videos, while the teachers were trained on only 27k labeled videos (LSVQ). Although Table 5 shows the W2S student can outperform the same architecture trained on LSVQ labels, a more controlled comparison—such as training the student on the same 200k videos using a simple regression baseline with teacher pseudo-labels—would more cleanly isolate the contribution of the novel ranking and iterative components from the benefit of simply having more data.
- **Under-discussed computational cost and limitations:** The strong student is a 7B-parameter LMM trained iteratively on up to 700k video pairs, requiring substantial resources (2 days on 8 A800 GPUs). While the paradigm aims for scalability, a discussion of this efficiency trade-off compared to smaller supervised models, and the inherent limitations of synthetic distortions in modeling complex real-world artifacts, would provide a more balanced view of the method's practicality.

## Nice-to-Haves
- A semi-supervised baseline, where the strong student is fine-tuned on a small set of human labels from LSVQ, would help contextualize the performance gap between the proposed label-free method and a data-efficient supervised approach.
- A deeper qualitative analysis of failure cases or challenging samples selected by the gMAD strategy would provide clearer insight into what the method learns and where it still struggles.

## Novel Insights
The paper provides a novel and important insight: the weak-to-strong generalization effect, previously studied in NLP and vision classification, can be successfully harnessed for the subjective, perceptual task of video quality assessment. By demonstrating that a strong model can learn from weaker, noisily-labeled teachers and not only match but significantly surpass them—especially on out-of-distribution data—the work establishes W2S as a principled and effective paradigm to circumvent the annotation bottleneck in VQA. The iterative, difficulty-aware training further shows that the student can progressively evolve beyond the collective knowledge of its teachers.

## Suggestions
- To strengthen the core claim, add an ablation experiment where the student is trained on the same 200k videos using a direct regression loss on the ensemble teacher scores (without the ranking formulation or iterative difficulty sampling). This would help isolate the contribution of the proposed framework's novel components.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0]
Average score: 3.3
Binary outcome: Reject
