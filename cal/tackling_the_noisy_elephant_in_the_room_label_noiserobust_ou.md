=== CALIBRATION EXAMPLE 57 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title captures the core theme well: noisy labels and OOD detection, with loss correction and low-rank decomposition as the proposed ingredients. However, the “Tackling the Noisy Elephant in the Room” framing is more slogan-like than informative; the substantive contribution is really a robust OOD detection framework under label noise.
- The abstract clearly states the problem, the high-level method, and the claimed outcome. It does identify that simply combining label-noise-robust learning with OOD detectors is insufficient, which is a useful motivation.
- The main concern is that the abstract makes strong claims of being “a principled solution” and “significantly outperform[ing]” state-of-the-art methods, but the paper does not yet provide enough theoretical justification to support “principled” in a strong sense. The mechanism seems heuristic: a combination of loss correction, low-rank projection, and a quantile-based sample selection rule.

### Introduction & Motivation
- The problem is well-motivated and timely for ICLR: OOD detection is important, and the interaction with noisy labels is indeed underexplored. The paper also appropriately cites recent work showing noise hurts OOD detectors.
- The gap in prior work is identified more clearly than in many submissions: existing noisy-label methods optimize classification robustness, not ID/OOD separability. That is a genuine and relevant distinction.
- The contribution statements are mostly accurate at a high level, but “the first to offer a principled solution” is too strong given the method’s reliance on design choices that are not theoretically derived from a formal objective linking label-noise correction to OOD detection.
- The introduction slightly over-claims novelty by presenting the framework as broadly “robust” and “principled,” while the actual detector is still a distance-based heuristic built on cleaned features.

### Method / Approach
- The method is only partially clearly described. The overall pipeline is understandable: apply a noisy-label loss correction method, project features into a low-rank ID component via a PI-based approximation, define a residual as OOD-like content, score samples by their residual norm, select a subset, and use distance-based OOD scoring.
- There are important reproducibility and correctness issues in the description:
  - In Section 3, the low-rank/sparse decomposition is introduced as if it solves a matrix optimization problem resembling robust PCA, but the implementation actually uses a batch-wise power-iteration approximation with no explicit optimization over the stated objective. The paper never clarifies how the sparse term in Eq. (4) is operationalized within training beyond “regularization,” nor how the low-rank projection and sparse penalty jointly influence gradients.
  - The role of the target rank is ambiguous. The paper says the rank can be selected as the number of classes, but this is not justified. For CIFAR-100 and other large-label settings, this choice may be questionable, and the relationship between class count and feature rank is not established.
  - The “OOD-ness” score \(o(x_n)=\|h(x_n)-h_{ID}(x_n)\|_2\) is intuitively plausible, but the paper does not justify why this quantity should correlate with mislabeled or hard examples in a way that improves OOD detection rather than just filtering difficult samples.
  - The selection rule in Eq. (5) retains the top \((1-p)\%\) highest OOD-ness samples, but the rationale is somewhat counterintuitive: if those samples are more non-ID-like, why are they the right reference set for OOD detector construction? The paper suggests they are “challenging examples,” but the connection to better ID reference embeddings is not rigorously argued.
  - The paper defines \(H_{ID}=QQ^\top H\) and \(H_{OOD}=H-H_{ID}\), but this is only a projection residual, not a true decomposition with identifiability guarantees. The interpretation of \(H_{OOD}\) as “OOD” is therefore semantic and may be misleading.
- There are also edge cases/failure modes not discussed:
  - If the batch feature matrix is small or class-imbalanced, the low-rank approximation may be unstable.
  - For fine-grained datasets or open-set scenarios where ID classes are not naturally low-rank, the projection assumption may fail.
  - The method seems to depend on a validation threshold for 95% ID recall, which may itself be nontrivial under noisy supervision.
- For the theoretical claims, there are no real proofs. The paper uses signal-processing terminology, but does not establish conditions under which the batch-wise PI decomposition approximates robust PCA or why the residual norm should correspond to label noise or OOD-ness.

### Experiments & Results
- The experiments do address the paper’s main claim: they evaluate OOD detection under both synthetic noise and real noisy labels, using standard OOD datasets and metrics (FPR95, AUROC). This is appropriate for the problem.
- Baselines are broadly appropriate, including both OOD detectors and noisy-label learning methods. It is good that the paper compares against methods from both literatures.
- A major concern is fairness and comparability across methods:
  - The paper uses DenseNet-101 as the shared encoder, but some baselines are originally designed and tuned for different backbones or training regimes. The supplementary notes that CIDER and SSD+ use a replaced encoder, which is fine, but the paper does not show whether these baselines were retuned to comparable strength.
  - For the noisy-label baselines, it is unclear whether they are being used in the way those methods are intended for classification or adapted specifically for OOD detection. The paper says OOD performance is evaluated with a kNN metric for all these methods unless specified, but this still leaves open whether the comparison captures the best version of each baseline.
- The ablation studies are useful but incomplete:
  - The paper ablates loss correction choice, metric choice, \(\lambda\), \(K\), and \(p\). This helps.
  - However, the key conceptual claim of the method is the interaction between loss correction and low-rank decomposition. There is no ablation showing the effect of removing the low-rank/sparse component while keeping the same loss correction, nor comparing to a simpler projection baseline.
  - There is also no direct ablation of the “OOD-ness” selection heuristic versus using all samples, beyond the figure showing \(p=0\) vs \(p=0.5\). This is suggestive but not sufficient to establish that the selection rule itself is necessary.
- Error bars or statistical significance are not reported. Given the variability that can arise from noise injection and neural training, this is a meaningful omission.
- The results are generally strong and, on their face, support the claim that NOODLE often improves FPR95 under noise. But some of the gains are modest, and the paper often emphasizes average FPR95 without discussing trade-offs with ID accuracy in depth. Since the method depends on feature cleansing, a clearer account of whether ID classification suffers would matter.
- The reported datasets and metrics are standard and appropriate. That said, the paper’s strongest real-world evidence is on CIFAR-10N and Animal-10N, which are useful but still limited; more diverse domains would strengthen the ICLR-level claim.

### Writing & Clarity
- The main conceptual storyline is understandable, but several parts of the method section are hard to follow precisely. In particular:
  - The relationship among the low-rank projection, sparse residual, and final OOD detector is not cleanly disentangled.
  - The transition from Eq. (4) to the PI-based approximation is somewhat abrupt.
  - The interpretation of the selected subset \(S\) in Eq. (5) is not fully explained.
- Figures 1–5 are useful conceptually, especially Fig. 1 and Fig. 3, because they support the intuition that noise distorts representation geometry and that the proposed score correlates with sample difficulty. However, the paper leans heavily on visual intuition where a more formal or quantitative analysis would be helpful.
- Tables are informative and extensive. The main readability issue is not formatting but that the key takeaways are scattered across many large tables, making it hard to isolate which claims are central versus supplementary.

### Limitations & Broader Impact
- The paper does not meaningfully acknowledge key limitations. It briefly notes that estimating transition matrices becomes difficult as the number of classes grows, but that is more of an empirical observation than a limitation analysis.
- Important limitations that are missing:
  - Dependence on the assumption that ID features are approximately low-rank and that OOD-like content is sparse.
  - Sensitivity to the choice of rank \(K\), subset fraction \(p\), and regularization \(\lambda\).
  - Dependence on validation data for threshold selection at 95% recall.
  - Potential instability when label noise is instance-dependent rather than symmetric/crowdsourced noise.
- Broader impact is not discussed. While the work is methodologically oriented and likely positive in intent, deploying OOD detectors in safety-critical systems under noisy labels can have real consequences if the low-rank assumptions fail. The paper does not discuss possible failure modes or the risks of false reassurance from improved average metrics.

### Overall Assessment
This is a timely and relevant paper for ICLR, and the empirical problem it targets is real: OOD detectors can degrade substantially under noisy labels, and existing noisy-label methods are not obviously designed to fix that. The proposed NOODLE framework is interesting, especially the combination of loss correction with a low-rank feature projection and a sample-selection-based distance detector. That said, the paper currently relies more on plausible heuristics and empirical gains than on a clearly justified or theoretically grounded method. The main concerns are the lack of formal support for the low-rank/residual interpretation, incomplete ablations isolating the core mechanism, and missing statistical rigor. I think the contribution is promising, but at ICLR’s standard it would need stronger justification that the proposed decomposition and score are not just useful heuristics, but a well-motivated and robust solution to the stated problem.

# Neutral Reviewer
## Balanced Review

### Summary
This paper studies out-of-distribution (OOD) detection when the training set has noisy labels, a setting that is increasingly relevant but underexplored. The authors argue that simply plugging label-noise-robust classification methods into existing OOD detectors is insufficient, and propose NOODLE, which combines loss correction with a low-rank/sparse decomposition of latent features plus a distance-based OOD score built from a learned “OOD-ness” measure over training samples.

### Strengths
1. **Targets a genuinely relevant and underexplored problem.**  
   The paper directly addresses OOD detection under noisy labels, which is well motivated by practical data collection issues. This is aligned with ICLR’s interest in robustness and reliable ML, and the paper cites a recent prior work showing that label noise can hurt OOD detection (Humblot-Renaux et al., 2024), indicating awareness of the emerging problem setting.

2. **The proposed direction is conceptually plausible and potentially useful.**  
   The central idea—cleaning latent features rather than only correcting class probabilities—is well matched to distance-based OOD detectors such as kNN and Mahalanobis, which depend on feature geometry. The low-rank plus sparse decomposition is a reasonable inductive bias for encouraging class-structured embeddings, and the paper gives an intuitive explanation for why this may help OOD separability.

3. **Empirical evaluation is broad.**  
   The paper reports experiments on synthetic noise, multiple real noisy-label datasets (CIFAR-10N, CIFAR-100N, Animal-10N), and multiple OOD benchmarks (SVHN, FashionMNIST, LSUN, iSUN, Texture, Places365). It also compares against both OOD detectors and label-noise-robust learning methods, which is appropriate for this problem.

4. **The results appear consistently favorable.**  
   Across the tables, NOODLE often achieves the best average FPR95 and AUROC, especially at higher noise rates and on real noisy datasets. The paper also provides ablations over the low-rank rank parameter and selection fraction p, suggesting some effort to test sensitivity and isolate component contributions.

5. **The paper includes implementation and algorithmic details.**  
   The supplementary material provides algorithm pseudocode for the low-rank approximation routine and the overall training loop. This is better than many papers in terms of giving at least a starting point for reproduction.

### Weaknesses
1. **The methodological novelty is somewhat incremental.**  
   The method combines existing ingredients from disparate areas: loss correction from noisy-label learning, low-rank/sparse decomposition from robust PCA, and standard distance-based OOD scoring. While the combination is interesting, the paper does not clearly establish a deep new principle or theoretical insight beyond “feature cleaning helps OOD.” For ICLR, where novelty and conceptual contribution matter, the work may feel more like a careful hybridization than a fundamentally new learning framework.

2. **The core technical formulation is not fully convincing or clearly justified.**  
   The paper describes decomposing the latent feature matrix as low-rank plus sparse, but the actual training implementation appears to rely on a batch-wise power-iteration approximation rather than solving the stated optimization problem. The connection between the formal decomposition objective and the implemented procedure is not fully developed, and the role of the sparse term is somewhat opaque. It is not clear why the OOD-ness score computed from the residual should be a principled estimate of “difficulty” or non-ID content.

3. **Reproducibility is only moderate.**  
   Although the paper gives some settings, many details remain underspecified or potentially fragile: exact training schedules, optimizer settings per dataset, how the transition-matrix methods are estimated in practice, how validation sets are constructed under noisy labels, and how hyperparameters are chosen across datasets. The supplement is helpful, but the description is not yet at the level expected for easy independent reproduction.

4. **The evaluation protocol leaves some open questions.**  
   The paper uses DenseNet-101 and DenseNet-100 inconsistently in different places, which suggests possible reporting or implementation ambiguity. More importantly, the reported gains are often on average FPR95, but it is unclear how statistically stable these numbers are; there are no confidence intervals, repeated runs, or significance tests. Given that OOD results can vary substantially with random seeds and training noise, this weakens confidence in the magnitude of the claimed improvements.

5. **Baselines are not fully normalized in a way that removes concern about fairness.**  
   The paper does try to compare against many methods, but some baselines are OOD detectors with standard training, while others are noisy-label methods adapted to OOD by plugging in kNN. It is not fully clear that all methods receive equally strong tuning. The supplement notes that some methods use different encoders from prior work and were replaced for consistency, but this also suggests that baseline performance may not reflect each method’s best configuration.

6. **The paper’s claims may be somewhat overstated relative to the evidence.**  
   The abstract and conclusion suggest a “principled solution” and “substantial improvements,” but the work is mostly empirical and does not provide theoretical guarantees. The generality of the approach is also limited to distance-based detectors and vision benchmarks; it is not shown on text, multimodal, or non-classification OOD settings.

7. **Clarity is uneven.**  
   The narrative motivation is clear, but some parts of the technical exposition are hard to follow. In particular, the decomposition equations, the definition of the OOD-ness score, and the exact way the low-rank component interacts with the classifier and detector would benefit from a more precise, unified derivation.

### Novelty & Significance
For ICLR, the paper addresses an important and timely robustness problem, and the empirical setting is relevant. However, the novelty is moderate rather than high: the method is a combination of known techniques applied to a new setting, with a plausible but not deeply novel mechanism. The significance is also moderate: if the gains hold robustly across seeds and broader modalities, this could be practically useful for reliable OOD systems under annotation noise, but in its current form the paper does not yet demonstrate the kind of broad, conceptually new advance that is typically needed for a strong ICLR acceptance.

Overall assessment: promising and relevant, with solid empirical evidence, but likely below the ICLR bar for a clearly strong acceptance unless the authors can better justify the method, strengthen reproducibility, and sharpen the novelty.

### Suggestions for Improvement
1. **Tighten the method description and connect the optimization to the implementation.**  
   Provide a clearer derivation from the low-rank-plus-sparse objective to the batch-wise power-iteration algorithm, and explicitly explain what is optimized end-to-end, what gradients flow where, and what approximation error is introduced.

2. **Add stronger evidence of robustness and statistical reliability.**  
   Report mean and standard deviation over multiple random seeds, and ideally confidence intervals or significance tests, especially for the main claims on FPR95 improvements.

3. **Clarify hyperparameter selection and validation under noisy labels.**  
   Explain exactly how p, λ, and detector thresholds are tuned without using test OOD information, and whether the same settings transfer across datasets or are dataset-specific.

4. **Strengthen the baseline comparison.**  
   Ensure every baseline is given a competitive tuning budget and the same backbone/training protocol where applicable. If some methods are adapted from original papers, include a careful justification that these adaptations are fair.

5. **Improve the conceptual justification of the OOD-ness score.**  
   Provide an analysis showing that the residual norm correlates with true label noise, ambiguous samples, or OOD-like feature behavior, rather than only showing top/bottom examples. A quantitative correlation study would help a lot.

6. **Include ablations on the decomposition mechanism itself.**  
   For example, compare low-rank only, sparse only, random projection, PCA-style projection, and alternative feature-cleaning strategies. This would clarify whether the benefit comes from the specific decomposition or just from generic regularization.

7. **Broaden the evaluation beyond vision and kNN/Mahalanobis.**  
   Since the paper makes a fairly general robustness claim, it would be helpful to test on at least one different modality or one post-hoc detector not heavily tied to representation geometry, to better establish generality.



# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add comparisons against stronger and more recent OOD detectors under noisy labels, not just the standard post-hoc and a small set of noisy-label methods. ICLR reviewers will expect evidence against current best-in-class OOD baselines and against directly noise-robust OOD-specific baselines if any exist, because the core claim is “state-of-the-art under noisy labels.”

2. Add evaluations beyond CIFAR-style image classification, especially a non-vision setting or at least a different architecture family. The paper claims a general robust OOD framework, but all reported evidence is on DenseNet-based image benchmarks, so it is unclear whether the method transfers beyond one encoder and one problem regime.

3. Add a clean-label control and a “method-only” ablation showing whether NOODLE still helps when labels are clean versus noisy. Without this, it is hard to tell whether the method is genuinely noise-robust or just a stronger OOD regularizer that happens to work better in the reported setups.

4. Add a direct comparison to simpler alternatives: low-rank feature regularization alone, loss-correction alone, and post-hoc kNN/Mahalanobis on the same corrected features. The current results do not isolate whether the gain comes from the full framework or from one component, which weakens the claimed novelty.

5. Add a sensitivity study over noise type and noise severity beyond symmetric noise and a few real noisy sets, including instance-dependent or class-conditional noise. The main claim is robustness to label noise in practice, but the evidence is narrow and may not cover the hardest/noisiest regimes ICLR reviewers would ask about.

### Deeper Analysis Needed (top 3-5 only)
1. Provide an analysis of why the low-rank decomposition should preserve ID structure under noisy supervision, including failure cases. Right now the method assumes “OOD-like” residuals are sparse, but the paper does not justify when that assumption is valid or when it breaks, which is central to the method’s credibility.

2. Add quantitative analysis of the “OOD-ness” score as a proxy for label corruption or sample ambiguity. The paper uses this score for selection, but without correlation to actual noise labels or sample difficulty, the scoring rule looks heuristic rather than principled.

3. Analyze whether the selected top-(1-p)% samples are truly more informative or just harder samples that may hurt detector construction. This matters because the selection rule could inadvertently bias the detector toward borderline or minority-class points, and the paper currently does not show that this choice is safe.

4. Report training dynamics and optimization stability for the PI-based low-rank module. Since the method modifies the feature space during end-to-end training, reviewers will want to know whether the gains are robust across seeds and whether the decomposition causes instability or collapse.

5. Separate OOD-detection improvement from ID-classification improvement. The paper reports ID accuracy but does not analyze whether better OOD detection comes from better classification, better geometry, or both; without this, the mechanism behind the gains remains unclear.

### Visualizations & Case Studies
1. Show nearest-neighbor neighborhoods before and after the low-rank decomposition for correctly labeled, mislabeled, and OOD-like samples. This would directly reveal whether the method actually “cleanses” features or just compresses them.

2. Provide failure-case visualizations where NOODLE still assigns low OOD-ness to mislabeled or ambiguous samples, or where OOD samples are retained as ID-like. ICLR reviewers will want to see the boundary cases, not only the cherry-picked top-10/bottom-10 examples.

3. Plot score distributions for clean ID, noisy-ID, and OOD test samples, not just UMAPs. This would show whether the new score is truly discriminative and whether the selection threshold is meaningful.

4. Visualize how the low-rank rank estimate and sparse residual evolve over training. Without this, it is hard to assess whether the decomposition is learning a stable class subspace or merely overfitting to batch structure.

### Obvious Next Steps
1. Extend the framework to instance-dependent and class-conditional label noise, since symmetric noise is too limited for the paper’s central robustness claim. This is the most obvious next step for an ICLR-level contribution.

2. Test the method with stronger backbones and modern OOD embeddings, including ViT-style encoders. If the method is truly general, it should not depend on DenseNet-specific feature geometry.

3. Add a principled criterion for selecting rank K and selection fraction p without manual tuning. The current approach relies on dataset-specific hyperparameter sweeps, which weakens deployability and the claim of an end-to-end robust method.

4. Compare against a simple pipeline of noise-robust training followed by standard OOD score recalibration. This baseline is necessary to show that the proposed joint framework is better than composing existing parts.

# Final Consolidated Review
## Summary
This paper studies out-of-distribution detection when the training set contains noisy labels, a setting that is practically important but still underexplored. The authors propose NOODLE, which combines a noisy-label loss correction module with a low-rank/sparse feature decomposition and then uses a distance-based OOD score on the cleaned embeddings.

## Strengths
- The paper targets a genuinely relevant and underexplored problem: OOD detectors are usually evaluated on clean training labels, yet the paper correctly shows that label noise can distort feature geometry and hurt OOD performance.
- The empirical evaluation is broad within its chosen setting, covering synthetic noise, multiple real noisy-label datasets, and several standard OOD benchmarks; the results are consistently favorable for NOODLE, especially at higher noise rates.

## Weaknesses
- The method is a somewhat ad hoc combination of existing ideas rather than a clearly principled new framework. It mixes loss correction, low-rank projection, sparse residuals, and a quantile-based selection rule, but the paper does not provide a rigorous derivation showing why these pieces together should solve robust OOD detection.
- The core low-rank/sparse formulation is not tightly connected to the actual implementation. The paper states a robust-PCA-like objective, but the algorithm uses a batch-wise power-iteration approximation and leaves the sparse term’s role and gradient flow unclear; this makes the claimed “feature cleansing” mechanism more heuristic than fully justified.
- The justification for the new “OOD-ness” score and the sample-selection rule is weak. The paper shows a few illustrative examples, but does not quantitatively demonstrate that this score correlates with true label corruption, ambiguity, or genuinely useful reference samples for OOD detection.
- The evaluation, while extensive, is not fully convincing as an ICLR-level robustness study. There are no repeated-seed statistics or confidence intervals, baseline tuning fairness is not fully established, and the paper relies heavily on one backbone family and image classification benchmarks, limiting the strength of the generality claim.
- Important ablations are missing. In particular, the paper does not cleanly isolate the benefit of low-rank decomposition from loss correction alone, nor does it compare against simpler alternatives such as PCA-style projection or “corrected features + standard kNN/Mahalanobis.”

## Nice-to-Haves
- A cleaner derivation connecting the stated objective to the implemented PI-based training procedure would improve trust in the method.
- Reporting mean/std over multiple seeds would make the claimed improvements much more credible.
- A quantitative correlation analysis for the OOD-ness score would help justify the sample-selection heuristic.

## Novel Insights
The most interesting insight is that label noise harms OOD detection not just by degrading classification accuracy, but by directly warping the latent geometry that distance-based detectors rely on. The paper’s main conceptual move is therefore to “clean” the representation space itself rather than only correcting output probabilities. That is a sensible direction, but the current implementation still feels more like a carefully engineered hybrid than a deeply principled solution, and the paper does not fully establish that the low-rank residual really corresponds to the kind of non-ID content the method claims to isolate.

## Potentially Missed Related Work
- Humblot-Renaux et al. 2024 — directly studies whether OOD detectors are robust to label noise, and is highly relevant to positioning this work.
- Robust PCA / low-rank plus sparse decomposition literature — relevant because the method borrows this machinery, though the paper already cites classic references.
- None identified for a clearly stronger, directly competing noise-robust OOD detection method beyond what the paper already discusses.

## Suggestions
- Add a strong ablation that separates: loss correction only, low-rank projection only, full NOODLE, and corrected-features + standard OOD score.
- Report mean and standard deviation across multiple random seeds for the main tables.
- Provide a quantitative study showing how the proposed OOD-ness score relates to label corruption or sample ambiguity.
- Clarify exactly how the low-rank/sparse objective is optimized end-to-end, and what approximation error the PI procedure introduces.
- If possible, test one non-vision setting or a stronger encoder family to support the robustness/generalization claim.

# Actual Human Scores
Individual reviewer scores: [8.0, 2.0, 4.0, 6.0]
Average score: 5.0
Binary outcome: Reject
