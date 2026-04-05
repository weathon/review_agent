=== CALIBRATION EXAMPLE 37 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title is broadly accurate: the paper is about class-incremental semantic segmentation and proposes a knowledge-distillation-based method with parameter release and knowledge reuse. That said, “Parameter Release” and “Knowledge Reuse” are not standard terms in the field, so the title is somewhat more slogan-like than descriptive.
- The abstract clearly states the problem, the broad method idea, and the claimed empirical outcome. It does identify the key challenge: no access to prior training samples.
- A major issue is that the abstract makes strong claims that are not yet well-supported from the paper as presented, especially “average performance approaches that of joint learning” and “effectively reducing class confusion.” The paper reports strong results, but the evidence for “approaches joint learning” is uneven across settings and not always contextualized against the gap to the upper bound.
- The abstract also presents the method as a “minimization–maximization distribution strategy,” but the mechanism is not immediately intuitive and feels more like a conceptual framing than a crisp summary of the actual algorithm.

### Introduction & Motivation
- The motivation is relevant and well aligned with ICLR standards: continual learning under privacy/storage constraints is important, and class-incremental semantic segmentation is a meaningful setting.
- The gap in prior work is reasonably identified: most methods use distillation or pseudo-labeling, and the paper argues that this can crowd the parameter space and underutilize prior knowledge.
- However, the central problem statement is not yet fully convincing as stated. The “parameter competition” argument is plausible, but the introduction does not sufficiently separate it from ordinary forgetting/plasticity tradeoffs already known in continual learning. The paper needs to more clearly justify why this is a distinct bottleneck rather than a rephrasing of the standard stability-plasticity tradeoff.
- The contributions are stated clearly, but they are somewhat over-strong. In particular, the introduction claims that DKD “achieves near-upper-bound average performance” and “state-of-the-art average performance” without clearly delimiting the backbone, protocol, or step settings where this holds.
- The introduction does not sufficiently acknowledge that many of the comparison methods use different backbones or training assumptions, which weakens the claim of a clean advance unless those confounds are carefully controlled.

### Method / Approach
- The method has an ambitious scope, but the description is not fully crisp or reproducible in the main text.
- The overall framework in Fig. 3 is understandable:  
  1) prune/release low-sensitivity parameters of the previous model,  
  2) minimize old-knowledge distribution via \(L_{\text{Min}}\),  
  3) estimate reusable regions/features with Laplacian-based projection in \(L_{\text{Esti}}\),  
  4) maximize shared distribution with \(L_{\text{Max}}\).
- The main concern is that the paper mixes several distinct ideas—parameter pruning, pseudo-label adjustment, Laplacian-based feature projection, and entropy/information-maximization—without a sufficiently clean algorithmic specification. It is hard to reconstruct the full training pipeline precisely from Sections 3.2(a–c).
- The parameter-release mechanism is particularly under-specified. The paper says it computes layer-wise Euclidean norms and prunes units below a threshold \(\tau\), but it is not fully clear:
  - whether pruning is applied to the frozen teacher, the trainable student, or both;
  - how often pruning is recomputed;
  - whether pruned weights are permanently zeroed or just masked in distillation;
  - whether the same mask is used across steps or recalculated each step.
  These details matter substantially for reproducibility and for understanding the claimed “parameter release.”
- The logic of \(L_{\text{Min}}\) is plausible, but the derivation in the appendix is not fully satisfying as a theoretical guarantee. The argument appears to show gradient directions for logits, but the conclusion that this “releases parameters” is still indirect. It is more accurate to say that the loss encourages redistribution of probability mass away from old classes after pruning, not that it formally releases representational capacity.
- The Laplacian-based projection estimation is the least clear part of the method. The paper claims to compute second-order gradients to identify low-curvature regions and store them in a position map \(P_t(h,w)\), but it is not explained with enough precision how this is computed in practice, nor why “Laplacian-based” is the right term here. The equations around (4)–(6) are difficult to parse conceptually even after accounting for parser artifacts.
- The confidence map \(C_t(h,w)\) seems to use cosine-like similarity between an old-knowledge label vector and the current feature, but the notation is not fully coherent. It is unclear whether this operates on logits, class prototypes, or features in a way that is consistent with segmentation semantics.
- \(L_{\text{Max}}\) is presented as an information-theoretic objective, and the appendix gives an MI-style interpretation. However, the derivation appears somewhat loose: it is not fully clear that the implemented loss exactly corresponds to mutual information maximization in the claimed way, especially given the segmentation setting and the use of pixel-wise predictions.
- Edge cases and failure modes are not discussed enough. For example:
  - What happens when the pruning threshold removes useful shared structure too aggressively?
  - How sensitive is the method to the quality of the old model’s predictions after pruning?
  - What happens when new classes are very similar to old classes and the confidence map becomes unreliable?
- For an ICLR bar, the method is interesting but the paper would benefit from a clearer algorithm box and a more rigorous explanation of why the three components interact coherently rather than heuristically.

### Experiments & Results
- The experiments do test the paper’s main claims: performance on multiple incremental settings, comparison to prior methods, ablations, and some analysis of confusion/stability.
- The choice of datasets is appropriate for CISS: Pascal VOC and ADE20K are standard and meaningful.
- The inclusion of challenging settings like 10-1 and 2-2 is a strength, since they test long incremental sequences where forgetting is more severe.
- A major issue is comparability across baselines. The tables mix ResNet101-based methods and ViT-based methods, and some entries are reproduced with the authors’ own code while others are original numbers from prior papers. The paper does attempt to separate some comparisons, but the presentation still makes it difficult to judge how fair the headline “state-of-the-art” claim is.
- The most important experimental concern is that the strongest comparisons are not always apples-to-apples:
  - some baselines use different backbones;
  - some are reproduced via code while others are cited from papers;
  - the paper also compares its ViT-based method with older ResNet-based methods in the same table.
  For ICLR, this should be made much more explicit, and the main claim should be scoped carefully to comparable settings.
- The ablation study on Table 3 is useful, but the table as presented is hard to interpret because the grouping and component combinations are not clearly explained in the main narrative. The reader needs a more explicit mapping from Grp. 1–8 to the inclusion/exclusion of \(L_{\text{Min}}, L_{\text{Esti}}, L_{\text{Max}}\).
- The paper reports a small repeated-run study in Table 5 and Table 13, which is good. However, three runs are a minimal statistical basis, and the reported std values are quite small relative to the gains in some settings. The paper should be more careful about claiming robustness from such limited repetitions.
- There is no ablation isolating the parameter-release threshold in relation to the learned performance across all settings beyond the threshold sweep in Table 10. That sweep is helpful, but it does not fully answer whether the benefit comes from pruning itself, the altered distillation target, or the additional entropy/projection losses.
- The qualitative figures are consistent with the quantitative story, but they are mainly illustrative rather than diagnostic. The t-SNE plots are not strong evidence on their own for “reduced class confusion,” although they support the narrative.
- Overall, the experimental results are promising and likely meaningful, but the paper’s strongest claims need tighter qualification and cleaner apples-to-apples evaluation to meet ICLR’s standard for convincing empirical evidence.

### Writing & Clarity
- The paper is generally readable, and the high-level intuition is understandable.
- The largest clarity issue is the method section: the conceptual flow between parameter release, Laplacian-based estimation, and entropy maximization is not easy to follow. The paper would benefit from a compact algorithm listing and a diagram that maps each loss to the exact tensors it acts on.
- Theoretical appendices are lengthy and attempt to justify the losses, which is good in spirit. But the derivations are not always easy to follow and sometimes feel more like post hoc interpretation than a clean proof of the method’s properties.
- Figures 1–3 and the experimental tables are useful at a high level, but the presentation of some tables is too dense to be readily interpretable without substantial effort. The key issue is not formatting per se, but that the reader has to work hard to identify what the tables are actually demonstrating.
- The paper would also benefit from clearer distinctions between “old model,” “pruned old model,” “current model,” and “teacher signal,” since these roles are easy to conflate.

### Limitations & Broader Impact
- The paper includes ethics, reproducibility, and broader impact sections, which is commendable.
- However, the limitations are underdeveloped. The paper does not clearly state when DKD may fail or be less suitable.
- Important limitations that are not discussed enough include:
  - reliance on a carefully tuned pruning threshold \(\tau\) and weighting parameter \(\gamma\);
  - potential sensitivity to backbone choice and training protocol;
  - possible degradation when old and new classes are highly overlapping semantically;
  - the computational overhead of the additional losses and pruning logic during training.
- The broader impact section is reasonable but generic. It does not engage with a substantive negative societal risk beyond a general statement about misuse in surveillance. Since semantic segmentation itself can be used in privacy-sensitive settings, the paper could have discussed whether improved continual adaptation might enable more persistent monitoring systems.
- The paper does not acknowledge that “parameter release” could make models harder to interpret or audit, since previous knowledge is no longer cleanly preserved in a frozen teacher-student form.

### Overall Assessment
This is a relevant and potentially impactful paper for ICLR: it addresses a real and difficult problem in class-incremental semantic segmentation, proposes an interesting combination of parameter release and knowledge reuse, and reports strong empirical results across several standard settings. That said, the current submission still falls short of ICLR’s bar for fully convincing contributions because the method is somewhat underspecified, the theoretical justification is not fully rigorous, and the experimental comparisons are not always cleanly apples-to-apples. I would regard the paper as promising and likely competitive, but it needs a sharper algorithmic description, more careful scoping of claims, and clearer fair-comparison evidence before the main conclusions can be fully trusted.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes DKD, a distribution-based knowledge distillation framework for class-incremental semantic segmentation (CISS) that combines parameter release with knowledge reuse. The method uses a pruning-based “release” step to free low-sensitivity parameters from old knowledge, then applies Laplacian-based projection estimation and entropy-based objectives to better align old and new representations, aiming to reduce forgetting and class confusion without extra inference cost.

### Strengths
1. **Addresses an important and challenging problem in CISS.** The paper targets class-incremental semantic segmentation under realistic constraints where old data is unavailable, which is squarely within a meaningful ICLR-relevant continual learning setting.
2. **Clear high-level motivation beyond standard KD.** The authors identify two plausible failure modes of mainstream distillation in static architectures: parameter competition and underutilization of acquired knowledge. This framing is conceptually interesting and helps distinguish the method from prior KD variants.
3. **Strong empirical breadth across benchmarks and settings.** The paper evaluates on Pascal VOC 2012 and ADE20K, including standard and more challenging incremental configurations (e.g., 10-1, 2-2, 100-5, 100-10), and includes comparisons to many recent baselines.
4. **Ablation studies are extensive.** The paper studies the contributions of the three losses, threshold τ, hyperparameter γ, and pruning effects, which helps support the claim that each component contributes to performance.
5. **Compatibility claims are partially supported.** The appendix reports experiments integrating DKD into an existing method (CoinSeg) and also into a ResNet101-based setting, suggesting the approach is not limited to a single backbone.
6. **The paper reports training stability and error analyses.** Repeated runs, loss curves, and qualitative visualizations strengthen the empirical story, even if they do not fully resolve all concerns.

### Weaknesses
1. **The core technical novelty is somewhat incremental relative to existing distillation/regularization ideas.** DKD combines pruning, feature alignment, and entropy-based balancing, but the overall recipe resembles a composition of known mechanisms. The paper does not yet make a compelling case that this is a fundamentally new principle rather than a tailored heuristic for CISS.
2. **The method description and notation are difficult to follow.** Even allowing for parser artifacts, the presentation is dense and at times ambiguous, especially around the definitions of \(L_{Min}\), \(L_{Esti}\), and \(L_{Max}\), the role of the “position map” and “confidence map,” and how the Laplacian-based projection is actually computed in practice.
3. **Reproducibility is not fully convincing from the paper text alone.** Key implementation details appear incomplete or under-specified: pruning is described layer-wise, but the exact units affected, handling across architectures, exact training schedule per step, and how gradients flow through the proposed masks/maps are not fully transparent. The appendix claims code availability, but the paper itself would still leave many questions for an independent reimplementation.
4. **The empirical gains, while positive, are not always shown in a way that clearly establishes a strong ICLR-level advance.** The paper reports improvements over competitive methods, but the improvements appear moderate in many settings and the gap to joint training is still nontrivial in several tables. The paper would benefit from a clearer analysis of where DKD materially changes failure cases rather than only average mIoU.
5. **Limited evidence that the approach resolves the underlying mechanism claimed.** The main thesis is about “parameter competition” and “knowledge reuse,” but the paper mostly provides performance numbers and qualitative plots. There is little direct diagnostic evidence that parameter release actually reduces competition in a measurable way or that the reuse mechanism is uniquely responsible for the gains.
6. **Some claims seem stronger than the evidence warrants.** Statements such as “near-upper-bound average performance” and “state-of-the-art” are plausible in some settings, but the tables show that performance remains below joint training and that improvements vary by setting and backbone. ICLR reviewers typically expect carefully calibrated claims.
7. **Scalability and computational cost are only partially addressed.** The paper notes no extra inference cost and some training overhead, but it does not deeply analyze the cost of computing the proposed maps or the practicality of repeated pruning/release on larger models and more steps.
8. **The paper is somewhat benchmark-centric.** It is strong on standard CISS evaluation but lighter on broader analysis, such as failure modes, sensitivity to backbone choice beyond one additional experiment, or comparison to alternative ways of encouraging reuse (e.g., prototype-based, replay-free, or parameter-efficient methods).

### Novelty & Significance
**Novelty:** Moderate. The paper introduces a distinct framing—minimization-maximization distillation with parameter release and knowledge reuse—but the ingredients largely build on established ideas in distillation, pruning, entropy regularization, and representation alignment. The novelty lies more in the combination and CISS-specific formulation than in a clearly new learning paradigm.

**Significance:** Moderate to potentially good for the CISS subcommunity, but borderline for ICLR unless the method is shown to have broader implications beyond this niche. The empirical improvements and the no-inference-overhead property are attractive, yet the paper would need stronger conceptual clarity and stronger mechanistic evidence to meet ICLR’s usual bar for broadly meaningful and well-substantiated advances.

**Clarity:** Below average. The presentation is conceptually interesting but hard to parse, with dense math and several under-explained constructs.

**Reproducibility:** Fair but not excellent. The appendix improves matters, but the paper would still benefit from a cleaner algorithmic specification and more explicit implementation details.

### Suggestions for Improvement
1. **Provide a concise algorithm box with exact step-by-step operations.** Spell out how parameter release is computed, which tensors are masked, how maps \(P_t\) and \(C_t\) are formed, and how all losses are combined in each training step.
2. **Strengthen the mechanistic evidence.** Add diagnostics showing that parameter release actually changes model capacity allocation, e.g., layer-wise sparsity evolution, gradient conflict measures, or representational similarity before/after DKD.
3. **Clarify the mathematical formulation of \(L_{Esti}\) and \(L_{Max}\).** Right now, the explanation is too abstract. A small worked example or pseudocode would help readers understand what is optimized and why.
4. **Tighten and calibrate claims.** Replace broad statements like “near-upper-bound” with more precise language, and explicitly report where the method still falls short of joint training.
5. **Expand ablations to isolate each conceptual claim.** For example, compare pruning-only, entropy-only, projection-only, and alternative non-Laplacian reuse schemes to show that each specific design choice matters.
6. **Discuss computational overhead more rigorously.** Report training time, memory use, and any extra cost from map estimation and pruning across settings, not just a single per-epoch figure.
7. **Improve presentation for ICLR standards.** A cleaner narrative, simplified notation, and a clearer relationship to prior work would substantially increase readability and make the contribution easier to assess.
8. **Add broader generalization tests.** If possible, evaluate on additional segmentation or continual-learning benchmarks, or test robustness across different backbones and decoders more systematically.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a direct comparison to the strongest recent ICLR/CVPR-era CISS baselines under the same backbone and protocol, especially methods beyond the older KD family. ICLR reviewers will expect evidence that DKD beats not just legacy ResNet101 methods but also the best ViT-based and background-shift-aware methods under identical training settings.

2. Add a strict joint-training and fine-tuning sanity suite across all settings, including “upper bound,” “lower bound,” and “no-distillation” baselines. Without these, the claim that DKD “approaches joint learning” and that gains come from the proposed mechanism rather than extra optimization tricks is not convincing.

3. Add ablations that isolate the parameter-release component from the knowledge-reuse component in a stronger way, including “prune only,” “reuse only,” “entropy only,” and “Laplacian only” variants across multiple step settings. Right now the paper does not convincingly show that the gains require the full minimization–maximization design rather than one strong sub-loss.

4. Add compute and memory cost comparisons against baselines, including training time, peak GPU memory, and parameter count after pruning/release. ICLR places weight on whether a method is not just better but also practically preferable; without these numbers, the “no extra inference burden” claim is under-supported.

5. Add a more challenging cross-dataset or cross-backbone generalization study beyond a single ResNet101 plug-in example. The current evidence is too narrow to support the broader claim that DKD is a general strategy for CISS rather than a ViT-specific recipe.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify how much forgetting, plasticity, and class confusion each component reduces over incremental steps, not just final mIoU. The paper claims to balance stability and plasticity, but it does not show stepwise forgetting curves, backward transfer, or confusion matrices in a way that makes the mechanism credible.

2. Analyze whether the pruning threshold actually releases meaningful capacity or merely zeros weights without changing effective representational use. The current “parameter release” story needs evidence on sparsity patterns, which layers are pruned, and whether released parameters are reactivated or stay dead.

3. Validate the Laplacian/projection estimation mathematically and empirically with clearer diagnostics of the position map and confidence map. ICLR reviewers will likely question whether these maps are stable, whether they correlate with reusable regions, and whether the second-order construction is more than a heuristic.

4. Provide sensitivity analysis over the pruning threshold and γ on more than one representative setting, with error bars. The current tuning story looks ad hoc; the paper needs to show the method is not fragile to hyperparameter choice.

5. Report variance across more than three runs and include statistical significance against the strongest baselines. The current error analysis is too limited to support “state-of-the-art” or “near-upper-bound” claims in a competitive ICLR setting.

### Visualizations & Case Studies
1. Show per-step forgetting and confusion evolution, including class-wise confusion matrices for old vs. new classes at each incremental stage. This would reveal whether DKD truly reduces class confusion or mainly improves a few easy categories.

2. Visualize the pruning/release masks layer by layer and over training steps. Without showing where capacity is “released,” it is hard to believe the mechanism is doing more than standard regularization.

3. Add side-by-side feature-space plots for baseline, prune-only, reuse-only, and full DKD, with class-separation metrics. The current t-SNE figures are too qualitative; they do not establish that the representation geometry explains the gains.

4. Include failure-case segmentation examples where DKD still confuses semantically similar classes or small objects. ICLR reviewers will want to see whether the method breaks in the exact cases where CISS is hardest.

### Obvious Next Steps
1. Extend DKD to a standard continual-learning benchmark suite beyond Pascal VOC and ADE20K, with a unified protocol. Two datasets are not enough to support a broad claim of a general CISS method.

2. Test DKD against rehearsal-based, architecture-expansion, and modern background-shift methods under a consistent resource budget. Right now the paper does not show whether its improvement comes from better ideas or from a favorable comparison regime.

3. Replace the current heuristic description of parameter release with a more principled optimization or a data-driven criterion. At ICLR, the method needs a stronger justification for why the chosen pruning rule is the right one.

4. Provide an open, exact reproduction package with all hyperparameters, seeds, and evaluation scripts. The paper claims reproducibility, but the main results would be more credible if the full experimental protocol were immediately auditable.

# Final Consolidated Review
## Summary
This paper proposes DKD, a distillation-based approach for class-incremental semantic segmentation that tries to address two issues the authors identify in standard KD: parameter competition between old and new knowledge, and underuse of previously acquired knowledge. The method combines low-sensitivity parameter pruning/release, a Laplacian-based projection estimation module, and an entropy-based objective to encourage reuse of shared knowledge, and it reports strong results on Pascal VOC and ADE20K across several incremental settings.

## Strengths
- The paper targets a genuinely important and difficult problem in class-incremental semantic segmentation, especially in the realistic regime where old data is unavailable.
- The empirical scope is fairly broad: it evaluates on two standard datasets, multiple incremental splits including harder ones like 10-1 and 2-2, and includes ablations on the main losses, threshold \(\tau\), and \(\gamma\), plus repeated-run variance and qualitative examples.
- The method is not limited to one specific backbone in the authors’ own experiments; the appendix includes a plug-in-style test on CoinSeg and a ResNet101-based variant, which at least partially supports the claim of broader compatibility.

## Weaknesses
- The core method is underspecified and still hard to reconstruct precisely. The paper mixes pruning, pseudo-label adjustment, Laplacian/projection estimation, and entropy maximization, but the exact training pipeline, mask application, and the computation of the position/confidence maps are not crisply laid out in the main text. This is a real reproducibility and assessment problem.
- The main theoretical story is not fully convincing. The appendices provide gradient- and information-theoretic interpretations, but they read more like post hoc justification than a rigorous explanation of why the three components should work together. In particular, the claim that pruning “releases parameters” is only indirectly supported.
- The empirical comparison is not always apples-to-apples. The paper mixes ResNet101 and ViT baselines in the same tables, and some numbers are reproduced while others are taken from prior papers. The results look promising, but the strongest “state-of-the-art” and “near-upper-bound” language is somewhat overclaimed given the remaining backbone/protocol confounds.

## Nice-to-Haves
- A concise algorithm box with explicit pseudocode for each training step, including how masks and maps are computed and applied.
- A stronger diagnostic analysis of whether the pruning step באמת changes capacity allocation, e.g., layer-wise sparsity over time or gradient conflict measurements.
- A cleaner ablation that isolates prune-only, reuse-only, entropy-only, and Laplacian-only variants to show which part is actually doing the work.

## Novel Insights
The most interesting aspect of the paper is its reframing of continual segmentation distillation: instead of treating the old model as a fixed target to be preserved, DKD tries to actively reshape the distribution by pruning low-sensitivity parameters and then reusing the remaining shared structure to guide new-class learning. That is a plausible and potentially useful perspective, but in the current form it still feels like a heuristic composition of several known ingredients rather than a sharply principled new paradigm. The empirical gains suggest the recipe is competitive, yet the paper does not fully demonstrate that the proposed “parameter release” and “knowledge reuse” mechanisms are uniquely responsible for those gains.

## Potentially Missed Related Work
- None identified

## Suggestions
- Provide an explicit step-by-step algorithm and clarify exactly when pruning/masking is applied, to which model components, and how the Laplacian-based maps are computed in practice.
- Tighten the comparison protocol: report matched-backbone comparisons separately from cross-backbone comparisons, and scope the headline claims accordingly.
- Add mechanistic evidence that parameter release and knowledge reuse are doing more than generic regularization, ideally through layer-wise sparsity/activation analyses and stronger ablations.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 4.0]
Average score: 3.3
Binary outcome: Reject
