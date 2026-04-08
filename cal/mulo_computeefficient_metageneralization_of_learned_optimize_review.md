=== CALIBRATION EXAMPLE 27 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "µLO: Compute-Efficient Meta-Generalization of Learned Optimizers" accurately reflects the content. The abstract clearly states the problem (LOs failing to meta-generalize to wider tasks), the method (deriving µP for two LO architectures), and the results (width generalization, and unexpected depth/horizon generalization). The claim of "zero extra computational cost" compared to SP LOs is presented honestly as coming from the meta-training recipe rather than the parameterization itself. One minor concern: the abstract mentions "25× meta-training" generalization to longer horizons without clarifying this means 25× longer than the meta-training unroll length, which could be misread.

---

### Introduction & Motivation

The problem of meta-generalization is well-motivated. The paper connects µP (a tool for hyperparameter transfer across widths in hand-designed optimizers) to the analogous problem of LOs failing to generalize to wider networks, which is a natural and compelling bridge. The observation that even VeLO trained on 4000 TPU-months fails to generalize beyond meta-training widths (referencing Figures 6 and 9 of Metz et al., 2022b) provides concrete empirical motivation. Contributions are clearly enumerated. The framing as "zero extra computational cost" is both accurate and appropriately highlighted.

One gap: the introduction could more clearly articulate *why* SP LOs fail on wider tasks in contrast to why µP should help — the intuition that feature learning collapses under SP at large width is never explicitly connected to the gradient signal becoming zero or exploding, which would sharpen the technical motivation for a reader unfamiliar with the Tensor Programs literature.

---

### Related Work

The related work is thorough. The authors correctly distinguish their work from VeLO-style scale-up approaches, generalization-via-regularization approaches, and pure µP/hyperparameter-transfer work. The citation of concurrent work (CompleteP, Dey et al., 2025) is appreciated. The discussion of Everett et al. (2024), which shows that SP with per-layer learning rates can sometimes outperform µP, is honestly included even though it is potentially damaging to the paper's premise — this signals good scholarly integrity. The authors appropriately note that the question of optimal parameterization for LO meta-learning remains open.

---

### Method / Approach (Section 4)

**Strength of the derivation.** The µP modification for LOs is cleanly stated: (i) standard µP initialization, (ii) 1/FAN\_IN multiplier on output layer pre-activations, (iii) 1/FAN\_IN rescaling of the learned optimizer's update for hidden layers only (Eq. 3). This is the direct adaptation of µP to the specific update rule of small\_fc\_lopt and VeLO (Eq. 2).

**Assumptions in Propositions 4.1 and 4.2.** Both propositions require that "during training the optimizee's parameters and input data become aligned, leading to Law of Large Numbers (LLN) scaling." This is a substantive and non-trivial assumption. LLN scaling requires that inner products of weight rows/columns and data vectors concentrate around their means as width grows, which holds for random weights but is not guaranteed after many optimizer steps. The authors do not justify this assumption under the action of the learned optimizer (as opposed to, say, SGD where concentration has been studied). Since the proof is in Appendix A.2 (unavailable due to file truncation), it is unclear whether this assumption is used as an approximation or formally justified. This is the paper's most significant theoretical vulnerability — a reader must accept the propositions somewhat on faith unless the appendix proof is closely inspected.

**Underspecification of hidden vs. output layer distinction.** Eq. (3) gives two cases: hidden layers (rescale by 1/FAN\_IN) and "otherwise" (no rescaling for the optimizer update). For input layers, no optimizer update scaling is applied. This is consistent with standard µP, where input layers scale via initialization rather than update. However, the paper does not discuss what happens for embedding layers, attention projection weights, or other non-standard weight matrices that appear in ViTs and transformers — architectures that are evaluated but not covered in the formal treatment.

**Missing: VeLO-specific complexity.** VeLO additionally produces a tensor-level learning rate ε_W (Eq. 2). Proposition 4.2 extends Proposition 4.1 to this case, but the proof structure is not visible. The tensor-level learning rate introduces an extra learned multiplier that could itself depend on width; it is not clear whether this is addressed in the derivation.

---

### Experiments & Results

**Experimental setup.** The meta-training recipe (µLO_M: width ∈ {128, 512, 1024}, 1000-step inner problems) is simple and the FLOP-matched comparison with SP LOs is properly controlled. The 35-task evaluation suite spanning MLPs, ViTs, and LMs is appropriately diverse given the compute constraints. The baseline tuning effort (500+ configurations of µAdam and AdamW per task) is thorough and ensures hand-designed optimizers are not straw-manned.

**Figure 3 (meta-training distribution ablation).** The µLO_M vs µLO_S comparison cleanly shows that multi-width meta-training helps. More importantly, µLO_S still substantially outperforms LO_M, confirming that µP itself — rather than multi-width training — is the dominant factor. This is a useful and well-executed ablation.

**Figure 4 / Table 1 (width generalization, main result).** The key comparison — µLO_M outperforming SP LO_M on unseen wide tasks — is compelling and visually clear. The divergence of SP LOs at large widths while µLOs train stably is a qualitatively strong result. The evaluation extends to ViTs and LMs despite meta-training only on MLPs, which is a meaningful test of generalization.

However, three issues arise:

1. **Missing per-width oracle baseline.** AdamW and µAdam are tuned at width=1024 and transferred to larger widths without re-tuning. The paper acknowledges this as a limitation, but it means the comparison is not fully fair to AdamW at large widths: SP AdamW tuned fresh at width=8192 might perform quite differently. This is the most important missing ablation and prevents the authors from making the strong claim that µLOs outperform per-task-tuned hand-designed optimizers at test time widths.

2. **Training loss only.** All evaluation metrics report training loss. While this is appropriate for assessing optimizer quality per se, a reader interested in practical deployment of µLOs would want to see validation loss or test accuracy, which would indicate whether µLOs introduce any optimization artefacts (e.g., sharp minima) that hurt generalization.

3. **Internal reference inconsistency.** Section 5.2.3 says "Figure 3 compares the training loss after 1000 steps of SP learned optimizers to µ-parameterized learned optimizers for different widths," but Figure 3 is the meta-training distribution ablation (ImageNet-32 at 1000 and 5000 steps plus OOD dataset results). The width generalization comparison is apparently in Figure 4. This mislabeling makes the results harder to follow.

**Figure 2 (pre-activation stability).** This empirically verifies µP Desideratum J.1 (stable pre-activations across widths). SP Adam blows up immediately for large-width MLPs; SP LOs take longer but also blow up. µP variants remain stable. This is necessary but not sufficient to confirm full µP compliance — in particular, Desideratum J.2 (maximal updates, every parameter moves O(1)) is not verified.

**Depth and horizon generalization (Figures 5–6).** These are reported honestly as purely empirical findings with no theoretical backing. The results are striking: µLOs meta-trained on 3-layer MLPs generalize to 16-layer ViTs and LMs, and to 25,000-step training (vs 1,000-step meta-training). The authors hypothesize this is due to pre-activation stability (Section F.1.2 in the appendix), which is plausible but not demonstrated. The risk is that readers may over-interpret these results as a principled theoretical property rather than a surprising empirical regularity. Tightening the hypothesis — e.g., by showing that the variance of updates at deeper layers follows µP scaling — would substantially strengthen this section.

**Comparison against VeLO (Metz et al., 2022b) at full scale.** The original VeLO was trained for 4000 TPU-months. This paper's µLO_M is trained for 100 GPU hours. The paper argues that VeLO still fails on large OOD widths (referencing Metz et al.), but does not directly compare µVeLO_M against the full VeLO checkpoint in any experiment. A direct comparison — even on a single task — would significantly strengthen the claim that µP-based meta-training is more efficient than scale-based meta-training for addressing OOD width generalization.

---

### Limitations & Broader Impact

Section 6 honestly lists three limitations: (1) meta-training only on MLPs, (2) no evaluation beyond width 8192, (3) no per-width oracle SP AdamW baseline. These are appropriate and the authors do not oversell their results. However, two additional limitations are not acknowledged:

- **No multi-task or diverse-domain meta-training.** The µP derivation is theoretically principled for width transfer, but nothing in the theory guarantees that an LO meta-trained on image classification MLPs should generalize to language modeling. The empirical success is noteworthy but its mechanism is unexplained — it could be an artefact of the specific architectures tested.

- **Sensitivity of the LLN assumption to LO dynamics.** As noted above, the theoretical propositions assume LLN scaling during training, which is not guaranteed. It is unknown whether µLOs violate µP desiderata in practice for very long training runs or for non-standard architectures; the training curves in Figure 6 show stable training but do not rule out subtle violations.

---

### Overall Assessment

µLO makes a clear and practically valuable contribution: deriving µP for two prominent learned optimizer architectures and demonstrating that meta-training under µP substantially improves meta-generalization to wider networks at zero additional meta-training cost. The empirical results are comprehensive, the baselines are strong, and the qualitative findings on depth and horizon generalization go beyond the theoretical scope in a way the authors are appropriately cautious about. The principal weaknesses are: (i) the theoretical propositions rest on an LLN scaling assumption whose validity under learned optimizer dynamics is unverified; (ii) the update scaling in Eq. (3) does not address transformer-specific weight matrices, leaving the ViT/LM results somewhat theoretically under-motivated; (iii) the absence of a per-width oracle AdamW baseline weakens the strongest claim (µLOs outperform per-task-tuned hand-designed optimizers); and (iv) a mislabeled figure reference impedes navigation of the core results. Despite these issues, the paper's contribution stands: it offers a principled, low-cost approach to a long-standing failure mode of learned optimizers, and the average-rank results in Table 1 are sufficiently large-margin to be convincing even with the noted caveats. This is above the ICLR acceptance threshold provided the proof in Appendix A.2 satisfactorily addresses the LLN assumption.

# Neutral Reviewer
## Balanced Review

### Summary
This paper adapts the Maximal Update Parametrization (µP) framework to Learned Optimizers (LOs) by deriving width-dependent scaling rules for initialization, pre-activations, and update steps for two prominent LO architectures (VeLO and small\_fc\_lopt). The authors propose a straightforward multi-width meta-training recipe and empirically demonstrate that µ-parameterized LOs significantly outperform standard-parameterized LOs in generalizing to wider, deeper, and longer-horizon optimization tasks. At a fraction of the compute used by prior large-scale LOs, µLOs achieve competitive or superior performance against extensively per-task tuned hand-designed optimizers.

### Strengths
1. **Clear Theoretical Extension of µP to LOs:** The paper rigorously derives the necessary scaling factors for LO update mechanisms (Propositions 4.1 & 4.2), formally showing they satisfy µP desiderata under standard LLN alignment assumptions. This bridges two important but previously disconnected lines of work.
2. **Robust & Multi-Axis Empirical Evaluation:** The experimental suite systematically tests generalization across width, depth, unroll length, and architecture families (MLPs, ViTs, LMs). Evidence in Table 1 and Figures 3–6 consistently shows µLOs avoid divergence and maintain smooth training curves where SP baselines fail, outperforming or matching per-task tuned Adam/µAdam.
3. **High Compute-Efficiency & Accessibility:** Meta-training µLOs requires ~100 GPU hours, contrasting sharply with prior large-scale efforts requiring thousands of TPU months. This makes the approach highly practical for academic research. The accompanying code release further strengthens accessibility.
4. **Well-Motivated Ablation & Baseline Design:** The comparison between single-width and multi-width meta-training (Figure 3) justifies the proposed recipe. Per-task extensive grid search (~500 configs) for AdamW/µAdam ensures baseline fairness and highlights true zero-shot generalization capability of the LOs.

### Weaknesses
1. **Narrow Meta-Training Distribution:** LOs are exclusively meta-trained on 3-layer MLP image classification tasks. While results on ViTs and LMs are promising, the paper lacks ablation on more complex architectures during meta-training, leaving it unclear whether multi-width training alone suffices or if architectural priors are missing. This limits claims about broad meta-generalization.
2. **Missing Oracle Baseline Context:** The authors acknowledge not sweeping optimizer hyperparameters at every test width. This means the reported advantage of µLOs over hand-designed optimizers may be partially an artifact of suboptimal baseline tuning at larger widths. A swept baseline or sensitivity analysis would better contextualize the true performance gap.
3. **Mechanistic Gap for Depth/Length Generalization:** Improvements for deeper networks and 25x longer unrolls are noted as surprising and purely empirical. While activation stability is hypothesized, the paper lacks quantitative analysis (e.g., gradient variance, activation covariance spectra, or Hessian condition numbers over long horizons) to substantiate why µP yields these unexpected benefits.
4. **Assumption-Heavy Theoretical Grounding:** Propositions 4.1/4.2 assume feature alignment and LLN scaling, which hold asymptotically but may be violated during early or unstable training phases. The paper does not discuss how µLOs behave when these assumptions break down, nor does it analyze finite-width corrections relevant to practical meta-training widths (128–1024).

### Novelty & Significance
*Novelty:* Moderate-to-High. While µP is well-established for hand-designed optimizers, formally deriving it for the non-standard output structures of LOs (magnitude/direction decoupling + tensor-level learning rates) is a novel and non-trivial contribution. The multi-width meta-training strategy, though conceptually simple, is effectively tailored to unlock µP's theoretical guarantees in the L2O setting.
*Clarity:* High. The paper is logically structured, with a clean progression from theoretical derivation to empirical validation. Definitions and scaling rules are explicit, and figures/tables are well-annotated to support claims.
*Reproducibility:* High. The authors provide an open-source codebase, clearly define task distributions, baseline tuning protocols, and report exact compute budgets (~100 GPU hours). Appendix references contain necessary hyperparameters and architectural details, satisfying ICLR reproducibility standards.
*Significance:* High. Meta-generalization remains the primary bottleneck for scalable Learned Optimizers. Demonstrating a compute-efficient, theoretically grounded path to scaling LOs to larger widths and longer horizons directly addresses a community-wide challenge and lowers the barrier for L2O adoption in resource-constrained settings.

### Suggestions for Improvement
1. **Architectural Diversity in Meta-Training:** Include a lightweight CNN or shallow transformer in the meta-training distribution to disentangle whether µP's benefits on ViTs/LMs stem purely from width scaling or require some degree of architectural alignment. If computationally prohibitive, explicitly bound claims to width-scaling transfer.
2. **Contextualize Baseline Tuning Gap:** Add a small-scale experiment or analysis discussing how a width-swept Adam/µAdam oracle might perform, or report a sensitivity analysis showing baseline loss variance across width. This would clarify whether µLOs truly surpass optimizers or simply outperform poorly transferred hyperparameters.
3. **Mechanistic Analysis for Empirical Findings:** Provide quantitative diagnostics for the depth and length generalization results. Tracking metrics like activation covariance drift, gradient norm stability, or effective learning rate evolution over 25k steps would transform observations into mechanistic insights.
4. **Finite-Width & Assumption Discussion:** Add a short subsection discussing the validity of LLN alignment at meta-training widths (128–1024) and how µLOs might behave if alignment assumptions are violated. Citing finite-width corrections or providing a quick empirical robustness check (e.g., different initialization schemes) would strengthen theoretical rigor.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Add oracle baselines tuned at target width.** The claim that $\mu$LOs outperform hand-designed optimizers is undermined because Adam/$\mu$Adam were tuned only at width=1024, not at the target widths (e.g., 8192). Without an oracle baseline tuned specifically at the test width, you cannot distinguish superior generalization from superior hyperparameter transfer.
2. **Report wall-clock time efficiency, not just steps.** The title claims "Compute-Efficient," and the introduction promises reduced wall-clock time, yet results only show loss vs. steps. Learned optimizers have higher per-step overhead than Adam; without wall-clock comparisons, the efficiency claim is unsupported.
3. **Ablate individual $\mu$P components.** $\mu$P involves initialization, multipliers, and update scaling. You must ablate these individually to prove which component drives the depth/length generalization, otherwise the theoretical derivation appears unnecessary.
4. **Measure meta-gradient norms during meta-training.** Long unrolls (1000 steps) typically cause vanishing/exploding meta-gradients. You need to plot meta-gradient norms to demonstrate that $\mu$P stabilizes the meta-learning process itself, not just the inner optimization.

### Deeper Analysis Needed (top 3-5 only)
1. **Explain the mechanism for depth generalization.** $\mu$P theory guarantees stability across width, not depth. You must analyze layer-wise activation norms in deep networks to validate your hypothesis that width-stabilization indirectly fixes depth issues.
2. **Analyze feature distribution shift between MLP and ViT.** You meta-train on MLPs but test on Transformers. The optimizer's input features (momentum, gradients) must have compatible statistics across architectures for this generalization to be valid rather than coincidental.
3. **Verify meta-optimizer hyperparameter transfer.** $\mu$P enables optimizee HP transfer, but you must analyze if the *meta-learning* rate also transfers across widths. If the meta-optimizer requires re-tuning for larger meta-training tasks, the compute savings are negated.

### Visualizations & Case Studies
1. **Plot Pareto frontiers of Loss vs. Wall-Clock Time.** This visualization is required to substantiate the "Compute-Efficient" claim by showing if $\mu$LOs actually reach target loss faster in seconds than tuned Adam.
2. **Visualize meta-gradient norms over the unroll.** Plotting gradient norms over the 1000-step meta-training unroll would expose whether $\mu$P prevents the instability that typically limits LO unroll lengths.
3. **Show histograms of optimizer input features.** Compare distributions of LO inputs (e.g., gradient magnitudes) between MLP meta-training tasks and ViT test tasks to reveal if the model is operating out-of-distribution.

### Obvious Next Steps
1. **Meta-train on mixed architectures (MLP + Transformer).** Relying solely on MLP meta-training limits the claim of a "general-purpose" optimizer; including Transformers in the meta-training distribution is necessary for broader validity.
2. **Validate on actual large-scale LLMs.** Testing on width-8192 MLPs is not equivalent to training 8B parameter LLMs. Verification on realistic large-scale language models is required to prove practical utility.
3. **Investigate failure modes at extreme scales.** The paper acknowledges computational limits prevented testing beyond specific widths. You should explicitly identify where the method breaks (e.g., width > 10k) to define the boundary of applicability.

# Final Consolidated Review
## Summary

The paper derives the Maximal Update Parametrization (µP) for two learned optimizer architectures (VeLO and small_fc_lopt), providing explicit scaling rules for initialization, pre-activation multipliers, and optimizer updates. Under a proposed multi-width meta-training recipe, µ-parameterized learned optimizers (µLOs) are shown to generalize substantially better to wider, deeper, and longer-horizon optimization tasks than standard-parameterized LOs, at comparable meta-training compute cost (~100 GPU hours).

## Strengths

- **Principled theoretical contribution:** The derivation of µP scaling rules for learned optimizers (Propositions 4.1 and 4.2) is a non-trivial extension of prior work on µP for hand-designed optimizers. The adaptation handles the non-standard output structure of LOs (magnitude/direction decoupling, tensor-level learning rates), which requires careful treatment of how FAN_IN scaling interacts with learned updates.

- **Comprehensive empirical evaluation across multiple axes:** The evaluation systematically tests meta-generalization across width (128→8192 for MLPs, up to 12288 for transformers), depth (3→16 layers), training horizon (1K→25K steps), and architecture families (MLPs, ViTs, LMs). Table 1 and Figures 4–6 consistently show µLOs maintaining stable training where SP LOs diverge or fail to progress.

- **Compute efficiency is demonstrated for meta-training:** The µLO meta-training requires ~100 GPU hours, compared to the 4000 TPU-months used for the original VeLO. This makes the approach accessible to academic research groups.

- **Strong baseline tuning protocol:** The authors tune AdamW and µAdam across 500+ configurations per task at width=1024, ensuring the comparison against hand-designed optimizers is fair and not a straw-man baseline. The per-task tuning is explicitly noted in the methodology.

- **Clean ablation isolates µP's contribution:** Figure 3 shows µLO_S (single-width meta-training) substantially outperforms LO_M (multi-width SP meta-training), confirming that µP itself—not just the multi-width recipe—is the dominant factor in improved generalization.

- **Honest treatment of empirical findings:** The paper appropriately describes depth and horizon generalization results as "purely empirical" without claiming theoretical guarantees, and acknowledges limitations (Section 6) including restricted meta-training distribution, computational bounds on tested widths, and absence of per-width oracle baselines.

## Weaknesses

- **LLN scaling assumption in propositions lacks validation under LO dynamics:** Propositions 4.1 and 4.2 assume "during training the optimizee's parameters and input data become aligned, leading to Law of Large Numbers (LLN) scaling." While standard in µP theory for hand-designed optimizers, this assumption's validity under learned optimizer dynamics—where update magnitudes and directions are produced by neural networks trained via meta-learning—remains unverified. The proof in Appendix A.2 establishes sufficiency but the paper does not empirically or theoretically demonstrate that LLN conditions hold in practice under µLO optimization.

- **Missing theoretical treatment for transformer-specific weight matrices:** The derivation in Section 4 covers hidden, input, and output layers but does not explicitly address attention projection matrices (query, key, value, output projections) or embedding layers. Yet the evaluation includes ViTs and transformer LMs. The extension to these architectures is not trivially obvious from the presented theory, leaving the empirical success on transformers theoretically under-justified.

- **Only training loss is reported; no validation or test metrics:** All figures report training loss. Without validation loss or test accuracy, it is unclear whether µLOs introduce optimization artifacts (e.g., converging to sharp minima) that harm generalization performance. A practical practitioner would need to know whether the improved optimization translates to improved downstream metrics.

- **Mechanistic explanation for depth and horizon generalization is absent:** The empirical findings that µLOs generalize to 5× deeper networks and 25× longer training horizons are striking, but the paper offers only a brief hypothesis ("pre-activation stability, see Section F.1.2") without quantitative analysis. No measurements of layer-wise activation norms, gradient variance, or covariance spectra are provided to substantiate the claim. This leaves a significant empirical finding without convincing explanation.

- **Per-width oracle baseline missing; limits strongest claims:** The authors acknowledge they did not tune AdamW/µAdam at each target width. Table 1 shows µLOs achieving lower average rank than µAdam and AdamW on large-width tasks, but these baselines were tuned only at width=1024. A fair comparison at width=8192 would require per-width tuning. The current results demonstrate generalization from width-1024 hyperparameters, but do not establish that µLOs outperform optimizers that are properly tuned at the target scale.

- **"Wall-clock time" efficiency claim is unsupported:** The abstract claims learned optimizers "significantly reduce the wall-clock training time," but no wall-clock timing results are presented. Learned optimizers have higher per-step computational overhead than Adam; without timing data, it is unknown whether the reduced step count translates to reduced wall-clock time. The 100-GPU-hour figure refers to meta-training cost, not the inner optimization runtime.

## Nice-to-Haves

- Direct comparison against the original large-scale VeLO checkpoint (even on a single task) would strengthen the claim that µP-based meta-training is more compute-efficient than scale-based meta-training for OOD generalization.

- Ablation of individual µP components (initialization, multipliers, update scaling) would clarify which elements are necessary for the observed benefits, particularly for the empirical depth/horizon findings.

- Meta-gradient norm analysis during meta-training would demonstrate whether µP stabilizes the meta-learning process itself, addressing a known challenge in learned optimization.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Meta-optimizer hyperparameter transfer analysis"** — This demands analysis of whether the meta-learning rate transfers across widths, which is beyond the paper's stated scope. The paper addresses optimizee hyperparameter transfer, not meta-optimizer hyperparameters.

- **"Figure reference inconsistency is a critical error"** — While Section 5.2.3 references "Figure 3" when discussing width generalization (which is actually in Figure 4), this is a minor editorial error that does not affect the validity of results. It impairs readability but is not a substantive flaw.

- **"Desideratum J.2 (maximal updates) is not verified"** — While technically correct, J.1 (stable pre-activations) is the primary stability criterion. Empirical verification of J.1 (Figure 2) provides reasonable evidence that the parameterization is functioning as intended. Demanding J.2 verification goes beyond what is necessary for the claims made.

## Novel Insights

The most novel empirical finding is that µP confers benefits beyond its theoretical guarantee. µP is designed for width transfer, yet the experiments show unexpected improvements in depth generalization (5×) and training horizon generalization (25×). The authors hypothesize that "pre-activation stability" drives these benefits, but this remains conjecture. A deeper mechanistic investigation—tracking layer-wise statistics during deep network training, or analyzing update variance over long horizons—could reveal whether µP's stabilizing effect on activations incidentally addresses other failure modes of learned optimizers. This empirical phenomenon, if validated mechanistically, could motivate theoretical work extending µP-like guarantees beyond width scaling.

## Suggestions

- Add validation loss and/or test accuracy for at least one representative task in each architecture family (MLP, ViT, LM) to demonstrate that improved training dynamics translate to improved generalization.

- Provide wall-clock timing data comparing µLOs against Adam/µAdam on the target tasks, to substantiate the "wall-clock training time" claim in the abstract.

- Expand Section 5.2.4 (depth and horizon generalization) with quantitative analysis: report layer-wise activation variance, gradient norm stability, or effective learning rate evolution over training to provide mechanistic grounding for the empirical findings.

- Clarify how the µP scaling rules in Eq. (3) apply to attention projection matrices and embedding layers; a brief footnote or appendix discussion would suffice for completeness.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 2.0]
Average score: 5.0
Binary outcome: Accept
