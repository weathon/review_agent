=== CALIBRATION EXAMPLE 51 ===

# Harsh Critic Review
Now I have enough material to write a thorough review. Let me compose it.

---

## Section-by-Section Critical Review

---

### Title & Abstract

The title "µLO: Compute-Efficient Meta-Generalization of Learned Optimizers" is accurate and informative. The abstract claims are largely supported by the experiments, though the phrasing "surprisingly good generalization to deeper networks (5× meta-training)" is notable — the authors themselves acknowledge in the body that this is purely empirical and lacks theoretical grounding, yet the abstract presents it alongside theoretically motivated claims without that caveat. The qualifier "surprisingly" should cue readers, but a sharper epistemic distinction between the theoretically motivated width-generalization claim and the empirical depth/horizon results would improve the abstract.

---

### Introduction & Motivation (Section 1)

The motivation is clear and well-executed: VeLO (Metz et al., 2022b) fails to generalize to wider and deeper networks even after 4000 TPU-months of training, and the connection between µP's hyperparameter-transfer mechanism and LO meta-generalization is elegant. The key question — *Are existing LO architectures compatible with µP? Does meta-learning under µP improve meta-generalization?* — is sharply posed.

**Concern 1 (Framing of baselines):** The introduction states that VeLO "is known to have difficulty optimizing models much wider and deeper than those seen during meta-training." However, VeLO in the experiments is *not* the publicly released, 4000-TPU-month VeLO — it is a smaller VeLO_M meta-trained by the authors on a FLOP-matched budget. Conflating the two risks overselling the contribution. The authors should be more careful to distinguish between "VeLO as originally published" (a very strong baseline) and "VeLO_M meta-trained here on a small budget" throughout the introduction.

**Concern 2 (Missing related work):** The paper does not discuss any prior work on *why* SP learned optimizers fail on wider networks — i.e., the activation explosion problem. The related work section focuses on generalization broadly and µP broadly, but does not adequately situate what was previously *known* about the failure mode being targeted.

---

### Background (Section 2)

The background covers the meta-training objective (Eq. 1) and the update rules for small_fc_lopt and VeLO (Eq. 2) adequately. However, several aspects of the input features **u_t** are relegated entirely to the appendix (Tables 2–4), which are inaccessible in the parsed version but are critical for assessing whether the µP derivation covers all inputs correctly. The two constants ϑ₁ and ϑ₂ are described as "constant values set to 0.001 to bias initial step sizes to be small," but their role in the width scaling of the update (Eq. 3) is not discussed here — their interaction with FAN_IN scaling is left for the reader to sort out.

---

### µ-Parametrization for Learned Optimizers (Section 4) — **Most Critical Section**

This is the paper's main theoretical contribution, and it requires the closest scrutiny.

**The parameterization itself (Eq. 3)** amounts to: for hidden-layer weights, divide the learned optimizer's update by FAN_IN. This is essentially the standard µP update scaling already established for Adam/SGD, now applied to the black-box update `ε_W · ϑ₁ · d · exp(ϑ₂ · m)` produced by the learned optimizer. The conceptual leap is non-trivial — the learned optimizer is itself a neural network — but the derivation relies on a critical assumption.

**Concern 3 (LLN Alignment Assumption):** Propositions 4.1 and 4.2 both require that "during training the optimizee's parameters and input data become aligned, leading to Law of Large Numbers (LLN) scaling." This is a substantial and non-obvious assumption. In practice, this means the learned optimizer's input features (momentum accumulators, variance accumulators, adafactor accumulators) must scale in a specific way with width. The paper does not:
- Formally state what "aligned" means in this context
- Show empirically that the LLN scaling condition is satisfied during training
- Discuss when or why this assumption might *fail* (e.g., for very deep networks where activations compound)

The proofs (in Appendix A.2, not readable due to length) presumably establish the µP desiderata under this assumption, but the main paper presents only two proposition statements with no proof sketches. For a theory paper at ICLR, the main body should at minimum provide an intuitive argument for why LLN scaling holds for the specific input features of small_fc_lopt and VeLO. As written, the theoretical contribution rests on an unverified and poorly motivated assumption.

**Concern 4 (Scope of the derivation):** The paper claims to "derive µP for two popular learned optimizer architectures." However, the update rule (Eq. 3) applies only to the *optimizee* — it is a rule for how the learned optimizer updates the optimizee's weights. There is no discussion of whether the *learned optimizer itself* (small_fc_lopt's MLP or VeLO's LSTM) also needs to be parameterized differently with width. Since the learned optimizer processes features derived from the optimizee's gradients (whose scale changes with width), it is unclear whether the learned optimizer's internal weights are themselves in µP or SP. This gap should be addressed explicitly.

**Concern 5 (Novelty relative to existing µP):** The update rule in Eq. 3 is structurally identical to the µP update rule for Adam. The added complexity is that the learned optimizer *computes* the update, but the width-scaling applied is the same. The paper should explain more explicitly what technical challenges arise when applying µP to a *learned* update rule compared to a hand-designed one, and why the existing µP derivation does not immediately apply.

---

### Experiments & Results (Section 5)

**Setup:** The setup is systematic and well-described. The FLOP-matching across µLOs and SP LOs is an important design choice. The 500+ configuration grid search for hand-designed baselines makes those strong baselines credible.

**Concern 6 (Training loss only):** All results report *training loss*, never test/validation loss or accuracy. For image classification (CIFAR-10, ImageNet) and language modeling (LM1B), reporting final training loss conflates optimization quality with generalization. A learned optimizer that overfits aggressively will appear to perform well by this metric. For ICLR readers, this is a meaningful gap: the paper is ostensibly about optimizers that help train models that *generalize*, but generalization of the *model being optimized* is never measured.

**Concern 7 (Missing oracle baseline):** The paper acknowledges that "we did not include an oracle SP AdamW baseline whose hyperparameters are swept at every width." This is the most important missing baseline. Without knowing how per-width tuned AdamW (the strongest hand-designed optimizer for each specific width) performs, we cannot determine whether µLOs are actually better than a well-tuned hand-designed optimizer or merely better than an improperly-transferred one. The authors correctly identify this, but it should be elevated from a minor limitation to a central caveat on the results.

**Concern 8 (Section 5.2.1 — Meta-training ablation):** The ablation compares µLO_S (single width=128) to µLO_M (widths 128, 512, 1024). The conclusion that multiple-width meta-training improves generalization is reasonable, but the experiment does not control for the fact that µLO_M sees more *total compute* during meta-training (training on 3 widths vs. 1). Under a strict compute-matched comparison, the benefit of width diversity vs. compute budget is not separated. The paper's emphasis on "FLOP-matched" budgets elsewhere makes this inconsistency noticeable.

**Concern 9 (Width generalization results — Figure 3/4 reference confusion):** Section 5.2.3 ("Meta-generalization to wider networks") opens with "Figure 3 compares the training loss..." but Figure 3 as described in Section 5.2.1 is labeled as the meta-training task ablation on ImageNet-32. The description of subfigures (a)–(d) in 5.2.3 does not match the subfigure labels described for Figure 3 in 5.2.1. This is likely a figure numbering error (the results should refer to Figure 4), and while it may be a parsing artifact, it is also possible this reflects a genuine inconsistency in the paper that impedes reproducibility.

**Concern 10 (Depth generalization — Section 5.2.4):** The depth generalization result is presented as a key finding, but the paper offers only a vague hypothesis: "we hypothesize that µP's stabilizing effect on the optimizee's activations leads to this improvement." No ablation isolates whether this is due to (a) the µP initialization, (b) the FAN_IN update scaling, or (c) the multiple-width meta-training. Since the paper explicitly states there is no theoretical justification, at minimum an empirical ablation disentangling these components is needed to make this claim credible.

**Concern 11 (Horizon generalization — Section 5.2.4):** Similarly, generalizing to 25× the meta-training horizon is presented as a key finding. The authors are appropriately cautious ("we are uncertain of the exact cause"), but such a dramatic finding warrants more investigation. In particular, the comparison to SP LOs at 25,000 steps may be unfair if SP LOs diverge quickly due to optimizer instability that is architecturally unrelated to SP vs. µP — i.e., an SP LO with gradient clipping might also generalize. The comparison should include stability-focused SP baselines.

**Concern 12 (Scale of evaluation):** The main results in Table 1 are aggregated across only 5 tasks (3 MLP classification tasks, 1 ViT, 1 LM). While 5 seeds per task are used, a ranking over 6 optimizers on 5 tasks is statistically very thin — a single task where µLOs underperform could significantly change the average rank. No statistical significance test is applied to the rank differences.

---

### Writing & Clarity

**Concern 13:** There is a clear editing error at line 508: "While our main focus is meta-generalization to wider networks While the focus of our paper is..." — this is a duplicated/incomplete sentence in the body of Section 5.2.4 and suggests the paper was not fully proofread.

**Concern 14:** The paper describes "35 tasks spanning image classification and language modeling" in the evaluation suite (Section 5.1), but Table 1 and the main results cover only 5 tasks. The relationship between the 35 tasks and the 5 reported tasks is not well explained in the main body. Readers would benefit from a clearer explanation of which subset is used for the main results and why.

**Concern 15:** Standard error is defined in the caption as "≈ σ/√n where σ is the population standard deviation." This should be *sample* standard deviation, and the formula should be written unambiguously. This is a minor issue but reflects imprecise statistical language throughout.

---

### Limitations & Broader Impact (Section 6)

The authors acknowledge three limitations:
1. Meta-training only on MLP image classification tasks
2. No evaluation beyond width 8192/3072
3. No oracle per-width hyperparameter sweep

These are all legitimate and appropriately identified. However, the limitations section is very brief. Additional limitations worth acknowledging:
- **Assumption dependence:** The theoretical results depend on the LLN alignment assumption, which may not hold in all practical settings, especially for non-MLP architectures.
- **Training vs. generalization:** The paper only measures training loss; whether µLOs lead to better generalization of the trained model is unknown.
- **Single meta-training architecture family:** The meta-training uses only two LO architectures. It's unclear whether the µP derivation and training recipe would apply to more recent LO architectures.
- **Computational accessibility:** The paper claims results are obtainable in ~100 GPU hours, but this still requires non-trivial resources compared to simply running AdamW.

---

### Overall Assessment

This paper tackles a well-defined and practically important problem — the poor meta-generalization of learned optimizers to wider networks — with a principled theoretical tool (µP). The core idea of applying µP to learned optimizer architectures is elegant and well-motivated, and the empirical results showing that SP learned optimizers diverge at large widths while µLOs do not are compelling. The work is appropriately scoped and the code is open-sourced.

However, several concerns weaken the submission for ICLR. Most critically: (1) the theoretical guarantees in Propositions 4.1/4.2 rest on an unverified "LLN alignment" assumption that is neither empirically validated nor adequately explained in the main body; (2) all results are in terms of training loss, never test performance, which is a significant omission for a venue focused on learning representations; (3) the missing oracle per-width baseline makes it impossible to assess whether µLOs genuinely outperform strong hand-designed optimizers or merely outperform poorly-transferred ones; (4) the empirically observed depth and horizon generalization, while interesting, lacks any ablation to identify its cause. The editing error (duplicated sentence) and figure reference confusion further suggest the paper would benefit from another round of careful revision. The contribution is meaningful but currently sits below the clarity and rigor bar expected at ICLR — particularly in the theoretical treatment. With additional proofing of the theoretical assumptions, inclusion of at least one test-performance metric, and an ablation on the depth generalization finding, this paper could make a solid contribution to the learned optimization literature.

# Neutral Reviewer
## Balanced Review

### Summary
This paper derives the Maximal Update Parametrization ($\mu$P) for state-of-the-art Learned Optimizer (LO) architectures and proposes a meta-training recipe to address their poor meta-generalization to unseen network widths. Empirical evaluations demonstrate that $\mu$-parameterized learned optimizers ($\mu$LOs) substantially outperform standard parameterization baselines and per-task tuned hand-designed optimizers (e.g., AdamW, Adam) on wider, deeper, and longer-horizon tasks. The work effectively bridges stable parameterization theory with meta-learning for optimizers, offering a compute-efficient solution to a key bottleneck in learned optimization.

### Strengths
1.  **Novel Theoretical Extension:** The authors derive $\mu$P for specific LO architectures (VeLO and small_fc_lopt) and provide propositions validating the parameterization's adherence to $\mu$P desiderata (Section 4, Propositions 4.1 & 4.2). This connects a stable wide-limit theory with meta-learning optimizers, an area where it was previously unexplored.
2.  **Strong Empirical Validation on Generalization:** The paper provides an extensive evaluation suite (35 tasks including ViT, LM1B, and varied MLP widths) showing that $\mu$LOs generalize to widths far exceeding meta-training (up to 8192 for MLPs). Section 5.2.3 and Figure 4 show $\mu$LOs outperform per-task tuned AdamW and $\mu$Adam on large tasks, a significant claim given AdamW's typical dominance.
3.  **Reproducibility and Efficiency:** The authors open-source their code and explicitly match the compute budget against baselines (Abstract, Section 5.1). They also demonstrate zero extra computational cost compared to SP LOs (Section 7), making the improvement practical.
4.  **Stability Analysis:** Section 5.2.2 provides empirical evidence regarding pre-activation stability (Figure 2), linking the parameterization to theoretical stability goals (Coord-wise standard deviation of pre-activations), which is crucial for the validity of the meta-learning process.

### Weaknesses
1.  **Limited Meta-Training Distribution:** The meta-training distribution is restricted to MLPs for image classification (Section 5.1, Section 6). The authors admit in Section 6 that they do not meta-train on tasks other than MLPs, limiting the generality of the learned optimizer to architectures like Transformers during the training phase, despite testing them during evaluation.
2.  **Empirical Nature of Depth/Horizon Generalization:** While width generalization is theoretically supported by $\mu$P, the generalization to deeper networks and longer training horizons is acknowledged as "purely empirical" (Section 5.2.4, Section 7) with only hypothesized explanations regarding pre-activation stability. This lacks the theoretical rigor found in the width analysis.
3.  **Scalability Limits:** The evaluation is capped at specific width limits (e.g., 8192 for MLPs, 3072 for ViTs) due to computational constraints (Section 6). For a paper claiming breakthrough generalization to wide networks, testing up to industry-relevant scales (e.g., 65k+ widths) or discussing the theoretical limits under $\mu$P would strengthen the claims significantly.
4.  **Baseline Comparisons:** While per-task tuned AdamW/Adam is a strong baseline, Section 6 notes that an "oracle SP AdamW baseline whose hyperparameters are swept at every width" was excluded due to constraints. This leaves a slight gap in comparing against the absolute best possible tuned performance at each specific width.

### Novelty & Significance
The novelty of this work lies in successfully adapting Maximal Update Parametrization to the specific dynamics of meta-learned optimizers. While $\mu$P has been applied to standard training and SGD/Adam, its application to Learned Optimizers (which have complex internal momentum/variance structures) is a distinct contribution. The significance is high for the ICLR community; it addresses a persistent failure mode of Learned Optimizers (VeLO and others) regarding width generalization without increasing the meta-training compute budget. If validated, it provides a practical path to robust learned optimizers that do not require architecture-specific re-tuning.

### Suggestions for Improvement
1.  **Elaborate on Meta-Training Distribution Robustness:** The authors should discuss whether the MLP-only meta-training is a fundamental limitation or an artifact of compute constraints. A small ablation study adding one non-MLP architecture (e.g., a small ViT) to the meta-training suite would clarify if $\mu$P requires the underlying task distribution to remain homogeneous.
2.  **Deepen Theoretical Analysis for Depth:** To support the empirical claims on depth generalization (Section 5.2.4), a brief theoretical discussion or reference to tensor program literature regarding depth scaling under $\mu$P (even if non-residual) would strengthen the argument beyond hypothesis.
3.  **Clarify Budget Metrics:** While FLOPs are matched, ICLR reviewers often prefer a wall-clock time analysis or a breakdown of why the $\mu$-scaling does not impact training time for the LO itself but significantly improves the optimizee's training. A brief note on the overhead of applying $\mu$P scaling on the optimizee during meta-training would be valuable.
4.  **Expand Limitations Discussion:** Section 6 currently lists limitations as bullet points. This should be expanded to a subsection explaining how the current limitations prevent adoption in large-scale real-world scenarios (e.g., LLM training) and how future work plans to mitigate them (e.g., via the "CompleteP" or other parameterizations mentioned).

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Oracle Hand-Designed Baselines at Target Widths:** Evaluate AdamW/$\mu$Adam with hyperparameters tuned *directly* on the largest test widths (e.g., 8192). The current comparison against width-transferred baselines weakens the claim that $\mu$LOs outperform hand-designed optimizers, as transferred hyperparameters are known to be suboptimal compared to in-distribution tuning.
2. **Cumulative Compute Budget Accounting:** Quantify the total FLOPs/hours for baseline hyperparameter tuning (500 configs $\times$ 35 tasks) vs. $\mu$LO meta-training. The "compute-efficient" claim is ambiguous without comparing the total cost of ownership for the end-user versus the researcher.
3. **Ablation of $\mu$P vs. Multiple-Width Recipe:** Compare $\mu$LO (single-width meta-training) against SP LO (multiple-width meta-training). This isolates whether performance gains come from the parametrization theory or simply from exposing the LO to width variations during meta-training.

### Deeper Analysis Needed (top 3-5 only)
1. **Theoretical Justification for Depth Generalization:** $\mu$P theory covers width, yet the paper claims depth generalization (3→16 layers). Provide an analysis of gradient signal propagation vs. depth to explain this empirically observed but theoretically unsupported benefit.
2. **Parameter Semantics Across Architectures:** Explain how LOs meta-trained on MLP weights generalize to Transformer parameters (e.g., attention scores vs. FC weights). The input feature semantics differ significantly between architectures, risking silent failure modes that current metrics do not capture.

### Visualizations & Case Studies
1. **LO Input Feature Distributions (MLP vs. ViT):** Visualize the distribution of gradients and moments fed into the LO for MLP vs. ViT tasks. This exposes whether the LO is truly handling out-of-distribution input features during architecture transfer or if the features align unexpectedly.
2. **Instability Trajectories:** Plot the evolution of parameter norms and gradient norms during divergence for SP LOs vs. $\mu$LOs. This confirms whether $\mu$P prevents explosion specifically or just delays it, validating the claimed stability benefits.

### Obvious Next Steps
1. **Meta-Train on Mixed Architectures:** Include ViT/Transformer tasks in the meta-training distribution to verify if MLP-only training is a bottleneck or sufficient for architecture transfer.
2. **Integrate Depth-Specific Parameterizations:** Test combinations with Depth-$\mu$P or CompleteP to see if the unexplained depth generalization can be theoretically grounded and improved within this work.

# Final Consolidated Review
## Summary
The paper derives the Maximal Update Parametrization (µP) for two state-of-the-art learned optimizer architectures (VeLO and small_fc_lopt) and proposes a meta-training recipe that enables learned optimizers to generalize to wider networks far beyond those seen during meta-training. The core contribution connects µP's hyperparameter transfer mechanism to the meta-generalization problem in learned optimizers, yielding empirical improvements at zero additional computational cost.

## Strengths
- **Novel theoretical extension:** The derivation of µP for learned optimizer architectures (Propositions 4.1 and 4.2) connects stable parameterization theory with meta-learned optimizers for the first time. The paper provides formal propositions establishing that the proposed parameterization satisfies µP desiderata under specific assumptions.
- **Strong empirical demonstration of width generalization:** µLOs meta-trained on widths 128/512/1024 successfully optimize networks up to width 8192 (MLPs) and 4096 (Transformers/ViT), substantially outperforming SP LO baselines and per-task tuned AdamW/µAdam (Table 1, Figure 4). The consistency of improvements across tasks is notable.
- **Compute efficiency and reproducibility:** The approach requires zero additional computational cost compared to SP LOs. The authors open-source code and report meta-training in ~100 GPU hours, making results accessible for academic research.

## Weaknesses
- **Theoretical assumptions require more justification:** Propositions 4.1 and 4.2 depend on an "LLN alignment" assumption that "during training the optimizee's parameters and input data become aligned." The paper does not formally define what "aligned" means in this context, provide empirical validation that this assumption holds during training, or discuss failure modes. For theoretically motivated claims, this gap is significant—readers cannot assess whether the assumptions hold for their architectures.
- **All evaluations report training loss only:** The paper never reports test/validation loss or accuracy. For optimizers intended to train models that generalize, this is a meaningful gap—an optimizer that overfits aggressively could appear to perform well on training loss while producing models with poor test performance.
- **Missing oracle baseline limits conclusions:** As acknowledged in limitations, the paper lacks an AdamW baseline tuned separately at each target width. Without this comparison, we cannot determine whether µLOs genuinely outperform well-tuned hand-designed optimizers or merely outperform hand-designed optimizers with transferred (potentially suboptimal) hyperparameters.
- **Depth and horizon generalization lack causal attribution:** The paper presents generalization to deeper networks (3→16 layers) and longer horizons (1000→25000 steps) as key findings, but acknowledges these are "purely empirical" (Section 5.2.4). No ablation isolates whether these benefits arise from µP initialization, FAN_IN scaling, or multiple-width meta-training. The hypothesis about pre-activation stability remains untested.
- **Limited statistical support:** Results aggregate over 5 tasks with ranking comparisons across 6 optimizers. No statistical significance tests are applied to the ranking differences, limiting confidence in the robustness of conclusions.

## Nice-to-Haves
- An ablation comparing SP LO with multiple-width meta-training against µLO with single-width meta-training (compute-matched) would clarify whether performance gains stem from parametrization theory or task diversity.
- Test loss or accuracy metrics for at least one downstream task would establish that improved optimization transfers to model generalization.
- Brief empirical validation or discussion of when the LLN alignment assumption holds (e.g., showing scaling behavior of LO input features during training).

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Editing error nitpick:** The duplicated sentence "While our main focus is meta-generalization to wider networks While the focus of our paper is..." (Section 5.2.4) is a proofreading issue that does not affect technical content.
- **Standard error formula notation:** The minor technical point about population vs. sample standard deviation in the error bar definition.
- **Request for gradient-clipped SP LO baseline:** Asking for additional baseline variants (e.g., SP LO with gradient clipping) is scope creep beyond the paper's stated contributions.
- **Concern about LO internal weights:** The learned optimizer is meta-trained once and applied; its internal weights do not scale with optimizee width during deployment, making this question less relevant.
- **35 vs 5 tasks explanation:** The paper explains the relationship (35 tasks total, main results on 5 representative tasks); the critic's concern is adequately addressed.

## Novel Insights
An interesting tension emerges: the paper's strongest empirical findings—generalization to deeper networks and longer horizons—are precisely where µP theory offers no guidance. The authors hypothesize that µP's pre-activation stability explains these benefits, but this remains untested. A potentially insightful experiment would be to visualize the learned optimizer's input feature distributions (gradient magnitudes, momentum statistics) across MLP vs. ViT vs. Transformer architectures during training. This could reveal whether the MLP-only meta-training implicitly normalizes feature statistics across architectures, or whether µP induces deeper invariances in how gradient features scale.

## Suggestions
- Report at least one test loss or accuracy metric for the main evaluation suite to establish that optimization improvements do not come at the cost of model generalization.
- Add an ablation isolating the source of depth generalization: train µLO with single-width meta-training (matching total compute to multi-width) to disentangle parametrization effects from task diversity effects.
- In the theoretical section, either provide empirical validation that LLN alignment holds for the input features of small_fc_lopt and VeLO during training, or add a brief discussion of expected failure modes.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 2.0]
Average score: 5.0
Binary outcome: Accept
