Now let me carefully verify the key criticisms:

1. **Theoretical derivation conflating macrostate probabilities with point evaluations**: Let me check this carefully. Eq. 1 says ΔG = -kT(log p_bind - log p_unbind), with "the partition function eliminated by the two-state comparison." The text says p_bind and p_unbind represent "the probabilities of the protein complex structure being in the bound conformation X_bind and the unbound conformation X_unbind." In statistical thermodynamics, the correct expression is ΔG = -kT log(Q_bound/Q_unbound), where Q is the partition function (sum over all microstates). The paper then equates these macrostate probabilities with p(X_bind | S_AB), a conditional density evaluated at specific coordinates. This is indeed a conceptual issue, though the paper does acknowledge it as an "approximation" (they say "we simplify and approximate X_bind and X_unbind as the backbone structure"). However, it's not just about simplifying X to backbone - it's about substituting a macrostate probability (ratio of partition functions) with a conditional density evaluated at a single point. The paper's text on line 75 does say "where we simplify and approximate X_bind and X_unbind as the backbone structure of the protein complex" which acknowledges approximation. But the core theoretical claim of "physical inductive bias" rests on this derivation being physically grounded, when actually the step from macrostate probability ratios to conditional density evaluations is a significant leap. This is a real but not fatal concern - the method works empirically and the derivation provides motivation even if not a rigorous proof.

2. **No variance reported**: Confirmed - Table headers say "Mean results of 3-fold cross-validation" and no standard deviations are shown anywhere. This is a valid concern.

3. **ΔG prediction (Section 4.4.1, Eq. 13) lacks theoretical justification**: The paper itself acknowledges "we lack a comprehensive explanation for canceling out p(X_bnd) and p(X_unbnd) in Eq.5" (line 235). This is a valid concern, and the paper is partially transparent about it.

4. **The DPO comparison**: Looking at the numbers, DPO performs substantially worse (per-structure Spearman 0.3913 vs SFT 0.4769 and Boltzmann 0.5134). The harsh critic raises concern about the DPO implementation, but the paper references appendix A.1 for implementation details.

5. **The macrostate issue**: This is the most significant theoretical concern. Let me assess its severity carefully.

Now let me write the final consolidated review.</think>Now I have thoroughly read the paper and calibration anchors. Let me compose the final review.

## Summary

The paper proposes Boltzmann Alignment, a technique to transfer knowledge from pre-trained inverse folding models to ΔΔG (change in binding free energy) prediction for protein-protein interactions. The key idea is deriving a connection between ΔΔG and sequence log-likelihoods via the Boltzmann distribution and thermodynamic cycle, then using this to construct both an unsupervised estimator (BA-Cycle) and a supervised fine-tuning method (BA-DDG). The method achieves state-of-the-art results on SKEMPI v2, with Spearman coefficients of 0.3201 (unsupervised) and 0.5134 (supervised), and is demonstrated on binding energy prediction, docking, and antibody optimization tasks.

## Strengths

- **The thermodynamic cycle insight—explicitly accounting for the unbound state—is a genuine and well-motivated improvement over prior inverse folding-based methods.** Tables 1 and 2 confirm this: BA-Cycle (overall Spearman 0.4097) substantially outperforms ProteinMPNN (0.3112) and ESM-IF (0.2806), which only consider bound-state likelihoods. Subtracting unbound-state likelihoods is physically principled at the level of modeling intuition and produces measurable empirical gains.

- **Strong empirical results across all evaluation metrics on SKEMPI v2.** BA-DDG outperforms all baselines across all 7 metrics in Table 1 (per-structure Spearman 0.5134 vs. prior SoTA 0.4324; overall Spearman 0.6346 vs. 0.5910). The margins are substantial and consistent.

- **Per-structure evaluation metrics** are reported alongside overall metrics (Table 1, Fig. 3), which is important because overall correlations can be dominated by a few large structures. This provides a more practical assessment of prediction quality.

- **The unsupervised BA-Cycle method provides a training-free baseline with meaningful performance** (overall Spearman 0.4097), useful when no labeled ΔΔG data is available. Table 2 ablates the thermodynamic cycle contribution clearly.

- **Robustness to predicted structures:** Table 4 shows BA-DDG performs comparably with AlphaFold3-predicted structures vs. crystal structures, demonstrating practical applicability.

## Weaknesses

### Fatal

None.

### Major

- **The theoretical derivation from Boltzmann macrostate probabilities to point evaluations of conditional densities is an approximation presented as a physical derivation, overclaiming the "physical inductive bias."** In statistical thermodynamics, the free energy ΔG = −kT log(Z_bound/Z_unbound) involves partition functions—integrals over all microstates. The paper (Eq. 1→2) substitutes p_bind = p(X_bind | S), treating macrostate probabilities as conditional density evaluations at a single backbone structure. While the paper acknowledges the rigid-backbone approximation and structural simplification, the more fundamental conflation—replacing a ratio of partition functions with a ratio of conditional densities evaluated at specific coordinates—is not discussed. The paper frames the entire contribution as introducing principled "physical inductive bias" through this derivation. If the derivation serves primarily as motivation for a heuristic that works well empirically, this should be stated plainly. The empirical gains are real, but the theoretical novelty as presented is overstated.

- **No variance or statistical significance is reported for any experiment (Tables 1–5).** All results are means over 3-fold cross-validation with no standard deviations. Given SKEMPI v2 has only 348 complexes, fold-level variance can be substantial. For example, SFT and Boltzmann Alignment achieve nearly identical AUROC (0.7725 vs. 0.7726) in Table 3, and the thermodynamic cycle ablation (Table 2) hinges on Spearman gaps (ProteinMPNN 0.2741 vs. BA-Cycle 0.3201). Without variance estimates, it is impossible to assess whether these differences are meaningful. This is particularly critical for the paper's claim of "significantly surpassing" prior SoTA.

### Minor

- **The ΔG prediction extension (Section 4.4.1, Eq. 13) drops the p(X) terms without explanation.** The paper acknowledges this: "Although we lack a comprehensive explanation for canceling out p(X_bnd) and p(X_unbnd) in Eq. 5." Presenting this as a demonstrated application without any theoretical justification or ablation of what happens when these terms are dropped is misleading about scope, though the paper is transparent about the limitation.

- **The DPO baseline implementation details are deferred to the appendix, and DPO performs dramatically worse (per-structure Spearman 0.3913 vs. SFT 0.4769).** The paper does not discuss why DPO fails so severely on this regression task, which would strengthen the comparison. Since DPO is designed for preference-based rather than regression settings, this may be an unfair comparison that nonetheless does not undermine the main result.

- **The unbound state uses bound-state backbone coordinates (the same X_A, X_B extracted from the complex), not actual unbound structures.** The paper notes this is because unbound structures are "typically not explicitly provided," but this conflates bound-state structural features with unbound-state likelihood calculations. The paper acknowledges this limitation in Section 5, but the degree to which the thermodynamic cycle improvement comes from correctly modeling unbound energetics vs. simply adding a regularization term remains unclear.

### Trivial

- The abstract claims "significantly surpassing" prior results, but without statistical tests, "surpassing" would be more accurate.

## Nice-to-Haves

- Report standard deviations across folds to support significance claims.
- Ablate the KL penalty coefficient β and the learnable k_BT parameter to quantify their individual contributions.
- Analyze performance by mutation type (single-point vs. multi-point) to characterize when the rigid-backbone assumption breaks down.
- Compare against a simple baseline that subtracts individual-chain ProteinMPNN likelihoods without the Boltzmann framing, to disentangle the thermodynamic cycle contribution from the architectural benefit of including unbound terms.

## Removed Points

- **"Macrostate vs. point evaluation" as a Fatal flaw:** The harsh critic characterized this as a fundamental conceptual error that invalidates the paper's core claim. While the theoretical derivation does have a genuine gap (macrostate probabilities ≠ conditional densities at single points), the paper does acknowledge approximation steps, and the method's empirical success demonstrates that the resulting formula is useful regardless of the theoretical framing. Downgraded to Major. *(Note: This concern remains in the main review as a Major weakness, just not Fatal.)*

- **Demand for comparison against methods using unbound-state corrections without Boltzmann framing:** Partially kept as Nice-to-Have, as this would strengthen but is not required. The paper already compares against relevant baselines.

- **Demand for actual unbound crystal structures:** This is outside scope; the paper explicitly uses bound structures as a practical approximation and discusses this limitation.

- **Criticism that DPO comparison may be unfair (favors the proposed method):** The harsh critic suggested DPO may be poorly adapted, which would make the comparison unfair in a direction that *favors* the authors. Per the hard rules, criticisms about unfair comparisons that favor the baseline are removed; but this comparison actually favors the proposed method. However, DPO's poor fit for regression tasks is a known concern, so the severity is reduced—the comparison still has value, it just should not be over-interpreted. Kept as Minor.

- **Missing ablation on β (KL penalty)** and **analysis of learnable k_BT:** These are valid suggestions but are Nice-to-Haves rather than Major weaknesses—standard practice does not require ablations of every hyperparameter.

- **Nitpick about the abstract emphasizing per-structure Spearman over overall Spearman:** The per-structure metric is actually the more practically relevant one, as the paper argues. This is not a misleading choice.

## Novel Insights

The fundamental observation—that existing inverse folding methods correlate with ΔΔG implicitly because log p(S|X) encodes structural compatibility, and that this relationship can be made explicit by adding the thermodynamic cycle correction for the unbound state—is both correct and practically significant regardless of the rigor of the Boltzmann derivation. The empirical evidence (BA-Cycle outperforming ProteinMPNN by a large margin) suggests that the unbound-state correction captures real physical information, and the improvement from BA-DDG over SFT shows that aligning the fine-tuning objective with the final prediction formula is beneficial. However, the paper would be stronger if it presented the derivation as a motivated approximation rather than a derivation from first principles, since the gap from partition functions to point densities is not merely a simplification but a conceptual shift.

## Suggestions

- Reframe the theoretical contribution honestly: present Eqs. 1–8 as a *motivation* inspired by the Boltzmann distribution, explicitly noting that p_bind/p_unbind in thermodynamics are macrostate probabilities (partition function ratios), while the method uses conditional densities at specific structures as a computationally tractable proxy. This preserves the intellectual contribution without overselling the physical grounding.
- Report standard deviations across 3-fold cross-validation for all tables, or at minimum for the key comparisons (Table 2, Table 3).
- Add a discussion of when and why the rigid-backbone assumption fails (multi-point mutations causing backbone rearrangements), ideally with per-mutation-type performance breakdowns.

## Calibration Anchor Comparison

- **High anchor: PSiijdQjNU (ProfileBFN, avg 7.5, Accept Oral):** Strong mathematical derivation with impressive benchmarks. Compared to this paper, ProfileBFN has cleaner theoretical framing and reviewers found its math sound. BA-DDG's theoretical contribution is less rigorous.
- **Medium anchor: IxmWIkcKs5 (Light-DDG, avg 6.75, Accept Poster):** Similar domain (ΔΔG prediction), similar claim structure (SoTA on SKEMPI). BA-DDG has a more novel methodological angle (thermodynamic cycle) but weaker theoretical justification; Light-DDG is more straightforward. Both lack variance reporting, which reviewers in that domain tolerate.
- **Medium anchor: uMAujpVi9m (pocket pretraining, avg 5.25-6.0, Reject):** Self-supervised pretraining on protein structures with moderate SoTA improvements. BA-DDG's improvements are larger and more consistent.
- **Low anchor: kvCKoKfqTd (NCGAMI, avg 3.0, Reject):** Overclaimed "groundbreaking" framework invoking thermodynamic connections with weak theoretical grounding and reviewers flagging overclaims. BA-DDG has empirical substance that NCGAMI lacks, but shares the pattern of dressing up a heuristic in thermodynamic formalism.
- **Low anchor: kKXIYUi8ff (DynamicsDiffusion, avg 3.0, Reject):** Questioned claims about Boltzmann distribution sampling; lack of scalability evidence. BA-DDG has much stronger experimental evidence and practical utility.

BA-DDG sits above the medium anchors empirically but below them theoretically: the method demonstrably works well on a standard benchmark with consistent improvements, which the medium-scored papers also achieve, but the theoretical overclaim pulls it down. Relative to Light-DDG (6.75), BA-DDG has more methodological novelty but weaker theoretical grounding. The paper's core empirical contribution (unbound state subtraction + alignment) is sound and significant; the theoretical framing issue is real but not fatal.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>