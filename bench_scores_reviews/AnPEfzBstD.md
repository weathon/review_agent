## Summary
This paper presents a large-scale empirical benchmark comparing 1D (MolFormer/SMILES), 2D (MPNN), and 3D (Equiformer v2) molecular representations in Bayesian Optimization (BO) for molecular property optimization. Across four datasets (QM7, QM9, GEOM MoleculeNet, GEOM DRUGS) and 2100+ runs, the study reports that 1D/2D representations generally outperform 3D features in BO loops, with LLM-based features performing best in most settings, and that 3D models require substantially more data to match 2D performance.

---

## Strengths

- **Benchmark scale with statistical robustness.** 2100+ distinct runs across four datasets with 15 seeds per configuration, producing mean ± standard error optimization curves. Most BO benchmarks use far fewer seeds; the aggregate statistical robustness here is a genuine strength.
- **Sample complexity analysis (Section 5.2) is a specific, useful contribution.** The systematic sweep over N = 500, 1000, 10,000, 50,000 training samples, showing that 3D models fail to match 2D performance until data exceeds 10,000 observations even on QM7, is actionable for practitioners. This directly ties BO's inherently low-data regime to 3D models' poor BO performance — a connection not made explicitly in prior benchmarks.
- **Multi-surrogate evaluation.** Testing both GP and linearized Laplace approximation (LLA) surrogates across all representation types meaningfully extends prior work, which typically fixes one surrogate.

---

## Weaknesses

### Fatal
None that would immediately invalidate the entire paper, but the major pretraining confound below severely undermines the primary conclusion.

### Major

- **Pretraining confound undermines the central 1D vs. 2D vs. 3D conclusion.** MolFormer is described as a masked language model *pretrained on large-scale chemical databases*, while MPNN and Equiformer v2 are explicitly stated to be trained from scratch with ~1.5M parameters (Section 4). The paper's headline conclusion — "1D representations outperform 3D" — is therefore better described as "a large pretrained model outperforms non-pretrained models of fixed parameter count." These two claims have very different practical and scientific implications. The paper does not include any ablation that disentangles model scale and pretraining from representation dimensionality (e.g., pretrained 3D models, or scratch-trained LLMs). This confound makes the central conclusion insufficiently supported and potentially misleading.

- **3D conformer generation workflow is undocumented.** The paper never specifies how 3D coordinates are obtained for molecules in the virtual library or for BO-proposed candidates. QM7 and QM9 include DFT-optimized geometries, but this is not stated. More critically, the workflow for how 3D structures are provided to the surrogate in the BO loop is never described (Figure 1 shows "Oracle Call (DFT)" but this returns energies, not geometries). If candidate molecules are proposed by the acquisition function but 3D geometries must be generated via cheaper methods (e.g., RDKit ETKDG) for the surrogate input, the quality mismatch between surrogate inputs and DFT-optimized reference geometries would systematically penalize 3D models regardless of their true representational power. This is a direct threat to the validity of the 3D-vs-2D comparison.

- **Abstract's "consistently outperform" claim contradicts the QM9 results.** The abstract states "LLM methods consistently outperform," but Section 5.1 explicitly states: "Contrary to all other datasets, LLMs performed worse than 2D and 3D models" on QM9. An inconsistency between the abstract and a key result of the benchmark is a clarity/honesty problem that must be corrected.

- **Generalizing from one 3D architecture.** Only Equiformer v2 represents the "3D" category. The paper's conclusion that "3D features underperform" is generalized from the behavior of a single architecture. Equiformer v2 is a strong representative, but failure could be specific to its optimization behavior, conformer sensitivity, or training regime. Reporting results for even one additional 3D model (e.g., SchNet or PaiNN, which differ in symmetry assumptions and data requirements) would substantially strengthen the generalizability of the claim.

### Minor

- **No quantification of claimed computational overhead.** The paper repeatedly cites "computational overhead" of 3D models as a key trade-off but provides no wall-clock times, FLOPs, or memory measurements. The efficiency claim is central to the practical recommendation ("1D is a strong default"), yet it rests on a qualitative assertion. Even a simple table of feature extraction time per molecule would make this concrete.

- **Random search ≈ GP regression on GEOM DRUGS.** Section 5.1 notes this directly but does not diagnose it. When a BO surrogate performs no better than uniform random sampling on a large-molecule dataset, it suggests surrogate miscalibration, a pathological search landscape, or a poorly structured optimization task — not just "larger models would help." This undermines the validity of the BO comparison on this dataset.

- **Figure 2 aggregation obscures dataset-level contradictions.** Aggregating GAP curves across all datasets, surrogates, and seeds into a single 1D/2D/3D curve hides the fact that on QM9, 2D > LLM, while on other datasets LLM >> 2D. The aggregate narrative ("1D always best") does not hold within individual datasets and should not be the headline figure without this caveat.

- **Transfer learning section (5.3) only covers QM7 and QM9.** The claim "foundation models prove a good tool to leverage in molecular optimization" is made broadly, but Fig. 5 only covers two datasets. Conclusions about scalability to GEOM DRUGS are not supported.

### Tiny

- The framing "empirically answer *why* 3D features are underused" in the abstract and contributions overstates what is shown. The paper characterizes *when* and *whether* 3D features help in BO; isolating causal mechanisms would require controlled ablations. The paper could simply say "empirically characterize" rather than "explain why."
- The acquisition function used in the BO loop is never stated. While BoTorch is cited (implying EI or UCB defaults), this should be stated explicitly for reproducibility.

---

## Nice-to-Haves

- **Evaluate on at least one genuinely 3D-dependent task** (e.g., conformer energy ranking, chiral discrimination, or docking score). The tasks used (atomization energy, HOMO-LUMO gap, absolute energy) are largely topology-determined, and the paper acknowledges this gap in its conclusion. Including even one such task would allow the conclusions to be appropriately scoped rather than relying on the limitation disclaimer.
- **Performance vs. compute Pareto frontier.** A single plot showing GAP achieved vs. average feature extraction time per iteration would make the efficiency-accuracy trade-off concrete and directly support the practical recommendation.
- **Uncertainty calibration analysis per representation.** Since BO critically depends on surrogate uncertainty quality (not just point predictions), calibration plots (e.g., ECE or coverage curves) would reveal whether 3D models fail due to poor predictions or poor uncertainty — information relevant to acquisition function behavior.
- **Sensitivity of 3D results to conformer quality.** Testing whether using DFT-optimized vs. RDKit-generated conformers as 3D inputs changes the relative ranking of 2D vs. 3D methods would address a genuine confound and help readers understand robustness.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic] Demand for theoretical/causal analysis.** The paper is explicitly an empirical benchmark; demanding theoretical proofs or principled causal decompositions is not appropriate for this paper type and community standard.
- **[Harsh Critic] Criticism of the discrete vs. continuous BO formulation.** Virtual library BO (finite candidate pools) is standard for molecular benchmarking; criticizing this choice as a deficiency is scope creep.
- **[Harsh Critic] Overly specific demands on Laplace approximation implementation details** (diagonal vs. Kronecker Hessian, prior specification). While more detail would be appreciated, absent evidence of misconfiguration this is not a substantive weakness.
- **[Harsh Critic] Fingerprint/1D confusion in Figure 2.** The paper maintains a consistent taxonomy (GP with fingerprints = baseline, MolFormer = 1D, MPNN = 2D, Equiformer = 3D); while clarity could be improved, this does not constitute a factual error.
- **[Positive Reviewer Strength] "Reproducibility efforts" is generic** and applies to nearly all papers providing code; not a distinctive strength for this paper.
- **[Harsh Critic] Section 2.1 BO formulation is standard** — criticism of it as "not discussing discrete BO" is a minor scope issue, not a real weakness, since discrete virtual library BO is well-established.

---

## Novel Insights

The genuinely novel observation that emerges is the *interaction between BO's structural data scarcity and 3D models' higher sample requirements*. Equivariant 3D models are known to need more data to converge in supervised learning, but BO by design operates in a regime of extreme data scarcity (here, starting from 10 observations and acquiring hundreds). The sample complexity section makes this concrete: the performance gap between 2D and 3D is largest at N=500–1,000, which is precisely the regime where BO adds most value. This suggests that 3D models may be fundamentally mismatched to the BO setting not because of representation quality but because of the learning dynamics of equivariant architectures in the sequential low-data regime. However, this insight is clouded by the pretraining confound — it remains unclear whether the data hunger observed belongs to equivariant architectures generically or to non-pretrained 3D models specifically.

---

## Suggestions

1. **Add a pretrained 3D model baseline** (e.g., Uni-Mol, which is pretrained on 3D molecular conformers) and/or train MolFormer from scratch, to disentangle pretraining advantage from representation dimensionality. This is the single most impactful fix.
2. **Document the 3D conformer pipeline explicitly**: specify the source of 3D coordinates in the virtual library for each dataset, and clarify what geometry is fed to the 3D surrogate for BO-proposed candidates.
3. **Correct the abstract**: change "consistently outperform" to accurately reflect that LLMs underperform on QM9, and qualify the conclusion appropriately.
4. **State the acquisition function** used and whether surrogates are retrained at each BO step.
5. **Diagnose the random search ≈ GP result on DRUGS**: report whether calibration fails on this dataset, or whether the task structure (many similar energies) makes BO uninformative. This would either reveal a limitation of the evaluation or motivate a fix.
6. **Extend the transfer learning analysis** to GEOM MoleculeNet and DRUGS before drawing general conclusions about foundation models in BO.

---

**Overall assessment across axes:**
- *Novelty*: Moderate. The specific BO-loop framing for comparing molecular representation dimensions is novel, but the central conclusion is severely confounded by pretraining asymmetry.
- *Technical soundness*: Weak-to-moderate. The experimental scale and surrogate coverage are commendable, but the pretraining confound and undocumented conformer workflow are serious methodological issues.
- *Empirical support*: Moderate. The scale (2100+ runs, 15 seeds) is a genuine strength, but key results (Figure 2 aggregation) obscure internal contradictions.
- *Significance*: Moderate if the confounds are resolved; currently limited by the gap between the stated conclusion and what the experiments actually show.
- *Clarity*: Below expectations for ICLR — the abstract-QM9 inconsistency, vague conformer protocol, and unreported acquisition function are concrete gaps.