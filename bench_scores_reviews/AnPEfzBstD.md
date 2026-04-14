## Summary

This paper presents a large-scale benchmark (2100+ runs across 4 datasets and 35 setups) comparing 1D, 2D, and 3D molecular representations within a Bayesian optimization (BO) framework for materials discovery. Using MolFormer (1D/LLM), MPNN (2D), and Equiformer v2 (3D) combined with GP and Laplace-approximation surrogates, the paper finds that simpler representations generally match or outperform 3D equivariant GNNs, that 3D models require substantially more training data to be competitive, and that transfer learning is a viable strategy across modalities. The practical takeaway is that 1D and 2D representations offer a better cost-performance tradeoff for the surveyed tasks.

---

## Strengths

- **Sample-complexity analysis (Section 5.2) is a concrete and novel contribution.** Systematically varying training-set size from 500 to 50,000 and demonstrating that 3D models consistently lag 2D models at low-data regimes—while the gap narrows at ≥10,000 samples—directly operationalizes a mechanistic reason for 3D's underperformance in BO settings and advances beyond prior benchmarks that simply report end-to-end BO curves.

- **Benchmark breadth and statistical rigor are above average for this class of paper.** Four datasets spanning three orders of magnitude in size (QM7 ~7K to GEOM DRUGS ~318K), two surrogate families, and 15 random seeds with reported standard errors represent genuine experimental investment. The use of the normalized GAP metric to enable cross-dataset aggregation is sensible.

- **Inclusion of a transfer-learning condition alongside single-property prediction** adds a useful and underexplored dimension to BO benchmarking, showing that multitask pretraining can approach task-specific training quality—a practically relevant finding for practitioners who cannot afford extensive labeled data for every target property.

---

## Weaknesses

### Fatal
None.

### Major

- **Unmatched model scale between MolFormer and the GNNs fundamentally confounds the 1D-vs-3D conclusion.** The paper constrains MPNN and Equiformer v2 to ~1.5M parameters trained on QM9, while MolFormer is a pretrained masked language model trained on orders-of-magnitude more data. The headline finding "LLMs consistently outperform 2D and 3D models" cannot be attributed to representation dimensionality; it may simply reflect that a heavily pretrained large model beats smaller, less-pretrained models. Without a comparably pretrained 3D foundation model (e.g., Uni-Mol, pretrained SchNet/DimeNet) or a size-matched non-pretrained SMILES model as ablation, the 1D-vs-higher-dimensional comparison is uninterpretable for the paper's stated research question.

- **The chosen target properties are largely topology-determined, making the claim that "3D features are not useful" severely underscoped.** Atomization energy (QM7), HOMO-LUMO gap (QM9), and absolute energy (GEOM) are properties well-predicted from 2D graph topology. Tasks where 3D is genuinely differentiating—stereo-isomer discrimination, protein–ligand binding affinity, conformer-dependent solvation energy, or reaction selectivity—are absent. The paper's general-sounding conclusion ("3D is not useful") only holds for the specific properties tested, and the text does not adequately qualify this. The practical recommendation to chemists is therefore misleading for any workflow where 3D geometry is intrinsically necessary.

- **Acquisition function is never specified, yet is a critical BO hyperparameter.** Sections 2.1 and 4 introduce BO and describe the BO loop (including an "acquisition function" box in Fig. 1) but never state whether EI, UCB, Thompson sampling, or greedy selection was used. Different acquisition functions impose different exploration–exploitation regimes and may interact asymmetrically with representation quality and uncertainty calibration. This omission makes the results neither fully reproducible nor interpretable from the manuscript alone.

- **Cost-benefit claims are stated without any empirical cost data.** The paper repeatedly claims that 3D's "computational overhead outweighs predictive performance" and uses this as a primary reason for practitioners to avoid 3D. However, no wall-clock times, GPU hours, inference latency, or memory measurements are reported anywhere. Without these numbers, the cost-benefit argument is assertion, not evidence.

- **The conformer selection protocol for 3D inputs is undescribed.** GEOM provides multiple conformers per molecule; the paper never states which conformer is selected, how (lowest energy, random, RDKit-generated), or whether quality was checked. Since 3D model performance is known to be sensitive to conformer quality, using suboptimal conformers would artificially degrade 3D performance—directly undercutting the validity of the comparison. This is especially important for GEOM DRUGS, which the paper cites as a dataset emphasizing conformational flexibility.

- **Internal inconsistency between Section 5.1 and the Conclusion regarding QM9/LLM performance.** Section 5.1 explicitly states: "Contrary to all other datasets, LLMs performed worse than 2D and 3D models" for QM9. The Conclusion states: "Across all datasets examined LLMs consistently outperformed both 2D and 3D models." These are mutually contradictory. The Conclusion also contains a likely typo: the QM9 explanation reads "the task may have been the most dependent on information not captured by 2D and 3D representations," which should read "not captured by 1D representations" given the context. These inconsistencies undermine the reliability of the narrative synthesis.

### Minor

- **Section 5.2 (sample complexity) omits the LLM/1D baseline.** The 2D-vs-3D comparison is shown for varying training sizes, but since the paper's headline claim involves 1D/LLM dominance, excluding MolFormer from this analysis leaves unanswered whether the LLM's advantage is robust at low data regimes or only emerges once it has sufficient fine-tuning data. This is a material gap given the stated research question.

- **Transfer learning results are restricted to QM7 and QM9** (as confirmed by Fig. 5's caption), omitting MoleculeNet and GEOM DRUGS—the larger, more complex datasets where generalization across properties is presumably more valuable. The claim "foundation models prove a good tool to leverage in molecular optimization" is unsupported at the scale where it would most matter.

- **Training convergence of 3D models is not verified.** Equivariant GNNs like Equiformer v2 are known to be more difficult to optimize than MPNNs, and constraining both to ~1.5M parameters does not guarantee comparable training quality. Training loss curves or validation metrics confirming convergence for all modalities at each data regime are absent, leaving open whether 3D gaps reflect representation limits or optimization failures.

- **The experimental setup section is ambiguous about whether feature extractors are trained separately per dataset.** The text says "The models were trained on QM9," but BO is also run on QM7, MoleculeNet, and DRUGS. It is unclear whether 3D/2D models trained solely on QM9 are applied to other datasets (introducing domain shift) or whether separate training occurs per dataset. The current wording is inconsistent, and this matters for interpreting cross-dataset comparisons.

### Tiny

- The GAP metric is defined using both $y^*$ and $y_*$ inconsistently in the notation.
- The paper lacks a dedicated Limitations section, which would clarify the offline virtual-library setting versus true closed-loop experimental BO and the scope of the 3D conclusions.
- The sentence in Section 5.1 (QM9 paragraph) citing "information not captured by 2D and 3D representations" when the intended referent is clearly "1D representations" needs correction.

---

## Nice-to-Haves

- **Uncertainty calibration analysis (reliability diagrams) per model type.** BO acquisition quality depends jointly on predictive accuracy and calibrated uncertainty. Demonstrating whether 3D models produce well-calibrated uncertainty would clarify whether the 3D gap is fixable via better calibration methods or reflects a deeper representational mismatch.
- **Ablation separating pretraining from architecture for the LLM.** A non-pretrained transformer operating on SMILES (same architecture as MolFormer but trained only on QM9) would allow the paper to decompose "1D representation advantage" from "massive pretraining advantage."
- **Low-budget BO regime analysis (first 50–100 steps highlighted).** For expensive-oracle settings that motivate BO in the first place, the early optimization phase is most relevant; the main figures currently emphasize 1000-step convergence.
- **Correlation of 3D model relative performance with molecular characteristics** (conformational flexibility, rotatable bond count) to make the "when does 3D help" question actionable rather than purely dataset-dependent.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[REMOVED] Criticism that the related work section insufficiently covers 3D GNN property prediction literature and foundation model pretraining scale comparisons.** Per instructions, missing related work claims are not evaluated since external sources cannot be verified.
- **[REMOVED] Criticism demanding theoretical/mechanistic causal analysis of why 3D underperforms.** The paper explicitly scopes to an empirical benchmark; requiring theoretical proofs imposes standards not expected for empirical systems papers at ICLR.
- **[REMOVED] Criticism about "real-world chemistry datasets" vs. offline benchmarks.** This is a minor stylistic/framing nitpick; offline virtual-library BO is a standard and accepted evaluation paradigm in the community.
- **[REMOVED] Criticism about LLA choice vs. deep ensembles, MC dropout, sparse GPs, etc.** LLA is an established, principled Bayesian NN approximation with prior BO-specific validation (Kristiadi et al., 2023/2024); critiquing the surrogate choice is outside the paper's stated scope of comparing representations. The paper cites Li et al. 2024 as motivation for the two-hidden-layer LLA architecture.
- **[REMOVED] Criticism about "ensuring best 10 observations remain in virtual library" being a "problematic design choice."** This is a standard BO benchmarking technique to ensure the global optimum is always reachable during the BO loop; it is not a flaw.
- **[REMOVED] Criticism that "multiple conformers per molecule" should be tested.** Testing single-conformer vs. multi-conformer inputs is outside the stated scope and would be a distinct methodological contribution. (The missing description of *which* conformer is selected—flagged under Weaknesses—is distinct from this.)
- **[REMOVED] Strength: "The paper is well-written / the topic is important / the benchmark is extensive."** These are generic and apply to any paper in the area.

---

## Novel Insights

The most genuinely novel observation that emerges from cross-reading the reviews and paper is the **interaction between representation dimensionality and data regime as a structured, quantified phenomenon in BO** (Section 5.2). Prior work had noted qualitatively that equivariant models can be data-hungry, but grounding this within the BO loop—where the surrogate is incrementally updated—reveals a compounding effect: not only do 3D models need more data to reach parity with 2D models in supervised learning, but the BO acquisition step must compensate for a worse-calibrated surrogate early in optimization. The sample-complexity crossover observed above 10,000 training observations provides a concrete threshold that practitioners can use. This finding, if validated with proper conformer controls and cost measurements, has the potential to be a durable empirical result. The remaining insights (LLM dominance, transfer learning viability) are unfortunately confounded or underdeveloped as detailed above.

---

## Suggestions

1. **Include a pretrained 3D baseline (e.g., Uni-Mol) or a non-pretrained 1D transformer** to isolate representation dimensionality from pretraining scale in the LLM comparisons. This is the single highest-priority revision.
2. **Add at least one conformer-sensitive task** (stereo-isomer property discrimination, docking score, conformer-dependent solvation) so the scope of "3D is not needed" can be properly bounded.
3. **State the acquisition function explicitly** and, if multiple were tested, report sensitivity to acquisition function choice.
4. **Report wall-clock time per BO iteration** (or at minimum per feature-extraction call) for each modality to make the cost-benefit claim empirical rather than qualitative.
5. **Describe the conformer selection protocol** (tool, energy criterion, number of conformers retained) and ideally include a brief ablation or discussion of how conformer quality affects 3D performance.
6. **Harmonize the QM9/LLM finding** between Section 5.1 and the Conclusion, and fix the misidentified representation in the QM9 explanation.
7. **Extend transfer learning experiments to MoleculeNet and GEOM DRUGS** to support the "foundation model" framing.
8. **Include LLM/1D in the sample-complexity figure** (Section 5.2) to complete the picture of all modalities' data efficiency.

---

## Paper Evaluation

| Axis | Assessment |
|---|---|
| **Originality** | Moderate. No new algorithm or methodology; the benchmark itself is the contribution. Including 3D representations in BO benchmarking fills a genuine gap, but the experimental design leaves the most interesting comparisons confounded. |
| **Importance of research question** | High. Representation choice in molecular BO is a concrete, practically significant question with direct implications for computational chemistry pipelines. |
| **Claims well-supported** | Partially. The "2D ≥ 3D" finding is reasonably supported across datasets and settings, but the "1D/LLM dominates" claim is critically confounded by pretraining scale. The cost-benefit claim lacks empirical cost data entirely. |
| **Soundness of experiments** | Moderate. The scale and seed count are commendable, but missing protocol details (acquisition function, conformer selection) and the confound between model scale and representation type reduce confidence in the conclusions. |
| **Clarity of writing** | Below expectations for ICLR. Internal inconsistencies (QM9/LLM contradiction between Section 5.1 and Conclusion), missing method details, and ambiguous training setup descriptions impair reproducibility and interpretation. |
| **Value to the research community** | Moderate. The sample-complexity finding and transfer learning analysis are actionable. However, the headline takeaway as currently presented ("3D is not useful") risks being misapplied by practitioners working on tasks where 3D geometry is intrinsically necessary, because no such tasks are tested. |
| **Contextualized relative to prior work** | Adequate. The paper correctly identifies that existing BO benchmarks skip 3D representations and positions itself accordingly. The connection to broader supervised-learning literature on 3D vs. 2D molecular modeling is thin. |