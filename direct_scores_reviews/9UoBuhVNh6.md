## Summary

Megalodon is a scalable transformer-based architecture for de novo 3D molecule generation, combining a DiT backbone with lightweight EGNN structure layers and trained with a novel "co-design" objective that samples independent time variables for continuous (coordinates) and discrete (atom/bond types) data modalities. The paper demonstrates state-of-the-art performance on 2D topological and 3D distributional benchmarks, introduces new physics-grounded energy benchmarks based on GFN2-xTB relaxation, and establishes that transformer-based architectures generalize markedly better to large, out-of-distribution molecule sizes than EGNN-only baselines.

---

## Strengths

- **Energy benchmark introduction with physical grounding**: The paper introduces GFN2-xTB-based relaxation energy (ΔE_relax), bond length, and dihedral error benchmarks (Table 3) that are directly tied to the GEOM dataset's original generation process. Megalodon achieves a median ΔE_relax of ~3.17 kcal/mol, approaching the thermally meaningful threshold of 2.5 kcal/mol—a milestone no prior 3DMG model had reached. This benchmark is specific to this paper and fills a genuine gap in the field's evaluation practice.

- **OOD molecule size generalization (Figure 3)**: The scaling experiment clearly shows that EQGAT-diff collapses to ~0% validity beyond ~90 atoms, while Megalodon large maintains ≥40–50% validity up to 120 atoms. All models are trained on identical data with identical objectives; only the architecture differs. This is a crisp, well-controlled result that directly supports the transformer-backbone hypothesis.

- **Paired diffusion vs. flow matching study within a single architecture**: By training Megalodon with both objectives using identical hyperparameters and comparing against EQGAT-diff (diffusion) and SemlaFlow (FM), the paper provides the first controlled empirical comparison of these paradigms in 3DMG. The finding that diffusion excels on energy/structure metrics while FM achieves better topological validity with 5× fewer steps is practically informative.

- **Conditional structure generation without specialized training**: Megalodon achieves coverage-recall of 71.4% (mean) and 75.0% (median) in Table 2—competitive with Torsional Diffusion (75.3%/82.3%) and dramatically better than EQGAT-diff (0.8%/0.0%), despite Megalodon being trained as an unconditional generator. The co-design training objective is shown to enable this capability, and the mechanism (learning structure-given-2D-graph via structure-only noise half of the time) is experimentally supported.

---

## Weaknesses

- **Co-design training objective insufficiently formalized and unablated**: The paper's central methodological contribution—sampling independent time variables t_continuous and t_discrete plus augmenting with structure-only noise half the time—is described informally in one paragraph without a formal equation or algorithm box. This makes the method unreproducible from the main text alone. More critically, no ablation is provided to isolate the contribution of the dual-time-variable scheme from the transformer architecture: is it the independent t schedule, the structure-only augmentation, or the transformer trunk that drives the conditional generation improvement? Without a controlled ablation (e.g., Megalodon with a single shared time variable), the core methodological claim cannot be attributed to the proposed training objective rather than to architectural capacity.

- **SE(3) vs. E(3) equivariance not acknowledged**: The EGNN structure layer includes a cross-product term (shown explicitly in Figure 1's update equation: (a_i - a_j) × (a_i - a_j) / ||...|| × ...), which breaks reflection symmetry and makes the model SE(3)-equivariant rather than E(3)-equivariant. This is architecturally significant for drug design tasks involving chiral molecules and is nowhere acknowledged in the paper. At minimum, the equivariance group should be declared and its implications for chiral molecule generation discussed.

- **Flow matching energy gap not mechanistically explained**: Table 3 shows Megalodon-flow has a mean ΔE_relax of 46.86 kcal/mol versus Megalodon-diff's 5.71 kcal/mol—an 8× gap with identical loss functions. The paper hypothesizes that coordinate normalization (variance scaling to match the Gaussian prior) reduces local spatial precision for bond lengths. However, this is not tested: no ablation disables normalization in Megalodon-flow, and no bond-length error breakdown is provided. This matters because if FM is fundamentally unsuitable for precision 3D generation under this formulation, that is a significant negative result about a methodology the paper otherwise advocates for.

- **Headline "49× improvement" is at a baseline failure mode**: The abstract's claim of "49× more valid molecules at large sizes" occurs when EQGAT-diff degrades to near-0% validity (Figure 3, ~120 atoms), a regime representing <0.1% of the training data. While the OOD generalization result is genuine and important, presenting this specific ratio as the headline improvement is misleading about what the typical performance gap looks like. A more representative framing (e.g., average improvement across the 72–120 atom range) would be more honest.

- **Cross-product term claim unsupported by ablation**: The paper states the cross-product term is "critical for model performance," but Table 1's "EGNN + cross product" row confounds the cross-product term with the complete removal of the transformer trunk. An ablation of "Megalodon (large) without cross-product" is needed to support this specific claim; as written, the evidence supports only "the transformer is critical," not the cross-product term itself.

- **Memory efficiency claim is confusingly stated**: The paper claims Megalodon (4× more parameters) is "more memory efficient than EQGAT-diff" while "still having the quadratic dependency of fully connected edge features." This is never clearly explained—is the efficiency from a different GPU implementation, batch normalization differences, or something else? This claim needs substantiation (e.g., peak GPU memory comparison) or clarification of what precisely is being measured.

---

## Nice-to-Haves

- **Inference cost analysis**: A wall-clock time or FLOPs-per-molecule comparison between Megalodon-diff (500 steps), Megalodon-flow (100 steps), and baselines would help practitioners assess the actual deployment trade-off. The "25× fewer steps" claim for FM does not account for potential per-step cost differences.

- **Dual-schedule ablation at intermediate fractions**: The paper uses a 50/50 split between jointly noised and structure-only noised samples. Reporting performance at 0%, 25%, 75%, 100% structure-only fraction would reveal whether 50% is specifically important or whether any nonzero fraction helps.

- **Model scaling with intermediate sizes**: Only two model sizes (19M and 40M) are tested. Adding 1–2 intermediate checkpoints would help establish whether the scaling trend is monotone and predictable, supporting the paper's framing around scaling laws.

- **Diversity metrics for low-energy subset**: It would strengthen the drug-discovery relevance argument to confirm that the low-energy generated molecules are chemically diverse (e.g., FCD or Tanimoto diversity on valid + low-energy subset), ruling out mode collapse to a small set of stable scaffolds.

- **Computational cost for conditional generation at scale**: A brief runtime comparison for the conditional structure generation task (Table 2) would clarify whether Megalodon's approach—despite requiring no retraining—has practical latency advantages or disadvantages versus purpose-built conformer generators.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic: TorsionalDiffusion is an unfair baseline** — Per review rules, the comparison is asymmetric in favor of TorsionalDiffusion (it uses RDKit initialization, only rotates dihedrals, and is purpose-built for conformer generation). This asymmetry benefits the baseline, making Megalodon's competitive performance a stronger result, not a methodology flaw.

- **Harsh critic: 100% uniqueness is expected and uninformative** — The paper includes this as a routine completeness check following prior work definitions (Le et al., 2024), not as a novel metric. Removing it is a style preference.

- **Harsh critic: Background section too long** — Pure style/formatting critique with no bearing on scientific validity.

- **Harsh critic: Eq. (3) functions f, g, σ underspecified** — The paper explicitly states "functions f, g, and σ are defined for any noise schedule such as the cosine noise schedule used in Vignac et al. (2023)." This is a direct citation; the functions are defined in the referenced prior work and reproduced in the appendix. Not a real flaw.

- **Harsh critic: GCDM and GDM-AUG missing baselines** — Per rules, we do not flag missing related works without external confirmation. If cited baseline omission genuinely matters it should appear in other reviews; it does not.

- **Harsh critic: Figure 2 "Diffusion Data" curves appear linear** — The image description is auto-generated from a figure image and may not faithfully represent the actual curves. Cannot reliably judge from text representation alone.

- **Positive reviewer / Harsh critic: Conditional generation "first model" claim overstated** — The paper's specific claim is that Megalodon is the "first model capable of *unconditional* molecule generation *and* conditional structure generation without retraining." Given that EQGAT-diff (the closest prior art in the same setting) achieves 0% coverage in Table 2, this specific combination is factually novel. The criticism that "many diffusion models support conditional sampling via classifier-free guidance" misreads the claim—this is about joint generation capability, not guidance, and the empirical failure of EQGAT-diff validates the specificity of the claim.

- **Harsh critic: Scaling comparison lacks parameter-matched baseline** — While a parameter-matched comparison would be ideal, the paper's hypothesis is specifically about the architectural inductive bias (transformer vs. EGNN), and the authors explicitly note all hyperparameters, objectives, and data are identical. This is a reasonable experimental design given the goal of isolating architecture. Moved to nice-to-have-level at most.

- **Harsh critic: Missing QED/SA scores / ADMET properties** — This is explicitly outside the paper's stated scope. The paper focuses on 3D structural quality benchmarks, and ADMET evaluation would require a different experimental setup (property-conditioned generation). Evaluating the paper against standards it did not set is scope creep.

- **Harsh critic: Energy computed only over valid molecules is cherry-picking** — This is standard practice in the field since energy calculations on invalid/disconnected structures are physically meaningless. The paper discloses this explicitly in Table 3's caption.

---

## Novel Insights

The most genuinely novel observation across the three reviews—partially surfaced but not fully emphasized in the paper itself—is that the failure mode of EQGAT-diff in conditional structure generation (0% coverage, Table 2) stems from a fundamental training-time coupling issue: when using a single shared time variable with a weighted cosine schedule, the model never observes structured 2D graphs at high noise levels for the coordinates, causing discrete edge features to contribute no learning signal for half of training. The paper's independent time variable co-design directly addresses this by constructing training instances where the 2D graph is clean while the 3D structure is noisy, forcing the model to explicitly learn the mapping from topology to geometry. This insight—that jointly training with mixed-modality noise levels is necessary to achieve usable conditional generation without specialized training—is non-obvious and suggests a design principle relevant beyond this specific architecture. A second insight, underappreciated in the paper itself, is that flow matching's coordinate normalization (variance rescaling to unit Gaussian) may introduce a precision ceiling for high-fidelity 3D generation tasks where bond-length accuracy is energetically decisive—a potential fundamental limitation of direct FM application to metric geometry generation that deserves independent investigation.

---

## Suggestions

1. **Formalize the co-design objective**: Add an algorithm box or numbered equation explicitly defining how (t_continuous, t_discrete) are sampled, how the structure-only noise augmentation is applied, and what the combined training loss looks like. This is the paper's novel methodological contribution and must be reproducible from the main text.

2. **Add a single critical ablation**: Train Megalodon-large with a shared single time variable (matching the EQGAT-diff training objective) and report conditional generation performance. This directly demonstrates that the dual-schedule design—not the transformer architecture—drives the conditional generation capability.

3. **Acknowledge SE(3)-equivariance explicitly**: State that the cross-product term makes the model SE(3)-equivariant (not E(3)), and briefly discuss implications for chiral molecule generation (e.g., whether the model can produce both enantiomers).

4. **Reframe the "49×" headline**: Replace this with a more representative statistic, such as the average validity improvement over molecules above 80 atoms (top 0.5% of training data), which would still be striking but more honest about the operating range.

5. **Investigate FM coordinate normalization**: Add a brief experiment disabling or reducing coordinate normalization in Megalodon-flow and report the effect on bond length error and ΔE_relax. This would either validate the paper's hypothesis or reveal that FM's energy gap has a different cause.

6. **Clarify memory efficiency**: Either provide a peak GPU memory comparison (GB) between Megalodon and EQGAT-diff at identical batch sizes, or remove the efficiency claim if it cannot be precisely substantiated.

---

## Evaluation

- **Novelty**: Moderate-to-good. The transformer-EGNN co-design architecture is a competent synthesis of existing components rather than a fundamental innovation, but the independent time variable training objective and the energy benchmark suite are genuinely novel contributions to the 3DMG field.
- **Technical soundness**: Good, with one notable gap—the co-design objective is the paper's central technical claim but lacks formal specification and direct ablation. The architecture and experiment design are otherwise well-executed.
- **Empirical support**: Good. The improvements in Table 1, Figure 3, and Table 3 are convincing and the effect sizes are large. The conditional generation result (Table 2) is particularly impressive given no task-specific training. Main caveats: small conditional test set (200 molecules), and the size-scaling experiment uses only 100 molecules per bucket.
- **Significance**: High. The energy benchmarks represent a genuine step toward physically grounded evaluation in a field over-reliant on 2D metrics. The OOD generalization results raise important questions about backbone architecture choice for molecular generative models.
- **Clarity**: Adequate for most sections; poor for the co-design objective, which is underspecified and scattered across §3 and §4.2.

MY FINAL SCORE: <pineapple>6.4</pineapple>