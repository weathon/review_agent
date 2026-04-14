## Summary
This paper presents a large-scale empirical benchmark comparing 1D (MolFormer/LLM), 2D (MPNN), and 3D (Equiformer v2) molecular representations for Bayesian Optimization in materials discovery. Across four datasets (QM7, QM9, GEOM MoleculeNet, GEOM DRUGS) with multiple surrogates (GP, LLA), the authors find that simpler 1D and 2D representations generally match or outperform 3D, and that 3D representations require considerably more training data to close the gap. The benchmark spans over 2100 runs and also investigates transfer learning and sample-complexity regimes.

---

## Strengths
- **Fills a genuine, documented gap in BO benchmarking**: Prior BO benchmarks for molecular discovery (Olympus, Summit, Griffiths et al. 2024) explicitly omit 3D representations. This work is the first systematic study that adds equivariant GNNs (Equiformer v2) to the comparison, across both GP and LLA surrogates—a non-trivial engineering effort not attempted elsewhere.
- **Sample complexity analysis has a concrete mechanistic anchor**: The finding in Section 5.2 that 3D (equivariant) models are substantially less data-efficient than 2D models is grounded in existing theory (Elesedy & Zaidi, 2021) and is demonstrated across all four datasets with four training-set sizes. This is the most original and actionable finding in the paper.
- **Multi-axis evaluation**: The study simultaneously varies representation dimensionality, surrogate type (GP vs. LLA), data regime (four sizes), and task type (single-property vs. transfer learning), producing structured evidence rather than a single-condition comparison.
- **Reproducibility practice**: 15 seeds per condition with reported standard errors, and an anonymous code repository, are above the norm for this class of benchmarking paper.

---

## Weaknesses

### Fatal
None. The core empirical findings are real, but a central methodological confound severely limits the scope of the strongest headline claims (see Major #1).

### Major

- **The 1D vs. 3D comparison conflates representation dimensionality with model scale and pretraining**. MolFormer is a masked language model pretrained on 1.1 billion SMILES strings (Ross et al. 2022), while the 2D/3D GNNs are constrained to ~1.5 million parameters trained from scratch on the benchmark tasks. The paper never accounts for this asymmetry. The dominant finding—"LLM/1D outperforms 2D and 3D"—is therefore at least as likely to reflect large-scale pretraining as it is to reflect anything about 1D representation dimensionality. This conflation is not minor: it makes the paper's central framing ("is 3D a step too far?") largely unanswerable from the presented experiments as the comparison is not isolating the dimensionality axis. The 2D vs. 3D comparison (both at ~1.5M parameters) is the paper's most internally fair comparison, and its conclusions should be foregrounded accordingly.

- **No computational cost measurements, despite cost being a core claim**. The paper repeatedly argues that 3D's "computational overhead" outweighs its gains—this framing appears in the abstract, introduction, results, and conclusion. Yet no wall-clock times, GPU hours, or FLOPs are reported anywhere. The cost claim is entirely qualitative. For a paper whose thesis is explicitly about cost–accuracy trade-offs, this is not a stylistic gap: without cost numbers, the trade-off cannot be evaluated.

- **Conformer handling for 3D models is never described**. GEOM datasets provide multiple conformers per molecule. The paper does not state which conformer is used for Equiformer v2 inputs—lowest-energy, random, or some other selection. This decision materially affects 3D model performance; if ground-truth minimum-energy conformers are used, 3D has an oracle advantage unavailable in real BO settings. If poor conformers are used, 3D's underperformance may reflect data quality rather than dimensionality. The confound directly undermines the interpretation of 3D vs. 2D results.

- **Task selection bias undermines the generality of the main conclusion**. All four benchmark targets—atomization energy (QM7), HOMO-LUMO gap (QM9), absolute energy (MoleculeNet/DRUGS)—are quantum mechanical scalar properties that are primarily determined by molecular topology and composition, not by specific 3D conformation. The paper itself acknowledges in the conclusion that "future research should focus on tasks where 3D information might be more important, e.g. protein docking." This acknowledgment, however, is not sufficient: it means the paper's headline "3D is a step too far" is tested only on tasks where 3D is not theoretically expected to win. The finding is valid for these tasks, but should not be presented as a general verdict on 3D representations in BO.

- **The acquisition function is never specified**. Section 4 describes datasets, feature extractors, and surrogates in detail, but never names the acquisition function (EI, UCB, Thompson sampling, etc.) or its hyperparameters. For a BO benchmark, this is a reproducibility-critical omission that prevents independent replication of any individual run.

### Minor

- **Only one 3D architecture tested**. The paper draws conclusions about "3D representations" using only Equiformer v2. Poor results could reflect architecture-specific failure modes (e.g., insufficient expressive power for the GP/LLA interface, initialization sensitivity) rather than a dimensionality-level verdict. Including even one additional 3D model (SchNet, DimeNet) would substantially strengthen the claim.

- **Factual inconsistency between abstract/conclusion and body**: The abstract and conclusion state "LLMs consistently outperformed" all methods, but Section 5.1 explicitly says "LLMs performed worse than 2D and 3D models" on QM9. This is not a minor phrasing issue—it is a factual contradiction that misleads readers who read only the abstract or conclusion.

- **Transfer learning analysis is incomplete and overclaims**: Section 5.3 and Fig. 5 show transfer learning results only for QM7 and QM9. Yet the text draws general conclusions about transfer learning and invokes "foundation model" potential. The claim "Foundation models prove a good tool" is overstated from two datasets with a limited fine-tuning protocol (only the final layer).

- **Potential MolFormer data leakage not investigated**: MolFormer was pretrained on 1.1 billion SMILES from ZINC and PubChem. QM9 and MoleculeNet molecules are small, well-known, and could plausibly appear in those corpora. If so, MolFormer's strong performance could partly reflect memorization rather than generalization. A membership overlap check is warranted.

- **Sample complexity analysis (Section 5.2) excludes the 1D/LLM comparison**: Despite MolFormer being a key performer, the sample complexity plots compare only 2D vs. 3D. If the paper's goal is a comprehensive 1D/2D/3D benchmark, omitting LLM from this axis is inconsistent.

- **GP kernel on learned embeddings is not specified**. For the GP surrogate using pretrained/trained feature embeddings, the kernel (RBF, Matérn, ARD, etc.) is never stated. This matters because kernel choice interacts with embedding geometry and directly affects uncertainty calibration.

### Tiny

- **Laplace approximation notation is inconsistent**: The paper writes $p(\theta|\Omega_t) \approx \mathcal{N}(\theta_*, \Sigma_*^{-1})$ and then defines $\Sigma_*^{-1} = -\nabla_\theta^2 \log p(\theta|\Omega_t)$, treating $\Sigma_*^{-1}$ simultaneously as a covariance parameter and as the Hessian. Standard convention is to write $\Sigma_* = H^{-1}$ where $H$ is the (positive-definite) negative log-posterior Hessian. The current notation will confuse readers.
- **"35 setups per dataset" is never broken down**. The abstract and introduction cite this number prominently, but the main text never enumerates the exact combination of representation × surrogate × regime × seed that generates the count. An explicit table in the appendix would make the benchmark auditable.
- **GAP metric notation inconsistency**: The definition uses $y_i$, $y_0$, $y_*$ but the body text refers to $y^*$. Minor but worth fixing.

---

## Nice-to-Haves

- **Include at least one genuinely conformation-dependent task** (e.g., docking score, stereoselective reaction yield, conformer-dependent binding affinity). Even a single dataset where 3D is theoretically expected to win would transform the negative results into a more principled and bounded statement rather than a potentially task-specific finding.
- **Uncertainty calibration evaluation**: BO performance depends on calibrated posteriors, not just point prediction accuracy. Reliability diagrams or ECE plots per model type would help distinguish whether 3D underperforms because its features are uninformative vs. because its uncertainty estimates are poorly calibrated—a distinction with distinct implications for practitioners.
- **Matched-pretraining controls**: A pretrained 2D GNN foundation model (e.g., from graph self-supervised pretraining) alongside a pretrained 3D model compared to MolFormer would allow cleaner disentanglement of the pretraining vs. dimensionality effect.
- **Additional 3D architectures (SchNet, DimeNet, or SphereNet)** to ensure the 3D conclusions generalize beyond Equiformer v2.
- **Per-subset breakdowns**: Reporting results for subsets of molecules by size, flexibility, or chirality would help practitioners understand when (not just whether) 3D helps.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"Why do nobody use them" is too informal** (Harsh Critic, Introduction): This is a pure style/tone complaint with no scientific content. Removed.
- **Fingerprints are described as "unique identifiers"** (Harsh Critic, Section 2.3): While technically ECFP fingerprints can have collisions, the characterization is conventional shorthand used throughout cheminformatics and does not affect any experimental result. Removed as inconsequential.
- **Comparison with Tanimoto GP is "unfair"**: The Tanimoto GP uses simple fingerprints and is explicitly used as a baseline *to prove a stronger point* for alternative methods. Any asymmetry favors the baseline, which strengthens rather than undermines the authors' claims. Removed per editorial rules.
- **Writing quality nitpicks** (subject-verb agreement, mid-sentence cuts in parsed PDF): These are artifacts of PDF text extraction or minor grammatical issues that do not affect scientific content. Removed as formatting/style nitpicks.
- **Requests for theoretical proofs of sample complexity bounds** (Spark Finder): This is an empirical benchmarking paper; demanding sample complexity theorems is not standard for this paper's scope or community setting. Moved to Nice-to-Have at most.
- **Criticism that "MoleculeNet is not a standard headline target"**: The paper uses absolute energy from GEOM's augmented version of MoleculeNet, and GEOM is a published dataset. This is a legitimate benchmark target. Removed.

---

## Novel Insights

The most genuinely novel finding beyond the benchmark results themselves is the *sample-complexity asymmetry* between 2D and 3D representations in BO: the paper provides converging evidence, consistent with equivariant model theory (Elesedy & Zaidi 2021), that equivariant 3D GNNs require substantially more training data before matching 2D performance, and that this crossover threshold (~10,000 examples in the cleaner comparisons) is well above what is typically available in realistic molecular BO campaigns with expensive oracle evaluations. This finding is actionable and not obvious: 3D models might be expected to learn more efficiently due to stronger inductive biases, but the data requirements of equivariance appear to dominate in practice. However, a clean mechanistic explanation—separating equivariance overhead from conformer noise and architecture-specific factors—remains an open question the paper does not fully answer.

---

## Suggestions

1. **Reframe the headline finding**: Present the key contribution as "2D outperforms 3D under realistic BO data budgets" (the most defensible and interesting result), rather than "1D/LLM outperforms everything," since the latter conflates pretraining scale with representation dimensionality.
2. **Report wall-clock time or GPU hours for each model class**—even a single table in the appendix comparing training and inference time per step would make the cost–accuracy trade-off argument empirically grounded.
3. **Explicitly state the acquisition function, its hyperparameters, and the GP kernel** in the main experimental setup section.
4. **Describe conformer selection policy for 3D models** (e.g., "we use the lowest-energy conformer from GEOM's precomputed ensemble") and, ideally, include a sensitivity analysis with random conformer selection.
5. **Fix the abstract/conclusion claim about "consistent LLM outperformance"** to acknowledge the QM9 exception, and qualify all major claims to specify that results hold for topology-dominated quantum chemistry properties under the data budgets studied.
6. **Add a breakdown table of the 35 setups** (representation × surrogate × task type × data regime) so the benchmark scope is auditable.