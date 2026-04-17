---
job_id: bd586356-e7c3-4a0b-8fce-6404c6792bd2
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: ulXBsAYSwU.pdf
paper: MolMiner: Towards Controllable, 3D-Aware, Fragment-Based Molecular Design
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper presents a fragment-based autoregressive generative model for molecular design with transformers, geometry-aware attention, and multi-property conditioning, which fits squarely within ICLR’s scope on generative models, learning on graphs, and applications to physical sciences.

## Minimum Quality
Pass ✅.  
All major sections (Abstract, Introduction, Related Work, Method, Experiments, Results/Analysis, Limitations, Conclusion) are present and in English. The method is clearly described, math is mostly consistent, and experiments with quantitative tables and figures are provided. While there are notable weaknesses (limited baselines, single dataset, some underspecified choices), they do not rise to the level of an automatic desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any text in the paper attempting to instruct or manipulate automated reviewers or hidden prompts; figures and appendices appear standard.

---

# Expected Review Outcome:

## Summary

The paper proposes **MolMiner**, a fragment-based, order‑agnostic autoregressive model for molecular generation. Molecules are decomposed into ring and bond fragments, and generation proceeds by attaching fragments at open sites while periodically relaxing the 3D geometry via a force field; attention is biased by inter-fragment distances. The model supports conditioning on up to 12 RDKit‑computed properties, uses a Gaussian mixture model to sample unspecified properties, and is evaluated on unconditional distribution matching and conditional calibration on a ZINC subset.

## Strengths

1. **Unified take on several desirable capabilities.**  
   MolMiner brings together fragment-based generation, order‑agnostic rollouts, an explicit (if simple) 3D-aware attention bias, and multi-property conditioning in a single framework. This combination is nontrivial and useful for practice, even if each ingredient individually has precedent.

2. **Clear probabilistic formulation and order‑agnostic training.**  
   The rollout likelihood in **Equation (1)** and the Jensen lower bound in **Equation (3)** follow the order‑agnostic training framework of Uria et al. and Hoogeboom et al. and are cleanly presented. Sampling a random rollout per epoch is a straightforward but effective form of data augmentation, supported empirically in **Figure 8** (resampling vs no-resampling).

3. **Symmetry‑aware fragment attachment is thoughtfully handled.**  
   The symmetry issues in fragment canonicalization are real headaches in fragment-based models. The authors’ procedure in Section 3.2 and Appendix A.6, using Morgan fingerprints plus constrained cyclic permutations to produce a standard map (see **Equations (6)–(7)** and **Figures 13–14**), is technically careful and, as far as I can tell, mathematically consistent for single-cycle fragments. This is one of the more solidly engineered parts of the work.

4. **Property‑calibrated conditional generation is reasonably strong.**  
   The calibration plots in **Figure 2** show that, for most properties (logP, SAS, FractionCSP3, HBD/HBA, ring count, rotatable bonds, chiral centers), the model tracks the prompted values quite well across the range \(\mu \pm 2\sigma\). Deviations in molWt and MR are systematic but interpretable. This multi-property calibration is more ambitious than typical one- or two-property controllable models.

5. **GMM-based partial conditioning is conceptually clean and mathematically well-specified.**  
   The conditional sampling from a Gaussian mixture in **Equations (4)–(5)** is done correctly, with explicit formulas for \(w_k\), \(\mu_{k,\mathrm{miss|obs}}\) and \(\Sigma_{k,\mathrm{miss|obs}}\). The validation experiment in **Figure 4** (q‑q plots and Wasserstein distances for reconstructing one property from the others) demonstrates that the GMM is a reasonable approximation of the joint property distribution, which is important since this mechanism drives “unconditional” sampling and partial conditioning.

6. **Extensive sampling ablations.**  
   The paper does a commendable job exploring decoding strategies. **Table 3** systematically varies condition source (dataset vs GMM), top‑k seed fragment selection, weighting, and greedy vs stochastic decoding. This is more thorough than is typical and clarifies trade‑offs between distributional match, diversity, and uniqueness.

7. **Useful evaluation diagnostics.**  
   The authors advocate Wasserstein distances for unconditional distribution comparison and calibration plots for conditional performance. While not conceptually new, the combination used here is appropriate and well-presented. **Table 1** and **Figure 15** together give a reasonably detailed picture of how MolMinerD/S compare to HierVAE on 12 properties.

8. **Ablation evidence that geometry and richer conditioning help.**  
   The training curves in **Figures 5–7** strongly suggest that (i) conditioning on 12 properties vs 3 improves reconstruction loss (“tomographic effect”) and (ii) the learned geometric bias with a positive initialization substantially improves both train and validation loss, while large negative initializations destabilize training. These ablations support the design choices rather than leaving them as arbitrary.

9. **Visualization of conditional behavior beyond scalar metrics.**  
   The flow plots in **Figure 9** showing how the initial fragment distribution changes with each individual conditioning property are a nice interpretability touch, giving some intuition about how conditioning is realized at the fragment level.

## Weaknesses

1. **Empirical evaluation is narrow and baseline coverage is weak.**  
   All experiments are on a **single, relatively small ZINC subset (~200k molecules)** with RDKit properties. There is no evaluation on alternative datasets (e.g., MOSES, QM9, ChEMBL-like sets) or genuinely property-focused tasks (e.g., activity/affinity, quantum properties). For unconditional generation, the only strong baseline is **HierVAE**, a fairly old VAE; MoLeR is reported only in the appendix and in a clearly broken regime, and MARS is dismissed rather than adapted for a fair comparison. Diffusion-based 3D or graph models and more recent fragment/motif-based architectures are entirely absent as baselines. This makes it hard to judge how competitive the model really is as a generative model beyond this specific setting.

2. **Unconditional performance is noticeably worse than HierVAE on core properties, without a convincing remedy.**  
   In **Table 1**, MolMinerD/S have substantially larger Wasserstein distances than HierVAE for molWt, TPSA, and MR, which are central to drug-like chemistry. For instance, HierVAE has molWt WD of 15 vs 47/65 for MolMinerD/S, and similar gaps for TPSA and MR. The paper attributes this mainly to early termination bias caused by many termination tokens in order‑agnostic rollouts (Section 5), but this explanation is speculative and not tested. Simple baselines like reweighting termination actions, curriculum on rollout depth, or RL-based termination tuning are not attempted, even though the authors explicitly suggest them. The result is a clear weakness in unconditional generation that undercuts the claim of “competitive” unconditional performance.

3. **3D modeling is relatively shallow and not properly evaluated as a 3D model.**  
   The model’s “3D awareness” is implemented through a scalar learned bias \(\theta D_{ij}\) in **Equation (2)**, where \(D_{ij}\) is a radial Gaussian of inter-fragment distance. This is not SE(3)-equivariant and does not exploit orientation or local geometry beyond scalar distances. More importantly, there is **no evaluation of generated 3D structures** as such: no metrics on RMSD to known conformers, internal steric clashes, or energy distributions, and no comparison to 3D models like G-SchNet or equivariant diffusion models on actual 3D generative quality. The only “3D” effect shown is improved reconstruction loss in **Figures 6–7** and better property match in aggregate, which could stem from exploiting distance correlations rather than genuinely good 3D generative behavior. For a paper that heavily markets 3D-awareness, this is a substantial gap.

4. **Conditional generation is only evaluated using surrogate RDKit properties, not task-relevant objectives.**  
   All 12 conditioning targets are RDKit‑computed “cheap” properties (logP, QED, SAS, MR, etc.). While these are standard, they are relatively easy to approximate, and many are strongly correlated. There is no demonstration that conditional control transfers to more realistic design tasks, such as docking scores, DFT-level properties, or bioactivity predictions. The calibration plots in **Figure 2** are encouraging as diagnostics but do not answer whether MolMiner is actually useful in a realistic inverse design loop.

5. **Use of GMM for sampling missing properties can induce inconsistencies, and its interaction with generation is under-analyzed.**  
   Appendix A.2 validates the GMM by reconstructing a *single* missing property from the remaining 11 and reporting low Wasserstein distances in **Figure 4**. However, the actual use case often conditions on *small* subsets (1–3 properties) and samples the rest. In such regimes, GMM extrapolation in sparse regions can easily produce unrealistic tuples. The authors attribute part of MolMinerS’s degradation relative to MolMinerD in **Table 1** to GMM approximation error, but do not quantify how often GMM-sampled property vectors fall outside the empirical support or how sensitive MolMiner is to such out‑of‑support conditions. There is also no calibration of conditional accuracy when conditioning on only a subset of properties, even though this is a main advertised feature.

6. **Order-agnostic modeling claims are stronger than the supporting evidence.**  
   The paper argues that order‑agnostic rollouts improve flexibility and act as a regularizer (Section 3.3, 4.1), but the only actual evidence for the regularization effect is **Figure 8**, which compares resampled vs fixed rollouts and finds better validation loss with resampling. However, this is not disentangled from the order‑agnostic factorization itself: there is no comparison to a fixed canonical order (e.g., BFS) with the same transformer, nor any ablation on the termination distribution. Moreover, the “drawback” of early termination bias (Section 5) seems to be a direct side effect of this rollout design, yet no mitigation strategies are evaluated. The net effect on *generation quality* (not just loss) is thus unclear.

7. **Mathematical / probabilistic details are glossed over in a few important places.**  
   - In **Equation (3)**, \( \mathcal{L}(\theta\mid\mathcal{M})\) is defined as the log of the expectation over products of probabilities, then lower-bounded by \( \mathbb{E}_R[ \sum_i \log p_\theta(x_i^{(R)} | \cdot)]\). This is fine, but in practice they sample exactly one rollout per molecule per epoch. This corresponds to an unbiased Monte Carlo estimator of the **expectation**, but only an estimator of the bound; the variance and its effect on optimization are not discussed. It would be good to make explicit that the training objective is *stochastic* and that the lower bound is not tightly optimized.  
   - In **Equation (2)**, the distance kernel \(D_{ij} = e^{-\|x_i-x_j\|^2 / (2\sigma^2)}\) has a fixed \(\sigma\); the choice of \(\sigma\), its units relative to force-field relaxed coordinates, and whether it is learned or tuned are not specified. This matters because when \(\sigma\) is too small or too large the kernel collapses to local or global uniform weighting. Without this, reproducibility and interpretation suffer.  
   - In Appendix A.2, Equation (5) is labeled as \(f(\vec{x}_{\mathrm{obs}}|\vec{x}_{\mathrm{miss}})\) instead of \(f(\vec{x}_{\mathrm{miss}}|\vec{x}_{\mathrm{obs}})\); the subsequent text and formulas clearly refer to the latter. This is a minor notational error but symptomatic of some sloppiness in math editing.

8. **Fragment decomposition and vocabulary design are only qualitatively evaluated.**  
   The fragmentization uses SSSR rings plus leftover bonds (Appendix A.6 and **Figures 11–12**). This yields a vocabulary of single-cycle fragments, but there is no analysis of: (i) vocabulary size, (ii) frequency distribution of fragments, (iii) how many fragments can be reconstructed uniquely from the data, and (iv) how this choice compares to alternative motif vocabularies (e.g., ring‑system + linker decompositions, frequent motifs). Since the model is fragment-based, the fragment vocabulary is a central design choice; the paper treats it mostly as given.

9. **Limited comparison to closely related contemporary work.**  
   The Related Work section focuses on JTNN, HierVAE, G-SchNet, and MoLeR, but omits several relevant recent efforts that combine fragments/motifs with 3D awareness or transformers on 3D molecular graphs. This weakens claims about being the first to “unify” various capabilities. See “Potentially Missing Related Work” below for concrete omissions.

10. **Some figures and results are diagnostic but do not connect back to concrete design guidance.**  
    For example, **Figure 9** (effect of conditioning properties on seed fragment selection) and **Figure 10** (SeedFragNet hyperparameter search) are interesting but the paper does not clearly state what practical conclusions a practitioner should draw. Does controlling particular properties reliably select particular substructures? Are there failure modes where the seed fragment systemically disagrees with the desired property profile? More interpretation would make these analyses more than just visualizations.

11. **Single-task, single‑dataset focus limits generality.**  
    All 12 properties are RDKit‑computable, so both the GMM and the conditional likelihood are trained and evaluated on the same surrogate feature space. There is no test on transfer to a *different* distribution (e.g., another subset of ZINC, ChEMBL) or evidence that the conditional generator can extrapolate beyond the training range (e.g., target properties outside \(\mu \pm 2\sigma\)). This limits the evidence that MolMiner is truly a robust conditional generator rather than a sophisticated re-sampler of the training set.

12. **Some reporting inconsistencies and typos in the tables.**  
    In **Table 1**, several column headers appear garbled (“SagP”, “PractCSP3”, “Rfothondt”), and in **Table 3** many columns are mis-typed (“mgP”, “OED”, “PPSA”, “eRings”, etc.). While likely copy‑paste or LaTeX issues, they hurt clarity and make cross-referencing with Appendix tables (**Table 4**) unnecessarily confusing.

## Potentially Missing Related Work

1. **Zhang, O., Huang, Y., Chen, S. (2024). “FragGen: Towards 3D Geometry Reliable Fragment-Based Molecular Generation.”**  
   This work focuses on fragment-based molecular generation with strong emphasis on 3D geometry reliability, which is very close in spirit to the “geometry-aware fragment-based” positioning of MolMiner. It should be cited and contrasted in **Section 2 (Related Work)** and in the discussion of 3D inductive biases in **Section 3.4**, including differences in how 3D information enters the model and how 3D quality is evaluated.

2. **Wu, F., Radev, D., Li, S. Z. (2021). “Molformer: Motif-Based Transformer on 3D Heterogeneous Molecular Graphs.”**  
   Molformer uses transformers over motif-based 3D graphs, making it highly relevant to MolMiner’s transformer‑based, fragment-centric, 3D‑aware architecture. It should be discussed in **Section 2**, especially when presenting transformer-based and geometry-aware molecular generators, and possibly compared empirically if feasible.

3. **Flam-Shepherd, D., Zhigalin, A., Aspuru-Guzik, A. (2022). “Scalable Fragment-Based 3D Molecular Design with Reinforcement Learning.”**  
   This paper uses reinforcement learning for fragment-based 3D molecular design. Since MolMiner mentions possible RL-based fine‑tuning for termination in **Section 5**, it would be appropriate to connect these ideas and clarify differences in problem setup and learning paradigm in **Section 2** and the **Limitations** discussion.

4. **Martínez León, A., Ries, B., Hub, J. S. (2025). “Moldrug Algorithm for an Automated Ligand Binding Site Exploration by 3D Aware Molecular Enumerations.”**  
   Moldrug performs 3D‑aware molecular enumeration for ligand binding site exploration, closely related to fragment-based enumeration plus 3D geometry. It would be relevant to mention in **Section 2** when discussing 3D-aware enumeration strategies, and potentially in **Section 6** when speculating about downstream applications of MolMiner.

5. **Zhang, K., Lin, Y., Wu, G. (2025). “Sculpting Molecules in Text-3D Space: A Flexible Substructure Aware Framework for Text-Oriented Molecular Optimization.”**  
   This work deals with 3D‑aware, substructure‑aware molecular optimization, which overlaps conceptually with MolMiner’s fragment-based controllable generation. It should be referenced in **Section 2** when describing recent 3D + substructure‑focused generative models and in **Section 6** as a complementary direction for integrating natural language or higher-level objectives.

## Questions

1. **3D evaluation and equivariance.**  
   - Have you evaluated the physical plausibility of generated 3D structures beyond force‑field relaxation (e.g., steric clashes per molecule, distribution of force‑field energies, or RMSD to re-optimized conformers)? If such results exist, please summarize them.  
   - Given that the geometric bias in **Equation (2)** is not SE(3)-equivariant, how sensitive is MolMiner to global rotations or translations of the starting fragment’s conformation? Could you report performance with randomized initial orientations?

2. **Termination bias and unconditional distribution gaps.**  
   - The limitations section hypothesizes that early termination bias explains the underestimation of molWt, TPSA, MR. Can you run a quick ablation where termination tokens are downsampled or reweighted during rollout sampling to test this hypothesis? Even a small‑scale experiment would strengthen the argument.  
   - Alternatively, have you tried imposing a minimum or target range of fragment counts during generation to align molecular size with the training distribution?

3. **Partial-conditioning robustness.**  
   - All calibration plots in **Figure 2** are for the case where *all* 12 properties are conditioned via the GMM-completed vector. How does calibration behave when you condition only on 1–3 properties and sample the rest? Can you show at least one example (e.g., conditioning only on logP and molWt) to demonstrate robustness in the intended interactive design scenario?

4. **Fragment vocabulary characterization.**  
   - What is the size of the fragment vocabulary, and what is the frequency distribution (e.g., top‑10 fragments vs long tail)?  
   - Have you compared your SSSR + bond decomposition to other fragmentization schemes (e.g., ring systems, BRICS, or motif mining) in terms of generation quality or property control?

5. **Choice of \(\sigma\) in the distance kernel.**  
   - How is \(\sigma\) in **Equation (2)** chosen? Is it fixed or learned, and what is its numeric value relative to typical inter-fragment distances after UFF relaxation?  
   - Have you examined sensitivity to \(\sigma\), e.g., by varying it and tracking Wasserstein distances in **Table 1** or calibration in **Figure 2**?

6. **Baselines for conditional generation.**  
   - Could you compare MolMiner’s conditional calibration against at least one existing conditional model (e.g., a simpler graph‑based conditional VAE or diffusion model) on the same 12 RDKit properties? Even if these baselines do not support 12‑property control natively, adapting them to multi‑target regression objectives would help contextualize your results.

7. **Out-of-distribution conditioning.**  
   - Have you explored conditioning outside \(\mu \pm 2\sigma\) for any property? If so, does the generator still produce chemically sensible molecules, or does it collapse or ignore the target? A small extrapolation study could clarify the limits of your method.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The core generative mechanism is technically sound, and the math (Equations (1)–(5), (6)–(7)) is mostly correct, but the empirical evidence is constrained to a single surrogate-property setting with limited baselines and missing 3D structure evaluation; several important design choices (termination bias, kernel \(\sigma\), fragment vocabulary) are under-analyzed.

## Presentation Rating

3: good.  
The paper is generally well-written and structured, with clear figures like **Figure 1** (overall architecture) and **Figure 2** (calibration). However, there are multiple typos in table headers, some notational slips (Equation (5)), and several sections (e.g., SeedFragNet hyperparameter search) that could be more tightly integrated into the main narrative.

## Contribution Rating

2: fair.  
The work combines several known ideas into a coherent framework and pushes multi-property conditioning to a somewhat higher dimensionality, but the incremental empirical validation, narrow benchmark, and lack of rigorous 3D evaluation limit the overall impact. The symmetry-aware fragment canonicalization is a notable technical contribution.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper has interesting ingredients (symmetry-aware fragment handling, order‑agnostic rollouts, multi-property calibration) and is technically competent, but the empirical section is too narrow and the 3D and unconditional claims are under-supported for ICLR standards. With stronger baselines, more diverse datasets, and proper 3D evaluation, this line of work could be much more compelling.

## Reviewer Confidence

4: confident.  
I am familiar with fragment/motif-based molecular generation, 3D-aware generative models, and conditional property control, and I carefully checked the mathematical derivations and experimental setup within the constraints of the provided text.