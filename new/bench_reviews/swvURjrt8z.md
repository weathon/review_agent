## Summary
This paper proposes TABDIFF, a mixed-type diffusion model for tabular data that jointly handles numerical features with Gaussian diffusion and categorical features with masked diffusion in a single continuous-time framework. Its main claimed contributions are feature-wise learnable noise schedules, a stochastic sampler intended to correct sticky categorical decoding errors, and a classifier-free-guidance extension for conditional imputation. Empirically, the paper is strong overall, especially on correlation-oriented fidelity metrics, but some of its headline claims are stronger than what the visible evidence cleanly supports.

## Strengths
- **Well-motivated problem and meaningful technical framing.** The paper directly addresses a real difficulty in tabular synthesis: mixed numerical/categorical types plus strong feature heterogeneity. Modeling numerical and categorical variables in their native spaces within one diffusion framework is a sensible and technically relevant design.
- **Feature-wise schedule learning is a plausible and useful contribution.** Section 2.3 is the paper’s most distinctive component, and the ablation in Table 5 supports that learnable schedules help, especially on Trend (from 1.93 to 1.80 when comparing fixed vs. learned schedules under the stochastic sampler).
- **Strong empirical performance on fidelity, especially pairwise correlations.** Table 2 is consistently favorable to TABDIFF across all seven shown datasets, with an average Trend error of 1.80 vs. 2.33 for TaBSYN. This is a substantive result and aligns with the paper’s intuition that joint mixed-type modeling should help capture inter-column dependencies.
- **The stochastic sampler addresses a real issue in absorbing categorical diffusion.** The paper correctly notes that once a categorical feature is decoded, it becomes hard to revise under the basic reverse process, and the restart-style stochastic sampler is a reasonable mechanism for self-correction. Table 5 supports that it helps in practice.
- **The paper evaluates both fidelity and utility.** Beyond Shape/Trend, the paper includes downstream MLE and an imputation setting, and it honestly notes that MLE may not always be a reliable proxy for overall data quality. That self-awareness improves credibility.
- **Writing is generally clear at the high level.** The motivation, high-level method description, and overall experimental narrative are easy to follow.

## Weaknesses

###: Fatal
None.

### Major:
- **The paper overstates the strength and uniformity of its empirical superiority.** The abstract and introduction repeatedly claim “superior average performance ... across all eight metrics,” but the main paper only shows a subset of those metrics, and even within the visible tables TABDIFF is not uniformly best on every dataset. For example, in Table 1 (Shape), TaBSYN is numerically better on Default (1.01 vs. 1.24) and News (2.06 vs. 2.35), even though the narrative remains very strong. Likewise, Table 3 shows competitiveness rather than universal dominance. The method looks strong, but the rhetoric should be toned down to match the evidence shown in the paper body.
- **The evidence for the central novelty—learnable feature-wise schedules—is not isolated as convincingly as it should be.** The ablation in Table 5 is useful but limited: it reports only average Shape/Trend, only one fixed schedule choice is used as the comparator (\(\rho_i\equiv 7\), \(k_j\equiv 1\)), and there is no comparison to alternative non-learned but feature-specific schedules or other reasonable schedule families. As written, the results support that this particular learned scheme helps over this particular fixed choice, but they do not fully establish that schedule learning itself is the key reason for the gains.
- **The experimental comparison protocol is not fully clean from the main text.** Table 1 explicitly states that most baseline numbers are taken from Zhang et al. (2024), with TaBSYN reproduced and Diabetes newly included. For a paper making strong SOTA claims based on relatively modest margins on some metrics, mixed provenance of baseline results weakens the evidential standard unless all methods are rerun under a unified pipeline. The paper should more clearly specify, across all tables, which results are inherited vs. rerun and under what matching setup.
- **The conditional-generation claim is broader than the evidence shown.** Section 2.5 frames classifier-free guidance as a general conditional generation extension, but the experimental evidence in Table 4 is limited to one narrow imputation formulation: treating the target label/response column as missing and conditioning on the rest. That is a useful use case, but it is narrower than arbitrary missing-column imputation or broader conditional tabular generation.

### Minor
- **Some methodological choices remain heuristic from the main text.** The power-mean schedule for numerical features and \(1-t^{k_j}\) categorical schedule are reasonable but only lightly justified beyond “flexibility and robustness.” More explanation or analysis of why these restricted families are appropriate would strengthen the contribution.
- **The stochastic sampler description is somewhat unclear in notation.** In Algorithm 2, the network is queried with \(\mu_\theta(\mathbf{x}_t, t^+)\) while transitions are performed from \(\mathbf{x}_{t^+}\) in some places. This is likely fixable presentation-wise, but the sampler is novel enough that exact state usage should be unambiguous.
- **The paper does not analyze the learned schedules themselves.** Since feature-wise schedules are a core claim, it would be valuable to show what \(\rho_i\) and \(k_j\) become across features or datasets and whether they correlate with feature difficulty, sparsity, or cardinality.
- **The imputation comparison set is limited.** Table 4 compares only against TaBSYN and XGBoost, not the broader baseline set used for unconditional generation, so the claim of superior conditional generation is only partially supported.
- **There is no efficiency or sampling-cost comparison.** This is not fatal, but it matters for practical relevance: the stochastic sampler adds extra forward perturbation at every reverse step, and continuous-time diffusion methods are often slower than latent-space or simpler generative baselines.

### Trivial
- **There is a small dataset-accounting inconsistency in Section 4.1.** The text lists “Adult, Default, Shoppers, Magic, Faults, Beijing, News, and Diabetes,” i.e., eight datasets, while the main result tables show seven and omit Faults without explanation. This should be clarified.

## Nice-to-Haves
- Add per-dataset ablations for learnable schedules and the stochastic sampler, not just averaged results.
- Provide plots or tables of learned \(\rho_i\) and \(k_j\) values to support the mechanistic claim about feature heterogeneity.
- Include a simple runtime / memory / sampling-speed comparison against TaBSYN and TabDDPM.
- Expand conditional-generation evaluation to arbitrary missing-column patterns, not just target-column imputation.
- Clarify whether the reported variance comes only from repeated synthetic sampling or also from repeated model training across seeds.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The main claim cannot be verified because appendices are not present here.”** Removed because this is an artifact of the review setting, not a flaw in the paper itself. The paper explicitly states that additional metrics are in the appendix.
- **Criticism based on boldface inconsistencies in Table 1.** Removed as a formatting/parser artifact rather than a substantive research issue.
- **Claims questioning the existence/release status of concurrent or cited systems.** Removed by policy.
- **Complaints about missing related work not verifiable from the paper alone.** Removed by policy. I therefore do not treat the lack of direct comparison to specific external papers beyond those already cited as a formal weakness unless the paper itself foregrounds them.
- **Generic demand for formal theory proving continuous-time superiority over discrete-time/latent-space methods.** Weakened/removed as a field-nonstandard demand for an empirical generative modeling paper; the absence of such proof is not a core flaw.

## Novel Insights
The strongest signal in this paper is not generic “better tabular generation,” but specifically **better preservation of inter-column dependency structure**. The Trend results are more convincingly strong than the utility results, suggesting that TABDIFF’s native mixed-type formulation may be most valuable when correlation fidelity matters more than pure downstream-label utility. At the same time, the paper’s own evidence implies that the gain likely comes from a combination of choices—native-space mixed diffusion, schedule learning, and stochastic correction—rather than from the learnable schedules alone, which is important for how the contribution should be framed.

## Suggestions
- Tone down the headline claims: say the method is **state-of-the-art on average and especially strong on correlation fidelity**, rather than implying uniform superiority everywhere.
- Make the comparison protocol fully explicit for every table: which baselines were rerun, which were inherited, and whether preprocessing/splits/evaluation were matched.
- Strengthen the ablation on learnable schedules by adding:
  - per-dataset breakdowns,
  - alternative fixed schedule families,
  - global-vs-feature-wise learning,
  - and visualizations of learned schedule parameters.
- Clarify Algorithm 2 so the state passed to the denoiser and the state transitioned by the sampler are unambiguous.
- Reframe Section 2.5/4.3 more narrowly unless broader conditional-generation experiments are added.
- Add a concise efficiency table with training time, generation time, and step count for key baselines.

## Score and Decision
**Originality:** Moderate. The paper is not radically new—its ingredients build on known continuous and discrete diffusion ideas—but the mixed-type native-space integration plus feature-wise schedule learning is a meaningful contribution for this problem setting.

**Importance of research question:** High. High-quality tabular synthesis is practically important, and modeling heterogeneous mixed-type data remains challenging.

**Whether the claims are well supported:** Moderately supported. The paper clearly supports that the method is strong, especially on Trend, but some stronger claims of broad and consistent superiority are overstated relative to the visible evidence and the comparison protocol.

**Soundness of experiments:** Good but not airtight. Breadth is solid, but the baseline provenance issue and limited ablations reduce confidence in the strongest conclusions.

**Clarity of writing:** Good overall, with some local notation/sampler clarity issues.

**Value to the research community:** Good. Even if somewhat incremental, the method seems practically useful and likely relevant to future tabular diffusion work.

### Calibration
I calibrated against:
- **CDTD** (`QPtoBPn4lZ.md`, scores 6/5/5/6, accepted poster): very similar topic and similar pattern of strengths/weaknesses—strong empirical results, some novelty concerns around mixing existing diffusion ingredients, and feature-adaptive schedules. TABDIFF appears somewhat stronger empirically in the visible results, but also somewhat overclaims; this places it around or slightly above CDTD rather than far above it.
- **TabSyn** (`4Ay23yeuz0.md`, scores 8/6/8/5, accepted oral): a stronger anchor on the high end. That paper was also somewhat incremental, but reviewers perceived the empirical package and practical advantages as stronger and more decisively acceptance-worthy. TABDIFF does not feel as cleanly above-bar as TabSyn.
- **TabDAR** (`kkGIbmpCHU.md`, scores 5/3/5/6, rejected): a lower anchor with strong empirical results but more unresolved methodological/presentation concerns. TABDIFF is clearly stronger and better supported than that rejection case.
- **TabUnite** (`Zoli4UAQVZ.md`, mostly 3–5, rejected/withdrawn): a lower anchor where novelty and analysis were judged insufficient despite experiments. TABDIFF is materially better than this.

Relative to these anchors, this paper lands in the **weak accept / borderline accept** zone: stronger than rejected incremental tabular-generation papers, but not as convincing as the strongest accepted ones.

**Final score: 6.5 / 10**  
**Decision: Accept**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>