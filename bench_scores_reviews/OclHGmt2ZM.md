---

## Summary

CMamba is a Mamba-based model for multivariate time series forecasting that addresses vanilla Mamba's lack of cross-channel dependency modeling. It introduces three components: (1) M-Mamba, a modified Mamba block tailored for time-series patches that removes the convolution branch, uses a feature-independent transition matrix A, and makes the skip connection D data-dependent; (2) GDD-MLP, a global data-dependent MLP for efficient cross-channel mixing; and (3) Channel Mixup, a training-time augmentation that linearly combines channels within a sample to reduce overfitting in channel-dependent models. Experiments across seven standard forecasting benchmarks show competitive performance with significantly lower computational overhead than attention-based channel-mixing alternatives.

---

## Strengths

- **Demonstrated computational efficiency of GDD-MLP.** Table 5 provides direct FLOPs measurements showing a 0.35–1.35% overhead increase even for the 862-channel Traffic dataset. This concretely supports the efficiency claim over self-attention-based channel mixing, and is a specific, quantified contribution rather than a qualitative assertion.

- **Modular versatility validated across diverse architectures.** Table 4 shows consistent improvement when GDD-MLP and Channel Mixup are inserted into four architecturally distinct models — iTransformer (CD/Transformer), PatchTST (CI/Transformer), RLinear (CI/Linear), and TimesNet (CD/Conv) — with an average ~5% gain. The especially large gains on CI models (PatchTST: 17.8% MSE improvement on Electricity) suggest the modules address a genuine gap in CI architectures. This cross-model portability is a meaningful contribution that distinguishes the paper from pure CMamba-architecture engineering.

- **Insightful diagnosis of Mamba component suitability for time-series patching.** The ablation in Table 2 reveals that vanilla Mamba's convolution branch is indeed redundant in a patch-based regime, and that making the skip-connection D data-dependent provides incremental benefit. While the ablation is narrow (Weather only), the findings are conceptually coherent: patching already provides local aggregation, making conv redundant, and D-dependence addresses within-patch variation across channels. This is a non-obvious, empirically-grounded contribution that differs from simply transplanting Mamba to a new domain.

- **Clear explanation and concrete ablation of Channel Mixup's role.** Table 3 directly demonstrates that GDD-MLP alone degrades performance on Traffic (MSE 0.479 → 0.525) while GDD-MLP + Channel Mixup recovers and surpasses the baseline (0.444). This is an honest and useful empirical finding, and the optimization loss curves in Fig. 4 visually corroborate the overfitting story.

---

## Weaknesses

### Fatal
None.

### Major

- **Absence of Mamba-based baselines in Table 1.** The paper explicitly discusses S-Mamba, Bi-Mamba+, and Time-SSM in the Related Work section and even criticizes S-Mamba for "large computational overhead." Yet none of these appear in the main comparison table. For a paper whose core claim is a *better adaptation of Mamba* for multivariate forecasting, the absence of the closest competitors leaves the most important performance claim (superiority within the SSM family) entirely unsubstantiated.

- **Table 1 highlighting errors that misrepresent the state-of-the-art claim.** From the text-extracted table, CMamba's results are highlighted as best (red) in settings where they are not:
  - ETTh2 MSE: CMamba 0.273 vs. ModernTCN 0.228 — ModernTCN is clearly better, yet CMamba is highlighted red.
  - ETTm2 MSE: CMamba 0.468 vs. TimesNet 0.412 / RLinear 0.422 — CMamba is markedly worse, yet highlighted red.
  The paper claims "top 1 in 65 out of 70 settings" — this count needs careful reexamination. These errors directly undermine the paper's primary empirical contribution claim.

- **Ambiguity in GDD-MLP's cross-channel interaction mechanism.** Eq. 5 applies MLP_1 and MLP_2 to `Pooling(H_t)`, where pooling over the embedding dimension gives descriptors F^l ∈ R^{V×N}. The critical question — whether the MLP processes the full V×N descriptor *jointly* (enabling cross-channel information flow) or operates *independently per (v,n)* entry (reducing it to SE-style per-channel dynamic scaling with no actual cross-channel dependency) — is never made explicit in the paper. If the latter, GDD-MLP does not capture cross-channel dependencies at all, only performs dynamic per-channel modulation, which contradicts the central contribution claim. The paper must clarify the operating dimension of MLP_1/MLP_2 and provide evidence that cross-channel information flows through the weight generation process.

- **Severe GDD-MLP failure on Traffic is under-analyzed.** Using GDD-MLP alone degrades MSE on Traffic from 0.479 to 0.525 (a ~10% regression). The paper attributes this to CD overfitting but provides no further analysis: no train/validation gap comparison, no exploration of whether the failure correlates with channel count (Traffic has 862 channels), and no discussion of when one can expect GDD-MLP to help vs. hurt. This failure mode is significant enough that practitioners cannot safely apply GDD-MLP standalone without Channel Mixup, and the conditions under which it is safe are unknown.

### Minor

- **M-Mamba ablation is limited to a single dataset (Weather).** The three design choices (remove conv, feature-independent A, data-dependent D) are core architectural contributions of M-Mamba. Validating them only on Weather — where effect sizes are tiny (0.240→0.237 MSE) — is insufficient to support the strong architectural claim that "convolution operation and gated z-branch are redundant for time series forecasting." Multi-dataset validation is needed.

- **GDD-MLP equation block (Eq. 11) suggests a sequential layout inconsistent with Fig. 2's parallel illustration.** Eq. 11 feeds M-Mamba output directly into GDD-MLP with only a post-GDD-MLP residual, while Fig. 2(b) depicts them as parallel branches. The actual data flow needs to be unambiguously stated to ensure reproducibility.

- **A ∈ R^3 for M-Mamba is stated but not explained.** The paper says "A ∈ R^3 in M-Mamba" without clarifying whether this means the state size S=3 or a different parameterization. If S=3, this is an unusually small state and the rationale for this choice is missing.

- **No end-to-end runtime or memory comparison with key competitors.** While Table 5 reports FLOPs for GDD-MLP in isolation, the paper never reports end-to-end training/inference time or memory footprint against iTransformer or S-Mamba. Since efficiency is a primary motivation for GDD-MLP over self-attention, this comparison should be direct and system-level.

### Tiny

- **Channel Mixup hyperparameter σ is not discussed in the main paper.** The distribution of λ is Gaussian with mean 0 and std σ, but σ's value and its sensitivity are deferred entirely to the appendix. Even a single sentence noting the default value and its robustness would help assess reproducibility from the main text.

- **Notation inconsistency in Section 3.1.** The paper writes "X_{v,v}" for the sequence of channel v, which should presumably be "X_{:,v}." This is a minor typo but affects readability of the preliminary setup.

---

## Nice-to-Haves

- **Controlled experiment isolating data-dependence vs. global receptive field in GDD-MLP.** The paper argues MLP fails because it lacks *both* data-dependence *and* global receptive field. A 2×2 ablation (local/global × static/dynamic) would directly validate which factor drives performance and would substantially strengthen the paper's explanatory contribution.

- **Sensitivity analysis for Channel Mixup σ.** Since the λ ~ N(0, σ²) can generate negative values and arbitrarily large perturbations, understanding how σ affects performance would help practitioners configure the method and clarify whether it is functioning as true channel mixup or as noise injection.

- **Analysis of GDD-MLP channel dependency patterns.** Weight heatmaps from GDD-MLP across correlated vs. uncorrelated channel pairs (e.g., in ETT where HULL/MULL relationships are explicitly cited as motivation) would provide interpretable validation that the module captures the claimed type of cross-channel dependency.

- **Discussion of limitations in the conclusion.** The paper should acknowledge: (a) semantic risks of Channel Mixup for heterogeneous physical variables; (b) conditions under which GDD-MLP alone may degrade performance; (c) scope of efficiency claims (per-module FLOPs vs. end-to-end cost).

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Eq. 8 contradicts Algorithm 1" (Reviewer 2).** Eq. 8 describes the formula for one virtual channel X' ∈ R^{L×1} resulting from mixing channel i and j. Algorithm 1 implements the vectorized version that applies this to all V channels simultaneously (X' = X + λ * X[:,perm]), yielding shape L×V. These are not contradictory — one is a per-channel formula, the other the vectorized implementation.

- **"Table Traffic MAE shows CMamba=0.645 vs iTransformer=0.262, CMamba highlighted as best."** This is a PDF extraction artifact producing garbled column alignment on the Traffic row. The paper's text explicitly states CMamba performs well on Traffic ("as well as or better than iTransformer"), and the claim about 65/70 top-1 rankings — while needing verification — is likely based on the actual formatted table. This should be verified against the original PDF, not penalized as an authorial error.

- **"GDD-MLP resemblance to SE-blocks overstates novelty" (Reviewer 2).** While the resemblance is real and the paper should acknowledge it, SE-blocks in CNNs operate on spatial feature maps and are not typically applied as cross-sequence channel mixers in temporal/sequence modeling. The application context and specific adaptation (avg+max pooling over patch×embedding for sequence-structured data) are different enough that this is not a fatal novelty concern. It is a reasonable acknowledgment to add, not a critical weakness.

- **"Semantic invalidity of Channel Mixup for heterogeneous channels" (Reviewer 1) — weakened.** The paper explicitly addresses this: channels share temporal characteristics (CI strategy effectiveness is cited), and λ ~ N(0,σ²) with mean 0 ensures the channel's own characteristics are preserved on average. The empirical ablation (Table 3) shows it consistently helps across heterogeneous datasets like Traffic and Weather. The concern is legitimate as a limitation to acknowledge, but does not invalidate the method.

- **"Requesting confidence intervals/variance statistics as a fatal weakness."** Across seven datasets and 70 settings with three runs, requesting standard deviations in the main table is desirable but not standard practice in the MTSF literature at current norms. The lack of variance reporting is a minor concern (especially given the small effect sizes in Table 2), not a major one.

- **"Using baseline numbers from iTransformer weakens comparability"** — weakened. The paper explicitly reruns the three baselines (MICN, TimeMixer, ModernTCN) whose experimental settings differed from iTransformer's. Reusing others from iTransformer under confirmed-identical settings is acceptable practice in the MTSF literature.

---

## Novel Insights

The most genuinely novel observation in this paper — beyond its own stated contributions — is the empirical finding that **vanilla Mamba's convolution branch becomes redundant *specifically because of patching***: patching already provides local temporal aggregation, making the conv1d redundant (and potentially harmful by creating conflicting inductive biases). This insight about the interaction between tokenization strategy and architectural redundancy has implications beyond CMamba — it suggests that researchers transplanting Mamba-style SSMs to other structured sequence domains should revisit which internal components are load-bearing given their specific tokenization. The Channel Mixup failure-then-recovery on Traffic (Table 3: 0.479 → 0.525 with GDD-MLP alone → 0.444 with both modules) also surfaced a useful empirical principle: channel-dependent methods appear to require explicit regularization at high channel counts, and within-sample channel augmentation is one effective form.

---

## Suggestions

1. **Add S-Mamba and Bi-Mamba+ to Table 1.** This is the most critical gap. At minimum, report CMamba vs. S-Mamba on a representative subset of datasets to substantiate the claim of improved Mamba adaptation.

2. **Audit and correct Table 1 highlighting.** Verify the ETTh2 MSE and ETTm2 MSE rankings in the camera-ready version; if the numbers extracted here are accurate (ModernTCN 0.228 < CMamba 0.273 on ETTh2, TimesNet 0.412 < CMamba 0.468 on ETTm2), the highlighting must be corrected and the 65/70 count revised.

3. **Clarify the operating dimension of MLP_1/MLP_2 in GDD-MLP.** Add one sentence (and ideally a shape annotation in an equation) specifying whether the MLP processes the V×N descriptor jointly across channels or per-patch/per-channel. If the intention is joint processing, make this explicit as the source of cross-channel information flow.

4. **Extend the M-Mamba design ablation to at least 3 additional datasets** (e.g., ETTh1, Electricity, Traffic) to validate that removing the conv branch generalizes beyond Weather.

5. **Specify σ for Channel Mixup in the main paper** and include at least a brief sensitivity plot or range, as this is a key hyperparameter of a core contribution.

6. **Explicitly discuss the relationship to SE-Net/Squeeze-and-Excitation** in the related work or methodology, and articulate why GDD-MLP's formulation (avg+max pooling over N×E for temporal sequences) is better suited to the MTSF setting than a direct SE-block application.

---

**Evaluation summary:** The paper addresses a real and timely problem with a computationally efficient design that shows genuine empirical promise. The modular portability result (Table 4) and the efficiency result (Table 5) are its cleanest contributions. However, the paper is held back by: missing Mamba-baseline comparisons that are essential to its core claim, apparent table-highlighting errors that call its performance numbers into question, and insufficient clarity on whether GDD-MLP actually achieves cross-channel information mixing versus per-channel dynamic scaling. Novelty is moderate; technical soundness is adequate but with reproducibility gaps; empirical support is broad in dataset coverage but has notable methodological gaps (narrow ablations, missing baselines). As submitted, it falls short of a confident acceptance at ICLR but has the elements of a solid paper with targeted revisions.