=== CALIBRATION EXAMPLE 34 ===

# Harsh Critic Review
Now I have enough material to write a thorough review. Let me compose my analysis.

---

## Section-by-Section Critical Review

### Title & Abstract

The title claims "Frequency-Aware Channel-Guided Multivariate Time Series Forecasting," yet the paper's own acronym expansion is **F**requency-**A**daptive **C**omplex **T**ransformer—a mismatch that is not a parser artifact. More substantively, the abstract concludes that FACT "achieves **competitive** performance," which is conspicuously weak language for a conference paper and turns out to be an accurate characterization (see Table 1 analysis below). The abstract also claims the method "effectively balances accuracy against regularization constraints for both computational efficiency and interpretability," yet Table 4 shows the full fusion variant is consistently *worse* in MSE than the simpler concatenation baseline on every dataset. These tensions between the framing and the actual results run throughout the paper.

---

### Introduction & Motivation (Sections 1–2)

The motivation for operating in the frequency domain and distinguishing amplitude from phase is physically intuitive and well-articulated. The "channel–frequency cell" framing is a useful conceptual contribution. However, several issues undermine the stated novelty:

1. **Complex-valued neural networks for time series are not new.** The paper does not engage with the literature on complex-valued deep learning (e.g., Trabelsi et al., "Deep Complex Networks," ICLR 2018) to explain what FACT adds beyond applying known complex-NN recipes to MTS forecasting.

2. **The CI/CD dichotomy framing is overstated.** Several recent baselines already occupy the middle ground (SOFTS via soft cluster routing, CARD via channel alignment), yet the paper positions FACT as uniquely solving a problem those methods "still" leave open without empirical evidence that the specific gap it closes is responsible for its performance differences.

3. Bullet three of the contributions states FACT "yields consistent improvements across datasets compared to raw-channel mixing." But the main table (Table 1) shows SOFTS consistently outperforming FACT—often by large margins—and FACT's improvement over raw channel methods is inconsistent across datasets (e.g., Traffic, ETTm2).

---

### Methodology (Section 4)

**Dynamic Frequency Band Decomposition (Section 4.2):** The Gaussian filter design is reasonable and the soft-mask idea is sound. However, the normalization scheme (Eq. 13 in Appendix F.3) raises a question: each band's filter is normalized by the sum across all bands at each frequency, which ensures partition-of-unity but means the filters are not truly "Gaussian" in the usual sense—their shapes will distort depending on how many other bands overlap. The paper should clarify whether learnable (µ, σ) parameters can collapse to near-identical filters, and whether any regularization prevents this degeneracy.

**Channel Prior Mixer (Section 4.3):** Equation 5 (the mixing matrix M_mix) is partially garbled in the text—the softmax argument is presented but the left-hand assignment appears detached. More critically, the coherence γ = Corr(|X_fft|) is computed from the **input** spectrum, meaning the mixing matrix is data-dependent but **not** learned jointly in an end-to-end sense beyond the α, β, τ scalars. This is essentially a gating mechanism informed by fixed batch statistics, not a learned representation of channel interactions. The claim that this provides "physical ground truth" is therefore question-begging: if the priors are derived from the input, the model cannot discover non-trivial structure beyond what amplitude correlations already encode.

**Feature Alignment (Section 4.5):** The complex cross-attention in Eq. 8 queries the raw spectrum against multi-band tokens. It is unclear why the raw spectrum (length F) querying compressed tokens (length K, where K=128 < F in most datasets) does not lose information. A discussion of how the compression ratio affects results is absent.

**Interpretability Regularization (Section 4.7):** L_coh constrains learned correlations γ̂ to match input amplitude correlations γ. This is a consistency loss, not an interpretability guarantee—the model is pushed to agree with the input statistics rather than to discover genuinely causal structure. L_phase similarly constrains learned phase differences to match input phases. There is no theoretical or empirical argument that these constraints improve generalization rather than simply preventing the model from ignoring its own priors (which would arguably indicate the priors are unhelpful).

---

### Experiments & Results (Section 5)

**Main Performance (Tables 1 & 2):** This is the most serious weakness. On the 8 long-term datasets:
- SOFTS wins **16 MSE / 23 MAE** first-place slots; FACT wins **3 MSE / 2 MAE**.
- FACT ranks 2nd or 3rd on several datasets but is often beaten by PatchTST, iTransformer, or TiDE—methods much simpler and older than FACT.
- On ETTm2 FACT averages 0.298/0.340 vs. SOFTS 0.283/0.327—a non-trivial gap.
- On Traffic, FACT (avg MSE 0.453) lags SOFTS (0.388) substantially.

The claim in the intro that FACT delivers "consistent gains across backbones" is not supported relative to the strongest baseline (SOFTS), only relative to weaker ones. Several relevant spectral methods cited in the related work—FreTS, FredFormer, TimeMixer, TimeMixer++—are **entirely absent from the comparison tables**. TimeMixer++ (ICLR 2025) and TimeFilter/TQN (ICML 2025) are mentioned in related work but not benchmarked, which is a significant omission for a paper positioning itself at ICLR 2026.

**Ablation Study (Table 4):** This is the most damaging finding in the paper. On every dataset:

| Config | Weather MSE | Solar MSE | Electricity MSE |
|---|---|---|---|
| FACT (concat) | **0.737** | **0.501** | **0.453** |
| FACT (fusion) | 0.783 | 0.523 | 0.468 |

The proposed Adaptive Fusion mechanism is **consistently outperformed by plain concatenation**. The paper's response is to emphasize runtime efficiency (10.23s vs. 58.55s on Electricity), which is legitimate but does not overcome the core issue: the method's featured architectural innovation hurts prediction accuracy. Furthermore, these ablations are conducted on a 4,096-sample **subset**, not the full datasets—this makes the results non-representative. The main results in Table 1 apparently use the full data and the fusion variant, meaning the comparison between FACT (concat) and the reported main results is apples-to-oranges: we do not know whether FACT (concat) with full data would also outperform the reported FACT (fusion) numbers.

**Regularization Sensitivity (Table 5):** Only **two** values of λ are tested (0.01 vs. 0.02). This is insufficient to characterize sensitivity. The conclusion that "stronger enforcement helps generalization" from two points is statistically unjustifiable.

**Backbone Generalizability (Table 6):** The claim of consistent backbone-agnostic gains is tested on only **2 datasets** (Electricity, Solar), the two datasets where FACT performs best. The claim would be more credible with ETTm2 or Traffic, where FACT is weaker.

**Lookback Window:** L=96 is used throughout (following SOFTS), but many competitive methods benefit from L=336 or L=512. There is no experiment exploring whether FACT's frequency decomposition advantage changes with longer history, which seems directly relevant to the method's core claims.

**Statistical Significance:** No variance across runs, no confidence intervals. Some of FACT's claimed improvements are in the third decimal place of MSE and may not be statistically meaningful.

---

### Interpretability Claims (Section 5.4–5.5)

The paper claims "positive correlation between interpretability metrics and accuracy" (Introduction, paragraph 4) and cites Section 5 as validation, but Section 5.4 provides only three qualitative visualizations on the Solar dataset. There is no quantitative evaluation of interpretability quality, no user study, no comparison against other interpretable baselines (CARD provides channel alignment penalties; SOFTS uses a soft global core). The coherence map "aligning with geographical proximity" is asserted without a ground-truth geographic distance comparison. These are suggestive but not rigorous evidence.

---

### Limitations & Broader Impact (Section 6)

The paper acknowledges quadratic complexity, which is honest. However, it does not acknowledge the more pressing issue surfaced by the ablation: that the fusion mechanism hurts performance. The limitation of only testing on standard benchmarks—without distribution shift, missing data, or non-stationary conditions—is not discussed. Given that the method claims robustness via frequency disentanglement, testing on corrupted or out-of-distribution inputs would be valuable.

---

### Writing & Clarity

The paper is mostly readable. However, the placement of Table 6 (backbone generalizability, referenced in Section 5.6) after the conclusion on pages 11–12 is structurally confusing. Section 5.6 references numbers that appear nowhere nearby. The description of the mixing matrix (Eq. 5) as formatted in the paper is incomplete without context, making the Channel Prior Mixer difficult to follow without consulting the appendix. Several claims in the body (e.g., "approximately 1.3%–2.1% sMAPE reduction" from Gaussian vs. fixed bands) are not backed by a dedicated table, only mentioned in passing.

---

### Overall Assessment

FACT addresses a legitimate and well-motivated problem—moving inter-channel interaction from the raw time domain to fine-grained frequency components with explicit phase modeling. The technical ingredients (complex-valued operations, Gaussian adaptive band decomposition, prior-guided gating) are coherent and the code is released. However, the paper faces three compounding problems that, in combination, do not meet the ICLR bar. First, empirical performance is modest: SOFTS, a simpler model from 2024, outperforms FACT by a wide margin across most benchmarks, and numerous recent frequency-domain methods (TimeMixer++, FredFormer, FreTS) are omitted from comparison entirely. Second—and most critically—the ablation in Table 4 shows that the key proposed module (Adaptive Fusion) consistently degrades MSE relative to simple concatenation, conducted on a non-representative data subset; the paper pivots to an efficiency narrative without resolving this accuracy regression. Third, the interpretability claims are qualitative and unverified quantitatively. The contribution is real but insufficiently validated: the paper would require substantially expanded baselines, a credible reconciliation of the concat vs. fusion accuracy gap on full data, and stronger evidence of the phase-modeling hypothesis to be ready for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes FACT (Frequency-Adaptive Complex Transformer), a method for multivariate time series forecasting that operates in the frequency domain using complex-valued representations to model channel-specific magnitude and phase priors. The framework utilizes Adaptive Band Decomposition for signal disentanglement and a complex-valued mixer to guide interactions, aiming to balance accuracy, efficiency, and interpretability. Experimental results on standard benchmarks demonstrate competitive performance and highlight the method's plug-in nature across different backbones.

### Strengths
1.  **Physically Grounded Interaction Mechanism:** Unlike standard real-valued attention, FACT explicitly models amplitude coherence ($\Gamma$) and phase offsets ($\Phi$) using complex numbers. This addresses a critical physical nuance (temporal alignment/phase shift) often ignored in neural baselines, offering a stronger inductive bias for periodic data (Sec 4.3).
2.  **Adaptive Frequency Decomposition:** The introduction of Dynamic Frequency-Band Decomposition (DynFBD) using learnable Gaussian filters avoids the rigidity of fixed band thresholds. This allows the model to automatically focus on salient frequencies, as evidenced by the performance drop when removing this module in ablation studies (Sec 4.2, Table 4).
3.  **Reproducibility and Extensibility:** The authors provide an anonymous code repository and detailed reproducibility checklists (Appx H, I). Furthermore, the design is presented as a backbone-agnostic plug-in (Transformer/MLP/Linear), with evidence of efficiency gains (up to 2.3x speedup) when paired with lighter backbones (Table 6).
4.  **Interpretability Focus:** The paper includes regularization losses ($L_{coh}, L_{phase}$) to enforce physical constraints, moving beyond "black-box" predictions. Visualizations of gating trajectories and coherence maps provide evidence of this transparency (Fig 6, Sec 5.4).

### Weaknesses
1.  **Inconsistency Between Claims and Results:** The text claims "distinct superiority on periodic datasets (Solar-Energy, Weather)" (Sec 5.2), but Table 1 shows lower MSE for competing models in several cases (e.g., Solar Avg MSE: 0.265 for FACT vs 0.233 for SOFTS). Additionally, the "Count (1st)" summary indicates competing methods (SOFTS, iTransformer) achieve more first-place rankings than FACT, suggesting the empirical evidence for superiority is not robust across all metrics.
2.  **Scalability in High-Dimensional Settings:** While the method claims efficiency, performance on high-channel datasets like Traffic (862 channels) is not superior to baselines (Traffic Avg MSE: 0.453 for FACT vs 0.428 for SOFTS). The text acknowledges this as an area for "further scaling" (Sec 6), which weakens the current empirical contribution for large-scale industrial use cases.
3.  **Complexity Analysis vs. Reality:** The theoretical complexity in Table 3 suggests efficiency ($O(n \cdot heads \cdot K \cdot d^2)$), but the ablation study notes a specific implementation (FACT-concat) required reduced batch sizes for memory. While the core method is claimed to be efficient, the overhead of complex operations and spectral decomposition warrants more critical analysis regarding wall-clock time versus training stability on massive datasets.
4.  **Novelty of Frequency Domain Application:** Frequency-domain approaches (e.g., FEDformer, TimesNet) are well-established by now. While FACT adds complex phase modeling, the core paradigm (spectral decomposition) is not fundamentally new. The incremental gains over existing frequency-enhanced models (like SOFTS, which shows superior rankings in the table) need clearer differentiation from prior work.

### Novelty & Significance
*   **Novelty:** The specific adaptation of **complex-valued** representations for **magnitude-phase priors** in MTS interaction modeling is a meaningful improvement over real-valued spectral methods. However, the general framework of frequency-domain decomposition is not novel.
*   **Significance:** The emphasis on **interpretability via physical regularizers** aligns well with current trends towards trustworthy AI. If the method proves robust for phase-sensitive data, it could significantly improve forecasting reliability in domains like energy and climate.
*   **Overall:** The paper presents a solid engineering contribution with a clear physical motivation, but the empirical advantage over state-of-the-art frequency methods (like SOFTS) is ambiguous based on the provided tables.

### Suggestions for Improvement
1.  **Clarify Empirical Claims:** Reconcile the text claiming superiority on Solar/Weather with the Table 1 data where SOFTS achieves lower average MSE. If the asterisks in Table 1 denote FACT's best performance, the numerical values (0.265 vs 0.233) contradict this. Please clarify the metric definitions or provide the missing context (e.g., different horizon averages) to support the "superiority" claim.
2.  **Strengthen Interpretability Quantification:** Beyond heatmaps, provide quantitative metrics (e.g., correlation between learned coherence matrices and ground-truth spatial graphs or physical constraints) to validate that the interpreted features actually map to real-world phenomena.
3.  **Expand Scalability Analysis:** Provide detailed runtime comparisons on high-dimensional datasets (Traffic/PEMS) comparing FACT against efficient baselines. Clarify if the "2.3x speedup" holds at scale without memory overflow issues mentioned in the ablation text.
4.  **Refine Baseline Comparison:** Ensure comparisons with recent frequency-domain architectures (e.g., TimeMixer++, FEDformer) are consistent and up-to-date in the main table, as the paper acknowledges these works in Related Work but the main results table formatting is confusing regarding which specific versions are compared.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Compare against a real-valued frequency baseline (e.g., FreTS or FITS) with the identical architecture to isolate whether the "complex number" component actually contributes to performance or if gains come solely from frequency decomposition.
2. Add synthetic noise robustness experiments (e.g., Gaussian noise injection) to empirically validate the claim that the method "disentangles meaningful signals from noise" better than time-domain counterparts.
3. Report runtime and memory comparisons against efficient SOTA models (DLinear, PatchTST) rather than internal ablation baselines, as the current efficiency claims are meaningless without external context.
4. Include statistical significance tests (e.g., Diebold-Mariano) for all main results, as the reported MSE differences are often marginal and could be due to random variance.

### Deeper Analysis Needed (top 3-5 only)
1. Quantify interpretability using a metric (e.g., correlation between learned attention weights and known physical dependencies) instead of relying solely on qualitative heatmaps which are subjective.
2. Analyze failure cases on non-periodic datasets (e.g., Traffic) where frequency assumptions may not hold, explaining why performance degrades compared to Channel-Independent models.
3. Provide a sensitivity analysis for the regularization weight $\lambda$ across all datasets, not just Weather, to prove the physical constraints generalize beyond a single domain.

### Visualizations & Case Studies
1. Visualize predicted vs. ground truth phase shifts for specific channels to prove the model actually learns temporal alignment rather than just amplitude matching.
2. Show the dynamic evolution of the Gaussian filter parameters ($\mu, \sigma$) over training steps to demonstrate that the "Adaptive Band Decomposition" actually learns dataset-specific frequencies.

### Obvious Next Steps
1. Address the admitted quadratic complexity bottleneck for high-dimensional channels (e.g., Traffic) by integrating sparse attention or low-rank projection before the interaction module.
2. Validate the "physical grounding" claim by comparing the learned channel coherence matrix $\Gamma$ against known physical graph structures (e.g., sensor proximity in traffic networks).

# Final Consolidated Review
## Summary
FACT proposes operating in the frequency domain with complex-valued representations to explicitly model both magnitude coherence and phase offsets for multivariate time series forecasting. The method uses Dynamic Frequency Band Decomposition (DynFBD) with learnable Gaussian filters, a Channel Prior Mixer for physics-guided channel interaction, and a backbone-agnostic architecture that can be paired with Transformer, MLP, or Linear encoders.

## Strengths
- **Physically-grounded complex-valued modeling:** FACT explicitly represents amplitude (energy intensity) and phase (temporal alignment) separately, addressing a real limitation of real-valued spectral methods that collapse phase information. This is particularly relevant for periodic phenomena (solar irradiance, weather) where lead-lag relationships matter.

- **Adaptive frequency decomposition:** The DynFBD module uses learnable Gaussian filters (Eq. 13 in Appendix) to isolate salient frequency bands adaptively rather than relying on fixed thresholds. Table 4 shows removing DynFBD degrades performance, confirming its contribution.

- **Backbone-agnostic plug-in design:** Table 6 demonstrates the frequency-aware interaction module can be paired with Transformer, MLP, or Linear backbones with only marginal performance drops (e.g., <5% MSE increase on Electricity) while achieving up to 2.3× speedup. This modularity is a practical contribution.

- **Reproducibility:** The paper provides an anonymous code repository, detailed hyperparameter specifications (Table 7), and clear reproducibility checklists (Appendix H, I).

## Weaknesses
- **Empirical performance is not dominant:** Across Tables 1 and 2, SOFTS achieves 16 MSE first-place rankings versus FACT's 3. On Traffic (862 channels), FACT averages MSE 0.453 vs. SOFTS's 0.388—a substantial gap. On Solar, FACT averages 0.265 MSE vs. SOFTS's 0.233. The paper claims "distinct superiority on periodic datasets" but the numerical evidence contradicts this for Solar. This undermines the core claim that the method advances the state of the art.

- **Key proposed module (Adaptive Fusion) hurts accuracy:** Table 4's ablation reveals that FACT (concat) consistently outperforms FACT (fusion) on MSE across all three tested datasets (Weather: 0.737 vs. 0.783; Solar: 0.501 vs. 0.523; Electricity: 0.453 vs. 0.468). The paper emphasizes efficiency gains instead, but the fact that the method's featured architectural innovation degrades accuracy is a fundamental concern that warrants deeper investigation.

- **Ablation conducted on non-representative subset:** The ablation in Table 4 uses only 4,096 samples, whereas main results use full datasets (e.g., Electricity has 26,104 training samples). This makes it impossible to determine whether FACT (concat) would also outperform the reported main results on full data.

- **Missing relevant frequency-domain baselines:** TimeMixer++, FredFormer, and FreTS are discussed in related work but absent from the comparison tables. These are directly relevant methods that also operate in the frequency domain. Their absence makes it difficult to assess FACT's contribution relative to current frequency-based approaches.

- **No statistical significance testing:** MSE differences are often in the third decimal place (e.g., ETTh1 0.451 vs. 0.453), without confidence intervals or standard deviations. Given the variance typical in time series benchmarks, some claimed improvements may not be statistically meaningful.

- **Interpretability claims remain qualitative:** The paper claims "positive correlation between interpretability metrics and accuracy" but Section 5.4 provides only visualizations. There is no quantitative evaluation (e.g., correlation between learned coherence and ground-truth spatial/physical relationships) to validate that the visualized patterns correspond to real phenomena.

## Nice-to-Haves
- Analysis of failure cases on non-periodic or high-channel datasets to understand when frequency assumptions break down
- Statistical significance tests (e.g., multiple seeds, Diebold-Mariano tests)
- Comparison against a real-valued frequency baseline with otherwise identical architecture to isolate the contribution of complex representations
- Extended regularization sensitivity analysis beyond λ ∈ {0.01, 0.02}

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Title acronym mismatch:** The critic notes "Frequency-Aware" vs. "Frequency-Adaptive Complex Transformer" but this is a minor phrasing inconsistency, not a substantive flaw.
- **Demand for complex-valued neural network literature review:** While citing Trabelsi et al. (2018) would strengthen the paper, the complex-valued application to MTS interaction modeling with phase priors is sufficiently novel.
- **Request for user studies on interpretability:** User studies are not standard practice in ML systems papers; quantitative correlation with physical structure would be sufficient.
- **Claim that CI/CD framing is overstated:** The paper correctly positions FACT as a middle ground and provides empirical comparison; the critic's demand for "evidence that the specific gap closed is responsible for performance" is an unrealistic standard of causal proof.
- **Complexity overhead complaint:** The paper accurately reports both theoretical complexity (Table 3) and empirical runtime (Table 4); the concat baseline's memory issues on full Electricity are acknowledged.

## Novel Insights
The explicit separation of amplitude coherence (Γ) and phase offset (Φ) as physically-grounded priors for channel interaction is conceptually valuable. The finding that learnable Gaussian band decomposition provides smooth gradient flow and a closed feedback loop with feature alignment (Figure 4) is an interesting design pattern that could extend to other spectral methods. The backbone-agnostic results suggest that much of the performance comes from the frequency-aware preprocessing rather than the encoder architecture—this has implications for the broader design of time series models.

## Suggestions
- Run the concat vs. fusion ablation on full datasets (not 4,096-sample subsets) to determine whether the fusion mechanism's accuracy penalty persists at scale, or whether the efficiency-efficiency trade-off is justified.
- Add missing frequency-domain baselines (TimeMixer++, FredFormer, FreTS) to the comparison tables; these are directly relevant and their absence weakens the empirical contribution.
- Include standard deviations or confidence intervals across multiple runs to strengthen statistical claims.
- Quantify interpretability by computing correlation between learned Γ matrices and known physical relationships (e.g., sensor distances in PEMS, geographic proximity in Solar) rather than relying solely on visual heatmaps.
- Clarify the experimental protocol: Table 4 should specify whether the same hyperparameters and random seeds were used for both concat and fusion variants to ensure fair comparison.

# Actual Human Scores
Individual reviewer scores: [4.0, 2.0, 4.0, 2.0]
Average score: 3.0
Binary outcome: Reject
