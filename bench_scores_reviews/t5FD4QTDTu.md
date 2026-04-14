## Summary
NoTS (Narratives of Time Series) proposes an autoregressive pre-training framework that reframes time series modeling from next-period prediction to next-function prediction. Rather than slicing a signal into temporal patches, it applies degradation operators (averaging kernels and sinc low-pass filters) at increasing intensity levels to construct a coarse-to-fine sequence of signal variants, and trains a transformer to autoregressively recover the original from the most simplified variant. The paper provides an approximation-theoretic motivation and validates the method across classification, anomaly detection, and imputation tasks on 22 real-world datasets.

---

## Strengths

- **Genuinely novel pretraining objective.** The shift from next-patch to next-function prediction via data-dependent degradation operators is a concrete and creative departure from the dominant patching paradigm. Unlike fixed-basis approaches (Fourier, Koopman), the degradation sequence is data-dependent and requires no pre-specified basis, which is a meaningful design choice.

- **Theoretically motivated.** Theorem 1 demonstrates a concrete failure mode: sampling from the differential operator produces a discontinuous sequence-to-sequence function that standard transformers cannot uniformly approximate. While limited in scope (see Weaknesses), this is a non-trivial negative example that provides honest, falsifiable motivation for the functional sequence construction—more targeted than the typical "patching breaks temporal structure" intuition.

- **Broad empirical evaluation with consistent direction.** Testing across 22 real-world datasets over three distinct task types (classification, anomaly detection, imputation), and demonstrating that NoTS-lw consistently outperforms SimMTM, bioFAME, and next-period prediction in average error rate (15.10 vs. 16.05 best competitor under full fine-tuning) is a substantive empirical contribution.

- **Parameter-efficient adaptation.** The context-aware adaptation pipeline achieving 82% average performance with <1% of parameters trained is a practically significant result for the foundation model use case, and the separation between channel adaptors and task adaptors is a clean design.

- **Ablation confirms component necessity.** Table 3 shows that removing the latent consistency term, AR masking, or cross-augmentation connections each individually degrades performance, validating that the design is not redundant.

---

## Weaknesses

### Fatal
None.

### Major

- **Table 2 "+NoTS" rows are uninterpretable.** The rows for "+NoTS (Ours)" applied on top of PatchTST and iTransformer report values of ~11–16 for classification (where all baselines report ~62–85) and ~1.003 for imputation (where baselines report ~0.11–0.35). These cannot be in the same units as the baseline rows, yet no annotation distinguishes them. The only consistent interpretation is that these rows report relative improvement ratios or normalized metrics, not raw task metrics—but this is never stated. The average error rate column (18.33 and 15.70) does suggest that "+NoTS" improves on PatchTST (21.78) and iTransformer (16.07), but the per-task breakdown is completely unverifiable. This is not a parsing artifact; it prevents any evaluation of whether specific gains are real, large, or uniform across tasks, and directly undermines two of the paper's central claims.

- **Forecasting is entirely absent.** The paper frames NoTS as "a viable alternative for building foundation models for time series" yet does not include forecasting—the most widely benchmarked and practically dominant time series task. Standard long-term forecasting benchmarks (ETTh, ETTm, Weather, Traffic) are widely used in the ICLR time series community. The imputation experiments use ETTm1/2 and ETTh1/2, so the data is already available. Omitting forecasting leaves the "foundation model" claim without empirical grounding in the community's primary evaluation paradigm.

- **Theory-method gap in Proposition 1.** Proposition 1 provides two sufficient conditions for approximability (an expressive constructed sequence, or an expressive tokenizer), but neither is proven nor empirically verified for the specific operators used in NoTS (averaging kernels and sinc filters). The paper states "see Appendix A.3 for an example solution for differential operator" but does not prove that the concrete hyperparameter schedules used in experiments satisfy either condition. The theory motivates that *a* solution could exist in the functional sequence framework, not that NoTS *is* that solution. This disconnect is significant because the theoretical justification is listed as a primary contribution.

### Minor

- **Ablations restricted to one synthetic task.** Table 3 ablates only the H-index synthetic regression task. Given that the paper's main contribution is a broadly applicable pretraining method, key design choices—number of degradation levels K, kernel parameter schedule {p_k}, local-only vs. global-only degradation, and the latent consistency weight—should be ablated on at least one real-world downstream task. The current ablations do not allow readers to understand sensitivity to the most critical hyperparameters.

- **Eq (3) loss directionality is unexplained.** The training loss minimizes L_recon(S'_{k+1}, S_k): the *predicted* less-degraded representation S'_{k+1} is trained to match the *actual more-degraded* signal S_k. This is the opposite of the naïve expectation (predict the next step, compare to the next step's ground truth). This may be an intentional design for stability, but it is never explained. Because the autoregressive direction and the reconstruction target are central to understanding what the model learns, this needs explicit justification or clarification, and should not be left as a possible notation error.

- **"26% improvement" in the abstract is selectively scoped.** Looking at Table 1, the improvements range from 0.98% to 37.80%. The 26% figure is the average over the three fBm features (37.80%, 8.41%, 31.44%). The abstract states this without qualification ("synthetic feature regression experiments"), implying it covers all experiments. Since SSC and WAMP are discontinuous sequence-to-sequence functions by design—exactly the failure mode the method is built to address—improvements on these synthetic features are expected; the real question is whether this translates broadly, which the abstract's framing obscures.

- **Theorem 1 is a negative example, not a general impossibility result.** The proof constructs two specific sequences X_1, X_2 from g_M(t)=sin(Mt)/M under adversarially matched sampling intervals. The result holds for that specific construction; the abstract and introduction generalize it to "sequences of time periods" broadly. Sampling with typical fixed intervals and common data distributions need not trigger this failure mode. The paper should be clearer that this is a sufficiency-style negative example illustrating a potential risk, not a proven limitation of all patching-based transformers.

- **Computational overhead not analyzed.** Constructing K degraded variants expands the transformer's input sequence length by a factor of K (both local and global smoothing variants). No training time, memory, or FLOPs comparison is provided against SimMTM, MAE, or PatchTST baselines. Scalability claims require knowing the cost.

### Tiny

- The degradation hyperparameters {p_k} are described as "selected as hyperparameters" without the specific values used in experiments, which impedes reproducibility.
- "Context-aware adaptation" overstates the mechanism slightly; what is adapted is prompt tokens and channel embeddings, not a context-conditioned model.

---

## Nice-to-Haves

- Include long-term forecasting benchmarks (e.g., ETTh1/2, Weather, Traffic) to substantiate the "foundation model" framing and enable comparison with the broader literature.
- Provide a deeper theoretical or empirical comparison to cold diffusion / deterministic degradation models, rather than treating this as a future direction after a one-line ablation.
- Add proof or empirical verification (e.g., measuring Lipschitz constants or reconstruction error as a function of k) that the specific averaging/sinc operators satisfy Proposition 1's continuity condition for the task operators encountered in practice.
- Include failure case visualizations — examples where NoTS's coarse-to-fine structure collapses or underperforms — to help characterize when the inductive bias is appropriate.
- Ablate the number of degradation levels K and the kernel schedules {p_k}; even a 2×2 grid on a real-world dataset would significantly strengthen confidence in hyperparameter robustness.
- Scale to larger model sizes (>10M parameters) with more data to provide a meaningful power-law analysis; the current 4-point curve over 127k–2.1M parameters is insufficient to claim power-law behavior with confidence.

---

## Removed Points

*These points are flagged for removal; treat with caution.*

- **"Missing related works" criticisms** (from harsh critic): Per reviewer instructions, claims about missing citations are removed as we cannot verify external sources.
- **"No broader impact discussion"** (harsh critic): This is a structural/formatting complaint, not a scientific weakness. Removed.
- **Demand for confidence intervals in Table 2 for the NoTS-lw rows**: The reported results are averaged and the first 8 rows of Table 2 are comparable. Given that the "+NoTS" rows are already uninterpretable for different reasons, adding this as a separate weakness would pile on. Removed as redundant.
- **Comparison with large-scale modern TS foundation models** (Spark Finder, harsh critic): TimesFM, Moirai, Lag-Llama operate at much larger scale with different training regimes. Since NoTS is a pretraining method applied to lightweight models, direct comparison would be asymmetrically unfavorable to the baselines. The paper's comparison scope is appropriate. Removed as scope creep.
- **Strawman framing of prior work** (harsh critic, claiming prior work is unfairly characterized as "naïve chunk concatenations"): The paper does cite multi-resolution, frequency-domain, and decomposition methods (Das et al., 2023; Woo et al., 2024; Liu et al., 2023a; Ansari et al., 2024; Rasul et al., 2023) and acknowledges these are partial solutions. The framing is somewhat rhetorical but not egregiously inaccurate. Removed as a style nitpick.
- **"The channel adaptor is insufficient for heterogeneous semantics"** (harsh critic): The paper does not claim the channel adaptor handles arbitrary semantic heterogeneity; it handles differing channel topologies via a linear mixing layer and reinitialized embeddings. Criticizing its insufficiency for harder cases outside the paper's tested scope is scope creep. Removed.
- **Claims that VQVAE and MAE are disadvantaged by the experimental pipeline** (harsh critic): The paper explicitly states all methods use "the same architecture and pre-training pipeline." If there is a pipeline design advantage to NoTS, it would need specific evidence. As stated, this is speculative. Removed.

---

## Novel Insights

The most thought-provoking observation across the three reviews—not explicitly synthesized in the paper itself—is the following: the relationship between NoTS's deterministic convolution-based degradation sequence and cold diffusion models is more than analogical. Cold diffusion (Bansal et al., 2024) replaces stochastic Gaussian diffusion with arbitrary deterministic degradation operators and learns a denoising AR process—structurally identical to NoTS's AR reconstruction from coarse to fine. The fact that Gaussian degradation underperforms (Table 3, row 4 vs. NoTS) recapitulates findings in audio diffusion suggesting that signal-domain smoothing outperforms noise-domain diffusion for signals with structured spectral content. This implies NoTS might be reinterpreted as a deterministic cold diffusion model operating in representation space, with the practical implication that score-function theory, denoising consistency regularization, and diffusion-based generation could all be imported into the NoTS framework—a more productive framing than treating the connection as a limitation to mention in passing.

---

## Suggestions

1. **Fix Table 2 immediately.** Either present "+NoTS" rows in the same absolute metric units as all other rows, or add a clearly labeled sub-table reporting relative improvement (e.g., Δ over base model). The current presentation makes the benefit on PatchTST and iTransformer empirically unverifiable.

2. **Add at least two standard forecasting benchmarks** (e.g., ETTh1 and ETTm1 long-term forecasting with prediction horizons 96/192/336/720). The data is already used for imputation; the experimental investment is modest. This single addition would substantially validate the "foundation model" framing.

3. **Clarify Eq (3) explicitly.** Add one sentence explaining why L_recon(S'_{k+1}, S_k) targets S_k rather than S_{k+1}, with the intuition (e.g., "predicting the next-level latent that can reconstruct the current-level signal ensures the AR predictions carry sufficient information about intermediate functional components while preventing trivial solutions").

4. **Add a single ablation table on one real-world dataset** varying K (e.g., K ∈ {2, 3, 5, 7}) and the degradation type (local only, global only, local+global). This would address the deepest reviewability gap at relatively low experimental cost.

5. **Either prove or empirically certify** that the sinc/averaging degradation operators create a functionally expressive sequence satisfying Proposition 1's continuity condition for at least the tasks studied, or reframe the theory as motivational rather than justificatory.

6. **Report training time and memory** for NoTS vs. baseline methods (even as a single row in the appendix) to allow readers to contextualize the K-factor overhead.

---

**Overall assessment:** NoTS introduces a genuinely novel and well-motivated pretraining perspective. The core idea—AR learning over functional degradation sequences—is specific, testable, and backed by theory and ablations. However, Table 2's presentation failure is severe enough that the two most novel experimental claims (NoTS on top of PatchTST/iTransformer) cannot be verified from the main paper. Combined with the absence of forecasting benchmarks and the theory-practice gap in Proposition 1, the paper as submitted sits below the bar for a confident accept. The contribution is strong enough that it would likely clear the bar after targeted revisions, but those revisions address substantive scientific communication failures, not cosmetic issues.