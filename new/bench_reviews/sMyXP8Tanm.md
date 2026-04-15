Now let me read one of the past reviews in full for calibration:Now I have enough information to write the consolidated meta-review. Let me synthesize.

---

## Summary

This paper introduces RADD (Reparameterized Absorbing Discrete Diffusion), built on a key theoretical observation (Theorem 1): for absorbing discrete diffusion, the concrete score for the only relevant reverse transitions (masked → unmasked) factorizes exactly into an analytic time-dependent scalar times a time-independent conditional clean-data probability. This explains SEDD's previously mysterious "scaling trick" and motivates removing time conditioning from the score network entirely. A caching strategy then naturally follows from time-independence, with the expected NFEs quantified in closed form (Eq. 3.4). Theorem 2 further unifies absorbing discrete diffusion with any-order autoregressive models (AO-ARMs) in the limit σ̄(T) → ∞, producing a chain of four equivalent loss functions. Empirically, all four RADD variants consistently outperform SEDD on average across five zero-shot language modeling benchmarks at GPT-2 scale.

---

## Strengths

- **Theorem 1 is a clean, nontrivial structural insight specific to absorbing diffusion.** The factorization in Eq. (3.2) — that the concrete score at a masked position equals an analytic time-dependent scalar times the clean conditional distribution — is the kind of observation that simultaneously explains a puzzling empirical practice (SEDD's scaling trick) and unlocks a simpler model design. The proof is specific to the absorbing matrix structure (Eq. 2.4) and does not trivially generalize to other noise processes.

- **The E-NFEs formula (Eq. 3.4) is a rigorously derived and empirically validated quantification of caching efficiency.** Fig. 1a shows tight agreement between the theoretical curve and experimental measurements. This is more thorough than concurrent work (Sahoo et al., 2024), which proposed caching empirically without such analysis.

- **Theorem 2 and the four-loss equivalence chain (Eq. 3.7) provide a principled alternative to treating DSE as merely an upper bound.** By showing DSE ↔ t-DCE ↔ λ-DCE ↔ AO-ARM (in the large-T limit), the paper reframes absorbing diffusion perplexity evaluation as an exact AO-ARM NLL rather than an upper bound with unknown gap — a meaningful conceptual advance with practical implications for training loss selection.

- **The ablation between SEDD-Scale and RADD-DSE is appropriately controlled.** Both use the same DSE loss and comparable parameter counts; the only change is removal of time conditioning. The consistent win for RADD-DSE across Tables 1 and 2 (e.g., WikiText2 small: 41.84 → 38.83; PTB medium: 87.12 → 75.16) provides direct empirical support for the theoretical claim that time conditioning is unnecessary in the absorbing case.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Novelty is significantly diluted by concurrent work.** The paper acknowledges that Shi et al. (2024) derived a nearly identical weighted cross-entropy loss and a proposition connecting score and mean parameterizations resembling Theorem 1, and Sahoo et al. (2024) independently removed time conditioning and proposed caching. The paper's unique additions — the explicit scalar factorization in Theorem 1 (simplifying Shi et al.'s conditional expectation to a time-zero distribution), the E-NFEs closed-form analysis, and the full four-loss equivalence chain — are genuine but incremental over a crowded concurrent space. The paper should not be faulted for concurrent discovery, but reviewers and authors should both be clear-eyed that this substantially reduces the novelty score relative to a standalone contribution.

- **The finite-T gap for Theorem 2 is unexamined.** Theorem 2 requires σ̄(T) → +∞ for the four losses to be equivalent. The practical experiments use finite T. The paper does not analyze how large T must be for the equivalence to hold approximately, nor does it characterize how sensitive empirical results are to this limit. Given that Table 1 shows non-trivial differences between theoretically equivalent losses (e.g., RADD-λ-DCE vs. RADD-AO on 1BW small: 72.99 vs. 74.28; RADD-t-DCE vs. RADD-λ-DCE on PTB medium: 78.77 vs. 82.08), the paper attributes these gaps vaguely to "variations in gradient estimation on finite data" without principled analysis. A variance analysis or discussion of why certain losses converge to different local optima would considerably strengthen confidence in the theoretical equivalence's practical relevance.

- **Caching benefits under batched generation are not evaluated.** The E-NFEs formula (Eq. 3.4) assumes a single sequence — each token independently unmasks according to the noise schedule. In batched generation, sequences in a batch unmask at different rates, and a cache hit on one sequence does not help others. The paper includes a brief note on batch size ablations in Appendix J.4, but the main paper does not address how caching efficiency degrades with batch size. Since practical inference uses batching, this is a meaningful gap between the theoretical speedup claim and real-world utility.

### Minor

- **Scale is limited to GPT-2 small/medium, and this is a real constraint.** The paper explicitly acknowledges this limitation. Given that concurrent works (and follow-up masked diffusion scaling papers) suggest the efficiency gap between diffusion and autoregressive models may widen at scale, the GPT-2-scale results leave open whether RADD's advantages persist. This is a genuine limitation, appropriately disclosed but worth emphasis.

- **Fixed-length generation limits practical applicability.** The paper acknowledges (Section 6) that RADD "can only generate full-length outputs, unlike auto-regressive models that can produce variable-length outputs." This is a significant restriction for real deployment. The limitation is honestly stated but no roadmap is offered for resolving it.

- **No direct benchmark against concurrent works in tables.** Tables 1 and 2 do not include Shi et al. (2024) or Sahoo et al. (2024) as baselines, which makes the marginal contribution of RADD over these specific works difficult to quantify empirically for readers.

### Trivial

- The perplexity comparison between RADD and GPT-2 is not perfectly apples-to-apples: RADD-DSE is evaluated using the DSE loss (acknowledged as having an "unknown gap" to true NLL), while GPT-2 reports exact perplexity. The AO-ARM connection (Theorem 2) provides a principled justification for RADD-AO perplexities being exact NLL values, but the DSE/t-DCE variants do not have this clean interpretation at finite T. This is a minor presentational issue; the paper's best-performing numbers on most benchmarks come from RADD-DSE/RADD-AO, and the latter has the cleaner likelihood interpretation.

---

## Nice-to-Haves

- A variance/sensitivity analysis explaining why theoretically equivalent losses converge to different local optima, with guidance on which loss to prefer in practice.
- Batched-generation caching evaluation: report actual cache hit rate and wall-clock time as a function of batch size, to bound the practical speedup.
- At minimum one experiment at larger scale (GPT-2 large or beyond) to test whether the architectural simplification benefits scale.
- A cleaner ablation isolating the effect of removing time conditioning from the simplification of the adaptive layer norm structure (since RADD removes both simultaneously from SEDD-Scale). This is not a fatal gap since removing both is principled under Theorem 1, but it would sharpen the causal argument.

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **"SOTA perplexity claim is not well-supported due to non-comparable evaluation"** (Harsh Critic, Weakness 1): Partially misfires. The paper provides explicit justification in Section 3.3 that the loss functions are valid for likelihood estimation via the AO-ARM equivalence. For RADD-AO specifically, the perplexity is an exact AO-ARM NLL. The concern is more narrowly applicable to RADD-DSE/t-DCE at finite T, and is retained as a Trivial weakness — but the broad framing that "Table comparisons are ambiguous" is too strong given the paper's principled response.

- **"The time-conditioning is unnecessary claim is empirically overstated"** (Harsh Critic, Weakness 4): This is a strawman. The paper's ablation (SEDD-Scale vs. RADD-DSE under the same DSE loss) is a fair controlled comparison. The paper does not claim this generalizes to all absorbing diffusion architectures, only that in this well-matched setting, time conditioning is unnecessary. The claim is appropriately scoped.

- **"Inference speed advantage over AR models not convincingly demonstrated"** (Human Finder, Weakness 5): The paper's sampling comparison in Fig. 1b is explicitly RADD vs. SEDD, not RADD vs. GPT-2. The paper references Fig. 1 of Lou et al. (2024) for the SEDD-vs-AR comparison and does not make new claims about AR comparison. The criticism confuses the paper's scope.

- **"Generated text samples not provided in main paper"** (Spark): The paper states samples are in Appendix K.1. Requesting them in the main paper is a formatting nitpick.

- **"Error bars / multiple seeds required"** (Spark): Single-run evaluation is the norm for GPT-2-scale language modeling benchmarks. The differences between RADD and SEDD-Scale are consistent across five datasets and two model sizes, providing sufficient internal replication.

- **"Training compute (FLOPs/GPU-hours) not reported"** (Spark): A reproducibility detail not standard for reporting in this area at this scale.

---

## Novel Insights

The most underappreciated contribution of this paper is the conceptual reframing it enables for the discrete diffusion literature: the DSE loss, previously treated as an upper bound with unknown gap on NLL (and therefore a somewhat unsatisfying evaluation criterion), can be reinterpreted through Theorem 2 as the exact expected NLL across all orderings of an AO-ARM ensemble. This means the perplexity numbers in prior SEDD papers are not upper bounds but are exact likelihoods under a distributional interpretation — a point with implications for how the whole field evaluates and compares discrete diffusion models. The paper identifies this but does not fully foreground it as its own contribution, focusing instead on training objectives and architecture. Surfacing this reframing more prominently would strengthen the paper's position in the literature.

---

## Suggestions

1. **Quantify the finite-T gap**: Run an experiment varying T and reporting how quickly the four losses converge toward equivalent perplexities. Even a single curve showing RADD-AO vs. RADD-DSE perplexity gap as a function of σ̄(T) would make Theorem 2's practical scope concrete.
2. **Batched caching evaluation**: Report cache hit rate and wall-clock time as a function of batch size (e.g., 1, 8, 32, 128) for the medium model to characterize real-world speedup.
3. **Foreground the NLL reframing**: Explicitly note in the abstract or intro that Theorem 2 resolves the longstanding concern that DSE perplexity is only an upper bound — this is a genuine service to the community.
4. **Add Shi et al./Sahoo et al. baselines to the main tables**, even with a note on training differences, to help readers position RADD's marginal contribution numerically.

---

## Evaluation on Key Axes

- **Novelty**: Moderate. Theorem 1 is a clean and specific insight, but Shi et al. derived a closely related result concurrently. The E-NFEs analysis and full loss-equivalence chain are genuinely original. Overall novelty is real but reduced by the crowded concurrent space.
- **Technical soundness**: Good. Theorems 1 and 2 appear correct, the E-NFEs formula is verified experimentally, and the model design follows cleanly from the theory. The finite-T caveat in Theorem 2 is a notable limitation that should be better characterized.
- **Empirical support**: Moderate. Results are consistent across five benchmarks and two model sizes, which is encouraging. But the scale is limited (GPT-2 only), the caching evaluation is narrow (single sequence, one batch setting), and there is no direct comparison with concurrent works in the tables.
- **Significance**: Moderate-to-good. The theoretical insights are genuinely useful for the absorbing diffusion community. The practical impact (simplified architecture, caching acceleration) is real. Significance is bounded by the GPT-2 scale and the concurrent discovery of similar ideas.
- **Clarity**: Good. The paper is well-organized, the proof roadmap is clearly presented, and limitations are honestly disclosed. The concurrent work section is unusually candid.

---

## Score and Decision

**Calibration against past reviews:**

- **mMPaQzgzAN (6.5, Accept)**: JumpReLU SAE paper with a clean KDE-STE theoretical contribution and solid multi-site empirical validation. RADD has a comparably clean theoretical insight (Theorem 1) and consistent empirical validation, but faces a significantly larger concurrent-work problem (two papers independently discovering the same ideas) and a more limited empirical scope (one scale only vs. multiple models/sites). RADD is *below* this benchmark.
- **1F8xTfv6ah (5.5, Weak Accept)**: KAN OOD paper with genuine empirical contributions but framing issues and base-method failure modes. RADD has a stronger theoretical foundation and fewer framing problems, but comparable scale limitations and a bigger novelty-dilution issue. RADD is *above* this benchmark.

RADD sits between 5.5 and 6.5, closer to 6.0. The theoretical insight is real and contributes to understanding an important model class; the architecture simplification is elegant and well-motivated; the empirical results are consistent. Against this, the concurrent-work situation is substantive, the empirical scope is narrow (GPT-2 only), and the key theorem's practical regime is undercharacterized. This is a clear weak accept: solid work that advances the community's understanding without being a breakthrough.

**Score: 6.0 — Weak Accept**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>