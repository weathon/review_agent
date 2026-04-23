## Summary

The paper introduces LNE-Blocking, a two-part method for (1) detecting data contamination in LLMs using Length Normalized Entropy (LNE) and (2) mitigating contamination effects during evaluation by adaptively blocking the top-predicted token at the first *n* positions, where *n* is determined by the LNE score. The method requires only two forward passes (vs. 50 samples for the prior SOTA TED), achieving a claimed 25x speedup with competitive or better mitigation performance across code generation and arithmetic reasoning tasks.

## Strengths

- **Strong improvement on the hardest detection setting (mild contamination):** LNE achieves F1=0.775 on Mild Contamination (Table 1), significantly outperforming Min-k% Prob (0.706), Perplexity (0.627), and CDD (0.648). Mild contamination is the most practically important regime since it is hardest to detect and most realistic, making this a meaningful advance.

- **LNE-Blocking succeeds where TED catastrophically fails on heavily contaminated models:** On Llama 3.1 GSM8K Heavy Contamination (Table 3), TED produces PG=0.694 (essentially failing) while LNE-Blocking achieves PG=0.065. Similarly, on CodeLlama Heavy Cont. (Table 2), LNE-Blocking PG=0.045 vs TED PG=0.137. This is because sampling-based methods cannot produce diverse outputs when the memorized answer dominates the distribution, while blocking deterministically disrupts it.

- **25x computational efficiency over TED:** LNE-Blocking requires only two forward passes (one greedy decoding to compute LNE, one with adaptive blocking), as shown in Algorithm 1, versus 50 sampling iterations for TED. This is a clear practical advantage for deployment scenarios.

- **Adaptive blocking intensity validated as superior to any fixed level:** Table 4 demonstrates that no single fixed blocking count works across contamination levels (Fixed Blocking 1 is best for mild but poor for heavy; Fixed Blocking 3 is best for heavy but over-corrects for mild). LNE-Blocking achieves the best average PG (0.037) by dynamically adjusting intensity via Equation 10, providing end-to-end validation of the LNE→blocking pipeline.

- **LNE outperforms Perplexity as a signal for controlling blocking:** Table 4 shows PPL-Blocking achieves average PG=0.042 versus LNE-Blocking's 0.037, with the gap driven by mild contamination (0.044 vs 0.035). This traces back to LNE's superior detection at mild levels (Table 1).

- **LNE requires neither ground truth nor multiple sampling for detection:** Unlike Perplexity (needs ground truth) and CDD (needs multiple samples), LNE uses only the output probability distribution from a single greedy inference, making it more practical for real-world deployment where ground truth is unavailable.

## Weaknesses

### Fatal
None.

### Major

- **All experiments rely on artificial contamination simulation, limiting generalizability claims.** The contamination is simulated via LoRA fine-tuning on test data. For 3/4 code models (Llama 2, CodeLlama, CodeGen), the paper reuses TED-provided weights with a 1:1000 data ratio, which is somewhat realistic. However, for Llama 3.1 (HumanEval) and both models on GSM8K, the paper fine-tunes for 20 epochs directly on test data (Section 5.2), which produces concentrated, localized memorization—precisely the regime where entropy-based detection and top-token blocking work best. Real contamination, where test data appears once as a negligible fraction of a trillion-token pre-training corpus, may produce subtler probability shifts that neither lower entropy dramatically nor concentrate memorization at the top token. While this is a field-wide challenge (TED uses the same simulation), the paper's claims of "obvious SOTA" and robust generalizability are not established without at least testing on one case of naturally occurring contamination or a more realistic simulation (e.g., injecting test data into a large pre-training mix at low concentration).

- **The design choice of blocking only the first *n* tokens lacks justification and may be suboptimal for many contamination scenarios.** Equation 9 defines blocking at positions 1 through *n*, and Section 4.3.1 states "Starting from the first token, the blocking operation is applied *n* times." No theoretical or empirical justification is provided for why memorization should be disrupted at the *beginning* of the output. While blocking early tokens can cascade through autoregressive generation (each subsequent token is conditioned on the prefix), this cascading argument is never made or tested. If a contaminated model memorizes an answer that appears in the middle or end of its generation (e.g., the final step of a math solution), blocking the first few tokens may corrupt the prefix and cascade into completely wrong outputs rather than disrupting the memorized content. The paper should compare against blocking at random positions, middle positions, or high-confidence positions to validate this design choice.

### Minor

- **The β=2 justification in Equation 10 is mathematically incorrect.** Section 4.3.2 states "dividing by 2 helps much by making an even distribution within the range of 0 to 1," but LNE ranges from 0 to log(V) (where V≈32K–100K), so LNE/2 does not lie in [0,1] in general. In practice, observed LNE values (Figure 1: ~0.1–0.6) make (1−LNE/β) roughly in [0.7,0.95], which is in [0,1] but not "evenly distributed." Additionally, (1−LNE/β) can yield negative values for high-entropy outputs, requiring implicit clipping that is not acknowledged. The choice β=2 is empirically validated in Table 4, so the method works, but the theoretical justification is wrong and should be corrected.

- **The "obvious SOTA" claim in the abstract is overstated.** For detection, the overall F1 advantage is marginal (0.854 vs 0.839 for Min-k% Prob, Table 1), and LNE actually *underperforms* Min-k% Prob on Moderate and Heavy contamination. For mitigation, the advantage is demonstrated against a single baseline (TED). The improvement on mild contamination detection and on heavily contaminated mitigation is meaningful, but characterizing the overall results as "obvious SOTA" overstates the evidence.

- **Limited ablation study.** Table 4 ablates only on CodeLlama for the code generation task. No ablation on GSM8K or other models is provided, which would strengthen confidence that the adaptive blocking mechanism generalizes across tasks and model families.

- **The original uncontaminated model's performance is not directly reported in Tables 2 and 3.** PG is defined as |E(Y_eva) − E(Y_M_origin)| (Eq. 12), but E(Y_M_origin) must be inferred from the PG values and mitigated performance. Explicitly reporting the original model's performance would make the results easier to verify and interpret.

### Trivial
None.

## Nice-to-Haves

- Testing on at least one model with naturally occurring contamination (e.g., GPT-4 on GSM8K) would substantially strengthen the paper's claims, even as a case study.
- Ablation comparing blocking at different positions (first-n, random-n, high-confidence-n) would validate or improve the first-n design choice.
- Per-position entropy and memorization analysis (visualizing token-level entropy and overlap with ground truth across positions) would reveal whether memorization is concentrated at the beginning of sequences.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"The entire experimental validation uses models contaminated via LoRA fine-tuning on the test set for up to 20 epochs"** — This is factually incorrect for 3/4 code models. The paper explicitly states that for Llama 2, CodeLlama, and CodeGen, it reused TED-provided LoRA weights that mix the HumanEval test set with StarCoder data at a 1:1000 ratio (Section 5.2), which is a more realistic contamination scenario. The 20-epoch claim only applies to Llama 3.1 and the GSM8K experiments. The concern is partially valid but the characterization is inaccurate.

- **"The PG metric confounds mitigation quality with performance degradation"** — While catastrophic forgetting from fine-tuning is a valid concern for the 20-epoch experiments, the relative comparison with TED (which uses the same PG metric and the same contamination simulation) is still meaningful. The PG metric's absolute interpretation has limitations, but the comparison showing TED's failure on heavily contaminated models while LNE-Blocking succeeds is robust.

- **"The human analogy in Section 2 is misleading"** — The analogy is presented as motivational, not as a formal argument. The low Pass@1 scores for some configurations are more relevant to the method's limitations than to the validity of the analogy.

- **"The paper assumes HumanEval is uncontaminated for selected models"** — This assumption follows from the initial publications of these models (Section 5.1) and is standard practice in this line of work. The paper acknowledges GSM8K contamination risk. This is a known limitation, not a novel criticism.

- **"Missing experiments on models with naturally occurring contamination"** — Moved to Nice-to-Haves above. This is a valid concern for strengthening the paper but is extremely difficult to execute in a controlled manner, as contamination levels in real models are unknown. This is a field-wide challenge.

- **"Analysis of where memorized content appears in the output"** — Moved to Nice-to-Haves. This is an important suggestion but requesting it as a required experiment overstates its necessity.

- **"Verification that blocked outputs reflect generalization, not luck"** — While interesting, this goes beyond the paper's stated scope of contamination mitigation evaluation, which measures how closely the mitigated performance aligns with the original model's performance (PG metric).

## Novel Insights

The paper reveals an interesting asymmetry in contamination mitigation: sampling-based methods (TED) fundamentally fail when the memorized answer dominates the probability distribution, because no amount of random sampling can escape a near-deterministic output. Deterministic top-token blocking sidesteps this failure mode entirely by explicitly suppressing the memorized answer. This insight—that detection + deterministic intervention can be more reliable than diverse sampling in heavily contaminated regimes—is the paper's most important conceptual contribution, and it is supported by the dramatic TED failures on heavily contaminated models (PG=0.694 for Llama 3.1 GSM8K Heavy).

## Suggestions

- Add a comparison with blocking at random or high-confidence positions (not just the first *n*) to either validate or refine the current design. This is a simple experiment that would significantly strengthen the paper.
- Add one experiment on a model suspected of natural contamination (even a single case study) to move beyond purely artificial validation.
- Soften the "obvious SOTA" language in the abstract and conclusion to reflect the nuance in the results (strong on mild detection and heavy mitigation, but marginal or underperforming in other regimes).
- Correct the β=2 justification in Section 4.3.2—explain that β=2 is empirically chosen and that practical LNE values happen to fall in a range where the formula produces reasonable blocking counts, rather than claiming a theoretical [0,1] normalization property.
- Report the original uncontaminated model's performance explicitly in Tables 2 and 3 to improve transparency and verifiability.

## Calibration

| Anchor Paper | Avg Score | Comparison |
|---|---|---|
| Min-K%++ (ZGkfoufDaU) | 7.50 | Theoretically motivated, comprehensive experiments across 5 models/2 benchmarks. This paper is weaker: less theoretical grounding, artificial contamination only, limited ablation. |
| Proving Test Set Contamination (KS8mIvetg2) | 7.50 | Strong theoretical guarantees (exact false positive rates), tested on real models. This paper is weaker: no theoretical guarantees, no testing on real contaminated models. |
| Time Travel in LLMs (2Rwq6c3tvr) | 7.00 | Practical detection method tested on real models (GPT-4). This paper is weaker: artificial contamination only, but adds the novel mitigation component. |
| Mitigating Memorization (MGKDBuyv4p) | 7.33 | Comprehensive evaluation of 17 mitigation methods with TinyMem suite, transferable to production models. This paper has narrower scope but a cleaner, more efficient method. |
| Elephants Never Forget (lwtaEhDx9x) | 4.75 | Multiple detection methods but weak validation, unclear methodology. This paper is stronger: cleaner methodology, direct comparison with SOTA, efficiency advantage. |
| BYOD Self-Supervised (zH6zBoktYO) | 4.50 | Interesting approach but weak experimental validation. This paper is stronger: more focused methodology, stronger empirical results. |
| CopyLens (Mez2No9lHj) | 2.33 | Deeply flawed methodology, no evidence the method works for the claimed task. This paper is far stronger: the method demonstrably works within its experimental framework. |

This paper sits between the medium anchors (4.5–4.75) and high anchors (7.0–7.5). It is clearly stronger than the rejected contamination papers (cleaner methodology, SOTA comparison, efficiency advantage) but weaker than the accepted ones (artificial contamination only, less theoretical grounding, limited ablation). The novel mitigation contribution and 25x efficiency gain are meaningful, but the generalizability to real-world contamination remains unvalidated.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>