## Summary
This paper investigates whether training Large Language Models using a Prover-Verifier Game (PVG) can improve the legibility of chain-of-thought solutions to grade-school math problems. The authors demonstrate that optimizing solely for correctness degrades human legibility (the "legibility tax"), while their proposed checkability training maintains human evaluation accuracy while improving model accuracy. The paper provides empirical results on verifier robustness, prover dynamics across training rounds, and human evaluations for both helpful and sneaky provers, suggesting that training for checkability against smaller models is a promising avenue for scalable oversight.

## Strengths
- **Clear formalization of checkability:** Section 3.1 cleanly adapts completeness and soundness to a probabilistic LLM setting and links them to a verifier-leading Stackelberg equilibrium via Theorem 1, providing a rigorous theoretical foundation for the iterative game.
- **Empirical mitigation of the legibility-accuracy trade-off:** Figures 1 and 8 demonstrate that the proposed method improves prover accuracy over initialization (~0.57 to ~0.63) while maintaining or improving human accuracy (~0.76 to ~0.84). This stands in sharp contrast to the correctness-only baseline (Fig. 4), where human accuracy drops to ~0.72–0.74 as model accuracy increases.
- **Actionable insights on verifier sizing:** The empirical exploration of verifier sizes (Sec. 4.2) identifies a "sweet spot" ( $10^{-3}$ to $10^{-1.8}$ of prover compute), noting that matched-size verifiers fail to become more robust while very small ones lack capacity. This provides practical guidance for scalable oversight implementations.
- **Transparency in experimental constraints:** The limitations section (Sec. 5.1) is unusually thorough, candidly disclosing heterogeneous stopping criteria, untuned hyperparameters, dataset constraints, and the potential flaws in the sneaky prover human evaluation pilot. This transparency allows for a more grounded interpretation of the results.
- **Emergent structured reasoning:** Figure 2 provides qualitative evidence that checkability training produces naturally structured solutions with explicit calculation tags, suggesting the model learns to externalize reasoning steps rather than just formatting superficially.

## Weaknesses

### Fatal
None.

### Major
- **The claim that sneaky provers learn to deceive humans relies on a flagged pilot study.** The paper asserts in Abstract and Sec. 4.3.2 that "the sneaky prover generates increasingly subtle flaws, challenging both verifiers and humans," citing Figure 7b as primary evidence. However, the authors footnote this figure: *"Fig. 7b is from a pilot study with potential design flaws."* Relying on a self-admittedly flawed campaign to support a core theoretical claim about human transfer is a significant methodological gap. While the authors' transparency is commendable, this result cannot be taken as established evidence without a rigorously controlled replication.
- **Potential confounding of the "legibility tax" with verbosity.** The claimed legibility penalty of correctness optimization is not disentangled from uncontrolled solution length. Section 4.1 notes that solutions "become longer and less coherent as training progresses," and Figure 4a shows a sharp accuracy drop at 8k steps when solutions hit max context limits. Without matching output length or token budget between the baseline and the proposed method, it remains unclear whether decreased human performance stems from correctness optimization or simply from the absence of length constraints. The authors acknowledge in Sec. 5.1 that early stopping could have been applied but lacked a "principled way" to do so, leaving the central trade-off claim partially unverified.

### Minor
- **Lack of ablation isolating logical clarity from superficial format hacking.** The hypothesis that verifier checkability transfers to human legibility is untested against the alternative that the verifier rewards stylistic or formatting cues (e.g., markdown, explicit step labels, verbosity). While the increasing verifier robustness (Fig. 6a) and improved prover accuracy (Fig. 8a) partially counter this, the mechanism linking verifier optimization to human legibility is never strictly disentangled from superficial readability signals. A controlled evaluation holding formatting constant could strengthen the claim that genuine logical transparency improves.
- **Gap between theoretical guarantees and practical RL implementation.** Theorem 1 assumes $\mathcal{H}$ contains all deterministic mappings, but the practical implementation uses PPO with KL regularization, finite parameterization, and a Bradley-Terry verifier loss. The paper cites the theorem as motivation but does not analyze or address how the theoretical constraints translate to the empirical dynamics, leaving a conceptual gap between the equilibrium guarantee and the observed RL trajectories.

### Trivial
None.

## Nice-to-Haves
- A failure matrix comparing verifier vs. human judgments (e.g., cases where the verifier accepts but humans reject, and vice versa) would help clarify what "legibility" captures in practice beyond aggregate accuracy.
- Analyzing distribution shift in solution embeddings between helpful and sneaky provers could verify whether the verifier learns robust logical features or collapses to pattern matching.
- Clarifying the human evaluation protocol in-text (time limits, contractor instructions, quality control measures) would improve interpretability of the human accuracy metrics.
- Extending the experiments to harder math datasets or fuzzier domains (e.g., human preference data) would test generalizability.

## Removed Points
- Criticism questioning the existence of the pilot study or citing unreleased models/tools: All cited works and datasets are assumed to exist. The paper is transparent about its flaws.
- Claims about missing related works: Dropped as no external verification is possible.
- Criticisms about the exact human evaluation protocol being in the appendix: Appendix sections are stripped by the parser; they exist in the original submission.
- Nitpicks about hyperparameter tuning: The authors explicitly disclose in Sec. 5.1 that hyperparameters were not thoroughly tuned and detail the 2400 A100 hours used. Demanding full tuning logs is a reproducibility nitpick for an empirical paper.
- Criticism of uncontrolled verbosity in the baseline: Addressed above in weaknesses; while the authors acknowledge it in limitations, it does not invalidate the main contribution.

## Novel Insights
The paper provides a rare empirical demonstration that the tension between model accuracy and output legibility is not just theoretical but measurable in standard LLM training loops. The finding that there is an optimal verifier size—neither too small (insufficient capacity) nor matched (prone to simulation)—offers a practical heuristic for future scalable oversight work. Moreover, the observation that legibility emerges naturally from verifiability constraints without explicit formatting rewards is a compelling result for the alignment community, suggesting that "checkability" might be a more scalable proxy for human oversight than previously demonstrated. However, the reliance on a flawed pilot for the sneaky prover's human transfer and the uncontrolled verbosity in the baseline limit the strength of these insights.

## Suggestions
- Replicate the time-constrained human evaluation for sneaky solutions with a clean protocol to validate the claim that adversarial solutions deceive humans.
- Run the correctness-only baseline with an explicit length penalty or early stopping at matched output token counts to verify whether the "legibility tax" persists when controlling for verbosity.
- Include a failure matrix or case studies of verifier-human disagreement to clarify what features of the solutions drive human judgment versus verifier scores.
- Add a brief discussion in the main text on the human evaluation protocol to improve standalone interpretability.

## Score and Decision
**Score: 5.5**
**Decision: Reject**

The paper addresses a highly relevant question in scalable oversight and presents a clean, well-motivated formal framework. The main contribution (Fig. 1, Fig. 8) that checkability training mitigates the "legibility tax" while improving prover accuracy is compelling and aligns with the broader goals of the alignment community. The authors are commended for their thorough transparency in the limitations section.

However, the empirical foundation has significant gaps. The critical claim that sneaky provers learn to deceive humans relies on a self-admittedly flawed pilot study, and the central "legibility tax" narrative is confounded by uncontrolled verbosity in the baseline. While the primary results on the helpful prover remain strong, these weaknesses prevent the paper from fully establishing its broader claims about adversarial dynamics and human transfer. If the human evaluation were rigorously replicated and the baseline confounders addressed, this could easily be a strong acceptance. For now, the paper is promising but incomplete.