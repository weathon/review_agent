=== CALIBRATION EXAMPLE 45 ===

# Final Consolidated Review
## Summary

This paper investigates how the Indirect Object Identification (IOI) circuit in GPT-2 small behaves on two novel prompt variants — DoubleIO and TripleIO — where the original IOI algorithm should theoretically fail. The authors find that the base IOI circuit outperforms the full model on these variants due to a newly identified evaluation artifact they term *S2 Hacking*, arising from mean ablation knocking out all non-S2 input paths. When circuits are freshly discovered for the variants, they reuse 100% of the base IOI circuit's nodes and 91.7%/84.6% of its edges, demonstrating strong structural generalization and providing, to the authors' knowledge, the first systematic demonstration of circuit generalization through circuit reuse.

---

## Strengths

- **S2 Hacking is a genuinely novel methodological insight.** The paper demonstrates, through concrete tracing from Duplicate Head 3.0 → Induction Heads 5.5/5.9 → S-Inhibition Head 8.6, that mean ablation can artificially inflate circuit faithfulness on out-of-distribution prompts by eliminating competing input paths. Faithfulness scores of 1.285 (DoubleIO) and 2.586 (TripleIO) for the base circuit confirm the magnitude of this artifact. This finding has direct implications for how circuits are evaluated across the broader mechanistic interpretability literature, well beyond the IOI task.

- **100% node reuse and 91–92% / 85% edge reuse is a strong, specific empirical result.** Table 2 shows that the discovered DoubleIO and TripleIO circuits add only 10 and 20 edges respectively to the 110-edge base circuit, all corresponding to paths from the newly duplicated IO tokens. This is the paper's core contribution and it is well-supported: path-patching identifies the same Name Mover, S-Inhibition, Duplicate, Induction, and Previous Token heads with significant causal effects as in the base circuit.

- **The "first come, first serve" finding in Head 2.2 is a specific and interesting mechanistic observation.** Figure 8 shows a clear ordering effect (logit diff ~2.5 when IO appears first vs ~0.9 when S appears first) and demonstrates that head 2.2's attention strongly tracks which name appears first in the prompt. This is a concrete, quantified behavioral finding that identifies a previously unrecognized decision point in the circuit.

- **The paper introduces a principled, replicable experimental framework for studying circuit generalization.** The use of normalized faithfulness, functional faithfulness, and confidence ratio metrics, combined with systematic edge-restoration experiments (Figure 5), provides a methodology that future circuit generalization studies can build on.

---

## Weaknesses

### Fatal
None.

### Major

- **The "decision point" claim about Head 2.2 rests solely on attention patterns, not causal intervention.** Section 5.3 identifies Head 2.2 as implementing a "first come, first serve" mechanism based on attention weight distributions. However, no ablation or patching experiment is performed on Head 2.2 itself to confirm that removing or redirecting its output causally changes the IO-first vs. S-first performance gap. Attention weight correlation does not establish causation; the head could be downstream of the actual decision mechanism. This is a substantive gap given that the "decision point" is one of the three key mechanistic findings of Section 5.

- **S2 Hacking is demonstrated only under mean ablation and not tested under alternative ablation protocols.** The paper identifies S2 Hacking as an artifact of mean ablation (stated explicitly: "a byproduct of the knockout procedure"). However, zero ablation and resample ablation are standard alternatives in circuit evaluation. If S2 Hacking does not appear under resample ablation — which uses out-of-distribution prompts rather than the mean activation — the phenomenon might be narrower than presented, or alternatively its prevalence across protocols could make the methodological warning stronger. Without this comparison, the paper cannot characterize how broadly its warning applies.

### Minor

- **The faithfulness of discovered variant circuits (~0.77) is notably lower than the base IOI circuit (0.895), and this gap is under-analyzed.** Table 2 shows this clearly. The paper frames the result as "strong generalization" (from Figure 1), but this categorization in Figure 1 refers to circuit overlap, not faithfulness. The faithfulness gap does deserve more discussion: does it reflect unidentified backup paths, incomplete circuit coverage, or fundamental limits of the structural reuse hypothesis? Even a paragraph in the conclusion addressing this would clarify what the "generalization" claim does and does not entail.

- **The abstract creates a framing confusion that persists into the introduction.** The abstract states the circuit "generalizes even to prompt variants where the original algorithm should fail" without immediately clarifying that this behavior is an evaluation artifact. Section 4 resolves this, but the abstract and parts of the introduction conflate S2 Hacking (an artifact of the evaluation procedure) with genuine model generalization. The paper's own data show the base circuit is *unfaithful* on the variants — faithfulness of 1.285 and 2.586 are explicitly described as "far from the ideal value of 1." Clearer abstract framing would prevent readers from misattributing the base circuit's apparent performance to the model's underlying mechanisms.

- **The absence of a causal explanation for why IO1 paths are inert while IO2 paths are causally effective** is noted (Section 5.1: "adding paths from just the IO1 token has little impact"). A plausible positional explanation exists (IO2 is closer to S2 and END), but this is not stated. Understanding this asymmetry is mechanistically important because it determines which duplicate instances feed into the circuit.

### Tiny

- **Inconsistency in sample sizes:** The main datasets use 200 prompts per variant, but Figure 4's confidence intervals are computed from only 50 samples. No explanation is given for this discrepancy.

- **The term "S2 Hacking" implies an active mechanism in the circuit**, whereas the paper itself clarifies it is an evaluation artifact. A name like "S2 Ablation Artifact" or "S2 Knockout Artifact" would be more technically precise, though this is a terminological preference.

---

## Nice-to-Haves

- **Causal knockout of Head 2.2** (e.g., mean-ablating its output or patching its attention pattern from IO-first to S-first prompts) would convert the "first come, first serve" observation from correlational to causal, substantially strengthening Section 5.3.

- **Checking S2 Hacking prevalence in other circuits** (e.g., Greater-Than) under mean ablation would establish whether this is an IOI-specific artifact or a systemic issue affecting multiple circuits in the literature.

- **Broader structural variants** (e.g., different sentence templates, passive constructions) beyond the duplication-count axis would probe whether circuit reuse extends beyond syntactic variants that share the same discourse structure.

- **Running automated circuit discovery (e.g., ACDC) from scratch on DoubleIO/TripleIO** without initializing from the base IOI circuit would provide an independent verification that the reuse result is not a confirmation bias artifact of the Wang et al. methodology.

- **Statistical significance test for the edge overlap percentages.** It would be useful to show that 91-100% overlap is significantly above a random circuit subgraph baseline of the same size, ruling out the possibility that dense transformer attention layers inflate overlap by default.

- Testing generalization of key findings on **GPT-2 Medium or XL** would narrow the gap between the paper's findings and its abstract's reference to "LLMs."

---

## Removed Points

*These points are flagged for removal; treat them with caution.*

- **"200 prompts is too small"** (Harsh Critic): The paper follows the exact data generation strategy of Wang et al. (2023), which also used 200 prompts. This is the community standard for this task.

- **Faithfulness ~0.77 is inconsistent with "strong generalization" framing** (Harsh Critic): This misreads the paper. Figure 1's "strong generalization" axis is *circuit overlap*, not faithfulness. With 91–100% node and edge overlap, calling this "strong generalization" per the paper's own Figure 1 framework is internally consistent. The faithfulness issue is a separate (valid) point already captured in the Minor weaknesses above.

- **Scope limited to GPT-2 small, but abstract claims about LLMs** (multiple reviewers): The abstract says circuits "may be more flexible" — a hedged claim appropriate for a first demonstration. Evaluated as a minor framing issue only; this does not undermine the findings.

- **Broader structural variants (passive voice, negation, relative clauses) should be tested** (Harsh Critic): The paper's stated scope is circuit generalization to *duplication-count variants* of IOI. Demanding broader linguistic variation is scope creep; moved to Nice-to-Haves.

- **Single-model limitation invalidates the paper's contributions** (Harsh Critic): GPT-2 small and IOI are the accepted sandbox for mechanistic interpretability; the paper is fully within scope for its community. The scope is a limitation worth noting, not a fatal flaw.

---

## Novel Insights

The most genuinely novel insight — one with implications well beyond this paper — is the identification of *S2 Hacking* as a systematic way in which mean ablation can make a circuit appear far more capable than the model it purports to explain, specifically on out-of-distribution prompts. The mechanism is that mean ablating paths from all non-circuit input tokens effectively eliminates competing signals, leaving the wrong-answer token (S2) as the sole non-ablated input to Duplicate and Induction heads. This can cause the circuit to "hack" the evaluation and appear to correctly solve variants the base algorithm should fail on. This is not merely a curiosity about the IOI circuit — it suggests that the standard circuit evaluation methodology produces systematically misleading faithfulness scores for any circuit evaluated on prompts that change which tokens are "competitors," and prior circuit faithfulness claims on generalization tasks may need to be revisited.

---

## Suggestions

1. **Verify Head 2.2 causally:** Add a single ablation experiment (e.g., mean-ablate Head 2.2 output or patch its attention from IO-first to S-first) and report the change in logit difference, stratified by name order. This would elevate Section 5.3 from observation to mechanistic claim.

2. **Replicate key metrics under resample ablation:** At minimum, recompute Table 1 (base circuit performance on variants) under resample ablation. If faithfulness > 1 persists, S2 Hacking is a general phenomenon; if it disappears, the methodological scope narrows. Either way, the result is informative.

3. **Revise the abstract and introduction framing:** Distinguish clearly between (a) the base circuit's evaluation performance being an artifact (S2 Hacking), and (b) the genuine generalization finding (circuit reuse). The current abstract risks being read as claiming the model's circuit genuinely adapts, when the paper's main finding is actually about the discovered variant circuits.

4. **Add a discussion paragraph on faithfulness degradation:** Address directly why the discovered variant circuits achieve 0.765–0.778 faithfulness rather than the 0.895 of the base circuit. Even a hypothesis (e.g., existence of additional IO-processing paths not captured by the circuit, positional noise from extra tokens) would strengthen the paper's overall account of how fully the circuits explain model behavior.

5. **Explain the IO1 inertness:** Add a brief mechanistic account of why paths from IO1 are causally ineffective compared to IO2. A positional or sequential account (IO2 is adjacent to the duplicate handling sub-circuit's token positions) is plausible and worth stating explicitly.

---

**Overall evaluation:** This is a well-executed paper that makes a specific, empirically grounded contribution to mechanistic interpretability. The S2 Hacking finding has genuine methodological significance for the field, and the circuit reuse results are substantive. The paper is limited in scope (single model, single task, single ablation protocol) and leaves the Head 2.2 decision point mechanistically unverified, but these are addressable. The core claims are supported by data and the experimental methodology is sound. The paper merits acceptance with the revisions above.

**Novelty:** High within its subdomain — S2 Hacking in particular is a novel, field-relevant methodological insight.
**Technical soundness:** Good; methodology is rigorous and follows established practices, with the gap around Head 2.2 causal verification being the main concern.
**Empirical support:** Solid for circuit reuse and S2 Hacking; weaker for the decision-point claim.
**Significance:** Meaningful for mechanistic interpretability; S2 Hacking has implications for prior and future circuit evaluation work.
**Clarity:** Generally clear, with localized framing confusion in the abstract and early introduction.

# Actual Human Scores
Individual reviewer scores: [5.0, 5.0, 3.0, 3.0]
Average score: 4.0
Binary outcome: Reject
