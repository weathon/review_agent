## Summary

This paper identifies and quantifies a "priming vulnerability" specific to Masked Diffusion Language Models (MDLMs), where affirmative tokens appearing at intermediate denoising steps can steer even safety-aligned models toward harmful responses. The authors propose Recovery Alignment (RA), which trains models to generate safe responses from intentionally contaminated intermediate states, and derive a tractable lower bound (Theorem 4.1) that enables efficient optimization-based attacks (First-Step GCG) without requiring denoising-process intervention.

## Strengths

- **Novel vulnerability identification with clean problem formulation.** The paper precisely defines the priming vulnerability as an MDLM-specific phenomenon arising from the iterative denoising mechanism, distinct from ARM prefilling attacks. The two-threat-model analysis (intervention vs. no intervention) is well-designed: the anchoring attack enables controlled, quantitative evaluation (Figure 2 shows ASR scaling cleanly with intervention step), while First-Step GCG demonstrates realistic exploitability. The key insight—that standard alignment trains only from clean initial states (Eq. 5) and therefore cannot constrain behavior at contaminated intermediates (Eq. 6)—is both simple and powerful.

- **Strong theoretical contribution enabling practical attacks.** Theorem 4.1's derivation of a tractable first-step lower bound resolves the gradient intractability caused by stochastic remasking. This is not merely a computational trick; it yields a 20× speedup and substantially higher ASR than MC GCG (Table 1: 58% vs. 20% on LLaDA Instruct), demonstrating that targeting the priming mechanism specifically is more effective than optimizing the full trajectory. The empirical validation of the monotonicity assumption (Appendix C.2, Figure 6) across three models adds credibility.

- **Recovery Alignment is well-motivated and effective.** The core idea—training on contaminated intermediate states rather than only from fully masked sequences—is directly motivated by the identified vulnerability mechanism. The ablation "RA w/o inter" cleanly isolates this contribution: without contaminated-state training, ASR at t_inter=4 exceeds 20% across models, while full RA reduces it to 0–1.3% (Table 2). The linear curriculum scheduling is sensible, and the ablation (Figure 3b) shows it outperforms constant and uniform alternatives.

- **Comprehensive evaluation across models, attacks, and evaluators.** Three MDLMs (LLaDA Instruct, LLaDA 1.5, MMaDA MixCoT), seven attack methods (four priming-based, three conventional), three safety evaluators (GPT-4o, LlamaGuard, keyword matching), two safety datasets, and eleven utility benchmarks. The consistency of results across this matrix is a genuine strength.

## Weaknesses

### Major:

- **Late-stage intervention remains largely undefended.** Even with RA, the anchoring attack at t_inter=32 achieves 50.7% ASR on LLaDA Instruct and 43.0% on LLaDA 1.5 (Table 2). The paper acknowledges this ("generating a fully safe response becomes challenging") but frames the overall result as "mitigated." Given that late-stage contamination with many anchor tokens is precisely where the vulnerability is most severe, the defense's partial failure at the hardest setting substantially qualifies the contribution. The paper should more explicitly characterize the boundary conditions under which recovery is feasible vs. impossible.

- **Limited generalization to certain conventional jailbreaks.** Against ReNeLLM, RA achieves 72.3% ASR on LLaDA Instruct and 71.7% on LLaDA 1.5 (Table 3)—barely improving over MOSA (77.7%) and worse than some might expect given RA's strong performance on priming attacks. The paper's proposed mechanism (harmful tokens necessarily emerge at intermediate steps, enabling re-detection) does not appear to hold for attacks like ReNeLLM that paraphrase harmful content into forms not readily detected from surface tokens. This suggests the "recovery" capability is more narrowly applicable than claimed, and the paper's discussion of generalization (end of Section 6.2) should more honestly delineate where RA helps and where it does not.

- **The monotonicity assumption lacks failure-mode characterization.** Theorem 4.1 underpins First-Step GCG and, by extension, the paper's claim that the vulnerability is exploitable without intervention. While Figure 6 shows the mean monotonicity gap is positive, the paper does not report what fraction of individual prompts violate the assumption or how the attack performs on those cases. If the assumption fails on a non-trivial subset of harmful queries, the lower bound—and thus the attack's theoretical guarantee—does not hold for those cases. A per-prompt analysis (even just reporting the violation rate) would significantly strengthen the theoretical contribution.

### Minor:

- **Utility degradation on specific benchmarks is understated.** Table 4 shows HumanEval dropping from 22.0 to 17.1 for LLaDA (a ~22% relative decrease), and PIQA from 74.4 to 71.6. The paper states "we do not observe substantial degradation" and highlights improvements on TruthfulQA and MBPP. While the average remains stable, the HumanEval drop is non-trivial for a code generation benchmark, and the claim of "minimal impact" would be more credible with explicit acknowledgment of this trade-off—particularly whether safety-oriented recovery training might systematically suppress the confident, deterministic generation required for code.

- **Over-refusal on benign queries is not evaluated.** The paper measures utility on standard benchmarks (MMLU, ARC, etc.) but does not assess whether RA increases false-positive refusal rates on borderline or sensitive-but-benign queries. Safety alignment methods are known to cause over-refusal, and the absence of this evaluation leaves the "minimal impact" claim incomplete. This is a notable gap given that RA trains on harmful contaminated states, which could shift the model's refusal boundary.

- **Computational cost of RA relative to baselines is not discussed in the main text.** Appendix C.4 reports ~16 hours on 4 H100 GPUs for 2,500 steps, but the main text claims RA is "practical and scalable" (Section 5) without comparing this cost to SFT or DPO. Practitioners need this comparison to assess whether the safety gains justify the overhead of on-policy rollouts from contaminated states.

### Trivial:

- The "Limitations" section mentions the DPO-style alternative but dismisses it due to data-construction cost. A brief discussion of why the RLHF instantiation was chosen over this alternative (beyond data availability) would clarify the design rationale.

## Nice-to-Haves

- Evaluate RA on benign instruction-following datasets (e.g., AlpacaEval, JustAsk) to measure over-refusal rates explicitly.
- Compare RA against an inference-time filtering baseline (e.g., reward-model-based output rejection) to justify the training-time cost.
- Visualize denoising trajectory token probabilities for RA vs. original models on a recovered example, to verify that the model actively overwrites harmful anchors rather than simply ignoring them.
- Test First-Step GCG suffixes for cross-model transfer to probe whether the vulnerability is architectural or model-specific.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **(Weakness) "Limited evaluation scope and generalization concerns"** — claiming only 2 datasets and 3 models is insufficient. The paper evaluates on 3 models, 7 attacks, 3 evaluators, 2 safety datasets, and 11 utility benchmarks. This is above-average comprehensiveness for the area.
- **(Weakness) "Incomplete ablation analysis of RA components"** — the paper includes ablations on t_max (Figure 3a), scheduling strategies (Figure 3b), and RA w/o inter (Table 2). While more ablations could always be added, the core design choices are tested.
- **(Weakness) "Missing comparison with ARM safety techniques adapted to MDLMs"** — the paper compares against SFT, DPO, and MOSA (the only existing MDLM-specific safety method). These are the relevant baselines.
- **(Weakness) "Threat model realism for anchoring attack"** — the paper explicitly addresses this by designing two threat models (Section 4.1 with intervention, Section 4.2 without). The anchoring attack is explicitly framed as a tool for "comprehensive evaluation," not a realistic attack.
- **(Weakness) "Clarify that only unmasked subset acts as anchor"** — re-reading Section 4.1, the anchoring attack applies the masking strategy m_{t_inter} to the harmful response r, producing r_{t_inter}. Since the masking strategy retains (t_inter/T) fraction of tokens unmasked, only those tokens act as anchors. The paper describes this correctly; the confusion arose from the reviewer's misreading.
- **(Weakness) Formatting/style issues with equations** — parser artifacts, not paper problems.

## Novel Insights

The most interesting structural insight across the reviews is that Recovery Alignment's mechanism may be dual-natured: it succeeds against priming attacks because contaminated intermediate states explicitly appear during training, but its partial failure against attacks like ReNeLLM (where harmfulness is obscured in surface form) suggests that the "recovery" is more of a pattern-matching response to known harmful token configurations than a deep re-evaluation of response semantics. This hints at a fundamental limitation: training on contaminated states teaches the model to resist *its own* harmful trajectories, but not necessarily to detect novel harmfulness that emerges through semantically subtle rephrasing. The distinction between "trajectory-level recovery" and "semantic-level safety" may be an important axis for future MDLM safety work.

## Suggestions

- Report the per-prompt monotonicity violation rate for Theorem 4.1 (even as a single number in the appendix) to clarify the theoretical bound's reliability.
- Add a brief table or paragraph in the main text comparing RA's training cost to SFT/DPO baselines (wall-clock time and approximate FLOPs), since the "practical and scalable" claim currently rests only on appendix data.
- Include 2–3 qualitative failure cases where RA fails to recover (e.g., at t_inter=32 or under ReNeLLM), with the actual generated text, to help readers understand the defense's boundaries.
- Evaluate over-refusal explicitly on a benign instruction-following benchmark; this is a low-cost addition that would significantly strengthen the utility-preservation claim.

---

**Axis Evaluations:**

- **Novelty:** High. The priming vulnerability is a genuinely new concept distinct from ARM prefilling, and Recovery Alignment is a well-motivated MDLM-specific defense. Theorem 4.1 is a clean theoretical result.

- **Technical Soundness:** Good with caveats. The core empirical analysis is rigorous, but the monotonicity assumption lacks failure-mode characterization, and the defense has clear boundary limitations that are under-discussed.

- **Empirical Support:** Strong on coverage (models × attacks × evaluators), but weakened by the substantial residual vulnerability at late intervention steps and under ReNeLLM, which tempers the "mitigation" framing.

- **Significance:** Significant. As MDLMs gain traction as ARM alternatives, establishing their distinct safety failure modes and tailored defenses is important and timely. The work sets a clear foundation for MDLM safety research.

- **Clarity:** Good. The paper is well-organized with a clean narrative arc from vulnerability identification → theoretical analysis → defense proposal → evaluation. The two-threat-model structure is particularly effective.