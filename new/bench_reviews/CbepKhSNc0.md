Now I have a good calibration sense. Let me synthesize the final review.

## Summary

LIAR reformulates the LLM jailbreak problem as an alignment problem, proposing that safety-aligned models can be "misaligned" by using an unsafe reward signal with alignment-style objectives. The implemented method uses best-of-N sampling from a small, unaligned GPT-2 model as an adversarial prompter, selecting the suffix that maximizes an unsafe reward when appended to harmful queries. The method requires no training, operates black-box on the target, and achieves competitive attack success rates at ASR@100 with very low perplexity (~2) and fast time-to-attack (seconds rather than hours).

## Strengths

- **Practical efficiency and speed advantage genuinely demonstrated**: TTA of 45 seconds vs. hours for AdvPrompter/GCG/AutoDAN is a clear practical win for red-teaming scenarios. Table 1 shows this convincingly across multiple target LLMs.
- **Extensive ablations**: Tables 2–5 systematically explore adversarial LLM choice, temperature, suffix length, and response length, providing actionable insight into which design choices matter.
- **Low perplexity is a meaningful contribution**: Perplexity ~2 (vs. 12–92000 for baselines) directly challenges the practicality of perplexity-based defenses, which is relevant to the safety community.
- **Black-box operation**: Unlike GCG and AutoDAN, LIAR requires no access to target model logits/gradients, making it applicable to API-only models — a genuine practical advantage.
- **The alignment framing, while overstated, offers a useful lens**: Connecting jailbreak vulnerability to RLHF-style objectives (Eq. 4) and deriving the optimal prompter distribution (Eq. 5) provides a principled starting point for understanding why aligned models remain vulnerable.

## Weaknesses

### Fatal
None.

### Major

- **The core conceptual claim — "jailbreaking via alignment" — is misleading relative to the actual algorithm**: The paper's central narrative is that jailbreaking can be formulated as an alignment problem and that alignment techniques can be leveraged for attacks. However, the actual algorithm (Eq. 6) is simply best-of-N sampling from a fixed, unmodified GPT-2: sample N suffixes, query the target, keep the one with highest reward. No model is trained, no RLHF is performed, no parameters are updated. The alignment formulation (Eqs. 3–5) is explicitly acknowledged as too costly ("a significant drawback... requires a costly training process") and is never instantiated. The presented method does not "misalign the model using alignment" — it performs random sampling with reward-guided selection. While the alignment framing provides context, claiming the method "leverages alignment" and "performs jailbreaking via alignment" overstates what the algorithm actually does. This mismatch between the conceptual framing and implementation undermines the paper's primary novelty claim.

- **Theoretical results are largely disconnected from the implemented method**: Theorem 1 bounds a "safety net" $\Delta_{\text{safety-net}}(\mathbf{x})$ defined between two hypothetical RLHF-optimized models ($\pi_u^*$ and $\pi_{\text{algo}}^*$), neither of which exists in or is relevant to the LIAR procedure. The bound itself is simply the range of $(R_u - R_s)$ over all outputs, which holds trivially for any two policies and does not meaningfully characterize jailbreakability of real aligned models. Theorem 2 provides a suboptimality bound $\frac{1}{N-1}\mathrm{KL}(\rho_u^*, \rho_0)$ relative to a hypothetical optimal prompter $\rho_u^*$ that is never constructed, and the KL term is never estimated — making the bound vacuous without knowing KL is small. Neither theorem is empirically validated (no estimates of safety nets or KL terms). The claim of "theoretical insights into inherent vulnerabilities in current alignment strategies" is unsupported by the actual theory presented.

- **LIAR performs poorly on strongly aligned models and ASR claims are misleading**: On LLaMA-2-7b — the most safety-aligned model tested — LIAR achieves only 3.85% ASR@100, compared to GCG's 23.7% and AutoDAN's 20.9% at ASR@1. The abstract claims "competitive attack success rates" and "up to 99% on Vicuna-7b," but this 99% figure comes from a model with known weak alignment and requires 100 attempts. The paper does not clearly qualify that its competitiveness holds only under a high-query-budget threat model against specific (not strongly aligned) targets. This is a significant limitation for a venue focused on alignment and safety.

- **ASR evaluation methodology has notable issues**: The paper uses keyword matching on only the first 32 tokens of target output (vs. the standard 150 tokens), which inflates ASR by missing refusals that appear after token 32. Table 5 acknowledges higher ASR with shorter responses but doesn't quantify this bias against baselines. The underlying reward $R_u$ is a binary keyword-based heuristic, not a learned safety classifier, undermining claims about "safety" and "alignment" while the theory assumes a well-defined reward function.

### Minor

- **Perplexity advantage is largely a design choice artifact**: Using a 124M-parameter unaligned GPT-2 naturally produces low-perplexity text. Baselines like GCG optimize loss directly (producing gibberish), while AutoDAN and AdvPrompter add explicit perplexity regularization. LIAR's low perplexity follows from sampling an off-the-shelf LM, not from any methodological insight. This should be acknowledged as a tradeoff of the approach rather than presented as a framework advantage.

- **TTA comparison is asymmetric**: TTA1 is reported for ASR@100 for LIAR but ASR@1 for other methods (acknowledged in a footnote of Table 1). While timing-matched comparisons would favor LIAR, the current presentation makes direct comparison difficult for readers seeking equal-budget analysis.

- **Limited to AdvBench only**: All experiments use only the AdvBench dataset (104 test samples). Generalization to other safety benchmarks (HarmBench, JailbreakBench) is not established.

- **No evaluation against defenses**: While the paper claims low perplexity "challenges the effectiveness of perplexity-based jailbreak defenses," it does not test against any defense mechanism (e.g., SmoothLLM, input/output classifiers, guard models).

### Trivial

- The acronym "LIAR" (LeveragIng Alignment to jailbReak) is a stretch — the 'I' and 'R' come from different words than expected.

## Nice-to-Haves

- Evaluate on strongly-aligned commercial or open-source models (e.g., Llama-3.1-Instruct, GPT-4) to establish the method's limits more clearly.
- Estimate the KL term in Theorem 2 empirically to show whether the suboptimality bound is tight.
- Test against active defenses (SmoothLLM, perplexity filtering, input/output guard models) to validate the practical attack threat.
- Use a learned harmfulness classifier or LLM judge for ASR evaluation rather than keyword matching, and evaluate at standard 150-token output length.

## Removed Points

- **"LIAR is just a rebranding of best-of-N"**: While the harsh critic characterizes this as structural deception, the paper does explicitly state (Section 3, end) that LIAR uses best-of-N and acknowledges the optimal solution (Eq. 5) requires fine-tuning. The alignment framing is a genuine, if imperfect, conceptual contribution — it provides an RLHF-style objective that motivates the approach. The issue is overstating the alignment connection, not that the method is wholly without novelty.

- **"Citation of models/tools that don't exist"**: All referenced models and datasets (GPT-2, LLaMA variants, AdvBench, etc.) are well-known and available. No issues here.

- **"Missing related works"**: Per instructions, not included.

- **"Unfair comparison — TTA computed differently"**: The asymmetry (TTA1 for ASR@100 vs. TTA1 for ASR@1) *favors* the baselines in terms of what constitutes a "successful" attack, not the authors. The authors' approach requires 100 queries for one success while baselines need only 1. This is noted as a minor presentation issue, not an unfair advantage.

- **"Demanding evaluation on closed-source/strongly-aligned models beyond paper scope"**: The paper tests on open-source models as stated. Requesting commercial model evaluation is reasonable as a nice-to-have but not a core flaw.

- **"The method has limited novelty because best-of-N is straightforward"**: The simplicity of a method does not determine its contribution value. Best-of-N has known theoretical properties and practical speed advantages that are meaningful contributions when applied to the jailbreak setting.

## Novel Insights

The most interesting observation is the tension between the alignment framing and the actual algorithm: the paper's strongest practical contribution is not "jailbreaking via alignment" but rather the demonstration that even the simplest possible sampling-plus-filtering approach from a tiny unaligned model can be surprisingly effective against many open-source LLMs. This suggests that much of the vulnerability comes from the target model's distribution having high-probability unsafe responses in the neighborhood of natural-language suffixes, rather than from any sophisticated alignment manipulation. The fact that LIAR nearly fails on LLaMA-2-7b (the strongest aligned model) while excelling on weakly aligned models implies that the effectiveness is mostly a function of how well the target model was aligned, not how clever the attack is — a finding that actually supports the safety community's emphasis on robust alignment rather than undermining it.

## Suggestions

1. **Reframe the contribution honestly**: Position LIAR as an efficient black-box attack using reward-guided best-of-N sampling, with the alignment formulation as motivating context. Avoid claiming the method "performs alignment" or "jailbreaks via alignment."
2. **Clearly scope the threat model**: State upfront that LIAR operates in a high-query-budget setting and is less effective per-query than gradient-based methods or against strongly aligned models.
3. **Report ASR at matched query budgets**: Present ASR as a function of k for all methods, or provide time-budget-matched comparisons, so readers can assess efficiency fairly.
4. **Evaluate with standard 150-token outputs and a more robust ASR judge**: Even a small validation experiment would strengthen the empirical claims.
5. **Empirically estimate the safety net and KL quantities in Theorems 1–2**: This would bridge the theory-practice gap and show whether the theorems have predictive value.

## Score and Decision

**Calibration anchors:**
- AutoDAN (ICLR, scores 5/10/5/5, rejected): Novel gradient-based adversarial suffix method with interpretable prompts. Similar domain, but AutoDAN had genuine algorithmic novelty and higher per-query effectiveness. Received mixed-to-negative reviews.
- PAIR (scores 3/6/5/5, rejected): Black-box jailbreak using LLM-as-attacker, similar "simple but practical" spirit. Rejected partly due to limited novelty of the core mechanism and unfair comparison framing.
- AutoDAN-Turbo (scores 8/8/8/8/8/3, accepted spotlight): Much stronger novelty — automatic strategy discovery, comprehensive evaluation including GPT-4, genuinely novel algorithmic framework.
- Simple Adaptive Attacks (scores 6/5/6/6/6/8/6, accepted poster): Demonstrated 100% ASR on many models including strong aligned models, but noted as "straightforward" with limited novelty.
- KDA (scores 1/3/3/3, rejected): Overclaimed novelty, evaluation issues, similar "distillation from existing attacks" pattern.
- Stochastic Monkeys (scores 5/3/6/5/5, withdrawn/rejected): Simple random augmentation method, similar "cheap attack" theme, limited novelty.

LIAR sits between PAIR/KDA (rejected for limited novelty and overclaimed contributions) and Simple Adaptive Attacks (accepted for empirically convincing results despite methodological simplicity). The core algorithm (best-of-N from GPT-2) is quite simple, the alignment framing is overstated, the theory is disconnected from practice, and empirical claims are misleading about competitiveness. However, there is a genuine practical contribution in demonstrating that this simple black-box approach achieves competitive ASR with remarkable speed and low perplexity. The weaknesses are significant but not fatal — the paper presents real empirical results that advance practical understanding of LLM vulnerability to simple attacks.

MY FINAL SCORE: <pineapple>4</pineapple>
MY FINAL DECISION: <orange>Reject</orange>