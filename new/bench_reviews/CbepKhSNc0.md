## Summary
This paper proposes LIAR, a training-free jailbreak method that reframes adversarial prompt generation as an alignment problem using Best-of-N sampling. The method achieves competitive attack success rates (ASR@100 ~97% on Vicuna-7b), significantly lower perplexity (~2.14) than baselines, and fast per-query latency (~0.033s) using a small GPT-2 adversarial model. The paper includes theoretical bounds on jailbreak feasibility and Best-of-N suboptimality, plus ablations on temperature and query length.

## Strengths
- **Training-free with genuinely fast per-query latency**: The method requires no training phase and achieves ~0.033s per query using GPT-2 (124M params), enabling 100+ attempts in ~14 minutes versus 22 hours training for AdvPrompter. Table 1 and Table 2 provide concrete timing measurements across 6 target models.
- **Low perplexity adversarial prompts**: LIAR achieves perplexity ~2.14 on Vicuna-7b, orders of magnitude lower than GCG (92,471) and notably lower than AdvPrompter (12.09), supporting claims about bypassing perplexity-based detection. This is empirically validated in Table 1 across all target models.
- **Competitive ASR with sufficient attempts**: At ASR@100, LIAR achieves 97.12% on Vicuna-7b and 100% on Falcon-7b/Pythia-7b, comparable to GCG's ASR@1 (99.10%). The scaling analysis in Figure 1 and Table 1 demonstrates the method's effectiveness when allowed multiple attempts.
- **Comprehensive ablations**: Section 5.3 analyzes temperature and query length effects, showing ASR@100 peaks at temperature 0.6 and query length 30, providing practical guidance for practitioners.

## Weaknesses

### Fatal
None

### Major
- **Confusing and potentially misleading TTA metric definition**: The Table 1 caption states "TTA1 for our method is computed for ASR@100, whereas TTA1 for all other methods are computed for ASR@1." This is internally inconsistent—TTA1 should mean "time for 1 attempt," but the paper appears to report the time for 100 attempts (45 seconds) while labeling it TTA1. This makes cross-method comparison difficult and undermines the "45 seconds" headline claim. If 45 seconds represents 100 queries, the per-query time is ~0.45s, not the 0.033s shown in Table 2. This discrepancy needs clarification.
- **Overstated "seconds" claim without success rate context**: The Abstract claims "Time-to-Attack measured in seconds" and Figure 1 highlights "~1 sec," but Table 1 shows ASR@1 is only 12.55% on Vicuna-7b. To achieve reliable jailbreaks (~97% success), LIAR requires 100 attempts taking 14 minutes (TTA100). The headline framing emphasizes single-query speed while obscuring that a *successful* attack typically requires minutes, not seconds. This is particularly problematic for security evaluations where time-to-*success* matters more than time-per-attempt.

### Minor
- **Lightweight theoretical contributions**: Theorem 1 bounds the "safety net" by the maximum difference between unsafe and safe rewards—a result that follows directly from the definitions. Theorem 2 (Best-of-N suboptimality) applies known RL sampling results to the jailbreak setting. While formally correct, these provide limited novel insight beyond framing existing knowledge in alignment terminology.
- **Low perplexity not validated against actual detectors**: The paper claims low perplexity "challenges the effectiveness of perplexity-based jailbreak defenses" (Section 5.1) but does not evaluate against specific detectors (e.g., Alon & Kamfonas, 2023). Demonstrating actual bypass rates would strengthen this claim.

### Trivial
- **Inconsistent TTA notation in Figure 1 vs Table 1**: Figure 1 shows "~1 sec" for LIAR at k=10, but Table 1 shows TTA1=45s. The figure appears to show per-query time while the table shows setup + queries, but this is not clearly explained in the caption.

## Nice-to-Haves
- Evaluate against a specific perplexity-based detector to demonstrate actual bypass rates, not just low perplexity scores.
- Report expected time-to-success (TTA / ASR@1) for all methods to normalize speed against success probability.
- Clarify whether the reward computation uses target model output text classification (true black-box) or requires any probability access, and explicitly state this in the method section.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **"Black-box claim contradicts method dependency on logits"**: The harsh critic claimed Equation 1 requires target model logits. However, the paper explicitly states (line 47) "our method is fully black-box as it does not depend on any logits or probabilities from the TargetLLM." Equation 1 is the problem formulation from prior work (GCG), not LIAR's implementation. LIAR uses Best-of-N sampling where the reward is computed from output text (unsafe/not unsafe via keyword matching), not logits. The black-box claim is valid. **REMOVED** - reviewer misread the paper.

- **"Unfair evaluation protocol penalizing training-based baselines"**: The critic argued including AdvPrompter's 22-hour training cost in TTA1 is unfair. However, the paper is transparent about this metric definition (line 176-177), and for one-off attack scenarios the training cost IS relevant. The paper also reports TTA100 showing scaling. This is a valid framing concern but not an unfair comparison—the asymmetry is intentional to show training-free advantage. **MOVED to Minor/Nice-to-have** - the metric is clearly defined, though amortization discussion would help.

- **"Best-of-N is not alignment, just inference-time optimization"**: The critic argued the "alignment" framing is semantically loose since LIAR doesn't update model weights. However, the paper explicitly frames this as "jailbreaking via alignment" using the RLHF objective formulation (Eq. 4), and Best-of-N is a recognized inference-time alignment technique (Amini et al., 2024, cited). This is a framing choice, not an error. **REMOVED** - the alignment framing is defensible given the RLHF-style objective.

- **"Theorem 2 is a known RL result"**: While True, applying known results to a new setting (jailbreaking) is a valid theoretical contribution. The bound on Best-of-N suboptimality in terms of KL divergence and N samples provides formal grounding for the method's scaling behavior. **REMOVED** - not a weakness, just incremental theory.

- **Generic strengths about "important problem" or "interesting question"**: Dropped per instructions—only concrete, evidence-backed strengths retained.

## Novel Insights
The paper's core insight—that jailbreaking can be framed as inverse alignment using an unsafe reward signal—is a useful conceptual reframing that connects two previously separate literatures. However, this insight is primarily presentational; the actual method (Best-of-N sampling from a fixed adversarial model) is standard. The genuinely novel observation is that training-free sampling can achieve competitive ASR@100 with dramatically lower perplexity than optimization-based methods, suggesting the jailbreak space contains many low-perplexity solutions that gradient-based methods miss due to their optimization trajectory. This empirical finding has practical implications for defense design.

## Suggestions
1. **Fix the TTA metric labeling**: Either relabel "TTA1" as "TTA100" for LIAR (since 45s represents 100 queries), or report true TTA1 (time for 1 query, ~0.45s including any setup amortization). Ensure Figure 1 and Table 1 use consistent definitions.
2. **Reframe the "seconds" claim**: State clearly that per-query time is ~0.033s but time-to-reliable-success (~97% ASR) is ~14 minutes. Consider adding a "Time-to-97%-Success" metric for fair cross-method comparison.
3. **Add detector evaluation**: Include a small experiment showing LIAR prompts bypass a specific perplexity-based detector (e.g., Alon & Kamfonas, 2023) to validate the stealth claim.
4. **Clarify reward computation**: Explicitly state in Section 3.1 that the reward $R_u$ is computed from TargetLLM output text (keyword matching or classifier), not logits, to preempt black-box concerns.

## Calibration and Score
I compared this paper against the following calibration anchors:

| Paper | Avg Score | Comparison to LIAR |
|-------|-----------|-------------------|
| **7B9mTg7z25** (6.0) | 6.0 | Stronger: Comprehensive adaptive attack framework breaking 12 defenses with large-scale human red-teaming. LIAR has narrower evaluation scope. |
| **ilnKzaQSCh** (5.5) | 5.5 | Stronger: Novel dialectic game-theoretic framing with theoretical guarantees. LIAR's Best-of-N is more standard. |
| **JDtIrWYB4o** (5.2) | 5.2 | Comparable: Training-free API-focused jailbreak with strong empirical results. JULI has more novel vulnerability (logit manipulation); LIAR has better perplexity results. |
| **mVcQXHRK8x** (4.5) | 4.5 | Similar: Test-time alignment method comparing against Best-of-N. Both have solid experiments but limited novelty. LIAR's metric confusion is a similar-level issue. |
| **Bzbu5czqMY** (4.0) | 4.0 | Similar: Strong experiments but with overclaiming and methodological issues (invalid proofs vs. confusing metrics). LIAR's issues are less severe than invalid proofs. |
| **OkjB6PWJEA** (3.0) | 3.0 | Weaker: Incomplete evaluation, missing SOTA baselines, unfair setup. LIAR has more comprehensive evaluation. |
| **e8b3fRYeDt** (0.67) | 0.67 | Much weaker: Single-turn evaluation only, missing key baselines, grandiose claims. LIAR is substantially stronger. |

LIAR sits between **mVcQXHRK8x (4.5)** and **JDtIrWYB4o (5.2)**. It has stronger empirical results than mVcQXHRK8x (more target models, better perplexity), but the TTA metric confusion is a genuine clarity issue similar to weaknesses in 4.0-4.5 papers. It lacks the novelty of JDtIrWYB4o (which targets a specific API vulnerability with a new attack vector). The paper's strengths (training-free, low perplexity, good ASR@100) are real, but the metric overclaiming prevents it from reaching 5.5+ territory.

**Final Score: 4.5** — Borderline. The empirical contributions are solid and the method works, but the confusing TTA definition and overstated "seconds" claim undermine the paper's clarity and trustworthiness. With metric corrections and tempered claims, this could reach 5.0-5.5.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>