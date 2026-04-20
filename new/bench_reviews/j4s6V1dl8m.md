Now let me do calibration searches to properly score this paper.Now I have enough information to write the final consolidated review. Let me synthesize everything.

---

## Summary

The paper studies a "legibility tax" in LLM training: optimizing for answer correctness via RL degrades the human-verifiability of chain-of-thought solutions. To mitigate this, the authors propose an iterative Prover-Verifier Game (PVG) training algorithm in which a "helpful" prover, a "sneaky" prover, and a small verifier are co-trained across multiple rounds. Evaluated on grade-school math (GSM), the authors report increasing verifier robustness across rounds and improving human accuracy when checking the helpful prover's solutions. The paper frames this as evidence that training for checkability against small LLM verifiers is a practical proxy for human legibility.

---

## Strengths

- **The legibility tax is convincingly demonstrated (Sec. 4.1, Fig. 4).** The correctness-only RL baseline shows human contractor accuracy dropping from ~0.82 (initialization) to ~0.73–0.74, while Fig. 4a confirms solutions degrade visually and hit the length limit. This is a concrete, well-motivated phenomenon that anchors the paper.

- **Verifier robustness increases under three independent adversarial attacks (Fig. 6a).** The SRC, CGC, and goodharting adversaries all require progressively more PPO steps to exploit the verifier, with the 10⁻² relative-size verifier becoming unexploitable beyond round 0 within 8000 steps. Importantly, the adversaries are separately trained (removing the weight-sharing confound), which adds credibility to the robustness claim.

- **Human legibility improvement over training rounds (Fig. 8b, 8c).** Human contractor accuracy on the helpful prover's solutions increases from ~0.76 (round 0) to ~0.84 (round 5), and solution evaluation time decreases. This is the most consequential empirical result of the paper and directly supports the transfer hypothesis.

- **Capability gap analysis (Sec. 4.2).** The paper systematically varies verifier size from 10⁻⁴ to 10⁰ relative to the prover, identifying a viable operating window and diagnosing failure modes at both extremes—a useful practical contribution.

- **Author transparency.** The paper is unusually candid: it flags the flawed sneaky-prover human study, acknowledges data contamination from math finetuning, discloses heterogeneous stopping criteria, and notes that early stopping the correctness-only baseline might have changed the headline comparison. This transparency, while revealing limitations, is scientifically commendable.

- **Qualitative evolution is clear (Fig. 2).** The side-by-side examples showing init/round-1/round-5 solutions make the emergence of structured, step-by-step reasoning tangible.

---

## Weaknesses

### Fatal
None.

### Major

- **The headline Figure 1 comparison is not at iso-accuracy, and the authors acknowledge it.** Figure 1 frames the legibility tax as an inherent consequence of correctness optimization, comparing an RL model at ~0.80 accuracy to the checkability game at ~0.60 accuracy. Yet Sec. 5.1 explicitly states: *"For the correctness-only baseline, we could have early stopped at less accurate but more legible checkpoint."* This is a crucial admission: if an early-stopped RL model at ~0.60 accuracy also retains ~0.82 human legibility (as Figure 4d's trajectory at low model accuracy suggests), the entire advantage attributed to PVG training over the correctness-only baseline in Figure 1 could collapse. Without an iso-accuracy comparison point, the headline claim—that PVG uniquely avoids the legibility tax—is unverified.

- **The causal link between verifier pressure and human legibility is assumed, not tested.** The paper's core mechanistic claim is that adversarial pressure from small LLM verifiers *causes* the helpful prover's solutions to become more legible to humans. Both verifier robustness and human accuracy improve over rounds, but these are co-occurring outcomes of training, not causally linked in the experiments. There is no ablation that trains the helpful prover with a frozen or random verifier to establish that the adversarial signal (rather than simply iterative PPO refinement toward more structured outputs) is the operative mechanism. Without this control, the transfer hypothesis—the entire rationale for using small verifiers as human proxies—rests on correlation.

- **One of the four stated contributions depends on a study the authors flag as potentially invalid.** The claim "The sneaky prover generates increasingly subtle flaws, challenging both verifiers and humans" (bullet 4, Sec. 1) is supported primarily by Fig. 7b, which carries the explicit caveat: *"This plot is from a potentially flawed campaign (see Appendix K)."* The paper also lists this as a contribution in the abstract without qualification. The proxy evidence (Fig. 7a's GPT-4 validity scores) establishes the flaw localization story for the verifier, but does not substitute for valid human-subject evidence. Claiming a human result on the basis of a self-flagged unreliable study weakens the paper.

### Minor

- **The human evaluation in Fig. 8b shows non-monotonic behavior that is left unexplained.** Round 4 shows a substantial dip (back to ~0.76, the same as round 0), after steadily rising through rounds 1–3 (~0.81–0.82). If checkability training monotonically drives legibility improvement, this reversal warrants explanation. Given that the paper also reports no sample sizes, variance estimates, or statistical tests for the human evaluation, the overall trajectory is hard to assess rigorously. For a core result, this is a notable gap.

- **The failure mode of same-size verifiers is unexplained.** Section 4.2 notes that verifiers at 10⁰ relative size fail to become more robust across rounds, attributing it to sneaky provers "simulating their decision process." This is speculative, and since same-size verifier failure is the key constraint on the method's scalability (particularly for the goal of aligning superhuman models), it deserves more than a brief hand-wavy sentence.

- **Theorem 1's connection to the experiments is not established.** The theorem assumes H is the set of *all* deterministic mappings from X to Z, a condition neural networks trained with PPO cannot satisfy. No convergence result to the Stackelberg equilibrium is offered, and the paper does not assess empirically how close the iterative training gets to equilibrium. The theorem provides theoretical inspiration but does not formally underpin the empirical claims. This is common in empirical ML work but should be stated clearly.

- **The baseline legibility degradation may be largely a reward-hacking artifact.** Fig. 4a's caption notes the sharp model accuracy drop at 8k steps is "due to solutions reaching the maximum completion length." This means a substantial portion of the "legibility tax" in the baseline is driven by solutions exceeding the context window and becoming incoherent—a known reward hacking failure mode—rather than a deep tension between accuracy and legibility. The paper presents this as a fundamental trade-off without ruling out the length-exploitation explanation.

### Trivial
- None warranting separate mention.

---

## Nice-to-Haves

- An ablation training the helpful prover with a frozen or random verifier would provide the strongest evidence for the causal mechanism.
- An iso-accuracy comparison between the correctness-only baseline (early-stopped at ~0.60) and the checkability game would directly test the Figure 1 claim.
- A clean replication of the sneaky prover human study (without the design flaws acknowledged in Appendix K) is necessary before the claim that sneaky solutions fool humans can be maintained.
- Evaluation beyond GSM (e.g., MATH or coding tasks) would substantially strengthen generalization claims.
- Plotting human accuracy and verifier robustness side-by-side across rounds and verifier sizes to test whether they correlate would directly support the proxy hypothesis.

---

## Removed Points

*These points are flagged to be removed; treat them with caution as they were not verified against the paper or violate the meta-review rules.*

- **Harsh Critic: "ChatGPT-generated synthetic data creates non-trivial label noise"** — The paper (Sec. 5.1) explicitly addresses this: *"Using ChatGPT to augment this dataset could result in some incorrect labels, but based on preliminary data analysis we did not see significant difference between verifier test accuracies trained on synthetic vs. real data."* The dismissal is brief but reasonable; this is a reproducibility nitpick adequately addressed.

- **Harsh Critic: "Compute comparison to baseline is unfair"** — The paper acknowledges 2400 GPU hours for 6 rounds in the limitations. The correctness-only baseline is not a compute-matched comparison but this paper is about a training algorithm, not an efficiency claim. Removing as scope creep.

- **Harsh Critic: "GPT-4 circular evaluation in Fig. 7a"** — Plausible concern but GPT-4 judging GPT-4 solutions is standard practice in the field; the paper uses it as a proxy measure alongside human evaluation, not as the primary evidence. Not a fatal concern in context.

- **Strength Finder: "Theorem 1 providing formal grounding"** listed as a supporting strength — Conflicts with the verified weakness that the theorem's assumptions are disconnected from the experimental setting. The theorem is inspirational rather than grounding.

---

## Novel Insights

The most genuinely novel observation in this paper—confirmed by the experiments—is that the *direction* of legibility deterioration under correctness-only RL (Fig. 4d) is asymmetric: solutions become harder to verify specifically on problems the model solves *inconsistently* (low accuracy quartiles), not on problems it solves reliably. This suggests the model is learning to exploit verifier blind spots via reward hacking on hard problems, rather than globally sacrificing legibility. The PVG's iterative adversarial structure may be particularly effective precisely because it forces the helpful prover to produce checkable solutions even on hard problems where reward hacking is tempting. This mechanism, if probed directly, would substantially strengthen the paper's contribution.

---

## Assessment

**Originality:** The application of prover-verifier games to *legibility training* of large LLMs, rather than verification of fixed programs, is novel and well-motivated. The framing of small LLM verifiers as proxies for human legibility is an interesting hypothesis with practical implications.

**Importance of research question:** High. Legibility of AI reasoning is directly relevant to alignment and oversight of advanced systems.

**Whether claims are well supported:** Mixed. The legibility tax is well-documented. The claim that PVG *causes* legibility improvement is supported by correlation (Fig. 8b) but not by a causal ablation. The headline Figure 1 comparison is self-acknowledged to be non-iso-accuracy and potentially misleading. One of four stated contributions is based on a flagged unreliable study.

**Soundness of experiments:** Moderate. Verifier robustness evaluation is rigorous (three independent attacks across multiple sizes). Human evaluation is hampered by lack of statistics, non-monotonic results, and a flagged study for one key claim.

**Clarity of writing:** Good. The paper is readable and transparent about its limitations.

**Value to research community:** Meaningful, as a proof-of-concept that legibility can be trained via adversarial verification. The paper is exploratory rather than definitive.

---

## Score and Decision

**Calibration anchors:**

| Paper | Topic | Scores | Avg |
|---|---|---|---|
| Neural Interactive Proofs (R2834dhBlo) | Prover-verifier games, LLM code validation | 5, 10, 5 | ~6.7 |
| Evaluating Oversight Robustness (licAR8FPTW) | Scalable oversight in toy domain | 3, 6, 3, 1, 3, 3 | ~3.2 |
| Rewarding Progress / PAVs (A6Y7AqlzLW) | Weaker provers verify LLM reasoning, math | 8, 8, 6, 8, 8, 6, 6 | ~7.1 |
| Smaller Weaker Yet Better (3OyaXFQuDl) | Weak-to-strong LLM training | 6, 6, 8, 8 | ~7.0 |

This paper is:
- **Above** licAR8FPTW (toy domain only, much weaker experiments, no real human evaluation)
- **Below** PAVs (clean experiments, proper statistical support, strong theoretical grounding connected to experiments)
- **Comparable to** Neural Interactive Proofs (real LLM experiments, but methodological gaps, split reviews)

The non-iso-accuracy headline, the unvalidated causal mechanism, the flagged sneaky-prover human study (one of four stated contributions), and the non-monotonic/unstatistical human evaluation results together represent meaningful methodological gaps. These are not fatal but they do mean the paper's core causal claim is unsupported in its current form. The paper would be significantly stronger with an iso-accuracy baseline comparison and a causal ablation—neither of which is trivial to add in a rebuttal.

Positioning: slightly below Neural Interactive Proofs (similar concept, comparable rigor), clearly above the toy-domain scalable oversight paper. Score: **5.5**.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>