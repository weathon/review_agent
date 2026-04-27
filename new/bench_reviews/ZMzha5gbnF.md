Now let me run calibration searches to anchor the score.Now I have sufficient information to write the consolidated review. Let me synthesize everything.

---

## Summary

This paper identifies and formalizes the *priming vulnerability* in Masked Diffusion Language Models (MDLMs): if affirmative tokens toward a harmful query appear at intermediate denoising steps, subsequent generation is steered toward a harmful response even in safety-aligned models. The paper introduces: (1) an anchoring attack under a hypothetical intervention threat model to characterize this vulnerability; (2) First-Step GCG, a theoretically grounded and computationally efficient realistic jailbreak that exploits this vulnerability without process intervention; and (3) Recovery Alignment (RA), an RLHF-style defense that trains models to recover from adversarially contaminated intermediate states. Experiments across three MDLMs and multiple attack families demonstrate RA significantly outperforms existing alignment baselines with minimal capability degradation.

---

## Strengths

- **Novel, MDLM-specific vulnerability with clear mechanistic explanation.** The paper identifies a structural difference between MDLMs and ARMs — the parallel, bidirectional denoising process — and concretely shows why standard alignment (training from fully masked sequences) fails to protect against affirmative tokens injected at intermediate steps. This is a genuine contribution to emerging MDLM safety research.

- **First-Step GCG is both theoretically motivated and empirically validated.** Theorem 4.1 establishes that first-step log-likelihood is a lower bound on the full generation probability, providing principled justification for the surrogate objective. Table 1 validates this: First-Step GCG is ~20× faster (0.2h vs 4.3h per prompt) and achieves 2–4× higher ASR than Monte Carlo GCG (e.g., 58.0% vs 20.0% on LLaDA Instruct), making this both a theoretical and practical advance.

- **Recovery Alignment achieves strong and consistent gains across three architectures.** Table 2 shows RA reduces the anchoring attack ASR from 44.0% (original) and 24.0% (best baseline MOSA) to 1.3% on LLaDA at t=4; similar gains hold for LLaDA 1.5 and MMaDA. The ablation "RA w/o inter" (which omits contaminated-state training) fails similarly to MOSA, directly validating that the contaminated-state conditioning is the key ingredient.

- **Generalization to conventional jailbreaks.** Table 3 shows RA reduces PAIR ASR from 44.3% to 10.0% and Crescendo ASR from 81.3% to 45.0% on LLaDA, outperforming all baselines — suggesting the recovery capability learned from intermediate-state training generalizes beyond the priming mechanism. The paper provides a plausible mechanism for this.

- **Curriculum ablation directly supports the design choice.** Figure 3b shows that linear scheduling outperforms both constant and uniform schedules. Constant scheduling at large $t_{\max}$ fails entirely due to training instability, providing a concrete justification for the curriculum design.

- **Capability preservation validated on 11 benchmarks.** Table 4 shows LLaDA and LLaDA 1.5 average accuracy essentially unchanged (52.2→52.6 and 52.7→52.8 respectively), and MMaDA actually improves, suggesting RA does not meaningfully trade off utility.

---

## Weaknesses

### Fatal
None.

### Major

- **RA's defense is not evaluated against adaptive attacks, and the training/evaluation distribution overlap is uncontrolled.** RA trains on harmful query-response pairs from BeaverTails, and the anchoring attack evaluation uses harmful responses from the same class of non-aligned model. An adversary aware of RA's training distribution who constructs contaminated states from a different source — or adaptively optimizes harmful responses to bypass RA's training coverage — is never tested. The conventional jailbreak experiments (Table 3) do not probe this because they do not exploit the priming mechanism. The robustness gap against ReNeLLM (72.3% on RA-LLaDA vs 77.7% MOSA) hints that adversaries who bypass surface-form detection can circumvent RA in ways the current evaluation cannot detect. Without any adaptive evaluation, the true robustness boundary of RA is unknown.

### Minor

- **Residual vulnerability at late intervention steps ($t_{\min}=32$) is underanalyzed.** Table 2 shows RA achieves 50.7% ASR for LLaDA and 43.0% for LLaDA 1.5 at $t_{\min}=32$. The paper attributes this to "practically impossible to generate a contextually safe response due to many anchors" but does not explain why the training schedule ($t_{\max}=32$) does not cover this regime or what would happen with a larger $t_{\max}$ (Figure 3a shows reward hacking at large $t_{\max}$). This leaves the upper boundary of RA's coverage genuinely unclear.

- **The monotonicity assumption in Theorem 4.1 is stated without validated scope for the safety-critical setting.** The lower bound derivation assumes $\log\pi_\theta(\tilde{r}_{t+1}=r|q,r_t) \geq \log\pi_\theta(\tilde{r}_1=r|q,r_0)$ for all steps. The paper validates this in Appendix C.2 "across a broad range of models," but does not confirm it holds specifically for well-aligned models responding to borderline harmful queries — the exact regime where alignment is strongest and the assumption is most likely to fail. The practical success of First-Step GCG (58% ASR) provides strong empirical evidence the approach works, but the theoretical justification has scope limitations the paper should surface in the main text.

- **HumanEval performance drop is not discussed.** Table 4 shows HumanEval declining from 22.0% to 17.1% on LLaDA with RA (~22% relative decrease), and PIQA from 74.4% to 71.6%. The paper claims "no substantial degradation" and this holds on average, but the coding benchmark drop is non-trivial and unaddressed. This is especially notable since HumanEval tests instruction-following and generation quality — capacities directly relevant to alignment training.

### Trivial

- **Framing of severity in the abstract and introduction.** The abstract's phrase "can be readily bypassed" and the general urgency framing rest primarily on the hypothetical intervention experiments. The paper does clearly distinguish the two threat models in Section 4, but a reader who only reads the abstract or introduction may overestimate the ease of exploitation in realistic deployment settings. Anchoring the severity claim in the First-Step GCG results (58% ASR) as the lead practical evidence would improve accuracy.

---

## Nice-to-Haves

- **DPO-style variant of RA.** The paper acknowledges this in the Limitations section. Since DPO is already a baseline in Table 2, applying DPO to contaminated intermediate states (data-augmented RA) would be a lightweight experiment that clarifies whether GRPO optimization is necessary or whether any training on contaminated states, regardless of optimizer, is sufficient for mitigation.

- **Qualitative examples of RA recovery.** Showing a concrete case where a contaminated intermediate state (with affirmative tokens) is followed by a safe final response from the RA-trained model would concretize the recovery mechanism and distinguish genuine recovery from output suppression.

- **Discussion of which deployment scenarios make the intervention attack relevant.** Since the anchoring attack requires internal process access, clarifying which real-world deployment contexts (e.g., diffusion models with user-accessible sampling APIs) enable this would help readers contextualize the hypothetical threat.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "MC GCG baseline may be under-resourced."** The paper runs both methods for 500 iterations under the same protocol. First-Step GCG is faster because it avoids stochastic gradient estimation, not because MC GCG gets fewer resources. The comparison is methodologically fair — First-Step GCG is architecturally more efficient. Removed as not a genuine flaw.

- **Harsh Critic: "Qualitative analysis of harmful response quality from unaligned model."** Whether BeaverTails harmful responses are "mild" or "maximal" in harmfulness affects interpretation of absolute ASR numbers but not the relative comparisons between methods or the validity of the priming mechanism demonstration. Removed as scope creep.

- **Harsh Critic: "Narrative overstates severity."** The paper explicitly labels the intervention threat model as "hypothetical" in both Section 4.1 and the Introduction ("we assume a hypothetical attacker who can intervene in the denoising process for comprehensive evaluation"). The framing is transparent, and the practical First-Step GCG results (58% ASR) are independently compelling. Downgraded to Trivial framing note; the "inflated severity" criticism was too strong.

- **Strength Finder: "Identical capability preservation for LLaDA 1.5."** Technically confirmed (52.7→52.8) but the claim of "no impact" is weakened by the HumanEval drop, which applies to LLaDA. Partially removed — the average preservation is real but incomplete as a strength claim given task-level variation.

---

## Novel Insights

The most genuinely novel insight in this paper is the theoretical and empirical unification of a two-phase analysis: the anchoring attack demonstrates that the priming vulnerability is inherent to the MDLM denoising mechanism (not a quirk of any specific model), while First-Step GCG shows that an attacker who cannot intervene in inference can still exploit the same structural weakness through gradient-based prefix optimization. The connection between these — via Theorem 4.1 and Figure 2 — reveals that the vulnerability is not just about intervention capability but about the fundamental relationship between early-step predictions and full-sequence generation in MDLMs. RA then exploits the converse: training from early contaminated states propagates safety recovery across later steps, generalizing even to attacks that do not directly exploit the priming mechanism.

---

## Suggestions

1. **Address adaptive attacks in at least a discussion section.** Characterize what an adaptive adversary (aware of RA's training distribution or training objective) would need to do to circumvent RA, even without running full experiments. This would help readers calibrate how robust the defense actually is.

2. **Acknowledge and analyze the HumanEval drop explicitly.** The 22% relative decrease on a coding benchmark after RLHF-style alignment is a meaningful side-effect that should be discussed alongside the aggregate "no degradation" claim.

3. **Bring the monotonicity assumption validation scope into the main text.** Briefly note what types of models or query distributions Appendix C.2 covers and whether safety-aligned models on harmful queries are included in the validation set, so readers can judge the theorem's applicability.

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Human Score | Comparison |
|---|---|---|
| `/home/wg25r/review_agent/human_reviews/r42tSSCHPh.md` | 7.0 (spotlight) | "Catastrophic Jailbreak" — novel attack on ARMs, simple method, very broad impact across 11 models; less theoretical, no new alignment method. Stronger impact due to broader model coverage. |
| `/home/wg25r/review_agent/human_reviews/aSy2nYwiZ2.md` | 6.67 (poster) | "JailbreakEdit" — backdoor injection, accepted poster; comparable novelty but narrower contribution (attack only, no defense). |
| `/home/wg25r/review_agent/human_reviews/sULAwlAWc1.md` | 7.0 (poster) | "ArrAttack" — robust jailbreak prompt generation; strong empirical results but again attack-only. |
| `/home/wg25r/review_agent/human_reviews/Nazzz5GJ4g.md` | 5.4 (rejected) | "Weak-to-Strong Jailbreaking" — novel framing, rejected; had missing adaptive experiments and unclear scope. |
| `/home/wg25r/review_agent/human_reviews/u08UxVNdIo.md` | 4.75 (withdrawn/rejected) | "Diffusion Attacker" — uses diffusion models for LLM jailbreak generation; unclear benefits, weak empirical validation. |
| `/home/wg25r/review_agent/human_reviews/UWuTZYPSxJ.md` | 2.5 (rejected) | "KDA" — jailbreak via knowledge distillation; no novel insight, weak contribution. Clearly below the paper under review. |

**Positioning:** The paper under review falls between the 6.67–7.0 cluster (accepted posters/spotlight) and the 5.4 cluster (rejected). It combines an attack and defense contribution with theoretical backing — more than most 7.0-range papers in this domain — but targets a narrower, emerging model class (MDLMs vs. widely deployed ARMs). The major weakness (no adaptive attack evaluation) is present in some accepted papers in this space. The theoretical contribution (Theorem 4.1), multi-model evaluation, and ablation quality push it above the 5.4 rejection threshold. I place it at **6.0** — solid poster-level, with the adaptive attack gap being the main hold-back from higher.

**Final Score: 6.0 / Accept (Poster)**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>