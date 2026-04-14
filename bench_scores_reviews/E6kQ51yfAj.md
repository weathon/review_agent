## Summary

This paper proposes Game-theoretical Preference Optimization (GPO), which frames LLM alignment as a two-player zero-sum game between an adversarial agent (generating challenging prompts) and a defensive agent (generating safe responses). The adversarial agent is equipped with diversity rewards (SelfBLEU and sentence-embedding-based) to prevent collapse onto a narrow attack distribution. The authors prove that an idealized variant of the iterative algorithm converges to an ε-approximate Nash equilibrium at rate O(T^{-1/2}), and demonstrate empirical gains over RLHF and static baselines on safety and jailbreak benchmarks.

---

## Strengths

- **Adaptive prompt distribution as a principled RL objective.** Unlike prior two-player alignment work (SPIN, self-play reward-hacking mitigation) that fixes the prompt set and lets the two agents compete on responses, this paper makes *prompt generation itself* a learned RL objective. The max-min formulation in Eq. (3.1) is a clean formalization of this idea, and it addresses a real gap in standard RLHF where D_PPO is static and cannot target the current defender's blindspots.

- **Diversity mechanism with clear empirical motivation.** The combination of SelfBLEU and semantic-embedding diversity rewards is concretely motivated and Table 2 / Figure 2 provide unambiguous evidence that without diversity rewards the adversary collapses to a narrow, high-toxicity distribution—reducing its utility as a training partner for the defender. The ablation over diversity intensity (k ∈ {0,1,5,10}) provides useful practical guidance.

- **Convergence analysis with FTRL grounding.** The no-regret/FTRL argument connecting Algorithm 1 to Nash equilibrium convergence is technically sound for the idealized variant. Leveraging the fact that the zero-sum objective is linear in both π and μ (enabling the minimax theorem and CCE→NE reduction) is the right approach. This is more principled than most heuristic adversarial training papers.

- **Attack transfer evaluation.** Testing the trained adversarial agent on three *held-out* target models (Llama-2-7b-chat, Vicuna-7b, RLHF model) rather than only the opponent it trained against is a strong experimental design choice. GPO+Div consistently achieves higher ASR with competitive or better diversity than RLHF+Div, suggesting the game dynamic—facing a stronger, co-evolving defender—induces better generalization in the attacker than training against a fixed weaker opponent.

---

## Weaknesses

### Fatal
None.

### Major

- **No comparison against MART (Ge et al., 2023), the most directly comparable baseline.** The paper explicitly calls out MART in related work ("MART iteratively conducts red teaming and safety enhancements but relies on supervised fine-tuning, which makes it difficult to balance the capabilities of attackers and defenders") and positions GPO as its successor with RL-based co-training. Yet MART is never included in Tables 1–3. Without this comparison, it is impossible to determine whether GPO's gains stem from the *game-theoretic RL co-training* or simply from doing any form of iterative red-teaming with an RL objective. This is the central empirical gap.

- **Cosine similarity formula in Eq. (3.5) appears incorrect.** The denominator is written as ‖φ(x)‖² ‖φ(x′)‖², whereas cosine similarity requires ‖φ(x)‖ · ‖φ(x′)‖ (product of norms, not sum of squared norms). If the formula as written is what was implemented, the embedding diversity reward is not measuring cosine similarity but something else entirely, which would affect training dynamics and the interpretation of diversity results in Table 2. The authors should clarify what was actually implemented and correct the formula if it is a typo.

- **Diversity reward magnitude grows unboundedly with history.** In Eq. (3.5), the sum runs over *all* previously generated attack prompts X. As training progresses, |X| grows, so the diversity reward's scale increases monotonically while the safety reward r(x,y) is bounded. This creates a non-stationary optimization landscape where later-round adversaries are penalized for diversity-deficiency at a much higher scale than early-round adversaries. The paper does not describe any normalization (e.g., dividing by |X|, windowing) or discuss this instability. Combined with the asymmetric step schedule (200 defense / 400 attack), this could explain why "selecting a moderate intensity is found to be more effective"—at high k, unbounded diversity penalty would eventually dominate—but no analysis is provided.

### Minor

- **Theory-practice gap is acknowledged but under-analyzed.** Section 3.3 explicitly states that the theoretical version of Algorithm 1 differs from the practical one in four ways: averaged vs. last-iterate policies, uniform initialization, exact argmax/argmin, and no optimization error. The practical PPO procedure satisfies none of these. The paper treats this honestly, but stops short of discussing whether the NEGap actually decreases empirically during training—reporting this quantity (or even qualitative training curves of both agents' rewards) would substantially strengthen the connection between theory and practice.

- **Adversarial agent action space ambiguity.** Equation (3.1) treats μ_φ as an unconditional prompt generator (x ~ μ_φ(·)), but the experimental setup describes the adversary as *conditionally transforming* original harmful prompts into "similar but more harmful variations." These are materially different setups: the former is open-ended generation, the latter is conditional rewriting. The discrepancy affects the interpretation of the theoretical result (which assumes unconditional generation) and needs explicit reconciliation.

- **Potential reward circularity unaddressed.** Both training rewards and evaluation metrics (ASR, r_safe) appear to derive from the same safety classifier family (Llama-Guard / ToxiGen-based). Whether safety improvements reflect genuine robustness or calibration to that specific classifier is not evaluated. Testing against a second, independent safety classifier or using human judgments for a subset of outputs would substantially increase confidence in the results.

- **Computational cost unquantified.** The paper acknowledges in the conclusion that training two LLM agents simultaneously is a limitation, but provides no wall-clock time, GPU-hour, or total parameter-update comparison against standard RLHF. For readers considering adoption, this information is necessary.

### Tiny

- The claim "as far as we know, our work is the first to investigate two-player games from this perspective" in Section 5 is plausible but should be more carefully qualified, given the breadth of adversarial training, automated red-teaming, and self-play alignment literature.
- Figure 2 reports training curves and diversity-intensity ablation without error bars, making it difficult to assess whether differences between k values are meaningfully larger than training variance.
- No ablation on iteration count T or the specific step schedule (200 defense / 400 attack)—both are critical hyperparameters whose sensitivity is entirely unexplored.

---

## Nice-to-Haves

- **Qualitative prompt evolution examples.** The "natural curriculum of increasing complexity" is a central motivational claim, but no qualitative evidence is provided. Showing how adversarial prompts evolve across game iterations—in terms of attack strategy, phrasing, and semantic coverage—would make this concrete and potentially reveal whether the agent discovers genuinely novel attack strategies or merely rephrase known patterns.

- **Embedding-space visualization of prompt distribution.** A t-SNE/UMAP plot of adversarial prompts at different training iterations, comparing GPO vs. GPO+Div, would directly substantiate the diversity reward's effect on prompt coverage—a cleaner demonstration than the scalar diversity metric alone.

- **Evaluation against qualitatively different attack strategies.** The paper tests on held-out prompt *datasets* (OOD in distribution), but does not evaluate whether the GPO-trained defender withstands qualitatively different attack *strategies* (e.g., GCG suffix attacks, multi-turn manipulation). This would more robustly support the "enhanced generalization" claim.

- **DPO as an alternative defense backbone.** Since DPO avoids RL entirely and hence sidesteps reward hacking, a brief exploration of whether the adversarial prompt generation framework can drive DPO-based defense updates (as foreshadowed in the conclusion) would broaden the paper's impact.

- **Evaluation on a non-safety task.** The conclusion explicitly acknowledges the intent to extend to helpfulness and reasoning; even a single experiment on GSM8K or similar would clarify whether the framework is safety-specific or genuinely general.

---

## Removed Points

*These points were flagged for removal; treat them with caution if revisiting.*

- **No variance estimates / single-run statistics (Harsh Critic).** At the scale of LLM RLHF training, single-run evaluation is standard in the field. Demanding confidence intervals or multiple seeds is not a standard expectation for this setting and does not constitute a meaningful weakness.

- **Weak scope criticism re: broader alignment tasks (Harsh Critic / Spark Finder).** The paper explicitly scopes to safety ("this work primarily focused on prototyping our idea using safety-related tasks") and lists non-safety extension as future work. Faulting the paper for not yet doing experiments in reasoning/helpfulness is scope creep. The overstated "optimal LLM alignments" in the title is noted in Minor above.

- **DPO/SimPO missing as alignment baselines (Spark Finder).** DPO is an off-policy, prompt-distribution-agnostic method. Integrating it as a defense agent is a different systems design requiring non-trivial work. Comparing defense-side performance against DPO as a drop-in is not standard practice here, and the paper acknowledges DPO as a future direction.

- **Missing related work demands.** Per the review instructions, claims about missing related works cannot be verified without external sources and are not included.

- **Formatting and venue-tag concerns.** Not relevant to scientific content.

- **Criticism that the diversity term in the shared payoff is "conceptually odd" for the defender (Harsh Critic).** Inspection of Eq. (3.1) shows that R_div(x) depends only on x, not y. The defender optimizes over y given the prompt x, so for the defender, -β_div R_div(x) is a constant and the effective objective is just maximizing E[r(x,y)]. There is no conceptual oddity.

- **Unfair comparison: RLHF adversary trained against a weaker opponent than GPO adversary (Harsh Critic).** This asymmetry favors RLHF (an easier training environment), not the proposed method. Comparing GPO-adversary (stronger opponent) vs. RLHF-adversary (weaker opponent) puts RLHF at an advantage, making GPO's superior attack transfer a *stronger* result, not a confounded one. This criticism is removed per the review instructions.

---

## Novel Insights

The most genuinely novel observation across the reviews—one not fully discussed in the paper itself—concerns the **interaction between diversity reward non-stationarity and training stability**. Because the diversity penalty in Eq. (3.5) grows with |X| throughout training while the safety reward is bounded, the effective balance between attack aggressiveness and diversity shifts systematically across iterations. This implicit curriculum—early rounds are more aggressiveness-driven, later rounds increasingly diversity-driven—may be a hidden mechanism behind the "moderate intensity is preferable" finding. If true, explicitly scheduling the diversity coefficient (rather than holding it fixed at k) could yield a more principled and effective training procedure. The paper presents this intensity effect empirically but does not recognize it as a consequence of the unbounded sum formulation. Addressing this could simultaneously resolve the normalization issue and improve practical performance.

---

## Suggestions

1. **Benchmark against MART.** This is the single most impactful experiment missing from the paper. Reuse the same training setup and evaluation datasets; report ASR and r_safe for MART's defender and ASR/diversity for MART's attacker.

2. **Fix or clarify Eq. (3.5).** Verify whether the implementation uses standard cosine similarity (divide by ‖φ(x)‖ · ‖φ(x′)‖) or the formula as written (divide by squared norms), and correct the equation accordingly. Include a note in the appendix on numerical behavior.

3. **Normalize or window the diversity reward sum.** Divide Eq. (3.5) by |X| or restrict the reference set to a sliding window of recent prompts to prevent reward magnitude from growing with training length. Report how this affects training dynamics.

4. **Reconcile unconditional vs. conditional adversary formulations.** Add a paragraph in Section 3.1 or Appendix B explicitly describing the conditional prompt-transformation setup used in experiments, and note which theoretical results still apply (or which assumptions they require).

5. **Report empirical NEGap during training.** Add a subplot to Figure 2 showing NEGap(π̂_T, μ̂_T) computed on a held-out set as a function of training steps. This bridges the theory-practice gap and gives concrete evidence that the algorithm is actually converging toward equilibrium in practice.

6. **Add a cross-classifier safety evaluation.** Pick 200–300 examples from the held-out Anthropic set and evaluate defensive agent outputs using a *different* classifier (e.g., if training used Llama-Guard, evaluate with a ToxiGen-based classifier, or with human raters via crowdsourcing). This would substantially strengthen the claim that safety gains are genuine rather than classifier-specific.

7. **Quantify computational overhead.** Report total GPU-hours, wall-clock training time, and peak memory for GPO+Div vs. standard RLHF on the same base model and dataset, so readers can assess practical viability.

---

**Evaluation summary:** The paper introduces a genuinely interesting and novel approach—treating the prompt distribution itself as a learnable adversarial policy co-trained against a defending LLM under an RL objective with diversity constraints. The convergence analysis is technically sound for its idealized scope. Empirically, results are consistent and the transfer-attack evaluation design is strong. However, the paper has meaningful technical issues (cosine similarity formula, diversity reward scaling), a critical missing baseline (MART), and an insufficiently analyzed theory-practice gap. In its current form, the contribution is solid but the evidence base needs to be shored up before it fully supports its claims. The paper's significance is **moderate-to-high** for the safety alignment community, its novelty is **genuine and specific** in the prompt-generation-as-game framing, its technical soundness is **moderate** (pending the formula correction and normalization), and its empirical support is **moderate** (strong design, but key baseline absent).