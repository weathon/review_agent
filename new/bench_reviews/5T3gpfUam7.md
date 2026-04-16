## Summary
This paper proposes **label annealing**, a simple finetuning regularizer for open-weight LLMs that preserves the initial model’s predictions on finetuning inputs via a KL penalty to a frozen copy of the pretrained model. The paper studies this idea across math finetuning, code finetuning, instruction tuning, and post-instruction niche-domain adaptation, and supplements the empirical study with a linear-regression analysis intended to explain why output-space regularization can preserve pretrained knowledge better than weight decay toward initialization.

## Strengths
- **Addresses a real and practically important problem.** The paper is well-motivated by a realistic setting: adapting open-weight LLMs when the original pretraining data are unavailable, making replay difficult or impossible. This is clearly stated in the introduction and is a meaningful systems/LLM finetuning problem.
- **The method is simple, easy to implement, and well matched to the setting.** Keeping a frozen copy of the initial model and regularizing prediction drift on finetuning inputs is straightforward and conceptually aligned with the goal of retaining prior capabilities without needing old data.
- **The core empirical results are directionally promising.** In the strongest settings, label annealing does materially improve the retention/adaptation tradeoff relative to direct finetuning and plain \(L_2\) regularization. For example, in math finetuning (Table 1), it recovers most of the TriviaQA and MMLU loss while maintaining strong target-task gains; in code finetuning (Table 2), it substantially mitigates the severe collapse on math benchmarks while retaining much of the HumanEval improvement.
- **The paper evaluates more than one finetuning regime.** It includes continued pretraining-like settings (math, code), instruction tuning, and post-instruction niche-domain adaptation, which is better than demonstrating the method on a single narrow case.
- **The theory section provides useful geometric intuition.** Although simplified, the linear-regression analysis helps explain the distinction between parameter-space regularization and output-space regularization on finetuning inputs, and gives a plausible intuition for why label annealing can preserve information “within the span” of finetuning data better than direct tuning.

## Weaknesses

###: Fatal
- **The main empirical claims are undermined by benchmark-tuned model selection on the same benchmarks used for reporting.**  
  Section 3.1 explicitly states that for both \(L_2\) regularization and label annealing, the authors sweep hyperparameters, **filter choices based on target benchmark performance**, and then **select the one with best source benchmark performance**. Those same target/source benchmarks are then reported in Tables 1–2 and used to support the central claim that the method preserves source performance while retaining target improvements. This is a serious evaluation flaw because the headline tradeoff is being optimized directly on the reported metrics rather than validated on held-out benchmarks or a separate validation split. Since the paper’s primary contribution is empirical, this substantially weakens the credibility of the reported gains.

### Major:
- **The paper overgeneralizes beyond what its experiments support.**  
  The empirical study is entirely on **Llama 3 8B** (base or instruct), using one fixed training recipe (5 epochs, fixed optimizer schedule), with very limited reporting of variance. The evidence supports that the method can work in these case studies, but claims such as improving target domains “without sacrificing other capabilities” are too broad. The paper itself shows task-dependent tradeoffs: e.g., in Table 2, HumanEval drops from 54.53 (direct FT) to 51.06 (label annealing), and in Section 3.3 the authors explicitly frame results as a **tradeoff** rather than no-cost preservation.
- **Baseline coverage is too weak to establish relative advantage convincingly.**  
  The main comparisons are only against direct finetuning and uniform \(L_2\) regularization toward initialization. Given the modest novelty of the core regularization idea, stronger weight-only forgetting-mitigation baselines would be important to establish whether label annealing is genuinely preferable or just better than a weak baseline. As written, the paper mainly shows superiority over direct tuning and simple shrinkage, which is not enough to make a strong comparative claim.
- **The replay comparison in Section 5 materially weakens the broader practical narrative.**  
  The paper’s motivation for a weight-only method is valid, but Table 3 shows that when a replay-style approximation is available, replay is already very competitive and often better on target metrics, with replay + label annealing only slightly improving over replay. This does not invalidate the contribution, but it does narrow the paper’s practical claim: label annealing is better viewed as a useful fallback for the no-replay setting, not as a generally strongest solution to forgetting.
- **The hyperparameter selection protocol is not only circular but also impractical.**  
  The method requires access to source benchmarks during tuning, yet one of the paper’s practical premises is that the user wants to preserve broad prior capabilities without necessarily having a comprehensive retained-capability validation suite. The paper does not provide a realistic strategy for choosing \(\lambda\) and \(T\) without oracle access to the same benchmark families used for final evaluation.
- **The computational overhead is nontrivial and under-discussed.**  
  The method requires a frozen copy of the original model and an additional forward pass every training step. For 8B-scale models this is already meaningful; for larger models it could be a major adoption barrier. A paper advocating a practically attractive finetuning method should quantify the cost-benefit tradeoff.

### Minor
- **The role of temperature \(T\) is underexplored.**  
  Equation (2) introduces temperature scaling as a key part of the method, but there is no focused ablation showing how \(T\) affects the tradeoff. That omission is notable because temperature is one of the few distinctive knobs in the method definition.
- **Some methodological claims are stated too strongly.**  
  In Section 2.2, the statement that the regularization “reduces to label smoothing” as \(T \to \infty\) is too loose as written; the connection is intuitive but not literally equivalent in the simple practical sense implied by the text.
- **The theory should be framed more carefully as motivation, not explanation.**  
  Section 4 replaces a transformer with linear regression and replaces KL over token distributions with an \(L_2\)-style penalty on predictions. This is a useful toy model, but it does not strongly explain actual transformer finetuning dynamics. The paper sometimes leans a bit too hard on the theory as if it substantively validates the empirical mechanism.
- **The benchmark suite is a limited probe of “general capabilities.”**  
  MMLU, TriviaQA, GSM8K/MATH, HumanEval, AlpacaEval, IFEval, and QuALITY are reasonable choices, but they still only sample a subset of capabilities. The paper should phrase retention claims more narrowly around the measured benchmarks rather than broader “pretraining knowledge” or “general capabilities.”
- **Some empirical interpretations are anecdotal.**  
  For example, in the code finetuning section, the dramatic MATH drop under direct finetuning is attributed to losing few-shot prompt-following ability, but this diagnosis is not directly evaluated.

### Trivial
- The name “label annealing” is somewhat potentially confusing since there is no annealing schedule described; the method is really fixed-strength KL regularization to the initial model throughout training.

## Nice-to-Haves
- Add a principled hyperparameter-selection strategy that does not rely on evaluating on the same source/target benchmarks used in the final tables.
- Report training-time and memory overhead relative to direct finetuning and \(L_2\) regularization.
- Include ablations on temperature \(T\) and robustness to \(\lambda\).
- Test at least one additional model family or scale to support broader generalization claims.
- Provide clearer reporting of variance / multiple seeds, especially for the tradeoff curves in Figures 2–3.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing related work / Learning without Forgetting not cited.”**  
  I am not including this as a review weakness because the instructions explicitly disallow criticizing missing related works when external verification is uncertain. It is fair to say the novelty is modest and the core idea is conceptually familiar, but I avoid turning that into a missing-citation complaint.
- **“Need comparison to LoRA/DoRA/PEFT specifically.”**  
  I weakened this into a broader baseline-coverage criticism. Demanding a specific family of baselines risks scope creep; the substantive issue is that the comparative evaluation is narrow.
- **“Comparison is unfair because direct finetuning uses fixed hyperparameters while the proposed method is tuned.”**  
  This is only partially valid. Direct finetuning indeed has fewer extra hyperparameters than label annealing or \(L_2\), so asymmetry here is not by itself a decisive flaw. The real problem is the circular use of evaluation benchmarks for selecting the regularized models.
- **“Contamination/overlap risk in QuALITY synthetic corpus.”**  
  The paper says the corpus is synthetic data related to QuALITY articles and uses QuALITY QA as target evaluation; while it would be helpful to clarify overlap more explicitly, the current text is not enough to conclude a concrete contamination problem from the paper alone.
- **Pure formatting/style complaints.**  
  Omitted per instructions.

## Novel Insights
The paper’s strongest contribution is not that KL-to-initial-model regularization can help—that is an intuitively unsurprising result—but that in LLM continued finetuning, the method seems especially effective when forgetting is concentrated on capabilities that are strongly activated by the finetuning inputs themselves. The linear-regression discussion, though simplified, suggests a useful way to think about this: output-space preservation on the finetuning distribution regularizes the subspace where standard fine-tuning is most destructive. That perspective helps explain why the method can dramatically recover some capabilities in Tables 1–2 while still admitting a tunable tradeoff in alignment-style settings.

## Suggestions
- **Fix the evaluation protocol first.** Use held-out validation data or separate tuning benchmarks for selecting \(\lambda\) and \(T\), and reserve Tables 1–2 / Figures 2–3 benchmarks for final reporting only.
- **Narrow the claims.** Rephrase the abstract/introduction to say the method is effective in the tested case studies and often improves the retention/adaptation tradeoff, rather than broadly claiming preservation “without sacrificing” other capabilities.
- **Strengthen baseline comparisons.** Add stronger non-replay weight-only baselines so the paper can establish comparative value rather than only beating direct tuning and plain \(L_2\).
- **Quantify overhead.** Report wall-clock slowdown, memory overhead, and any engineering tricks needed to make the frozen-reference pass practical.
- **Ablate \(T\) and hyperparameter robustness.** Since temperature is central to the formulation, the paper should show whether performance is robust or highly tuned.
- **Clarify the theory’s role.** Present Section 4 explicitly as geometric intuition rather than as a close model of transformer finetuning dynamics.

In terms of the evaluation axes: **originality** is moderate at best, as the method is simple and conceptually familiar even if applied to a timely setting; **importance of the research question** is high; **support for claims** is mixed because the empirical results are promising but the selection protocol significantly weakens them; **experimental soundness** is the paper’s main issue due to benchmark-based hyperparameter selection and limited breadth; **clarity of writing** is generally good; **value to the community** could be meaningful if the evaluation were tightened, because the problem setting is real and the method is easy to adopt.

## Score and Decision
**Calibration papers consulted:**
- **/home/wg25r/review_agent/human_reviews/IhbZytsinc.md (Minifinetuning)** — scores 5/6/5/8, reject. This is a close anchor because it is also a self-distillation / teacher-student approach for mitigating forgetting in LLM finetuning without replay. That paper had broader model coverage and more ablations, but still landed in reject territory due to novelty and evaluation concerns.
- **/home/wg25r/review_agent/human_reviews/VQ7Q6qdp0P.md** — scores 3/8/5/3, treated as reject. This is another close anchor on distillation-style forgetting mitigation. Human reviews split on whether the empirical study was enough to justify an incremental idea.
- **/home/wg25r/review_agent/human_reviews/NQEe7B7bSw.md** — scores 5/6/6/8, accept poster. This is a high-side anchor: a single-backbone LLM finetuning paper with some generalization concerns, but with stronger methodological framing and no equally severe core evaluation flaw.
- **/home/wg25r/review_agent/human_reviews/6N5OM5Duuj.md** — scores 5/6/5/8, accept poster. This is a broader continual-learning anchor where novelty is moderate but experiments and positioning were sufficient for some reviewers.

Relative to these anchors, this submission has a **promising and practical idea**, but its central empirical evidence is weakened more seriously than the accepted anchors because the reported tradeoff is selected on the same benchmarks used to prove it. That is not a minor hygiene issue; it directly affects the main contribution. I therefore place it **below the accept-poster anchors and closer to the reject-range self-distillation/forgetting papers**, though not at the very bottom because the paper is coherent, useful, and has some strong directional results.

**Score: 4.5 / 10**  
**Decision: Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>