## Summary

This paper investigates whether example selection methods for in-context learning (ICL) amplify social biases in large language models. Through extensive experiments across eight LLMs and four selection strategies on a new paraphrased sentiment benchmark (*EEC-paraphrase*), the authors find that accuracy-optimized example selection can increase maximum bias across random seeds even when mean bias decreases. To mitigate this, they propose ReBE (Remind with Bias-aware Embedding), a prompt-tuning approach that learns soft prompt embeddings via a joint accuracy and bias-contrastive loss.

## Strengths

- **Timely and understudied empirical observation.** The paper identifies an important gap: existing example selection research optimizes for accuracy while ignoring fairness risks. The observation that high-accuracy selections frequently land in a "high accuracy, high bias" region (Figure 1, scatter plot) is a useful cautionary finding for practitioners.
- **Large-scale empirical evaluation.** The experiments span eight models (LLaMA-2, OPT, GPT-J, GPT-neo) and four selection baselines (Random, Similarity, Perplexity, DPP), lending breadth to the bias-variance observation. Table 2 reports max and mean bias across all combinations.
- **Novel benchmark construction.** *EEC-paraphrase* paraphrases the Equity Evaluation Corpus via GPT-3.5-Turbo to create more naturalistic test cases than template-based stimuli, which may be of independent utility to the fairness community.
- **Empirical effectiveness of ReBE.** Table 3 and Figure 6 demonstrate that ReBE reduces maximum bias metrics (AvgGF, MaxTG, MaxFG) across selection methods without catastrophic accuracy drops, and the ablation in Table 4 isolates the role of the bias-contrastive loss.

## Weaknesses

### Fatal

None.

### Major

- **ReBE abandons the zero-update ICL paradigm that frames the paper.** The Introduction explicitly defines ICL as adaptation "without parameter updates" (Section 1), contrasting it with fine-tuning. Yet ReBE is prompt tuning: it adds virtual tokens, freezes original LLM weights, and updates soft-prompt embeddings via back-propagation and gradient descent on task data (Section 4.2: "trainable parameters … updated to minimize the loss"; Section 4.3: "optimize the bias-aware embedding via back-propagation"). This is parameter-efficient fine-tuning, not ICL. The paper repeatedly claims ReBE "retains the flexibility of ICL" and is "compatible with existing example selection methods," but a method requiring task-specific training cannot be deployed in the zero-update setting that motivates the work. The solution therefore does not address the stated problem within its own paradigm and needs to be reframed.
- **The attribution of spurious correlations to example selection is under-supported.** The paper observes gendered disparities in confusion matrices under ICL (Figure 3: male sadness→fear misclassification at 0.54 vs. female at 0.08) and attributes them to spurious correlations introduced by example selection, citing null-prompt tests (Figure 4) as isolation. However, null prompts only measure P(label | demographic token) in isolation; they do *not* rule out parametric bias in how the model jointly processes demographic tokens and sentiment words. Moreover, the paper never demonstrates that the *selected* few-shot examples actually exhibit demographic–label correlations—the defining property of a spurious correlation—nor does it analyze the content of selected sets. The inference from null prompts to "example selection contributes to spurious correlations" (Observation ❸) is insufficiently justified.

### Minor

- **Methodologically mismatched baseline comparison in Section 5.3.** ReBE is compared against counterfactual and gender-balanced context augmentations, but those baselines require no training whereas ReBE requires gradient-based optimization. A fair comparison would include a standard prompt-tuning baseline (soft prompt + cross-entropy only) trained under identical conditions across all selection methods. Table 4 includes an $\mathcal{L}_{acc}$ ablation but only for one model and without full bias metrics across all selectors.
- **Unconventional contrastive loss design is unjustified.** Equation 4 defines a bias-contrastive loss in which positives are same-label/different-demographic pairs and negatives are different-label/same-demographic pairs, while different-label/different-demographic samples are entirely excluded from the denominator. The paper does not explain why cross-demographic negative pairs are ignored, which is non-standard in supervised contrastive learning and could limit the loss's ability to push apart undesired representations.

### Trivial

- Table 3's layout with interleaved models and parenthetical deltas is difficult to parse; clearer formatting (e.g., separate rows per model or explicit variance estimates) would improve readability.
- The semantic similarity model used to compare generated outputs against answer options (Section 3.2) is not specified in the main text.

## Nice-to-Have

- **Variance decomposition across seeds.** Full distributions (violin or box plots) of bias across random seeds for each model–baseline pair, rather than just mean/max tuples, would help readers assess whether the amplification is a tail event or systematic shift.
- **Qualitative analysis of selected examples.** Concrete prompts for seeds producing extreme vs. low bias would make the "spurious correlation" hypothesis concrete rather than speculative.
- **A training-free debiasing intervention.** Since the motivating scenario is zero-update deployment, an example-selection or prompting strategy that mitigates bias without gradient-based training (e.g., balanced retrieval) would more directly address the problem as posed.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Amplification is based on an asymmetric comparison between maximum bias over random seeds and a deterministic zero-shot baseline"** — *Removed because factually wrong.* The paper explicitly states: "we have also collected the experiment results of zero-shot under various random seeds and plotted the red dashed line 'Bias of zero-shot' with the maximum bias value in Figure 1." The comparison is max-to-max across seeds, not max-to-single-point.
- **"The evidence shows mean bias actually decreases" presented as a hidden flaw** — *Removed because the paper explicitly acknowledges this.* The text states: "It is evident that, although example selections reduce the mean bias value, the LLMs tested exhibit varying degrees of increase in the maximum gender or race bias value."
- **"A proper maximum-bias baseline for zero-shot"** — *Removed because the paper already samples zero-shot under multiple seeds and reports the maximum.* Additional template variations could strengthen robustness but are not required to invalidate the existing comparison.
- **"Training with the bias loss alone drops accuracy to 0.26" presented as a flaw** — *Removed because this is the purpose of an ablation.* Showing that $\mathcal{L}_{bias}$ alone collapses accuracy while the joint objective recovers it is standard and actually supports the design.
- **"EEC-paraphrase validity"** — *Weakened to trivial because the paper notes quality validation in Appendix A (stripped by the parser).*
- **Typos, formatting, and parser artifacts** — *Removed per hard rules.*

## Novel Insights

Beyond the paper's own contributions, the review synthesis highlights that the most durable contribution of this work may be its empirical characterization of bias *variance* under example selection—specifically, that accuracy-driven selectors can produce high-variance fairness outcomes where worst-case seeds exceed zero-shot maxima. This distributional insight is more robust than the "amplification" framing and could motivate future training-free interventions (e.g., robust retrieval or ensemble prompting) that operate within the true ICL paradigm.

## Suggestions

- **Reframe the problem and solution.** Either (a) reposition the work as studying bias risks in ICL example selection and then proposing a separate parameter-efficient *fine-tuning* debiaser (ReBE) that uses ICL-selected contexts, with appropriately matched baselines; or (b) develop a genuinely training-free debiasing strategy and reserve ReBE for a follow-up paper.
- **Strengthen the spurious-correlation evidence.** Analyze the actual selected example sets for demographic–label correlations, or design a stronger isolation test (e.g., compare zero-shot with demographic cues in the query vs. ICL with selected examples) to disentangle parametric bias from selection effects.
- **Add a matched prompt-tuning baseline** (soft prompt + cross-entropy only) across all experimental settings in Table 5 to fairly assess the marginal value of the bias-contrastive loss.

## Score and Decision

**Calibration papers used:**
- *High:* `/home/wg25r/review_agent/human_reviews/oZtt0pRnOl.md` (8.00, Accept) — differentially private ICL with clear problem-solution alignment and theoretical guarantees; our paper lacks this alignment. `/home/wg25r/review_agent/human_reviews/IUmj2dw5se.md` (7.50, Accept) — comprehensive bias benchmark; our paper's benchmark is narrower. `/home/wg25r/review_agent/human_reviews/kynD1UUk6q.md` (6.75, Accept) — theoretically grounded fairness method; our paper's theoretical justification is weaker.
- *Medium:* `/home/wg25r/review_agent/human_reviews/MyVC4X5B2X.md` (5.75, Accept) — SEBRA debiasing via contrastive learning, accepted despite a strong assumption; our paper has a more severe framing issue. `/home/wg25r/review_agent/human_reviews/FEDnzAhIT4.md` (5.75, Reject) — test-time causal debiasing, rejected due to limited practical impact and presentation; comparable methodological ambition but clearer scope. `/home/wg25r/review_agent/human_reviews/WszeEzjcq2.md` (5.33, Reject) — problem-solution mismatch (NAR→AR) most analogous to our ICL→prompt-tuning gap; our empirical findings are more extensive but the mismatch is similarly damaging. `/home/wg25r/review_agent/human_reviews/7GKbQ1WT1C.md` (5.25, Accept) — causality-guided prompting, accepted despite weak theory-practice connection; our paper's connection between diagnosis and solution is weaker.
- *Low:* `/home/wg25r/review_agent/human_reviews/M7CblLwJB8.md` (2.60, Reject) — overclaimed metric and weak baselines. `/home/wg25r/review_agent/human_reviews/Y8DClN5ODu.md` (3.40, Reject) — missing critical baselines and wrong task choices. `/home/wg25r/review_agent/human_reviews/OcXsdBo6vK.md` (4.00, Reject) — unclear problem setting.

Our paper sits between the medium and low bands. It has genuinely novel empirical findings and extensive experiments that place it above the low anchors, but the core mismatch between the ICL problem framing and the prompt-tuning solution—combined with under-supported causal claims—makes it weaker than the accepted medium anchors (MyVC4X5B2X, 7GKbQ1WT1C) and comparable to the rejected ones (FEDnzAhIT4, WszeEzjcq2). The empirical contribution is salvageable with reframing.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>