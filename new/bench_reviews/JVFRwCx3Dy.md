Now I have all the information I need. Let me synthesize the final review.

## Summary

This paper investigates factors driving the emergence and stability of in-context learning (ICL) in autoregressive models. Through controlled experiments on image classification with a 3-layer GPT-2, the authors identify two key factors: (1) conceptual repetitions (iCopy) in training sequences, which create a strong in-context look-up mechanism and reduce ICL transiency, outperforming previously emphasized properties like burstiness and long-tail distributions; and (2) the complexity of the in-weight learning (IWL) objective, where harder IWL tasks (via more classes, noisy labels, or instance discrimination) lead to more stable, non-transient ICL performance.

## Strengths

- **The iCopy mechanism is a clean, well-motivated experimental manipulation that yields the paper's strongest result.** Figure 3a shows that even low-burstiness sequences with a single repetition (Q-A-B-C-D-E-F-G iCopy) achieve ICL performance comparable to high-burstiness (3xQ-3xA-B-C), while low-burstiness without repetitions shows no ICL ability. This directly challenges prior work's emphasis on burstiness as the primary driver of ICL.

- **The demonstration that IWL task complexity stabilizes ICL is a meaningful contribution.** Figure 7c provides the most striking evidence: switching from supervised to instance discrimination on the same 3600 samples/200 classes transforms a model with no ICL into one with ~80% stable ICL. This offers a new explanation for why prior properties (more classes, skewed distributions) help ICL — they make the IWL task harder.

- **The transiency analysis addresses a real gap in the literature.** The finding that combining iCopy with instance discrimination nearly eliminates ICL transiency (Figure 4a, ~80% stable across 500K steps) directly addresses the transiency problem identified by Singh et al. (2023) and aligns with the non-transient ICL observed in real LLMs.

- **Induction head analysis provides mechanistic grounding.** Figure 5 shows that iCopy+burstiness produces clear induction head circuitry (L1: previous-token attention; L2: query-to-matching-label attention), while burstiness alone does not. This connects the behavioral findings to the specific circuit architecture predicted by prior theory (Olsson et al., 2022).

- **Generalization beyond Omniglot.** Figure 4b demonstrates strong ICL on CIFAR-100, Caltech-101, and DTD with iCopy+burstiness, where burstiness alone fails entirely, showing the practical relevance of the repetition mechanism beyond a single dataset.

- **No harm to IWL from repetitions.** Figure 3b shows iCopy sequences do not degrade IWL performance compared to baseline, addressing a key limitation of prior setups where strong ICL came at the expense of IWL.

## Weaknesses

### Fatal
None.

### Major

- **The paper overclaims its relevance to LLMs without any language-model experiments.** The abstract states "we systematically uncover properties present in LLMs that support the emergence of ICL," but all experiments use a 3-layer, 1-head GPT-2 on image classification with synthetically constructed sequences. No experiment tests any claim on a language model or naturalistic text data. The paper acknowledges in the limitations that "showing a similar analysis on real-world sequential data is out of scope of this work," yet the framing throughout (abstract, Section 1, Section 6) presents the findings as explaining ICL in LLMs. Even a small-scale experiment — training a tiny LM with and without n-gram deduplication and measuring ICL — would substantially narrow this gap. As it stands, the LLM claims remain speculative, which is inconsistent with the authoritative framing. The paper would be much stronger if it narrowed its claims to match the evidence: "a controlled study of ICL emergence with potential implications for LLMs."

- **The iCopy-to-LLM analogy is imperfect and insufficiently justified.** The paper's central bridge to LLMs is that "conceptual repetitions could refer to n-gram repetitions in textual data" (Abstract). However, iCopy duplicates the exact query-answer pair in context, enabling trivial copy-by-exact-match. N-gram repetition in text means a substring appears twice in a context window — this is surface-form repetition, not repetition of a complete input-output mapping where the answer is trivially retrievable. The mechanisms these phenomena would induce are qualitatively different. The paper does not justify why these should be considered equivalent beyond the shared notion of "repetition." Section 6 uses careful language ("we speculate"), but the abstract and introduction do not.

### Minor

- **The "IWL task complexity" unifying explanation may conflate distinct mechanisms.** Section 5 proposes that four manipulations (more classes, skewed distribution, noisy labels, instance discrimination) all promote ICL through the same mechanism: making IWL harder. However, these may operate through different causal pathways — noisy labels make IWL unreliable (not just hard), instance discrimination changes the task fundamentally (28,800 classes vs. 200), and more classes makes individual predictions less frequent. The paper does not disentangle complexity per se from these confounded factors (e.g., by testing harder class boundaries at constant class count and distribution). The common observation that all four reduce IWL accuracy is consistent with a shared mechanism, but does not establish it.

- **No error bars or variance reporting despite acknowledged high variance.** The limitations section states "we observed a large variance in the ICL performance curves w.r.t. random seeds," yet no learning curve plot shows error bars or confidence intervals. This makes it difficult for readers to assess the significance of differences between conditions, particularly where curves are close (e.g., Figure 3a near peak ICL).

- **Induction head analysis is limited to a single snapshot.** Figure 5 shows attention maps at peak ICL performance only. Given the transiency phenomenon — where ICL emerges and then degrades — showing how induction heads form and dissolve over training in both conditions would be far more informative about the causal relationship between induction heads and ICL stability.

### Trivial
None.

## Nice-to-Haves

- An ablation comparing iCopy (exact copy of query-answer pair) against a single non-copy same-class instance would isolate whether the mechanism depends on exact duplication or mere same-class presence, strengthening the "repetitions" vs. "burstiness" argument.
- A time-series of attention map evolution showing induction head formation/dissolution dynamics.
- Testing iCopy in a text domain, where repeating an exact query-answer pair in context would directly test the analogy to LLMs.
- Disentangling "harder IWL" from "less reliable IWL" by comparing noisy labels against equally hard but reliable tasks (e.g., harder decision boundaries at same class count).

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Critic's claim that "the 90% in-context / 10% standard split is far from the LLM setting."** This is a design choice for a controlled study, not a flaw. The paper's goal is to understand what drives ICL in isolation, not to replicate LLM training conditions exactly. Moreover, the paper explicitly discusses how the mix of standard/in-context sequences is a hyperparameter.
- **Critic's claim that Figure 3's comparison is "not fully fair" because iCopy has 1 query-class instance vs. 3 in high-burstiness.** The comparison is explicitly designed to test whether a single repetition (with lower burstiness) can match high burstiness — the answer (yes) is the paper's key finding. This is not an unfair comparison; it's the point of the experiment.
- **Critic's claim about "missing ablation with single non-copy query-class instance."** This is a reasonable suggestion but falls under nice-to-have rather than a weakness; the paper already shows that low-burstiness without iCopy (Q-A-B-C-D-E-F-G) shows no ICL, which addresses the same question from the opposite direction.
- **Strength Finder's claim about "empirical motivation from real pretraining corpora" (Figure 1).** While Figure 1 does quantify n-gram repetitions in LLM pretraining data, this only establishes correlation (n-grams repeat), not causation (n-gram repetition causes ICL in LLMs). Given that this is precisely the gap the paper fails to close with experiments, listing this as a strength that directly supports the main claims would be misleading. Moved to a supporting observation rather than a core strength.
- **Critic's claim about "no validation on any actual language model" as a Fatal/Evidential issue.** This is real but better categorized as Major rather than Fatal — the paper does make genuine contributions within its controlled setting; the issue is the mismatch between claims and evidence, not a fundamental flaw in the methodology itself. The paper's findings (iCopy promotes ICL, harder IWL stabilizes ICL) are valid within the controlled setup.

## Novel Insights

The paper's most insightful contribution is the decomposition of ICL emergence into two independent levers — data-sequence-level (repetitions enabling look-up) and objective-level (IWL complexity preventing transiency) — which provides a cleaner framework than prior work that focused primarily on data distribution properties. The observation that these two factors interact (iCopy + instance discrimination nearly eliminates transiency) suggests that ICL stability in real LLMs may arise from the conjunction of repetitive structures in text AND the inherent complexity of language modeling, rather than from either factor alone. This dual-factor view is a genuine conceptual advance over prior single-factor explanations.

## Suggestions

- Narrow the abstract and introduction claims from "we systematically uncover properties present in LLMs" to "we systematically uncover properties that support ICL emergence in controlled settings, with potential implications for LLMs." This single change would substantially improve the paper by aligning claims with evidence.
- Add at least one small-scale language model experiment (e.g., train a tiny LM with and without n-gram deduplication and measure ICL performance) to provide direct evidence for the LLM analogy.
- Report standard deviations across 3-5 seeds for key metrics (peak ICL accuracy, final-step ICL accuracy) to address the acknowledged variance concern.

## Score and Decision

**Calibration anchors:**

- **aN4Jf6Cx69** (avg 9.0, Accept Oral): Studies ICL + data properties + induction heads in a minimal attention-only network. Much deeper mechanistic analysis (three-parameter phenomenological model, progress measures). Also uses controlled experiments but makes appropriately scoped claims. The paper under review is clearly below this — less mechanistic depth, overclaimed scope.

- **ekeyCgeRfC** (avg 7.0, Accept Oral): Controlled study of ICL on discrete Boolean functions. Includes some LLM validation (LLaMA-2, GPT-4). The paper under review lacks this direct LLM connection but has a broader range of controlled manipulations. Below this due to the overclaim gap.

- **79ZkWgY2FI** (avg 5.25, Accept Poster): Studies small-to-large generalization; questioned for insufficient insight and unclear message. The paper under review has clearer contributions (iCopy finding, transiency reduction) but similarly overclaims generalization. Roughly comparable quality.

- **JopVmAPyx6** (avg 4.5, Reject): IWL vs ICL in synthetic task; rejected for limited technical contribution and unclear writing. The paper under review has significantly more experiments and clearer contributions, so it is above this.

- **Hikig6ljsd** (avg 2.5, Withdrawn): Overclaims LLM generalization from limited evidence. The paper under review has stronger methodology and genuine findings, clearly above this.

The paper makes genuine contributions within its controlled setting but overclaims LLM relevance. It sits between the medium-quality anchors (5.25-5.5) and the higher-quality ICL papers (7.0-9.0). The overclaim is the primary factor holding it down from a higher score; the actual experimental contributions are solid.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>