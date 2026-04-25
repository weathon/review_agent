## Summary

The paper investigates whether latent “belief directions” extracted from LLM representations encode context‑sensitive truth‑value judgments and causally mediate inference. Using multiple probing methods (CCS, CCR, MMP, LR) on two datasets (EntailmentBank, SNLI) and three models, the authors show that these directions respond to premise context, define four error scores to quantify consistency violations, and perform an intervention that moves premise representations along the belief direction to measure effects on hypothesis probabilities. The paper introduces the CCR probing variant and reports that instruction‑tuning shifts error patterns. The core claims are that belief directions are context‑sensitive and act as causal mediators of in‑context information.

## Strengths

- **Systematic comparison across probing methods, models, and datasets.** The paper evaluates CCS, CCR, MMP, and LR on Llama2-7B/13B and OLMo-7B (± instruction tuning) using both EntailmentBank and SNLI, providing a comprehensive view of how belief probes behave under varied conditions (Section 4, Table 2, Figures 2–3).  
- **Novel CCR probing method with stable convergence.** CCR replaces CCS’s equidistance requirement with a reflection constraint, avoiding degenerate solutions without training multiple probes (Section 3.1, Equation 2).  
- **Causal intervention experiment takes a step beyond correlation.** The authors manipulate premise representations along the belief direction and measure resulting changes in hypothesis probabilities, finding expected shifts for entailed and contradicted hypotheses (Section 4.2, Figure 4).  
- **Nuanced error‑characterization framework.** The four error scores (E1–E4) quantify distinct kinds of inconsistency, enabling detailed analysis of how probes incorporate irrelevant contexts or mishandle negation (Section 3.3, Table 1).  
- **Open‑source release.** All code is made available, supporting reproducibility.

## Weaknesses

### Fatal
None.

### Major

1. **Overstated causal‑mediation claim.** The abstract and conclusion assert that belief directions are “(one of the) causal mediators in the inference process.” However, the intervention only shows that artificially moving premise representations along the belief direction influences hypothesis probabilities; it does **not** establish that the direction *mediates* the natural causal effect of the premise on the hypothesis. Proper causal mediation requires (a) the premise affecting the mediator, (b) the mediator affecting the hypothesis, and (c) the premise→hypothesis effect being blocked when the mediator is blocked—none of which is demonstrated. The evidence supports a weaker claim: belief directions can *causally influence* hypothesis representations when directly manipulated. (Section 4.2; Figure 4; abstract)

2. **Inadequate baselines to distinguish belief from spurious correlations.** The paper does not include essential control probes to rule out that identified directions encode properties correlated with truth in the training data but unrelated to belief (e.g., sentiment, specificity, lexical overlap). The LM‑head baseline measures token probabilities, not representation geometry, and therefore does not address this concern. Missing are baselines such as (i) probes trained with permuted truth labels, (ii) probes for unrelated attributes (sentence length, formality), or (iii) directions obtained from random sentence pairs. Without such controls, the identification of “belief directions” is circular: directions are defined as those separating true/false sentences and then said to represent belief. (Section 4.1; Table 2)

3. **Problematic operationalization of “truth” and “belief”.** The evaluation relies on datasets where “truth” is dataset‑specific and may not reflect real‑world factuality. SNLI labels describe image content unseen by the model; EntailmentBank contradictions are artificially constructed by mixing correct premises with incorrectly answered questions. This raises the possibility that probes are learning dataset‑specific annotation artifacts (e.g., hypothesis‑only biases in SNLI) rather than a general truth representation. While the paper acknowledges hypothesis‑only bias, cross‑dataset consistency does not fully eliminate this concern because both datasets could share similar artifacts. Moreover, the term “belief” is used throughout without justification, conflating truth‑value judgment with psychological belief. (Section 4; Footnote 3)

4. **Inconsistent mediation evidence across probing methods.** Figure 4 shows that the supervised LR probe exhibits minimal intervention effects despite showing premise sensitivity and good accuracy. If belief directions are causal mediators, why does a method that accurately separates true/false sentences fail to mediate? This discrepancy undermines the universality of the mediation claim and is not addressed by the authors. (Section 4.2; Figure 4)

5. **Error scores rely on debatable normative assumptions.** E3 and E4 assume a rational agent that treats premises as truthful and bases beliefs on its own evaluation of premise truth. These assumptions may not hold for LLMs, which can represent truth‑values without committing to premise truth. While the scores are presented as descriptive metrics, framing them as “errors” implies normative shortcomings that may be irrelevant to how LLMs operate. The paper would be stronger if these were reframed as patterns rather than errors. (Section 3.3; Table 1)

### Minor

- The reported intervention effect sizes are modest (≈0.05–0.10 absolute probability changes), and the paper does not discuss practical significance.  
- Layer‑wise analyses (Figures 2–3) show high variability; more precise discussion of where key transitions occur and why would be helpful.  
- The paper does not test whether belief directions generalize to out‑of‑distribution domains beyond the two datasets used (e.g., factual QA, mathematical reasoning).  
- The theoretical distinction between prior/conditional/marginal beliefs (Section 3.2) is not cleanly mapped to the empirical operationalization; the connection could be elaborated.  
- Some figure labels are small (e.g., axis labels in Figure 4), reducing readability.

### Trivial
None beyond formatting artifacts.

## Nice‑to‑Haves

- **Control experiments:** (i) train probes with permuted labels to show specificity; (ii) train probes to predict unrelated attributes (sentence length, formality); (iii) test whether belief directions correlate with lexical‑overlap features.  
- **Intrinsic analysis of directions:** e.g., top‑activating examples for neurons/components aligned with belief directions, or projection onto known semantic axes.  
- **Evaluation on factual benchmarks:** Use datasets with verifiable ground truth (e.g., TrueDate) to reduce subjectivity.  
- **Theorize method differences:** Explain why different probing methods (CCR/MMP vs. LR) yield different mediation strengths.  
- **Orthogonality checks:** Test whether belief directions are orthogonal to spurious‑correlation directions.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Removed 1:** Critic claims CCS/CCR’s unsupervised assumption is misleading because negation may change meaning in other ways. This is invalid because the paper uses a meta‑statement negation scheme that sidesteps the issue (Section 4: bracketed “in/out” avoids presupposition problems).  
- **Removed 2:** Critic says the TVJ analogy to child language acquisition is problematic. This is a minor framing issue, not a substantive flaw.  
- **Removed 3:** Any criticism about missing appendices, proofs, or references is automatically excluded per guidelines.  
- **Removed 4:** Critic suggests that Table 2 accuracy is misleading due to hypothesis‑only bias. The paper explicitly acknowledges this and includes the LM‑head baseline; the point is addressed.

## Novel Insights

Beyond the paper’s own contributions, the results reveal an inherent **trade‑off between robustness to irrelevant context and proper handling of premise negation**. No probing method simultaneously achieves low E1/E2 (sensitivity to corrupted/unrelated premises) and low E3/E4 (appropriate response to premise polarity); methods that reduce irrelevant sensitivity tend to increase E4 errors, and vice versa (Table 2). This suggests that the features used to judge truth‑value may be partitioned into distinct representational factors—one that captures general contextual relevance and another that captures polarity‑sensitive reasoning.

## Suggestions

- **Revise causal language:** Replace “causal mediators” with “causal influences” or “play a causal role” in the abstract and conclusion; optionally include a discussion of mediation vs. direct causal influence.  
- **Add control baselines:** Even a small‑scale control (e.g., permuted labels, sentence‑length probe) would strengthen the belief‑direction identification claim.  
- **Explain LR’s weak intervention effects:** Provide analysis (e.g., alignment of LR direction with layer representations) to reconcile the discrepancy between LR’s premise sensitivity and its minimal causal influence.  
- **Clarify normative assumptions:** Re‑frame E3/E4 as descriptive patterns rather than normative errors; discuss how LLMs might legitimately violate these patterns.  
- **Improve figure readability:** Increase font sizes for axis labels and legends in Figures 2–4.

## Score and Decision

**Score rationale:**  
- **Low anchors** (e.g., LNLr8WXDEh avg 4.5; sZq3lDDETp avg 4.2) were penalized for weak baselines, overselling, and limited experiments—issues also present here.  
- **Medium anchors** (qIN5VDdEOr avg 6.0; gsShHPxkUW avg 5.75) share systematic evaluation but avoid fatal causal overclaim or include stronger controls.  
- Our paper lies between: it is more comprehensive than the low‑scoring probes but suffers from a major overclaim (causal mediation) and lacks standard controls (permuted labels, unrelated‑property probes). The intervention is interesting but does not salvage the mediation claim. The systematic cross‑method/dataset analysis and new CCR method are solid contributions, but they do not outweigh the interpretive weaknesses.

MY FINAL SCORE: <pineapple>5.0</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>