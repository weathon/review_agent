## Summary
The paper proposes a new logit-space pseudo-label loss, sparse cross-logit (sparse-CL), and a complementary k-hardness negative loss (k-NL) for online test-time adaptation (TTA). By replacing entropy minimization in SAR with these losses, the method aims to enable large, stable learning-rate updates and reports substantial gains on ImageNet-C under standard, imbalanced, and batch-size-1 settings.

## Strengths
- The problem setting—online TTA with noisy pseudo-labels—is important and well-motivated (Sec. 1, Sec. 2).
- The proposed objectives are extremely simple: sparse-CL is just \(-\sum_i \hat y_i h_i\) and k-NL is \(\sum_i \bar y_i h_i\) (Eqs. 9, 14), making them easy to implement and drop into existing TTA pipelines (Algorithm 1).
- The gradient analysis is explicit: entropy minimization, CE, sparse-CL, and k-NL are all differentiated, and their logit-level L1 gradient norms are derived (Eqs. 3, 6, 11, 16), with Figure 3 empirically showing much more stable gradient norms for the proposed loss.
- Empirical results on ImageNet-C level-5 corruptions are consistently strong: in all three settings (Tables 1–3), “SAR* + sparse-CL” and “SAR* + L_final” significantly outperform SAR† and TENT†, with average improvements of ~3.8–6.7 points for sparse-CL alone and up to 8.1 points with k-NL.
- The qualitative logit-space analysis in Figure 4 is thoughtfully constructed and suggests that sparse-CL + k-NL creates clearer separation between true and noisy pseudo-labels and increases the “absolute true labels” count compared to entropy minimization.

## Weaknesses

### Fatal
None.

### Major
- **Theoretical justification and interpretation of sparse-CL are weak and somewhat misleading.**  
  The central “derivation” replaces \(p_i = \exp(h_i)/\sum_j \exp(h_j)\) by \(p_i \approx \exp(h_i)\) (Eqs. 7–8), then plugs this into CE to obtain \(\mathcal L_{\text{sparse-CL}} = -\sum_i \hat y_i h_i\) (Eq. 9). This effectively discards the softmax normalization and changes the loss into linear logit maximization. Calling this a “surrogate” to CE and claiming it is “inspired by cross-entropy” while treating (8) as an approximation is mathematically very loose, and there is no discussion of the implications (e.g., loss of probabilistic interpretation, calibration, potential for unbounded logit growth). The claim that constant logit-level L1 norm “indicates smaller gradient variance and stable learning” (Eq. 11 and surrounding text) is asserted without analyzing how this translates to parameter-space gradients. This undermines the strength of the theoretical story, even though the empirical results are solid.

- **Causal narrative about mitigating memorization and confirmation bias is overclaimed relative to the evidence.**  
  The abstract and intro repeatedly assert that “quick learning” via large LR and sparse-CL reduces memorization and confirmation bias and leads to a lower pseudo-label noise ratio (e.g., lines 35–41, 58, and the conclusion). However, the experiments never directly measure memorization dynamics, pseudo-label noise rates over time, or error accumulation. Figure 4 provides a descriptive logit-space analysis conditioned on the proposed method, but there is no comparison against CE/entropy at matched final accuracy, nor any time-course analysis showing that faster convergence indeed avoids memorization or confirmation bias. As a result, the causal claims are substantially stronger than what the data actually support.

- **Effect of the loss is entangled with learning-rate and protocol choices; baseline tuning fairness is unclear.**  
  The core empirical contrast is SAR† (entropy minimization) versus SAR* + sparse-CL (and +k-NL) (Sec. 4, Tables 1–3). The text states that they “replaced the entropy minimization loss in SAR with our loss function” (line 269) and Figure 2 shows that scaled-SAR with a larger LR becomes unstable, whereas the proposed method is stable at that LR. However, it remains unclear whether SAR† and other baselines were properly re-tuned for the same adaptation budget and allowed their own best stable LRs, or whether only the proposed method is given aggressive LR while baselines stay close to their original settings. There is no LR sweep or systematic search for entropy minimization with stabilizing tricks (e.g., smaller parameter subsets, gradient clipping), only a single “scaled-SAR” curve. This makes it difficult to firmly attribute the reported SOTA margins to the loss design rather than to more favorable hyperparameters.

### Minor
- **Scope mismatch between claims and evaluation breadth.**  
  The paper claims improvements “in a diverse set of TTA experiments” and “various real-world data settings” (abstract; line 58; Table 3 caption, line 371), but all experiments are on ImageNet-C level-5 corruptions with a single backbone, in three protocol variants. This is a reasonable starting point for a TTA paper, but the breadth does not quite match the ambition of the narrative; additional benchmarks or backbones would better support the generality claim.

- **Limited empirical dissection of k-NL’s design choices.**  
  Section 3.3 carefully derives k-NL and argues that skipping top-s negatives and choosing k hard negatives yields a “hardness-aware” yet stable loss (Eqs. 14–16, lines 192–196). Empirically, k-NL yields modest additional gains over sparse-CL alone (e.g., +1.4 points in Table 1; +1.3 in Table 2; +1.7 in Table 3), but there is no ablation over k and s, nor comparison to simpler logit regularizers or to tuned NL+. Thus, while k-NL clearly helps, it is not yet shown that this particular form is necessary rather than one of many viable negative-loss variants.

- **Claims linking constant logit-gradient norms to “stable large-LR adaptation” are under-supported.**  
  The paper emphasizes that sparse-CL and k-NL have zero variance in L1 gradient norm over logits (Eqs. 11, 16) and presents Figure 3 to show reduced norm fluctuations. However, there is no empirical analysis of parameter-space gradient variance, no comparison of divergence rates or instability across seeds, and no direct study of how far LR can be increased before failure for each loss. As a result, the leap from “stable logit-level L1 norms” to “stable optimization that permits significantly larger learning rates” is only partially substantiated.

- **Experimental protocol details are somewhat underspecified in the main text.**  
  Algorithm 1 mentions trainable parameters \(\phi \subseteq \theta\), but the main body does not state which layers are actually adapted in the experiments (BN only, classifier, or more). SAR and TENT typically restrict adaptation to BN affine parameters; if this method adapts a larger subset, that could interact with the new loss and LR. Similarly, the distinction between SAR† and SAR* is only indirectly conveyed through the tables and captions; explicit description in the main text would help disentangle implementation differences from loss effects.

- **No variability reporting.**  
  All results are single-run averages over corruptions (Tables 1–3) without error bars or multiple seeds. For the main SAR† vs. SAR* + sparse-CL differences (3–8 points), this is likely fine, but for the incremental effect of k-NL (∼1–2 points), it is difficult to assess robustness.

### Trivial
- Some conceptual statements (e.g., “This indicates that this loss will yield a smaller gradient variance during updating and a stable gradient norm in the backward steps. As a result, we can adapt the model with this loss using a high learning rate,” lines 128–130) could be phrased more cautiously to distinguish what is proven analytically versus what is observed empirically.
- The mention of viewing the classifier as prototypes (line 68) is not used later and could be safely removed or more tightly integrated to streamline exposition.

## Nice-to-Haves
- Directly measure pseudo-label accuracy and noise rate over the adaptation trajectory for entropy minimization, CE, sparse-CL, and sparse-CL + k-NL at comparable LRs, to test the memorization/confirmation-bias hypotheses.
- Add controlled comparisons at matched adaptation protocols: same parameter subset \(\phi\), LR grid-search per loss, and perhaps a “best stable LR” experiment for each loss to isolate the contribution of the loss vs. LR choice.
- Provide ablations over k and s, and compare k-NL against simpler alternatives (e.g., margin-based logit penalties, top-k negative CE variants, tuned NL+), to justify the specific k-NL design.
- Explore at least one additional benchmark or backbone (e.g., CIFAR-C or a ViT-based model) to support claims of broad applicability.

## Removed Points
These points are flagged to be removed, treat them with caution.

- Any criticism claiming the paper fails to include certain prior work or related methods beyond what is visible in the parsed text has been omitted, since we cannot verify the completeness of references or appendices.
- Formatting/notation/typo complaints are excluded by instruction, as the parsed text may not faithfully reflect the original PDF.
- Hypothetical concerns about non-release or non-existence of baselines or datasets are removed; the paper clearly specifies standard baselines and ImageNet-C, which we must assume exist and are available.

## Novel Insights
None beyond the paper’s own contributions; the main insights revolve around the proposed constant-logit-gradient losses and their empirical behavior, which are already articulated in the paper.

## Suggestions
- Reframe the theoretical section around what is actually being optimized: present sparse-CL as a deliberate linear-logit objective, not as a cross-entropy “approximation” via \(p_i \approx \exp(h_i)\). Clarify the pros/cons (e.g., no normalization, overconfidence) and why they may be acceptable in TTA.
- Soften and better align the memorization/confirmation-bias narrative with the evidence. Either add targeted experiments (noise dynamics, error accumulation) or present those aspects as hypotheses and intuition rather than proven effects.
- Make the experimental protocol fully explicit: which parameters are adapted (BN-only vs. more), exact LR and optimizer settings for all baselines, and what distinguishes SAR† from SAR*.
- Add, at minimum, LR-sweep plots and “failure LR thresholds” per loss (EM, CE, sparse-CL) to convincingly show that the proposed objectives truly expand the stable LR range, not just for a single chosen LR.
- Include multi-seed results (mean ± std) for key settings, especially when quantifying the incremental benefit of k-NL over sparse-CL.

### Axes Evaluation
- **Originality:** Moderately high; the specific logit-linear loss and constant-norm k-NL in TTA are simple but not standard, and the gradient-norm–stability angle is interesting.
- **Importance of question:** High; robust, efficient online TTA with noisy pseudo-labels on large-scale datasets is practically relevant.
- **Support for claims:** Mixed; empirical SOTA gains are strong, but theoretical framing and the causal story about memorization/confirmation bias are overstated.
- **Soundness of experiments:** Reasonably strong for ImageNet-C SOTA comparison, but lacking in diagnostic ablations and LR-fairness analyses.
- **Clarity of writing:** Generally clear and well-organized, with concrete equations and algorithms, but some theoretical claims are not carefully bounded.
- **Value to community:** Good, primarily as a simple, strong-performing loss for ImageNet-C TTA; with tightened theory and more diagnostic experiments, it could be quite impactful.

## Score and Decision

### Calibration anchors consulted
- High-score anchors (>7):
  - `/home/wg25r/review_agent/human_reviews/9w3iw8wDuE.md` (avg 7.0): TTA loss/selection (entropy vs PLPD), strong empirical and well-calibrated theoretical story; more careful causal claims than this paper.
  - `/home/wg25r/review_agent/human_reviews/TPZRq4FALB.md` (avg 8.0): multi-modal TTA with robust optimization; very strong, broad evaluation and solid theory.
  - `/home/wg25r/review_agent/human_reviews/BmG88rONaU.md` (avg 7.5): TTA for cross-modal retrieval with both strong experiments and tight claims.
  - `/home/wg25r/review_agent/human_reviews/d8w0pmvXbZ.md` (avg 8.0): high-LR / instability analysis with rigorous experiments and theory.
  - `/home/wg25r/review_agent/human_reviews/Kl9CqKf7h6.md` (avg 7.25): LR scheduling with robust empirical/theoretical alignment.  
  Compared to these, the current paper has comparably strong empirical margins but noticeably weaker and looser theory and causal interpretation, suggesting it should score below 7–8.

- Medium-score anchors (4–6):
  - `/home/wg25r/review_agent/human_reviews/N0ETIi580T.md` (avg 5.25, Accept): TTA adversarial vulnerability; solid but with notable gaps, accepted as poster.
  - `/home/wg25r/review_agent/human_reviews/xqxG5WogN6.md` (avg 5.67, Reject): TTA with good ideas but overclaims and missing analyses.
  - `/home/wg25r/review_agent/human_reviews/7iuFxx9Ccx.md` (avg 6.0, Reject): resource-efficient test-time training; strong empirical story but missing key ablations.
  - `/home/wg25r/review_agent/human_reviews/ws0F5NTzGw.md` (avg 4.5, Reject): TTA for tabular data; promising but underdeveloped empirically.
  - `/home/wg25r/review_agent/human_reviews/eXrUdcxfCw.md` (avg 4.8, Reject): continual TTA; decent experiments but methodological concerns.  
  The present paper looks stronger than the weaker medium anchors (4.5 range) due to its consistent and sizeable gains on a standard benchmark, and comparable or slightly stronger than the 5–6 range papers that had solid but imperfect experimental setups. Its main issues are overclaiming and lack of diagnostic analysis, similar to several 5–6 score anchors.

- Low-score anchors (<3):
  - `/home/wg25r/review_agent/human_reviews/pdzHpQbGrn.md` (avg 2.5): TTA prompt learning; significant methodological flaws and weak evidence.
  - `/home/wg25r/review_agent/human_reviews/L1BXvqwsMv.md` (avg 2.5): online adaptation of SAM; limited evaluation and conceptual issues.
  - `/home/wg25r/review_agent/human_reviews/ctzGqxE3O0.md` (avg 2.5): malware adaptation; poor methodology.
  - `/home/wg25r/review_agent/human_reviews/fsmEuS5ZNg.md` (avg 3.0): video domain adaptation; underdeveloped experiments.  
  The current paper is clearly above this band: the methodology is standard, the experiments are extensive on ImageNet-C, and the main flaws are about theory and attribution rather than correctness.

Positioning relative to anchors: the paper has very strong empirical results like some 7–8 papers but weaker, somewhat sloppy theoretical framing and causal claims, as in several mid-range works. It is stronger than many rejected TTA papers in the 4–5 range because of its consistent, sizable improvements with simple methods on a standard benchmark, but below the 7–8 range due to overclaiming and missing diagnostic evidence.

**Final score:** 6.0  
This reflects a paper with solid empirical contributions and clear practical value, but with overreaching theory and causal narrative that keep it below the top tier.

**Decision:** Borderline but leaning to accept: I would accept it as a solid empirical TTA contribution, provided the authors clarify and temper their theoretical and causal claims.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>