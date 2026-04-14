## Summary
This paper investigates how the training details of the classification head ("probe") affect post-hoc attribution quality in models with frozen pre-trained backbones. The central finding is that training linear probes with Binary Cross-Entropy (BCE) instead of Cross-Entropy (CE) yields substantially more localized and class-specific attribution maps across diverse pre-training paradigms (supervised, MoCov2, DINO, BYOL, CLIP), model architectures, and explanation methods. The mechanism is traced to CE's softmax shift-invariance, which creates an equivalence class of probes with identical loss but drastically different attribution behavior. Additionally, replacing linear probes with interpretable B-cos MLP probes further improves both accuracy and localization simultaneously.

---

## Strengths

- **Novel identification of a concrete failure mechanism in CE-trained probes.** The softmax shift-invariance argument (Eqs. 2–3) cleanly demonstrates that infinitely many functionally equivalent CE probes exist, with attribution quality varying dramatically. Figure 4 makes this vivid: two manually constructed equivalent probes achieve identical CE loss (4.08) but GridPG scores of 65.7% vs. 11.9% under LRP. This is a sharp and underappreciated observation specific to the probing setting, not a generic observation about training affecting attributions.

- **Large, consistent, and reproducible empirical improvements.** The gains from switching CE→BCE are not marginal: LRP localization on DINO improves by 40 p.p. (52%→92%), on MoCov2 by 30 p.p. (50%→80%), on CLIP by 32 p.p. (19%→51%). These improvements hold across 4+ pre-training paradigms, both ResNet and ViT architectures, multiple localization metrics (GridPG, EPG), pixel deletion, and compactness/complexity measures. The scale and breadth of the effect makes it credible without requiring confidence intervals.

- **B-cos MLP probes achieve simultaneous accuracy and localization gains.** Unlike prior non-linear probe work that trades off interpretability, the paper shows B-cos MLPs consistently improve both dimensions. E.g., MoCov2+IntGrad goes from (66.9%, 69.0 GridPG) with a linear probe to (71.3%, 83.0) with a 3-layer B-cos MLP. This is a distinctive finding specific to this architecture choice, distinguishing the paper from generic "more layers help" claims.

- **Demonstration that B-cos explanations are compatible with SSL backbones.** The paper shows for the first time that B-cos attribution quality on SSL-pre-trained models closely matches that of supervised models when BCE is used (Figure 6b), broadening the applicability of B-cos networks.

---

## Weaknesses

- **The theoretical argument does not explain why SGD empirically converges to poor CE solutions.** Equations (2)–(3) establish that CE training is theoretically indifferent among an infinite equivalence class of probes. However, the paper does not show that gradient descent with typical regularization (weight decay) actually samples from the "bad" region of this class. With standard L2 weight decay, the minimum-norm CE solution is unique. The paper constructs two deliberately extreme equivalent probes to illustrate the problem but never shows that typical SGD runs produce diverse outcomes. A simple empirical test—training multiple CE probes with different random seeds and measuring variance in GridPG scores—would directly demonstrate whether the theoretical failure mode manifests in practice.

- **Localization is used as a proxy for faithfulness without validation.** The entire improvement story rests on localization metrics (GridPG, EPG, pixel deletion). Yet BCE probes are, by construction, trained to independently maximize/minimize the probability of each class (via sigmoid), which directly biases probe weights toward class-specific directions. For linear and near-linear attribution methods (LRP, B-cos, IxG, IntGrad), attributions are linear functions of probe weights. Therefore, more class-specific probe weights mechanistically produce more localized attributions—independent of whether those attributions better reflect the backbone's actual computation. If a backbone genuinely uses context or background cues for classification (which is plausible for models trained with contrastive objectives), BCE probes may suppress those background attributions, producing maps that are localized but unfaithful to the true decision process. The pixel deletion results partially address this but, as noted below, may be partially circular. No ROAR/ROAD-style faithfulness evaluation (model retrained on masked features) is provided.

- **Potential circularity in the pixel deletion evaluation.** BCE attributions are sparser and more compact (as confirmed by Gini/entropy metrics). Under pixel deletion, the most important pixels (identified by attribution magnitude) are removed first. If BCE attributions concentrate importance into a smaller region, then "unimportant" pixels—removed later—have less residual effect on the prediction, leading to a flatter deletion curve and higher retained probability. The paper does not disentangle whether this reflects genuinely more faithful importance estimates or simply the greater sparsity of BCE maps. A most-to-least deletion curve alongside the reported least-to-most would help disambiguate.

- **IxG and GradCAM fail to benefit consistently on conventional models.** The paper notes (Section 5.1) that "I×G and GradCAM only show consistent improvements for B-cos models" for conventional backbones. Since IxG and GradCAM are among the most widely used attribution methods in practice, this significantly limits the generality of the core recommendation for practitioners using conventional (non-B-cos) models. The paper's explanation cites shattered gradients, but does not test or quantify this. The abstract and conclusion do not adequately caveat this important limitation.

- **Inconsistent behavior of conventional MLP probes across datasets.** Section 5.2 reports that conventional (non-B-cos) MLPs improve EPG on COCO but show a consistent *decrease* in GridPG on ImageNet. This dataset-dependent inconsistency is only minimally discussed. The distinction between the two metrics (bounding box EPG vs. synthetic-grid GridPG) may reveal something meaningful about what conventional MLP probes learn—e.g., features that respond to object extent but not to class-distinctiveness—but this is not analyzed.

- **Theoretical grounding for B-cos MLP probes on conventional backbones is incomplete.** Section 3.3 attributes B-cos MLP improvements to "weight-input alignment" (citing Böhle et al., 2022). But that alignment property was established when the B-cos network is trained end-to-end; when a B-cos MLP head is placed on frozen conventional backbone features (not trained with the B-cos objective), it is unclear why the same alignment should emerge or why it would improve localization. The paper treats this as an empirical discovery without informal mechanistic reasoning for why alignment in the probe head propagates to input-level attribution quality on a conventional backbone.

---

## Nice-to-Haves

- **Discussion of end-to-end fine-tuning.** The paper is scoped to frozen backbone + probe, which is a well-established SSL evaluation protocol. Still, the practical relevance to practitioners who fine-tune end-to-end would be enhanced by a brief ablation or discussion of whether BCE advantage persists under partial or full fine-tuning.

- **Training cost and hyperparameter sensitivity for B-cos MLP probes.** The paper states B-cos MLPs comprise ~10% of model parameters, but does not report training time overhead or sensitivity to B-value and number of layers. Practitioners adopting this recommendation need this information.

- **Quantitative variance decomposition: probe-induced vs. pre-training-induced variance.** The claim that probe training objective matters "much more" than pre-training paradigm is visible in scatter plots (Figure 5) but is only visually argued. A formal variance decomposition or attribution-quality ANOVA across pre-training × loss function would sharpen this claim.

- **Visualization of probe weights projected back to input space.** Confirming that BCE probe weights are intrinsically more class-localized (independent of the attribution method applied) would strengthen the mechanistic interpretation and directly connect the theoretical shift-invariance argument to empirical observations.

- **Comparison with losses explicitly designed for interpretability or sparsity.** BCE removes one known failure mode (shift-invariance), but whether it is optimal among alternatives (e.g., focal loss, max-margin objectives) is left open.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Contribution (1) is not entirely new."** The harsh critic claims that the general observation that training affects attributions is already established by DINO, Böhle et al., and Tsipras et al. However, the paper explicitly distinguishes its contribution: those prior works report the effect for *specific* methods/paradigms, while this paper isolates the *mechanism* (probe loss function, shift-invariance) and demonstrates it *systematically* across many paradigms, architectures, and metrics. This is a genuine incremental contribution, not a re-observation.

- **"Abstract oversells challenge to post-hoc independence."** The statement that findings "challenge the very notion" of post-hoc independence is mild rhetorical framing and not a substantive error. Not worth flagging as a weakness.

- **"BCE's 999-vs-1 class imbalance problem on ImageNet."** While technically valid, BCE imbalance on large-scale datasets is well-known and common mitigations are standard. The paper already observes strong empirical gains with BCE; if this imbalance were seriously harmful, it would appear in the results. This concern does not undermine the paper's claims.

- **"Accuracy-interpretability trade-off under-explored."** The accuracy drop cited as concerning (73.2% → 72.8% for ViT-B/16 supervised) is 0.4 p.p. and within noise. For DINO, accuracy *increases* (77.2 → 77.6). This is not a meaningful trade-off. Removed.

- **"Human user study required."** Demanding user studies for a purely algorithmic/metrics-based contribution is not a standard expectation for this subfield at ICLR. Removed.

- **"Missing related works"** claims: Removed per instructions; external knowledge cannot confirm or deny.

- **"Unfair baseline comparisons."** Not applicable here; comparisons are fair within the paper's scope.

---

## Novel Insights

The most genuinely novel insight in this work—underemphasized in all three sub-reviews—is that the softmax shift-invariance failure is *structurally* more than an edge case: it suggests that *any* standard CE-trained linear probe on SSL features is essentially underdetermined with respect to attribution quality, not by chance but by the geometry of the loss landscape. The corollary is unexplored but interesting: probe weight diversity across random seeds should be substantially higher under CE than BCE for attribution maps even when classification accuracy is nearly identical, implying that evaluation of SSL backbones via explanation quality in prior work may have inadvertently measured probe training variance as much as backbone quality. This calls into question any prior study that compared SSL representations using attribution map quality and trained probes with CE.

---

## Suggestions

1. **Empirically validate the SGD convergence claim.** Train 5–10 CE and BCE linear probes with different random seeds on the same backbone. Report variance in GridPG scores across seeds. If CE probes show substantially higher variance than BCE probes, this directly confirms that the theoretical indifference manifests in practice and greatly strengthens the mechanistic argument.

2. **Add a faithfulness evaluation that is not reducible to localization.** For example, run a perturbation-based completeness/sufficiency test: select top-k% pixels by attribution magnitude, mask the rest, and measure classification accuracy. Do BCE probes identify pixels that are *more sufficient* for classification? This is conceptually distinct from localization and would address the most important open question raised by the work.

3. **Clarify the mechanism for B-cos MLP improvements on conventional backbones.** Even one paragraph of informal reasoning—or a small experiment showing that B-cos MLP weights are more class-specific (e.g., lower inter-class cosine similarity between weight vectors) compared to ReLU MLP weights after training on the same features—would substantially strengthen the theoretical account of Section 3.3.

4. **More prominently caveat IxG and GradCAM limitations.** Given that these are among the most widely deployed attribution methods, the abstract, conclusion, and practical recommendations should clearly state that BCE improvements are not guaranteed for these methods on conventional (non-B-cos) backbones.

5. **Decouple sparsity from faithfulness in the pixel deletion evaluation.** Report both most-to-least and least-to-most pixel deletion curves, and additionally report AUC for both. If BCE maps are more faithful, they should outperform CE on both orderings; if the advantage is due to sparsity alone, the effect may be ordinal-dependent.

---

**Novelty:** Moderate to good — the specific mechanism (softmax shift-invariance in probe training) and its empirical consequences are novel; the general finding that training affects attributions has prior grounding.

**Technical soundness:** Moderate — the theory is elegant and correct, but the gap between the theoretical equivalence class argument and empirical SGD behavior is not closed. The B-cos MLP mechanism on conventional backbones is asserted but not fully grounded.

**Empirical support:** Strong — large and consistent improvements across many paradigms, architectures, and metrics make the core finding robust. The exceptions (IxG, GradCAM, conventional MLPs on ImageNet) are acknowledged if under-analyzed.

**Significance:** Good — the finding directly affects how SSL representations should be evaluated via attributions and provides actionable, low-cost improvements. The scope is broad enough to be relevant to the XAI community at large.

**Clarity:** Good — the paper is well-organized, the key figures clearly support the narrative, and the theoretical motivation is accessible.

MY FINAL SCORE: <pineapple>6.8</pineapple>