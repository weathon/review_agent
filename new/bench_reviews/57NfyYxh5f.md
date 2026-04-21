## Summary

This paper identifies that the training objective of the classification layer (probe) on frozen pre-trained backbones significantly impacts the quality of post-hoc explanations. The key finding is that using Binary Cross-Entropy (BCE) instead of Cross-Entropy (CE) loss for linear probe training consistently and substantially improves attribution localization (18–40 percentage points on GridPG across settings). The paper attributes this improvement to the shift-invariance of the CE loss (Equations 2–3), demonstrating that two functionally equivalent CE probes can yield vastly different attributions (Figure 4). Additionally, the paper shows that B-cos MLP probes further improve both accuracy and localization simultaneously, and demonstrates B-cos compatibility with self-supervised learning for the first time.

## Strengths

- **Striking motivating observation**: Figure 4 shows two functionally equivalent CE-trained probes (identical loss, identical predictions) producing GridPG scores of 65.7% vs. 11.9%. This is a powerful demonstration that CE shift-invariance creates fundamental ambiguity for attribution methods, regardless of whether it fully explains the BCE improvements.

- **Exceptional empirical breadth**: The paper evaluates across 5 pre-training paradigms (supervised, MoCov2, DINO, BYOL, CLIP), 2 architectures (ResNet-50, ViT-B/16), 10+ attribution methods spanning gradient-based (IntGrad, I×G), backpropagation-based (LRP), activation-based (GradCAM, ScoreCAM), perturbation-based (LIME), inherently interpretable (B-cos), and ViT-specific (Rollout, CGW1, CausalX), with 3 datasets and 5 evaluation metrics. The consistency of the BCE improvement across this sweep makes the core finding hard to dismiss.

- **Large and consistent magnitude of improvement**: The gains are substantial, not marginal. For example, Figure 5 shows LRP GridPG improvements for conventional models: CLIP +32 p.p. (19%→51%), MoCov2 +30 p.p. (50%→80%), DINO +40 p.p. (52%→92%). These hold across attribution methods and architectures (Table 1 for ViT).

- **High practical utility with near-zero cost**: Switching from CE to BCE for probe training requires changing one line of code and has negligible impact on accuracy (Table 1 shows mixed but small accuracy differences). This is a rare paper where the proposed intervention is both highly impactful and trivially adoptable.

- **First demonstration of B-cos compatibility with SSL**: The paper shows that inherently interpretable B-cos models are compatible with self-supervised learning approaches (contribution 5, Section 5.2), preserving both performance and interpretability — a result of independent interest.

## Weaknesses

### Fatal
None.

### Major

- **The shift-invariance mechanism for BCE improvements is claimed but not experimentally validated.** Section 3.2 presents shift-invariance as the explanation for why CE-trained probes yield poor attributions, and the paper uses this to motivate BCE as the solution (lines 132, 146, 152). While Figure 4 convincingly shows that shift-invariance *can* produce arbitrarily different attributions, it does not establish that shift-invariance is the *actual cause* of the observed BCE improvements in practice. BCE differs from CE in at least two ways: (a) it removes shift-invariance, and (b) it optimizes each class independently (per-class sigmoid) rather than competitively (softmax), which independently encourages class-specific weights. The paper acknowledges the latter on line 152 ("the linear probe is penalized for adding a constant positive shift to non-target classes and thus biased towards focusing on class-specific features") but does not disentangle these mechanisms. A straightforward control experiment — taking CE-trained probes, subtracting the mean weight vector across classes, and re-evaluating attributions — would test whether removing the shift component alone produces comparable improvements. Without such validation, the theoretical contribution remains an untested hypothesis rather than an established explanation, despite being presented as the central insight (Figure 4 caption: "Due to the shift-invariance of softmax, one cannot expect positive and negative attributions to be well calibrated").

- **The abstract overclaims that probe training matters "much more" than pre-training.** The data does not cleanly support this. In Figure 5 (conventional models, LRP), CE probes' GridPG scores range from 19% (CLIP) to 52% (DINO) — a 33 p.p. spread across pre-training methods. Under BCE, the spread is 41 p.p. (51%–92%). The across-method variation is *comparable to or larger than* the within-method BCE improvement for some settings. The more nuanced statement in Section 5.1 (line 283: "the choice of pre-training method has a limited impact only on explanation quality, with no particular method consistently outperforming others") is defensible — it's about consistency, not magnitude. But the abstract's "much more than the pre-training scheme itself" (line 27) conflates these and is not supported by the data. Both factors contribute substantially and interact.

### Minor

- **"Better explanations" is operationalized primarily as localization, not faithfulness.** The evaluation metrics (GridPG, EPG, pixel deletion, compactness, complexity) all measure concentration of attribution in the correct region. A method that correctly localizes to the dog region but highlights background texture within that region rather than the dog's diagnostic features would score well. Pixel deletion is the closest proxy to faithfulness, but it measures whether removing top-attributed pixels hurts prediction — related to but distinct from attribution correctness. The paper does not engage with this distinction, which limits how strongly one can interpret the results as showing "better explanations" rather than "more localized attributions."

- **BCE improvements are not universal across attribution methods.** The paper notes (line 257) that I×G and GradCAM only show consistent improvements for B-cos models, not conventional ones. GradCAM is one of the most widely used attribution methods; this exception is discussed only briefly in the main text with a reference to Appendix E. The paper should more clearly delineate when and why BCE helps vs. doesn't help.

- **B-cos MLP improvements may be partially circular.** B-cos layers are architecturally designed to increase weight-input alignment, which directly increases localization. The paper shows B-cos MLPs improve localization (Section 5.2) but does not fully disentangle whether this stems from better feature extraction or from B-cos's built-in alignment property. The finding that conventional MLPs *decrease* GridPG on ImageNet (line 305) is important and suggests the improvement is not just about capacity, but more analysis of the B-cos-specific inductive bias would strengthen the interpretation.

### Trivial
None.

## Nice-to-Haves

- **Mean-subtraction control experiment**: Take CE-trained probes, subtract the mean weight vector across classes, and re-evaluate attributions. This would directly test whether shift-invariance is the mechanism and would be a very cheap experiment.

- **Disentangling shift-invariance from per-class optimization**: An ablation that keeps per-class optimization but adds back shift-invariance (or vice versa) would clarify the mechanism.

- **Analysis of when BCE hurts accuracy**: The paper reports some accuracy drops (Table 1: ViT-B/16 MoCov3 drops from 76.3% to 75.1%). A more systematic analysis of the accuracy–interpretability tradeoff would strengthen practical recommendations.

- **Attribution quality beyond localization**: Showing cases where BCE and CE attributions are both localized but highlight different features within the correct region would reveal whether BCE changes *what* the model attends to, not just *where*.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Post-hoc methods are independent of how models are trained" mischaracterizes the field** (from Harsh Critic): The critic argues no serious work claims post-hoc methods are independent of the trained model; they are independent of the *training process*. However, the paper's statement is clearly about independence from training details/process, not from the model itself — the entire paper is about how attributions depend on the model. This is a misreading of the paper's claim, not a valid criticism.

- **Reproducibility concerns about models/datasets cited** (implicit in general reviewer guidelines): All cited models (ResNet-50, ViT-B/16, CLIP, DINO, MoCov2, BYOL) and datasets (ImageNet, VOC, COCO) are well-known and publicly available. Not a concern.

- **Missing related works**: Per rules, I do not have external sources to confirm existence of specific missing references.

- **Formatting and presentation nitpicks**: Removed per rules about parser artifacts.

- **Undisclosed hyperparameters / implementation details**: Per rules, trivial implementation details are not valid criticisms.

- **End-to-end BCE training as a weakness**: The paper explicitly scopes its contribution to probe training on frozen features. Requesting end-to-end training is scope creep — it would broaden the impact but is not a flaw in the current work.

## Novel Insights

The paper reveals an underappreciated structural issue: the classification layer, often treated as a trivial appendage in representation learning pipelines, can dominate the quality of post-hoc explanations. This is particularly significant given that linear probing is the standard evaluation protocol for self-supervised learning. The implication is that the interpretability community may have been drawing conclusions about backbone representations from explanations that are largely artifacts of the probe's training objective. This reframes the question from "which attribution method is best?" to "under what probe conditions can any attribution method be trusted?"

## Suggestions

- Run the mean-subtraction control experiment and report results — this would either validate the shift-invariance mechanism (strengthening the theoretical contribution) or reveal that per-class optimization is the key factor (which would shift the framing but still be valuable). Either outcome strengthens the paper.

- Soften the abstract claim from "much more than the pre-training scheme itself" to something like "as much as or more than the pre-training scheme," which is better supported by the data.

- Add a brief discussion distinguishing localization from faithfulness, and explicitly acknowledge that the current evaluation primarily measures the former.

## Evaluation Axes

- **Originality**: The shift-invariance analysis of CE loss in the context of attribution quality is novel and important. The empirical finding about probe training impact is surprising and underexplored.

- **Importance of research question**: High — post-hoc attribution methods are widely used, and the finding that their results depend critically on probe training has broad practical implications.

- **Claim support**: The empirical claims are well-supported by extensive experiments. The theoretical mechanism claim (shift-invariance as the cause) is not experimentally validated. The "much more than pre-training" claim is overstated.

- **Soundness of experiments**: Very strong breadth; the experimental design (frozen backbone, only probe varies) is clean and well-controlled. The main gap is the missing mechanism-validation experiment.

- **Clarity of writing**: Generally clear and well-organized. The shift from "hypothesize" (line 132) to assertive presentation in Figure 4 caption could be more consistent.

- **Value to research community**: High — the BCE recommendation is immediately actionable, and the finding raises important questions about the reliability of current evaluation practices.

## Score and Decision

**Calibration anchors compared against:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| UNI (attribution baseline) | `/home/wg25r/review_agent/human_reviews/PBjCTeDL6o.md` | 8.0 | Much stronger theoretical grounding + empirical validation; our paper lacks mechanism validation |
| Rethinking Self-Distillation | `/home/wg25r/review_agent/human_reviews/EJfLvrzh2Q.md` | 7.0 | Theoretical analysis of linear probing with validated mechanism; our paper has broader empirical scope but weaker theory |
| Backdoor XAI Evaluation | `/home/wg25r/review_agent/human_reviews/cObFETcoeW.md` | 6.75 | Novel evaluation framework with strong motivation; comparable practical impact |
| AttEXplore | `/home/wg25r/review_agent/human_reviews/FsVxd9CIlb.md` | 5.75 | Moderate novelty; our paper has stronger empirical evidence and broader scope |
| Butterfly Effects (EMA) | `/home/wg25r/review_agent/human_reviews/CgPs04l9TO.md` | 5.33 | Similar pattern: simple intervention, unvalidated mechanism, accepted poster; our paper has broader evaluation and more striking motivating example |
| Probe Training Variability | `/home/wg25r/review_agent/human_reviews/4Qz9BT4mpM.md` | 5.75 | Related topic (probe training effects); rejected due to limited novelty; our paper has stronger contribution |
| AdamE | `/home/wg25r/review_agent/human_reviews/5nldnvvHfw.md` | 2.50 | Theoretical errors in proofs; our paper has no such fundamental flaws |

The paper sits above Butterfly Effects (5.33) and AttEXplore (5.75) due to its exceptional empirical breadth and more striking motivating observation, but below the self-distillation linear probing paper (7.0) and UNI (8.0) due to the unvalidated mechanism. The overclaimed abstract and unvalidated shift-invariance mechanism are the main factors preventing a higher score, but the empirical contribution is strong enough to place this solidly in the acceptable range.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>