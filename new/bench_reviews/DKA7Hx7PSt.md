## Summary

The paper proposes LELP (Learning Embedding Linear Projections), a knowledge distillation method tailored to binary and few‑class classification. LELP computes per‑class PCA directions of teacher embeddings (after an optional null‑space projection and random rotation), defines pseudo‑subclasses via projections in these directions, and trains the student on an expanded SC‑way soft label distribution. Experiments on binarized CIFAR‑10/100 and several NLP benchmarks suggest that LELP is competitive with, and often slightly better than, strong KD baselines, especially Subclass Distillation, without requiring teacher retraining.

## Strengths

- **Clear, simple idea grounded in representation structure.**  
  The method is mechanically straightforward: compute per‑class PCA directions of teacher embeddings, optionally project out teacher weight directions and random‑rotate, then factor class probabilities into S subclasses per class using softmax over the projected coordinates (Sec. 3.1–3.3). This is easy to implement in existing KD pipelines and aligns with empirical observations from Neural Collapse that final‑layer embeddings retain fine‑grained structure beyond logits.

- **Addresses a real gap for few‑class / binary KD.**  
  The introduction correctly notes that logit‑based KD conveys less information when there are few classes, and that many sophisticated KD methods are tuned for vision multi‑class scenarios. The binary CIFAR‑10/100 and few‑class NLP settings (Amazon Reviews, Sentiment140, GLUE) are appropriate and important application domains for such a method.

- **Insightful use of “Oracle Clustering” as an upper bound.**  
  The binarized CIFAR‑10/100 experiments in Sec. 4.2 use the original (10/100‑way) labels to define an oracle subclass partition. Training students on these oracle subclasses then evaluating on the binary task both provides a meaningful upper bound and shows that richer subclass supervision can make a student outperform its teacher (Table 1 and associated discussion). This is an elegant experimental design that clarifies what “good subclass structure” can buy in distillation.

- **Empirical evidence that naïve clustering is unreliable but LELP is robust.**  
  Table 1 and Fig. 3 compare Agglomerative, K‑means, and t‑SNE+K‑means subclassing against LELP. Naïve clustering methods do not reliably outperform Vanilla KD and can even underperform it, whereas LELP consistently does better than Vanilla KD and other clustering choices across multiple teacher–student pairs on binarized CIFAR‑10/100. This supports the claim that the *way* pseudo‑subclasses are constructed matters significantly.

- **Broad set of baselines and teacher–student pairs.**  
  The paper compares against a wide range of KD methods: Vanilla KD, Embedding/Feature distillation, FitNet/VID‑style methods, Relational KD, CRD, DKD, and Subclass Distillation, and it evaluates several teacher–student architectures: ResNet→ResNet, ResNet→MobileNet, MobileNet→MobileNet, ALBERT‑Large/XXL→ALBERT‑Base, and ALBERT‑XXL→a 2‑layer MLP over frozen Sentence‑T5 features (Sec. 4.1). This shows that LELP is not tied to a single architecture family and can cope with mismatched embedding dimensions.

- **Honest limitations section.**  
  Sec. 5 openly notes that subclasses need not be linearly separable, that LELP is most useful when logit information is limited (few classes), and that for large‑class datasets (e.g., ImageNet‑1K) it is not expected to help. This appropriately bounds the claimed scope.

## Weaknesses

### Fatal

None. The paper presents a coherent method, and the experiments, while imperfect, do provide genuine positive evidence. There are, however, important issues that significantly weaken the strength of the claims.

### Major

- **Non‑standard evaluation regime: α fixed to 0 for all methods.**  
  In Sec. 4.1, the authors state: “we always set α = 0 in equation 1,” so the student loss is purely distillation without any cross‑entropy to ground‑truth labels, even in fully supervised settings. This is explicitly done “to reduce the variance between methods” and to match semi‑supervised scenarios. However, in most practical supervised KD, practitioners use α>0 and exploit ground‑truth labels together with KD. Some baselines (e.g., Vanilla KD, embedding/feature‑based methods) are known to benefit substantially from mixing CE and KD. Under the presented experiments:
  - No sensitivity analysis over α is provided for any method.
  - We do not know whether the reported ranking would hold under a standard CE+KD setup with α tuned per method.
  Given that many of LELP’s gains over the strongest baselines in Table 2 are small (often ≤1 point over Subclass Distillation and much smaller over non‑subclass baselines), it is very plausible that in a more realistic α>0 regime some baselines would close or reverse the gap. As a result, the strong claim that LELP is “typically superior to existing SOTA distillation algorithms for binary and few‑class problems” is not fully supported; what is actually demonstrated is superiority in a particular α=0 regime.

- **Claims of consistent superiority over Subclass Distillation are stronger than the evidence.**  
  Subclass Distillation is the main strong baseline in both binary CIFAR and few‑class NLP experiments. The paper highlights that LELP “achieves performance that is always on par with, and typically exceeding, Subclass Distillation” (Sec. 2) and repeatedly emphasizes superiority in the abstract and conclusion. Looking at the presented numbers:
  - On binary CIFAR (Table 1 and text referencing Table 3 in Appendix B), LELP outperforms Subclass Distillation in several configurations, but gains are modest and sometimes comparable to other baselines.
  - In Table 2 (few‑class NLP without subclass structure), LELP clearly beats Subclass Distillation on some datasets (e.g., Amazon Reviews 5‑class: 78.06 vs 76.28; Sentiment‑60 Bin: 87.60 vs 85.93) and is essentially tied or slightly worse in others (e.g., QGLUE/sst2: 92.81 vs 92.85). The “Avg. gain over the best baseline” row is uniformly small (+0.02 to +0.05) and somewhat opaque, but at least from the per‑dataset entries it is clear that LELP is *competitive* and sometimes better, rather than clearly and universally superior.
  Moreover, the authors themselves acknowledge that the Subclass Distillation teacher differs from the main teacher (Sec. 4.1: “the accuracy of the teacher model in Subclass Distillation usually differs from the one used for LELP (and the other baselines)... comparing them directly might not be entirely fair”). This makes the direct “beats Subclass Distillation” narrative less clean. Overall, LELP is convincingly shown to be on par with Subclass Distillation while being much cheaper (no teacher retraining), which is already a strong result; the stronger “typically superior” claim overstates the evidence.

- **Hyperparameter sensitivity of core design choices is under‑exposed in the main text.**  
  LELP’s distinctiveness hinges on several specific decisions:
  - Projecting teacher embeddings to the null‑space of the teacher output weights before PCA (Sec. 3.1).
  - Applying a random orthonormal rotation to the top PCA directions to equalize variance (Sec. 3.1).
  - Choosing the number of projections/subclasses per class S and subclass temperature β (Sec. 3.2).
  These are described as “often helps” and claimed to be validated in Appendix C, but the main paper does not provide quantitative ablations or robustness analyses. Given that:
  - A very simple alternative (plain PCA on embeddings without null‑space projection or rotation) is conceptually close.
  - Another simple alternative (random projections) is mentioned but not systematically compared in the main body.
  It is not clear how much each design choice actually contributes, nor how sensitive performance is to S and β across tasks. Since the algorithmic novelty is relatively incremental (linear operations on embeddings plus a modified loss), the empirical justification for *why this particular combination* is needed is important to establish the contribution as more than a tuned variant of “cluster and split.”

- **Evaluation of modality‑independence and cross‑architecture generality is narrower than the claims.**  
  Sec. 3 lists “Modality‑independent” and “Compatibility between differing student/teacher architectures” as key desiderata, and the paper describes LELP as “modality‑independent” and “uniquely versatile.” In reality:
  - Only two modalities are tested: vision images (CIFAR) and English text (various reviews and GLUE). These are standard, high‑resource settings with well‑studied encoders.
  - Architecture mismatch is probed via ResNet→MobileNet and MobileNetwd2→MobileNet in vision, and ALBERT→ALBERT and ALBERT‑XXL→MLP over frozen Sentence‑T5 in NLP. These are useful, but still relatively limited.
  The presented results do show that LELP transfers across these cases, which is a genuine positive. However, the breadth of experimentation does not justify very strong claims of modality independence or unique versatility; the evidence supports “works in both vision and NLP, and across several teacher–student families,” not universal applicability.

### Minor

- **Statistical support for reported gains is weak.**  
  The tables report means ± standard deviations over three runs. Differences between LELP and the best baselines on several NLP datasets (Table 2) are small relative to these standard deviations, and there is no hypothesis testing or deeper variance analysis. For example, in QGLUE/cola and QGLUE/sst2, the means differ by at most a few tenths of a point with overlapping error bars. A more cautious phrasing (e.g., “competitive and sometimes better”) would better reflect this uncertainty.

- **Choice and explanation of subclass factorization.**  
  The subclass probabilities are defined as  
  \( p_{c,s}^{\text{Teacher}} = p_c^{\text{Teacher}} \cdot \frac{e^{z_{c,s}/\beta}}{\sum_{j} e^{z_{c,j}/\beta}} \),  
  so the per‑class logits and per‑subclass projected logits interact multiplicatively. This ensures that subclass probabilities sum to the original class probability, which is reasonable, but the text offers little intuition or comparison with alternatives (e.g., training the student on an SC‑way softmax directly from concatenated logits). Some brief discussion of why this particular factorization works empirically and how it relates to prior subclass methods would improve clarity.

- **Clarity on the teacher‑side computation cost.**  
  The paper notes that the PCA computation is dominated by the cost of forward‑passing the dataset through the teacher (Sec. 3.1) and thus is effectively O(N). This is true but somewhat understates that a full pass through a very large teacher can still be costly in absolute terms, albeit a one‑time cost. A short discussion of this cost versus the cost of retraining teachers (as in Subclass Distillation) would make the trade‑off clearer.

### Trivial

- Minor typographical/notation issues (e.g., occasional mismatches in names like “Feature”, “Retained KD”, “CKD” vs. the terms used in Sec. 2) likely stem from the extraction process and are not substantive.  
- Some table captions and descriptions could be made more explicit about which baselines are included in the “best baseline” versus “best non‑subclass baseline” summaries.

## Nice-to-Haves

- Experiments with α>0 (combining CE with KD) on a subset of the NLP tasks, to show that LELP’s advantages persist in the standard supervised regime.
- A main‑paper ablation plot showing performance vs. S (number of subclasses per class) and vs. β for at least one vision and one NLP dataset, to give readers a sense of robustness and practical tuning guidance.
- A small experiment with moderate‑class datasets (e.g., 10–20 classes) to empirically illustrate the transition from “LELP helps” to “LELP comparable to Vanilla KD,” complementing the limitation discussion in Sec. 5.
- Additional embedding visualizations for an NLP dataset (analogous to Fig. 4 for CIFAR‑10bin) to directly demonstrate that LELP induces more structured student embeddings there as well.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **“Marginal gains make the method ineffective overall.”**  
  While some gains over baselines are small, the paper also reports clear improvements in harder or more practically relevant settings (e.g., Amazon Reviews, Sentiment140, binary CIFAR‑100 with strong teacher–student mismatch), and the method has appealing simplicity and cost advantages. It would be incorrect to conclude that the method is overall ineffective solely due to marginal improvements in some configurations.

- **“LELP does not scale at all to larger models or LLM‑like teachers.”**  
  The paper does not include experiments with modern decoder‑only LLMs, but it does demonstrate LELP with large ALBERT models and a Sentence‑T5‑11B encoder as feature extractor. There is no evidence in the text that LELP fundamentally fails to scale; the limitation is about evaluation breadth rather than a demonstrated scaling failure.

- **“The improvements are purely due to increased classifier output size (capacity), not to the projections.”**  
  This is speculative. The paper explicitly compares against other subclass‑forming or clustering‑based methods that also increase the number of targets (e.g., K‑means, t‑SNE+K‑means, Oracle Clustering) and shows LELP outperforming them (Table 1), indicating that its particular way of splitting classes matters beyond just output size.

## Novel Insights

The most genuinely novel insight here is the systematic demonstration that, in low‑label‑information regimes (binary or few‑class classification), carefully constructed pseudo‑subclasses derived from the *geometry* of teacher embeddings can substantially improve distillation—sometimes even surpassing the teacher—while naïve clustering of embeddings often fails to help or is unstable. By using Oracle Clustering on binarized CIFAR as an upper bound, and then showing that a simple, linear‑algebraic construction (null‑space PCA plus subclass factorization) gets close to this upper bound without retraining the teacher, the paper gives a concrete, empirically grounded perspective on how and when final‑layer representation structure can be leveraged effectively in KD.

## Suggestions

- Temper the strongest empirical claims. Rephrase statements such as “typically superior” or “achieving an improvement of 1.85% and 0.88% over the best baseline” to emphasize that LELP is *competitive with and often slightly better than* strong baselines—especially Subclass Distillation—under the presented α=0 setting, and that some improvements are within or close to the reported standard deviations.

- Add at least one main‑paper ablation figure on the effect of S and β and, if space allows, include a small comparison between full LELP vs. (i) plain PCA without null‑space projection or rotation and (ii) random projections, for a representative dataset. This would solidify the empirical justification for the specific design choices.

- For a subset of datasets (ideally one vision and one NLP), run experiments with α>0 (e.g., α∈{0.25, 0.5, 0.75}) for LELP and a key baseline such as Vanilla KD or Subclass Distillation. Even if done at a coarse level, this would materially strengthen the case that LELP is beneficial in more standard supervised KD regimes.

- Clarify in the exposition that the “Subclass Distillation Teacher” row represents a differently trained teacher and explicitly discuss how that affects fairness of comparison, perhaps moving the most direct LELP‑vs‑Subclass‑Distillation claims to a more nuanced, cost–benefit framing.

- If possible, include at least one moderate‑class (e.g., 10–20 classes) experiment to illustrate where LELP stops providing advantages, making the scope outlined in Sec. 5 more concrete and useful to practitioners.

Regarding the standard evaluation axes:

- **Originality:** Moderate. The idea of pseudo‑subclasses builds on prior work, but using teacher‑embedding linear projections (with null‑space PCA and rotation) to form subclasses without teacher retraining is a neat, non‑obvious twist.
- **Importance of question:** Solid. Improving KD in binary/few‑class regimes, especially in NLP, is practically important.
- **Support for claims:** Mixed. The method is clearly beneficial in some settings, but the strongest superiority claims are not fully justified, especially given the α=0 constraint and small margins on some benchmarks.
- **Soundness of experiments:** Reasonably sound but incomplete. The experimental setup is coherent and covers multiple domains and baselines, yet key hyperparameters and the role of supervised loss are under‑explored.
- **Clarity:** Good. The method is described clearly, with helpful figures and a well‑structured narrative.
- **Value to the community:** Moderate to good. LELP is easy to implement and likely to be useful for practitioners working on few‑class or binary KD, but the current paper needs stronger, more balanced empirical validation to be impactful at a top venue.

## Score and Decision

### Calibration

I compared this paper against several KD‑related submissions:

- **“Improving Language Model Distillation through Hidden State Matching” (IcVSKhVpKu, scores 6, 8, 3, Accept Poster):** This paper proposed CKA‑based hidden state matching, had clear novelty and reasonably strong empirical validation across tasks, but also some mixed reviews. It demonstrated robust gains across several LM tasks in standard supervised regimes. The present LELP paper is somewhat less thoroughly validated (α=0 throughout, weaker ablations) and the novelty is comparable or slightly lower. I assess LELP as somewhat weaker overall than this paper.

- **“Efficient Unsupervised Knowledge Distillation with Space Similarity” (QHVTxso1Is, scores 5, 3, 6, 6, 5, Reject):** This paper had a simple method and extensive experiments but was ultimately rejected due to issues like limited robustness analysis and missing comparisons. LELP is on par or slightly stronger in terms of conceptual grounding (Neural Collapse, pseudo‑subclasses) but exhibits similar experimental limitations (sensitivity to design choices under‑analyzed, some strong claims over‑stated).

- **“Medium-Difficulty Samples Constitute Smoothed Decision Boundary for KD on Pruned Datasets” (Rz4UkJziFe, scores 8, 3, 6, 6, Accept Poster):** This paper combined a solid conceptual contribution with targeted experiments and had a clearer story on when and why it helps. LELP’s contribution feels somewhat narrower and its empirical claims are less rigorously substantiated.

- **“PROGRESSIVE KNOWLEDGE DISTILLATION (PKD)” (GHaoCSlhcK, scores 3, 3, 5, 5, 3, Reject):** PKD proposed an architecture‑agnostic KD framework but was criticized for unclear necessity of some components and insufficient ablations. The present paper is stronger than PKD in clarity and in directly demonstrating benefits in its target regime, but still shares the issue of under‑justified design choices.

Considering these anchors, I place this paper around the borderline between weak accept and reject. The idea is interesting and potentially useful, but the current empirical evaluation and claim calibration are not strong enough for acceptance at a top venue.

**Final score:** 5.5 (between “borderline accept” and “borderline reject”; leaning reject due to over‑strong claims and incomplete evaluation).

Given typical selectivity and the issues noted, my **decision recommendation** is Reject in its current form. With more balanced claims, α>0 experiments, and stronger ablations, a future version could be competitive.

MY FINAL SCORE: <pineapple>5.5</pineapple>  
MY FINAL DECISION: <orange>Reject</orange>