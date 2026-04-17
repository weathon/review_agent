# Regulating Internal Alignment Flows for Robust Learning Under Spurious Correlations

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 8

## Abstract
Deep models often exploit spurious correlations (e.g., backgrounds or dataset artifacts), hurting worst-group performance. We propose \textbf{Alignment-Gated Suppression (AGS)}, a lightweight, plug-in regularizer that intervenes inside the network during training. AGS tracks a class-conditional, confidence-weighted contribution for each neuron (more negative $\Leftrightarrow$ stronger support) and applies a percentile-based, multiplicative decay to the most extreme contributors, reducing overconfident shortcut pathways while leaving other features relatively more influential. AGS integrates with standard ERM, requires no group labels, and adds $<5\%$ training overhead. We provide analysis linking AGS to minority-margin gains, path-norm-like capacity control, and stability benefits via EMA-smoothed gating. Empirically, AGS improves worst-group accuracy and calibration vs.\ ERM and is competitive with state-of-the-art methods across spurious-correlation benchmarks (e.g., Waterbirds, CelebA, BAR, COCO), while maintaining strong average accuracy. These results suggest that regulating internal alignment flow is a simple and scalable route to robustness without group labels.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes tracking the contribution of each penultimate-layer feature to the classification prediction via an evidence score which is very negative for features which strongly support the prediction. The proposed Evidence Gated Scoring (EGS) algorithm estimates the evidence energy via an exponential moving average, then “gates” (zeros out with a stop-gradient) the features with the largest evidence score (in absolute magnitude). Experiments are provided which show EGS acts as a regularizer for overconfident “shortcut” neurons; empirical analyses, benchmarks, and a theoretical study are also provided.

### Strengths
1. The paper is very well-written with clear, intuitive explanations as well as rigorous experimental and theoretical justification. The “connections to broader ML principles” section in the Appendix is also a nice touch. Overall, beyond the proposed EGS method, this paper provides what I would term a “mechanistic lingua franca” for theorizing about how individual features affect group robustness.

2. The analysis and ablation studies on the EGS method are well-done, especially the Section 5 discussion, and I appreciate the additional correlation studies in the Appendix. Overall, the EGS method is elegant and each of its components are strongly justified.

3. The theoretical results complement the empirical analyses, with justified and clearly denoted assumptions.

### Weaknesses
The main weaknesses of the paper relate to the performance, evaluation, and comparison of the proposed EGS method.

1. Worst-group accuracy performance is quite poor on Waterbirds; I disagree with the paper’s assessment that it has “near-top WGA” on this dataset (lines 370, 448). State-of-the-art methods should achieve 90% or higher WGA [1, 2]. More importantly, ERM with simple class-balancing can attain 82.9% WGA [3, Table 2], meaning that EGS is worse than a standard data balancing baseline on this dataset.

2. More generally, the performance comparisons in Table 1 do not reflect the state-of-the-art; only a single reference from after 2023 is listed. Please consider including more recent competitive methods, e.g., [2, 4, 5, etc].

3. I’m a bit confused on the difference between _conflicting_ accuracy and _worst-group_ accuracy, and why the former is reported for CelebA/COCO while the latter is reported for Waterbirds. It would be nice to also report WGA for CelebA/COCO to compare with results from the literature which also report WGA.

4. If the main claim on the performance of EGS is that it “advances the Pareto frontier” of group robustness methods, then a method for interpolating between different average vs. worst-group trade-offs should be provided. In particular, a method which achieves Pareto gain at an _a priori_ undetermined point on the curve is not very useful in practice, as the user is unsure of what trade-off to expect before deployment, and it may lose out to techniques which more explicitly target WGA (e.g., as EGS does on Waterbirds).

5. Only a single data domain (computer vision) and model architecture (ResNet50) are examined. While not strictly necessary, it would improve the claims to benchmark EGS using BERT on language datasets such as CivilComments or MultiNLI, as is common in the group robustness literature.

[1] Kirichenko et al. Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations. ICLR 2023.

[2] Noohdani et al. Decompose-and-Compose: A Compositional Approach to Mitigating Spurious Correlation. CVPR 2024.

[3] Idrissi et al. Simple data balancing achieves competitive worst-group-accuracy. CLeaR 2022.

[4] Yang et al. Identifying Spurious Biases Early in Training through the Lens of Simplicity Bias. AISTATS 2024.

[5] Han et al. Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference. ICML 2024.

### Questions
1. EGS is predicated on the assumption that any feature which disproportionately contributes to a certain class is likely to be a spurious feature (and so would be gated by EGS). What evidence is there that this assumption holds? I am concerned this assumption may not be justified in situations where the invariant and spurious features have similar “complexity”. For example, [1] show that the invariant and spurious features on CelebA (hair color and gender respectively) contribute roughly equally to the class prediction (Section 5.5).

2. I’m curious how EGS relates to the line of work on feature diversification and ensembling, e.g., [2, 3, 4, 5], which similarly try to avoid overconfident shortcut-driven features.

3. I wonder whether EGS can be formulated via a min-max objective which minimizes the maximum evidence energy contributed by any neuron, thus encouraging a roughly uniform contribution from each feature. If not, what is the difference with the EGS objective?

4. Is EGS robust to model selection without group annotations, e.g., by selecting based on worst-class accuracy [6] or the bias-unsupervised validation score of [7]?

5. Some clarity/grammatical improvements:
    1. It would increase clarity to properly utilize \citep to put parentheses around citations.

    2. Missing semicolon or period in line 159

    3. The order of Section 3.4 and Section 3.5 could potentially be switched since Section 3.4 references Section 3.5

[1] Vasudeva et al. Mitigating Simplicity Bias in Deep Learning for Improved
OOD Generalization and Robustness. TMLR 2024.

[2] Addepalli et al. Feature Reconstruction From Outputs Can Mitigate Simplicity Bias in Neural Networks. ICLR 2023.

[3] Rame et al. Model Ratatouille: Recycling Diverse Models for Out-of-Distribution Generalization. ICML 2023.

[4] Zhang et al. Rich Feature Construction for the Optimization-Generalization Dilemma. ICML 2022.

[5] Lin et al. Spurious Feature Diversification Improves Out-of-distribution Generalization. ICLR 2024.

[6] Yang et al. Change is hard: A closer look at subpopulation shift. ICML 2023.

[7] Tsirigotis et al. Group robust classification without any group information. NeurIPS 2023.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Evidence-Gated Suppression (EGS), a regularizer that computes a per-neuron, class-conditional "evidence" score and suppress the feature neurons that are most likely spurious. The method tracks an EMA of the evidence energy and reduces those with over confidence. It achieves improvement across Waterbirds, CelebA, BAR and COCO with 5% computation overhead.

### Strengths
1. The paper presents a clear conceptual framework: regulating internal, class-conditional confidence flows to mitigate shortcut reliance. Unlike many debiasing methods that act on the data level (e.g., reweighting, DRO), EGS directly intervenes on neuron–class weight inside the model.

2. The paper supports its design with theoretical properties: (1) scale-invariance under logit reparameterization, (2) stability from EMA-based gating, (3) class-wise proximal shrinkage behavior, and (4) selective suppression of spurious features with preservation of robust ones.

3. The paper shows strong empirical results on multiple spurious-correlation benchmark, demonstrating its effectiveness.

### Weaknesses
1. **The claimed contribution of defining evidence energy is highly concerning**. The formulation closely mirrors that of Eq. (7) in EvA [1], which already introduced a similar concept with exactly the same name evidence energy. Moreover, EvA-E also evaluates evidence energy on a class-wise basis to suppress spurious features. Both definitions essentially measure neuron confidence toward predictions. The primary distinction is that EGS integrates this process during training, whereas EvA applies it post-hoc. Although EvA-E is included in the experiments, the paper fails to explicitly discuss the differences and instead claims the definition of evidence energy as its own contribution.

2. While the paper highlights low computational overhead, it omits critical details such as training FLOPs and total GPU hours. The claim that EGS is “plug-and-play” is weaker than that of previous post-hoc approaches, as it still requires full model retraining and tracking throughout the training process.

3. The paper does not include experiments on large-scale or challenging datasets, such as spurious-feature variants on ImageNet-level [2, 3]. Such evaluations are essential to substantiate the claimed generalization and robustness benefits.

[1] Erasing Spurious Correlations with Activations, ICLR 2025.

[2] Salient ImageNet: How to discover spurious features in Deep Learning? ICLR 2022.

[3] Large-Scale Detection of Harmful Spurious Features in ImageNet, ICCV 2023.

### Questions
1. The proposed evidence energy appears nearly identical to the formulation in EvA [1]. Could the authors clarify how their definition fundamentally differs from that of Eq. (7) in EvA? A more explicit comparison and proper attribution seem necessary.

2. Have the authors evaluated EGS on large-scale benchmarks such as ImageNet or its spurious-feature variants [2, 3]?

3. Have you compared with other classical penalties including weight decay, dropout or class weights adjusting?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a method called Evidence-Gated Suppression (EGS).

The core idea is to track each neuron’s confidence-weighted evidence toward different classes and softly suppress those neurons that consistently provide strong but potentially spurious signals. 

It uses a percentile-based gating mechanism and EMA smoothing to apply multiplicative decay to low-tail features, to weaken shortcut pathways while preserving robust ones.

EGS achieves strong and consistent performance across CelebA, Waterbirds, BAR, and COCO.

### Strengths
- conceptually easy to understand and simple to implement. 
- works without any group or environment annotations, which makes it practical for real-world applications.
- strong and consistent improvements on multiple spurious-correlation benchmarks (CelebA, Waterbirds, BAR, COCO).

### Weaknesses
- I noticed that the definition and formulation of evidence energy in Sections 3.1–3.2 look very similar to what was introduced in He et al. (2025). It would be helpful if the authors could clarify how their “evidence energy” differs conceptually or technically from that prior work.
- it would be interesting to see how EGS works on Transformer architectures, since the paper mainly analyzes CNNs. A brief discussion or experiment on Transformers could make the work stronger.

### Questions
- I also how this kind of spurious correlation behaves in vision–language models (VLMs). Since VLMs often learn strong cross-modal associations, will it be different to the normal classification task like what are defined here?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Evidence-Gated Suppression (EGS), a lightweight, plug-in regularizer designed to combat shortcut learning in deep models without requiring group labels. EGS operates inside the network during training by tracking a class-conditional, confidence-weighted "evidence energy" for each neuron to identify which neurons contribute most strongly to the model's predictions. It then applies a percentile-based multiplicative decay to the weights of these extreme contributors, selectively suppressing overconfident shortcut pathways while leaving more robust features relatively influential. The authors demonstrate across several spurious correlation benchmarks (such as Waterbirds and CelebA) that EGS improves worst-group accuracy and calibration, achieving competitive performance with state-of-the-art methods while maintaining strong average accuracy and adding minimal training overhead.

### Strengths
While many methods address spurious correlations by reweighting data (e.g., JTT, LfF) or modifying the loss function (e.g., GroupDRO), EGS introduces a fundamentally different locus of intervention: the internal evidence pathways of the network itself. The concept of a dynamic, training-time, *neuron-level* regularizer that is both class-conditional and group-agnostic is original. The "evidence energy" metric, which combines model confidence (`pk(x)`) with feature-weight alignment (`Wjk * φj(x)`), provides a simple yet powerful signal for identifying and suppressing over-reliant pathways.

### Weaknesses
The central heuristic of EGS is that the most negative (i.e., highest-confidence, class-aligned) evidence corresponds to spurious features. While this holds true in many shortcut-learning scenarios, it is not guaranteed. A genuinely robust and highly discriminative feature could also consistently produce very strong evidence and be inadvertently suppressed by the percentile gate. The paper would be strengthened by an analysis that more directly validates this core assumption. For example, the authors could run EGS on a dataset known to have minimal spurious correlations (e.g., a balanced version of a dataset or even standard CIFAR-10) to demonstrate that the method does not harm performance when strong shortcuts are absent.

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

### Contribution
4
