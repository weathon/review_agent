# Boosting Open Set Recognition Performance through Modulated Representation Learning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
The open set recognition (OSR) problem aims to identify test samples from novel semantic classes that are not part of the training classes, a task that is crucial in many practical scenarios. However, the existing OSR methods use a constant scaling factor (the temperature) to the logits before applying a loss function, which hinders the model from exploring both ends of the spectrum in representation learning -- from instance-level to class-specific features. In this paper, we address this problem by enabling temperature-modulated representation learning using a set of proposed temperature schedules, including our novel negative cosine schedule. Our temperature schedules allow the model to form a coarse decision boundary at the beginning of training by focusing on fewer neighbors, and gradually prioritizing more neighbors to smooth out the rough edges. This gradual task switching leads to a richer and more generalizable representation space. While other OSR methods benefit by including regularization or auxiliary negative samples, such as with mix-up, thereby adding a significant computational overhead, our schedules can be folded into any existing OSR loss function with no overhead. We implement the novel schedule on top of a number of baselines, using cross-entropy, contrastive and the ARPL loss functions and find that it boosts both the OSR and the closed set performance in most cases, especially on the tougher semantic shift benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses open-set recognition by replacing the fixed training temperature used in common OSR losses with epoch-wise temperature schedules that modulate representation learning from instance-level to semantic-level focus. 

It proposes a generalized cosine framework and a new negative cosine schedule (NegCosSch) that starts at a low temperature to form coarse, separative boundaries and then increases to compact within-class clusters; linear and exponential increasing variants are also considered. The schedules plug into Cross-Entropy, Supervised Contrastive, and ARPL losses without extra computational cost. 

Experiments on TinyImageNet and the Semantic Shift Benchmarks (CUB, FGVC-Aircraft, Stanford Cars) demonstrate consistent gains over constant-temperature and prior cosine schedules in closed-set accuracy and open-set metrics (AUROC, OSCR), with improvements that tend to increase as the number of training classes increases. 

The claimed contributions are (i) analysis of temperature effects for OSR, (ii) plug-and-play schedules, especially NegCosSch, applicable across losses, and (iii) empirical improvements on tougher benchmarks.

### Strengths
The paper’s strengths span four fronts. On originality, it repurposes temperature scheduling, previously explored mostly for closed-set/self-supervised contrastive learning, into a unified, loss-agnostic mechanism for open-set recognition, with a negative cosine schedule that explicitly traverses instance→semantic regimes. This is a simple yet creative removal of the fixed-τ limitation that has constrained prior OSR training dynamics. 

In terms of quality, the method is evaluated across multiple datasets of varying difficulty, multiple loss families (CE, SupCon, ARPL), and reports averages over five random seeds, showing consistent AUROC/OSCR and accuracy gains without requiring extra computation or architectural changes. 

The paper clearly motivates the role of temperature, presents the proposed schedules using a generalized cosine formulation, and supplements intuition with gradient analyses and UMAP visualizations that illustrate how clusters evolve during training. 

In terms of significance, the contribution is pragmatically valuable: it provides a drop-in schedule that improves both open- and closed-set metrics, scales favorably as the number of training classes increases, and can be integrated into existing OSR pipelines with negligible engineering overhead, thereby lowering the barrier to adoption on more challenging semantic-shift benchmarks.

### Weaknesses
The paper’s novelty is primarily curatorial, adapting known temperature-scheduling ideas (e.g., cosine schemes) to OSR, without a stronger theoretical account tying the proposed schedules to measurable properties of open-set boundaries; parameter choices such as (τ+, τ−), the period P, and the “finish high-τ” heuristic are fixed largely by heuristic guidance rather than principled sensitivity analyses. 

Empirically, improvements are modest and reported without dispersion or significance testing; although the text states five seeds were used, tables report means only, so it is unclear whether +1–3% AUROC/OSCR gains exceed run-to-run noise. 

Evaluation breadth is limited: results focus on TinyImageNet and SSBs with a narrow architecture set (VGG-like, ResNet-50; ViT deferred to the appendix) and do not probe domain shift or cross-dataset OSR, leaving external validity uncertain. Comparisons emphasize constant-τ and cosine baselines; several competitive contemporary OSR methods in the related work are not reproduced under the same protocol, making relative advantage ambiguous. 

Finally, the “no computational overhead” claim overlooks the practical cost of tuning τ-ranges and schedule shapes, as well as the UMAP visualizations. While intuitive, these visualizations lack quantitative cluster diagnostics (e.g., intra/inter-class scatter, uniformity) to substantiate the representation-learning narrative.

### Questions
1. Describe the dispersion and significance of improvements. Report mean±std over the five seeds you mention and include 95% CIs and paired tests for AUROC/OSCR/accuracy across all tables; currently tables list means only.

2. Run factorial sweeps over (τ+, τ−) and period P for NegCosSch and monotonic variants, showing Pareto frontiers of closed- vs open-set metrics. Your choices (e.g., τ+∈{0.3–0.4}, τ−=0.1; “finish with high τ”) are presently heuristic. Quantify the robustness of performance to these settings.

3. Justify why starting low-τ then increasing is fundamentally better for OSR.

4. You use max-logit for unknown detection under SupCon/CE. Compare against energy score, ODIN-style perturbations, and cosine-margin heads, holding the backbone/schedule fixed. Does NegCosSch still win irrespective of the scoring rule?

5. Breadth and strength of baselines
   Reproduce stronger contemporary OSR baselines under your exact protocol/splits, not only constant-τ or CosSch: e.g., BackMix (TPAMI 2025), iCausalOSR (PR 2024a). This clarifies whether scheduling alone is competitive at today’s frontier.

6. You claim plug-and-play for “any OSR loss”. Test at least one additional family—e.g., OpenAUC-style objectives or prototype-based methods—and report whether schedule benefits persist without retuning.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores the influence of the temperature parameter in loss of OSR and introduces a novel temperature scheduling strategy NegCosSch. It varies the temperature during training, enabling the model to balance instance-level and semantic-level features in representation learning without additional computational cost. Extensive experiments on the CIFAR and SSB benchmarks demonstrate consistent performance improvements.

### Strengths
+ This paper is well-written and easy to follow. I favor studies that derive methodological innovations from experimental observations.

+ The details provided in the Appendix are clear and crucial, enhancing the paper’s overall logical flow and strengthening the credibility of some experiments.

+ The empirical evaluation is extensive, covering multiple benchmarks and ablation studies.

### Weaknesses
+ The choice of $P$, $(\tau ^{+}, \tau ^{-})$ in the method is mainly heuristic. A more thorough theoretical analysis would improve the quality of the paper.

+ Some of the authors’ statements lack experimental or theoretical support. For example, the discussion regarding empty regions in the representation space, the significant computational overhead in existing methods, and the statement *'Therefore, the methods that demonstrate improvement on smaller datasets…'*(lines 096–098).

+ From my perspective, Sec. 4.2 constitutes one of the key sections for motivation. However, it remains unclear how the subsequent discussion is derived from Eq. (4), further clarification is needed.

+ The figures (especially Fig.1 and Fig. 2) are not positioned near the relevant text, which hinders readability.

+ Typos issue: line 135 *Section, 3*

### Questions
My questions are mainly focused on the experimental section:

+ Why is label smoothing introduced in Sec. 5.2? If it is only because “As label smoothing (LS) has shown performance improvements,” the purpose is not clearly justified.

+ Why are the results of the Vision Transformer model on TinyImageNet presented? If this section is necessary, shouldn’t results on larger-scale datasets be shown?

+ I would like to see the results about the metric OpenAUC [1]. 

+ The paper does not mention a validation set. How were the hyperparameters determined? Was the test set used for tuning, and are the final results reported based on that? I am mainly concerned.

Minor point:

+ Do some of the terms refer to the same concept (semantic-level, class-level, group-wise)? If so, please unify them.

[1] Wang, Zitai, et al. "Openauc: Towards auc-oriented open-set recognition." Advances in Neural Information Processing Systems 35 (2022): 25033-25045.

If authors address the Weaknesses and Questions, I will increase my score.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose temperature schedules for reprsentation learning
in open set recognition.  They use supervised contrastive loss for
representation learning.  For each sample in a batch, they create two
augmentations. Positive pairs are from the same class.  They use
temperature in softmax to regulate the representation of intra-class
samples (relative to inter-class samples).  According to the
Familiarity Hypothesis (D&G, 2022), most OSR methods flag novelty
based on the absence of semantic features and recommends to extract
"interesting content features" beyond the semantic features. The
authors propose varying the temperature to learn semantic features and
instance ("content") features.  Based on CosSch (starting with higher
temperature and going toward lower), they propose NegCosSch, which
starts from lower temperature and going higher.  Both follows a cosine
function with a certain period.  During the first half of a cosine
cycle, increasing temperature encourages semantic features.  During
the second half of a cosine cycle, decreasing temperature encourages
instance features.  Besides the periodic version (P-NegCosSch), they
propose a monotonic version (M-NegCosSch) which only increases in
temperature.

For evaluation, they compare 6 existing methods over 4 datasets.  On
methods, they added exponential increase and linear increase as well
in addition the two versions of NegCosSch.  The empirical results in
Table 1 indicate their 4 methods generally outperform compared
methods.  However, among their four methods, a generally more
effective methods is not apparent.  When used with different loss
functions, P-NegCosSch and M-NegCosSch generally performs better than
constant temperature, while P-NegCosSch generally outperform the
others.  They also found that their method can benefit from more
classes.

### Strengths
1.  For open set recognition, Negative Cosine Schedule (NegCosSch) is
proposed for temperature scheduling in softmax used in supervised
contrastive learning.

2.  The proposed variations of P-NegCosSch and M-NegCosSch together
with exponential increase and linear increase generally outperforms
the other 6 existing methods over 4 datasets.

3.  The paper is generally well written.

### Weaknesses
1.  The proposed method is a minor variation of the existing NegCosSch
method for temperature scheduling.  Hence, the novelty level is not
high.

2.  The P-NegCosSch and M-NegCosSch do not seem to perform better than
the simpler exponential increase and linear increase.

3.  The motivation for the periodic temperature schedule could be
further discussed.  Also, it does not seem to outperform the monotonic
increasing version.

4.  Part of the motivation is from semantic features vs
instance/content features, further analysis with NegCosSch would be
significant.

### Questions
1.  Since P-NegCosSch does not seem to generally outperform
M-NegCosSch, what are the main reasons?  Could the 2nd half of a cycle
"undo" the 1st half of a cycle?

2.  p6: "Otherwise, P in Eq. (7) can be chosen by dividing E by the
number of cycle?"  How does one choose the number of cycles?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the open-set recognition (OSR) problem, aiming to enhance the model's ability to identify samples of novel semantic classes unseen during training when tested. The authors propose a modulated representation learning method that dynamically adjusts the temperature parameter in the loss function by introducing temperature scheduling, especially a novel negative cosine scheduling. This enables the model to focus on instance-level features to form rough decision boundaries in the early stages of training and gradually transition to semantic-level features to smooth the boundaries. Experiments demonstrate that the proposed method improves both OSR and closed-set performance across various loss functions and benchmark datasets without additional computational overhead.

### Strengths
The idea of temperature scheduling is novel and insightful. In particular, the negative cosine scheduling achieves a smooth transition of task switching by reverse-adjusting the temperature, which is more effective than fixed temperature or traditional cosine scheduling. The method does not require introducing additional regularization or generating synthetic samples, reducing computational costs and providing a concise yet efficient solution to the OSR problem.

### Weaknesses
Although comparisons with baseline methods (e.g., ARPL, SupCon) have been conducted, some recent OSR methods based on augmentation or multi-expert models (e.g., Wang et al. 2024) are not included. It is recommended to incorporate a broader range of comparisons in the rebuttal or future work to fully demonstrate the superiority.
The performance improvement is limited on benchmarks with a small number of training classes (e.g., CIFAR), indicating that the method may be more suitable for scenarios with a rich number of classes. The authors should explicitly discuss this limitation and explore ways to extend it to small-scale data.
The selection of the temperature range (τ⁺, τ⁻) and period P relies on heuristic rules (e.g., τ⁺=τ+Δ). Despite parameter tuning, there is a lack of theoretical guidance. It is suggested that the authors analyze the boundaries of parameter sensitivity or provide adaptive adjustment strategies to enhance robustness.

### Questions
1. On Page 1 (near line 242), does "group-wise features" specifically refer to "class-wise features" (i.e., class-level features)? If so, it is recommended to uniformly use the standard terminology "class-wise" to avoid terminology confusion. If not, please explain the specific differences between "group-wise features" and "class-wise features" to help readers better understand the technical perspectives in the paper.
2. The summation index a in Formula (2) is not defined. Does a iterate over the set A(i), representing all samples in the batch except the anchor i?

### Soundness
3

### Presentation
3

### Contribution
3
