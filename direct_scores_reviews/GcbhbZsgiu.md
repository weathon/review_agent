## Summary
MixUnlearn proposes a generator-unlearner adversarial framework for machine unlearning that leverages mixup samples to regularize the unlearning process and mitigate "catastrophic unlearning," wherein erasing specific data inadvertently damages model generalizability in the interpolation space between forgetting and remaining data. A learnable MixBlock generator is trained via a novel contrastive objective in an adversarial direction—producing hard mixed samples that challenge the unlearner—while the unlearner is trained with two complementary contrastive losses on both synthetic and real samples. A key practical contribution is a label-agnostic variant that replaces ground-truth labels with sharpened pseudo-labels from the initial model, enabling unlearning in semi-supervised or weakly-labeled settings.

---

## Strengths

- **Adversarial mixup formulation specifically targeting the interpolation zone.** Unlike prior augmentation-based unlearning methods (UNSIR, GLI) that perturb retained samples, MixUnlearn explicitly synthesizes samples in the feature-space boundary between forgetting and remaining data and optimizes the generator adversarially to produce maximally hard mixtures. This is a mechanistically distinct and motivated approach, supported by the toy example and t-SNE visualizations in Figure 3 showing the restoration of class structure around the airplane/bird boundary that LAF fails to recover.

- **Label-agnostic capability with minimal degradation.** The use of sharpened f_D predictions as pseudo-labels (Eq. 4) allows the full framework to function without any ground-truth labels. Across Tables 1 and 2, the label-agnostic variant closely tracks the label-aware variant (e.g., CIFAR-10 class-level: Test_r 86.32% vs 87.10%), demonstrating this is not merely a degenerate fall-back but a fully functional operating mode. This is practically significant for semi-supervised or sparsely-annotated deployment scenarios.

- **Comprehensive ablation isolating loss components.** Table 3 systematically removes L_real, L_mix, MixBlock, and the sharpening operation, revealing clear contributions: removal of L_real causes catastrophic collapse (Test_r drops to ~29% on CIFAR-10 and ~27% on SVHN), while removal of L_mix degrades representation quality as confirmed in Figure 3(d). This granularity distinguishes the paper from ablation-lite unlearning work.

- **KDE loss distribution analysis.** Figure 4 shows MixUnlearn uniquely recovers the Retrain's loss distribution alignment between forgetting and unseen samples—a diagnostic that directly quantifies the "catastrophic" effect the paper claims to address, and that LAF and the initial model both fail to achieve.

---

## Weaknesses

- **Likely notation error in Eq. 5 that affects reproducibility.** In Eq. 3 (L_gen), the outer summation runs over `x_j ∈ B_r` (remaining), where x_j is the remaining sample used to form x_ij_mix = g(x_i, x_j, λ). In Eq. 5 (L_mix), the outer summation appears to run over `x_j ∈ B_f` (forgetting)—but the paper then states that this loss "retains the knowledge of x_j," which contradicts x_j being a forgetting sample. The paper's own explanation of Eq. 5 ("directs f_U to remove information about x_i while retaining the knowledge of x_j") only makes sense if x_j is still a remaining sample and the outer sum should be over B_r. This appears to be a typo, but it is in the critical technical contribution of the paper and a reader attempting to reimplement from the text alone would be misled. This must be corrected and clarified.

- **Ablation cannot isolate the adversarial objective from the MixBlock architecture.** The primary ablation ("w/o MB") replaces MixBlock with vanilla linear interpolation while keeping the contrastive losses. This compares "adversarial MixBlock-based mixing" against "adversarial vanilla mixing"—but there is no ablation that keeps MixBlock while removing the adversarial generator training (i.e., training MixBlock with a non-adversarial or reconstruction objective). The performance gains attributed to the adversarial strategy may in part be attributable to MixBlock's attention-based spatial mixing producing structurally richer samples, not to the adversarial loss direction per se. The paper should add an ablation that fixes MixBlock but removes or reverses the adversarial objective.

- **SCRUB's anomalously low performance raises baseline validity concerns.** SCRUB (Kurmanji et al., 2024) achieves Test_r = 34.12% on CIFAR-10 and 20.33% on SVHN for class-level unlearning—essentially random on a 9-class problem. For a published method at this level, this performance is implausibly poor and strongly suggests a hyperparameter misconfiguration or implementation error. If baselines are not adequately tuned, the magnitude of MixUnlearn's advantage is overstated. The paper should either confirm SCRUB settings explicitly in the main text or acknowledge that competitive baselines may require additional hyperparameter search.

- **ASR metric definition is ambiguous.** Section 5.3 states metrics "closer to Retrain" indicate better unlearning, but the Retrain's ASR on CIFAR-10 is 67.98%—well above 50% random-guessing. This suggests ASR is evaluated on the remaining (or full test) data as a utility measure, not specifically on D_f to verify that membership information about the forgetting set has been erased. For the ASR to serve as a forgetting verification metric, it should be evaluated specifically on D_f, where a value near 50% would indicate genuine forgetting. The paper should clarify the exact data population used for MIA, and whether it measures forgetting quality or general model utility.

- **Narrow margins at data-level unlearning, particularly in label-agnostic setting.** For CIFAR-10 label-agnostic data-level unlearning, MixUnlearn achieves Test=84.82±1.39 versus L-Mix's 84.56±1.46—a difference of 0.26%, well within one standard deviation of both methods. Similarly, SVHN label-agnostic: Ours Test=92.46±0.47 vs L-Mix Test=92.44±0.69. These narrow margins undermine the paper's language of "consistently outperforms" and "significantly greater gains." The contribution for data-level unlearning in the label-agnostic setting appears modest; the paper should be more calibrated in its claims for this regime.

- **Large-scale experiments (ImageNet + ViT) are absent from the main body.** The only mention is a single sentence in §5.4 deferring to Appendix A.11. For ICLR-level claims about general machine unlearning, results on CIFAR-10/SVHN with ResNet-18 are necessary but not sufficient to establish scalability. This is particularly important given that the adversarial training loop and MixBlock feature mixing operate in the feature space of f_D, and the behavior may differ substantially under large-scale transformers.

---

## Nice-to-Haves

- **Ablation on generator update frequency.** The generator is updated once every 4 iterations "for efficiency," but there is no sensitivity study on this schedule. Given that adversarial training dynamics can be sensitive to update ratios, a brief sweep would help readers understand the design envelope.

- **Stronger privacy evaluation.** The sole privacy metric is MIA (Shokri et al., 2017). More recent approaches, such as likelihood ratio tests or shadow-model attacks, would better characterize the privacy guarantees of the method, especially as the paper invokes GDPR motivation. This would not change the method but would strengthen the privacy framing.

- **Visualization of generated mixup samples.** Appendix A.9 reportedly visualizes a mixed sample, but making this visible in the main paper (or expanding it) would provide intuitive evidence that MixBlock generates semantically meaningful interpolations rather than adversarial noise. This directly validates the paper's core geometric intuition.

- **Testing robustness to f_D feature quality.** The method relies on h_D(·) from the initial model for both MixBlock input and pseudo-label generation. A brief experiment restricting or corrupting f_D feature access would show how robust the method is to degraded initial model quality, which matters for practical deployments.

- **Discussion of single-sample unlearning.** All experiments remove substantial fractions of data (e.g., an entire class, or 40% of some classes). The behavior for removing a single data point—the most privacy-sensitive and GDPR-relevant case—is unexamined and could be noted as a scope clarification or future work.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"The claim of 'significantly outperforms' is overstated"** — while some margins are narrow (handled in Weaknesses), in the class-level unlearning setting and MNIST/FASHION-MNIST the advantages are substantial and consistent across 4 datasets. The language is reasonable.

- **"Temperature asymmetry in Eq. 3 is unexplained"** — the temperature τ_gen appears only in the denominator (forgetting terms), while the mixing ratio λ controls weighting in the numerator. The paper explicitly explains that λ controls the weight between the two SimLoss terms. The asymmetric structure is a deliberate design choice tied to the mixing ratio semantics, not a standard contrastive loss error.

- **"The cat/dog example assumes nearby classes; this is unstated and not always valid"** — this is scope creep. The paper uses the cat/dog illustration as a motivating example; it does not restrict the method to semantically similar classes, and the experiments span diverse class pairs across CIFAR-10, SVHN, MNIST, and Fashion-MNIST.

- **"Theoretical bounds / certified forgetting are missing"** — the paper is a systems/empirical contribution. Demanding theoretical privacy guarantees or differential privacy bounds is not a standard expectation for this class of approximate unlearning papers at ICLR. Moved to Nice-to-Have.

- **"LAF+R comparison is unfair because LAF+R is fundamentally label-agnostic"** — the paper explicitly categorizes LAF+R as a "Label-Aware" baseline (Table 1) under the condition that "Full label information is provided to both these baselines and MixUnlearn during our comparisons." This is a deliberate choice to give LAF+R the advantage of labels, making it an intentionally conservative/asymmetric comparison that strengthens MixUnlearn's label-agnostic case. The asymmetry favors the baseline.

- **"Missing confidence intervals for large-scale benchmarks"** — the paper already provides ±std across 5 seeds throughout. This is standard practice.

- **"Data-level unlearning setup (40%, classes 5–9) is unjustified"** — this setup follows Shen et al. (2024), which the paper explicitly cites. Criticizing the setup absent knowledge of the source paper's rationale is not well-founded.

- **"The paper claims to be among the first to use mixup for catastrophic unlearning but DSMixup, UNSIR, GLI predate it"** — the paper explicitly distinguishes its contribution from DSMixup (exact vs. approximate unlearning, different mixing targets), UNSIR (class-level only, uses artificial noise not mixup), and GLI (noise perturbation, not interpolation between forgetting/remaining). The claim specifically concerns mixup for *catastrophic* unlearning in *approximate* unlearning, which is defensible.

---

## Novel Insights

The most genuinely novel analytical contribution is the identification that catastrophic unlearning arises specifically from insufficient regularization in the *interpolation zone* between forgetting and remaining data—the region where forgetting and retention operations produce conflicting gradient signals on unseen samples. This goes beyond prior work, which primarily retains knowledge on *observed* remaining samples without addressing the interpolation manifold. The adversarial framing operationalizes this insight: by training a generator to maximize exploitation of the unlearner's interpolation-zone vulnerability, the method achieves self-correcting regularization. The KDE analysis in Figure 4 provides an empirical diagnostic for this effect that could be adopted as a standard evaluation tool by the unlearning community.

---

## Final Evaluation

**Novelty:** The adversarial generator specifically targeting the forgetting-remaining interpolation zone is a genuinely new mechanism in approximate unlearning. The label-agnostic formulation is meaningfully differentiated from prior work.

**Technical soundness:** The framework is conceptually coherent and the loss design is motivated, but the likely notation error in Eq. 5 and the inability to cleanly separate adversarial optimization from MixBlock architecture in ablations leave important questions about whether the method is fully reproducible and what drives the gains.

**Empirical support:** Strong in class-level unlearning across four datasets; weaker for data-level unlearning in the label-agnostic setting where margins are within noise. The anomalously poor SCRUB reproductions and the ambiguous ASR metric reduce confidence in the quantitative magnitude of the improvements.

**Significance:** Catastrophic unlearning is a real and underappreciated failure mode; the proposed framework addresses it with practical applicability (label-agnostic, efficient) and strong positive results in the main settings.

**Clarity:** The paper is generally well-written; however, the critical loss equations (Eqs. 3 and 5) have clarity issues that must be resolved before this can be considered reproducible.

MY FINAL SCORE: <pineapple>5.8</pineapple>