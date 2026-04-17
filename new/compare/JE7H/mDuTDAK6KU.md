---
job_id: db93f08a-b245-44f2-9ace-264a2c9bc75f
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: mDuTDAK6KU.pdf
paper: KoALA: KL-L0 Adversarial Detector via Label Agreement
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes an adversarial example detector for deep networks, with formal analysis and experiments on CIFAR-10 and Tiny-ImageNet; this fits squarely within adversarial robustness, metric learning, and safety in ML, all core ICLR topics.

## Minimum Quality
Pass ✅.  
The paper includes Abstract, Introduction, Related Work, Methodology, Experiments (with quantitative results in Tables 1–5), theory sections with detailed proofs, an Ethics statement, and a Reproducibility statement. The method is nontrivial, math is substantial, and experiments are reasonably detailed, with no obvious fatal methodological flaw in the main text.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden prompts or attempts to steer automated reviewers; the text is standard academic writing without manipulative instructions.

---

# Expected Review Outcome:

## Summary

The paper introduces KoALA, an adversarial example detector that replaces a classifier head with a nearest-prototype “detector head” operating on two metrics: KL divergence and an $L_0$-based distance on normalized feature embeddings. KoALA flags an input as adversarial when the class predictions induced by these two metrics disagree, and abstains from classification in that case. The authors provide a detailed theoretical analysis (Theorem 1 and supporting propositions) establishing conditions under which no $\ell_2$-bounded perturbation in feature space can fool both metrics simultaneously, and present experiments on ResNet-18/CIFAR-10 and CLIP/Tiny-ImageNet under PGD, CW, and AutoAttack to validate detection behavior and study metric combinations.

## Strengths

1. **Clear conceptual idea and detector design**  
   The central idea, to use two complementary metrics (KL on normalized features and a thresholded $L_0$-style count of “large” coordinate changes) and flag inputs when their induced nearest-prototype labels disagree, is simple and well articulated in Section 3.1. Equation (3) precisely defines the two labels $\hat y_{\text{KL}}$ and $\hat y_{L_0}$, and Equation (4) encodes the detector’s decision logic. This is easy to implement as a plug-in head, independent of the backbone architecture.

2. **Strong theoretical development (even if somewhat heavy-handed)**  
   The theoretical part (Section 3.2, Appendix B) is unusually detailed for an adversarial detection paper. Proposition 2 (Page 13–14) derives a necessary condition for a successful KL-based attack, involving the inner product $\sum_i (\hat c_i - c_i^*) \frac{\delta_i}{p_i^*} &gt; \Delta KL(p^*)$. Proposition 3 characterizes how many coordinates must move by at least a certain amount to flip the $L_0$ decision, and Theorem 1 then formalizes a condition on the gap $|c_i^* - \hat c_i| &gt; \Gamma_i(\epsilon)$ under which no $\|\delta\| \le \epsilon$ can simultaneously fool both metrics. The derivation is long but internally consistent and uses Taylor expansion (Equations (11)–(14)) and careful bounding of the remainder term.

3. **Explicit, nontrivial training objective**  
   The fine-tuning objective in Section 3.3 is concretely specified. Equation (5) defines a binary cross-entropy loss on the KL-based similarity $\mathrm{sim}_{KL}(\mathbf c,\mathbf p) = \exp(-KL(\mathbf c\|\mathbf p))$, and the $L_0$ similarity uses a smooth surrogate $\widehat{L_0}$ with a sigmoid, leading to $\mathrm{sim}_{L_0} = 1 - \widehat{L_0}/d$. The combined loss in Equation (6) is straightforward and practical. This gives a clear recipe to reproduce the reported results.

4. **Some insightful empirical analysis of metric combinations and “geometry”**  
   The ablation in Experiment 2 (Table 2, Page 8) is useful. On ResNet/CIFAR-10, the KL+L0 combination clearly dominates other pairings in Recall and F1 (e.g., at $\ell_\infty^{2/255}$, Accuracy 0.88 vs 0.73–0.78, F1 0.87 vs ≤0.74). For CLIP/Tiny-ImageNet, the KL+L0+Cosine combination gives higher Recall but for reasons the authors themselves critically dissect (essentially by breaking the classifier so metrics “randomly” disagree). This shows the authors do not cherry-pick and are willing to interpret results that cut against their initial design.

5. **Tight link between theorem and experiment (Experiment 1)**  
   Experiment 1 is carefully designed to validate the theorem’s conditions empirically. Table 1 splits each dataset into “Theorem-compliant” vs “Non-compliant” samples (based on prototype separations). For compliant samples, all metrics are exactly 1.0 (Acc, Prec, Rec, F1) under PGD for both backbones and $\epsilon$ values, while non-compliant subsets have significantly lower performance. Table 5 (Appendix A) gives the raw confusion counts, showing for compliant samples TP = sample size and FP = FN = 0. This is a nice alignment between theory and practice, albeit only under the internal notion of feature-space perturbations and prototype gaps.

6. **Architecture and pipeline clearly illustrated**  
   **Figure 2** (Page 3–4) is a good high-level visualization of the training and inference phases of the KoALA head. It shows perturbation flowing through the backbone encoder to an embedding, then through two nearest-prototype classifiers (KL and L0), with separate loss paths for $Loss_{KL}$ and $Loss_{L_0}$, and finally the agreement-based attack detection logic. The color-coding (red for KL band, blue for L0 band) helps connect with the intuition from **Figure 1** (Page 2), where dense vs sparse perturbations are shown as traveling outside complementary “stability bands.” Together they make the overall mechanism relatively intuitive, even for readers who will not follow every line of the proof.

7. **Empirical robustness gains on ResNet/CIFAR-10**  
   Table 3 (Page 9) is interesting beyond detection: it shows that fine-tuning ResNet-18 with the KL+L0 objective improves adversarial accuracy substantially relative to the baseline classifier head without explicit adversarial training. For example, under PGD with $\ell_\infty^{4/255}$, the baseline has 33.11% accuracy while KL+L0 yields 54.60%; similarly, for AutoAttack $\ell_\infty^{4/255}$, baseline is 31.95% vs 51.12% with KL+L0. Clean accuracy remains high (94.78%), so the extra robustness is not bought by severe clean performance collapse. While this is not the main claim, it is a meaningful empirical side-effect.

## Weaknesses

1. **Threat model and feature-space vs input-space mismatch**  
   The theorem and assumptions operate in *feature space*, with a bounded $\|\delta\|\le \epsilon$ applied to $p = f_\theta(I)$ and assumptions like $|\delta_i| \le \tfrac{3}{2}|p_i^*|$ (Assumption A3, Page 5). However, attacks in the experiments (PGD, CW, AutoAttack) are defined in *input space* under $\ell_\infty$ constraints (Section 4.1). There is no explicit bound connecting $\|\delta_x\|_\infty$ to the induced $\delta_p$ in feature space beyond a hand-wavy reference to “Lipschitz continuity” of the encoder. The theoretical statements, including Theorem 1, are therefore not directly about the actual threat model used in experiments. Without explicit Lipschitz constants or bounds on $f_\theta$, it is unclear when real image-space attacks satisfy the crucial inequality $|\delta_i| \le \tfrac{3}{2}|p_i^*|$ or the energy budgets assumed in the proofs. This gap materially affects how much weight one can assign to the claimed “formal guarantee.”

2. **Extremely complex and arguably impractical condition in Theorem 1**  
   Despite being “a proof of correctness,” Theorem 1 relies on a threshold $\Gamma_i(\epsilon)$ whose expression is buried in a long chain of inequalities (Pages 23–28). The final bound (Equation (78)–(85) and the definition of $\Gamma(\epsilon)$ at the end) involves norms of $v$, $\Delta KL(p^*)$, prototype differences, $p_j^*$, $\mu(\cdot,\cdot)$, and $\|\delta\|_1$, and it is not shown how one would *compute* or even *approximate* this threshold in practice to certify a given model. There is no numerical illustration of the magnitude of $\Gamma_i(\epsilon)$ for the actual ResNet or CLIP models used. As a result, the theorem reads more like a qualitative separation argument than an actionable condition for real-world detectors.

3. **Nonstandard and somewhat ad hoc $L_0$ distance definition**  
   The $L_0$ distance in Equation (2) (Page 4) is not the usual count of coordinates with $|c_i - p_i| &gt; \tau$ but rather compares each $|c_i - p_i|$ against $\tau\cdot\mu(c,p)$, where $\mu$ is the mean absolute difference. This has two implications:
   - The effective threshold is *input dependent*, so a single coordinate can be classified as “large change” or “small change” depending on global average distance, which complicates the interpretation of “sparse, high-impact perturbations.” There is no justification that this variant is the right surrogate for classic $L_0$-type sparsity, and the theoretical analysis in Proposition 3 is correspondingly very intricate.
   - From a robustness standpoint, an adversary might influence $\mu$ by spreading tiny perturbations, altering the threshold for what counts as “sparse.” There is no discussion or experiment investigating such adaptive strategies, even though the theoretical analysis essentially assumes fixed $\tau$ and a clean baseline $\mu(c,p^*)$.

4. **Lack of comparisons to existing adversarial detectors**  
   The experiments only compare different *metric combinations* within the KoALA head (KL+L0, KL+Cosine, etc.) but not against any established adversarial detectors. Given the Related Work (§2), obvious baselines include Mahalanobis-based detection (Lee et al., 2018), MagNet (Meng & Chen, 2017), detection via LID (Ma et al., 2018), NIC (Ma & Liu, 2019), or feature squeezing (Xu et al., 2018). There is no quantitative comparison in any testbed. For instance, Table 2 and Table 1 exclusively evaluate KoALA variants, giving no sense of how competitive the method is relative to modern detectors on CIFAR-10, which is a well-trodden benchmark. This is a major omission for an ICLR-level empirical paper.

5. **Very limited evaluation of adaptive attacks or white-box awareness of the detector**  
   The attacks used are “standard” PGD, CW, and AutoAttack applied to the backbone; there is no evidence of attacks *targeting* the KoALA detector. In particular, because KoALA declares an attack when $\hat y_{KL}\neq \hat y_{L_0}$, an adaptive adversary could explicitly optimize to keep these two predictions aligned while misclassifying (for example, the loss could be $L = \ell_{\text{cls}}(y_{\text{target}}) + \lambda \big|KL(\cdot) - L_0(\cdot)\big|$ or a surrogate that forces the nearest prototypes to match under both metrics). The theory argues that this is impossible when prototype gaps and $\epsilon$ satisfy certain bounds, but as noted earlier those bounds are not tied quantitatively to real networks. Without adaptive-attack experiments, it is unclear how brittle KoALA is once the defense is known.

6. **“Semantics-free” claim is overstated, especially for CLIP**  
   The paper calls KoALA “semantics-free” several times (Abstract, Page 2–3), arguing it operates purely on representation geometry. However, for CLIP/Tiny-ImageNet prototypes are *text embeddings* of class names (“a photo of [CLASS]”, Page 6–7). This is very much a semantic construction that leverages label text; the detector’s geometry is anchored by language semantics. The method is also heavily classification-centric and requires well-defined class prototypes, so calling it semantics-free or “modality agnostic” is misleading. A more accurate claim would be that KoALA does not require *additional* semantic side information beyond what is already implicitly available via the model or labels, and even that should be nuanced.

7. **Inconsistent story about robustness gains on CLIP**  
   For ResNet/CIFAR-10, KL+L0 clearly improves adversarial accuracy (Table 3). For CLIP, however, Table 4 shows that the best adversarial robustness is obtained by the *L0-only* objective (e.g., PGD $\ell_\infty^{2/255}$: 60.02% for KL vs 53.31% for L0; but under CW/AutoAttack the patterns are different, and overall L0-only or KL-only seem preferable to KL+L0). The narrative in Section 4.4 attributes this to CLIP’s pretraining, but it also undercuts the central claim that *simultaneously* optimizing KL+L0 is the right training recipe. The conclusions about which objective to recommend for practitioners become muddled: on CLIP, KL+L0 appears inferior to L0-only, thereby weakening the universality of the proposed strategy.

8. **Ambiguities in the experimental protocol and metrics**  
   - In Experiment 3 (Section 4.4, Tables 3–4), “PGD attack (%)” columns are presented, but it is not entirely clear if these are accuracies *after* discarding detected samples or accuracies of the backbone classifier irrespective of detection. The text says “adversarial accuracy (performance on successfully attacked images that were not detected),” but the tables list just “PGD attack (%)” without distinguishing detection vs classification behavior. This needs clarification to interpret how much improvement stems from better classification vs more abstentions.
   - In Experiment 1, the split into Theorem-compliant and Non-compliant samples is not operationally defined in the main text; we are told that Theorem 1 requires “sufficient inter-class prototype separation,” but not exactly which inequality is measured for each sample. This is critical because the central empirical claim (“recall = 1.0 on all compliant samples”) hinges on this construction. Without a clear, computable rule in Section 4.2, the reader has to trust that this identification was done correctly.

9. **Mathematical exposition is very heavy and sometimes error-prone in notation**  
   While the math appears internally consistent, the exposition is dense and occasionally sloppy in notation and typesetting, which hampers understanding:
   - In Proposition 2, Equation (7) the text defines both $\Delta KL(p^*)$ and $\Delta KL(\hat p)$ but then uses $\tilde c$ in Equation (9) instead of $\hat c$, suggesting minor but confusing typos.
   - In Proposition 3, the sets $\mathbb{S}^{\text{unchange}}$ vs $\mathbb{S}^{\text{exchange}}$ / $\mathbb{S}^{\text{change}}$ vs “c e m a i n” are inconsistently spelled across pages (e.g., Page 21 has apparent OCR glitches: “\mathbb{S}^{\text{c e m a i n}}”). This is not just cosmetic; these sets are central to defining the optimization in Proposition 4, and the reader has to work to infer which is which.
   - In Proposition 4 and subsequent derivations (Pages 18–23), the projection problem (Equations (50)–(54)) is correct in spirit but explained with a proliferation of indices and redefinitions; for instance, $a_i^*$ is introduced after Equation (58) without clear motivation. A more structured presentation with clearly separated lemmas and consistent notation would improve rigor and accessibility.

10. **No discussion of runtime/overhead and deployment considerations**  
    KoALA requires computing distances to all class prototypes under two metrics, plus a fairly expensive $L_0$-surrogate involving sigmoids and global means per prototype. There is no measurement of computational overhead relative to a standard softmax head. In large-scale settings (e.g., ImageNet-1k or language models with many classes/tokens), this may be nontrivial, and it is not addressed.

11. **Missing or under-discussed related work on sparse attacks and $L_0$ robustness**  
    The Related Work section omits several works specifically on sparse or one-pixel attacks that directly motivate $L_0$-sensitive detection. This weakens the framing around why $L_0$ is a natural second metric and how KoALA compares to detectors tailored to sparse perturbations (see next section for specific missing references).

## Potentially Missing Related Work

1. **Su et al., “One Pixel Attack for Fooling Deep Neural Networks” (2019)**  
   This paper introduces sparse, one-pixel adversarial attacks that align directly with KoALA’s notion of “sparse, high-impact perturbations” and $L_0$ geometry. It should be discussed in Section 2 (likely under “Detectors utilizing intrinsic statistics of attacks”) as a canonical example of sparse $L_0$-style attacks and could be used as an explicit threat model when motivating the L0 metric in Section 3.1.

2. **Nguyen-Son et al., “OPA2D: One-Pixel Attack, Detection, and Defense in Deep Neural Networks” (2021)**  
   This work not only uses one-pixel attacks but also proposes detection and defense mechanisms, making it directly relevant to KoALA’s detection angle. It should be cited and contrasted in Section 2, with a short discussion about how KoALA’s geometric disagreement criterion differs from OPA2D’s detection strategies, and ideally included as a baseline or at least as a conceptual comparison in CIFAR-10 experiments.

3. **Doan et al., “TnT Attacks! Universal Naturalistic Adversarial Patches Against Deep Neural Network Systems” (2022)**  
   This paper is about patch attacks, which are another form of spatially sparse but semantically significant perturbations. It is relevant when the authors explain sparse vs dense attacks in Figure 1 and Section 3.1. A brief discussion in Related Work could clarify whether KoALA’s $L_0$-based metric would be expected to catch such patch-based attacks and whether they are in scope.

4. **Li et al., “Towards Adversarial-Resilient Deep Neural Networks for False Data Injection Attack Detection in Power Grids” (2023)**  
   This is in a different domain (power systems) but is about adversarially resilient detectors in safety-critical infrastructure. Given the Introduction’s emphasis on safety-critical applications, this paper could be cited to contextualize KoALA among adversarial detection strategies in security-sensitive domains, perhaps in Section 1 or at the end of Section 2.

5. **Xiao et al., “Fooling Deep Neural Detection Networks with Adaptive Object-Oriented Adversarial Perturbation” (2021)**  
   This paper develops adaptive attacks targeting *detection* architectures. It is directly relevant to KoALA’s missing evaluation of adaptive, detector-aware adversaries. It should be discussed in Section 2, with explicit acknowledgment that KoALA has not been stress-tested under such adaptive scenarios and possibly as inspiration for future work.

6. **Yu et al., “Adversarial Parameter Attack on Deep Neural Networks” (2023)**  
   This explores parameter-space attacks rather than input-space ones, but still relates to the general question of how adversarial changes manifest in the representation geometry and how detectors might sense them. A brief mention in Section 2 could help situate KoALA among broader adversarial paradigms (input vs parameter attacks), even if parameter attacks remain out of scope here.

7. **Malatyński & Jaworek-Korjakowska, “Robust Detection of Directional Adversarial Attacks in Deep Neural Networks for Radiological Imaging” (2025)**  
   This is an example of adversarial detection in a critical domain (medical imaging), which the paper mentions as a motivation. Including it in Section 2 would help strengthen the link to real-world deployment scenarios and clarify how KoALA’s geometry-based approach differs from domain-specific detectors in radiology.

## Questions

1. **How exactly are “Theorem-compliant” samples identified in Experiment 1?**  
   Please provide a precise operational criterion in the main text (not just appendix) for deciding whether a test sample satisfies Theorem 1. What quantities are computed (e.g., $|c_i^* - \hat c_i|$ vs $\Gamma_i(\epsilon)$) and what thresholds are used?

2. **Can you quantify feature-space perturbation norms for the attacks used?**  
   For PGD with $\ell_\infty^{2/255}$ and $\ell_\infty^{4/255}$, what are the empirical distributions of $\|\delta_p\|_2$ and coordinate-wise $|\delta_i|/|p_i^*|$ on the attacked embeddings? Do they in fact satisfy Assumptions A2–A3 in practice, or are there many violations?

3. **How would you mount an explicit adaptive attack on KoALA?**  
   Suppose the attacker optimizes over $\delta_x$ with a loss encouraging misclassification *and* agreement between $\hat y_{KL}$ and $\hat y_{L_0}$. Have you tried such an attack, and if so, what are the detection/accuracy numbers? If not, what makes you confident that the geometric separation conditions in Theorem 1 are sufficient to withstand such adaptive strategies in realistic networks?

4. **Why not compare to Mahalanobis- or LID-based detectors?**  
   On CIFAR-10, it would be straightforward to include Mahalanobis (Lee et al., 2018), LID (Ma et al., 2018), or feature-squeezing (Xu et al., 2018) as baselines. Is there a technical or resource-based reason this was omitted? Please clarify, and if possible, add at least one such baseline.

5. **Clarification on Table 3–4 metrics: does “PGD attack (%)” include detected samples?**  
   When you report, e.g., 54.60% PGD accuracy for KL+L0 on ResNet (Table 3), is this the accuracy on *all adversarially perturbed inputs* (including ones flagged as attacks), or only on “successful, undetected attacks”? How does this relate to the confusion matrix definitions in Section 4.2?

6. **Sensitivity to $\tau$ and $\phi$ in the $L_0$ surrogate**  
   How sensitive are the results in Table 2 and Table 3 to the choice of $\tau=0.75$ and $\phi=0.5$? Have you explored other values or adaptive thresholding schemes? Given that $\tau$ is also central in the theory (Proposition 4, Theorem 1), some empirical robustness analysis w.r.t. $\tau$ would be informative.

7. **Scalability to larger class counts**  
   Have you tried KoALA on datasets with many more classes than CIFAR-10 / Tiny-ImageNet (e.g., full ImageNet)? If not, can you comment on the expected computational overhead of computing both KL and $L_0$ distances to all prototypes and whether approximate nearest neighbor methods or class subset pruning might be needed?

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The high-level idea and math are nontrivial and mostly correct, but key assumptions (feature-space threat model, prototype gaps, bounded coordinate perturbations) are not convincingly tied to real input-space attacks, and there is no evaluation under adaptive, detector-aware adversaries.

## Presentation Rating

2: fair.  
The central concept is clearly presented and figures/tables (**Figures 1–2**, Tables 1–5) are informative, but the theoretical exposition is extremely dense with inconsistent notation, and important experimental design details (Theorem-compliant criteria, metric definitions in tables) are under-specified in the main text.

## Contribution Rating

2: fair.  
The work offers an interesting geometric viewpoint and an analytically rich study of dual-metric prototype heads, with some promising empirical behavior on ResNet/CIFAR-10, but the absence of comparisons to existing detectors and the impracticality of the formal conditions reduce the impact.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The idea of leveraging disagreement between KL and a sparsity-sensitive metric in a nearest-prototype head is conceptually appealing and theoretically explored in depth, and the ResNet/CIFAR-10 experiments (including Table 3’s robustness gains and Table 2’s ablations) suggest some practical benefit. However, the theoretical guarantees are tied to feature-space perturbations with strong assumptions that are not quantitatively connected to real image-space attacks, there is no comparison to standard adversarial detectors or adaptive attacker baselines, and the CLIP results weaken the universality of the KL+L0 objective. With more empirical baselines, adaptive-attack evaluations, and a clearer, more actionable formulation of Theorem 1, this could reach ICLR standard; in its current form I lean to a (slightly) negative recommendation.

## Reviewer Confidence

4: confident.  
I am comfortable with adversarial robustness and representation learning, have read the math and experiments carefully (though not mechanically re-deriving every line), and I am confident in the main points of this assessment, while acknowledging that some technical details in the lengthy proof could be subtle.