---
job_id: d8a1fc67-dcc4-4684-9d62-b5c8f503f909
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: VgVeQpagf7.pdf
paper: High Performance Differentially Private Fine-Tuning Using Dataset Distillation
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is squarely about differentially private learning, dataset distillation, and representation learning for vision models, which fits ICLR’s core topics (privacy, optimization, generative modeling, transfer/fine-tuning).

## Minimum Quality
Pass ✅.  
The paper is written in English, has all key sections (Abstract, Introduction, Background/Related Work folded into Section 2, Method, Experiments/Results, Discussion/Conclusion), presents a nontrivial algorithmic contribution with explicit equations and pseudocode, and provides substantial empirical evidence on standard benchmarks (CIFAR-10/100, CAMELYON17, Tiny-ImageNet). I do not see any obviously fatal theoretical or experimental flaws that would justify immediate desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not detect any hidden prompts, instructions to reviewers/LLMs, or other manipulative artifacts in the provided main paper content.

---

# Expected Review Outcome:

## Summary

The paper introduces SPS and SPS+, differentially private dataset distillation algorithms that summarize intermediate activations of a public pretrained network on the private data, privatize these statistics via the Gaussian mechanism, and then synthesize a reusable synthetic dataset by optimizing images to match the privatized statistics. SPS+ further introduces multistage clipping and grouped pseudo-classes to improve noise efficiency, especially for many-class problems. On CIFAR-10/100, SPS+ fine-tuning achieves or exceeds state-of-the-art DP-SGD accuracy across multiple privacy budgets, and the synthetic datasets enable ensembling, federated learning, and continual learning without further privacy cost.

## Strengths

1. **Substantive empirical advance over DP-SGD on canonical image benchmarks.**  
   - Table 1 (Page 7) is central: SPS+ with WRN34-10 ensembles reaches 96.2% / 76.6% accuracy on CIFAR-10/100 at ε=1, compared to 94.8% / 70.3% for the DP-SGD baseline of De et al. (2022). At higher ε, SPS+ remains at least competitive and often slightly better, especially when leveraging ensembles, while all DP guarantees are inherited from the single measurement step.  
   - Figure 2 (Page 7) further shows consistent gains from increasing the number of multistage clipping steps M across architectures and privacy budgets, especially in the strict privacy regime.

2. **Well-motivated shift from model-based DP to data-based DP with clear advantages.**  
   - The paper articulates a compelling motivation: once privatized statistics are released, arbitrary numbers of models can be trained, ensembling is trivial, and continual/federated settings are naturally supported by reusing or aggregating synthetic data (Sections 1, 5.4–5.6).  
   - Figure 5 (Page 9) visually supports this: panel (a) shows high accuracy with up to 10× compression, (b) similar for CIFAR-100, and panels (c–e) show federated and continual learning scenarios where SPS+ is competitive or better than DP-FL baselines under the same or stronger privacy constraints.

3. **Careful algorithmic design to make DD compatible with DP constraints.**  
   - Section 3.2 explains how the method avoids a private teacher model and instead uses a fixed public model θ_P, collecting class-conditional and global activation statistics at certain layers and privatizing them with a single Gaussian mechanism.  
   - Equation (2) defines a global + per-class KL objective that replaces the cross-entropy + soft-label machinery in D3S, which would be problematic when no private teacher is allowed.  
   - Section 3.2.2 and Equation (4) specify the vectorization, clipping, and noise-injection scheme, with tunable statistic dimensionality via D_G, D_C, and |L_C|. This is a neat way to decouple DP noise from full gradient dimensionality.

4. **SPS+ innovations (multistage clipping and grouped pseudo-classes) are empirically important and reasonably motivated.**  
   - Section 4 explains multistage clipping (MC) as iterative recentring/clipping of the statistic vector based on synthetic-data estimates and shows its privacy accounting via straightforward RDP composition (Theorem 4.1, Section 4.3).  
   - Grouped pseudo-classes (Section 4.2) reduce effective per-class noise by aggregating statistics over random groups of classes; Table 8 (Page 23) shows that adding GPC and MC dramatically boosts CIFAR-100 performance at low ε (e.g., ε=1: from 48.9% to 71.0%).  
   - Figure 2’s trends in M reinforce that these are not marginal tweaks; they materially close the gap to or surpass DP-SGD.

5. **Solid and fairly broad experimental validation.**  
   - Core benchmarks: CIFAR-10 and CIFAR-100 at ε ∈ {1,2,4,8}, δ=10⁻⁵, multiple architectures (WRN22/28/34) and ensembles.  
   - Out-of-domain: CAMELYON17 (Table 2) where SPS with ε=8 attains 92.6% accuracy, outperforming DP-Diffusion and DP-SGD run at ε≈10 despite domain shift between ImageNet pretraining and histopathology.  
   - Higher resolution: Tiny-ImageNet at 64×64 (Section G.1) with ε=8, getting 49.5% with WRN28-10 ensembles, suggesting scalability beyond 32×32.  
   - Extensive ablations:  
     * Table 4 and Table 5 (Pages 21–22) show the necessity of the SPS loss, class rescaling, smooth activations, and GSAM.  
     * Tables 6–7 (Page 22) quantify robustness to choices of D_C, D_G and L_C.  
     * Table 8 isolates the effect of multistage clipping and grouped pseudo-classes.  
     * Table 13 compares against many DP generation and DP-SGD baselines, and SPS+ clearly stands out on CIFAR-10 accuracy.

6. **Use of figures to support interpretability and trade-offs.**  
   - Figure 1 (Page 3) clearly visualizes the SPS pipeline: real data summarized through θ_P into statistics, then privatized, then used to synthesize a new dataset by minimizing KL divergence; this makes a complex algorithm much easier to follow.  
   - Figure 4 (Page 8) displays SPS+ distilled CIFAR-100 images under different ε values, with FIDs annotated. The progression from noisy, texture-like patterns at ε=1 to recognizable objects at ε=8 aligns nicely with the quantitative accuracy improvements in Table 1.  
   - Figure 3 (Page 8) connects FID to accuracy, showing a monotonic negative correlation across ε and M. This is a useful sanity check that the synthetic data are not just “FID good” but also classification-useful.

7. **Privacy accounting is straightforward and grounded in established tools.**  
   - Theorem 4.1 (Section 4.3) and the background in Appendix C derive the RDP cost of M Gaussian mechanisms composing over multistage clipping. The use of the standard RDP accountant and explicit b₀ values in Table 11 helps reproducibility and transparency.  
   - The whole DP argument reduces to bounding the ℓ₂ sensitivity of v via clipping, then composing Gaussian mechanisms, which is conceptually clean.

8. **Reproducibility and implementation detail are better than typical.**  
   - Appendices A and D give explicit pseudocode (Algorithms 1–8), hyperparameters (Tables 9–12), and runtime estimates (Section D.4, F.1).  
   - The authors report b₀ for each (ε, M) (Table 11), data augmentation details, optimizer settings, and batch sizes both for synthesis and validation.

## Weaknesses

1. **Some mathematical and algorithmic details are difficult to parse or have inconsistencies/typos.**  
   - Equation (4) uses noise variance `b_0^{il}` in the text (“$\mathcal{N}(0, b_0^{\mathrm{il}} \|v\|_{\max}^2 I)$”), which appears to be a typo given Theorem 4.1 depends on σ = b₀ ‖v‖\_{\max}. The relationship between σ, b₀, and ε should be consistently stated and clarified.  
   - Algorithm 2’s notation is at times garbled (e.g., “$m_{clj}$”, “$z_{clj}$”, repeated “$RDG$”) and partially conflicts with Equation (3) on Page 5. For a privacy-sensitive algorithm, the exact construction of vᵢ, including scaling by √S and class indicators, matters a lot.  
   - In Algorithm 3, several lines appear corrupted: e.g., `μc← C/N ∂c| {Additionally unscale by √S}`, and `Σc← C/N ∂c| + (uC) - μcμG`. Similar issues appear in Algorithms 6–8 for SPS+. While one can infer the intended operations from the narrative (Sections 3.2.2–3.2.4 and 4.1–4.2), the current pseudocode is not precise enough for a faithful reimplementation without the authors’ code.  
   - For a paper hinging on precise DP accounting, these inconsistencies are not cosmetic; they obscure exactly what is privatized and how class/group statistics are normalized and recombined.

2. **Theoretical analysis of utility is minimal; SPS+ heuristics lack formal justification.**  
   - The only substantive theorem is Theorem 4.1, which restates the RDP guarantee of an M-fold composition of Gaussian mechanisms. There is no analysis of how distortions in the privatized statistics (norm of noise, spectral perturbation of Σ) translate to excess risk or classification accuracy.  
   - Grouped pseudo-classes (Section 4.2, Appendix A.5) are argued to reduce “noise rate” from O(C/N) to O(C/(N N_{C/p})), but this conflates statistical variance with DP noise: the per-release DP noise is fixed by ε and sensitivity; grouping changes the conditioning of the KL objective and optimization landscape, not the privacy noise distribution itself. The text does acknowledge that privacy on original class statistics is not improved, but the discussion is still somewhat hand-wavy about when and why this trick works (e.g., invertibility of P, conditioning of Σ, effect of eigenvalue clipping).  
   - Multistage clipping (Section 4.1, Appendix A.6) borrows inspiration from DP mean-estimation work, but there is no adaptation of those theoretical bounds to this high-dimensional covariance estimation context. It would be valuable to at least provide a proposition bounding the bias/variance trade-off achieved by recentering and re-clipping for v in this specific construction.

3. **Scope of experimental validation, while strong, is still relatively narrow.**  
   - Most of the evaluation is on small-resolution image benchmarks: CIFAR-10/100 (32×32) with 10 and 100 classes, and Tiny-ImageNet (64×64, 200 classes). Given the paper’s ambition as an alternative to DP-SGD, results on a more realistic high-resolution dataset (e.g., full ImageNet-1K) or at least a clear argument why the current scheme scales (memory/sensitivity of high-dimensional covariances, cost of image optimization) would strengthen the claim.  
   - For SPS+, the only “many-class” experiment beyond CIFAR-100 is Tiny-ImageNet with a single ε=8 configuration (Section G.1). It would be illuminating to see ε=1–4 on Tiny-ImageNet to verify that GPC and MC truly scale in harsher privacy regimes.

4. **Fairness and completeness of baseline comparisons raise some questions.**  
   - Table 1 compares primarily to De et al. (2022) and Private Evolution (Lin et al., 2024). Other DP image classification works are only compared in Table 13 in the appendix, and some comparisons are indirect (different ε, different δ, no public pretraining, etc.). Given SPS+ depends heavily on public ImageNet pretraining, it would be fair to more systematically separate “public-pretrained” vs “from scratch” baselines and to be explicit wherever the baselines lack such pretraining (e.g., DP-RandP and some older DP-SGD setups).  
   - For federated learning (Section 5.5, Figure 5 panels (d–e) and Appendix E), FedDM is non-DP and FedLAP-DP uses multi-round communication, so their comparison to “one-shot” SPS+ is not apples-to-apples. The text does acknowledge this to some extent in Appendix E, but the main-text narrative might give an over-strong impression that SPS+ dominates existing FL methods under identical assumptions.

5. **DP assumptions regarding public data and pretraining are critical but somewhat under-discussed.**  
   - SPS and SPS+ inherently assume a powerful public pretrained model θ_P, trained on a disjoint non-sensitive corpus (ImageNet32/64 variants). The paper cites works (Mehta et al., 2022; De et al., 2022; Ganesh et al., 2023) justifying this assumption, but does not deeply explore sensitivity to mismatch.  
   - Section 5.2 partially addresses out-of-domain mismatch using CAMELYON17, but only with ε=8 and a single pretraining corpus; it would be valuable to see degraded pretraining (e.g., lower accuracy) or entirely unrelated public data to understand robustness of SPS to imperfect θ_P.  
   - In practice, the availability of high-quality public models often drives performance more than clever privatization; disentangling how much of SPS+’s gains over DP-SGD come from better use of θ_P vs. inherently better noise-efficiency would be insightful.

6. **Computational cost is high and partially limits practical advantage.**  
   - Section D.4 notes that generating 50k images takes 8–21 GPU-hours on an H100 per stage, and SPS+ with M>1 multiplies this by roughly \(1 + (M-1)/2\). Meanwhile, Section F.1 estimates that comparable DP-SGD training requires about 45 H100 hours. So SPS+ is “roughly comparable” but not clearly more efficient, especially if one wants multiple synthetic datasets or extensive hyperparameter search.  
   - The paper frames SPS as enabling reusability across many models and tasks, which is valid, but in many applications one still needs multiple synthetic datasets for tuning or ablations. A more quantitative cost–benefit analysis (e.g., cost per trained model under DP-SGD vs. cost per reusable SPS dataset amortized across K models) would clarify where SPS+ is truly advantageous.

7. **Clarity of SPS+ description is weaker than for SPS, making the enhanced variant harder to understand.**  
   - The core SPS loss and privatization (Sections 3.2.1–3.2.3) are reasonably clear, aided by Figure 1 and Equation (2). In contrast, SPS+’s multistage clipping and grouped pseudo-classes are largely pushed to Section 4 and Appendix A.5–A.6, where notation is inconsistent and some equations are corrupted (e.g., Equation (5) uses repeated badly formatted subscripts; Equation (6) on Page 20 has unmarked red-highlight commentary references that did not transfer cleanly).  
   - For instance, the construction and normalization of the pseudo-class matrix P, and how P and P⁻¹ are used to aggregate and recover statistics, are not fully clear from the main text. Since GPC is central to SPS+’s CIFAR-100 performance (Table 8), a cleaner, self-contained mathematical description in the main paper would significantly help readers.

8. **Privacy–utility trade-offs for federated and continual learning are only partially quantified.**  
   - Figure 5 panels (c–e) and Figures 7–9 (appendix) show encouraging performance for federated and class-incremental continual learning. However, the privacy accounting in these multi-source settings is under-specified in the main text.  
   - In the federated experiments, each client runs SPS+ with its own ε and δ, and the server aggregates synthetic datasets. While this satisfies each client’s local DP guarantee (no central raw-data access), the effective privacy for the union dataset or for a user whose data exists in multiple silos is not clearly discussed.  
   - Similarly, in continual learning (Section 5.6), each task’s data are privatized once; the claim that no further privacy cost is incurred when reusing synthetic data is correct, but the compounding of ε across tasks at the data level (each user may appear in multiple tasks) could be acknowledged more explicitly.

9. **Some experimental design details could be clarified.**  
   - For SPS+, hyperparameters like (P, N_{C/p}), stage counts M per ε, and clipping factors K_clip^m are given in Appendix D.2.1 and Table 9, but the rationale for these specific choices is largely empirical. It would be helpful to highlight which hyperparameters are most sensitive in practice, based on Figure 6/7 and the ablations in Tables 6–7.  
   - For the Tiny-ImageNet experiment (Section G.1), more detail on synthetic dataset size, compression ratio, and whether grouped pseudo-classes are used would help interpret the 49.5% accuracy result.

## Potentially Missing Related Work

Below I list directly related works that appear not to be cited or discussed and should be integrated into the related work / comparison:

1. **Liu et al., “Model Conversion via Differentially Private Data-Free Distillation” (2023)**  
   - Relevance: Proposes converting pretrained models into privacy-preserving counterparts via data-free distillation, closely related to using synthetic data and public models for DP. Similar in spirit to SPS in leveraging a public backbone and data-free / synthetic approaches for private model training.  
   - Suggested integration: Discuss in Section 2.2 / 2.3 when surveying DP synthetic data and DP distillation methods, and clarify differences: SPS operates on privatized activation statistics and produces reusable datasets, whereas Liu et al. focus on model conversion.

2. **Zheng et al., “Improving Noise Efficiency in Privacy-preserving Dataset Distillation” (2025)**  
   - Relevance: Targets precisely the challenge addressed by SPS+, namely how to inject noise efficiently when distilling datasets under DP. Their techniques and assumptions should be compared to the multistage clipping and grouped pseudo-class strategies in Section 4.  
   - Suggested integration: Add to the DP dataset distillation paragraph in Section 2.3 and extended discussion in Appendix B.2, and if feasible, include at least a qualitative comparison of the mechanisms in Section 4.

3. **Shi et al., “DP-GENG: Differentially Private Dataset Distillation Guided by DP-Generated Data” (2025)**  
   - Relevance: Introduces a framework where DP-generated data guide distillation, sitting at the intersection of DP generative modeling and dataset distillation. This is conceptually close to SPS, which uses privatized statistics rather than full DP-generated images.  
   - Suggested integration: Discuss alongside DP-Diffusion and DP-KIP in Section 2.2–2.3 and in the appendix comparison (Section F), to position SPS relative to other hybrid generation–distillation approaches.

4. **Zheng & Li, “Differentially Private Dataset Condensation” (2026)**  
   - Relevance: Proposes DP algorithms explicitly for dataset condensation, which is essentially the same goal as SPS. Their mechanisms and empirical results should be contrasted with SPS/SPS+ on similar benchmarks if available.  
   - Suggested integration: Include in Section 2.3 as a direct alternative to SPS, and if numerical results are reported on CIFAR or similar, consider adding to Table 13 or a new comparison table.

5. **Flemings & Annavaram, “Differentially Private Knowledge Distillation via Synthetic Text Generation” (2024)**  
   - Relevance: Though focused on text, this is highly conceptually aligned with the idea of DP knowledge distillation via synthetic data, which the introduction already points to for language models.  
   - Suggested integration: Mention in the paragraph on text-domain DP synthetic data (currently citing Xie et al., 2024; Amin et al., 2024) to broaden the context and to clarify differences between text-focused and vision-focused methods.

6. **Khadem et al., “DP-OPD: Differentially Private On-Policy Distillation for Language Models” (2026)**  
   - Relevance: Another example of using DP distillation techniques (here on-policy) in the language domain. It underlines the growing literature on DP distillation that SPS is now extending to images.  
   - Suggested integration: Add to the related work discussion in Section 2.2–2.3 or the discussion section, contrasting off-policy synthetic data distillation (SPS) with on-policy distillation paradigms.

These additions would better situate SPS/SPS+ within the rapidly evolving landscape of DP dataset distillation and DP synthetic-data-based training.

## Questions

1. **Clarification of SPS+ pseudocode and exact operations.**  
   - Could you provide a cleaned-up version (possibly in the rebuttal as a figure) of Algorithms 3, 6, and 7 showing:  
     * The exact formulas for recomputing μ_l^c and Σ_l^c from the privatized v, including scaling by C/N, N_{C/p}, and √S,  
     * How the pseudo-class matrix P is constructed and used (forward and inverse),  
     * The precise noise covariance in Equation (4) for each component?  
   - This would increase confidence that the implementation matches the described math and that the DP proof applies to the actual code.

2. **Impact of public pretraining quality and domain mismatch.**  
   - How does SPS/SPS+ perform if θ_P has lower accuracy or is trained on a less related public dataset? For example, have you tried replacing ImageNet32 pretraining with a substantially weaker or different corpus and measuring the degradation in Table 1 / Figure 2 results?  
   - Any quantitative evidence about the sensitivity of SPS to θ_P quality (beyond the one CAMELYON17 experiment) would help practitioners gauge when SPS is viable.

3. **Scalability to higher resolutions and larger class sets.**  
   - For Tiny-ImageNet, could you report results at multiple ε values (e.g., ε=1,2,4) and compression ratios, similar to Figure 5?  
   - Additionally, what are the memory and runtime bottlenecks as a function of D_G, D_C, and L_C (e.g., per Table 10)? A small table showing d_tot and wall-clock time per configuration would help characterize scaling beyond 32×32.

4. **Fairness of federated learning baselines and privacy accounting.**  
   - In Figure 5(d–e), can you clarify:  
     * The exact ε, δ settings used for FedLAP-DP and FedDM,  
     * Whether the same public model and pretraining were available to these baselines,  
     * How you account for privacy across multiple communication rounds for FedLAP-DP?  
   - Also, how would you formalize user-level privacy guarantees when multiple silos may contain overlapping users? A brief discussion would help.

5. **Utility analysis perspective.**  
   - Have you empirically explored the relationship between the norm of the added noise in v (controlled by b₀ and K_clip) and the spectral properties of the resulting Σ (e.g., condition number, smallest eigenvalue after clipping), and how these relate to the KL loss and downstream accuracy?  
   - Even partial empirical plots (e.g., KL vs. ε, or eigenvalue distributions before/after clipping) could support the heuristic explanation of why multistage clipping and pseudo-class grouping are beneficial.

6. **Hyperparameter sensitivity and tuning strategy.**  
   - In practice, how many SPS/SPS+ runs on the *real* data were required to arrive at the hyperparameters in Tables 9–10?  
   - Could you describe a tuning protocol that keeps privacy overhead small, e.g., by doing most tuning on public data / surrogate datasets and only a small search on the private data?

Clear answers to these questions would likely increase my confidence in both the soundness and the practical impact of the work.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The work focuses on improving differentially private training for image classification with standard public and benchmark datasets; no specific additional ethical, fairness, or safety concerns beyond the usual DP context are apparent.

## Soundness Rating

3: good.  
The core DP mechanism (Gaussian with clipping, RDP accounting) is standard and correctly applied. The empirical evidence is extensive and convincing on several datasets. However, the utility analysis is largely empirical, and some SPS+ pseudocode/math inconsistencies reduce confidence in reproducibility and exact algorithmic understanding.

## Presentation Rating

3: good.  
The high-level narrative, figures (particularly Figures 1–5), and main equations (especially Equation (2)) are clear and informative. Yet, the appendices and SPS+ pseudocode contain several notation issues and minor corruptions that hinder full clarity of the enhanced algorithm.

## Contribution Rating

4: excellent.  
The paper is, to my knowledge, the first to demonstrate a generation-based DP method that matches or exceeds state-of-the-art DP-SGD on CIFAR-10/100, while enabling flexible downstream usage (ensembles, FL, continual learning). This is a substantial and timely contribution to the DP learning community.

## Overall Rating

8: Accept, good paper (poster).  
The work offers a meaningful advance in differentially private learning by turning dataset distillation into a high-performance, reusable DP synthetic dataset generator for image classification, surpassing or matching DP-SGD on key benchmarks and enabling new usage modes. There are some issues in mathematical clarity and limited theoretical utility guarantees, but the empirical results, design insights, and breadth of experiments make this paper well above the bar for acceptance.

## Reviewer Confidence

4: confident.  
I am familiar with differential privacy, DP-SGD, and dataset distillation literature; I carefully checked the main equations, algorithm descriptions, and tables. Some SPS+ implementation details remain opaque due to notation issues, which is why I stop short of “absolutely certain,” but the overall assessment feels well-founded.