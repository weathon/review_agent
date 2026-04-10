=== CALIBRATION EXAMPLE 24 ===

# Final Consolidated Review
## Summary
This paper proposes DenseFace, a post-training method to mitigate demographic bias in face recognition models without retraining. The core idea is to model face embeddings via von Mises-Fisher distributions, estimate local inter-class embedding densities using a balanced anchor set, and perform density-aware probabilistic matching that adjusts similarity scores (lower in dense regions, higher in sparse regions). The method is evaluated on standard benchmarks (RFW, RB-WebFace) using a NIST-style protocol, showing consistent reduction in false-positive rate disparities across racial groups while preserving verification accuracy.

## Strengths
- **Novel density-aware probabilistic matching formulation:** The paper introduces a well-motivated, novel procedure that links the von Mises-Fisher concentration parameter (κ) estimated from a *balanced inter-class anchor set* to demographic bias. The local distortion with margin *m* (Eq. 6-7) is a clever solution to handle near-orthogonal embeddings, and the probabilistic matching (Eq. 9) provides a principled way to adjust scores based on local density.
- **Extensive empirical evaluation across varied models and datasets:** The method is validated on multiple strong face recognition models (AdaFace, CosFace) with different architectures (ResNet-50/100) trained on diverse large-scale datasets (MS1MV2, Glint360K, WebFace4M/12M). Results show consistent bias reduction according to the proposed NIST-style metric (FPR at a fixed threshold) without degrading overall verification accuracy (Table 4).
- **Critical adoption of improved bias metrics:** The paper cogently argues for moving beyond standard deviation of accuracy (RFW protocol) and adopts a more rigorous NIST-aligned evaluation (FPR at a threshold fixed by the Caucasian group's performance), which better reflects real-world deployment constraints where per-group thresholds are impractical.

## Weaknesses
### Major:
- **Missing comparisons to state-of-the-art post-processing bias mitigation methods:** The paper claims DenseFace advances post-training calibration, but all experiments (Tables 2-5) compare only against the cosine similarity baseline of the same model. There are no head-to-head results against relevant prior post-processing methods such as PASS (Dhar et al., 2021), score normalization (Linghu et al., 2024), Fair Score Normalization (Terhörst et al., 2020a), or von Mises-Fisher loss projection (Conti et al., 2022). This evidential gap severely undermines the claim of superiority or meaningful advance over existing techniques.
- **Dependence on a carefully constructed, demographically balanced anchor set:** The method's efficacy hinges on a balanced anchor set (54k identities from Glint360K, balanced by race and gender using attribute classifiers). This requires access to a large, labeled dataset and reliable attribute classifiers, which may not be available in practice. The paper provides limited analysis of how anchor set imbalance or classifier errors might degrade performance, making the method's robustness and practicality uncertain.

### Minor:
- **Limited analysis of impact on correlated attributes and potential "bias leakage":** The qualitative analysis (Figure 7) suggests nearest neighbors from the anchor set often share attributes like hairstyle and expression. This raises the concern that density adjustment could inadvertently reinforce or amplify biases tied to other correlated attributes (e.g., age, accessories). The paper does not examine the method's effect on bias across multiple intersecting attributes or other protected classes.
- **Computational and memory overhead for the anchor-set version:** While a learning-based variant reduces inference cost, the core method requires storing a large anchor set (54k embeddings) and performing nearest-neighbor searches for each test sample. The probabilistic matching (Eq. 9) is also 1.5× slower than cosine similarity even after optimizations (Sec. 4.8). This overhead may limit deployment in latency-sensitive applications, and a comparison of total system cost against other post-hoc methods is lacking.

### Trivial:
- **Lack of integration with per-subgroup threshold calibration:** The evaluation uses a fixed threshold (Caucasian FPR=0.001). Combining DenseFace with optimal threshold calibration per subgroup is a logical next step but not a flaw in the current work.

## Nice-to-Haves
- **Ablation studies on anchor set parameters:** Systematic analysis of the impact of anchor set size (*K*), degree of demographic balance, and source dataset would clarify the method's robustness and requirements.
- **Deeper investigation into the density-bias correlation mechanism:** While correlation is shown (Fig. 3, Table 2), further analysis could explore *why* inter-class density differs demographically and whether adjusting it directly addresses root causes or correlated symptoms.
- **Evaluation on additional benchmarks with diverse covariates:** Testing on datasets like IJB-C or MegaFace, which include variations in age, pose, and occlusion, would strengthen claims of general robustness.

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Strength:** "The paper is well-written and the topic is important." → *Removed as generic.*
- **Weakness:** "The introduction overstates novelty by not acknowledging prior post-training methods." → *Removed as factually wrong; Section 2.3 explicitly discusses Terhörst et al., Dhar et al., Conti et al., and Linghu et al.*
- **Weakness:** "The margin *m* is heuristic and unjustified." → *Removed as an overstatement; the paper introduces *m* to handle near-orthogonal embeddings (Sec. 3.2, Eqs. 6-7, Fig. 4) and shows it improves density estimation, though sensitivity analysis is lacking.*
- **Weakness:** "Reproducibility concerns due to undisclosed hyperparameters." → *Removed per rule against nitpicks on trivial implementation details; Sec. 4.2 describes key hyperparameters (K=128, anchor set construction).*
- **Weakness:** "The method requires demographic labels for the anchor set, which may be impractical." → *Partially kept in Major weaknesses, but removed as a standalone point to avoid duplication.*

## Suggestions
- **Add direct comparisons to post-processing baselines:** Include experiments comparing DenseFace against state-of-the-art post-hoc bias mitigation methods (e.g., PASS, score normalization, Fair Score Normalization) on the same models and metrics to substantiate claims of advancement.
- **Analyze the impact on correlated attributes and intersectional bias:** Extend the evaluation using attribute-labeled datasets (e.g., CelebA) to examine whether DenseFace affects bias related to age, expression, or accessories, and assess performance across intersections (e.g., Black females).
- **Provide an ablation study on anchor set sensitivity:** Systematically vary the anchor set's size, balance, and source to demonstrate the method's robustness and inform practical deployment requirements.

# Actual Human Scores
Individual reviewer scores: [2.0, 6.0, 2.0]
Average score: 3.3
Binary outcome: Reject
