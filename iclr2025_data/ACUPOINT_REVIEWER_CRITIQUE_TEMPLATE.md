# Reviewer Critique Template: Acupoint Landmark Augmentation with Diffusion

## Weakness 1: Lack of Explicit Geometric Constraints for Landmark Preservation

**Critical Issue:**
The paper uses Stable Diffusion 1.5 with IP-Adapter for identity preservation and IC-Light for illumination consistency, but provides no evidence that these mechanisms preserve anatomical landmark positions. Unlike IC-Light which explicitly enforces light transport consistency through physics-based constraints, this work implicitly assumes that preserving "identity" automatically preserves fine-grained acupoint locations.

**Evidence from Literature:**
IC-Light paper demonstrates that "without well-suited regularization and constraints, the model can easily degrade to random behaviors" and emphasizes "careful design of training objectives and constraints" for preserving intrinsic properties. This is not merely theoretical—their ablation shows removing light transport consistency causes "noticeable issues with color saturation" and loss of fine details. By analogy, removing explicit landmark constraints could cause acupoint position drift.

**Specific Questions for Authors:**
- How do you ensure acupoint positions don't shift during diffusion?
- Have you measured Euclidean distance between landmarks in original vs. augmented images?
- Did you experiment with anchor-point loss functions or landmark detection guidance during generation?
- What is the maximum acceptable position error for clinical use, and do you measure against it?

---

## Weakness 2: CLIP-Score Filtering is Insufficient for Medical Applications

**Critical Issue:**
KnowData paper identifies two failure modes that CLIP filtering catches: (1) text refinement errors and (2) global image-class mismatch. However, these fail to catch medical-specific failures like:
- Acupoint subtle position distortions (1-2 pixel shifts undetectable by CLIP)
- Artifacts in anatomically sensitive regions (eyes, mouth, facial structure)
- Anatomically implausible landmark configurations (e.g., landmarks outside anatomical bounds)

KnowData explicitly states: "our primary goal...is not to perform precise quality ranking, but rather to eliminate obviously mismatched samples." This is inadequate for medical imaging where subtle errors propagate.

**Specific Questions for Authors:**
- What percentage of "CLIP-filtered" augmented images have acupoints within ±2 pixels of the original?
- Did you implement anatomical validity checks (e.g., convex hull constraints, inter-landmark distance consistency)?
- What is your false negative rate for detection of small acupoint displacements?
- Did human raters validate a sample of "passed" augmented images?

---

## Weakness 3: Uncontrolled Synthetic Data Distribution

**Critical Issue:**
The paper doesn't report the distribution of facial variations in augmented data. Without explicit control, Stable Diffusion may generate:
- Biased representation of skin tones or facial structures
- Limited pose diversity despite aiming for "diverse" augmentation
- Out-of-distribution acupoint positions that differ from real clinical data

IC-Light achieved robustness by "utilizing all available types of data sources" and "expanding the dataset to over 10 million diverse samples," but crucially, they unified data into a "common format for neural network training" with explicit quality controls. This work provides no equivalent distribution validation.

**Specific Questions for Authors:**
- What is the distribution of: face poses (frontal? profile? angled?), skin tones (Fitzpatrick scale), expressions?
- Did you measure whether synthetic landmarks cluster in regions of feature space absent from real data?
- How does augmented data distribution compare to the original training set?
- Do different acupoints benefit equally from augmentation, or do some cluster unnaturally?

---

## Weakness 4: Missing Per-Landmark Accuracy Metrics

**Critical Issue:**
Standard classification metrics (accuracy, F1-score) hide whether specific acupoints are preserved. A detector could achieve 95% accuracy while systematically mislocating LI-10 (Shousanli) across all augmented images. IC-Light reports landmark-relevant metrics (PSNR, SSIM, LPIPS) and ablation studies showing component impact on specific properties (albedo preservation, shadow consistency). This work should report per-acupoint metrics.

**Specific Questions for Authors:**
- Do you report per-acupoint localization error (L2 distance from ground truth)?
- Is detector performance uniform across all 14+ acupoints, or do some degrade significantly?
- Do anatomical invariants (e.g., relative distances between acupoints) remain constant in augmented data?
- Can you provide an error distribution showing which acupoints are most vulnerable to augmentation?

---

## Weakness 5: Unvalidated Component Interactions

**Critical Issue:**
The paper combines three components: (1) Stable Diffusion 1.5, (2) IP-Adapter for identity, (3) IC-Light for illumination. IC-Light itself is a recent method requiring physical constraints to avoid "random behaviors." No ablation study demonstrates that the three-component combination preserves landmarks, nor that the combined loss surface doesn't introduce new failure modes.

**Specific Questions for Authors:**
- Did you run ablations isolating each component's effect?
- What is the performance delta: SD1.5 → +IP-Adapter → +IC-Light?
- Do spatial artifacts increase in any configuration (e.g., does IC-Light's illumination adjustment sometimes distort face geometry)?
- How sensitive is landmark preservation to hyperparameters of each component?

---

## Weakness 6: No Principled Data Ratio Optimization

**Critical Issue:**
CFDG paper explicitly identifies as a limitation: "determining the optimal ratio for the three types of data...remains an open challenge." They found ratio "can significantly impact performance in different environments." Yet this paper likely either:
- Uses a fixed ratio without justification, or
- Doesn't report ratio sensitivity analysis

For medical applications where data scarcity motivated this work, the optimal synthetic:real ratio is critical.

**Specific Questions for Authors:**
- How much augmented data did you use? (synthetic:real ratio)
- Did you experiment with different ratios? If so, provide a curve.
- At what ratio does augmented data help most? Does it plateau or degrade?
- Is the optimal ratio stable across acupoint detection benchmarks, or does it vary?

---

## Weakness 7: Lack of Real Clinical Validation

**Critical Issue:**
The paper validates on academic benchmarks (likely constructed from annotations on clean images). Real clinical use requires validation that:
- Augmentation generalizes to images from different acquisition settings
- Practitioners actually benefit from the detector trained on augmented data
- Acupoint localization is clinically accurate enough for acupuncture practice

IC-Light demonstrates this with broad testing across "in-the-wild scenarios" including "artistic and composed lighting effects." This work should validate across clinical settings.

**Specific Questions for Authors:**
- Did you test on images from a different clinical center or acquisition protocol?
- Did practitioners (acupuncturists) validate the detector on real patients?
- How does performance degrade for underrepresented groups (skin tones, face shapes)?
- Is augmentation most beneficial for any specific subpopulation?
- What is the false negative rate (missed acupoints) in real clinical conditions?

---

## Overall Assessment of Soundness

The work addresses a real problem (acupoint detection data scarcity) with a reasonable approach (diffusion-based augmentation). However, it makes strong implicit assumptions:

1. **IP-Adapter preserves fine-grained anatomy** (unvalidated)
2. **CLIP filtering ensures medical quality** (known to be insufficient)
3. **Implicit constraints equal explicit constraints** (contradicted by IC-Light)
4. **Academic benchmarks predict clinical utility** (unvalidated)

The strongest related work (IC-Light) proves that assumption #3 is false: they show that "without well-suited...constraints, the model...degrade[s] to random behaviors" and "removing light transport consistency...significantly decreased" key properties. By analogy, removing explicit landmark constraints likely causes landmark drift.

---

## Recommended Revisions (Priority Order)

**REQUIRED (address before acceptance):**
1. Add per-landmark localization error metrics and per-acupoint F1 scores
2. Implement multi-stage quality filtering with landmark detector validation
3. Conduct ablation studies on each component (SD1.5 → +IP-Adapter → +IC-Light)
4. Provide visual evidence of landmark preservation (before/after position plots)

**STRONGLY RECOMMENDED:**
5. Analyze synthetic data distribution; compare to original training set
6. Experiment with data ratio optimization (try 1:1, 1:2, 1:5 synthetic:real)
7. Test generalization to images from different clinical settings

**RECOMMENDED FOR IMPACT:**
8. Implement explicit landmark preservation constraint (geometric loss or anchor-point masking)
9. Validate with clinical practitioners on real patient images
10. Analyze per-acupoint performance to identify vulnerable landmarks

---

## Mapping to Published Work

| Paper | Key Lesson | Application to Acupoint Work |
|-------|-----------|---------------------------|
| **IC-Light** | Implicit constraints insufficient; explicit physical constraints needed | Add geometric/anatomical losses |
| **KnowData** | CLIP filtering catches global but not local failures; multi-stage filtering needed | Implement landmark-specific validation |
| **CFDG** | Optimal data ratios vary by task and must be empirically determined | Run ratio sensitivity analysis |

