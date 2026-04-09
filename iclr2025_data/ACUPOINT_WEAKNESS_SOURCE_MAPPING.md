# Weakness Source Mapping: How Each Weakness Appears in Reference Papers

This document maps the 7 identified weaknesses to their manifestations in the three reference papers, showing how lessons from each apply to acupoint augmentation work.

---

## Weakness 1: Lack of Explicit Landmark Preservation Constraint

### Where It Appears

| Paper | Where | How It Manifests |
|-------|-------|-----------------|
| **IC-Light** | Section 1 (Introduction) | Core motivation: "preserving underlying image details and intrinsic properties, such as albedo...requires careful design of training objectives and constraints" |
| **IC-Light** | Section 3.2 (Method) | Explicit solution: IC-Light constraint that "ensures the model modifies only the illumination aspects of an image while preserving other intrinsic properties" |
| **IC-Light** | Section 4.2 (Ablation) | Proof by contradiction: "removing light transport consistency...significantly decreased" fine-grained properties |

### The Lesson

IC-Light explicitly shows that implicit constraints fail: a vanilla diffusion model without light transport constraints produces "random behaviors...color mismatch, incorrect details." The IC-Light constraint mathematically enforces one specific property (light transport consistency) to prevent unwanted deviations.

### Application to Acupoint Paper

Acupoint detection requires preserving anatomical landmark positions, analogous to how IC-Light must preserve albedo. The paper should:
1. Define a landmark preservation constraint mathematically (e.g., optical flow consistency, spatial anchor loss)
2. Add this constraint to the diffusion training objective (not just post-hoc filtering)
3. Include ablation showing that removing this constraint degrades landmark accuracy
4. Report before/after landmark position statistics

### Key Quote
> "Without appropriate constraints, the training may produce a structure-guided random image generator, resulting in outputs that do not align with the desired illumination editing requirements... This necessitates careful design of training objectives and constraints to guide the learning process effectively." (IC-Light, Section 1, p. 2)

---

## Weakness 2: Inadequate Quality Filtering for Anatomical Accuracy

### Where It Appears

| Paper | Where | How It Manifests |
|-------|-------|-----------------|
| **KnowData** | Section 3.3 (Quality Filtering) | Identifies CLIP score filtering as insufficient: "our primary goal...is not to perform precise quality ranking, but rather to eliminate obviously mismatched samples" |
| **KnowData** | Section F (Appendix) | Documents two failure modes that CLIP catches: (1) text refinement errors, (2) global domain mismatch |
| **KnowData** | Section F (Appendix) | Shows failure examples: EuroSAT synthetic images "did not resemble actual satellite images" |

### The Lesson

KnowData explicitly acknowledges CLIP score filtering's limitation: it catches **global** failures (class mismatch) but not **local** failures (domain-specific quality issues). For satellite imagery, it misses subtle domain violations. For medical imaging, it would similarly miss subtle anatomical errors.

### Application to Acupoint Paper

CLIP-based filtering alone cannot detect:
- Acupoint position shifts (1-2 pixels undetectable to global CLIP score)
- Subtle anatomical distortions (e.g., asymmetric facial landmarks)
- Anatomically implausible configurations (landmarks outside valid ranges)

The paper should implement multi-stage filtering:
1. **Stage 1 (Global):** CLIP score filtering for basic quality
2. **Stage 2 (Anatomical):** Verify landmarks stay within anatomically valid bounds
3. **Stage 3 (Consistency):** Check inter-landmark distances remain constant
4. **Stage 4 (Human):** Manual review of high-risk samples

### Key Quotes

> "It is worth noting that our primary goal in using CLIP scores is not to perform precise quality ranking, but rather to eliminate obviously mismatched samples or failed generations for the targeted class." (KnowData, Section 3.3, p. 4)

> "We identify two major failure patterns in low-quality images before filtering: (1) Inadequate Text Refinement...and (2) Failed Synthetic Image Generation: Due to the randomness of diffusion model generation, synthetic images sometimes fail to meet the specific dataset requirements." (KnowData, Section F, p. 17)

---

## Weakness 3: Uncontrolled Data Distribution Diversity

### Where It Appears

| Paper | Where | How It Manifests |
|-------|-------|-----------------|
| **IC-Light** | Section 1 (Introduction) | Problem statement: "learning the large-scale, complicated, and noisy data is challenging. Without well-suited...constraints, the model can easily degrade to random behaviors" |
| **IC-Light** | Section 3.1 (Data) | Solution: Explicit data unification into "common format for neural network training" with multiple sources (light stages, 3D renders, in-the-wild) |
| **IC-Light** | Section 4.4 & 4.5 | Demonstrates robustness across diverse lighting conditions through explicit data control |

### The Lesson

IC-Light achieves robustness through **explicit data distribution management**, not just large-scale training. They:
1. Mix multiple data types (real light stages, synthetic 3D, in-the-wild augmentations)
2. Unify into common format with standardized components (appearance, albedo, normal, light, background)
3. Use scheduled probability to balance data sources during training
4. Validate generalization across diverse illumination distributions (rim lighting, backlighting, artistic effects)

### Application to Acupoint Paper

Without explicit distribution control, Stable Diffusion may generate:
- Limited facial pose diversity (clustering toward frontal faces)
- Biased ethnicities or facial structures (mode collapse)
- Out-of-distribution acupoint positions that don't match real clinical data

The paper should:
1. Explicitly control synthesis parameters (pose range, ethnicity distribution, expression variety)
2. Report distribution analysis comparing synthetic vs. original data
3. Measure whether augmented landmarks cluster in feature space regions absent from real data
4. Validate that augmentation helps equally for diverse face types

### Key Quotes

> "Our goal is to learn a robust and generalized model to handle in-the-wild illumination patterns. Nevertheless, learning the large-scale, complicated, and noisy data is challenging. Without well-suited regularization and constraints, the model can easily degrade to random behaviors, e.g., color mismatch, incorrect details, etc." (IC-Light, Section 3.2, p. 5)

> "This method allows us to achieve a maximized setup: expanding the dataset to over 10 million images, adopting stronger backbones like SDXL and Flux, and utilizing all available types of data sources, including real photos captured from light stages, rendered images, and in-the-wild natural or artistic images with synthetic illumination augmentations." (IC-Light, Section 1, p. 2)

---

## Weakness 4: Missing Per-Landmark Accuracy Metrics

### Where It Appears

| Paper | Where | How It Manifests |
|-------|-------|-----------------|
| **IC-Light** | Section 4.4 (Quantitative Evaluation) | Reports multiple metrics: PSNR, SSIM, LPIPS addressing different properties |
| **IC-Light** | Section 4.2 (Ablation Study) | Per-property ablation: "removing light transport consistency...red and blue differences vanished in some images, and noticeable issues with color saturation are observed" |
| **IC-Light** | Section 4.5 (Visual Comparison) | Per-component analysis: "normal maps produced by this model exhibit higher quality for human than alternatives" |

### The Lesson

IC-Light doesn't rely on single global metric. Instead, they:
1. Use multiple metrics targeting different properties (PSNR for reconstruction, SSIM for structure, LPIPS for perception)
2. Provide ablation results showing impact on specific properties (albedo, shadow, color saturation)
3. Analyze per-component performance (quality of different elements in the image)
4. Conduct both quantitative and visual analysis

### Application to Acupoint Paper

Standard classification metrics (accuracy, precision, F1-score) hide whether specific acupoints are preserved. For example:
- 95% accuracy could mask systematic mislocalization of acupoint LI-10
- F1-score could hide that 2 out of 14 acupoints consistently fail

The paper should report:
1. **Per-acupoint metrics:** Localization error, F1-score for each individual acupoint
2. **Anatomical consistency:** Whether inter-landmark distances remain constant
3. **Error distribution:** Which acupoints are most vulnerable to augmentation?
4. **Landmark detection confidence:** Does augmentation change model confidence in predictions?
5. **Ablation by acupoint:** Performance degradation for each acupoint when component removed

### Key Quote

> "We conducted quantitative comparisons using metrics such as Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS)... The full method, which combines multiple data sources and enforces light transport consistency, produces a well-balanced model capable of generalizing across a range of scenarios. It also retains fine-grained image details and intrinsic properties, such as albedo, while reducing errors in output images." (IC-Light, Section 4.4, p. 8-9)

---

## Weakness 5: Unvalidated Component Interactions

### Where It Appears

| Paper | Where | How It Manifests |
|-------|-------|-----------------|
| **KnowData** | Section 2 (Related Work) | Notes that existing methods "fail to effectively incorporate the relevant knowledge of the class itself into the generation process" |
| **KnowData** | Section 3.1-3.4 (Method) | Describes four separate components: structured knowledge extraction, LLM expansion, unstructured knowledge integration, text refinement |
| **KnowData** | Table 1 & Section 4 (Ablation) | Systematically ablates each knowledge source to show individual contribution |
| **IC-Light** | Section 4.2 (Ablation) | Tests removing light transport consistency alone to isolate its effect |

### The Lesson

Both papers validate component interactions through systematic ablation:
- KnowData progressively adds knowledge sources and measures impact on image quality
- IC-Light removes single constraint and measures degradation across multiple properties
- Neither paper assumes component interactions are benign without evidence

### Application to Acupoint Paper

The paper combines three components: (1) Stable Diffusion 1.5, (2) IP-Adapter, (3) IC-Light. Without ablation, it's unclear:
- Does IP-Adapter interfere with IC-Light's constraints?
- Does IC-Light's illumination adjustment sometimes distort face geometry and thus landmarks?
- Which component is most important for landmark preservation?

The paper should:
1. Report baseline performance: SD1.5 alone
2. Add IP-Adapter and measure landmark preservation change
3. Add IC-Light and measure change
4. Test alternative component combinations
5. Measure performance delta at each stage

### Key Quote

> "Existing studies employ various generation tricks for diffusion models to enhance the diversity and quality of generated images and then finetune the CLIP model, but they fail to effectively incorporate the relevant knowledge of the class itself into the generation process." (KnowData, Section 2, p. 2)

---

## Weakness 6: No Data Ratio Optimization

### Where It Appears

| Paper | Where | How It Manifests |
|-------|-------|-----------------|
| **CFDG** | Section 1 (Introduction) | Identified problem: "studying the relationship between offline and online data and determining how to augment these two types of data are crucial problems" |
| **CFDG** | Section 6 (Conclusion) | Explicit limitation: "the ratio of offline to online data can significantly impact performance in different environments. Additionally, determining the optimal ratio for the three types of data, including the generated data, remains an open challenge." |
| **CFDG** | Section 4 (Experiments) | Reports fixed 1:1 ratio but acknowledges it "can significantly impact performance in different environments" |

### The Lesson

CFDG explicitly shows that data augmentation ratio is:
1. **Task-dependent:** "can significantly impact performance in different environments"
2. **Environment-dependent:** Different tasks need different ratios
3. **Unsolved:** "remains an open challenge" (acknowledged limitation of the work)

### Application to Acupoint Paper

The paper likely uses a fixed synthetic:real ratio without justification (e.g., 1:1 or 1:2). But optimal ratio depends on:
- Available real data quantity
- Model capacity
- Acupoint detection difficulty (rare vs. common landmarks)
- Clinical setting specificity

The paper should:
1. Experiment with multiple ratios (1:1, 1:2, 1:5, 1:10 synthetic:real)
2. Plot performance as function of ratio
3. Identify optimal ratio for their specific benchmark
4. Acknowledge ratio is likely dataset-dependent
5. Provide guidance for practitioners on ratio selection

### Key Quotes

> "The main task of O2O RL is to improve sample efficiency and performance... online data is often limited in the traditional O2O RL setting." (CFDG, Section 1, p. 1)

> "Although our method achieved the expected performance, there are still some limitations. One of these is the setting of the data ratio... However, according to some of our tests, the ratio of offline to online data can significantly impact performance in different environments. Additionally, determining the optimal ratio for the three types of data, including the generated data, remains an open challenge." (CFDG, Section 6, p. 10)

---

## Weakness 7: No Realistic Clinical Validation

### Where It Appears

| Paper | Where | How It Manifests |
|-------|-------|-----------------|
| **IC-Light** | Section 1 (Introduction) | Scope: "diffusion-based image generators...hold unique industrial value for visual content creation and manipulation" |
| **IC-Light** | Section 4.3 (Applications) | Beyond-benchmark applications: normal maps, background-conditioned illumination, artistic effects |
| **IC-Light** | Section 4.5 (Visual Comparison) | Cross-domain testing: "when compared to alternative models trained on smaller or more structured datasets, our approach generalizes to a wider variety of illumination distributions" |

### The Lesson

IC-Light validates across multiple distributions and applications, not just a single benchmark. They test:
1. Different data sources (light stages, 3D renders, in-the-wild)
2. Different lighting conditions (rim lighting, backlighting, artistic effects)
3. Different model backbones (SD1.5, SDXL, Flux)
4. Different applications (illumination editing, normal map generation, background harmonization)

### Application to Acupoint Paper

Academic benchmarks (fixed pose, frontal lighting, high-quality images) differ from clinical reality where:
- Lighting conditions vary (clinical exam rooms, field clinics)
- Facial poses are less controlled (patient comfort varies)
- Image quality varies (different camera equipment, operators)
- Skin conditions vary (scars, acne, sensitivity)
- Practitioner backgrounds vary (different experience levels, training)

The paper should:
1. Test on multiple clinical datasets (ideally from different centers)
2. Validate with practitioners (not just algorithmic metrics)
3. Analyze performance on underrepresented groups (skin tones, face shapes, ages)
4. Report failure modes and confidence intervals
5. Compare to baseline (how much improvement is clinically meaningful?)

### Key Quote

> "We provide pretrained illumination editing models to facilitate illumination editing applications in content creation and manipulation across diverse domains... We showcase the method's ability to handle more in-the-wild illumination scenarios, including artistic and composed lighting effects." (IC-Light, Section 1, p. 2)

---

## Summary: How Lessons Transfer

| Weakness | Source Paper | Core Lesson | Acupoint Application |
|----------|--------------|-----------|-------------------|
| 1. Explicit constraints | IC-Light | Implicit ≠ Explicit: Need mathematical constraint | Add geometric landmark loss to diffusion objective |
| 2. Quality filtering | KnowData | Single metric insufficient for domain-specific tasks | Multi-stage filtering with anatomical validation |
| 3. Distribution control | IC-Light | Explicit data management beats implicit scaling | Control and analyze facial variation distribution |
| 4. Per-component metrics | IC-Light | Don't use single global metric; analyze per-property | Report per-acupoint, not just overall accuracy |
| 5. Component ablation | KnowData + IC-Light | Validate interactions systematically | Test each augmentation component separately |
| 6. Ratio optimization | CFDG | Optimal data mix is task-dependent and open problem | Experiment with synthetic:real ratios |
| 7. Clinical validation | IC-Light | Benchmark success ≠ real-world utility | Test across clinical settings and practitioners |

---

## Using This Mapping

### For Authors Responding to Reviews:
Use this document to understand **why** each weakness matters by seeing how it manifests in published work. Show reviewers you understand the underlying principles by referencing source papers.

### For Reviewers:
This mapping provides specific paper citations and page numbers for each concern. Use these to strengthen review comments with published evidence.

### For Method Development:
The lesson column shows the underlying principle behind each weakness. Use these principles to guide method design beyond just addressing specific weaknesses.

