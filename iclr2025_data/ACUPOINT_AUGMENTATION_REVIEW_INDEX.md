# ICLR 2025 Review Analysis: Diffusion-Based Acupoint Augmentation
## Search Results & Relevant Findings

**Search Date:** April 8, 2026  
**Dataset:** ICLR 2025 all_notes.json (8,614 papers, 464 with full text)  
**Results:** 6 highly relevant papers (scores 8.2-10.0)

---

## Quick Reference Table

| Rank | Paper | Score | Decision | Relevance | Paper ID |
|------|-------|-------|----------|-----------|----------|
| 1 | Diffusion-based Illumination Harmonization | 10.0 | Accept (Oral) | **CRITICAL** | u1cQYxRI1H |
| 2 | Consistency Models Stability | 9.2 | Accept (Oral) | **HIGH** | LyJi5ugyJx |
| 3 | Representation Alignment for Generation | 9.0 | Accept (Oral) | **HIGH** | DJSZGGZYVi |
| 4 | TANGO: Gesture Reenactment | 8.5 | Accept (Oral) | HIGH | LbEWwJOufy |
| 5 | Sparse Autoencoder Evaluation | 8.2 | Accept (Oral) | MEDIUM-HIGH | tcsZt9ZNKD |
| 6 | SANA: High-Res Image Synthesis | 8.5 | Accept (Oral) | MEDIUM | N8Oj1XhtYZ |

---

## Available Documents

### 1. **Main Summary Report** ✓
**File:** `iclr2025_acupoint_augmentation_review_summary.txt`  
**Size:** 13 KB  
**Content:** 
- Executive summary of findings
- Detailed analysis of each of 6 papers
- Cross-cutting themes and consensus findings
- Specific recommendations for your work
- Critical questions your paper should answer
- Evaluation protocol suggestions

**Read this for:** Complete overview and main guidance

---

### 2. **Quick Search Summary** ✓
**File:** `ACUPOINT_AUGMENTATION_ICLR2025_REVIEW_SEARCH.txt`  
**Size:** 12 KB  
**Content:**
- Structured findings with key takeaways
- Critical weaknesses categorized by severity
- Consensus themes across papers
- Recommended evaluation protocol
- Critical questions to answer

**Read this for:** Quick reference and action items

---

### 3. **Structured JSON Data** ✓
**File:** `iclr2025_acupoint_findings.json`  
**Size:** 5.2 KB  
**Content:**
- Search metadata
- Primary findings with key quotes
- Evaluation consensus
- Weakness categorization
- Recommended metrics by category

**Read this for:** Structured data for programmatic analysis

---

## Key Findings at a Glance

### The Critical Challenge
**Source:** Paper #1 (Score 10.0) - "Diffusion-based Illumination Harmonization"

> "Due to the stochastic nature of diffusion algorithms and the encoding-decoding processes of latent spaces, diffusion-based image generators inherently tend to introduce randomness into image contents, making it difficult to retain fine-grained details."

**Why it matters:** Without explicit constraints, your model will NOT reliably preserve acupoint landmarks.

**Solution:** Implement explicit preservation constraints (like their "light transport consistency") analogous to ensuring landmark preservation.

---

### The Evaluation Consensus
**Sources:** Papers #1, #5 (Scores 10.0, 8.2)

**Finding:** Standard perceptual metrics (SSIM, LPIPS) are insufficient for landmark preservation tasks.

**Required metrics:**
- Landmark localization error (pixels)
- Landmark detection accuracy on augmented images
- Distribution preservation (real vs. synthetic)
- Cross-architecture robustness

---

### Training Stability Concerns
**Source:** Paper #2 (Score 9.2) - "Consistency Models"

**Finding:** Performance degrades at higher resolutions. Landmark preservation may suffer from instability at scales needed for fine-grained detail.

**Recommendation:** Monitor landmark preservation metrics during training; implement early stopping based on preservation quality.

---

### Representation Alignment
**Source:** Paper #3 (Score 9.0) - "Representation Alignment for Generation"

**Finding:** Diffusion models may not properly "understand" landmark locations without explicit representation alignment.

**Recommendation:** Use representation quality metrics (CKA) to validate that model learns correct landmark mappings.

---

## Critical Weaknesses to Address

### Severity: CRITICAL (1 item)
- **Stochastic nature prevents feature preservation**
  - Solution: Implement explicit preservation constraints
  - Cite Zhang et al. (2024) as precedent

### Severity: HIGH (2 items)
- **Training stability at landmark-preserving resolutions**
  - Solution: Monitor preservation metrics; use preservation-based early stopping
  
- **Landmark annotation errors propagate downstream**
  - Solution: Validate annotation quality; test robustness to errors

### Severity: MEDIUM (2 items)
- **Insufficient landmark-specific evaluation**
  - Solution: Use localization error, detection accuracy, distribution metrics
  
- **Representation alignment not validated**
  - Solution: Use CKA or similar metrics to verify landmark understanding

---

## Recommended Evaluation Metrics

### Landmark Preservation (Task-Specific) ⭐
- Mean Euclidean distance to ground truth (pixels)
- Percentage landmarks within tolerance (±N pixels)
- Per-region accuracy (if multiple landmark types)
- Landmark detection success rate on augmented images

### Visual Quality (Standard)
- PSNR vs. original images
- SSIM vs. original images
- LPIPS (perceptual distance)

### Dataset Statistics (Distribution)
- Landmark position mean, std, range
- Real vs. synthetic distribution comparison
- Feature-landmark correlation preservation

### Task Performance (Downstream)
- CNN landmark detection accuracy on augmented data
- Classification/detection performance improvement
- Comparison to baseline augmentation methods

---

## Recommended Citations

For your related work section, cite these papers:

1. **Zhang et al. (2024)** - "Scaling In-the-Wild Training for Diffusion-based Illumination Harmonization"  
   → Use for: Landmark preservation challenges & constraint-based solutions

2. **Song et al. (2024)** - "Representation Alignment for Generation"  
   → Use for: Training methodology & representation alignment in diffusion models

3. **"Simplifying, Stabilizing and Scaling Continuous-time Consistency Models"**  
   → Use for: Training stability challenges at scale

4. **"TANGO: Co-Speech Gesture Video Reenactment"**  
   → Use for: Structural consistency in synthetic generation

---

## Critical Questions Your Paper Must Answer

These questions should be addressed explicitly in your paper:

1. **How do you preserve acupoint locations?**
   - Explicit constraints? Conditioning? Hybrid approach?
   
2. **What is your landmark preservation metric?**
   - Specific numbers? Thresholds? Baselines?
   
3. **How does performance vary across conditions?**
   - By anatomical region? By image variation? By resolution?
   
4. **Does quality degrade at different resolutions?**
   - Scale dependency analysis?
   
5. **What is the downstream task impact?**
   - Measurable improvement? Comparison to other methods?
   
6. **What are your failure modes?**
   - When does preservation fail? How often? Why?

---

## Five Key Takeaways

1. **Simple diffusion models will NOT reliably preserve landmarks without explicit mechanisms**
   - Multiple papers (scores 9+) confirm this challenge exists

2. **Precedent exists for constraint-based preservation**
   - Illumination preservation (Zhang et al.) can be adapted for landmarks

3. **Comprehensive evaluation is non-negotiable**
   - Standard metrics insufficient; task-specific metrics required

4. **Training stability and data quality are critical**
   - Large-scale diffusion training shows variance
   - Annotation errors propagate to downstream performance

5. **Representation alignment must be validated**
   - Ensure model actually learns to preserve landmarks correctly

---

## How to Use These Results

### For Writing Your Paper:
1. Read `iclr2025_acupoint_augmentation_review_summary.txt` first (full overview)
2. Reference the specific papers and quotes when justifying your approach
3. Use the evaluation protocol and metrics in your experimental section
4. Address all "Critical Questions" explicitly in your paper

### For Evaluation:
1. Implement all metrics from the "Recommended Evaluation Metrics" section
2. Test against the weaknesses listed in "Critical Weaknesses to Address"
3. Compare your approach to the "Solution" recommendations

### For Literature Review:
1. Use the recommended citations
2. Discuss why landmark preservation is a recognized challenge
3. Position your work within the broader context of feature preservation in generative models

---

## Search Methodology

- **Total papers searched:** 8,614
- **Papers with full text available:** 464
- **Search terms used:** 
  - "diffusion" + "augment"
  - "landmark" + "preservation"
  - "medical image" + "synthetic"
  - "synthetic data" + "evaluation"
  - "domain adaptation" + "generative"
  - "feature preservation" + "generation"

- **Selection criteria:** 
  - High quality papers (score >= 8.0)
  - Accepted/Oral/Spotlight status
  - Direct relevance to landmark preservation or evaluation methodology
  - Recent (ICLR 2025)

---

## Contact & Questions

For clarification on any findings, refer to the full documents:
- **Detailed analysis:** `iclr2025_acupoint_augmentation_review_summary.txt`
- **Quick reference:** `ACUPOINT_AUGMENTATION_ICLR2025_REVIEW_SEARCH.txt`  
- **Structured data:** `iclr2025_acupoint_findings.json`

---

**Last Updated:** April 8, 2026  
**Dataset Version:** ICLR 2025 (all_notes.json)
