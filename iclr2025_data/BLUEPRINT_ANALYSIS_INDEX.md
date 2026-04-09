# Blueprint-Bench Weakness Analysis — Complete Index

## Quick Navigation

### For Reviewers (Start Here)
- **[BLUEPRINT_WEAKNESSES_EXECUTIVE_SUMMARY.md](BLUEPRINT_WEAKNESSES_EXECUTIVE_SUMMARY.md)** ⭐ START HERE
  - 1-page overview of 6 weaknesses
  - How to use in your review
  - Specific reviewer language to use

### For Detailed Analysis
- **[BLUEPRINT_WEAKNESSES_STRUCTURED.md](BLUEPRINT_WEAKNESSES_STRUCTURED.md)**
  - All 6 weaknesses with full descriptions
  - Failure modes and diagnostic tests
  - Summary table and validation recommendations

### For Quick Reference
- **[BLUEPRINT_WEAKNESSES_QUICK_REFERENCE.txt](BLUEPRINT_WEAKNESSES_QUICK_REFERENCE.txt)**
  - Bullet-point format for all weaknesses
  - Quick diagnostic checklist
  - Expected failure modes table

### For Deep Dive
- **[BLUEPRINT_BENCH_SPECIFIC_WEAKNESSES.md](BLUEPRINT_BENCH_SPECIFIC_WEAKNESSES.md)**
  - 7 weaknesses with extensive context
  - Multiple supporting papers per weakness
  - Detailed relevance to Blueprint-Bench

---

## The 6 Critical Weaknesses (Summary)

| # | Weakness | Source Paper | Severity | Review Relevance |
|---|----------|--------------|----------|------------------|
| 1 | Multi-View Inconsistency | MM-R (70YeidEcYR) | **CRITICAL** | Models give different floor plans for same apartment from different angles |
| 2 | Visual Perturbation Sensitivity | MM-R (70YeidEcYR) | **CRITICAL** | Close-up vs far-away photos produce different results |
| 3 | Spatial Hallucination | MERLIM (49qqV4NTdy) | **CRITICAL** | Models invent rooms/doors not in images |
| 4 | Visual Instruction Failure | PIXELATED (DiRJUdmZoK) | HIGH | Open-source models fail at visual text in images |
| 5 | Information Underutilization | MMVP (5E6VOD7W0z) | HIGH | Models don't reference spatial info during generation |
| 6 | Complex Reasoning Failure | BigCodeBench | HIGH | Models fail on apartments with 5+ rooms |

---

## How to Use This Analysis

### For Reviewers Writing a Review

**Step 1**: Read BLUEPRINT_WEAKNESSES_EXECUTIVE_SUMMARY.md (5 min)

**Step 2**: Identify which weaknesses Blueprint-Bench paper fails to address:
- Does it validate consistency across 20 images? ❌ Missing = add to review
- Does it test perturbation robustness? ❌ Missing = add to review
- Does it report hallucination rate? ❌ Missing = add to review
- Does it ablate visual inputs? ❌ Missing = add to review
- Does it analyze complexity effects? ❌ Missing = add to review
- Does it validate metric alignment? ❌ Missing = add to review

**Step 3**: Write your weaknesses section using this template:
```
"The paper should address the following diagnostic tests to validate that
Blueprint-Bench reliably measures spatial reasoning (informed by related work
on spatial understanding failures):

[List 2-3 of the 6 weaknesses that the paper doesn't address]

For example, [weakness 1] is not validated. The authors should [specific test].
[Supporting quote from this analysis].

Additionally, [weakness 2] remains unaddressed...
```

**Step 4**: (Optional) If paper is strong on most dimensions:
```
"The paper demonstrates rigor in [specific aspects], but should extend
validation to [2-3 weaknesses from the 6] following best practices in
related spatial reasoning benchmarks (MM-R, MERLIM, BigCodeBench)."
```

---

### For Benchmark Developers

If you're designing a spatial reasoning benchmark, validate against all 6:

1. **Multi-View Consistency**: Run Model(A, Image_i) and Model(A, Image_j) where both are of apartment A. Outputs should be identical.

2. **Perturbation Robustness**: Apply transformations (zoom, rotation, brightness) to images. Score distributions should remain stable.

3. **Hallucination Detection**: For each room/door in output, verify ≥1 supporting image. False positive rate should be <5%.

4. **Information Utilization**: Ablate visual inputs: Model(text only) vs Model(text+images). Performance should drop significantly.

5. **Complexity Calibration**: Bin apartments by room count. Performance should remain stable across bins (or report findings clearly).

6. **Metric Validation**: Correlate graph similarity scores with expert human judgment. Correlation should be >0.8.

---

## Paper Reference Information

### Source Papers Analyzed
- **MM-R** (70YeidEcYR.txt) - "Consistency Benchmark for Multimodal Large Language Models"
- **PIXELATED INSTRUCTIONS** (DiRJUdmZoK.txt) - "Visual Modality Instruction Following"
- **MERLIM** (49qqV4NTdy.txt) - "Visual Grounding and Hallucination Analysis"
- **MMVP / VLM Action Games** (5E6VOD7W0z.txt) - "Spatial Reasoning in Vision Language Models"
- **BigCodeBench** - "Complex Instruction Following Benchmark"
- **PhysBench** - "Physical Scene Understanding Benchmark"

### Full File Paths
- Executive Summary: `/home/wg25r/review_agent/iclr2025_data/BLUEPRINT_WEAKNESSES_EXECUTIVE_SUMMARY.md`
- Structured Analysis: `/home/wg25r/review_agent/iclr2025_data/BLUEPRINT_WEAKNESSES_STRUCTURED.md`
- Quick Reference: `/home/wg25r/review_agent/iclr2025_data/BLUEPRINT_WEAKNESSES_QUICK_REFERENCE.txt`
- Detailed Analysis: `/home/wg25r/review_agent/iclr2025_data/BLUEPRINT_BENCH_SPECIFIC_WEAKNESSES.md`
- Original Synthesis: `/home/wg25r/review_agent/iclr2025_data/BLUEPRINT_BENCH_WEAKNESS_SUMMARY.md`

---

## Key Quotes for Review Writing

### On Multi-View Inconsistency
> "the models' outputs may vary with the phrasing of a query rather than its actual intent, which undermines their reliability" — MM-R

### On Perturbation Sensitivity
> "We find that SoTA MLLMs while often quite competitive in accuracy can differ substantially in their consistency of responses" — MM-R

### On Hallucination
> "A notable problem is the tendency of MLLMs to produce responses that are not factually grounded in the visual input...This undermines the trustworthiness of MLLMs in many practical applications" — MERLIM-class

### On Visual Instruction Failure
> "models that haven't seen visual instruction training are not robust enough at visual-modality instruction following" — PIXELATED INSTRUCTIONS

### On Information Underutilization
> "This underscores the potential to enhance model performance by employing better extraction and utilization techniques with the same pretrained image encoder" — MMVP

### On Complex Reasoning
> "complex instruction following remains a fundamental limitation of current models" — BigCodeBench

---

## Reviewing Tips

1. **Most Important**: Check that the paper validates consistency across the 20 images per apartment. This is the foundation of the benchmark.

2. **Second Priority**: Verify that hallucination rates are reported (should be near 0%).

3. **Third Priority**: Look for complexity analysis showing performance remains stable as apartment complexity increases.

4. **If Paper is Strong**: Still recommend adding metric validation (correlation with human judgment).

5. **Recommended Strength to Highlight**: "Provides 20 images per apartment, enabling consistency validation" (if done).

6. **Major Weakness to Flag**: "No validation that graph similarity metrics correlate with expert spatial reasoning quality."

---

## FAQ

**Q: Should Blueprint-Bench address all 6 weaknesses?**
A: Ideally yes, but at minimum: consistency, hallucination rate, and metric validation.

**Q: Are these weaknesses specific to floor plan generation?**
A: No—they apply to any spatial reasoning benchmark. These are general MLLM failures that spatial tasks expose.

**Q: Can a strong paper on other dimensions compensate?**
A: Partially, but missing validation on spatial reasoning fundamentals (consistency, hallucination) is a major red flag for a spatial benchmark.

**Q: What if the paper doesn't mention these papers?**
A: That's okay—but the diagnostic tests should still be present. If absent, note "Missing validation that [weakness]."

---

## Document History

- Created: 2024-04-08
- Synthesis: Analysis of 6 related papers on spatial reasoning in MLLMs
- Intended Use: Review writing for Blueprint-Bench and similar spatial reasoning benchmarks
- Version: 1.0 (Complete)

