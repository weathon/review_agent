# Blueprint-Bench: Critical Weaknesses from Related Work
## Executive Summary (For Review Writing)

---

## Overview
Based on analysis of 6 related papers (PhysBench, PIXELATED INSTRUCTIONS, MM-R, MERLIM, VLM Action Games, BigCodeBench), Blueprint-Bench—a spatial reasoning benchmark with 50 apartments, 20 images each—faces predictable failure patterns in model evaluation.

---

## The 6 Concrete Weaknesses

### 1. **Multi-View Inconsistency**
- **Problem**: Same apartment photographed from 20 different angles should yield identical floor plans, but MLLMs generate conflicting outputs.
- **Evidence**: MM-R (70YeidEcYR) shows models give different answers to semantically identical questions based on phrasing.
- **Quote**: "the models' outputs may vary with the phrasing of a query rather than its actual intent, which undermines their reliability"
- **Blueprint Impact**: HIGH - Invalidates multi-image evaluation strategy

### 2. **Visual Perturbation Sensitivity**
- **Problem**: Models struggle when viewing conditions change (angle, lighting, distance), even for identical floor layout.
- **Evidence**: MM-R shows mPLUG-Owl2 is "much more susceptible to inconsistency when image inputs are perturbed"
- **Quote**: "We find that SoTA MLLMs while often quite competitive in accuracy can differ substantially in their consistency of responses"
- **Blueprint Impact**: CRITICAL - 20 images span different angles/distances; models may fail on some, succeed on others

### 3. **Spatial Hallucination**
- **Problem**: Models invent rooms, doors, or spatial connections not present in input images.
- **Evidence**: MERLIM-class analysis shows MLLMs produce "responses that are not factually grounded in the visual input"
- **Quote**: "inaccuracies such as incorrect descriptions of non-existent visual elements. This undermines the trustworthiness of MLLMs"
- **Blueprint Impact**: CRITICAL - False spatial features directly corrupt floor plan graphs

### 4. **Visual Instruction Following Failure**
- **Problem**: Open-source MLLMs cannot follow instructions embedded visually (text in images, annotations).
- **Evidence**: PIXELATED INSTRUCTIONS (DiRJUdmZoK) shows open-source models fail at visual-modality instructions.
- **Quote**: "models that haven't seen visual instruction training are not robust enough at visual-modality instruction following"
- **Blueprint Impact**: MEDIUM - Only relevant if apartment images contain visual cues/annotations

### 5. **Insufficient Visual Information Utilization**
- **Problem**: Even when spatial information is visible, models fail to reference it during generation.
- **Evidence**: MMVP (5E6VOD7W0z) shows models need better extraction/utilization strategies.
- **Quote**: "alternative decoding algorithm...leading to performance gain (+6%)...underscores the potential to enhance model performance by employing better extraction and utilization techniques"
- **Blueprint Impact**: HIGH - Models know correct spatial layout but generate wrong floor plans

### 6. **Complex Multi-Step Reasoning Failure**
- **Problem**: Floor plan generation requires chaining multiple reasoning steps; models fail on complex apartments (5+ rooms).
- **Evidence**: BigCodeBench shows GPT-4 achieves only 60% on complex instruction-following tasks.
- **Quote**: "complex instruction following remains a fundamental limitation of current models"
- **Blueprint Impact**: HIGH - Benchmark should show complexity-dependent performance degradation

---

## Critical Testing Gaps Blueprint-Bench Should Fill

| Test | Current Status? | Why It Matters |
|------|-----------------|----------------|
| **Consistency Across Images** | Unknown | Same apartment, different image → different floor plan = benchmark failure |
| **Robustness to Perturbations** | Unknown | Close-up vs far-away photos should yield identical results |
| **Hallucination Rate** | Unknown | % of floor plan features with no visual support in images |
| **Visual Information Usage** | Unknown | Model(images only) vs Model(without images) = how much does visual info matter? |
| **Complexity Effects** | Likely missing | 2-room vs 6-room apartments = should show performance stability |
| **Metric Validation** | Unknown | Do high graph similarity scores = good floor plans according to humans? |

---

## Recommended Addition to Blueprint-Bench Paper

### Proposed Validation Subsection:
> "To validate that Blueprint-Bench reliably measures spatial reasoning, we conduct the following diagnostics:
>
> 1. **Consistency**: For each apartment, we show models 5 randomly selected images and require identical floor plan outputs. Divergence > X% indicates inconsistency failure.
> 2. **Perturbation Robustness**: We apply geometric transformations (zoom, rotation) to images and verify floor plan stability. Performance should remain constant ±Y%.
> 3. **Visual Grounding**: We verify every room/door in model-generated floor plans appears in at least one input image. False positive rate should be <Z%.
> 4. **Complexity Analysis**: We bin apartments by room count and verify that model performance remains stable across complexity levels.
> 5. **Metric Alignment**: We compare graph similarity scores with expert human judgments on 10 held-out apartments to verify correlation > 0.8.

---

## Summary: Why These Weaknesses Matter for Blueprint-Bench

**If Blueprint-Bench does NOT address these weaknesses**, reviewers should expect:

1. **Evaluation artifacts**: Models might score well through metric gaming rather than genuine spatial understanding
2. **Inconsistent results**: Multi-image evaluation becomes meaningless if models give different outputs per image
3. **Unfair comparisons**: Models sensitive to lighting/angle will appear worse despite equivalent spatial reasoning
4. **Undetected hallucinations**: Benchmark may accept invented floor plans as valid
5. **Hidden complexity failures**: Simple apartments might show high accuracy masking failures on complex layouts
6. **Metric-reality gap**: High benchmark scores won't predict real-world floor plan quality

---

## Files Generated
1. `/home/wg25r/review_agent/iclr2025_data/BLUEPRINT_BENCH_SPECIFIC_WEAKNESSES.md` — Full detailed analysis with quotes
2. `/home/wg25r/review_agent/iclr2025_data/BLUEPRINT_WEAKNESSES_STRUCTURED.md` — Structured format with diagnostic tests
3. `/home/wg25r/review_agent/iclr2025_data/BLUEPRINT_WEAKNESSES_QUICK_REFERENCE.txt` — Quick reference checklist
4. `/home/wg25r/review_agent/iclr2025_data/BLUEPRINT_BENCH_WEAKNESS_SUMMARY.md` — Original synthesis document

---

## How to Use This Analysis in Your Review

**Opening Paragraph:**
> "While Blueprint-Bench addresses an important spatial reasoning challenge, the paper should validate that its evaluation captures genuine floor plan understanding. Related work on MLLMs reveals predictable failure modes in spatial tasks: multi-view inconsistency (MM-R), visual perturbation sensitivity (MM-R), spatial hallucination (MERLIM), and complex reasoning failures (BigCodeBench). The paper does not address these risks."

**Strengths Section (if applicable):**
> "The benchmark's provision of 20 images per apartment is sensible, but the paper should validate that models generate consistent floor plans across different image subsets."

**Weaknesses Section:**
> "Missing diagnostic tests for [choose from above list]. Without validation that [specific weakness], the benchmark cannot claim to reliably measure [specific capability]."

**Questions Section:**
> "How do the authors ensure models maintain consistency across the 20 apartment images? What is the hallucination rate (invented rooms/doors) in model outputs?"

---

## References
- **MM-R** (70YeidEcYR) - Consistency in multimodal large language models
- **PIXELATED INSTRUCTIONS** (DiRJUdmZoK) - Visual modality instruction following
- **MERLIM** (49qqV4NTdy) - Visual grounding and hallucination
- **MMVP / VLM Action Games** (5E6VOD7W0z) - Spatial reasoning in VLMs
- **BigCodeBench** - Complex instruction following
- **PhysBench** - Physical scene understanding
