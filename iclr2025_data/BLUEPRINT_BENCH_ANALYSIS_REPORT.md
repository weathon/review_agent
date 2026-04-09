# Blueprint-Bench Relevant Weakness Analysis Report

**Analysis Date:** April 2026
**Dataset:** ICLR 2025 Human Reviews (368 reviews analyzed)
**Target Application:** Benchmark weaknesses applicable to Blueprint-Bench paper

## Executive Summary

This analysis identified **9 most relevant human reviews** containing **10 key weakness patterns** that are directly applicable to the Blueprint-Bench paper. The Blueprint-Bench paper evaluates LLMs, image generation models, and agents on converting apartment photographs into floor plans - a task combining spatial reasoning, visual understanding, and generative modeling.

The identified weaknesses cluster around:
1. **Dataset scale and evaluation scope** (most critical)
2. **Evaluation metrics appropriateness**
3. **Baseline and comparison comprehensiveness**
4. **Statistical rigor**
5. **Real-world applicability**

---

## Key Weakness Patterns Identified

### 1. **Limited Dataset Size Inadequate for Reliable Evaluation**
**Severity: CRITICAL**
**Relevance to Blueprint-Bench:** DIRECT

Papers with 50-100 samples for benchmarking face the same challenges Blueprint-Bench may encounter with limited spatial reasoning examples.

**Representative Quotes:**
- "Generating only 16 videos seems too few... to demonstrate that the proposed method works generally" (qxRoo7ULCo.md)
- "Ablation studies are performed on only 32 videos... results may have low statistical significance" (DyyLUUVXJ5.md)
- "Only 4 different images are used to showcase the results" (IqGVIU4rvM.md)

**Implications for Blueprint-Bench:**
- With ~50 floor plan samples, statistical significance of performance differences becomes questionable
- Ablation studies may lack power to detect meaningful improvements
- Generalization claims to unseen apartment types need explicit validation

---

### 2. **Metrics Inappropriate for Task or Dataset Characteristics**
**Severity: CRITICAL**
**Relevance to Blueprint-Bench:** DIRECT

Evaluation metrics may not capture the full complexity of spatial reasoning tasks.

**Representative Quotes:**
- "FID and KID metrics used on small reference set may be meaningless" (qxRoo7ULCo.md)
- "CLIP fine-tuning is not entirely clear... how does the generalization work" (FqWtMGw8tt.md)

**Implications for Blueprint-Bench:**
- Standard metrics (SSIM, pixel accuracy) may not reflect whether generated floor plans preserve spatial relationships
- Metrics should be evaluated for alignment with human judgment of floor plan quality
- Need validation that metrics capture architectural correctness, not just visual similarity

---

### 3. **Insufficient Baseline Comparisons and Existing Work Comparison**
**Severity: HIGH**
**Relevance to Blueprint-Bench:** DIRECT

Comprehensive baselines essential for credibility of contributions.

**Representative Quotes:**
- "Only one baseline (Im2Vec) was used for experiments... This would have made the evaluations thorough" (RBp0x7rkMO.md)
- "Missing comparisons with relevant baselines like Efficient4D and 4DGen" (qxRoo7ULCo.md)

**Implications for Blueprint-Bench:**
- Compare against simple baselines (e.g., CNN encoder-decoder)
- Include comparisons with both vision-only and language+vision approaches
- Benchmark against human-level performance where available

---

### 4. **Small Sample Size Leads to Unreliable Statistical Conclusions**
**Severity: HIGH**
**Relevance to Blueprint-Bench:** DIRECT

Sample sizes must be sufficient for statistical power.

**Representative Quotes:**
- "Evaluation on only 20 scenes per dataset... too small for statistical significance" (IXOoltTofP.md)
- "With only six participants, it is difficult to ascertain the statistical significance" (CpQegoH1Fn.md)
- "Visually inspecting only 100 examples about 0.002% of the data is a poor quality metric" (70YeidEcYR.md)

**Implications for Blueprint-Bench:**
- Report confidence intervals and statistical tests, not just means
- Conduct error analysis on representative subsets
- Cross-validate findings across different apartment types and layouts

---

### 5. **Evaluation Lacks Quantitative Rigor**
**Severity: HIGH**
**Relevance to Blueprint-Bench:** DIRECT

Missing error bars, confidence intervals, and statistical tests.

**Representative Quotes:**
- "It is incorrect to report just the mean... Standard deviation (std) values should be included" (DyyLUUVXJ5.md)
- "For the ablation study, I cannot find any quantitative comparisons" (RBp0x7rkMO.md)

**Implications for Blueprint-Bench:**
- Report all metrics with standard deviations and confidence intervals
- Include statistical significance tests for performance comparisons
- Provide per-sample analysis showing which floor plans are harder/easier

---

### 6. **Unfair Baseline Comparisons - Different Training Conditions**
**Severity: HIGH**
**Relevance to Blueprint-Bench:** DIRECT

Experimental fairness is essential for valid comparisons.

**Representative Quotes:**
- "The Adapt. row has been trained for the specific tasks... while the Ori. models are operating in a zero-shot manner" (70YeidEcYR.md)
- "Claims zero-shot but uses augmented training data" (FqWtMGw8tt.md)

**Implications for Blueprint-Bench:**
- Ensure all baselines use the same training data and optimization procedures
- Control for model size and capacity differences
- Document exact training procedures for reproducibility

---

### 7. **Limited Evaluation Scope**
**Severity: MEDIUM-HIGH**
**Relevance to Blueprint-Bench:** DIRECT

Single model/dataset/domain limits generalization claims.

**Representative Quotes:**
- "Testing on single pre-trained MLLM model (GPT-4o) on a very limited set of scenes" (IXOoltTofP.md)
- "Experiments on only one medical dataset" (CpQegoH1Fn.md)
- "Focus on relatively small datasets limits scalability" (RBp0x7rkMO.md)

**Implications for Blueprint-Bench:**
- Test on diverse apartment styles (modern, vintage, industrial, etc.)
- Include different room layouts and complexities
- Evaluate across different geographic regions if data allows

---

### 8. **Qualitative Results Inconsistent with Quantitative Claims**
**Severity: MEDIUM-HIGH**
**Relevance to Blueprint-Bench:** DIRECT

Visual quality should align with metrics.

**Representative Quotes:**
- "AdaCache achieves better VBench results... However, qualitative comparison shows a different result" (DyyLUUVXJ5.md)
- "Although AdaCache achieves higher acceleration speedup... results are significantly worse" (tccML2tDd4.md)

**Implications for Blueprint-Bench:**
- Include qualitative visual analysis of generated floor plans
- Show failure cases and how they relate to metric scores
- Include human evaluation to validate quantitative metrics

---

### 9. **Ground Truth or Annotation Quality Concerns**
**Severity: MEDIUM**
**Relevance to Blueprint-Bench:** DIRECT

Annotation quality critical when ground truth is hand-created.

**Representative Quotes:**
- "The subjects were laypersons rather than experts... their personal opinion is highly questionable" (CpQegoH1Fn.md)
- "Lack of IRB approval discussion and recruitment process description" (CpQegoH1Fn.md)

**Implications for Blueprint-Bench:**
- Document how ground truth floor plans were created (expert vs. crowdsourced)
- Report inter-rater agreement if multiple annotators
- Discuss any systematic biases in annotations

---

### 10. **Practicality of Benchmark Tasks Questioned for Real-World Applications**
**Severity: MEDIUM**
**Relevance to Blueprint-Bench:** DIRECT

Benchmark should map to realistic use cases.

**Representative Quotes:**
- "I can not imagine any real-world condition where a captured image is so distorted" (70YeidEcYR.md)
- "There is limited experimentation with real robots... how well models generalize to real-world sensor data" (sahQq2sH5x.md)

**Implications for Blueprint-Bench:**
- Validate that apartment photo collection matches real-world conditions
- Show example use cases (e.g., interior design, real estate, architecture)
- Document failure modes and when the benchmark might not apply

---

## Blueprint-Bench Specific Recommendations

### High Priority (Address to Avoid Major Criticism)

1. **Increase dataset diversity and document composition**
   - Show breakdowns by apartment type, layout complexity, room count
   - Ensure representative coverage of different architectural styles

2. **Establish quantitative evaluation framework**
   - Justify choice of metrics (why these metrics for spatial reasoning?)
   - Include multiple evaluation dimensions (structure, geometry, rooms, connections)
   - Report confidence intervals and statistical tests

3. **Provide comprehensive baseline comparisons**
   - Include vision-only, language-only, and multimodal baselines
   - Compare different agent architectures if applicable
   - Report results from multiple independent runs

4. **Document ground truth creation process**
   - Describe how floor plans were annotated
   - Report inter-rater agreement if applicable
   - Show example annotations and quality control measures

### Medium Priority (Strengthen Claims)

5. **Include qualitative analysis**
   - Show successful floor plan generations
   - Analyze failure cases and their characteristics
   - Include human evaluation studies

6. **Validate metrics against human judgment**
   - Show correlation between automatic metrics and human ratings
   - Identify cases where metrics misalign with human preferences

7. **Assess generalization**
   - Test on held-out apartment types
   - Evaluate transfer across different photo conditions
   - Report performance breakdown by apartment characteristics

### Lower Priority (Best Practices)

8. **Provide computational analysis**
   - Report inference time and memory requirements
   - Compare computational costs across methods

9. **Make results reproducible**
   - Provide code and dataset splits
   - Document all hyperparameters
   - Make model checkpoints available if possible

---

## Conclusion

The analysis of ICLR 2025 human reviews reveals consistent patterns of criticism around:
- **Evaluation methodology rigor** (most emphasized)
- **Dataset adequacy and diversity**
- **Fairness of comparisons**
- **Statistical validity**

Blueprint-Bench should address these concerns proactively to withstand peer review. The most critical issues are:
1. Dataset size and coverage documentation
2. Metric justification and validation
3. Comprehensive baseline comparisons
4. Statistical rigor in reporting results

By addressing these weakness patterns, the paper can significantly strengthen its contributions and increase likelihood of acceptance.

---

## References

This report analyzed the following review files:
- `qxRoo7ULCo.md` - 4K4DGen evaluation
- `IqGVIU4rvM.md` - Vision tokenizer evaluation
- `sahQq2sH5x.md` - MetaUrban benchmark
- `RBp0x7rkMO.md` - GRIMOIRE graphics generation
- `FqWtMGw8tt.md` - KnowData augmentation
- `70YeidEcYR.md` - MM-R3 robustness benchmark
- `IXOoltTofP.md` - 3DAxisPrompt spatial reasoning
- `CpQegoH1Fn.md` - rRBF network evaluation
- `DyyLUUVXJ5.md` - AdaCache diffusion acceleration
- `tccML2tDd4.md` - Object detection in foggy conditions

