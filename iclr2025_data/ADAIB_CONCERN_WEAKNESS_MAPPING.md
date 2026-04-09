# AdaIB Weakness Patterns from Related Papers

## Overview
This document maps specific weaknesses identified in ICLR 2025 reviews to key concerns for the AdaIB paper (Adaptive Information Bottleneck for Multimodal Attribution).

---


## OVERFITTING TO NOISY PAIRS

**Concern**: Overfitting to noisy or mismatched image-text pairs

**Related Topics**: noisy_misaligned_data, label_noise

**Papers Addressing This**: 

### Identified Weaknesses


## ALIGNMENT ASSUMPTION

**Concern**: Assuming reliable cross-modal alignment without verification

**Related Topics**: vision_language_robustness, noisy_misaligned_data

**Papers Addressing This**: Multimodal Instruction Tuning

### Identified Weaknesses


- **From**: Multimodal Instruction Tuning (Vision Language Robustness)
  
  > language alignment training for the adapter, then instruction tuning for the LLM backbone. Train-short-infrence-long is not a new technique that is used for input length extrapolation in LLMs [1]. The hybrid model structure is from Jamba. To sum, I don’t think the novelty is enough for an ICLR paper. 2. It’s better to show the inference resolution 


## COMPRESSION FITTING TRADEOFF

**Concern**: Trade-off between compression and fitting terms in information bottleneck

**Related Topics**: information_bottleneck, label_noise

**Papers Addressing This**: Multimodal Instruction Tuning

### Identified Weaknesses


- **From**: Multimodal Instruction Tuning (Vision Language Robustness)
  
  > language alignment training for the adapter, then instruction tuning for the LLM backbone. Train-short-infrence-long is not a new technique that is used for input length extrapolation in LLMs [1]. The hybrid model structure is from Jamba. To sum, I don’t think the novelty is enough for an ICLR paper. 2. It’s better to show the inference resolution 


## OPEN WORLD ROBUSTNESS

**Concern**: Robustness in open-world settings with unseen distributions

**Related Topics**: vision_language_robustness, label_noise

**Papers Addressing This**: 

### Identified Weaknesses


## THEORETICAL JUSTIFICATION

**Concern**: Lack of theoretical analysis and justification

**Related Topics**: information_bottleneck

**Papers Addressing This**: 

### Identified Weaknesses


---

## Summary of Patterns

Based on the comprehensive review analysis, the following patterns emerge for AdaIB:

1. **Robustness Concerns**: Papers consistently raise concerns about robustness to distribution shifts and unseen data. AdaIB should emphasize adaptive mechanisms that handle diverse image-text distributions.

2. **Alignment & Reliability**: Multiple reviews question assumptions about data alignment and reliability. AdaIB's adaptive weighting can address this by down-weighting unreliable pairs.

3. **Hyperparameter Sensitivity**: The information bottleneck trade-off introduces hyperparameter tuning challenges. Reviews show this is a common concern across related work.

4. **Theoretical Gaps**: Several papers lack theoretical justification for their approaches. AdaIB should provide theoretical analysis of the adaptive mechanism.

5. **Generalization**: Cross-dataset and cross-domain generalization is frequently questioned. AdaIB should demonstrate robustness across diverse multimodal datasets.

