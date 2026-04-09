# AdaIB-Relevant Weaknesses from ICLR 2025 Reviews

## Search Scope

- **Topics Searched**: 5
  1. Multimodal attribution/interpretability methods
  2. Vision-language models and CLIP robustness
  3. Information bottleneck approaches
  4. Handling noisy or misaligned image-text data
  5. Robustness to label noise

- **Total Reviews Processed**: 368
- **Relevant Reviews Found**: 75
- **Papers with Detailed Weaknesses Extracted**: 5

---

## KnowData: Knowledge-Enabled Data Generation

**Paper ID**: `FqWtMGw8tt`  
**Topic**: Noisy Misaligned Data  
**Relevance to AdaIB**: Review provides insights into noisy misaligned data challenges that AdaIB addresses

### Extracted Weaknesses

1. Weaknesses - Improvements on ImageNet, although better than previous work, are rather marginal.

---

## Data Shapley in One Training Run

**Paper ID**: `HD6bWcj87Y`  
**Topic**: Label Noise  
**Relevance to AdaIB**: Review provides insights into label noise challenges that AdaIB addresses

### Extracted Weaknesses

1. sentence "The results of this experiment..." appears twice in a row

2. limitations which they claim will be a part of future work.

3. Weaknesses I do not see any major weaknesses of the paper.

4. it is unclear whether the method only outperforms leave-one-out (approximated by influence functions) or is also better than the standard data Shapley value.

---

## Multimodal Instruction Tuning

**Paper ID**: `cagNCwQEEN`  
**Topic**: Vision Language Robustness  
**Relevance to AdaIB**: Review provides insights into vision language robustness challenges that AdaIB addresses

### Extracted Weaknesses

1. language alignment training for the adapter, then instruction tuning for the LLM backbone. Train-short-infrence-long is not a new technique that is used for input length extrapolation in LLMs [1]. The hybrid model structure is from Jamba. To sum, I don’t think the novelty is enough for an ICLR paper. 2. It’s better to show the inference resolution 

2. concerned with the novelty of this paper.

---

## Adaptive Length Image Tokenization

**Paper ID**: `mb2ryuZ3wz`  
**Topic**: Vision Language Robustness  
**Relevance to AdaIB**: Review provides insights into vision language robustness challenges that AdaIB addresses

### Extracted Weaknesses

1. 100 (rather than full ImageNet-1K) for the adaptive tokenization training, making it difficult to assess the method's capabilities at scale fully. The authors acknowledge this limitation in Table 1's footnote, but don't sufficiently explore whether the performance gap with fixed tokenizers (particularly in FID scores for low token counts) is fundam

2. the paper lacks a formal framework for analyzing the optimality of these allocations.

3. limitation in current visual representation learning systems: their use of fixed-length representations regardless of image complexity.

4. Weaknesses The paper's primary weakness lies in its validation strategy and scaling limitations.

5. limitation in Table 1's footnote, but don't sufficiently explore whether the performance gap with fixed tokenizers (particularly in FID scores for low token counts) is fundamental to the approach or simply due to training scale.

6. It is unclear if the authors backpropagate gradients from future iterations all the way back or if they stop the gradient flow between each iteration.

7. It is unclear if it is enabled for all experiments or some subset.

---

