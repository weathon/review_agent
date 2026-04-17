# Review

## Summary
This paper introduces Quantized Zeroth-order Optimization (QZO), a method for memory-efficient training of large language models (LLMs) by combining zeroth-order optimization with model quantization. The key innovation is perturbing the continuous quantization scale instead of the discrete weights to estimate gradients, allowing QZO to avoid de-quantization and re-quantization. The paper also proposes a directional derivative clipping method to stabilize training. The authors demonstrate that QZO can reduce memory usage by more than 18× compared to 16-bit fine-tuning, enabling the fine-tuning of 4-bit LLMs within a single 24GB GPU.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper is well-written and easy to follow.
2. The idea of perturbing the continuous quantization scale instead of the discrete weights is novel and makes sense.
3. The proposed method is orthogonal to both scalar-based and codebook-based post-training quantization methods.

## Weaknesses
1. The experiments are only conducted on 4-bit and 2-bit quantized models, and more experiments on different bit-widths should be included.
2. The performance of QZO lags behind full-parameter fine-tuning and MeZO on some tasks, indicating that the accuracy of QZO's gradient estimation is still limited.
3. The paper lacks a detailed analysis of the computational overhead and convergence behavior of QZO compared to other methods.

## Questions
1. How does the computational overhead of QZO compare to other memory-efficient training methods?
2. Could the authors provide more insights into the limitations of QZO's gradient estimation and how they affect the learning process?
3. How does QZO's convergence behavior compare to other methods, particularly in deeper and more complex networks?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4