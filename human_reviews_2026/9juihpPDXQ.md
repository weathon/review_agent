# Differential Gated Self-Attention

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Transformers excel across a large variety of tasks but remain susceptible to corrupted inputs, since standard self‐attention treats all query-key interactions uniformly. 
Inspired by lateral inhibition in biological neural circuits and building on the recent Differential Transformer’s use of two parallel softmax subtraction for noise cancellation, we propose Multihead Differential Gated Self-Attention (M-DGSA) that learns per‐head input-dependent gating to dynamically suppress attention noise. Each head splits into excitatory and inhibitory branches whose dual softmax maps are fused by a sigmoid gate predicted from the token embedding, yielding a context-aware contrast enhancement. M-DGSA integrates seamlessly into existing Transformer stacks with minimal computational overhead. We evaluate on both vision and language benchmarks, demonstrating consistent robustness gains over vanilla Transformer, Vision Transformer, and Differential Transformer baselines. Our contributions are (i) a novel input-dependent gating mechanism for self‐attention grounded in lateral inhibition, (ii) a principled synthesis of biological contrast‐enhancement and self‐attention theory, and (iii) comprehensive experiments demonstrating noise resilience and cross-domain applicability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a new approach called Multihead Differential Gated Self-Attention (M-DGSA), designed to enhance the robustness of Transformers against noisy inputs. By introducing an input-dependent gating mechanism for each attention head, M-DGSA dynamically combines excitatory and inhibitory branches, allowing it to more effectively suppress noise. Experimental results demonstrate that M-DGSA consistently boosts accuracy and robustness, particularly under noisy conditions, across both vision and language tasks.

### Strengths
The paper introduces M-DGSA, a method that learns per-head, input-dependent gating to dynamically suppress attention noise. The results demonstrate consistent improvements in noisy environments, showcasing the method's effectiveness. Additionally, the paper is well-written and accessible.

### Weaknesses
The paper presents a relatively straightforward idea and lacks significant originality. Compared to the approach in Ye et al. (2024), there are no major innovations or departures in the proposed method.

The experiments have some notable limitations. The CIFAR and MNIST datasets are relatively small and simple, which may not fully showcase the model's capabilities. Additionally, the reported ImageNet accuracy (Table 2) is low. Due to these factors, the claims regarding the algorithm's effectiveness remain somewhat unconvincing.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose multihead differential gated self-attention (M-DGSA), a modification of the differential transformer (DT). M-DGSA replaces the scalar \lambda in DT with an input-dependent sigmoid gate. The method is evaluated in both vision and language tasks, showing modest accuracy gains over the DT baseline on (clean) datasets.

### Strengths
- Replacing DT's static $\lambda$ with an input-dependent gate is a reasonable modification
- Evaluations are averaged over 5-seeds with sd reported, which is excellent
- Empirical evaluation is fairly sound, though shows relatively modest results (ImageNet is convincing, mod concerns about memory/compute being held equal)
- Something only mentioned in the appendix: they got rid of DT's $\lambda$ schedule, using a fixed value of 0.8. This seems like a potentially useful contribution, especially if it applies only to the new gated version. Tuning the schedule sounds like a hassle
- Qualitative results for images seem convincing

### Weaknesses
- More discussion of compute/memory requirements would be appreciated; it's unclear if the relatively small gains on ImageNet are worth potential additional training/inference time. The appendix mentions it's roughly equal, but seems somewhat offhand. 
- Relatively small gains compared to DT itself, except the Newsgroup dataset where DT performs suspiciously badly.
- Undertrained CIFAR-10 baselines -- 75% accuracy on CIFAR-10 is super low, makes it hard to trust the comparisons. You could use something like mimetic initialization to compensate for small datasets (I know it's hard to train ViTs on small datasets)

### Questions
- Are the comparisons to DT and vanilla ViT fair wrt the use of SwiGLU? Should there be additional ablations for this?
- Do you have any more analyses of memory/compute requirements compared to baselines? 
- The method is motivated as improving noise robustness to some extent -- have you done experiments on this in particular?

I'd increase my score pretty easily if some of these things were cleared up.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Multihead Differential Gated Self-Attention (M-DGSA), a novel self-attention mechanism for Transformers inspired by lateral inhibition in biological neural circuits. M-DGSA splits each attention head into excitatory and inhibitory branches, fusing their outputs via a learned, input-dependent gating mechanism to dynamically suppress attention noise. The approach is designed to enhance robustness to corrupted or noisy inputs and integrates seamlessly into existing Transformer and Vision Transformer (ViT) architectures with minimal computational overhead. Experiments on both vision (e.g., CIFAR-10/100, ImageNet) and language (e.g., IMDB, MNLI) benchmarks show that M-DGSA improves accuracy and noise resilience over standard and Differential Transformer baselines.

### Strengths
+ The motivation of the propopsed method is resonable introducing an interpretable gating mechanism based on lateral inhibition.

+  M-DGSA shows improved accuracy and noise resilience across several vision and language tasks, outperforming baselines. It also produces sharper, more focused attention maps.

+ M-DGSA can be incorporated into existing Transformer architectures with negligible computational or memory cost.

### Weaknesses
- The gating mechanism, while lightweight, adds more complexity to the attention computation and may require careful tuning. It is not clear how the proposed method can effectively and efficiently scale up: effects on training stability, convergence speed, or performance on very large-scale or long-sequence tasks.

- The evaluations are limited to synthetic noise. Most robustness experiments use synthetic corruptions, while real-world noise and other modalities e.g., cross-attention, multimodal or diffusion tasks are not explored.

- While gains are consistent obtained on several benchmarks, the margin over Differential Transformer and ViT baselines is marginal, especially on saturated or simple benchmarks.

### Questions
Please refer to the detailed questions raised in Weakness section above.

### Soundness
3

### Presentation
2

### Contribution
2
