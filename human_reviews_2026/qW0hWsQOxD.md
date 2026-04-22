# Lightweight Image-to-3D Shape Generation via Vitality-Aware Pruning and Quantization

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
We propose the first compression framework for image-to-3D generative models that substantially reduces model size while preserving synthesis fidelity.
Recent advances in 3D shape generative modeling, particularly Diffusion Transformer (DiT) architectures, have achieved remarkable progress in synthesis fidelity and controllability. 
However, the substantial computational cost of large DiT-based image-to-3D models hinders their practical application in resource-constrained settings. 
While existing efficiency-oriented approaches improve inference speed, they leave the underlying model size and computational cost of synthesis largely unchanged.
To address this challenge, we propose a systematic compression framework that physically reduces model size while preserving the fidelity of 3D shape synthesis. 
Our approach builds on the observation that Transformer layers in 3D DiT models exhibit non-uniform importance, with only a subset of layers contributing significantly to geometry generation. 
Leveraging this insight, we introduce a vitality-guided framework that integrates structured pruning, adaptive quantization, and targeted fine-tuning to balance efficiency and quality. 
Experimental results show that our method achieves up to 66% model-size reduction across state-of-the-art 3D generative models with minimal loss in synthesis fidelity. 
This highlights the potential of our framework as a plug-and-play solution for efficient 3D shape generation across diverse models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a compression framework for image-to-3D DiT models using layer vitality analysis, structured pruning, adaptive quantization, and targeted finetuning. Experiments on Step1X-3D, Hunyuan3D 2.0, and 2mini achieve 44-66% size reduction while maintaining synthesis quality.

### Strengths
1. This work addresses a practical problem with good results - 44-66% compression enables deployment in resource-constrained environments.
2. The evaluations are systematic and comprehensive, with results on different backbone models showing its generalization ability.
3. The idea of vitality analysis from T2I (Avrahami et al. 2025) to 3D generation using EMD on point clouds is a reasonable domain adaptation.

### Weaknesses
1. The technical novelty is relatively limited, as the core techniques (layer pruning, quantization, and distillation) are mostly from existing ones. The main contribution is applying vitality analysis (from Avrahami et al. 2025) to 3D with EMD. This is essentially applying a 2D technique to 3D generation with minimal domain-specific innovation. It would be better if there are some novel insights about 3D geometry or generation processes. 
2. Different thresholds per architecture undermine "plug-and-play" claims. How sensitive are results? Can this be automated?
3. Why does selective finetuning of only the "Min-vital" layer work? There is no ablation study about this strategy.

### Questions
Refer to weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper has introduced the first study of prunning and quantization of I23D models. Specifically, by leveraging the structured pruning, vitality-aware adaptive quantization, and lightweight finetuning, fidelity can be well maintained under compression. Extensive experiments on several 3D base models and benchmarks have demonstrated the effectiveness of the proposed method.

### Strengths
1. With the rapid progress of 3D AIGC, the prunning and quantization of these models is indeed necessary.
2. The analysis and experiments of this paper is comprehensive.

### Weaknesses
1. I appreciate the The main concern is whether the proposed method is unique on 3D, or can be directly applied to any DiT/transformer architecture. Considering 3D generation all follow the same design, just applying existing prunning and quantization tricks on 3D DiTs can only be a weak contribution.
2. Some discussions of 3D generative models are missing, like 3DTopia-XL (CVPR 25), GaussianAnything (ICLR 25), EG3D / pi-GAN (GAN-based 3D generative models), and OpenAI's Shape-E (the first 3D diffusion model).

### Questions
1. I did not fully understand why the single-layer and double-layer block needs separate threshold and processing. Any insight behind?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a compression framework for image-to-3D generative models based on Diffusion Transformers (DiT). The core contribution is introducing "layer vitality" analysis to quantify each layer's contribution to generation quality using Earth Mover's Distance (EMD) between outputs of the full model and layer-ablated models. Based on this analysis, the method combines: Structured pruning of low-vitality layers, Vitality-aware adaptive quantization (8-bit for vital layers, 4-bit for others) and Selective finetuning targeting minimally-vital layers. Experiments on Step1X-3D, Hunyuan3D 2.0, and Hunyuan3D 2mini demonstrate effective model size reductions while maintaining synthesis quality comparable to full models.

### Strengths
1. Have analysis of layer importance in image-to-3D DiT models and first work to achieve actual model size reduction in this domain.
2. Achieves substantial compression (up to 66%) with minimal quality degradation, making high-quality 3D generation more accessible for resource-constrained environments.

### Weaknesses
1. Limited Novelty
This is essentially a straightforward application of existing compression techniques to 3D models, not a novel method:

"Layer vitality" via ablation+distance is directly borrowed from 2D image/video work (Avrahami 2025, Kim 2025)
Adaptive quantization based on importance scores is standard practice
The paper doesn't explain what makes 3D generation uniquely challenging for compression

2. No Theoretical Analysis
The paper is purely empirical without explaining why compression works:

Why do certain layers have low vitality? Is it due to training data, architecture design, or something else?
Why do different models (Step1X-3D vs. Hunyuan3D) show different vitality patterns?
Without understanding the underlying causes, it's unclear when this approach will succeed or fail

3. Missing Critical Experiments
Failure cases: No analysis of when/why compression degrades quality. What types of objects fail? At what compression ratio does quality collapse?
Computational cost: No reported training time, memory usage, or actual inference speedup. How expensive is the vitality analysis (requires N forward passes)? How long does finetuning take?
Baseline comparisons: No comparison with standard compression methods: Teacher-student distillation(e.g.,DMD) from scratch and other structured pruning approaches

4. Generalization Concerns

Different models need different hyperparameters (learning rates: 10⁻⁸ vs 10⁻⁴, different thresholds, different finetuning iterations)
Only 210 images used for vitality calculation—how stable are these scores?
Only tested on DiT models—unclear if it works for other 3D architectures such as AR-based models.

5. Insufficient Ablations

How does each component (pruning, quantization, finetuning) contribute individually?
How sensitive is performance to threshold selection

### Questions
Why not try sparse-voxel-based methods such as Trellis? What are the challenges?
How sensitive is the method to the number of samples used for vitality calculation?
Since 3D generation models are not particularly large, speedup seems less critical to me. Why not focus on general model compression task or video generation instead, which is more crucial?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a compression pipeline for large image-to-3D DiT models. The central idea is to estimate each layer’s *vitality* by measuring the 3D output degradation via Earth Mover’s Distance between point clouds after temporarily removing that layer. Layers with low vitality are pruned, while remaining ones are quantized with adaptive bit widths proportional to vitality. A small distillation fine-tuning step attempts to recover performance. Experiments on Step1X-3D, Hunyuan3D 2.0, and Hunyuan3D 2-mini report parameter reduction with minimal drop in embedding-based metrics (Uni3D-I, OpenShape-I).

### Strengths
**Clear motivation.**
3D DiT models are large, so compression is valuable.

**Practical meanings.**
The parameter reduction without catastrophic quality loss is practical.

**Good writing.**
The core idea of this paper is clearly written and explained in the method section. The ablation study cleanly separates pruning, quantization, and finetuning effects, and main comparison experiments are clearly demonstrated.

### Weaknesses
**Marginal novelty.**
The approach reuses well-known pruning and quantization frameworks from 2D diffusion literature, adding no new theoretical formulation, and its contribution mainly lies in empirical replication on 3D models. It only substitutes the 2D similarity metric with a 3D EMD and apply the same concept to 3D DiTs. The quantization and finetuning steps are standard, and the final pipeline is an incremental adaptation rather than a new principle.

**Computational cost and scalability not addressed.**
The vitality analysis itself appears computationally heavy. The paper does not disclose GPU hours, runtime, or scaling properties. Without these, claims of *lightweight* or *efficient* are debatable and the significance of this approach would suffer.

**Insufficient evaluation.**
All quantitative metrics are embedding-based and no direct 3D geometry comparisons are provided, which leaves uncertainty about how much detail or structural accuracy is lost during pruning. The reported metrics are calculated from only 200 test samples, which is too small to support broad claims and hinders credibility.

### Questions
1. What is the actual computational cost of vitality analysis?

2. Is there a particular reason why the reported metrics can only be evaluated with 200 pairs?

3. Can you report Chamfer Distance or other geometry-level metrics on a small benchmark?

4. What are the real inference-time savings (latency and memory) after compression?

### Soundness
3

### Presentation
3

### Contribution
2
