# SpikingLLM: Spiking Large Language Models with Causal Spiking Self-Attention and Spike-Form Knowledge Distillation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Spiking Neural Networks (SNNs) offer promising energy-efficient alternatives to large language models (LLMs) due to their event-driven nature and ultra-low power consumption. However, to retain representation capacity, most existing spiking LLM approaches rely on integer activations or softmax, which involve intensive floating-point operations and undermine inference efficiency. Moreover, the intrinsic spatial-temporal optimization of spiking networks further increase the direct training cost and difficulty. To address these challenges, we propose \textbf{SpikingLLM}, the first fully binary spike-driven spiking LLM framework developed from random initialization, without reliance on floating-point matrix multiplications or softmax. At the core of SpikingLLM is the \textbf{Causal Spiking Self-Attention (CSSA)} mechanism, which replaces conventional softmax with binary spike-based operations and thereby enables autoregressive language modeling in the spiking domain, ensuring low-cost inference. To support cost-efficient training under constrained computational budgets, we further introduce \textbf{Spike-Form Knowledge Distillation (SKD)}, a multi-level distillation strategy that aligns ANN teacher and SNN student across embeddings, attention maps, intermediate features, and output logits. SKD framework allows SpikingLLM to achieve competitive performance with ANN counterparts using substantially fewer training tokens (e.g., 1.0B tokens for a 0.125B model and 10.0B tokens for a 1.3B model), resulting in effective training. As a result, SpikingLLM achieves ANN-level performance at only \textbf{4.16\%–5.87\%} of the computational cost on natural language generation tasks. Our results highlight the feasibility and effectiveness of fully binary spike-driven LLMs and establish the distillation as a promising pathway for energy-efficient, brain-inspired spiking NLP.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents SpikingLLM, a fully spike-driven large language model that eliminates floating-point operations using a new Causal Spiking Self-Attention mechanism and a Spike-Form Knowledge Distillation framework. CSSA enables autoregressive modeling with binary spikes, while SKD distills multi-level knowledge from an ANN teacher, improving training efficiency. Experiments show strong energy savings, and competitive accuracy up to ~93% of ANN baselines with far fewer training tokens.

### Strengths
1. Introduces a novel spike-based attention mechanism that supports causal modeling without softmax.
2. SKD provides an effective approach for training large SNNs from scratch.

### Weaknesses
1. The experiments mostly benchmark against smaller or less comparable baselines; a head-to-head test with modern quantized or efficient transformer variants would strengthen claims.

2. It’s unclear how the spike-based training framework would scale to multi-billion parameter ranges beyond 1.3B without major engineering adjustments.

3. Although impressive for an SNN, SpikingLLM’s accuracy still falls short of ANN models by a notable margin on reasoning benchmarks.

### Questions
1. How does the proposed Causal Spiking Self-Attention handle long-context dependencies, especially when firing sparsity increases? Is there a performance drop for extended sequences?

2. Could the Spike-Form Knowledge Distillation method generalize across architectures, or does it rely heavily on alignment with the OPT-family structure?

### Soundness
2

### Presentation
2

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
This paper presents SpikingLLM, a fully binary, spike-driven large language model designed for energy-efficient inference without relying on floating-point operations or softmax. It introduces a Causal Spiking Self-Attention (CSSA) mechanism that enables autoregressive language modeling entirely in the spiking domain. To address training challenges, the authors propose a multi-level Spike-Form Knowledge Distillation (SKD) framework that allows direct training from random initialization. Experimental results show that SpikingLLM achieves competitive performance to ANN-based models with drastically reduced computational and energy costs, highlighting its potential for efficient, brain-inspired NLP.

### Strengths
S1: This paper proposes a directly trained spiking language model that can be pre-trained with a decoder-only architecture, effectively shortening the spike length compared to ANN-to-SNN conversion methods.
S2: This paper trains an SNN-based language model from scratch, demonstrating the language modeling capability of spiking neural networks.
S3: The method in this work is simple and easy to follow.

### Weaknesses
S1: As a pre-trained model, training on only 1B/10B tokens for generative language tasks is insufficient—the model has not converged to a usable level and may even fail to conduct effective conversations (a case study could help verify this). Therefore, larger-scale experiments are recommended to increase the confidence of the conclusions.
S2: The proposed model architecture lacks ablation studies. Further discussion is needed on whether components such as softmax removal are truly necessary in SNNs.

### Questions
How can the 1-bit activation LLMs be ensured to converge effectively? Based on experience with quantized models, directly pretraining a decoder-only model with 1-bit activations generally fails to achieve stable convergence. Therefore, a case study or empirical verification is needed to demonstrate that the proposed model can indeed be successfully trained.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel Spiking LLM framework with the CSSA mechanism to improve efficiency and reduce computation.
The idea is interesting and potentially impactful for energy-efficient LLMs.
However, the experimental evidence is not sufficient to validate the claimed advantages within SNNs.
The comparisons with ANNs and the role of discrete values need clearer justification.
Overall, the work is promising but requires stronger ablation and scalability analysis.

### Strengths
The paper proposes the Spiking LLM framework, which improves performance while reducing computational cost.

### Weaknesses
The authors introduce CSSA; however, the ablation studies compare only against ANN counterparts, which cannot effectively demonstrate the validity of this method within SNNs. The ablation analysis requires further clarification.
The proposed method employs discrete values, and what is the essential difference between this and quantized ANN methods? The experiments lack direct comparisons with quantized ANNs.
The number of parameters is small compared to SOTA methods.

### Questions
Does introducing an ANN teacher increase training memory usage and time, thereby limiting scalability?
Why does designing a CSSA lead to performance improvement? This technique is already a very common approach in ANNs, so I find it difficult to understand its specific contribution to SNNs.

### Soundness
2

### Presentation
3

### Contribution
2
