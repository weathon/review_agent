# Universal Properties of Activation Sparsity in Modern Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Activation sparsity is an intriguing property of deep neural networks that has been extensively studied in ReLU-based models, due to its advantages for efficiency, robustness, and interpretability. 
However, methods relying on exact zero activations do not directly apply to modern Large Language Models (LLMs), leading to fragmented, model-specific strategies for LLM activation sparsity and a gap in its general understanding. 
In this work, we introduce a general framework for evaluating sparsity robustness in contemporary LLMs and conduct a systematic investigation of this phenomenon in their feedforward~(FFN) layers.
Our results uncover universal properties of activation sparsity across diverse model families and scales.
Importantly, we observe that the potential for effective activation sparsity grows with model size, highlighting its increasing relevance as models scale. 
Furthermore, we present the first study of activation sparsity in diffusion-based LLMs. 
Overall, our work provides a comprehensive perspective and practical guidance for harnessing activation sparsity in LLM design and acceleration.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a comprehensive study on the properties of sparsification in the FFN layers of transformer-based language models and diffusion language models. Specifically, they use a simple top-$p$ sparsification strategy and analyze the robustness against sparsity of various models from the Gemma3, Qwen2.5, and LLaMa3.1/3.2 family, by measuring at what sparsity levels does downstream task performance drop below 99% of its original performance.

Some interesting findings from this paper include: input-based sparsification is often more attractive than gate-based sparsification, larger models are more robust to sparsification, and diffusion models are more robust to sparsification. The findings of this paper are valuable for designing language models with better robustness against sparsification or methods that better leverage sparsification to achieve inference speedup.

### Strengths
- The paper is well-presented and generally easy to follow.
- The findings of this paper is very interesting and highly valuable to researchers interesting in language models with sparse activations. 
- The methods and experimental setup is well designed, making the results reliable and convincing.

### Weaknesses
- How sparse activation patterns evolve across diffusion steps is an interesting phenomenon and the paper does a good job in bringing it up (starting with Line 406, and Figure 6). However, the conclusions of the experiments in this regard is inconclusive and it is unclear how they are useful for the community.
- Considering how sparse mixture-of-experts (MoE) models have become very popular, I would love to see investigations in how the findings of this paper transfers to MoE models. I suggest repeating some of the experiments with models such as MoE variants of the Qwen3 series, GPT-OSS, GLM4.5, etc.
- While the top-$p$ sparsification strategy is reasonable, I would like to see a comparison of it against an even simpler, top-$k$ sparsification strategy.

### Questions
- What does "Normalized accuracy" in Figure 5 mean?
- Can you provide more insights into why MMLU-Redux is less robust to sparsification compared to GSM8K and TruthfulQA?
- It seems that you apply the same $p$ value across all layers. However, I think using a different $p$ for different layer is more reasonable. Is it possible to repeat the experiments, but with a layer-specific $p$ value such that each layer can have its own critical sparsity?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work proposes a unified framework for evaluating sparsity robustness in large language models and systematically examines activation sparsity in their feedforward layers. The study identifies consistent sparsity patterns across different model families and sizes, revealing that larger models exhibit greater potential for effective activation sparsity. It also provides the first analysis of sparsity in diffusion-based LLMs. Overall, the work offers insights and guidance for leveraging sparsity to improve LLM efficiency and design.

### Strengths
- The paper uncovers some universal properties of activation sparsity across diverse model families and scales. This could be quite helpful to the community.
- The paper is well written.
- Experiments are quite quantitive and informative.

### Weaknesses
- Pure experimentation-driven work without any theoretical analysis or others.
- Larger model leads to higher critical sparsity. This point is actually very straightforward. For instance, usually larger model and smaller model are trained on the same pile of data. Of course the larger model admit more redandancy.

### Questions
- What is  "functional sparsity"? The authors not seem to provide good explanation across the paper, though this term is very important.
- What is "effective rank"? Need short explanation in the paper to be self-contained. For instance, what "effective rank of 0.1" means? How it corresponds to your conclusion.

Personally, I like this type of paper giving inspiration. Please work on the quesitons and weakness points if appropriate.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the lack of a general understanding of activation sparsity in modern Large Language Models (LLMs) that utilize non-ReLU activations, which prohibits the use of traditional zero-based sparsity methods. To solve this, the authors introduce a novel, general framework to evaluate sparsity robustness in contemporary LLMs, focusing on the Feedforward Network (FFN) layers. Through systematic investigation across diverse models and scales, the work reveals a key universal property: the potential for effective activation sparsity increases with model size. Furthermore, the study presents the first analysis of activation sparsity in diffusion-based LLMs, ultimately offering a comprehensive perspective and practical guidance for leveraging this phenomenon in LLM design and acceleration.

### Strengths
1. The paper introduces a simple, training-free top-p sparsification method and the metric of critical sparsity (maximum sparsity retaining 99\% performance), providing a unified and fair way to compare sparsity tolerance across different LLM architectures and FFN components.

2. The work systematically confirms that the critical sparsity increases with model size. This is consistently reinforced by the finding that the effective rank of activations decreases with model size, offering strong evidence that larger models inherently possess greater exploitable redundancy.

3. The investigation into diffusion-based LLMs (LLaDA) is novel, revealing that they exhibit substantial activation sparsity and even slightly more favorable sparsity-performance trade-offs than autoregressive models, highlighting a new acceleration opportunity.

4. Based on empirical data, the authors provide the practical insight that input activation sparsification is the most effective training-free approach, as its sparsity tolerance is comparable to or greater than gate or up-projection activations.

### Weaknesses
1. A major finding is that the critical sparsity varies substantially across different downstream tasks and training recipes (e.g., instruction-tuning). This challenges the core assumption of many prior acceleration methods that sparsification rules calibrated on an auxiliary dataset will generalize universally without overfitting

2. The paper acknowledges that the effective speedups from activation sparsity methods are practically limited to a factor of 1.3x to 1.5x, which is less compelling when compared to alternative lossless techniques like speculative decoding that can achieve up to 4x speedups.

3. The analysis is explicitly constrained to only the FFN layers, intentionally excluding the Multi-Head Attention (MHA) module. While a cost justification is provided, this limits the comprehensiveness of the "universal properties" claim within the entire Transformer architecture.

4. The use of effective rank as a theoretical proxy for redundancy is weakened by the observation that gate activations show a high effective rank yet exhibit a low empirical capacity for sparsification, suggesting that this metric is insufficient to fully capture robustness to sparsification

### Questions
none

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates activation sparsity in modern large models. Beyond ReLU-based networks that produce exact zeros, using SiLU/GELU activations in FFNs yields functional or approximate sparsity, with many activations near zero. The authors address the current fragmented understanding of this phenomenon by introducing a general, training-free framework to evaluate sparsity robustness.

### Strengths
They propose a simple top-p sparsification to induce sparsity in various activations (input, gate, up-projection, intermediate). This allows for the introduction of critical sparsity, i.e., the maximum sparsity level that causes less than 1% performance drop. Through extensive experiments on models like Gemma, Llama, and Qwen across different scales (e.g., 1B to 32B parameters), they find that the potential for effective activation sparsity increases with model size. They also find that input-based sparsification is as effective as, or even better than, the more commonly studied gate-based methods, making it a more practical, predictor-free approach. The study also shows that critical sparsity is task-dependent, varying significantly across different downstream evaluations. It also persists across different model types, including instruction-tuned and reasoning-specialized variants. All these provide a comprehensive perspective and practical guidance for harnessing activation sparsity in model design and acceleration.

### Weaknesses
This paper's methodology introduces sparsity via the top-p rule and defines critical sparsity as the level at which performance degradation is less than 1%. This is similar to the paper "Sparsing Law: Towards Large Language Models with Greater Activation Sparsity". Further comparison of different sparsity definitions is crucial, as a better definition of sparsity will result in less reduction in model performance.

### Questions
Please refer to "Weaknesses".

### Soundness
3

### Presentation
3

### Contribution
3
