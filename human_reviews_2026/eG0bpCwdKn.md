# QuRL: Low-Precision Reinforcement Learning for Efficient Reasoning

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has become a trending paradigm for training reasoning large language models (LLMs). 
However, due to the autoregressive decoding nature of LLMs, the rollout process becomes the efficiency bottleneck of RL training, consisting of up to 70\% of the total training time. 
In this work, we propose Quantized Reinforcement Learning (QuRL) that uses a quantized actor for accelerating the rollout. 
We address two challenges in QuRL. First, we propose Adaptive Clipping Range (ACR) that dynamically adjusts the clipping ratio based on the policy ratio between the full-precision actor and the quantized actor, which is essential for mitigating long-term training collapse.
Second, we identify the weight update problem, where weight changes between RL steps are extremely small, making it difficult for the quantization operation to capture them effectively. 
We mitigate this problem through the invariant scaling technique that reduces quantization noise and increases weight update.
We evaluate our method with INT8 and FP8 quantization experiments on DeepScaleR and DAPO, and achieve 20% to 80% faster rollout during training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces QuRL (Quantized Reinforcement Learning), a novel method to accelerate the rollout phase in reinforcement learning (RL) for large language models (LLMs). The authors address the critical bottleneck in RL training—where the autoregressive decoding process consumes most of thel training time—by quantizing the actor model during rollout while maintaining full-precision parameters for policy updates.

### Strengths
(1) it identifies and systematically addresses under-explored challenges at the intersection of quantization and RL, such as the divergence between behavior and proximal policies. The proposed ACR mechanism is both intuitive and effective, dynamically adjusting clipping bounds to sustain training stability over extended horizons. 
(2)UAQ cleverly leverages invariant scaling to amplify weight updates relative to quantization noise, enabling the quantized model to reflect training dynamics more faithfully.
(3) the experimental validation is extensive, covering diverse model sizes (7B to 32B), tasks, and quantization formats (INT8/FP8), with consistent improvements over strong baselines like FlashRL.

### Weaknesses
(1) he UAQ method, though effective, introduces a hyperparameter (scale factor ss) that requires tuning, adding complexity. Also the adaption may add more computational overhead
(2)while QuRL is positioned between PTQ and QAT, its dependence on one-time scaling may limit adaptability to highly dynamic training regimes.

### Questions
(1) What are the implications of quantization on RL exploration-exploitation trade-offs, especially in tasks requiring high creativity or diversity?
(2) How would QuRL perform under more aggressive quantization (e.g., 4-bit), and what additional adaptations would be necessary?

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
3

### Summary
This paper introduces Quantized Reinforcement Learning (QuRL), a method to accelerate the rollout phase in reinforcement learning (RL) for large language models (LLMs) by quantizing the actor model during inference. The authors identify two key challenges: (1) training instability due to policy divergence between the full-precision and quantized actors, and (2) the mismatch between the magnitude of weight updates and quantization errors, which hinders the quantized model from capturing training dynamics. To address these, QuRL proposes Adaptive Clipping Range (ACR) to dynamically adjust trust region bounds based on policy divergence, and Update-Aware Quantization (UAQ) to amplify weight updates relative to quantization noise using invariant scaling. Experiments on GSM8K, AIME 2024, and DeepScaleR benchmarks demonstrate that QuRL achieves 20–80% faster rollout with INT8/FP8 quantization while maintaining competitive accuracy compared to full-precision baselines and outperforming naive quantization approaches.

### Strengths
1. The paper tackles a critical bottleneck in RL training for LLMs -- rollout latency -- by integrating quantization into the RL loop. The idea of decoupling quantized rollout from full-precision updates is innovative and addresses a timely problem in scalable RL.

2. The proposed solutions (ACR and UAQ) are well-motivated by empirical analyses of training instability and weight update mismatches. The ablation studies validate the contributions of each component.

3. Experiments span multiple RL algorithms (PPO, DAPO, GRPO), model sizes (7B–32B), and reasoning benchmarks, demonstrating broad applicability. Throughput gains (20–80%) are substantial and empirically validated.

4. The paper is well-structured, with clear explanations of challenges and methodologies. Figures effectively illustrate key issues like training collapse and weight update mismatches.

### Weaknesses
1. The work primarily contrasts QuRL with full-precision RL and naive quantization. Including comparisons to other efficiency methods (e.g., pruning, distillation) could better contextualize QuRL’s advantages.

2. Although ACR mitigates collapse up to 1,200 steps, the scalability to even longer training horizons (e.g., thousands of steps) is not tested. Stability under extended training remains an open question.

3. Robustness: How sensitive are ACR and UAQ to hyperparameters like the clipping bound or scale factor ? Are there guidelines for tuning them on new tasks?

### Questions
See weakness

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
4

### Summary
This paper presents QuRL, a practical recipe to accelerate reinforcement learning for reasoning LLMs by performing rollouts with a low-precision actor (INT8/FP8) while keeping full-precision weights for policy updates, which effectively decouples the behavior (quantized) and proximal (full-precision) policies and necessitates careful importance sampling and trust-region control. They propose Adaptive Clipping Range (ACR), which absorbs the truncation factor into PPO’s clipping term, shrinking the effective range (with a fixed upper cap) to stabilize gradients and prevent late-stage collapse. The paper also introduces Update-Aware Quantization (UAQ), a one-time invariant scaling of transformer linear layers that simultaneously reduces quantization error and amplifies effective updates, yielding an s² improvement in update-to-noise ratio. On GSM8K with PPO, QuRL closes much of the gap between naive INT8 and BF16 (53.55% vs 55.35%) and also improves FP8 robustness relative to baselines. Overall, QuRL stabilizes decoupled PPO (via ACR) and makes low-precision rollouts learnable (via UAQ), offering a simple, engine-compatible path to faster RL for reasoning LLMs.

### Strengths
1. This paper studies an important problem of model quantization in reinforcement learning, which can help speed up the RL training process. The work decouples rollout and optimization by quantizing the actor for sampling while keeping full-precision weights for updates; the paper cleanly frames this as decoupled/off-policy PPO and motivates the need for trust-region/IS control.

2. Adaptive Clipping Range stabilizes late-stage collapse by adapting the clipping bounds to proximal-to-behavior ratios; Update-Aware Quantization (invariant scaling) reduces quantization error and amplifies effective updates with a simple, engine-compatible change.
 
3. The paper conducts a fair comparison to demonstrate the effectiveness of QuRL for RL training.

### Weaknesses
1. Marginal accuracy improvements in places: on GSM8K the final gap vs BF16 remains (~53.55 vs 55.35), and DeepScaleR’s reported average gain is modest (+1.7%), which may feel incremental relative to the engineering complexity.

2. Scale of core effectiveness studies: the detailed PPO accuracy study uses a 0.5B base model (Qwen2.5-0.5B); while throughput is shown for 7B–32B, end-to-end accuracy results beyond small models are limited. Therefore, the authors are suggested to conduct experiments on larger models and even MoE models to demonstrate the effectiveness of their method.

3. Nowadays more models focus on reasoning or agentic capabilities, which requires stronger long-context capabilities of model. The paper does not provide direct evaluations on thinking models or agentic models to study whether the proposed methods works under these settings.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
