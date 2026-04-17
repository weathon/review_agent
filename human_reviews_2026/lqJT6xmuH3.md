# Equilibrium Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Large Language Models (LLMs) excel across diverse applications but remain impractical for edge deployment due to severe memory bottlenecks at the edge devices. We propose Equilibrium Language Models (ELMs), a novel compression framework that replaces groups of Transformer layers with a lightweight fixed-point network, reinterpreting deep computation as solving for an equilibrium state. To achieve ELMs, We introduce *Group Pruning Policy Optimization*, which automatically learns optimal pruning intervals. Moreover, we propose *One-Step KV-Cache*, which drastically reduces memory overhead by storing only the final iteration cache without compromising the accuracy, to enable effective deployment at the edge devices. Across different tasks such as common sense reasoning, mathematical problem solving, and code generation, ELMs prune 28\% of parameters while retaining 99\% of the accuracy of dense fine-tuned LLMs, establishing a new direction for memory-efficient edge deployment of large models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This submission proposes Equilibrium Language Models (ELMs), a novel compression framework for Transformer-based large language models targeting edge deployment. The core idea is to replace groups of consecutive Transformer layers with a lightweight fixed-point network, reinterpreting deep computation as solving for the equilibrium state of a fixed-point system. To enable ELMs, the authors introduce two key components: (1) Group Pruning Policy Optimization (GPPO), a reinforcement learning-based method to automatically identify optimal layer pruning intervals; and (2) One-Step KVCache, which reduces memory overhead by storing only the final iteration’s cache without accuracy loss. Experiments on Qwen2.5 and Llama3.2 models show that ELMs prune 28% of parameters while retaining 99% of the accuracy of dense fine-tuned LLMs across commonsense reasoning, mathematical problem-solving, and code generation tasks.

### Strengths
1. The work breaks from traditional LLM pruning (e.g., weight-level, head-level, or heuristic layer pruning) by leveraging fixed-point network theory (from Deep Equilibrium Models, DEQs) to replace layer groups. 

2. GPPO avoids the limitations of heuristic pruning metrics (cosine similarity, perplexity) by framing layer interval selection as a policy optimization problem. 

3. One-Step KVCache is a critical optimization for edge deployment

### Weaknesses
1. All experiments use LLMs with exactly 28 layers (Qwen2.5-1.5B/7B, Llama3.2-3B). It remains unclear how ELMs perform on models with different depths.

2. The paper highlight the significance for edge deployment, however, the paper shows no data on latency results for edge devices. 

3. While One-Step KVCache reduces memory, the submission only reports KV cache size (Table 3) but not total inference memory (including model parameters, activations). Edge devices often have strict RAM limits (e.g., 4GB on low-end phones), so total memory data is essential.

### Questions
1. What happens if the inherited Transformer layer is replaced with a simple FFN? Or if W_h is not initialized as an identity matrix? Such ablations would clarify which components drive performance retention.

2. How do you choose the convergence threshold for One-Step KVCache? 

3. The paper mentions using Stochastic Jacobian-Free Backpropagation (SJFB) to reduce training cost , How does ELMs’ training time/GPU memory compare to baselines?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Equilibrium Language Models (ELMs), a new family of Transformer-based language models designed to reduce inference-time memory usage and improve computational efficiency without sacrificing performance.

### Strengths
- Well-written and clearly structured; the mathematical formulations are concise and supported by clear intuition.

- Novel modeling approach: Reformulating decoding as a fixed-point equilibrium problem is original and elegant.

- Parameter efficiency: Demonstrates substantial parameter pruning without requiring fine-tuning.

### Weaknesses
- While the paper emphasizes parameter pruning (~28%), it does not report key deployment metrics such as inference latency, memory consumption, or speedup over dense baselines. This makes it difficult to verify the real-world efficiency gains, especially since equilibrium solvers may incur higher per-step computation even if they reduce parameter count.

- Although the paper presents an elegant formulation of Equilibrium Language Models (ELMs) and demonstrates strong pruning performance, it does not discuss the cost, stability, or convergence of obtaining these models. In particular, the Group Pruning Policy Optimization process likely introduces significant computational overhead, as it involves iterative search and training steps that may require multiple retraining cycles.

- In Table 2, ELMs are not consistently better than all baselines under the same compression level. For instance, LLM-Streamline (Layer) outperforms ELMs on MMLU at M=8, and Sheared LLM performs better on HumanEval. This suggests that ELMs may not be universally superior. A more systematic comparison across different  M values and tasks is necessary to understand when ELMs are advantageous or potentially weaker than existing methods.

### Questions
- What happens when the pruned layers are not contiguous? For example, instead of pruning 12 consecutive layers, what if we prune two separate groups of six layers distributed across the model? 


I am open to discussing this further during the rebuttal and will be happy to increase my score if my concerns are addressed.

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
This paper proposes ELMs, a model compression framework for LLMs. Instead of pruning individual weights or layers, ELMs replace groups of Transformer layers with a fixed-point network. This approach reinterpretes deep sequential computation as solving for an equilibrium state. The paper introduces: 1. Group Pruning Policy Optimization (GPPO): a reinforcement learning–based policy that learns optimal layer intervals for conversion to fixed-point modules; 2. One-Step KV Cache: a technique that stores only the final iteration’s cache to reduce memory cost during inference without sacrificing accuracy; 3. Adaptive Solvers (Broyden/Anderson acceleration): used to speed up convergence of fixed-point iterations.

Experiments on Qwen2.5-1.5B, Qwen2.5-7B, and Llama3.2-3B across commonsense reasoning, math, and code benchmarks show that ELMs prune ~28% of parameters while maintaining ≈99% of the dense model’s accuracy.

### Strengths
1. The paper reformulates deep computation as solving for an equilibrium state, which is quite novel.
2. The approach consistently outperforms all baselines, including recent strong pruning methods. The method still reaches good performance on hard reasoning tasks (e.g., GSM8K, MATH, HumanEval), where most compression methods fail.

### Weaknesses
1. The GRPO approach seems to only search for the start of the layers to be pruned under a fixed number of pruned layers. In this case, the search space is not very huge. How is the computation cost comparing to going through all of the possible start layers? Also, GPPO adds a non-trivial training loop with policy optimization, which might not be required for simpler pruning methods. 
2. Evaluations are limited to post-finetuned LLMs. It remains unclear how ELMs generalize to instruction-tuned multi-task setups.

### Questions
1. How does the choice of fixed-point iteration solver (simple, Broyden, Anderson) affect latency in real-time inference on GPU/edge devices?
2. Can GPPO be applied jointly to multiple layer groups (non-contiguous intervals), or is it limited to a single interval per model?
3. What is the training cost of GPPO, and how is it comparing to enumerating method?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a compression method for LLMs called "Equilibrium Language Models" (ELMs). The approach replaces a continuous block of layers in the model with a single lightweight equilibrium layer, which computes its output through iterative solving, thereby achieving parameter compression. To support this framework, the paper introduces three auxiliary components: 1) a reinforcement learning-based "Group Pruning Policy Optimization" (GPPO) method for automatically selecting the layer blocks to be replaced; 2) a "single-step KV cache" mechanism designed to reduce memory overhead during iterative inference; and 3) the application of existing numerical solvers to accelerate convergence. Experiments demonstrate that this method maintains performance close to the original model across multiple benchmarks after pruning approximately 28% of the parameters, outperforming several baseline approaches.

### Strengths
A New Perspective on Compression
The paper rethinks model pruning as an implicit depth problem. This offers a new way to approach this field, showing that we can use iterative computation to replace a large number of parameters.

Strong Empirical Results 
The paper shows excellent results in balancing parameter reduction and accuracy, outperforming strong existing baselines. This demonstrates the great potential of the ELM framework.

Efficient Memory Optimization
The "One-Step KV Cache" is a simple but effective innovation. It directly solves the huge memory overhead that the iterative method could cause during LLM inference. This is especially valuable for long sequence generation.

### Weaknesses
Lack of Key Inference Latency Evaluation
The authors claim the method is friendly for edge deployment, but they don't provide any real-world speed data (like tokens/sec) to support this. The iterative process, the solver, and the convergence checks all add extra overhead. The final latency might even be worse than the original model. Without this data, the practical value of the paper is questionable.

Missing Analysis of GPPO's Practicality and Cost
Automatic search is nice, but its cost is critical to determine if it's feasible. The authors need to quantify the search cost of GPPO (e.g., how many GPU-days it takes) and discuss its cost relative to the final model training. 

Limited Pruning Strategy
The method is limited to pruning a single, contiguous block of layers. This is a strong assumption and may not be the best approach. The authors should discuss why they made this choice (e.g., technical limitations or to simplify the problem) and explore the possibility of extending it to more flexible pruning patterns, like multiple, non-contiguous blocks.

### Questions
Same as weakness.

### Soundness
3

### Presentation
4

### Contribution
3
