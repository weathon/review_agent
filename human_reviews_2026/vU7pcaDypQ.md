# Partial Parameter Updates for Efficient Distributed Training

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
We introduce a memory- and compute-efficient method for low-communication distributed training. Existing methods reduce communication by performing multiple local updates between infrequent global synchronizations. We demonstrate that their efficiency can be significantly improved by restricting backpropagation: instead of updating all the parameters, each node updates only a fixed subset while keeping the remainder frozen during local steps. This constraint substantially reduces peak memory usage and training FLOPs, while a full forward pass over all parameters eliminates the need for cross-node activation exchange. Experiments on a $1.3$B-parameter language model trained across $32$ nodes show that our method matches the perplexity of prior low-communication approaches under identical token and bandwidth budgets while reducing training FLOPs by $15$% and peak memory by up to $47$%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a partial parameter updates scheme based on the DiLoCo framework, which reduces memory use and communication costs by freezing parameters on each node. Unlike Streaming DiLoCo which divide the model in a pipeline-parallel manner, this work slice the training in a tensor-parallel way to achieve a similar results.

### Strengths
This paper is well-structured and the idea is easy to follow, and it provides targeted optimizations to address the problem of high peak memory consumption in Streaming DiLoCo. Under low-bandwidth settings, proposed method converges faster than standard DDP.

### Weaknesses
1. It looks like a natural extension to Streaming DiLoCo with frozen weights, also lacks convergence guarantee in theory. As seen in Table 1, the ppl goes up quickly when N is larger than 8.
2. The improvement compared to Streaming DiLoCo is marginal, seen in Figure 3, while the curve for Streaming DiLoCo is smoother, indicating a more stable run.

### Questions
1. Could you provide experimental results for a Streaming DiLoCo synchronization strategy that has the same communication budget as 1/4 MLP? I think it might be a better baseline when comparing convergence.
2. Is there any clear pattern in the strategy selection for parameter slicing (eg. training a model with different shape), and could you provide ablation experiments that only slice attention heads?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a method to improve the efficiency of distributed Transformer training. Specifically, it freezes a subset of parameters on each node, allowing local updates and an all-reduce operation on the local weight changes (similar to DiLoCo), followed by an outer optimization step. By omitting optimizer states for the frozen parameters, the approach reduces memory usage and achieves better performance per FLOP.

### Strengths
The proposed method in the paper is both interesting and practical. Since adaptive optimizer states typically consume more memory than the model parameters themselves, the resulting memory savings are substantial. These savings can be leveraged to train larger models or use bigger batch sizes, leading to more efficient resource utilization.

The paper is well written, and the proposed approach is simple yet effective. It can be easily implemented with existing communication-efficient distributed training frameworks. The experimental results are promising and demonstrate the potential of the method in improving both memory efficiency and training performance.

### Weaknesses
While the proposed method is promising, its main limitation lies in the lack of experimental breadth. Evaluating only a single model size, in a single distributed configuration, and on a single dataset is insufficient to convincingly demonstrate the method’s generality or robustness. Additional ablations and experiments under diverse settings would significantly strengthen the paper’s claims.

Moreover, the reported memory savings are highly dependent on the choice of optimizer. For instance, modular optimizers such as Muon [1] and its variants are gaining traction and have shown that momentum-based SGD can perform comparably to adaptive optimizers like AdamW. In such cases, the memory advantage offered by the proposed approach becomes less substantial.

[1] https://arxiv.org/abs/2502.16982

### Questions
1. It would be valuable to include an ablation showing how the proposed method performs as the number of nodes (or model replicas) increases. This would clarify how communication efficiency and performance scale with distributed configurations.

2. Including training and validation curves plotted against the number of processed tokens for all compared baselines would be helpful. Such curves would help assess convergence behavior, stability, and sample efficiency of the method.

3. Including the performance of standard DDP training in Table 1 would provide a stronger reference point and make it easier to contextualize the improvements of the proposed approach.

4. Additional experiments across multiple datasets, network depths, and hidden dimension sizes would significantly strengthen the empirical section. These would help demonstrate that the observed gains are consistent across varying model scales and data regimes.

### Soundness
2

### Presentation
3

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
This paper introduces an efficient algorithms for low-communication distributed data-parallel training that performs local updates on a node-specific subset of parameters. The author(s) validate the effectiveness of the method by training a 1.3B LLMs on 32 nodes and compare it with Streaming DiLoCo on Perplexity and training FLOPs.

### Strengths
1. This paper describes the research method in detail.
2. This paper describes the experimental setup in detail.

### Weaknesses
1. There is no discussion on block coordinate optimization. There is plenty of literacture on block coordinate optimization [1-3] and it is unclear what is the main technical contribution of the paper.
2. There is no comparison with block coordinate optimization and parameter-efficient fine-tuning (PEFT) in the experiments. Given that there aer a lots of existing methods on block coordinate optimization, they should be included as baselines. Meanwhile, the proposed method has similar performance advantages to PEFT methods such as LoRA [4], requiring only communicating matrices smaller than the model weights during parameter synchronization. So PEFT methods should be included in the comparison.

[1] Accelerating Block Coordinate Descent for LLM Finetuning via Landscape Correction

[2] Memory-Efficient Block Coordinate Descent for Hessian-Informed Zeroth-Order Optimizer

[3] How to Train a Model on a Cheap Cluster with Low Cost using Block Coordinate Descent

[4] Lora: Low-rank adaptation of large language models.

### Questions
N/A

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
This paper proposes partial parameter updates for local SGD and then global synchronization with diloco/fedavg to reduce communication cost between workers in a data parallel setting. The core idea is that each worker optimises a subset of model parameters (predetermined) and other frozen parameters are only updated through the outer optimizer after the global sync. The experiments show comparable performance with streaming diloco wrt to flops.

### Strengths
1. Even though incremental, the idea is simple and seems effective. The way of determining the subsets is similar to MoE and makes sense. Due to partial parameter updates, per-node memory usage is reduced, enabling larger models to be trained in the DP setting.
2. Language model experiment shows performance matching streaming diloco when compared against flops processed.

### Weaknesses
1. Limited experiments: Only one dataset and one datasets and one baseline method compared. Specifically, sparta (Beton et al) is a relevant baseline, and diloco has many variants -- for instance, outer gradients can be compressed using quantization/low-rank to reduce communication cost. These methods could be compared to correctly position the paper wrt convergence speed and communication cost. 
    - As the main idea is heuristic and experiments with multiple datasets and model sizes would strengthen the paper.
2. Even though wrt flops, the results match streaming diloco, wrt to training steps it is slow.
3. To reduce memory usage and fit larger models into smaller gpus, one can adopt pipeline parallelism with low-rank compression of activations and gradients as shown in [1]. This is a relevant comparison on the aspect of fitting larger models in smaller gpus.

[1] Ramasinghe, Sameera, et al. "Protocol Models: Scaling Decentralized Training with Communication-Efficient Model Parallelism." arXiv preprint arXiv:2506.01260 (2025).

### Questions
1. Fig.1 shows that the outer optimizer only updates the frozen parameters, but this contradicts line 15 of Algorithm 1. Also, there are common trainable parameters between workers, so they will also be updated during the outer step. Please clarify

### Soundness
3

### Presentation
3

### Contribution
2
