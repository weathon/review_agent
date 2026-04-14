# Routing Experts: Learning to Route Dynamic Experts in Existing Multi-modal Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Recently, mixture of experts (MoE) has become a popular paradigm for achieving the trade-off between modal capacity and efficiency of multimodal large language models (MLLMs). Different from previous efforts, we are dedicated to exploring the dynamic experts in existing MLLMs and showing that a standard MLLM can also be a mixture of experts. However, achieving this target is still notoriously challenging. The well-trained MLLMs are more accustomed to the fixed pathway and a drastic change in its inference manner also greatly impedes its performance. To address these issues, we propose a novel dynamic expert routing method for existing MLLMs, termed Routing Experts (RoE), which can achieve example-dependent optimal path routing without obvious structure tweaks. Meanwhile, a new structure sparsity regularization is also introduced to force the well-trained MLLMs to learn more short-cut pathways. In addition, we also address the alignment of the training and inference of MLLMs in terms of network routing. To validate RoE, we apply it to a set of existing MLLMs, including LLaVA-1.5, LLaVA-HR and VILA, and conduct extensive experiments on a bunch of VL benchmarks. The experiment results not only show the effectiveness of our RoE in improving MLLMs' efficiency, but also yield obvious advantages over MoE-LLaVA in both performance and speed, e.g.,  an average performance gain of 3.3% on 5 benchmarks while being 1.61 times faster. Our code is anonymously released at https://github.com/DoubtedSteam/RoE

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Routing Experts (RoE), a novel approach to transform existing multimodal LLMs into mixture-of-experts models without significant architectural changes. The key innovation is treating each layer of pre-trained MLLMs as a potential expert that can be dynamically routed or skipped based on input complexity.

### Strengths
- The paper is well-written and good in coherence.
- The authors present innovative architectural designs that effectively integrate Mixture-of-Depths into Multimodal Large Language Models
- The empirical validation is comprehensive, encompassing diverse model architectures and benchmark datasets.
- The proposed approach is computationally efficient, requiring minimal fine-tuning overhead for implementation.

### Weaknesses
**Major**
- From Table 1, RoE-LLaVA-HR shows a large drop in performance. While the authors note that "LLaVA-HR is more sensitive to network skipping ... Nevertheless, RoE can further improve the compactness." They should explain why this happens and whether the improved compactness is worth the performance loss.
- From Table 2, comparing RoE to *Router* that entirely skips model layers may not be fair enough. The study needs separate tests for each part of RoE (adapter, regularization, and router token) to show how each contributes.
- The sparsity ratio in Table 4 and 5 is not clearly stated, and the inference speed improvements are not very impressive. This raises questions about how well RoE can handle more complex tasks and higher sparsity levels.

**Minor**
- Formatting: Too much empty space under figures and between sections.
- Inconsistent Terms: "L1-Distance" is written differently in Figure 1 and its caption.

### Questions
- While the study demonstrates thorough experiments across different LLaMA-based MLLMs, the generalizability to non-LLaMA architectures (e.g., Qwen) remains unexplored. Testing RoE on diverse language model backbones would better validate its broader applicability.

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
This paper explores how to make a MLLM (multimodal large language model) more efficient. They find that different layers of the MLLM contribute differently to each sample; therefore, the paper proposes adaptively skipping over less important layers. They do this by replacing full layers with adapters, and learning a routing function at each layer that makes a binary decision to use the adapter or the full layer. This effectively converts an existing MLLM into a "mixture of experts". They find that this approach, called RoE, results in significant speedups at inference time while suffering only a slight degradation in accuracy.

### Strengths
Originality: simple way of taking an existing model and making it more efficient/into a MOE by swapping out existing layers with adapters and training a router. 

Quality: results show fairly consistent speedups in MLLMs. RoE also outperforms existing MOEs in accuracy on downstream tasks.

### Weaknesses
Quality: 
- Only one scenario (SQA in table 5) has RoE being strictly better than other models in both accuracy and speed. All other settings exhibit a tradeoff, and it is not unclear how good/bad this tradeoff is. It would be nice if the paper could visualize the pareto frontier. There are also some cases where RoE is slower than dense MLLM counterparts.
- Some additional ablations would be helpful, such as adapter+no router+no reg; that is, just using the adapter at each layer. 
- More analysis would be better, i.e., what is being routed between the adapter and the existing layer (Figure 3d touches on this) 

Clarity: 
- Notation like $\{G_1, G_2, \dots, G_n\}$ are layers chosen by the router. Its number is smaller than the default length $n$ is ambiguous. Should use separate $n$'s. 
- Equation (3) is not properly explained - what is $I$? What is $T$?
- What are the training objectives for Stage 1 and Stage 2 of RoE?

Significance: there are many tradeoffs in the proposed method. Therefore, it is unclear how much people would use this in practice (i.e., do people want to spend ~93.6 GPU hours to make an existing model faster but worse---and not clear by how much)

### Questions
1. Can you provide more visualization on the accuracy/speed tradeoff of each model?
2. Can you provide additional ablations, such as showing what happens when a subset of the layers is deterministically set to use the adapter.
3. Can you provide more examples that are routed to adapters more often versus routed to the existing layer more often? Do these correspond with easier/harder samples?
4. Clarity suggestions.
5. Would welcome more discussion/results on improving and quantifying the tradeoffs made in RoE and its costs.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a method of introducing sparsity in multimodal LLMs by skipping some transformer layers. Importantly the skipped layers have a low-rank adaptor applied, and the exact set of layers skipped or not depends on the input due to a learned routing function. The work is in essence combining ideas of model pruning with MoEs and applying it to multimodal LLMs. The authors introduce several techniques to effectively train this model, such as warming up the routers and adaptors, and enforcing sparsity in the router. They show their model can maintain very close to SOTA accuracy across a variety of multimodal LLMs while increasing throughput by 10-20%.

### Strengths
The paper is clearly written, and the main ideas are simply presented. The work shows good speedups of up to 20% without much degradation of model accuracy.

### Weaknesses
The paper focuses mostly on the connection of their work to MoEs, but not as much on the connection to existing model pruning / layer removal efforts. Also while the paper compares accuracy & speed-up compared to the baseline models, they don't compare to baseline pruning or distillation techniques.

### Questions
Suggestion for improvement is to compare against another technique for model pruning or distillation.

### Soundness
4

### Presentation
4

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
This paper proposes RoE, which skips layers in an existing MLLM achieve efficiency and effectiveness. The router for managing the layer skipping is expected to skip layers that have redundancy. The skipped layer is substitute by an adapter for mitigating feature gap.

### Strengths
1. The paper is well-written and easy to follow

2. The problem solved is interesting and meaningful

3. The proposed method seems to be interesting and effective

### Weaknesses
1. Even though the paper is motivated through MoE, the method is more focusing on layer skipping, which is a generally well-studied field for LLM. There should be a subsection in the related work talking about this field. Moreover, these two papers [1,2] seem to be very relevant and should be compared or discussed. The current version makes it hard to judge the novelty or contribution.

[1] Raposo, David, et al. "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models." arXiv preprint arXiv:2404.02258 (2024).
[2] Tan, Zhen, et al. "DLO: Dynamic Layer Operation for Efficient Vertical Scaling of LLMs." arXiv preprint arXiv:2407.11030 (2024).

2. The major motivation lie in the feature redundancy of layers in the MLLM, as shown in Fig 1. Can the author plot similar figures for the learned RoE model, to show that the redundancy is mitigated?

3. There seem to be no direct supervision signal for calculating the feature similarity and guiding the learning of the router. How to make sure the skipped layers are indeed redundant? Also, can the paper show the training loss to indicate that the convergence of the method?

4. Even though the paper focuses on VLLMs, the major design seems can also be applied to LLMs. It would be interesting to see how this will impact LLMs.

5. Since the redundancy is highly correlated to the hardship of the input instance, how to decide the sparsity before the training? If it's a hyper-parameter for tuning across datasets / tasks, then this might heavily impact the applicability of the proposed method for unseen tasks. Can the authors provide some insights on how to choose the sparsity? Also, the current tasks are more focusing on easier tasks like VQA. Is the method still effective / necessary for newer or harder tasks and benchmarks like grounding or segmentation?

6. Since the efficiency is the major target, can the author provide comparison of actual averaged FLOPs in the experiments, to explicitly show the effectiveness and importance of the proposed method?

### Questions
Please see above.

### Soundness
3

### Presentation
3

### Contribution
2
