# AMDP: Asynchronous Multi-Directional Pipeline Parallelism for Large-Scale Models Training

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 4, 6

## Abstract
Pipeline parallelism has become a critical technique for scaling up the training of large models. However, existing asynchronous pipeline approaches often suffer from degraded convergence due to parameter mismatch between forward and backward passes. To address this, we propose Asynchronous Multi-Directional Pipeline parallelism (AMDP). AMDP limits stage 0 of each pipeline to read only two minibatches before initiating the first backward pass, thereby reducing the number of parameter updates that occur between the forward and backward passes of each minibatch. To mitigate the pipeline bubbles introduced by this restriction, AMDP instantiates multiple concurrent pipelines and adapts their number according to pipeline depth. Furthermore, AMDP accumulates gradients across minibatches and applies them in a single parameter update, ensuring that only a small number of minibatches (bounded by the pipeline depth) encounter parameter mismatch, which is constrained to within one step. Experiments on GPT- and BERT-style models demonstrate that AMDP significantly accelerates the training of large-scale models while preserving convergence. The source code based on Megatron-LM is avaiable at https://anonymous.4open.science/r/Megatron-AMDP-BB23

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces AMDP, a new asynchronous pipeline parallel training technique for LLMs. Asynchronous pipelines improve the hardware utilization while avoiding the degraded convergence caused by parameter mismatch. AMDP imposes a constraint on minibatch preloading (stage 0 reads only two minibatches before its first backward) to bound mismatch, while launching multiple directionally-interleaved pipelines to offset bubble overhead. A gradient accumulation strategy lowers communication cost and further limits mismatch propagation. A Zero-Redundancy Optimizer (ZeRO) is incorporated to reduce optimizer memory overhead. Experiments on GPT- and BERT-like models, running on 8×A800 GPUs, show up to 17% throughput improvement over asynchronous baselines (PipeDream-2BW) with comparable convergence to synchronous methods.

### Strengths
- The paper describes well the trade-off between asynchronous pipeline throughput and convergence degradation due to deep parameter mismatch.
- Multi-directional scheduling with constrained minibatch preload is distinct from prior 1F1B asynchronous strategies and directly combats mismatch scaling with depth.
- Results include comparisons against several state-of-the-art synchronous and asynchronous baselines, covering both throughput and convergence.

### Weaknesses
- The method is categorized as asynchronous, yet Table 1 asserts “stable convergence.” The paper explains that AMDP bounds parameter mismatch to one step through constrained minibatch preload and gradient accumulation, but this justification is primarily empirical. There is no theoretical analysis demonstrating that a one-step mismatch is universally sufficient for stability, nor evaluation on deeper or multi-node pipelines where mismatch and communication delays may increase. The current results suggest stability under the tested configuration, rather than establishing a general convergence guarantee.
-  In relation to the point above, the bound on mismatch (1 step) is argued via scheduling behavior, but there is no formal convergence analysis or error bound.
- Multi-directional scheduling scales the number of pipelines with depth (d/2). Evaluation does not include large-cluster settings, where communication contention and topology effects could be significant.
- Introduction of multiple pipelines and reduce/broadcast synchronization is claimed to be modest overhead, but no detailed breakdown or sensitivity study is provided.

### Questions
- How does AMDP behave with significantly deeper pipelines (e.g., d > 16) and multiple nodes with InfiniBand instead of NVLink?
- Is stage imbalance handling general or dependent on Transformer-specific layer normalization and embedding behavior?
- What are the conditions where the middle bubble in multi-directional schedules dominates and limits benefit?
- Would mismatch-aware adaptive learning-rate scaling further improve stability?

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
The authors propose an asynchronous pipeline parallelism approach that bounds the parameter mismatch between forward and backward passes. To this end, they apply multiple pipelines in parallel where each GPU holds multiple stages of the model. They also propose a zero redundancy optimizer where the updates for a particular stage only happens in a dedicated GPU. The throughput gain shown is non trivial and impactful.

### Strengths
Increasing the efficiency of pipeline parallelism is  an important problem in distributed training. Synchronous methods suffer from GPU bubbles and many asynchronous methods suffer from delayed gradients and parameter mismatch between forward and backward passes. Therefore, the proposed method in the paper is compelling and the results are encouraging. If released in an easy-to-use software package, I believe this work will be useful to the community.

### Weaknesses
The method seems to be heavily reliant on the multi directional pipeline concept proposed in Chimera (although Chimera is synchronous). This limits the contribution of the work in my opinion. Also, although the authors cite Chimera, in the introduction the narrative presents multi directional pipeline assignment as one of the contributions of this paper. It is better to refine the introduction to make this dependency more clear. Also, some recent asynchronous pipeline parallel methods are missing from the experiments.

### Questions
Can you compare the proposed work with the method presented in "Nesterov Method for Asynchronous Pipeline Parallel Optimization"?

There is an extra communication time associated with zero redundancy optimizer. How does this affect the overall throughput. More ablations around this would be helpful.

Is the method scalable, since multiple stages have to be assigned to the same GPU?

In Fig. 8, why is there a gap between your method and the synchronous method at the beginning which gradually closes as the training progresses? Can the authors provide any insight into this?

### Soundness
2

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
3

### Summary
AMDP is an async pipeline-parallel scheme building upon previous works like DAPPLE/Inter1F1B/Chimera (sync) and PipeDream(2BW)/XPipe (async). It is empirically convergent via a staleness cap between a minibatch's forward and backward passes at most one update step. Additionally, the author employs a technique termed "multi-directional scheduling", where the algorithm runs two stage-0 forward reads before the first backward pass, followed by multiple concurrent oppositely directed pipelines. Then, gradients are accumulated and updated per-stage in a synchronized manner during bubbles, and optimizers are sharded ZeRO-style. On GPT/BERT-style models, AMDP has marginal extra memory, and up to 17% higher throughput over strong baselines, and is reportedly numerically convergent.

### Strengths
* The paper is fairly well written with solid experimental results, good schematics, and no major grammatical flaw. This is a satisfactory system work, and I in particular like the related work section.
* The proposed method shows understanding of PP and is both novel and simple enough to implement.

### Weaknesses
* I can't access https://anonymous.4open.science/r/AMDP-4915 so I can't verify whether the method actually works or produces the reported speedup. None of the files appear to be available. Megatron-LM is a fairly complex codebase and I'm curious to see where the authors have made changes
* In my opinion, PP is overall a complicated technique that only has great benefits at very large scale, and in those regime, numerical stability and healthy learning dynamics really matters. Asynchronous methods fundamentally has the challenge of disrupting such dynamics and adds risks to a large model training run, which is arguably why async methods rarely are considered in larger runs. A major weakeness of this work is that the convergence is only empirically observed in a small-scale experiment, with no theoretical groundings, and the additional gain is only 17% over (async) baselines. For many use cases, this would be not good enough.
* Another minor nit is that the hardware and software descriptions are too vague to make replications meaningful. A server cannot simply be described by the GPU make and "NVLink."

### Questions
I don't have additional questions beyond the inaccessibility of the code.

### Soundness
3

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
4

### Summary
The paper proposes an idea to improve pipeline utilization while keeping the staleness low for asynchronous PP methods. The main idea is to use multi-directional microbatch scheduling to improve utilization. The results validate the method and at a high level it make sense from a convergence perspective as staleness is kept low.

### Strengths
1. The idea is clever -- an engineering solution via multi-directional pipeline scheduling to keep staleness low and increase pipeline utilization. Although the implementation might be complicated.
2. Results show that the proposed method achieves higher throughput compared to other synchronous and asynchronous PP approaches, while ensuring similar convergence.
3. The main message is conveyed clearly with illustrations.

### Weaknesses
1. Some staleness correction methods not discussed (eg., [1] and citations thereof). This is relevant, as [1] showed that async pp methods can match synchronous methods in terms of per-iteration convergence. 
2. Related to the above, Pipedream and similar methods (including [1]) can increase the update interval (updating every K microbatches instead of 1) to keep staleness low while ensuring high pipeline utilization. This is an important comparison and needs to be added to position this paper correctly.
3. In the multi-directional scheduling, each GPU needs to hold other stage weights and process forward-backward for multiple stages in an alternating fashion. How is this implemented in practice without increasing the peak memory usage? If the weights are offloaded to cpu and back, wouldn't it be an additional overhead?

[1] Ajanthan, T, et al. "Nesterov Method for Asynchronous Pipeline Parallel Optimization." ICML, 2025.

### Questions
1. What is the peak memory usage and computation time, including off-loading to cpu and back (if)?

### Soundness
3

### Presentation
3

### Contribution
3
