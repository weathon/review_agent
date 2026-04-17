# Scaling Large Vision-Language Model RL Training via Efficient Load Balancing

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Reinforcement learning (RL) is increasingly used to align vision--language models (VLMs), yet scaling RL for VLMs is bottlenecked by multimodal data handling and extreme workload skew. In typical RL pipelines, visual data loading and preprocessing are centralized, creating severe I/O and CPU/memory stragglers, while batches that mix short image-text prompts with long video contexts lead to large cross-GPU imbalance during rollouts, inference, and training. We present FlexRL, an end-to-end system that removes these bottlenecks. FlexRL introduces: (1) ShadowLoader, a distributed, metadata-driven pipeline that keeps only lightweight visual metadata on the controller, pushes decoding and preprocessing to worker-side preprocessors, and asynchronously materializes tensors to overlap I/O with GPU computation; (2) FlexUlysses, a cost-aware sub-sequence sharding and execution engine that adaptively splits sequences to balance compute and memory. Our evaluation shows that across multiple VLM scales and multimodal datasets on 128-GPU clusters, FlexRL improves end-to-end throughput by up to 8.47$\times$ over state-of-the-art RL systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents FlexRL, a system designed to address data loading bottlenecks and computational load imbalance in RL training for large multimodal models. Its core contributions are a decentralized data pipeline and a novel hybrid sequence sharding mechanism. The authors claim up to 4.2× speedup on a 128-GPU cluster.

### Strengths
- Accurately identifies key systemic bottlenecks in multimodal RL training.

- The proposed hybrid sequence sharding mechanism is ingenious and offers a promising direction for handling workload skew from heterogeneous sequences.

- The design of a dynamic execution engine to manage the scheduling complexity introduced by hybrid sharding is a significant engineering step.

### Weaknesses
- [Mandatory] There are minor typos, formatting inconsistencies, and grammatical errors. The authors should carefully proofread the manuscript.

- [Mandatory] The paper does not separately evaluate the individual contributions of the two core components: the "Decentralized Data Pipeline" and the "Hybrid Sequence Sharding." It is unclear which component drives the performance gains.

- [Mandatory] The hybrid sharding introduces complex All-to-All communication. While overlap strategies are qualitatively mentioned, a quantitative analysis of the communication overhead's impact on end-to-end performance under different cluster scales and network topologies is missing.

- [Mandatory] Key details regarding the scheduling heuristic and cost estimation model are missing, hindering reproducibility.

### Questions
Please refer to Weaknesses. Btw, I have some optional questions:

- [Optional] Dynamic sequence packing techniques, which concatenate short sequences into longer ones during training, have emerged recently. How does FlexRL fundamentally compare to such methods in terms of load balance efficiency and memory utilization? What are the relative advantages and disadvantages?

- [Optional] The title claims "Universal," but the method heavily relies on All-to-All communication within the attention mechanism. If future VLM architectures shift towards SSMs or other non-attention-based mechanisms, would FlexRL's core sharding mechanism remain effective? What is your view on this architectural dependency risk?

### Soundness
3

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
This paper presents a distributed training pipeline for RL training stage of large-scale VLMs involving multimodal data (images, videos, texts). The key challenge in the multimodal RL training is the highly diverse data length (short text, long text, long image&video tokens), which is hard to be properly scheduled in a distributed training system. It mainly proposes a decentralized data pipeline to properly schedule the data with a single controller, and a hybrid sequence sharding technique to partition sequences into finegrained chunks to enable sub-sequence level load balancing. Existing technique Ulysses Sequence Parallelism is used to enable sequence parallel training.

### Strengths
-	The proposed hybrid sharding technique is novel and alleviate the issue of imbalanced loading.
-	Experiments show the proposed approach outperform the speed of existing approach such as verl on video understanding tasks.

### Weaknesses
-	The speed of running a batch on a gpu should be clarified more. How does the gpu handles sequence with different length in a batch? From figure2, a gpu will pack samples with different lengths into groups and conduct their attention operation separately. While in some implementations using generic sequence packing and masked attention, the running time is irrelevant to the sequence length of each sample since a global masked attention of all tokens in a batch is conducted. How much speed up does the proposed approach have compared to the global-attention method?
-	The paper only provides numbers for speed, while accuracy numbers are not provided. The accuracies of using the proposed approach is also required to show the method’s robustness for different applications.

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
FlexRL is an end-to-end optimization system built on the verl framework to improve the efficiency of RL training for MLLMs. It addresses two primary bottlenecks: (1) a Decentralized Data Pipeline that distributes multimodal data loading and preprocessing across worker nodes while the control node handles only lightweight metadata, eliminating centralized I/O bottlenecks; and (2) a Hybrid Sequence Sharding mechanism that partitions sequences into fine-grained chunks to achieve subsequence-level load balancing, mitigating uneven GPU utilization caused by extreme length disparities across modalities such as text, images, and video.

### Strengths
The paper systematically analyzes practical bottlenecks across the entire RL training pipeline for MLLMs rather than focusing on a single stage, and it demonstrates strong system-level completeness.

### Weaknesses
I must first note that I am not very familiar with mlsys, while I only offer a limited perspective on this paper.

1. Why not compare against other verl-based optimized frameworks, such as [1], which also targets long-video scenarios?
2. The paper lacks concrete ablations; for example, it does not separately quantify the contributions of the Decentralized Data Pipeline and the Hybrid Sequence Sharding components.
3. Is the framework primarily intended for highly imbalanced workloads? The design appears to degenerate to conventional parallelism, but comparisons under balanced workloads (e.g., image-only or pure-text) are missing.
4. Performance under different settings is not reported, e.g., varying batch size and tensor/pipeline/sequence parallelism (TP/PP/SP) sizes.
5. As an mlsys work, the paper does not provide an engineering code release or community usage feedback, which I consider a weakness.

[1] Scaling RL to Long Videos. NeurIPS 2025.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
