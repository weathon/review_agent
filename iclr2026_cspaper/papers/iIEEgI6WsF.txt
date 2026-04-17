# Revisiting Parameter Server In Llm Post- Training

Xinyi Wan1,2∗
, Penghui Qi1,2∗
, Guangxing Huang1, Chaoyi Ruan2, Min Lin1 **& Jialin Li**2 1Sea AI Lab 2National University of Singapore

## Abstract

Modern data parallel (DP) training favors collective communication over parameter servers (PS) for its simplicity and efficiency under balanced workloads. However, the balanced workload assumption no longer holds in large language model (LLM) post-training due to the high variance in sequence lengths. Under imbalanced workloads, collective communication creates synchronization barriers, leading to under-utilization of devices with smaller workloads. This change in training dynamics calls for a revisit of the PS paradigm for its robustness to such imbalance. We propose **On-Demand Communication (ODC)**, which adapts PS into Fully Sharded Data Parallel (FSDP) by replacing collective all-gather and reduce-scatter with direct point-to-point communication. Compared to FSDP, ODC reduces the synchronization barrier from once per layer to once per minibatch and decouples the workload on each device so that faster workers are not stalled. It also enables simpler and more effective load balancing at the minibatch level. Across diverse LLM post-training tasks, ODC consistently improves device utilization and training throughput, achieving up to a 36% speedup over standard FSDP. These results demonstrate that ODC is a superior fit for the prevalent imbalanced workloads in LLM post-training. Our implementation of ODC and integration with FSDP is open-sourced at https://github.com/sail-sg/odc.

## 1 Introduction

The development of DP distributed training (Krizhevsky, 2014; Goyal et al., 2017; Li et al., 2020)
has followed two main approaches: the PS architecture and collective communication. Early largescale systems such as DistBelief used the PS model to train deep neural networks across heterogeneous hardware and networks with variable latencies (Dean et al., 2012). In this setup, servers stored the model parameters while workers handled computation, enabling asynchronous or loosely synchronous training that tolerated slower or unreliable machines. Later work expanded on this design by enabling different consistency policies and exploring elastic scalability with continuous fault tolerance (Li et al., 2014). With the emergence of dense, homogeneous GPU clusters and highbandwidth interconnects, collective communication became the mainstream approach for distributed DP. A prominent advantage of this paradigm was the opportunity it created for communicationefficient algorithms. Ring-based methods, as demonstrated in Baidu AllReduce (Research, 2017) and Horovod (Sergeev & Del Balso, 2018), reduced bandwidth requirements while scaling predictably. This trend was further reinforced by vendor-optimized libraries like NCCL (NVIDIA, b), which made high-performance collectives broadly accessible and easy to integrate into modern training frameworks. It is important to note that the high efficiency of collective communication fundamentally relies on balanced workloads. This presumption was largely valid for many dominant deep learning domains, including vision, speech, and early NLP. As a result, the dependency on workload balance was frequently taken for granted or neglected in system design. Recently, the post-training of LLMs (Ouyang et al., 2022; Guo et al., 2025) breaks the long-standing assumption of balanced workloads that collective communication relies on. Real-world text corpora contain sequences of widely varying lengths (Bai et al., 2024; Yang et al., 2025). As the cost of attention grows quadratically with sequence length (Vaswani et al., 2017) while activation memory grows linearly, this variation leads to persistent computational imbalance across devices. Although
∗Equal Contributors 1 a line of work has focused on mitigating this issue with sophisticated packing strategies (Krell et al., 2021; Kundu et al., 2024; Yao et al., 2025; Wang et al., 2025), these methods can only reduce the skew, but cannot remove it entirely, especially under memory constraints that force minibatches to be split into smaller microbatches (Huang et al., 2019; Qi et al., 2024). This not only narrows the solution space for effective packing, but also increases the number of synchronization points, further amplifying the inefficiency due to imbalanced workloads. This inefficiency from workload imbalance is particularly severe in contemporary sharded DP, exemplified by ZeRO (Rajbhandari et al., 2020) and PyTorch's FSDP (Zhao et al., 2023). By sharding parameters, gradients, and optimizer states across devices, FSDP enables memory-efficient scaling to trillion-parameter models, making it the standard choice for LLM post-training and reinforcement learning (RL) pipelines (Hu et al., 2024; Sheng et al., 2025; Fu et al., 2025; Liu et al., 2024). However, this memory efficiency comes at the cost of increased synchronization (Figure 1). FSDP relies heavily on collective communication: per-layer parameters are reconstructed via *all-gather* before the forward pass, and gradients are aggregated via *reduce-scatter* after the backward pass. This fine-grained, layer-level synchronization implicitly assumes balanced workloads, which is precisely the assumption violated in LLM post-training. Our evaluation shows that even with state-of-the-art packing strategies, workload imbalance can still result in device idle times of up to 50% during long-sequence supervised fine-tuning (see Table 6). To bridge the gap between fine-grained synchronization and workload imbalance in LLM posttraining, we revisit the PS idea, and adapt it to the modern sharded DP paradigm through Ondemand Communication (ODC). We replace the per-layer collectives with point-to-point primitives, allowing devices to fetch parameters and push gradients independently (Figure 2). This reframes FSDP as a decentralized PS where server and worker roles are colocated, thus preserving its memory and scaling advantages. While preserving the synchronous optimization semantics, we relax synchronization from the layer level to the minibatch level. This decoupling of device progress significantly mitigates straggler effects and enables a more flexible space for workload balancing. In summary, this paper presents a novel perspective: compared to collectives, the PS architecture is naturally better suited for LLM post-training due to its tolerance for heterogeneous workloads. To retain the key benefits of modern DP schemes, we do not build a standalone PS. Instead, we propose ODC, a communication scheme that brings the workload-tolerance of classic PS into FSDP. Our evaluation demonstrates that ODC substantially improves device utilization and end-to-end throughput across diverse LLM post-training tasks, including supervised fine-tuning (SFT) and RL, achieving up to 36% speedup over conventional FSDP.

![1_image_0.png](1_image_0.png)

Figure 1: Collective communications introduces per-layer synchronization barriers in FSDP.

![1_image_1.png](1_image_1.png)

## 2 Background 2.1 Minibatch, Microbatch And Gradient Accumulation

In deep learning, a minibatch refers to the set of training samples processed in a single optimizer step. However, training LLMs often exceeds the memory capacity required to process the desired minibatch in one forward–backward pass. A common remedy is to divide the minibatch into M microbatches and accumulate gradients before performing the optimizer update. For each microbatch m ∈ 1*, . . . , M*, we compute the forward and backward passes to obtain per-parameter gradients g
(m), and then accumulate g¯ =PM
m=1 wm g
(m), where wm encodes the aggregation policy (e.g.,
wm = 1 for summation, or proportional weighting when averaging by tokens or samples).

## 2.2 Synchronization Barriers In Fsdp

![2_image_0.png](2_image_0.png)

In FSDP, both parameters and gradients are partitioned across devices. FSDP primarily uses allgather to materialize parameters and *reduce-scatter* to aggregate gradients. The mechanics of reduce-scatter and *all-gather* are illustrated in Figure 3. The communication pattern unfolds as follows. During the forward pass, before computation on a specific layer begins, its full parameters are reconstructed on each device via an *all-gather* operation. These reconstructed parameters are then discarded immediately after use to save memory. A similar all-gather process occurs during the backward pass. Additionally, after gradients are computed for a layer, they are aggregated and distributed using a *reduce-scatter* operation, leaving each device with only its corresponding shard of the total gradient. The overall communication flow is shown in Figure 4. In practice, modern implementations overlap these communications with computation (e.g., pre-fetching parameters for the next layer during the current layer's execution) to hide the latency, but this overlap does not remove the underlying synchronization points.

![2_image_1.png](2_image_1.png)

Figure 4: Communication pattern of FSDP within a microbatch. The left panel shows forward communication (*all-gather* parameters), and the right shows backward communication (*all-gather* parameters & *reduce-scatter* gradients). AG = *all-gather*; RS = *reduce-scatter*.

These per-layer collectives create fundamental synchronization barriers that are the root cause of inefficiency under imbalanced workloads. All devices must complete the *all-gather* before a layer's forward computation can begin, and they must all complete the *reduce-scatter* before gradient accumulation can proceed. This tight coupling forces all devices to advance at the same pace, meaning faster devices must idle and wait for the slowest one before moving to the next layer.

More formally, let a batching solution PM specify the assignment of training samples to M microbatches on each device. Denote by T*m,d,l*(PM) the time to execute layer l of microbatch m on device d under PM. For a model with L layers, the minibatch runtime is bounded by the slowest device at each per-layer step:

$$T({\mathcal P}_{M})\;=\;\sum_{m=1}^{M}\sum_{l=1}^{L}\operatorname*{max}\;T_{m,d,l}({\mathcal P}_{M}).$$

$$(1)$$
dT*m,d,l*(PM). (1)
A significant body of research has focused on finding an optimal batching solution, P
⋆, that minimizes T(PM). However, as we detail in Section 4, these approaches face fundamental limitations.

## 3 On-Demand Communications

To address the inefficiency of FSDP caused by imbalanced workload, we step back from the prevailing focus on complex batching strategies and re-examine a first principle of data parallelism: per-device computations are independent. Standard FSDP violates the spirit of this independence by using collective communication, which imposes fine-grained synchronization barriers. These barriers, which force devices to wait for the slowest one, are the direct cause of idle time. They are an artifact of the communication model, not a requirement of the training algorithm itself, and are therefore fundamentally **avoidable**. To address this root cause, we propose ODC, a new communication scheme that relaxes synchronization to a much coarser granularity without altering the training semantics (Figure 2). ODC preserves FSDP's memory layout and computational graph but replaces its synchronous collectives with point-to-point operations. Specifically, we decompose the collective calls. An *all-gather* is replaced by a series of targeted *gather* requests, where a device fetches only the specific parameter shards it needs from its peers. Similarly, a *reduce-scatter* is broken down into a series of scatteraccumulate operations, where a device pushes its computed gradients directly to the devices that own the corresponding gradient shards. This process is illustrated in Figures 5. With ODC, each device operates independently, fetching parameters or pushing gradients as soon as it is ready, thereby eliminating the synchronization-induced stalls. A critical feature of ODC is that these point-to-point data transfers are non-intrusive. When one device initiates a gather or *scatter-accumulate* request to another, it does not interrupt the ongoing computation on the target device. We show how this is enabled in Section 3.2.

![3_image_0.png](3_image_0.png)

## 3.1 Odc As A Decentralized Parameter Server

The classic PS architecture (Dean et al., 2012; Li et al., 2014) separates model state from model computation, where a set of server nodes is responsible for storing the model's parameters and optimizer states. Meanwhile, a set of worker nodes pulls parameters from the servers, performs the forward and backward computations on its local data, and then pushes the resulting gradients back to the servers. The servers then aggregate these gradients and apply the updates. This design decouples the progress of individual workers and provides a natural tolerance for stragglers, which is a key advantage for the imbalanced workloads common in LLM post-training. As shown in Figure 6, ODC paradigm reframes FSDP as a modern, decentralized PS. Instead of using dedicated server nodes, we colocate the server and worker roles by evenly partitioning parameters, gradients, and optimizer states across all devices. Each device acts as a server by owning and managing a shard of the model's parameters and optimizer state. Simultaneously, it acts as a worker by executing the forward and backward passes on its assigned data. This decentralized, co-located design mirrors the memory layout of FSDP and avoids the network bottlenecks of a centralized PS.

While colocated roles has precedent in some PS systems (Jiang et al., 2020), our approach is novel in its direct integration with FSDP's sharding mechanism. Ultimately, by replacing FSDP's per-layer collectives with on-demand point-to-point communication, our method gains the imbalance tolerance of a PS while retaining the core benefits of FSDP: memory efficiency, decentralization, scalability, and simplicity.

![4_image_0.png](4_image_0.png)

## 3.2 Implementation

ODC workers often push or pull data to servers while colocated workers concurrently perform computations, making it essential to minimize server interference. Communication primitives must also support ODC's on-demand nature, where workers control the flow and servers cannot anticipate requests. Existing message-based libraries like MPI (Gabriel et al., 2004) and NCCL(NVIDIA, b)
require explicit, ordered participation from both sender and receiver, making them neither transparent nor on-demand, and prone to deadlocks if not carefully scheduled. ODC instead leverages native RDMA-based interfaces: CUDA IPC (NVIDIA, a) for intra-node and NVSHMEM (NVIDIA, c) for inter-node communication. RDMA enables transparent data transfers without active server involvement, except for gradient accumulation, which is handled by a lightweight daemon. The communication kernel is built on Triton-Distributed (Zheng et al., 2025), a Triton (Tillet et al., 2019) wrapper that exposes RDMA functionalities directly in Python Triton kernels, eliminating the need for low-level CUDA C code. We put more implementation details at Appendix B, and will open-source our implementation for community usage. Integrating ODC into FSDP is straightforward: it only requires replacing collective communication calls with ODC primitives and retrieving accumulated gradients at the minibatch end.

## 4 Simplified Load Balancing With Odc

Due to the variation in sequence lengths, a naive padding strategy significantly suffers from computation waste. To mitigate this, Krell et al. (2021) introduced the strategy of sequence packing, which concatenates multiple samples into a single sequence with appropriate attention masks, improving utilization and balancing workload across microbatches. This approach has been broadly adopted and extended by subsequent work (Bai et al., 2024; Kundu et al., 2024; Yao et al., 2025; Wang et al., 2025), with efficient support in modern libraries like FlashAttention (Dao et al., 2022; Dao, 2023). However, existing sequence packing methods operate at the microbatch level, which faces several fundamental limitations under FSDP. First, the size of a microbatch is bounded by device memory, limiting the number of samples per microbatch and leaving substantial variance in workload across devices. This effect is amplified in long-sequence training regimes, such as LongAlign (Bai et al., 2024) and RL for LLM reasoning (Guo et al., 2025), where extended contexts further constrain per-device capacity. Second, for a sample of sequence length s, activation memory typically scales as O(s) while runtime scales as O(s 2) (e.g., due to attention), creating a fundamental mismatch between memory and compute. Consequently, compute alignment can be infeasible under memory constraints. For instance, if a microbatch contains a single sample at the maximum sequence length, no feasible packing of shorter samples can match its runtime. By replacing collective operations with ODC, our approach decouples the execution of microbatches across devices. This eliminates synchronization barriers inherent in FSDP and removes the implicit requirement for a uniform number of microbatches per device. This insight allows for a significant simplification of workload balancing strategy. Specifically, our strategy shifts the balancing objective from the fine-grained microbatch level to the coarser minibatch level. We first partition the global set of training samples across devices with the sole goal of balancing the total computational load. Subsequently, each device independently packs its local subset of samples into microbatches, governed only by its local memory constraints. This shift in granularity not only simplifies the packing algorithm, but also achieves superior load balancing by operating on a larger, less constrained set of samples. We leave the detailed packing algorithms in Appendix C.

## 5 Evaluations 5.1 Setup

We evaluate ODC on two major LLM post-training tasks: SFT and RL. For SFT, we use a) LongAlign (Bai et al., 2024), a dataset for extending LLM context windows, and b) open-source trajectories from SWE-Smith (Yang et al., 2025), an agent model for software engineering tasks released by the SWE-Bench team (Jimenez et al., 2023). For RL, we run GRPO (Guo et al., 2025; Liu et al., 2025) implemented in verl (Sheng et al., 2025) on AIME prompts (Li et al., 2024), which includes problems from Olympiad-level math contest. Notably, we only record the model training time in RL, ignoring forward-only parts like actor rollout. The sequence length distributions of these datasets are shown in Figure 7.

![5_image_0.png](5_image_0.png)

We evaluate ODC on the DeepSeek-R1-Distill-Qwen family of models (Team, 2024; Guo et al., 2025), with varying size from 1.5B to 32B. The models are trained on up to 32 NVIDIA A100 80G GPUs, with NVSwitch for intra-node communication and RoCE RDMA (800 Gbps per node) for inter-node communication. Notably, for RL experiment we run only up to 14B model using 16 GPUs, as the inference time would be too long for a 32B model. Additionally, we validate the correctness of ODC by verifying the training convergency in Appendix F. Each method in our evaluation is a combination of communication scheme and load balancing algorithms. For communication scheme, we have a) *Collective* - baseline using collective *all-gather* and reduce-scatter; b) ODC - our approach introduced in Section 3; For load balance algorithms, we include a) *LocalSort* - adapted from Bai et al. (2024); within each device's minibatch, sequences are sorted by length but not packed. b) *LB-Micro* - a heuristic-based packing baseline designed to minimize workload imbalance across devices within the same microbatch. In RL experiments, we show that it is substantially faster than the native implementation in verl (Sheng et al., 2025), underscoring its effectiveness as a strong baseline. c) *LB-Mini* - our algorithm introduced in Section 4, which balances workload at the minibatch level. As LB-Mini can produce different number of microbatches for different devices, it applies only to ODC. Detailed implementations can be found in Appendix C. Unless otherwise specified, the maximum number of tokens in a microbatch is constrained by the maximum sequence length of a single sample in the dataset.

![6_image_0.png](6_image_0.png) 

![6_image_1.png](6_image_1.png)

## 5.2 Main Results

Figure 8 presents the evaluation results on SFT tasks. ODC consistently improves throughput over the collective baseline in both unpacked (LocalSort) and packed (LB-Micro, LB-Mini) settings, with the most pronounced gains observed under packing, reaching up to a 36% speedup. All methods perform similarly when the minibatch size is one, since in this case ODC synchronizes after every sample, just like collective. Figure 9 shows in RL tasks ODC achieves up to 10% speedup over collective baseline, although the gains are less pronounced than in SFT. This is primarily due to: a) implementation constraints in verl, which require identical numbers of samples per device and thus limit the effectiveness of LB-
Mini. While relaxing this constraint is feasible, we did not do so, as the current solution is easier to integrate; and b) a less long-tailed sequence length distribution compared to SFT datasets (Figure 7). At small minibatch sizes, LB-Mini often outperforms LB-Micro. This reflects the benefits of its minibatch-level balancing, which permits devices to process different numbers of microbatches. As the minibatch size increases, however, LB-Micro has more flexibility to balance workloads effectively, which narrows the performance gap between the two methods. The detailed timing data as well as bubble rate is reported in Appendix G.

## 5.3 Parametric Study

The effectiveness of ODC compared to collectives depends on several factors: a) Minibatch size: the number of samples per minibatch per device; b) Max length: the maximum sequence length in the dataset; to control this factor while maintaining the overall distribution, we adjust each sample by uniformly truncating or repeating tokens at a fixed ratio; c) Packing ratio: the maximum number of tokens allowed in a microbatch divided by the max sequence length (e.g., with a max sequence length of 16K and packing ratio of 2, a microbatch may contain up to 32K tokens); d) Devices: the total number of devices. To isolate the impact of each factor, we adopt a controlled methodology: starting from a fixed golden setting (Table 1), we vary one factor at a time while holding others constant. As shown in Figure 10, the acceleration ratio peaks at moderate minibatch sizes before declining as larger batches give the baseline more flexibility; it increases with sequence length, since longer sequences amplify the quadratic compute cost and exacerbate imbalance; it decreases with packing ratio, which improves the baseline's packing efficiency; and it grows with the number of devices, as more devices introduce greater heterogeneity.

| Model   | Dataset             | minibatch Size   | Devices   | Packing Ratio   |
|---------|---------------------|------------------|-----------|-----------------|
| 1.5B    | LongAlign (Max 64K) | 4                | 8         | 1               |

![7_image_0.png](7_image_0.png)

Table 1: Golden setting for the parametric study. Each experiment varies at most one factor.

## 5.4 Benchmark On Communication Primitives

![7_image_1.png](7_image_1.png)

We compare the bandwidth of ODC primitives (*gather* and *scatter-accumulate*) against collectives
(*all-gather* and *reduce-scatter*) in NCCL. For fairness, ODC primitives are launched synchronously: each device issues operations in the same order, with barriers inserted before and after each primitive. Results are shown in Figure 11. Within a single node (up to 8 devices), ODC achieves bandwidth comparable to collective. However, once communication spans multiple nodes, ODC lags significantly behind collective. We leave more discussion and how to mitigate this inter-node inefficiency in Section 6.

8

## 6 Discussion 6.1 Challenges On Inter-Node Communication Efficiency

Collective primitives are often highly optimized by exploiting hierarchical interconnects in multinode settings. For example, an all-gather operation might first perform an inter-node broadcast followed by an intra-node broadcast to minimize costly inter-node traffic. ODC does not increase communication volume, but changes the topology: it uses point-to-point RDMA and thus forgoes these hierarchical optimizations (see Appendix D). However, we argue that larger DP scale typically amplifies straggler effects under imbalance, increasing the benefit of ODC's decoupled progress (see Figure 10). Furthermore, several ways can effectively mitigate this communicate overhead. Overlapping Communication with Computation. ODC retains the standard FSDP optimization of overlapping communication with computation. This is particularly effective because communication volume per microbatch is constant with sequence length (s), whereas computation scales as O(s 2). For long sequences, the large computational cost effectively hides the communication latency. Consequently, despite using a non-hierarchical communication pattern, ODC shows no significant slowdown in our long-context evaluations (see Section 5.2). Hybrid Sharding. When the tokens per microbatch is too small to hide communication costs, hybrid sharding provides an effective solution. Similar to ZeRO++ (Wang et al., 2024), parameters and gradients are sharded only *within* a node, while optimizer states remain sharded *across* nodes. This design eliminates cross-node parameter *gather* and gradient *scatter-accumulate*, at the cost of higher per-node memory usage, which is a manageable trade-off given that activation memory requirements are lower. As shown in Appendix E, this strategy effectively mitigates ODC's additional overhead.

## 6.2 Future Work

ODC is an initial effort toward adapting PS to modern sharded DP. We believe this is a foundational step that opens several promising directions for future research. ODC-specific Optimizations While our current ODC implementation uses direct point-to-point communication, its communication graph can be further optimized. For instance, a device could fetch a parameter shard from a peer on the same node that has already cached it, effectively creating a hierarchical communication path similar to topology-aware collectives. Relaxing Synchronization Guarantees Our current design intentionally preserves a synchronous update at the minibatch boundary to maintain identical training semantics. However, this barrier could be relaxed. Extending ODC to support classic asynchronous SGD schemes (Recht et al., 2011), such as bounded-staleness updates (Chen et al., 2016; Ho et al., 2013), could further reduce idle time and improve hardware utilization, particularly in highly heterogeneous environments. This would, however, require a careful analysis of the convergence implications for LLM training.

Elasticity and Fault Tolerance A significant advantage of PS-style architectures is their natural support for elasticity and fault tolerance (Dean et al., 2012; Li et al., 2014). Collective-based systems, in contrast, are notoriously brittle and difficult to resize (Jiang et al., 2020; Narayanan et al., 2021; Duan et al., 2024). Integrating these capabilities into ODC would improve the resilience and flexibility of large-scale, long-running LLM training jobs.

## 7 Conclusion

This paper revisits PS and adapts its principles to solve a critical bottleneck in modern sharded DP training for LLM post-training. We identified that the per-layer *all-gather* and *reduce-scatter* collectives in FSDP create fine-grained synchronization barriers, which amplify the straggler effects caused by workload imbalance.

We proposed ODC to replace these collectives with point-to-point operations, effectively relaxing synchronization from the layer level to the minibatch level. This approach, which reframes FSDP as a decentralized PS, decouples device execution and enables more effective load balancing. Empirically, ODC delivers consistent throughput and utilization gains across a range of long-sequence SFT and RL tasks.

## Reproducibility Statement

To ensure reproducibility of our experiments, we open-source our implementation, including: a) the core communication library of ODC, and b) the code patch that integrates ODC into FSDP at https://github.com/sail-sg/odc..

## References

Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. Longalign: A recipe for long context alignment of large language models. arXiv preprint arXiv:2401.18058, 2024.

Jianmin Chen, Xinghao Pan, Rajat Monga, Samy Bengio, and Rafal Jozefowicz. Revisiting distributed synchronous sgd. *arXiv preprint arXiv:1604.00981*, 2016.

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. *arXiv* preprint arXiv:2307.08691, 2023.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory- ´
efficient exact attention with io-awareness. *Advances in neural information processing systems*,
35:16344–16359, 2022.

Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Marc'aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang, et al. Large scale distributed deep networks. Advances in neural information processing systems, 25, 2012.

Jiangfei Duan, Shuo Zhang, Zerui Wang, Lijuan Jiang, Wenwen Qu, Qinghao Hu, Guoteng Wang, Qizhen Weng, Hang Yan, Xingcheng Zhang, et al. Efficient training of large language models on distributed infrastructures: a survey. *arXiv preprint arXiv:2407.20018*, 2024.

Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, Jiashu Wang, et al. Areal: A large-scale asynchronous reinforcement learning system for language reasoning. *arXiv preprint arXiv:2505.24298*, 2025.

Edgar Gabriel, Graham E Fagg, George Bosilca, Thara Angskun, Jack J Dongarra, Jeffrey M
Squyres, Vishal Sahay, Prabhanjan Kambadur, Brian Barrett, Andrew Lumsdaine, et al. Open mpi: Goals, concept, and design of a next generation mpi implementation. In *European Parallel* Virtual Machine/Message Passing Interface Users' Group Meeting, pp. 97–104. Springer, 2004.

Priya Goyal, Piotr Dollar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, An- ´
drew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. *arXiv preprint arXiv:1706.02677*, 2017.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

Qirong Ho, James Cipar, Henggang Cui, Seunghak Lee, Jin Kyu Kim, Phillip B Gibbons, Garth A
Gibson, Greg Ganger, and Eric P Xing. More effective distributed ml via a stale synchronous parallel parameter server. *Advances in neural information processing systems*, 26, 2013.

Jian Hu, Xibin Wu, Wei Shen, Jason Klein Liu, Zilin Zhu, Weixun Wang, Songlin Jiang, Haoran Wang, Hao Chen, Bin Chen, et al. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework. *arXiv preprint arXiv:2405.11143*, 2024.

Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, et al. Gpipe: Efficient training of giant neural networks using pipeline parallelism. *Advances in neural information processing systems*, 32, 2019.

Yimin Jiang, Yibo Zhu, Chang Lan, Bairen Yi, Yong Cui, and Chuanxiong Guo. A unified architecture for accelerating distributed {DNN} training in heterogeneous {GPU/CPU} clusters. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20), pp. 463–479, 2020.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

Narendra Karmarkar and Richard M Karp. *The differencing method of set partitioning*. Computer Science Division (EECS), University of California Berkeley, 1982.

Mario Michael Krell, Matej Kosec, Sergio P Perez, and Andrew Fitzgibbon. Efficient sequence packing without cross-contamination: Accelerating large language models without impacting performance. *arXiv preprint arXiv:2107.02027*, 2021.

Alex Krizhevsky. One weird trick for parallelizing convolutional neural networks. arXiv preprint arXiv:1404.5997, 2014.

Achintya Kundu, Rhui Dih Lee, Laura Wynter, Raghu Kiran Ganti, and Mayank Mishra. Enhancing training efficiency using packing with flash attention. *arXiv preprint arXiv:2407.09105*, 2024.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. *Hugging Face repository*,
13(9):9, 2024.

Mu Li, David G Andersen, Alexander Smola, and Kai Yu. Communication efficient distributed machine learning with the parameter server. *Advances in neural information processing systems*, 27, 2014.

Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, et al. Pytorch distributed: Experiences on accelerating data parallel training. *arXiv preprint arXiv:2006.15704*, 2020.

Zichen Liu, Changyu Chen, Xinyi Wan, Chao Du, Wee Sun Lee, and Min Lin. Oat: A researchfriendly framework for llm online alignment. https://github.com/sail-sg/oat, 2024.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, et al. Efficient large-scale language model training on gpu clusters using megatron-lm. In Proceedings of the international conference for high performance computing, networking, storage and analysis, pp. 1–15, 2021.

NVIDIA. Cuda c++ programming guide. https://docs.nvidia.com/cuda/
cuda-c-programming-guide/, a. n.d.

NVIDIA. Nvidia collective communications library (nccl). https://developer.nvidia.

com/nccl, b. n.d.

NVIDIA. Nvidia openshmem library (nvshmem) documentation. https://docs.nvidia.

com/nvshmem/api/index.html, c. n.d.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. *Advances in neural information processing systems*, 35: 27730–27744, 2022.

Penghui Qi, Xinyi Wan, Guangxing Huang, and Min Lin. Zero bubble (almost) pipeline parallelism.

In *The Twelfth International Conference on Learning Representations*, 2024.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–16. IEEE, 2020.

Benjamin Recht, Christopher Re, Stephen Wright, and Feng Niu. Hogwild!: A lock-free approach to parallelizing stochastic gradient descent. *Advances in neural information processing systems*, 24, 2011.

Baidu Research. Baidu allreduce. https://github.com/baidu-research/
baidu-allreduce, 2017.

Alexander Sergeev and Mike Del Balso. Horovod: fast and easy distributed deep learning in tensorflow. *arXiv preprint arXiv:1802.05799*, 2018.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pp. 1279–1297, 2025.

Qwen Team. Qwen2 technical report. *arXiv preprint arXiv:2407.10671*, 2024. Philippe Tillet, Hsiang-Tsung Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In *Proceedings of the 3rd ACM SIGPLAN International* Workshop on Machine Learning and Programming Languages, pp. 10–19, 2019.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Guanhua Wang, Heyang Qin, Sam Ade Jacobs, Xiaoxia Wu, Connor Holmes, Zhewei Yao, Samyam Rajbhandari, Olatunji Ruwase, Feng Yan, Lei Yang, et al. Zero++: Extremely efficient collective communication for large model training. In The Twelfth International Conference on Learning Representations, 2024.

Zheng Wang, Anna Cai, Xinfeng Xie, Zaifeng Pan, Yue Guan, Weiwei Chu, Jie Wang, Shikai Li, Jianyu Huang, Chris Cai, et al. Wlb-llm: Workload-balanced 4d parallelism for large language model training. *arXiv preprint arXiv:2503.17924*, 2025.

John Yang, Kilian Leret, Carlos E Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, Binyuan Hui, Ofir Press, Ludwig Schmidt, and Diyi Yang. Swe-smith: Scaling data for software engineering agents. *arXiv preprint arXiv:2504.21798*, 2025.

Yongqiang Yao, Jingru Tan, Kaihuan Liang, Feizhao Zhang, Yazhe Niu, Jiahao Hu, Ruihao Gong, Dahua Lin, and Ningyi Xu. Hierarchical balance packing: Towards efficient supervised finetuning for long-context llm. *arXiv preprint arXiv:2503.07680*, 2025.

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. *arXiv preprint arXiv:2304.11277*, 2023.

Size Zheng, Wenlei Bao, Qi Hou, Xuegui Zheng, Jin Fang, Chenhui Huang, Tianqi Li, Haojie Duanmu, Renze Chen, Ruifan Xu, et al. Triton-distributed: Programming overlapping kernels on distributed ai systems with the triton compiler. *arXiv preprint arXiv:2504.19442*, 2025.