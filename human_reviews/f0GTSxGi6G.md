# Hardware Scaling Trends and Diminishing Returns in Large-Scale Distributed Training

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3, 5, 3

## Abstract
Dramatic increases in the capabilities of neural network models in recent years
are driven by scaling model size, training data, and corresponding computational
resources. To develop the exceedingly large networks required in modern applications, such as large language models (LLMs), model training is distributed
across tens of thousands of hardware accelerators (e.g. GPUs), requiring orchestration of computation and communication across large computing clusters. In this
work, we demonstrate that careful consideration of hardware configuration and
parallelization strategy is critical for effective (i.e. compute- and cost-efficient)
scaling of model size, training data, and total computation. We conduct an extensive empirical study of the performance of large-scale LLM training workloads
across model size, hardware configurations, and distributed parallelization strategies. We demonstrate that: (1) beyond certain scales, overhead incurred from
certain distributed communication strategies leads parallelization strategies previously thought to be sub-optimal in fact become preferable; and (2) scaling the
total number of accelerators for large model training quickly yields diminishing
returns even when hardware and parallelization strategies are properly optimized,
implying poor marginal performance per additional unit of power or GPU-hour.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors performed extensive experiments on hardware utilizations (MFU, tokens/w, etc) when number of accelerators (e.g., GPUs) scales to hundres of GPU server nodes (thousands of GPUs) and found:
1. Combination of different parallelism, e.g., FSDP and TP/PP can leverage the heterogeneous intra/inter-node interconnect BW more efficiently than FSDP along
2. Yet the system becomes communication (latency or bw) bound and has diminished return of MFU beyond certain scale

### Strengths
+ Extensive experiments on most popular 3D parallelims on modern models/hardwares at large scale (thousands of GPUs)
+ Real world evaluation metrics other than MFU, e.g,. tokens per watt etc
+ Could serve as a quantative handbook for researchers/engineers optimizing distribributed system

### Weaknesses
- The 2nd conclusion from the study is well know, i.e, distributed system is typically communication bound. The authors touch a bit on all-reduce could be BW bound while reduce-scatter and all-gather can be latency bound, e.g., Figure 3, but didn't go further provide any quantitative/analytical model or techical details on why the latter two are latency bound

- The 1st conclusion regarding parallelism strategies (and a mitigation to point above) is also relatively well known, e..g, the 3D (FSDP/TP/PP) is quite standard in modern LLM training, and their design and relation to network topogies (NVLink vs Infiniband) is also well studied, e.g., in Megatron series of papers.

- While it is true that distributed system total throughput scales sublinearly at scale, the "scale" the auhors begin to see diminsing returns (e.g., 2000 GPUs, both 7B/70B) are far behind industry leading results, e.g., Llama 3 training can reach 40% MFU at 20, 000 GPUs scale (https://arxiv.org/pdf/2407.21783)

### Questions
1. why is there a sudden jump in CCL time from 128 to 256 nodes in last plot of Figure 4?
2. what are the yellow and blue dots in Figure 5? My guess is one is PP and the other is TP.
3. what's the training framework used? Megatron/Deepspeed or in-house developed, how does it compare to these baselines?

### Soundness
2

### Presentation
3

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
The authors conduct a large-scale empirical study on large-scale LLM training workloads and presents the following insights:

1. As the total number of accelerators used for training large models increases, there are diminishing returns in performance. This means that even with optimized hardware and parallelization strategies, adding more accelerators yields progressively smaller improvements in throughput and efficiency.

2. The overhead incurred from certain distributed communication strategies can lead to scenarios where parallelization methods previously considered sub-optimal become preferable. This highlights the critical role of communication efficiency in large-scale training.

3. The study characterizes the scaling properties of sharded training, demonstrating that the efficiency of distributed training is heavily influenced by the balance between computation and communication costs. As model sizes and batch sizes increase, the communication volume also rises, making the training process increasingly communication-bound rather than compute-bound.

4. The experiments show that hardware utilization and throughput regress as the number of devices increases for a fixed computational workload. This indicates that scaling up the number of accelerators does not guarantee improved performance.

### Strengths
The thorough empirical study bridges the gap between existing scaling laws that connect loss to training FLOPs and real-world training configuration decision-making. 

The insights presented in the paper answer many open questions in training configuration selection where no prior work has been done systematically studied on.

### Weaknesses
1. "Training one large model is less power-per-token efficient than training many smaller ones." The part on energy efficiency seems a bit out of place. If there is a need/workload to train a large model, then training many smaller ones would not replace the need to train a larger one. I am not sure if this takeaway fits into the theme of the rest of the paper, which answers the important question of how different parallelism configurations affect training throughput.

2. While I understand this may not be feasible from a resource perspective, knobs such as global batchsize/sequence length would affect loss/downstream accuracy, which leads to a new space of tradeoffs. This regime is unexplored and may impact the quality of the trained model. Some results in this line would further strengthen the paper, such as if you pick the parallelism configuration you find to be most efficient, how does the training loss curve differ from existing known configurations used to train other open models of the same size? Or how and why existing models have been using suboptimal configurations during training.

3. "Additional scale only marginally improves throughput." The author argues that with fixed global batchsize, scaling has marginal returns. However, this seems to be an unrealistic setting. Using more nodes serves the exact purpose of accommodating larger global batchsize on many occasions with the assumption that using larger batchsize for LLM pretraining has minimal/acceptable impact on downstream performance/loss.

### Questions
My concerns are raised above. If the authors can address my concerns above, I would be happy to raise my rating.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors address three questions in this paper：
1. How do hardware scaling and parallelization strategies affect the efficiency of large-scale model training?
2. What’s the impact of different parallelism methods(data, tensor, pipeline) on training performance?
3. How do communication costs limit scalability, and when do we see diminishing returns?

To answer these questions, the authors measured performance metrics, including throughput (words per second, tokens per second), power efficiency, and communication overhead (time and exposed communication) using the Llama 2 model families (1B and 70B) across different generations of NVIDIA GPUs (V100, A100, and H100) with combined parallelization strategies of data-, tensor-, and pipeline-parallelism. These measurements are from a few to 2,048 GPUs.

The experiments compare the scalability of data, tensor, and pipeline parallelism, evaluating the impact of different hardware configurations and GPU counts on training speed and efficiency. Additionally, the authors examined how communication overhead constrains scalability as system size increases.

Based on the performance measurements, the authors draw three conclusions:
1. Communication limits scaling: as hardware scales up, performance gains slow down due to communication overhead.
2. Model parallelism helps reduce some overhead but has limitations, especially across multiple nodes.
3. Increasing the total number of accelerators for large model training quickly leads to diminishing returns, even with optimized hardware and parallelization strategies, indicating poor marginal performance per additional unit of power or GPU-hour.

The authors conclude the paper with five trends in scaling and implications in Section 5.

### Strengths
1. It is promising to use real-world LLMs for performance measurements
2. The testbed with 2,048 GPUs is sufficient to justify the scaling trends.

### Weaknesses
None of the scaling trends in Section 5 is new or unexpected. 
1. Not All FLOPs Are Equal: The authors point out that existing compute-optimal scaling laws and performance metrics primarily rely on FLOPs or derived metrics, which fail to take into consideration underlying massively parallelized distributed hardware which requires communication to execute these workload. 

This statement overlooks the latest LLaMA 3.1 paper (https://arxiv.org/abs/2407.21783), which considers the impact of communication in addition to FLOPs-based metrics.

2. Communication-Computation Dynamics Change at Scale. The authors claim that it is the low scalability of the collective communication primitives that motivates alternative parallelization strategies. This statement is incorrect, as the tensor-, pipeline-, and context-parallelism are motivated by the need of higher learning capability in LLMs. 

3. Additional scale only marginally improves throughput. The paper overlooks the opportunity of context-parallelism, which will be significantly improved with additional computing resources. This idea has been demonstrated in Llama 3.1 paper, where Meta trains the 405B model with 16,384 H100 GPUs using ring-attention. 

4. Training one large model is less power-per-token efficient than training many smaller ones. This is obvious as smaller models have less communication requirement, which results in better computation efficiency and, thus, power-per-token efficiency. This is not a new insight.

5. Improvements in networking within nodes improves scale-out performance. This statement is misleading, as an enhanced intra-node network would improve the overall training performance. However, the contribution of intra-node networks is independent of inter-node communication. 

6. Performance benchmarking fails to extrapolate across scales and hardware generations: The paper concludes that performance benchmarks cannot be extrapolated across different scales and hardware generations, but the experiments presented do not provide sufficient evidence to support this claim.

There exists ambiguity in Scaling Model Parallelism (Section 4.2): When scaling model parallelism, the degree of model parallelism is described as the product of tensor and pipeline parallelism, however, the mapping between a model-parallelism value to the tuple of tensor- and pipeline-parallelism is not unique. It is unclear if the authors examine all possible combinations of tensor- and pipeline-parallelism for a given model-parallelism setting. For example, the total degree of 16 can be 2-way tensor-parallel and 8-stage pipeline, or 4-way tensor-parallel and 4-stage pipeline. This ambiguity makes it unclear how the scaling experiment was conducted.

### Questions
Please address all questions in the Weakness section.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The authors of this paper stated that recent advancements in neural network capabilities, particularly in large language models (LLMs), are largely driven by increases in model size, training data, and computational resources. Training these expansive networks often involves distributing tasks across thousands of hardware accelerators like GPUs, necessitating effective orchestration of computation and communication across large clusters. This study highlights the importance of hardware configuration and parallelization strategies for efficient scaling. Key findings include: (1) certain distributed communication strategies that were previously deemed sub-optimal may become more favorable beyond specific scales due to overheads; and (2) increasing the number of accelerators can lead to diminishing returns in performance, even with optimized strategies, indicating a decline in the marginal benefits of additional computational power.

### Strengths
1. This paper provides evidence that model parallelism yields improved global throughput although previous works suggest the opposite.
2. The experiment setups are well-described and well-established in Section 3.
3. This paper brings up some observations suggesting that data parallelism could be less efficient compared to what people previously thought. Model parallelism in large-scale training is actually not causing many efficiency drops in a large-scale situation due to communication overlapping.

### Weaknesses
1. There are some minor concerns and suggestions about this paper. I have listed them in the Questions section.
2. There is no discussion on the training framework which could be buggy and thus making the experiment results less convincing.
3. Figure 6 seems to have the wrong labels, it is using pipeline parallel size 16 and model parallel 1 which is impossible.
4. The current trend in finding optimal MP and DP is testing configs one by one, also MP size vs efficiency is supposed to be a bell curve since model size couldn't be infinitely large. DP helps to scale up the training process and MP is designed to fit the larger models. Some DP and MP comparisons in this paper seem not fair since they do not have the same target in the first place. As in Figure 10, MP helps reduce computation per GPU well DP does not (not considering FSDP).

### Questions
1. Can you also add a citation to Sequence Parallel in 2.1.1?
2. The legend arrangement in Figure 3 seems confusing. Also, a legend of the second graph in Figure 4 is missing. Figure 5 is not well explained, like the color difference.
3. All the experiments are conducted with a Megatron-inspired framework as mentioned in Appendix E, is there any pros and cons of this framework? Are there any concerns about this framework? Could it affect the experiment results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper provides an empirical analysis of the performance of training LLMs with a variety of different parallelization configurations. The study includes data- and model-parallelism across a variety of GPU scales, as well as different GPU generations (A100 vs H100) and model sizes. A key takeaway is that many parallelization techniques become communication-bound at scale, and given that compute performance is rapidly outstripping communication performance, this may necessitate rethinking training paradigms for models.

### Strengths
1. Large-scale training is becoming increasingly common and having good studies and guidelines for this available to the community is valuable.
2. The paper studies a number of situations and identifies limitations in commonly-used parallelization schemes at scale.
3. The paper is clear and well-written, and its points are easily understood.

### Weaknesses
1. The paper's key point is that scaling a fixed model across more accelerators leads to rapidly diminishing returns due to communication overheads. Yet it seems to me that this is just recapitulating the well-known challenges of strong scaling and over-decomposition (_any_ fixed problem becomes communication-bound when it is scaled, unless it is embarrassingly parallel). Indeed, the scientific and high-performance computing communities have long recognized this and have typically preferred weak-scaling paradigms where possible (see, e.g., Gustafson's Law; the classic citation being Gustafson, "Reevaluating Amdahl's Law", CACM 1988). It is not clear to me that the paper is making any real contribution here; perhaps incorporating weak scaling studies would benefit the paper.
2. Some of the insights in the paper in terms of configuration for parallelization strategies (tensor vs pipeline vs data parallelism, etc.) seem to reprise the advice given in Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM", Supercomputing 2021. How does the paper compare with Narayanan et al., and does it draw different conclusions? (Note there are some configurations, e.g., sequence parallelism, considered here but not in Narayanan et al.)
3. Related to the above two points, it seems that many of the configurations the paper evaluates are not actually realistic. For example, due to the communication requirements, FSDP is typically not performed over many nodes as in Figure 4 and instead "hybrid shard" versions are employed where the model is sharded only over a small number of GPUs (e.g., one node) and standard data-parallelism is employed across shards. A similar strategy is used with tensor parallelism. It is thus not clear what the practical value of many of the configurations evaluated is, considering one would not use them in practice as they over-decompose.

These are less critical points, but seem to indicate some underlying issues with the paper:

4. In Section 2.2, the paper states that ring allreduces are "bandwidth-balanced" (I assume this means optimal?) whereas tree allreduces have suboptimal bandwidth but better (logarithmic) latency. This is not fully accurate. It is indeed the case that a tree-based recursive doubling algorithm for allreduce takes time (ignoring computation; $p$ processors, message size $n$, latency $\alpha$, inverse bandwidth $\beta$) $\log p \alpha + n \log p \beta$ and a ring allreduce takes time $2 (p-1) \alpha + 2 \frac{p-1}{p} n \beta$. However, the classic Rabenseifner algorithm for allreduce is tree-based (recursive halving/doubling) and bandwidth-optimal, with time $2\log p \alpha + 2 \frac{p-1}{p} n \beta$ (although in practice ring-based algorithms can perform better depending on the network topology/configuration). See, e.g., Thakur et al., "Optimization of Collective Communication Operations in MPICH", IJHPC 2005; Chan et al., "Collective communication: theory, practice, and experience", Concurrency and Computation: Practice and Experience, 2007.
5. Also in Section 2.2., the paper states that "AllGather and ReduceScatter can only use ring algorithms". This is incorrect: tree-based algorithms (with logarithmic latency terms) are known. See the recursive-doubling or Bruck algorithms for allgather and the recursive-halving algorithm for reduce-scatter. (See above references for details.)
6. In Section 2.1.1, when discussing data parallelism, the paper states that the gather operations for FSDP delay computation. While it is true that FSDP introduces communication on the critical path, the situation is not quite like this: given some additional memory, the gathers can be "double-buffered" by fetching subsequent layers while performing computation. Some implementations, e.g., PyTorch's, are able to do this.
7. Minor: The paper uses "words per second" for throughput; is this really tokens per second?

### Questions
Please see the comments/questions above under "weaknesses".

Please focus in particular on whether there are actionable insights beyond the limits to strong scaling and the guidelines in Narayanan et al.; I would be willing to raise my score if a convincing case is made for this.

### Soundness
2

### Presentation
4

### Contribution
2
