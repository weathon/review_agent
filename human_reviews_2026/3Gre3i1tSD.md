# GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 4, 6

## Abstract
Sparse Mixture of Experts (SMoE) performs conditional computation by selectively activating a subset of experts, thereby enabling scalable parameter growth in large language models (LLMs). However, the expanded parameter scale exceeds the memory capacity of a single device, necessitating distributed deployment for inference. This setup introduces two critical challenges: (1) *Communication Issue*: Transferring features to devices with activated experts leads to significant communication overhead. (2) *Computational Load Issue*: Skewed expert activation overloads certain GPUs, resulting in load imbalance across devices. Among these, communication overhead is identified as the main bottleneck in SMoE inference. Nevertheless, reducing communication between devices may exacerbate load imbalance, leading to device idleness and resource waste. Therefore, we present **GRACE-MoE**, short for **G**rouping and **R**eplic**a**tion with Lo**c**ality-Awar**e** Routing for S**MoE** inference. **GRACE-MoE** is a co-optimization framework that jointly reduces communication overhead and alleviates computational load imbalance. Specifically, the framework comprises two key phases: ① *Grouping & Replication*: This phase groups experts based on their affinity to reduce cross-device communication. Additionally, dynamic replication is applied to address load skewness, improving computational load balance across GPUs. ② *Routing*: This phase employs a locality-aware routing strategy with load prediction. It prioritizes local replicas to minimize communication overhead and balances requests across remote replicas when necessary. Experiments on diverse models and multi-node, multi-GPU environments demonstrate that **GRACE-MoE** efficiently reduces end-to-end inference latency, achieving up to **3.79×** speedup over state-of-the-art systems. Code for **GRACE-MoE** will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents GRACE-MoE, a co-optimization framework designed to address two major bottlenecks in distributed Sparse Mixture-of-Experts inference: communication overhead and computational load imbalance. Experiments across multiple MoE models and multi-node GPU clusters show that GRACE-MoE achieves up to 3.79× end-to-end inference speedup over state-of-the-art systems such as DeepSpeed, Tutel, Megablocks, and C2R, without compromising accuracy.

### Strengths
1.	GRACE-MoE effectively integrates three orthogonal techniques (grouping, replication, and routing) into a unified framework. The design is both theoretically motivated and practically relevant for distributed MoE inference.
2.	The paper demonstrates consistent and substantial performance improvements across diverse models, datasets, and hardware configurations, including multi-node environments, which is a challenging and underexplored setting for MoE inference.
3.	The combination of controlled non-uniform grouping and load-aware dynamic replication provides a solid general foundation for future optimization of distributed sparse models beyond MoE.

### Weaknesses
1.	Although the paper acknowledges increased memory usage due to replication, it does not quantify this overhead or explore its impact on GPU memory-limited settings. A detailed trade-off analysis would improve completeness.
2.	The experiments are conducted on relatively modest clusters (up to 2 nodes × 8 GPUs). It remains unclear whether the method scales linearly or maintains efficiency on industrial-scale infrastructures with high interconnect heterogeneity.
3.	The grouping and replication schemes rely on profiling expert activation statistics. The paper does not detail how frequently profiling must be updated, nor how GRACE-MoE adapts to non-stationary input distributions during long-term inference.

### Questions
1.	A concise summary table comparing GRACE-MoE’s components with existing systems (FasterMoE, FlexMoE, C2R, etc.) would help contextualize novelty.
2.	How about comparing GRACE-MoE with COMET[1] and Triton-distributed[2]?

[1] https://arxiv.org/pdf/2502.19811
[2] https://github.com/ByteDance-Seed/Triton-distributed

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
4

### Summary
MoE architectures have recently enabled scaling of LLMs to even larger sizes without introducing the same amount of computation. However, expert parallelism can lead to workload imbalance and high communication overhead during inference. To mitigate these issues, the authors propose allowing GPUs to host varying numbers of experts to improve expert affinity. They also explore replicating hot experts to reduce resource idle time. Empirical results show that the proposed method achieves improvements in certain scenarios.

### Strengths
- The author has chosen to study the inference of the mixture-of-experts LLM models, which is a very important subject with major implications to the deployment in the industry.

### Weaknesses
- **Lack of novelty.** The paper failed to distinguish itself from works in the same area. The author claimed that busy expert replication is novel because it "allocates replicas according to the load skew of the maximum load group in each layer, replicating only its busiest expert". But just as the authors themselves noticed in L127-136, the replication strategy has been adopted by many works. For instance, He et al. (2022), which is cited in this manuscript, also studied a multi-node, topology-aware setup. The main difference appears to be that this work allows a different number of experts per device, which alone does not sufficiently establish novelty. Its effect could also be questionable (see below).

- **Poor writing and missing definitions.** The paper frequently introduces non-standard concepts without proper definition, making it difficult for readers to follow. For example:
  - the first few paragraphs failed to discuss what is the so-called "non-uniform grouping strategy" and "uniformity constraint", so it's unclear to me what could be the takeaway from figure 1 on line 150.
  - L231 mentions that the proposed strategy involves selecting the knee from the (S(r), U(r)) joint plot. However, it's unclear at all how this plot would look like from the collected data, and the authors did not provide such an example in the manuscript.
  - L253: what is a collaborative expert? This word was taken as granted by the authors and only shown once before, in caption of Fig 1b, also without a clear explanation. If the author was referring to a definition introduced in Suo et al. 2025 / Zhang et al. 2025, the referenced paper should be cited here.
  - L359: The author mentioned two types of clusters, one with 3090, the other with 4090. Yet it's unclear that for the multi-node setup, whether there exists cross-cluster-type setup or not. 
  - L398: The average grouping mentioned here is confusing. Is the author trying to ensure the load on each GPU is averaged? If so, the author should describe it at the beginning of sec 5.3.

- **Unrealistic workload and insufficient experimentation.** The workload configuration (L352) fixes batch size, prefill, and decode lengths, which is an unrealistic assumption. In practice, requests vary in length and frequency, and continuous batching or dynamic scheduling can significantly alter performance behavior. Moreover, the experiments use only one dataset, limiting the generalizability of the results. The paper also omits memory profiling, leaving potential OOM risks unexamined.

- **Significant Proofreading issues**. Lines 145 and 232 contain question marks indicating missing references, suggesting that the manuscript was not properly proofread. Such oversights are unacceptable in a submission.

- Code is not released with the manuscript.

### Questions
See weaknesses for some of the concerns regarding this manuscript. In addition:

In section 1, authors claimed that "More critically, prior work typically addresses only one of these two issues, yet optimizing one often worsens the other. For instance, reducing communication overhead usually aggravates computational load imbalance."This statement requires clarification and evidence. For example, reducing load imbalance could result in more evenly distributed communication volume across devices, thereby reducing communication overhead rather than increasing it. Please elaborate on this apparent contradiction.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper provides optimizations on expert placement and routing to improve the efficiency of distributed SMoE inference. The experimental results show that it achieves significant end-to-end speedup compared to the framework without optimizations on expert placement.

### Strengths
1. The manuscript is well-structured and easy to follow.
2. The experimental results demonstrate significant benefits.

### Weaknesses
1. The manuscript provides insufficient coverage of existing research within the relevant fields.
2. Lack of novelty.
3. The evaluations and analyses presented are inadequate. The manuscript does not sufficiently explain the reasons behind the performance differences among the methods shown in Figure 4. And a comprehensive comparison with existing expert placement methods is lacking.
4. Lack of analysis regarding the impact and benefits of the proposed approach under varying model and hardware configurations.

### Questions
1. Since flexible expert placement is a widely adopted approach for improving the efficiency of MoE models, please provide a more detailed review and comparison with state-of-the-art methods, for example, EPLB.
2. Based on the results presented in Figure 4, please provide an in-depth analysis of the performance differences between different methods. Additionally, analyze the impact of these methods across various model architectures and hardware platforms, including a breakdown of overheads or modeling, to substantiate the general benefits it can achieve.
3. Some references, such as those to figures and sections, are missing.

### Soundness
2

### Presentation
2

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
This paper alleviates both the communication overhead and computational load imbalance issues in SMoEs. Experimental results show that affinity-based expert grouping, dynamical replicaion based on load skew, and topology-aware routing can greatly reduce MoE layer time.

### Strengths
1. The logic of the research is well-understandable.
2. The presented method shows good improvement in MoE layer time.

### Weaknesses
1. In Observations, the fluid relationships of co-activation are not described in detail.
2. There are several errors in the LaTex hyperlink.
3. More ablation studies are required in the current setup of the parameters/weights for each methods used to alleviate each of the two issues in SMoE.

### Questions
1. What would the alternative approach of this algorithm be, for smaller nodes and devices or larger models, that prehibit the best-case scenario from happening?

### Soundness
2

### Presentation
3

### Contribution
3
