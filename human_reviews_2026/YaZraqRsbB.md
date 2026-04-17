# FlexHiNM-GP: Flexible Hierarchical Pruning via Region Allocation and Channel Permutation

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
N:M sparsity has emerged as a hardware-friendly pruning strategy, notably supported by NVIDIA’s Sparse Tensor Cores. While efficient, its fixed sparsity ratio restricts flexibility, making it difficult to adapt pruning granularity to varying weight importance across layers and architectures.
To overcome this limitation, we propose FlexHiNM, a hybrid framework that adaptively partitions each layer into three regions: dense, vector-pruned, and N:M sparse, enabling finer-grained control while preserving hardware compatibility. To better preserve salient weights, we extend this to FlexHiNM-GP, which incorporates Gyro-Permutation, an iterative channel-rearrangement algorithm. Through successive sampling, clustering, and assignment, Gyro-Permutation aligns high-importance weights with structured sparsity patterns and mitigates suboptimal configurations in multi-level pruning.
During gradual pruning, FlexHiNM-GP further employs a differentiable masking mechanism based on the Hard Concrete distribution, enabling gradient-based mask learning and preventing over-aggressive early pruning. Experiments on vision and language benchmarks demonstrate that FlexHiNM-GP consistently surpasses strong structured baselines and approaches the performance of unstructured pruning, validating the effectiveness of combining hybrid sparsity with learned masks and permutation strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presented FlexHiNM, a framework to consider a hierarchical N:M sparsity (HiNM) using Gyro-permutation. Difference from HiNM that has 0:4 and 2:4 pruning, the proposed method considers 4:4 pruning that preserves entire vectors with high importance scores. Gyro-permutation based channel reordering optimizes the order of input and output channels so that important weights are well aligned with the n:M constraint. Hard concrete module enables differentiable mask learning under the N:M sparsity constraint. By applying DeiT, BERT, and LLaMA-2, the proposed method shows better performance than other previous methods such as HiNM and OVW (Outer-vector-wise Pruning)

### Strengths
- Compared with other methods dealing with N:M sparsity, this paper introduces a hierarchical structure of 0:4, 2:4, 4:4 pruning to provide variable sparsity. This ratio can be automatically adjusted based on the importance of each layer and channel to minimize accuracy loss.
- Gyro-permutation makes better aligned input and output channels with N:M sparsity. 
- Hard concrete module shows better importance representation and stable convergence.
- By redesigning the GPU kernel to handle permutation operation, the gyro-permutation can be applied efficiently in real hardware environments.
- It shows better performance than HiNM when applying to DeiT, BERT, LLaMA2.

### Weaknesses
- The learning scheme in this paper has complex structure so that model training and implementation may be difficult. Due to various hyperparameters and boundary search (vs, ps), the training time should be increased.
- The additional memory and computational overhead are necessary for the process of permutation and learnable masks.
- Comparison with other methods considering N:M sparsity is necessary to show the effectiveness of the method, for example, Venom (Castro et al., 2023)
- There is no comparison of inference time or throughput, FLOPS.

### Questions
- Gyro-permutation for N:M sparsity is proposed in arxiv paper, Toward Efficient Permutation for Hierarchical N:M Sparsity on GPUs (2024). Why did not you cite and compare this paper?

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
4

### Summary
This paper addresses the lack of flexibility in N:M sparsity by proposing the FlexHiNM framework. The method partitions weights into three regions: dense (4:4), sparse (2:4), and pruned (0:4). To accommodate this partitioning, the authors design the Gyro-Permutation (GP) algorithm to reorder channels. The authors claim their method outperforms the compared structured pruning baselines on DeiT, BERT, and LLaMA-2 in terms of accuracy.

### Strengths
1. The motivation is clear and practically relevant. For hardware-supported N:M sparsity, the fixed 50% sparsity ratio is indeed a severe limitation in practice. The paper's direction in addressing this is correct.

2. The "three-region" partitioning (4:4, 2:4, 0:4) is a logical and incremental improvement over existing HiNM (2:4 and 0:4), providing finer-grained control for layers with different redundancies.

3. The paper conducts extensive experiments on accuracy metrics. The experimental results show that the proposed FlexHiNM-GP method does consistently surpass baselines like OVW and HiNM-V in accuracy on vision and language tasks, and it narrows the gap with unstructured pruning.

### Weaknesses
1. The paper is lack of practical inference latency evaluation. The paper's premise is hardware-friendliness and efficiency, yet the entire manuscript contains no data on end-to-end latency, throughput, or actual wall-clock speedup. The custom kernel proposed in the appendix is merely a theoretical design; its practical overhead from mixed execution and cross-stream synchronization is unknown. Lacking this core speed evidence, the paper's accuracy improvements are unconvincing, as they might be achieved at the cost of sacrificing inference performance.

2. The algorithmic proposal appears overly complex, like a hodgepodge of techniques. Its core innovation is limited to the three-region partitioning, which is a minor incremental improvement. Other parts of the framework, such as Hard Concrete mask learning (Algorithm 3) and gradual pruning, are off-the-shelf techniques in the field. The core Gyro-Permutation (GP) algorithm (Algorithm 2) feels like an over-designed product, seemingly designed as an extremely convoluted process (sampling + K-means + Hungarian matching) just to force the weight distribution to fit the three-region partitioning. This combination of boundary search, iterative GP, and gradual learning leads to extremely high pruning process overhead. The authors do not quantify this cost, making its practical feasibility questionable.

3. The ablation studies fail to sufficiently demonstrate the necessity of its core components. The most critical comparison is missing: a comparison between FlexHiNM (with only three-region partitioning) and FlexHiNM-GP (which adds Gyro-Permutation). We cannot ascertain whether the accuracy gain comes from the (relatively simple) three-region partitioning or from the (extremely complex) GP algorithm. If the improvement from the GP algorithm is negligible, its existence only serves to add unnecessary complexity.

### Questions
1. Why did you not provide any end-to-end inference latency or throughput data? This is the most critical metric for evaluating a hardware-friendly pruning method. Can you provide the actual speedup of FlexHiNM-GP versus HiNM-V at 75% and 87.5% sparsity, measured on an RTX 4090 using the kernel design from Appendix D.1?

2. Can you quantify the total computational overhead of the complete FlexHiNM-GP pruning pipeline (including boundary search, GP reordering, and mask learning)? For example, how many GPU-hours are required to prune the LLaMA-7B model once?

3. Can you add an ablation study comparing only FlexHiNM (without GP) versus FlexHiNM-GP? This is essential to justify the necessity of the complex Gyro-Permutation algorithm.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to address the effectiveness degradation caused by wide variance of important scores in N:M sparsity patterns. The authors propose a three-level hierarchical pruning framework that partitions model weights into dense, vector-pruned, and N:M-sparse regions, enabling balanced sparsity and accuracy. A boundary search algorithm is introduced to determine the optimal configuration. This paper also introduces Gyro-Permutation, a channel reordering method designed to enhance information aggregation and improve pruning performance. Combined with iterative pruning, the approach achieves further performance gains. 
To mitigate the computational overhead introduced by permutation, a custom GPU kernel is developed to execute dense and sparse regions concurrently, ensuring hardware efficiency. The proposed framework is evaluated on both vision and language models, showing strong pruning performance.

Contributions

1. A three-level hierarchical pruning framework and a dynamical boundary searching algorithm.

2. The Gyro-Permutation algorithm that improves pruning efficiency through channel reordering.

3. A hardware-aware co-design, integrating a custom GPU kernel to offset the online computational cost of permutation.

### Strengths
**Originality.**
The paper provides original and meaningful improvements over prior structured pruning methods. It introduces an effective boundary search algorithm to adaptively determine three-level pruning ratios, a more robust channel permutation scheme (Gyro-Permutation) to enhance information concentration, and an efficient kernel implementation that avoids costly online computation.
Beyond these individual components, the work’s novelty lies in how it integrates algorithmic and hardware aspects into a unified co-design framework, achieving both flexibility and practical deployability.

**Quality.**
The technical quality of the paper is strong, particularly in the boundary search algorithm and the GPU kernel design. Both components are clearly formulated, well-motivated, and experimentally validated.

**Clarity.**
The motivation and high-level logic are clearly stated, helping readers understand how each module contributes to the framework.

**Significance.**
The work addresses a practically important challenge, improving the effectiveness of N:M sparsity under high variance, through a coherent combination of pruning and permutation.
The cross-domain applicability to both vision and language models increases its potential impact.

### Weaknesses
**Clarity of figures and presentation.**
Several figures (e.g. figure 6) are difficult to interpret and in some cases make the workflow harder to follow. A few textual descriptions are also inaccurate or ambiguous, which reduces readability and clarity (i.e. lambda in equation 3, what is the definition and what is the value in experiment).

**Insufficient detail on Gyro-Permutation.**
As a core component of the proposed framework, the Gyro-Permutation method is described too briefly in the main text and lacks sufficient algorithmic detail even in the appendix. Providing a clearer formulation and rationale would significantly strengthen the technical contribution.

### Questions
In Figure 1, why is the N:M range of R for the HiNM-P model bounded at 0.7?

The Figure 2 and Figure 3 are confusing to me. Which part in figure 2 corresponds to figure 3? There is a sorting logic in figure 3, but no corresponding illustration in figure 2.

The description of Figure 7 is confusing. It states that HiNM-GP is HiNM with gyro-permutation, but it is unclear which “HiNM” this refers to.
If HiNM-V is defined as FlexHiNM without gyro-permutation and is said to be “equivalent to Venom,” then the relationship among HiNM, Venom, HiNM-V, and HiNM-GP becomes ambiguous.
Does HiNM here correspond to Venom, or is it a different baseline?
If HiNM-GP = HiNM + gyro-permutation, and HiNM-V = Venom, then logically HiNM-GP should be equivalent to HiNM-V + gyro-permutation, which seems to describe FlexHiNM.
Please clarify the naming and hierarchical relationship among these variants.

What is the motivation behind gyro-permutation’s sampling and clustering?  In the appendix, it states that the number of samples is dynamically chosen without any detail. What is the method for choosing this number of samples hyperparameter? Also, there is no detail about how balanced K-means works.

For CUDA kernel, it seems when you move the input and weight from global memory to shared memory, you need to do indexing to select the corresponding partition. However, I wonder whether this indexing will cause an uncoalesced problem or not which may degrade the actual running speed? Since you have implemented the CUDA kernel, it would be more convincing if you provide experiment results on speed.

### Soundness
3

### Presentation
2

### Contribution
3
