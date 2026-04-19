# Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

- Decision: Accept (Poster)
- Scores: 8, 6, 5

## Abstract
Large Language Models (LLMs) with the Mixture-of-Experts (MoE) architectures have shown promising performance on various tasks. However, due to the huge model sizes, running them in resource-constrained environments where the GPU memory is not abundant is challenging. Some existing systems propose to use CPU resources to solve that, but they either suffer from the significant overhead of frequently moving data between CPU and GPU, or fail to consider distinct characteristics of CPUs and GPUs. This paper proposes Fiddler, a resource-efficient inference system for MoE models with limited GPU resources. Fiddler strategically utilizes CPU and GPU resources by determining the optimal execution strategy. Our evaluation shows that, unlike state-of-the-art systems that optimize for specific scenarios such as single batch inference or long prefill, Fiddler performs better in all scenarios. Compared against different baselines, Fiddler achieves 1.26 times speed up in single batch inference, 1.30 times in long prefill processing, and 11.57 times in beam search inference. The code of Fiddler is publicly available at https://github.com/efeslab/fiddler.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper discusses how to optimize the inference performance of MoE in a resource-constrained environment but with heterogeneous hardware. Because the model size is too large to fit into the GPU memory, existing systems offload weights or computations to CPU resources. However, they didn’t consider the MoE model properties or different device characteristics of CPUs and GPUs, leading to suboptimal performance. To optimize the performance, this paper proposes Twiddler that finds the optimal execution strategy using both GPU and CPU resources. Its evaluation demonstrates that Twiddler outperforms existing systems in different applications and hardware settings.

### Strengths
+ This paper is well-written and clearly defines the scope.
+ This paper analyzes the fundamental limitations of existing systems for MoE inference in a resource-constrained environment
+ This paper discusses the design space of MoE inference in a resource-constrained environment: 1/ expert weight is on GPU memory; 2/ expert weight is not on GPU and it is copied copied from CPU to GPU; and 3/ expert weight is not on GPU but the activation is copied from GPU to CPU. It then devises a cost model to dynamically determine whether to execute computation on GPU or CPU.
+ This paper evaluates different input and output length on two hardware settings to demonstrate the effectiveness of Twiddler.

### Weaknesses
- The used metrics can be further improved: 1/ the definition of end-to-end latency is unclear; 2/ Typically TTFT and inter-token latency (ITL) will be both used as metrics for inference performance evaluation, but only TTFT is used in this paper. Although token per second is used, its definition still mingles with TTFT.
- Only single batch size is used for evaluation. Curious if Twiddler still works for scenarios with batch size over 1, such as 2, 4, and 8
- The performance improvement of Twiddler over the best baselines is marginal. In terms of TTFT, it only outperformed DeepSpeed-MII by 1.07x.

### Questions
N/A

### Soundness
3

### Presentation
4

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
The authors propose a resourceefficient inference system for MoE models with limited GPU resources,which can strategically utilizes CPU and GPU resources by determining the optimal execution strategy. It performs better in all scenarios.Additionally, the  authors design a specialized computation kernel for expert processing on the CPU using the AVX512_BF16 instruction set,

### Strengths
1. The algorithm in the experiment achieved good results.Twiddler achieves 1.26 times speed up in single batch inference, 1.30 times in long prefill processing, and 11.57 times in beam search inference.
2. it is good to  analyze the sensitivity of Twiddler’s performance on input datasets to discover whether  MoE models’ routing behavior may be affected by the characteristics of input data distribution.
3. it seems good to  design a specialized computation kernel for expert processing on the CPU using the AVX512_BF16 instruction set

### Weaknesses
1. The experiments are a bit sparse and insufficient. There should be experiments where experts have better data from CPU to GPU, which can better explain the reasons for the performance improvement.
2. A dedicated computing core is designed for expert processing on the CPU. However, its impact was not considered during the experiment.
3. More details about the way to find the suitable experts in the GPU rather than CPU are needed, and how the inference can comparable with SOTA inference framework like TensorRT etc.

### Questions
1. In section 2.2, when it states, “such approaches show suboptimal performance for important use cases,” what specific metrics are being referred to as suboptimal? Is it in terms of execution time, accuracy, resource utilization, or something else?
2. At runtime, Twiddler identifies the optimal configurations for executing expert layers across the GPU and CPU. Could you provide more details on how Twiddler determines which expert(s) to use for each token after executing the gating function for each layer? What methods are used to obtain the optimal configuration? Are there any experimental studies to support this?
3. Could you explain the concepts of beam search and speculative decoding? How does speculative decoding function within this system? More experimental results would be helpful for clarification.
4. What is the memory usage for this work? How does it compare in terms of memory savings with approaches that do not implement switching?
5. How does this work compare with state-of-the-art LLM inference methods such as vLLM, TensorRT, or other LLM serving frameworks? While I understand some of them may not support the Mixture of Experts (MoE) framework, how does the baseline performance compare?
6. If one expert is too slow, what is the overhead associated with loading this expert into GPU memory? How are the costs related to switching experts between GPU and CPU managed? Is there a specific switching policy in place?
7. The author mentions that a router is used to decide when to use the CPU and GPU for the next token. What is the overhead cost of the router? Additionally, if the tokens are broadcast to each expert, can the proposed approach still function effectively

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes Twiddler, an efficient inference system for MoE models with access to a constrained number of GPUs. Twiddler applied heterogeneous computing mechanisms to leverage the capabilities of both CPUs and GPUs to accelerate inference, given optimal execution strategies. Twiddler claims salient performance improvements in three scenarios, single batch inference, long prefill processing, and beam search inference.

### Strengths
The paper systematically approached scenarios where mixed use of CPUs and GPUs can outperform performance of purely relying on GPUs for MoE models.

### Weaknesses
Most industry-level inference systems rely predominantly on GPUs for training and inference. At least for current system demands, CPUs are less often adopted in practical heterogeneous clusters, considering the tremendous flops to be performed. Although there may be use cases where communicating weights of small models across CPUs and GPUs may have a slight improvement, this formulation has little practical implications for models of medium and large sizes due to the tremendous computational latencies. Secondly, input size is only but one aspect that contributes to the single batch latency in inference as model capacity also contributes significantly to the total flops. Therefore, this assumption on input size requires refinement. Thirdly, the sequence lengths/widths used in the experiments are not representative enough to demonstrate the generalizability of your formulation. Hence, the paper is recommended for rework.

### Questions
Both the formulation and experiments are recommended for rework.

### Soundness
1

### Presentation
2

### Contribution
1
