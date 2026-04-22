# DynamicInfer: Runtime-Aware Sparse Offloading for LLMs Inference on a Consumer-Grade GPU

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Large Language Models (LLMs) have achieved remarkable success in various NLP tasks, but their enormous memory footprints pose significant challenges for deployment on consumer-grade GPUs.
Prior solutions, such as PowerInfer, combine offloading and sparse activation to reduce memory and computational overhead, but suffer from static neuron partitioning, leading to suboptimal GPU utilization and increased latency.
In this work, we present DynamicInfer, a runtime neuron offloading framework that dynamically adapts neuron scheduling based on input-dependent activation patterns. DynamicInfer introduces (1) a hierarchical neural caching strategies, (2) a load-aware neuron activation mechanism tailored to heterogeneous hardware, and (3) an activation-aware prefetching pipeline that overlaps data transfer with computation.
Extensive experiments on ReluLLaMA and Prosparse models across multiple hardware platforms demonstrate that DynamicInfer achieves up to 253\% speedup over llama.cpp and 59\% over PowerInfer, while retaining model accuracy. Our approach offers a practical and scalable solution for high-performance LLM inference on resource-constrained devices.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes a CPU–GPU collaborative inference method for large language models on a consumer-grade GPU by leveraging activation sparsity. Specifically, it introduces a hierarchical neural caching strategy that determines neuron placement based on sentence-level and token-level activation patterns. This differs from the baseline, PowerInfer, which also performs CPU–GPU collaborative inference but relies on offline profiling and a static hot–cold neuron partitioning. Given a layer’s hidden state, the predictor forecasts the sparsity of a single later layer located several layers deeper in the network. To hide host-to-device weight-transfer overhead, the paper applies optimization techniques such as I/O pipelining with overlapping of computation and data transfer. To maintain load balancing between GPU and CPU, it adaptively adjusts the threshold based on the observed loads of each device. The neuron-placement strategy must satisfy memory, communication, and macro-residency constraints; the paper formulates this as an integer linear programming (ILP) problem. Evaluated with ReluLLaMA and Prosparse on the Alpaca dataset, the method achieves a 253% speedup over llama.cpp and a 59% speedup over PowerInfer.

### Strengths
- This paper targets an important research problem: democratizing LLM inference and preserving users’ data privacy by enabling inference on a consumer-grade GPU.

- The plots and figures are very helpful for understanding the proposed method, when each component operates, and where the speedup comes from.

- The achieved speedup over the baseline is substantial, and the results show the method works across several model sizes (7B, 13B, 70B).

- Figure 1, which describes the offloading system, is very helpful for understanding the baselines, prior work, and how data transfer occurs in each case.

- The rationale behind the method is sound—both in terms of systems optimization (I/O pipelining, overlapping, and balancing the load) and algorithmic design (dynamic, runtime prediction versus offline, static prediction).

### Weaknesses
- The evidence supporting the claim that the method preserves accuracy is insufficient. The paper evaluates only two model architectures (ReluLLaMA and ProSparse). To demonstrate generality, additional experiments on other architectures (e.g., Qwen, DeepSeek) are needed to show that accuracy is maintained.

- The method targets models whose activation functions induce high sparsity by design (e.g., ReLU). The paper should evaluate whether comparable accuracy and speedups hold for architectures using other activations such as ReGLU and SwiGLU.

- The experimental setup needs clearer description and stronger justification. The paper states that a “low-end PC” uses 120 GB of host memory, but 120 GB is arguably not low-end. Clearly specifying the exact memory configuration would improve transparency.

- A sensitivity study is needed for the hyperparameter k and for how speedup varies with input length, output length, and batch size, to establish robustness across diverse settings. Since k controls how far ahead the predictor targets and influences predictor accuracy, its impact should be quantified.

- Stronger baselines are needed. Although “Transformers” is included, it typically shows slower performance than highly optimized inference frameworks (e.g., vLLM) and baselines like SpecInfer that employ speculative decoding.

### Questions
- Accuracy/generalization: Beyond ReluLLaMA and ProSparse, can you report accuracy on additional architectures (e.g., Qwen, DeepSeek) to demonstrate that accuracy is preserved more generally?

- Activation functions: Do the proposed gains (accuracy and speedup) hold for models that use non-ReLU activations such as ReGLU or SwiGLU?

- Experimental setup clarity: Could you clarify the “low-end PC” configuration—especially the 120 GB host memory claim—and provide the exact memory specifications to justify this designation?

- Sensitivity/robustness: How sensitive is performance to the hyperparameter k? And how do speedups vary with input length, output length, and batch size?

- Baselines: Can you include stronger baselines (e.g., vLLM, SpecInfer with speculative decoding) and compare throughput/latency directly to these systems?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
DynamicInfer improves LLM inference throughput served with memory-constrained GPUs via sparsification. It does this by sparsifying the neurons in the Feed Forward Network (FFN) stages of transformers. The key contribution is a scheme for 'online sparsification', where the sparsity map of the next layer is estimated with the previous layer activations, giving enough time for the CPU-GPU memory transfers of the FFN weights to happen. DynamicInfer Is tested with LLM architecture that utilize ReLU activations and does not markedly drop LLM accuracy on four datasets.

### Strengths
Thank you for submitting your work. I overall like the paper. In terms of strengths:
* I like the idea of online sparsity. It's a complicated technique, but there are few alternatives when memory restrictions are aggressive (except for quantization, which should be compared to).
* The paper is mostly well-presented. §2 lays the groundwork for the rest of the discussion, and §3 conveys the general idea (though I have suggestions for improving §3).
* DynamicInfer is evaluated on two machines, 5 models, and compared to 3 other schemes.

### Weaknesses
I find 3 major flaws in the paper:
* From what I know, most modern LLMs do not use ReLU activation functions. Older LLMs used GeLU. The previous generation, such as Llama3, Qwen2 used SiLU, and the newest generation such as gpt-oss use SwiGLU. These families of activations always contribute to the next layer neurons, regardless of their value. ReLU, on the other hand, is akin to a on/off switch; some neurons do not affect the next layer at all, which is why sparsification would be a prime optimization candidate. This makes me worry if DynamicInfer would incur more substantial accuracy losses with commonly used LLMs, such as the original Llama family.
* The problem setup is a memory constrained GPU, and model compression is allowed (sparsification that skips neurons is technically model compression). So a natural contender is quantization. The PC-low setup with 7B models and PC-high setup with 13B models can be compared to int8 quantized versions (done properly, as simply casting would break the model) of the original models. I understand that quantization cannot work with 70B models, but they are still simpler alternatives than sparsification.
* The input/output sequence lengths are too small compared to real workloads. Even the relatively old ShareGPT dataset has input/output sequence lengths in the 300/200 range, and more modern datasets have even higher input lengths (due to longer prompts) and output lengths (due to reasoning chains). I wonder if the benefits of sparsification still hold at longer sequence lengths, since the attention runtime will increase and become a bigger share of the full inference pass.

Some minor comments:
* I found the explanation of the figures in §3 to be confusing. For example, why are there two bars in every group in Figures 2b and 2c? I expected a single bar to show the Jaccard distance between global hot neurons and per sentence hot neurons (Also, please cite a reference for Jaccard distance, or use the more common IoU metric which is quite similar). As another example, the oracle is defined to be $\frac{\text{\\# activated neurons}}{\text{\\# GPU neurons}}$. If this is the definition, then what is "Oracle-GPU" and "Oracle-CPU", and why do they sum to 1?
* The paper has formatting issues (figures have inconsistent reference formatting) and needs proofreading (typos).

### Questions
I am willing to increase my score if the authors can either provide reasonable answers to, or provide empirical evidence for the following questions:
* Can DynamicInfer improve the throughput of modern LLMs, e.g., Llama3 family or gpt-oss?
* How does DynamicInfer compare against vLLM with quantized models (throughput and accuracy)?
* How do DynamicInfer's improvements change at ISL/OSL 2K/128?

### Soundness
2

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
The paper proposes DynamicInfer, a runtime-aware system for efficient LLM inference on resource-constrained devices. It enhances efficiency by dynamically and finely scheduling FFN neurons between CPU and GPU. Its core mechanisms involve using a lightweight sparse predictor to forecast activation patterns online and dynamically adjusting the sparsity threshold based on the overlapping window of computation and data transfer, thereby optimizing hardware utilization. Experiments show that this method significantly reduces end-to-end latency compared to baseline approaches.

### Strengths
1. The work addresses the critical challenge of deploying massive LLMs under tight memory constraints by combining model offloading with sparse activation. Unlike prior systems such as PowerInfer that rely on static neuron partitioning based on offline profiling, DynamicInfer introduces a dynamic, input-aware scheduling mechanism that adapts neuron placement between CPU and GPU at runtime. The method dynamically adjust the activated neuron with optimization from system level to minimize the latency. Practical and Timely Problem: The focus on enabling high-performance LLM inference on consumer-grade hardware is highly relevant and impactful. With growing interest in on-device AI, this work provides a compelling solution to a real-world deployment bottleneck.
2. Strong Technical Innovation: The shift from static to runtime-adaptive scheduling is a key conceptual advance. The design of cross-layer sparsity prediction, combined with a greedy yet effective optimization framework, enables fine-grained, input-dependent control over neuron placement—a clear improvement over fixed partitioning schemes.
3. Well-Rounded System Design: The integration of three synergistic components—dynamic caching, load balancing via threshold tuning, and I/O pipelining—forms a cohesive and robust system. The attention to engineering details (e.g., contiguous memory layout, asynchronous transfers) enhances practicality and performance. The method constrains the memory, communication and macro residency to balance dynamic adaptability

### Weaknesses
1. As far as I know, the I/O similarity between the initial layers and deep layers of LLMs is quite low—in some other models, this phenomenon may be even more pronounced. In such cases, does cross-layer sparsity prediction still work effectively for these layers or hidden states?
2. Moreover, do you have any additional mechanisms or fallback strategies specifically designed for those exceptional layers with very low similarity?
3. The data transfer between CPU and GPU is rather slow compared with GPU memory access. In your article, does the cross layer prediction fully cover the time of weight transfer? It'll be helpful if you provide detailed data.
4. There are some typos and mismatch of your illustrations with your pictures that may cause misunderstanding. The "Figure3 B" in Observation one seems to be Figure2 B?

### Questions
1. Does cross-layer sparsity prediction still work effectively for layers that have low I/O similarity?
2. Moreover, do you have any additional mechanisms or fallback strategies specifically designed for those exceptional layers with very low similarity? Besides, can you provide some data to show cross-layer prediction has mere impact on prediction accuracy?
3. The data transfer between CPU and GPU is rather slow compared with GPU memory access. In your article, does the cross layer prediction fully cover the time of weight transfer? It'll be helpful if you provide detailed data.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents DynamicInfer, a runtime neuron offloading framework for efficient LLM inference on consumer-grade GPUs. The key innovation is dynamic neuron scheduling that adapts to input-dependent activation patterns, in contrast to PowerInfer's static offline partitioning. The system achieves significant speedups (up to 253% over llama.cpp and 59% over PowerInfer) through three main mechanisms: hierarchical neural caching, load-aware neuron activation, and activation-aware prefetching.

### Strengths
**Well motivated Problem**
- Figure 2,3 underlines the importance of dynamic neuron activations.
- The two observations clearly clarify the drawbacks of static partitioning that PowerInfer uses.

**Comprehensive Engineering efforts**
- Authors implement the core inference engine in C++ and CUDA for efficiency. The three component approach is well integrated to minimize overhead between various parts.
- Constraint design shows a thorough understanding of latency profiling.
- Contiguous memory layout (Appendix E.2.1) for FFN neurons is interesting. 
- Benchmarked on two different generations of NVIDIA GPUs with different memory bandwidths.

### Weaknesses
**Need Additional Comparisons.**
-   As authors note, this is not the first work to dynamically detect activations across the layers. SparseInfer (Shin et al., 2025) (cited by authors) also provides training-free prediction for ReLU-activated LLMs. A direct comparison is essential since both papers target the same models and claim similar benefits. At minimum, discuss architectural differences and when each approach is preferable.
-   Section D.1 states that the open source implementation of PowerInfer differs from the main paper. The results are hence unfair to compare. Authors should ideally contact the authors to implement and/or discuss the implications to the results.
-   Latency evaluation on at least one more dataset can help show robustness.
  
**Insufficient details on the proposed approach.**
-   How accurate is the trained sparse predictor for new samples? across layers? across models?
-   A number of constants are introduced w.r.t the optimization problem and constraints in E.3. How are they measured? How much do they depend on the hardware?
-   E.4.1 is overly simplified. Lacks justification for Bernoulli distribution. “average error” can vary across layers and models. How is k decided for the sparsity predictor beforehand? Is it data/hardware/model agnostic? what is the procedure?
-   Can you explain the difference between your Macro-level residency constraint and PowerInfer? Although it makes sense to introduce, it does not seem very important from Fig10.
-   Line 1048 abruptly switches from the optimization problem to the greedy strategy. Can you discuss the differences in optimal solution and how much it affects? At the very least, an empirical comparison on some instances would be helpful.
-   Can you provide a table showing peak memory usage for GPU and CPU for PowerInfer and DynamicInfer?
    
**Writing needs polish.**
-   The references are not clickable.
-   The visual and text layer of the document are mismatched, making it hard to select text.
-   The majority of details about the method are in the appendix.
-   Captions should contain more details.
-   A Grammar check would be helpful.

### Questions
-   While the comparison with PowerInfer is reasonable, can you justify comparing your method against llama.cpp? It supports a much larger class of models (non-ReLU, quantization, etc) and was not tailored for efficient inference. It seems unfair.
-   How sensitive is the method w.r.t to λ, the neuron importance metric.
-   Line 361, why did you mention “SparseInfer” ?

### Soundness
2

### Presentation
2

### Contribution
3
