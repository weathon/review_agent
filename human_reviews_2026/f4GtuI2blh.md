# CudaForge: An Agent Framework with Hardware Feedback for CUDA Kernel Optimization

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Developing efficient CUDA kernels is increasingly critical for AI applications such as large-scale LLM training. However, manual kernel design is both costly and time-consuming, motivating automatic approaches that leverage LLMs for code generation. Existing methods for automatic kernel generation, however, often produce low-efficiency kernels, incur high computational overhead, and fail to generalize across settings. 

In this work, we propose CudaForge, a training-free multi-agent workflow for CUDA kernel generation and optimization. Our workflow is inspired by the iterative workflow of human experts, which contains steps such as developing initial kernels, testing correctness, analyzing hardware feedback, and iterative improvement.
More specifically, CudaForge employs two LLM agents -- a Coder and a Judge -- that iteratively generate, correct, and optimize CUDA kernels, while integrating hardware feedback such as Nsight Compute (NCU) metrics. In our extensive evaluations, we show that CudaForge, by leveraging base models like OpenAI-o3, achieves 97.6\% correctness of generated kernels and an average 1.68$\times$ speedup over PyTorch baselines, substantially surpassing state-of-the-art models including OpenAI-o3 and Kevin on KernelBench. Beyond accuracy and speed, CudaForge demonstrates strong generalization across GPUs (A100, RTX 6000, 4090, 3090) and base models (OpenAI-o3, GPT5, gpt-oss-120B, Claude-Sonnet-4, QwQ-32B), while maintaining high efficiency. In particular, generating an optimized kernel takes about $25$ minutes on one RTX6000 and incurs \$0.30 API cost. Our results highlight that multi-agent, training-free workflows can enable cost-effective, generalizable, and high-performance CUDA kernel optimization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents CudaForge, a training-free multi-agent framework for CUDA kernel optimization. By integrating GPU kernel performance metrics directly into its feedback loop, CudaForge efficiently refines generated kernels, achieving notable speedups while remaining cost-effective.

### Strengths
- The paper clearly identifies the limitations of prior work and demonstrates significant performance improvements over existing approaches.
- The evaluation is thorough and well-executed. In particular, the ablation study is clearly presented and highlights the key NCU metrics that drive performance gains in the generated CUDA kernels.
- The work clearly shows that incorporating NCU metrics is fundamental to improving CUDA kernel performance, and this insight has broader applicability to other agent-based optimization workflows.
- The inclusion of complete prompts for both the generator and judge models, along with detailed model specifications, makes it easy to produce the results.

### Weaknesses
- One of the paper's central claims is the use of a two-agent workflow, where one agent judges and the other codes. While the results do indicate that separating these roles yields better outcomes than a single agent handling both tasks, this observation is not particularly novel. Similar findings have been reported in prior work (e.g., [1]) and in domains beyond CUDA kernel generation. As a result, the paper’s unique contribution in this regard is unclear.
- Another claimed contribution is the exploration of which NCU metrics to incorporate into the judge's feedback. However, the identification and use of representative NCU metrics has been studied extensively in prior literature (e.g., [2], [3], [4]). This work does not appear to provide significant innovation beyond existing methods. A stronger positioning would involve explicitly situating the approach within this prior body of work, clarifying what is new, and citing relevant references to give readers a more complete perspective.
- The main takeaway seems to be that incorporating a subset of NCU metrics into feedback can improve kernel efficiency. While this is a useful observation, the paper offers limited additional insights or deeper analysis beyond this point.
- The paper suggests that reducing the number of NCU metrics prevents the coder from being overwhelmed. However, the evidence is conducted in a single task (Figure 5). It remains unclear whether this effect generalizes across a broader set of tasks.
    - The paper does not explore what the "sweet spot" is for t
he number of NCU metrics. Is trial-and-error the intended approach, or are there principled guidelines?
    - It is also unclear whether the observed performance gains stem from the quality of the selected metrics (i.e., stronger correlation with performance) or simply from reducing the volume of NCU metrics.
- Minor issues
    - In the section “Comparison with O3-10-O (optimization-only Judge)”, it would be helpful to include concrete numbers to quantify how much correctness is compromised.
    - On line 362, Appendix E is missing its reference.
    - There are inconsistencies in spelling: some plots use "optimisation" while the main text uses "optimization."

[1] Lange, Robert & Sun, Qi & Prasad, Aaditya & Faldor, Maxence & Tang, Yujin & Ha, David. (2025). Towards Robust Agentic CUDA Kernel Benchmarking, Verification, and Optimization. 10.48550/arXiv.2509.14279.   
[2] S. Che et al., "Rodinia: A benchmark suite for heterogeneous computing," 2009 IEEE International Symposium on Workload Characterization (IISWC), Austin, TX, USA, 2009, pp. 44-54, doi: 10.1109/IISWC.2009.5306797. keywords: {Kernel;Multicore processing;Parallel processing;Application software;Yarn;Benchmark testing;Central Processing Unit;Energy consumption;Microprocessors;Computer architecture},  
[3] B. Hu and C. J. Rossbach, "Altis: Modernizing GPGPU Benchmarks," 2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), Boston, MA, USA, 2020, pp. 1-11, doi: 10.1109/ISPASS48437.2020.00011. keywords: {Runtime;Graphics processing units;Focusing;Production;Computer architecture;Benchmark testing;Hardware;GPGPU},  
[4]  Che S, Skadron K. BenchFriend: Correlating the performance of GPU benchmarks: Correlating the performance of GPU benchmarks. The International Journal of High Performance Computing Applications. 2013;28(2):238-250. doi:10.1177/1094342013507960

### Questions
1. The paper adopts 24 NCU metrics, but the rationale for this specific number is unclear. Why is 24 considered optimal? Would a smaller subset (e.g., the top 10 most representative metrics) yield comparable performance? Conversely, is there evidence of a threshold beyond which adding more metrics begins to degrade kernel speedup—either gradually or abruptly? A sensitivity analysis would strengthen the argument here.
2. It remains ambiguous whether the observed kernel speedup improvements are primarily driven by the quality of the selected NCU metrics (i.e., their correlation with performance outcomes) or simply by the quantity of metrics included in the feedback. Clarifying this distinction would provide deeper insight into the mechanism behind the improvements.
3. Results in Table 4 suggest that the choice of Coder model has a stronger influence on the quality of generated code than the Judge. If this is the case, could a smaller or less capable Judge, when paired with a more powerful Coder, achieve comparable performance?

### Soundness
3

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
CudaForge proposes multi agent framework with iterative refinement to produce performant CUDA kernels. Authors propose use of two models working independently 1) coder: focuses on generating code given a prompt and optionally feedback, 2) judge: focuses on analysing execution and hardware feedback and generate cues for coder model to improve correctness or performance. Authors also provide a methodology to parse relevant information from hardware feedback and this helps in removing redundant and irrelevant metrics that might jeopardise producing good kernels. Authors have presented comparison with other baselines and good enough ablation study.

### Strengths
- Propose train-free iterative refinement based approach for CUDA kernel generation.
- Demonstrate the use of LLMs with different identities (coder & judge) in producing performant CUDA kernels.
- Provide a systematic methodology to extract and refine the output of Nsight profiler. 
- Provide detailed analysis of related methods and bring forth interesting observations.
- Method has been shown to work across various frontier and open source models.

### Weaknesses
- Efficacy of this approach is not demonstrated by authors on popular but low resource languages such as Triton.
- Unlike other approaches, there is no clear methodology of evaluation specified in the paper. Precise evaluation setup is extremely important in such tasks.
- Performance measurement with native pytorch implementation without torch.compile does not reflect a comparison with a true baseline.

### Questions
- Does CudaForge scale to Triton programming? What performance improvements can be achieved there?
- How does CudaForge methodology evaluate speedup of generated kernels?
- How does this approach compare against implementations like AlphaEvolve/OpenEvolve?
- How does CudaForge performance metrics look like when compared against torch.compile version of pytorch?

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
3

### Summary
CudaForge proposed two agents: Coder and Judge. The Judge analyzes runtime errors and nsight compute performance metrics, identifies bottlenecks such as memory stalls or low occupancy. It provides optimization feedback to the Coder. Coder regenerates an improved kernel. This loop continues until convergence. CudaForge does not requrie reinforcement learning or training.

### Strengths
1. Originality
Incorporating nsight compute profiling data is an effective approach. The outcome is highly verifiable. This bridges a gap between abstract code generation and hardware-aware tuning, mimicking expert workflows in a systematic way. The separation of roles into a Coder and Judge are sound and understoodable. CudaForge performs optimization purely at inference time, showing that meaningful performance gains are achievable without learning-based fine-tuning.

2. Quality
The evaluation covered tasks from KernelBench, multiple difficulty levels, and A100 / RTX 6000 / 4090 / 3090 setup.
The comparisons against OpenAI-o3, Kevin are fair. Ablations isolate the impact of correction vs. optimization feedback, demonstrating sound causal reasoning behind design choices. The authors report correctness, speedup and detailed prompts. It is helpful for other researchers to reproduce the result.

3. Clarity
human vs. agent workflow diagram effectively illustrate the iterative process. The case study of CrossEntropyLoss optimization makes the workflow intuitively understoodable. The breakdown of Judge behavior, metric selection algorithms, and prompt templates offers transparency.

4. Significance
The work may establish a new design that uses nsight compute profiling data from real hardware. This is applicable beyond applicable beyond CUDA. Automating CUDA kernel optimization directly addresses one of the most labor-intensive bottlenecks in deep learning systems development. The framework could democratize performance engineering and lower the barrier for custom GPU optimization. Most importantly, it's training free.

### Weaknesses
the overall multi-agent refinement structure follows a familiar template used in prior agent-based code generation frameworks using  self-refine. The core advance lies in the feedback modality rather than a fundamentally new learning or reasoning principle. The paper draws a hard line between training-free and RL-based paradigms but doesn’t explore hybrid approaches. In practice, the line can be blurred. If the kernel perf is verifable, it's possible to train the model for better answers

While the paper reports end-to-end time (≈25 min per kernel), it doesn’t dissect where that time is spent — compilation, profiling, or LLM inference. This makes it difficult to assess which parts of the workflow dominate cost and how well it scales for large codebases.

### Questions
How do we know the kernel is 100% numerically correct? It's hard to just run a few input and claim it's right, because the input are limited samples. 

If perf and correctness are verifiable, is it possible to extend to RL style training to improve the model

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes CudaForge, a framework for LLM-based kernel generation that incorporates separate Coder and Judge models, with the judge being guided by actual hardware profiling feedback.

### Strengths
The inclusion of hardware execution feedback seems like a critical step in improving autogenerated kernels, and the investigation into which metrics actually matter as feedback is generally useful beyond this paper (not only to limit the context to be put into the LLM, but also, gathering reduced statistics makes profiling faster).

The method achieves very good correctness scores, while being computationally cheaper than competitors; in particular, it is training-free.

Having an example of how the LLM achieves its improvements and how it fails in certain cases  (appendix A) is very good; though I'd like to  see the actual kernel implementations, to verify whether the improvements proposed in figure 4 are actually implemented.

Also, thanks for including the actual prompts used in this work; this should make it more easily reproducable than some of its competitors.

### Weaknesses
Unfortunately, KernelBench, as a benchmark, is quite flawed, because many of its tasks use shapes that are too small, exacerbating the overheads induced by not using torch.compile as the baseline.

It seems hard to believe that on something as essential as cross-entropy, there'd be a 4x speedup left on the table;

Figure 4 suggests that the framework is doing something promising, but I am very skeptical about the reported speedups reflecting meaningful scenarios.

I'm not sure the "Comparison with O3-10-O (optimization-only Judge)" paragraph adds anything substantial to the paper; this might be better placed in the appendix, and Figure 4 moved to the main part.

### Questions
How does the model perform on larger input shapes. Have you independently verified that the model is not exploiting some weakness in the evaluation procedure, especially in cases where very large speed-ups are reported.

What fraction of speed-of-light is achieved by the baseline, what by the generated kernels.

### Soundness
2

### Presentation
3

### Contribution
2
