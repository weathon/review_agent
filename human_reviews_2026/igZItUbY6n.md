# CUDA-L1: Improving CUDA Optimization via  Contrastive Reinforcement Learning

- Avg Score: 3.50
- Decision: Accept (Poster)
- Scores: 4, 8, 0, 2

## Abstract
The exponential growth in demand for GPU computing resources has created an urgent need for automated CUDA optimization strategies. 
While recent advances in LLMs show promise for code generation, current state-of-the-art models achieve low success rates in improving CUDA speed. In this paper, we introduce CUDA-L1, an automated reinforcement learning (RL) framework for CUDA optimization that employs a novel contrastive RL algorithm. 

CUDA-L1 achieves significant performance improvements on the CUDA optimization task: trained on NVIDIA A100, it delivers an average speedup of {\bf ×3.12} with a median speedup of {\bf ×1.42} against default baselines over across all 250 CUDA kernels of KernelBench, with peak speedups reaching {\bf ×120}. In addition to the default baseline provided by KernelBench, CUDA-L1 demonstrates {\bf ×2.77} over Torch Compile, {\bf ×2.88} over Torch Compile with reduce overhead, and {\bf ×2.81} over CUDA Graph implementations. Furthermore, the model also demonstrates portability across GPU architectures, achieving average speedups of {\bf ×3.85} (median {\bf ×1.32}) on H100, {\bf ×3.13} (median {\bf ×1.31}) on L40, {\bf ×2.51} (median {\bf ×1.18}) on RTX 3090, and {\bf ×2.38} (median {\bf ×1.34}) on H20 despite being optimized specifically for A100. 

Beyond these benchmark results, CUDA-L1 demonstrates several properties: CUDA-L1 1) discovers a variety of CUDA optimization techniques and learns to combine them strategically to achieve optimal performance; 2) uncovers fundamental principles of CUDA optimization, such as the multiplicative nature of optimizations; 3) identifies non-obvious performance bottlenecks and rejects seemingly beneficial optimizations that actually harm performance.
The capabilities demonstrate that, RL can transform an initially poor-performing LLM into an effective CUDA optimizer through speedup-based reward signals alone, without human expertise or domain knowledge. 
In this process, it identifies CUDA optimization patterns, discovers new techniques, synthesizes them to achieve speedups, and more importantly, 
extends the acquired reasoning abilities to new kernels.
This paradigm opens possibilities for automated optimization of CUDA operations, and holds promise to substantially promote GPU efficiency and alleviate the rising pressure on GPU computing resources.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CUDA-L1, a comprehensive framework for automating the optimization of CUDA kernels using large language models (LLMs) enhanced through contrastive reinforcement learning. The framework targets the challenge of improving GPU computing efficiency by enabling LLMs to autonomously generate optimized CUDA implementations that are both correct and performant.
CUDA-L1 adopts a three-stage training pipeline—supervised fine-tuning with data augmentation, self-supervised refinement, and contrastive reinforcement learning—to progressively enhance the model’s ability to reason about, generate, and evaluate CUDA code. Through comparative learning, the model integrates execution feedback directly into its reasoning process, allowing it to learn from past examples of successful and unsuccessful optimizations.
The system is evaluated on a large and diverse set of CUDA kernels across multiple GPU architectures. The paper further contributes a set of CUDA Graph implementations for KernelBench, enhancing community baselines.

### Strengths
- The extensive evaluation across all 250 KernelBench kernels—covering multiple GPU architectures and varying complexity levels—provides strong empirical validation of the proposed approach.
- The enrichment of the KernelBench dataset with CUDA Graph implementations represents a valuable and reusable contribution to the research community.
- The identification of non-obvious optimization techniques (e.g., mathematical short-circuiting) highlights the potential of reinforcement learning to uncover previously unexplored performance improvements.

### Weaknesses
- The paper conflates CUDA kernel optimization with PyTorch-level program optimization. Case studies such as LSTM and 3D convolution reveal that the “optimized” code primarily alters high-level operator invocation and execution patterns, leaving the underlying CUDA kernels unchanged and thus creating a misleading impression of kernel-level innovation.
- While the paper demonstrates impressive success and speedup rates, it lacks an in-depth analysis of cases where the optimization failed or underperformed.
- The work does not discuss the computational resources required to train the CUDA-L1 system. Given the multi-stage training and RL fine-tuning involved, such costs could be substantial.
- The paper focuses heavily on empirical results but provides limited theoretical explanation for why the contrastive-RL approach outperforms alternatives.
- Although the paper claims that the reward checking model “successfully identifies reward hacking over 60% of the time,” it does not quantify how much this detection reduced actual reward hacking, nor whether the system can recognize previously unseen hacking strategies.
- While the internal baselines and ablations are comprehensive, the work lacks direct comparison with external, published state-of-the-art methods.

### Questions
- Given that the paper's core contribution is framed as "CUDA optimization" but the evidence suggests most improvements come from higher-level framework patterns, how do you distinguish between these two distinct types of optimization in your evaluation methodology?
- Since CUDA evolves over time, how would CUDA-L1 handle updates to the CUDA toolkit? Would it require full retraining, or does the model have mechanisms to adapt to new features incrementally?
- The paper builds upon deepseek-v3-671B. How critical is this specific large-scale model to CUDA-L1’s performance? Have experiments been conducted with smaller open-source models, and if so, how does performance degrade?
- The case studies show the model discovering both high-level algorithmic improvements and low-level CUDA implementation optimizations. Does CUDA-L1 include an explicit mechanism to decide the level of abstraction for optimization, or does this behavior emerge naturally from the RL process?
- To aid reader understanding, could the authors include a concrete example (e.g., a filled-in mid-training prompt) in the appendix? While Table 9 outlines the general format, an actual example with real—possibly truncated—code snippets and corresponding performance scores would better illustrate how comparative exemplar guidance works in practice.
- Will the training code, reward-checking implementation, or exemplar-guided prompts be released to facilitate replication by other groups?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces CUDA-L1, a multi-stage training pipeline designed to enhance large language models’ (LLMs) capability to generate efficient CUDA kernels. The proposed framework comprises three stages:
- Supervised Fine-Tuning (SFT): The model is trained on synthetic data generated by various state-of-the-art LLMs to increase its exposure to diverse CUDA programming patterns.
- Self-Supervised Learning: The model is further refined by leveraging its own successful generations as additional training data, thereby improving the overall success rate.
 - Contrastive Reinforcement Learning (RL): The model is trained with contrastive rewards applied both at the parameter level and within the LLM context, targeting execution efficiency.
The first two stages primarily aim to maximize correctness and success rate, while the final stage focuses on improving kernel execution speed.

Empirical evaluations demonstrate that CUDA-L1 achieves significant acceleration and near-optimal success rates compared to both human-designed and LLM-based compiler baselines. Moreover, the method shows strong generalizability across different GPU architectures.

### Strengths
- The paper is well-written and easy to follow, presenting all critical information clearly in the main text.
- The proposed approach, particularly the contrastive RL stage, is well-motivated. Training the LLM to reason about previous code performance is a thoughtful design, and ablation studies demonstrate its effectiveness.
- Outstanding performance: Empirical results show near 3× improvement on KernelBench with a near-perfect success rate, highlighting the method’s substantial practical impact.

- Comprehensive and transparent analysis: The evaluation and supporting analyses are detailed and informative, with some examples:
  - Detailed performance comparison: Main results include detailed distributions across different difficulty levels in KernelBench.
  - Ablation studies: Systematic studies illustrate the contributions of each stage in the training pipeline. Comparisons with straightforward alternatives (e.g., evolutionary methods or simple RL such as Stage 1+2+GRPO in Table 2) confirm the superiority of the proposed pipeline.
  - Cross-hardware evaluation: Experiments on diverse hardware demonstrate consistent success rates, emphasizing the generalizability advantage of LLM-based methods over traditional human-designed compilers.
  - Transparency: The appendix provides a lot of details, e.g. examples of reward hacking, showing the model attempting to manipulate profiler timers via asynchronous streams.
  - Case studies: Illustrative examples in the appendix demonstrate how the model improves CUDA code generation in practice.

### Weaknesses
- It is not entirely clear how the model is evaluated. Specifically, does the evaluation involve iterative refinement over multiple reasoning steps, or is it conducted in a one-shot generation manner? Clarifying this would help assess the robustness of the reported results.

- In Table 1, the average acceleration of nearly 3× appears inconsistent with the P50 speedup of only ~1.2× for the CUDA Graph rows. This raises concerns that the reported average may be skewed by a few poor baseline implementations, resulting in an inflated speedup. Additionally, for Level 2 and Level 3 tasks, which involve more complex operations, the P50 and speedup ratios do not seem to approach a neutral baseline. However, I do not have a better idea on how to present the distribution of acceleration to be more convincing.

### Questions
- How is the mean acceleration metric in Table 1 computed? Is it calculated as the mean of per-task acceleration ratios, or as the ratio of mean run times between CUDA-L1 and the baseline? The former could potentially exaggerate the metric, as individual acceleration ratios may vary widely (e.g., from 0 to 100).
- When transferring the model to other GPUs, such as H100, is it necessary to re-run the training pipeline, or can the trained model generalize directly?
- Given that the model is primarily trained in the context of A100 GPUs, does it demonstrate the ability to adapt to new hardware functionalities, such as Tensor Memory Access (TMA) units in H100?
- Are there any plans to open-source the model or training pipeline, allowing the community to build upon this work?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper tackles CUDA code optimization by fine-tuning a large language model through a sequence of supervised fine-tuning (SFT), self-supervised learning, and contrastive reinforcement learning. The authors first construct a dataset for SFT by prompting diverse LLMs with reference PyTorch code from the KernelBench dataset to produce optimized code variants of the original dataset. They then SFT the model on this corpus and subsequently apply self-supervised learning; the model generates code and successfully executed code samples are fed back for further training. Finally, they employ contrastive reinforcement learning driven by code performance scores to refine the model’s preferences toward faster implementations. The resulting model achieves a 3.12× speedup over the KernelBench code.

### Strengths
The authors proposed a contrastive reinforcement learning strategy that incorporates reward scores into prompts, potentially enhancing the LLM’s ability to generate optimized code on KernelBench tasks.

### Weaknesses
- LLM is both trained and evaluated solely on the same reference code from KernelBench without splitting the training/test dataset. We cannot verify that the finetuned model can create faster optimized versions of code for other tasks or datasets.

- The comparison between CUDA-L1 and the baseline is unfair. CUDA-L1 is fine-tuned on the code that is known to be successfully compiled and executable, benefiting from multiple inference rounds, whereas other baseline LLMs are limited to only five attempts. Since CUDA-L1 requires additional training costs, a proper assessment of the training method should compare it against alternative fine-tuning strategies, not just untuned baselines.

- There is no direct evaluation of the generated CUDA kernels; only the PyTorch code from the KernelBench is measured. Although the end-to-end evaluation may implicitly include the performance of the CUDA kernels, it is not possible to isolate and assess their contribution. The performance improvements observed could instead originate from optimizations in the surrounding Python code (e.g., removing Python overhead). Consequently, the experiment does not provide evidence that the fine-tuned model can optimize CUDA kernels themselves.

- The only novel element in the finetuning pipeline is prompting with a reward score. The methodology does not encode any prior knowledge of CUDA (or PyTorch) into the LLM. Therefore, it seems largely unrelated to CUDA optimization, contrary to its title.

- The authors use terminology that could mislead readers. The second stage of the fine-tuning pipeline should not be described as self-supervised learning but rather as supervised fine-tuning, since failed trials are excluded from the training data. Because this process depends on feedback from an external execution environment, it is more appropriate to regard it as a form of labeling. Moreover, the fine-tuned LLM does not generate optimized CUDA kernels; it produces optimized PyTorch code written in Python, which is fundamentally different from generating efficient CUDA implementations.

### Questions
- Is the reward function defined solely in terms of execution time? Are there any components accounting for memory usage or other resource efficiency metrics?

- Why was the KernelBench dataset chosen for this study? KernelBench is designed to evaluate whether an LLM can generate CUDA code from given PyTorch implementations, not to optimize the PyTorch code itself. Therefore, comparing CUDA-L1’s results with the vanilla KernelBench code and reporting speedups appears to be an inappropriate or uninformative evaluation.

### Soundness
1

### Presentation
2

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
The paper introduces CUDA L1, a three stage system (SFT followed by self supervised filtering followed by contrastive RL) to optimize CUDA code, where the RL stage injects scored prior variants into the prompt to force comparative reasoning and then updates the policy with a GRPO style objective. On KernelBench (250 tasks), the authors report 3.12× mean speedup over the original Pytorch implementation and consistent improvements over torch.compile variants.

### Strengths
•	Strong speedups on a community benchmark and across GPUs. 

•	Clear prompt/exemplar design and informative case studies.

### Weaknesses
•	One of my concerns with the paper is that it does not guarantee the correctness of optimized code. Optimizing code to generate highly optimized CUDA code is well studied in the compiler literature, and many techniques that guarantee the generation of highly optimized, correct code exist. The fact that this proposed method does not guarantee the correctness of the generated code is an important limitation.

•	Automatic code optimization is well studied in compilers. While the comparison with Pytorch torch.compiler is interesting, the paper should also discuss and compare to other state-of-the-art compilers such as TVM, AutoTVM, Ansor, and Triton, which are the de facto baselines for automated GPU optimization.

•	Several wins stem from graph level optimizations (CUDA Graphs, static reuse, stream control) rather than low level kernel transformations. This is fine in practice but should be made explicit. Consider reporting the proportion of gains due to kernel level vs. graph level vs. PyTorch level changes.


•	Contrastive prompts with scored exemplars are compelling, but the policy optimization is standard GRPO. Please clarify more what is algorithmically new.

•	The authors compare to the Pytorch implementations of KernelBench, but it is unclear what backend library is used to accelerate the Pytorch operators. Are the operators accelerated on GPU ? Which library is used to accelerate the operators? Specifying this is very important since this constitutes the baseline and the difference in performance between the different variants is significant.


•	The paper lacks critical training and compute details (GPU hours, steps, batch sizes). Without these, reproducing the reported speedup is unlikely.

### Questions
•	What are the full training hyperparameters and compute for each stage? 

•	What fraction of improvements stems from kernel level vs graph level vs framework level optimizations? 

•	Can you include TVM/Ansor/Triton comparisons?

### Soundness
2

### Presentation
2

### Contribution
2
