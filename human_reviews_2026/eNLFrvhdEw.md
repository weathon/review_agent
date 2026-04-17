# GASDU: Gauss--Southwell Dynamic Update for Efficient LLM Fine-Tuning

- Decision: Reject
- Scores: 8, 4, 2, 8

## Abstract
Parameter-efficient fine-tuning (PEFT) is crucial for adapting large language models (LLMs), yet existing methods trade off accuracy, latency, and compute: some add inference-time modules, others fix a static parameter set that can drift from evolving gradients, and dynamic variants can be costly. We propose **Gauss–Southwell Dynamic Update (GASDU)**, which performs *periodic Gauss–Southwell-k selection: every M steps it uses the current gradients to select the (k) largest-magnitude coordinates and updates only those entries while reusing the mask until the next refresh. The Top-(k) selection is implemented in a streaming, tile-wise way to avoid materializing dense gradients, making the amortized refresh cost negligible. Theoretically, under a local Polyak–Łojasiewicz condition, we prove that GASDU enjoys a linear convergence rate scaled by a measurable gradient-retention factor and show that this factor degrades sublinearly within each refresh window. This sublinear decay implies that a moderate (M) can maintain a high retention factor, which in turn explains GASDU’s near–full–fine-tuning behavior. Empirically, GASDU sustains high retention between refreshes at an extreme parameter budget (0.01%) and consistently outperforms strong PEFT baselines and closely tracks or exceeds full fine-tuning across diverse commonsense and arithmetic reasoning benchmarks and LLMs (LLaMA-2/3 and GPT-OSS-20B).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a parameter-efficient fine-tuning (PEFT) method for large language models (LLMs) that updates only a dynamically selected subset of parameters. Instead of using static masks or low-rank adapters or incurring high overhead from dynamic sparse methods, GASDU periodically performs Gauss–Southwell–k selection. Linear convergence is proved under a local Polyak–Łojasiewicz (PL) 
condition. Empirically, GASDU is shown to match full fine-tuning using only 0.01% of parameters, achieving speedup and memory savings.

### Strengths
- The paper is well written. Anonymous reproducibility repository provided.

- The use of Gauss–Southwell-$k$ coordinate selection in a PEFT context is original and well-motivated.

- Provides a clean convergence proof under a local PL condition, which introduces the gradient retention factor, a useful measurable diagnostic linking sparsity, update cadence, and convergence rate.

- Benchmarks span arithmetic reasoning and commonsense tasks. The ablation on refresh period $M$ is convincing.

### Weaknesses
- Only 0.01% update budget tested; performance across larger budgets would help understand scaling.

- The authors could clarify the effect of $k$.

- On line 777, one citation is not properly rendered.

### Questions
- Theoretically, how does the choice of $k$ impact the convergence behavior of the proposed algorithm?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a PEFT method based on dynamic sparse updates, i.e., dynamically selecting the top-k largest-magnitude gradients for updates and retaining/refreshing the selected mask every M steps. The paper provides both theoretical analysis and an efficient implementation of the method. The approach is evaluated on LLaMA and GPT-OSS models, showing improved fine-tuning accuracy and comparable throughput.

### Strengths
1). The paper is well written, and the proposed method is reasonably novel.

2). It provides theoretical proofs that the proposed sparse update achieves a linear convergence rate and that the masks can be reused during each refresh period.

3). The paper presents an efficient implementation that avoids materializing the full gradient matrix, achieving throughput comparable to other PEFT methods.

### Weaknesses
1). Unlike LoRA, the proposed dynamic update method appears to require one full model per downstream task, which limits its scalability for multi-task deployment in practice.

2). The baselines used for comparison are relatively weak. The paper relies on basic baseline settings, whereas both LoRA-based and fixed-mask methods have advanced significantly over the past year (see a few selected references below). Without comparison to state-of-the-art methods, it is difficult to assess the true advantages of the proposed approach.

References:

•	LoRA-One: http://arxiv.org/abs/2502.01235

•	LoRA-Pro: http://arxiv.org/abs/2407.18242

•	LoRA-GA: http://arxiv.org/abs/2407.05000

•	SMT: https://openreview.net/forum?id=GbgCRJedQ7

•	Diablo: http://arxiv.org/abs/2506.03230

### Questions
In addition to the weaknesses noted above:

1). In Table 3, the ablation on M shows large variance—results for M between 1 and 100 appear to have little impact and show no clear trend. Can the authors provide an explanation?

2). Is the proposed method (GASDU) implemented with DeepSpeed’s FusedAdam optimizer? If so, how are the first and second moments handled?

3). Can this algorithm be applied in tensor-parallel or fully sharded data-parallel settings?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes GASDU, a dynamic sparse fine-tuning method that periodically selects top-k gradient coordinates (Gauss–Southwell rule) and updates only those parameters, reusing the mask for several steps. It provides convergence analysis under a local PL condition and conducted numerical experiments.

### Strengths
Simple and well-motivated dynamic sparse update rule.
Clear theoretical analysis with convergence guarantees under a PL assumption.

### Weaknesses
1. The proposed method appears to work only with plain SGD. When combined with adaptive optimizers such as Adam, the dynamic masking conflicts with momentum updates: maintaining moment estimates for all parameters contradicts the claimed memory efficiency, while periodically resetting them typically leads to instability and poor convergence. If only plain SGD is used, there is little need for a separate algorithm—SGD updates can be directly fused into backpropagation (i.e., updating parameters as gradients are computed and then clearing them), which essentially replicates the proposed masking behavior.
2. The paper lacks comparisons with recent LoRA variants (e.g., LoRA-One, LoRA-GA, MiLoRA), which substantially improve fine-tuning quality and represent stronger baselines.
3. Because the proposed algorithm updates entries across the original weight matrices rather than using lightweight adapters, it loses one of the key benefits of PEFT—the ability to store and switch between multiple small task-specific adapters. GASDU cannot easily support such modularity.
4. The theoretical analysis mainly restates the classical PL convergence result with minor modifications for a masked gradient. The provided theory does not offer clear insight into why the proposed method should outperform existing approaches or how it explains the observed empirical gains.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces a new fine-tuning method that updates only a tiny fraction of model weights by refreshing a sparse mask every few/M steps using a Gauss–Southwell selection. By selecting only top-k gradients, the method retains training efficiency through lower memory and compute costs. The authors further show improved convergence speeds and that the approach can maintain on par performance (or at least better than previous PeFT) with full fine-tuning despite using a subset of the parameters. Along with the performance improvement comes comparable (to PeFT methods) speedup and memory efficiency over full-finetuning.

### Strengths
1. Broad model compatibility: GASDU demonstrates consistent effectiveness across several large language model families, including LLaMA-2, LLaMA-3, and GPT-OSS-20B. This cross-architecture success indicates that the method is not tightly coupled to any specific model design, highlighting its robustness and potential for general adoption in diverse fine-tuning scenarios.
2. Theoretical foundation: The method is supported by a convergence analysis under the Polyak–Lojasiewicz condition, providing formal assurance of stable and predictable optimization behavior. By introducing the gradient-retention factor as a measurable quantity, the authors establish a link between theoretical guarantees and empirical performance, strengthening confidence in the method’s reliability.
3. Practical and efficient design: GASDU’s streaming top-k selection mechanism eliminates the need for dense gradients, substantially lowering per-iteration computational cost and memory usage. This design enables the method to achieve performance that lands somewhere between lightweight approaches like LoRA and more resource-intensive full fine-tuning, maintaining both efficiency and accuracy.

### Weaknesses
1. The current experiments cover commonsense reasoning tasks. To better assess the generality of GASDU, the authors should also evaluate it on benchmarks involving longer input sequences and contextual dependencies.

2. The discussion of sparsity-based PEFT methods is missing. This work focuses on updates to a sparse selection of parameters, along with some online update refreshing. There are some recent relevant works in the domain of PeFT, such as S2FT (NeurIPS 2025)[1] and SparseLoRA (ICML 2025)[2], Galore[3]. Including these would strengthen the discussion on sparsity and help illustrate the novelty of refresh-based updates on a sparse set of weights.

References:

[1] Xinyu Yang, Jixuan Leng, Geyang Guo, Jiawei Zhao, Ryumei Nakada, Linjun Zhang, Huaxiu Yao, Beidi Chen, "S2FT: Efficient, Scalable and Generalizable LLM Fine-tuning by Structured Sparsity", NeurIPS 2025

[2] Samir Khaki, Xiuyu Li, Junxian Guo, Ligeng Zhu, Chenfeng Xu, Konstantinos N. Plataniotis, Amir Yazdanbakhsh, Kurt Keutzer, Song Han, Zhijian Liu, "SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity", ICML 2025

[3] Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, Yuandong Tian, "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection", Arxiv 2024

### Questions
1. Since most tested tasks produce short outputs, such as those in commonsense reasoning, it would strengthen the paper to include evaluations on longer-context and multi-step generation tasks -- like MT-Bench for dialogue or HumanEval for code generation, and arithmetic reasoning benchmarks -- to show how well GASDU scales to extended contexts and more complex output generations. 

2. Could the authors provide some insights into why the retention factor decreases for the first half of training, followed by a steep increase, and maintains a high value for the remainder of training? Previous works, such as SparseLoRA[2] and STEP[4] have shown that sparsity is more sensitive in the early stages and hence choose to keep it dense (i.e, have full retention ratios on PeFT at the beginning), before introducing sparsity in later stages. However, interestingly, the trend in Figure 4.0 points to a less aggressive sparsity in the later stage. 

[4] Yucheng Lu, Shivani Agrawal, Suvinay Subramanian, Oleg Rybakov, Christopher De Sa, Amir Yazdanbakhsh, "STEP: Learning N:M Structured Sparsity Masks from Scratch with Precondition" ICML 2023

### Soundness
3

### Presentation
3

### Contribution
3
