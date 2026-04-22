# LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Recent studies have shown that combining parameter-efficient fine-tuning (PEFT) with mixture-of-experts (MoE) is an effective strategy for adapting large language models (LLMs) to the downstream tasks. However, most existing approaches rely on conventional TopK routing, which requires careful hyperparameter tuning and assigns a fixed number of experts to each token. In this work, we propose LD-MoLE, a Learnable Dynamic routing mechanism for Mixture of LoRA Experts that enables adaptive, token-dependent, and layer-wise expert allocation. Our method replaces the non-differentiable TopK selection with a differentiable routing function and a closed-form solution. Moreover, our design allows the model to adaptively determine the number of experts to activate for each token at different layers. In addition, we introduce an analytical sparsity control objective to regularize the number of activated experts. Extensive experiments on the Qwen3-1.7B and Llama-3.2-3B models show that LD-MoLE achieves the highest average scores compared to state-of-the-art baselines, across a diverse set of benchmarks. Our method not only achieves superior performance, but also demonstrates the ability to learn token-dependent and layer-wise expert allocation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LD-MoLE, a Mixture-of-LoRA-Experts framework with learnable dynamic routing based on a closed-form Sparsegen function. A shared MLP predicts a token-wise sparsity factor λ , allowing the model to dynamically adjust the number of activated experts per token in a differentiable and stable manner. The method also includes an analytical sparsity loss for expert control.

### Strengths
1. Originality - The paper proposes a Sparsegen-based dynamic routing method with a mathematically proven sparsity loss, showing novelty and clear motivation.

2. Quality and Clarity – The methodology is well-designed and theoretically sound, with clear writing that makes the approach easy to follow.

3. Significance – The visualization of LoRA experts selected per token clearly demonstrates the effectiveness of the proposed dynamic routing mechanism.

### Weaknesses
1. Insufficient Ablation Studies and Experiments
The paper lacks an ablation study on the load balancing loss weight and its impact on the model's performance. It would be beneficial to explore how varying this parameter affects the stability and performance of the model, especially since load balancing plays a crucial role in expert distribution.  
The paper also does not compare against important baselines like:  
[1] Harder Task Needs More Experts: Dynamic Routing in MoE Models  
[2] HMoRA: Making LLMs More Effective with Hierarchical Mixture of LoRA Experts Including these baselines would provide a more comprehensive evaluation and highlight the specific advantages of the proposed approach.

2. Lack of Large-Scale Model Comparison
The experiments are limited to relatively smaller models (Llama-3.2-3B and Qwen-3-1.7B). A comparison against larger-scale models would help validate the scalability of the proposed method and its ability to handle more complex tasks. This would provide a clearer picture of how the method performs in more resource-intensive settings.

3. Unconventional Benchmark Selection
The choice of experimental benchmarks seems somewhat unusual. The paper compares against a small subset of tasks, whereas other multi-task MoLE baselines (e.g., [3] KASA: Knowledge-Aware Singular-Value Adaptation of Large Language Models and [4] MTL-LoRA: Low-Rank Adaptation for Multi-Task Learning) involve a broader range of tasks. The rationale behind selecting this specific subset is unclear, and a more diverse set of benchmarks could provide a more robust evaluation. A clarification on why this subset was chosen would improve the overall experiment design.

4. Unclear Necessity of Sparsegen Routing
The relationship between Sparsegen routing and the Mixture-of-Experts (MoE) architecture is not fully explained. It is unclear why Sparsegen routing was specifically chosen over other potential methods. Has the paper explored alternatives like Gumbel-Softmax or directly using softmax without Top-K selection? These methods could provide valuable insights and potentially simplify the routing process. A discussion on why Sparsegen is preferred, and a comparison to other discrete or continuous routing methods, would strengthen the theoretical justification for this choice.

5. Computational cost
The paper lacks a quantitative evaluation of the additional computational cost introduced by the Sparsegen routing (such as the sorting operation and MLP prediction), and does not provide direct comparisons with baseline methods in terms of training/inference time, FLOPs, or latency.

References:  
[1] Harder Task Needs More Experts: Dynamic Routing in MoE Models.  
[2] HMoRA: Making LLMs More Effective with Hierarchical Mixture of LoRA Experts.  
[3] KASA: Knowledge-Aware Singular-Value Adaptation of Large Language Models.  
[4] MTL-LoRA: Low-Rank Adaptation for Multi-Task Learning.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

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
This paper proposes LD-MoLE, a learnable dynamic routing framework for Mixture of LoRA Experts (MoLE) that replaces the traditional non-differentiable Top-K selection with a Sparsegen-based routing mechanism. The model adaptively determines the number of activated experts per token and per layer through a shared MLP predicting a sparsity factor λ, allowing token-dependent and layer-wise routing. The authors also introduce an analytical sparsity control loss to regularize the number of activated experts. Extensive experiments on Llama-3.2-3B and Qwen-3-1.7B show consistent improvements over MoLA (Top-K routing) and ReMoLE (ReLU-based routing) across multiple NLP benchmarks. The results demonstrate improved efficiency, stability, and interpretability of dynamic routing in parameter-efficient fine-tuning.

### Strengths
**Novel Differentiable Routing Mechanism:** The paper presents a clean and theoretically grounded alternative to Top-K routing using Sparsegen, providing both differentiability and guaranteed non-zero expert activation, addressing stability issues observed in prior dynamic routing work (e.g., ReMoE).

**Comprehensive Analysis:** The authors provide solid theoretical derivations, ablations on sparsity and λ prediction, and empirical validation on two base LLMs across 8 diverse benchmarks, clearly showing robustness and consistency.

**Strong Practical Value:** The design achieves dynamic, token-aware expert selection without increasing parameter count significantly (thanks to the shared MLP), making it practical for real-world PEFT and instruction-tuning applications.

### Weaknesses
1. **Gains are modest and transferability isn’t established.**
Reported improvements over Top-K/RELU routers (e.g., MoLA, ReMoLE) are incremental on small–mid LLMs and a limited task suite; the paper doesn’t convincingly show zero-shot / cross-dataset transfer (e.g., to harder or different domains such as math/code/MMLU) or robustness under distribution shift. Including a brief analysis (no extra training) of cross-dataset generalization would better support the method’s claims. 

2. **No scale-up evidence (3B→8B→30B+) and limited backbone diversity.**
Experiments focus on small models (e.g., 1.7B–3B), leaving open whether learnable sparsity remains stable/efficient as parameters and depth grow (routing entropy, expert utilization balance, convergence speed, memory/latency). By contrast, scalable MoE work (e.g., Switch Transformer, DeepSeek-V3) validates routing and load balancing at tens–hundreds of billions of parameters; LoRA-MoE baselines like MixLoRA also report on a broader set of bases (e.g., LLaMA/Gemma/Mistral), aiding claims of generality. Adding at least partial scaling curves and results on a more diverse set of backbones would strengthen the paper.

3. **Why choose Sparsegen over other differentiable routers isn’t empirically settled.**
Theoretical appeal aside (deterministic, truly sparse, λ-controllable), the paper lacks a side-by-side comparison under the same budget against other differentiable routing families widely used in practice:
– Soft routing + Top-K variants from large-scale MoE (e.g., Switch/DeepSeek-V3 and modern aux-loss-free balancing);
– Gumbel-Softmax/Concrete reparameterization for differentiable categorical choices;
– ReLU-based differentiable routers.
A controlled ablation (same base/rank/FLOPs) on accuracy, stability, load balance, latency, and activated-expert count would clarify whether Sparsegen offers practical advantages beyond its theory.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes LD-MoLE, a novel **Learnable Dynamic** routing mechanism for **M**ixture **o**f **L**oRA **E**xperts. The method addresses key limitations in existing MoLE approaches by replacing the non-differentiable Top-K routing with a fully differentiable routing function based on Sparsegen projection. LD-MoLE introduces a lightweight shared MLP that predicts a token- and layer-dependent sparsity parameter (λ), enabling adaptive expert allocation. An analytical sparsity loss is derived to explicitly control the number of activated experts. Extensive experiments on Llama-3.2-3B and Qwen3-1.7B demonstrate state-of-the-art performance across diverse benchmarks, outperforming strong baselines like MoLA (Top-K) and ReMoLE (ReLU-based routing).

### Strengths
- The idea of combining Sparsegen-based differentiable routing with a learned λ-predicting MLP is novel and impactful. It elegantly solves the non-differentiability of Top-K and the instability of ReLU routing, advancing the MoLE paradigm.
- Theoretical grounding such as closed-form solutions, proofs is solid. Experiments are comprehensive, covering multiple models, tasks, and ablation studies including sparsity control, λ analysis and zero-activation issue.
- The paper is well-structured, clearly written, and figures effectively illustrate the methodology and findings.
- LD-MoLE provides a principled, efficient, and high-performing framework for dynamic expert routing, with potential influence on both PEFT and MoE research communities.

### Weaknesses
- **Computational Overhead**: While parameter-efficient, the computational cost (e.g., latency, FLOPs) of the dynamic routing mechanism—especially the Sparsegen projection—compared to simple Top-K is not quantified. A brief analysis would strengthen the efficiency claim.
- **Baseline Comparison**: Including a fixed-λ Sparsegen baseline would more directly isolate the contribution of the learned λ (via the MLP) from the contribution of Sparsegen itself.
- **Hyperparameter Sensitivity**: The impact of the new hyperparameters (α, β, and target sparsity k) on performance and stability is not discussed. A sensitivity analysis would improve practical usability.

### Questions
1. The shared MLP predicts λ for the same module type across all layers. Does this design potentially limit the model's ability to learn highly specialized, layer-specific routing strategies? What is the trade-off between parameter efficiency and layer-wise flexibility here?

2. In the sparsity control analysis (Sec. 4.4), using the sparsity loss (β>0) reduces expert count at a performance cost. Do you see a pathway for the model to learn a task-optimal sparsity level automatically, rather than relying on a pre-defined k?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes LD-MoLE, a learnable dynamic routing mechanism for Mixture of LoRA Experts. Built on Sparsegen, the method introduces a differentiable and token-wise adaptive expert allocation strategy, with a learnable sparsity parameter predicted by a shared MLP. The framework also includes a sparsity control objective to regulate the number of activated experts. Experiments across several downstream tasks show improved performance compared to TopK and ReLU-based routing baselines.

### Strengths
The paper presents a clear motivation for improving expert allocation in MoE models and proposes a differentiable routing mechanism with controllable sparsity. The method is well-integrated into the LoRA-based MoE setting, and the overall presentation is clear. Experiments cover multiple tasks and include comparisons against relevant baselines, along with some ablation studies.

### Weaknesses
* **Applicability to Standard MoE Architectures**: While LD-MoLE is instantiated within the MoLE setting, it remains unclear whether the proposed routing mechanism can be seamlessly integrated into conventional MoE architectures with full FFN experts or applied during pretraining. Clarifying its plug-and-play compatibility with standard MoE training and inference pipelines would help better demonstrate the generality and practical advantages of the approach.

* **Insufficient Efficiency Evidence**: The paper claims that LD-MoLE effectively reduces the number of activated experts, but the evaluation largely reports task accuracy and qualitative activation plots. To substantiate efficiency, more concrete system metrics (e.g., FLOPs, latency, throughput and peak memory) would better support the conclusion that reduced activation translates into real compute savings.

* **Missing Discussion Regarding Sparse Routing.**: the paper lacks comparison with prior methods that target similar goals through dynamic routing or explicit sparsity control [1-3]. A clearer conceptual and empirical comparison with related works would strengthen the contribution.

[1] Team M L C, Li B, Lei B, et al. Longcat-flash technical report[J]. arXiv preprint arXiv:2509.01322, 2025.

[2] Zeng Z, Miao Y, Gao H, et al. Adamoe: Token-adaptive routing with null experts for mixture-of-experts language models[J]. arXiv 
preprint arXiv:2406.13233, 2024.

[3] Yue T, Guo L, Cheng J, et al. Ada-k routing: Boosting the efficiency of moe-based llms[C]//The Thirteenth International Conference on Learning Representations. 2024.

### Questions
Please refer to Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
