# ComPhy: Composing Physical Models with end-to-end Alignment

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
Real-world phenomena typically involve multiple, interwoven dynamics that can be elegantly captured by systems of Partial Differential Equations (PDEs).   However, accurately solving such systems remains a challenge. In this paper, we introduce ComPhy (CP), a novel modular framework designed to leverage the inherent physical structure of the problem to solve systems of PDEs. CP assigns each PDE to a dedicated learning module,  each capable of incorporating state-of-the-art methodologies such as Physics-Informed Neural Networks or Neural Conservation Laws. 
  Crucially, CP introduces an end-to-end alignment mechanism, explicitly designed around the physical interplay of shared variables, enabling knowledge transfer between modules, and promoting solutions that are the result of the collective effort of all modules. 
  CP is the first approach specifically designed to tackle systems of PDEs, and our results show that it outperforms state-of-the-art approaches where a single model is trained on all PDEs at once.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a modular framework to tackle the optimization problem of PINNs caused by multiple loss constraints, which is named ComPhy. Specifically, ComPhy proposes to split multiple loss functions into several subsets with shared physical quantities. Then, several modules are configured as PINN or NCL models, and each module will be optimized based on one subset of loss functions. Besides, an alignment loss is newly proposed to align the shared physical quantities in different modules. As for inference, ComPhy only needs to infer the module with complete quantities once. Such a modular framework is expected to ease the difficulty in joint optimization of multiple losses and ensure the physical alignment in the final results. The authors provide sufficient experiments to verify the effectiveness of the proposed ComPhy.

### Strengths
(1)	I think the proposed idea is interesting and novel, especially in assigning different loss subsets to different modules.

(2)	The experiments are sufficient to deliver a comprehensive evaluation of the proposed method.

(3)	Rich implementation details are included.

### Weaknesses
### (1) The motivation of ComPhy. Why does this design work?

I think the authors fail to elaborate on why ComPhy performs well. The only statement is in Lines 99-101, that is, “State-of-the-art models like PINNs may suffer from optimization problems when multiple PDEs are involved. ComPhy avoids this issue by using different modules to optimize the different PDEs of the system separately.” However, suppose one of the modules in ComPhy contains complete physics quantities. In that case, it will also be optimized by the newly added alignment losses, which is still a multiple-loss optimization problem. Why is the module optimized in this way better than the model optimized from multiple PDE losses? I think an intuitive understanding is required.

Besides, let us consider the example in Section 2.3. If ComPhy employs two PINN models as modules, at the beginning of training, it is really hard for the second module optimized with Eq.~(8) to generate a reliable solution. Why can the alignment loss help the first module be optimized better?

Therefore, I cannot understand why ComPhy can help with the training. **Maybe the visualization of training curves (training, alignment and test losses) can be a good choice for elaboration. In addition, some theoretical analyses or thought experiments are expected.**

### (2) Too many unjustified or vague claims.

I think this paper contains many unsupported claims or statements, which seriously damage the scientific rigor. Here are some examples:

-	Abstract: “CP is the first approach specifically designed to tackle systems of PDEs”. Suppose the authors refer to “systems of PDEs” as the combination of multiple PDE equations. In that case, I think there are many related works that tackle the optimization problem of balancing multiple PINN losses, such as [1].

-	Line 99: “State-of-the-art models like PINNs may suffer from optimization problems when multiple PDEs are involved”. What kind of “optimization problems” do you mean? I think if the authors cannot detail this statement, the motivation of this paper is unclear.

-	Line 111: “The solution to a PDE system is unique only if all the PDEs are satisfied at once (Evans, 2022).” Although this is not a core statement, it should be noted that many PDEs contain multiple solutions.

-	Eq. (3): The authors do not provide a clear definition for the L_{align}, since in Eq. (3), each row defines one type of L_{align}. I do not know what the final version used in Eq. (4) is.

-	Table 1: It is really hard to understand the last column. After a long time thinking, I understand that each row of the last column represents one configuration in ComPhy. I think a more direct description is required. 

-	All the numbers, like 100.000 or 600.000, should be 100,000 or 600,000.

-	Line 356: “The authors show that when PINNs are optimized correctly, the gradients tend to be evenly distributed across all layers.” This claim is not correct, since in Wang et al. (2021), the main focus is on the imbalance among different losses. Thus, this statement should be “the gradients of multiple PDEs”.

[1] Wang et al. When and why pinns fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 2022.

### (3) How to decide the configuration of ComPhy, such as pure PINNs, pure NCLs, or a combination of PINN and NCL? 

As presented in Table 2, different configurations lead to quite different results. When using ComPhy, it may take a long time to tune the concrete configuration of the modular framework.

###  (4) About related work. 

I think this paper is not related to the neural operator, which is purely data-driven. The authors should spend more time reviewing papers about PINN optimization, such as [1,2,3].

[1] Wang et al. When and why pinns fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 2022.

[2] Daw et al. Mitigating propagation failures in physics-informed neural networks using retain-resample-release (r3) sampling, ICML 2023.

[3] Wu et al. RoPINN: Region Optimized Physics-Informed Neural Networks, NeurIPS 2024.

### Questions
Please see Weaknesses. To highlight, I think the authors should answer the following questions carefully:

-	Why does ComPhy work well?

-	How to decide the configuration of ComPhy?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces ComPhy, a framework for solving systems of PDEs. Instead of training a single monolithic model on all equations, ComPhy assigns each PDE to a dedicated learning module (e.g., a PINN or NCL) and enforces alignment losses between modules that share physical variables. These alignment losses encourage consistency across modules and improve convergence.

### Strengths
1. The paper introduces a novel modular design. The decomposition of multi-PDE systems into specialized modules is elegant and well-motivated both computationally and physically.

2. Introducing Sobolev-inspired alignment effectively transfers physical information between modules and leads to empirical gains.

3. Gradient distribution studies convincingly explain why ComPhy’s modular approach stabilizes training compared to conventional PINNs.

### Weaknesses
See questions below.

### Questions
1. Managing multiple interacting modules may increase computational and memory overhead, particularly for systems with many PDEs. It would be helpful if the authors could clarify how they address this issue.

2. Since the overall objective combines both alignment losses and module-specific PDE/BC/IC losses, it would be useful to report how sensitive the method is to the relative weighting of these terms. Does performance degrade significantly if the alignment coefficient is varied?

3. Can the modular design generalize to systems where PDEs share only partial or implicit variables?

4. The figures, particularly Figure 1, could be made more intuitive and easier to interpret.

### Soundness
3

### Presentation
2

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
This paper presents ComPhy, a novel modular framework designed to leverage the inherent physical structure of the problem to solve systems of PDEs. ComPhy assigns each equation a dedicated learning module, like PINN or NCL, and introduces an end-to-end alignment mechanism to enable knowledge transfer between different modules. The results show that it outperforms state-of-the-art approaches where a single model is trained on all PDEs at once.

### Strengths
- From my perspective, this is a novel method designed for solving systems of PDEs. It is an interesting research field which have not been explored.
- The proposed method is novel and seems elegant for solving systems of PDEs. The experiment results also demonstrate that it outperforms plain PINNs.
- The paper is well written and easy to follow.

### Weaknesses
- It may be hard, but some theoretical understandings, even intuitive ones, could make the method more convincing.
- After training, we can use a subset of the trained networks to predict all physical variables. Can some networks sharing the same variables have conflicts?
- The paper lacks analysis on the efficiency of the model. For a system of N PDEs, each network requires (N+2) loss terms, and the whole system requires N*(N+2) loss terms; how does it influence the training time compared to plain PINNs.
- Can we replace the current PINNs and NCLs with neural operators? The current framework does not seem to support such modules.

### Questions
See weaknesses.

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
This paper introduces ComPhy (CP), a modular framework for solving systems of Partial Differential Equations (PDEs) using machine learning. The core innovation involves assigning each PDE in a system to a dedicated learning module (such as PINNs or Neural Conservation Laws) and connecting these modules through an end-to-end alignment mechanism. The alignment process enforces consistency between modules that predict the same physical variables, with particular emphasis on derivative-based alignment losses (Sobolev norm and derivative-only alignment). The authors demonstrate that this compositional approach outperforms standard methods where a single model learns all PDEs simultaneously. Experiments span systems ranging from two to five equations, including Navier-Stokes, acoustic wave equations, and magnetohydrodynamics, showing consistent improvements in accuracy. The paper also provides gradient analysis suggesting that CP's modular structure leads to more balanced gradient distributions during training, which may explain its superior performance.

### Strengths
Strengths

The paper demonstrates several notable strengths that support its contribution to the community.
Clear motivation and intuitive approach: The paper effectively motivates the problem of solving coupled PDE systems and presents an intuitive solution. The compositional structure mirrors the mathematical structure of the underlying physical system, making the approach both theoretically appealing and practically sensible. The gradual build-up from methodology to the concrete Navier-Stokes example in Section 2.3 aids understanding.
Strong and consistent empirical results: The experimental evaluation is comprehensive, covering multiple physical systems with increasing complexity. 

Thorough experimental methodology: The authors compare against multiple relevant baselines including PINN with gradient reweighting and adaptive point resampling. The experiments are well-documented with detailed problem setups, boundary conditions, and reference solutions in the appendices. 

Valuable gradient analysis: Section 3.4's gradient histogram analysis provides meaningful insight into why the modular approach succeeds. The observation that CP produces more balanced gradient distributions across layers compared to standard PINNs offers an empirical explanation for the performance gains and could inform future research.
Generalization beyond divergence-free equations: Unlike NCL which is specifically designed for divergence-free fields, ComPhy's framework applies to general PDE systems, demonstrating particular value in experiments like acoustics and MHD where NCL-only approaches would be insufficient.

### Weaknesses
Weaknesses and Concerns

 Insufficient analysis of hyperparameter selection and sensitivity
The paper does not provide clear guidance on choosing the critical hyperparameter λ_align. While Table 2 and Table 3 show results with fixed hyperparameters, there is no ablation study examining sensitivity to this choice or methodology for setting it. 

 Module assignment strategy not systematically addressed
The paper does not provide principled guidance on how to assign PDEs to modules. For the NS-Euler experiment (Section 4.1), the authors test multiple configurations (2xPINN, PINN+NCL, 2xNCL, 3xPINN) but offer no systematic approach for making this choice. Different assignments can lead to different architectures, but the selection process appears ad-hoc.
The paper would benefit from either developing heuristics for module assignment (e.g., based on PDE type, coupling strength, or variable sharing patterns) or demonstrating that the method is robust across reasonable assignment choices.

 Comparison fairness and architecture choices
The baseline single PINN appears to use the same architecture size as individual CP modules (Table 5), meaning the total CP model has substantially more parameters across all modules. It is unclear whether a larger single PINN with comparable total parameter count would close the performance gap. I wonder whether the observed gains arise specifically from the modular structure or merely from the increased model capacity.

### Questions
Statistical significance and error bars
The results tables (Tables 2 and 3) report point estimates without error bars or confidence intervals. Given that neural network training involves stochastic elements (random initialization, batch sampling), reporting means and standard deviations across multiple runs would strengthen the claims.  

Ablation on alignment losses
The authors should include ablation studies showing performance across a range of λ_align values, demonstrate the method's robustness (or lack thereof) to hyperparameter choices

Computational Efficiency and Cost–Performance Analysis 
The training time overhead of CP models is mentioned but not quantitatively justified. It would be useful to discuss whether the additional training cost is proportionate to the observed performance improvement. Presenting these metrics side by side would improve the completeness and transparency of the experimental analysis.

Parameter-matched baseline  
Compare against a single PINN baseline that is parameter-matched to the full CP system (same total parameter count). Alternatively, show scaling curves where single PINN and CP models are compared across increasing total parameter budgets. This will clarify whether gains are due to modularity or simply model capacity.

Alignment learning
Recent studies [1][2] have incorporated alignment learning into PINN-like frameworks, demonstrating its potential to enhance physical consistency and optimization efficiency. Therefore, the authors should discuss how their proposed method relates to these works, highlighting the key differences or advantages.

[1]Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective

[2]Physics-informed Temporal Alignment for Auto-regressive PDE Foundation Models

### Soundness
3

### Presentation
3

### Contribution
3
