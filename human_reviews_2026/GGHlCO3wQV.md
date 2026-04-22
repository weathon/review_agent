# DEMO:Diffusion-based Evolutionary Optimization for 3D Multi-Objective Molecular Generation

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Optimizing multiple objective properties while satisfying structural constraints is a major challenge in 3D molecular discovery. This difficulty arises because optimization objectives can be non-differentiable and the structure–property relationship is often unknown. Evolutionary algorithms (EAs) are widely used for multi-objective optimization to find Pareto fronts and can naturally handle structural constraints without any explicit modelling; however, in the 3D molecular space they lack mechanisms to guarantee chemical validity and are therefore prone to producing invalid structures. Conversely, diffusion models excel at generating chemically valid 3D molecules but typically require modifying the model and retraining to incorporate structural constraints. Moreover, diffusion models are not inherently designed for direct multi-objective optimization and struggle to explore the Pareto front of the learned property distribution — a critical capability for discovering novel, high-performing molecules. To bridge this gap, we propose a novel 3D molecular multi-objective evolutionary algorithm that leverages the generative power of a pretrained diffusion model. Instead of manipulating molecules directly in the complex chemical space, our method performs crossover operations in the noise space defined by the diffusion model's forward process, thereby enabling parental features or desired fragments to be fused into offspring. The pretrained model's denoising process then restores structural validity. The approach is highly composable and, requiring no retraining, can be readily integrated with existing guidance methods to improve discovery. Experimental results demonstrate strong performance on single-objective, multi-objective, and structurally constrained optimization tasks. Notably, our hybrid method successfully and rapidly explores and captures the Pareto front of the learned property distribution, effectively overcoming a key limitation of using diffusion models alone.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The proposed DEMO combines the strengths of evolutionary algorithms (EA), which are effective for multi-objective optimization, with diffusion models capable of generating valid 3D molecules. It efficiently performs 3D molecular multi-objective optimization without requiring any retraining process.

### Strengths
- DEMO effectively combines diffusion models with EA, successfully addressing the challenging problem of 3D molecular optimization.
- It utilizes a pretrained diffusion model without any retraining, reducing computational cost and performing multi-objective optimization, which has not been actively explored in previous 3D molecular generation studies.

### Weaknesses
- The combination of EA and diffusion models is interesting, but each component is based on well-known techniques, so the overall novelty is limited.
- Although the diffusion model’s forward and denoising processes are well integrated with EA operators, further validation is needed to see whether it works reliably on new objectives or unseen (OOD) samples without retraining.

### Questions
- While DEMO’s no-retraining technique is one of its main strengths, wouldn’t the diffusion model still require some additional training when applied to new objectives or molecular datasets?

- Could the DEMO framework be extended to other diffusion models[1] or flow-matching approaches[2]? Although only two baselines are used in this paper, various 3D molecular generation models[1–4] exist. As mentioned in the paper, TFG may be difficult to apply in this setting, but this also seems to highlight a limitation in the model’s extensibility.

- Could you clarify the difference between SOP-MT and SOP-ST? In addition, what distinguishes SOP-MT from MOP-MO? It seems that SOP-MT involves multiple properties that are scalarized into a single objective function — is this interpretation correct? A more detailed explanation would be helpful.

- In Figure 1, do SOP and MOP respectively denote single-objective and multi-objective optimization? If so, the abbreviations may be a bit confusing. Also, placing this figure in the appendix rather than on the main page might be more appropriate.

- Could the proposed DEMO method be applied to optimization problems involving more than three objectives? Since multi-objective optimization typically considers three or more objectives, such an evaluation would help demonstrate the generality of the model. Additionally, including Pareto front visualizations in the discussion section could further aid in interpreting the results.

- In the MOP setting, would it be possible to compare with other fitness design strategies such as Chebyshev scalarization sampling[5] or Dirichlet-weight sampling[6]?

- Is there a particular reason for conducting the ablation study on Crossover (CO) and Mutation (MT) in Table 2? As these are core components of EA, the experiment feels somewhat unconventional. If the goal is to analyze the effects of crossover and mutation within the diffusion latent space, it might be helpful to compare with a naive EA (without diffusion). (Is “TopN” intended to represent such a baseline?)

[1] Le, Tuan, et al. "Navigating the design space of equivariant diffusion-based generative models for de novo 3d molecule generation." arXiv preprint arXiv:2309.17296 (2023).

[2] Lin, Haowei, et al. "TFG-Flow: Training-free Guidance in Multimodal Generative Flow." arXiv preprint arXiv:2501.14216 (2025).

[3] Irwin, Ross, et al. "Efficient 3d molecular generation with flow matching and scale optimal transport." ICML 2024 AI for Science Workshop. 2024.

[4] Irwin, Ross, et al. "SemlaFlow--Efficient 3D Molecular Generation with Latent Attention and Equivariant Flow Matching." arXiv preprint arXiv:2406.07266 (2024).

[5] Chugh, Tinkle. "Scalarizing functions in Bayesian multiobjective optimization." 2020 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2020.

[6] Shin, Dong-Hee, et al. "Offline Model-based Optimization for Real-World Molecular Discovery." Forty-second International Conference on Machine Learning.

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
This paper presents DEMO (Diffusion-based Evolutionary Molecular Optimization), which is a framework that integrates pretrained 3D diffusion models into evolutionary algorithms for molecular discovery tasks. The method addresses two key issues: (1) evolutionary algorithms in 3D molecular space struggle to generate chemically valid structures due to violation of chemical laws during genetic operations, and (2) diffusion models are inflexible for multi-objective optimization and require expensive retraining when adapting to new properties or constraints. DEMO performs crossover operations in the noise space defined by the diffusion model's forward process by adding Gaussian noise to parent molecules, performing random one-point crossover on the noised representations (both coordinates and features), and then using the pretrained diffusion model's denoising process to restore chemical validity. The noise level is empirically determined through grid search to balance valid crossover against efficient reconstruction. For mutation, the method adds noise to a single parent and denoises it, enabling local exploitation. The framework uses a linear schedule that favors crossover in early generations for global exploration and shifts toward mutation in later generations for local refinement.

### Strengths
1. The combination of evolutionary algorithms with pretrained 3D diffusion models for molecular generation is understandable, as it addresses the validity issues inherent in applying genetic operations directly to 3D molecular coordinates.

2. The framework operates in a training-free manner, which can provide practical utility when working with molecular property evaluators that lack gradient information.

3. I appreciate the empirical analysis of noise level selection (t') across different datasets and models.

### Weaknesses
**1. Limited Technical Novelty**
- While combining evolutionary algorithms with diffusion models is reasonable from an engineering standpoint, the core contribution offers limited algorithmic innovation. The method essentially applies standard genetic operators (crossover and mutation) to noise-space representations and relies entirely on the pretrained diffusion model's denoising capability. 
- In the proposed framework, the crossover operation might be considered merely a linear interpolation in noise space, followed by denoising—a straightforward application of existing techniques rather than a novel methodological advance. 
- The individual components (EAs, diffusion models, noise-space operations) are all well-established, and their integration does not introduce new insights.

**2. Questionable Multi-Objective Claims**
- The authors claim to tackle multi-objective optimization, but their experiments only address two-objective problems throughout the paper. Optimizing two objectives is not substantially more difficult than single-objective optimization and does not truly validate the framework's capability for complex multi-objective scenarios. 
- For genuine multi-objective optimization validation, the method should handle at least 4 objectives where the complexity of the Pareto front increases dramatically and the trade-offs become significantly more challenging. 
-The current two-objective setting raises serious questions about whether this framework is truly beneficial for multi-objective settings or if the claimed advantages would disappear with more realistic problem complexity.

**3. Overstated Pareto Front Discovery Claims**
- In two-objective settings, the Pareto front is relatively simple and easy to approximate. The authors' emphasis on "successfully
and rapidly explores and captures the Pareto front" appears overstated given this limited complexity.
- If the authors want to highlight the method's general performance for 3D molecular generation, they should reduce the emphasis on multi-objective optimization.
- Conversely, if multi-objective optimization is a core contribution, they must demonstrate performance on problems with 4+ objectives where Pareto front approximation is genuinely challenging.

**4. Insufficient Baseline Comparisons**
- The paper lacks comparisons against several relevant baseline approaches. In particular, there appears to be no comparison with state-of-the-art genetic algorithm–based methods with advanced multi-objective optimization techniques, making it unclear how much the diffusion model component actually contributes. If the authors believe they have included sufficient baselines, please clarify.

**5. Unclear Generalization and Applicability**
- The method's performance heavily depends on empirically determined noise levels that vary significantly across datasets. This dataset-dependent hyperparameter tuning requirement raises concerns about the framework's generalizability to new molecular datasets or chemical spaces not covered during training.

- The paper provides no principled method for setting noise level beyond expensive grid search, limiting practical applicability.

**6. Limited Analysis of Component Contributions**
- The ablation studies (w/o CO, w/o MT) show inconsistent results across tasks, with neither crossover nor mutation being uniformly beneficial.  In SOP-MT tasks, removing crossover often improves performance, while in MOP tasks, removing mutation is sometimes better. This inconsistency suggests the framework lacks a coherent design principle and that the components may not synergize as claimed.
- The paper does not adequately explain when and why each operator is beneficial, raising questions about the method's reliability.

### Questions
**1. How are multiple objectives scalarized?**
- For SOP-MT tasks (Table 2), the goal is to find molecules that simultaneously match multiple target property values. Equation 7 shows that deviation is calculated as the Euclidean distance. Does this imply a scalarization scheme that assigns approximately equal importance to all property deviations? If so, how are cases handled where certain properties should be prioritized over others? The authors do not clearly discuss or specify how such weighting would be determined.

- Have the authors considered alternative scalarization methods such as Chebyshev scalarization, which can better handle conflicting objectives and avoid bias toward certain regions of the objective space?

**2. Clarification needed on the theoretical justification in Appendix A.2**
- I am confused about the geometric interpretation of “on-manifold” versus “off-manifold” presented in Appendix A.2. The authors claim that the crossover mean is “off-manifold,” while the mutation mean is “on-manifold.” However, both seem to be scaled versions of points that may or may not lie on the molecular manifold. I may be mistaken, but I find this interpretation unclear and would appreciate further clarification.

- If the molecular manifold is defined as the set of chemically valid molecules, then scaling a valid molecule would also produce an "off-manifold" point?

- The theoretical framework relies on the assumption of "Local Fitness Landscape Continuity" (Assumption 1). How do the authors know this assumption holds for molecular fitness landscapes, which can be highly rugged and discontinuous?

**3. Does the "no retraining" advantage truly hold for new objectives and datasets?**
- The authors emphasize that DEMO requires no retraining as a key advantage over existing methods.  However, when applying this framework to entirely new molecular objectives or new chemical datasets,  wouldn't the pretrained diffusion model also require retraining or fine-tuning to properly capture the distribution of valid structures in these new domains?

- The current evaluation uses diffusion models pretrained on QM9 and GEOM-Drugs and tests on the same datasets. This does not validate the "no retraining" claim for genuinely out-of-distribution scenarios. Can the authors demonstrate that their framework works with a diffusion model pretrained on Dataset A but applied to optimize molecules from a completely different Dataset B without any retraining?

**4. Can this approach be applied to 2D molecular generation/optimization?**
- The authors emphasize 3D molecular generation throughout the paper. However, I wonder whether the core idea of performing genetic operations in diffusion noise space could also be applied to 2D molecular generation. There has been substantial research on multi-objective 2D molecular generation and optimization, including benchmarks such as PMO (Practical Molecular Optimization) introduced by Gao et al. at NeurIPS 2022. Considering this, exploring 2D molecular generation would be an important and natural extension. Have the authors considered this direction?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work has proposed a 3D single and multi-objective molecular evolutionary algorithm that leverages a pretrained diffusion model. DEMO integrates diffusion models into an evolutionary loop to improve validity and accelerate search. This also combined 3D diffusion backbone (EDM, …) with classical operators executed in noise space. The framework is also training-free and demonstrates improvements across multiple tasks.

### Strengths
To my knowledge this is the first framework that combines evolutionary algorithms with 3D diffusion models, which is novel and interesting.
The problem formulation is clear and grounded in the goals of multi objective molecular design.
The effectiveness of the framework is shown across four tasks using the same pretrained model

### Weaknesses
More detailed explanations of the reported metrics in the tables, including Table 1 and 2, are needed.

The approach is interesting and reaches state of the art on some metrics, yet results are not fully consistent, as different diffusion backbones share or swap the best scores across metrics.

In the Tasks and Datasets section, please specify the exact objectives for SOP and MOP, for example which properties are optimized and how they are computed.

Performance of crossover and mutation depends strongly on the noise level t prime, chosen by empirical tuning. The authors mentioned that this as balancing manifold linearity for valid crossover against information retention for reconstruction. This task dependent tuning slightly weakens the plug and play story.

### Questions
The authors mentioned that the 3D crossover is a simple one-point split on the atom sequence, leading to a “random partitioning”. Given the arbitrary nature of the atom list, how can this unguided fragmentation effectively combine meaningful molecular fragments to do chemically valid exploration that is superior to a well-designed chemically-aware crossover operator?

The noise level t' is reduced linearly across generations (Algorithm 1, line 4). Did the authors explore a dynamic, fitness-dependent t' scheduling, where the noise level is adapted based on the current population's fitness, rather than a fixed linear decrease?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes to replace the mutation and crossover operations in evolutionary algorithms with diffusion-based denoising, enabling property-guided molecular generation. The method is evaluated on small-molecule datasets, such as QM9 and GEOM-DRUG datasets, and on a protein–ligand optimization dataset where Vina scores are used as the oracle.

### Strengths
Overall, the manuscript is well-structured and reads smoothly, making it relatively easy to follow the proposed methodology.

### Weaknesses
## Experimental Design
The paper attempts to optimize certain QM9 properties as the objective, but the motivation for those is unclear. QM9 is originally designed as a regression benchmark, and I do not see a strong justification for treating these properties as optimization targets. It is also unclear how the reference scores or target ranges for these QM9 objectives were determined.

In contrast, the Vina score–based optimization task appears to be much more meaningful and realistic for molecular design. In fact, I would argue that this should have been the main experimental setting, rather than QM9.

## Multi-objective Setting
For multi-objective optimization, the more standard and practically relevant setup involves objectives such as Vina score + QED + SA, as used in recent works such as Saturn [1] and fRAG [2]. I believe adopting such a setting would provide a stronger and more convincing evaluation of the proposed method.

## Number Objectives
The paper considers the optimization of only two objectives for the multi-objective experiments, yet refers to this as the Pareto front and multi-objective optimization, and this setup feels somewhat limited. Recent studies typically explore 4–6 objectives simultaneously to better reflect the complexity of real-world molecular design [3]. Extending the evaluation to higher-dimensional objective spaces would make the results more compelling.


[1] Saturn: Sample-efficient Generative Molecular Design using Memory Manipulation (Arxiv 2024) 

[2] Drug Discovery with Dynamic Goal-aware Fragments (ICML 2024)

[3] Efficient Evolutionary Search Over Chemical Space with Large Language Models (ICLR 2025)

### Questions
A recent ICML 2025 paper [4] pretrained neural network-based operators for crossover, which seems conceptually related to DEMO. However, that work explicitly pretrains the neural network to learn the crossover operator, while it is unclear whether DEMO applies any GA-specific training to align the diffusion model with the evolutionary algorithm. It would be helpful if the authors could clarify how the diffusion model was trained or adapted to effectively serve as a mutation/crossover operator within the EA framework.

[4] Offline Model-based Optimization for Real-World Molecular Discovery (ICML 2025)

### Soundness
3

### Presentation
2

### Contribution
2
