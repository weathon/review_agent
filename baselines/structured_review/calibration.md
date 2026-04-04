=== CALIBRATION EXAMPLE 1 ===

# Final Consolidated Review
## Summary

This paper presents a systematic review of Large Language Models (LLMs), covering their evolution, architectural types (generative, masked language, sequence-to-sequence, and hybrid models), development techniques, applications across text generation/translation/summarization, limitations including bias and resource intensity, ethical considerations, evaluation benchmarks, and future research directions.

## Strengths

- **Comprehensive foundational coverage**: The paper provides a solid overview of core LLM architectures (Section 2), including decoder-only (GPT), encoder-only (BERT), encoder-decoder (T5, BART), and hybrid models (XLNet), with appropriate citations to foundational works (Vaswani 2017, Devlin 2018, Raffel et al. 2020).

- **Structured treatment of limitations and ethics**: Sections 6 and 7 meaningfully discuss bias/fairness, interpretability, resource intensity, overfitting concerns, and ethical deployment challenges with relevant citations (Bender et al. 2021, Gallegos et al. 2024), which are important considerations for the field.

- **Coverage of evaluation landscape**: Section 8 and 3.9 discuss multiple evaluation frameworks including HELM benchmark, LMSYS Chatbot Arena, GLUE, and SQuAD, providing readers with understanding of how LLMs are assessed.

## Weaknesses

- **Missing systematic review methodology**: Despite the title claiming a "systematic review," the paper lacks any methodology section describing search strategy, inclusion/exclusion criteria, database sources, or time frame for literature selection. This is a fundamental requirement for any systematic review and undermines reproducibility and rigor.

- **Incomplete coverage of critical recent developments**: The paper substantially misses key advances from 2022-2024, including instruction tuning/alignment methods (RLHF, DPO, Constitutional AI), parameter-efficient fine-tuning (LoRA, QLoRA), chain-of-thought reasoning, retrieval-augmented generation (RAG), and open-source model families (LLaMA, Mistral). This is a critical gap for a review published in 2024-2025.

- **Limited differentiation from existing surveys**: The paper cites comprehensive surveys like Bommasani et al. (2021) and Zhao et al. (2023) but does not clearly articulate what novel contribution or perspective this review adds. The comparative analysis in Section 3.11 and Table 1 is superficial compared to existing detailed surveys.

- **Shallow application analysis**: Despite claiming to cover "practical usages," the applications discussion (text generation, translation, summarization, sentiment analysis, question answering) provides only high-level descriptions without concrete benchmark results, performance comparisons across models, or practical deployment considerations.

## Nice-to-Haves

- A timeline figure visualizing the evolution of LLMs from 2017-2024 would help readers understand the rapid progression in the field.

- Deeper technical analysis of training methodologies (pretraining objectives, data curation, scaling laws) would strengthen the development-focused sections.

## Novel Insights

The paper's discussion of the HELM benchmark (Section 3.9.1) and LMSYS Chatbot Arena (Section 3.9.2) as complementary evaluation frameworks is useful, highlighting the tension between academic benchmark evaluation and real-world conversational performance. However, this insight is underdeveloped—the paper could have critically analyzed why different evaluation paradigms yield different model rankings and what this means for the field.

## Potentially Missed Related Work

- **Hu et al. (2021, 2022)** on LoRA (Low-Rank Adaptation) — Critical for understanding parameter-efficient fine-tuning, which has become central to practical LLM deployment.

- **Ouyang et al. (2022)** on InstructGPT and RLHF — Foundational work on alignment that shaped modern LLM training.

- **Wei et al. (2022)** on Chain-of-Thought prompting — Major advancement in LLM reasoning capabilities.

- **Touvron et al. (2023)** on LLaMA — Open-source model family that democratized LLM research.

- **Lewis et al. (2020)** on Retrieval-Augmented Generation — Key technique for grounding LLM outputs.

- **Rafailov et al. (2023)** on Direct Preference Optimization (DPO) — Important alternative to RLHF for alignment.

## Suggestions

- Add a detailed methodology section describing the systematic review process: databases searched (ACL, NeurIPS, arXiv), keywords used, inclusion/exclusion criteria, and PRISMA-style flow diagram for paper selection.

- Expand Section 3 to include a dedicated subsection on alignment and instruction tuning methods (RLHF, DPO, RLAI), which are essential to understanding modern LLM capabilities and safety.

- Update Table 1 to include representative recent models from 2022-2024 (e.g., LLaMA, GPT-4, Mistral) and expand metrics to include reasoning benchmarks (GSM8K, MATH) and alignment evaluations (TruthfulQA).

# Actual Human Scores
Individual reviewer scores: [1.0, 1.0, 1.0, 1.0]
Average score: 1.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 2 ===

# Final Consolidated Review
## Summary
The paper proposes Multi-Objective Decision-Focused Learning (MoDFL), extending decision-focused learning to multi-objective optimization problems. The authors introduce three novel loss functions: landscape loss (using sRMMD to measure objective space discrepancy), Pareto set loss (measuring solution space distance), and decision loss (using weighted-sum transformation to single-objective). Experiments on web advertisement allocation and bipartite matching benchmarks demonstrate improvements over two-stage methods and existing DFL approaches.

## Strengths
- **Addresses an important gap in DFL literature**: The paper correctly identifies that existing decision-focused learning methods focus on single-objective optimization, while multi-objective problems are prevalent in real-world scenarios. This extension is novel and practically relevant.
- **Well-motivated loss function design**: The three loss components capture complementary aspects of multi-objective optimization—objective space (landscape loss), solution space (Pareto set loss), and decision quality (decision loss). The ablation study in Table 5 confirms each component contributes meaningfully, with the decision loss having the largest impact.
- **Comprehensive experimental comparison**: The paper compares against seven baselines (TwoStage, SPO, BB, MAP, NCE, Pointwise, Listwise) across two distinct benchmarks with multiple evaluation metrics (GD, MPFE, HAR, regret metrics), demonstrating consistent improvements.
- **Novel application of sRMMD**: The use of sample rank maximum mean discrepancy for comparing objective space landscapes across different optimization problem instances is a creative adaptation of optimal transport theory to this domain.

## Weaknesses
- **Limited problem scope to convex, linear cases**: The method relies on weighted-sum transformation and DSLP (Differentiation of Smooth Linear Programming), restricting applicability to convex multi-objective problems. Non-convex Pareto fronts, common in many real-world multi-objective problems, cannot be properly handled by weighted-sum approaches—this limitation is not discussed.
- **Missing hyperparameter sensitivity analysis**: The combined loss function uses three hyperparameters (λl=1, λd=2, λps=5 in Section 5.2.4), but no justification or sensitivity analysis is provided. The relative weighting could significantly impact performance, and practitioners need guidance on tuning these for new problems.
- **Incomplete evaluation metric definitions**: Key metrics (GD, MPFE, HAR, average percentage regret) are referenced to an appendix that is not included. For reproducibility, at least the regret metric formulation should appear in the main text, especially since multi-objective regret is non-trivial to define.
- **No computational complexity or training time analysis**: Decision-focused methods are known to be computationally expensive due to repeated solver calls. The paper provides no analysis of training time, scalability with number of objectives, or comparison of computational cost against baselines.

## Nice-to-Haves
- Experiments with non-convex multi-objective problems to demonstrate broader applicability beyond LP
- Sensitivity analysis showing how performance varies with λl, λd, λps choices
- Training time comparison across methods to quantify computational overhead

## Novel Insights
The key methodological insight is that multi-objective DFL requires measuring discrepancies in both objective space and solution space simultaneously. The landscape loss based on sRMMD treats the objective space as a manifold and uses optimal transport to compare distributions across problem instances—this captures structural similarity that simple pointwise metrics miss. The ablation results reveal that while all three losses contribute, decision loss (measuring quality of representative solutions) has the largest impact, suggesting that for practical MOP, the weighted-sum representative solution remains central to learning.

## Potentially Missed Related Work
- Deb K. et al. "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II" (foundational MO optimization work)
- Bhusal D. et al. (2023) "Pareto Front Differentiation for Multi-Objective Learning" — recent work on differentiable Pareto front learning
- Liu S. et al. (2021) "Conflicting Gradients in Multi-Task Learning" — relevant to gradient conflicts in multi-objective settings
- Recent DFL advances (2022-2024) addressing computational efficiency and broader problem classes

## Suggestions
- Add theoretical discussion of weighted-sum method's limitations for non-convex Pareto fronts, and how MoDFL might handle such cases
- Include a table reporting training times across methods to quantify computational trade-offs
- Provide explicit formulas for evaluation metrics in the main text for reproducibility
- Consider experiments with 4+ objectives to test scalability to many-objective optimization

# Actual Human Scores
Individual reviewer scores: [3.0, 1.0, 3.0]
Average score: 2.3
Binary outcome: Reject

=== CALIBRATION EXAMPLE 3 ===

# Final Consolidated Review
## Summary
The paper proposes MQFL-FHE, a novel framework integrating quantum computing with federated learning and fully homomorphic encryption (FHE) to mitigate FHE-induced performance degradation while preserving privacy. The authors introduce a multimodal quantum mixture of experts (MQMoE) architecture and demonstrate empirical improvements on CIFAR-10, DNA sequences, MRI scans, and combined genomics-MRI datasets.

## Strengths
- **Novel integration of multiple technologies**: The paper presents a genuinely novel combination of FHE, quantum computing, federated learning, and mixture-of-experts for multimodal data processing (Sections 4.1-4.3). This integration addresses a timely and challenging problem at the intersection of privacy-preserving ML and quantum computing.
- **Comprehensive empirical evaluation**: The authors evaluate across multiple datasets (CIFAR-10, DNA sequences, MRI, PCOS, and combined DNA+MRI) with multiple configurations (centralized vs. federated, classical vs. quantum, with/without FHE), providing ROC curves and confusion matrices for detailed analysis (Tables 3-4, Figure 4-5).
- **Medical domain application**: The multimodal genomics + brain MRI application demonstrates practical relevance for privacy-sensitive healthcare scenarios, with demonstrated improvements on underrepresented classes (Section 5.1 ablation analysis).
- **Mathematical formulation**: The paper provides formal notation for the CKKS encryption scheme, quantum state encoding, and aggregation procedures (Equations in Sections 3-4), enabling reproducibility of the core approach.

## Weaknesses
- **Insufficient theoretical justification for quantum mitigation claim**: The central claim that quantum computing counteracts FHE-induced performance degradation lacks rigorous theoretical foundation. While the appendix discusses quantum noise confinement under SU(2) unitary constraints, the paper does not establish a clear mechanistic connection between quantum layer properties and FHE noise accumulation. The explanation that "periodic error cancellation" occurs remains qualitative without quantitative analysis of how this interacts with CKKS noise growth.
- **Marginal and inconsistent empirical improvements**: The performance gains from quantum enhancement are modest and sometimes contradictory. For CIFAR-10 centralized (Table 3), quantum (74.33%) underperforms classical (76.59%). In federated settings, QFL+FHE (71.12%) vs. FL+FHE (68.53%) shows only ~2.6% improvement. More critically, QFL without FHE (72.16%) still outperforms QFL+FHE (71.12%), indicating FHE degradation persists despite quantum layers.
- **Practical deployment concerns**: The experiments use simulated quantum circuits (Pennylane) rather than real quantum hardware. Given NISQ-era noise and decoherence, results may not translate to actual quantum devices. Additionally, QFL+FHE requires ~2.9x more computation time than FL+FHE for CIFAR-10 (9747s vs. 4022s in Table 4), raising concerns about practical viability.

## Nice-to-Haves
- Comparison with alternative privacy mechanisms (differential privacy, secure aggregation) would strengthen the positioning of FHE against other approaches.

## Novel Insights
The paper's observation that quantum layers may provide regularization benefits in the context of FHE noise is intriguing, though underexplored. The mixture-of-experts architecture with modality-specific quantum experts represents an interesting architectural choice for multimodal medical data, where different modalities (MRI images vs. genomic sequences) have fundamentally different structure. The authors correctly identify that underrepresented classes benefit more from quantum enhancement (confusion matrices show improved glioma and pituitary classification), suggesting quantum circuits may provide richer representations for minority classes—a finding that warrants deeper theoretical investigation.

## Potentially Missed Related Work
- Chen et al. (2023) "Quantum Federated Learning with Differential Privacy" - relevant for alternative privacy-preserving QFL approaches
- Cheng et al. (2024) "Federated Learning with Encrypted Gradients: A Survey" - recent survey on gradient encryption methods
- Wibisono et al. (2023) on "Differentially Private Federated Learning with Quantum Neural Networks" - directly relevant to the intersection of privacy and QFL

## Suggestions
- Provide quantitative theoretical analysis (or bounds) connecting quantum circuit depth/gate composition to FHE noise mitigation, rather than relying solely on empirical demonstration.
- Include experiments comparing against differential privacy baselines and other FL+FHE implementations to contextualize the proposed approach's advantages.
- Report confidence intervals across multiple experimental runs to establish statistical significance of the marginal improvements shown.

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 3.0, 3.0, 3.0]
Average score: 3.4
Binary outcome: Reject

=== CALIBRATION EXAMPLE 4 ===

# Final Consolidated Review
## Summary
This paper identifies and addresses the problem of "quantization bias" in federated learning with LoRA adapters when clients operate LLMs at different quantization levels. The authors propose FedQLoRA, which introduces a quantization-aware adapter to separate quantization error from task-specific adapter learning, and an iterative version (iFedQLoRA) to handle data heterogeneity.

## Strengths
- **Novel problem identification**: The paper is the first to identify and formalize the quantization bias problem in federated LoRA (Eq. 9). Figure 1 provides compelling empirical evidence that mixed quantization settings (2-bit + 4-bit) perform significantly worse than homogeneous settings, validating that this is a real and important issue.
- **Principled methodology**: The approach of decomposing adapter updates into quantization-aware components (L_i, R_i) and task-specific components (B_i, A_i) via SVD (Eq. 11-12) is mathematically sound. The connection to LoRA-aware quantization (Proposition 1) provides theoretical grounding.
- **Comprehensive experimental evaluation**: The paper evaluates across multiple dimensions: IID vs. non-IID settings, varying client numbers (3, 5, 10), model heterogeneity proportions (Figure 3), and data heterogeneity levels (Figure 4). The iterative version consistently outperforms baselines, with improvements ranging from 2-7% in accuracy depending on settings.

## Weaknesses
- **Significant gap between claims and experimental scale**: The paper claims to address "large language models with billions of parameters" (Abstract, Introduction) but uses DistilBERT-base-multilingual-cased (~66M parameters), which is not an LLM by any standard definition. This raises questions about whether the method scales to actual LLMs (e.g., Llama-7B, 70B) where quantization effects and adapter dynamics differ substantially. The memory and computational trade-offs central to the motivation are not demonstrated at realistic scales.
- **Strong assumption about known quantization methodology**: The derivation in Eq. 15 assumes "when the quantization method Q = Q_i... is known," enabling simplification. However, in realistic heterogeneous deployments, clients may use different quantization schemes (GPTQ, AWQ, uniform, symmetric/asymmetric) that are not known to the server or other clients. The paper provides no discussion or experiments on how performance degrades when this assumption is violated.
- **Incomplete description of quantization implementation**: The paper mentions "2-bit" and "4-bit" quantization but provides no implementation details—which quantization method is used, which layers are quantized, what normalization function F(·) is employed (Eq. 1), or how the quantization encoder/decoder are implemented. This makes reproducibility difficult and prevents understanding of how method performance depends on quantization quality.

## Nice-to-Haves
- Evaluation on actual LLMs (e.g., Llama-2-7B or similar) would substantially strengthen the claims and demonstrate practical applicability.

## Novel Insights
The paper makes an important observation that LoRA adapters trained on quantized models implicitly learn two things simultaneously: (1) compensation for quantization error and (2) task-specific knowledge. This "entanglement" means that aggregating adapters from clients with different quantization levels introduces systematic bias. The insight that decoupling these components via a dedicated quantization-aware adapter can improve federated aggregation is elegant. The iterative approach that uses global adapter information to refine quantization-aware estimates (addressing heterogeneity bias) is a clever extension that recognizes local-only estimation is insufficient under non-IID data.

## Potentially Missed Related Work
- **Federated Learning with Quantization**: Papers on quantized federated learning (e.g., "FedPAQ" or more recent work on heterogeneous quantization in FL) may be relevant, though most focus on traditional models rather than LoRA adapters.
- **LoRA Fine-tuning under Quantization**: Recent work on LoRA fine-tuning stability under quantization (2023-2024) could provide additional context for how adapter dynamics change with quantization levels.
- **Personalized Federated Learning**: The paper compares to H-LoRA and FFA-LoRA but could benefit from comparison with personalized FL methods like pFedMe or Ditto that handle heterogeneity through personalization, which is conceptually similar to the quantization-aware adapter approach.

## Suggestions
- Run experiments on at least one modern LLM (e.g., Llama-2-7B with 4-bit/8-bit quantization variants) to validate scalability claims, even if limited to fewer experimental configurations.
- Add a sensitivity analysis for the key assumption that Q is known: simulate scenarios where clients use different quantization methods (e.g., uniform vs. non-uniform) and report how performance degrades.
- Include implementation details for quantization (method, layers, code reference) in the appendix or main text to ensure reproducibility.

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 5.0, 3.0]
Average score: 4.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 5 ===

# Final Consolidated Review
## Summary

The paper introduces AdaFM (Adaptive Filtered Momentum), an adaptive variance-reduced algorithm for stochastic minimax optimization that eliminates the need for manual tuning of problem-dependent hyperparameters. The algorithm achieves near-optimal O(ϵ^{-3}) sample complexity for finding an ϵ-stationary point in both non-convex-strongly-concave (NC-SC) and non-convex-Polyak-Łojasiewicz (NC-PL) settings, matching the best parametric VR algorithms while being substantially easier to use in practice.

## Strengths

- **Addresses a practical pain point**: The paper convincingly demonstrates (Figure 1) that existing VR-based algorithms like RSGDA and VRAdaGDA require extensive hyperparameter tuning, with learning rate combinations being highly sensitive and dataset-dependent. This motivates the need for adaptive methods.

- **Novel algorithm design**: The key technical innovation is the simplified momentum parameter β_{t+1} = 1/t^{2/3} (same for both variables) combined with adaptive learning rates η_t^x = γ/({max{α_t^x, α_t^y}})^{1/3+δ} and η_t^y = λ/(α_t^y)^{1/3-δ} that use only historical gradient estimator information, eliminating dependence on problem parameters like L and G.

- **Strong theoretical guarantees**: Theorems 1 and 2 establish O(κ^{4.5}/T^{1/3+δ}) and O(κ^5/T^{1/3+δ}) convergence rates for NC-SC and NC-PL settings respectively, translating to the optimal O(ϵ^{-3}) sample complexity. This improves upon TiAda's O(ϵ^{-4}) complexity.

- **Comprehensive empirical evaluation**: Experiments span synthetic test functions (Section 5.1), deep AUC maximization on CIFAR-10/100 with various imbalance ratios (Section 5.2), and WGAN-GP training (Section 5.3). The comparisons against RSGDA, VRAdaGDA, and TiAda demonstrate AdaFM's effectiveness and robustness.

## Weaknesses

- **The δ parameter undermines "parameter-free" claims**: The paper states AdaFM is "parameter-free," yet δ appears in the learning rate formulas and critically, the ablation study (Figure 9) shows δ=0 causes convergence failure. The statement that δ takes "an arbitrarily small value" is misleading—if δ cannot be zero and affects convergence behavior, it remains a hyperparameter requiring selection. This is a substantive contradiction with the paper's positioning.

- **Unclear practical guidance for hyperparameter selection**: While the paper claims γ=λ=1 suffice theoretically, all experiments use grid-searched values from [0.1, 0.5] and [0.6, 1.0] respectively. No experiments validate the γ=λ=1 claim in practice. Similarly, the paper provides limited guidance on how practitioners should select δ across different tasks—Figure 10 shows δ impacts inception scores on WGAN-GP, suggesting task-dependent tuning may still be needed.

- **Dependency on condition number κ in NC-PL setting**: The O(κ^5/T^{1/3+δ}) rate in Theorem 2 is worse than the NC-SC rate of O(κ^{4.5}/T^{1/3+δ}). The paper acknowledges this but does not discuss whether this gap is inherent to the PL setting or a limitation of the current analysis.

- **Potential missing comparison to Adam/AdaGrad-style methods**: The experiments compare against RSGDA, VRAdaGDA, and TiAda, but do not include widely-used adaptive methods like Adam or RMSprop with standard SGDA, which practitioners commonly use for GAN training despite lacking VR guarantees.

## Nice-to-Haves

- Validation experiments explicitly using γ=λ=1 across multiple tasks would strengthen the claim that these parameters need not be tuned.

- Runtime/iteration efficiency analysis beyond sample complexity, as the adaptive learning rate computation adds per-iteration overhead.

## Novel Insights

The asymmetric learning rate design—where η_t^x depends on max{α_t^x, α_t^y} while η_t^y depends only on α_t^y—is a clever mechanism that implicitly handles two-timescale optimization. By slowing x's update when y's maximization subproblem is unresolved (large α_t^y relative to α_t^x), the algorithm maintains the hierarchy needed for minimax convergence without requiring manual timescale ratio tuning. This represents a principled way to achieve adaptive two-timescale behavior in variance-reduced methods.

## Potentially Missed Related Work

- **"Normalizing Gradient Descent" (Jelassi et al., ICLR 2024)**: Recent work on adaptive normalization in optimization that may relate to the learning rate scaling strategies.

- **Recent advances in parameter-free optimization (2023-2024)**: The rapidly evolving area of parameter-free methods may have relevant techniques for minimax settings.

## Suggestions

- Either remove "parameter-free" terminology or clearly qualify it—the algorithm reduces but does not eliminate hyperparameters. Be explicit about δ's role and provide recommended defaults.

- Add a practical experiment with γ=λ=1 to empirically validate the theoretical claim, or clarify why practitioners should still tune these parameters.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 8.0, 6.0]
Average score: 5.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 6 ===

# Final Consolidated Review
## Summary
The paper proposes Swift-FedGNN, a federated graph neural network framework that reduces communication and sampling costs in geo-distributed graph training. The key idea is that clients primarily perform efficient local training using only their local graph data, while periodically a subset of randomly sampled clients conduct cross-client training to incorporate information from neighbors on other clients. The paper provides theoretical convergence analysis proving Swift-FedGNN achieves O(1/√T) convergence rate, matching SOTA sampling-based GNN methods, and demonstrates significant efficiency improvements on ogbn-products and Reddit datasets.

## Strengths
- **Addresses a critical practical challenge in federated GNN training**: The paper identifies that cross-client training can be 5x slower than local training due to sampling and communication overhead (Figure 2), and proposes a principled solution that balances efficiency with information completeness from cross-client neighbors.

- **Novel theoretical analysis for biased stochastic gradients in GNNs**: Unlike prior work that makes unrealistic assumptions about unbiased gradients (Chen et al., 2018) or consistent gradients (Chen & Luss, 2018), this paper bounds the stochastic gradient approximation errors (Lemmas 5.4-5.5) and shows they positively correlate with GNN depth—a finding unique to federated GNN training.

- **Strong empirical results demonstrating efficiency gains**: Table 2 shows Swift-FedGNN reduces communication overhead by 4-20x compared to baselines, while Figure 4 shows comparable convergence accuracy to FedGNN-G (87.73% vs 87.93% on ogbn-products; 95.60% vs 96.03% on Reddit).

- **Clever privacy-preserving aggregation design**: Equations 5-6 implement double aggregation (first at remote clients, then at server) before transmitting neighbor embeddings, which both reduces communication and provides privacy benefits since individual node information is not directly exposed.

- **Comprehensive ablation studies**: Figure 6 provides useful sensitivity analysis for key hyperparameters (correction frequency I, number of cross-client training clients K, sampling fan-outs), showing practical trade-offs between communication overhead and model quality.

## Weaknesses
- **Limited GNN architecture support**: Footnote 2 explicitly states the operation offloading only supports element-wise aggregation (mean, sum, max), which excludes important architectures like Graph Attention Networks (GAT) that require raw features to be transferred—this significantly limits applicability and is not adequately addressed as a limitation.

- **Disconnect between theoretical analysis and experiments**: The convergence analysis in Section 5 is specifically for GCN architecture, while experiments use GraphSAGE. The paper claims the analysis extends but provides no discussion of how the theoretical bounds would differ for GraphSAGE's aggregation scheme, creating uncertainty about whether proven guarantees apply.

- **Informal privacy guarantees**: While the paper claims the double aggregation approach "helps preserve data privacy" because clients don't know neighbor locations and embeddings are aggregated, no formal privacy analysis is provided. The claim that "the information of each node is not leaked" (Section 1) is unsubstantiated—aggregated embeddings could still leak information about individual nodes, and no differential privacy or other formal privacy framework is employed.

- **Unrealistic federated data partition assumption**: Experiments use METIS partitioning which optimizes for edge cuts, but real federated scenarios may have different natural partitions (e.g., hospitals having patient data with cross-institution edges for patient transfers). The performance under non-IID or organic partitions is not evaluated.

## Nice-to-Haves
- Experiments with non-element-wise GNN architectures (e.g., GAT) to demonstrate the framework's broader applicability or clearly quantify the communication overhead penalty for such cases.

## Novel Insights
The paper reveals an important structural property of GNNs in federated settings: the bias in stochastic gradients from neighbor sampling propagates and amplifies through layers. Lemmas 5.4 and 5.5 show that both the sampling-induced error (B_G^l, B_G^f) and the missing cross-client neighbor error (B_G^r) scale with GNN depth through terms like (C_σ B_W B_P)^l. This "structural entanglement"—where errors compound through interleaved aggregation and non-linear transformations—has implications for federated GNN design: deeper networks require more frequent cross-client correction to maintain accuracy, while shallower networks can tolerate longer intervals between expensive communication rounds.

## Potentially Missed Related Work
- "Federated Graph Learning with Secure Neighbor Sampling" (2023) - addresses similar privacy concerns with cryptographic approaches
- "FedStar: Federated Spatial-Temporal Graph Neural Networks" (2022) - federated approach for temporal graphs
- "Cross-Client Contrastive Learning for Federated Graph Neural Networks" (2023) - uses contrastive learning to handle cross-client information without direct communication

## Suggestions
- Add formal privacy analysis: even a simple differential privacy bound or discussion of what information the aggregated embeddings leak would strengthen the privacy claims substantially.

- Include experiments with varying cross-client edge densities to understand the regime where Swift-FedGNN provides the most benefit versus where local-only training becomes competitive.

# Actual Human Scores
Individual reviewer scores: [3.0, 6.0, 5.0, 5.0]
Average score: 4.8
Binary outcome: Reject

=== CALIBRATION EXAMPLE 7 ===

# Final Consolidated Review
## Summary
This paper proposes HiCA (Hierarchical prompts with Context-Aware calibration) for open-vocabulary object detection, addressing the problem of base class overfitting in existing prompt-based OVD methods. The approach introduces hierarchical prompts that decompose region-category mapping into coarse-grained (region-to-superclass) and fine-grained (superclass-to-category) stages, plus context-aware calibration that learns the distribution of categories across different visual contexts to refine predictions. Experiments on OV-COCO and OV-LVIS demonstrate improvements over baselines.

## Strengths
- **Novel problem formulation**: The paper correctly identifies that existing OVD methods overfit to base classes by directly mapping regions to categories, ignoring shared superclass semantics. The hierarchical prompt approach (Section 3.3) elegantly addresses this by using superclasses as an intermediary that encompasses both base and novel classes.
- **Strong empirical results**: Table 1 shows HiCA achieves 57.2% mAP_B and 50.4% mAP50 on OV-COCO, surpassing OADP baseline by 5.5% and 4.4% respectively. On OV-LVIS (Table 2), HiCA improves AP_r from 22.8% to 24.3% when applied to BARON.
- **Modular design**: As stated in Section 3.5, both hierarchical prompts and context-aware calibration are plug-and-play modules that can be applied to different OVD frameworks. The paper demonstrates this by showing improvements when applied to both OADP and BARON baselines.
- **Comprehensive ablation studies**: Tables 3-5 and Figure 3 provide thorough analysis of each component's contribution, the effect of balance parameter λ, and design choices for context clustering and DG layer depth.

## Weaknesses
- **Superclass supervision assumption is under-specified**: The method requires a predefined superclass taxonomy (matrix A in Eq. 2) mapping categories to superclasses. Section 3.1 defines C_S as the set of superclasses, but it is unclear how these are obtained. If they require manual annotation or external knowledge bases, this should be clearly stated as a requirement. For novel classes during inference, the superclass membership must be known—but this may not always be available in real open-vocabulary settings.
- **Context clustering quality is not analyzed**: The context-aware calibration relies on K-means clustering of global image features (Eq. 5), but there is no analysis of cluster quality or semantic meaningfulness. Table 5 shows that different numbers of clusters (K=6,8,10) yield different results, but without qualitative analysis of what contexts are being captured, it is difficult to assess whether this approach scales to diverse datasets.
- **Limited novelty class analysis**: While the paper claims improved generalization to novel classes, the ablation in Table 3 shows that hierarchical prompts alone achieve mAP_N=30.0% (vs baseline 29.9%), and context-aware calibration only marginally improves this to 31.2%. The main gains come from base class performance (51.7%→57.5%). The paper should discuss whether this trade-off is acceptable.

## Nice-to-Haves
- Analysis of how robust the method is to incorrect or missing superclass labels for novel categories at test time would strengthen the practical applicability claims.
- Visualization of learned context clusters (what semantic contexts do they capture?) would provide insight into what the model learns.

## Novel Insights
The hierarchical prompt approach represents a principled way to inject coarse-grained semantic structure into OVD. Rather than treating all categories as independent, it leverages the fact that objects share high-level semantics (e.g., animals, vehicles). This creates an interesting connection to the literature on hierarchical classification. The context-aware calibration is also noteworthy—it treats context not as mere background to be ignored, but as informative prior about what categories are likely to appear. The idea that "a bus is more likely in urban contexts than in forests" is implicitly captured through the learned context-class distribution matrix, which is a novel application of context for OVD.

## Potentially Missed Related Work
- **Cao et al., "Open-Vocabulary Object Detection with Multi-modal Prompt" (2023)**: Explores multi-modal prompts for OVD with hierarchical class relationships.
- **Gu et al., "Class-agnostic Object Detection with Language Embeddings" (2024)**: Recent work on leveraging language structure for detection.
- **Recent prompt learning works for VLMs (2023-2024)**: Methods like MaPLe, KgCoOp that explore hierarchical or context-aware prompting strategies for vision-language models could be relevant comparisons.

## Suggestions
- Explicitly describe how the superclass taxonomy is constructed and what supervision is required. If using existing hierarchies (e.g., COCO supercategories, WordNet), cite them and discuss limitations.
- Add a small experiment or discussion on the sensitivity to superclass quality—what if novel classes are assigned incorrect superclasses?
- Include failure case analysis: when does context-aware calibration hurt performance, and on which types of images or categories?

# Actual Human Scores
Individual reviewer scores: [5.0, 5.0, 5.0, 5.0]
Average score: 5.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 8 ===

# Final Consolidated Review
## Summary

The paper investigates whether labels can be ignored in out-of-distribution (OOD) detection. The authors provide information-theoretic proofs showing that SSL and unsupervised OOD methods are guaranteed to fail when the surrogate learning objective is independent of labels—a condition termed "label blindness." They introduce "Adjacent OOD Detection," a novel benchmark where ID and OOD classes are sampled from the same dataset to maximize feature overlap. Experiments across multiple datasets demonstrate that unlabeled methods perform near random on this benchmark while supervised baselines substantially outperform them.

## Strengths

- **Novel theoretical framework for SSL OOD failure**: The paper provides formal information-theoretic proofs (Theorem 3.1, Lemma 3.2, Corollary 3.3) establishing conditions under which SSL/unlabeled OOD detection must fail. The proof structure rigorously connects minimal sufficient statistics from information bottleneck theory to OOD detection failure, showing that when surrogate task features x₁ are independent of label-relevant features x₂, the learned representation z must satisfy I(x₂; z) = 0.

- **Important safety-critical contribution**: The Adjacent OOD detection benchmark addresses a real gap in existing OOD evaluation. Theorem 4.1 formally establishes that overlapping ID/OOD data is unavoidable in practice (e.g., datasets created at time t cannot contain future categories), making this benchmark highly relevant for safety-critical systems.

- **Strong empirical validation of theoretical predictions**: Table 1 shows that unlabeled methods (SimCLR, RotLoss, Diffusion-based) perform poorly on Adjacent OOD tasks (AUROC ~45-55%, near random) while supervised MSP achieves substantially better performance (AUROC 69-79%). This directly validates the label blindness theory.

- **Clear evidence that existing benchmarks miss this failure mode**: Table 3 demonstrates that the same SimCLR KNN method achieves near-perfect AUROC (99.7%) on far OOD tasks, confirming that standard OOD benchmarks can mask the label blindness problem the authors identify.

## Weaknesses

- **The theoretical independence assumption is strong**: The main result (Corollary 3.3) requires strict independence between surrogate task features (x₁) and label features (x₂). While "Approximate Label Blindness" is mentioned via Fano's inequality, the paper provides limited analysis of how often this independence condition holds in practice or how to quantify "approximate" blindness in real datasets. The CIFAR-10 results in Table 2 (SSL achieving 73-77% AUROC on adjacent OOD) suggest the condition may not always hold strongly, warranting deeper analysis.

- **Minimal sufficient statistic assumption may not reflect practice**: Theorem 3.1 assumes representations are minimal sufficient statistics for the surrogate task. Modern deep networks are overparameterized and empirically learn "superfluous information" beyond the minimal sufficient statistic (section 2.2 acknowledges this). The theory should address whether non-minimal sufficient statistics could retain label information incidentally.

- **Limited exploration of why some SSL objectives work better**: The paper shows that different SSL methods (SimCLR vs. RotLoss) perform differently across datasets, but provides little analysis of which SSL objectives are more likely to learn label-relevant features. This limits the actionable guidance for designing better SSL methods for OOD detection.

- **The "unavoidable risk" theorem adds limited insight**: Theorem 4.1 states that finite training sets have non-zero probability of encountering unseen labels. While mathematically sound, this is well-understood in the OOD literature and doesn't substantially advance understanding beyond what Theorem 3.1 and Corollary 3.3 already establish.

## Nice-to-Haves

- Analysis comparing recent SSL methods (MAE, DINO, DINOv2) which may learn more semantically meaningful representations for the adjacent OOD benchmark would strengthen the empirical claims about the generality of label blindness.

- A systematic study measuring the degree of feature overlap between surrogate tasks and labels across different datasets/dmethods would help quantify "approximate" label blindness in practice.

## Novel Insights

The Adjacent OOD detection formulation elegantly exposes a fundamental blind spot in current OOD evaluation. By randomly splitting classes within the same dataset, the benchmark ensures maximum feature-space overlap between ID and OOD samples—precisely the scenario where label-blind representations must fail. This is a clever operationalization of the theoretical failure mode. The finding that zero-shot CLIPN's success depends critically on pretraining data alignment (Table 1: strong performance on Cars/Food where CC3M contains similar images, weak on Faces where "angry" captions don't match facial expressions) is an important empirical observation about when foundation models can compensate for lack of training labels versus when they cannot.

## Potentially Missed Related Work

- **"Do Vision Transformers See Like Convolutional Neural Networks?" (Raghu et al., 2021)** and subsequent ViT analysis papers — relevant for understanding whether different architectures learn more or less semantically meaningful representations without labels.

- **"Understanding failures in out-of-distribution detection with deep generative models" (Zhang et al., 2021)** — This paper analyzes why generative models fail at OOD detection and discusses related failure modes of likelihood-based methods.

- **"Is Out-of-Distribution Detection Learnable?" (Fang et al., NeurIPS 2022)** — Directly addresses learnability of OOD detection and provides theoretical analysis that complements this work.

- **Recent work on semantic features in SSL (e.g., "What does CLIP know about a red circle?" and similar probing studies)** — Relevant for understanding what semantic information SSL representations actually capture.

## Suggestions

- Expand the theoretical analysis to characterize when non-minimal sufficient statistics (overparameterized representations) might retain label information incidentally. This would help bridge the gap between the clean theoretical result and empirical observations where SSL sometimes works reasonably well.

- Provide guidance on quantifying "approximate label blindness" in practice. For a given SSL method and dataset, what metrics could indicate whether the surrogate task has sufficient mutual information with labels?

# Actual Human Scores
Individual reviewer scores: [6.0, 5.0, 8.0]
Average score: 6.3
Binary outcome: Accept

=== CALIBRATION EXAMPLE 9 ===

# Final Consolidated Review
## Summary
The paper presents LocoVR, a large-scale VR-collected dataset of 7000+ two-person indoor trajectories across 131 home environments, capturing social navigation behaviors (proxemics) like yielding and maintaining personal distance. The authors demonstrate utility through three tasks—global path prediction, trajectory prediction, and goal prediction—showing models trained on LocoVR outperform those trained on existing datasets when evaluated on real-world test data.

## Strengths
- **Novel dataset addressing a clear gap**: LocoVR provides 131 diverse indoor scenes with two-person socially-aware trajectories, filling a gap left by existing datasets that are either limited in scene variation (GIMO: 19 scenes, THOR-MAGNI: 4 scenes) or lack multi-person social dynamics in home environments. Table 1 provides a clear comparison with existing datasets.

- **Rigorous evaluation with real-world test data**: The authors create LocoReal, a physical-room test dataset, demonstrating that LocoVR-trained models achieve lower errors (e.g., ADE of 0.11m vs. 0.19m for U-Net on trajectory prediction, Table 3) than models trained on other datasets. Appendix C ablation studies quantify contributions of dataset scale, multi-person data, and heading direction.

- **Comprehensive dataset statistics**: Figures 10-17 and Tables 12-13 provide detailed statistics on trajectory lengths, durations, speed distributions, inter-person distances, and participant demographics, enabling informed use by future researchers.

- **Demonstrated learning of social behaviors**: Figure 7 and Appendix E show qualitative evidence that models trained on LocoVR learn to predict socially-aware paths (maintaining distance, yielding), which the ablation study in Table 6 quantitatively confirms by showing performance drops when removing the second person's trajectory input.

## Weaknesses
- **Limited model architecture exploration**: The paper uses only U-Net and Ynet architectures, dismissing more recent methods (NSP, Goal-GAN, SoPhie) in Section 4 without empirical comparison. While the authors argue these methods compress geometric features, lack of comparison with transformer-based or attention-based trajectory prediction models (e.g., Trajectron++, AgentFormer) limits understanding of whether dataset improvements are architecture-agnostic and misses an opportunity to benchmark against state-of-the-art.

- **Artificial task protocol**: The data collection uses random goal assignment rather than realistic home activities (e.g., cooking sequences, retrieving objects). While Section 3.2 describes the protocol, the "walk to random marker" approach may not capture the naturalistic goal-directed behaviors in home environments that robots would need to predict. The trajectories may prioritize coverage over behavioral authenticity.

- **Demographic and environmental limitations**: As acknowledged in Appendix A, the 32 participants are predominantly college-aged (18-42), limiting generalizability to children, elderly, or mobility-impaired populations. The environmental context also excludes real-world factors like pets, ambient noise, or varying furniture densities that affect navigation in actual homes.

## Nice-to-Haves
- Empirical comparison with at least one transformer-based trajectory prediction architecture to demonstrate improvements generalize across model families
- Quantitative metrics for social compliance (e.g., personal space violation rates, collision prediction accuracy) beyond position error metrics

## Novel Insights
This work demonstrates an important methodological finding: VR-collected data at sufficient scale can generalize better to real-world scenarios than smaller real-world datasets. The key insight is that scene diversity and sample quantity matter more than data collection modality for indoor trajectory prediction. The ablation studies (Tables 5-7) systematically decompose the value of LocoVR: scale reduction ("data-size-G", "data-size-T") causes notable performance drops, and removing the second person's trajectory (Table 6, "wo/p2") degrades trajectory prediction, confirming that multi-person social interactions are learned. The path efficiency analysis (Figure 11, mean 0.81 comparable to THOR-MAGNI) shows VR-collected trajectories exhibit realistic navigation complexity despite the controlled environment.

## Potentially Missed Related Work
- **Trajectron++ (Salzmann et al., 2020)**: Scene-compliant multi-agent trajectory prediction model relevant for benchmarking
- **AgentFormer (Yuan et al., 2021)**: Transformer architecture for multi-agent trajectory prediction
- **Social-NCE (Liu et al., 2022)**: Contrastive learning framework for social navigation, relevant for learning social behaviors from trajectory data
- **ST-P3 (Shen et al., 2023)**: Recent work on socially-aware trajectory prediction with implicit planning
- **CROWD-VIC (Su et al., 2023)**: Recent dataset capturing human interactions in crowded VR environments

## Suggestions
- Include empirical comparison with at least one transformer-based trajectory prediction method to demonstrate that LocoVR's benefits are not specific to U-Net/Ynet architectures
- Add quantitative social compliance metrics (e.g., rate of predicted paths maintaining >0.5m distance from other agents, collision avoidance success rate) to complement position error metrics and directly measure whether social behaviors are captured

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0, 6.0, 3.0]
Average score: 5.8
Binary outcome: Accept

=== CALIBRATION EXAMPLE 10 ===

# Final Consolidated Review
## Summary

The paper introduces Fine-tuned Score Deviation (FSD), a novel method for detecting pretraining data in large language models. The key insight is that fine-tuning an LLM on a small amount of non-member data from a specific domain causes perplexity scores to shift differently for members versus non-members—non-members exhibit larger decreases. By measuring this score deviation between the pre-trained and fine-tuned model, the method amplifies the distinction between seen and unseen data, improving upon existing scoring functions like Perplexity and Min-k%.

## Strengths

- **Strong empirical improvements across diverse settings.** Table 1 shows substantial AUC improvements: Min-k% improves from 0.62 to 0.91 on WikiMIA under OPT-6.7B, and from 0.76 to 0.86 on ArXivTection under LLaMA-7B. TPR@5%FPR improvements are equally dramatic, e.g., perplexity improves from 0.10 to 0.81 on ArXivTection under LLaMA-7B (Table 12).

- **Data efficiency demonstrated clearly.** Figure 4 shows that FSD achieves significant improvements with as few as 100 non-member samples, improving perplexity-based AUC from 0.63 to 0.91 on WikiMIA—a 44% relative improvement. This makes the method practically deployable.

- **Orthogonal to existing scoring functions.** The method can be applied on top of Perplexity, Min-k%, Zlib, and Lowercase methods (Tables 1-4), demonstrating broad applicability rather than proposing yet another scoring function in isolation.

- **Comprehensive ablation studies.** The paper investigates: (1) fine-tuning with members vs. non-members (Table 5), (2) different fine-tuning methods like AdaLoRA and IA3 (Table 7), (3) temporal shift mitigation (Table 6), (4) cross-domain fine-tuning (Table 16), and (5) fine-tuning parameter sensitivity (Table 17).

- **Addresses concerns about distribution shift in evaluation benchmarks.** Section 5 explicitly tackles the temporal shift issue by removing/replacing timestamps and shows FSD still provides meaningful improvements (e.g., perplexity improves from 0.54 to 0.71 on the Replacement dataset).

## Weaknesses

- **Domain specificity limits practical applicability.** Table 16 shows that FSD fails to improve performance when fine-tuning data comes from an unrelated domain (fine-tuning on WikiMIA while evaluating on ArXivTection yields near-random AUC around 0.50-0.64). This means practitioners must collect domain-matched non-member data for each domain of interest, which may be difficult for proprietary or diverse training corpora.

- **Requires model fine-tuning capability.** The method assumes access to fine-tune the target model. While the authors note this is feasible for open-source models and some commercial APIs, most production API providers (e.g., OpenAI, Anthropic) do not expose fine-tuning interfaces with arbitrary user data, severely limiting real-world applicability.

- **Evaluation setup may underestimate deployment challenges.** The experiments assume practitioners can reliably identify non-member data via temporal cutoffs. However, the exact training data cutoff dates for many LLMs are not publicly disclosed, and training data may span varying time periods across domains, making it non-trivial to construct clean non-member datasets in practice.

## Nice-to-Haves

- Analysis of how sensitive the method is to the quality and diversity of the fine-tuning data (e.g., what proportion of the 100 samples need to be truly non-members?).

- Comparison with recent reference-model-based MIA methods that train a shadow model or use a smaller reference model for comparison.

## Novel Insights

The core insight—that fine-tuning on non-members asymmetrically affects the perplexity distribution for members versus non-members—is elegant and well-motivated. The empirical finding that non-members show substantially larger perplexity decreases after fine-tuning (Figure 2) provides a clean signal that can be exploited. Notably, the method essentially leverages the differential "forgetting" or "adaptation" dynamics of the model: members are already well-optimized during pretraining, so additional fine-tuning has minimal effect, whereas non-members benefit more from gradient updates in their domain.

## Potentially Missed Related Work

- **Mattern et al. (2023) "Membership Inference Attacks against Language Models via Neighbourhood Comparison"** - Uses reference models to compare likelihoods, which shares conceptual similarities with FSD's comparative approach.

- **Carlini et al. (2022) "Quantifying Memorization in LLMs"** - More recent work on detecting memorization that could serve as additional baseline context.

## Suggestions

- Provide a more detailed discussion of failure modes: specifically, analyze cases where domain-mismatched fine-tuning data harms performance and quantify how much domain overlap is needed for the method to work.

- Include an analysis of computational overhead: report the time/cost of fine-tuning for FSD compared to baseline inference-only methods, as this affects practical deployment decisions.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 6.0, 5.0]
Average score: 6.2
Binary outcome: Accept

=== CALIBRATION EXAMPLE 11 ===

# Final Consolidated Review
## Summary

This paper introduces Process Advantage Verifiers (PAVs), a novel approach to process reward modeling for LLM reasoning. The key insight is that process rewards should measure "progress" (advantage) rather than absolute Q-values, and should be computed under a distinct "prover" policy—not the base policy itself. Theoretically, the paper characterizes "complementary provers" that distinguish steps without being too misaligned with the base policy, showing that even weaker provers can improve stronger base policies. Empirically, PAVs improve test-time search compute efficiency by 1.5–5× and accuracy by 8–10% over ORMs, and enable 6× sample efficiency gains in online RL—a claimed first for PRMs over ORMs.

## Strengths

- **Novel conceptual insight about prover policies**: The paper provides a counter-intuitive and well-motivated insight that using the base policy's own advantages as process rewards is ineffective (A^π ≈ 0 for poor policies, and equivalent to ORM rewards otherwise). The theoretical analysis (Theorem 3.1) formalizes why "complementary" provers—those that can distinguish actions without being too misaligned—improve the base policy. The observation that weaker provers can sometimes outperform stronger ones (e.g., Gemma 2B prover outperforms Gemma 9B prover for Gemma 27B base policy, Figure 5c) is genuinely surprising and well-explained by the theory.

- **Strong theoretical grounding**: Theorem 3.1 provides a lower bound on policy improvement that decomposes into distinguishability (variance of A^μ) and alignment (correlation between A^μ and A^π) terms. Remark 3.1 specifically analyzes Best-of-K policies, showing K^2 scaling for distinguishability with only O(K) alignment degradation. The proof is complete and the analysis connects naturally to the experimental findings.

- **Substantial empirical gains with comprehensive evaluation**: The paper demonstrates consistent improvements across three model scales (2B, 9B, 27B Gemma models) on the MATH benchmark. The 6× sample efficiency improvement for online RL (Figure 7c) is indeed one of the first demonstrations that PRMs can substantially outperform ORMs in RL training, addressing a key limitation noted in prior work (Shao et al., 2024 reported only 1–2% gains). The didactic experiments (Figure 3) provide clean ablations validating the theoretical claims.

- **Well-designed ablation studies**: The paper systematically ablates prover choice (Figure 5b-c), compares advantages vs Q-values (Figures 2, 6), and analyzes discounted vs undiscounted rewards for strong provers (Appendix E, Figure 16). The finding that Q-based process rewards lead to degenerate "rephrase the problem" solutions (Appendix G) is an informative negative result.

## Weaknesses

- **Limited guidance on practical prover selection**: While the paper shows that Bo4 works well empirically and provides theoretical intuition for why intermediate-strength provers are optimal, there is no principled method for determining the best prover for a new base policy. The recommendation to use BoK(π) with K > 1 is useful but the optimal K varies by model (Figure 5b shows Bo4 dominates, but this may not generalize). A more systematic approach for prover selection would strengthen the practical applicability.

- **Monte Carlo estimation noise is not rigorously analyzed**: The PAV training procedure relies on Monte Carlo estimates of Q-values (Q_mc in Appendix D), which can be noisy—especially for early steps where many rollouts are needed. While Figure 13 analyzes the n_mc/n_cov tradeoff, the paper does not characterize how estimation error propagates through the advantage computation or affects the theoretical guarantees. The theoretical analysis assumes oracle access to Q^π and A^μ, but learned approximators introduce errors that could affect the distinguishability-alignment tradeoff.

- **Hyperparameter α requires tuning without principled guidance**: The mixing coefficient α between outcome rewards and process advantages is tuned separately for test-time search (α ∈ [0.1, 0.7]) and RL (α ∈ [1.5, 5.5]). The paper acknowledges tuning α on a validation set (Appendix E) but provides no principled method for setting it. Given that α controls the exploration-exploitation tradeoff, this limits reproducibility and transfer to new tasks.

## Nice-to-Haves

- The paper could benefit from evaluation on additional benchmarks beyond MATH (e.g., GSM8K, GPQA, or code reasoning tasks) to demonstrate generalization of the approach.

- A more detailed comparison with recent PRM approaches like Math-Shepherd (Wang et al., 2024) in terms of data efficiency and compute would strengthen the positioning. The comparison in Figure 12 is useful but limited to test-time search.

## Novel Insights

The paper makes a genuinely novel observation that the ineffectiveness of prior PRM approaches stems from using Q-values (which confound state promise with action progress) or from using the base policy itself to estimate process rewards. The key insight—that progress should be measured as advantage under a complementary prover policy—reframes the PRM problem from "what makes a step correct?" to "what step-level signal can accelerate exploration?". This shift in perspective explains why the same prover can help both test-time search (by guiding beam exploration) and RL (by providing dense, informative gradients), and connects process supervision to classic RL concepts like potential-based reward shaping (Ng et al., 1999). The theoretical characterization of complementary provers as those with high variance in A^μ but positive correlation with A^π provides a principled framework for understanding when process rewards help.

## Potentially Missed Related Work

- **"Process Supervision for LLM Reasoning: Theoretical Analysis and Practical Implications"** (work on understanding when/why process supervision helps) — may provide additional theoretical perspectives on the advantages of PRMs.

- **"Scaling Test-Time Compute with Sequential Verification"** (recent work on compute-optimal inference scaling) — could be relevant for comparison on test-time efficiency claims.

- **"Self-Taught Reasoner (STaR) and Re-STaR"** (Zelikman et al.) — while cited, the connection to iterative improvement could be explored further, particularly whether PAVs could be combined with expert iteration.

## Suggestions

- **Provide a practical algorithm for prover selection**: Even a simple heuristic based on the base policy's accuracy (e.g., choose K such that BoK(π) achieves ~60-80% accuracy on a validation set) would improve usability. The current approach of testing multiple K values is not scalable for practitioners.

- **Analyze the variance-bias tradeoff in Monte Carlo estimates**: A theoretical or empirical analysis of how many MC rollouts (n_mc) are needed per step to achieve stable advantage estimates would strengthen the methodology. Consider comparing with learned value networks as an alternative to MC estimation.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 8.0, 8.0, 6.0]
Average score: 7.1
Binary outcome: Accept

=== CALIBRATION EXAMPLE 12 ===

# Final Consolidated Review
## Summary
The paper introduces the task of audio difference explanation—generating natural language descriptions of differences between two audio recordings. The authors create two datasets (ACD and CLD) derived from AudioCaps and Clotho with three tiers of explanations (concise, brief, detailed) generated via LLM prompting and verified by humans on the test set. They propose ADIFF, a prefix-tuning approach with cross-projection and three-stage training, demonstrating improvements over a naive baseline and Qwen-Audio.

## Strengths
- **Novel task formulation**: Audio difference explanation is a meaningful and previously unexplored problem with applications in forensics, quality assessment, and generation (Section 1). The tiered explanation framework (concise/brief/detailed) provides a useful granularity hierarchy for evaluation.
- **Comprehensive empirical evaluation**: The paper includes objective metrics (BLEU, METEOR, SPICE, CIDEr, SPIDEr) and human evaluation across correctness, granularity, and readability dimensions (Table 3), as well as extensive ablations on cross-projection, LM scaling, position captioning, and finetuning (Section 5).
- **Practical insights from ablations**: The finding that smaller LMs (GPT-2 base) can outperform larger ones under limited compute constraints (Section 5.3, Table 5, Figure 4) provides valuable guidance for resource-constrained audio-language model development.
- **Hallucination detection mechanism**: The proposal to leverage frozen HTSAT encoder's audio event predictions for verifying generated explanations (Section 6, Figure 5) offers a practical tool for grounding validation.

## Weaknesses
- **Dataset quality concerns from LLM generation**: The datasets are constructed by prompting an LLM to generate difference explanations from existing human captions (Section 2.1, Appendix E). While human verification is performed on the test set, the training data inherits any hallucinations or artifacts from LLM generation, and the quality of explanations depends entirely on the quality of underlying captions. The paper notes AudioCaps annotators had access to visual cues (Section 2.1), which could introduce systematic biases that propagate through the pipeline.

- **Limited SoTA comparison**: The paper compares only against Qwen-Audio, justified as "the only ALM in literature that supports two audio inputs" (Section 4.1). However, other ALMs like SALMONN, GAMA, or LTU could potentially be adapted by concatenating audio embeddings or using similar multi-audio strategies, which would provide a more comprehensive baseline comparison.

- **GPT-2 backbone limitations**: Using GPT-2 (124M parameters) as the language model backbone is a significant architectural limitation. While Section 5.3 explores scaling, the final ADIFF model uses GPT-2 base, which may explain some of the quantitative limitations (e.g., ADIFF scores around 3.5/5 on human evaluation). The choice is attributed to "compute limitations" but leaves open whether stronger LLM backbones would yield substantially better performance.

- **Cross-projection analysis remains speculative**: The analysis in Section J and Table 11 interprets cross-projection outputs by mapping to vocabulary tokens, concluding that "text prefix gets used for...storing information about attributes to be used for comparison." However, this interpretation is approximate and relies on dot product similarity with vocabulary embeddings, lacking rigorous validation of whether this mechanism truly enables comparative reasoning.

## Nice-to-Haves
- Direct comparison against recent ALMs (SALMONN, GAMA, AudioFlamingo) adapted for dual-audio input would strengthen the empirical contribution.
- Error analysis categorizing common failure modes would provide actionable insights for future work.

## Novel Insights
The paper reveals that cross-projection enables the text prefix to store comparative attributes rather than simply steering generation—a departure from typical prefix-tuning behavior where prefixes primarily condition the model. The position captioning approach, where models are trained to caption "audio 1" or "audio 2" individually, demonstrates a simple yet effective method for improving grounding in multi-audio scenarios. The three-tier explanation framework also provides a useful evaluation scaffold for distinguishing surface-level from deep audio understanding.

## Potentially Missed Related Work
- **Audio difference learning for audio captioning (Komatsu et al., 2024)**: The paper cites this but could more explicitly contrast with it—Komatsu et al. focus on improving captioning via difference learning rather than explaining differences.
- **CLAP-style contrastive audio-language models (Elizalde et al., 2023; Wu et al., 2023)**: These could provide alternative baselines for audio comparison tasks.
- **Qwen2-Audio (Chu et al., subsequent work)**: More recent versions may offer improved dual-audio handling.

## Suggestions
- Include explicit analysis of LLM-generated explanation quality in the training data—for instance, by having human annotators rate a sample of training explanations or by comparing LLM-generated vs. human-verified explanations on the test set to quantify systematic biases.
- Add experiments adapting a stronger LM backbone (even LoRA-adapted) to demonstrate whether architectural gains from ADIFF components transfer to more powerful language models.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept

=== CALIBRATION EXAMPLE 13 ===

# Final Consolidated Review
## Summary
The paper introduces SAM 2, a foundation model for promptable visual segmentation in images and videos, extending SAM to the temporal domain. The authors contribute a streaming transformer architecture with memory attention, the large-scale SA-V dataset (35.5M masks, 53× larger than previous VOS datasets), and an iterative data engine that achieves 8.4× faster annotation. SAM 2 demonstrates strong performance across video segmentation benchmarks with 3× fewer interactions than prior approaches, and improves upon SAM for image segmentation while being 6× faster.

## Strengths
- **Unified architecture for images and videos**: The streaming memory design elegantly generalizes SAM to video while maintaining backward compatibility. When memory is empty (single frame), the model behaves like SAM. The architecture processes frames sequentially with cross-attention to memories of past frames, enabling real-time processing (Section 4, Figure 3).

- **Substantial dataset contribution**: The SA-V dataset with 35.5M masks across 50.9K videos is 53× larger than YouTube-VOS (Table 3). The dataset includes object parts (not just whole objects), has high disappearance rate (42.5%) indicating occlusion challenges, and demonstrates geographic diversity across 47 countries (Figure 10).

- **Impressive empirical results across multiple settings**: SAM 2 outperforms SAM+XMem++ and SAM+Cutie baselines on 9 zero-shot video datasets in both offline and online interactive evaluation (Figure 5, Tables 13-14). On image segmentation, SAM 2 achieves 58.9 mIoU vs SAM's 58.1 mIoU while being 6× faster (Table 5).

- **Rigorous ablation studies**: The paper provides comprehensive ablations on data mixtures (Table 7), data quantity (Figure 6), architectural choices including memory size, resolution, and positional encoding (Tables 9-11), demonstrating that both the model design and training data contribute to performance.

- **Data engine innovation demonstrates practical impact**: The phased data engine shows progressive efficiency gains from Phase 1 (37.8 s/frame) to Phase 3 (4.5 s/frame), with maintained quality as measured by Phase 1 Mask Alignment Score (Table 1). This represents meaningful practical advancement for annotation workflows.

## Weaknesses
- **Limited multi-object efficiency analysis**: The paper states SAM 2 processes each object separately with shared per-frame embeddings but no inter-object communication (Section D.1). However, no experiments or analysis are provided on how performance degrades with increasing number of objects, which is critical for practical video understanding applications with multiple entities.

- **Incomplete comparison to unified video segmentation paradigms**: While the paper briefly mentions Video Instance Segmentation and Video Panoptic Segmentation in Appendix I, there is no experimental comparison on VIS/VPS benchmarks. Given the claimed foundation model status, evaluating these related unified video segmentation tasks would strengthen the paper's positioning.

- **Training data reproducibility gap**: The model is trained on a mix including an "Internal" dataset (62.9K videos, 69.6K masklets) that is not released (Section 5.2, Table 3). While SA-V is open-sourced, the internal data creates a reproducibility gap between the released model and what others can achieve.

- **Streaming constraint may limit performance on long videos**: The streaming memory design, while enabling real-time processing, limits temporal context. Although the 16-frame fine-tuning helps (Section D.2.2), the ablation in Table 9c shows performance varies with memory size. The paper lacks analysis on very long videos beyond LVOSv2 (average 1.14 minutes).

## Nice-to-Haves
- Analysis of memory bank scaling for videos longer than those in LVOSv2 would help characterize limitations for extended video understanding applications
- A comparison table showing inference time vs. accuracy trade-offs for different memory bank configurations would help practitioners choose appropriate settings

## Novel Insights
The paper makes an important architectural observation: treating image segmentation as a special case of video segmentation (empty memory bank) yields both better performance and efficiency compared to the original SAM. This suggests that video pretraining transfers positively to static images—counter to concerns that temporal modeling might introduce noise for single-frame tasks. The data engine finding that automatically-generated masklets filtered through human verification effectively augment manual annotations (Table 2: "Auto" row shows improvement) is a practical insight for scaling segmentation dataset collection. The object pointer mechanism (Table 11), which adds lightweight semantic vectors to spatial memory features, provides surprisingly large gains on LVOSv2 (+4.6 J&F) despite minimal impact on other benchmarks, suggesting it specifically addresses long-horizon object identity preservation.

## Potentially Missed Related Work
- **Video-kmax (Shin et al., 2024)** - A unified approach for online and near-online video panoptic segmentation that could provide relevant comparison for unified video segmentation capabilities.
- **OMG-Seg (Li et al., 2024)** - "Is One Model Good Enough for All Segmentation?" proposes a unified model for image and video segmentation tasks including VIS and VPS, presenting an alternative unified approach worth discussing in relation to SAM 2's design choices.
- **Tube-Link (Li et al., 2023b)** - A flexible cross-tube framework for universal video segmentation that addresses multiple video segmentation tasks with a unified architecture.

## Suggestions
- Add experiments on VIS and VPS benchmarks (e.g., YouTube-VIS 2019, VIPSeg) to demonstrate the broader applicability of SAM 2 as a foundation model. The current VOS-only evaluation limits claims about unified segmentation capabilities.
- Provide per-object inference time analysis showing how SAM 2 scales with the number of objects in a video, and discuss potential efficiency improvements for multi-object scenarios.

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 8.0, 10.0]
Average score: 9.0
Binary outcome: Accept

=== CALIBRATION EXAMPLE 14 ===

# Final Consolidated Review
## Summary
The paper introduces "shallow safety alignment" as a unifying concept explaining why current LLM safety mechanisms are vulnerable to diverse attacks. The authors demonstrate that alignment primarily modifies only the first few output tokens, and propose two mitigation strategies: (1) a data augmentation approach using "safety recovery examples" to deepen alignment, and (2) a constrained fine-tuning objective that protects initial token distributions during downstream fine-tuning. Empirical results show both approaches improve robustness against multiple attack types while maintaining utility.

## Strengths
- **Unified conceptual framework**: The paper introduces "shallow safety alignment" as a principled explanation for multiple previously disconnected vulnerabilities (adversarial suffix attacks, prefilling attacks, decoding parameter exploits, fine-tuning attacks). This conceptual unification is valuable for the field.

- **Strong empirical characterization**: Figure 1 provides compelling evidence that KL divergence between aligned and unaligned models concentrates heavily in initial token positions. Table 1 shows that even unaligned base models achieve safety comparable to aligned models when forced to start with refusal prefixes, demonstrating the "safety shortcut" phenomenon.

- **Effective mitigation proposals with solid evaluation**: The data augmentation approach (Table 3) substantially reduces ASR against prefilling attacks (42.1%→2.8% at 5 tokens), GCG attacks (36.5%→18.4%), and decoding parameter exploits. The constrained fine-tuning objective (Table 4) preserves safety during adversarial fine-tuning while maintaining utility on benign tasks.

- **Theoretical grounding with practical insights**: Appendix F provides rigorous theoretical analysis of the constrained objective's limiting behaviors and gradient properties, while Section 2.3.2's per-token dynamics analysis (Figure 3) reveals why early tokens are critical—their higher initial loss and larger gradients make them vulnerable during fine-tuning attacks.

## Weaknesses
- **Limited evaluation against adaptive attacks**: While Table 3 and Table 4 demonstrate effectiveness against existing attacks, the paper lacks evaluation against adaptive attacks specifically designed to counter the proposed defenses. For the data augmentation approach, an attacker aware of the "recovery" training could potentially craft prefixes that specifically circumvent it.

- **Suboptimal implementation of data augmentation**: The authors acknowledge in Section 3.1 and Appendix B.3 that they can only fine-tune an already-aligned Llama-2-7B-Chat rather than integrating the augmentation into the original alignment pipeline. This limits conclusions about how effective "deep safety alignment" could be when properly implemented from scratch.

- **Potential generalization concerns**: The approach's reliance on specific refusal prefix tokens (e.g., "I cannot", "I apologize") raises questions about generalization. Table 1 shows these prefixes are specific to certain model families; models with different refusal styles or non-English training may require different approaches without clear guidance.

## Nice-to-Haves
- Evaluation on additional model families beyond Llama-2 and Gemma to assess generalization
- Analysis of whether the constrained fine-tuning objective's computational overhead (storing π_aligned probabilities for all training data) is practical for large-scale commercial deployment
- Investigation of whether the approach works in multilingual settings

## Novel Insights
The paper's core insight—that safety alignment operates as a "gate" on initial tokens rather than a comprehensive modification of harmful response distributions—provides a mechanical explanation for why diverse attack methodologies all succeed. The per-token dynamics analysis (Figure 3) reveals that during fine-tuning attacks, early tokens have both higher initial cross-entropy loss AND larger gradient norms, creating a compounding vulnerability: the alignment learned most at early positions is also most vulnerable to unlearning. The connection between benign fine-tuning dynamics and safety regression (Appendix C) is also noteworthy—overconfidence in affirmative prefixes from alignment training creates large gradient signals when fine-tuning on datasets without such prefixes, potentially explaining "benign" safety regression.

## Potentially Missed Related Work
- Xu et al. (2024) "SafeDecoding" - Proposes a similar insight about amplifying safety-relevant tokens during decoding, though from a different angle
- Rosati et al. (2024) "Representation noising" - Related approach to preserving safety during fine-tuning
- Zou et al. (2024) "Short circuiting" - Concurrent work on disrupting harmful generation through representation-space intervention

## Suggestions
- Conduct adaptive attack evaluation: Design attacks that explicitly target the data augmentation defense by finding prefixes that lead to harmful continuations even after the model's recovery training
- Provide clearer guidance on computational overhead for deployment: While Appendix G.2 addresses this, include memory requirements and any preprocessing costs in the main text for production considerations
- Evaluate multilingual models: Test whether the shallow alignment phenomenon and proposed mitigations extend beyond English-only models to understand cross-lingual applicability

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 10.0, 10.0]
Average score: 9.5
Binary outcome: Accept
