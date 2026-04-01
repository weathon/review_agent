=== CALIBRATION EXAMPLE 1 ===

# Final Consolidated Review
Summary: This paper presents a review of Large Language Models, covering their types (generative, masked, sequence-to-sequence, hybrid), deep learning techniques used in their development, comparative analysis of major models, limitations, ethical considerations, evaluation metrics, and future directions. The paper aims to provide a comprehensive overview of the LLM landscape for readers seeking an introduction to the field.
Strengths:
- **Broad topical coverage**: The paper addresses a reasonable breadth of LLM-related topics, from architectural foundations (Transformer, self-attention) to contemporary concerns (hallucination in Section 3.10, adversarial robustness in Section 5, ethical considerations in Section 7).
- **Inclusion of recent benchmark frameworks**: The paper introduces readers to HELM (Section 3.9.1) and LMSYS Chatbot Arena (Section 3.9.2), which are relevant evaluation frameworks for understanding modern LLM assessment.
- **Reference list with recent work**: The bibliography includes relevant and reasonably current citations (through 2024), pointing readers to key papers including Zhao et al. (2023) and Bommasani et al. (2021).
- **Comparative table**: Table 1 provides a structured side-by-side comparison of BERT, GPT-3, T5, and BART, summarizing their pros, cons, training datasets, and relevant metrics.
Weaknesses:
- **Title misrepresents methodology**: The paper is titled 'Systematic Review of Large Language Models' but does not follow systematic review methodology. A genuine systematic review requires a documented search strategy, databases queried, inclusion/exclusion criteria, quality assessment framework, and typically a PRISMA flow diagram. The paper appears to be a narrative literature review with ad hoc citation selection.
- **Abstract-content mismatch on applications**: The abstract promises to 'delve into specific applications such as text generation, translation, summarization, and more,' but the paper contains no dedicated applications section. These topics receive only brief, scattered mentions within the literature review rather than substantive treatment.
- **Internal content duplication**: The paper contains significant duplicated text. For example, 'Denoising Autoencoders' and 'Masked Language Modeling' appear as bullet points in Section 3.1 and then again verbatim after Section 3.3. The Transformer architecture description repeats across Sections 1, 3.2, and after 3.3. This indicates insufficient editorial care.
- **No clear differentiation from existing surveys**: The paper cites Zhao et al. (2023) and Bommasani et al. (2021), both comprehensive and highly-cited surveys, but does not articulate what unique contribution this paper provides. Section 3.11 'Comparison with Recent Reviews' is only two paragraphs and offers no substantive differentiation.
- **Limited and outdated model coverage**: Table 1 covers only BERT, GPT-3, T5, and BART while omitting major developments including LLaMA, Mistral, PaLM, Claude, instruction tuning, RLHF, and retrieval-augmented generation that are central to current LLM research.
- **Table 1 lacks quantitative data**: The 'Metrics' column lists metric names (Accuracy, F1, Perplexity, BLEU, ROUGE) without actual numerical benchmark results, limiting its utility for readers seeking performance comparisons.
Nice-to-haves:
- **Related work suggestions**: The authors may wish to consider citing recent work on instruction tuning, chain-of-thought reasoning, retrieval-augmented generation (RAG), parameter-efficient fine-tuning (LoRA, Adapters), and open-source model families (LLaMA, Mistral, Falcon) to better represent the current LLM landscape.
- **Methodology clarification**: If the intention is a true systematic review, adding a methods section with search terms, databases, inclusion/exclusion criteria, and a PRISMA diagram would strengthen credibility.
- **Quantitative synthesis**: Adding actual benchmark numbers to Table 1 or including a meta-analysis aggregating results from multiple papers would provide more value to readers.
Novel insights: The spark finder report suggested several directions that could strengthen the work, including: developing a novel taxonomy connecting architecture choices to downstream capabilities; conducting original small-scale empirical comparisons; and narrowing focus to a specific aspect (e.g., efficiency-accuracy trade-offs, hallucination mitigation) to enable deeper, more differentiated analysis. These represent opportunities for future development rather than requirements for the current submission.
Potentially missed related work:
- Recent instruction tuning methodologies (e.g., Wei et al. on fine-tuning approaches)
- RLHF/RLAIF techniques for alignment
- Retrieval-augmented generation (RAG) frameworks
- Parameter-efficient fine-tuning methods (LoRA, Adapters)
- Open-source model families: LLaMA, Mistral, Falcon
- Multilingual LLMs and cross-lingual transfer
Suggestions:
- Consider revising the title to accurately reflect the paper type (e.g., 'A Survey of Large Language Models' rather than 'Systematic Review') or adopt genuine systematic review methodology.
- Add a dedicated applications section to fulfill the abstract's promise, with specific examples and performance benchmarks for tasks like text generation, translation, and summarization.
- Expand model coverage to include recent architectures and techniques that define the modern LLM landscape.
- Provide quantitative benchmark data in the comparative analysis rather than just listing metric names.
- Remove duplicated content and improve editorial consistency throughout the manuscript.

# Merger Subscores
Novelty: 3.0
Technical soundness: 5.0
Empirical support: 3.0
Significance: 3.0
Clarity: 5.0

# Actual Human Scores
Individual reviewer scores: [1.0, 1.0, 1.0, 1.0]
Average score: 1.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 2 ===

# Final Consolidated Review
Summary: This paper proposes Multi-Objective Decision-Focused Learning (MoDFL), extending the decision-focused learning paradigm from single-objective to multi-objective optimization problems. The authors introduce three novel loss functions—landscape loss (using sRMMD to measure objective space discrepancy), Pareto set loss (measuring solution space distance), and decision loss (measuring representative solution quality)—to align predictive model training with downstream multi-objective optimization. Experiments on two benchmark problems demonstrate improved performance over two-stage methods and adapted single-objective DFL baselines.
Strengths:
- **Addresses an important and underexplored gap**: The paper correctly identifies that existing decision-focused learning methods focus almost exclusively on single-objective optimization, while multi-objective problems are prevalent in real-world applications. This extension is non-trivial due to Pareto front complexity and conflicting gradient directions.
- **Well-motivated loss function design**: The three proposed loss functions are thoughtfully designed to capture different aspects of MOP structure: (1) landscape loss treats the multi-dimensional objective space as a manifold and uses sRMMD to measure discrepancies, (2) Pareto set loss directly measures solution space distance using concepts from inverted generational distance, and (3) decision loss handles representative solution quality via weighted-sum scalarization.
- **Comprehensive experimental comparison**: The paper compares against seven baseline methods (TwoStage, SPO, BB, MAP, NCE, Pointwise, Listwise) using multiple appropriate MOP evaluation metrics (GD, MPFE, HAR, regret).
- **Ablation studies validate design choices**: Table 5 demonstrates that removing any of the three loss components degrades performance, with decision loss having the largest impact, providing empirical support for the multi-component design.
- **Differentiation approach is properly addressed**: Section 4.2 clearly explains how gradients are computed through reparameterization (for landscape loss) and DSLP (for decision/Pareto set losses), addressing the core technical challenge of non-differentiable optimization mappings.
Weaknesses:
- **Limited to linear programming problems**: The method relies on DSLP (Wilder et al., 2019) for differentiation, which is designed for linear programming. The paper does not address how to extend the approach to mixed-integer programs, nonlinear optimization, or convex programs, limiting the scope of applicability.
- **Weighted-sum scalarization has known limitations**: The decision loss uses weighted-sum scalarization to transform MOP to single-objective problems. This approach cannot find solutions on non-convex portions of the Pareto front—a well-known limitation in the MOP literature that the paper does not acknowledge or discuss.
- **Missing statistical significance testing**: Despite claiming experiments were "repeated 5 times for consistency," Tables 1-5 report only single values without standard deviations, confidence intervals, or statistical significance tests. This undermines claims of "significant" improvements.
- **Incomplete related work on multi-objective optimization**: The related work section covers DFL literature thoroughly but omits discussion of classical MOP literature and recent work on neural network approaches to multi-objective optimization (e.g., neural Pareto set learning). This makes it difficult to assess the paper's positioning relative to existing MOP+learning work.
- **Weak scalability demonstration**: The experiment with three objectives (Section 5.3.3) uses a third objective that is a weighted sum of the first two, which does not introduce genuinely independent conflicting objectives. Scalability to many-objective problems (4+ objectives) remains untested.
- **Hyperparameter sensitivity not analyzed**: The loss combination weights (λ_l=1, λ_d=2, λ_ps=5) are provided without justification or sensitivity analysis. Practitioners have no guidance on how to tune these for different problem types.
- **Computational cost not discussed**: The method requires solving MOPs during training and maintaining solution caches, but no analysis of training time or memory overhead compared to baselines is provided.
Nice-to-haves:
- Testing on non-linear multi-objective problems to demonstrate broader applicability beyond LP
- Comparison with alternative scalarization methods (e.g., Chebyshev, ε-constraint) that can handle non-convex Pareto fronts
- Analysis of how the three loss components interact during training and potential conflicts between them
- Theoretical analysis such as convergence guarantees or conditions under which the proposed losses improve decision quality
Novel insights: The landscape loss approach—treating the multi-dimensional objective space as a manifold and using sRMMD to measure discrepancy—provides an interesting perspective that could inspire similar approaches in other learning-for-optimization contexts. The insight that single-objective DFL methods adapted via weighted-sum loss may fail to balance multiple objectives effectively (as shown by MAP, Pointwise, and BB in Table 1) is a valuable empirical finding that motivates dedicated multi-objective approaches.
Potentially missed related work:
- The paper may benefit from engaging with classical multi-objective optimization literature (e.g., Deb et al.'s work on NSGA-II, MOEA/D) and recent neural network approaches to multi-objective optimization such as neural Pareto set learning methods. These connections could strengthen the positioning and clarify what is novel versus adapted from existing MOP techniques.
Suggestions:
- Add standard deviations or confidence intervals to all experimental results and include statistical significance tests when claiming improvements over baselines
- Acknowledge the weighted-sum limitation for non-convex Pareto fronts and discuss potential remedies or the class of problems for which this approach is suitable
- Include a dedicated limitations section discussing: (1) restriction to LP problems, (2) computational cost of Pareto set computation during training, (3) lack of scalability validation beyond 3 objectives
- Expand related work to cover multi-objective optimization literature and clarify positioning relative to existing work on learning for MOP
- Provide hyperparameter sensitivity analysis or principled guidance for setting λ weights

# Merger Subscores
Novelty: 6.5
Technical soundness: 6.0
Empirical support: 6.0
Significance: 6.0
Clarity: 7.0

# Actual Human Scores
Individual reviewer scores: [3.0, 1.0, 3.0]
Average score: 2.3
Binary outcome: Reject

=== CALIBRATION EXAMPLE 3 ===

# Final Consolidated Review
Summary: This paper proposes MQFL-FHE, a framework that integrates quantum computing with fully homomorphic encryption (FHE) in multimodal federated learning to mitigate FHE-induced accuracy degradation during model aggregation. The authors introduce a Multimodal Quantum Mixture of Experts (MQMoE) architecture with modality-specific quantum experts and attention-based gating, evaluating the approach on CIFAR-10, DNA sequences, MRI scans, PCOS, and combined DNA+MRI datasets.
Strengths:
- **Novel integration of emerging technologies**: The combination of quantum computing, FHE, and multimodal federated learning addresses a genuine gap. The paper correctly identifies that FHE causes accuracy degradation during FL aggregation and proposes an innovative approach combining three technologies that have not been jointly explored.
- **Comprehensive experimental evaluation**: The framework is evaluated across multiple datasets (CIFAR-10, DNA Sequence, MRI Scan, PCOS, DNA+MRI) and experimental configurations (classical centralized, quantum centralized, classical FL, QFL, FL+FHE, QFL+FHE), demonstrating consistent patterns across diverse settings.
- **Demonstrated empirical improvements**: Results show QFL+FHE outperforms FL+FHE across datasets. The ROC analysis in Figure 4 shows macro-average AUC improvements from 0.92 to 0.95 (DNA) and 0.93 to 0.97 (MRI) when adding quantum enhancements. The multimodal DNA+MRI experiments demonstrate real-world applicability in medical domains.
- **Thoughtful MQMoE architecture**: Figure 3 presents a well-designed architecture with modality-specific experts (MRI through convolutional layers, DNA through linear layers), 6-qubit quantum circuits for both modalities, and multi-head attention-based gating for expert selection.
Weaknesses:
- **Weak theoretical foundation for core claim**: The central claim that quantum computing counteracts FHE-induced noise lacks rigorous theoretical grounding. Section 7.1 discusses quantum noise confinement through SU(2) norm preservation, but this analysis addresses quantum noise rather than establishing how quantum operations specifically mitigate CKKS FHE noise accumulation from homomorphic addition/multiplication. The claim that "rotation-induced discrepancies can counterbalance FHE-induced errors" is asserted without formal derivation.
- **Missing statistical significance testing**: Tables 3-4 report mean ± standard deviation for timing but provide no confidence intervals or statistical tests for accuracy metrics. Given the modest improvements observed (often 2-4%), statistical significance is essential to validate that improvements are not due to random variation.
- **Limited baseline comparisons**: The Related Work section mentions FheFL (Rahulamathavan et al. 2023) and FedSHE (Pan et al. 2024) as prior FHE-FL methods, but no direct comparison with these approaches is provided. Comparing against established FHE-FL baselines would better position the contribution.
- **Superficial ablation study**: Section 5.1 provides qualitative observations rather than quantitative ablation results. The paper lacks systematic analysis isolating contributions from: (a) quantum layers vs classical layers, (b) MoE architecture vs single model, (c) attention-based fusion vs simpler fusion methods.
- **Reliance on quantum simulation**: All experiments use PennyLane quantum simulators on classical hardware. While this is common practice, the paper does not address how NISQ-era quantum noise, decoherence, or limited qubit connectivity would affect results on real quantum hardware.
- **Significant computational overhead**: Table 4 shows QFL+FHE requires approximately 2-3× more time than classical FL (e.g., CIFAR-10: 9747s vs 3406s). The trade-off between modest accuracy improvements and substantial computational cost is mentioned but not analyzed.
Nice-to-haves:
- Testing with non-IID data distributions (e.g., Dirichlet partitions) would demonstrate robustness to realistic heterogeneous FL scenarios.
- Comparison with other privacy-preserving mechanisms (differential privacy, secure aggregation) would contextualize the privacy-utility-computation tradeoff.
- Communication overhead analysis (ciphertext sizes, bandwidth per round) would help assess practical deployment feasibility.
- Convergence analysis showing loss/accuracy curves over communication rounds would validate optimization stability.
Novel insights: **Quantum operations as regularizers in privacy-preserving learning**: The empirical observation that quantum layers improve accuracy specifically in FHE settings suggests an interesting interaction between quantum representations and encrypted data processing. Even without complete theoretical understanding, this phenomenon merits further investigation and could inspire new approaches to mitigating encryption-induced degradation.
Potentially missed related work:
- The related work coverage is adequate. The paper cites relevant prior work on QFL (FedQNN), FHE in FL (FheFL, FedSHE), and multimodal FL (CreamFL). No significant omissions were identified.
Suggestions:
- Add statistical significance testing (confidence intervals, hypothesis tests) for accuracy metrics across multiple experimental runs.
- Deepen the ablation study with quantitative results systematically varying: quantum vs classical experts, MoE vs single model, and attention fusion vs simpler alternatives.
- Include direct comparison with at least one prior FHE-FL method to demonstrate relative performance.
- Strengthen theoretical analysis by formally connecting CKKS noise growth mechanics to quantum state evolution properties, or alternatively, acknowledge the current limitations of the theoretical argument.

# Merger Subscores
Novelty: 7.5
Technical soundness: 5.5
Empirical support: 6.0
Significance: 6.0
Clarity: 6.5

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 3.0, 3.0, 3.0]
Average score: 3.4
Binary outcome: Reject

=== CALIBRATION EXAMPLE 4 ===

# Final Consolidated Review
Summary: This paper identifies and addresses quantization bias in federated learning with LoRA adapters, where clients have heterogeneous quantization levels (e.g., mixing 2-bit and 4-bit models). The authors propose FedQLoRA, which introduces a quantization-aware adapter to separate quantization error from the task-relevant LoRA adapter, and an iterative version (iFedQLoRA) to handle data heterogeneity in non-IID settings. Experiments on text classification tasks demonstrate consistent improvements over baselines including LoRA, FFA-LoRA, and H-LoRA.
Strengths:
- Novel problem identification: The paper identifies quantization bias as an important and previously unaddressed issue in federated LoRA when clients use different quantization levels, supported by compelling empirical evidence in Figure 1 showing that mixed quantization (32.0% accuracy) performs worse than either homogeneous setting (35.9% for all 2-bit, 38.9% for all 4-bit).
- Clear problem decomposition: Equations 5-9 formally decompose the aggregation error into quantization error (endogenous to each client) and quantization bias (arising from heterogeneous quantization levels), providing theoretical insight into why naive aggregation fails.
- Reasonable solution design: The two-adapter approach (quantization-aware adapter for compensation + LoRA adapter for task learning) logically separates the two concerns. The iterative version addresses the additional challenge of data heterogeneity.
- Comprehensive experimental settings: Evaluations cover two datasets (XGLUE NC, 20 NewsGroup), multiple client configurations (3, 5, 10 clients), both IID and non-IID data partitions, and include model heterogeneity analysis, data heterogeneity analysis, and convergence studies.
- Consistent empirical improvements: FedQLoRA and iFedQLoRA outperform baselines across most settings, with iFedQLoRA achieving 2-5% accuracy improvements in non-IID scenarios.
Weaknesses:
- Model scale mismatch: The paper claims to address 'large language models with billions of parameters' but experiments use DistilBERT (~66M parameters). Quantization dynamics differ substantially for billion-parameter models, limiting the validity of conclusions for the claimed target domain.
- Limited quantization scope: Only 2-bit and 4-bit quantization are tested with a 1:1 ratio. Practical deployments use various quantization methods (GPTQ, AWQ, GGUF) and bit-widths (8-bit, 3-bit, etc.). The narrow scope reduces confidence in generalizability.
- No statistical significance reported: Tables 1-2 report single accuracy values without standard deviations or confidence intervals, despite experiments involving randomness from data partitioning and initialization.
- Limited task diversity: Only text classification is evaluated. No experiments on generation tasks, question answering, or other common LLM benchmarks that would better showcase applicability.
- Theoretical assumptions need better justification: Equation 7-8's assumption that the adapter difference equals the weight quantization error is stated without rigorous justification, and the circularity concern in Equation 13 (using locally-trained adapters to estimate the unquantized model) warrants discussion.
- Baseline performance concerns: FFA-LoRA achieves surprisingly poor performance (9.3% accuracy in some settings), which may indicate implementation issues or hyperparameter problems that should be investigated.
Nice-to-haves:
- Experiments with larger LLMs (e.g., LLaMA-7B, Mistral-7B) to validate scalability claims.
- Communication cost analysis quantifying bandwidth savings compared to full model sharing.
- Ablation studies on the rank of the quantization-aware adapter and its impact on performance and overhead.
- Additional NLP tasks beyond classification (e.g., generation, question answering).
- Per-client performance analysis comparing benefits for high-precision vs. low-precision clients.
- Theoretical convergence guarantees for the iterative version iFedQLoRA.
- Computational overhead analysis (training time, memory usage) for the additional quantization-aware adapter.
Novel insights: The core insight that aggregating LoRA adapters from models with different quantization levels introduces quantization bias is novel and practically important. The decomposition of aggregation error into quantization error (client-specific) and quantization bias (arising from heterogeneity) provides useful theoretical structure. The idea of using a separate adapter to compensate for quantization effects while keeping the main LoRA adapter focused on task-relevant learning is conceptually elegant.
Potentially missed related work:
- None
Suggestions:
- Include experiments with at least one billion-parameter model to validate that the method scales as claimed.
- Add error bars or standard deviations to experimental results to enable statistical comparison.
- Expand quantization experiments to include 8-bit and 3-bit levels, and different ratios beyond 1:1.
- Discuss the practical constraint that the method requires knowledge of each client's quantization method for re-quantization, and analyze robustness when this information is unavailable.
- Investigate and explain the poor performance of FFA-LoRA baseline to strengthen confidence in experimental validity.
- Add computational overhead analysis comparing FedQLoRA to baselines in terms of training time and memory.

# Merger Subscores
Novelty: 7.5
Technical soundness: 6.0
Empirical support: 6.0
Significance: 6.5
Clarity: 7.0

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 5.0, 3.0]
Average score: 4.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 5 ===

# Final Consolidated Review
Summary: This paper introduces AdaFM (Adaptive Filtered Momentum), an adaptive variance-reduced algorithm for stochastic minimax optimization. The key innovation is adapting momentum parameters based solely on iteration count (β_{t+1} = 1/t^{2/3}) and learning rates based on cumulative historical estimator information, eliminating the need for problem-dependent parameters like smoothness constants or gradient bounds. The algorithm achieves near-optimal O(ϵ^{-3}) sample complexity for finding ϵ-stationary points in both non-convex-strongly-concave (NC-SC) and non-convex-Polyak-Łojasiewicz (NC-PL) settings. Experiments validate the method on synthetic test functions, deep AUC maximization, and WGAN-GP training.
Strengths:
- **Addresses a practical gap in variance-reduced minimax methods**: The paper correctly identifies that existing VR algorithms require tuning numerous problem-dependent hyperparameters (learning rates for both primal and dual variables, momentum parameters), which limits practical adoption. Figure 1 effectively demonstrates this sensitivity using RSGDA on CIFAR-10/100.
- **Novel algorithmic design for primal-dual interaction**: The design choice of η_t^x = (max{α_t^x, α_t^y})^{1/3+δ} (Equation 4) ensures the primal variable updates cautiously when the inner maximization is unresolved—a principled mechanism for handling two-timescale dynamics in minimax optimization.
- **Strong theoretical contributions**: The paper provides rigorous convergence analysis for both NC-SC (Theorem 1: O(κ^{4.5}/T^{1/3+δ})) and NC-PL (Theorem 2: O(κ^5/T^{1/3+δ})) settings. The proofs properly handle a four-case analysis for bounding gradient and error terms, matching best-known rates among parametric algorithms.
- **Comprehensive experimental evaluation**: The method is validated across three diverse tasks with different problem structures: synthetic test functions, deep AUC maximization (NC-SC), and WGAN-GP training (NC-PL). The experiments demonstrate consistent improvements over RSGDA, VRAdaGDA, and TiAda.
- **Useful ablation study**: Section A.4 provides analysis of how δ affects convergence, revealing the trade-off between faster stepsize ratio adjustment and overall convergence rate.
Weaknesses:
- **'Parameter-free' claim is overstated**: The algorithm requires three hyperparameters (γ, λ, δ). While the theory states γ = λ = 1 and δ → 0 work, the ablation (Figure 9) shows δ = 0 causes convergence failure, and Figure 10 shows δ significantly impacts WGAN-GP performance. The terminology should be revised to 'adaptive with minimal tuning' or 'reduced-hyperparameter.'
- **Theory-practice gap regarding δ**: The theoretical analysis assumes δ can be 'arbitrarily small,' but experiments require δ ≈ 0.001–0.1 for stability. This discrepancy between theoretical assumptions and practical requirements needs reconciliation—either through modified theory or clearer practical guidance.
- **Limited empirical rigor**: No error bars or statistical significance are reported across multiple runs for any experiment. This makes it difficult to assess the reliability of the reported improvements.
- **Missing standard baseline for GAN experiments**: The WGAN-GP experiments compare against RSGDA, VRAdaGDA, and TiAda but exclude Adam/AdamW, which are standard practice for GAN training. Including these would strengthen the practical utility argument.
- **Incomplete analysis of hyperparameter robustness**: The ablation focuses on δ but provides limited analysis of γ and λ sensitivity across different tasks. A more comprehensive sensitivity study would better support the claim of minimal tuning requirements.
Nice-to-haves:
- Comparison with Adam/AdamW for WGAN-GP experiments, since these are standard baselines for GAN training.
- FID scores alongside Inception Score for GAN evaluation.
- Error bars across multiple random seeds for all experiments.
- Wall-clock time comparison to assess computational overhead of maintaining momentum estimators.
Novel insights: The key insight—that learning rates for primal and dual variables should be coupled through their cumulative estimator magnitudes to ensure proper two-timescale dynamics without manual tuning—is a valuable contribution. The max{α_t^x, α_t^y} mechanism elegantly ensures that the primal update slows when the inner maximization is unresolved, addressing a fundamental challenge in adaptive minimax methods.
Potentially missed related work:
- **PES (Guo et al., 2023)**: Cited in related work but not experimentally compared. PES addresses NC-PL objectives and could serve as an additional baseline.
- **Adaptive methods for adversarial training**: Works like Madry et al.'s adversarial training framework could be mentioned as a prominent minimax application, though the paper already covers diverse applications.
Suggestions:
- Revise terminology from 'parameter-free' to 'adaptive' or 'reduced-hyperparameter' throughout the paper, and explicitly acknowledge the three hyperparameters (γ, λ, δ) with recommended defaults.
- Address the δ tension: either modify the theory to handle δ = 0, or provide clearer guidance on selecting δ in practice (e.g., δ = 0.001 as a safe default).
- Add error bars across multiple runs for all experiments to improve empirical rigor.
- Include Adam as a baseline in WGAN-GP experiments, since it is standard practice for GAN training.
- Expand the ablation study to include γ and λ sensitivity to demonstrate robustness across tasks.

# Merger Subscores
Novelty: 7.5
Technical soundness: 7.0
Empirical support: 6.5
Significance: 7.0
Clarity: 7.5

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 8.0, 6.0]
Average score: 5.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 6 ===

# Final Consolidated Review
Summary: The paper proposes Swift-FedGNN, a federated graph neural network training framework for geo-distributed graph data. The key innovation is having clients primarily perform local training using only local graph data, while periodically conducting cross-client training on a subset of randomly sampled clients to incorporate cross-client neighbor information. The paper provides theoretical convergence analysis proving an O(T^{-1/2}) rate and validates the approach on ogbn-products and Reddit datasets.
Strengths:
- Well-motivated problem with concrete use case (hospital networks) and empirical evidence showing that cross-client training is 5× slower than local training (Figure 2), making a compelling case for the proposed solution.
- Novel algorithmic design: The periodic cross-client training strategy (Algorithms 1-3) with double aggregation at remote clients and server is a sensible approach to balance accuracy-efficiency tradeoffs, reducing both communication overhead (Table 2 shows 4-20× reduction) and memory requirements.
- Theoretical contributions: The convergence analysis (Theorem 5.6) provides bounds on stochastic gradient approximation errors caused by both neighbor sampling and missing cross-client neighbors (Lemmas 5.4-5.5), without resorting to unrealistic assumptions like unbiased or consistent gradient estimators used in prior work.
- Empirical validation on ogbn-products (the largest dataset in federated GNN literature) demonstrates that Swift-FedGNN achieves comparable accuracy to full cross-client training (87.73% vs 87.93%) while significantly reducing wall-clock training time.
- Comprehensive hyperparameter sensitivity analysis (Figure 6) provides useful insights for correction frequency I, number of cross-client training clients K, and sampling fanouts.
Weaknesses:
- No statistical significance reporting: The experiments lack error bars, confidence intervals, or specification of the number of runs, making it difficult to assess whether reported accuracy differences are meaningful (e.g., 87.73% vs 87.93% on ogbn-products).
- Limited architectural diversity: Only GraphSAGE is tested experimentally, while the theoretical analysis assumes GCN architecture. The method also relies on element-wise aggregation operations, limiting support for attention-based GNNs like GAT (acknowledged in Footnote 2).
- Idealized experimental setup: Experiments simulate federated learning on a single machine using shared memory with simulated 1Gbps bandwidth. Real federated settings involve network latency, client heterogeneity, device failures, and stragglers—none of which are evaluated.
- No formal privacy guarantees: The paper claims that 'aggregated embeddings help preserve data privacy' but provides no formal privacy analysis (e.g., differential privacy guarantees). The privacy benefits are stated informally without substantiation.
- Missing non-IID analysis: The paper uses METIS partitioning which creates balanced partitions but does not analyze performance under heterogeneous (non-IID) data distributions, which are fundamental challenges in federated learning.
Nice-to-haves:
- The theoretical insight about positive correlation between gradient error and number of layers (Section 5) is novel and interesting, but empirical validation across different GNN depths would strengthen this contribution.
- Ablation study on the two-level aggregation strategy would help isolate the benefits of this design choice versus simpler alternatives.
- Testing on heterophilic graphs or graphs with different structural properties would demonstrate robustness beyond the two homophilic benchmark datasets.
- Analysis of server scalability as the number of clients grows would address potential deployment concerns.
- Formal privacy analysis or empirical privacy leakage evaluation would strengthen the privacy claims.
Novel insights: The paper provides a new theoretical insight that stochastic gradient approximation errors in GNNs are positively correlated with the number of layers, which arises from the structural entanglement of neighbor aggregation and non-linear transformations. This finding is unique to federated GNN training and could inform future algorithm design.
Potentially missed related work:
- Recent federated GNN surveys (2023-2024) could strengthen the positioning and provide readers with broader context.
- FedGraphNN (He et al., 2021a) and SpreadGNN (He et al., 2021b) are mentioned in related work but not included as experimental baselines, which could provide a more complete empirical comparison.
Suggestions:
- Add error bars and report the number of experimental runs to enable readers to assess statistical significance of reported results.
- Consider testing GCN explicitly to bridge the gap between theoretical analysis and empirical evaluation.
- Include experiments under more realistic federated conditions (varying bandwidth, client heterogeneity, straggler effects) to validate practical applicability.
- Add a dedicated limitations section discussing: the element-wise operation constraint, lack of formal privacy guarantees, and the residual error from infrequent cross-client training.
- Consider formal privacy analysis (even basic differential privacy bounds) to substantiate privacy claims, or clearly state the privacy limitations of aggregated embeddings.

# Merger Subscores
Novelty: 7.0
Technical soundness: 7.5
Empirical support: 6.5
Significance: 7.0
Clarity: 8.0

# Actual Human Scores
Individual reviewer scores: [3.0, 6.0, 5.0, 5.0]
Average score: 4.8
Binary outcome: Reject

=== CALIBRATION EXAMPLE 7 ===

# Final Consolidated Review
Summary: This paper proposes HiCA (Hierarchical prompts with Context-Aware calibration) for open-vocabulary object detection. The method introduces two main components: (1) hierarchical prompts that decompose region-category mapping into a two-step process using coarse-grained superclass knowledge as an intermediary, and (2) context-aware calibration that leverages visual context through unsupervised clustering to calibrate detection logits. The approach is evaluated on OV-COCO and OV-LVIS benchmarks, demonstrating consistent improvements when applied to multiple baselines (OADP and BARON).
Strengths:
- Strong empirical results: The method achieves 57.2% mAP_B on OV-COCO (surpassing reproduced OADP baseline by 5.5%) and 50.4% mAP50 overall. On OV-LVIS, HiCA improves over the OADP baseline by 4.6% AP, demonstrating consistent gains across benchmarks.
- Well-motivated problem formulation: The paper correctly identifies that existing OVD methods overfit to base classes by directly mapping regions to categories, neglecting shared superclass knowledge that could benefit novel class generalization.
- Comprehensive ablation studies: Tables 3, 4, and 5 systematically evaluate each component (hierarchical prompts, context-aware calibration, different prompt types, balance parameter λ, clustering settings). The visualizations in Figure 4 and the appendix provide qualitative support for the claims.
- Plug-and-play flexibility: The authors demonstrate that both hierarchical prompts and context-aware calibration can be applied to multiple baselines (OADP and BARON), supporting the claim of framework-agnostic design.
- Principled treatment of visual context: Unlike prior work that treats context purely as negative examples, this paper leverages context-superclass distributions to calibrate predictions, representing a more nuanced approach to contextual information.
Weaknesses:
- Insufficient explanation of superclass taxonomy: The paper does not clearly explain how superclasses are defined or obtained. The subordinate matrix A maps categories to superclasses, but the paper lacks details on whether this taxonomy is manually constructed, derived from external knowledge bases (e.g., WordNet), or learned. This is critical for reproducibility and for understanding how novel classes (unavailable during training) are mapped to superclasses at inference time.
- Context selection mechanism underspecified: Equation 7 applies the context-aware matrix via Hadamard product, but the paper does not clearly explain how the specific context index k is selected for a given image during inference. Section 3.5 mentions 'selecting the context-aware vector based on the context to which the current image belongs' without specifying the selection procedure.
- Limited analysis of novel class generalization: While the paper claims improved generalization to novel classes, the gains are modest (1.3% mAP_N improvement on OV-COCO). Figure 5 in the appendix shows that novel classes (indices 48-53) do not achieve maximal diagonal similarity, suggesting the method still struggles with novel class detection. Deeper analysis of why superclass knowledge transfers (or fails to transfer) would strengthen the contribution.
- No statistical significance reporting: The paper reports single-run results without error bars or confidence intervals, making it difficult to assess whether improvements (e.g., 31.2% vs 29.9% mAP_N) are statistically meaningful.
- Undefined notation: The visual prompt operation ⊕ in Equation 4 is not explicitly defined, creating ambiguity for reproducibility.
Nice-to-haves:
- Cross-dataset generalization evaluation (e.g., zero-shot transfer from OV-COCO to OV-LVIS) would better demonstrate open-vocabulary capability.
- Computational cost analysis (training time, inference speed, memory) would help practitioners understand practical trade-offs.
- Per-class performance breakdown showing which specific classes benefit most from hierarchical prompts would provide deeper insight.
- Evaluation with stronger VLM backbones (ViT-L/14, ViT-H/14) would demonstrate scalability.
- Analysis of what the K context clusters semantically represent (e.g., indoor vs. outdoor scenes) would clarify the mechanism.
Novel insights: The key insight is that directly mapping visual regions to fine-grained category prompts leads to overfitting on base classes. By introducing superclass-level prompts as an intermediary, the model learns coarse-grained visual-linguistic alignments that transfer better to novel classes sharing the same superclass. The context-aware calibration further leverages scene-level priors to adjust predictions based on environmental context. While individually these ideas are not revolutionary, their combination and application to OVD represents a meaningful contribution.
Potentially missed related work:
- HierKD (Ma et al., 2022) appears in Table 1 but is not discussed in Related Work despite sharing the hierarchical knowledge distillation concept—comparing approaches would clarify differences.
- Recent foundation models like GLIP, Grounding DINO, and FIBER represent alternative approaches to open-vocabulary detection that could contextualize this work's positioning.
- Hierarchical classification literature (e.g., Deng et al., ICML 2010; Zhu & Bain, NeurIPS 2017) provides theoretical grounding for why hierarchical structures improve generalization that could strengthen the paper.
Suggestions:
- Add explicit details on superclass construction: Include a table or appendix showing the superclass taxonomy used for each dataset and the mapping procedure for novel classes at inference.
- Clarify the context selection mechanism in Equation 7—specify how context index k is determined for each image.
- Report results with error bars over multiple runs to establish statistical significance.
- Add a limitations section acknowledging constraints: dependence on predefined superclass taxonomy, assumption that base class context distributions transfer to novel classes, and hyperparameter sensitivity.
- Define the ⊕ operation explicitly in the notation section.

# Merger Subscores
Novelty: 6.5
Technical soundness: 6.0
Empirical support: 7.0
Significance: 6.5
Clarity: 6.0

# Actual Human Scores
Individual reviewer scores: [5.0, 5.0, 5.0, 5.0]
Average score: 5.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 8 ===

# Final Consolidated Review
Summary: This paper investigates whether labels can be safely ignored in out-of-distribution (OOD) detection. The authors provide a theoretical proof showing that self-supervised and unsupervised OOD detection methods will fail when the surrogate learning objective is independent of the labels needed for OOD detection—a condition they term 'label blindness.' They introduce a novel 'Adjacent OOD' benchmark where ID and OOD data share significant feature overlap, and empirically demonstrate that existing unlabeled OOD methods perform poorly under these conditions while supervised baselines remain effective.
Strengths:
- **Theoretical framework**: The Label Blindness Theorem (Corollary 3.3) provides rigorous information-theoretic proofs for when unlabeled OOD detection is guaranteed to fail. The proofs in Appendix D are detailed and correctly apply information bottleneck principles to OOD detection failure conditions.
- **Novel benchmark contribution**: The Adjacent OOD detection task addresses a genuine safety gap not captured by existing near/far OOD benchmarks. Theorem 4.1 proves that overlapping ID/OOD data is unavoidable in real-world systems, making this benchmark highly relevant for safety-critical applications.
- **Comprehensive empirical validation**: Experiments span multiple datasets (Faces, Cars, Food-101, CIFAR10/100), multiple methods (SimCLR, Rotation Loss, Diffusion inpainting, CLIPN), and multiple scoring mechanisms (KNN, SSD). Results consistently show SSL methods performing near random-guess level on Adjacent OOD while supervised MSP achieves substantially higher AUROC.
- **Important practical implications**: The paper directly challenges the growing body of work claiming unlabeled OOD detection can replace supervised methods, providing both theoretical justification and empirical evidence. Section 7.2 appropriately acknowledges scenarios where unlabeled OOD remains useful.
- **Scientific integrity**: The authors include Appendix F.1 showing better SSL performance on CIFAR10/100 Adjacent OOD and Appendix F.2 confirming strong far-OOD performance, demonstrating appropriate reporting of cases where SSL methods succeed.
Weaknesses:
- **Strong theoretical assumptions with incomplete bridge to practice**: The strict label blindness condition requires exact independence I(x₁; x₂) = 0, which rarely holds for real-world images. The extension to 'approximate label blindness' via Fano's inequality is mentioned but not rigorously developed—readers cannot precisely quantify when approximately zero mutual information becomes problematic in practice.
- **Limited SSL method coverage**: Experiments focus primarily on SimCLR and Rotation Loss, omitting other prominent approaches like MAE, DINO, MoCo, or BYOL. These methods have different inductive biases that may affect label-relevant feature retention differently.
- **Modest supervised baseline**: Table 1 shows supervised MSP achieves AUROC of ~70% on the Adjacent OOD tasks, which is itself modest. Comparison against stronger supervised baselines (e.g., ODIN, Energy-based detection, Deep Mahalanobis) would strengthen the claim that labels are necessary.
- **Partial success of SSL methods under-analyzed**: Table 1 shows SSL methods achieving 52-65% AUROC on Cars/Food—above random but below supervised. The paper does not analyze what features enable this partial success or what determines when SSL objectives happen to align with label-relevant features.
- **No empirical exploration of mitigation strategies**: Section 7.1 briefly mentions that 'few or one shot methods' could help overcome approximate label blindness, but provides no experiments testing how much label information (e.g., 1%, 5%, 10% labeled data) suffices to avoid failure.
Nice-to-haves:
- Testing additional SSL baselines (MoCo, BYOL, DINO, MAE) would clarify whether label blindness is general across SSL paradigms or specific to certain objectives.
- Experiments varying the degree of ID/OOD overlap (beyond the 75%/25% split) would help characterize how detection performance degrades as overlap increases.
- Quantitative comparison of feature overlap between Adjacent OOD and existing near OOD benchmarks would substantiate the claim that current benchmarks miss this failure mode.
- Semi-supervised or few-shot experiments would provide practical guidance on how much label information suffices to avoid label blindness.
Novel insights: The Adjacent OOD benchmark concept is genuinely novel and addresses a critical safety scenario: when OOD inputs share significant visual features with ID data (e.g., same face with different expression, same car model with different year). The theoretical insight that minimal sufficient statistics for an independent surrogate task cannot encode label information correctly explains why SSL methods trained only on reconstruction or contrastive objectives may fail to distinguish semantically distinct but visually similar classes.
Potentially missed related work:
- The paper does not cite recent work on feature memorization in SSL (e.g., Jing et al., 'Understanding Dimensional Collapse in SSL') which may be relevant to understanding when SSL representations retain label-relevant information.
- Multi-view information bottleneck theory (Federici et al., 2020) is cited but the deeper connection between multi-view assumptions and label blindness could be further explored.
Suggestions:
- Develop the approximate label blindness case more rigorously—providing theoretical bounds or empirical thresholds for how small I(x;y) must be before OOD detection performance degrades significantly would increase practical impact.
- Add stronger supervised baselines (ODIN, Energy, Deep Mahalanobis) to more convincingly demonstrate the performance gap between labeled and unlabeled approaches.
- Include representation analysis experiments estimating mutual information between learned representations and labels to empirically validate theoretical predictions.
- Analyze why SSL methods show partial success on Cars/Food (52-65% AUROC)—understanding what features are captured could inform better SSL-OOD designs.

# Merger Subscores
Novelty: 7.0
Technical soundness: 7.5
Empirical support: 7.0
Significance: 8.0
Clarity: 7.5

# Actual Human Scores
Individual reviewer scores: [6.0, 5.0, 8.0]
Average score: 6.3
Binary outcome: Accept

=== CALIBRATION EXAMPLE 9 ===

# Final Consolidated Review
Summary: The paper presents LocoVR, a VR-based dataset of 7,000+ two-person trajectories collected across 131 diverse indoor home environments. The dataset addresses a genuine gap in the literature by combining large-scale scene diversity with social navigation dynamics (proxemics-based behaviors). The authors demonstrate the dataset's utility through three tasks—global path prediction, trajectory prediction, and goal prediction—showing that models trained on LocoVR outperform those trained on existing indoor datasets when tested on real-world data.
Strengths:
- Significant scale and scene diversity: With 131 scenes and 7,000+ trajectories, LocoVR substantially exceeds existing indoor motion datasets (GIMO has 19 scenes, THOR-MAGNI has 4 scenes). The ablation study (Tables 5-7) confirms that dataset scale contributes meaningfully to performance gains.
- Novel data collection methodology: The VR-based approach effectively addresses practical challenges of indoor data collection—camera obstructions and high costs of 3D scanning—while providing accurate spatial information and trajectory data. The system captures both geometric constraints and social motion behaviors.
- Focus on social navigation dynamics: Unlike most datasets that focus on single-person motion or outdoor crowd dynamics, LocoVR captures two-person proxemics-based behaviors. Approximately 70% of trajectories involve participants coming within 2m of each other (Figure 16), providing rich social interaction data.
- Real-world transfer validation: The authors created LocoReal (450 trajectories in 4 physical room layouts) to test VR-to-real transfer, demonstrating that models trained on LocoVR outperform those trained on physically-collected datasets (GIMO, THOR-MAGNI).
- Comprehensive experimental evaluation: Three distinct tasks with appropriate baselines, ablation studies isolating contributions of dataset scale, multi-person data, and heading direction, and testing on both VR and real-world data.
Weaknesses:
- Limited participant diversity: Only 32 participants (21 male, 11 female), all aged 18-42, all presumably able-bodied (Appendix I.2). This homogeneity restricts generalizability to broader populations. While acknowledged in Appendix A, the demographic narrowness is significant for a dataset claiming to model 'human locomotion' and proxemics behaviors that vary with cultural background.
- VR-real gap validation limited by test set size: LocoReal contains only 450 trajectories across 4 scenes, which constrains the strength of real-world transfer claims. The paper acknowledges the VR-real gap (Section 5) but the analysis is limited.
- Baseline model coverage limited: For trajectory prediction, only U-Net and YNet are compared. Modern approaches like transformer-based models (e.g., Trajectron++, Social-STGCNN) or diffusion-based models are not included. The paper claims that existing models 'compress geometric features into small sets' but does not empirically validate this assertion.
- Insufficient quantification of social behaviors: While the paper claims rich social navigation dynamics, systematic annotation or classification of specific social behaviors (yielding, passing side preferences, collision avoidance patterns) is absent. The appendix shows minimum distance distributions but lacks behavioral taxonomy.
Nice-to-haves:
- The ablation study could include a more controlled scene-count comparison (e.g., training on 19 random LocoVR scenes vs. GIMO with matched scene counts) to isolate the contribution of data quality from quantity.
- Quantitative analysis of systematic differences between VR and real trajectories (walking speed distributions, turning behaviors, hesitation patterns) would help users understand transfer limitations.
- Annotation of specific social interaction types (passing, yielding, following, avoidance) would enable more targeted research on social navigation.
- The participant pool could include wider age ranges and mobility profiles to strengthen claims about modeling general human locomotion.
Novel insights: The paper makes a genuine contribution by addressing the intersection of three gaps: (1) the lack of large-scale indoor trajectory datasets with diverse scene geometries, (2) the absence of social navigation dynamics in existing indoor datasets, and (3) the challenge of collecting accurate spatial information alongside trajectory data. The VR-based methodology is well-motivated and the demonstration of real-world transfer—while limited by test set size—provides evidence that VR-collected data can be practically useful.
Potentially missed related work:
- The authors may want to consider discussing or comparing against transformer-based trajectory prediction methods (e.g., Trajectron++, Social-STGCNN, AgentFormer) which have become standard baselines for trajectory prediction.
- Recent work on diffusion-based trajectory generation (e.g., MID, SceneDiffuser) could provide additional relevant baselines for the trajectory prediction task.
Suggestions:
- Expand baseline comparisons to include modern transformer-based and diffusion-based trajectory prediction models.
- Provide quantitative analysis of social behavior types observed in the dataset, beyond proximity statistics.
- Acknowledge more prominently the limited demographic diversity and its implications for proxemics modeling given the cultural factors mentioned in the introduction.

# Merger Subscores
Novelty: 7.0
Technical soundness: 7.5
Empirical support: 7.0
Significance: 7.5
Clarity: 8.0

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0, 6.0, 3.0]
Average score: 5.8
Binary outcome: Accept

=== CALIBRATION EXAMPLE 10 ===

# Final Consolidated Review
Summary: The paper proposes Fine-tuned Score Deviation (FSD), a method for detecting whether a text was included in an LLM's pretraining data. The key insight is that fine-tuning on a small set of non-member data causes larger perplexity decreases for non-members than for members, creating a detectable gap. FSD measures the score deviation between pre-trained and fine-tuned models to improve upon existing scoring functions like Perplexity and Min-k%.
Strengths:
- **Strong and consistent empirical improvements**: FSD achieves substantial AUC gains across all tested scenarios. For example, on WikiMIA with OPT-6.7B, Min-k% improves from 0.62 to 0.91 AUC; on ArXivTection with LLaMA-7B, Perplexity improves from 0.68 to 0.92 AUC (Table 1).
- **Novel and practical insight**: The observation that fine-tuning on non-members differentially affects members vs. non-members is intuitive yet previously unexplored for this task. The approach leverages readily available data (content published after model release), making it practically applicable.
- **Comprehensive experimental validation**: The authors test across 7 models (LLaMA-7B/13B/30B, OPT-6.7B, GPT-J-6B, Pythia-6.9B, NeoX-20B), 5 benchmark datasets (WikiMIA, ArXivTection, BookMIA, BookTection, Pile), and 4 baseline scoring functions.
- **Thorough ablation studies**: The paper investigates the impact of fine-tuning data size (Figure 4), different fine-tuning methods (Table 7), model sizes (Table 3), and addresses temporal distribution shift concerns (Table 6).
- **Data efficiency**: The method works well with very few non-members—as few as 100 examples yield dramatic improvements (Figure 4), enhancing practical utility.
Weaknesses:
- **Limited theoretical grounding**: The paper empirically observes that fine-tuning on non-members decreases scores for other non-members more than for members, but provides no formal or theoretical explanation. Analysis connecting this to memorization dynamics, gradient behavior, or distribution shift would substantially strengthen the contribution.
- **No statistical significance reporting**: All AUC and TPR@5%FPR results are reported as point estimates without error bars, confidence intervals, or significance tests across multiple random seeds. Given the stochastic nature of fine-tuning, this omission reduces confidence in the reported improvements.
- **Temporal shift in benchmarks partially inflates results**: The authors acknowledge that WikiMIA suffers from temporal differences between members (older text) and non-members (newer text). While they attempt mitigation via timestamp deletion/replacement (Table 6), performance degrades substantially (e.g., Perplexity AUC drops from 0.92 to 0.71 with replacement), suggesting the original benchmark results may be somewhat optimistic.
- **Domain-specific requirement limits applicability**: FSD requires collecting non-member data from the same domain as the query text. Table 16 shows that when fine-tuning on data from an unrelated domain (e.g., WikiMIA data for detecting ArXivTection members), the method fails to improve over baselines. This constrains practical deployment scenarios.
- **Incomplete baseline comparisons**: The paper compares with Perplexity, Lowercase, Zlib, and Min-k%, but does not empirically compare with more recent methods like RECALL (Xie et al., 2024) or neighborhood-based attacks (Mattern et al., 2023), which are cited but not tested.
- **Fine-tuning data construction lacks realism**: The experimental setup samples 30% of the benchmark dataset to construct the fine-tuning set (Section 4.1), rather than independently collecting non-member data. This may not reflect realistic scenarios where one must gather domain-specific data without access to the benchmark's non-member pool.
Nice-to-haves:
- Consider adding comparisons with more recent MIA methods (RECALL, Min-k%++, neighborhood-based approaches) to strengthen claims of improvement over state-of-the-art.
- Provide theoretical analysis of why fine-tuning creates asymmetric score shifts, potentially connecting to gradient dynamics or information-theoretic perspectives.
- Include computational cost analysis (fine-tuning time, memory requirements) to enable practical assessment.
- Test on more recent open-source models (LLaMA-2, LLaMA-3, Mistral) to demonstrate continued relevance.
- Analyze failure cases systematically—some Pile subsets in Table 15 show marginal gains or occasional base method superiority, which deserves discussion.
- Consider reporting results with bootstrap confidence intervals across multiple random seeds.
Novel insights: The core insight—that exposing an LLM to a few non-member examples via fine-tuning creates measurable asymmetry in how scores change for members vs. non-members—is novel and previously unexplored. This transforms the detection problem from comparing absolute score values to comparing relative score changes, which is more robust to confounding factors like text frequency or repetitiveness.
Potentially missed related work:
- None
Suggestions:
- The authors should consider systematically analyzing what dataset characteristics predict FSD effectiveness (e.g., comparing strong results on WikiMIA vs. weaker results on some Pile subsets).
- Developing adaptive threshold selection that does not require labeled validation data would enhance real-world deployability.

# Merger Subscores
Novelty: 7.0
Technical soundness: 7.0
Empirical support: 7.5
Significance: 7.0
Clarity: 8.0

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 6.0, 5.0]
Average score: 6.2
Binary outcome: Accept

=== CALIBRATION EXAMPLE 11 ===

# Final Consolidated Review
Summary: This paper introduces Process Advantage Verifiers (PAVs), which define process rewards as the 'progress' (advantage) made by a reasoning step under a 'prover' policy distinct from the base policy. The key insight is that advantages under complementary provers better distinguish good and bad steps than Q-values or advantages computed under the base policy itself. The authors provide theoretical characterization of good provers (Theorem 3.1) and demonstrate empirically that PAVs achieve 1.5-5× compute efficiency gains in test-time beam search and 6× sample efficiency gains in online RL compared to outcome reward models—a substantial improvement over prior PRM work showing only 1-2% gains.
Strengths:
- **Strong theoretical foundation**: Theorem 3.1 formally characterizes how prover policies should balance 'distinguishability' (variance in advantages under the base policy) and 'alignment' with the base policy. This provides principled guidance for prover selection, grounding the empirical observations in theory.
- **Significant empirical improvements**: The 6× sample efficiency gain in online RL (Figure 7) directly addresses a key limitation in prior work where PRMs provided only marginal improvements (Shao et al., 2024 reported 1-2%). The 1.5-5× compute efficiency gains for test-time search (Figure 4) are practically meaningful.
- **Counter-intuitive insight with theoretical support**: The finding that weaker provers can improve stronger base policies (e.g., Gemma 2B prover helping Gemma 9B base in Figure 19; 9B prover helping 27B base in Figure 5c) challenges intuitive assumptions and is well-supported by the theoretical framework in Proposition F.1.
- **Clear motivation for advantages over Q-values**: The beam search example in Figure 2a effectively illustrates why Q-values conflate state promise with action quality, while advantages properly isolate step-level progress. Appendix G demonstrates the failure mode of using Q^μ instead of A^μ, showing degenerate 'REPHRASE THE PROBLEM' solutions.
- **Comprehensive empirical validation**: Results are consistent across three model scales (2B, 9B, 27B), and the paper includes systematic ablations on prover choice (Figure 5b-c), comparisons with prior PRM approaches (Figure 12), and appropriate baselines (ORM-based best-of-N, PRM with Q-values from Snell et al.).
Weaknesses:
- **Limited task diversity**: All experiments are conducted on the MATH dataset. No evaluation on other reasoning domains (code generation, multi-hop QA, logical reasoning) is provided to assess whether the approach generalizes beyond mathematical problem-solving. This limits confidence in broader applicability.
- **Hyperparameter sensitivity without principled tuning method**: The α weight requires substantially different values across settings (α=0.5 for 2B/9B vs α=0.2 for 27B in search; α=5.0 for 2B vs α=3.0 for 9B in RL). While Appendix E describes a manual tuning procedure on validation sets, no automated or principled method is provided for setting this critical hyperparameter in new domains.
- **Prover selection remains heuristic**: Despite Theorem 3.1 providing qualitative guidance (distinguishability + alignment), practical prover selection defaults to BoK(π) with K=4, chosen empirically. The paper acknowledges this limitation (Appendix J) but does not provide a constructive procedure for identifying optimal provers for new settings.
- **Limited failure mode analysis**: The paper focuses on aggregate improvements but provides minimal analysis of problem types where PAVs fail to help, or when the 'complementary prover' assumption breaks down. Understanding these boundaries would strengthen practical applicability.
- **Theory-to-practice gap**: Theorem 3.1 assumes oracle access to Q^π and A^μ values, softmax parameterization, and tabular MDP. How prediction errors from learned PAVs compound during RL remains unanalyzed, limiting the direct applicability of the theoretical guarantees.
Nice-to-haves:
- Evaluation on additional reasoning benchmarks beyond MATH (e.g., GSM8K, GPQA, code generation) would strengthen claims about broad applicability.
- Analysis of failure cases: a characterization of problem types or settings where PAVs fail to improve over ORMs would help practitioners understand when to apply this approach.
- Comparison with human-annotated PRMs (e.g., PRM800K from Lightman et al.) would clarify how PAVs perform relative to the 'gold standard' of human process supervision.
- Cross-model-family experiments (beyond Gemma) would strengthen the claim that PAVs are a general approach.
- A principled method for setting α based on observable quantities (e.g., advantage variance, policy entropy) would reduce the tuning burden for new applications.
Novel insights: The conceptual reframing of process rewards from 'correctness' to 'progress' is genuinely insightful. The realization that a prover policy distinct from the base policy is necessary—because advantages under the base policy provide no new signal—is elegant and well-justified. The formal characterization of 'complementary provers' through the lens of variance and alignment provides a useful theoretical lens for thinking about process supervision that could influence future work on multi-agent or curriculum-based training methods.
Potentially missed related work:
- The paper compares against automated PRMs but could acknowledge concurrent work on tree search with process supervision (e.g., MCTS-based approaches) more explicitly.
- The connection to intrinsic motivation and exploration bonuses in RL could be discussed—the advantage-based reward resembles count-based or prediction-error-based exploration methods.
- Recent work on self-generated process supervision (beyond the cited Math-Shepherd and PRM800K) may be relevant for context.
Suggestions:
- Include at least one non-math reasoning benchmark to demonstrate generalization of the approach.
- Provide qualitative examples showing actual MATH problems with step-by-step PAV scores to help readers build intuition about what 'progress' looks like in practice.
- Consider developing an adaptive prover selection mechanism that adjusts K during RL training as the base policy improves—this natural extension is hinted at in limitations but not explored.
- Explicitly commit to releasing code and trained PAV checkpoints to enable reproduction and community extension.

# Merger Subscores
Novelty: 7.8
Technical soundness: 7.5
Empirical support: 7.2
Significance: 7.6
Clarity: 7.3

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 8.0, 8.0, 6.0]
Average score: 7.1
Binary outcome: Accept

=== CALIBRATION EXAMPLE 12 ===

# Final Consolidated Review
Summary: This paper introduces the novel task of audio difference explanation—generating natural language descriptions that explain differences between two audio recordings. The authors create two benchmark datasets (ACD and CLD) derived from AudioCaps and Clotho with three tiers of explanation granularity (concise, brief, detailed), propose ADIFF (a prefix-tuning model with cross-projection, separator token, and three-stage training), and demonstrate improvements over naive baselines and Qwen-Audio through extensive experiments and ablations.
Strengths:
- Novel and well-motivated task formulation: Audio difference explanation addresses a meaningful gap in audio-language modeling with clear applications in forensics, quality assessment, and generation evaluation. The three-tier framework (concise, brief, detailed) enables fine-grained evaluation of model capabilities.
- Comprehensive benchmark creation: The ACD (~48k train) and CLD (~19k train) datasets are derived from established audio captioning datasets with human verification on test sets. Dataset statistics, vocabulary analysis, and entropy measurements provide thorough characterization.
- Extensive and well-designed ablation studies: The paper systematically examines baseline architecture, cross-projection effects, language model scaling (128M to 1.5B parameters), position captioning, and stage-3 finetuning. The finding that smaller LMs perform comparably or better under limited compute is practically valuable.
- Strong empirical results: ADIFF outperforms both naive baselines and QwenAudio (7B parameters) on most metrics across all three tiers, despite using only a 128M parameter GPT-2. Human evaluation across three domains confirms improvements in correctness, granularity, and readability.
- Insightful cross-projection analysis: Appendix J provides mechanistic understanding showing how cross-projection repurposes text prefixes to store difference attributes, explaining the architectural contribution.
- Practical hallucination detection mechanism: By keeping the HTSAT encoder frozen, the paper enables users to cross-reference generated descriptions against audio event probabilities, addressing a real concern in audio-language models.
Weaknesses:
- LLM-generated training data with limited quality analysis: The training data is entirely generated by LLMs using human captions as prompts, with only test sets receiving human verification. The paper does not analyze the quality of the training explanations or compare LLM-generated vs. human-authored training data, leaving open questions about cascading errors and distributional biases.
- Lack of statistical rigor: No error bars, confidence intervals, or statistical significance tests are reported for any quantitative results. For a paper with extensive multi-way comparisons across tiers, datasets, and model variants, this omission limits confidence in reported improvements.
- Limited analysis of failure modes and edge cases: The paper presents successful qualitative examples but lacks systematic error analysis. Importantly, the model's behavior on identical/similar audio pairs (where no difference exists) is not evaluated—a critical capability for practical applications.
- Audio ordering invariance not tested: Swapping audio 1 and audio 2 would reveal whether the model learns symmetric difference understanding or relies on position-specific patterns. This is essential for verifying robust comparative reasoning.
- Single audio encoder used throughout: All experiments rely on HTSAT as a frozen encoder. Testing with alternative encoders (CLAP, Whisper) would demonstrate whether findings generalize across representations.
- Inconsistent Tier 2 CLD performance under-explained: QwenAudio (F) substantially outperforms ADIFF on Tier 2 CLD (SPIDEr: 0.958 vs 0.692). The paper attributes this to linguistic simplicity and larger LLM, but this explanation is not rigorously tested and the discrepancy warrants deeper investigation.
- Human evaluation details incomplete: Only 5 annotators evaluate models across three scenarios, with no inter-annotator agreement reported. This limits statistical confidence in subjective evaluation conclusions.
Nice-to-haves:
- SUGGESTION: Consider evaluating on curated subsets of perceptually similar sounds (same events, different acoustic conditions) to better characterize discriminative capabilities.
- SUGGESTION: A factorial ablation separating the separator token contribution from the cross-projection layer would clarify each component's role.
- SUGGESTION: Attention pattern visualization when processing dual audio inputs would provide insight into how comparison is performed.
- SUGGESTION: Cross-dataset generalization (train on ACD, test on CLD) would demonstrate robustness to distribution shifts.
- SUGGESTION: Systematic hallucination quantification—what percentage of generated events lack grounding in audio—would strengthen the practical utility assessment.
Novel insights: The cross-projection analysis revealing that the text prefix stores difference attributes rather than just steering generation is a notable mechanistic insight. The finding that smaller LMs with proper training can match or exceed larger models under compute constraints provides practical guidance for resource-limited settings. The three-tier explanation framework creates a novel evaluation dimension for audio-language models beyond standard captioning tasks.
Potentially missed related work:
- None
Suggestions:

# Merger Subscores
Novelty: 7.5
Technical soundness: 6.5
Empirical support: 7.0
Significance: 7.0
Clarity: 7.5

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept

=== CALIBRATION EXAMPLE 13 ===

# Final Consolidated Review
Summary: This paper presents SAM 2, a foundation model for promptable visual segmentation in both images and videos. The authors introduce a streaming transformer architecture with memory that processes video frames sequentially while maintaining temporal context. They collect the SA-V dataset—the largest video segmentation dataset to date with 35.5M masks across 50.9K videos—through an iterative data engine. SAM 2 achieves superior performance on video segmentation with 3× fewer interactions than prior methods while also being 6× faster and more accurate than the original SAM on image segmentation tasks.
Strengths:
- **Comprehensive task formulation**: The Promptable Visual Segmentation (PVS) task elegantly generalizes the Segment Anything task to videos, allowing prompts (clicks, boxes, masks) on any frame with iterative refinement. This formulation naturally subsumes both interactive VOS and semi-supervised VOS as special cases (Section 3).
- **Substantial dataset contribution**: SA-V contains 53× more masks than any existing VOS dataset (Table 3), with 190.9K manual masklets plus 451.7K automatically generated ones. The dataset is released under a permissive CC BY 4.0 license, representing a significant community resource.
- **Strong empirical performance**: SAM 2 achieves 75.3 J&F on 17 video datasets with 3 clicks (vs. 70.1 for SAM+Cutie, Table 4), outperforms prior work on SA-V val/test by significant margins (Table 6), and matches or exceeds SAM on image benchmarks while being 6× faster (Table 5).
- **Efficient data collection methodology**: The data engine achieves 8.4× speedup over per-frame annotation while maintaining comparable quality (Table 1), demonstrating the practical value of model-in-the-loop annotation.
- **Extensive ablation studies**: The paper includes thorough analyses of data mixtures (Table 7), model architecture choices (Tables 9-11), and demonstrates that SA-V data improves other models like Cutie (Table 17), showing the dataset's broader utility beyond SAM 2.
- **Fairness evaluation and responsible AI practices**: The authors conduct and report performance across demographic groups (Table 13, Section E.1.1), showing minimal discrepancy, and include detailed model/dataset cards following established standards (Appendix J).
- **Real-time inference capability**: The streaming architecture achieves 43.8 FPS (Hiera-B+) on a single A100 (Section 7), enabling practical applications.
Weaknesses:
- **Object pointer mechanism explanation could be clearer in main text**: While Section 4 mentions object pointers, the key implementation detail that 'the mask token corresponding to the output mask as the object pointer token' is deferred to Appendix D.1. This central mechanism for temporal propagation deserves more prominent exposition.
- **No inter-object communication**: The model processes multiple objects independently without sharing object-level context (Section D.1), which limits efficiency for multi-object scenarios and contributes to confusion with similar-looking objects—a known failure mode acknowledged in Appendix C.
- **Limited quantitative failure mode analysis**: While Appendix C describes failure cases (shot changes, crowded scenes, long occlusions, thin/fast objects), the paper lacks quantitative analysis of failure frequency or severity across different challenging scenarios.
- **Evaluation protocol assumptions**: The offline/online evaluation protocols rely on specific time assumptions (T_loc=1s, T_click=1.5s, T_exam=30s per 300 frames) that may not generalize to all real-world annotation scenarios. While the sensitivity to these parameters is not analyzed, this is a minor concern given the comprehensive evaluation across varied settings.
Nice-to-haves:
- **DEVA comparison in interactive setting**: DEVA achieves strong performance in Table 6 but is not evaluated in the interactive promptable segmentation experiments (Figure 5). Including this comparison would provide a more comprehensive baseline against modular decoupled approaches.
- **Long-video scalability analysis**: While LVOS/LVOSv2 results are reported, explicit analysis of how accuracy, memory usage, and latency scale with video length beyond training sequences would help practitioners understand deployment constraints.
- **Temporal attention visualization**: Visualizing what the cross-attention in memory attention focuses on (which past frames, which spatial regions) would help readers understand when and why the model succeeds or fails at temporal propagation.
- **Error propagation analysis**: Analysis of error recovery after corrective prompts, and characterization of 'hard to recover' failure modes, would help users understand when additional prompts are most needed.
- **Computational cost reporting**: Reporting FLOPs and memory footprint across model variants would help practitioners choose appropriate models for their compute constraints.
Novel insights: The key insight is that a unified streaming architecture with memory-based conditioning can effectively generalize SAM's promptable segmentation paradigm from images to videos, while maintaining strong performance on both domains. The finding that spatial memory features combined with lightweight object pointers outperform recurrent mechanisms like GRU (Table 11) is notable—the model learns to leverage temporal context without explicit recurrent state. The data engine design, where model-in-the-loop annotation achieves 8.4× speedup while targeting challenging failure cases, demonstrates an effective iterative methodology for building large-scale datasets.
Potentially missed related work:
- None
Suggestions:
- Consider adding a brief example or visualization of the memory bank evolution during a challenging scenario (e.g., occlusion followed by reappearance) to help readers build intuition for the temporal reasoning mechanism.
- Ablation on the number of auto-generated vs. manual masklets in training could help clarify the trade-offs of using automatically-generated data at scale.
- Quantifying failure rates across different challenge categories (e.g., occlusion duration thresholds, object size ranges) would provide users with clearer expectations about when the model is most reliable.

# Merger Subscores
Novelty: 7.0
Technical soundness: 8.5
Empirical support: 9.0
Significance: 9.0
Clarity: 7.5

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 8.0, 10.0]
Average score: 9.0
Binary outcome: Accept

=== CALIBRATION EXAMPLE 14 ===

# Final Consolidated Review
Summary: This paper introduces the concept of 'shallow safety alignment'—the observation that safety alignment in LLMs primarily affects only the first few output tokens, leaving later tokens largely unmodified from the base model. The authors demonstrate this phenomenon empirically and show how it explains multiple vulnerabilities (prefilling attacks, adversarial suffix attacks, decoding parameter exploits, fine-tuning attacks). They propose two mitigation strategies: (1) data augmentation with 'safety recovery examples' that train models to recover from harmful prefixes, and (2) a token-position-weighted constrained fine-tuning objective that limits distribution shifts on initial tokens during downstream fine-tuning.
Strengths:
- **Unified conceptual framework**: The 'shallow safety alignment' concept elegantly explains multiple seemingly disparate LLM vulnerabilities through a single underlying mechanism. The per-token KL divergence analysis (Figure 1) compellingly demonstrates that alignment primarily affects early tokens, and the prefilling experiments (Table 1) show that even unaligned models appear safe when forced to start with refusal prefixes.
- **Strong theoretical grounding**: The constrained fine-tuning objective (Equation 3) is well-motivated with detailed mathematical derivations in Appendix F, including limiting behavior analysis (Theorems 1-3), gradient interpretation, and RL-based reformulation. This theoretical rigor exceeds typical empirical safety papers.
- **Comprehensive empirical evaluation**: The paper evaluates across multiple models (Llama-2-7B-Chat, Gemma-7B-IT), multiple attack types (prefilling, GCG, decoding exploits, fine-tuning attacks), and multiple benchmarks (HEx-PHI, AdvBench, MaliciousInstruct). The per-token dynamics analysis in Figure 3 clearly shows how fine-tuning attacks concentrate changes in initial tokens.
- **Practical mitigation with demonstrated effectiveness**: The data augmentation approach reduces ASR on prefilling attacks from 42.1% to 2.8% (5 tokens prefilled) while largely maintaining utility. The constrained SFT reduces fine-tuning attack ASR from 88.9% to 4.6% while preserving downstream task utility.
- **Thorough ablation studies**: Appendices D and E provide ablations on data augmentation hyperparameters, β_t configurations, and warmup effects, strengthening empirical claims.
Weaknesses:
- **Limited model scale**: Experiments are conducted only on 7B-scale models (Llama-2-7B-Chat, Gemma-7B-IT). It remains unclear whether shallow safety alignment manifests similarly at larger scales where alignment procedures and model capabilities may differ.
- **No adaptive attack evaluation**: The proposed defenses are not tested against attackers specifically optimized to circumvent them (e.g., targeting tokens beyond constrained positions, distributing harmful content across positions).
- **Hyperparameter selection lacks principled guidance**: The specific β_t values (0.5, 2.0, 0.1) and token-position assignments appear somewhat arbitrary. While ablations show biased constraints are important, there is limited guidance on how to set these for new models.
- **Incomplete defense coverage for open-weight scenarios**: The constrained SFT requires the defender to control the fine-tuning process, which may not apply when users can freely fine-tune open-weight models. The augmented model still shows 55.2% ASR after harmful fine-tuning (Table 10).
- **Utility trade-offs exist**: Measurable utility drops occur (AlpacaEval: 51.8→49.5; constrained SFT GSM8k: 41.7→37.4). While the paper claims utility preservation, a more thorough discussion of these trade-offs would be appropriate.
Nice-to-haves:
- Evaluation on larger or more recent models (e.g., Llama-3, Mistral) would demonstrate whether shallow alignment persists across architectural advances.
- Formal characterization or metric for quantifying 'alignment depth' would strengthen the conceptual contribution.
- Mechanistic interpretability analysis of why models learn shallow alignment shortcuts during optimization would deepen understanding.
- Testing whether the data augmentation and constrained SFT compose synergistically when applied together would provide a more complete solution.
- Extension to multi-turn conversation settings where 'depth' might need to span across dialogue turns.
Novel insights: The paper's key insight—that diverse LLM vulnerabilities (adversarial suffix attacks, prefilling attacks, fine-tuning attacks) share a common root cause in shallow token-position effects—is genuinely valuable. The per-token gradient analysis during fine-tuning attacks (Figure 3) is particularly illuminating, showing that early token positions experience dramatically larger gradients, explaining why safety can be undone in just a few fine-tuning steps. The connection to the 'Superficial Alignment Hypothesis' is appropriately noted while distinguishing the safety-specific contributions.
Potentially missed related work:
- None
Suggestions:
- Provide more principled guidance on β_t selection based on model characteristics or safety requirements.
- Conduct at least preliminary evaluation against adaptive attacks designed to circumvent the proposed defenses.
- Clarify the threat model assumptions more explicitly—when each defense applies and what scenarios remain vulnerable.
- Consider releasing the augmented model checkpoints and training code to enable community building on this work.

# Merger Subscores
Novelty: 7.8
Technical soundness: 7.8
Empirical support: 7.3
Significance: 7.7
Clarity: 7.2

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 10.0, 10.0]
Average score: 9.5
Binary outcome: Accept
