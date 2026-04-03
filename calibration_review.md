=== CALIBRATION EXAMPLE 1 ===

# Review
## Review of "Systematic Review of Large Language Models: Applications, Limitations, Practical Usages and Future Directions"

### 1. Summary
This paper presents a survey of Large Language Models (LLMs), aiming to provide a comprehensive overview of their types, applications, limitations, and future directions. The authors categorize LLMs into generative, masked, sequence-to-sequence, and hybrid models, and provide a literature review covering deep learning techniques, core architectures (Transformer, BERT, GPT), and training methodologies. The paper further discusses limitations such as bias, resource intensity, and interpretability, alongside ethical considerations and adversarial robustness. Finally, it outlines evaluation metrics and benchmarks before concluding with future research directions.

### 2. Strengths
- **Broad Coverage:** The paper attempts to cover a wide spectrum of topics relevant to LLMs, ranging from the fundamental Transformer architecture to ethical considerations and future directions. This makes it a potential entry point for readers new to the field.
- **Structure:** The manuscript is organized logically, moving from types of models to literature, limitations, and finally future outlooks. The separation of model types (Generative vs. Masked vs. Seq2Seq) follows standard pedagogical approaches in NLP.
- **Relevant Topics:** The inclusion of sections on "Adversarial Robustness" and "Hallucination" highlights awareness of current critical issues in LLM deployment beyond just raw performance metrics.

### 3. Weaknesses
- **Lack of Novelty and Differentiation:** There are already numerous high-quality, comprehensive surveys on LLMs (e.g., Zhao et al. 2023, cited in the paper; Bommasani et al. 2021). This submission does not offer a distinct new perspective, a novel taxonomy, or a significantly deeper analysis than existing works. It largely summarizes established knowledge without providing the critical synthesis expected of a top-tier venue review.
- **Missing Critical Modern Developments:** The content feels dated for a paper submitted to a modern ML venue. While it covers BERT and GPT-3 extensively, it lacks substantial discussion on the paradigm shifts defining the current state-of-the-art, specifically:
    - **Instruction Tuning and Alignment:** There is almost no discussion on instruction tuning, RLHF (Reinforcement Learning from Human Feedback), or the shift from raw autoregressive models to helpful/aligned assistants (ChatGPT era).
    - **Efficient LLMs:** While "resource intensity" is mentioned as a limitation, there is no detailed discussion on recent architectural innovations aimed at solving this, such as Mixture-of-Experts (MoE), FlashAttention, or quantization techniques, which are crucial for modern LLM research.
    - **Emergent Abilities:** The concept of emergent abilities in scaling laws is missing.
- **Methodological Concerns regarding "Systematic Review":** The title claims this is a "Systematic Review," but the methodology section is missing. A systematic review typically requires a defined protocol (search terms, databases, inclusion/exclusion criteria, PRISMA flow diagram). Without this, the paper is merely a narrative review, and the title is misleading.
- **Repetitive and Poorly Edited Content:**
    - There is significant duplication between the Introduction and Section 3.2. For instance, the description of the Transformer architecture replacing RNNs/LSTMs and the introduction of BERT appears verbatim or near-verbatim in both sections.
    - There is a missing citation placeholder in the text (`**?**` on page 1), which indicates a lack of proofreading.
    - Formatting artifacts (e.g., `> - SYSTEMATIC REVIEW...`, `005 006`) suggest the manuscript was not properly polished before submission.
- **Superficial Comparative Analysis:** Table 1 and the surrounding analysis are quite shallow. Listing "Large model size" and "inference cost" as "Cons" for BERT without contextualizing these against modern standards or parameter-efficient fine-tuning (PEFT) methods reduces the utility of the comparison. Furthermore, comparing BERT (encoder-only) directly against GPT-3 (decoder-only) and T5 without nuanced discussion of their specific use-cases (understanding vs. generation) is too simplistic.

### 4. Questions
1.  **Novelty:** How does this survey improve upon or differentiate itself from existing comprehensive surveys like "A Survey of Large Language Models" (Zhao et al., 2023)? What new taxonomy or synthesis does it offer?
2.  **Methodology:** Can the authors clarify the methodology used to select papers for this "systematic" review? What databases were searched, and what were the inclusion criteria?
3.  **Modern Context:** Why are critical modern topics such as RLHF, Instruction Tuning, and Chain-of-Thought prompting largely omitted from the discussion? Are the authors planning to expand the scope to include the "ChatGPT era" innovations in detail?
4.  **Editing:** There are duplicate paragraphs between the Introduction and Section 3.2/3.3. Is this intentional, or is it an editing error that needs to be cleaned up to improve conciseness?
5.  **Evaluation:** The paper mentions benchmarks like HELM. Could the authors provide a more critical analysis of how well these benchmarks capture the *reasoning* capabilities of modern LLMs, rather than just listing them?

# Actual Human Scores
Individual reviewer scores: [1.0, 1.0, 1.0, 1.0]
Average score: 1.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 2 ===

# Review
# Review: Differentiation of Multi-Objective Data-Driven Decision Pipeline

## Summary

This paper proposes Multi-Objective Decision-Focused Learning (MoDFL), extending the decision-focused learning paradigm from single-objective to multi-objective optimization problems. The key insight is that traditional two-stage methods (predict then optimize) can lead to suboptimal decisions due to misalignment between prediction and optimization objectives, and this problem is compounded in multi-objective settings where solutions form Pareto fronts rather than unique optima.

The authors contribute three novel loss functions tailored for MOP:
1. **Landscape loss**: Uses sample rank maximum mean discrepancy (sRMMD) to measure discrepancy in objective space between predicted and true problems
2. **Pareto set loss**: Measures distance between Pareto sets in solution space using inverted generational distance concept
3. **Decision loss**: Transforms the MOP into a single-objective problem via weighted sum and measures decision quality

The method is validated on two benchmark problems: web advertisement allocation and bipartite matching among scientific papers.

---

## Strengths

**1. Novel Problem Extension**: Extending decision-focused learning to multi-objective optimization is a meaningful contribution. The paper correctly identifies that existing DFL methods focus predominantly on single-objective problems, leaving a gap for the numerous real-world multi-objective scenarios.

**2. Thoughtful Loss Design**: The three-component loss structure (landscape, Pareto set, decision) captures different aspects of multi-objective solution quality—objective space distance, solution space distance, and representative decision quality. This is more comprehensive than naively applying single-objective losses.

**3. Technical Innovation**: The use of sRMMD for landscape loss is creative, treating the objective space as a manifold and leveraging optimal transport concepts to measure distributional discrepancies.

**4. Empirical Validation**: The paper compares against seven baseline methods across multiple metrics (GD, MPFE, HAR, regret) and includes ablation studies demonstrating the contribution of each loss component.

---

## Weaknesses

**1. Writing and Presentation Quality**: The paper suffers from significant formatting and clarity issues:
- Duplicate text in the introduction (entire paragraphs repeated verbatim)
- Garbled/placeholder text in equations (e.g., "[The problem under study can be formulated as follows:]")
- Inconsistent notation and unclear mathematical definitions
- Reference numbering appears corrupted (e.g., "141142", "225226")
- Missing figure descriptions

**2. Methodological Concerns with Decision Loss**: The weighted sum transformation for decision loss is problematic:
- A single weighted solution cannot represent the full Pareto front quality
- The choice of weights (uniform weights mentioned) is arbitrary and not justified
- Weighted sum methods have well-known limitations—they cannot find solutions on non-convex portions of Pareto fronts

**3. Limited Baseline Comparison**: The baseline methods (SPO, BB, NCE, etc.) are all single-objective DFL methods adapted by summing weighted losses. This is a naive adaptation that doesn't account for multi-objective structure. A more principled multi-objective baseline comparison is needed.

**4. Experimental Limitations**:
- Only two benchmark problems, one described opaquely as "Anonymous App"
- Both problems are LP-based; no evaluation on other problem classes (MIP, convex, non-convex)
- Limited discussion of computational cost; sRMMD computation scales quadratically with sample size
- No scalability analysis with respect to number of objectives or problem size

**5. Hyperparameter Sensitivity**: The method introduces multiple hyperparameters (λl, λd, λps, kernel bandwidths σ, τ, ε) with limited discussion of sensitivity or principled selection methods.

**6. Incomplete Pareto Set Computation**: The paper does not adequately explain how Pareto sets are obtained during training for computing Pareto set loss. Computing true Pareto sets for each problem instance during training would be computationally prohibitive.

---

## Questions

1. **Pareto Set Computation**: How exactly are the Pareto sets P_{y*} computed during training? Are these approximated, and if so, how does approximation quality affect the method?

2. **Scalability**: How does computational cost scale with (a) number of objectives, (b) problem size, and (c) number of samples for sRMMD? What are practical limits?

3. **Weighted Sum Limitations**: The decision loss uses weighted sum aggregation. How does the method handle non-convex Pareto fronts where weighted sum methods fail to capture all optimal solutions?

4. **Baseline Fairness**: Is comparing against single-objective DFL methods with summed losses a fair comparison? Have the authors considered comparing against multi-objective optimization-specific approaches?

5. **Hyperparameter Selection**: What guidance exists for selecting λl, λd, λps? The ablation shows all components help, but is there a principled way to balance them?

6. **Number of Objectives**: Experiments only test bi-objective and tri-objective cases (with the third being a weighted combination of the first two). How does performance degrade with more objectives?

7. **Anonymous App Details**: Can the authors provide more details about the web advertisement allocation problem? The current description is vague about the specific optimization formulation.

8. **Reproducibility**: Will code be released? The mathematical notation inconsistencies make implementation challenging without reference code.

---

## Additional Comments

The paper addresses an important problem but needs substantial revision before publication. The core ideas are sound, but the execution—in terms of writing quality, experimental rigor, and methodological completeness—needs improvement. The authors should particularly address the limitation of weighted sum for representing multi-objective quality and provide clearer problem formulations.

# Actual Human Scores
Individual reviewer scores: [3.0, 1.0, 3.0]
Average score: 2.3
Binary outcome: Reject

=== CALIBRATION EXAMPLE 3 ===

# Review
# Review of MQFL-FHE: Multimodal Quantum Federated Learning Framework with Fully Homomorphic Encryption

## 1. Summary

This paper proposes MQFL-FHE, a novel framework that integrates quantum computing (QC) with fully homomorphic encryption (FHE) in federated learning (FL) settings. The central thesis is that while FHE provides strong privacy guarantees in FL by enabling computations on encrypted model updates, it introduces noise and quantization errors that degrade model accuracy during aggregation. The authors propose using quantum neural network layers to counteract this performance degradation.

The paper makes three main contributions: (1) a QFL-FHE framework that integrates FHE with quantum federated learning, (2) a multimodal quantum federated learning (MQFL) extension for handling heterogeneous data modalities, and (3) a Multimodal Quantum Mixture of Experts (MQMoE) architecture that uses quantum layers within an expert routing mechanism. The authors validate their approach on CIFAR-10, DNA sequences, MRI scans, PCOS datasets, and a combined DNA+MRI multimodal dataset, reporting improved accuracy when quantum enhancements are applied to FHE-based FL.

## 2. Strengths

**Novelty of Integration:** The paper presents an ambitious integration of three emerging technologies—quantum computing, federated learning, and fully homomorphic encryption. The combination of QC with FHE to address performance degradation is a genuinely novel direction that could open new research avenues in privacy-preserving machine learning.

**Comprehensive Experimental Setup:** The authors conduct experiments across multiple datasets (CIFAR-10, DNA Sequence, MRI Scan, PCOS, and DNA+MRI combined) with various configurations (classical centralized, quantum centralized, classical FL, QFL, FL+FHE, QFL+FHE). This thoroughness allows for meaningful comparisons across different settings.

**Real-World Application Focus:** The choice to validate the approach on healthcare-related datasets (genomics and brain MRI) demonstrates attention to practical applications where both privacy and accuracy are critical. This domain relevance strengthens the paper's potential impact.

**Detailed Ablation Analysis:** The paper includes ROC curves and confusion matrices that help visualize where quantum enhancements provide the most benefit (particularly for underrepresented classes in DNA sequences and challenging MRI categories like glioma).

**Mathematical Appendix:** The inclusion of Appendix 7.1 on error propagation analysis, which discusses how quantum unitary constraints (SU(2)) might bound noise propagation, provides theoretical grounding for the empirical claims.

## 3. Weaknesses

**Unclear Causal Mechanism:** The central claim that quantum computing "counteracts" or "mitigates" FHE-induced performance degradation lacks rigorous theoretical justification. The paper states that quantum operations "bound error propagation" through unitary constraints, but this argument is not convincingly developed. How exactly do quantum rotations prevent or reduce CKKS noise accumulation? The mathematical appendix discusses quantum state norm preservation, but the connection to FHE noise cancellation remains speculative and is not empirically validated through controlled experiments.

**Unfair Baseline Comparisons:** The paper compares classical and quantum models but does not clearly establish whether the model architectures are equivalent in terms of parameter count and capacity. Tables 3 and 4 show quantum models sometimes outperforming classical ones (e.g., CIFAR-10: 74.33% vs 76.59% in centralized; 72.16% vs 71.03% in FL), but it is unclear whether this is due to quantum advantage or simply a more expressive architecture. The parameterized quantum circuits described in Section 4.1 use rotation gates and CNOTs, but classical baseline architecture details are sparse.

**Simulation vs. Real Quantum Hardware:** All quantum experiments are conducted using PennyLane simulators on classical hardware (NVIDIA A100 GPUs). The paper makes claims about quantum advantages but does not validate these on actual quantum devices or discuss the implications of simulation. This raises questions about whether the reported "quantum" benefits would transfer to real quantum systems where noise and decoherence are significant factors.

**Computational Overhead Not Adequately Addressed:** Table 4 shows dramatic increases in computation time: classical FL on CIFAR-10 takes ~3405 seconds while QFL+FHE takes ~9747 seconds—nearly 3x slower. While the paper acknowledges this overhead, it does not provide a cost-benefit analysis. Is a 1-2% accuracy improvement worth a 3x computational cost increase? This trade-off should be explicitly discussed.

**Inconsistent Performance Claims:** The results show mixed patterns. For instance, in centralized settings (Table 3), classical models outperform quantum ones on CIFAR-10 (76.59% vs 74.33%) and DNA Sequence (94.50% vs 92.63%), while in FL settings, quantum sometimes wins. The paper claims QC "mitigates FHE degradation" but the evidence is inconsistent across datasets and settings. A more nuanced discussion of when and why quantum helps versus hurts is needed.

**Limited Baselines for FHE Comparison:** The paper compares FL+FHE vs QFL+FHE but does not include important baselines such as: (a) FL with differential privacy, (b) FL with secure aggregation (without FHE), or (c) other privacy-preserving techniques. Without these comparisons, it is difficult to assess whether the proposed approach is the best way to achieve privacy in FL.

**MQMoE Architecture Details Insufficient:** The MQMoE description in Section 4.3 is too brief. How many experts are used? What are the classical preprocessing layers for each modality? How is the gating network trained? Figure 3 provides a high-level view but lacks architectural details such as layer dimensions, activation functions, and quantum circuit depths for each expert.

**No Statistical Significance Testing:** The results report single accuracy values with time standard deviations, but no confidence intervals or statistical significance tests for accuracy comparisons. Given that some improvements are small (e.g., 0.9% for DNA in QFL+FHE vs FL+FHE), significance testing is essential to determine if differences are meaningful.

## 4. Questions

1. **Mechanism Clarification:** Could the authors provide a more precise mathematical explanation of how quantum operations specifically mitigate CKKS-related noise? The current explanation about unitary constraints and bounded error propagation is high-level. Is there experimental evidence showing reduced FHE noise variance when quantum layers are present?

2. **Architecture Equivalence:** What are the exact parameter counts for classical vs. quantum model architectures? Are they designed to be comparable, or do quantum models have more expressivity by design? Please provide layer-by-layer details for fair comparison.

3. **Quantum Circuit Specifics:** The PQC definition in Section 4.1 mentions L layers with RX gates and CNOTs. What value of L was used for each dataset? How many qubits? What encoding method was used for classical data?

4. **FHE Parameter Sensitivity:** Table 5 in Appendix 7.2 shows FHE parameter trade-offs, but the main experiments use fixed parameters. How sensitive are the claimed benefits to FHE parameters like polynomial modulus degree and coefficient modulus size? Does the "quantum advantage" vary with encryption depth?

5. **Real-World Feasibility:** Given that current quantum computers have limited qubits and high error rates, what are the practical implications of this work? Would the proposed approach work on near-term NISQ devices, or is this a long-term vision?

6. **Baselines Missing:** Why were differential privacy and secure aggregation baselines not included? These are standard privacy-preserving FL techniques that should be compared against.

7. **Noise Propagation Experiments:** The appendix discusses theoretical error propagation, but are there empirical experiments showing that FHE noise behaves differently in quantum vs. classical networks? For example, measuring model parameter distributions after multiple encrypted aggregation rounds.

8. **Scalability:** How does the approach scale with more clients, more modalities, or larger models? The experiments use 10 clients—what happens with 100 or 1000 clients?

9. **MQMoE Gating Mechanism:** How exactly does the quantum layer contribute to the gating mechanism in MQMoE? Is the quantum layer part of the gating network, or only within the experts? The description is ambiguous.

10. **Ablation on Quantum Contribution:** Is there an experiment that isolates the quantum contribution? For instance, comparing: (a) classical FL+FHE, (b) classical FL+FHE with an expanded architecture (same parameter count as quantum), and (c) QFL+FHE. This would help distinguish architectural improvements from quantum-specific benefits.

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 3.0, 3.0, 3.0]
Average score: 3.4
Binary outcome: Reject

=== CALIBRATION EXAMPLE 4 ===

# Review
# Review of "FedQLoRA: Federated Quantization-Aware LoRA for Large Language Models"

## Summary

This paper addresses an important but previously overlooked problem in federated learning (FL) with large language models: **quantization bias** that arises when clients use LLMs quantized to different precision levels. The authors first demonstrate that naively aggregating LoRA adapters from clients with heterogeneous quantization (e.g., mixed 2-bit and 4-bit models) leads to worse performance than using uniformly lower-precision models. To address this, they propose **FedQLoRA**, which introduces a **quantization-aware adapter** to estimate and separate quantization error from the learned task knowledge. The quantization-aware adapter remains personalized (not aggregated), while the LoRA adapter is shared across clients. They further propose an **iterative version (iFedQLoRA)** that alternates between updating the quantization-aware adapter using global knowledge and training the LoRA adapter on local data, addressing data heterogeneity bias in non-IID settings. Experiments on text classification tasks demonstrate improved performance over baselines.

---

## Strengths

1. **Novel problem identification.** The paper identifies and clearly articulates the quantization bias problem in federated LoRA. The motivating experiment (Figure 1) effectively demonstrates that mixing quantization levels leads to unexpected performance degradation—worse than using all low-precision models. This is a real and practical concern given the trend toward deploying quantized LLMs on heterogeneous edge devices.

2. **Principled problem formulation.** The derivation of quantization bias in Equations 7-9 provides clear theoretical grounding, decomposing the aggregation error into quantization error and quantization bias components. The insight that quantization error is endogenous (determined by the original model and quantization method) motivates the separation strategy.

3. **Well-motivated solution.** The proposed quantization-aware adapter is a sensible approach to isolate quantization-specific corrections from task-specific knowledge. The connection to LoRA-aware quantization (LoftQ) through Proposition 1 is elegant and provides theoretical justification.

4. **Comprehensive experimental design.** The authors evaluate across multiple dimensions: IID vs. non-IID settings, varying numbers of clients, different proportions of quantization levels, and convergence analysis. The ablation on model heterogeneity (Figure 3) and data heterogeneity (Figure 4) provides useful insights.

5. **Good convergence properties.** The iterative version shows faster convergence than H-LoRA (Figure 5), which is valuable in FL settings where communication rounds are costly.

---

## Weaknesses

1. **Limited model scale.** The experiments use DistilBERT, a relatively small model (~66M parameters). While this facilitates experimentation, the core premise of the paper is about *large* language models with "billions of parameters." It is unclear whether the findings generalize to truly large-scale models (7B+ parameters) where quantization is most impactful and where quantization error characteristics may differ significantly.

2. **Strong assumptions about quantization method.** The proposed method assumes that the quantization method Q is known and identical across clients (e.g., deriving Equation 15 from 14). However, in realistic heterogeneous deployments, different clients may use different quantization algorithms (uniform vs. non-uniform, different calibration datasets, etc.). The paper does not discuss this scenario or analyze sensitivity to this assumption.

3. **Missing relevant baselines.** The paper compares against FFA-LoRA, H-LoRA, and H-LoRA-T, but does not compare with other personalized FL approaches that could potentially handle heterogeneity, such as FedPer, FedRep, or recent personalized FL methods designed for LLMs. Additionally, a simple baseline of personalized LoRA (no aggregation) should be included to isolate the benefits of federation.

4. **Incomplete analysis of FFA-LoRA results.** FFA-LoRA performs poorly in the experiments (sometimes near random), which the authors attribute to hyperparameter sensitivity. This warrants deeper investigation—is the failure fundamental to the method under mixed quantization, or is it a tuning issue? The comparison may not be fair.

5. **Memory and computational overhead not analyzed.** The quantization-aware adapter adds additional parameters that must be stored locally. The paper does not quantify the memory overhead, which is critical for resource-constrained edge devices. Similarly, the iterative version requires alternating optimization—what is the computational cost compared to the non-iterative version?

6. **SVD approximation is heuristic.** The use of SVD to obtain the quantization-aware adapter (Equation 12) assumes the quantization error matrix is low-rank. No justification or empirical validation of this assumption is provided. The rank m is a hyperparameter whose impact is not analyzed.

7. **Limited quantization settings.** Only 2-bit and 4-bit quantization levels are tested. Real-world deployments may involve more varied precision levels (e.g., 3-bit, 8-bit, mixed-precision). The performance with a wider range of quantization combinations remains unexplored.

8. **Ablation on the rank of adapters missing.** The paper does not study how the rank of the LoRA adapter and quantization-aware adapter affect performance. Since both adapters use low-rank decomposition, understanding their interplay is important.

---

## Questions

1. **Scalability to larger models:** Have the authors tested or do they have plans to validate FedQLoRA on larger language models (e.g., LLaMA-7B or larger)? The characteristics of quantization error may change significantly with model scale, and the computational feasibility of the iterative version needs to be demonstrated.

2. **Handling different quantization methods:** How would FedQLoRA handle the case where clients use fundamentally different quantization algorithms (not just different bit-widths)? Does the method require knowledge of the exact Q function, or is an approximation sufficient?

3. **Hyperparameter sensitivity:** What are the ranks (r for LoRA, m for quantization-aware adapter) used in the experiments? How sensitive are the results to these choices? Are they tuned separately for each client or shared?

4. **Memory requirements:** What is the total parameter overhead of the quantization-aware adapter compared to standard LoRA? For edge deployment, this is a critical consideration.

5. **Personalized LoRA baseline:** What is the performance of training LoRA locally without any federation (i.e., purely personalized, no aggregation)? This would help quantify the trade-off between personalization and collaborative learning.

6. **Performance gap explanation:** Could the authors explain why FFA-LoRA performs so poorly in Table 1 and 2? Is there a fundamental incompatibility with mixed quantization, or is it a tuning issue?

7. **Convergence of iterative version:** Does iFedQLoRA always converge? The optimization alternates between two objectives—are there any theoretical guarantees or conditions for convergence?

8. **Inference-time overhead:** During inference, both adapters must be applied. What is the latency overhead compared to standard LoRA, particularly for batched inference?

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 5.0, 3.0]
Average score: 4.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 5 ===

# Review
# Review of "ADAFM: Adaptive Variance-Reduced Algorithm for Stochastic Minimax Optimization"

## Summary

This paper introduces AdaFM (Adaptive Filtered Momentum), an adaptive variance-reduced algorithm for stochastic minimax optimization. The key contribution is designing a method that automatically adjusts hyperparameters based on historical estimator information, eliminating the need for problem-dependent parameter tuning that plagues existing VR-based algorithms in minimax settings.

The algorithm builds on STORM-style momentum-based variance reduction but makes two key adaptations: (1) momentum parameters β_t are simplified to 1/t^{2/3} for both primal and dual variables, and (2) learning rates η_t^x and η_t^y are computed adaptively using cumulative gradient estimator norms, with a max operation ensuring proper interaction between the two variables' learning rates.

The authors provide theoretical analysis showing that AdaFM achieves O(ε^{-3}) sample complexity for finding ε-stationary points in both Non-Convex-Strongly-Concave (NC-SC) and Non-Convex-PL (NC-PL) settings, matching the optimal rate of parameter-dependent algorithms. The paper includes experiments on synthetic test functions, deep AUC maximization, and WGAN-GP training, comparing against RSGDA, VRAdaGDA, and TiAda.

## Strengths

1. **Addresses a practical problem**: The paper correctly identifies that existing VR methods for minimax optimization require careful tuning of many problem-dependent hyperparameters (learning rate ratios, momentum parameters, etc.), making them impractical for real-world use. This is a genuine and important problem.

2. **Novel algorithmic design**: The approach of using cumulative gradient norms with momentum-weighted denominator to automatically adjust learning rates, combined with the max operation to couple the two learning rates, is a clever adaptation of AdaGrad-style ideas to the minimax setting. The theoretical justification via error dynamics analysis is sound.

3. **Solid theoretical results**: The paper provides convergence guarantees for both NC-SC and NC-PL settings with rigorous proofs. The O(ε^{-3}) sample complexity is indeed optimal for the non-convex minimax setting, and achieving this without knowledge of problem parameters is a meaningful theoretical advance.

4. **Comprehensive experimental evaluation**: The experiments cover diverse settings (synthetic functions, deep AUC on imbalanced data, GAN training) and multiple datasets (CIFAR-10, CIFAR-100). The hyperparameter sensitivity analysis for RSGDA in Figure 1 effectively motivates the need for adaptive methods.

5. **Clear algorithmic contribution over TiAda**: The comparison with TiAda shows that AdaFM converges faster while maintaining adaptivity. The test function experiments demonstrate that AdaFM adjusts stepsize ratios more quickly than TiAda.

## Weaknesses

1. **Overstated "parameter-free" claim**: The algorithm still requires three hyperparameters: γ, λ, and δ. While the paper argues these can be set to default values, the ablation study (Figure 9) shows that δ=0 causes divergence. This contradicts the claim that "any arbitrarily small δ" works. The statement "adjusting these three parameters presents no difficulty" needs stronger empirical support.

2. **Missing baselines and comparisons**: The paper only compares against RSGDA, VRAdaGDA, and TiAda. Recent adaptive minimax methods like NeAda (NeAda for nested optimization) and PES (cited in related work) are not included as baselines. Additionally, comparing against simpler adaptive methods like Adam-SGDA would strengthen the practical evaluation.

3. **Ablation study reveals concerning behavior**: Figure 9a shows that AdaFM diverges when δ=0, which raises questions about the robustness of the algorithm. The paper should explain why this happens and whether there are safer alternatives.

4. **Gap between theory and practice**: The theory assumes bounded gradients (Assumption 2), but in deep learning applications, gradients can grow unbounded. The experiments use neural networks but don't discuss this discrepancy. Gradient clipping might be needed in practice.

5. **Incomplete analysis of the δ parameter**: The ablation study on δ is limited. The paper claims δ can be "arbitrarily small," but then Figure 10 suggests smaller δ leads to better performance on WGAN-GP. More systematic analysis of δ's effect across different tasks would be valuable.

6. **Computational overhead not discussed**: Computing cumulative sums of gradient norms adds memory and computation costs. The paper should quantify this overhead, especially for large-scale problems.

7. **Narrow theoretical improvement over TiAda**: The paper claims TiAda achieves O(ε^{-4}) complexity, but this comparison may not be entirely fair. TiAda is designed for a more general setting and doesn't assume bounded gradients. The comparison should clarify whether the complexity improvement comes from assumptions or algorithmic innovation.

## Questions

1. **On the δ parameter**: The ablation study shows failure when δ=0, but the theory suggests any small δ suffices. Can you explain this discrepancy? What is the minimum δ that guarantees convergence in practice, and how should users select it?

2. **On hyperparameter defaults**: The experiments use different values of δ across tasks (δ=0.1 for test functions, δ=0.001 for deep AUC and WGAN-GP). How should practitioners choose δ, γ, and λ? Is there a principled approach beyond tuning?

3. **On comparison with other adaptive methods**: Why weren't recent adaptive minimax methods like NeAda included in the experiments? How would AdaFM compare against Adam-SGDA or other simpler adaptive schemes?

4. **On practical implementation**: The algorithm requires maintaining cumulative sums of gradient norms divided by β_i. For very long training runs, these sums can grow large. Are there practical numerical stability concerns, and how do you handle them?

5. **On the learning rate max operation**: The use of max{α_t^x, α_t^y} for η_t^x is motivated by ensuring x updates more conservatively when y optimization is incomplete. Could you provide intuition for why this particular formulation works better than other coupling mechanisms?

6. **On convergence verification**: The theory guarantees convergence to stationary points. How does this translate to practical metrics like AUC score or inception score in the experiments? Are there gaps between theoretical convergence and practical performance?

## Additional Comments

The writing is generally clear, though there are some minor issues:
- The notation δ_x = 1/3 + δ and δ_y = 1/3 - δ could be confusing; consider using different symbols.
- Some equations in the appendix are hard to parse due to formatting.

Overall, this is a solid contribution to adaptive optimization for minimax problems. The paper makes meaningful progress on an important practical challenge, though the claims about being fully parameter-free are somewhat overstated. Addressing the weaknesses, particularly the δ parameter behavior and missing baselines, would significantly strengthen the work.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 8.0, 6.0]
Average score: 5.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 6 ===

# Review
# Review of "Swift-FedGNN: Federated Graph Learning with Low Communication and Sample Complexities"

## Summary

This paper proposes Swift-FedGNN, a federated learning framework for training Graph Neural Networks (GNNs) on geo-distributed graph data. The key challenge addressed is the high communication and sampling costs incurred by cross-client neighbors in federated GNN training. The proposed solution has clients perform local training using only local graph data in most iterations, while periodically allowing a subset of clients to conduct cross-client training that incorporates neighbor information from other clients. To further reduce communication overhead, neighbor embeddings are aggregated twice before transmission: first at remote clients and then at the server. The authors provide theoretical convergence analysis showing an O(1/√T) rate, matching state-of-the-art sampling-based GNN methods, and validate the approach on ogbn-products and Reddit datasets.

---

## Strengths

**1. Clear Problem Formulation:** The paper identifies an important practical challenge in federated GNN learning—the prohibitive communication costs from cross-client neighbor sampling. The motivation is well-grounded, with a realistic healthcare scenario example and empirical evidence (Figure 2) demonstrating that cross-client operations dominate training time.

**2. Solid Theoretical Contributions:** The convergence analysis is a notable strength. Unlike prior work that relies on strong assumptions (unbiased or consistent stochastic gradients), this paper explicitly bounds the stochastic gradient approximation errors caused by neighbor sampling and missing cross-client neighbors. The connection between gradient bias and GNN depth (number of layers) is an interesting insight unique to federated GNN training.

**3. Principled Algorithm Design:** The approach of periodic cross-client training with a subset of clients is intuitive yet effective. The double-aggregation mechanism (at remote clients and server) serves dual purposes: reducing communication and providing a degree of privacy protection by hiding individual node embeddings.

**4. Comprehensive Experimental Evaluation:** The experiments on ogbn-products (large-scale) and Reddit (dense) datasets provide good coverage of different graph characteristics. The comparison with LLCG, FedGNN-PNS, and FedGNN-G is appropriate, and the analysis of computation vs. communication time ratios is informative.

**5. Complexity Analysis:** The communication overhead per iteration (Table 2) clearly demonstrates the efficiency gains, with Swift-FedGNN achieving 4-20× reduction compared to baselines.

---

## Weaknesses

**1. Limited Baseline Comparison:** The paper misses several relevant federated GNN methods. Notably, FedGraphNN (He et al., 2021), SpreadGNN (He et al., 2021), and FedGCN (Yao et al., 2023a) with its one-time communication strategy deserve comparison. The latter is mentioned in Related Work but not included in experiments, which is concerning since it addresses similar issues.

**2. Privacy Analysis is Superficial:** While the paper claims privacy benefits from double aggregation, no formal privacy guarantees (e.g., differential privacy) or empirical privacy analysis are provided. The claim that "information of each node is not leaked" is misleading—aggregated embeddings can still leak information. This is a significant gap given the paper's focus on privacy-preserving learning.

**3. Limited Task Scope:** The evaluation is restricted to node classification. GNNs are widely used for link prediction and graph classification tasks. The algorithm's applicability to these tasks, particularly link prediction where edge information across clients is critical, should be discussed or evaluated.

**4. Architectural Limitations:** As noted in footnote 2, the approach only supports element-wise aggregation operations (mean, sum, max). This excludes important GNN architectures like GAT (Graph Attention Networks) that use attention mechanisms. The practical implications of this limitation should be more prominently discussed.

**5. Unrealistic Simulation Environment:** The experiments simulate federated learning on a single machine with shared memory and synthetic network bandwidth constraints. Real-world deployments involve network latency, asynchronous communication, and system heterogeneity, which could significantly impact the results.

**6. Missing Non-IID Analysis:** Federated learning research typically addresses non-IID data distributions across clients. The paper uses METIS partitioning, which creates structured splits but doesn't explore heterogeneity scenarios (e.g., different degree distributions or class distributions per client).

**7. Hyperparameter Sensitivity Gaps:** The analysis of correction frequency I and number of cross-client training clients K is limited to communication ratios. How do these parameters affect final model accuracy and convergence? What are good default values?

**8. Server Bottleneck Concern:** During cross-client training, the server aggregates embeddings from multiple remote clients for each training node. This could create a computational bottleneck on the server, but this aspect is not analyzed.

**9. Convergence Rate Interpretation:** The O(1/√T) rate matches sampling-based GNN methods, but the residual error term in Theorem 5.6 grows with I and decreases with K. A more detailed discussion of how to set I and K to balance efficiency and accuracy would strengthen the practical applicability.

---

## Questions

1. **Privacy Guarantees:** Can the authors provide any formal or empirical analysis of the privacy protection offered by the double aggregation mechanism? Would adding differential privacy noise be necessary for strong privacy guarantees?

2. **Comparison with FedGCN:** Why was FedGCN (Yao et al., 2023a) not included in the experimental comparison despite being discussed in Related Work? Its one-time full cross-client neighbor communication strategy seems like an important baseline.

3. **Scalability with Number of Clients:** How does Swift-FedGNN scale with a large number of clients? The current experiments use only 10-20 clients. Does the server bottleneck become problematic when hundreds of clients participate?

4. **Non-IID Data Scenarios:** How does Swift-FedGNN perform when client data distributions are heterogeneous (e.g., varying class distributions or graph structural properties)? This is crucial for real-world federated learning applications.

5. **Link Prediction Task:** Is Swift-FedGNN applicable to link prediction tasks where predicting edges across clients may require cross-client information? How would the algorithm handle this scenario?

6. **GAT and Non-element-wise Operations:** Can the authors elaborate on potential modifications to support attention-based GNN architectures like GAT? What would the communication overhead look like in that case?

7. **Asynchronous Setting:** The current design assumes synchronous training. How would Swift-FedGNN perform in asynchronous federated settings where clients may have different update frequencies?

8. **Real-world Network Latency:** Have the authors considered testing in a more realistic distributed environment with actual network latency? The shared memory simulation may not capture realistic communication patterns.

---

## Minor Comments

- The notation is generally clear but could benefit from a more compact representation in some sections.
- The related work section on federated GNNs could be more comprehensive, particularly regarding recent developments in this rapidly evolving area.
- Figure quality is good, but some figures (e.g., Figure 4) would benefit from confidence intervals or error bars across multiple runs.

# Actual Human Scores
Individual reviewer scores: [3.0, 6.0, 5.0, 5.0]
Average score: 4.8
Binary outcome: Reject

=== CALIBRATION EXAMPLE 7 ===

# Review
# Review of "Hierarchical Prompts with Context-Aware Calibration for Open-Vocabulary Object Detection"

## Summary
The paper proposes **HiCA**, a method for Open-Vocabulary Object Detection (OVD) designed to improve generalization to novel classes while maintaining strong performance on base classes. The authors identify two main limitations in existing knowledge distillation-based OVD methods: (1) the direct mapping of regions to class prompts ignores shared semantic knowledge (superclass information), leading to overfitting on base classes, and (2) visual context is often ignored or treated purely as negative background, failing to capture scene-object correlations.

To address these, HiCA introduces two components:
1.  **Hierarchical Prompts**: This module decomposes the standard region-category alignment into a two-step process: region-to-superclass (coarse-grained) and superclass-to-category. It utilizes learnable prompts for both levels and fuses them during inference to share semantic knowledge across classes.
2.  **Context-Aware Calibration**: This module leverages unsupervised clustering of global image features to define visual contexts. It trains a Distribution Generation (DG) Layer to predict the probability distribution of superclasses within a given context, using ground-truth annotations as supervision. During inference, this predicted distribution calibrates the classification logits via an element-wise product.

Experiments are conducted on OV-COCO and OV-LVIS benchmarks. The method demonstrates consistent improvements over baselines like OADP and BARON, particularly boosting base class performance.

## Strengths
1.  **Motivation and Insight**: The paper correctly identifies a significant issue in current OVD methods: the overfitting to base classes due to fine-grained prompt learning. Introducing hierarchical semantic structures (superclasses) is a logical and effective strategy to regularize the feature space and share knowledge between base and novel classes.
2.  **Novelty in Context Modeling**: The context-aware calibration module offers a sensible way to utilize global image information. Instead of merely treating context as background, modeling the conditional probability of object occurrence based on scene features is a sound approach for disambiguating objects and suppressing false positives.
3.  **Plug-and-Play Design**: The proposed modules are largely independent of the specific detector architecture. The demonstration that HiCA can be applied to both OADP and BARON baselines effectively shows its versatility and potential for broader application.
4.  **Strong Empirical Results**: The method achieves state-of-the-art results on OV-COCO (50.4 mAP50) and competitive results on OV-LVIS. The ablation studies clearly isolate the contributions of the hierarchical prompts and the context-aware calibration, validating the design choices.

## Weaknesses
1.  **Reliance on Superclass Definitions**: The proposed hierarchical prompt mechanism fundamentally relies on a predefined taxonomy where classes are grouped into superclasses (e.g., "dog" $\in$ "animal"). The paper does not clearly explain the source of this hierarchy. Is it derived from the COCO/LVIS category definitions, WordNet, or manually constructed? This dependency limits the method's applicability to scenarios where such hierarchies are unavailable or where novel classes do not fit cleanly into existing superclasses. The assumption that the superclass hierarchy for *novel* classes is known during inference needs explicit clarification.
2.  **Heuristic Nature of Calibration**: The context-aware calibration relies on a Hadamard product (element-wise multiplication) between the region-class similarity scores and the predicted context-class distribution (Eq. 7). While intuitive (acting as a gating mechanism), this interaction is heuristic. A justification for why a multiplicative operation is preferred over an additive bias (log-odds adjustment) or a softmax fusion is lacking. Additionally, the "Distribution Generation Layer" predicting context-superclass probabilities requires supervision from ground-truth co-occurrence statistics, adding an extra preprocessing step that is not thoroughly discussed.
3.  **Modest Novel Class Gains**: While the overall performance is strong, the improvement on novel classes (mAP_N) is relatively modest compared to the gains on base classes (mAP_B). For instance, on OV-COCO with OADP, the method improves mAP_N by 1.3% but mAP_B by 5.5%. This suggests the method primarily stabilizes base-class learning rather than dramatically improving zero-shot transfer, which somewhat contradicts the primary motivation of enhancing generalization to novel classes.
4.  **Implementation Complexity**: The training pipeline appears complex, involving multiple loss terms (hierarchical alignment, distillation, context prediction) and unsupervised clustering updates. The paper mentions updating visual context embeddings "in a fixed number of rounds divisible by one thousand," but the sensitivity of the method to this update frequency or the number of clusters $K$ (only briefly mentioned in the appendix) could be a concern for reproducibility.

## Questions
1.  **Superclass Derivation**: How exactly are the superclasses defined for the base and novel classes? If the novel classes are truly unseen during training, how is their superclass mapping determined during inference? Is a fixed taxonomy assumed, or does the model infer the superclass?
2.  **Context Calibration Mechanism**: In Equation 7, you use a Hadamard product to calibrate the logits. What is the theoretical or empirical justification for this specific operation? Have you experimented with additive bias or attention-based mechanisms?
3.  **Comparison with Hierarchical Zero-Shot Methods**: How does this work differ from or improve upon standard hierarchical zero-shot learning techniques that have used similar superclass-subclass decompositions? A brief discussion on the novelty relative to prior hierarchical embedding works would be helpful.
4.  **Novel Class Supervision in Context**: The context-aware matrix $M_{dst}$ is supervised by ground-truth annotations. Since novel classes are absent during training, how does the model learn to associate context features with novel class concepts? Does the superclass association serve as the sole bridge for this transfer?

# Actual Human Scores
Individual reviewer scores: [5.0, 5.0, 5.0, 5.0]
Average score: 5.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 8 ===

# Review
# Review: "Can We Ignore Labels in Out-of-Distribution Detection?"

## Summary

This paper investigates a fundamental question in OOD detection: whether self-supervised and unsupervised learning methods can replace supervised approaches for detecting out-of-distribution samples. The authors provide a theoretical analysis showing that such methods are guaranteed to fail when there exists zero mutual information between the learning objective and the task-relevant labels—a phenomenon they term "label blindness."

The main contributions are threefold: (1) the Label Blindness Theorem, proving that SSL/unsupervised OOD detection fails when the surrogate task is independent of label information; (2) the introduction of "Adjacent OOD detection," a new benchmark that tests OOD detection when ID and OOD data have significant overlap in the input space but differ in labels; and (3) empirical validation showing that existing SSL and unsupervised OOD methods perform poorly on this benchmark, contrasting with their strong performance on conventional far/near OOD benchmarks.

## Strengths

1. **Novel theoretical contribution.** The formalization of "label blindness" using information-theoretic arguments is a valuable addition to the OOD detection literature. The proofs connecting information bottleneck principles to OOD detection failure are rigorous and provide concrete conditions for when failures are guaranteed.

2. **Practical motivation.** The Adjacent OOD benchmark addresses a real safety gap that conventional OOD benchmarks overlook. The argument that real-world systems may encounter OOD samples with substantial feature overlap with ID data (e.g., new disease variants, novel objects from existing categories) is compelling and practically relevant.

3. **Theorem 4.1 provides theoretical grounding.** The proof that overlapping OOD data is unavoidable in finite training sets strengthens the argument for why this benchmark matters in practice.

4. **Comprehensive empirical evaluation.** The authors evaluate multiple SSL methods (SimCLR, Rotation Loss), unsupervised methods (Diffusion inpainting), and zero-shot methods (CLIPN) across three datasets (Faces, Cars, Food-101) with consistent methodology.

5. **Clear exposition of limitations.** The paper honestly discusses when unlabeled OOD methods may still be useful and provides guidance for future research directions.

## Weaknesses

1. **Strong theoretical assumptions.** The theoretical results rely on the assumption that features \(x_1\) (relevant to the surrogate task) and \(x_2\) (relevant to labels) are independent. This strict independence is rarely satisfied in real-world data. While the authors acknowledge "approximate label blindness," the connection between theoretical guarantees under strict independence and empirical behavior under approximate independence remains unclear. The paper would benefit from quantifying how sensitive the results are to violations of this independence assumption.

2. **Baseline performance is also poor.** The supervised MSP baseline achieves only ~70% AUROC on the Adjacent OOD benchmark, which is not particularly strong. This suggests the Adjacent OOD task is inherently difficult, but it also raises the question of whether the gap between supervised and unsupervised methods is as dramatic as claimed. The paper should discuss what performance levels are achievable on this benchmark with stronger supervised methods.

3. **Inconsistency with CIFAR10/100 results.** The appendix shows that SSL methods perform substantially better on CIFAR10/100 adjacent OOD tasks compared to Faces/Cars/Food. The AUROC values for SimCLR KNN on CIFAR10 adjacent OOD (73.0%) are much closer to supervised MSP (85.3%) than the gaps observed in other datasets. The authors briefly attribute this to "more visually dissimilar" classes, but this deserves deeper analysis. Does this contradict the theoretical predictions?

4. **Limited analysis of GradCAM visualization.** Figure 1 provides a qualitative example showing that SSL features may not attend to label-relevant regions. However, this is presented as anecdotal evidence without systematic evaluation. Quantifying the attention discrepancy between SSL and supervised representations across multiple examples would strengthen this argument.

5. **Zero-shot method analysis is incomplete.** The paper includes CLIPN as a zero-shot baseline but attributes its varying performance to pretraining data alignment without thoroughly investigating this hypothesis. The analysis in Appendix G showing CC3M examples is useful but remains qualitative.

6. **Missing discussion of mitigation strategies.** While the paper establishes that pure SSL/unsupervised methods can fail, it provides limited guidance on practical solutions. The discussion mentions "few or one shot methods" but does not explore how much label information is needed or how to combine SSL with limited supervision effectively.

## Questions

1. **Independence assumption in practice:** Can the authors provide empirical estimates of the mutual information between features relevant to SSL objectives (e.g., rotation prediction, contrastive similarity) and features relevant to semantic labels for the datasets studied? This would help readers understand whether the theoretical assumptions are approximately satisfied.

2. **Dataset-specific differences:** Why does SimCLR perform relatively well on CIFAR10/100 adjacent OOD compared to Faces/Cars/Food? Is it because SimCLR's learning objective inadvertently captures label-relevant features in these datasets, or is there another explanation?

3. **Quantifying overlap:** How exactly is "overlap" defined and measured in the Adjacent OOD benchmark? The paper describes the construction method but does not provide a quantitative measure of overlap between ID and OOD distributions.

4. **Label efficiency:** The paper suggests incorporating limited label information as a solution. Have the authors conducted preliminary experiments on how many labeled samples are needed to achieve acceptable performance? Even qualitative observations would be valuable.

5. **Comparison with stronger supervised baselines:** Would methods like ODIN, Mahalanobis distance-based detection, or energy-based detection (which are standard supervised OOD methods) substantially outperform MSP on the Adjacent OOD benchmark? This would help contextualize the performance gap.

6. **Extension to other SSL methods:** The experiments focus on SimCLR and Rotation Loss. How would masked autoencoders (MAE) or other reconstruction-based SSL methods perform on Adjacent OOD? These methods may learn different features than contrastive approaches.

---

**Overall assessment:** This paper makes a valuable theoretical contribution by identifying conditions under which SSL-based OOD detection is guaranteed to fail. The Adjacent OOD benchmark is a meaningful addition that exposes weaknesses in current methods. However, the assumptions underlying the theory are strong, and the empirical results show some inconsistency across datasets. Addressing the questions above would significantly strengthen the paper's impact.

# Actual Human Scores
Individual reviewer scores: [6.0, 5.0, 8.0]
Average score: 6.3
Binary outcome: Accept

=== CALIBRATION EXAMPLE 9 ===

# Review
# Review of LocoVR: Multiuser Indoor Locomotion Dataset in Virtual Reality

## Summary

This paper presents LocoVR, a large-scale dataset of two-person indoor locomotion trajectories captured in virtual reality. The dataset contains over 7,000 trajectories across 131 diverse indoor home environments, capturing social navigation dynamics such as proxemics-based behaviors, collision avoidance, and path negotiation. The authors develop a VR-based data collection system where two participants simultaneously navigate shared virtual spaces toward assigned goals. They demonstrate the utility of LocoVR through three downstream tasks: global path prediction, trajectory prediction, and goal area prediction. The results show that models trained on LocoVR outperform those trained on existing datasets (GIMO, THOR-MAGNI) when evaluated on a separate real-world test dataset (LocoReal), suggesting that VR-collected data can effectively transfer to real-world scenarios.

## Strengths

**Novel and Timely Dataset.** The paper addresses a clear gap in existing human motion datasets. While outdoor pedestrian datasets are abundant, large-scale indoor datasets with multi-person interactions are scarce. LocoVR's scale (131 scenes, 7,000+ trajectories) significantly exceeds comparable indoor datasets like GIMO (19 scenes, 187 trajectories) and THOR-MAGNI (4 scenes).

**Social Motion Behaviors.** The focus on two-person social dynamics is valuable for practical applications in home robotics and AI agents. The dataset captures proxemics-driven behaviors (yielding, maintaining social distance, negotiating narrow passages) that are often missing from single-person motion datasets.

**Rigorous Experimental Design.** The evaluation includes three distinct tasks with multiple baselines, an ablation study analyzing dataset scale, multi-person data, and heading information contributions, and cross-dataset generalization experiments (training on different datasets, testing on LocoReal and GIMO).

**Real-World Validation.** The authors created LocoReal, a small real-world test dataset, to demonstrate that models trained on VR data can transfer to physical environments. This addresses a common concern about VR-to-real transfer.

**Comprehensive Dataset Analysis.** The appendix provides detailed statistics on trajectory distances, durations, speeds, and inter-personal distances. The analysis showing ~70% of trajectories involve participants within 2m of each other quantifies the social interaction content.

## Weaknesses

**Model Architecture Limitations.** The baseline models (U-Net and Y-Net) are relatively simple compared to recent advances in trajectory prediction. The paper would benefit from comparisons with modern approaches such as transformer-based models (e.g., Trajectron++, AgentFormer) or graph neural networks that are specifically designed to capture social interactions. Without these comparisons, it is unclear whether the improvements are due to the dataset quality or the chosen architecture's suitability for the task.

**VR-to-Real Domain Gap Underexplored.** While the authors acknowledge this limitation, deeper analysis is needed. The paper mentions differences in walking speed and non-verbal communication but does not quantify these gaps. Key questions remain: How does the distribution of LocoVR trajectories compare to real-world indoor motion patterns? Are social distances maintained similarly in VR versus physical spaces?

**Limited Participant Diversity.** With only 32 participants (21 male, 11 female, ages 18-42), the dataset may not capture the full spectrum of human locomotion patterns. Cultural backgrounds, age-related mobility differences, and physical abilities could significantly affect social navigation behaviors. The paper briefly mentions this in Appendix A but does not quantify potential biases.

**Task Design Constraints.** The goal-reaching task, while controlled, may not capture the diversity of natural indoor behaviors. Real home navigation involves searching for objects, multitasking, interruptions, and varied emotional states. The "at least 2m" goal placement constraint and random goal assignment may produce trajectories that differ from purposeful daily activities.

**Missing Analysis of Social Behavior Types.** The paper claims LocoVR captures socially-motivated movements but lacks quantitative analysis of what types of social behaviors are present and in what proportions. Are there specific annotation labels for "yielding," "passing," or "following"? Without this, researchers cannot selectively train on specific social interaction types.

**Limited Baselines for Some Tasks.** The goal prediction task only compares against U-Net with simple random and nearest baselines. More sophisticated goal prediction methods should be included to better establish the benchmark.

## Questions

1. **Social Behavior Annotation:** Does LocoVR include explicit annotations of social behavior types (e.g., yielding, collision avoidance, following)? If not, are there plans to add such labels, and how can users identify specific social interaction patterns in the data?

2. **Trajectory Prediction Baselines:** Why were more recent trajectory prediction models (e.g., Social-STGCNN, Trajectron++, AgentFormer) not included as baselines? Would these architectures show similar improvements when trained on LocoVR?

3. **VR Behavior Validation:** Have the authors compared VR-collected trajectories against real-world indoor trajectories to quantify the domain gap? Even a small comparison study would strengthen the claim that VR behaviors transfer to real settings.

4. **Scale Ablation:** The ablation study shows performance drops with smaller datasets, but is there a saturation point where additional scenes/trajectories provide diminishing returns? This would help practitioners decide how much data is sufficient for different applications.

5. **Participant Pairing Effects:** Given that some pairs were friends and others were strangers, did the authors observe differences in social navigation behaviors? Is this information encoded in the dataset for analysis?

6. **Multi-Person Scalability:** The paper focuses on two-person interactions. What modifications would be needed to extend this framework to multi-person scenarios (3+ people), and are there plans for such extensions?

7. **Scene Semantic Information:** The paper mentions semantic maps are available. How do semantic scene features (furniture types, room functions) correlate with navigation patterns? This could provide valuable insights for future work.

8. **Training Details:** What is the training time comparison between models trained on LocoVR versus smaller datasets? Is there a trade-off between dataset scale and computational requirements?

## Summary

LocoVR is a valuable contribution to the field of human trajectory prediction and social navigation. The dataset's scale and focus on indoor multi-person interactions fill an important gap. However, the evaluation would be strengthened by including more modern baseline architectures and deeper analysis of the VR-to-real transfer. The paper opens promising directions for future work in socially-aware AI systems for indoor environments.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0, 6.0, 3.0]
Average score: 5.8
Binary outcome: Accept

=== CALIBRATION EXAMPLE 10 ===

# Review
# Review: Fine-tuning Can Help Detect Pretraining Data from Large Language Models

## Summary

This paper proposes Fine-tuned Score Deviation (FSD), a novel method for detecting pretraining data in large language models. The key insight is that fine-tuning a model on a small set of non-member data (data known not to be in the pretraining corpus) creates asymmetric changes in model scores: non-members experience larger perplexity decreases than members. The method computes the deviation between scores from the original and fine-tuned models as a membership signal.

The approach consists of three steps: (1) collect non-member data from the same domain as the target text (e.g., Wikipedia articles published after model release), (2) fine-tune the model using LoRA on this auxiliary data, and (3) use the score deviation between original and fine-tuned models as the membership metric. FSD can be applied on top of existing scoring functions like Perplexity, Min-k%, Zlib, and Lowercase.

The authors evaluate FSD on multiple benchmarks (WikiMIA, ArXivTection, BookMIA, BookTection, Pile) across various open-source LLMs (LLaMA, GPT-J, OPT, Pythia, NeoX), demonstrating substantial AUC improvements over baselines.

---

## Strengths

**Novelty and Insight.** The core observation—that fine-tuning on non-members creates differential perplexity shifts—is intuitive yet underexplored in the literature. The method cleverly leverages temporal information (data timestamps vs. model release dates) to construct reliable non-member sets, which is a practical assumption for many real-world scenarios.

**Strong Empirical Results.** The improvements are substantial across datasets and models. For instance, AUC increases from 0.62 to 0.91 on WikiMIA with OPT-6.7B, and TPR@5%FPR improves from 0.10 to 0.81 on ArXivTection with LLaMA-7B. These are not marginal gains but significant performance jumps.

**Comprehensive Evaluation.** The paper evaluates on multiple datasets (WikiMIA, ArXivTection, BookMIA, BookTection, Pile with 20 subsets), multiple models (5 different architectures), and multiple scoring functions. The ablation studies (fine-tuning data size, different fine-tuning methods, member vs. non-member fine-tuning) are thorough.

**Practical Applicability.** The method is data-efficient—achieving strong results with as few as 30-100 fine-tuning examples—and compatible with any existing scoring function. The assumption that one can collect post-release domain data is realistic for many applications like detecting benchmark contamination or copyrighted books.

**Addressing Temporal Shift.** The authors acknowledge and empirically address the temporal distribution shift issue in WikiMIA by conducting experiments with timestamp removal and replacement. This is a notable strength as recent work has highlighted this confound in existing benchmarks.

---

## Weaknesses

**Assumption on Non-Member Availability.** The method requires collecting non-member data from the *same domain* as the target text. While this works for Wikipedia events and arXiv papers (where publication dates provide clear temporal boundaries), it is less clear how well it generalizes to domains where such boundaries are ambiguous or unavailable. The paper could more explicitly discuss failure modes when domain-specific non-member data is scarce.

**Computational Overhead.** Fine-tuning—even with LoRA—adds computational cost compared to standard scoring methods. The paper does not provide timing or resource comparisons. For practitioners deciding between methods, understanding this trade-off is important.

**Limited Analysis on False Positives.** While AUC and TPR@5%FPR are reported, the paper lacks analysis on *why* certain members and non-members are misclassified. Understanding failure cases would strengthen the paper.

**Baselines Are Sometimes Near Random.** On the Pile dataset (Table 2), baseline methods achieve AUC ~0.50, suggesting near-random performance. This raises questions about the baseline implementations or the inherent difficulty of the dataset. The authors should comment on why baselines perform so poorly and whether FSD's gains are more about fixing baseline failures than a fundamental improvement.

**Dependence on Fine-Tuning Quality.** The ablation on fine-tuning parameters (Table 17) shows sensitivity to learning rate. More discussion on robustness to hyperparameter choices and potential for overfitting to the fine-tuning set would be valuable.

---

## Questions

1. **Domain Mismatch Robustness.** In Table 16, the method fails when fine-tuning on WikiMIA and evaluating on ArXivTection. Could the authors quantify how much domain overlap is necessary for FSD to be effective? Is there a principled way to estimate domain similarity?

2. **Hyperparameter Sensitivity.** The results in Table 17 show learning rate sensitivity. Could the authors provide guidance on learning rate selection without access to ground-truth validation labels? In realistic deployment, how would one tune these hyperparameters?

3. **Threshold Selection.** The paper mentions determining the threshold ϵ by maximizing accuracy on a validation set. How sensitive is performance to this threshold? How many labeled examples are needed for threshold calibration?

4. **Comparison to Reference-Based Methods.** Recent work on reference-based MIAs (e.g., using smaller reference models) also aims to improve membership inference. Could the authors compare FSD to these approaches or discuss complementarity?

5. **Non-Member Contamination.** What happens if some collected "non-member" data inadvertently contains members (e.g., due to incomplete temporal information)? How robust is FSD to noise in the fine-tuning set?

6. **Computational Cost.** Could the authors provide wall-clock time comparisons between FSD and baseline methods? This would help readers understand the practical cost-benefit trade-off.

7. **Effect of Fine-Tuning Duration.** The paper uses 3 epochs. Would fewer epochs (less overfitting to non-members) or more epochs change the relative score deviation in predictable ways?

---

## Overall Assessment

This paper presents a clever and well-motivated approach to pretraining data detection with strong empirical results. The core insight—that fine-tuning on non-members reveals membership information—is intuitive and effectively leveraged. The method is simple, data-efficient, and broadly applicable. However, the practical limitations around domain-specific non-member availability and the computational overhead merit clearer discussion. The paper would benefit from additional analysis of failure cases and hyperparameter robustness.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 6.0, 5.0]
Average score: 6.2
Binary outcome: Accept

=== CALIBRATION EXAMPLE 11 ===

# Review
# Review: Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning

## Summary

This paper introduces **Process Advantage Verifiers (PAVs)**, a novel approach to designing process rewards for improving LLM reasoning via test-time search and online reinforcement learning. The key insight is that effective process rewards should measure **progress**—the change in likelihood of producing a correct response before and after taking a reasoning step—computed under a **prover policy** distinct from the base policy being optimized.

The authors argue that conventional process reward models (PRMs) based on Q-values have fundamental limitations: Q-values conflate the "promise" of previous states with the value of the current action, making it difficult to distinguish good steps from promising prefixes. Instead, using **advantages** (the difference between Q-values of consecutive states) provides a cleaner measure of step-level progress.

A central theoretical contribution is the characterization of **complementary provers**: policies that can distinguish steps taken by the base policy (high variance in advantage values) while remaining reasonably aligned with the base policy's preferences. Theorem 3.1 provides a lower bound on policy improvement, showing improvement scales with prover distinguishability and alignment. Notably, the analysis reveals that weaker provers can sometimes improve stronger base policies—a counter-intuitive finding with practical implications.

Empirically, the paper demonstrates:
- **Test-time search**: PAVs achieve >8% higher accuracy and 1.5-5× better compute efficiency compared to ORM-based best-of-N re-ranking across Gemma 2B/9B/27B models
- **Online RL**: Dense rewards from PAVs yield 6× better sample efficiency and +7% accuracy improvement over outcome-only rewards

## Strengths

**1. Conceptual Clarity and Novel Framework.** The paper provides a principled theoretical framework for understanding why advantages (progress) rather than Q-values should be used as process rewards. The distinction between "exploiting" high-value states versus "exploring" steps that make progress is clearly articulated and motivated through both intuitive examples (Figure 2) and formal analysis.

**2. Theoretical Grounding.** Theorem 3.1 and the subsequent analysis provide formal justification for the design choices. The characterization of complementary provers through distinguishability and alignment terms offers concrete guidance for prover selection. The observation that Best-of-K policies provide a natural prover class with theoretical guarantees (Remark 3.1) is practically useful.

**3. Counter-Intuitive Empirical Findings.** The result that weaker provers (e.g., 2B for a 9B base) can outperform stronger provers is intriguing and well-supported by both theory and experiments. This challenges the natural assumption that stronger verifiers always help more.

**4. Comprehensive Evaluation.** The paper evaluates both test-time search and online RL applications, showing consistent benefits. The comparison with prior PRM approaches (Math-Shepherd, etc.) and the ablation over prover strengths adds depth.

**5. Practical Workflow.** Appendix D provides a clear recipe for collecting training data and training PAVs, including guidance on the n_mc/n_cov ratio and the "first pit" sampling strategy.

**6. Connection to Reward Shaping.** Appendix I correctly identifies that PAV advantages satisfy the potential-based reward shaping condition, ensuring policy invariance—a theoretically sound property.

**7. Didactic Analysis.** The planted sub-sequence experiment in Section 3.3 provides clean empirical validation of theoretical claims in a controlled setting.

## Weaknesses

**1. Limited Evaluation Scope.** All experiments are conducted on the MATH dataset. While math reasoning is an important benchmark, the paper would benefit from evaluation on additional reasoning domains (e.g., code generation, logical reasoning, or multi-step tasks like GSM8K, AIME problems) to assess generalizability.

**2. Model Scale Limitations.** The largest model tested is Gemma 27B. Given current frontier model capabilities, it remains unclear whether PAVs would provide similar benefits for models in the 70B-500B parameter range, or for proprietary models like GPT-4/Claude where the prover-base policy relationship may differ.

**3. Hyperparameter Sensitivity.** The paper acknowledges tuning α (the weighting between Q-values and advantages) for each base policy, but provides limited guidance on how to set it for new domains or models. The "robust range" claimed is still quite wide (0.1-0.7 for search, 1.5-5.5 for RL), and practitioners need validation sets for tuning.

**4. Prover Selection Remains Open.** While BoK policies with K>1 are identified as reasonable provers, the paper does not provide a principled method for selecting the optimal K. Figure 5(b) shows K=4 is best for one setting, but this requires empirical tuning. The question of whether prover policies should evolve alongside the base policy during RL is not addressed.

**5. Computational Overhead Under-Reported.** Appendix H discusses training costs but presents the comparison somewhat favorably. PAV training requires ~10× more data than ORM training (due to per-step annotations), and scoring requires processing all prefixes rather than just the final output. While amortized costs are claimed to be favorable, practitioners should be aware of upfront costs.

**6. Simplified Theoretical Setting.** Theorem 3.1 is proven in the tabular setting with oracle access to Q-values and advantages. The gap between this idealized analysis and the practical setting where verifiers are learned (with potential approximation errors) is not formally addressed.

**7. Limited Discussion of Failure Modes.** The paper focuses on positive results but provides minimal discussion of when PAVs might fail. For instance, Appendix G shows that using Q[μ] instead of A[μ] leads to degenerate "REPHRASE THE PROBLEM" behaviors, but more analysis of failure modes would be valuable.

**8. Missing Comparisons.** The paper compares against PRM baselines but does not compare against other exploration strategies for LLMs, such as intrinsic motivation methods, curiosity-driven exploration, or recent tree search methods beyond beam search.

**9. Discount Factor Experiment Under-Explained.** The experiment in Appendix E on discounted strong provers is interesting but the negative result is not thoroughly analyzed. Why exactly does discounting not help strong provers distinguish steps?

**10. Data Collection Complexity.** The "first pit" strategy requires substantial engineering and hyperparameter decisions (thresholds for "high value" states, etc.). The paper could benefit from more ablations on these design choices.

## Questions

**1. Prover Evolution During RL.** As the base policy improves during online RL, the optimal prover may change. The paper uses fixed PAVs during RL training. Have the authors experimented with periodically updating the prover as the base policy improves? Would this provide additional gains?

**2. Scaling to Frontier Models.** Given the theoretical requirement that provers be "complementary" to base policies, how would the authors recommend applying PAVs when the base policy is already near frontier-level? Would using a stronger model (e.g., GPT-4) as prover for a weaker base model still satisfy complementarity, or would the alignment term suffer?

**3. Sensitivity to K in BoK Provers.** Remark 3.1 suggests K>1 is beneficial, but empirical results show an optimal K around 4. Is there a theoretical or practical guidance for selecting K that generalizes across model families and scales?

**4. Computational Cost for Large-Scale Deployment.** For practitioners considering PAVs, what is the rough breakdown of computational costs: (a) training data collection (rollouts), (b) PAV training, (c) test-time inference? How do these compare to ORM-based approaches end-to-end?

**5. Handling Multi-Step Reasoning with Variable Step Counts.** The current formulation assumes a fixed maximum number of steps H. How does the approach handle problems where optimal reasoning requires highly variable step counts?

**6. Robustness to Distribution Shift.** PAVs are trained on a specific distribution of problems. How robust are they to out-of-distribution test problems? Have the authors evaluated on held-out mathematical domains or different reasoning tasks?

**7. Relationship to Other RL Techniques.** The advantages A[μ] resemble baseline subtraction in actor-critic methods. How does PAV training compare to using learned value functions as baselines in standard policy gradient methods for LLMs?

**8. Explainability of Prover Selection.** Figure 5(c) shows that 9B policy benefits from 2B prover, and 27B benefits from 9B prover. Is there a pattern (e.g., optimal prover is ~3× smaller than base) that generalizes? What about the case where base and prover have similar scales?

**9. Comparison with AlphaLLM Exploration.** Appendix K shows PAVs outperform AlphaLLM's uncertainty-based exploration. However, the comparison uses the authors' implementation of AlphaLLM. Would it be possible to compare against the original AlphaLLM implementation or other recent exploration methods?

**10. Reproducibility of α Tuning.** The paper describes a two-layer search procedure for α. Could the authors provide more details on how many validation examples and compute were required for this tuning? Is there potential to develop a theoretical guidance for α based on policy/prover properties?

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 8.0, 8.0, 6.0]
Average score: 7.1
Binary outcome: Accept

=== CALIBRATION EXAMPLE 12 ===

# Review
# Review of "ADIFF: Explaining Audio Difference Using Natural Language"

## 1. Summary

This paper introduces a new task—audio difference explanation—which aims to generate natural language descriptions comparing two audio files. The authors make several contributions: (1) They create two new datasets (ACD and CLD) derived from AudioCaps and Clotho, using LLMs to generate three tiers of explanations ranging from concise event descriptions to comprehensive analyses including semantics and listener emotions. (2) They propose ADIFF, a prefix-tuning based model that incorporates a cross-projection module with separator tokens, position captioning for better audio grounding, and a three-stage training process. (3) They conduct extensive ablations studying cross-projection effects, language model scaling, and training stages. The model is evaluated using both standard captioning metrics and human evaluation, showing improvements over a naive baseline and the Qwen-Audio model.

## 2. Strengths

- **Novel task formulation**: The paper identifies an important and previously unexplored problem—generating natural language explanations of audio differences. This has practical applications in forensics, quality assessment, and generative audio evaluation.

- **Comprehensive dataset creation**: The authors create two datasets with three tiers of explanations of increasing complexity, providing a benchmark for future work. The human verification of test sets adds reliability.

- **Thoughtful model design**: The cross-projection layer with separator tokens is a sensible approach to help the model distinguish between two audio inputs. The analysis showing that this layer stores difference attributes in the text prefix is insightful.

- **Thorough ablation studies**: The paper systematically investigates multiple architectural choices (cross-projection, LM scaling, position captioning, fine-tuning stages), providing useful insights for future audio-language model development.

- **Interesting finding on LM scaling**: The observation that smaller LMs (128M, 256M) perform better than larger ones (774M, 1.5B) under limited compute is counter-intuitive and valuable for practitioners with resource constraints.

- **Multiple evaluation methods**: Combining objective metrics with human evaluation across multiple dimensions (correctness, granularity, readability) provides a more complete picture than metrics alone.

## 3. Weaknesses

- **Dataset quality concerns**: The training data is entirely LLM-generated from existing captions, not human-annotated for this specific task. While test sets are human-verified, the quality of training data—and potential propagation of LLM hallucinations into the training set—is not analyzed. The authors should report what percentage of generated explanations required modification during verification.

- **Limited baselines**: Qwen-Audio is the only compared Audio-Language Model, selected primarily because it supports dual audio inputs. The paper would benefit from implementing additional baselines, such as concatenating audio embeddings and feeding them to other ALMs, or using a single ALM to independently caption both audios and then using an LLM to compare them.

- **Unfair comparison with Qwen-Audio**: The paper compares ADIFF (trained on ACD/CLD) against Qwen-Audio zero-shot. While the fine-tuned Qwen-Audio variants are included, it's unclear whether the comparison is fair given Qwen-Audio's 7B parameters versus ADIFF's 128M LLM. The human evaluation shows Qwen-Audio (F) outperforming ADIFF in some metrics (readability, granularity on FSD50K), suggesting the larger model may have advantages the paper downplays.

- **Tier 3 performance**: The model struggles significantly with detailed (Tier 3) explanations, achieving notably lower SPIDEr scores compared to Tiers 1 and 2. This limits practical utility for applications requiring comprehensive audio analysis.

- **Hallucination not quantified**: The paper mentions hallucination detection using audio event probabilities from HTSAT, but provides no quantitative analysis of hallucination rates or comparison between models.

- **Human evaluation methodology unclear**: The paper mentions five professionals for human evaluation but provides no information on inter-annotator agreement, which is critical for subjective evaluations.

- **Position captioning results mixed**: While position captioning improves ACD results, it yields mixed results on CLD. The explanation that AudioCaps contains more speech is reasonable, but this suggests the approach may not generalize well to diverse audio types.

- **Missing analysis on similar audio pairs**: A key challenge mentioned is distinguishing perceptually similar sounds, but there's no dedicated analysis on how well the model handles audio pairs that are genuinely similar versus obviously different.

## 4. Questions

1. **Training data quality**: What percentage of LLM-generated explanations required modification during human verification? Were there systematic issues in the generated explanations?

2. **Human evaluation details**: What was the inter-annotator agreement (e.g., Krippendorff's alpha) across the three dimensions? How were disagreements resolved?

3. **Dataset pair selection**: How were audio pairs selected for the datasets? Were they random pairs, or were they curated to include challenging comparisons (similar sounds, same audio events)?

4. **Hallucination analysis**: Can you provide quantitative analysis of hallucination rates across models? Does cross-projection reduce hallucinations?

5. **Computational cost**: What are the training and inference costs of ADIFF compared to Qwen-Audio fine-tuning?

6. **Error analysis**: Beyond qualitative examples, can you provide systematic error analysis? What types of audio pairs or difference types does the model consistently fail on?

7. **Tier 2 anomaly**: Why does Tier 2 consistently achieve higher scores than Tiers 1 and 3? The explanation about linguistic structure is plausible, but shouldn't the ground truth quality affect this?

8. **Position captioning**: Does the model trained with position captioning actually learn to distinguish "audio 1" from "audio 2," or does it still make attribution errors?

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept

=== CALIBRATION EXAMPLE 13 ===

# Review
# Review of "SAM 2: Segment Anything in Images and Videos"

## Summary

This paper presents SAM 2, a foundation model for promptable visual segmentation in both images and videos. The work extends the original Segment Anything Model (SAM) from static images to the video domain through three main contributions:

1. **Task Definition**: The Promptable Visual Segmentation (PVS) task, which generalizes image segmentation to video by allowing prompts (clicks, boxes, masks) on any frame to define objects that are tracked throughout the entire video.

2. **Model Architecture**: A unified transformer architecture with streaming memory that processes video frames sequentially. Key components include a Hiera image encoder, memory attention modules for conditioning on past frames, a memory bank (storing spatial features and object pointers), and a mask decoder. The model handles occlusion prediction and multi-mask ambiguity resolution.

3. **Data Engine and Dataset**: An iterative data collection pipeline using model-in-the-loop annotation across three phases, producing the SA-V dataset containing 50.9K videos with 642.6K masklets (35.5M masks total)—significantly larger than existing video segmentation datasets.

The experiments demonstrate that SAM 2 achieves state-of-the-art performance on video object segmentation benchmarks (DAVIS, MOSE, LVOS), interactive video segmentation tasks, and also improves upon SAM for image segmentation while being 6× faster. The model shows strong zero-shot transfer capabilities across 17 video and 37 image datasets.

## Strengths

**Novelty and Contributions:**
- The PVS task formulation elegantly unifies interactive image segmentation and semi-supervised video object segmentation, providing a coherent framework for both domains.
- The streaming memory architecture is a thoughtful generalization of SAM to video, enabling real-time processing of arbitrarily long videos while maintaining temporal consistency.
- The introduction of object pointers as lightweight semantic representations, combined with spatial memory features, provides an effective mechanism for maintaining object identity across frames.

**Dataset and Data Engine:**
- The SA-V dataset represents a substantial contribution—at 53× larger than previous VOS datasets in terms of mask count, it significantly advances resources available for video segmentation research.
- The focus on object parts (not just whole objects) and the inclusion of challenging scenarios (occlusions, disappearances) makes the dataset particularly valuable for developing general segmentation capabilities.
- The three-phase data engine demonstrates practical efficiency gains (8.4× faster annotation than per-frame SAM).

**Experimental Rigor:**
- Comprehensive evaluation across 54 datasets (17 video + 37 image) provides convincing evidence of generalization.
- The offline and online interactive evaluation protocols fairly simulate real-world annotation scenarios.
- Ablation studies systematically justify architectural choices (memory size, positional encoding, object pointers).
- Fairness evaluation across demographic groups is a welcome addition often missing from segmentation papers.

**Reproducibility and Impact:**
- Open-source release of models, dataset, and code under permissive licenses.
- Detailed model, dataset, and annotation cards following established documentation standards.

## Weaknesses

**Methodological Concerns:**

1. **Baseline Comparison Strategy**: The paper constructs baselines by combining SAM with XMem++/Cutie, which while reasonable, may not represent optimal integration. For instance, the baseline appears to restart tracking from scratch when corrections are needed, whereas SAM 2's memory allows incremental refinement. This design choice favors SAM 2's architecture but doesn't necessarily reflect what a purpose-built interactive VOS system could achieve.

2. **Training Sequence Length**: The model is trained on 8-frame sequences (with 16-frame fine-tuning), yet evaluated on much longer videos. While the streaming architecture theoretically handles arbitrary lengths, the short training sequences may not fully prepare the model for long-range temporal dependencies, particularly for re-identification after extended occlusions.

3. **Auto Masklets**: A significant portion of the training data (451.7K/642.6K masklets) comes from automatic generation using earlier versions of SAM 2. While verified by annotators, this bootstrapping approach could propagate model biases and create a confirmation loop where the model learns to reproduce its own failure modes in less challenging cases.

4. **Object-Level Independence**: SAM 2 processes each object independently, missing opportunities for inter-object reasoning that could help in disambiguating similar objects or handling crowded scenes—limitations acknowledged in Appendix C but not extensively analyzed.

**Evaluation Concerns:**

5. **Internal Dataset**: Training includes an "Internal" dataset (62.9K videos) unavailable to the community. This makes full reproducibility impossible and raises questions about what unique characteristics this data provides. The paper should clarify what performance gap would result from training only on publicly available data.

6. **Limited Baseline Comparison for Image Tasks**: For the Segment Anything image task, comparisons focus on SAM and HQ-SAM. Given the substantial literature on promptable segmentation, including additional recent methods would strengthen the evaluation.

7. **Click Sampling Protocol**: The interactive evaluation uses an Oracle protocol where clicks are placed optimally based on ground-truth error regions. While this provides a consistent evaluation framework, it may overestimate practical usability where users must identify errors visually.

**Missing Analyses:**

8. **Memory Scalability**: The paper lacks analysis of how memory requirements scale with video length and number of tracked objects. The FIFO queue with N=6 recent frames may be insufficient for very long videos or multi-object scenarios.

9. **Real-world Deployment Considerations**: The paper claims "real-time" processing (43.8 FPS for Hiera-B+), but this is measured on A100 GPUs. Discussion of CPU/GPU performance for deployment scenarios (mobile, edge devices) is absent.

10. **Comparison with Prompt-Based Tracking**: Recent works on promptable video tracking (e.g., point tracking methods) share conceptual similarities with PVS. The paper would benefit from discussing relationships to this broader literature.

## Questions

1. **Memory Bank Design**: The memory bank maintains N=6 recent frames and M prompted frames. What happens when M exceeds the design limit in very long interactive sessions? Is there a mechanism for intelligent memory pruning or prioritization?

2. **Training on Auto-Generated Masklets**: Can the authors provide statistics on the quality difference between manually annotated and auto-generated masklets? How sensitive is model performance to the proportion of auto-generated data?

3. **Internal Dataset Impact**: What performance degradation would occur if training excluded the internal dataset? This is critical for reproducibility claims.

4. **Multi-Object Efficiency**: For scenes with many objects, processing each independently with separate memory banks seems inefficient. Have the authors explored sharing computations across objects, and what trade-offs exist?

5. **Long-Range Dependencies**: The 16-frame fine-tuning addresses longer sequences partially, but videos can be hundreds of frames. How does performance degrade with video length, and is there a critical length beyond which the model's streaming approach shows limitations?

6. **Occlusion Handling**: The occlusion prediction head is mentioned but not extensively analyzed. What is the accuracy of occlusion detection, and how do occlusion prediction errors propagate through the memory bank?

7. **Temporal Coherence Without Explicit Motion Modeling**: The paper mentions that incorporating explicit motion modeling could help with similar-looking objects. Have the authors experimented with optical flow integration, and what were the results?

8. **Dataset Splits and Potential Contamination**: The SA-V val/test sets are annotated using Phase 1 (per-frame SAM), while training uses Phase 3 (full SAM 2). While this mitigates some bias, could there be systematic differences in object types or video characteristics between splits?

---

**Overall Assessment**: This paper makes significant contributions to video segmentation through a unified architecture and a large-scale dataset. The methodology is generally sound, and the empirical results are impressive. However, the concerns above—particularly regarding baseline comparison, data provenance (internal dataset), and certain practical considerations—should be addressed to strengthen the work. The release of models and data will undoubtedly catalyze further research in this area.

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 8.0, 10.0]
Average score: 9.0
Binary outcome: Accept

=== CALIBRATION EXAMPLE 14 ===

# Review
## Summary

This paper investigates the robustness of safety alignment in Large Language Models (LLMs) and identifies a unified root cause for various vulnerabilities: **shallow safety alignment**. The authors argue that current alignment methods primarily shift the model's distribution over the first few output tokens to produce refusal prefixes, while leaving the harmful content distribution for subsequent tokens largely unmodified. This "shortcut" explains why models are susceptible to prefilling attacks, adversarial suffixes (GCG), decoding parameter manipulation, and fine-tuning attacks.

The paper makes three main contributions:
1.  **Characterization:** It empirically demonstrates that aligned models differ from base models primarily in the KL divergence of the first few tokens, and that unaligned models become "safe" simply by forcing refusal prefixes. It analyzes fine-tuning dynamics to show that gradients are largest for initial tokens.
2.  **Deep Alignment Method:** It proposes a data augmentation strategy ("safety recovery examples") where models are trained to refuse after starting with harmful prefixes. This "deepens" the alignment and empirically improves robustness against inference-time attacks.
3.  **Constrained Fine-tuning Defense:** It introduces a token-wise constrained loss function to preserve safety during downstream fine-tuning by constraining the distribution shift of the first few tokens. This effectively mitigates harmful fine-tuning attacks while preserving utility on benign tasks.

## Strengths

**Conceptual Unification:** The paper provides a compelling and intuitive unifying explanation ("shallow alignment") for a wide array of disparate vulnerabilities. Connecting prefilling attacks, GCG attacks, and fine-tuning attacks through a single mechanism (the distribution of initial tokens) is a significant insight that advances the understanding of LLM safety failure modes.

**Strong Empirical Diagnosis:** The analysis is grounded in strong empirical observations. Figure 1 (KL divergence concentrated on early tokens) and Table 1 (unaligned models appearing safe with forced refusal prefixes) are simple yet powerful experiments that effectively validate the core hypothesis. The analysis of per-token gradient norms during fine-tuning (Figure 3) further strengthens the argument.

**Practical Mitigations:** The proposed solutions are well-motivated by the diagnosis. The constrained fine-tuning objective is particularly novel and practically relevant for platforms offering fine-tuning APIs. The theoretical derivation of the loss function (limiting behaviors) is rigorous and adds depth to the methodology section.

**Comprehensive Evaluation:** The authors evaluate their proposed defenses against a variety of threats (GCG, prefilling, decoding exploits, harmful fine-tuning, identity shifting, backdoors). The preservation of utility on standard benchmarks (MMLU, GSM8K, etc.) is also thoroughly checked.

## Weaknesses

**Incompleteness of the Defense:** While the proposed "deep alignment" improves robustness against GCG attacks, the Attack Success Rate (ASR) remains notable (e.g., ~18-20% on AdvBench in Table 3). This suggests that while deepening alignment is necessary, it may not be sufficient for robust defense against stronger adaptive attacks.

**Limited Evaluation of Stronger Attacks:** The paper focuses on GCG and prefilling for inference attacks. It does not evaluate against more complex, recent jailbreaks like TAP (Tree of Attacks with Pruning), AutoDAN, or multi-modal attacks. A more comprehensive robustness evaluation would strengthen the claims about the efficacy of deep alignment.

**Potential Utility Trade-offs:** The paper claims utility preservation, but the results in Table 4 show a noticeable drop in accuracy for GSM8K (41.7% $\to$ 37.4%) when using the constrained loss. This suggests a tension between preserving the initial token distribution and learning complex reasoning tasks. The authors should discuss this trade-off more explicitly.

**Heuristic Nature of Data Augmentation:** The data augmentation strategy relies on synthetic "safety recovery" examples. The quality of these examples depends on the specific harmful prefixes used. The paper uses prefixes from a specific dataset; it is unclear how sensitive the method is to the diversity or quality of these prefixes. Additionally, training on incoherent text (harmful prefix followed by refusal) might have unintended effects on model coherence in other contexts.

**Evaluation of Adaptive Attacks for Constrained Fine-tuning:** The constrained loss assumes the defender controls the training process. However, if the attacker knows the defense, could they not fine-tune on data that specifically targets later tokens or uses a different "harmful" prefix structure to bypass the constraint? The robustness against an adaptive attacker aware of the defense mechanism is not discussed.

## Questions

1.  **Generalization of Data Augmentation:** How does the "deep alignment" model generalize to harmful prompts or attack types not seen during the augmentation? Does the model learn a general "refusal after harm" behavior, or does it overfit to the specific recovery examples?
2.  **Adaptive Attacks on Constrained Loss:** If an attacker is aware of the constrained loss defense, could they construct a fine-tuning dataset where the harmful content is delayed (e.g., "Sure, [filler text], here is the bomb...") or structure the attack to optimize later tokens? How would the constrained loss perform in such a scenario?
3.  **Utility Trade-off Details:** You report a drop in GSM8K accuracy. Do you observe similar drops in tasks that require specific output formatting (e.g., code generation or translation) where the initial tokens might carry significant semantic weight?
4.  **Comparison with Other Defenses:** How does your constrained fine-tuning approach compare to parameter-efficient fine-tuning methods (e.g., LoRA) or other defense mechanisms specifically designed for fine-tuning (e.g., SafeLoRA) in terms of safety-utility trade-offs?
5.  **Scalability of KL Measurement:** You use KL divergence to measure alignment depth. Is this metric robust across different model architectures or tokenizers? For example, does a single token in Llama-2 carry the same semantic weight as in Gemma?

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 10.0, 10.0]
Average score: 9.5
Binary outcome: Accept
