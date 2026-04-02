=== CALIBRATION EXAMPLE 1 ===

# Review
# Review of "Systematic Review of Large Language Models: Applications, Limitations, Practical Usages and Future Directions"

## Summary

This paper presents itself as a systematic review of Large Language Models (LLMs), covering their types (generative, masked, seq2seq, and hybrid models), technical architectures, applications (translation, summarization, text generation), limitations (bias, resource intensity, interpretability), ethical considerations, evaluation metrics, and future directions. The paper attempts to provide a comparative analysis of major LLMs including BERT, GPT-3, and T5 through a summary table. While the paper has reasonable breadth, it lacks depth, systematic methodology, and any novel contribution to the field.

## Strengths

1. **Broad Coverage**: The paper covers a wide range of LLM topics, from foundational architectures to ethical considerations, making it a reasonable starting point for newcomers to the field.

2. **Clear Structure**: The paper is organized into logical sections (Types, Literature Review, Limitations, Ethics, etc.) that provide a navigable overview of the LLM landscape.

3. **Inclusion of Figures**: The paper includes diagrams illustrating BERT input representations and Transformer architecture, which can be helpful for educational purposes.

4. **Comparative Table**: Table 1 attempts to compare different LLMs across dimensions (pros, cons, datasets, metrics), which is useful for high-level understanding.

5. **Inclusion of Recent Topics**: The paper touches on relatively recent topics such as hallucination, adversarial robustness, and multimodal applications (CLIP).

## Weaknesses

### Critical Methodological Issues

1. **No Systematic Review Methodology**: The paper claims to be a "systematic review" but lacks any of the standard protocols: no search strategy is described, no inclusion/exclusion criteria are provided, no PRISMA guidelines are followed, and no protocol was registered. This is a fundamental misrepresentation.

2. **No Novel Contribution**: The paper does not present any new analysis, methodology, experiments, or insights. It is purely a descriptive summary of existing literature without any critical synthesis or original contribution.

### Major Content and Organization Problems

3. **Severe Duplicate Content**: Sections 3.1 and 3.3 contain identical text for "Denoising Autoencoders" and "Masked Language Modeling." Section 3.3 is malformed—it begins discussing BERT's contributions but then abruptly switches to duplicated content from Section 3.1, followed by a partial figure caption that cuts off mid-sentence. This suggests either extreme carelessness or automated text generation without proper editing.

4. **Incomplete/Missing Citations**: Multiple placeholders remain in the text, including "**?**" and "?**" in Sections 1 and 2. These should have been replaced with proper citations before submission.

5. **Figure 2 is Illegible**: The caption and content for Figure 2 ("Transformer architecture and training objectives") contain garbled text ("Textime 7 Text Task Classification..." and "200 201 202203 Multiple..."), making it completely unusable.

6. **Figure 1 Plagiarism Concerns**: Figure 1 (BERT input representation) appears to be reproduced directly from the original BERT paper (Devlin et al., 2018) without any transformation or proper attribution beyond a general citation.

### Missing Coverage of Important Topics

7. **No Discussion of Instruction Tuning**: The paper completely omits discussion of instruction tuning, RLHF (Reinforcement Learning from Human Feedback), DPO (Direct Preference Optimization), or similar techniques that are fundamental to modern LLM development (e.g., InstructGPT, ChatGPT, GPT-4).

8. **No Coverage of Open-Source Models**: Models like LLaMA, LLaMA-2, Mistral, Falcon, and their derivatives—which have been transformative for the field—are entirely absent.

9. **Missing Chain-of-Thought and Reasoning**: The paper does not discuss chain-of-thought prompting, reasoning capabilities, or mathematical problem-solving in LLMs.

10. **No Discussion of Tool Use and Agents**: Modern LLM applications involving tool use, function calling, and autonomous agents are not covered.

11. **Missing Major Benchmarks**: Important benchmarks such as MMLU, HumanEval, BIG-Bench, GSM8K, and HELM are only briefly mentioned without meaningful discussion. The LMSYS Chatbot Arena is mentioned but not critically evaluated.

12. **No Coverage of Fine-Tuning Advances**: Beyond the brief mention of adapters, the paper does not cover LoRA, QLoRA, or other parameter-efficient fine-tuning methods that have democratized LLM customization.

### Technical Depth Issues

13. **Surface-Level Treatment**: The paper describes concepts at a superficial level without deep technical analysis. For example, the section on "Adversarial Robustness" is only four sentences long and lacks any technical substance.

14. **No Critical Analysis**: The paper presents information without critically evaluating findings or synthesizing conflicting results across studies.

15. **Missing Quantitative Synthesis**: A systematic review should ideally include meta-analysis or at least quantitative summaries of findings; this paper has neither.

### Formatting and Presentation Issues

16. **Inconsistent Citation Style**: Some citations are formatted inconsistently (e.g., "Vaswani (2017)" vs. "Vaswani (2017)" without proper author names in references).

17. **Table 1 Issues**: The table compares LLMs but includes only three models (BERT, GPT-3, T5) and omits many others. The columns are unbalanced, and BART (discussed extensively in the text) is missing from the comparison table.

18. **Repetitive Text**: Sections 9 and 10 (Future Directions and Conclusion) contain significant overlap in phrasing and content.

## Questions for Authors

1. What was the systematic review protocol used for this paper? Please provide the search strategy, databases searched, inclusion/exclusion criteria, and PRISMA flow diagram.

2. Why does Section 3.3 contain duplicate content from Section 3.1? This appears to be a significant editing error.

3. The paper contains placeholder citations marked with "?". Could you clarify which papers these should reference?

4. Figure 2 contains garbled text. Please provide the correct version of this figure.

5. Why are instruction tuning methods (RLHF, DPO) and their importance for modern LLMs completely absent from this review?

6. Why are open-source models (LLaMA, Mistral, Falcon) not included despite their significant impact on the field?

7. What is the target audience for this paper? Given its surface-level treatment, it does not appear suitable for researchers or practitioners familiar with LLMs.

## Recommendation

This paper requires substantial revision before it could be considered for publication. The most critical issues are:

1. The paper must either follow proper systematic review methodology or rename itself as a "survey" or "overview" paper.

2. All duplicate content, placeholder citations, and corrupted figures must be corrected.

3. The paper must include coverage of modern developments including instruction tuning, open-source models, and current state-of-the-art benchmarks.

4. The authors should provide original analysis and synthesis rather than merely summarizing existing literature.

Without addressing these fundamental issues, this paper does not meet the standards expected for a top ML venue.

# Actual Human Scores
Individual reviewer scores: [1.0, 1.0, 1.0, 1.0]
Average score: 1.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 2 ===

# Review
## Review: Differentiation of Multi Objective Data Driven Decision Pipeline

### 1. Summary

This paper proposes **Multi-Objective Decision-Focused Learning (MoDFL)**, a framework that extends decision-focused learning (DFL) from single-objective to multi-objective optimization problems (MOPs). The key contribution is a set of three novel loss functions designed for MOPs:

1. **Landscape loss** (based on sample rank-based Maximum Mean Discrepancy/sRMMD): Measures discrepancy between objective spaces
2. **Pareto set loss**: Measures distance between Pareto sets in solution space using minimum distance approximation
3. **Decision loss**: Uses weighted sum transformation to obtain a representative solution and measures its quality

The authors combine these losses (weighted by hyperparameters λₗ, λ_d, λ_ps) and use Differentiation of Smooth Linear Programming (DSLP) to enable end-to-end training through the optimization layer. Experiments on web advertisement allocation and bipartite matching benchmarks demonstrate improved performance over two-stage methods and existing DFL baselines.

### 2. Strengths

- **Addresses an important gap**: The paper tackles the under-explored problem of extending DFL to multi-objective settings, which is relevant for many real-world applications with conflicting objectives.
- **Thoughtful loss design**: The three-pronged approach (landscape, Pareto set, and decision loss) captures different aspects of multi-objective quality: objective space distribution, solution set coverage, and representative decision quality.
- **Solid theoretical grounding**: Uses principled techniques—entropy-regularized optimal transport (sRMMD), KKT conditions (DSLP), and reparameterization—for gradient computation.
- **Comprehensive evaluation**: Multiple benchmarks (web ad allocation, bipartite matching with 2 and 3 objectives), comparison against 7 baseline methods, and ablation studies validating each component's contribution.
- **Ablation analysis**: Demonstrates that each loss component contributes positively, with decision loss having the largest impact.

### 3. Weaknesses

**Methodological concerns:**

1. **Weighted sum limitation**: The paper uses weighted sum transformation for decision loss, which is known to fail for non-convex Pareto fronts (cannot represent all Pareto-optimal solutions). The paper does not discuss or evaluate this limitation.

2. **Pareto set loss approximation**: Equation 8 approximates Pareto set distance as the minimum over individual solution pairs. This is a crude approximation that may not capture true set-to-set distance, and its effectiveness depends heavily on Pareto set discretization quality.

3. **Limited problem scope**: Experiments are restricted to linear programming problems. The applicability to nonlinear, combinatorial, or integer MOPs is unclear.

4. **Gradient computation concerns**: The reliance on DSLP requires specific regularity conditions (Hessian full-rank, etc.). The paper does not discuss when these conditions might fail or the robustness of gradients near non-differentiable points.

**Evaluation concerns:**

5. **Table 1 and Table 2 appear identical**: Both tables show identical numerical results with all seven methods having the same GD, MPFE, HAR, r1, r2, r values across all rows. This suggests either an error in the paper or that the tables are meant to represent different experimental settings but were not properly updated. Clarification is essential.

6. **Marginal improvements**: The performance gains are relatively modest (e.g., GD improves from 12.29 to 11.85, ~3.6% improvement). Without standard deviation or statistical significance testing across the 5 runs, it's difficult to assess whether improvements are reliable.

7. **Metric ambiguity**: The paper mentions "average percentage regret" as a key metric but uses notation "r" without clear formulation in the main text. The relationship between different metrics (GD, MPFE, HAR, r) needs more explicit explanation.

8. **Hyperparameter sensitivity**: Loss weights (λₗ=1, λ_d=2, λ_ps=5) are set without justification or sensitivity analysis. Different weight configurations might yield substantially different results.

**Presentation concerns:**

9. **Missing baseline**: A natural baseline would be applying existing multi-objective evolutionary algorithms (MOEAs) with learned coefficients, but none are included.

10. **Limited discussion of computational overhead**: The landscape loss via sRMMD involves iterative Sinkhorn computations. The paper doesn't discuss training time comparison with baselines.

### 4. Questions for Authors

1. **Tables 1 and 2**: Please confirm whether these tables contain identical data or if there was an error in the manuscript. If they are different experiments, please correct the values.

2. **Pareto set loss implementation**: How exactly is the Pareto set PS_yi* computed in practice? What multi-objective solver is used? How many solutions are retained?

3. **Handling non-convex Pareto fronts**: The weighted sum transformation cannot represent all Pareto-optimal solutions for non-convex fronts. How does your method handle such cases?

4. **Statistical significance**: Could you report mean ± standard deviation for key metrics across the 5 runs? Are the improvements statistically significant?

5. **Why does landscape loss help?** The ablation shows it helps, but could you provide more intuition about why measuring objective space discrepancy improves decision quality?

6. **Extension to non-linear/non-convex problems**: DSLP relies on linear programming. How would the method extend to nonlinear MOPs or mixed-integer MOPs?

# Actual Human Scores
Individual reviewer scores: [3.0, 1.0, 3.0]
Average score: 2.3
Binary outcome: Reject

=== CALIBRATION EXAMPLE 3 ===

# Review
# Detailed Review: MQFL-FHE: Multimodal Quantum Federated Learning Framework with Fully Homomorphic Encryption

## Summary

This paper proposes a novel framework combining multimodal quantum federated learning with fully homomorphic encryption (FHE) to address the trade-off between privacy preservation and model performance in federated learning systems. The authors introduce three main contributions: (1) QFL-FHE, integrating quantum computing with FHE in federated learning; (2) MQFL-FHE, extending this to multimodal data handling; and (3) MQMoE-FL-FHE, a mixture-of-experts architecture leveraging quantum layers for enhanced representation learning. Experiments are conducted on CIFAR-10, DNA sequences, MRI scans, and PCOS datasets across centralized and federated settings with various configurations (Classical FL, QFL, FL+FHE, QFL+FHE). The authors claim that quantum enhancements mitigate FHE-induced performance degradation and improve classification accuracy.

## Strengths

1. **Novel Integration**: The combination of quantum computing, federated learning, and fully homomorphic encryption addresses an important research gap. While each component exists independently, their unified treatment is relatively underexplored.

2. **Multimodal Handling**: The extension to multimodal data (DNA + MRI) with a mixture-of-experts architecture is a reasonable approach for handling heterogeneous data types across clients.

3. **Comprehensive Experimental Suite**: The paper tests multiple configurations (centralized vs. federated, with/without quantum, with/without FHE) across diverse datasets, providing a broad empirical evaluation.

4. **Ablation Analysis**: The ablation study systematically examines the contribution of each component (QC and FHE), offering insights into their individual and combined effects.

5. **Medical/Biological Focus**: Applying the framework to medical domains (genomics, brain MRI) is timely and addresses high-stakes applications where privacy is critical.

## Weaknesses

1. **Lack of Actual Quantum Hardware**: The paper uses PennyLane for quantum simulations on classical hardware. There is no evidence of execution on actual quantum processors. Simulated quantum circuits do not capture real quantum noise, decoherence effects, or hardware limitations. This fundamentally limits the paper's claim of "quantum-enhanced" performance, as the simulations cannot provide genuine quantum advantages.

2. **Questionable Mechanism for FHE Noise Mitigation**: The paper claims quantum computations "counteract" FHE-induced performance degradation, but the mechanism is unconvincing. The appendix discusses SU(2) constraints on Bloch sphere rotations and periodic error oscillation, but this theoretical justification does not clearly connect to how quantum layers actually reduce FHE noise in practice. The argument appears post-hoc rather than rigorously derived.

3. **Unclear FHE Implementation**: The mathematical formulations contain significant issues:
   - Equations appear duplicated and corrupted throughout the paper (e.g., lines 160-165, 260-265 are repeated)
   - The CKKS encryption and aggregation equations lack clarity
   - The description of how quantum model updates are encrypted ("hybrid lattice-based encryption") is vague
   - No formal security analysis or threat model is provided

4. **Limited Baseline Comparisons**: The paper compares only against itself (FL vs. FL+FHE vs. QFL vs. QFL+FHE). There are no comparisons with:
   - Other privacy-preserving FL methods (differential privacy, secure aggregation)
   - State-of-the-art multimodal FL frameworks mentioned in related work (CreamFL, Gong et al.)
   - Classical multimodal mixture-of-experts approaches without quantum components

5. **Poor Dataset Justification**: 
   - CIFAR-10 is a toy image dataset that seems mismatched with the paper's focus on medical/biological applications
   - PCOS is a medical condition affecting women, making the framing of "underrepresented categories" in Section 1 misleading
   - The RAVDESS dataset appears only in the appendix table but is not used in experiments

6. **Performance Gains Are Marginal**: The quantum improvements over FL+FHE are modest (e.g., DNA: 93.75% → 95.31%, MRI: 83.33% → 87.26%). Meanwhile, computation times increase dramatically (QFL+FHE takes ~3× longer than FL+FHE). The trade-off between marginal accuracy gains and substantial computational overhead is not convincingly justified.

7. **MQMoE Novelty Is Limited**: The MQMoE architecture is essentially a standard mixture-of-experts with quantum layers replacing some classical layers. The gating mechanism uses standard multi-head attention. The paper does not clearly articulate what is novel about the quantum gating or why quantum layers provide benefits beyond classical alternatives.

8. **Encryption Formalism Issues**: The paper states encryption parameters (polynomial degree 8192, bit sizes [60,40,40,60], 128-bit security) but:
   - Does not justify these specific choices
   - Does not discuss the impact of these parameters on the reported results
   - Does not address ciphertext depth limitations for multiple FL rounds

9. **Missing Technical Details**:
   - Specific PQC architectures are not detailed (number of qubits, gate types, entanglement patterns)
   - Quantum layer initialization is not specified
   - Training hyperparameters for quantum layers (learning rates, optimizers) are not discussed separately
   - How gradient updates work through quantum layers is not explained

10. **Writing Quality**: The manuscript contains numerous typographical errors, duplicated text blocks, and corrupted equations, suggesting rushed preparation. This undermines confidence in the technical rigor of the work.

## Questions for Authors

1. **Quantum Hardware**: Were any experiments conducted on actual quantum hardware? If not, what is the justification for claiming "quantum-enhanced" performance when only classical simulations are performed?

2. **FHE Noise Mitigation Mechanism**: Please provide a clear, formal explanation of HOW quantum operations reduce FHE noise accumulation. The SU(2) argument in the appendix does not convincingly connect to FHE error propagation. What is the specific mechanism?

3. **Security Model**: What is the precise threat model? Does the CKKS encryption protect against membership inference attacks, gradient leakage, or only communication channel eavesdropping? Please provide a security analysis.

4. **Baseline Comparisons**: Why were other multimodal FL frameworks (e.g., CreamFL) not compared? How does the proposed method compare to simpler approaches like differential privacy with FL?

5. **Scalability**: The computation times are already high (~10,000 seconds for QFL+FHE on CIFAR-10). How does the method scale to larger models, more clients, and more communication rounds? What are the FHE ciphertext depth limitations?

6. **MQMoE Design Choices**: What motivated the specific expert architecture? Why use separate quantum layers for each modality rather than shared quantum layers? How are the gating weights learned and aggregated in the federated setting?

7. **Why CIFAR-10?**: Given the paper's emphasis on medical applications and privacy-sensitive data, why include CIFAR-10 (a toy image dataset) rather than additional medical datasets?

8. **Error Analysis**: The reported standard deviations are very small (±0.02-±0.13), but the paper does not clearly state how these were computed or over how many random seeds. Please clarify the experimental reproducibility protocol.

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 3.0, 3.0, 3.0]
Average score: 3.4
Binary outcome: Reject

=== CALIBRATION EXAMPLE 4 ===

# Review
## Review: FedQLoRA: Federated Quantization-Aware LoRA for Large Language Models

### 1. Summary

This paper proposes **FedQLoRA**, a novel federated learning framework designed to address the **quantization bias** problem that arises when clients use LLMs with heterogeneous quantization levels (e.g., some clients use 2-bit quantization, others use 4-bit). The core insight is that when adapters trained on differently quantized models are aggregated, the quantization error creates an adverse bias that degrades overall performance.

The proposed method introduces a **quantization-aware adapter** that estimates and separates quantization error from the task-specific information learned by the LoRA adapter. Additionally, the paper presents an **iterative version (iFedQLoRA)** that alternates between improving the quantization-aware adapter and refining the LoRA adapter using global aggregated information, thereby also addressing heterogeneity bias in non-IID settings.

### 2. Strengths

1. **Novel Problem Identification**: The paper is the first to identify and formalize the "quantization bias" problem in federated learning with heterogeneous quantized LLMs. This is a genuine and important contribution to the field.

2. **Clear Theoretical Analysis**: The mathematical formulation of quantization bias (equations 7-9) is well-structured. The decomposition of the aggregated adapter difference into quantization error and quantization bias components provides clear insight into why mixed quantization settings perform poorly.

3. **Theoretical Justification**: Proposition 1 establishes a sound theoretical connection between LoRA-aware quantization and the optimal quantization-aware adapter, providing mathematical grounding for the approach.

4. **Comprehensive Experimental Evaluation**: 
   - Experiments across two datasets (XGLUE NC, 20 Newsgroups)
   - Multiple client configurations (3, 5, 10 clients)
   - Both IID and non-IID data partitions
   - Ablation studies analyzing model heterogeneity and data heterogeneity separately

5. **Practical Solution**: By separating quantization error from local data learning, the method effectively mitigates the performance degradation in mixed quantization scenarios while maintaining the communication efficiency benefits of adapter-sharing approaches.

### 3. Weaknesses

1. **Scale of Experiments**: The paper claims to address LLMs with "billions of parameters," yet experiments use only **DistilBERT** (66M parameters). There is no validation on actual large-scale LLMs (e.g., LLaMA, GPT models). This significantly weakens the practical relevance of the claims.

2. **Limited Baseline Comparison**: The baselines compare only against other LoRA-based FL methods (LoRA, FFA-LoRA, H-LoRA, H-LoRA-T). The paper would benefit from comparisons with:
   - Vanilla federated QLoRA without the proposed bias correction
   - Other FL aggregation methods (e.g., FedProx, SCAFFOLD adapted to LoRA)
   - Methods that align quantization levels before aggregation

3. **Insufficient Algorithmic Details**: The practical implementation of quantization-aware learning is unclear. Specifically:
   - How is the rank *m* of the quantization error matrix determined?
   - What SVD implementation is used given that the quantization error matrix is not directly observable?
   - The optimization in Eq. 15 requires knowledge of the quantization method *Q*, which may not be available in practice.

4. **Communication Overhead**: While the paper emphasizes reduced communication costs by sharing only adapters, it does not quantify the overhead of the additional quantization-aware adapter parameters. This is important for a fair comparison with other methods.

5. **Missing Convergence Analysis**: Section 4.3 shows convergence curves but provides no theoretical convergence guarantees for iFedQLoRA. The iterative optimization involves alternating between two non-convex objectives (Eq. 19 and Eq. 20), and the paper does not discuss whether convergence is guaranteed or under what conditions.

6. **Moderate Performance Improvements**: The performance gains of iFedQLoRA over baselines are modest—typically 1-3% in accuracy/F1. Given the added complexity of the iterative framework, the practical significance of these improvements warrants further discussion.

7. **Notation and Presentation Issues**: The paper contains several instances of duplicated, overlapping, or corrupted text (particularly in equations and Figure 2), making some mathematical formulations difficult to follow. This suggests the paper may need additional proofreading and formatting attention.

8. **Scope of Quantization Methods**: The experiments focus on 2-bit and 4-bit quantization with specific quantization methods (quantile quantization and LoRA-aware quantization). The generalizability to other quantization schemes (e.g., GPTQ, AWQ) is not discussed.

### 4. Questions for Authors

1. **Scale of Experiments**: Why were experiments conducted only on DistilBERT (66M parameters) rather than actual LLMs? How do the authors expect the quantization bias problem and the proposed solution to behave with models in the billions of parameters range?

2. **Algorithmic Implementation**: Can you provide more details on how the quantization-aware adapter is implemented in practice? Specifically, how is the SVD applied when the quantization error matrix *E* is not directly observable, and how is the rank *m* determined?

3. **Privacy and Quantization Knowledge**: Equation 15 assumes the quantization method *Q* is known. In practice, if different clients use proprietary quantization schemes, how would this assumption hold? Does this limit the applicability of the method?

4. **Communication Overhead**: What is the total number of parameters communicated per round? How does the quantization-aware adapter size compare to the LoRA adapter size, and does this affect the overall communication efficiency?

5. **Convergence Guarantees**: The iterative optimization alternates between two non-convex objectives. Can the authors provide any theoretical convergence guarantees or discuss the conditions under which convergence is ensured?

6. **Comparison with Simple Truncation**: H-LoRA-T achieves reasonable performance by simply truncating all models to the lowest precision level. Can the authors provide a more detailed cost-benefit analysis of why FedQLoRA is preferable despite the added complexity?

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 5.0, 3.0]
Average score: 4.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 5 ===

# Review
# Review: ADAFM: Adaptive Variance-Reduced Algorithm for Stochastic Minimax Optimization

## 1. Summary

This paper introduces AdaFM (Adaptive Filtered Momentum), a novel adaptive variance-reduced algorithm for stochastic minimax optimization problems. The key contribution is developing an algorithm that automatically adjusts its hyperparameters (momentum parameters and learning rates) based solely on historical estimator information, eliminating the need for manual tuning of problem-dependent parameters.

The algorithm is inspired by STORM (Cutkosky & Orabona, 2019) and incorporates variance reduction with momentum correction. AdaFM uses a simplified momentum parameter schedule β_{t+1} = 1/t^{2/3} for both primal (x) and dual (y) variables, making it tuning-free. Learning rates are adapted based on cumulative estimator values divided by momentum parameters, with a mechanism ensuring x's learning rate accounts for y's learning rate to maintain proper step size ratios.

Theoretical results demonstrate that AdaFM achieves near-optimal sample complexity of O(ε^{-3}) for finding ε-stationary points in both Non-Convex-Strongly-Concave (NC-SC) and Non-Convex-Polyak-Łojasiewicz (NC-PL) settings. Experiments on test functions, deep AUC maximization, and WGAN-GP validate the algorithm's effectiveness and robustness compared to RSGDA, VRAdaGDA, and TiAda.

## 2. Strengths

**Novelty and Contribution:**
- First adaptive variance-reduced algorithm for stochastic minimax optimization, addressing a significant practical gap between theory and implementation
- Elegant simplification of momentum parameters (β_{t+1} = 1/t^{2/3}) that eliminates the need for separate β^x_t and β^y_t tuning required by prior VR-based methods
- Theoretically principled approach that maintains optimal convergence rates while being parameter-free

**Theoretical Depth:**
- Comprehensive analysis covering both NC-SC and NC-PL settings
- Detailed error dynamics analysis showing how the momentum and learning rate choices control estimation errors
- Four-case analysis strategy in the NC-SC setting based on cumulative error and gradient magnitudes is thorough
- Comparison with existing methods (TiAda achieves O(ε^{-4}), VRAdaGDA requires tuning) clearly positions the contribution

**Practical Impact:**
- Addresses a real problem: VR-based minimax algorithms are notoriously sensitive to hyperparameter choices (demonstrated in Figure 1 showing RSGDA's sensitivity)
- Claims γ = λ = 1 works in theory and the remaining hyperparameter δ can be set to small values (0.001), making the algorithm truly easy to use
- The hyperparameter grid search comparison (Figure 5 vs Figure 1) effectively demonstrates robustness

**Experiments:**
- Diverse evaluation across synthetic (test functions), NC-SC (deep AUC), and NC-PL (WGAN-GP) settings
- Ablation study on δ provides useful practical guidance
- Comparison with multiple baselines including the only existing parameter-free method (TiAda) strengthens the contribution

## 3. Weaknesses

**Theoretical Concerns:**

1. **Incomplete δ = 0 Analysis**: The paper states that δ can be "arbitrarily close to 0" but the ablation study (Figure 9) shows that δ = 0 causes the algorithm to fail to converge. This contradiction needs clarification—does "arbitrarily close to 0" mean strictly positive, and what is the practical minimum δ value for guaranteed convergence?

2. **κ^5 vs κ^4.5 Gap in PL Setting**: Theorem 2 shows a degradation from O(κ^4.5) in NC-SC to O(κ^5) in NC-PL. The paper attributes this to using the PL condition indirectly through the quadratic growth condition, but this degradation is significant for ill-conditioned problems (large κ). The paper doesn't explore whether this gap can be closed.

3. **Sample Complexity Interpretation**: The paper claims O(ε^{-3}) sample complexity, but this is per variable (x and y each require samples). For a fair comparison with single-variable optimization, the total sample count should be considered. The statement "AdaFM only needs two samples, i.e., O(1), to compute estimators" requires clarification—does this mean each iteration requires 2 stochastic gradient evaluations?

4. **Missing Comparison with Modern Adaptive Methods**: The paper compares with TiAda (parameter-free but O(ε^{-4})), RSGDA, and VRAdaGDA, but doesn't compare with other adaptive methods like AdaGDA (Huang et al., 2023) which also has adaptive learning rates but with different adaptivity mechanisms.

**Experimental Concerns:**

1. **Limited Deep Learning Scale**: Experiments are conducted on CIFAR10/CIFAR100 with relatively small models (ResNet20, 4-layer CNNs). The practical benefits of AdaFM's adaptivity may become more or less apparent at larger scales (e.g., ImageNet, large language models).

2. **δ Sensitivity Not Fully Explored**: While ablation studies are provided, the paper doesn't give clear guidance on selecting δ for different problem types. The finding that smaller δ works better for WGAN-GP contradicts the intuition that larger δ helps adjust learning rate ratios faster.

3. **Inception Score Metric**: For WGAN-GP experiments, the paper uses Inception Score, which is known to have limitations and can be insensitive to mode collapse. Frechet Inception Distance (FID) would provide a more complete picture of generation quality.

4. **Missing Baselines**: The deep AUC experiments don't include standard minimax optimization methods like SGDA or basic Adam variants, making it difficult to assess the practical overhead of AdaFM versus simpler approaches.

**Presentation Issues:**

1. **Algorithm Description Gaps**: Equations 2 and 3 (the estimator update rules) appear as omitted pictures, making it difficult to fully understand the algorithm. These are central to the contribution and should be clearly stated.

2. **Notation Inconsistencies**: The paper uses both κ and L/μ notation for the condition number. Consistency would improve readability.

3. **Missing Pseudocode Formatting**: Algorithm 1 is presented as a numbered list rather than proper pseudocode, reducing clarity.

## 4. Questions for Authors

1. **On δ Selection**: Your ablation study shows that δ = 0 causes non-convergence (Figure 9a), yet the theoretical results claim δ can be "arbitrarily close to 0." What is the minimum practical δ value for guaranteed convergence? Is there a theoretical lower bound on δ?

2. **Sample Complexity Clarification**: Each iteration requires computing vt and wt using stochastic gradients. Please clarify the exact number of stochastic gradient oracle calls per iteration and confirm that the total sample complexity is indeed O(ε^{-3}) rather than O(2ε^{-3}) or similar.

3. **PL Setting Gap**: The NC-PL setting shows O(κ^5) complexity versus O(κ^4.5) for NC-SC. Can this gap be closed, or is it fundamental to parameter-free algorithms under PL conditions? Is there a lower bound showing this degradation is necessary for parameter-free methods?

4. **Comparison with AdaGDA**: Huang et al. (2023) propose AdaGDA with adaptive learning rates. How does AdaFM differ from AdaGDA, and why is AdaGDA not included as a baseline in experiments?

5. **Robustness to Initial Learning Rates**: While γ and λ can be set to 1, the experiments use different learning rate search ranges for AdaFM versus other methods. Is there any guidance on appropriate initial learning rate ranges for different problem types (NC-SC vs NC-PL)?

6. **Computational Overhead**: AdaFM requires maintaining cumulative sums of estimator norms (α_t^x, α_t^y) across all iterations. What is the memory and computational overhead, and how does this scale compared to VRAdaGDA?

7. **Theoretical Tightness**: The bound in Theorem 1 contains multiple O(·) terms with different κ dependencies. Is this bound tight, or can the constants and κ-exponents be improved?

8. **Practical Implementation**: The paper claims γ = λ = 1 works in theory, but practical implementations (Figure 3) use different search ranges. What are the recommended default values for γ and λ in practice?

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 8.0, 6.0]
Average score: 5.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 6 ===

# Review
# Review: Swift-FedGNN: Federated Graph Learning with Low Communication and Sample Complexities

## 1. Summary

This paper proposes Swift-FedGNN, a novel federated graph neural network (GNN) framework designed to train GNNs on geo-distributed graph data while preserving data privacy. The key insight is to reduce the heavy communication overhead of cross-client neighbor sampling by having most clients perform efficient local training in parallel, with only a small, randomly sampled subset of clients periodically conducting the more expensive cross-client training. The paper provides rigorous theoretical convergence analysis showing that Swift-FedGNN achieves O(1/√T) convergence rate, matching state-of-the-art sampling-based GNN methods, despite operating in the more challenging federated setting. Extensive experiments on real-world datasets (ogbn-products and Reddit) demonstrate significant improvements in convergence speed and communication efficiency compared to existing federated GNN approaches.

## 2. Strengths

**Novel Algorithmic Contribution:** The paper makes a genuine algorithmic contribution by introducing a hybrid local/cross-client training strategy that strategically schedules expensive cross-client operations. This is a principled approach that addresses a real bottleneck in federated GNN training.

**Theoretical Rigor:** The paper provides a non-trivial convergence analysis that handles the key challenge of biased stochastic gradients in GNNs. Notably, the analysis does not rely on unrealistic assumptions (like unbiased gradient estimators) that are commonly used in prior work. The derivation of gradient error bounds that scale with the number of layers L is a valuable theoretical insight.

**Privacy Preservation:** The two-stage aggregation design (remote clients → server → training client) provides architectural privacy benefits by preventing direct feature sharing between clients. This is a meaningful practical contribution.

**Comprehensive Experiments:** The paper evaluates on two large-scale real-world datasets with distinct characteristics (large/sparse vs. dense). The comparison against multiple baselines (LLCG, FedGNN-PNS, FedGNN-G) is thorough, and the communication overhead analysis in Table 2 is particularly compelling.

**Clear Problem Formulation:** The paper clearly motivates the problem of cross-client neighbor sampling overhead and establishes the trade-off between information loss and communication cost through the hyperparameters I (correction frequency) and K (number of cross-client clients).

## 3. Weaknesses

**Experiment-Theory Mismatch on Architecture:** The theoretical analysis is conducted for GCN, but the experiments use GraphSAGE. While both are sampling-based GNNs, the architectural differences (GraphSAGE uses concatenation and different aggregation functions) could affect the analysis. The paper should clarify whether the convergence guarantees extend to GraphSAGE or if additional analysis is needed.

**Hyperparameter Sensitivity:** While Figure 6 shows sensitivity analysis for I, K, and sampling fanouts, the paper does not provide clear guidance on how to set these critical hyperparameters in practice. The trade-off between I, K, and convergence quality is not quantified, making it difficult for practitioners to apply the method.

**Scalability Concerns:** The experiments use only 10-20 clients. Real-world federated learning scenarios may involve hundreds or thousands of clients. The paper does not discuss how the approach scales, particularly regarding the server-side aggregation overhead when many clients are involved.

**Limited GNN Architecture Support:** The paper explicitly notes (footnote 2) that operation offloading only supports element-wise operations (GCN, SGC). Supporting non-element-wise operations like GAT would require transferring raw features, negating some privacy benefits. This is a significant practical limitation not sufficiently discussed.

**Privacy Guarantees Are Architectural, Not Formal:** While the two-stage aggregation design helps preserve privacy, the paper does not provide formal privacy guarantees (e.g., differential privacy analysis). For privacy-critical applications (healthcare, as motivating example), formal guarantees would strengthen the contribution.

**Handling of Cross-Client Edges:** The paper assumes nodes are disjointly partitioned across clients, but does not discuss how edges between clients are handled in practice. Are these edges discovered during sampling, and if so, how is this communicated efficiently?

## 4. Questions for Authors

1. **Architecture mismatch:** The theory is for GCN but experiments use GraphSAGE. Can you clarify whether the convergence analysis applies to GraphSAGE, and if not, what modifications would be needed?

2. **Hyperparameter guidance:** The correction frequency I and number of cross-client clients K are critical parameters. Is there a principled way to set these based on the dataset characteristics (e.g., cross-client edge density)? The paper mentions a trade-off but does not provide concrete guidance.

3. **Scalability:** How does the approach scale with the number of clients? The server aggregates from all clients (line in Algorithm 1), and this could become a bottleneck with many clients. Have the authors tested with larger client numbers?

4. **Edge discovery:** How are cross-client edges discovered? In the healthcare example, patients at different hospitals have interactions—but how does the system know which clients contain a node's neighbors without excessive communication?

5. **GAT compatibility:** You mention GAT requires raw feature transfer. For privacy-critical applications, what is the practical impact of this limitation? Are there alternatives (e.g., privacy-preserving aggregation) you considered?

6. **Convergence to suboptimal solution:** Theorem 5.6 shows convergence to a neighborhood of the optimal solution. The residual error depends on B_Δ terms. Can you provide more intuition or quantitative bounds on how large this neighborhood could be in practice?

7. **Client heterogeneity:** The paper assumes i.i.d. or balanced data distribution implicitly through METIS partitioning. How would the approach perform under non-IID data distributions across clients, which is the typical FL scenario?

## 5. Minor Issues

- Figure 2 shows per-iteration time breakdown but the text mentions Figure 2 multiple times with references to captions that don't fully match.
- The notation in some places is confusing (e.g., double subscripting like Bv[m] could be clearer as B_v^{(m)}).
- Some equations in the appendix appear truncated or improperly formatted, making verification difficult.

# Actual Human Scores
Individual reviewer scores: [3.0, 6.0, 5.0, 5.0]
Average score: 4.8
Binary outcome: Reject

=== CALIBRATION EXAMPLE 7 ===

# Review
# Detailed Review: Hierarchical Prompts with Context-Aware Calibration for Open-Vocabulary Object Detection

## Summary

This paper proposes HiCA (Hierarchical prompts with Context-Aware calibration) for Open-Vocabulary Object Detection (OVD). The method addresses the limitation that existing OVD approaches primarily learn direct mappings between visual regions and category names, ignoring shared knowledge between base and novel classes as well as visual context information. 

The main contributions are:
1. **Hierarchical prompts**: Decompose region-category mapping into coarse-grained (region-to-superclass) and fine-grained (superclass-to-category) steps, where superclass prompts capture shared semantic knowledge across both base and novel classes.
2. **Context-aware calibration**: Use unsupervised clustering on visual context features to build a context-superclass distribution matrix, which calibrates detection results based on environmental context.
3. **Multi-modal prompts**: Combine learnable text and visual prompts for improved knowledge distillation.

Experiments on OV-COCO and OV-LVIS show competitive results, achieving 57.2% mAP_B on OV-COCO and 32.4% AP on OV-LVIS.

## Strengths

1. **Well-motivated problem**: The paper correctly identifies that existing OVD methods overfit to base classes during training, leading to poor generalization to novel classes. The hierarchical approach to incorporate coarse-grained knowledge is conceptually sound.

2. **Comprehensive experiments**: Extensive ablation studies demonstrate the contribution of each component. The balance parameter analysis (Figure 3) effectively shows how coarse/fine-grained knowledge affects performance.

3. **Visualization and analysis**: The paper provides good qualitative analysis through similarity matrices (Figure 6) and prompt projection visualizations (Figure 4), helping understand why hierarchical prompts work.

4. **Plug-and-play design**: The hierarchical prompts and context-aware calibration are designed as modular components independent of the knowledge distillation framework, making them potentially applicable to other OVD methods.

5. **Improved novel class generalization**: The method achieves better balance between base and novel class performance compared to methods like CORA that sacrifice base class performance for novel class gains.

## Weaknesses

1. **Unclear superclass definition**: The paper does not clearly explain how superclasses are defined. It appears to be manually constructed (Figure 2 shows "animal", "vehicle" as superclasses). For different datasets or deployment scenarios, how should users determine appropriate superclasses? This limits the method's generalizability.

2. **Simplistic subordinate matrix**: The matrix A with binary 0/1 assignment (category belongs to superclass or not) seems overly simplistic. Many objects could reasonably belong to multiple superclasses. The paper does not explore more nuanced assignment strategies.

3. **Balance parameter λ is dataset-dependent**: The ablation study shows λ affects performance differently, but the paper does not provide guidance on selecting λ for new datasets. This hyperparameter requires tuning on validation data, which may not be available for novel classes.

4. **Context clustering K requires tuning**: Table 5 shows performance varies significantly with K (number of context clusters). The paper does not explain how to determine optimal K, and the performance gap (27.6% to 31.2% mAP_N) is substantial.

5. **Limited novelty**: The overall framework combines existing techniques (knowledge distillation, learnable prompts, visual prompts from prior work). The hierarchical prompt concept, while reasonable, is a straightforward extension of CoOp's learnable prompts.

6. **Missing baselines**: Some recent strong baselines (e.g., BARON+HiCA achieves 59.8% mAP_B but only 36.0% mAP_N) are compared but the paper does not discuss whether the improvement on novel classes is statistically significant across runs.

7. **Context-aware calibration complexity**: The method introduces several new components (DG Layer, context clustering, distribution generation) with limited ablation showing modest gains (1.2% improvement on mAP_N from context-aware calibration alone).

## Questions for Authors

1. **Superclass construction**: How exactly are superclasses determined for a given dataset? Is it purely based on common semantic groupings (animals, vehicles, etc.)? What happens if appropriate superclasses don't exist or overlap significantly?

2. **Generalization to different superclass granularities**: Have the authors experimented with different levels of superclass granularity (e.g., "living things" vs. "animals" vs. "mammals")? How sensitive is the method to superclass choice?

3. **Novel class prompts**: During inference, how are prompts for novel classes generated? Are they learned from base classes or constructed from class names? The paper mentions "we can only get the name of the novel classes" but doesn't detail prompt construction.

4. **Context clustering stability**: The context clustering is performed via K-means on global image features. How stable is this across different initializations? Does it need to be re-run for each dataset?

5. **Why does context-aware calibration hurt base class performance?**: The ablation shows base class mAP drops from 57.5% to 57.2% with context calibration. The paper explains this is expected but doesn't explore how to mitigate this trade-off.

6. **Comparison with other hierarchical/structured classification methods**: How does this approach compare with methods that explicitly model class hierarchies (e.g., hierarchical softmax, taxonomy-based classification) in the OVD setting?

7. **Training efficiency**: The method adds visual prompts and context-aware calibration. What is the additional computational overhead compared to baseline methods like OADP?

## Minor Issues

- Some formatting issues in equations (e.g., lines 231-232, 294-295 appear corrupted)
- Figure 2's description is quite long and overlaps with the main text
- The paper claims SOTA on OV-COCO but BARON+HiCA's novel class performance (36.0%) is still below CORA's (35.1% on their reproduced baseline, though CORA's reported is 35.1%)

## Conclusion

This is a solid contribution to the OVD literature that addresses an important problem (base class overfitting). The hierarchical prompts provide a principled way to incorporate coarse-grained semantic knowledge, and the ablation studies are thorough. However, the reliance on manually-defined superclasses and the number of new hyperparameters (λ, K, DG layer depth) may limit practical applicability. The paper would benefit from clearer guidelines on hyperparameter selection and analysis of robustness to superclass definitions.

# Actual Human Scores
Individual reviewer scores: [5.0, 5.0, 5.0, 5.0]
Average score: 5.0
Binary outcome: Reject

=== CALIBRATION EXAMPLE 8 ===

# Review
# Review: Can We Ignore Labels in Out-of-Distribution Detection?

## Summary

This paper addresses a fundamental question in out-of-distribution (OOD) detection: whether labels are necessary for effective OOD detection. The authors make three main contributions: (1) a theoretical "Label Blindness Theorem" proving that SSL/unsupervised methods will fail when the learning objective is independent of in-distribution labels, (2) a new "Adjacent OOD Detection" benchmark that tests OOD detection when there is significant overlap between ID and OOD data, and (3) experiments demonstrating that existing unlabeled OOD methods fail under label blindness conditions. The paper concludes that labels cannot be safely ignored for OOD detection.

## Strengths

1. **Theoretically motivated contribution**: The information-theoretic framing of the Label Blindness Theorem provides a principled understanding of why unlabeled OOD detection can fail. The derivation from information bottleneck principles to guarantee failure is well-structured.

2. **Addresses an important gap**: The question of whether labels are necessary for OOD detection is crucial for safety-critical applications. The paper correctly identifies that existing benchmarks (far/near OOD) may inadvertently hide label blindness failure due to low feature overlap between ID and OOD.

3. **Novel benchmark**: The Adjacent OOD Detection task is a valuable contribution. By bootstrapping class splits from the same dataset, it creates a maximally challenging scenario that existing benchmarks do not capture.

4. **Comprehensive theoretical analysis**: The paper connects its work to existing theorems (No Free Generalization, Fano's Inequality) and provides rigorous proofs for all main claims in the appendix.

5. **Practical implications discussed**: The paper appropriately acknowledges scenarios where unlabeled OOD methods may still be acceptable (e.g., when adjacent OOD risk is low, or when consequences of failure are tolerable).

## Weaknesses

1. **Limited empirical breadth**: The experiments on Adjacent OOD focus primarily on three datasets (Faces, Cars, Food) where the problem appears nearly unsolvable (all FPR95 values ~88-96%). The paper would benefit from additional datasets where SSL methods exhibit varying degrees of label blindness.

2. **Missing recent baselines**: Several recent unlabeled OOD detection methods are mentioned in the related work but not evaluated (e.g., CSI by Tack et al., 2020; CADET by Guille-Escuret et al., 2024). This creates an incomplete picture of the current state of unlabeled OOD detection.

3. **Theoretical assumptions may be strong**: The strict independence assumption (x₁ ⊥ x₂) and the information bottleneck framework may not perfectly characterize how neural networks actually learn. The extension to "approximate label blindness" via Fano's inequality is weaker and less rigorous than the strict case.

4. **Limited guidance on solutions**: While the paper argues labels cannot be ignored, it provides minimal concrete guidance on how many labels are sufficient or what form of label supervision helps. The mention of "few-shot methods" lacks specifics or experiments.

5. **CLIPN evaluation complexity**: The analysis of CLIPN's performance relies on observing alignment between pretraining captions and ID labels, which is post-hoc and qualitative. A more systematic evaluation would strengthen the claims.

6. **Adjacent OOD may conflate with fine-grained classification**: On datasets like Stanford Cars (196 fine-grained classes), Adjacent OOD may be measuring the ability to distinguish visually similar classes rather than true OOD detection capability.

7. **CIFAR results present a nuanced picture**: The paper shows SSL methods perform substantially better on CIFAR adjacent OOD (Table 2), but this nuance is not fully explored. The reason for this difference should be explained more carefully.

## Key Clarifications Needed from Authors

1. **Minimal label requirements**: The paper states labels are necessary but does not specify how many labels would suffice to overcome label blindness. Is one labeled example per class sufficient? What is the theoretical minimum?

2. **Why CIFAR adjacent OOD works better**: The paper shows SimCLR achieves 77.6 AUROC on CIFAR10 adjacent OOD but only 52% on Faces/Cars. What explains this discrepancy? Is it the visual similarity of classes or something about the learned representations?

3. **Scope of theory for generative models**: The theory focuses on representation learning via information bottleneck. How does it extend to generative methods like diffusion models (Liu et al., 2023) or VAEs (Xiao et al., 2020)?

4. **Role of multi-view/supervised contrastive learning**: The paper mentions supervised contrastive learning (Khosla et al., 2020) as a way to improve OOD detection. Does the Label Blindness Theorem suggest this would help, or is this still limited?

5. **Practical recommendations**: For practitioners, what concrete guidance do the authors offer? Should they always use supervised methods? When is unlabeled acceptable? The discussion section touches on this but lacks specifics.

6. **Relation to recent concurrent work**: The paper cites Du et al. (2024a,b) as supporting the importance of labels. How does the theoretical framework here compare or extend theirs?

## Minor Concerns

- Some notation in proofs is difficult to parse (e.g., inconsistent formatting of mutual information terms)
- The gradient visualization in Figure 1 is illustrative but qualitative; more rigorous feature analysis would strengthen the empirical case
- Section 4.3's distinction between far/near/adjacent OOD could be more clearly defined
- The paper's conclusion that "the answer is no" to ignoring labels is strong; a more nuanced discussion of trade-offs would be appropriate

## Summary Assessment

This is a valuable paper that makes a solid theoretical contribution to understanding the limitations of unlabeled OOD detection. The Label Blindness Theorem is well-motivated and rigorously derived. The Adjacent OOD benchmark fills an important gap in evaluation methodology. However, the empirical evaluation would benefit from broader coverage of methods and datasets, and the paper could provide more practical guidance for practitioners. The core message—that labels cannot be safely ignored for OOD detection in safety-critical applications—is important and well-supported by theory, though the conditions under which unlabeled methods might still be useful could be explored more thoroughly.

# Actual Human Scores
Individual reviewer scores: [6.0, 5.0, 8.0]
Average score: 6.3
Binary outcome: Accept

=== CALIBRATION EXAMPLE 9 ===

# Review
## Review: LOCOVR: Multiuser Indoor Locomotion Dataset in Virtual Reality

---

### Summary

This paper introduces LocoVR, a novel dataset of 7000+ two-person trajectories captured in virtual reality across 131 diverse indoor home environments. The authors argue that existing indoor trajectory datasets lack sufficient scale, scene variation, and social navigation dynamics (proxemics). LocoVR addresses these limitations by leveraging VR technology for efficient data collection of geometrically and socially-aware human motion. The paper demonstrates the dataset's utility through three indoor tasks: global path prediction, trajectory prediction, and goal prediction, showing that models trained on LocoVR outperform those trained on existing datasets (GIMO, THOR-MAGNI) when evaluated on both synthetic and real-world test data.

---

### Strengths

1. **Novel dataset scale and diversity**: 7000+ trajectories across 131 indoor scenes represents a significant advancement over existing indoor trajectory datasets (e.g., GIMO with 19 scenes, THOR-MAGNI with 4 scenes). This scale is crucial for generalization to unseen environments.

2. **VR-based collection methodology**: The paper effectively argues for VR's advantages in data collection—fast scene variation, accurate spatial data, natural locomotion, and controlled social scenarios. The alignment between virtual and physical space (10m × 10m) enables realistic walking behavior.

3. **Social proxemics capture**: The dataset explicitly captures two-person interactions including yielding behaviors in narrow spaces, distance maintenance, and path negotiation. The analysis of minimum distance distributions (~70% within 2m) demonstrates meaningful social interactions.

4. **Comprehensive evaluation design**: Three distinct tasks with multiple baselines provide thorough validation. The ablation studies effectively demonstrate the contribution of dataset scale, multi-person data, and heading direction information.

5. **Real-world validation**: The LocoReal dataset collected in physical space addresses the VR-to-real transfer concern and shows meaningful performance gains, supporting the paper's claim about VR data utility.

6. **Detailed ablation study**: The analysis of data-size-G/T variants isolates the impact of scale versus scene diversity, providing valuable insights into dataset composition effects.

---

### Weaknesses

1. **Limited participant diversity**: Only 32 participants (21 male, 11 female, ages 18-42), all able-bodied adults. This significantly limits the dataset's representativeness for home environments, which often include children, elderly, and people with mobility impairments.

2. **VR-real gap remains a concern**: While the paper addresses this with LocoReal, the test dataset is small (450 trajectories, 4 layouts). More extensive validation would strengthen confidence in transferability. The psychological differences between navigating real versus virtual spaces are not fully explored.

3. **Task methodology concerns**: The U-Net model representing trajectories as image-based probability distributions is unconventional for trajectory prediction. Standard approaches (LSTM/Transformer-based sequence models, graph neural networks) would be more typical and may better capture temporal dependencies.

4. **Insufficient social behavior analysis**: Despite emphasizing social proxemics as a key contribution, the paper lacks quantitative analysis of social behavior occurrences. How often do yielding, distance maintenance, or path negotiation behaviors occur? Without this, it's difficult to assess whether social dynamics are well-represented.

5. **Data augmentation methodology**: The 8× augmentation (horizontal flip + rotations) may create unrealistic scenarios—rotating trajectories 180° in asymmetric scenes could produce physically implausible navigation patterns.

6. **Limited baseline diversity**: The compared datasets (GIMO, THOR-MAGNI) are not ideal controls—they have different characteristics (single vs. multi-person, scene count). A synthetic dataset baseline or randomly initialized model would better isolate dataset quality effects.

7. **Task design limitations**: The goal-conditioned proxy task may not reflect natural indoor locomotion patterns. Daily home movements often involve less purposeful goal-seeking and more exploratory behavior.

8. **Missing comparison**: The paper does not compare against GTA-IM or HUMANISE, which also use synthetic data and include multiple scenes, despite these appearing in the related work table.

---

### Questions for Authors

1. **Dataset novelty claim**: The paper states LocoVR is the "first large-scale indoor trajectory dataset featuring two-person motions." How does this differ from THOR-MAGNI, which also contains multi-person indoor trajectories? Please clarify the distinguishing characteristics.

2. **Social behavior quantification**: What percentage of trajectories exhibit clear social navigation behaviors (e.g., detouring to avoid other person, yielding in narrow spaces)? How were these behaviors identified and labeled?

3. **Trajectory encoding rationale**: Why choose image-based U-Net representation over sequence models (LSTM/Transformer) for trajectory prediction? What motivated this architectural decision?

4. **Augmentation realism**: How do 180° and 270° rotations affect model learning? These rotations may create physically implausible scenarios in asymmetric scenes. Were these tested for negative impact?

5. **Participant instructions**: Were participants given any instructions about social behavior, or were all proxemic interactions naturally emergent? This affects the ecological validity of social motion data.

6. **Dataset accessibility**: What are the licensing terms and access procedures for LocoVR? Is the dataset fully available or subject to restrictions?

7. **Generalization boundaries**: Beyond indoor home environments, how well would models trained on LocoVR transfer to other indoor settings (offices, hospitals, retail)? Are there domain-specific features that would limit transfer?

8. **Statistical significance**: Are the performance improvements between training datasets statistically significant? The paper reports standard deviations but doesn't provide p-values or confidence intervals.

9. **Avatar influence**: How might the virtual avatars (SMPL meshes) affect participant behavior compared to bare VR controllers? Did you conduct any validation on avatar realism's impact on behavior?

10. **Path efficiency discrepancy**: Path efficiency (0.81) is comparable to THOR-MAGNI despite LocoVR being collected in "natural" (uncontrolled) settings. Why might this similarity exist, and what does it imply about the data quality?

---

### Additional Notes

The paper's treatment of two-person interactions as the primary social unit is reasonable given home environment statistics (60% single or two-person households), but limits applicability to crowded scenarios. The ethical implications section appropriately acknowledges demographic limitations, though more aggressive data collection strategies to diversify participants would strengthen the contribution.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0, 6.0, 3.0]
Average score: 5.8
Binary outcome: Accept

=== CALIBRATION EXAMPLE 10 ===

# Review
# Review of "Fine Tuning Can Help Detect Pretraining Data from Large Language Models"

## Summary

This paper introduces **Fine-tuned Score Deviation (FSD)**, a novel method for detecting whether text was part of an LLM's pretraining data. The key insight is that after fine-tuning an LLM on a small set of previously unseen (non-member) data from a specific domain, the perplexity scores decrease significantly more for other non-members compared to actual members. FSD exploits this differential by measuring the score deviation between the pretrained and fine-tuned models as a membership signal. The method is designed to be compatible with existing scoring functions (Perplexity, Min-k%, Zlib, Lowercase) and consistently improves their detection performance. Extensive experiments across multiple benchmarks (WikiMIA, ArXivTection, BookMIA, BookTection, Pile) and models (LLaMA, GPT-J, OPT, Pythia, NeoX) demonstrate substantial AUC improvements (e.g., from 0.62 to 0.91 on WikiMIA with OPT-6.7B).

## Strengths

1. **Novel and well-motivated insight**: The observation that fine-tuning on unseen data causes differential perplexity shifts between members and non-members is insightful and represents a new direction in membership inference for LLMs.

2. **Strong empirical results**: The paper demonstrates significant and consistent improvements across diverse datasets, models, and scoring functions. The improvements are substantial (AUC gains of 0.15-0.40+ in many cases) and the method scales across model sizes (7B to 30B).

3. **General framework**: FSD is presented as a general methodology that can enhance any scoring function, which is valuable for practical deployment and future extensions.

4. **Data efficiency**: The method achieves good results with relatively small fine-tuning datasets (100-300 non-members), demonstrating practical applicability.

5. **Comprehensive ablation studies**: The paper thoroughly investigates the impact of fine-tuning data size, different fine-tuning methods (LoRA, AdaLoRA, IA3), model sizes, cross-domain generalization, and temporal shift mitigation.

6. **Real-world relevance**: The method addresses important concerns about data contamination and copyright infringement in LLM training.

## Weaknesses

1. **Domain-matching assumption**: The method critically depends on obtaining non-member data from the *same domain* as the test data. This is a significant practical limitation. The paper acknowledges it but doesn't adequately address scenarios where domain-matched unseen data is unavailable. The cross-domain experiments (Table 16) show complete failure when training on WikiMIA and testing on ArXivTection.

2. **Temporal shift confounding**: The paper acknowledges temporal differences between members and non-members in WikiMIA (e.g., 2014 events vs. 2023 events). While experiments with timestamp manipulation (Deletion/Replacement) show reduced but still positive performance, it's unclear what fraction of the improvement stems from detecting true membership versus temporal artifacts. This is a fundamental validity concern.

3. **Missing stronger baselines**: The paper compares against Perplexity, Min-k%, Zlib, and Lowercase, but doesn't compare against more recent methods like Min-k%++ (Zhang et al., 2024), which could serve as stronger baselines.

4. **Limited discussion of why the mechanism works**: The paper provides empirical observation but lacks theoretical justification for why fine-tuning affects member and non-member scores differently. This limits understanding of when and why the method might fail.

5. **Black-box limitation**: The method assumes access to model weights and the ability to fine-tune, which limits applicability to commercial APIs that don't expose these capabilities, despite the paper's claims about commercial API compatibility.

6. **Validation set dependency**: The threshold ε is determined using a validation set. In real-world detection scenarios, this labeled validation set may not be available.

7. **Potential overfitting concern**: Fine-tuning on a small, specific set of non-members could lead to overfitting to that particular data distribution rather than learning generalizable membership signals.

## Questions for Authors

1. **Domain generalization**: The cross-domain experiment shows complete failure (Table 16, ArXiv/Wiki row). Can you elaborate on the failure mode and whether this is a fundamental limitation of the approach?

2. **Temporal shift quantification**: What percentage of the AUC improvement on WikiMIA can be attributed to detecting temporal differences versus actual membership? Have you conducted experiments with datasets where temporal confounds are controlled?

3. **Baseline comparisons**: Why wasn't Min-k%++ (Zhang et al., 2024) included as a baseline? This seems like a significant omission given it's a stronger recent baseline.

4. **Mechanism explanation**: The observation that fine-tuning "memorizes" non-members more than members is interesting but not fully explained. Is this due to overfitting dynamics, loss landscape differences, or something else?

5. **Practical deployment**: For a practitioner who wants to detect contamination in a new domain, what is the recommended strategy if domain-matched non-members are not available?

6. **Computational cost**: What are the GPU hours required for fine-tuning across different model sizes, and how does this compare to just computing baseline scoring functions?

7. **Threshold selection**: How robust is the method to threshold choice, and what strategies would you recommend when a validation set is unavailable?

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 6.0, 5.0]
Average score: 6.2
Binary outcome: Accept

=== CALIBRATION EXAMPLE 11 ===

# Review
# Review: Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning

## Summary

This paper introduces **Process Advantage Verifiers (PAVs)** — a novel approach to automated process reward models for improving LLM math reasoning. The core contribution is the insight that process rewards should measure "progress" — specifically, the change in likelihood of producing a correct answer before and after taking a step, as computed under a *prover* policy distinct from the base policy. The paper provides:

1. Theoretical analysis characterizing "good" prover policies as those that can distinguish steps taken by the base policy (high variance in advantages) while remaining aligned with the base policy's assessments
2. A training procedure for PAVs using rollouts from the prover policy
3. Empirical validation on Gemma 2B/9B/27B models showing PAVs improve test-time search by 8% accuracy and 1.5–5× compute efficiency, and online RL by 6× sample efficiency with +7% accuracy gains

## Strengths

**Conceptual novelty**: The key insight that process rewards should capture *progress* (advantages) rather than absolute Q-values or step correctness is original and well-motivated. The paper makes a compelling case that Q-values conflate state evaluation with action evaluation, leading to poor exploration in beam search.

**Theoretical contribution**: Theorem 3.1 provides a formal lower bound on policy improvement that depends on two key quantities: (i) the variance of prover advantages (distinguishability) and (ii) alignment between prover and base policy advantages. This characterization is meaningful and guides practical prover selection.

**Comprehensive evaluation**: The experiments are thorough:
- Multiple model sizes (2B, 9B, 27B)
- Both test-time search and online RL settings
- Meaningful baselines including ORM, PRM-Q (Snell et al.), PAV-as-ORM, and AlphaLLM
- Clear quantitative improvements with confidence intervals

**Practical insights**: The empirical finding that Best-of-K policies with K=4 serve as good provers is actionable. The surprising result that a *weaker* 9B prover outperforms a stronger 27B prover for the 27B base policy nicely validates the theoretical claim about complementary provers.

**First significant RL results**: The paper correctly claims this is "one of the first results" showing substantial gains from PRMs in online RL (6× sample efficiency), addressing a key gap in prior work where PRMs only improved 1–2%.

## Weaknesses

**Hyperparameter sensitivity**: The choice of α requires tuning (α=0.5 for test-time search, α=3–5 for RL), and the paper lacks guidance on transferring this to new settings. The robustness claims ("any α in range X works") need more systematic evaluation.

**Prover selection is still somewhat heuristic**: While Theorem 3.1 characterizes good provers theoretically, the paper does not provide a practical algorithm for automatically selecting provers. The BoK(π) recommendation with K=4 is empirically validated but lacks principled automated selection.

**Dataset limitation**: All experiments are on the MATH dataset. Generalization to other reasoning domains (coding, logical reasoning, multi-step QA) is unclear and potentially non-trivial.

**Training overhead**: The paper acknowledges PAVs require ~10× more training data than ORMs. The discussion of computational cost (Appendix H) shows favorable amortized cost but the absolute training cost may be prohibitive for some applications.

**Limited comparison with related work**: The paper could better situate PAVs against other recent automated PRM approaches (e.g., Math-Shepherd, ARM). The comparison in Figure 12 is helpful but could be expanded.

**Theoretical gaps**: The tabular analysis assumes oracle access to advantages and Q-values. The extension to learned verifiers in practice involves approximation errors that are not formally analyzed.

## Questions for Authors

1. **Prover selection across settings**: The paper recommends Bo4(π) as a prover, but this choice seems to vary (9B prover best for 27B base). How should practitioners automatically determine the optimal prover for their setting?

2. **Generalization to other tasks**: Are the PAV gains specific to mathematical reasoning, or do the authors expect similar benefits for code generation, logical reasoning, or other multi-step tasks?

3. **Training efficiency**: Can the 10× training data requirement for PAVs be reduced through curriculum learning, better data collection strategies, or architectural improvements?

4. **Alpha transfer**: The paper shows α needs tuning across search vs RL. How robust is α to changes in base policy strength, model size, or task difficulty?

5. **Failure modes**: Are there cases where PAVs actually hurt performance or are comparable to ORMs? What determines when PAVs are worthwhile?

6. **Alternative prover policies**: Beyond BoK, could learned verifier ensembles or mixture-of-experts provers serve as better complementary provers?

7. **RL training details**: The claim of 6× sample efficiency is impressive. Could the authors clarify whether this accounts for the additional compute per iteration when scoring multiple steps with PAVs vs. a single score with ORMs?

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 8.0, 8.0, 6.0]
Average score: 7.1
Binary outcome: Accept

=== CALIBRATION EXAMPLE 12 ===

# Review
# Review: ADIFF: Explaining Audio Difference Using Natural Language

## 1. Summary

This paper introduces the novel task of **audio difference explanation** — generating natural language descriptions that explain the differences between two audio recordings. The authors make three main contributions: (1) a new task definition with three tiers of explanation complexity (concise, brief, and detailed), (2) two new datasets (ACD and CLD) derived from AudioCaps and Clotho using LLM-generated explanations with human verification on test sets, and (3) ADIFF, a prefix-tuning-based model with a cross-projection layer and three-stage training process. The model combines HTSAT audio encoder with GPT2 language model and incorporates position captioning for improved audio grounding. Extensive experiments with both objective metrics (BLEU, METEOR, SPIDEr) and human evaluation demonstrate improvements over naive baselines and Qwen-Audio (a state-of-the-art audio-language model).

## 2. Strengths

**Novelty and Task Definition:**
- The task of audio difference explanation is genuinely novel and fills a gap in the audio-language literature
- The three-tier hierarchy (concise → brief → detailed) provides a principled way to evaluate models at different granularity levels
- The task has practical applications in audio forensics, quality assessment, and audio generation

**Comprehensive Analysis:**
- Extensive ablation studies covering: baseline architecture, cross-projection effects, language model scaling, position captioning, and stage-3 finetuning
- Language-only ablation (random encoder weights) to isolate linguistic learning from true audio understanding
- Analysis of entropy and information density across tiers validates the tiered design

**Practical Contributions:**
- The hallucination detection mechanism using frozen HTSAT predictions is practical and well-motivated
- Position captioning (training on "caption audio 1" vs "caption audio 2") addresses a real problem of distinguishing between similar audio inputs
- Code and datasets will be publicly released

**Evaluation Rigor:**
- Both objective metrics and human evaluation with clear rubrics (correctness, granularity, readability)
- Evaluation across multiple audio domains (studio recordings, FSD50K, GTZAN genres)

## 3. Weaknesses

**Architectural Innovation:**
- The core architectural contribution (cross-projection layer with separator token) is relatively straightforward — essentially applying transformer layers to concatenated audio prefixes
- Heavy reliance on existing pretrained components (HTSAT, GPT2) with standard prefix-tuning
- The interpretation of cross-projection outputs (Table 11, Appendix J) is qualitative and uses approximate nearest-neighbor matching in embedding space

**Dataset Quality Concerns:**
- Training data relies entirely on LLM-generated explanations without human verification; only test set explanations are verified
- Limited details on prompting (exact prompts for tier generation not shown in main text, relegated to appendix)
- Table 23 shows LLM-based density scoring rather than human evaluation for dataset quality validation

**Comparative Baselines:**
- Only one comparison model (Qwen-Audio) — this is understandable since most ALMs don't support two audio inputs, but limits the competitive landscape
- Qwen-Audio uses 7B parameters vs ADIFF's 128M parameters — the comparison is somewhat unfair, though authors acknowledge this
- No comparison with recent image difference captioning methods adapted to audio

**Performance Limitations:**
- Human evaluation scores average ~3.5/5 even for the best model, indicating significant room for improvement
- Tier 3 explanations consistently underperform Tier 2 (Table 2), suggesting the model struggles with detailed semantic/emotional descriptions
- Cross-projection ablation (Table 14, Appendix O) shows diminishing advantage when LLM is finetuned

**Evaluation Methodology:**
- The paper uses audio captioning metrics (BLEU, SPICE, CIDEr) for evaluation, which may not be ideal for comparative reasoning tasks
- SPIDEr as the "primary metric" is not well-justified for difference explanation

## 4. Questions

1. **Dataset Generation Protocol:** How exactly are audio pairs selected for prompting? The paper mentions excluding i to i+4th audio — what is the rationale for this exclusion, and does it introduce any selection bias in the dataset?

2. **Tier Difficulty Analysis:** The paper attributes Tier 2's higher scores to linguistic simplicity. Could an alternative explanation be that the model overfits to Tier 2's linguistic patterns without genuine audio understanding? Have the authors analyzed error patterns across tiers?

3. **Cross-Projection Mechanism:** The interpretation in Appendix J shows the text prefix storing attributes, but the analysis is qualitative. Is there quantitative evidence (e.g., attention visualization, probing classifiers) that the cross-projection layer learns meaningful difference representations?

4. **Generalization:** All evaluation is on AudioCaps/Clotho-derived datasets. How does the model perform on audio difference pairs from completely different domains (e.g., comparing environmental recordings vs. musical performances)?

5. **Hallucination Analysis:** The hallucination detection relies on frozen HTSAT predictions. What is the detection accuracy, and how often do the model's errors not align with HTSAT prediction errors?

6. **Tier-3 Performance Gap:** What specific aspects of Tier 3 explanations does the model struggle with most? Is it the emotional/psychological components, fine-grained acoustic descriptions, or comparative reasoning structure?

7. **Training Efficiency:** Figure 4 shows larger LMs need more epochs. Was any attempt made to scale training tokens alongside model size (as in Chinchilla scaling laws), or is the compute scaling analysis incomplete?

8. **WavCaps Degradation:** Appendix P shows performance degradation when adding WavCaps data. This is attributed to distribution shift, but could this also indicate that the model learns to rely on linguistic patterns from the diverse WavCaps captions rather than true audio differences?

## Minor Concerns

- The Boston Marathon anecdote in the introduction is compelling but lacks a direct connection to the paper's methodology
- Several equations in Section 3.2 appear to have formatting issues (missing content)
- Table numbering inconsistency: Table 3 header spans multiple columns that should be clearly delineated
- The paper would benefit from more diverse qualitative examples showing failure cases

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept

=== CALIBRATION EXAMPLE 13 ===

# Review
# Review: SAM 2: Segment Anything in Images and Videos

## Summary

This paper presents Segment Anything Model 2 (SAM 2), a foundation model that extends the original SAM architecture to handle video segmentation in addition to image segmentation. The authors introduce a streaming memory architecture that allows the model to maintain object context across video frames, enabling interactive segmentation and tracking. A key contribution is the SA-V dataset—the largest video segmentation dataset to date with 35.5M masks across 50.9K videos—collected through a multi-phase data engine. SAM 2 demonstrates strong zero-shot transfer capabilities across 17 video and 37 image benchmarks, achieving 3× fewer interactions needed compared to prior approaches while maintaining real-time processing speeds (43.8 FPS with Hiera-B+).

## Strengths

**Novel Architecture and Task Formulation:**
The Promptable Visual Segmentation (PVS) task provides an elegant unification of image segmentation, semi-supervised VOS, and interactive VOS under a single framework. The streaming architecture with memory attention, memory encoder, and object pointers is well-designed for the video domain. The occlusion prediction head is a thoughtful addition that addresses the challenge of objects disappearing and reappearing in videos.

**Dataset Scale and Quality:**
The SA-V dataset represents a substantial contribution with 53× more masks than existing video segmentation datasets. The three-phase data engine demonstrates a principled approach to data collection, achieving 8.4× speedup in annotation time while maintaining quality. The combination of manual and automatic annotations, along with quality verification, addresses both scale and reliability concerns.

**Comprehensive Evaluation:**
The paper presents extensive experiments across diverse benchmarks. Zero-shot evaluations on 17 video datasets and 37 image datasets demonstrate strong generalization. The comparison against SAM+XMem++ and SAM+Cutie baselines under both offline and online evaluation settings provides meaningful insights into the interactive segmentation experience.

**Technical Details:**
Ablation studies on data mixtures, model architecture (resolution, memory size, encoder capacity), and training strategies (16-frame fine-tuning) are thorough. The power-law relationship between training data quantity and accuracy (Figure 6) provides useful insights for future data collection efforts.

## Weaknesses

**Baseline Comparisons:**
The primary baselines (SAM+XMem++ and SAM+Cutie) combine SAM with existing video trackers rather than comparing against end-to-end video segmentation models. While this is reasonable given SAM 2's design philosophy, a comparison against state-of-the-art specialized VOS models (e.g., AOT, DEVA) in their native settings would strengthen the paper.

**Memory Mechanism Complexity:**
The memory architecture, while effective, relies on FIFO queues of recent frames without more sophisticated temporal modeling. The paper notes that GRU-based recurrent memory did not improve results (Table 11), but other temporal modeling approaches (e.g., learned temporal aggregation, motion cues) are not explored. The ablation shows object pointers help SA-V but not average zero-shot performance, suggesting potential brittleness.

**Occlusion Handling:**
The paper acknowledges limitations with long occlusions, but the specific behavior of the occlusion prediction head deserves more analysis. How reliable is occlusion detection? Does the model struggle with partial occlusions versus complete occlusions?

**Fairness Evaluation Scope:**
The fairness analysis is limited to the Ego-Exo4D dataset focusing on gender and age groups. While the results show minimal discrepancy, this evaluation covers only people segmentation and one dataset. A broader fairness assessment across different object categories and more diverse scenarios would strengthen the paper's responsible AI considerations.

**Generalization to Extreme Scenarios:**
The paper shows strong results on standard benchmarks, but generalization to edge cases like very long videos (hours-long), rapid camera motion, severe compression artifacts, or domain shift to significantly different video styles is not thoroughly explored.

## Questions for Authors

1. **Multi-object handling:** The paper mentions processing each object independently with shared image embeddings. How does SAM 2 handle cases where two objects of the same class overlap or merge? Is there any mechanism for inter-object communication during segmentation?

2. **Memory footprint:** For extremely long videos (thousands of frames), how does the memory bank scale? Is there any mechanism to summarize or compress memories beyond the fixed FIFO queue?

3. **Shot changes:** The paper mentions SAM 2 may fail across shot changes. How many frames of context are typically needed to recover after a cut? Does the model provide any mechanism to detect shot boundaries automatically?

4. **Training data efficiency:** Figure 6 shows a power-law relationship, but what is the saturation point? Would additional data continue to improve performance, or are there diminishing returns?

5. **Object parts vs. whole objects:** The SA-V dataset includes both whole objects and parts. How does the model decide whether to segment a part or whole object when a user clicks? Is there ambiguity resolution during inference?

6. **Automatic annotation quality:** The automatic masklet generation uses SAM 2 in the loop. What is the estimated quality of these automatic annotations compared to manual ones? Are there systematic failure modes in the automatic generation?

7. **Encoder backbone:** The paper uses Hiera instead of ViT (used in SAM). What motivated this architectural choice, and would a larger ViT backbone provide similar benefits?

## Minor Comments

- Section D.2.1 mentions 7 correction clicks during training but the text says 8 in SAM. The paper should clarify this discrepancy.
- The annotation time estimates in the interactive evaluation (Section F.1.2) rely on assumptions from prior work. A user study validating these timing estimates would strengthen the claims about interaction efficiency.
- Table 1's "Mask Alignment Score" definition could be clarified—it measures the percentage of masks with IoU > 0.75 compared to Phase 1, but it's unclear if this is frame-wise or masklet-wise comparison.

## Conclusion

SAM 2 represents a significant contribution to the field of visual segmentation, successfully extending the foundation model paradigm to video. The combination of a well-designed architecture, massive-scale dataset, and comprehensive evaluation establishes a strong baseline for "segment anything in videos." The main areas for potential improvement are the baseline comparisons (which could include more specialized VOS models), the fairness evaluation scope, and deeper analysis of failure modes. Overall, this is a strong paper that will likely have substantial impact on video segmentation and related research areas.

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 8.0, 10.0]
Average score: 9.0
Binary outcome: Accept

=== CALIBRATION EXAMPLE 14 ===

# Review
# Review: Safety Alignment Should Be Made More Than Just A Few Tokens Deep

## Summary

This paper identifies and characterizes a fundamental vulnerability in current LLM safety alignment: the alignment effect is concentrated almost exclusively on the first few output tokens, which the authors term "shallow safety alignment." Through systematic experiments on Llama-2-7B and Gemma-7B models, they demonstrate that this shallowness explains the effectiveness of multiple attack vectors (prefilling attacks, GCG adversarial suffix attacks, decoding parameter exploits, and fine-tuning attacks). The paper makes three main contributions: (1) a unified analysis of how shallow safety alignment underlies various vulnerabilities, (2) a data augmentation approach using "safety recovery examples" to create deeper alignment, and (3) a token-wise constrained fine-tuning objective that protects early token distributions during downstream customization. Both proposed methods show meaningful improvements in robustness while preserving utility.

## Strengths

1. **Valuable Unifying Concept**: The notion of "shallow safety alignment" provides a genuinely useful unifying framework that connects previously disparate attack vectors. Rather than treating each vulnerability in isolation, the paper identifies a root cause that explains multiple phenomena, which is a significant contribution to the field.

2. **Systematic Empirical Analysis**: The paper presents thorough empirical evidence for shallow alignment, including:
   - KL divergence analysis showing concentration on early tokens
   - Per-token gradient dynamics during fine-tuning attacks
   - Systematic testing across multiple attack types and models

3. **Practical Mitigation Strategies**: Both proposed solutions are practical and implementable:
   - The data augmentation approach requires minimal changes to training pipelines
   - The constrained fine-tuning objective can be applied by API providers without requiring users to change their fine-tuning data

4. **Strong Theoretical Foundation for Constrained Fine-tuning**: Section 4 includes rigorous theoretical analysis with theorems characterizing the limiting behaviors of the loss function (small/large βt regimes), gradient analysis, and connections to KL-regularized RL. The interpretation from an RL perspective is particularly illuminating.

5. **Comprehensive Ablation Studies**: The appendix contains extensive ablation studies on hyperparameters (C, p for augmentation; βt configurations for constrained fine-tuning), warmup effects, and comparisons with related work (Vaccine), which strengthen confidence in the results.

6. **Responsible Disclosure**: The paper appropriately discusses the dual-use nature of the work and justifies publication on grounds of improving model safety rather than enabling attacks.

## Weaknesses

1. **Limited Model Scope**: Experiments are confined to 7B parameter models (Llama-2-7B-Chat, Gemma-1.1-7B-IT). It remains an open question whether similar shallow alignment dynamics exist in larger models (70B+), though the authors' reasoning suggests the phenomenon likely generalizes.

2. **Suboptimal Augmentation Implementation**: Due to lack of access to the original alignment data and pipeline, the authors perform augmentation by fine-tuning an already-aligned model rather than training from scratch. This limits the maximum achievable depth of alignment since the model has already converged to the shallow optimum. A more thorough evaluation of deep alignment would require aligning from the base model.

3. **Coherence of Augmented Data**: The safety recovery examples are explicitly acknowledged to be "not coherent in natural language" - the paper shows examples like "Step 1: Gather phosphorus **I cannot fulfill your request." This raises questions about whether this is truly the right approach or merely a workaround. Future work should explore more principled ways to generate such data.

4. **Evaluation via GPT-4 Judge**: The paper relies on GPT-4 as an automatic judge for safety evaluation. While this is common practice, GPT-4 itself may have biases and inconsistencies in safety assessment, particularly for edge cases. Some human evaluation or multiple automated judges would strengthen the evaluation.

5. **Limited Exploration of Attack Adaptations**: The paper demonstrates improved robustness against the attacks considered, but adaptive attackers who know about deep alignment may develop new strategies. The paper acknowledges this limitation but doesn't explore it.

6. **Constrained Fine-tuning Requires Reference Model Access**: The proposed constrained fine-tuning objective requires access to π_aligned to compute the regularization term. For fully open-source models this is fine, but it may not be applicable in all deployment scenarios.

## Questions for Authors

1. **Generalization to Larger Models**: Do you have any evidence (even preliminary) that the shallow safety alignment phenomenon extends to larger models (e.g., Llama-2-70B, GPT-4 class models)? The gradient dynamics and KL divergence patterns might differ at scale.

2. **Why Does KL Divergence Concentrate on Early Tokens?**: Your analysis attributes this to optimization dynamics (no incentive during RLHF to address deeper tokens), but is there any evidence from mechanistic interpretation about whether early tokens are fundamentally easier to modify, or is it purely an optimization artifact?

3. **Data Augmentation from Scratch**: You mention the limitation that your augmentation is applied post-hoc to an already-aligned model. Would the augmentation approach be equally effective if integrated into the initial alignment process? What challenges would arise?

4. **Trade-offs with Legitimate Refusals**: The deep alignment approach trains the model to "recover" from non-refusal prefixes. Could this inadvertently cause the model to backtrack on legitimate refusals that happened to start with unusual phrasing? Have you observed any such behavior?

5. **Connection to Representation Engineering**: Your paper mentions the concurrent work on "short-circuiting" (Zou et al., 2024) as sharing conceptual foundations. Could you elaborate more on how deep safety alignment differs from or complements representation-level interventions?

6. **Scaling the Augmentation**: For production deployment, what would be the recommended scale of safety recovery examples? Is there a risk of overfitting to specific harmful patterns, or does the approach generalize broadly?

7. **Constrained Fine-tuning Overhead**: While you show the computational overhead is marginal (~5% in Table 12), have you evaluated the memory overhead for very large datasets or models where storing π_aligned probabilities might become significant?

8. **Relationship to Benign Fine-tuning Safety Regression**: Your analysis in Appendix C suggests that benign fine-tuning causes safety regression partly due to the same gradient dynamics. Does deep alignment also help preserve safety during benign fine-tuning, and if so, how?

## Conclusion

This is a strong paper that makes a significant contribution by identifying and characterizing the shallow safety alignment problem and providing practical mitigation strategies. The unifying conceptual framework and comprehensive empirical analysis are the paper's main strengths. The proposed solutions show meaningful improvements and the theoretical analysis of the constrained fine-tuning objective is rigorous. The main limitations relate to the scope of experiments and the implementation constraints due to lack of access to alignment pipelines. Overall, this work provides valuable insights for the community and should stimulate important follow-up research on deeper safety alignment approaches.

# Actual Human Scores
Individual reviewer scores: [10.0, 8.0, 10.0, 10.0]
Average score: 9.5
Binary outcome: Accept
