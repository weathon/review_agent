=== CALIBRATION EXAMPLE 85 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title accurately reflects the core contribution. The abstract is excellent: it clearly states the problem (long, complex multimodal policies), identifies the research gap (prior work focuses on text-only safety or simple templates), and concisely summarizes the contributions (new MPI task, two datasets, TriMPI framework with PolicyRollout). The claimed results ("70.7% and 79.4% absolute gains") are specific and set high expectations for the experiments.

**Introduction & Motivation**
The motivation is compelling and well-grounded. The computational cost argument (fixed policy tokens vs. short queries) is practical, and the shift to multimodal policies is timely. The related work is appropriately surveyed, clearly distinguishing MPI from prompt compression and deliberative alignment. The unique challenges of MPI are well-articulated. The contributions are explicitly listed at the end of the section.

**Method / Approach**
*   **Problem Formulation (Section 2):** Clear and precise.
*   **Baselines (Section 4.1):** Direct SFT and CoT SFT are appropriate and standard baselines.
*   **TriMPI Framework (Section 4.2):** The three-stage pipeline is logically structured. The Visually-Masked Continual Pretraining (VM-CPT) stage is a straightforward adaptation of CPT for multimodal data, though the justification for masking *all* visual tokens is empirical rather than principled. A brief ablation or discussion on this choice would strengthen the method.
*   **PolicyRollout (Section 4.3):** This is the key algorithmic innovation. The idea of augmenting the rollout space with policy-conditioned responses while computing policy gradients only on the no-policy path is clever and directly addresses the train-inference misalignment. However, the description of how the advantages are computed for the combined group of rollouts is slightly ambiguous. Equation 3 and the text imply that the \(2G\) responses (G from no-policy, G from policy) are pooled into a single group for advantage estimation, but this should be stated explicitly. The phrase "group-based advantage estimation" is used without specifying the group size. This detail is important for reproducibility.

**Experiments & Results**
*   **Datasets (Section 3):** A major strength. ClevrPolicy provides controlled complexity, and GTAPolicy offers a realistic, low-data scenario. The zero-shot in-context results (Table 1) effectively establish benchmark difficulty. The dataset creation is well-documented in the appendix.
*   **Main Results (Section 5.1):** The improvements are substantial and clearly presented in Table 2. The ablations convincingly show the contribution of each TriMPI component (RL, VM-CPT, PolicyRollout). The efficiency analysis (Figure 6) is compelling. The discussion on DAPO vs. GRPO performance across datasets is insightful.
*   **Generalization Evaluations (Sections 5.2-5.4):** These evaluations are thorough and go beyond standard accuracy. Policy Override tests adaptability to new in-context rules. Policy Referral (LLM-as-judge) is a clever way to probe the internalization of policy *knowledge*, not just behavior. Policy In-Context demonstrates that TriMPI's benefits are not solely due to prompt removal. Results consistently favor TriMPI.
*   **Robustness to Catastrophic Forgetting (Section 5.5):** Evaluating on MMMU-Pro and MMLU-Pro (and safety in the appendix) addresses a critical concern. The finding that TriMPI is more robust, especially on the small GTAPolicy dataset, is significant.
*   **Error Analysis (Section 5.6):** Provides valuable qualitative and quantitative insights. The branching error analysis (Figure 14 in Appendix) quantitatively shows that TriMPI leads to more policy-grounded reasoning.
*   **Reproducibility:** The appendix provides extensive details: hyperparameters (Table 5), dataset statistics (Table 7), CoT generation process (Appendix D), and evaluation prompts (Figure 9). The promise of releasing code meets ICLR standards.

**Writing & Clarity**
The paper is exceptionally well-written and structured. The figures are informative. Minor points:
1.  Figure 4's caption ("Direct SFT CoT SFT") on the right side is confusing as the right diagram illustrates TriMPI.
2.  The description of PolicyRollout could be more precise regarding the advantage estimation over the pooled rollouts.
3.  Acronym RLVR is used in Section 4.2 without definition (though inferable).

**Limitations & Broader Impact**
The limitations section appropriately notes the need for larger real-world datasets, more sophisticated CPT strategies, and handling task mixtures. The ethics statement is standard. A brief discussion on the societal implications of efficient policy internalization (e.g., potential for misuse, benefits for accessibility) could be added but is not essential.

### Overall Assessment
This is a strong, well-executed paper that makes a clear and novel contribution. It introduces a well-motivated task (Multimodal Policy Internalization), provides two high-quality benchmarks, and proposes an effective training framework (TriMPI) with a clever RL extension (PolicyRollout). The experimental evaluation is comprehensive, demonstrating substantial gains in performance, generalization, and robustness. The main weaknesses are minor: the description of PolicyRollout's advantage computation could be more precise, and the VM-CPT masking strategy lacks a deeper justification. These do not undermine the paper's core contributions. The work is novel, impactful, and meets the high standards for ICLR acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces a new task, Multimodal Policy Internalization (MPI), which aims to embed complex, reasoning-intensive multimodal policies (governing behavior like decision-making and tool use) directly into a model's parameters. This allows the model to follow the policy without having the lengthy policy text present in the input context during inference, improving both efficiency and faithfulness. The authors contribute two new datasets (ClevrPolicy and GTAPolicy) and propose TriMPI, a three-stage training framework featuring a novel RL algorithm called PolicyRollout. Experiments show TriMPI significantly outperforms strong baselines and in-context learning.

### Strengths
1.  **Well-Motivated and Novel Task Formulation:** The paper clearly identifies a practical and emerging problem: the computational inefficiency and performance challenges of lengthy, in-context multimodal policies in conversational agents. It effectively distinguishes MPI from prior work on text-only safety alignment and simpler prompt compression.
2.  **High-Quality Benchmark Construction:** The creation of two new datasets, ClevrPolicy (synthetic, controlled complexity) and GTAPolicy (real-world, low-data regime), is a substantial contribution. They are well-designed to probe different aspects of the problem (reasoning, tool use) and will be valuable for future research.
3.  **Strong Empirical Results and Comprehensive Evaluation:** The proposed TriMPI method achieves impressive performance gains (e.g., ~70% absolute improvement over SFT baselines). The evaluation is thorough, including not just end-task accuracy but also analyses of generalization (policy override), policy knowledge retention (policy referral), efficiency gains, robustness to catastrophic forgetting, and ablation studies. The consistent gains across model sizes and policy complexities are convincing.
4.  **Clear and Reproducible Methodology:** The paper is well-written, the TriMPI framework is clearly explained, and the appendices provide substantial detail on datasets, training procedures, hyperparameters, and prompts, supporting reproducibility.

### Weaknesses
1.  **Modest Algorithmic Novelty:** The core algorithmic contribution, PolicyRollout, is a simple and effective extension to existing GRPO/DAPO algorithms. While it works well, its conceptual simplicity may be seen as incremental. The three-stage framework (CPT + SFT + RL) is a sensible composition of existing techniques.
2.  **Limited Scale and Diversity of Real-World Evaluation:** Although GTAPolicy uses real images, its size (451 training instances) is small. The policies, while complex, are still crafted for research. The paper would be stronger with validation on a larger-scale, more diverse real-world policy internalization task to fully demonstrate practical impact.
3.  **Potential Overfitting in Low-Data Regime:** The results on GTAPolicy show that some baselines (CoT SFT + DAPO) suffer from catastrophic forgetting, indicating overfitting. While TriMPI mitigates this, the problem persists to a degree. More analysis or techniques specifically for robust low-data internalization would be beneficial.

### Novelty & Significance
**Novelty:** The paper is the first to formally define and tackle the problem of internalizing *multimodal*, *reasoning-intensive* policies. The datasets and the PolicyRollout algorithm are novel contributions.
**Significance:** The work addresses a critical bottleneck for deploying efficient and reliable multimodal agents. The performance improvements are substantial, and the provided datasets, training framework, and analysis establish a strong foundation for an important new research direction. The demonstrated efficiency gains (reduced prompt tokens and latency) have direct practical relevance.

### Suggestions for Improvement
1.  **Scale Up the Real-World Benchmark:** The most important future step is to create or evaluate on a larger, more complex real-world dataset, perhaps in collaboration with industry partners, to stress-test the method under conditions closer to deployment.
2.  **Deeper Analysis of PolicyRollout:** Provide more insight into *why* PolicyRollout works so well. For instance, analyze the quality/diversity of the policy-augmented rollouts compared to standard rollouts, or show how the advantage estimates differ.
3.  **Explore Alternative Architectures or Training Schemes:** Briefly discuss or experiment with whether other model architectures (e.g., more modular designs) or alternative training paradigms (e.g., iterative distillation) could be even more effective for MPI, framing TriMPI within a broader design space.

**Overall, this is a strong paper suitable for ICLR. It introduces a well-motivated new task, provides valuable resources (datasets), and presents a simple yet highly effective method validated through extensive experiments. The work is timely, reproducible, and likely to influence future research on efficient and aligned multimodal agents.**

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **No comparison to state-of-the-art prompt compression or parameter-efficient fine-tuning (PEFT) methods.** The paper dismisses soft prompting and similar methods but provides no experimental comparison to approaches like LoRA fine-tuning *with* the policy kept in context. A crucial baseline is: train with LoRA (or full fine-tuning) on (policy + query) -> answer, then at inference, use the same adapted model but *without* the policy. This tests if simple fine-tuning can internalize the policy, which would undermine the need for the complex TriMPI framework.
2. **Missing ablation on the necessity of the three-stage pipeline.** The contributions of each stage (VM-CPT, CoT-SFT, RL) are shown, but a critical ablation is missing: **training RL directly from the base model (or after VM-CPT) on non-CoT data**. This would test if the CoT-SFT stage is truly necessary or if RL can learn policy compliance from scratch with sufficient exploration (aided by PolicyRollout). Its absence leaves the role of CoT-SFT unclear.
3. **No experiment scaling policy length/complexity within a single dataset.** The paper uses policies of fixed complexity per dataset (e.g., N=6 trees). To robustly claim that TriMPI is better for *complex* policies, an experiment within ClevrPolicy is needed: sweep policy depth (N=2,4,6,8) and show the performance gap between TriMPI and baselines widens with complexity. Currently, the claim relies on cross-dataset comparison.
4. **Lack of a "policy distillation" baseline.** A strong, obvious baseline is to use a powerful teacher model (e.g., Claude-4) with the policy in-context to generate (query -> answer) training data, then distill this into the student model via SFT. This tests if the problem is simply one of data quality versus the need for the proposed multi-stage RL exploration.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of *what* is internalized: policy rules versus input-output mappings.** The Policy Referral metric is a good start but is a black-box LLM judge score. A more rigorous analysis is needed: after internalization, probe the model's ability to **explicitly recite or answer questions about the policy rules** in isolation (e.g., "What does Condition 1.2 specify?"). This distinguishes true rule internalization from learning sophisticated input-output correlations.
2. **Causal analysis of PolicyRollout's improvement.** The paper claims PolicyRollout enables "more grounded exploration." This needs verification: analyze the reward distribution (e.g., average reward) of policy-conditioned rollouts versus standard rollouts during training. If policy-conditioned rollouts consistently yield higher rewards, it confirms they provide better learning signals. Without this, the mechanism is speculative.
3. **Sensitivity analysis of the visual masking in VM-CPT.** The VM-CPT stage masks all visual tokens from the loss. An analysis is needed to understand if this prevents the model from internalizing the *visual components* of multimodal policies (e.g., the demo images in ClevrPolicy-M). Compare embedding similarity or retrieval accuracy for policy images before/after VM-CPT.

### Visualizations & Case Studies
1. **Visualization of the exploration space during RL with/without PolicyRollout.** Show a t-SNE or similar projection of the generated response embeddings (or their semantic features) from the rollout phase, color-coded by whether they were generated with or without the policy in-context. This would visually demonstrate that PolicyRollout expands the coverage of the policy-compliant response space.
2. **Case studies of "policy override" failures.** The paper shows override results but does not dissect *why* models fail. Provide concrete examples where an internalized model incorrectly prioritizes its internalized policy over a conflicting in-context update, revealing the limits of the generalization claim.

### Obvious Next Steps
1. **Apply the method to a real, proprietary conversational agent policy.** The paper's motivation is grounded in real systems with long policies. The most obvious next step is to partner with the industrial co-authors to test TriMPI on an actual Alexa+ or similar policy, even if results are anonymized. This is necessary to validate the practical utility beyond constructed benchmarks.
2. **Investigate mixture-of-policies internalization.** The limitation section mentions internalizing mixtures of tasks. A direct next step is to train a single model on multiple policies (e.g., from ClevrPolicy and GTAPolicy) and evaluate cross-policy interference and the need for task identifiers, which is a core challenge for deployment.
3. **Benchmark the inference computational savings end-to-end.** The efficiency analysis only measures prompt token reduction and prefill time. The true system saving should include reduced memory bandwidth and total latency per query, especially for long-context models. A simple comparative throughput (queries/second) benchmark under a fixed hardware budget is needed.

# Final Consolidated Review
## Summary
This paper introduces Multimodal Policy Internalization (MPI), a new task aiming to embed complex, reasoning-intensive multimodal policies (e.g., for decision-making or tool use) directly into model parameters, eliminating the need for lengthy in-context policy prompts during inference. The authors contribute two new benchmarks (ClevrPolicy for controlled analysis, GTAPolicy for real-world low-data regimes) and propose TriMPI, a three-stage training framework featuring a novel reinforcement learning extension called PolicyRollout. Experiments show substantial performance gains, improved generalization, and robustness to catastrophic forgetting.

## Strengths
- **Well-motivated and novel task definition.** The paper clearly identifies a practical gap: prior work focuses on text-only safety or simple template compression, while complex multimodal policies for conversational agents are unexplored. The formulation is timely and addresses both computational inefficiency and policy-following reliability.
- **High-quality benchmark creation.** The two new datasets, ClevrPolicy (synthetic, controlled complexity) and GTAPolicy (real-world, low-data), are carefully designed to probe different aspects of the problem and are accompanied by thorough documentation (e.g., policy generation, evaluation metrics). The zero-shot in-context baselines (Table 1) effectively establish their difficulty.
- **Comprehensive and convincing empirical evaluation.** The proposed TriMPI framework achieves large absolute gains (up to ~70% over strong baselines). Evaluation goes beyond accuracy to include generalization (policy override), policy knowledge probing (policy referral), efficiency gains (93.9% prompt token reduction), robustness to catastrophic forgetting, and detailed ablations. Results are consistent across policy complexities and model sizes.

## Weaknesses
- **Algorithmic simplicity of the core innovation.** PolicyRollout, while effective, is a relatively straightforward extension to existing GRPO/DAPO algorithms. The three-stage framework (VM-CPT + CoT-SFT + RL) composes established techniques. The paper would be stronger with deeper analysis of *why* PolicyRollout works so well (e.g., analyzing reward distributions or rollout diversity) and a more principled justification for the VM-CPT visual masking strategy.
- **Limited scale and diversity of real-world validation.** GTAPolicy, while using real images and realistic tool-use policies, is small (451 training instances) and crafted for research. Demonstrating the method's effectiveness on a larger, more diverse real-world policy (e.g., from an actual conversational agent) would strengthen the practical impact claim.
- **Missing baseline comparisons to simple fine-tuning and distillation.** The paper dismisses soft prompting but does not compare against a strong baseline of fine-tuning (e.g., full or LoRA) on (policy + query) → answer pairs and then removing the policy at inference. Similarly, a distillation baseline using a strong teacher model with policy-in-context to generate training data is absent. These comparisons are needed to fully establish the necessity of the proposed multi-stage RL approach.

## Nice-to-Haves
- Scale up the real-world benchmark to include more policies and tasks, or partner with industrial co-authors for a deployment-scale validation.
- Provide a more detailed analysis of PolicyRollout's mechanism, e.g., by visualizing the response embedding space or comparing reward distributions of policy-conditioned vs. standard rollouts.
- Briefly discuss the design space for MPI, such as alternative architectures (modular designs) or training paradigms (iterative distillation), to position TriMPI within a broader context.

## Removed Points 
*These points are flagged to be removed, treat them with caution*
- **"The description of PolicyRollout's advantage computation is ambiguous."** — The paper states the \(2G\) responses are pooled for group-based advantage estimation; Equation 3 and Figure 5, supplemented by appendix details (e.g., rollout batch size), are sufficient for reproduction.
- **"The VM-CPT masking strategy lacks a deeper justification."** — The approach is empirically motivated and shown to work; demanding a theoretical justification is not standard for this type of empirical systems contribution.
- **"Include an experiment scaling policy length/complexity within a single dataset."** — The paper already varies policy complexity (N=2,4,6) in ClevrPolicy and shows TriMPI's gains are more pronounced on complex policies (Table 8). A finer-grained sweep is an extension.
- **"Ablate training RL directly from the base model on non-CoT data."** — The paper includes a Direct SFT baseline on non-CoT data and shows RL on top of CoT SFT significantly improves over it. This variant's absence does not undermine the demonstrated pipeline efficacy.

## Novel Insights
The paper introduces the novel insight that internalizing complex multimodal policies requires not just behavior cloning but also mechanisms to ground exploration in the policy rules during training. The PolicyRollout algorithm operationalizes this by augmenting the RL rollout space with policy-conditioned responses while computing gradients only on the no-policy path, thus aligning training and inference. This approach, combined with policy knowledge injection via continual pretraining, leads to models that not only perform better but also show improved generalization to policy updates and more accurate internal referral to policy rules (as measured by the novel Policy Referral metric).

## Suggestions
- Add a baseline comparison where the model is fine-tuned (with LoRA or full) on (policy + query, answer) pairs and evaluated without the policy at inference, to directly test if simpler fine-tuning can internalize policies.
- Include a distillation baseline where a strong teacher model (with policy in-context) generates training data for SFT, to disentangle the contribution of data quality versus the proposed RL exploration.
- In the error analysis, quantify the proportion of failures due to perception errors (object misidentification) versus reasoning errors (rule hallucination) to better guide future work.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0]
Average score: 7.3
Binary outcome: Accept
