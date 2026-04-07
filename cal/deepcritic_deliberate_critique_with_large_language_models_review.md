=== CALIBRATION EXAMPLE 65 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title**: Appropriately reflects the core contribution (deliberate critique via a two-stage framework).
- **Abstract**: Clearly states the problem (shallow critiques), proposed solution (two-stage training with SFT on deliberate critiques and RL), and key results (outperforms existing critics on error identification and aids refinement). Claims are specific and appear supported by the experiments. No major issues.

### Introduction & Motivation
- Well-motivated: highlights the need for scalable oversight and identifies the weakness of current LLM critics (superficial critiques in math reasoning).
- Contributions are clearly stated: (1) a two-stage pipeline (SFT on curated deliberate critiques, then RL) that teaches LLMs to critique with multi‑perspective verification and meta‑critiquing; (2) extensive experiments showing improved error identification and refinement capabilities; (3) demonstration of test‑time scaling for both critics and generators.
- One minor concern: the claim that the framework can be effectively applied to subjective domains is only briefly supported in Appendix M with a small summarization experiment. This should be either toned down in the introduction or supported with more substantial evidence.

### Method / Approach
- **Problem formulation**: Clear and standard.
- **SFT data generation**: 
  - The iterative process (initial critique → in‑depth critique → merging) is innovative and well‑described.
  - However, the seed data heavily relies on Qwen2.5‑72B‑Instruct. While self‑improvement experiments (Section 4.4) show some gains when using the same 7B model, performance is lower, raising questions about the generality of the approach when a strong teacher is unavailable. The authors should discuss this limitation.
  - Filtering criteria for SFT data (keep only solutions where in‑depth judgments match ground‑truth for all steps) ensures high quality but may reduce diversity. The impact of this filtering is not ablated.
- **RL data and training**:
  - Two RL data sources: human‑labeled (PRM800K) and auto‑annotated via Monte Carlo sampling. The auto‑annotation method follows prior work (Wang et al., 2024) and includes careful filtering. However, the actual noise level in the auto‑annotated data is not quantified; the robustness experiment (Appendix K) uses 20% synthetic noise, which may not reflect real conditions.
  - Reward design is simple (accuracy‑only). The authors tried an informativeness‑aware reward but found no improvement (Appendix L). This is acceptable, but a more detailed analysis of why the additional reward failed would be helpful (e.g., reward hacking leading to longer but uninformative critiques).
  - The RL algorithm (GRPO) is appropriate, and hyper‑parameters are provided.
- **Overall**: The methodology is technically sound and reproducible, but there are some open questions about data dependency and reward design.

### Experiments & Results
- **Benchmarks and baselines**: A comprehensive set of error‑identification benchmarks (MR‑GSM8K, PRM800K, ProcessBench) and a wide range of baselines (PRMs, LLM critics, reasoning‑enhanced models) are used. This is a strength.
- **Main results (Table 1)**:
  - The SFT model (DeepCritic‑7B‑SFT) shows a large improvement over the base model (20‑point F1 gain), validating the quality of the seed data.
  - RL further boosts performance, with the best model (DeepCritic‑7B‑RL‑PRM800K) outperforming all LLM critics, including GPT‑4o and DeepSeek‑R1‑Distill models of similar size.
  - However, Qwen2.5‑Math‑PRM‑7B (a PRM baseline) achieves higher F1 scores on several benchmarks. The authors argue that this model is trained on larger in‑house data and lacks actionable feedback, but this is not fully substantiated. A direct comparison on refinement tasks with Qwen2.5‑Math‑PRM‑7B would strengthen the claim that DeepCritic provides more useful critiques.
- **Ablation studies (Section 4.3)**: Effectively demonstrate that both step‑wise critique generation and in‑depth critiques contribute to performance. The analysis of self‑correction behavior during inference (Figure 3) is insightful.
- **Test‑time scaling (Section 5)**:
  - Majority voting improves critic performance, and DeepCritic shows better scaling than baselines (Appendix G).
  - Verified majority voting and critique‑based refinement experiments show that DeepCritic can enhance generator performance, even when supervising larger models (72B, GPT‑4o). This is a strong result.
  - However, the refinement comparison with DeepSeek‑R1‑Distill‑Qwen‑7B is confounded by answer leakage (poor instruction‑following), as noted by the authors. This is appropriately flagged.
- **Statistical rigor**: The main evaluation uses a single sample per input (Pass@1). No confidence intervals or statistical significance tests are reported. For ICLR, it is important to provide some measure of variability (e.g., standard deviations over multiple runs or bootstrap confidence intervals) to ensure the improvements are reliable.
- **Generalization claims**: The paper focuses on math reasoning. The summarization experiment in Appendix M is a positive step but is limited in scale (200 test samples) and may not sufficiently support broad claims about subjective domains. Either temper these claims or provide more extensive evaluations.

### Writing & Clarity
- The paper is generally well‑written, with clear figures and tables. The methodology and experiments are described in sufficient detail.
- Some sections are dense (e.g., the data generation pipeline) but remain understandable.
- The appendix provides comprehensive details (prompts, hyper‑parameters, additional results), aiding reproducibility.

### Limitations & Broader Impact
- **Limitations**: Not explicitly discussed in a dedicated section. The paper should acknowledge:
  1. Dependence on a strong teacher model for high‑quality seed data (though self‑improvement is possible with a performance drop).
  2. The auto‑annotation pipeline’s reliance on a separate generator, which may introduce noise and bias.
  3. The focus on step‑by‑step math solutions; generalization to other formats (e.g., free‑form reasoning) is untested.
  4. Computational cost of the two‑stage training and the Monte Carlo sampling for auto‑annotation.
- **Broader impact**: Briefly mentioned in the ethics statement. Should also discuss potential negative societal impacts (e.g., misuse for generating misleading critiques, environmental costs of training) and mitigations.

### Overall Assessment
This paper presents a novel and effective two‑stage framework for enhancing the critique capabilities of LLMs, specifically for mathematical reasoning. The core idea—teaching models to generate deliberate critiques through iterative, multi‑perspective verification and meta‑critiquing—is innovative and well‑executed. Experiments demonstrate substantial improvements over strong baselines in error identification and show promising test‑time scaling and refinement utility. However, the paper would be strengthened by: (1) more rigorous statistical reporting, (2) a clearer comparison with the strongest PRM baseline (Qwen2.5‑Math‑PRM‑7B) on refinement tasks, (3) a more tempered or better‑supported claim about generalization to subjective domains, and (4) an explicit discussion of limitations. Despite these concerns, the contribution is significant and meets ICLR’s standards for novelty, technical soundness, and empirical validation. With revisions addressing the points above, the paper would be a strong candidate for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces DeepCritic, a two-stage framework to enhance the math critique ability of Large Language Models (LLMs). The method first generates a dataset of 4.5K "deliberate" critiques via an iterative process using a strong teacher model (Qwen2.5-72B-Instruct), incorporating multi-perspective verification and meta-critiquing for each reasoning step, and uses this for Supervised Fine-Tuning (SFT). A second Reinforcement Learning (RL) stage further improves performance using either human-labeled (PRM800K) or automatically constructed data. The resulting 7B critic significantly outperforms existing LLM critics and process reward models on multiple error identification benchmarks and aids in refining generator outputs.

### Strengths
1. **Well-motivated and clear problem formulation**: The paper clearly identifies a critical limitation of current LLM critics—their tendency to produce shallow, echoic critiques—and provides empirical examples. This establishes a strong foundation for the proposed solution.
2. **Innovative and methodical pipeline**: The two-stage SFT+RL approach is sound. The novel iterative critique generation process (initial critique → in-depth critique → merged deliberate critique) is a clever way to inject multi-perspective reasoning and self-correction capabilities into the training data, moving beyond simple distillation.
3. **Strong and comprehensive empirical results**: The model (DeepCritic-7B-RL-PRM800K) achieves state-of-the-art or highly competitive performance on multiple benchmarks (MR-GSM8K, PRM800K, ProcessBench), outperforming much larger models (GPT-4o, Qwen2.5-72B) and specialized reasoning models (DeepSeek-R1). The paper includes extensive ablations (direct distillation, initial-critic-only), self-improvement experiments, test-time scaling analysis (majority voting), and refinement applications, convincingly demonstrating the framework's effectiveness.
4. **Thorough analysis and ablation studies**: The paper provides valuable insights through component ablations (showing the necessity of both initial and in-depth critiques), analysis of self-correction behavior, investigation of label noise robustness, and experiments on reward design. The case studies in the appendix effectively illustrate the model's deliberate reasoning process.
5. **Demonstration of practical utility**: Beyond error identification, the paper shows the critic's value in downstream applications: improving generator performance via verified majority voting and, more importantly, via actionable feedback for refinement, including in a weak-to-strong supervision setting.

### Weaknesses
1. **Limited discussion of computational cost and scalability**: While the method is presented as a step towards scalable oversight, the computational cost of the data curation process (using a 72B model for iterative generation and Monte Carlo rollouts for RL data) is not quantified. The RL stage's constraint on solution length due to GPU memory is noted but not deeply analyzed for its impact.
2. **Comparison with the strongest baseline could be more nuanced**: The Qwen2.5-Math-PRM-7B model outperforms DeepCritic on some benchmarks. The authors correctly note it was trained on a larger in-house SFT dataset and lacks refinement utility, but a more direct comparison or discussion on the trade-offs (capability vs. data efficiency, critique quality vs. judgment accuracy) would strengthen the paper.
3. **Generalization claims could be more substantiated**: The paper mentions the framework can be applied to subjective domains (Appendix M) and shows preliminary results on summarization. However, this experiment is relatively lightweight compared to the main math focus. A more rigorous evaluation across diverse domains would better support the claim of generalizability.
4. **Potential overfitting to benchmark formats**: The evaluation relies heavily on existing benchmarks (GSM8K, MATH, etc.). While the performance gains are impressive, there is limited analysis of how the critic performs on genuinely novel or out-of-distribution problem types not represented in these datasets.
5. **Some technical details are deferred to appendices**: Key prompts for data generation and evaluation are in the appendix. While this is acceptable, integrating the core prompt strategies into the main methodology section could improve readability for those wanting to understand the core innovation quickly.

### Novelty & Significance
**Novelty**: The core novelty lies in the structured, iterative process for generating "deliberate" training critiques that teach the model to reason from multiple perspectives and critique its own initial assessments (meta-critiquing). This goes beyond prior work that focuses on direct verification or single-pass critique generation. The two-stage pipeline combining this curated SFT data with RL (using both human and automated data) is also a well-executed synthesis of existing techniques for a new purpose.
**Significance**: The work addresses a critical challenge in the evolution of LLMs: scalable and accurate oversight. By significantly improving the critique ability of a relatively small model (7B), it demonstrates a promising path toward automated supervision and refinement, which is highly relevant for the AI safety and alignment community. The strong empirical results, test-time scaling properties, and refinement applications make it a substantive contribution likely to influence future research.

### Suggestions for Improvement
1. **Provide a more detailed analysis of computational efficiency and scaling**: Include estimates of the compute cost for data generation (teacher model inference, Monte Carlo rollouts) and discuss the feasibility of scaling this approach. Could the pipeline be made more efficient (e.g., with smaller teacher models)?
2. **Deepen the comparison with Qwen2.5-Math-PRM-7B**: Conduct a targeted analysis or head-to-head comparison on a subset of tasks focusing on critique quality (e.g., informativeness, usefulness for refinement) rather than just accuracy, to better articulate the advantages of the DeepCritic approach.
3. **Strengthen the generalizability evaluation**: Expand the experiments on subjective domains (e.g., code critique, factual reasoning) with the same rigor as the math evaluation, using established benchmarks to more convincingly demonstrate the framework's broad applicability.
4. **Explore and discuss failure modes**: Include a qualitative analysis of cases where DeepCritic fails. Does it fail on particular types of errors (logical vs. arithmetic) or problem complexities? This would provide valuable insights for future improvements.
5. **Clarify the relationship between model size and critique ability**: The paper shows strong results with a 7B model. A brief discussion or experiment on how the gains from the DeepCritic pipeline might scale with the base model's size (e.g., applying it to a 1B or 30B model) would be insightful.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare with state-of-the-art critique training methods beyond simple distillation.** The paper only compares with PRMs and prompted LLMs. Stronger baselines like Critique-CoT or recent preference-based critique training methods should be included. Without this, the claimed superiority of the two-stage pipeline is not fully validated.
2. **More rigorous refinement experiments across diverse generators and datasets.** The refinement tests use only two generators (Qwen2.5-7B/72B) on two math datasets. To substantiate the claim that the critic "helps the LLM generator refine erroneous steps," experiments with more diverse generators (including GPT-4, Claude) and more challenging/out-of-domain benchmarks are needed.
3. **Ablation on RL data scale and mixture.** The impact of RL data size and the mix of human vs. auto-annotated data is not studied. A scaling curve showing performance vs. RL data size would clarify the data efficiency of the method and whether gains are simply due to more data.
4. **Self-improvement comparison with other self-training methods.** The self-improvement experiment only shows improvement over the base model. To claim the pipeline is effective for self-improvement, a comparison with standard self-training or self-rewarding methods is necessary.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantify the occurrence of claimed "deliberate" behaviors.** The paper asserts the model performs multi-perspective verification and meta-critiquing, but only provides a limited manual analysis (Figure 3). A systematic, quantitative evaluation of how often these behaviors appear in generated critiques (e.g., via automated tagging or LLM judgment on a large sample) is needed to validate the core mechanism.
2. **Analyze the quality and noise of auto-annotated RL data.** The auto-annotation process is central to scalability. The paper should report the accuracy of the auto-generated labels (e.g., via human audit on a sample) and analyze how label noise correlates with RL performance. This is critical to trust the "automated scalable oversight" claim.
3. **Error analysis of where the critic fails.** The paper lacks a breakdown of failure modes. Categorizing errors (e.g., missing subtle mistakes, hallucinating errors, misidentifying error step) would reveal the model's limitations and whether "deliberate" critiquing actually addresses them.

### Visualizations & Case Studies
1. **Side-by-side comparison of critique depth against strong baselines.** Show a few challenging examples where baseline critics (GPT-4o, DeepSeek-R1) produce shallow or incorrect critiques, and DeepCritic's critique is deeper/correct. This would visually substantiate the claimed advantage.
2. **Failure case studies.** Include examples where DeepCritic produces overly verbose, unhelpful, or incorrect critiques. This would provide a honest assessment of limitations and whether the "deliberate" process can sometimes lead to overthinking or reward hacking.

### Obvious Next Steps
1. **Scale up the auto-annotation and RL data significantly.** The paper uses only 14.3K auto-annotated samples. A logical next step is to show that performance continues to improve with more auto-annotated data (e.g., 100K+), reinforcing the scalability claim.
2. **Apply the pipeline to at least one more complex, non-mathematical domain.** The summarization experiment in the appendix is superficial. To convincingly argue for generalizability, the full two-stage training should be applied to a domain like code or scientific reasoning, with proper benchmarks.
3. **Integrate the critic into an online RL loop.** The paper mentions this as a future direction but does not provide even preliminary results. Demonstrating that DeepCritic can be used for online policy improvement via iterative critique-and-refine would strengthen the contribution.

# Final Consolidated Review
## Summary
This paper introduces DeepCritic, a two-stage framework to enhance the critique ability of large language models (LLMs) for mathematical reasoning. The method first curates a dataset of "deliberate" critiques through an iterative process that combines initial and in-depth critiques with multi-perspective verification and meta-critiquing, used for supervised fine-tuning. Reinforcement learning then further boosts performance using human or automatically annotated data. The resulting 7B model outperforms existing LLM critics on error identification benchmarks and effectively aids in refining generator outputs through detailed feedback.

## Strengths
- **Innovative and well-motivated pipeline**: The iterative critique generation process (initial critique → in-depth critique → merged deliberate critique) explicitly teaches models to reason from multiple perspectives and self-correct, moving beyond simple distillation. Ablation studies confirm that both components contribute to performance gains.
- **Strong and comprehensive empirical validation**: The DeepCritic-7B-RL-PRM800K model achieves state-of-the-art or highly competitive results across multiple benchmarks (e.g., MR-GSM8K, PRM800K, ProcessBench), outperforming larger models (GPT-4o, Qwen2.5-72B) and specialized reasoning models (DeepSeek-R1). Experiments include thorough ablations, self-improvement tests, and analysis of test-time scaling.
- **Demonstrated practical utility**: Beyond error identification, the critic enhances generator performance via verified majority voting and, more importantly, through actionable feedback for refinement. This is shown in weak-to-strong supervision settings, improving outputs of 72B and GPT-4o generators.

## Weaknesses
- **Limited support for generalization claims**: The paper asserts that the framework "can also be effectively applied to subjective domains," but only provides a lightweight summarization experiment (Appendix M) with 200 samples. This insufficient evidence weakens the claim of broad applicability beyond mathematical reasoning.
- **Dependence on a strong teacher model for optimal performance**: Self-improvement experiments (Section 4.4) show that using the same-sized model for data curation yields lower gains, indicating that the pipeline's effectiveness may rely on access to a more capable critic. This limitation is not explicitly discussed in the main text.
- **Incomplete analysis of auto-annotated data quality**: While robustness to synthetic label noise is tested (Appendix K), the actual noise level in the auto-annotated RL data—critical for the "automated scalable oversight" claim—is not quantified via human audit or similar validation.
- **Computational cost and scalability concerns are underdeveloped**: The pipeline involves expensive steps (e.g., using a 72B teacher model for iterative generation, Monte Carlo rollouts for auto-annotation), but resource requirements, trade-offs, and scaling implications are not analyzed, hindering assessment of practical adoption.
- **Comparison with the strongest baseline lacks depth**: Qwen2.5-Math-PRM-7B outperforms DeepCritic on some benchmarks (Table 1). The authors note it uses larger in-house data and lacks refinement utility, but a direct comparison on critique quality (e.g., informativeness, usefulness for refinement) would better substantiate DeepCritic's advantages.

## Nice-to-Haves
- Quantitative analysis of how often the model exhibits "deliberate" behaviors (e.g., multi-perspective verification, meta-critiquing) on a larger scale to validate the core mechanism.
- Ablation studies on RL data scale and mixture to inform data efficiency and optimal curation strategies.
- Expansion of refinement experiments to include more diverse generators (e.g., different architectures) and non-mathematical domains to better assess general utility.
- Error categorization to identify common failure modes (e.g., missing subtle errors, hallucinating mistakes) and link them to the critique process.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Formatting nitpick about technical details in appendices**: The prompts and hyperparameters are adequately provided in appendices, which is standard for reproducibility.
- **Demand for statistical significance tests or confidence intervals**: In LLM evaluation for large-scale benchmarks, single-run evaluation is common practice; imposing additional statistical rigor is not a standard requirement for this paper's context.
- **Request to compare with all state-of-the-art critique training methods beyond those included**: The paper compares with relevant PRMs and LLM critics; demanding exhaustive comparison is outside its stated scope.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add a dedicated limitations section in the main text to explicitly discuss teacher model dependency, computational costs, domain generalizability, and potential overfitting to benchmark formats.
- Conduct a human audit or systematic LLM judgment on a sample of auto-annotated RL data to quantify label accuracy and report it, strengthening the scalability claim.
- Perform a head-to-head comparison with Qwen2.5-Math-PRM-7B on critique informativeness, using human or LLM judges to assess feedback quality for refinement tasks.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 4.0]
Average score: 4.7
Binary outcome: Reject
