# Composition-Grounded Data Synthesis for Visual Reasoning

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Pretrained multi-modal large language models (MLLMs) demonstrate strong performance on diverse multimodal tasks, but remain limited in reasoning capabilities for domains where annotations are difficult to collect. In this work, we focus on artificial image domains such as charts, rendered documents, and webpages, which are abundant in practice yet lack large-scale human annotated reasoning datasets. We introduce COGS (COmposition-Grounded data Synthesis), a data-efficient framework for equipping MLLMs with advanced reasoning abilities from a small set of seed questions. The key idea is to decompose each seed question into primitive perception and reasoning *factors*, which can then be systematically recomposed with new images to generate large collections of synthetic question-answer pairs. Each generated question is paired with subquestions and intermediate answers, enabling reinforcement learning with factor-level process rewards. Experiments on chart reasoning show that COGS substantially improves performance on unseen questions, with the largest gains on reasoning-heavy and compositional questions. Moreover, training with a factor-level mixture of different seed data yields better transfer across multiple datasets, suggesting that COGS induces generalizable capabilities rather than dataset-specific overfitting. We further demonstrate that the framework extends beyond charts to other domains such as webpages. We release the code and data at https://cogsynthesis.github.io.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes COGS (COmposition-Grounded instruction Synthesis), a three-stage pipeline to improve VLMs’ reasoning in artificial image domains (charts, rendered documents, webpages). Stage 1 decomposes a seed set of questions into interpretable perception and reasoning factors; Stage 2 recomposes sampled factors with new images to synthesize new questions; Stage 3 fine-tunes VLMs using GRPO with the generated questions. Experiment demonstrates the effectiveness of COGS on ChartQAPro and MMC-Bench.

### Strengths
1.	The pipeline of COGS is clear and easy to follow. The method is intuitive.

2.	Factor-level sub-questions make the supervision more transparent and are practical for error analysis.

### Weaknesses
1.	Evaluation protocol risks leakage. The use of 33% of the test set as seeds violates the test-only usage and can tune the pipeline to the test distribution even without answer leakage. I checked the ChartQAPro benchmark, and it has 1341 charts for 1948 questions. Different questions may target the same image chart; this makes image-level leakage likely.

2.	The process reward uses LLM-as-a-judge, which can cause significant training overhead. The paper should report efficiency statistics

### Questions
Please see Weaknesses

### Soundness
2

### Presentation
2

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
This paper proposes COGS (Composition-Grounded Instruction Synthesis), a data-efficient pipeline to endow MLLMs with visual reasoning skills in artificial image domains like charts, documents, webpages. The method (i) decomposes a small set of seed questions into atomic perception/reasoning factors, (ii) recomposes sampled factors with new images to synthesize QA pairs with sub-questions and sub-answers, and (iii) fine-tunes an MLLM via GRPO using process rewards computed at the factor level.  Experiments on ChartQAPro and VisualWebBench show consistent gains over strong open baselines, ablations on mixing factor pools across datasets, reward model, factor break down further demonstrate the effectiveness and scalability of the method.

### Strengths
1. Clear, modular idea with practical upside. Factorizing seed questions then reusing those factors across unlabeled images is straightforward and domain-agnostic. The pipeline is well-motivated and broadly applicable.
2. Process reward design with both theoretical and experimental justification. The ProcessRM-max objective is motivated by a simple but persuasive order-preservation argument and turns out to be effective.
3. Transfer via data-level and factor-level mixing. The author provides a valid ablation on generalization over mixture of datasets, indicating the method captures reusable structure, not just surface patterns.

### Weaknesses
**1. Positioning in prior data-synthesis work is thin.**

 The introduction and related work don’t really situate the paper within the broader line of data-synthesis methods that first generate sub-questions/functions and then recombine them into new examples. It would help to spell out what’s borrowed vs. what’s new, why existing approaches fall short for this setting, and how the paper’s factorization addresses those limits. Please cite a few representative strands to make the attribution explicit. See [1-3].

**2. Comparisons skip broader synthesis and RL frameworks.**

 The experiments compare against chart/GUI-specific generators, but there are notable data-synthesis pipelines in RL and process-supervised training that should be part of the picture. Even a small, controlled head-to-head with one or two representative frameworks (same base model, same budget) would clarify what the proposed method adds beyond existing synthesis strategies. Some works can be found in [4-5].

**3. Reliability of synthetic sub-questions/answers isn’t audited.**

Decompositions can be inconsistent or non-minimal, and recomposition relies on an LLM to produce sub-answers. The order-preserving argument assumes a particular noise structure, but we don’t see concrete diagnostics. Although the results demonstrate the effectiveness of the pipeline, we still need more diagnostics like sub-answer error rates; factor-label drift across seeds/images; sensitivity of GRPO to sub-reward noise. Right now, reliability is assumed rather than demonstrated.

**4. Model ablation is narrow.**

Results are limited to Qwen2.5-VL-7B. It’s unclear whether the gains persist across families (and weaker/stronger baselines) or scale with model size. A light grid—e.g., a smaller and a larger open model, plus a different family—would make the real-world applicability much clearer.

[1]. Self-Instruct: Aligning Language Models with Self-Generated Instructions

[2]. WizardLM: Empowering large pre-trained language models to follow complex instructions

[3]. Automatic Instruction Evolving for Large Language Models

[4]. ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search

[5]. Self-Rewarding Language Models

### Questions
1. Decomposition quality: What is the inter-run agreement for factor labels/sub-questions on the same seed? Any filtering or self-consistency checks?


2. Mixture strategy: In factor-level mixing, how often do cross-dataset factors actually co-occur in recomposed questions? Can you provide some domain drift failure cases?


3. Factor generation: When MLLM decomposes the main question and generate the subquestion with the Factor, how to make sure the number of the generated Factor is within a reasonable range?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents COGS, a framework for augmenting the reasoning capabilities of multimodal large language models (MLLMs) in domains lacking large annotated datasets, such as charts and webpages. The key idea is to decompose seed questions into primitive factors, and then systematically recompose these factors with new images to generate diverse, compositional training data.

### Strengths
COGS factorizes a small seed set into reusable perception/reasoning factors, recomposes them with new images to create diverse, grounded QA pairs , and uses the factor structure to supply process-level rewards, yielding richer supervision than final-answer matching alone.

### Weaknesses
1. The authors state that the framework *does not require ground-truth subquestion answers*, but this appears to hold only for the *seed set*: during data synthesis, each sample includes LLM-generated sub-answers that are then used for process-level supervision in RL. If these pseudo-labels are noisy, errors may accumulate and be amplified through factor-level recomposition.

2. The manuscript does not isolate the factor pool’s *direct* contribution, emphasizing cross-dataset ablations instead. Clarifying its substantive, incremental value, and distinguishing it from direct decomposing/recomposing method, would make the contribution more transparent.

3. The title is *COMPOSITION-GROUNDED **INSTRUCTION SYNTHESIS**  FOR **VISUAL REASONING***, yet the manuscript does not clearly foreground *instruction synthesis* and *visual reasoning*; it predominantly centers on chart QA. The related-work section likewise emphasizes task descriptions rather than surveying these two threads. Moreover, while the introduction mentions tables and documents, the experiments are confined to a relatively narrow setting. Chart QA can serve as a proxy for visual reasoning, but the heavy focus on charts leaves the broader visual-reasoning aspect underdeveloped.

4. The approach is mainly applicable when a task admits a reliable decomposition into subquestions; for complex cases that are not readily decomposable, its utility is limited.

5. Minor editorial issues: on lines 71 and 205, ***Grouped Rollout Policy Optimization*** should be ***Grouped Relative Policy Optimization***; on line 184, the enumerator should be ***(i)***.

### Questions
1. Does the manuscript describe any validation mechanism for the generated subquestion–answer pairs to prevent intermediate errors from propagating and compounding through the pipeline?

2. Can the authors provide empirical results or error analyses examining the quality/diversity of the generated synthetic data? 

3. Is there any analysis of how the choice/size of the initial seed set $\mathcal{Q}^0$ affects performance or coverage? For instance, how does model generalization degrade when the seed is minimal or unrepresentative?

### Soundness
2

### Presentation
3

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
This paper introduces COGS, a data-efficient framework for enhancing visual reasoning capabilities in multimodal large language models  for artificial image domains like charts and webpages. The key innovation is decomposing seed questions into primitive perception and reasoning "factors," then systematically recombining these factors with new unlabeled images to generate synthetic training data. Each generated question includes subquestions and intermediate answers, enabling reinforcement learning with process-level rewards. The authors evaluate COGS primarily on chart reasoning (ChartQAPro, MMC-Bench) and webpage understanding (VisualWebBench), demonstrating substantial improvements over baseline models, with particularly strong gains on reasoning-heavy and compositional questions.

### Strengths
- Novel compositional approach: The factor decomposition and recomposition strategy is intuitive and well-motivated, providing a principled way to scale up training data from limited seed examples while maintaining diversity and complexity.

- Strong empirical results: COGS achieves meaningful improvements across multiple benchmarks (52.02% on ChartQAPro vs. 47.36% base model; 88.04% on VisualWebBench vs. 85.65% base), outperforming both general-purpose models and domain-specific baselines.

- Transferability across datasets: The factor-level mixture experiments (Section 4.1.2) demonstrate that COGS induces generalizable reasoning capabilities rather than dataset-specific overfitting, with factor-level mixing outperforming data-level mixing.

- Theoretical contribution: Proposition 3.1 provides valuable theoretical insight into why ProcessRM-max preserves policy ordering while ProcessRM-sum does not, backed by empirical validation.

- Comprehensive evaluation: The paper includes thorough ablations (reward models, question complexity, factor types) and extends beyond charts to webpages, demonstrating generalizability.

### Weaknesses
- The paper focuses exclusively on artificial image domains (charts, webpages). It's unclear whether this approach would transfer to natural images or other multimodal reasoning tasks. The restriction to domains with "abundant unlabeled images" may limit applicability.

- Dependency on high-quality decomposition: The entire framework relies on an MLLM's ability to accurately decompose questions into factors. The paper doesn't thoroughly analyze decomposition quality or failure modes. What happens when decomposition is incorrect or incomplete?

- Limited baseline comparisons: While the paper compares against several data synthesis approaches, it doesn't compare against other compositional reasoning methods or more sophisticated prompting techniques (e.g., self-consistency, tree-of-thoughts).

### Questions
Can factors learned from one domain (e.g., charts) transfer to another domain (e.g., webpages) without redecomposition? Would a shared factor pool improve multi-domain performance?

### Soundness
3

### Presentation
3

### Contribution
4
