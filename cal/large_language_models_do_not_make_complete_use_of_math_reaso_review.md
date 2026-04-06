=== CALIBRATION EXAMPLE 36 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title is clear and directly reflects the core claim. The abstract succinctly summarizes the key finding—that adding more math reasoning data during fine-tuning causes a significant portion of previously correct test samples to become incorrect—and links it to the broader phenomenon of predictive multiplicity. The claims are specific and set appropriate expectations for the paper.

**Introduction & Motivation:** The introduction effectively motivates the problem by situating it within the context of data scarcity for LLMs and the scaling paradigm. The central research question ("Do deep neural networks make complete use of the data they are provided?") is well-posed. The contributions are clear: identifying the "newly incorrectly answered" phenomenon, showing its persistence despite test-time scaling, and linking it to predictive multiplicity/Rashomon sets via empirical and theoretical analysis.

**Method / Approach (Sections 3 & 3.2):** The experimental methodology is generally sound and described with sufficient detail for reproducibility (models, datasets, PEFT methods, hyperparameters). The core protocol—training on nested supersets of data and tracking per-sample correctness across steps—is appropriate for the research question.
*   **Major Concern (Statistical Rigor):** A critical weakness is the limited number of random seeds. Most SFT experiments use only 3 seeds, and the RL experiments use only 1 seed. For claims about variability across seeds (predictive multiplicity), this is inadequate. The results, especially the intersection calculations in Fig. 5 and 6, could be highly sensitive to seed choice with such a small *n*. This undermines the robustness of the central empirical finding.
*   **Theoretical Analysis (Section 4.2):** The link to Rashomon sets is a valuable conceptual contribution. However, the theoretical framework (Settings 1 & 2) feels underdeveloped. The derivations for the number of permissible models rely on strong, simplified assumptions (e.g., independent per-sample strategy choices, the specific structure of mistakes in Setting 1). More importantly, the connection between the derived combinatorial formulas and the actual *training dynamics* of LLMs is not established. The theory describes a *possibility* (a large Rashomon set) but does not explain *why* SGD on these tasks consistently finds such diverse members of the set. The strategy counting ("average number of unique strategies per test sample was 5.32") is a good start, but the methodology for extracting "strategies" from reasoning traces is not detailed and may be overly simplistic (e.g., is an operation sequence a robust proxy for a reasoning strategy?).

**Experiments & Results (Sections 3, 3.1, 4, 4.1):**
*   **Core Phenomenon:** The primary result—the significant fraction of "newly incorrectly answered" samples—is demonstrated across multiple models (Llama3, Gemma3, Qwen2.5), datasets (GSM8K, MAWPS, MATH), and training paradigms (SFT, RL). Showing that this persists even with majority voting (Fig. 4) effectively counters the potential objection that it's just test-time sampling noise.
*   **Fixed Set Analysis (Fig. 6):** This is strong evidence supporting the predictive multiplicity claim. Showing that models trained on the *same fixed dataset* with different seeds achieve similar aggregate accuracy but correctly answer different subsets of the test set is compelling.
*   **Missing Analyses & Baselines:**
    1.  **What about the data?** The paper focuses on model behavior but does not analyze the *training data* itself. Is the effect correlated with certain problem types, difficulty, or the presence of conflicting reasoning patterns in the added data? A qualitative analysis of samples that flip would greatly strengthen the work.
    2.  **Comparison to Standard Fine-Tuning:** All SFT experiments use PEFT (LoRA/LoftQ). While Appendix A.2 shows the effect persists in one full fine-tuning experiment, this needs to be more central. A key question is whether the observed multiplicity is exacerbated by the low-rank adaptation constraint.
    3.  **Beyond Math Reasoning:** The title and claims are about "math reasoning data." Is this phenomenon specific to reasoning tasks, or would it occur in standard language modeling or classification? A discussion or a small experiment on a non-reasoning task would help scope the contribution.
*   **Presentation of Results:** The figures are central, but due to the parsing artifacts, their exact values and labels are often unreadable in the provided text. The review must assume the underlying data is sound, but this makes a precise evaluation of the magnitudes of effects (e.g., the exact percentages in Fig. 1) impossible from the provided content.

**Writing & Clarity:** The paper is generally well-written and logically structured. The narrative from the empirical observation to the fixed-set analysis to the Rashomon set explanation is clear. However, Section 4.2 is dense and could benefit from a more intuitive, narrative explanation alongside the formal definitions.

**Limitations & Broader Impact:** The Limitations section is implicit but insufficient. Key limitations that should be explicitly discussed include: (1) the small number of random seeds, as noted; (2) the exclusive focus on math reasoning tasks; (3) the use of PEFT in most experiments; (4) the simplified theoretical model. The broader impact section is absent. While the work is primarily diagnostic, a discussion on the implications for data curation, training stability, and ensemble methods would be appropriate. The societal impact is likely neutral/positive (improving efficient data use).

### Overall Assessment

This paper identifies a fascinating and counter-intuitive failure mode in LLM fine-tuning for reasoning: performance plateaus not just because of diminishing returns, but because new data actively disrupts existing correct capabilities. Linking this to predictive multiplicity/Rashomon sets is a conceptually important step. However, for ICLR, the work currently has significant weaknesses. The empirical foundation is shaky due to an insufficient number of random seeds, which is fatal for a paper whose central claim hinges on variability across seeds. The theoretical analysis, while a good start, is more of a suggestive formal analogy than a mechanistic explanation. The contribution is novel and potentially important, but in its current form, it feels preliminary. Major revisions addressing statistical rigor, deepening the theoretical connection, and adding analysis of the training data are needed for the contribution to stand firmly at the ICLR acceptance bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates whether large language models (LLMs) fully utilize training data for mathematical reasoning tasks. Through experiments in supervised fine-tuning (SFT) and reinforcement learning (RL), the authors demonstrate that adding more training data can cause models to incorrectly answer a significant portion (10-15%) of test samples they previously answered correctly. They link this phenomenon to high predictive multiplicity (the Rashomon effect), where models trained on identical data with different random seeds learn vastly different functions, each correctly solving only a small, non-overlapping subset of the test set. The core claim is that LLMs do not make "complete use" of their math reasoning data.

### Strengths
1.  **Important and Counter-Intuitive Empirical Finding:** The central observation—that adding data harms performance on a subset of previously mastered examples—is well-documented across multiple models (Llama3, Gemma3, Qwen2.5), datasets (GSM8K, MAWPS, MATH), and training paradigms (SFT, RL). This challenges the standard scaling law narrative and is highly relevant to the community, especially given concerns about data scarcity.
2.  **Rigorous and Extensive Experimental Design:** The study is thorough. It uses consistent subset scaling (supersets), multiple random seeds, and investigates the failure of common mitigation strategies like majority voting. The fixed-set analysis (training on the same data across seeds) directly supports the predictive multiplicity hypothesis. Ablations on sample order and LoRA dropout further strengthen the argument.
3.  **Clear Practical Implications:** The work successfully highlights a novel failure mode in contemporary LLM training for reasoning. The "Union vs. Final Model" comparison (Figure 3) effectively visualizes the potential performance ceiling that is missed due to this incomplete data usage, framing a clear problem for future research.

### Weaknesses
1.  **Underdeveloped Theoretical Connection:** While the link to the Rashomon effect and the derivation of permissible model counts are good starting points, the theoretical analysis remains somewhat superficial. The "strategy set" (Definition 3) is loosely defined (extracting operation sequences), and the analysis relies on strong independence assumptions between samples. A deeper theoretical exploration of *why* the hypothesis space for math reasoning is so large and leads to such high multiplicity is needed.
2.  **Lack of Causal Analysis and Proposed Solutions:** The paper is excellent at diagnosing the problem but offers little direction for solutions. The conclusion states the finding but does not suggest concrete architectural, optimization, or data curation strategies to mitigate the issue. For ICLR, a discussion of promising research directions would significantly strengthen the impact.
3.  **Clarity and Presentation Issues:** The manuscript suffers from significant formatting problems (e.g., broken tables, misplaced figure captions, garbled text in Sections 3 and 4). While some are noted as parser artifacts, they severely hinder readability. The description of experiments, especially the RL setup and the exact "Step" definitions in figures, could be clearer. Some figures (e.g., Figure 5) are referenced before being fully explained.

### Novelty & Significance
**Novelty:** The paper identifies and systematically characterizes a previously underexplored phenomenon in LLM fine-tuning for reasoning. The connection between diminishing marginal returns from data, "catastrophic forgetting" of specific test samples, and predictive multiplicity in this context is novel.
**Significance:** The findings are highly significant for the field. They question the efficiency of current data scaling approaches for reasoning and reveal a fundamental limitation in how LLMs consolidate knowledge from diverse examples. This has direct implications for how synthetic data generation and curriculum learning might be designed. The work meets ICLR's bar for presenting a clear, well-supported empirical insight that challenges common assumptions and opens new research avenues.

### Suggestions for Improvement
1.  **Deepen the Theoretical Analysis:** Move beyond defining the Rashomon set and counting strategies. Analyze the loss landscape or gradient dynamics for reasoning tasks. Could the multiplicity be linked to the existence of many near-optimal but distinct reasoning "paths" (Chain-of-Thoughts)? A more formal analysis of why SFT/RL on math data leads to a wider Rashomon set than, say, language modeling would be valuable.
2.  **Explore and Discuss Mitigations:** The paper should dedicate a section to discussing potential solutions. For example: Does ensembling models from different seeds recover the "Union" performance? Could consistency-based regularization (encouraging similar outputs for semantically equivalent questions) help? Would a different optimizer or learning rate schedule reduce variance? Even speculative discussions would guide future work.
3.  **Major Clarity Revisions:** The authors must thoroughly clean the manuscript to address parser artifacts. All figures and tables need clear, legible captions and axis labels. The experimental details section (3.2) should be expanded to unambiguously define evaluation steps, subset sizes, and how "newly incorrect" samples are counted. Consider moving some details from the appendix (e.g., full fine-tuning results) into the main text for a more complete narrative.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Test larger-scale models (e.g., 70B+ parameters).** The paper's central claim is about "large language models," but experiments are limited to models up to 8B parameters. For ICLR, it's critical to show the phenomenon persists at the scale where data scarcity is most acute.
2. **Compare against the standard SFT → RLHF pipeline.** The paper studies SFT and ZeroRL separately. A key missing baseline is sequential SFT-then-RL, which is the dominant paradigm for aligning reasoning models. Without this, the claim about RL's data inefficiency is incomplete.
3. **Ablate different PEFT methods and full fine-tuning more thoroughly.** The appendix shows one full fine-tuning experiment, but the core analysis uses LoRA/LoftQ. A systematic ablation is needed to rule out that the observed effects are artifacts of parameter-efficient tuning and not inherent to model learning.
4. **Include more diverse and complex reasoning datasets (e.g., TheoremQA, Olympiad-level MATH).** The claim is about "math reasoning data." Testing only on GSM8K, MAWPS, and a MATH subset is narrow. Performance on highly complex, symbolic reasoning tasks is necessary to generalize the finding.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze *why* specific test samples flip from correct to incorrect.** The paper shows the "what" but not the "why." A qualitative analysis of the reasoning traces for flipped samples is essential to distinguish between catastrophic interference, strategy substitution, or learning of incorrect heuristics.
2. **Quantify the relationship between Rashomon set size and dataset properties.** The theoretical argument hinges on the number of unique strategies `|K|`. This should be empirically measured and correlated with dataset characteristics (e.g., problem diversity, solution length) across the three datasets used.
3. **Investigate the role of data quality and curriculum.** The performance drop when adding data could be due to noisy or harder later samples. An analysis of sample difficulty/quality order and its effect on forgetting is needed to rule out trivial explanations.
4. **Measure predictive multiplicity directly with metrics like Rashomon Ratio or discrepancy.** The paper informally observes high multiplicity. Formal metrics should be computed and reported to quantify the effect's strength across models and datasets.

### Visualizations & Case Studies
1. **Side-by-side examples of reasoning traces for the same problem solved correctly by a smaller-data model and incorrectly by a larger-data model.** This visualization is the most direct way to validate the core claim and show how the model's reasoning strategy degrades or changes.
2. **A visualization of the "strategy space" for a few representative problems.** The paper counts unique strategies but doesn't show them. Illustrating the diverse correct/incorrect paths a model can take for a single problem would make the Rashomon set concept concrete.
3. **Error case studies grouped by mathematical operation or problem type.** This would reveal if forgetting is systematic (e.g., models lose proficiency in division problems) or random, impacting the interpretation of "incomplete use."

### Obvious Next Steps
1. **Propose and evaluate a simple mitigation.** For ICLR, a paper identifying a problem should at least sketch and test a preliminary solution (e.g., replay buffers, elastic weight consolidation). Without this, the work is purely diagnostic and lacks constructive value.
2. **Connect findings directly to synthetic data generation.** The introduction mentions synthetic data as a solution to scarcity. The paper must discuss how its findings (models don't fully use data) should inform the generation or curation of synthetic math data.
3. **Discuss implications for ensemble methods.** If different seeds yield diverse correct answers, an ensemble of cheaply trained models might outperform a single heavily trained one. This is a direct, practical implication that should be explored and reported.

# Final Consolidated Review
## Summary
This paper investigates whether large language models (LLMs) fully utilize their training data for mathematical reasoning tasks. Through fine-tuning experiments on multiple models and datasets using supervised fine-tuning (SFT) and reinforcement learning (RL), the authors demonstrate that adding more training data causes a model to incorrectly answer a significant portion (10-15%) of test samples it previously answered correctly. They link this phenomenon to high predictive multiplicity (the Rashomon effect), showing that models trained on identical data with different random seeds learn very different functions, each mastering only a small, non-overlapping subset of the test set.

## Strengths
- **Compelling and Counter-Intuitive Empirical Discovery:** The core finding—that performance plateaus arise not just from diminishing returns but from new data actively degrading performance on previously solved problems—is robustly demonstrated across multiple models (Llama3, Gemma3, Qwen2.5), datasets (GSM8K, MAWPS, MATH), and training paradigms (SFT and RL). This is a significant, well-supported observation that challenges standard scaling narratives.
- **Methodologically Rigorous Analysis:** The experimental design is thorough. The use of nested supersets of data, the fixed-set analysis (showing different seeds on the same data yield different correct-answer sets), and the demonstration that the effect persists even with test-time scaling (majority voting) provide strong, multi-faceted evidence for the claim. The ablation studies (sample order, LoRA dropout) further strengthen the argument that training randomness leads to divergent functions.

## Weaknesses
- **Limited Theoretical Explanation:** While the connection to Rashomon sets is a valuable conceptual link, the theoretical analysis (Section 4.2) remains more of a suggestive formal analogy than a mechanistic explanation. The derivations rely on strong simplifying assumptions (e.g., independent per-sample strategy choices), and the analysis does not explain *why* standard LLM training dynamics consistently find such diverse members of the Rashomon set for reasoning tasks.
- **Insufficient Statistical Support for RL Findings:** The RL experiments (ZeroRL) are conducted with only a single random seed. For claims about model behavior and variability, this is inadequate. While the SFT results with three seeds are more convincing, the RL findings lack statistical robustness and should be interpreted with caution.

## Nice-to-Haves
- A qualitative analysis of the reasoning traces for samples that flip from correct to incorrect would provide deeper insight into the nature of the "forgetting" (e.g., is it a change in strategy, a collapse to a heuristic, or interference?).
- A discussion of potential mitigation strategies or research directions (e.g., the utility of ensembling, implications for synthetic data curation) would enhance the paper's impact, though proposing a solution is not required for this diagnostic contribution.
- Testing on a broader range of model capacities (beyond 8B parameters) would help generalize the claim, though the phenomenon is already established across a meaningful set of models.

## Novel Insights
The paper provides a novel and important synthesis: the observed plateau in performance when scaling math reasoning data is not merely due to saturation but is actively driven by a loss of previously acquired capabilities. This is convincingly linked to the inherent predictive multiplicity (Rashomon effect) in the hypothesis space for multi-step reasoning tasks, where many distinct near-optimal functions exist. The finding that standard training methods only recover a small, seed-dependent subset of these functions reveals a fundamental inefficiency in how current LLMs consolidate knowledge from data.

## Suggestions
- Conduct the RL experiments with multiple random seeds (at least 3) to substantiate the claims about variability and predictive multiplicity in that setting.
- Clarify and potentially expand the methodology for extracting and counting "unique strategies" from reasoning traces (Definition 3), as this is a key component of the theoretical argument.
- Explicitly discuss the limitations of the theoretical framework (e.g., independence assumptions) and its role as a conceptual model rather than a precise predictor of training dynamics.

# Actual Human Scores
Individual reviewer scores: [2.0, 6.0, 6.0, 2.0]
Average score: 4.0
Binary outcome: Reject
