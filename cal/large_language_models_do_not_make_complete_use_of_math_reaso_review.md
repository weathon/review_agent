=== CALIBRATION EXAMPLE 39 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title is clear and makes a strong claim. The abstract succinctly summarizes the core finding: that increasing training data for math reasoning leads to models "forgetting" previously correct answers, a phenomenon not resolved by test-time scaling. It directly addresses a relevant question in the era of data scarcity. The claims are specific and appear to be supported by the experiments listed.

**Introduction & Motivation**
The introduction effectively frames the problem: while scaling data is a primary lever for improvement, it's unclear if models utilize all data completely. The motivation, tied to data scarcity and the cost of procuring/generating data, is compelling for the ICLR audience. The central question is clearly stated. The introduction also correctly previews the main empirical observations (newly incorrect answers, test-time scaling failure) and the proposed explanation (predictive multiplicity).

**Method / Approach**
The experimental methodology is sound and reproducible, with detailed hyperparameters in Section 3.2. The core analysis—training on nested subsets and tracking per-sample correctness across seeds—is appropriate to answer the posed question. The extension to RL (ZeroRL) and test-time scaling (majority voting) strengthens the claim's generality.

**Key concerns regarding the method/approach:**
1.  **Theoretical Framework (Section 4.2) is Underdeveloped:** The connection to Rashomon sets and the derived expressions for the size of the permissible model set are a positive step. However, the analysis makes a strong independence assumption (strategy per sample is independent) that is almost certainly violated in practice; the learning of one strategy likely influences others. The derived bounds, while illustrative, are not used to make quantitative predictions or to guide the design of subsequent experiments. This section feels more like a post-hoc justification than a guiding theoretical insight. A more rigorous discussion of the Rashomon effect in overparameterized models fine-tuned on compositional tasks would strengthen the paper.
2.  **Definition and Measurement of "Strategy":** The operational definition of a "strategy" (sequence of mathematical operations) is reasonable but coarse. The paper claims an average of 5.32 unique strategies per test sample, with 3.15 being incorrect. It is unclear how robust this counting is to superficial variations in formatting or wording within the reasoning trace that do not change the underlying logic. A brief analysis or justification of this method's stability would be helpful.
3.  **Scope of Ablations:** The ablations in Fig. 7 (fixing sample order, removing LoRA dropout) convincingly show that these sources of randomness contribute to learning different functions. However, this list is likely incomplete (e.g., optimizer state initialization, batch composition). The paper would benefit from a more systematic discussion of the sources of variance that lead to the Rashomon effect in this setting.

**Experiments & Results**
The empirical results are the core strength of the paper. The phenomenon of "newly incorrectly answered" samples is demonstrated across multiple models (Llama3, Gemma3, Qwen2.5), tasks (GSM8K, MAWPS, MATH), and training paradigms (SFT, RL). The fixed-set analysis (Fig. 6) is particularly compelling, showing low overlap in correctly answered samples across seeds trained on the *same* data. The appendix results on model capacity and full fine-tuning (Figs. 8, 9) help rule out simple explanations.

**Key concerns regarding experiments & results:**
1.  **Statistical Significance and Seeds:** For the primary SFT experiments (Llama3-8B, Gemma3-4B), the paper states results are over 3 seeds, which is minimal. While the trends are clear, reporting confidence intervals or standard deviations for key metrics (e.g., the size of the "newly incorrect" set at each step) would bolster confidence. The RL experiments are reported for only 1 seed, which is a notable weakness.
2.  **Lack of a "Best-Possible" Baseline:** The "Union" metric in Fig. 3 is insightful, showing a significant gap between the final model and the ensemble of all subset-trained models. However, a stronger baseline would be an *ensemble* of models trained from different seeds on the *full* dataset. This would directly test whether the predictive multiplicity on the full dataset could be harnessed to recover the "Union" performance, which is a natural practical question arising from this work.
3.  **Limited Discussion of Why Math Reasoning?** The paper convincingly shows the effect exists in math reasoning. A brief discussion or experiment hinting at *why* this domain is particularly susceptible would be valuable. Is it due to the multi-step, compositional nature? The existence of many solution strategies? A comparison to a simpler task (e.g., sentiment classification or QA on factual knowledge) where one might expect less multiplicity would help contextualize the contribution.

**Writing & Clarity**
The paper is generally well-written and logically structured. However, due to the PDF parsing issues, many figure references are broken (e.g., "Fig. 1 (Left)" appears with garbled table text, references to Figs. 4 & 5 are confusing). The core narrative remains understandable, but the reader must make significant effort to piece together which visual result is being discussed. The authors must ensure the final submission has correctly referenced and legible figures. The theoretical section (4.2) could be clearer in its assumptions and implications.

**Limitations & Broader Impact**
The paper explicitly acknowledges its scope (math reasoning, SFT/RL) which is appropriate. A more thorough limitations section could discuss: the limited number of random seeds; the focus on correctness rather than reasoning quality (could a "forgotten" sample be answered with a different but valid reasoning path?); and the potential that this is a feature of PEFT/LoRA, though Appendix A.2 partially addresses this. Broader impact is briefly alluded to (aiding in improving data scaling) but could be expanded: these findings suggest that simply collecting more data may have diminishing and unpredictable returns for fine-tuning on reasoning tasks, and that efforts might be better spent on improving training stability or leveraging multiplicity via ensembling.

### Overall Assessment
This paper identifies a novel, important, and counter-intuitive phenomenon in LLM fine-tuning for reasoning: adding more data causes regression on a significant subset of previously solved problems, which is linked to high predictive multiplicity (Rashomon effect). The empirical evidence across models and tasks is robust and compelling, constituting a solid contribution suitable for ICLR. The main weaknesses are the underdeveloped theoretical analysis (which currently provides intuition more than falsifiable predictions) and some methodological limitations (low seed count for RL, missing ensemble baseline). Addressing these, particularly by strengthening the theoretical framing or adding the suggested ensemble experiment, would significantly elevate the paper. The core empirical finding is sufficiently strong and relevant to the community's focus on data-efficient tuning to warrant acceptance, provided the major clarity issues with figures are resolved.

# Neutral Reviewer
## Balanced Review

### Summary
This paper investigates whether large language models (LLMs) fully utilize training data for mathematical reasoning tasks. Through experiments with supervised fine-tuning and reinforcement learning on multiple models and datasets, the authors observe that adding more training data often causes models to incorrectly answer a significant portion of test samples they previously answered correctly. They attribute this phenomenon to high *predictive multiplicity* (the Rashomon effect), where models trained on the same data with different random seeds learn very different functions, leading to low overlap in the sets of correctly answered test samples.

### Strengths
1. **Relevant and Timely Research Question**: The work directly addresses a critical issue in the era of data scarcity for LLMs, questioning a fundamental assumption (more data always helps) and providing a novel perspective on scaling laws. This aligns with ICLR's interest in foundational understanding of model behavior.
2. **Extensive and Rigorous Empirical Analysis**: The paper provides comprehensive experiments across multiple model families (Llama3, Gemma3, Qwen2.5), datasets (GSM8K, MAWPS, MATH), and training paradigms (Supervised Fine-Tuning, Reinforcement Learning/ZeroRL). The use of parameter-efficient fine-tuning (LoRA, LoftQ) and full fine-tuning, as well as ablations on sample order and dropout, strengthens the evidence.
3. **Connection to Established Theory**: The authors effectively link their empirical observations to the concept of predictive multiplicity and Rashomon sets, providing a theoretical framework to explain why many equally good models exist for math reasoning tasks due to multiple viable (and non-viable) reasoning strategies.
4. **Controlled Analysis of Test-time Scaling**: The demonstration that the issue persists even with majority voting (a common test-time scaling technique) rules out simple inference-time non-determinism as the cause and strengthens the core claim about a training-level phenomenon.

### Weaknesses
1. **Limited Statistical Analysis and Quantitative Rigor**: While the trends are clear, the paper lacks statistical significance tests for key claims (e.g., the "10-15%" of samples affected). Reporting confidence intervals or p-values for the differences in "newly incorrect" samples across seeds or steps would strengthen the evidence.
2. **Under-explored Scope and Generalizability**: The work is focused exclusively on mathematical reasoning tasks. While this is a justified and important domain, the title and abstract present a broader implication ("LLMs do not make complete use of data"). The paper would be stronger if it included a preliminary discussion or experiment on whether this phenomenon occurs in other reasoning or non-reasoning tasks (e.g., code generation, factual QA) to better scope its claims.
3. **Theoretical Framework Could Be Deepened**: The theoretical discussion in Section 4.2 (Settings 1 & 2) is a good start but feels somewhat detached from the empirical results. A more formal connection, perhaps showing how the empirically measured strategy counts (*|K|*, *|M|*) plug into the derived bounds to explain the observed discrepancy, would be more impactful.
4. **Practical Implications and Mitigations Are Underdeveloped**: The paper successfully diagnoses a problem but offers only minimal guidance on "improving a model’s ability to effectively scale its performance with more data" (as mentioned in the abstract). A discussion or preliminary experiments on potential mitigations (e.g., ensembling, different optimization strategies, or data curation informed by predictive multiplicity) would significantly increase the practical significance of the work.

### Novelty & Significance
**Novelty:** The core finding—that adding data can cause forgetting of previously learned correct solutions in math reasoning due to inherent predictive multiplicity—is novel and counter-intuitive. The systematic connection of this observation to the Rashomon effect in the context of modern LLM fine-tuning is a fresh and valuable contribution.

**Significance:** The work is highly significant for the ICLR community. It challenges a standard assumption in scaling and provides a crucial caveat for synthetic data generation and continual learning efforts. It also offers a new lens (predictive multiplicity) for analyzing and interpreting model performance on reasoning tasks, which could influence future evaluation and training methodologies.

**Clarity:** The paper is generally well-written and logically structured. However, the extracted text contains severe formatting artifacts (broken figures, misplaced tables, garbled captions) that hinder the assessment of visual communication. Based on the captions and context, the intended figures seem central to the argument. *[Reviewer Note: Per instruction, these formatting issues are considered parser artifacts and not held against the paper.]*

**Reproducibility:** The experimental details (models, datasets, hyperparameters, compute) are sufficiently detailed for reproducibility. The code and specific data splits used for incremental subsets are not provided but could be reasonably inferred.

### Suggestions for Improvement
1. **Strengthen Quantitative Evidence**: Add statistical tests to key results (e.g., bootstrapped confidence intervals for the proportion of "newly incorrect" samples). Report the exact intersection-over-union metrics for the correctly answered sets across seeds in the fixed-set analysis (Fig. 6) to quantify the multiplicity more precisely.
2. **Broaden the Discussion of Scope**: Include a section discussing the limits of the findings. Acknowledge that the study is confined to math reasoning and hypothesize whether similar effects might be observed in other domains with high solution multiplicity (e.g., code generation, creative writing) versus those with single answers (e.g., factual recall). This will better frame the contribution.
3. **Deepen the Theoretical Synthesis**: More tightly integrate the empirical measurements (average number of unique strategies per sample) with the derived combinatorial formulas for the Rashomon set size. A small simulation or calculation using the empirical numbers would make the theory section more concrete and compelling.
4. **Explore and Discuss Mitigations**: Propose and, if possible, conduct preliminary experiments on one or two simple methods to reduce harmful predictive multiplicity. For example, does ensembling models from different seeds recover the "Union" performance shown in Figure 3? Does modifying the training objective (e.g., with consistency regularization across augmented samples) help? Even a thoughtful discussion of such directions would greatly enhance the paper's utility.
5. **Improve Presentation of Key Results**: Ensure the final version has clearly labeled and legible figures. The central finding (Fig. 1, 2, 4) should be presented with unambiguous bar charts or line plots showing "Newly Correct" vs. "Newly Incorrect" across training steps. The current parsed text with placeholders like `|Col1|Col2|...` is unusable for evaluation.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Full fine-tuning (non-PEFT) for larger models**: The paper relies heavily on PEFT (LoRA/LoftQ) for SFT experiments. It must include full fine-tuning of a model like Llama3-8B to rule out that the observed effects are not an artifact of constrained parameter updates from PEFT, which could artificially increase predictive multiplicity.
2. **Systematic model scale ablation**: While the appendix includes Gemma3-1B/12B, a rigorous scale analysis across multiple model families (e.g., 0.5B, 2B, 7B, 13B) is needed to show whether the phenomenon persists or diminishes with increased capacity, which is central to claims about LLMs in general.
3. **Non-mathematical reasoning tasks**: The claim is about "math reasoning data," but to argue that the issue is specific to reasoning (or general), experiments on other reasoning benchmarks (e.g., logical deduction, commonsense QA) are essential. Without this, the scope of the claim is unsupported.
4. **Hyperparameter sensitivity analysis**: The effect may depend heavily on hyperparameters (learning rate, epochs, dropout). The paper only ablates sample order and LoRA dropout. Testing a range of hyperparameters is necessary to confirm the robustness of the phenomenon and rule out that it's a byproduct of suboptimal training configurations.

### Deeper Analysis Needed (top 3-5 only)
1. **Characterization of flipped samples**: The paper counts flips but does not analyze *why* specific test samples flip. An analysis of sample difficulty, similarity to newly added training data, or the nature of reasoning steps is critical to distinguish between catastrophic interference and random flips, which would inform the underlying mechanism.
2. **Quantification of predictive multiplicity using established metrics**: The paper visually shows intersection sizes but does not compute standard predictive multiplicity metrics (e.g., discrepancy, Rashomon set size estimates). Without these, the claim of "high predictive multiplicity" is not quantitatively substantiated and cannot be compared across settings.
3. **Analysis of strategy independence assumption**: The theoretical framework assumes per-sample strategies are independent, which is unrealistic for sequential reasoning. The analysis must discuss how dependencies would affect the Rashomon set size and whether the theoretical explosion of models still holds under more realistic assumptions.

### Visualizations & Case Studies
1. **Concrete examples of flipped test samples**: Showing specific questions, the model's reasoning traces before and after adding data, and the incorrect strategy adopted would make the phenomenon tangible and help readers assess whether flips are due to meaningful strategy changes or superficial errors.
2. **Visualization of diverse strategies across seeds**: For a few test problems, illustrate the different operation sequences (strategies) generated by models trained with different seeds. This would directly support the claim that models learn different functions and clarify what "different strategies" means in practice.

### Obvious Next Steps
1. **Investigate mitigation techniques**: The paper identifies a problem but does not explore solutions. Testing simple mitigations like ensembling, data curation, or regularization (e.g., weight averaging) is a logical next step to show whether the incomplete data use can be alleviated, which would strengthen the paper's impact.
2. **Extend test-time scaling beyond majority voting**: The paper only uses majority voting to address non-determinism. To rule out that more advanced test-time compute (e.g., verifiers, reward-based ranking) could recover lost information, experiments with these methods are necessary to fully substantiate the claim that test-time scaling doesn't resolve the issue.

# Final Consolidated Review
## Summary
This paper investigates whether large language models fully utilize their training data for mathematical reasoning tasks. Through experiments with supervised fine-tuning and reinforcement learning across multiple models and datasets, the authors demonstrate that adding more training data causes models to incorrectly answer a significant portion (10-15%) of test samples they previously answered correctly. They link this phenomenon to high predictive multiplicity (the Rashomon effect), showing that models trained on the same data with different seeds learn very different functions, leading to low overlap in correctly answered test sets.

## Strengths
- **Multi-faceted empirical demonstration:** The core phenomenon is robustly shown across multiple model families (Llama3, Gemma3, Qwen2.5), datasets (GSM8K, MAWPS, MATH), and training paradigms (Supervised Fine-Tuning and Reinforcement Learning/ZeroRL). This thorough experimentation rules out simple explanations tied to a single setup.
- **Novel connection to predictive multiplicity:** The paper effectively links an observed training instability (forgetting with more data) to the established concept of the Rashomon effect, providing a theoretical lens to explain why many equally good models exist for math reasoning due to multiple viable and non-viable reasoning strategies.
- **Ruling out test-time artifacts:** The authors show the issue persists even with majority voting, a common test-time scaling technique, demonstrating it is a fundamental training phenomenon and not merely a consequence of inference-time non-determinism.

## Weaknesses
- **Limited investigation of practical consequences and mitigations:** The paper successfully diagnoses a problem but offers minimal guidance on how to improve data scaling. A natural and impactful experiment—ensembling models from different seeds trained on the *full* dataset to see if it recovers the "Union" performance shown in Figure 3—is missing. This leaves the practical significance underdeveloped.
- **Insufficient statistical reporting and seed counts:** While trends are clear, quantitative rigor is lacking. Key results (e.g., the proportion of "newly incorrect" samples) lack measures of variance. More importantly, the RL experiments are reported for only a single seed, and the primary SFT experiments use only three seeds, which is minimal for robustly claiming "high predictive multiplicity" across the training process.
- **Under-discussed scope:** The title and claims are about "math reasoning data," but the paper does not contextualize whether this phenomenon is particularly acute for math reasoning due to its compositional, multi-strategy nature, or if it might generalize to other domains. A discussion of the task characteristics that might drive predictive multiplicity would better frame the contribution's boundaries.

## Nice-to-Haves
- A more formal quantitative link between the empirically measured number of unique strategies per sample and the theoretical bounds on the Rashomon set size.
- Analysis of whether "forgotten" samples are answered with qualitatively different (but still incorrect) reasoning strategies versus superficial errors.
- Exploration of whether hyperparameters other than sample order and dropout (e.g., learning rate, number of epochs) significantly influence the observed predictive multiplicity.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Weakness:** "Theoretical framework is underdeveloped due to its independence assumption." → The paper explicitly states this assumption is for simplicity and uses it to provide an intuitive combinatorial explanation, not a precise quantitative model. It is a reasonable simplifying step for the paper's purpose.
- **Weakness:** "Strategy measurement is not robust to superficial variations." → The paper provides a clear, operational definition (sequence of mathematical operations). Demanding a validation study for this definition is a methodological nitpick not required for the core claim.
- **Weakness:** "Must include experiments on non-mathematical reasoning tasks." → The paper explicitly scopes its contribution to "math reasoning tasks" due to data scarcity concerns in this domain. Demanding experiments in other domains is scope creep.
- **Weakness:** "Must include full fine-tuning for larger models." → The paper already addresses this in Appendix A.2, showing the effect persists without PEFT for Llama3.2-3B.
- **Weakness:** "Must perform systematic hyperparameter sensitivity analysis." → This is a generic request for more ablation that is not standard practice for establishing a core phenomenon. The provided ablations (sample order, dropout) are sufficient to show sources of variance.

## Novel Insights
The paper's key novel insight is identifying and rigorously documenting a counter-intuitive failure mode in LLM fine-tuning for reasoning: adding more data systematically causes regression on a subset of previously solved problems, which is not an artifact of unstable inference. It then insightfully reframes this not as simple catastrophic interference but as a manifestation of high predictive multiplicity (the Rashomon effect) inherent to the task, where the training data supports many different, equally accurate predictive functions. This connects a practical observation about scaling to a fundamental theoretical concept in machine learning.

## Suggestions
- Conduct and report an ensemble experiment: train multiple seeds on the *full* dataset and compare the ensemble accuracy to the "Union" accuracy from the incremental training analysis. This directly tests a key practical implication of your multiplicity finding.
- Increase the number of random seeds for the RL experiments to at least three and report variance metrics (e.g., standard deviation) for the size of the "newly incorrect" set across seeds in the primary SFT experiments.
- Add a dedicated subsection discussing the scope of your findings, explicitly hypothesizing why math reasoning (with its multi-step, multi-strategy nature) might be particularly susceptible to this effect compared to tasks with fewer valid solution paths.

# Actual Human Scores
Individual reviewer scores: [2.0, 6.0, 6.0, 2.0]
Average score: 4.0
Binary outcome: Reject
