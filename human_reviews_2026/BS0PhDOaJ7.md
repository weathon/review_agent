# CUARewardBench: Benchmark for Evaluating Reward Models on Computer-using Agent Trajectories

- Avg Score: 4.50
- Decision: Reject
- Scores: 8, 2, 4, 4

## Abstract
Computer-using agents (CUAs) enable task completion through natural interaction with operating systems and software interfaces. While script-based verifiers are widely adopted for evaluation, they suffer from limited scalability and inability to provide step-wise assessment. Reward models offer promising alternatives, but their effectiveness on CUA evaluation remains largely underexplored. To address this gap, we present CUARewardBench, comprising four key contributions: (1) First-ever Comprehensive CUA Reward Benchmark: We introduce the first benchmark for evaluating both outcome reward models (ORM) and process reward models (PRM) on CUA tasks, enabling systematic assessment across trajectory-level and step-level evaluation. (2) Diverse, Practical and Reliable Dataset: CUARewardBench encompasses trajectories from 10 software categories and 7 agent architectures with varying performance levels (25.9%-50.8% success rates). All trajectories are expertly annotated through carefully designed protocols, with rigorous quality control to ensure reliability and practical applicability. (3) Comprehensive Analysis and Insights: Through extensive experiments across 7 vision-language models and 3 prompt templates, we reveal critical limitations of current CUA RMs, including insufficient visual reasoning capabilities, knowledge deficiencies, and the superiority of general VLMs over specialized CUA models for reward evaluation. (4) Unanimous Prompt Ensemble (UPE): Based on the insights from our comprehensive analysis, we propose UPE, a novel ensemble method that significantly enhances reward model reliability through strict unanimous voting and strategic prompt-template configurations. UPE achieves 88.0% precision and 95.3% NPV for ORM, and 83.1% precision and 86.2% NPV for PRM, substantially outperforming single VLMs and traditional ensemble approaches. In a short, this work introduces both a comprehensive benchmark and a novel ensemble method that substantially enhances CUA reward model reliability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the first benchmark designed to evaluate **reward models** for computer-use agents.  It systematically annotates, categorizes, and analyzes the common failure cases of existing reward models when applied to complex, multi-step computer-interaction tasks.  The benchmark provides a much-needed standardized evaluation setup for this emerging subfield of embodied and tool-using language models, offering metrics, data splits, and error analyses that can serve as a foundation for future work.  

Section 3.4, which studies the effects of prompt variations, and Section 4, which presents a detailed error analysis, are particularly strong and insightful contributions that deepen understanding of model behavior.

In sum, my judgement is that this is a good and needed paper and likely to become a reference point in the evaluation of reward models for computer use agents.  The empirical analysis is thoughtful, the benchmark is well-motivated, and the contributions are meaningful.  Addressing the statistical and presentation issues can further solidify its impact.

### Strengths
Overall, this work is very good.  I like it.

1. **Originality and significance.**  
   This is the first known benchmark focusing specifically on reward models for computer-use agents.  The contribution is timely, as the community increasingly explores agents capable of manipulating real user interfaces or software environments.

2. **Quality and depth of analysis.**  
   The benchmark is carefully annotated and exhibits thorough evaluation of multiple open models.  The inclusion of rich failure categorizations and detailed qualitative analyses demonstrates deep engagement with the data rather than superficial benchmarking.  
   Section 4’s analysis of error modes helps readers understand why models fail.  **I am quite surprised that there's no error coming from hallucination**, by the way.  Is this not a problem at all here?

3. **Clarity and interpretability.**  
   The paper is overall well-organized and easy to follow.  
   The visualizations help illustrate the impacts of prompt phrasing and context-window limitations.

4. **Practical utility.**  
   The benchmark will likely become a valuable diagnostic tool for researchers developing reward models for computer use agentic LMs.  
   Its release could catalyze a line of work around better reward alignment for interactive agents.
   Section 3.4, which explores how prompt design affects reward-model outputs, is particularly instructive to me.

### Weaknesses
This is a good paper and there are not many weaknesses.  All weaknesses below are easy to solve.

1. **Lack of closed-source baselines.**  
   The evaluation currently excludes leading proprietary models.  Including at least one closed-source model via API, would strengthen claims of comprehensiveness and help contextualize open-weight performance.  This is probably not that expensive, but would improve benchmark credibility and adoption.

2. **Absence of statistical uncertainty.**  
   The paper reports single-run results without confidence intervals or variance estimates.  It is difficult to judge statistical significance.  The authors should specify the number of evaluation runs per model and report confidence intervals in the tables.  Otherwise, it's hard to interpret the numbers.

3. **Overstated claims about scale versus training quality.**  
   The paper claims (l. 350) that “training quality becomes more critical than parameter scale beyond 7B,” yet the largest model, GLM-4.5V-106B, also achieves the best performance overall, so the data are suggesting that **having more parameters does help a lot, doesn't it?**   This is an example of the numerous claims made in section 3.3 that are simply too strong in tone.  These are more like (valuable and useful) intuitions and not concrete conclusions supported by data.  The authors should either provide additional evidence—such as matched-scale ablations or training-data comparisons—or moderate their claim to reflect the limits of their observations.  My take is that the paper as it stands right now is strong enough, so just rewriting section 3.3 to make the tone more moderate is enough.  You can do more experiments and perform serious hypothesis testing about the claims in 3.3 in the future.

4. **Language and style issues.**  
   There are various grammatical and typographical errors throughout.  Please proofread and fix all of them. 
   Examples include:
   - Line 266: extra space after “selection.”  
   - Line 349: the comma should be a semicolon, connecting two complete clauses.  
   - Quotation marks uses are wrong; the authors make the beginners' LaTeX mistake of never typing any left quotation marks.  Please check how to write left quotation marks in LaTeX, or just use some package like `csquotes` that fixes this for you. 

5. **Benchmark reporting issues.**
   This is a good benchmark.  Please follow the guidelines in figure 4 of Zhu et al 2025 *Establishing Best Practices for Building Rigorous Agentic Benchmarks* to properly and systematically describe and report this benchmark.  Right now, from looking at their requirements, the current benchmark and the supplementary materials fall short in several aspects
   - There is no README document explaining how to use the code.
   - A large portion of the code is in Chinese (not just the annotations, but also the descriptions of errors), so non-Chinese speakers can find it hard to understand or use the benchmark. 
   - There is no  analysis of potential flaws.
   - There's no reporting or calculation of statistical significance.  
   - etc.

   The work is already high-quality, and it can be made more impactful and cleaner, if the authors follow the best practices for benchmark reporting. 

I'm okay if you don't solve problem 1.  Please solve problems 2, 3, 4, and 5

### Questions
1. **Generalization to closed-source models.**  
   Can the authors test whether the benchmark’s reward-model failure taxonomy also applies to closed models (e.g., GPT-4.1 or Claude-3.5)?  
   Would such models show qualitatively similar failure types or novel ones?

2. **Confidence interval reporting.**  
   How many evaluation runs were used to compute the reported averages?  
   Could the authors include confidence intervals to quantify robustness?

3. **Claim moderation.**  
   Could the authors clarify the empirical basis for claiming that training quality outweighs model size?  
   Is there a controlled experiment or only observational correlation?

4.  **Is there no hallucination?**
   I am just surprised that the error analysis does not contain any of those.  Do reward models never hallucinate?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes CUARewardBench, a benchmark for evaluating reward models (RMs) on computer-using agents (CUAs), aiming to address the limitations of script-based verifiers in CUA evaluation. The benchmark covers both outcome reward models (ORM) and process reward models (PRM), includes trajectories from 10 software categories and 7 agent architectures, and relies on expert annotations for evaluation.

### Strengths
⦁	As the first benchmark specifically targeting CUA reward models, it fills the gap of dedicated evaluation tools for desktop software-oriented CUA RMs.
⦁	The expert annotation process incorporates cross-checking and quality control mechanisms, ensuring basic reliability of the benchmark’s annotation data for subsequent CUA RM evaluation.

### Weaknesses
⦁	Limited innovation with partial similarities to existing work: Core framework is similar to the existing ICML work 'Boosting Virtual Agent Learning and Reasoning: A Step-Wise, Multi-Dimensional, and Generalist Reward Model with Benchmark'—both construct process reward model benchmarks to address traditional outcome-based evaluation limitations. CUARewardBench only adjusts the scenario to desktop CUAs; it uses expert annotation instead of the ICML paper’s automatic annotation (MCTS-P) to reduce costs, may leading to annotation inefficiency.
⦁	Poor and awful figures and citation presentation: Extremely few visualizations—only one ambiguous and unaesthetic figure. More clear visualizations would facilitate readers' understanding. Additionally, the paper appears to exclusively use \citet citation format, causing method names and authors to be conflated (e.g., "OSWorld Xie et al. (2024)"), which creates significant reading difficulties.
⦁	Limited and shallow experimental design: Evaluates few VLMs (mostly Qwen2.5VL series) and 3 prompt templates, missing comparisons with mainstream CUA models and powerful closed-source models (e.g., GPT-5, Claude 3.7) with strong agent capabilities.
⦁	Unjustified trajectory selection thresholds: Excludes trajectories with <1 or >7 successful agent configurations to control difficulty, but provides no rationale for this threshold (e.g., why 8+ successes are "too easy"). Also lacks analysis of how the threshold impacts the benchmark’s ability to distinguish RM performance.

### Questions
⦁	Explanation for experimental results: The experimental analysis only focuses on overall precision/recall, without in-depth exploration of performance differences across software categories (e.g., why Thunderbird has the lowest successful trajectory count but no analysis of underlying reasons) or model parameter scale vs. performance correlation (e.g., Qwen2.5VL-72B underperforming 32B but no in-depth technical explanation).
⦁	Lack of annotation quality metrics: The paper claims to have "rigorous expert annotation" but does not provide key metrics such as inter-annotator agreement (e.g., Cohen’s kappa coefficient). Please explain whether the annotation quality meets the standards of top conference benchmarks, and provide specific data to prove the reliability of the annotation.
⦁	More exploration of low-resource scenarios: The paper’s dataset only includes 272 trajectories, and it is highly recommended to explore the performance of RMs in low-resource scenarios (e.g., few-shot fine-tuning with limited trajectories). This is crucial for the practical application of the benchmark but is not mentioned at all.
⦁	Explanation for incomplete cross-software analysis: The paper covers 10 software categories, but the experimental results only focus on 5 categories (VS Code/GIMP/etc.) and relegate the rest to supplementary materials without analysis. Please explain why the performance differences across software categories (e.g., Thunderbird has the lowest successful trajectory count) are not discussed, and how these differences affect the benchmark’s generalization ability.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces CUARewardBench, the first benchmark designed to evaluate reward models for computer-using agents operating in desktop environments. The authors construct a dataset of 272 trajectories, collected from 7 different agent architectures performing tasks across 10 diverse software categories. The benchmark facilitates a dual evaluation of both Outcome Reward Models for trajectory-level success and Process Reward Models for step-level correctness. Through extensive experiments on 7 vision-language models, the paper reveals several key findings, notably that general-purpose VLMs outperform specialized CUA models in the reward modeling task, and that strong visual reasoning capability is the most critical factor for success.

### Strengths
Novel and Important Problem: This is the first work to systematically benchmark reward models for CUAs in diverse desktop environments. It successfully bridges the gap between agent execution environments like OSWorld and web-focused RM benchmarks like AgentRewardBench, addressing a critical need in the community.

Comprehensive Experimental Design: The evaluation is thorough, covering 7 different VLMs that span a spectrum of architectures (general-purpose, reasoning-focused, CUA-specialized) and testing them with 3 distinct prompt templates. This design allows for a robust and insightful analysis of the key factors that influence RM performance.

### Weaknesses
Insufficient Scale and Statistical Generalizability: The primary weakness of this work is the small size of the dataset. With only 272 trajectories distributed across 10 software categories and 7 agent models, the data per condition is extremely sparse. For some categories, such as Thunderbird, there are only 16 trajectories in total. This small sample size makes it difficult to draw robust, generalizable conclusions and raises serious questions about whether the observed performance differences between models are statistically significant. The absence of any statistical significance testing is a major oversight for a paper introducing a new benchmark.

Unverifiable Annotation Reliability: The paper's core contribution is its expert-annotated dataset, yet its reliability is not empirically demonstrated. The protocol for selecting "key" steps for PRM annotation is defined with subjective terms like "as large as possible," which is not a reproducible scientific standard. More critically, the paper fails to report Inter-Annotator Agreement (IAA), the standard method for validating the consistency and reliability of human-generated labels. Without IAA, the quality of the ground truth remains unknown, which fundamentally undermines the validity of the entire benchmark and the conclusions drawn from it.

### Questions
On Scale: Given the small number of trajectories per software category, could the authors perform statistical significance tests (e.g., bootstrapping or permutation tests) to validate that the performance differences reported between models are not simply due to chance? How do the authors reason about the generalizability of their conclusions from this limited dataset?

On Annotation Protocol: Could the authors provide a more concrete, operationalized definition for identifying "key" good/bad actions to improve reproducibility? Crucially, was any form of Inter-Annotator Agreement (IAA) calculated during the annotation process? If so, what were the scores? If not, how can the community be confident in the reliability of the benchmark's ground-truth labels?

On PRM Scope: The exclusion of redundant actions seems to significantly narrow the scope of PRM evaluation, limiting it to error detection rather than a holistic assessment of step quality that includes efficiency. Could the authors elaborate on the reasoning behind this decision and discuss how this limitation impacts the benchmark's utility for training agents via reinforcement learning, where rewarding efficiency is paramount?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces CUARewardBench, the first benchmark for evaluating both outcome reward models (ORM) and process reward models (PRM) on computer-using agents (CUAs). The benchmark contains 272 trajectories and 346 step annotations across 10 software categories, with data collected from 7 CUA architectures. The authors provide expert-validated annotations and analyze the performance of 7 vision-language models (VLMs) under 3 prompt templates. Key findings include that visual reasoning capability dominates reward model performance, general VLMs outperform CUA-specialized models, and current reward models still struggle on both trajectory- and step-level evaluations.

### Strengths
+ The paper addresses an important and timely gap in evaluating reward models for computer-using agents (CUAs).

+ The benchmark is well designed, covering both trajectory-level (ORM) and step-level (PRM) evaluation, and includes multiple software environments and agent types.

+ The empirical analysis is thorough and reveals clear weaknesses in visual reasoning and consistency across current reward models.

+ Writing and experimental presentation are clear and professional, enabling reproducibility.

### Weaknesses
+ While the benchmark clearly reveals important deficiencies in existing reward models, the paper does not explore or propose mechanisms to mitigate these issues. For instance, after identifying reasoning and visual comprehension failures, no ablation or refinement strategy is suggested to improve such capabilities.

+ The analysis remains largely diagnostic rather than prescriptive—it characterizes the current landscape effectively but stops short of translating the insights into new evaluation metrics or model improvements.

+ The results, though informative, focus primarily on descriptive comparison (precision/recall across models and prompts). The paper would benefit from further discussion on how these empirical findings could guide the design of future reward modeling frameworks or hybrid evaluators combining visual and symbolic verification.

### Questions
+ Since the benchmark conclusions rely heavily on observed precision and recall differences, how do the authors ensure that prompt sensitivity or label ambiguity did not confound these measurements?

+ The analysis points out that current RMs lack visual reasoning consistency, but the paper does not test any ablation or control study. Would it be possible to quantify which visual features or interface elements contribute most to model failures?

+ The study emphasizes the diagnostic role of CUARewardBench but not prescriptive validation. Could the authors provide additional empirical evidence (e.g., cross-correlation or regression analysis) showing that benchmark metrics meaningfully distinguish stronger from weaker reward models?

+ The paper mentions that screenshots alone limit observability for verifying outcomes. How do the authors control for unobservable success conditions when labeling trajectory outcomes to ensure reliability of the ground-truth annotations?

### Soundness
3

### Presentation
2

### Contribution
2
