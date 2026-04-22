# Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Large Language Models (LLMs) are widely used as judges to evaluate response quality, providing a scalable alternative to human evaluation. However, most LLM judges operate solely on intrinsic text-based reasoning, limiting their ability to verify complex constraints or perform accurate computation. Motivated by the success of tool-integrated reasoning (TIR) in numerous tasks, we propose TIR-Judge, an end-to-end RL framework for training LLM judges that integrates a Python executor for precise evaluation. TIR-Judge is built on three principles: (i) diverse training across verifiable and non-verifiable domains, (ii) flexible judgment formats (pointwise, pairwise, listwise), and (iii) iterative RL that enables bootstrapping directly from a base model without distillation. On six public benchmarks, TIR-Judge surpasses strong reasoning-based judges by up to 6.4% (pointwise) and 7.7% (pairwise), and achieves listwise performance comparable to Claude-Opus-4 despite having only 8B parameters. Remarkably, TIR-Judge-Zero—trained entirely without distillation—matches the performance of the distilled variants, showing that tool-augmented judges can self-improve through iterative reinforcement learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Proposes TIR-Judge, a framework to train (finetune) LLM judges. Specifically focuses on training them how to use tools (Python executor) using reinforcement learning, across verifiable and non-verifiable domains. Evaluate TIR-Judge on multiple benchmarks and show significant improvements over baseline, with just a small model.

### Strengths
1. Introduces a simple but effective framework to improve LLM-as-a-judge using RL on relatively small models.
2. Good figures and explanations.
3. Detailed evaluations and ablation comparisons. Well structured.

### Weaknesses
1. “eward-guided best-of-N inference over datasets” typo
2. Analysis on the code it does in Python? How often errors (runtime, syntax)?
3. Given they propose a framework, it is hard to know how well the training recipe and setup works in other setups. How well does it work on smaller models? Different model families?

### Questions
1. In Findeis 2025’s work, it was found that some of the benchmarks are too easy out of the box. How does your RL scale with complexity? E.g. does a better model still see improvements (on harder tasks), or do we only see gains here since it’s “low hanging fruit”?
2. "For safety and general helpfulness prompts, a positive format reward is granted only if the model produces a valid output without invoking tools" -- What is the impact of this?
3. Impact of training duration?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents TIR-Judge, a framework that trains LLM-based judges to integrate Python code execution with reinforcement learning, achieving strong empirical results across six benchmarks with notable parameter efficiency (8B model matching 32B baselines). The work demonstrates experimental coverage including pointwise/pairwise/listwise evaluation formats and best-of-N validation, with generally clear presentation and detailed appendices. The authors present the first approach to co-optimize inference and tool use (executing Python code) for LLM evaluator learning via reinforcement learning. Researchers achieved up to 6.4% performance improvement on pointwise and 7.7% performance on pairwise compared to robust inference-based evaluators on six public benchmarks, and 96% of Claudet-Opus-4 performance on 8B parameter models. TIR-Judge-Zero performs on par with or superior to distillation-based versions with only pure reinforcement learning without distillation from the teacher model.

### Strengths
1. Defining clear and practical issues
The practical limitations of LLM evaluators (accurate calculation, failure to verify constraints) are clearly presented with concrete examples.
The importance of evaluators has been convincingly described throughout the model development pipeline (post-training, inference, and evaluation).

2. Comprehensive Experimental Design
Researchers systematically evaluated it on six different benchmarks. They demonstrate generalizability by addressing all three evaluation formats: Pointwise, pairwise, and listwise.
Best-of-N experiments validate its practicality in downstream applications.

3. Thorough ablation study
Authors systematically analyzed the data mixing effect (Figure 3), tool usage vs text-only (Figure 4a), and progress of iterative RL (Figure 5).

4. Efficiency and Scalability
Parametric Efficiency: 8B model showed competitive performance with 32B model.
Computational Efficiency: We demonstrate no additional inference time overhead despite tool integration (Figure 6).

5. Efforts for Reproducibility
Addendum provided detailed prompt templates, dataset configurations, hyperparameters, and more.
It specified data pollution prevention efforts through 8-gram deduplication.

### Weaknesses
1. Problem: The key components of this paper (tool-integrated reasoning, RL-based judgments, and projection sampling) are all techniques covered in previous studies. Combining them together is meaningful, but there is a lack of theoretical analysis or deep insight into why this particular combination is effective.

Evidence: Feng et al. (2025), Li et al. (2025a), cited by Related Work, have already learned TIR as RL, while Chen et al. (2025b), Whitehouse et al. (2025) have dealt with RL-based judges. This paper is a combination of these, but does not explain why applying TIR to judges is essentially different from applying to policy models.


2. Lack of statistical rigor (Major)
Problem: All experimental results were reported as single run point estimates, and no confidence interval, standard deviation, or statistical significance tests were presented at all.

Evidence: All figures in Table 1 are shown up to a single decimal point, and there is no variance or p-value for multiple random seeds. In particular, we do not know if the performance difference between TIR-Judge-Zero and Distill (e.g., a 3.7% difference in 4B pointwise) is statistically significant.
Impact: The credibility of one of the main arguments, the conclusion that "distillation is unnecessary," is diminished.


3. Unfair comparison and baseline setting issues (Major)
Problem 1: The results of some strong baseline models (J1, GPT-4o, Claude 3.5, etc.) were cited in the original paper, and it is unclear whether they used exactly the same evaluation protocol and data preprocessing.
Problem 2: The "Tool-augmented" baseline (Qwen3-Tool, Gemini-2.5-Flash-Tool) seems to have induced tool use simply by prompting; for a fair comparison, they should have been fine-tuned with the same amount of tool use data.

Evidence: In Section 5.1, we said "For baseline, we consider... Qwen-3 (Team, 2025) (our backbone) that use the same tool as TIR-Judge," but they didn't specify how to learn. If you look at the prompt in Appendix B, it simply instructs you to "You may use Python code."
Impact: It's possible that the actual benefits of tool integration have been overestimated.



4. Lack of consistency in TIR-Judge-Zero's claim of excellence (Moderate)
Problem: While the paper emphasizes that TIR-Judge-Zero is effective without distillation, the real-world results are mixed with benchmarks and scales.

Evidence:
4B pairwise: Zero excels on 3 of 6 benchmarks
8B pairwise: Zero is only good on 2 of 6 benchmarks
4B pointwise: Zero is excellent in 7 out of 10 subtasks
8B pointwise: Distill excels in 6 out of 10 subtasks (Table 4)
Impact: The claim that "distillation is unnecessary" is exaggerated and should be corrected to "competitive performance can be achieved without distillation under certain conditions".

### Questions
1. Methodology Related (Critical)
Q1: Please clarify what exactly sθ(x, y) of pointwise evaluation means in Section 4.2 Equation (2). Is it the value in the <score> tag generated by the model or is it the scalar output inside the model? If it is the former, what do you do if an error occurs while parsing the generated text?

Q2: TIR-Judge-Zero's iterative learning said it would prefer to "keep only one trajectory per each prompt, with the shortest response or least tool calls," but how do you decide when these two criteria conflict (e.g., long response, but with fewer tool calls)? Do you have priorities?

Q3: Why did you choose the multiplication form from the reward function of Equation (3)? Have you tried the addition form (R = α₁ Rc + α₂Rf + α ₃ Rt)? Didn't the multiplication form cause the sparse reward problem early in learning?


2 Experimental Design Related (Critical)
Q4: How many random seeds did the results in Table 1 run? TIR-Judge-Zero 4B pairwise recorded 75.0% on PPE, what is the standard deviation or confidence interval for this? Specifically, did you test whether the performance difference between TIR-Judge-Zero and Distill was statistically significant?

Q5: How did you implement the "Tool-augmented Judges" baseline (Qwen3-Tool, Gemini-2.5-Flash-Tool)? Did you simply add "you may use Python code" to the prompt, or did you do new-shot prompting with some examples of tool usage? How much does learning them with data like TIR-Judge improve the performance?

Q6: Pairwise evaluation said you used "single random ordering", how did you control position bias? It's standard practice to evaluate and average both A-B and B-A orders, so why didn't you do this?


3. Interpretation of Results (Important)
Q7: Table 4 shows a mixture of cases where TIR-Judge-Zero is superior to distilled versions and cases where it is not. Under what conditions do you find a pattern that is better for Zero and under what conditions is better for Distill? Based on this, what would you recommend to practitioners?

Q8: In Figure 5, will the performance improve further or will it converge if we continue with an additional RL round after TIR-Judge-Zero-RL-2. What is the optimal number of iterations?

Q9: TIR-Judge on the RewardBench2 (Table 2) is higher than Claude-Opus-4 on the Instruction Followings, but far behind on Safety. What do you think is causing this difference? Could tool use be rather detrimental to safety assessments?
6.4 Regarding calculation costs (Important)

Q10: How do I compare the total learning cost (SFT + RL) of TIR-Judge-Distill to the total learning cost (RL + RS + SFT of multiple rounds) of TIR-Judge-Zero in GPU time? If the Zero method requires more computation, which is more efficient compared to the cost of generating a 10k sample of distillation data?

Q11: How many samples did Reception sampling generate on average per each prompt, and what percentage of them produced the correct format and answer? How did this acceptance rate change as learning progressed?


5. Generalization and Limitations Related (Moderate)
Q12: How do I modify my framework to extend it to other tools (web search, calculator, database query, etc.) other than the Python executor? Is the current reward function or learning procedure still applicable to other tools?

Q13: Has it occurred during learning or evaluation to generate malicious or dangerous code (e.g. file system access, infinite loop)? How did you detect and process it?

< Additional feedback for improvement >
1. Must-have 
Add statistical significance: Rerun the experiment with at least 3-5 random seeds, report the mean±standard deviation. For key arguments (for example, Zero > Distill), provide a paired t-test or bootstrap confidence interval.

Compare fair baselines: Reevaluate tool-augmented baselines by fine tuning them into the same learning data. Separate the actual contribution of RL by adding "TIR-Judge without RL (only SFT)" as at least ablation.

Position bias control: Perform Pairwise evaluation in both A-B and B-A directions and average; current single ordering can have a maximum bias of ±5%.

Add an experiment to compare the multiplication form and the addition form of the reward function ablation: expression (3). This will show the robustness of the method.

2 Strongly recommended improvements
Calculate Cost Analysis: Add a table comparing the total cost of learning in Distill vs Zero to GPU time and dollar amounts. This is important for practitioners to make decisions.
Add theoretical analysis: Add theoretical insights or analyses on why tool integration is particularly effective for judges and why iterative RL converges. There is currently a lack of explanation beyond "it works empirically".

Failed Case Analysis: Add a section to analyze when TIR-Judge still fails. For example, if you write a code but make an error due to incorrect logic, or if you need to use the tool but don't use it, and so on.
Data Efficiency Analysis: What percentage of 26k samples actually require tool use?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces TIR-Judge, a framework for training LLM-based judges that integrate tool execution (Python interpreter) within a reinforcement learning pipeline. Unlike prior text-only judges, TIR-Judge enables LLMs to generate, execute, and learn from code to perform verifiable evaluations.

Empirical results on six major benchmarks (e.g., PPE, IFBench, CodeJudgeBench, RewardBench) show consistent improvements over reasoning-based judges.

### Strengths
1. Novel integration of tool-use with reinforcement learning: The work goes beyond inference-time tool invocation by embedding code execution into the RL loop, allowing models to learn when and how to use tools effectively.
2. Strong empirical performance: Demonstrated improvements across multiple datasets and formats, outperforming larger models like 32B RRM on several metrics. Particularly impressive is the performance of TIR-Judge-Zero, which learns without teacher supervision.

### Weaknesses
1.Domain bias: Gains are largest in verifiable domains (math, code), while improvements in non-verifiable areas (helpfulness, safety) are marginal.
2. Generalization limitations: The framework is Python-specific; it is unclear how well the method would extend to multiple heterogeneous tools or symbolic engines.

### Questions
1. Could iterative RL amplify over-confidence or lead to reward hacking?
2. How is when not to call a tool learned? Is this behavior emergent from the RL reward or manually encoded through heuristics?
3. Would a multi-tool setup (e.g., Python + web search + symbolic solver) be stable under the same RL objective?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces TIR-Judge, an end-to-end RL framework for training LLM-based Judges that interleaves reasoning with Python code execution. Using this framework, the authors train several TIR-Judge models, which tend to outperform other LLM-based Judges without tools.

### Strengths
1. The paper is well-written, and the motivation and approach are clear. The figures and case study enhance readability.
2. The proposed method framework is effective, and the evaluation is comprehensive, spanning a variety of baselines and several relevant benchmarks.
3. The flexible judgement formats (e.g., pointwise, pairwise, and listwise) expand the functionality of the framework.

### Weaknesses
1. Prior work has identified several biases common in LLM-based judges, e.g., positional bias [1], verbosity bias [2], and self-preference bias [3]. Other work [4] has shown that training can increase the prevalence of such biases. Given this, this work should evaluate for bias before and after training.
2. It's unclear how the baseline tool-augmented judges (e.g., Qwen3-4B-Tool or Gemini-2.5-Flash-Tool) are evaluated. Are they expected to output tool calls in the same format as TIR-Judge, e.g., '''python...''', or do they use the native tool-calling features of the models? If it's the former, it's difficult to assess whether the improvement comes from learning how to make effective tool calls versus just learning how to make tool calls using the correct format. If it's the latter, it should be made clear in the paper.
3. The studies on efficiency could be better explained. Is code execution time factored in? Why are TIR-Judge-Zero 4B and TIR-Judge-Distill 4B 2x as efficient as Qwen-3-4B (their base model), yet TIR-Judge-Distill 8B has similar latency to Qwen-3-8B? Can you provide the average number of generated tokens for each?
4. (minor) Table 6 is confusing. It reports the number correct instead of accuracy, and the lack of precision makes comparisons difficult. Is this over a single run of 16 responses?
[1] https://arxiv.org/pdf/2406.07791
[2] https://arxiv.org/pdf/2310.10076
[3] https://arxiv.org/pdf/2410.21819
[4] https://openreview.net/pdf?id=JFTSZa2stt

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3
