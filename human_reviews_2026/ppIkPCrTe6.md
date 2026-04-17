# CC-Learn: Cohort-Based Consistency Learning

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Large language models excel at many tasks but still struggle with consistent, robust reasoning. We introduce Cohort-based Consistency Learning (CC-Learn), a reinforcement learning framework that trains on cohorts of similar questions instantiated from symbolic programmatic abstractions and executes a programmatic solution unchanged across each cohort. Our composite objective mixes execution-based signals with critique-based signals. The execution-based signals include cohort-level accuracy, retrieval usage, and penalties for invalid lookups. The critique-based signals come from a frozen judge that checks whether the program’s sub-questions cover the key factors and whether its reasoning logic moves closer to a higher-quality self-improvement. Optimized via reinforcement learning, this objective steers the policy toward uniform, generalizable procedures rather than instance-specific shortcuts. Across five in-domain benchmarks (ARC-Easy/Challenge, CSQA, StrategyQA, HotpotQA) and three out-of-domain benchmarks (OpenBookQA, PubMedQA, MMLU), at two model scales (3B/7B), CC-Learn delivers roughly 10–20 absolute-point gains over strong baselines under both lenient and strict criteria, improving accuracy and stabilizing reasoning. These results show that cohort-level RL with execution signals and external feedback effectively enforces cross-variant consistency in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces CC-LEARN, an RL framework designed to improve reasoning consistency in LLMs. It trains models on cohorts of similar questions that share the same reasoning path but differ in factual details, forcing one executable program to solve all variants. Using GRPO, the model optimizes a composite reward that combines execution-based and critique-based signals, including accuracy, retrieval efficiency, and structural alignment from a frozen judge. The approach yields significant gains in reasoning stability and generalization across multiple QA benchmarks.

### Strengths
Strengths

* Well written and easy to follow – The paper is clearly structured, making the methodology and training pipeline easy to understand.

* Good motivation – The authors clearly articulate the importance of improving reasoning consistency in LLMs and provide strong justification for their approach.

* Interesting approach – The use of executable program generation to enforce shared reasoning paths across cohorts is a fresh and creative idea for me, and I think this adds novelty to this paper. Also, although the reward function contains 6 different terms which can hinder the model to be learn properly due to the complex signal, each term seems to be reasonable.

* Reliable experimental results – The inclusion of standard deviation values in the results supports the reliability and robustness of the reported performance.

### Weaknesses
Weaknesses and Questions

- Complex Reward Design – The reward model contains too many distinct components (execution-based and critique-based signals), making the overall optimization signal quite complex. It is unclear whether each term meaningfully contributes to the final behavior or if some components might introduce noise during training.

- Heuristic Reward Scaling – The scaling coefficients of individual rewards (e.g., 0.2 per correct answer, 0.06 for decomposition score) appear to be set heuristically without principled tuning or sensitivity analysis. It would be helpful to understand how robust the method is to different reward weightings.

- Cohort Example Ambiguity (Appendix A.3) – In the provided examples, the answers across two questions seem swapped (the answer for the first question is written under the second question and vice versa :)). Anyway, the main point is that it is not entirely clear whether the generated program solves all cohort variants through the same reasoning steps as claimed. For me, it seems the model solves two cohorts in different logic. Clarification is needed on how consistency manifests across variants.

- StrategyQA-Specific Improvement – In Table results, other baselines perform poorly on StrategyQA, whereas the RL_{cohort}(Exec + Crit) variant shows a large jump. What specific aspect of StrategyQA or the reward composition enables this sharp improvement? A deeper analysis would help understand when the proposed RL is more helpful for training.

- Limited Analysis of Failure Cases – While the paper provides some qualitative examples, it lacks a systematic discussion of where CC-LEARN fails or produces inconsistent reasoning. Understanding failure modes could clarify the limitations of the method and guide future refinements.

### Questions
See the above section. If the authors provide convincing explanations to my questions, I would be willing to raise my score, as the overall idea and motivation are good.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces CC-LEARN, a cohort-based reinforcement learning framework aimed at improving consistency in reasoning for large language models. It abstracts similar questions into masked templates, generates reusable executable reasoning programs, and employs atomic retrieval operations with rejection filtering. Experimental results demonstrate substantial accuracy improvements and enhanced consistency across diverse benchmarks compared to baseline methods.

### Strengths
- The paper introduces a framework that groups semantically similar questions into cohorts and trains the model to produce a single executable reasoning program shared across them. This design directly targets consistency by forcing uniform reasoning across paraphrased inputs, reducing random output variance and improving stability in reasoning behavior.

- Training is guided by a composite reward that combines execution accuracy, retrieval efficiency, and structural critiques from a frozen judge model. This dual-signal design rewards both correct outcomes and logically complete, well-aligned reasoning, effectively encouraging the model to generate reliable and factor-aware programs.

- Using cohort-based K-of-N evaluation, the paper demonstrates that CC-LEARN maintains consistent reasoning across rephrased or structurally similar questions.

### Weaknesses
- K-of-N scoring rewards group success but may mask problematic individual cases. Absent calibrated confidence or human-in-the-loop triggers, ambiguous items can be mishandled. That limits suitability in high-stakes settings.

- Similar-question cohorts and SFT programs are synthesized by frontier LLMs, then only a small sample is human-checked, which risks template artifacts or label bias leaking into both train and test.

- Figure 2 contains overlapping text.

### Questions
Please check the weaknesses.

### Soundness
3

### Presentation
2

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
This paper proposes an approach to ensure that LLMs produce robust reasoning by training on a cohort of similar examples instead of a single example in which the model could be rewarded for arriving at the right answer through the wrong reasoning steps. The approach consists of using a composite objective function that combines execution based signals like accuracy and retrieval usage that are easily measurable, and critique based signals like factor-complete decomposition that require a frozen LLM judge. Using GRPO to train in this cohort-based way, the approach shows promising results on in-distribution and out-of-distribution benchmarks.

### Strengths
- The paper addresses an important issue in LLM reasoning, where the model can be potentially rewarded for providing an incorrect reasoning because the final answer it outputs passes problem-specific tests.
- The method proposed to address this issue is innovative and builds on prior work in deep learning literature in the context of LLM reasoning: ensuring that the solution produced generalises across a cohort of samples instead of a single sample, which makes it much harder for the model to chance upon a solution with reasoning (in this case a piece of code) satisfying the entire cohort.
- The use of code as the reasoning trace is an interesting and novel choice because it forces the reasoning to be fully objective, which makes it executable and also interpretable.

### Weaknesses
- For most results in table 1, the 7B model seems to show similar performance for RL-normal and RL-cohort with execution-based and critique-based rewards (within confidence interval). Since the gap is not statistically significant, it seems to indicate that the gains compared to the rest of the baselines for the 7B model are due to the reward function design, not cohort-based consistency enforcement.
- Most of the statistically significant gains in Table 1 seem to be for the 3B model, so studying the effect of the proposed method in the context of the model size might give more insights into the actual behaviour of the method.
- A key weakness of this paper is the lack of a discussion on what it means to impose human-interpretable consistency to constrain the behaviour of LLMs and the reasoning patterns they produce. The idea of enforcing consistency in the reasoning trace through similar examples semantically and expressing them as code is appealing, but LLMs have shown emergent properties as well, which may be impacted by this type of cohort-based learning.
- Choosing definitions for lenient accuracy to mean that 4 out of 6 tests passed and strict accuracy to mean 5 out of 6 tests passed seems very cherry-picked and is not backed up by any convincing reason. It might be more meaningful in terms of results to observe the entire spectrum of test pass rate.
- While there is discussion on prior work in LLM reasoning, this paper would benefit from a subsection on similar approaches to enforce consistency in pre-LLM deep learning literature.

### Questions
- The choice of strict accuracy as being able to pass 5 out of 6 tests seems quite arbitrary. Given that strict accuracy usually means that all tests pass, would the authors clarify why this particular decision was taken? It would be helpful to see results with strict accuracy meaning all tests passed.
- How do the retrieve function usage reward and the rejection penalty interact with each other, since one is rewarding correct decomposition and the other is penalising very simple queries? Is there a reason for why the coefficients for these are 0.6 and -0.1 respectively? More broadly, can the authors discuss the rationale behind the coefficients for the reward function and whether or not there were ablations conducted on different values of these coefficients?
- One interesting insight from the paper is that a larger retriever at eval time is always better than a smaller one. Have the authors tried a 7B-3B version, where the eval retriever is smaller than the one used in training?

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
2

### Summary
The paper proposes CC-Learn, a new reinforcement learning framework to fix a key weakness in large language models (LLMs): reasoning inconsistency. LLMs often answer one question correctly but fail on a rephrased or similar version because they use different, fragile reasoning steps. To address this, CC-Learn trains an LLM to produce one Python program that must solve a whole group (or “cohort”) of semantically similar questions. CC-Learn achieves 10–20+ point gains in both accuracy and reasoning consistency over strong supervised baselines on several benchmarks (e.g., ARC, CSQA, StrategyQA, MMLU, PubMedQA).

### Strengths
- The approach delivers large absolute improvements (10–20+ points) compared to strong baselines, consistent across various datasets, model scales (3B and 7B), and evaluation settings (Lenient and Strict).

- The model also performs strongly on out-of-domain benchmarks, indicating that the reasoning strategies learned during training generalize well to unseen tasks and domains.

### Weaknesses
1. **High reliance on subjective critique rewards:**
   CC-LEARN’s performance depends heavily on two critique-based rewards, ( $R_{fc}$ and $R_{sa}$ ), both provided by a “Judge” model. $R_{fc}$ is a qualitative score (1–10) reflecting how well the Judge thinks the program covers key factors—an inherently subjective judgment. $R_{sa}$ measures structural similarity to an “improved” program $p^{+}$ that the Judge itself creates, meaning the model is rewarded for resembling the Judge’s output rather than being truly correct. As a result, the policy’s performance is effectively limited by the Judge’s own reasoning and programming ability.

2. **Narrow applicability and rigid design:**
   The method is well-suited for tasks that can be neatly expressed as Python programs using atomic `retrieve` calls (e.g., QA or commonsense reasoning), but its generalization to open-ended or creative tasks remains unclear. Moreover, the strict enforcement of “atomic” retrievals—defined through a few-shot prompt—may reduce flexibility, causing the model to produce unnecessarily long or error-prone code when a slightly more complex retrieval would suffice.

### Questions
1. Since the policy's quality is guided by the Judge's critique ($R_{sa}$) and ablations show stronger Judges yield stronger policies, what happens if the Judge model is weaker than the Policy model? Is the framework capable of "self-improvement" where the policy can surpass the quality of its Judge?

### Soundness
2

### Presentation
2

### Contribution
3
