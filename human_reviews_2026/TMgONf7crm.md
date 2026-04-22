# THINK-Bench: Evaluating Thinking Efficiency and Chain-of-Thought Quality of Large Reasoning Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Large reasoning models (LRMs) have achieved impressive performance in complex tasks, often outperforming conventional large language models (LLMs). However, the prevalent issue of overthinking severely limits their computational efficiency. Overthinking occurs when models generate excessive and redundant tokens that contribute little to accurate outcomes, especially in simple tasks, resulting in a significant waste of computational resources. To systematically investigate this issue, we introduce Think-Bench, a benchmark designed to evaluate the reasoning efficiency of LRMs. We also propose novel efficiency metrics and conduct a comprehensive evaluation of various LRMs across multiple dimensions, including the reasoning process, outcome quality, and chain-of-thought (CoT) characteristics. Our analysis reveals that most LRMs exhibit overthinking in handling easy questions, generating unnecessarily lengthy reasoning chains. While many LRMs demonstrate high CoT quality, several suffer from low efficiency. We hope that Think-Bench can serve as a robust foundation for advancing research into LRMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper propose the Think-Bench benchmark, aiming to systematically evaluate the reasoning efficiency of CoT quality of large reasoning models. To accurately identify and quantitatively evaluate the thinking procedure of large reasoning models, the authors designed nine metrics to measure CoT efficiency and quality. Evaluation results on 11 LRMs show that most models exhibit overthinking behaviors.

### Strengths
S1: This work targets an important yet underexplored research question which is to evaluate the chain-of-thought (CoT) quality of large reasoning models. They have marched a step in quantitatively evaluating the efficiency and quality of CoTs.

S2: The collected dataset and designed metrics can be helpful for future research works to improve the CoT quality and efficiency.

### Weaknesses
W1: Limitations in model-generated and manually-annotated "groundtruth" intermediate steps. To evaluate the correctness of intermediate reasoning steps in LRM-generated solutions, the authors construct ground-truth key reasoning steps for all questions (Section 2.3). However, many scientific problems admit multiple valid solution paths—differing either in the sequence of reasoning steps or in the granularity with which the steps are decomposed. As a result, the proposed Recall and Precision metrics can only measure deviation from a single predefined “gold” reference, rather than alignment with the broader space of correct reasoning trajectories. The paper does not appear to account for this multi-solution nature, which raises concerns about the completeness and fairness of its evaluation framework.

W2: Several of the proposed evaluation metrics such as reflection quality, recall, and precision are based on judgments from Claude 3.7 Sonnet, itself a large reasoning model. However, the faithfulness of LRMs as evaluators has been questioned [1, 2]. Such models may hallucinate, produce inconsistent intermediate judgments, or even contradict their own final conclusions. The paper should include a more thorough discussion of this reliability issue and clarify whether additional validation or calibration steps were taken to ensure the trustworthiness of the LRM-based judgments.

---
Reference

[1] Chen Y, Benton J, Radhakrishnan A, et al. Reasoning Models Don't Always Say What They Think[J]. arXiv preprint arXiv:2505.05410, 2025.

[2] Lu J, Xu Z, Kankanhalli M. Reasoning LLMs are Wandering Solution Explorers[J]. arXiv preprint arXiv:2505.20296, 2025.

### Questions
Q1. The valid reflection seems to have a vague and arguable definition (Line 256 - Line 258). Can you give an example what is a valid reflection which confirm an earlier conclusion? What is an invalid reflection by contrast?

Q2. Several evaluation metrics are quite "subjective" to specific judges - if changing the judge to different models, are the evaluation results still consistent?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Think-Bench, a benchmark specifically designed to evaluate the thinking efficiency and CoT quality of LRMs. The dataset contains 1,375 questions covering mathematics, physics, and chemistry, each with manually annotated key inference steps, totaling over 13,000 annotations. The authors designed a comprehensive evaluation metric system, including six efficiency metrics such as total tokens, first-time correct tokens, efficiency, reflection tokens, and number of thinks, and two CoT quality metrics. Through evaluation of 11 representative LRMs, the paper reveals the prevalent over thinking problem in current models, particularly in handling simple problems, where excessively long inference chains are generated, leading to wasted computational resources.

### Strengths
S1. The problem addressed in this paper is indeed a real challenge faced by large-scale inference models, especially given the increasing popularity of test-time-scaling strategies. This paper not only focuses on the accuracy of the final answer but also delves into the efficiency and quality of the reasoning process, filling a significant gap in existing evaluation systems. The proposed six efficiency indicators and two quality indicators complement each other, forming a relatively complete evaluation system capable of characterizing the model's reasoning behavior from multiple perspectives.

S2. The paper employs a rigorous two-stage annotation process. First preliminary references are generated using Claude 3.7 Sonnet, and then a team of graduate students with backgrounds in mathematics, physics, and chemistry conducts manual review and corrections. This annotation method ensures both efficiency and quality.

S3. The paper evaluates 11 representative LRMs, covering models of different sizes from 1.5B to 235B parameters and types, with broad experimental coverage. The experiments not only report overall performance but also provide detailed analyses by difficulty level and subject, revealing some interesting patterns.

S4. The paper provides an anonymous code repository containing the complete dataset, evaluation code, and all experimental scripts, which is good for the community.

### Weaknesses
W1. The core assessment in this paper relies on Claude 3.7 Sonnet to determine the correctness of reasoning steps and their match with the reference answer, but the reliability of this assessor is not adequately verified. There is no report on the inter-annotator agreement between human assessment and LLM assessment, nor is there an analysis of potential systematic biases introduced by Claude. For example, Claude may have a preference for certain expression styles or reasoning paths, which could affect the fairness of the assessment.

W2. What kind of reasoning chain is considered "excessive"? Is it merely a matter of length, or does it also include repetitiveness and inefficiency? The paper primarily uses the percentage of correctly identified tokens on the first attempt to measure this, but this metric has significant limitations: if the model performs necessary exploration before providing the correct answer on the first attempt, is this also considered "excessive"? 

W3. The criteria for judging "reflective quality" are rather subjective and lack a clear operational definition. These conceptual ambiguities may lead to inconsistent understandings of the results among different researchers.

W4. ThinkBench only covers three subjects: mathematics, physics, and chemistry, which limits the representativeness of the benchmark. Many important reasoning scenarios, such as programming, legal reasoning, and common sense reasoning, are not covered. Is it because annotating key steps is not feasible for issues that are not fully well-defined?

W5. Section 4.3 mentions that some models produce empty outputs. The paper chooses to skip these samples directly to "maintain the integrity of the evaluation," but this approach may actually introduce bias. These failure cases themselves reflect the model's deficiencies in reasoning ability, and simply skipping them will make the evaluation results of these models appear better than they actually are.

W6. The paper mentions some possible directions for improvement in the conclusion such as dynamic inference paths and early exit mechanisms, but these are all general and lack specific technical solutions or preliminary experimental verification. Including ablation experiments or case studies to explore which factors affect inference efficiency would make the paper more in-depth and instructive.

W7. The paper's evaluation process relies on multiple calls to Claude 3.7 Sonnet for step matching and quality assessment, which is quite expensive in terms of both computation and time. This could be a practical obstacle for researchers who want to evaluate new models using Think-Bench. The paper does not report the total cost including API call fees, time consumption and so forth.

### Questions
Q1. Did you conduct manual verification to check the accuracy of Claude 3.7 Sonnet as a judge?

Q2. Regarding the empty output issue mentioned in Section 4.3, could you provide more detailed statistics: How frequently does each model produce empty outputs? How does the distribution of this problem differ between simple and hard problems? Is it related to a specific subject or problem type? Why are these cases not counted as errors but instead skipped? Does this approach overestimate the actual performance of these models? How would the rankings change if empty outputs were recalculated and counted as errors?

Q3. The efficiency metric is defined as the percentage of the total length where the first correct answer appears, but does this truly reflect the "efficiency" of the reasoning? Consider two scenarios: First, the model quickly provides the correct answer and then performs extensive verification; Second, the model provides the correct answer only after careful consideration. By your definition, scenario 1 is more efficient, but the verification process in scenario 2 may be necessary, especially for difficult problems. How do you view this trade-off?

Q4. The `reflection quality` metric requires judging whether a reflection step accurately identified previous errors or provided new insights, and these criteria are quite subjective. Different people may have different judgments. Have you conducted inter-annotator consistency checks? Can you provide some boundary cases illustrating what constitutes effective reflection and what constitutes ineffective reflection? Is this metric reliable enough?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Think-Bench, a benchmark to evaluate reasoning efficiency and the quality of CoT in LLMs. It aims to evaluate the overthinking problem. The paper introduces new efficiency metrics and conducts an empirical evaluation, highlighting inefficiencies in models despite high CoT quality.

### Strengths
1. The idea of evaluating LRMs on both reasoning efficiency and CoT quality is innovative, providing a new framework that looks beyond the final answer accuracy, which is a limitation of many existing benchmarks.

2. The paper performs a detailed evaluation of 11 models, including both proprietary and open-source LRMs, across various domains and difficulty levels, offering insights into current reasoning inefficiencies.

3. The proposed metrics for reasoning quality, like reflection quality, contribute valuable tools to the field, facilitating further exploration of reasoning models.

### Weaknesses
1. Numerous studies have already proposed solutions to address overthinking [1-3], such as dynamic reasoning path design or early-exit mechanisms. This paper merely cites these ideas as potential future directions without providing any concrete comparative analysis, resulting in a rather weak analytical foundation.

2. The definition of “efficiency” is quite ambiguous, and it is unclear which type of efficiency the paper refers to. For instance, if efficiency is defined from the user’s perspective, it should correspond to the number of tokens required to produce a correct answer. If it concerns the model’s own efficiency, it should logically be defined as the number of tokens required to generate the correct answer on the first attempt (i.e., the number of tokens spent in a successful initial exploration, which aligns more closely with the conventional notion of efficiency—the smaller, the better). Dividing this by the total sequence length seems conceptually inconsistent. Moreover, if the model arrives at the correct answer in the first attempt but requires three to four rounds of verification, this could still indicate efficient reasoning and exploration, with inefficiency lying instead in the critic’s quality or performance.

3. The method heavily depends on Claude’s critic capability. When facing highly complex problems that exceed the model’s reasoning boundaries, even if intermediate reasoning steps contain errors, the model may fail to detect them. A rigorous evaluation framework should take this issue into account.

4. The evaluation cost is extremely high, and since the approach relies on closed-source models, reproducibility becomes difficult. The robustness of the results also remains uncertain.

5. Most of the performance analysis is descriptive, lacking an in-depth examination of the trade-off between reasoning quality and efficiency. Although the paper claims to demonstrate a balance, it does not sufficiently discuss why certain models excel in efficiency while others perform better in reasoning quality—particularly in tasks such as GSM8K, where direct instruction-following models may actually be more effective.

6. The error analysis in Section 4.3 focuses only on specific failure cases in a few models. A more comprehensive exploration of error types (e.g., logical inconsistencies or misinterpretations of the problem) would yield deeper insights into the limitations of these models.

7. In fact, all traditional CoT benchmarks seem to be able to perform similar metrics for evaluation by Claude, and the data contribution in this work does not appear to be that important.

[1] Sui Y, Chuang Y N, Wang G, et al. Stop overthinking: A survey on efficient reasoning for large language models[J]. arXiv preprint arXiv:2503.16419, 2025.

[2] Shen Y, Zhang J, Huang J, et al. Dast: Difficulty-adaptive slow-thinking for large reasoning models[J]. arXiv preprint arXiv:2503.04472, 2025.

[3] Chen Q, Peng D, Liu J, et al. Aware First, Think Less: Dynamic Boundary Self-Awareness Drives Extreme Reasoning Efficiency in Large Language Models[J]. arXiv preprint arXiv:2508.11582, 2025.

### Questions
1. This paper should mention more concrete comparative analysis about efficient reasoning.

2. What exactly does the term refer to efficiency from the user’s perspective, or the model’s intrinsic efficiency? Can you further explain why you define the efficiency in such way.

3. Since the proposed method heavily depends on Claude’s critic capability, how does the evaluation framework handle cases where the task complexity exceeds the model’s reasoning boundary? If the critic fails to detect errors in intermediate steps, how can the reliability of the evaluation be ensured?

4. The paper should discuss how to mitigate these issues, or assess the robustness of its results under different evaluation settings.

5. Can you provide deeper analysis on why there is no deeper examination of the trade-off between reasoning quality and efficiency?

6. In Section 4.3, the error analysis focuses only on specific failure cases. Why does the paper not include a broader categorization of error types, which could offer more comprehensive insights into the models’ limitations?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Think-Bench, a benchmark for evaluating reasoning efficiency and chain-of-thought (CoT) quality in LRMs. It contributes (i) a dataset with human-curated key steps across math/physics/chemistry, (ii) a suite of efficiency metrics (e.g., First-Correct Tokens, Reflection Tokens, Thought Num) plus CoT recall/precision, and (iii) a multi-model empirical study showing widespread “overthinking,” especially on easy questions.

### Strengths
Process-level evaluation. Goes beyond final accuracy to step-level CoT quality (recall/precision), enabled by human key-step annotations and a judging pipeline. This addresses a real gap in current benchmarks.

Clear efficiency metrics. The definitions of Efficiency (first-correct/total), Reflection Tokens, and Thought Num are intuitive and operationalizable.

Empirical findings. Consistent evidence of “overthinking,” differences across subjects and difficulty, and model-specific trade-offs are well documented (Tables 2–6, Fig. 6).

### Weaknesses
1. Missing marquee closed models. The model list includes Claude 3.7 Sonnet, DeepSeek-R1(+distills), Qwen3-235B, Grok-3-mini, ERNIE-X1-Turbo, GLM-Z1-Air—but not GPT-4/5 or Gemini-2.5-Pro. For a paper about thinking efficiency of LRMs, omission of today’s most-used LLMs weakens external validity. Authors do note some models don’t expose CoT, but a discussion/ablation on “no-CoT models” or proxy evaluations would help.

2. Simplicity / metric specificity. While practical, some metrics (e.g., Thought Num via discourse markers) feel heuristic and dataset-specific. The work would be stronger if it tied metrics to theoretical desiderata (e.g., optimal stopping/early-exit) or validated metric stability under paraphrase/noise.

3. Adoption risk. The benchmark is limited to STEM QA (1,375 items) and relies on Claude-as-judge for step matching. Community uptake may hinge on (a) openness of the judging pipeline beyond a single vendor, and (b) extensibility beyond the three domains. The paper mentions extensibility but does not demonstrate it.

4. Human reasoning annotations enable interpretable, step-aware evaluation rather than outcome-only accuracy. However, inter-annotator agreement and robustness of the step set (multiple valid paths) need more quantification; currently it’s described but not stress-tested.

### Questions
- Can you justify model coverage (why omit GPT-4/5, Gemini-2.5-Pro)? If CoT visibility is the blocker, can you evaluate no-CoT models via approximate proxies (e.g., self-consistency runs, early-exit triggering, or token-budgeted answers) and report at least Efficiency/Accuracy?

- Connection to “token complexity” trade-offs. The paper reports an efficiency–performance trade-off qualitatively (and via Efficiency vs. Tokens). It would be compelling to plot accuracy vs. total tokens (and vs. first-correct tokens) per model and per difficulty, and connect to the token complexity phenonmen in [1] to test curve universality on Think-Bench. The data seem available (Tables 2–6).

[1] Lee, A., Che, E., & Peng, T. (2025). How well do llms compress their own chain-of-thought? a token complexity approach. arXiv preprint arXiv:2503.01141.

### Soundness
3

### Presentation
4

### Contribution
2
