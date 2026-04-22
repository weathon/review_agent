# RECAP: REwriting Conversations for Intent Understanding in Agentic Planning

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 6

## Abstract
Understanding user intent is essential for effective planning in conversational assistants, particularly those powered by large language models (LLMs) coordinating multiple agents. However, real-world dialogues are often ambiguous, underspecified, or dynamic, making intent detection a persistent challenge. Traditional classification-based approaches struggle to generalize in open-ended settings, leading to brittle interpretations and poor downstream planning. We propose RECAP (REwriting Conversations for Agent Planning), a new benchmark designed to evaluate and advance intent rewriting, reframing user-agent dialogues into concise representations of user goals. RECAP captures diverse challenges such as ambiguity, intent drift, vagueness, and mixed-goal conversations. Alongside the dataset, we introduce an LLM-based evaluator that assesses planning utility given the rewritten intent. Using RECAP, we develop a prompt-based rewriting approach that outperforms baselines. We further demonstrate that fine-tuning two DPO-based rewriters yields additional utility gains. Our results highlight intent rewriting as a critical and tractable component for improving agent planning in open-domain dialogue systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces **RECAP**, a benchmark designed to evaluate *intent rewriting* in open-ended, multi-turn dialogue systems, especially those involving agentic planning. The authors argue that existing dialogue benchmarks fail to capture the process of reformulating ambiguous or underspecified user intents, which is crucial for accurate downstream task planning. The work includes both a dataset (RECAP) and a lightweight environment (RECAP-Toy) for controlled experiments. The study further explores whether fine-tuning a large language model (GPT-4-class) on preference pairs can improve intent rewriting and, consequently, planning performance.

While the motivation is relevant, namely linking intent understanding with planning, the methodology, evaluation, and contribution boundaries remain unclear.

### Strengths
- **Problem framing**: The paper identifies 'intent rewriting' as a potential bottleneck in multi-turn, open-ended dialogue systems, especially where goal-oriented planning is required.
- **Benchmark focus:** Establishing a benchmark around this specific functionality (rewriting for planning) is an interesting step toward formalizing a neglected capability of LLM-based agents.
- **Initial empirical results:** The study provides preliminary evidence that better intent rewriting can positively influence plan generation quality.

### Weaknesses
1. **Unclear novelty and positioning**  
   The contribution of RECAP as a benchmark is insufficiently distinguished from existing datasets. Intent rewriting overlaps conceptually with prior dialogue understanding and task reformulation benchmarks (e.g., MultiWOZ, ALFRED, Tau-Bench). The authors should clarify what new capability RECAP measures that others cannot.

2. **Synthetic and limited data generation**  
   The benchmark heavily relies on **synthetic dialogues**, raising concerns about representativeness and realism. Comparable or richer examples could be derived from existing corpora, reducing the need for a fully synthetic setup.

3. **Questionable need for RECAP-Toy**  
   The introduction of a simplified environment (RECAP-Toy) seems to distract from the main benchmark. Its utility is unclear, and it reduces the ecological validity of the evaluation.

4. **Incomplete evaluation design**  
   - The evaluation lacks an end-to-end metric (e.g., task success or goal completion) connecting rewriting quality to actual execution outcomes.  
   - Reported metrics (pairwise win rates, plan quality improvement) remain qualitative and fail to quantify the true downstream impact.  
   - The paper assumes a rigid sequential flow, from *intent understanding* to *planning* and then *execution*, without exploring reactive or interleaved strategies that may render rewriting unnecessary.

5. **Narrow generalization and static planner assumption**  
   All experiments are conducted within a single planning setup. It remains unclear whether improvements generalize across planners, domains, or conversation styles.

6. **Limited interpretability of findings**  
   The main takeaway, showing that intent rewriting affects planning, is intuitively expected but not operationalized into actionable insights. The proposed methods (prompt engineering, preference fine-tuning) are well-known and not evaluated for robustness or transfer.

### Questions
- **Overall design:** Can user interaction occur in parallel with plan execution, or must all ambiguities be resolved beforehand? This affects the necessity of intent rewriting.
- **Figure 3:** Is there a rationale behind the specific topic arrangement? Clarifying the visual structure would help interpretability.
- **Line 199:** Please clarify what is meant by “pair-wise method” in the context of preference training.
- **Footnote 3:** The reviewer still believe that including a basic rewriter baseline would provide useful insights and facilitate a more comprehensive evaluation of the results.
- **Line 326:** The purpose of RECAP-Toy requires clearer justification, namely what gap does it fill beyond debugging convenience?
- **Lines 331–333:** The sentence appears duplicated; editing needed.
- **Lines 364–372:** The authors acknowledge a flaw in the benchmark but do not fix it before release. This undermines the dataset’s reliability.
- **Table 5:** For DPO:LLM on *Shifted Intent*, *Noisy Input*, and *Total*, the **bold font** should highlight the **Loss Rate**.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on intent rewriting to help planners understand ambiguous, long conversations. They propose a new benchmark, RECAP, to test this. They show that a prompt-based rewriter (Advanced) leads to better plans than naive baselines. Further, they DPO-fine-tune rewriter models using both human labels and labels from an automatic LLM-evaluator.

### Strengths
1. The work clearly identifies that user intent can change during a conversation which can confuse agentic planner. I like the idea of decomposing the task, clearly this is better than simply generating summary of the conversation. 
2. The authors introduce a new dataset specifically designed to capture these conversational challenges (shifted intent, multi-intent, etc.). Their sensitivity analysis also shows it's more challenging and is better at distinguishing between a dummy rewriter and advanced rewriter.

### Weaknesses
Main:
1. The work assumes that a better plan will lead to successfully completing a task. They do not evaluate this hypothesis and the work is missing downstream evaluation. 
2. It is a synthetically generated dataset, which might not work for human conversations. I think synthetic data is useful if you can show that it helps in tasks that involve real-human conversations.
3. The human evaluation is done on a small subset, and shows different results than LLM-as-a-judge results on the complete set -- showing that DPO:human has no significant gain over the advanced rewriter.
4. The work uses a single static planner. It remains to see how this module would help other models. Would this rewriter be useful for new models like GPT-5. An experiment to show this could be trying different parameter models from the same family or comparing cross family of models.

### Questions
1. What is the llm-as-a-judge performance on RECAP-toy?
2. What is the sensitivity analysis for basic vs advanced?
3. Why are the llm-as-a-judge evaluation so different than human judgements? What are human judges catching that the llm-as-a-judge doesn't?

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
3

### Summary
This paper introduces RECAP, a benchmark designed to evaluate and improve intent rewriting for large language model (LLM)-based agentic planning. The authors argue that existing intent classification methods struggle with ambiguity, drift, and multi-intent scenarios in real-world dialogues. RECAP contains 810 synthetic user–agent conversations covering diverse topics and intent challenges. The paper also proposes an LLM-based evaluator for assessing plan quality derived from rewritten intents. Through experiments comparing prompt-based and DPO-finetuned rewriters, the study shows that effective intent rewriting can improve downstream task planning quality and alignment with human preferences.

### Strengths
- This paper is easy to read and logically clear.
- The experimental framework is relatively systematic, with complete links.
- This paper is clearly presented with descriptive figures.

### Weaknesses
- The experimental scope is limited: although RECAP contains 810 dialogues, only 150 are used for evaluation, without disclosure of their composition across intent types, domains, or dialogue lengths. This weakens the representativeness of the results and prevents meaningful analysis of how dialogue length or topic affects model performance.
- The paper lacks a quantitative evaluation of intent understanding. Scenario categories such as Shifted, Noisy, and Underspecified Intent are described only by examples, not by formal definitions or measurable criteria. Without explicit intent-level metrics or annotation guidelines, it is unclear whether the reported improvements truly reflect better intent recognition.
- The evaluation framework conflates planning quality with overall task success. Results are based mainly on plan preferences judged by humans or LLMs using a single planner, without statistical significance tests or execution-based validation. This limits the reliability and generalizability of the claimed improvements.

### Questions
- How were the 150 dialogues selected from the 810 available in RECAP? Could you provide the distribution of these samples across intent types, dialogue lengths, and topics?
- How do you define and annotate the five intent challenge types (Shifted, Noisy, Underspecified, Multi-Intent, Perfect)? Are there formal annotation criteria or inter-annotator agreement statistics?
- Since intent understanding is the main focus, do you have any quantitative metric—such as intent classification accuracy or intent drift detection rate to complement plan preference evaluation?
- Have you tested whether the observed improvements hold across different planners or decoding settings (e.g., beyond GPT-4o)?
- Can you provide statistical significance tests or confidence intervals for human evaluation results to demonstrate the robustness of the reported improvements?

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
4

### Summary
This paper introduces the RECAP benchmark for evaluating intent rewriting in LLM-driven dialogue systems. This benchmark captures key challenges such as ambiguity, intent drift, and mixed-goal conversations. By reformulating the dialogue into a concise representation of the user's goal, rewriting enables more accurate downstream task execution planning.Experimental results demonstrate that both prompt-based and DPO-trained rewriters significantly improve planning effectiveness. These results provide guidance for building more effective and adaptive dialogue agents.

### Strengths
1. RECAP is a new benchmark dataset specifically designed to evaluate and improve intent rewriting, especially in open-domain dialogue systems.
2. The RECAP systematically captures multiple intent rewriting challenges, including intent ambiguity, intent drift, ambiguity, and multi-objective dialogue.

### Weaknesses
1.The RECAP dataset spans five distinct domains: cooking, programming, flights, restaurants, and health. These topics were selected to cover a diverse range of common, goal-oriented conversational scenarios that users frequently engage in with AI assistants.While these domains are representative of many real-world interactions, the areas covered are still not comprehensive enough to prove more generalization.

2.RECAP categorizes intent challenges into five types: shifted intent, noisy input, underspecified intent, multi-intent, and perfect intent. While these categories cover a broad range of intent-related challenges, other types such as contradictory intent (where users express conflicting goals) or emotionally charged intent (where emotional language affects goal clarity) may also be relevant.

3.The paper's core experiments used only 150 data points, representing only about 18.5% of the entire RECAP dataset (810 instances). This raises questions about the statistical significance and reliability of the experimental results. Is a method that performs well on a small subset also robust on the full dataset? The paper's appendix mentions an evaluation of DPO:human and Advanced on the full dataset, but the results show a very neutral preference rate, which is somewhat inconsistent with the paper's conclusion that DPO:human is significantly superior to Advanced. This requires further explanation.

### Questions
1.The RECAP dataset is constructed across five specific domains: cooking, programming, flights, restaurants, and health. Could the authors further elaborate on the criteria and rationale for selecting these particular themes? To what extent can these themes represent common task-planning scenarios in open-domain dialogue assistants? How well can the conclusions drawn from performance on these themes generalize to other important domains (e.g., finance, education, entertainment)? Have the authors considered the potential biases introduced by this domain selection? 

2.The paper defines five categories of intent challenges: shifted intent, noisy input, underspecified intent, multi-intent, and perfect intent. Why were these five types chosen as the core challenges for the benchmark? Are there other equally important and common intent-understanding challenges (e.g., contradictory intents, implicit intents, or intents obscured by emotional language) that are not covered by the current taxonomy? Could the authors provide a more detailed justification that the current five categories are both necessary and sufficient for evaluating planning utility?

### Soundness
3

### Presentation
3

### Contribution
3
