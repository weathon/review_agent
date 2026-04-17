# SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Large language models (LLMs) are increasingly tested for a "Theory of Mind" (ToM) — the ability to attribute mental states to oneself and others. Yet most evaluations stop at explicit belief attribution in classical toy stories or stylized tasks, leaving open the questions of whether LLMs can implicitly apply such knowledge to predict human behavior, or to judge an observed behavior, in diverse scenarios. We introduce SimpleToM, a benchmark that advances ToM evaluation along two novel axes. First, it probes multiple levels of ToM reasoning, from mental state inference (explicit ToM) to behavior prediction and judgment (applied ToM). Second, it situates these tasks in diverse, everyday scenarios — such as supermarkets, hospitals, schools, and offices — where information asymmetries naturally arise (e.g., hidden defects in grocery store items, incomplete information in provider–patient interactions, or restricted access to locked devices). SimpleToM contains concise stories (e.g., "The can of Pringles has moldy chips in it. Mary picks up the can in the supermarket and walks to the cashier."), each with three questions that test different degrees of ToM reasoning, asking models to predict: (a) mental states ("Is Mary aware of the mold?"), (b) behaviors ("Will Mary pay for the chips or report the mold?"), and (c) judgments ("Mary paid for the chips. Was that reasonable?"). Experiments reveal a striking gap: state-of-the-art models often reliably infer mental state (a), but fail at applying knowledge about the mental state for secondary predictions, with performance dropping sharply for behavior prediction (b) and further for behavior judgment (c). This exposes a critical fragility in LLMs’ social reasoning in terms of what they know (explicit ToM) versus how well they can implicitly apply that knowledge for predictions (applied ToM). By uniting assessment of different levels of ToM reasoning with diverse, everyday scenarios, SimpleToM opens new opportunities for rigorously evaluating and diagnosing ToM abilities in LLMs, and reveals surprising, new insights about current model capabilities, guiding efforts toward future generations of models capable of robust social understanding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces SimpleToM, a benchmark for testing whether LLMs can not only infer mental states (explicit ToM) but also use that understanding to predict and judge behavior (applied ToM).

Results show that (1) while models perform well in inferring mental states, their ability drops sharply for behavior prediction,
(2) model performances across different scenarios differ, and 
(3) prompting-based interventions do not fully close the performance gap between implicit and explicit ToM.

The presentation is clear, but my main concern is the originality and significance of the work. I will detail these below.

### Strengths
- The presentation is clear, and the experiments are thorough.
- The main distinction between SimpleToM and existing benchmarks appears to be that, based on the example figure and Section 6, SimpleToM does not explicitly describe the characters’ observations. Instead, models must rely on implicit commonsense knowledge. For instance, understanding that “Mary cannot open the chips before paying for them.” This setup is relatively novel and could be valuable for assessing ToM in more natural, less-structured everyday contexts.

### Weaknesses
The main concern is that the originality of this work is limited. To my knowledge, BigToM [1] also adopts real world scenarios and includes an “applied ToM” task (the forward action task), suggesting some overlap in contribution.

It is also unclear why models perform poorly on the behavior judgment task even when mental states are given. A possible issue I notice is ambiguous phrasing in the dataset. For example, the question “Next, Mary pays for the chips and not reports the moldy chips. The behavior of Mary can be best described as: A. reasonable, B. not reasonable.” does not clarify who the behavior is reasonable to: Mary herself, or the external observer (we know that there are moldy chips). Such ambiguity may confuse models and affect their judgment.

The authors could examine whether such ambiguity affects performance and consider clearer or more balanced phrasing.

[1] Gandhi, Kanishk, et al. "Understanding social reasoning in language models with language models." Advances in Neural Information Processing Systems 36 (2023): 13518-13529.

### Questions
Could you compare the behavior prediction task with the forward action task in BigToM more clearly, since you claim 'applied ToM' is the main contribution of your benchmark?

Could you also explain why the behavior judgment task is needed and what it adds? How is it different from behavior prediction? Is it a new task you designed, or is it based on existing work in psychology / cognitive science?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates a gap in the existing Theory of Mind (ToM) dataset, that most of them focus on the mental state inference while few address the issue of applying the knowledge about the mental state into actions. Inspired by this, this paper constructs a dataset to test the propriety LLMs on how they can infer the mental states as well as how they can apply such inference / knowledge into their actions. The experimental results reveal that behavior predictions present a significant performance drop from mental state inference's prediction. In addition, simple test-time intervention does not seem to help the behavior predictions.

### Strengths
- The dataset addresses the gap in ToM research, as few existing datasets investigate the action in ToM space. This paper can serve as the first study in ToM on both the mental state inference and the behavior side study.

- The examples in the dataset are short and easy to evaluate. This can disentangle confounding factors such as the length, verbosity, etc, in ToM evaluation.

- The authors have conducted experiments on proprietary LLMs as well as some open-source LLMs. The results mostly justify their findings.

### Weaknesses
- It would be nice for the authors to include more open-source LLMs. Currently, the table 2 includes many proprietary LLMs (cheers to the authors). However, only Llama 3.1 8B is involved. I would like to see how performance changes along the spectrum of LLM scales and how large the gap is between the open-source LLMs versus the commercial black-boxed LLMs.

- In Table 2, the judgement scores for Llama 3.1 8B is 54.6, which is significantly higher than Llama 3.1 405B's 10.0 and the scores for the other LLMs. Though the authors seem to indicate (in line 353-line 354) that the model is performing randomly, therefore it achieves a random score. I am not convinced in several aspects. 

    - First, how can you ensure that there is no randomness in other LLMs' answers? 

    - Second, if randomly selecting a choice would lead to a much higher performance, shall we double-check the dataset to prevent such shortcuts? Do we have better ways to design the dataset to mitigate randomness or shortcuts?

    - Since there is only one 8B LLM (Llama 3.1 8B) tested, are other small (compared to larger commercial LLMs) LLMs behave similarly in terms of choosing randomly? What would be a threshold in terms of scales for LLMs to stop choosing randomly?

### Questions
Please see the weakness section.

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
The paper presents SimpleToM, a two-sentence story benchmark that separates explicit ToM from applied ToM. The dataset contains 1,147 stories across 10 everyday information-asymmetry scenarios, with strict 3/3 annotator agreement filtering; a human baseline covers 50 stories / 150 questions. On SimpleToM, models achieve near-ceiling mental-state accuracy but large drops on behavior and judgment; targeted test-time guidance (mental-state reminders and ToM-specific CoT) can push strong models to ~95–97% averages.

### Strengths
1. The three-stage evaluation (mental state → behavior → judgment) plus first-failure analysis offers diagnostic value. 
2. Two-round LLM generation, strict 3/3 agreement, and consistent retention rates across generators. 
3. Clear empirical finding. Large gap between explicit and applied ToM across many models, including reasoning/inference-time models (o1/DeepSeek-R1). 
4.  Mental-state reminders and CoT sharply improve applied ToM, indicating carry-over/cueing issues.

### Weaknesses
1. No confidence intervals, significance tests, or variance across seeds/prompts; “below random” is not visualized with chance bands. 
2. Only 50 stories / 150 Qs, limiting precise human–model comparisons and per-scenario variance. 
3. Binary task format may compress nuance. All tasks are binary; judgments especially could benefit from graded scales and rationales. (No direct evidence of graded/rationale analysis is provided.) 
4.  Paper includes o1/DeepSeek-R1 and GPT-5 in tables, but analysis does not deeply compare inference-time design choices or situate results against recent ToM benchmarks.

### Questions
1. Can you add 95% CIs and paired tests/bootstraps for Tables 2/3 and error bars for Figs. 3/4 to substantiate “below random” and scenario effects? 
2. Can the human baseline be expanded beyond 50 stories and reported per scenario with uncertainty? 
3. Do MS and CoT gains persist on paraphrases/held-out scenarios and across reasoning-token settings for o1/DeepSeek-R1? 
4. Could you incorporate graded judgments and short rationales to reduce ceiling/floor and probe value alignment? (Motivated by current binary drops.) 
5. Please situate the analysis more fully within inference-based advances (o1/o3, DeepSeek-R1) and recent ToM benchmarks, clarifying why judgment lags persist despite reasoning tokens.

### Soundness
3

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
5

### Summary
This paper presents SimpleToM, a benchmark that tests large language models’ Theory of Mind (ToM) abilities through 1,100 short, everyday scenarios involving natural information asymmetries. It evaluates three reasoning levels: mental state inference (explicit ToM), behavior prediction, and judgment of behavior (applied ToM). Results show that while advanced models like GPT-5 and Claude-3.5 excel at recognizing others’ beliefs, they struggle to use this understanding for predicting or judging actions. This highlights a key gap between knowing and applying social reasoning, suggesting that future AI work must focus on bridging this explicit–applied ToM divide.

### Strengths
- The paper makes a valuable distinction between explicit and applied Theory of Mind (ToM), highlighting an overlooked but critical aspect of social reasoning in LLMs.
- SimpleToM covers diverse, realistic scenarios with naturally occurring information asymmetries, improving ecological validity over previous ToM datasets.
- High data quality: The combination of LLM-assisted story generation and rigorous human annotation yields concise, well-controlled, and reliable test cases.
- Results across 16 frontier models provide a broad and insightful comparison of current ToM capabilities and their limitations.

### Weaknesses
- The benchmark’s tightly templated two-sentence stories and binary question formats (yes/no, two-option, or “reasonable/unreasonable”) may oversimplify the complexity of social reasoning. This constrained setup may produce below-random accuracies in judgment tasks for strong models, likely reflecting design artifacts rather than genuine reasoning deficits.
- The “judgment” task assumes a single normative ground truth derived from crowdworker consensus. Because moral or social appropriateness varies culturally and contextually, this approach embeds hidden biases that could distort conclusions about model morality or social reasoning.
- Many stories rely on repeated structural motifs (e.g., containers, hidden information) that may conflate Theory of Mind reasoning with world-knowledge heuristics, making it unclear whether models are failing at perspective-taking or at commonsense inference.
- The study’s interventions (prompt reminders, system messages, chain-of-thought prompts) are limited and not systematically ablated, so the paper’s claim that the explicit–applied ToM gap is robust across prompting strategies remains only partially supported

### Questions
- Have you tried the interventions on reasoning models?
- The paper analyzes “first failure” along the MS → behavior → judgment chain. Do you observe non-monotonic cases (e.g., behavior correct when MS is wrong), and if so, how frequent are such patterns per scenario?

### Soundness
3

### Presentation
3

### Contribution
3
