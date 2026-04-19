# LongGenBench: Benchmarking Long-Form Generation in Long Context LLMs

- Decision: Accept (Poster)
- Scores: 8, 3, 5, 8, 8

## Abstract
Current benchmarks like ``$\textit{Needle-in-a-Haystack}$'' ($\textit{NIAH}$), $\textit{Ruler}$, and $\textit{Needlebench}$ focus on models' ability to understand long-context input sequences but fail to capture a critical dimension: the generation of high-quality long-form text. Applications such as design proposals, technical documentation, and creative writing rely on coherent, instruction-following outputs over extended sequences—a challenge that existing benchmarks do not adequately address. To fill this gap, we introduce $\textit{LongGenBench}$, a novel benchmark designed to rigorously evaluate large language models' (LLMs) ability to generate long text while adhering to complex instructions. Through tasks requiring specific events or constraints within generated text, $\textit{LongGenBench}$ evaluates model performance across four distinct scenarios, three instruction types, and two generation-lengths (16K and 32K tokens). Our evaluation of ten state-of-the-art LLMs reveals that, despite strong results on $\textit{Ruler}$, all models struggled with long text generation on $\textit{LongGenBench}$, particularly as text length increased. This suggests that current LLMs are not yet equipped to meet the demands of real-world, long-form text generation. We open-source $\textit{LongGenBench}$ to promote comprehensive evaluation and improvement in this critical area, with code and data available at ${anonymousurl}$.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes a new benchmark for long-form generation where the model-generated content, rather than the input context, is long. Specifically, the model is asked to roll out a detailed description of certain tasks such as diary over a year or floor plan for a skyscraper, subject to a few specific instructions that can be either singular or recurrent. Evaluation metrics include main task completion to test whether generation follows the expected format, as well as the success rate for specific instructions at both micro and macro level. Results demonstrate that the proposed benchmark is much more challenging than needle-in-the-haystack tasks for long context evaluation. Models generally do not perform very well within their claimed context lengths.

### Strengths
* The paper is the first to study long-form generation as opposed to long-context generation. The perspective is novel, interesting, and of practical value to unleash the potential of LLMs for more complicated tasks.
* Problems in the benchmark are constructed in a sound and intuitive way. While evaluating the quality of long text snippets is usually challenging and complex, the smart design in this paper enables reliable and accurate evaluation for long-form capability.

### Weaknesses
One of my major concerns is the clarity of writing in the evaluation part.
* The definition of STIC-1/STIC-2 isn't quite clear. Using the notation $T_S=(T_{S_1}, T_{S_2}, \dots)$ in Sec 2.3, my best guess is that STIC-1 means the average success rate over $(T_{S_1}, T_{S_2}, \dots)$, while STIC-2 counts the entire $T_S$ as successful only if all $(T_{S_1}, T_{S_2}, \dots)$ are successful, and gives 0 score otherwise.
* The abbreviation **CR** in Table3 isn't defined anywhere, though I can guess this is likely the Completion Rate in Main Task Completion.
* The terms "main task", "subtask",  "instruction task", "specific task" are used in a confusing way. It would be very helpful to unify them with clear definitions, and to associate the terms to symbols like $T_S$ or $T_{S_1}$.
* Missing x-ticks in Figure 2.

Apart that, there are a few technical details that are unclear to me.
* How do you determine whether the generated text at one specific point satisfies all task requirements? For example, given the generated diary of a particular day, how do you verify that all required activities (e.g. wedding, vacation, and etc.) are covered? I would imagine that a natural language inference (NLI) module should be involved but it's not mentioned in the paper.
* In Table 3, though the length to be evaluated at is 16K/32K respectively, the actual generation length seems only around half of the max length. How are the 16K/32K defined?

### Questions
* In L460-468, it is mentioned that there are significant repetitions in long-form generations. Are these repetitions correct with respect to the given instructions, or are they semantically wrong (i.e. violating the given instructions)? The former indicates that it's probably caused by how instructions in the prompt are designed, while the latter means that model struggles at producing long and meaningful generation.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes a benchmark to evaluate models' strength in long-span generation tasks. It constructs synthetic examples from a fixed set of scenarios and templates in three modes. The resultant instruction measures models' abilities to faithfully follow position-specific, range-specific, and periodic instructions.

### Strengths
1. The paper is the first attempt at long-context generation, requiring models to generate a long text that follows a combination of specific instructions as opposed to just answering questions pertaining to long prompts.
2. The benchmark does uncover a setting that seems to be challenging to SOTA models.

### Weaknesses
1. The paper is very sparse on details regarding the evaluation of correctness. How is matching and parsing done w.r.t. the templates? A model could generate several outputs matching the criteria
2. The benchmark is limited to a few domains and scenarios, and the paper's contributions seem quite limited overall

### Questions
Would an IFEval [1] style setting where the outputs or attributes of the outputs could be verified definitively using code checks be a better option for such a benchmark? For long generations, getting the model outputs to match specific output patterns in the prompt is a challenge unto itself.

[1] https://arxiv.org/abs/2311.07911

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces LongGenBench, a benchmark for evaluating large language models' (LLMs) ability to generate long-form text while adhering to complex instructions. It features tasks in four scenarios with different instruction types and lengths (16K and 32K tokens). The study finds that even advanced LLMs struggle with long-form generation, particularly as text length increases, highlighting the need for improvements in model architecture and training.

### Strengths
1. Interesting task design, which can evaluate the long text generation ability of large models from a certain perspective
2. The paper is well written.

### Weaknesses
1. The types of task scenarios are relatively limited, and it is impossible to comprehensively evaluate the long text generation capabilities of large models.
2. The evaluation metrics seem to be customized according to the scenario.
3. Limited number of models evaluated

### Questions
1. It seems that the evaluation metrics are designed for these scenarios. If there are new scenario tasks, do we need to update the evaluation metrics?
2. What other aspects of long text generation with LLM do you think need to be evaluated? It seems that your evaluation is more oriented towards some planning tasks or well-structured text. Is it difficult to evaluate the creation task? For example, it is difficult to design evaluation metrics for novel writing?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Existing LLMs' long-context benchmarks rely on understanding tasks. This paper proposes a benchmark targeting specifically on long-context *generation* ability of LLMs. The authors design 4 long-context instruction-following tasks, up to 16 or 32K tokens: (1) Writing Diary for a year; (2) Wrting menu for a year; (3) Design a 100/300-floor skyscraper; (4) Plan an urban layout. As a result, all existing LLMs don't work well on these tasks of the benchmark.

### Strengths
* This paper is overall clear and easy to understand.
* This proposed evaluation is novel, and the generation ability it benchmarks is not covered by previous metrics.

### Weaknesses
* Some of the details are possibly missing or hard to get by readers -- see "Questions".
* In the proposed benchmark, the way to form long content is to pile short answers to many sub-queries, while the sub-tasks are actually independent, to a large extent. For example, given all the demands on one-year dairies, it should be easy for LLMs to write a diary if it is assigned a specific day of the year, while this benchmark just require the LLM generate 365 diaries all at once. In this case, the challenge of this benchmark might majorly be forgetting the instruction under the influence of generated content, instead of keeping the conherence and content interaction among generated long content. That latter should be the one mostly desired by the community.

### Questions
* How do you split the long generation and match them to all the subtask instructions?
* How do you check if every sub-instruction is satisfied? is it by prompting another LLM or by word matching, etc.?
* What's the significant difference between STIC-1 and STIC-2? Looks they just have difference denominators. I don't quite get it although there is a paragraph in Sec. 2.4 as below for this. Is there any specific case where an LLM can get low STIC-1 while high STIC-2, or the other way around?

> STIC-1 is primarily concerned with the completion rate of instructions that result in sub-scenarios,
focusing on whether instructions are correctly executed. In contrast, STIC-2 assesses the overall completion of the specific instruction task, including the presence of sub-scenarios and their completion
status.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces LongGenBench, a benchmark for measuring LLMs' capacities, especially their long-context abilities, by generating long-form context from 16k to 32k with rather complex instructions. This new dataset departs from traditional benchmarks aiming at decoded length in four different scenarios. Preliminary evaluations are done with main streamed LLMs.

### Strengths
- First benchmark focusing on long-form generation during the test time
- The evaluation combines both complexities of evaluation prompts and different scenarios
- First batch of results on 10 mainstreamed LLMs
- The paper is easy to follow

### Weaknesses
- I am a little bit distracted from the main takeaways from the experimental studies, and not so convinced with failure cases. See question 1

I have other minor concerns regarding the experiment setup

- There has been much research showing that the prompt format matters, what's your thought? 

- Reasoning tasks are not well involved, as o1 seems to argue that longer decoded length is helpful with reasoning complex tasks, in your benchmark, you might want to add an axis of reasoning ability clearly or have some analysis around this topic?

- Most of the evaluation focused on existing transformer-based architecture, but models are presenting SOTA results, for example, mamba-based models. Are those models, with good inference time complexity, good at benchmark, or if not, why?

### Questions
- question 1: can you present a bit more concise takeaways from your benchmark, to me I feel like I was reading a lot of pieces, and no surprising results to me either. It might be good to have some failure cases to support your point

-  question 2: when you evaluate the prompt complexity, how do you choose the prompt formats?

-  question 3: Do you think complex prompts and instructions might need manyshots?

-  question 4: Do you think you can add some reasoning axes to your benchmark? 

- question 5: maybe consider adding some long-context recent models with SOTA results, not only looking at model parameter counts but also architectural differences.

### Soundness
3

### Presentation
3

### Contribution
2
