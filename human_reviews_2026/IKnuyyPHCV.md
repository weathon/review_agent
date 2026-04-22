# RECODE-H: A Benchmark for Research Code Development with Interactive Human Feedback

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Large language models (LLMs) show the promise in supporting scientific research implementation, yet their ability to generate correct and executable code remains limited. Existing works largely adopt one-shot settings, ignoring the iterative and feedback-driven nature of realistic workflows of scientific research development. To address this gap, we present RECODE-H, a benchmark of 102 tasks from research papers and repositories that evaluates LLMs through multi-turn interactions with human feedback. It includes structured instructions, unit tests, and a five-level feedback hierarchy to reflect realistic researcher–agent collaboration. We further present ReCodeAgent, a framework that integrates feedback into iterative code generation. Experimentswith leading LLMs, including GPT-5, Claude-Sonnet-4, DeepSeek-V3.1, and Gemini 2.5, show substantial performance gains with richer feedback, while also highlighting ongoing challenges in the generation of complex research code. RECODE-H establishes a foundation for developing adaptive, feedback-driven LLM agents in scientific research implementation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors present a new benchmark called RECODE, which is a set of 102 code execution tasks from research papers and their respective repositories. Their dataset is built via a clever combination of expert human annotation effort (using PhD-level experts to select the target code and repositories and provide code generation instructions) and LLM boostrapping using LLMs (e.g., for helping to annotate code explanations and construct on-the-fly unit tests). The particular focus of their dataset is on explicitly modeling multi-turn coding (intuitively, they say that the benchmark aims to capture standard programming where developers "interatively refine implementations through cycles of execution, debugging and feedback") and on testing directly the effect of different forms of feedback using a novel feedback hierarchy that they define in Section 3.2. This benchmark looks similar to those they cite in the RelatedWork, as well as some they don't cite, notably the work below (where the landmarks annotations appear to be similar in spirit to the types of feedback they collect):
 
[Bogin et al.] SUPER: Evaluating Agents on Setting Up and Executing Tasks from Research Repositories. EMNLP 2024

Most uniquely, however, their problems are annotated with different levels of feedback (generated using LLMs) using the feedback hierarchy from Section 3.2, and a set of concrete unit tests that allow them to measure functional code correctness. 

They couple their benchmark with a new agent design called ReCodeAgent. Based on how this agent is described, both in Figure 1 as well as in the begining of Section 4, is it really unclear how this agent is nothing more than a ReACT agent or a closely related variant such as a Reflexion agent. Indeed, their earlier claim (starting on line 035) that "existing benchmarks .. for evaluating LLMs in research code generation mainly adopt a one-shot setting, where models are expected to produce final code in a single interaction* simply seems incorrect given that the models in virtually all of the studies they cite involve ReACT agents, much like ReCodeAgent, that engage in precisely the kind of act-observe loop they show visually in Figure 1. This point needs to be directly addressed by the authors and is the main source of my  concern about this paper. 

Their main empirical results are show across seven LLMs (including 4 LLM model familities) in Table 2, where they also carefully report the performance effect of different types of feedback (not surprisingly, the most detailed feedback, level 4, is clearly the most hepful in improving end task performance).  This is coupled with other fairly expected conclusions, e.g., (citing the authors) *Model size and capability play a clear role in performance*. Further error analyis is provided (Table 3).

### Strengths
-- A new research coding benchmark with explicit feedback annotations that allow for more granular analysis of multi-turn coding. I could imagine this benchmark being used by others working in this area.

### Weaknesses
-- **Misleading motivation and discussion of past work**. As noted above, claims like "existing benchmarks for evaluating LLMs in research code generation mainly adopt a one-shot setting, where models are expected to produce final code in a single interaction" and "the ability of [LLMs] to generate correct and execute code remains limited" (line 011) seems inconsistent with virtually all of the papers cited. In the latter case, most studies involve REACT agents that are by design multi-turn agents. I would like to see the authors directly address. 

--  **Limited to No Novelty of their ReCodeAgent** As noted above, their proposed solution, and their sole modeling approach, seems to be nothing more than a ReACT or Reflexion agent. 

-- **Limited empirical validation** Experiments are limited only to their new dataset. Especially if they claim that their coding agent is unique, it would be expected that this approach shows improvements on other tasks.

### Questions
-- In what way is the agent workflow in Figure 1 (top left) not standard ReACT?

-- If it is different from ReACT, did you try to compare against a standard ReACT or relfexion approach?  Or compare your approach on other benchmarks?

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
This paper presents RECODE, a benchmark of 102 research coding tasks in AI/ML to study the interactions between LLM-based agents and user feedback. The user feedbacks are simulated with LLMs and controlled to have five different granularity levels of information. Experimental results show that LLMs benefit from additional feedback, especially straightforward ones, but still struggle with those requiring deeper understanding of the research tasks.

### Strengths
1. The paper adds a nice new dimension around user feedback to the evaluation of agents for research coding, which is helpful and adheres to the real-world use cases of such agents.
2. The paper presents a reasonable amount of analysis of experimental results. The error analysis and feedback adoption analysis are interesting and may facilitate future research.

### Weaknesses
1. While it is appreciated that the authors assembled a team of 26 annotators, their roles in the entire annotation process are not very clear to me. It seems that LLMs (Gemini 2.5 Pro and GPT-4o-mini) are used to perform many annotations. What are the jobs of the annotators and how their involvement ensures the benchmark’s quality and real-world utility?
2. Relatedly, the quality of LLM-generated unit tests is unclear and not thoroughly discussed in the paper, which is critical to the reliability of the experimental results. It is also unclear to me how the “unit tests” can be leveraged to evaluate some tasks, such as training machine learning models or analyzing data. 
3. The authors should list and cite all the papers adapted. Meanwhile, discussions and tests of data contamination are also missing.
4. The validity of “level 4” feedback, which provides ground truth code, may not be a valid setting since it directly “supervises” the code generation process and plays an exceptionally helpful role for most models. The evaluations would be more sound and clean by just stopping at “level 3.”
5. Some example tasks in the appendix would be appreciated to help the benchmark description be more grounded.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

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
The authors propose RECODE, a new benchmark for generating research code in the novel setting of multi-turn interactions with a (simulated) human, which iteratively provides hints/details/corrections to help steer the system. They show that interactive feedback helps, while also highlighting that models still struggle.

### Strengths
- Introducing the interactive multi-turn angle is novel and refreshing, and adds some extra realism to the coding challenge
 - novel framing, with levels of feedback
 - benchmark looks like a useful product of a lot of hard work
 - evaluation is thorough

### Weaknesses
- The interactive feedback is (for reproducibility) actually simulated so has a degree of artificality associated with it (even though humans helped create it). In what ways does this setup differ from a (truely) real setup with actual humans, e.g., noise, imprecision, etc. Could you include any of those elements in your framework?
 - The conclusions from the experiments seem obvious (e.g., more feedback helps). What did you learn that was surprising/interesting/informative? It seems to me that should be more nuanced findings. The error analysis is perhaps more informative. If you were to advise future researchers on where to invest their energy in building better coding agents, what would you tell them based on the learnings from your work?

Minor:
 - "evaluates LLMs through multi-turn interacations with human feedback" - makes it sound like there's a human in the eval loop. Perhaps "(pre-collected) human feedback" or "(simulated) human feedback" or something else to indicate the actual eval is fully automated.
 - Figure 2 would be more readable showing % rather than fractionals (e.g., "1.4" rather than "0.014" etc.)

### Questions
See weaknesses. Also:
 - Isn't level 4 feedback basically giving the system the answer? Why doesn't the coding agents score 100% as a result? Adding some discussion around this would be interesting, in particular that "coding" requires more than just knowing lines of code. What would you need to include in a "level 5" feedback to ensure the coding agents did score 100%?
 - Do your results suggest any advice for what *kinds* of feedback people should be giving to their coding agents, i.e., provide insights that make interactive coding agents more usable/effective?
  - The Conclusion seems pretty weak, surely there's more to conclude than just that LLMs continue to face coding challenges? What are the big insights you found?
   - Building/exanding this benchmark looks very labor intensive. Can you think of ways you might reduce the cost / semi-automate the process to extend it further?

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
4

### Summary
This paper presents RECODE, a benchmark for 102 code-generation tasks, based on LaTeX snippets from recent ML papers. They introduce a ReAct-style multi-turn ReCodeAgent that interacts with a code repo and receives feedback from a simulated expert (similar to LLM as a judge). The authors find that richer feedback substantially improves performance, though with diminishing returns. The paper also analyzes error types and how effectively models adopt feedback.

### Strengths
1. The empirical findings are valuable to broader coding agents: early feedback levels yield large improvements, while higher levels show smaller gains; and that most model failures stem from misunderstanding the paper or repo semantics.
2. The benchmark targets realistic, interactive research-coding scenarios, which are increasingly relevant as LLM agents enter scientific domains.
3. Solid engineering effort, clear and intuitive definitions and hierarchies of feedback

### Weaknesses
1. It is not clear how the papers were selected and the list of papers is not disclosed. How do you ensure that evaluated LLMs haven’t already seen the source repositories during pretraining?
2. There is no code editing tool in the system prompt. The replace tool replaces the entire file content, but the prompt mentions "aim for editing the code more" -- it is not clear how this is implemented.
2. LLM-generated feedback is clean and complete; real human feedback is often noisy or partial. But results with human-generated feedback on these tasks are missing.
2. Prior benchmarks like SciReplicate-bench and ResearchCodeBench already explore research-code generation; the paper needs a clearer statement of what’s uniquely new here.

### Questions
See above weaknesses.
I am curious if you had results with a single-turn format. It would be interesting to see if the multi-turn format gives a big lift.
Minor typos/issues: Fig 7 and 8 have same captions

### Soundness
3

### Presentation
2

### Contribution
3
