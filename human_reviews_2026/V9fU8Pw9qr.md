# WAFER-QA: Evaluating Vulnerabilities of Agentic Workflows with Agent-as-Judge

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Agentic workflows—where multiple large language model (LLM) instances interact to solve tasks—are increasingly built on feedback mechanisms, where one model evaluates and critiques another.
Despite the promise of feedback-driven improvement, the stability of agentic workflows rests on the reliability of the judge. However, judges may hallucinate information, exhibit bias, or act adversarially—introducing critical vulnerabilities into the workflow.
In this work, we present a systematic analysis of agentic workflows under deceptive or misleading feedback. 
We introduce a two-dimensional framework for analyzing judge behavior, along axes of intent (from constructive to malicious) and knowledge (from parametric-only to retrieval-augmented systems).
Using this taxonomy, we construct a suite of judge behaviors and develop WAFER-QA, a new benchmark with critiques grounded in retrieved web evidence to evaluate robustness of agentic workflows against factually supported adversarial feedback. We reveal that even strongest agents are vulnerable to persuasive yet flawed critiques—often switching correct answers after receiving misleading feedback. 
Taking a step further, we study 
how model predictions evolve over multiple rounds of interaction, revealing distinct behavioral patterns between reasoning and non-reasoning models.
Our findings highlight fundamental vulnerabilities in feedback-based workflows and offer guidance for building more robust agentic systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates vulnerabilities in feedback-based agentic workflows, where one or more LLMs act as judges providing critiques or evaluations to other models. The authors introduce a two-dimensional framework that characterizes judge behavior along intent (constructive → malicious) and knowledge access (parametric → grounded). Building on this, they propose WAFER-QA, a benchmark that augments QA datasets with web-retrieved “alternative” evidence supporting incorrect but plausible answers. Through experiments across multiple LLMs and workflows (generator–evaluator, round-table discussion, moderator setups), the authors show that even the strongest models (e.g., GPT-4o, o4-mini) are highly susceptible to deceptive or adversarial feedback, with large accuracy drops. They further analyze multi-round feedback and multi-agent discussions, highlighting oscillatory behaviors and partial robustness gains from reasoning models or moderator agents.

### Strengths
The experimental design is comprehensive and systematic, with evaluations across different models, judge types, and task settings. The results are consistent and well-presented, revealing interesting behavioral distinctions between reasoning and non-reasoning models. The multi-round and multi-agent analyses are particularly strong, showing that reasoning models exhibit greater stability across iterations, and that additional normal agents in a discussion can partially mitigate the influence of deceptive participants. Some good points - 

- Timely and relevant topic addressing reliability in multi-agent workflows.
- Comprehensive experimental coverage: parametric vs. grounded judges, reasoning vs. non-reasoning models, and single- vs. multi-agent setups.
- Sections 5.3 and 6 are particularly valuable: they provide agent-specific insights showing (a) multi-agent interactions can dampen deception, and (b) reasoning models are more stable across multi-round feedback.
- Clear, reproducible presentation of results with consistent quantitative reporting.

### Weaknesses
- Limited novelty :  the main findings extend well-known LLM vulnerabilities (knowledge conflict, sycophancy, adversarial context) into an agentic framing, without introducing new underlying mechanisms.
- Benchmark reliability : the WAFER-QA construction lacks clear human validation that “alternative” answers are actually incorrect. Some questions may have multiple valid interpretations, making the measured vulnerability ambiguous.
- No confidence or calibration analysis : in agentic settings, an agent’s susceptibility to external critique should strongly depend on its internal confidence. If confident generators resist incorrect feedback while uncertain ones flip, that provides causal understanding and an actionable defense. However, the paper never quantifies this relationship or reports how calibration correlates with robustness.
- Shallow multi-agent analysis : Section 5.3 is promising but largely descriptive. There is no deeper causal study of who influences whom, how consensus evolves, or whether the majority of normal agents consistently stabilize decisions. Understanding these dynamics is central to robustness in collaborative agents.
- No mechanistic explanation of vulnerability : the paper convincingly shows that models ‘flip’ under persuasive judges but doesn’t probe why. It’s unclear whether failures stem. 

Overall, while empirically strong, the paper stops short of deeper agentic-level insights that could make the results more explanatory or predictive.

### Questions
1. Benchmark reliability:
How do you make sure that the “alternative answers” in WAFER-QA are actually wrong? Some questions might have more than one valid answer. Did you check this with human annotation or any validation process?

2. Multi-agent dynamics:
The multi-agent study (Section 5.3) is interesting but mostly descriptive. Could you show which agents tend to influence others, how agreement is reached, or whether having more normal agents always helps stabilize the outcome?

3. Why models flip:
It would be helpful to understand why the models change their answers after receiving feedback. Is it because of wording overlap, trust in citations, or reasoning failures? Some controlled ablations could make this clearer.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the vulnerabilities of agentic LLM workflows, specifically focusing on systems that use a "judge" agent to provide feedback to a "generator" agent. The authors propose a two-dimensional framework for categorizing judge behavior based on intent (constructive to malicious) and knowledge (parametric-only to retrieval-augmented). The core contribution is a new benchmark, WAFER-QA, which evaluates an agent's robustness against deceptive feedback that is grounded in retrieved web evidence supporting plausible but incorrect answers. Experiments with several SOTA LLMs (e.g., GPT-4o, o3-mini, o4-mini) demonstrate that these models are highly susceptible to deceptive feedback, especially when it is backed by factual-sounding (even if fabricated) or genuinely retrieved web evidence.

### Strengths
The paper tackles a critical and timely issue. As feedback-based agentic workflows become more common, understanding their vulnerabilities to unreliable or malicious feedback is essential. The two-dimensional taxonomy of intent and knowledge is a key strength, providing a clear and extensible framework for analyzing and generating diverse judge behaviors.  The paper provides some insights, such as the distinction in robustness between reasoning-focused models (o4-mini) and other models (GPT-4o, Qwen), and the finding that models struggle to acknowledge ambiguity even when prompted.

### Weaknesses
I think the WAFER-QA construction method is clever, but its dependence on finding questions that already have some plausible web evidence for an alternative answer might make it less general. The paper even mentions that this approach is “infeasible for factually well-settled queries.” That makes me wonder how representative this benchmark really is of the kinds of problems agents might face in the real world—especially those that have one clear, unambiguous truth.

The main experiments use the same model as both the generator and the judge. That’s a common setup, but it doesn’t quite match real-world situations where agents built by different teams or using different base models interact. The appendix briefly looks at an asymmetric setup (a stronger judge and a weaker generator), but I wish there were a deeper exploration of how heterogeneous agents—ones with different knowledge bases—would behave.

The paper mainly focuses on showing the vulnerability. In Section 6.2, the authors try a “moderator” agent as a possible fix, but it only works partly, and that part of the analysis feels underdeveloped. Overall, the paper does a good job of highlighting the problem, but it doesn’t go very far in offering solid solutions, other than noting that reasoning-trained models tend to be more resilient.

### Questions
see weakness.

### Soundness
3

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
This paper focuses on evaluating LLMs ability to provide feedback to other models in a debate-style setting. In particular, the paper proposes a new benchmark, WAFER-QA, that allows evaluating judges with web-search tool-use access in adversarial settings. Their benchmark builds on a framework introduced by the authors that aims to disentangle judge intent (constructive/hypercritical/malicious) from judge knowledge (parametric/grounded). The authors report the results of models on multiple question-answering benchmarks with the various judge setups. The authors analyse a well-selected set of models.

### Strengths
1. **Well-written.** This paper is well and clearly written, overall pleasant to read through. The tables and figures are well constructed and easy to read.
2. **Well-selected experimental data.** Given the QA setting, the authors select a number of well-known and -used datasets as the basis for their new benchmark (.
3. **Extensive discussion of experimental results.** The paper includes an extensive discussion of their diverse experimental results. It's a nice read.

### Weaknesses
1. **Lack of mean/variance statistics across re-runs.** Currently the experiments appear to use a single set of observation for each metric. Would be interesting to see how high the variance of each of these metrics is, even just 3 seeds would provide quite a bit of additional context here. In particular, since most metrics are based on multi-turn interactions, also evaluating variance along a point estimate would be very helpful. Also, sampling/inference parameters appear to be not discussed (e.g. temperature).
2. **Focus on simple Q&A tasks.** The paper currently focuses on simple question answering tasks. Whilst I see the value of keeping the scenarios simple for practical reasons, it means that the results may not extend to scenarios that are more similar to realistic real-world (complex agentic tasks, such as coding or web interactions).
3. **There could be stronger/clearer motivation for the threat scenario.** I understand that judges can play an important role in debate setups but it remains somewhat unclear how exactly a malicious judge might arise or enter the picture. This part of the motivating scenario could be further explored/discussed. Currently, it is simply assumed that such judges exist and relied on the reader to motivate this for themselves.
4. **Judge knowledge dimension not considered in benchmark.** As far as I read it, the later part of the paper (introducing the WaferQA benchmark) appears to focus far more on the judge intent perspective rather than the judge knowledge part. None of the tables or figures in the main body actually vary the judge knowledge dimension. Nevertheless, this knowledge dimension is one of the two introduced earlier - this makes the experiments and earlier sections feel disconnected.

Minor (no impact on score, no need to respond):
1. Use of \citet citations are sometimes used instead of \citep (e.g. L34,L40). This makes some sections more difficult to read. Notably the related work section is not affected by this issue.
2. L108: Table one, font feels unnecessarily small
3. L249: citing "Team", though this should be "Gemma Team". "Team" is not a last name here, confusing and needs to be adjusted in bib file, e.g. by adding curly brackets {}.
4. L232: terms contextual vs non-contextual QA should be clarified/defined

### Questions
1. L343: How exactly is the acknowledgement detected and the corresponding acknowledgement rate computed? How do you detect whether a model "explicitly signals the presence of ambiguity"? Is this LLM-as-a-Judge? And, are the tasks up-front formulated in such a way that ambiguity is allowed?
2. Do you have an intuition how robust the benchmarks scores are under different random seeds (related to weakness above)?
3. Since you use such well-known benchmarks, do you think that the results might change if the underlying dataset was more "fresh", less likely to have (indirectly) leaked into the models' training data?
4. Would you be able to clarify how the knowledge dimension relates to the experiments? If it does not, would you be able to clarify why it is necessary to discuss in the taxonomy section (3).

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors present a dataset for evaluating the robustness of a system to a deceptive or hallucinating Agent-as-a-Judge. They find that existing systems are very vulnerable to such problematic judges.

### Strengths
- This is a very timely topic. With the growth in these kinds of judges and their unreliability, understanding how much a system can resist an unreliable judge is growing increasingly important.
- In line with that, a new benchmark is always appreciated. We're seeing models gaming benchmarks all the time now, so any new metrics we can use to help correct for this are very useful right now.

### Weaknesses
- This paper is all over the place. It seems like the two axes they are looking at are quite unrelated. So then, why these two axes? Is there some pattern in the literature? I would have expected a strong reason why whether a judge has database access and whether a judge is deceptive or helpful are the two axes used here. There seem to be many other characteristics a judge might have. I feel like the database access of a judge seems completely unrelated to the title of the work. The authors should focus on one and include the latter as a secondary consideration (this could be accomplished quite easily, actually).
- One that note, the title is bad. WAFER-QA is a dataset, so why does the title make it sound like a method? Also, Agent-as-Judge seems grammatically incorrect. I believe it should be Agent-as-a-Judge, like with LLM-as-a-Judge.
- They have Agent-as-Judge in their title and then cite a paper that includes Agent-as-a-Judge in the title. But they only mention it offhandedly. From my understanding, this paper is proposing an agentic judge---a direct instantiation of what the Agent-as-a-Judge work proposes, but then only mentions it in passing as a "constructive judge and/or adversarial judge without web
access"? If this is something I noticed, I worry what this says about the other papers they cite that I haven't read recently.
- There are a few statements that seem quite odd. For example: `For example, in response to the question “What is the capital of France in 2025?”, no credible web evidence exists to support any answer other than Paris, making web-based retrieval infeasible for factually well-settled queries.` But there are most definitely many things on the web that say otherwise (even though it may be incorrect). Do the authors mean to say that some questions will have an overwhelming quantity of web evidence leading to a particular conclusion and much less evidence proposing an alternative, as opposed to other facts where the evidence is more ambiguous (such as relating to a certain cooking technique being superior to another)? If so, they should be clearer about this and defend it. This also means "plausible supporting evidence" needs a more rigorous definition. Otherwise, it is very ambiguous what was or was not included.
- I think the above makes this not particularly useful to the field without quite a bit of a cleanup. It seems it needs to focus on the deceptive axis and include the web usage as a side quality being evaluated (or vice versa). Otherwise, it's trying to do two things at once.
- I understand that the above is quite challenging to meet in the timeline ICLR gives. However, if the above could all be addressed to a reasonable degree, I'm not opposed to changing my recommendation as I see the potential here.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
3
