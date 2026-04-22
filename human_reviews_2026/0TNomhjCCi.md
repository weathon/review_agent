# How Brittle is Agent Safety? Rethinking Agent Risk under Intent Concealment and Task Complexity

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
Current safety evaluations for LLM-driven agents primarily focus on atomic harms, failing to address sophisticated threats where malicious intent is concealed or diluted within complex tasks. 
We address this gap with a two-dimensional analysis of agent safety brittleness under the orthogonal pressures of intent concealment and task complexity. To enable this, we introduce OASIS (Orthogonal Agent Safety Inquiry Suite), a hierarchical benchmark with fine-grained annotations and a high-fidelity simulation sandbox. Our findings reveal two critical phenomena: safety alignment degrades sharply and predictably as intent becomes obscured, and a “Complexity Paradox” emerges, where agents seem safer on harder tasks only due to capability limitations. By releasing OASIS and its simulation environment, we provide a principled foundation for probing and strengthening agent safety in these overlooked dimensions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the "brittleness" of LLM agent safety. It argues that current tests are too simple and focus on direct, obvious harms. The authors propose a new benchmark called OASIS, which evaluates agents along two axes: "intent concealment" (how hidden the bad goal is) and "task complexity" (how many steps are in the task) . They test several models and find that agent safety gets much worse when the intent is hidden. They also find a "Complexity Paradox," where agents seem safer on very hard tasks, but only because they fail the task before they can do the harmful part.

### Strengths
I agree that we would need a more sophisticated benchmark to evaluate the AI agent safety issue. I think the idea of testing safety on these two dimensions of concealment and complexity is a good direction. The paper also has some interesting findings, like the "Complexity Paradox" and the fact that some models are "static" in their safety checks while others are "dynamic".

### Weaknesses
First, the paper says agents seem safer on complex tasks, but it could be due to their "planning capabilities" failing them, and they can't complete the task. Therefore,  how would the authors distinguish the results from to be a test of capability and a new insight about safety alignment?

Second, the way the benchmark (OASIS) was created seems a bit circular. The paper says the tasks were "synthesized using Gemini 2.5 Pro" and then validated by humans. Could the authors provide more information about who the annotators are? And what does the annotation process look like? Without those details, it's hard to argue whether the benchmark reflects "real-human" threats.

Third, I think the paper could benefit from a more thorough literature review. The idea of "intent concealment" in a multi-step task is explored in previous works (e.g., https://openreview.net/forum?id=KI1WQ6rLiy).

### Questions
see weakness

### Soundness
3

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
2

### Summary
This paper argues that current safety evaluations for LLM agents, which focus on "atomic harms," are insufficient. It proposes that agent safety must be evaluated along two orthogonal dimensions: "Intent Concealment" (obscuring malicious goals within benign narratives) and "Task Complexity" (diluting harmful steps within long, multi-step workflows).

To enable this, the authors introduce OASIS, a new hierarchical benchmark and stateful simulation sandbox. Through experiments on a suite of SOTA models, the paper presents several key findings:

Safety alignment degrades sharply and predictably as intent concealment increases. A "Complexity Paradox" emerges, where agents appear safer on more complex tasks; the authors demonstrate this is an illusion caused by capability limitations (i.e., the agent fails the task) rather than improved safety reasoning. The paper also identifies heterogeneous safety mechanisms, finding that most models rely on "static, pre-execution" checks (which are brittle), whereas the GPT-5 family, for example, exhibits a more robust "dynamic, in-workflow" monitoring.

### Strengths
- The paper's primary strength is its novel problem formulation. By shifting the focus from "atomic harms" to the more realistic, orthogonal dimensions of "intent concealment" and "task complexity," it reveals a critical and overlooked gap in safety research.
- The paper goes beyond reporting simple refusal rates. The discovery and classification of "static, pre-execution" vs. "dynamic, in-workflow" safety mechanisms is a key insight into *how* safety systems fail.

### Weaknesses
- While acknowledged by the authors, the curated set of 53 general-purpose tools is a limitation. Real-world agents will need to interact with thousands of dynamic, heterogeneous, and evolving third-party APIs. It is unclear how these findings (especially the "Complexity Paradox") will scale when the complexity of tool use itself.
- While the sandbox is described as "high-fidelity," the tasks are ultimately synthetic, and the tool outputs are pre-synthesized. This means the agent cannot elicit new harmful information from a "live" tool.

### Questions
The Harm Progression Score (HPS) measures the proportion of harmful steps executed, which implies all harmful steps are weighted equally. In a multi-step plan, the severity of harm seems non-linear (e.g., "emailing to purchase" a harmful item seems more severe than "searching" for it). Could the authors comment on this?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces OASIS, a benchmark and simulation framework for evaluating LLM-agent safety under varying levels of intent concealment and task complexity. The authors show that safety alignment degrades sharply when malicious intent is hidden and uncover a “Complexity Paradox,” where agents appear safer on harder tasks due to capability limits. They also distinguish performance between pre-execution and in-workflow safety mechanisms.

### Strengths
- The authors show how intent concealment and task complexity interact to influence the safety performance of language-model agents. They highlight the “complexity paradox” where the observation that agents may appear *safer* in more complex scenarios simply because they fail to act, reflecting capability limitations rather than genuine safety awareness.
- The paper introduces a diagnostic framework that evaluates both process and outcome through diverse metrics such as the *Hierarchical Refusal Rate* and *Harm Progression Score*. They also show the discrepancy between pre-execution and post-execution safety judgments, providing valuable insight into how agents handle dynamic risk.

### Weaknesses
The most critical weakness of this paper lies in its lack of clear explanations, definitions, and transparency. Many descriptions of the experimental setup are vague or informal—closer in tone to a blog post than an academic paper. As a result, the work falls short of reproducibility standards: it is difficult to fully understand the authors’ design decisions or replicate their experiments. In particular, no concrete examples of datasets, task instances, or tool usage are provided, which further limits interpretability. Please read my questions below.

### Questions
- **Line 212:** What exactly constitutes the “ground-truth plan”? Is this plan provided as an input to the model, or is it generated by the authors as a reference?
- **GPT-5-mini’s FPR:** Why is the false-positive rate notably high only in the *Idealized* scenario? Intuitively, less complex conditions should yield better performance according to the paper’s claims.
- **Qwen3 reasoning traces:** What do these traces actually show? Do models acknowledge safety risks but proceed anyway, or do they omit mention of them entirely? The lack of detailed qualitative analysis limits interpretability.
- **Tool selection:** How were the *53 tools* chosen? What are they specifically? For example, why were utilities such as `port_scanner` and `get_crypto_price` included? Please clarify the selection criteria and execution process, ideally in an appendix.
- **Code availability:** Why is the submitted code not accessible? Transparency here is crucial for validation.
- **Concealment levels:** Could the authors provide one example per *concealment* level? The annotation procedure for both concealment and complexity levels remains underexplained.
- **Post-execution refusals:** How are these judged? What are the precise inputs and outputs for evaluation?
- **Terminology consistency:** The paper uses *static* and *pre-execution* interchangeably, as well as *dynamic* and *in-workflow*. Since all evaluations appear to involve static text inputs, maintaining consistent terminology (e.g., *pre-execution* vs. *in-workflow*) would greatly improve clarity.
- **Related work:** The following appear to be missing and should be discussed for completeness:
    - https://arxiv.org/abs/2412.15701
    - https://arxiv.org/abs/2409.16427

### Soundness
1

### Presentation
1

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
The paper introduces OASIS benchmark to evaluate LLM agent safety along two dimensions simulateneously, intent concealment (how well malicious intent is hidden) and task complexity (e.g. length of tool chains). It investigates how these two orthogonal factors affect agent’s safety alignment and execution or refusal. Authors test several SOTA LLM in their defined realistic and idealized scenarios and identify that safety alignment degrade proportionally with intent concealment and a complexity paradox, where agents appear safer on complex tasks, but maybe due to capability limitation, rather than safety reason. Authors also study different LLMs show static and dynamic refusal.

### Strengths
1.	The paper introduces a new two-dimensional benchmark with per-step harm labels. The involvement of domain-experts and double verification by authors in benchmark preparation is good.

2.	The paper shows how concealed intent and task complexity jointly affect safety at different levels of either one, which seems logical for making comprehensive decision about agent’s safety capability than unidirectional measurement.

3.	The benchmark is evaluated on 8 different LLMs and identified several interesting phenomenon such as Complexity-Safety Tradeoff, static-dynamic refusal decision etc.

### Weaknesses
1.	The paper states that “all tasks were synthesized using Gemini 2.5 Pro,” but does not clarify how Gemini generated these tasks, or what prompting or control strategy was used. Overall, how they were generated. Without proper justification, it’s difficult to assess whether the resulting tasks are realistic or represent an actual real-world scenario.

2.	Although the evaluations are well-organized, with the small benchmark, it’s difficult to say if the findings are actually statistically generalizable.

3.	The paper primarily combines existing concepts into a new dataset and sandbox. While evaluations are interesting in their scope, some findings are somewhat intuitive, except maybe dynamic vs static refusal, but it’s narrow in scope.

### Questions
Please answer the generalization and benchmark question in the weakness section,

### Soundness
3

### Presentation
2

### Contribution
2
