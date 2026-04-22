# Towards Evaluating Fake Reasoning Bias in Language Models

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
Large Reasoning Models (LRMs), evolved from standard Large Language Models (LLMs), are increasingly utilized as automated judges because of their explicit reasoning processes. Yet we show that both LRMs and standard LLMs are vulnerable to Fake Reasoning Bias (FRB), where models favor the surface structure of reasoning even when the logic is flawed. To study this problem, we introduce \textbf{THEATER}, a comprehensive benchmark that systematically investigates FRB by manipulating reasoning structures to test whether language models are misled by superficial or fabricated cues. It covers two FRB types: (1) \textbf{Simple Cues}, minimal cues that resemble reasoning processes, and (2) \textbf{Fake CoT}, fabricated chains of thought that simulate multi-step reasoning. We evaluate 17 advanced LLMs and LRMs on both subjective DPO and factual datasets. Our results reveal four key findings: (1) Both LLMs and LRMs are vulnerable to FRB, but LLMs are generally more robust than LRMs. (2) Simple Cues are especially harmful, reducing accuracy by up to 15\% on the most vulnerable datasets. (3) Subjective DPO tasks are the most vulnerable, with LRMs suffering sharper drops than LLMs. (4) Analysis of LRMs' thinking traces shows that Simple Cues hijack metacognitive confidence, while Fake CoT is absorbed as internal thought, creating a “more thinking, less robust” paradox in LRMs. Finally, prompt-based mitigation improves accuracy on factual tasks by up to 10\%, but has little effect on subjective tasks, where self-reflection sometimes lowers LRM performance by 8\%. These results highlight FRB as a persistent and unresolved challenge for language models. Code and data are available at \url{https://anonymous.4open.science/r/fake-reasoning-bias-0B5A}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces THEATER, a benchmark designed to study Fake Reasoning Bias (FRB), a phenomenon where language models, including Large Reasoning Models (LRMs) and Large Language Models (LLMs), are misled by superficial reasoning cues. THEATER tests two FRB types: Simple Cues, which mimic minimal reasoning patterns, and Fake CoT, which imitates multi-step reasoning. Evaluating 17 advanced models on subjective and factual datasets, the study finds that both LRMs and LLMs are vulnerable, with LRMs being less robust. Simple Cues reduce accuracy most severely (up to 15%), especially in subjective DPO tasks. Analysis reveals that LRMs’ explicit reasoning increases susceptibility to misleading cues, leading to a “more thinking, less robust” paradox. Prompt-based mitigations offer modest gains for factual tasks but limited benefits for subjective ones.

### Strengths
1. The paper presents THEATER, a comprehensive and well-structured benchmark for systematically analyzing Fake Reasoning Bias (FRB) in language and reasoning models.

2. It offers novel empirical insights into how superficial reasoning cues undermine model robustness, highlighting the “more thinking, less robust” paradox in LRMs.

3. The evaluation is extensive and well-executed, covering 17 state-of-the-art models across diverse datasets and providing actionable findings on bias mitigation.

### Weaknesses
1. The observation that superficial cues can interfere with reasoning trajectories is not novel, as prior research has already documented this phenomenon. [1,2,3]

2. The "Semantic Cue" experiment inserts a fixed reasoning pattern between answer options, which seems unnatural for evaluating LLMs. A more meaningful setup would introduce such cues within the reasoning process itself. Besides, the finding that larger reasoning models (LRMs) experience greater degradation is also expected, given their training on prompts like “let me think” or “wait, wait” preceding solutions. This insertion "hacks" the models’ training data distribution rather than revealing a genuine reasoning weakness.

3. The purpose of FakeCOT (shown in Table 1) is unclear. Since the injected distractive reasoning sequences are not model-generated, they may lie outside the model’s internal token distribution. If these sentences indeed have high perplexity for the model, they are unlikely to be produced by it, and thus should not be interpreted as indicative of an inherent “bias.”


[1] https://aclanthology.org/2025.findings-acl.1006.pdf
[2] https://arxiv.org/pdf/2311.09702
[3]https://arxiv.org/pdf/2410.05229

### Questions
The fallacy in reasoning pattern is indeed an important phenomenon to be studied. In general it is expected to conduct research on problems like ``what contributes to such reasoning fallacy'' and ``how to improve the robustness of the LRMs''.

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
5

### Summary
The paper introduces the THEATER benchmark which augments existing MMLU tasks and preference datasets with injections of “fake reasoning”, to test if models are biased to select answers including reasoning structures. They include results on different open and closed source LLMs and LRMs and some analysis on their findings.

### Strengths
The paper tackles an important and well-posed problem – if models have biased preference for reasoning structures in answers – which is especially pertinent as reasoning models become prevalent. The presented benchmark evaluates this issue and authors present results on several different models and analysis on different axes (model family, LRM vs LLM, etc.) and try a basic prompt-based mitigation.

### Weaknesses
Some experiment design choices may be biasing the results presented in the paper – e.g. only introducing simple cues in the second answer choice, instead of randomizing, or only generating injection biases using a single model. Some analysis of their findings (e.g. why shallow cot is the most damaging) feels incomplete. Their bias injections are also limited to only a single answer choice and tasks such as MMLU are reformulated with a correct and only a single incorrect answer, which is not how this task is used in practice.

### Questions
- Table 1: For simple cues why is the thinking cue injected before the answer choice instead of within the answer choice?

- section 2.1 line 168: good that the model generating the bias injections wasn’t included in the eval, but using a single model to inject bias can introduce unique biases of its own. Can the dataset be augmented with injections from multiple models?

- section 2.2 line 229: Why is the reasoning cue only being injected for one choice, can this be scaled to include all of the choices included in the original MMLU questions? A more fair comparison would be to include some reasoning cue for each option (for the correct choice as well, but maybe slightly longer for some incorrect options) and see if the model gets distracted and to select an incorrect option.

- section 3.1 line 269: Why are the cues only injected before or in the second answer option? This mixes the reasoning bias with positional bias of the model. Better to randomly select the choice where it is injected.

- section 3.1 line 298: If LLMs not trained to reason are more robust to FRB than LRMs, then it would be good to test if not producing a CoT to compare answers makes the model more robust to FRB. Models may be more calibrated when producing a single token answer (https://arxiv.org/pdf/2207.05221).

- section 3.2 line 316: why is shallow CoT the most damaging? one might expect that longer reasoning chains would prove most distracting for the model

- section 3.2 line 320: for simple cues, it seemed that LLMs were more robust than LRMs for both factual and subjective tasks. Why are LRMs more robust than LLMs for factual tasks with fake CoT?

- section 3.2 line 323: it may be valuable to try to isolate the effects of reasoning training vs the data reasoning training was done with. If the evaluated LRMs are primarily trained for math and coding reasoning, their reasoning capabilities wouldn’t necessarily transfer to more subjective tasks. Would LRMs trained with in-domain data for subjective tasks be more robust to FRB than LLMs?

- section 3.4 line 415: how were these mitigation prompts selected? Trying a few different prompts and averaging results would increase the reliability of these results.

- section 3.4: It would be interesting to do the same CoT analysis as in the previous section after the prompt mitigations to understand how models respond to this mitigation.

- section 5: What could be the right way to scalably handle FRB if prompting is insufficient?

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
3

### Summary
This paper offers yet another critique of chain-of-thought reasoning, this time proposing a benchmark to measure what they term “fake reasoning bias.” While the idea of introducing a benchmark is potentially valuable (helping standardize evaluation and encourage more empirical rigor) the execution here is lacking.

On the positive side, the benchmark concept provides a starting point for measuring a specific kind of reasoning flaw, and even an imperfect benchmark can serve as a focal point for further refinement and community discussion. However, the actual presentation is confusing and the visualizations are poorly executed. There’s little statistical grounding, and it’s hard to discern any meaningful insights from the data they present. It feels like a rough sketch rather than a polished tool.

In the end, I am lukewarm on this paper. I lean toward acceptance mainly because it’s not worth a lengthy debate. It is a minor, incremental effort that might have some limited utility, but I am not particularly impressed, I will certainly forget this paper immediately after reading it. If it helps start a conversation about better benchmarks, that’s fine, but as it stands, it’s not a game-changer.

### Strengths
- Concrete benchmark proposal (THEATER) for “fake reasoning bias” in LLMs.
- Attempts cross-family, cross-scale evaluation (DeepSeek, Qwen, OpenAI).
- Distinguishes between simple cues and fake CoT, which could be empirically tractable.
- Offers first quantitative treatment of reflection cues as bias sources.

### Weaknesses
- Presentation confusing; figures and metrics unclear.
- Statistical methodology under-developed; no significance reporting, hard to epistemically ground the claims.
- “Fake reasoning bias” partly rebrands known prompt sensitivity phenomena.
- Claims of metacognitive distortion speculative; weak theoretical grounding.

### Questions
- What exactly is measured? is FRB separable from e.g. surface-form bias or position bias?
- How stable are FRB effects across temperatures, decoding schemes, and seeds? This could serve as foundation for statistical grounding.
- Can the benchmark’s metrics be integrated into training or are they post-hoc only?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigated a novel issue termed Fake Reasoning Bias (FRB), which posits that the “redundant” CoTs in LLM/LRM will degrade their performances. The authors considered two main branches of FRBs, and established a benchmark named THEATER to assess their effects. Extensive numerical experiments were conducted, which provide many interesting findings.

### Strengths
**The motivation behind the study is intriguing, and the findings are also valuable**. I appreciate this practical finding-driven motivation, which effectively addresses a gap in the understanding of the emerging trends in LLMs and LRMs. The experiments conducted and the corresponding results are ample and rich.

Besides, **the paper is well-organized**.

### Weaknesses
My main concern is that this paper primarily focuses on presenting findings **without offering any theoretical analysis or more fruitful insights**. While Section 3 outlines a series of valuable observations, the authors neither develop a unified theory to explain these observations nor explore the underlying factors that contribute to the findings in a more meaningful way.

This gives the impression that the paper highlights several intriguing issues, but ultimately remains at a surface level, lacking a deeper understanding of the underlying complexities.

### Questions
**Please try to answer the following questions:**

1. I’m curious as to why the authors focus solely on these two types of FRBs. Are there other potential forms of FRBs that could also be considered? I don't believe that these two alone fully encompass the range of FRBs.
2. The authors' THEATER framework introduces simple cues and fake CoTs by design. However, if these added contexts are inherently redundant, they could negatively impact performance, which may not necessarily be related to the FRB issue. Could the authors provide evidence to demonstrate that this is not the case?
3. Can authors also include a discussion on the limitations of the paper? I believe this is also important.
4. I noticed that a paper published recently explored the similar topic [1], making the basic idea **not** brand new anymore. Can authors provide the distinctions and connections?

[1] Does Thinking More always Help? Mirage of Test-Time Scaling in Reasoning Models. NeurIPS, 2025.

### Soundness
3

### Presentation
4

### Contribution
2
