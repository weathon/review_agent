# MoReBench: Evaluating Procedural and Pluralistic Moral Reasoning in Language Models, More than Outcomes

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
As AI systems progresses, we rely more on them to make decisions with us and for us. To ensure that such decisions are aligned with human values, it is imperative for us to understand not only what decisions they make but also how they come to those decisions. Reasoning language models, which provide both final responses and (partially transparent) intermediate thinking traces, present a timely opportunity to study AI procedural reasoning. Unlike math and code problems which often have objectively correct answers, moral dilemmas are an excellent testbed for process-focused evaluation because they allow for multiple defensible conclusions. To do so, we present MoReBench: 1,000 moral scenarios, each paired with a set of rubric criteria that experts consider essential to include (or avoid) when reasoning about the scenarios. MoReBench contains over 23 thousand criteria including identifying moral considerations, weighing trade-offs, and giving actionable recommendations to cover cases on AI advising humans moral decisions as well as making moral decisions autonomously. Separately, we curate MoReBench-Theory: 150 examples to test whether AI can reason under five major frameworks in normative ethics. Our results show that scaling laws and existing benchmarks on math, code, and scientific reasoning tasks (fail to) predict models' abilities to perform moral reasoning. Models also show partiality towards specific moral frameworks (e.g., Benthamite Act Utilitarianism and Kantian Deontology), which might be side effects of popular training paradigms. Together, these benchmarks advance process-focused reasoning evaluation towards safer and more transparent AI.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces MoReBench, a benchmark designed to test how well models can perform moral reasoning by assessing their chain-of-thought reasoning rather than their final decisions. They do this by collecting 1,000 moral scenarios (many from pre-existing datasets), and then creating expert-written rubrics containing the criteria which good reasoning traces ought to meet. They demonstrate that overall model performance (e.g. ChatbotArena or AIME) does not predict how good models are at moral reasoning. They also introduce MoReBench-Theory, a similar benchmark which assesses how well models can reason under 5 different moral frameworks (e.g. utilitarianism, virtue ethics, etc.). They show that models are generally more capable of utilitarian and deontological reasoning and less capable of reasoning under the other moral frameworks.

### Strengths
1. The paper identifies an interesting gap in the literature.  
2. The authors use LLMs as judges to assess how accurately reasoning traces comply with a given rubric. This raises the natural question of how accurate LLM judges are at assessing this. The authors answer this question in a lot of depth, including comparing their answers to human experts, and tasking human experts with writing low-quality, medium-quality, and high-quality reasoning traces and then showing the LLM judges can tell the difference. The results here are quite convincing.  
3. The curation of the rubrics in the dataset appears to be high quality. They hired 53 experts to do this, most of whom have doctorates in moral philosophy or a similar subject. They then use a second expert to review each rubric and make any necessary edits.  
4. The results, particularly Figure 4, are surprising and important. Insofar as the metric is accurate, it seems to suggest that models do not get meaningfully better at moral reasoning over time, which raises major concerns about the ability of frontier labs to teach their models how to reason in ethically challenging scenarios.  
5. The paper is clearly written, easy to follow, and reports most of the metrics that I’d like to see.

### Weaknesses
1. ​​The authors also introduce MoReBench-Hard, which penalizes longer reasoning traces. The choice of using a length-corrected score seems questionable — taking the example from the appendix, if a moral agent is acting as a federal judge, is it really undesirable for it to spend 1000 tokens on reasoning before deciding a sentence? In moral dilemmas with major consequences, reasoning for a long time seems appropriate, and yet the authors are penalizing it.  
2. Two of the metrics introduced in this paper are MoReBench and MoReBench-Hard, with the latter penalizing long reasoning traces. The authors argue that both of these meaningfully capture LLMs’ moral reasoning abilities. However, if this was the case, one would expect these metrics to correlate with each other at least somewhat strongly. This is not the case, the correlation is only r=0.09 (calculated from the results in table 6; the authors conveniently avoid reporting the correlation). Given that this is the case, it would seem that at least one of these metrics isn’t meaningfully measuring the models’ underlying moral reasoning capability. This also makes me a bit more sceptical of some of the other results the authors report, due to the ease of cherry picking one or the other metric depending on which tells a more convenient story. For instance, Fig. 5 reports how CoT quality correlates with the quality of final decisions, both measured by MoReBench-Hard. But the authors do not, as far as I can see, report the same statistic for MoReBench. (Though to their credit, the most important results, e.g. Fig. 4, do report both metrics)  
3. Looking at the examples in appendix B.1, some of the scenarios appear to be very contrived and unrealistic. E.g. the one on lines 1053–1073 has absurdly unrealistic sentencing guidelines (“either a 5-year sentence or a 25-year sentence with no intermediate options allowed”). The model’s behavior on obviously contrived scenarios like this may or may not be indicative of the model’s behavior in real-world situations, which limits the usefulness of this benchmark.  
4. The authors themselves discuss in sec 4.1 that there’s an “inverse scaling” phenomenon with this benchmark, and speculate (I think correctly) that larger models are capable of performing more reasoning steps implicitly without needing out output them in the CoT. This means that, as models get larger and more capable, this benchmark will become less and less capable of accurately capturing the quality model’s moral reasoning (indeed we are already seeing the start of this trend). This is, of course, inherent and unavoidable in any CoT-centric benchmark, but it is nonetheless a noteworthy weakness of the authors’ approach.  
5. The data referenced on lines 400–403 doesn’t appear to match the data in the corresponding table (Table 2). E.g. on avoiding harmful outcomes, the text says 77.5% while the table says 81.1%. The qualitative claims do match the data in the table though.  
6. Abstract is unclear about the results: “Our results show that scaling laws and existing benchmarks \[...\] (fail to) predict models’ abilities to perform moral reasoning”. This tells the reader nothing; this should be clearly stated as being one way or the other.  
7. Typo in the first sentence of the abstract: “As AI systems progresses” should be “As AI systems progress”

### Questions
See weaknesses, in particular points 1, 2, and 3\.

In addition:

1. Where can I see the actual dataset? As far as I can see, there are no supplementary materials in this submission, so the only thing I can see are the handselected examples in the appendices; this makes it hard to see for myself how high-quality the dataset is.  
2. Just to make sure, does the judge only have access to the CoT, or does it have access to both the CoT and the final output? If it’s only the CoT, this could be missing something; plausibly some models justify their reasoning in more detail in the final answer  
3. Is there a version of Figure 5 with MoReBench rather than MoReBench-Hard?  
4. Re data in Fig 4.3 — can you report the same data for the expert-written reasoning traces which they use in Sec. 3.3? I’m wondering if the judge model could just be bad at assessing the logical reasoning criteria.

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
3

### Summary
The paper presents MoReBench, a benchmark focusing on evaluating moral reasoning, using expert-developed, rubric-based scoring. It contains a good amount of scenarios and criteria that contribute to the field of model evaluation, especially in the area of moral reasoning. It presents a novel framework that includes scenario dataset capturing moral dilemma spanning 16 topics.The paper also showcase a good, detailed analysis on the performance of models such as different models of GPT-5, Claude, Gemini and Llama against their thinking traces, sizes and model capabilities.

### Strengths
The work shows strong, original focus on moral reasoning and provide a robust analysis, grounded in a wide variety of scenarios and topics. It also recruited the help of human experts in defining rubic criteria and writing the scenarios (besides other sources concerning moral advisory or agency roles) which is a good contribution to the field. The analysis of the correllation between the MoReBench and other benchmarks (as a proxy for representing moral reasoning vs other capacities), which illustrates the usefulness and differentiation of MoReBench to other datasets. Aspects of evaluation and coverage of topics are well designed and integrated into the benchmark.

### Weaknesses
The writing and explanation of the paper could be clearer. 

In Figure 1, while the diagram provides a clear illustration of components and processes within the benchmark, it could have distinguish the roles of the moral advisor and the moral agent more clearly (it was also slightly confusing since the advisor and agent use two different scenario, and the candidate reasoning process uses one of these only). 

There is quite a lot of information condensed in the paper, many deeply rooted in the field of philosophy and ethics. The paper could improve its readability with clear structuring of dataset creation, criteria creation and evaluation, along with where the elements fit into these (for example, how the pluralistic perspectives from five major frameworks in normative ethics mentioned in Figure 2 were implemented and how they help augment the usefulness or coverage of the benchmark; how is the level defined i.e. Morebench Hard and Regular).

### Questions
- Is there any potential bias from using only GPT-5 High as a judge?
- From Figure 6, there seems to be a wide range (lower bound to upper bound) of performance contained in all of the boxplots, which nearly cover a similar range of performance for Contractarianism, Contractualism, Denontology and Virtual Ethics. How robust is the conclusion on model performance given this range?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work presents an expert-written benchmark on moral reasoning, containing 1000 LLM-as-moral-agent/advisor problems and 150 theoretic problems. For each problem, authors curated multiple expert-written rubrics and evaluate with LLM-judge. They then benchmarked frontier models, discovering a "reverse-scaling-law", the models' weakness in generating both good thinking traces and results, and the models' inability to utilize moral reasoning frameworks.

### Strengths
The paper presented splendid human-resourcing and experimental workload. Curation of the questions, rubrics and rubric-testing experiments all include high-quality expert work and extensive experiments on frontier models. 

This is one of the first work on evaluating pluralistic moral reasoning traces, and is sound with quantifiable and multi-dimensional rubrics.

Detailed and inspiring conclusions are presented in both general and specific aspects.

### Weaknesses
One natural thing to question is the consistency of quality of human-written content. With such a large number of question/rubrics to write per expert, is there a way to ensure non-overlapping? Are the rubrics fair across all questions? This leads to questions in experiments as well. For example, in 3.3, can experts really write reasoning traces of distinctively "low, medium and high" quality?

There's a general lack of emphasis on models' final responses / decisions. Although experiments show quality of reasoning traces are in positive correlation with quality of responses, why isn't this correlation, or the sole quality of responses, part of the rubrics? For weaker / not aligned models running this benchmark, the moral soundness of its final response should also be considered for harmful outputs.

There should also be studies on the correlation between the models' implicit moral inclination and performances on the benchmark. The external evaluation ran are mainly in "general-domain/math/code reasoning", but much literature has supported the existence of implicit moral inclination of LLMs, and such inclination would probably affect the quality of moral reasoning.

### Questions
Questions are listed in the weaknesses section with descending importance.

### Soundness
2

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
This paper proposes a diverse benchmark together with a set of well-curated criteria to evaluate the morality of llm, not just in the final result, but also the intermediate steps.

### Strengths
1. it includes more than 1000 scenarios
2. clear and solid evaluation on whether llm-as-judge is reliable
3. includes philosopher-written criteria, which could be extremely useful
4. extensive model evaluation

### Weaknesses
There are a lot of papers mentioning that the chain of thought is not faithful, while this paper basically relies on analyzing the thinking process generated

I'm curious what the authors think about it and how this may impact the actual fidelity of the morality evaluation

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
4
