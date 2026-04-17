# Automated Capability Discovery via Foundation Model Self-Exploration

- Decision: Reject
- Scores: 8, 2, 0

## Abstract
Foundation models have become general-purpose assistants, exhibiting diverse capabilities across numerous domains through training on web-scale data. It remains challenging to precisely characterize even a fraction of the full spectrum of these abilities and potential risks in any new model. Existing evaluation approaches often require significant human effort, and it is taking increasing effort to design ever harder challenges for more capable models. We introduce Automated Capability Discovery (ACD), a framework that designates one foundation model as a scientist to systematically propose open-ended tasks probing the abilities of a subject model (potentially itself). By combining frontier models with ideas from the field of open-endedness, ACD automatically and systematically uncovers a diverse spectrum of surprising capabilities and failures in the subject model. We demonstrate ACD across a range of foundation models (including the GPT, Claude, and Llama series), showing that it automatically generates thousands of distinct tasks, which are then clustered to reveal dozens of broader capability areas and failure modes, that would be challenging for any single team to uncover. We further validate our method's automated scoring with extensive human surveys, observing high agreement between model-generated and human evaluations. By leveraging foundation models' ability to both create tasks and self-evaluate, ACD is a significant step toward scalable, automated evaluation of novel AI systems. 
All code and evaluation logs are open-sourced at https://anonymous.4open.science/r/ACD-D13E.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper develops an Automated Capability Discovery System whereby an AI (LLM) scientist model probes an AI subject model. Their systems can discover a wide range of unexpected capabilities and failures. ACD also generates a comprehensive report, describing the AI scientist's study of the AI subject model.

### Strengths
This was an exceptionally well-written paper. It was a very smooth and cohesive narrative. 
I think systems like this will be increasingly built in the future, and excited to see steps in this direction.

### Weaknesses
The text in Figure 4 is extraordinarily small (even when zooming in). 

It would be nice to compare the evaluations done by ACD to the evaluations done by existing scientists on existing models. Does ACD rediscover many of the capabilities or vulnerabilities that scientists identified in existing models? More discussion of how the weaknesses or capabilities discovered by ACD differ from human evaluators, for instance, they are more varied/diverse or less diverse. Even anecdotal evidence would suffice, but getting human experts to judge the quality of inputs (rather than general human participants) would be significant. 

A clearer discussion of automated red teaming and how this differs from previous automated systems would be helpful.

More discussion of the resource usage, ie, api prices, GPU usage, time, etc in the main body would be a significant improvement. Is this system cheap and cost-effective, or does it require resources that only frontier or well-resourced labs would have?

### Questions
Does ACD rediscover many of the capabilities or vulnerabilities that scientists identified in existing models?

What is the computational intensity and price to run this system? 

Can such a system be used to benchmark rank models, and if so, what is the ranking?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose Automated Capability Discovery, a framework where a “scientist” foundation model autonomously generates and evaluates tasks to probe the abilities and failure modes of a "subject" model. They claim that ACD surfaces diverse task families and reveals both strengths and "surprising" weaknesses across GPT-4o, Claude, and Llama models.

### Strengths
- The authors work on a timely challenge in scaling adaptive evaluation for foundation models.
- The approach demonstrates the ability to generate a wide variety of task types
- The method has potential practical value because it can produce reusable task repositories for evaluating new models.

### Weaknesses
- The paper does not sufficiently define or analyze the notion of “interestingness” and novelty used to filter task proposals; in particular, given that this is a key element in their approach, I'd appreciate a more detailed explanation of how these were evaluated (or supposed to be evaluated by the LLM). I saw the appendix but this still seems very high-level – was either of these concepts at least validated somehow during the task generation process?
- The authors do not discuss the risk of systematic bias (a) if the scientist and subject models are the same and (b) if the same model (or similar models) both propose tasks and evaluate correctness (see this paper for more problem context: https://arxiv.org/abs/2404.13076).
- The paper repeatedly claims that ACD reveals failure modes that "traditional" benchmarks miss but it doesn't provide any evidence for these claims, so their statements feel very anecdotal.
- The criteria for declaring a task “consistently solved” or “consistently failed” are not clearly specified or justified.
- The robustness of the human evaluation is uncertain because the reviewers’ domain knowledge across highly diverse task types is not established.
- In general, the authors claim that the tasks are valid but don't provide sufficient evidence that this is true (see this paper for a more in-depth explanation of how to establish validity in evals: https://arxiv.org/abs/2505.10573)
- Nit: abbreviation “FM” was introduced but is inconsistently used
- The related work & background section don't sufficiently detail how the authors' approach is novel compared to previous attempts (neither 2.1 nor the first paragraph of Section 3 establish this). I'm willing to update my score if this gets clarified.

### Questions
- In Section 5.3, was Claude 3.5 also the evaluator throughout the whole discovery process?
- What does “sufficiently” mean in line 225? What is the stop criteria?
- Does “consistently solved” mean that all n shot evals need to be correct?
- How is the validity of each of translated task code ensured?

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper prompts a large language model to generate questions to study the capabilities of a LLM (either the same or different). They claim that the resulting benchmark is highly diverse, full of novel tasks, and interesting.

### Strengths
The paper is clearly written.

### Weaknesses
At it's core, this paper fails to provide any real evidence that their methodology is useful. This paper is full of phrases like "novel task families" and "interestingly new," seeking to highlight the allegedly highly diverse and presumably useful questions. However at no point do they present any empirical analysis - qualitative or quantitative - of these aspects. While they do do a human study of the questions developed using their methodology, they only ask the humans if the tasks are "valid and coherent" and not if they are diverse, novel, interesting, new, etc.

Additionally, at no point do they compare the evaluation questions developed via their methodology to those using any other LLM-powered methodology or to static benchmarks. We have no way to know if this methodology is superior to the dozens of other similar papers, nor even to believe that it's significantly better than static benchmarks. They also fail to provide ablations on any aspect of their pipeline, leaving the door open for significant improvements via minor tweaks.

Finally they seem to scope their literature review in such a fashion as to render no other methodology a worthwhile comparison point, despite the fact that papers like "Discovering Language Model Behaviors with Model-Written Evaluations" and "AutoBench-V: Can Large Vision-Language Models Benchmark Themselves?" would make perfectly good points of comparison (neither is cited). If this is deliberate, it is academic misconduct.

### Questions
What evidence do you have that the tasks identified by this methodology are interesting or worthwhile, either in abstract or compared to other methods for producing benchmarks?

Did you know about the paper "Discovering Language Model Behaviors with Model-Written Evaluations"? If so, why did you not cite or compare to it?

### Soundness
1

### Presentation
3

### Contribution
1
