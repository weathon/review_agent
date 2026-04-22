# HugAgent: Evaluating LLMs in Simulating Human-Like Individual Reasoning on Open-Ended Tasks

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Simulating human reasoning in open-ended tasks has long been a central aspiration in AI and cognitive science. While large language models now approximate human responses at scale, they remain tuned to population-level consensus, often erasing the individuality of reasoning styles and belief trajectories. To advance the vision of more human-like reasoning in machines, we introduce HugAgent (Human-Grounded Agent Benchmark), which rethinks human reasoning simulation along three dimensions: (i) from averaged to individualized reasoning, (ii) from behavioral mimicry to cognitive alignment, and (iii) from vignette-based to open-ended data. The benchmark evaluates whether a model can predict a specific person's behavioral responses and the underlying reasoning dynamics in out-of-distribution scenarios, given partial evidence of their prior views. HugAgent adopts a dual-track design: a human track that automates and scales the think-aloud method to collect ecologically valid human reasoning data, and a synthetic track for further scalability and systematic stress testing. This architecture enables low-cost, extensible expansion to new tasks and populations. Experiments with state-of-the-art language models reveal persistent adaptation gaps, positioning HugAgent as the first extensible benchmark for aligning machine reasoning with the individuality of human thought. The benchmark, along with its complete data collection pipeline and companion chatbot, is open-sourced as HugAgent (https://anonymous.4open.science/r/HugAgent) and TraceYourThinking (https://anonymous.4open.science/r/trace-your-thinking).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a benchmark to evaluate LLMs at simulating individuals, predicting the person's belief state and belief updates. For benchmark creation, an LLM chatbot interacted with humans to infer their beliefs and updates over 8-20 QA pairs on 3 topics, where the survey also collects gold labels based on human reported stances and reasoning weights. The task for a language model is to then predict the structured belief state and updates of humans. Results show that global majority baseline leads to low performance, demonstrating individual reasoning is needed for the task. Evaluations are conducted on low-cost or old models, which with individual context perform near (best 79.12 for Qwen 2.5 32b instruct) the human baseline (80.6) for belief state inference, but are behind (best 63.38 for Llama 3.3 70b) for belief updates (human = 79.70). The paper includes a preliminary discussion of many possible design choices and details for the task of simulating individual reasoning.

### Strengths
1. The paper studies an interesting problem of LLMs simulating individual reasoning as beliefs and belief updates.

2. The paper conducts a human study to collect data, and also collects human performance estimates by computing test-retest agreement.

3. The paper shows evidence that models find it harder to simulate human belief updates, in comparison to predicting initial human beliefs given the context of their conversation.

### Weaknesses
1. The methodology is not written clearly and is hard to follow. For example, the dataset construction is almost entirely described in Appendix C. A chatbot is used to converse with humans to elicit their beliefs and belief updates, but it is unclear how this chatbot is created, what these conversations look like, and how effective this methodology is. Then in section 4.2, the term "attribution" is introduced for the first time as a metric being evaluated, even though section 3.5 on evaluation protocols has no mention of it. The paper could benefit from a significant rewrite to make it clear what the hypotheses and claims are, and the methodology used to study them.

2.  It seems like the results are extremely noisy, especially for the belief inference task where most models have very similar performance ranging from 74.5-79.1. There seems to be no clear interpretation of the results even for belief updates, where for example DeepSeek R1 has really poor belief inference performance with strong belief update prediction performance. The stated reason is "over-elaboration", but it is unclear what that means, and no evidence or justification is provided for why it is the correct explanation. Moreover, GPT 5 Mini results are reported but not Gemini 2.5 pro, GPT 5 High thinking, Claude etc. which makes it unclear what the state of the art with modern LLMs is. 

3. In the section on "main findings", the findings seem quite preliminary with not enough convincing evidence. For example in "more context doesnt always help", its unclear whether: does the dialogue provided as context already leak the prediction task answers? the claim that performance peaks at 5-10 questions before declining does not hold true for many datapoints in the tables eg Qwen 2.5 32B. For finding 2, I would be interested to konw what the human attribution accuracy generalization is, as I find it unclear why personalization transfer is expected from a single conversation about a person's beliefs about "zoning" to "healthcare".

These are just examples of my broader concern: at the end of reviewing this paper, I have little clear takeaways, and much confusion. To improve the presentation in future versions, I suggest a) reporting error bars b) reporting results as bar graphs instead of tables for clearer comparison c) clearly stating hypotheses, why the specific test used to test them is valuable, and what it shows d) giving details on benchmark construction early, and more analysis of benchmark quality and finally more discussion on prior work like [1] that studies individual simulations could help position the contribution.

[1] Generative Agent Simulations of 1,000 People
Joon Sung Park, Carolyn Q. Zou, Aaron Shaw, Benjamin Mako Hill, Carrie Cai, Meredith Ringel Morris, Robb Willer, Percy Liang, Michael S. Bernstein

### Questions
1. How good is the benchmark data constructed using a chatbot that elicits human beliefs and belief updates?

2. What is the alignment between the synthetic agent and human study? Why were only "50 agents" (where I believe agent means causal bayesian network?) used for the synthetic track if the motivation was scaling (the human study already has 36 participants). Can synthetic agents really be used to reliably scale the data? How ecologically valid are they?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of evaluating large language models (LLMs) in simulating human-like individual reasoning, moving beyond population-level consensus. The authors introduce HugAgent, a benchmark for "average-to-individual reasoning adaptation," which requires predicting a specific person's belief states and reasoning trajectories given partial evidence of their past views. HugAgent employs a dual-track design: a synthetic track for scalable, controlled stress tests and a human track for ecologically valid "out-loud" reasoning data. The benchmark formalizes two core tasks, i.e., Belief-State Inference and Belief Dynamics Update, and evaluates state-of-the-art LLMs, revealing persistent adaptation gaps. Contributions include a formalized task definition, baseline results, error analyses, and open-source release of the benchmark and data-collection chatbot.

### Strengths
1. The motivation is strong and interesting. The paper formalizes "average-to-individual reasoning adaptation" as a measurable task, addressing a critical gap in LLM evaluation.

2. The use of first-person, out-loud self-reports as ground truth enhances ecological validity compared to static survey responses and the dual-track design allows for both controlled stress tests (synthetic) and real-world validation (human), which supports robust evaluation.

3. Experiments cover multiple state-of-the-art LLMs. Providing broad baseline results and comprehensive ablation studies further reveal some meaningful findings toward cross-domain generalization, context-length, and so on.

4. Detailed pipeline, including benchmark data collection, annotation, and chatbot code, is well illustrated and released as open source, promoting community adoption and extension.

### Weaknesses
1. The human track includes only 36 participants after quality control, which may limit statistical power and generalizability. While the authors mention it in limitations, my primary concern is that are these participants sufficient to represent the thinking ways of people with different characteristics in the real society? To ensure diversity, one way is to increase the sample size. Otherwise, you can report more detailed information about these 36 participants (e.g., their occupations and characteristics) to show that the sample size is enough for diversity consideration. 

2. In Sec 4.2, the authors mention that `` ﻿﻿Tables 2 summarize performance. … Open-source LLaMA and Qwen rival GPT-4o’’, but there is no GPT-4o results in Table 2.

3. While the authors observe many findings through ablation studies, they do not analyze deeply for each finding and many of these points were glossed over without in-depth analysis. I think some ideas can be further developed. 

4. The update operator U in Eq. (2) is introduced without justification in Line 121.

### Questions
See Weaknesses

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
This paper introduces HugAgent, a benchmark for evaluating whether LLMs can simulate individual reasoning rather than just population-level consensus. The key innovation is framing "average-to-individual reasoning adaptation" as a measurable task: predicting how a specific person would reason and update their beliefs in novel scenarios given their past views. 

The benchmark features a dual-track design combining synthetic agents (for controlled testing) with human participants (for ecological validity), and evaluates two core tasks 1/ Belief-State Inference (predicting current stance and reasoning), and 2/ Belief Dynamics Update (predicting how beliefs change under counterfactual evidence). Experiments with 9 state-of-the-art LLMs reveal systematic failure modes, with models achieving 74-79% accuracy on belief-state inference but struggling more with belief dynamics.

### Strengths
The shift from simulating "the average" to "the individual" is critical for applications like digital twins, personalized AI assistants, and social simulation. This is well-motivated.

The paper goes beyond accuracy to include error analysis, cross-domain transfer tests, context ablations, and systematic failure mode analysis (directional bias, domain sensitivity, tail-driven errors).


Promising to release the full pipeline, including the chatbot, is valuable for the community.


The formalization using belief states, Bayesian updating, and structural causal models (Section 2) provides principled anchors. The four guiding hypotheses (H1-H4) make the evaluation partially interpretable

### Weaknesses
1. Missing important related work: The paper cites Agent Bank (Park et al., 2024) and discusses ToM benchmarks briefly, unless I miss it -several highly relevant recent works should be discussed more thoroughly:

PersonalLLM (ICLR 2025) - directly addresses personalization with heterogeneous preferences, very similar to HugAgent's goals

UniToMBench (2025) - unified ToM benchmark with multi-interaction tasks and evolving scenarios


2. 36 humans is quite limited for establishing generalizable conclusions about individual reasoning. The paper acknowledges this but doesn't adequately address:

 - Demographic diversity concerns

- Whether 36 participants provide sufficient coverage of reasoning patterns

- Statistical power for cross-domain transfer claims


3. I feel there is a theoretical grounding overclaim. Section 2.2 invokes Bayesian updating, PLoT, and SCMs as "anchors". But: 1) These aren't actually used in the evaluation or analysis, 2) The connection between theory and practice is unclear, and 3) H2 (cross-domain transfer) references graph similarity but admits it's "left as future work"

4. Table 2 shows that DeepSeek-R1 performs poorly (40.16% on belief-state inference, 42.25% on update). The explanation ("over-elaboration diverges from human reasoning") is speculative and not empirically validated.

### Questions
For the claim that "more context doesn't always help" is interesting but needs deeper analysis, 
- Why does update accuracy peak at 5-10 questions, then decline?
- Is this cognitive overload or increasing noise?


Section 7 proposes "guiding principles" but admits these are "left as future work." Did you test any of these principles? If so, what were the results?

The claim is that synthetic agents "approximate the structural statistics of human belief graphs." Can you provide evidence in the main paper that synthetic agents exhibit similar error patterns to humans when evaluated by LLMs?

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
2

### Summary
The paper proposes HugAgent, a dataset for evaluating LLM agents' capabilities on simulating human-like reasoning. It features a human track for ecological validity and a synthetic track for scalable stress tests. Results reveal persistent adaptation gaps: top LLMs trail human retest ceilings, with larger deficits on belief dynamics.

### Strengths
- Novel problem formulation assessing LLMs’ capability to simulate *individual* human reasoning.
- Results provide interesting insights into LLM failure modes
- Open-source pipeline (chatbot, data, evaluation code) supports reproducibility and extension.

### Weaknesses
- Confusing use of “open-ended”: in LLM evaluation, this often implies systems auto-generating novel tasks; here it seems to refer to diverse human reasoning traces in curated domains—clarify terminology to avoid misinterpretation.  
- Scalar score calibration: predicting 1–10 stance or 1–5 reason weights is known to be noisy in LLMs; no analysis of calibration error or distribution shift vs. human ratings.
- Synthetic track lacks human validation: 50 scripted agents use deterministic causal graphs. Is there a study that confirms humans perceive these as plausible or follow similar update paths?

### Questions
see above weakness

### Soundness
3

### Presentation
2

### Contribution
3
