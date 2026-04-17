# Computer Agent Arena: Toward Human-Centric Evaluation and Analysis of Computer-Use Agents

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 8

## Abstract
As Computer-Use Agents (CUAs) proliferate and grow increasingly capable, evaluation has become more challenging: static, manually curated benchmarks are narrow in domain, contamination-prone, and environment-heavy, and they diverge substantially from user-driven, real-world evaluation. We present Computer Agent Arena, an open-source platform for head-to-head CUA evaluation and a dynamic methodology that converts human preferences into structured feedback in realistic environments. The system (i) simulates real-world computer use via cloud-hosted, diverse, and dynamic environment initializations and customizations; (ii) ensures authentic, fair comparison by faithfully reproducing open-source CUAs and executing anonymously in matched, controlled environments; and (iii) extends evaluation beyond pairwise preference and correctness to capability- and behavior-oriented signals. Across 2,201 high-quality votes over 12 agents—spanning multi-app interactions, ambiguous instructions, and open-ended queries—we observe striking ranking reversals relative to static benchmarks. Further analysis shows that overall correctness mainly drives human preference; beyond that, agent-human interaction and self-correction boost user preference, even when overall task completion is comparable. Our error analysis reveals agent behavior errors, such as long-horizon memory and fine-grained action failures that static benchmarks fail to evaluate. We also contrast pure GUI agents with universal digital agents capable of tool use and coding, and discuss the trade-offs of these different design philosophies. We open source the full platform, collected dataset, and code of Computer Agent Arena to support future research on the evaluation and development of CUA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Computer Agent Arena, a platform where users can perform pairwise evaluation of agents on user-provided tasks. Analysing data generated in this way, they find that static benchmark performance does not necessarily correlate with user preference on these more ambiguous and open-ended tasks.

### Strengths
- A highly configurable environment allowing for diverse user-defined tasks
- Quick start tools reduce friction when initialising new tasks
- Very in-depth analysis in body and appendices
- Likely relevant for significantly longer than static benchmarks

### Weaknesses
- The number of votes is fairly small when spread over the task distribution.
- Evaluating agent traces and setting up new tasks is significantly more demanding than simply asking questions (as in LLMArena), which may limit the number of users willing to perform comparisons and hence the future relevance of the platform.

### Questions
- I find the poor performance of GPT-5 and Gemini 2.5 Pro quite surprising. Do you have hypotheses or analyses that suggest what the cause may be?
- You mention that long-horizon memory errors and fine-grained action failures are both common. Can you give per-agent error rates in these categories?
- Could you train a learned preference model on your data, incorporating correctness, latency, number of queries, etc, that would allow your human data to be reused in an offline setting (i.e. without requiring reranking for a quick oversight of a new model)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents COMPUTER AGENT ARENA (CAA), an open-source platform for evaluating Computer-Use Agents (CUAs) in realistic computer environments using human preference–based judgments. CAA moves beyond static and contamination-prone benchmarks such as OSWorld or WebArena by supporting dynamic, side-by-side evaluation of agents in identical cloud-hosted virtual machines. Each session runs two anonymized agents that perform the same user-defined task, and human participants rate their outcomes and behaviors. The collected preferences are aggregated into an Elo leaderboard through a Bradley–Terry model, with optional stepwise labels that capture correctness, safety, and self-correction.

The authors gather 2,201 high-quality votes from 1,058 users across 12 agents, including commercial, open-source, and multimodal models. Results show significant ranking reversals compared with static benchmarks: models strong on OSWorld often perform worse on CAA. Human preference is driven mainly by correctness, but users also value process quality such as self-correction, reasoning clarity, and judicious use of CALL_USER queries. Both excessive and absent user interactions reduce satisfaction. Error analysis highlights failure types that static benchmarks overlook, including long-horizon memory drift, insufficient clarification, and fine-grained GUI control errors. The paper also releases the platform, dataset, and codebase to support open and reproducible human-centric evaluation of CUAs.

### Strengths
1. **Originality and significance**: CAA introduces a new evaluation paradigm for computer-use agents. It extends the “Arena” concept pioneered by Chatbot Arena into the a more complex, multimodal, and stateful domain of GUI-based computer interaction. By converting crowd-sourced, real-world tasks and pairwise human preferences into structured feedback, CAA establishes a scalable, human-centric framework for CUA assessment. This contribution is conceptually innovative and practically significant, redirecting focus from benchmark scores to real user satisfaction and behavioral quality.
2. **Technical quality and system design**:  The system demonstrates great engineering practice. The cloud-based virtual machine infrastructure provides elastic scaling, hundreds of preset setups, and user-defined customizations for authentic computer use. Agents run in matched environments with identical AMI fingerprints and software versions, ensuring fair comparison. The ranking method is statistically sound and is supported by bootstrapping, permutation tests, and power analysis. It delivers reproducible Elo scores with confidence intervals. The paper also reports strong inter-annotator agreement and transparent filtering procedures.
3. **Human-centric insight and analytical depth**: The analysis section is insightful, and I highly appreciate it. It shows that users value execution process quality —such as clarity, responsiveness, and error recovery —more than speed or execution time. The observed “inverted U-shaped” relationship between the number of CALL_USER queries and win rate offers actionable guidance for designing interactive agents. Moreover, the comparison between tool-integrated and pure GUI agents reveals an important benchmark gap. Tool-centric agents perform well on scripted benchmarks but often struggle with open-ended real-world tasks due to tool misuse and hidden failures. These findings represent valuable behavioral diagnostics for the field.
4. **Clarity and communication**: The paper is well-written and illustrated. The methodology is described with approachable language, and the paper also offers full reproducibility details. This clarity significantly enhances the paper’s readability and utility to the community.

### Weaknesses
1. **Limited task coverage and representativeness**: Although the authors collect over 2,000 votes, some models receive relatively few comparisons, which may affect ranking stability. Tasks are crowd-sourced and primarily English-language, so the distribution may underrepresent non-English and specialized domains such as scientific or enterprise workflows. Explicit coverage metrics for task diversity would strengthen the evaluation.
2. **Cost and scalability**: The framework is efficient for benchmarking but remains costly for large-scale data collection (about $1.24 per vote). Scaling to preference-tuning datasets might require further automation or active sampling methods.
3. **Human-centric definition and interactivity**: While CAA is human-judged, the evaluation is not truly interactive because raters review replays rather than engage in real-time turn-taking. This limits its scope as a “human-in-the-loop” system. Future extensions could integrate live interaction to capture dynamic feedback and adaptation.
4. **Task bias and environmental variability**: Despite identical AMIs and matched virtual machines, minor runtime differences such as network latency or software updates could influence agent behavior. Moreover, crowd-generated tasks may skew toward common desktop operations rather than professional applications.

### Questions
1. **Task coverage and diversity**: How do you quantify the coverage of crowd-sourced tasks across domains and difficulty? 
2. **Influence of user expertise**: Did the authors analyze whether technical versus non-technical raters show different preference patterns? This could inform CUA design for different audiences.
3. **Complementary metrics**: Can CAA report additional task-level signals such as time-to-completion, error counts, or privacy violations to assist agent diagnosis?
4. **Score interpretation**: Given the high correlation between Elo and GenScore (r ≈ 0.95 in Appendix E.2), what distinct insight does GenScore provide beyond the main leaderboard?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present a new benchmarking platform for agentic systems that interact with computers (virtual machines). In doing so, they find that existing models perform differently from what would be expected from previously reported results.

### Strengths
- With the increasing amount of models gaming evaluations, it's always good to have more benchmarks.
- I like the direction of moving beyond easily codified evaluations and relying on more direct preferences. This has been the trend since InstructGPT, but many works regress on this.

### Weaknesses
- This feels a little weak on contributions. I would have expected some kind of algorithmic insights to go along with a benchmark for a conference paper. I would have expected to see something like this in a dedicated benchmark track.
- It seems like the main advantage of the benchmark is that it has more open-ended tasks. But are the other benchmarks with fewer open-ended tasks already more or less solved? Why do we need more open-ended tasks?
- I would want to see some more discussion relating to the shortcomings of other benchmarks here. I'm struggling a bit to understand why existing benchmarks are meaningfully inadequate here. This is really the only thing holding me back from accepting it.
- Adding humans in here makes the evaluations markedly more stochastic. It isn't, after all, the only way of doing more holistic evaluation (see, e.g., LLM-as-a-Judge or Agent-as-a-Judge). Some discussion of this or alternative ways of automating the human judges in the future could be useful. Maybe the humans are not needed at all if correctness is guiding human evaluations?
- Why are the mouse movements defined as actual movements and not clicks at certain points? This is more related to my own curiosity here.

### Questions
See Weaknesses.

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
The paper presents a platform for evaluating computer use agents (CUAs) via head-to-head evaluation strategies with real users. The platform provides preset initialized computer systems with apps in various start states. The users write task prompts against this environment and get to observe and evaluate the agent performance on the task. These preferences from more than 1000 users on about 1200 tasks are used to provide ELO style rankings of a large number of models. The results surface significant differences in human preference compared to previously observed rankings on existing CUAs evaluation environments (e.g OSWorld). Further analyses shows specific difficulties in these dynamic real user evaluations compared to static fixed set evaluations, as well as some insights in to the type of agent behavior that humans prefer.

### Strengths
This is a timely and much needed effort in evaluating computer user agents. The head-to-head evaluation using real users at scale provides strong empirical evidence and fine-grained information that can help distinguish between different model performance, and highlight where the models fail. The overall methodology is sound and has been done at scale. The framework looks extensible---can be easily used for other specific types of evaluation considerations.

### Weaknesses
I am not able to identify any significant weaknesses for this work. 

- The evaluation shows that the tasks being generated are diverse. It would help to either describe how this diversity comes about in the process. Figure 1 lays out the general process but some more details on the process could help clarify whether the scenarios within computers are somehow systematically guaranteed to be diverse or if the diversity comes from what the users do with the same scenarios. 

- While I appreciate the apples-to-apples comparison with standardized settings, it would also help to have tried the best settings for at least one or two models to see what is the best performance that could have been obtained. This is not a major issue and one that future work can fix.

### Questions
See weaknesses above.

### Soundness
4

### Presentation
4

### Contribution
4
