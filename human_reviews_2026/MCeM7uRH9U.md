# An Interactive Paradigm for Deep Research

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Recent advances in large language models (LLMs) have enabled deep research systems that synthesize comprehensive, report-style answers to open-ended queries by combining retrieval, reasoning, and generation. Yet, most frameworks rely on rigid workflows with one-shot scoping and long autonomous runs, offering little room for course correction if user intent shifts mid-process. We present **SteER**, a framework for steerable deep research that introduces interpretable, mid-process control into long-horizon research workflows. At each decision point, **SteER** uses a cost–benefit formulation to determine whether to pause for user input or proceed autonomously. It combines diversity-aware planning with utility signals that reward alignment, novelty, and coverage, and maintains a live persona model that evolves throughout the session. **SteER** outperforms state-of-the-art open-source and proprietary baselines by up to 22.80\% on alignment, leads on quality metrics such as breadth and balance, and is preferred by human readers in 85\%+ of pairwise alignment judgments. We also introduce a persona–query benchmark and data-generation pipeline. To our knowledge, this is the first work to advance deep research with an interactive, interpretable control paradigm, paving the way for controllable, user-aligned agents in long-form tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces STEER, a novel framework that brings interactive, user-steerable control to long-form research question answering systems. Current “deep research” pipelines (where an AI agent conducts multi-step web retrieval and reasoning to produce a comprehensive report) typically operate in a largely autonomous way: the system might ask one clarification at the start and then proceed to generate a lengthy answer. This one-shot approach can lead to misalignment with the user’s true needs if those needs are not fully specified upfront or evolve as new information emerges.

STEER aims to keep the user in the loop in a principled, minimally invasive manner throughout the research process. The framework is built around three key ideas: 

(1) Diversity-aware planning: At each step, the system generates several potential sub-questions or directions to investigate, explicitly promoting diversity of topics (using techniques like Maximal Marginal Relevance to avoid redundancy). This ensures a broad exploration of the query’s aspects. 

(2) Cost–benefit based pause decision: Rather than blindly pursuing all those directions or arbitrarily asking the user, STEER uses a decision module to determine whether to pause and ask the user to choose the next direction. It computes a utility for expanding each candidate branch (favoring those that align with the user’s interests, add novelty, and improve coverage of the topic) and an approximate “execution cost” for following that branch (e.g. how much work it entails). If the system continues autonomously, it will pick a subset of branches that maximize total utility minus cost. If it pauses, the user can select which branches they care about (and even suggest new ones). The expected benefit of pausing is calculated as the gain in utility from letting the user prune low-value branches and add relevant ones, minus a pause cost that models the burden of bothering the user. The pause cost is personalized: it increases as more questions are asked, according to a user-specific tolerance budget (so the system will only interrupt when it believes the user’s guidance is truly valuable).

3) Live persona modeling: STEER maintains a dynamic model of the user’s persona and preferences. Initially, it takes into account any provided profile or aspect checklist the user wants covered. After each interaction, it updates this persona – for instance, if the user selects certain subtopics or explicitly states new preferences, these are incorporated.

### Strengths
In experiments on a constructed persona-query benchmark, STEER outperforms state-of-the-art baselines (including both open-source pipelines and a proprietary system) by a large margin in terms of alignment to the user’s requested aspects (gains of up to 22.8% in alignment metrics are noted). Its answers also maintain strong overall quality, with higher breadth and balance of coverage. 

Human evaluators overwhelmingly prefer STEER’s outputs – in over 85% of cases when asked which answer is more tailored to their interests, and similarly high preference for focus and usefulness. Importantly, the framework allows tuning the trade-off between asking too often and potentially missing what the user wants: by adjusting the pause cost parameters, one can make the agent more cautious or more independent, as suited for the user.

The contributions of this work are the introduction of an interactive, interpretable decision policy in deep research agents and a demonstration that this leads to measurably more user-aligned results.

### Weaknesses
One possible downside is added complexity: STEER’s policy involves many components (planning, utility estimation, persona updates) and could require careful calibration of hyperparameters like the user’s interruption cost. I hope to hear detailed discussions about this matter (In terms of realistic application view). However, the authors provide a clear formulation for these and show that the system can effectively balance asking versus autonomy. 

Also, i want the authors to carefully investigate whether there are any other benchmarkable research application, which enables 1) pause decision or 2) live persona modeling. I evaluate these characteristic as beneficial, but i think that this paper requires extensive comparisons with current web-based research services. 

I am quiet convinced that Steer advances the deep research application via various interplay algorithms, but i also think that these are somewhat incremental. So, i would appreciate if the authors provide table-based view on comparisons of techniques. I am leaning toward rejecting this paper, while i could increase the score if above issues are well resolved.

### Questions
Discussed in weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduce Steer, which is a framework that incorporate mid-process control into the long-horizon research workflow. They compare their frameowkr with opensource and openAI's o4-mini-deep-research model. THe evaluation is done on persona-tailored quality and general quality of the report. It outperform the proprietary model on the persona alignment.

### Strengths
The task steerable deep research is interesting.
The system is well designed

### Weaknesses
1. there is no meta-evaluation on the LLM-as-judge, need to have high correlation with human judgements to backup the usage of LLM-as-judge.
2. the user study is flawed: first, missing inter-annotator agreement; second, currently the user needs to mimic the persona and query, which are not what the user's own query, it would be better to conduct the user study with user doing their own queries instead of mimicking some one.
3. the task is not novel, as in chatgpt, people can already do deep research in multi-turns.

### Questions
See weaknesses

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
This paper presents SteER, a human-in-the-loop framework designed for deep research. The framework introduces interpretable, mid-process control into long-horizon research workflows, combining diversity-aware planning with utility signals that reward alignment, novelty, and coverage. Experimental results show substantial performance improvements over proprietary baselines, and human evaluators consistently prefer SteER in pairwise alignment judgments.

### Strengths
- Introduces an interactive human-in-the-loop framework for deep research, enabling mid-process interpretability and control. Centered around three components: diversity-aware exploration, pause-decision, and persona modeling, the framework systematically explores different solution paths, keeping track of different costs and user personas.
- SteER demonstrates strong performance on Persona-tailored and Quality metrics.
- Ablation studies provide valuable insights into the contributions of different components. Removing Explore, InfoGain, and Diversity exploration degrades the performance across metrics except for Depth without InfoGain.
- User studies validate the framework’s effectiveness and alignment with human preferences. Even though evaluated on only 58 pairwise annotations, SteER is preferred in 86-90% cases with substantial gains in Coverage and Findability.

### Weaknesses
- Comparisons are limited to only three frameworks (GPT-Researcher, Open Deep Research, and OpenAI Deep Research model).
- Including additional baselines such as Gemini-2.5-Pro Deep Research, Perplexity Research, and Grok Deeper Search would strengthen the evaluation.
- Experiments are conducted solely on DeepResearchGym. Broader benchmarking, including datasets like DeepResearch Bench (Du et al., 2025), would enhance the generalizability of the results.

### Questions
Was any analysis conducted on the latency or runtime of SteER compared to the baseline frameworks?

### Soundness
3

### Presentation
4

### Contribution
3
