# A Personalized Conversational Benchmark: Towards Simulating Personalized Conversations

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 2

## Abstract
We present \textsc{PersonaConvBench}, a large-scale benchmark for evaluating personalized reasoning and generation in multi-turn conversations with large language models (LLMs). 
Unlike existing work that focuses on personalization or conversational structure in isolation, \textsc{PersonaConvBench} tightly integrates both, offering three core tasks: sentence classification, impact regression, and user-centric text generation, covering 10 diverse Reddit-based domains. 
This design enables systematic analysis of how personalized conversational context can shape LLM outputs in realistic, multi-user conversational scenarios. 
We systematically benchmark several commercial and open-source LLMs under a unified prompting setup, and observe that incorporating personalized conversational history yields substantial performance boosts—e.g.,  
achieving a 198\% relative gain over the best non-conversational baseline in sentiment classification.
By releasing \textsc{PersonaConvBench} with comprehensive evaluations and codes, we aim to facilitate research on LLMs that can adapt to individuals’ conversational styles, track long-term context, and generate more contextually rich and engaging responses.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces PersonaConvBench, a comprehensive benchmark designed to evaluate personalized reasoning and generation in multi-turn conversations with large language models (LLMs). Built upon Reddit posts and comments, PersonaConvBench leverages users’ interaction histories as personalized signals to predict new comments. The benchmark consists of 19,215 posts and over 111,239 conversations from 3,878 users, spanning 10 diverse Reddit-based domains. PersonaConvBench defines three core tasks:
- Sentiment Classification: Binary prediction of the polarity of user replies.
- Impact Forecasting: Regression-based prediction of community feedback scores.
- Personalized Text Generation: Generation of user-specific follow-up responses.

Experimental results demonstrate that incorporating personalized conversational history significantly improves the performance of state-of-the-art LLMs, including GPT-4.1, GPT-4o-mini, Claude-3.5, Llama-3.3, and DeepSeek-R1.

### Strengths
- The benchmark’s focus on evaluating LLM personalization using users’ past interaction histories is both well-motivated and highly realistic.
- The proposed dataset is large-scale and diverse, spanning 10 domains and encompassing varied conversation styles. Its thoughtful construction incorporates temporal constraints and a graph-based representation of conversations.
- Extensive experiments yield strong empirical results: leveraging users’ past interaction histories and dialog context consistently improves performance across all tasks, models, and domains.

### Weaknesses
- Evaluation Metrics: The evaluation of Personalized Text Generation primarily relies on n-gram overlap metrics and SBERT scores, using only a single reference response. Given the open-ended nature of dialog, there may be multiple valid responses for a given context, making these metrics potentially insufficient for capturing the full range of appropriate outputs. Additionally, the absence of human evaluation limits the assessment of response quality and relevance.
- Research Findings: The results indicate that incorporating dialog context and user interaction history improves the prediction of user responses. However, this outcome is somewhat expected, as removing dialog context or substituting interaction history with that of other users constitute relatively weak baselines. Therefore, the reported improvements are not particularly surprising.

### Questions
How many examples are used in the in-context learning setting? 
Given that some user might have long interaction history, will there be context limitation when doing few-shot learning?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a personalized benchmark dataset curated from Reddit posts. The authors construct a graph to capture the relationships between posts and their corresponding replies across conversation turns. In addition, they propose three benchmark tasks: sentiment classification, Reddit upvote prediction, and response generation.

### Strengths
1. The paper presents a multi-turn personalized dialogue benchmark derived from Reddit posts.

### Weaknesses
1. The paper provides limited ablation studies to support its experimental findings.

2. The dataset curation process based on Reddit data is not particularly novel.

3. Although the paper emphasizes conversational personalization, there is little evidence of incorporating personalization signals beyond dialogue history in the response generation process.

### Questions
N/A

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
PersonaConvBench introduces a large-scale, Reddit-derived benchmark that integrates personalization and multi-turn conversational structure across 10 domains and three tasks—sentiment classification, impact (score) regression, and user-specific next-text generation. The authors provide ~19k posts, evaluate multiple LLMs, showing large performance gains from personalized history.

### Strengths
+ Extensive, real-world dataset spans 10 Reddit domains—19,215 posts, ~111,239 conversations, 3,878 users, providing scale and diversity for robust evaluation of personalized conversational models. 
+ Novel formulation combines graph-structured multi-user, multi-turn conversations with three tasks—sentiment classification, impact regression, and user-centric next-text generation—plus standardized in-context prompting and evaluation protocols. 
+ Comprehensive LLM benchmarks reveal large personalization gains.

### Weaknesses
- The paper measures personalization mostly via performance deltas (P-Conv vs P-NonConv) and paired t-tests, rather than a direct “degree of personalization” metric or richer human judgments.
- Heavy Reddit preprocessing (Nu, Nr, Np thresholds) and class-imbalance filtering (initial ~11:1 skew reduced to ~5:1) retained only ~6k sentiment posts. Removal of deleted/short posts create selection bias toward highly active users, reducing representativeness and real-world robustness.
- Experiments run GPT-4.1, GPT-4o-mini, Claude-3.5, LLaMA3.3, DeepSeek-R1 but omit Qwen-family and reasoning-mode evaluations. Greedy, zero-shot decoding may understate reasoning gains. 
- Generation evaluation relies on automatic metrics (ROUGE, BLEU, METEOR, SBERT) without reported human evaluation; these metrics can miss conversational quality, personalization nuance, and pragmatic appropriateness.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces PERSONACONVBENCH, a large-scale benchmark that evaluates how LLMs perform personalized reasoning and generation in multi-turn conversations. It integrates personalization and dialogue context across ten domains and three tasks—classification, regression, and text generation—showing that using user-specific conversation history improves model performance.

### Strengths
The paper constructs the first benchmark that jointly models personalization and multi-turn dialogue, enabling systematic evaluation of LLMs’ ability to adapt to user-specific styles and evolving conversational context.
By representing multi-user conversations as directed temporal graphs, the benchmark captures realistic branching, temporal ordering, and inter-user dependencies—allowing for fine-grained personalization and contextual reasoning that go beyond flat dialogue datasets.

### Weaknesses
Problem formulation lacks clarity:
The notation is underspecified — in particular, while Cu​ (the user trajectory set) is later defined, the meaning of f is not clearly introduced where it first appears. This makes it difficult to precisely understand what constitutes the model input.

Ambiguity in task setup and visibility scope:
It is unclear whether the model has access to all users’ conversational trajectories or only those of the participants in the current dialogue. In real conversations, a user A replying to B might also draw on prior interactions with other users (e.g., C). The paper does not explicitly explain whether such cross-thread context is included, how it is implemented. If implemented, whether temporal constraints are also enforced in such cross-thread context (i.e., that a reply at time t can only use information from ≤ t – 1). Without a clear temporal or visibility restriction, the need for a graph-based formulation is weakened.

Line 218: The phrase “conditioned on the conversational trajectory and the user’s trajectory set” is conceptually ambiguous. Are these two distinct inputs to the model, or do they represent different levels of abstraction of the same information? My understanding is that the conversational trajectory refers to an abstract notion, while the user’s trajectory set denotes the concrete collection of conversations associated with a specific user. If they are indeed separate inputs, please clarify their respective definitions, roles, and how they differ in practice.

Evaluation limitations:
For the dialogue generation task, it appears that each message has only one reference reply as ground truth. Metrics such as ROUGE are thus poorly suited to capture the diversity and open-endedness of conversational responses, limiting the reliability of quantitative evaluation.

### Questions
see above

### Soundness
3

### Presentation
2

### Contribution
2
