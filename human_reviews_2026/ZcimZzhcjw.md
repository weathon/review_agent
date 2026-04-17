# Ada-Search: Balancing Parametric Knowledge and Search in Large Language Models via Reinforcement Learning

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Equipping large language models (LLMs) with search engines via reinforcement learning (RL) has emerged as an effective approach for building search agents. However, overreliance on search introduces unnecessary cost and risks exposure to noisy or malicious content, while relying solely on parametric knowledge risks hallucination. The central challenge is to develop agents that adaptively balance parametric knowledge with external search, invoking search only when necessary.
Prior work mitigates search overuse by shaping rewards around the number of tool calls. However, these penalties require substantial reward engineering, provide ambiguous credit assignment, and can be exploited by agents that superficially reduce calls. Moreover, evaluating performance solely through call counts conflates necessary and unnecessary search, obscuring the measurement of true adaptive behavior. 
To address these limitations, we first quantify the self-knowledge awareness of existing search agents via an F1-based decision metric, revealing that methods such as Search-R1 often overlook readily available parametric knowledge. Motivated by these findings, we propose **AdaSearch**, a simple two-stage, outcome-driven RL framework that disentangles problem solving from the decision of whether to invoke search, and makes this decision process explicit and interpretable. This transparency is crucial for high-stakes domains such as finance and medical question answering, yet is largely neglected by prior approaches. 
Experiments across multiple model families and sizes demonstrate that AdaSearch substantially improves knowledge-boundary awareness, reduces unnecessary search calls, preserves strong task performance, and offers more transparent, interpretable decision behaviors.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the challenge of training large language model (LLM) agents to use an external search tool only when necessary, rather than over-relying on search for every query. Overuse of search can incur unnecessary cost and risk exposure to irrelevant or malicious content. Previous approaches tried to curb excessive tool use by adding penalties for each search call, but this required complex reward engineering and led to ambiguous learning signals, agents might game the reward by avoiding calls even when needed.

### Strengths
The paper introduces a multi-task RL training framework (ADASEARCH) that explicitly optimizes both problem-solving and tool-use decision-making, instead of entangling them in a single reward.  It eliminates complex reward shaping – using a simple success/failure reward – yet achieves performance on par with or better than prior methods that used elaborate penalties.

### Weaknesses
* ADASEARCH  mainly re-frames existing tool-RL setups by partitioning prompts (decision vs. solving) and using simple outcome-only rewards to introduce a “should I search?” gate; this reads as an engineering refinement rather than a fundamentally new paradigm and thus feels incremental rather than a paradigm shift.

* ADASEARCH is consistently underperform by Search-R1 on EM across benchmarks. While the paper argues it reduces unnecessary tool calls, there is no accompanying cost/latency/token-use analysis (e.g., average tool calls per query, tokens per answer, wall-clock time, or $-cost) to demonstrate a concrete efficiency win that would justify the accuracy trade-off.

* Lack of qualitative case studies or error analysis. The paper would benefit from worked examples showing when the model correctly abstains from search, when it mistakenly avoids or overuses search, and typical multi-hop failure modes—with decision rationales and retrieved evidence.

* Limited evaluation breadth and realism. Testing on newer search benchmarks (e.g., BrowseComp, SimpleQA if applicable) and adding ablations with different retrievers and  responses numbers K would better demonstrate robustness.

### Questions
*  Please quantify average tool calls, latency, tokens, and $-cost.
*  Add a few worked examples showing correct abstentions, under-search, over-search, and typical multi-hop failures (with decision rationales).
*  Evaluate on newer browsing/search benchmarks (e.g., BrowseComp; SimpleQA if applicable) and vary retrievers，K and threshold  for robustness.

### Soundness
3

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
3

### Summary
This submission presents an RL framework designed to help LLMs decide when to rely on their internal knowledge and when to use external search tools. The authors’ primary goal is to reduce the number of unnecessary search calls, while maintaining task performance. 

To solve this problem, the authors introduce AdaSearch, an RL framework that trains LLMs to both solve tasks and make decisions about when to search. AdaSearch proceeds in two stages. In the first stage (problem solving), the model is trained with two prompts: one for parametric reasoning and one for search-based reasoning. The model is rewarded when it produces a correct final answer, as measured by an exact match with the ground truth. In the second stage (decision making), the model learns to decide whether to use search for a given question. Using “pseudo-labels” generated from the first stage’s solve rates, they train a binary classifier to predict whether external search is required. The authors also introduce an end-to-end variant that joints trains the problem-solving and decision-making modules. 

The authors empirically evaluate their framework using the Qwen2.5 and Llama3 models, conducting experiments using question-answering benchmarks, comparing to various baselines. For the search component, they use E5 on a 2018 Wikipedia page dump. 

The main results are presented in Table 2. The metrics the authors report are EM (task accuracy) and F1 score. Across the different tasks, AdaSearch performs comparable to or slightly better than the baselines.

### Strengths
AdaSearch is a clean RL framework for balancing an LLM’s internal knowledge with external search, without relying on hand-tuned reward functions. Their approach works by disentangling problem solving from deciding whether or not to search. I view the simplicity of their approach as a virtue and the main strength of this submission. Despite this simplicity, their method achieves comparable or slightly better performance across multiple QA benchmarks compared to more complicated baselines. Balancing search and internal knowledge is an important, and well-studied, problem. The paper writing is clear.

### Weaknesses
While I like the simplicity of this approach, the middling results when compared to the baselines make it unclear when the authors' approach should be used. I understand that previous approaches rely on reward engineering, but a discussion on when to use this approach versus other alternatives would be appreciated.

### Questions
n/a

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
This paper proposes a reinforcement learning framework aimed at training large language models to decide when to rely on parametric knowledge versus invoking external search tools. The approach uses a two-stage training process with simple binary outcome rewards instead of complex reward shaping. The authors claim to reduce unnecessary search calls while maintaining task performance. Experiments are conducted on small 3B-parameter models (Qwen2.5 and Llama-3.2).

### Strengths
- Reducing unnecessary search calls in tool-augmented large language models is important both, economically and for system reliability. 
- Empirical evaluation covers a diverse set of Q&A benchmarks. The decision-level F1 metric is a nice addition.
- The method significantly reduces the number of search calls.

### Weaknesses
- Missing ablation for the p-threshold and parameters of the reward-shaping based RL baselines. 
- Benchmarks do not directly reflect an advantage of reduced function calls as Search-R1 consistently outperforms AdaSearch. 
- Lack of hyperparameter settings (learning rate, ...), although promised to be in the appendix (page 6), limits reproducability. 
- There is no evaluation of AdaSearch-E2E. The only comparison is made in Table 5. I suggest the authors focus on one of the methods or extend the analysis in the paper. 
- The writing and formatting often seem rushed and incomplete. For example: 
	- Table placement should be improved. Table 1 is mentioned on page 2 but appears on page 5. 
	- Algorithm 2 is not mentioned at all. 
	- Page 5: "[...] as detailed in Appendix X."
	- Hyperparameters promised in the appendix are missing.

### Questions
- Can you provide more evaluation results for AdaSearch-E2E?
- Could you provide ablation studies for naive shaping and AdaSearch? How does the threshold for p affect performance and tool use? How does it compare to different values of lambda?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the problem of training LLMs using reinforcement learning to efficiently use search calls to leverage retrieval tasks. In particular, the paper focuses on eliminating the use of search calls when the model's internal knowledge is sufficient to solve the task. To do so, the authors propose a straightforward inference method in which the model is first prompted to decide whether to use its own parametric knowledge (i.e. no search calls), or to use search in its response. The model is then trained using RL to improve all three of these abilities (deciding whether to use search or not, solving without search, and solving with search). The authors show that the resulting method is roughly on par with standard search + RL methods (e.g. Search-R1) in accuracy, but improves in terms of the proportion of tasks that don't utilize search (f1 score).

### Strengths
The paper tackles a clear problem and the proposed method is easy to understand and well-presented. The experiments are also well-done and almost all baselines are considered.

### Weaknesses
I found the paper to be primarily weak because of the lack of importance of the proposed problem setting. The paper focuses on reducing the proportion of tasks in which search is invoked, but I don't see a strong argument for why this is actually a useful distinction over prior work that reduces the average number of tool calls. For example, one could argue that not invoking tool calls on some prompts can reduce latency, but the proposed method requires an additional prompt to the LLM. Furthermore, because no existing method focuses on this problem setting, it's no surprise that the proposed metric of F1 score is better optimized by Ada-Search. Even then, there is a missing reward-shaping baseline that more sharply penalizes the transition between 0 and 1 tool calls.

### Questions
1. Why is SubEM necessary for the decision reward? If there's actually a big difference with EM, then using SubEM for everything may be better. 
2. Stage 1 and 2 seems to be more of an ablation - I'd suggest keeping E2E as the main method, and introduce stage 1/2 only in the experiments section.
3. Beyond F1 score, are there results for average number of tool calls per problem for all methods?

### Soundness
3

### Presentation
3

### Contribution
2
