# How Do Coding Agents Spend Your Money? Analyzing and Predicting Token Consumptions in Agentic Coding Tasks

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 2

## Abstract
AI agents offer substantial opportunities to boost human productivity across many settings. However, their use in complex workflows also drives rapid growth in LLM token consumption. When agents are deployed on tasks that can require millions of tokens, a natural question arises: where does token consumption come from in agentic coding tasks, and can we predict how many tokens a task will require? In this paper, we use Openhands agent as a case study and present the first empirical analysis of agent token consumption patterns using agent trajectories on SWE-bench, and we further explore the possibility of predicting token costs at the beginning of task execution. We find that (1) more complex tasks tend to consume more tokens, yet token usage also exhibits large variance across runs (some runs use up to 10$\times$ more tokens than others); (2) unlike chat and reasoning tasks, input tokens dominate overall consumption and cost, even with token caching; and (3) while predicting total token consumption before execution is very challenging (Pearson’s $r<0.15$), predicting output-token amounts and the range of total consumption achieves weak-to-moderate correlation, offering limited but nontrivial predictive signal. Understanding and predicting agentic token consumption is a key step toward transparent and reliable agent pricing. Our study provides important empirical evidence on the inherent challenges of token consumption prediction and could inspire new studies in this direction.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses a timely and underexplored aspect of AI agents: the analysis and prediction of token consumption in agentic coding workflows. Using trajectories from Claude Sonnet 3.7 on the SWE-bench dataset, the authors provide empirical insights into token usage patterns and evaluate prediction methods. While the work offers novel findings on cost dynamics and highlights practical challenges, it is limited by its reliance on a single model and proprietary pricing schemes, which undermines generalizability. The prediction results are underwhelming, raising questions about the feasibility of the proposed approaches. Overall, this is an exploratory study with interesting observations, but it requires broader experimentation to strengthen its claims. I recommend rejection because it needs to address generalizability and overstatements of prediction accuracy.

### Strengths
The lack of transparency in AI agent pricing is a significant barrier to their widespread adoption. As the authors claim, this appears to be the first empirical study on agent token consumption. By focusing on agentic coding tasks (e.g., via SWE-bench), the paper potentially informs better pricing models and user expectations.

The core discovery that higher token costs are associated with lower accuracy (termed the "inverse test-time scaling paradox"), is compelling and counterintuitive. This challenges assumptions about scaling test-time compute and suggests inefficiencies in agent trajectories, such as excessive exploration leading to diminished performance.

The authors meticulously decompose token types (e.g., non-cached prompts, cache creation, cache reads, and completion tokens) and their contributions to costs across different phases (early, mid, and late).

### Weaknesses
The entire empirical analysis and prediction experiments are based solely on a single, closed-source LLM Claude Sonnet 3.7. Although the authors acknowledge this in the limitations section, it severely restricts the paper's scope. The title implies broad applicability to "AI Agents," but the conclusions may not extend to other models (for example, GPT-series or open-source alternatives). A more accurate framing might be "How Does Claude Sonnet 3.7 Spend Your Money?" to reflect the narrow focus.

The cost analysis (particularly in Figures 3 and 4) heavily depends on Anthropic's specific pricing scheme, which differentiates rates for cache creation, cache reads, and non-cached inputs. Switching to a different provider (e.g., GPT-4o) or an open-source model could yield entirely different cost dynamics. The methodology fails to decouple these findings from vendor-specific decisions, limiting the transferability of the results.

The latter half of the paper focuses on token prediction, but the core results indicate failure: Pearson's correlation for total token consumption is below 0.15, which is essentially negligible. The authors claim that predicting output tokens and ranges is "practical and reasonably accurate," but Figure 5 shows that even the best setting (PTDR with LogScale-FewShot) achieves only around 0.36 correlation for output tokens, a value typically considered "weak" to "moderate" in prediction tasks. Describing this as "reasonably accurate" overstates the findings and could mislead readers.

The authors do not evaluate the real-world utility of "self-prediction" approaches, such as the overhead costs involved. If a task requires only 50,000 tokens but the prediction process consumes 30,000 tokens, users end up paying for 80,000 tokens total, rendering the method economically unviable. Additionally, the noted tendency for self-prediction to "overestimate true costs" further diminishes its practicality, yet this is not critically discussed.

### Questions
The "inverse test-time scaling" finding is intriguing. Could this be linked to specific agent behaviors, like looping in exploration? Are there ablation studies on agent prompts or tools that might mitigate this?

### Soundness
2

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
4

### Summary
This paper presents the first empirical study to analyze and predict the token consumption of AI agents in complex coding tasks. The authors identify that unpredictable and often massive token usage is a major barrier to the adoption of agentic systems, creating non-transparent pricing and risk for users .

The paper's contributions are:


An empirical analysis of agent trajectories from the SWE-bench dataset. This analysis reveals several key findings: (a) Token consumption has extremely high variance, with some runs using 10x more tokens than others for the same task. (b) More token usage is correlated with lower accuracy, a phenomenon the authors call an "inverse test-time scaling paradox" . (c) Unlike chat models, input tokens dominate overall consumption and cost in coding agents, even with caching.





A prediction task formulation to estimate token consumption before task execution. The authors benchmark multiple prediction methods, including zero-shot and few-shot prompting of a single LLM, as well as letting the agent "self-predict" its own costs.


The paper concludes that while precisely predicting the total token consumption is extremely challenging (Pearson's r < 0.15), predicting output token counts and the log-scale range of total consumption is feasible and reasonably accurate.

### Strengths
The paper tackles a practical, high-impact problem (cost prediction) that is a major barrier to real-world agent adoption.

The "inverse scaling" paradox and the "input token dominance" are both major, counter-intuitive findings that will directly influence future research into agent efficiency.

 The analysis is based on strong, relevant data (SWE-bench) and the methodology is rigorous.

The paper provides immediate, actionable advice for agent providers (e.g., "offer log-scale ranges or budget alerts, not exact quotes") and for researchers (e.g., "focus on optimizing input token ingestion, not just generation").

### Weaknesses
1. The entire paper's analysis is based on one model (Claude Sonnet 3.7), one framework (OpenHands), and one task domain (coding on SWE-bench) . This is a significant limitation.



2. Do the findings generalize? Does the "inverse scaling" paradox hold for GPT-4o or Llama 3.1? Does "input token dominance" hold for other input-heavy tasks, like document analysis or web research, or is it unique to coding? The paper claims to analyze "agentic coding tasks", but it has only analyzed one agent's behavior.

3, While a negative result is a valid scientific contribution, the fact remains that the primary goal of predicting total token cost was not achieved (r < 0.15). This makes the paper more of an "analysis and problem-definition" paper than a "solution" paper.


4. Cost of Self-Prediction: The paper rightly explores "self-prediction" but Figure 7 shows this prediction process is itself very expensive, consuming a median of ~25,000 prompt tokens and ~400 completion tokens. This high cost to get a prediction makes the method impractical. The paper notes this as a challenge but doesn't fully analyze this trade-off (e.g., is the prediction cost correlated with the actual cost?).

### Questions
1. The paper's most significant findings (inverse scaling, input-token dominance) are based on a single model and benchmark. How confident are you that these findings are fundamental to agentic coding, rather than being an artifact of the OpenHands framework or Claude 3.7's specific reasoning patterns?

2. The "Inverse Scaling" Paradox: This is a fascinating finding. What is your hypothesis for the cause of this? The paper suggests "inefficient trajectories", but could it also be that longer context histories (from more tokens) actively degrade the agent's reasoning, causing it to get "lost in the middle" and fail? I am interested in it.

3. The "self-prediction" agent itself consumes a significant number of tokens to produce an estimate (shown in Figure 7). Did you find any correlation between the cost of the prediction and the actual cost of the task? It seems this prediction overhead makes the method impractical for all but the most expensive tasks.

4. Why was predicting input tokens so much harder than output tokens (Figure 5)? Is it because the agent's exploration path (which files to read, which tools to call) is fundamentally more stochastic than its generation (which just solves a given sub-problem)?

5. Your paper's key finding is that input tokens—from repository exploration and tool use—are the primary cost driver, not completion tokens. one paper AgentTaxo (https://openreview.net/pdf?id=0iLbiYYIpC) independently discovers this exact same phenomenon in multi-agent systems, finding that input tokens outnumber output tokens by 2:1 to 3:1 due to inter-agent communication. Given that both papers are pioneering the empirical analysis of agent tokenomics and have validated this same counter-intuitive finding in complementary domains (agent-to-environment vs. agent-to-agent), can you please address this gap by citing AgentTaxo and discussing how your work and theirs together provide a more complete picture of why agentic systems are so input-heavy?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper systematically investigates the token consumption patterns and cost predictability of coding agents driven by large language models (LLMs) when performing real-world software engineering tasks on the SWE-bench Verified dataset.

### Strengths
This is one of the first studies focusing on token-level cost analysis for LLMs. I find some of its conclusions particularly insightful for users who are attentive to token cost, especially when optimizing prompt design or inference budgeting.

### Weaknesses
1. Since this is an empirical analysis paper, it is important to justify the sufficiency and generalizability of using SWE-bench as the sole benchmark across all dimensions of investigation. To what extent do these conclusions generalize to other platforms?
2. The “accuracy-cost trade-off frontier” is not explored.
3. The observed negative correlation between token usage and accuracy may stem from problem difficulty, as the paper does not control for this variable.

### Questions
1. It would be better to expand the experimental datasets to include more diverse programming and reasoning tasks.
2. I am wondering if this token-consumption problem can be solved by optimization techniques.
3. I would suggest to establish a performance-cost trade-off curve to better quantify efficiency and accuracy relationships.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper makes use of LLM trajectories on SWE-bench to study how coding agents consume tokens for different tasks and explore the possibilities of predicting the execution cost. The problem is interesting and important given the recent proliferation of agentic AI. Also, some of the findings based on the token statistics of the agents are interesting. Yet all the prediction tasks and analytics tasks are carried out using prompts, and no learning algorithms are involved in this paper. Seems that its current contents better fit some other conferences than this conference focusing on representation learning.

### Strengths
The problem based on the SWE-bench data is interesting and could trigger more learning tasks related to Agentic AI to be formulated. The paper is asking some interesting questions.

### Weaknesses
The technical contribution of this paper is on making use of the trajectory data and carrying out some analytics tasks to answer the questions raised related to token consumption by coding agents. It does not propose any algorithm or formulation which are related to representation learning tasks. Even though I like the questions being asked, the discussion with details related to agentic AI is only the illustration of the prediction task instead of some research roadmap.

### Questions
Q1: Other than using LLMs to predict the execution cost, why not trying to learn some prediction models? 

Q2: Given the findings of the paper, can you see some learning tasks which can be defined? E.g., given that the same task could result in big differences in token consumption, other than just warning the user, can the task execution and/or input token for context ingestion be analyzed and optimized first (like query optimization in DBMS)? This is only one example. Will there be more related learng tasks related to the usage data which can be identified and formulated?

### Soundness
2

### Presentation
3

### Contribution
2
