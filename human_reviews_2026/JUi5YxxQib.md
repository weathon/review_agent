# LLM Agents Do Not Replicate Human Market Traders: Evidence from Experimental Finance

- Decision: Reject
- Scores: 2, 2, 8, 6

## Abstract
In this study, we compare Large Language Models (LLMs) with human traders in a classic experimental-finance paradigm where prices are determined endogenously. Using a well-established asset-trading design, we run homogeneous markets with single-model LLM agents and heterogeneous “battle-royale” markets with multiple LLM models. Our findings reveal that LLMs generally exhibit a “textbook-rational” approach, pricing the asset near its fundamental value and showing only a muted tendency toward bubble formation, while humans deviate substantially and generate bubbles consistently. Additional treatments, including dividend shocks and repeated-exposure/experienced runs, show that these differences persist across various experimental settings. Further analyses of LLM-generated strategy text indicate lower variance, reduced bias, and stronger reliance on fundamentals relative to humans’ more heuristic-driven trading. These results highlight the risk of using LLM-only agents to model human-driven market phenomena, as key behavioral features such as large, emergent bubbles are not reproduced.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This study compares Large Language Models (LLMs) with human traders in experimental markets where prices emerge endogenously. LLMs consistently price assets near their fundamental values and show little bubble formation, while humans generate substantial bubbles. These patterns persist across different market conditions. Analysis of LLM strategies shows lower variance, reduced bias, and stronger reliance on fundamentals. The authors conclude that LLMs are poor proxies for modeling human market behavior, as they do not replicate key phenomena like large bubbles.

### Strengths
1. The findings show clear behavioral differences: LLM-driven markets exhibit far more “textbook‑rational” pricing than human‑driven markets.
2. The paper shows that LLMs incorporate less bias into their price forecasts and rely more on fundamental‑value strategies, unlike the heuristic‑driven approaches common among human traders.
3. These results challenge the assumption that “out‑of‑the‑box” LLMs can reliably replicate human market dynamics, especially the emergence of phenomena like bubbles and crashes.

### Weaknesses
1. The experimental setup appears overly simplified, which undermines the strength of the authors’ claims. The simulated market environment does not adequately capture the complexity of real-world trading scenarios, making it difficult to draw robust conclusions about performance differences between human traders and LLMs.
2. The configuration of LLM agents in the experiment also seems simplistic and may not fully reflect the models’ ability to understand and develop trading strategies. The implementation approach resembles prompt engineering, but it is unclear to what extent prompt design was influenced by human intervention or biases, which could affect the validity of the results.
3. The paper does not provide sufficient detail about the human participants involved in the study. Without clear information on their backgrounds, trading experience, or demographic characteristics, it is difficult to assess the credibility of the reported differences and the generalizability of the findings.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper conducted LLM experiments on experimental finance settings where there's a known fundamental value and compare it with human experimental data. The paper finds that LLM stick close to fundamentals and rarely form large bubbles, which departs from human behaviors. The paper also tested settings on battle royale markes and forecasts, showing LLM-human differences.

### Strengths
The paper covers several topics, there's comparison to human baselines and careful analysis.

### Weaknesses
I'm a bit unsold on the motivation and contributions.

- First of all, the exact types of experimental data and 'correct'/'expected' behavior is well within the training data of the LLM. In addition, prompts in the experiment disclose redemption value / fundamental value mechanisms, so it is largely unsurprising to observe the discovered LLM phenomenon. In other words, "textbook-rational" seem unsurprising given that LLMs are trained on the said textbooks, as well as the CoT reasoning paradigm that they are trained to use. 
- Given the above, I’m not sure that the mere discovery of LLM "rationality" is new or persuasive for this community. Nowadays, models already hit near–top human accuracy on competitive math (gold medal in IMO) and similar benchmarks; so the fact that an LLM looks more rational than the average human-especially when the task is framed explicitly with a redemption value-doesn’t feel surprising. I do see value in documenting this within experimental finance, but the acceptance bar likely requires more than this. 
- Most experiments are run on (or include) older model generations. In line 1028 the authors noted that "Grok 2 has been deprecated, and
thus we were unable to run this treatment with that model. Gemini-1.5 had technical errors that will be rerun for camera-ready submission". I think especially given the framing of the papers in line 450, any deployment in real markets will minimally use one of the up to date reasoning models. 

Some minor points:
- Humans face strict per-screen time limits (20 seconds to trade and 30 seconds to forecast). There seem to be a difficulty translating that to time and budget for LLMs. In the prompt used in the paper, the authors described (line 971) "PLANS/INSIGHTS" files that agents make on top of chain of thought. That's a notable difference in experimental conditions.
- There are some inconsistent findings across models (e.g., Gemini-1.5 Pro vs. GPT-3.5 vs. GPT-4o), suggesting model-specific idiosyncrasies that may not hold across generations. The paper’s own summary tables/plots show materially different error patterns and dynamics by model.
- The paper also notes "Token usage for Gemini 1.5 Pro could not be accessed through the API console". This is a minor point but is not consistent with my experience with using it.

### Questions
- LLMs have inherent prompt sensitivity. Is there any ablations on how that will affect the results (e.g. one counterargument from the papers that 'claim' to use LLM for human simulation might say that they are explicitly asking LLM to take on human persona and roles - will that change behaviors systematically)

- Can the authors articulate the contribution beyond documenting "LLMs look rational under explicit FV prompts"? What is the core conceptual/empirical advance for experimental finance or agent-based markets, and how does it differ from prior work and adds new insights to the community?

- Results may hinge on PLANS/INSIGHTS + CoT and looser time budgets for models. Is there results on no-memory/no-cot agents, or, another extreme, on more up-to-date advanced reasoning models?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Experimental Design: Conducts homogeneous (single-model) and heterogeneous ("Battle Royale") LLM-agent markets.

Behavioral Divergence: Demonstrates that LLMs exhibit "textbook-rational" behavior, while humans consistently generate price bubbles.

Robustness Tests: Validates findings across dividend shocks, experienced sessions, and linguistic analysis of trading strategies.

Forecast Rationality: Shows LLMs produce more accurate and less biased price forecasts than humans.

### Strengths
1.  First systematic comparison of LLMs and humans in endogenous experimental markets, bridging behavioral finance and AI alignment

2. Rigorous methodology: a lot of markets vs. 6 LLM models (Claude-3.5, GPT-4o, etc.), with controls for dividend shocks and experience.

3. Well-structured with clear visualizations and statistical tests.

4. Challenges the use of off-the-shelf LLMs as human proxies in finance experiments.

### Weaknesses
Are larger models (e.g., GPT-4 Turbo, Claude 3 Opus) and LLMs that may exhibit different behaviors excluded?

Why LLMs are anchored to fundamentals is not explored.

The simplified single-asset design lacks real-world characteristics (e.g., short selling, information asymmetry).

The incentives of human participants (e.g., monetary rewards) may not align with the "profit maximization" imperative of LLMs.

The impact of imperative engineering (e.g., explicit bubble-inducing directives) is not explored.

### Questions
Would a larger LLM (e.g., Claude 3 Opus) or a modified RLHF model exhibit bubble-like behavior?

How do the results apply to markets involving high-frequency trading or multi-asset portfolios?

Does the human participants' prior financial knowledge influence the results?

Can LLMs more accurately simulate inexperienced traders?

### Soundness
3

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
This paper compares the trading behavior of humans and an array of LLMs in an experimental asset market. Subjects trade the asset with each other, and the asset price depends on trading activity, allowing for the possibility of bubble formation. The paper finds that the LLMs it tests are less prone to bubble formation than human traders, this result holds both in a "mono-agent" (only copies of agents based on the same LLM) and in a "battle royale" (different LLMs competing) setting. Moreover, the two most capable LLMs tested (GPT-4o, Claude 3.5 Sonnet) exhibit the least tendency towards bubble formation. The paper also conducts a textual analysis to uncover reasons behind differences in LLM and human trading behavior.

### Strengths
The experimental design allows for a sensible comparison of human versus LLM behavior. Even though the LLMs tested are relatively outdated at this point, the fact that a broad array of different LLMs from different providers are tested, and the separation between human and LLM behavior is so clear, means the results strike me as credible and generalizable.

### Weaknesses
The textual trading strategy analysis is interesting but perhaps a little rudimentary, focusing mostly on keyword matching. Moreover, the results at the start of Section 6.1 (that the LLMs and humans all write in different styles) is not really surprising. This analysis would be strengthened by, e.g. (1) a more fine-grained semantic text analysis, (2) additional experiments in the style of Section 7, e.g. checking how LLMs' trading behavior changes if the content of the insight/plan part of the prompt changes.

### Questions
In Section 8, it seems like the prediction task the LLMs face is easier, because the prices in the LLM markets exhibit less volatility than the prices in the human markets. Is there possibly a way to control for this? For example, perhaps one could task each LLM with the prediction tasks of all the other markets (including the human markets).

### Soundness
4

### Presentation
4

### Contribution
4
