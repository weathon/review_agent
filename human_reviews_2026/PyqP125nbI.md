# Fidelity Breeds Complexity: Simulating Stock Markets with Large-Scale Generative Agents

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Stock markets are one of the most complex systems in the modern world, where prices emerge from billions of decentralized interactions among heterogeneous participants in an ever-evolving information landscape. Building a high-fidelity stock market simulator is not only a cornerstone for understanding such complexity, but also offers a valuable testbed for anticipating and mitigating crises and disruptions. Despite decades of efforts, existing methods remain confined to an unresolved dilemma: structural fidelity often comes at the cost of non-intelligent agents, while large language model (LLM) agents can only participate in oversimplified market environments. To this end, we propose MarketSim, a large-scale stock market simulation framework with generative agents. Specifically, we first design a hierarchical multi-agent architecture. By decoupling agents’ strategic reasoning from their high-frequency actions, this architecture enables LLM agents to participate in a nanosecond-resolution, NASDAQ-like continuous double auction market. Building on this, we simulate over 15k diverse market participant agents, whose billions of interactions collectively create an evolving market environment in which agents learn from feedback and adapt their strategies accordingly. Furthermore, we ground these agents in a rich informational landscape that covers over 12k real-world news articles, policy documents, and earnings reports. To evaluate our proposed MarketSim, we develop a comprehensive benchmark that includes stocks from 8 GICS sectors and 3 representative real-world scenarios, along with 5 stylized facts for market complexity and 5 price-related statistical metrics. Extensive experiments demonstrate that MarketSim not only captures the complexity characterizing real-world markets, but also accurately tracks real-world high-frequency price dynamics with an average MAPE of 3.48\%. Overall, MarketSim not only offers direct applications in understanding and anticipating financial crises, but also provides evidence for a key tenet of complexity science: fidelity breeds complexity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
1

### Summary
Hierarchical Agent Design: Decouples strategic reasoning (LLM-driven managers) from high-frequency execution (trader agents)


Scalability: Simulates 15k+ diverse agents interacting in a dynamic environment grounded in real-world data.

Benchmarking: Evaluates using 8 GICS sectors, 3 real-world scenarios, and 5 stylized facts, achieving a 3.48% MAPE in tracking real price dynamics.

Ablation Studies: Validates that joint behavioral and structural fidelity is critical for emergent market complexity.

### Strengths
1.  Novel hierarchical agent architecture mirrors real-world institutional workflows

2. Ablations systematically validate design choices

3. Well-structured with clear figures and tables

### Weaknesses
1. Reliance on proprietary LLMs (DeepSeek R1) may hinder reproducibility. Results with smaller models (Qwen3-8b) show performance drops (Tab. S8).

2. Extreme shocks (e.g., VST’s 13.9% MAPE in DeepSeek scenario,Tab. S6) are less accurately modeled.

3. Simulating 15k agents with nanosecond resolution is resource-intensive.

4. Potential misuse for market manipulation is noted but not mitigated (e.g., agent poisoning).

### Questions
1. Will the code release include simplified demos for smaller-scale validation (e.g., single-stock simulations)?

2. Can MarketSim capture long-term market cycles (e.g., recessions), or is it limited to short-term shocks?

3. Have domain experts (e.g., fund managers) validated the realism of agent decisions?

4. Regarding the article's conclusion: Overall,  MarketSim not only offers direct applications for understanding and anticipating financial crises, but also provides evidence for a key tenet of complexity science:  Fidelity breeds complexity.

What practical implications does it offer traders? Does it help traders or quantitative traders achieve greater profits? Please provide actual trading results.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents MarketSim, a hierarchical approach for modelling multi-agent markets with LLM-based reasoning yet enabling high-scale/fast acting. This tackles an important problem of modelling more realistic market scenarios.

### Strengths
- The overall idea is well motivated: Represent a hierarchy, where the “thinking” is at a higher level, requiring less agent processing, then the acting is at a lower level, allowing scalability with many actors. 
This is in the spirit of shared policy learning, learning a generic (higher level) policy, and conditioning the policy on sub-agents, see e.g. “Towards multi-agent reinforcement learning-driven over-the-counter market simulations” for a RL example of a related (but not identical) idea in market modelling, and how this relates to Equation 3 in the proposed paper.
- The experiments are good, with much analysis and ablations. Several scenarios are analysed (Liberal day, Deep seek debut, earnings announcement), across multiple different stocks. 
- The paper is well written and clear

### Weaknesses
The weakness is counterintuitively how close the model matches to the real data. An extremely close fit is observed, which raises concerns about either data leakage into the LLMs (based on the ablation with lower powered LLMs), or too much revelation in the price setting process.  The price setting process is not disclosed. Repeatedly throughout the paper its said that pricing emerges from interactions, but an exact price formation equation should be given. 

Following above, an analysis should be added without any reference prices/without following a real price trend, to disentangle any of this leakage/revelation from endogenous effects.


Addressing the above comments and questions below is essential to move the paper into acceptance

### Questions
- How is the exact price determined? Is the reference value revealing too much information about the true price? Is the order book getting values which are close from the “true” price?
- Do all the stylised facts still emerge purely from the interaction (endogenous) component? Or are the sylised facts just occurring from a reference price/news arrival (exogenous)? 
- How is µ_t set, is this from a particular range, or unbounded?

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
This paper presents a well-structured multi-agent system for stock market simulation, supported by ablation studies and self-reflection mechanisms that help model responses to external events. I find the overall design logical and promising. However, the relatively short simulation period (12 weeks) and the limited focus on large-cap stocks raise questions about how the system would generalize to longer time horizons or to other stock categories, such as mid-cap and small-cap equities. Extending evaluation to broader and longer scenarios would make the contribution more compelling.

### Strengths
1. The three-layer multi-agent design is clearly presented and integrates key informational components necessary for predicting stock price movements.
2. The ablation study is thoughtfully constructed, demonstrating how external events and agent self-reflection mechanisms can influence simulated market trends.
3. The results are encouraging, showing that the system can capture stock trend responses to external events.

### Weaknesses
1. During the experiments, stock names are preserved. It would be helpful to clarify whether this might allow the model to implicitly memorize or exploit prior knowledge about specific stocks, even when the test period is outside the training data.
2. The experiment results from Figure 3 are based on a very short-period simulation. Stock prediction in real practice often relies on longer-term data; therefore, I am not very convinced whether such a short horizon yields meaningful insights for long-term forecasting.
3. The dataset focuses mainly on large-cap stocks, with little to no inclusion of mid-cap or small-cap equities. This limitation raises my concerns about the generalizability of the system.
4. While the proposed system shows promise, the paper does not compare its performance against established baseline models for stock prediction. It would be valuable to discuss how LLM-based agents add unique advantages beyond traditional models (through comparisons) for stock prediction.

### Questions
1. I wonder if the authors have conducted additional experiments on different stock types. In particular, small-cap stocks tend to be more volatile, and how well can the system capture such dynamics?
2. I am also curious to know how the system would perform over a longer time period, given that many investment decisions depend on medium- to long-term company trajectories.

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
4

### Summary
This paper propose a large-scale stock market simulation framework MarketSim. It employs a hierarchical multi-agent architecture, where LLM-based manager agents make strategic decisions and lightweight trader agents perform high-frequency executions. The framework simulates over 15,000 heterogeneous participants with billions of interactions, reproducing key market stylized facts across eight GICS sectors and three shock scenarios, and achieving an average MAPE of 3.48%, demonstrating strong realism and quantitative accuracy.

### Strengths
1. Applying LLMs to stock market simulation is a kind of apply innovative (minor). The multiple agent-based approach moves beyond traditional rule-based frameworks by enabling agents to reason, interpret information, and adapt dynamically, representing a substantial advancement in capturing realistic market behavior and enhancing behavioral fidelity within complex financial environments.
2. The experiments are comprehensive, incorporating diverse datasets and multiple evaluation metrics. Covering eight GICS sectors, three real-world shock scenarios, and both qualitative and quantitative indicators, the study provides robust validation of MarketSim’s performance.
3. By articulating the generalized principle that “fidelity breeds complexity,” the paper establishes a valuable conceptual insight for the field.

### Weaknesses
1. Lack a direct comparison with traditional modeling approaches (e.g., ABIDES) on quantitative metrics such as MAPE and DTW. Without such baselines, it is difficult to objectively assess how much MarketSim improves over existing high-fidelity simulators in terms of predictive accuracy and behavioral realism.
2. The author may need to provide detailed descriptions of the experimental environment. Because the study involves large-scale simulations (15k agents, nanosecond-level CDA, and over 12k text documents), the paper lacks disclosure of key computational details such as GPU/TPU usage, total runtime, and estimated computational cost.
3. The paper’s writing needs further standardization, such as adding a period at the end of the paragraph(minor, Page 9, Applications section).

### Questions
1、 Is there a theoretical or empirical basis for determining the proportion (Table S4) of different participant agent types (e.g., institutional investors, retail traders, market makers)? Or were these ratios primarily treated as hyperparameters and adjusted empirically based on experimental performance?
2、 Will you try running it in a production environment? As far as I know, NoF1 is also an attempt similar to the work described in this paper, although it focuses on the cryptocurrency market. What are the differences and advantages between your approach and theirs?

### Soundness
3

### Presentation
3

### Contribution
3
