# FinEvo: From Isolated Backtests to Ecological Market Games for Multi-Agent Financial Strategy Evolution

- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Conventional financial strategy evaluation relies on isolated backtests in static environments. Such evaluations assess each policy independently, overlook correlations and interactions, and fail to explain why strategies ultimately persist or vanish in evolving markets. We shift to an ecological perspective, where trading strategies are modeled as adaptive agents that interact and learn within a shared market.
Instead of proposing a new strategy, we present \textbf{FinEvo}, an ecological game formalism for studying the evolutionary dynamics of multi-agent financial strategies.
At the individual level, heterogeneous ML-based traders—rule-based, deep learning, reinforcement learning, and large language model (LLM) agents—adapt using signals such as historical prices and external news. At the population level, strategy distributions evolve through three designed mechanisms—selection, innovation, and environmental perturbation—capturing the dynamic forces of real markets. Together, these two layers of adaptation link evolutionary game theory with modern learning dynamics, providing a principled environment for studying strategic behavior.
Experiments with external shocks and real-world news streams show that FinEvo is both stable for reproducibility and expressive in revealing context-dependent outcomes. Strategies may dominate, collapse, or form coalitions depending on their competitors—patterns invisible to static backtests.
By reframing strategy evaluation as an ecological game formalism, FinEvo provides a unified, mechanism-level protocol for analyzing robustness, adaptation, and emergent dynamics in multi-agent financial markets, and may offer a means to explore the potential impact of macroeconomic policies and financial regulations on price evolution and equilibrium.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors introduce FinEvo, an ecological game framework for evaluating financial trading strategies in dynamic, multi-agent markets. Unlike traditional static backtests, FinEvo simulates diverse ML-based traders that interact and adapt to price data and real-world news, while the overall strategy population evolves through selection, innovation, and environmental change. Experiments incorporating external events demonstrate that FinEvo can reveal emergent behaviors—such as dominance, collapse, and coalition formation—providing deeper insights into strategy robustness, adaptability, and the potential impacts of policies and regulations.

### Strengths
1. The authors introduce FinEvo, an ecological game framework combining heterogeneous traders with evolutionary population dynamics via a stochastic differential equation (SDE). Its core mechanisms—selection, innovation, and environmental perturbation—move evaluation beyond individual returns to the underlying behaviors shaping market dynamics.
2. Under mild conditions, the authors prove minimal theoretical guarantees for the FinEvo SDE, including simplex invariance, positivity, and existence/uniqueness. The authors further derive a macro-level variance decomposition that attributes overall system volatility to the forces of selection, innovation, and environmental perturbation.
3. Experiments with market shocks and real-world signals show that FinEvo uncovers context-dependent performance, robustness, and emergent coalition dynamics that static backtests fail to capture.

### Weaknesses
1. For the simulation results, could the authors provide references or justification for the metrics used to assess how closely the simulation reflects real-world market behavior? Including such references would improve the credibility of the evaluation and make the simulation more convincing.
2. The paper would benefit from a deeper analysis of the rationale behind the chosen simulation scenarios. Explaining the reasoning and context for these scenarios would help readers better understand their relevance and connection to real-world market conditions.

### Questions
I am curious about the rationale for selecting these two specific scenarios. Could the authors explain why these were chosen and whether other possible cases were considered? Providing this context would strengthen the justification for the experimental design.

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
3

### Summary
The paper presents an ecological perspective on financial markets, for modelling the evolutionary dynamics of multi-agent strategies. Rather than analysing strategies in isolation, e.g. through back testing, this provides a way to simulate effects taking into account interdependencies.

### Strengths
- The market dynamic is clearly formalised
- The population evolution is explicit
- A range of simulations and ablations are provided under varying market conditions (bull vs bear) and scenarios

### Weaknesses
On the formalism
- [Uncorrelated] The assumption of uncorrelated payoffs and environment shocks is quite fundamental to all the closed forms provided. However, in realistic scenarios, this is unlikely to be the case. Having the derivations for the limiting case of uncorrelated is okay, but showing something in the correlated case would be more useful.
- [Normal] Environment shocks are assumed to be normally distributed. 
- [Latent params] The formalism relies on knowing N (number of traders) and K (number of strategies), which are often unknown. More crucially, this also seems to rely on knowing the distribution of traders across all strategies, through f_{k,t} (line 106/107).

For simulation: 
- The main limitation here is that it is not clear what is the exact goal to demonstrate in this section. Addressing this is crucial for motivating the approach and experiments (see questions below)
- There is a comparison to ABIDES and PyMarketSim, but it’s just in terms of different aggregated metrics, not e.g. comparing to some realistic baseline.
- N=20 is relatively low for a market. Based on the above comment on params, seeing some sensitivity across N would be useful.

Overall, a clearer motivation and demonstrated use of the approach is required, simply modelling the ecology in a simulated scenario is insufficient

### Questions
Related to the weaknesses above:

- [Uncorrelated] What happens under the case of the correlated payoffs and shocks?
- [Normal] Do we still see the emergence of crucial stylised facts such as fat tails, volatility clustering, etc through the endogenous part of the price setting? If so, this is likely okay assumption, but explicit tests for important stylised facts should be shown.
- [Latent params] Could these be inferred under realistic settings? How sensitivity are results to misspecification?

- [Simulation] What is the exact goal of this section, what is trying to be demonstrated with the approach? Is it the impact of shocks on the market? The effect? To evaluate the introduction of a new strategy under the existing ecology? To model real markets?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes FinEvo, a financial strategy evaluation framework which aims to improve model testing from static, isolated backtests into a dynamic ecological game formalism. The authors suggest markets are modeled as adaptive ecosystems where different kinds of trading agents interact, evolve, and compete.

### Strengths
* A theoretically grounded ecological modeling framework for financial markets and agent-based traders.
* Experimentation across synthetic and empirical environments.
* Integration of game theory, RL, agent models and ecosystem dynamics
* Some potentially interesting observations, especially about system shocks

### Weaknesses
* arbitrary choice of SDEs governing the ecology of the simulations
* hyperparameter choices not validated, so may need excessive tuning
* empirical validity questionable
* economic meaning and usefulness questionable
* alternate approaches not baselined
* baselining against other methods tested not in terms of financially meaningful metrics
* overrelience on synthetic scenarios

### Questions
Though I can see the point of creating a large agent simulation, there is the alternative – which is still dynamic, in that the LOB is altered by a trading agent – of simply making an OB simulator and training a single agent in historic market data, so orders are combinations of real order flow and agent trades. Work like the below considers this and seem very effective.
Frey et al. JAX-LOB: A GPU-Accelerated limit order book simulator to unlock large scale reinforcement learning for trading. ICAIF '23: Proceedings of the Fourth ACM International Conference on AI in Finance. 
Peter Belcak et al. 2022. Fast Agent-Based Simulation Framework with Applications to Reinforcement Learning and the Study of Trading Latency Effects. In Multi-Agent-Based Simulation XXII

There is also some prior work on financial ecosystems, similar in spirit to this work – for example: Mahdavi. Introducing the HFTE Model: A Multi-Species Predator-Prey Ecosystem for High-Frequency Quantitative Financial Strategies. Wilmott, Volume 2017, Issue 89, May 2017
Your agents are conditioned on previous prices alone – presumably just some mid-price. Real trading agents look at order books in depth, and include multiple price levels and – importantly – volume of trades.

The SDE for the value process is a standard drift diffusion SDE, the same as price evolution. I see no reason for this choice. Likewise the SDE for selection, innovation and perturbation is chosen rather than derived. Why this choice?

How are all the hyperparameters in the models set? 

The comparisons with ABIDES and PyMarketSim use "ecological" metrics (entropy, HHI) but not financial performance measures (returns, volatility, or PnL distributions etc). Hence the claim of FinEvo’s superiority is qualitative rather than quantitative.

your “phase change” measure is interesting but undefined in economic terms. is it volatility clustering, or strategy regime switching or what?

You present results for just 1 day of real data. This should be extended to show the approach works in general. Does it also work across markets/assets?

While the paper includes “real-world news” experiments, the price series and market dynamics are synthetic. The mapping from news sentiment to returns lacks empirical grounding - is there a deeper economic understanding? The conclusions about LLM agents’ dominance could therefore be effects of simulation assumptions.

You mention scientific investigation of markets, not building trading algorithms. What other investigation of market dynamics is there if not to build trading models?

The financial realism remains limited: The market clearing and execution mechanisms are simplified. No empirical calibration or validation against actual historical price series beyond qualitative correspondence. Economic feedbacks such as liquidity constraints, transaction costs, or regulatory requirements are minimal. As a result, FinEvo is a conceptual testbed but not yet a predictive or policy-relevant model. How could you extend the approach to offer more realism?

Minor point – protect capitals in references, eg {GPT}

### Soundness
2

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
This study presents a novel examination of the interactive and evolutionary mechanisms among strategies in financial markets. By proposing a multi-agent financial strategy evolution framework termed FinEvo, this paper investigates how populations of heterogeneous agents employing diverse strategies interact and engage in game-theoretic dynamics. Furthermore, the evolutionary dynamics of the population are decomposed into three primary components: selection, innovation, and environmental perturbation.

### Strengths
1. Significance of the Contribution: This paper offers a significant improvement over traditional, isolated strategy backtesting by proposing a comprehensive evolutionary framework. This framework is designed to study strategy interaction and evolution across diverse market environments.

2. Rigorous Theoretical Framework: The paper introduces a set of strategy evolution equations built upon a rigorous mathematical foundation. These equations decompose market dynamics into three interpretable components—selection, innovation, and perturbation—thereby enhancing the model's interpretability.

3. Comprehensive Experimental Design:
 - a. Heterogeneity: The study incorporates four distinct agent types (rule-based, Deep Learning, Reinforcement Learning, and LLM), 
reflecting the realistic composition of modern markets.
- b. Multiple Scenarios: The model is tested across a variety of conditions, including artificial shocks, real-world news events, and different market cycles (bull/bear), which provides comprehensive evidence of its robustness.
- c. Ablation Study: An ablation study targeting the three core mechanisms (selection, innovation, and perturbation) compellingly demonstrates that all three are indispensable for maintaining a realistic and adaptive ecosystem

### Weaknesses
1. Limited Scope of the Evolutionary Framework: While the framework is presented with rigorous definitions, its evolutionary mechanism is predominantly payoff-driven. This approach overlooks micro-level strategy propagation dynamics (such as diffusion through adjacent nodes). Furthermore, the study confines its analysis to competition among a fixed set of predefined strategies, rather than employing methods like genetic algorithms to evolve new, and potentially superior, hybrid strategies.

2. Opaque Initial Parameter Design: The paper lacks a detailed justification for the initial parameter settings of the agents (e.g., learning rates, information window lengths). This omission is critical because, given a high sensitivity to initial conditions, the agents' subsequent performance could be significantly impacted by these choices. This influence may introduce experimental artifacts rather than reflecting the intrinsic dynamics the study aims to capture.

3. Potential Data Contamination for the LLM: The LLM's training window potentially overlaps with the experimental time period. The study's market data includes the bear market phase from January to October 2022. The training corpus of GPT-4o-mini likely contains data from this period, which could provide it with prior information unavailable to the other agents. This 
data

### Questions
See the weakness

### Soundness
3

### Presentation
3

### Contribution
3
