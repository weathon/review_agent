# AlphaAgentEvo: Evolution-Oriented Alpha Mining via Self-Evolving Agentic Reinforcement Learning

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 2, 2, 6, 6

## Abstract
Alpha mining seeks to identify predictive alpha factors that generate excess returns relative to the market from a vast and noisy search space; however, existing evolution-based approaches struggle to facilitate the systematic evolution of alphas. Traditional methods, such as Genetic Programming (GP), cannot interpret natural language instructions and often fail to extract valuable insights from unsuccessful attempts, leading to low interpretability and inefficient exploration. Analogously, without mechanisms for systematic evolution, e.g., long-term planning and reflection, existing multi-agent approaches may easily fall into repetitive evolutionary routines, resulting in inefficient evolution. To overcome these limitations, we introduce AlphaAgentEvo, a self-evolving Agentic Reinforcement Learning (ARL) framework for alpha mining, which moves alpha mining beyond the brittle search-backtest-restart cycle toward a continuous trajectory of evolution. Guided by a hierarchical reward function, our agent engages in self-exploration of the search space, progressively learning basic requirements (e.g., valid tool calls) and then harder objectives (e.g., continuous performance improvements). Through this process, the agent acquires advanced behaviors such as long-horizon planning and reflective reasoning, which enable it to actively react to the underlying state (e.g., market regime shifts) and realize a self-evolving agent, marking a step toward more principled and scalable alpha mining. Extensive experiments demonstrate that AlphaAgentEvo achieves more efficient alpha evolution and generates diverse and transferable alphas, consistently surpassing a wide range of baselines. Notably, with only 4B parameters, it outperforms LLM-driven evolution methods configured with state-of-the-art closed-source reasoning models, highlighting the promise of ARL for next-generation alpha mining.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces AlphaAgentEvo, a framework for quantitative alpha mining that uses Agentic Reinforcement Learning. This agent interacts with a financial backtesting tool over a multi-turn trajectory to progressively refine an initial "seed" alpha. The learning process is guided by a sophisticated, hierarchical reward function that balances tool usage correctness, structural consistency, exploration, performance improvement, and sustained progress. Experiments show that AlphaAgentEvo outperforms traditional Genetic Programming methods and state-of-the-art LLMs in generating alphas with higher success rates.

### Strengths
1. The hierarchical reward function is a key strength. Instead of relying on a single, sparse performance metric, it provides the agent with dense, multi-faceted feedback.
2. AlphaAgentEvo demonstrates superiority over a range of baselines, including GP and large reasoning models like DeepSeek-R1.

### Weaknesses
1. The core technical method is an adaptation of an existing RL algorithm (GRPO) to an agentic setting. The primary innovation lies in the problem formulation and reward engineering. This makes the contribution more of an applied nature than a fundamental advance.
2. True self-evolution might imply an agent that modifies its own learning algorithm or core reasoning architecture, whereas this work uses RL to improve performance on a task. The "evolution" observed is in the alpha's quality, not a fundamental evolution of the agent itself.
3. The experiment is not sufficient. It should be considered to compare models trained on this task. Currently, all the baselines are training-free.

### Questions
1. Why design this method just for alpha mining? Can it be performed on general tasks? What part of the work (except the reward) is specifically designed for alpha mining?
2. Why do you call your method "self"-evolving? It seems that the authors just use RL to train the LLM on the alpha mining task. There's no highlighted "self" part in the method. And the performance increase is evolving?

### Soundness
2

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
3

### Summary
The paper addresses the problem of _alpha mining_ — optimising financial factors (alphas) that demonstrate strong predictive power in markets. To this end, the authors propose _AlphaAgentEvo_, a reinforcement learning–based framework that learns an evolutionary process, essentially mapping an initial seed factor $f_{{seed}}$ to an optimal factor $f^*$, a process they term _alpha evolution_. Through experiments on real-world financial market data (CSI500), the authors report that AlphaAgentEvo outperforms existing baselines. Further analysis indicates that AlphaAgentEvo produces more diverse solutions and demonstrates stronger generalization to unseen data.

### Strengths
1. The paper is clearly written and accessible. Although I am not deeply familiar with the field of alpha mining, it effectively defines both the problem and the proposed solution methodology.

2. The application of RL + LLM evolution to the field of alpha-mining is interesting and novel.

3. The results are promising, particularly given that the evaluation was conducted on real-world, previously unseen financial data.

### Weaknesses
1. The paper argues that alpha evolution is a more effective approach for discovering \( f^* \) than static methods, as it “looks into past feedback.” It further claims that alternative “search–test–restart” strategies (which includes RL)—are “brittle.” However, RL methods, particularly off-policy methods, do in fact leverage prior experience to improve over time. Thus, the characterization of these methods as “naive” seems unconvincing. Further, the paper does not provide any evidence to support the claim of their “brittleness”.

    a. Further, in practice, AlphaAgentEvo uses 3 turns (which is quite low), so I’m curious as to how important the “evolution” is. For this, I suggest an ablation on the number of steps with t = 0 (predict f^* directly), t=1 (f_init -> f^*) and maybe longer horizons, to validate the effectiveness of “evolution”.


2. It is unclear how the reward function is “hierarchical.” From its definition in Eq. 5, it appears to simply combine multiple objectives. Moreover, the use of fractional forms for the first three rewards is not well justified—why not employ a simpler linear combination instead?

    a. (Minor) The reward function introduces several additional hyperparameters. An analysis of their robustness, along with a complete ablation study on all components of the reward function, would strengthen the practical validity of the proposed algorithm.

    b. What is the rationale for enforcing proximity to \( f_{\text{init}} \)? If the objective is solely to identify the optimal alpha \( f^* \), why constrain the search to remain close to the initial factor?

    c. How are the initial factors generated, and what is their quality or performance baseline?

3. (Datasets) The paper provides no details on how the dataset was constructed, which makes reproducibility and thorough analysis of the results difficult.

4. (Evaluation) Why do you use score > f_seed as the main evaluation and not score directly? 

5. While I am not deeply familiar with the specifics of alpha mining, evaluating the model on training data (2023-01-01 to 2024-01-01) is a very poor choice for assessment. Moreover, dividing “seeds” into train, validation, and test sets is unconventional. The standard practice is to perform such splits on the data itself (i.e., train/validation/test data partitions). Therefore, I believe the results presented in Tables 1 and 2 (left), as well as Figure 3(a), should be interpreted with caution or potentially excluded from the evaluation.

6. (Unfair evaluation of baselines) I am concerned that the baseline methods may not have been allocated a comparable training budget. What is the training budget for your proposed method (e.g., 150 steps × k_t)? If so, what value of k_t was used? Was an equivalent budget applied to the baselines? From my understanding, it appears that the baselines have been given a significantly smaller budget (only three turns)

7. The paper lacks a baseline comparison with state-of-the-art LLM-based evolutionary methods, such as FunSearch, AlphaEvolve and EvoTune. Additionally, such work has not been discussed at all.

8. Figure 4: what is direction aware reward?

FunSearch: Romera-Paredes, B., Barekatain, M., Novikov, A. et al. Mathematical discoveries from program search with large language models. Nature

AlphaEvolve: Novikov, Alexander, et al. "AlphaEvolve: A coding agent for scientific and algorithmic discovery." arXiv preprint arXiv:2506.13131.

EvoTune: Šurina, A. et al. Algorithm discovery with large language models: Evolutionary search meets reinforcement learning. Second Conference on Language Modelling

### Questions
1. How important is the number of turns in the training of AlphaAgentEvo?
2. See Weakness 2 b,c, 4, 7 and 8.
3. How are the datasets constructed?
4. What is the training budget used for AlphaAgentEvo and the baselines?

### Soundness
2

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
The paper proposes AlphaAgentEvo, a self-evolving Agentic RL (ARL) framework that turns alpha mining from a brittle search–backtest–restart loop into a multi-turn evolution process guided by a hierarchical reward (tool-use validity, direction-aware consistency to the seed, exploration diversity, performance, and improvement streak). The policy LLM plans, proposes multiple offspring factors per turn, queries a backtesting tool, reflects on outcomes, and updates via a GRPO-style objective adapted to multi-turn, tool-in-the-loop trajectories. Experiments on AlphaEvo500 (new) and Alpha158 across bearish/bullish periods report higher valid ratios and pass rates than GP, multi-agent, and strong LLM baselines, with 1.7B–4B models outperforming larger closed models on several metrics.

### Strengths
1.	Well-motivated paradigm: Precise problem framing of alpha evolution and limitations of GP / prompt-only methods; clear move from single-shot search to multi-turn evolution with reflection. 
2.	Methodical ARL design: A GRPO-style objective extended to multi-turn trajectories and masked tool tokens; multiple offspring per turn with group-normalized advantages. 
3.	Hierarchical reward that encodes domain priors: direction-aware similarity, exploration via AST-based structural diversity, performance and streak bonuses, with caps to prevent domination. 
4.	Strong empirical evidence across two libraries and two regimes, with consistent gains in VR and pass@T; AlphaAgentEvo-4B reaches pass@5 of 0.76/0.94 on AlphaEvo500 and 0.725/0.994 on Alpha158.

### Weaknesses
1.	Many hand-set caps/weights and thresholds (e.g., similarity thresholds, reward caps C*, α*)—no sensitivity study is shown; robustness to these knobs is unclear. 
2.	Heavy use of pass@T and VR; fewer statistics on effect sizes with uncertainty (CI/std) and transaction-cost / turnover analyses. Some violin plots are helpful but more rigorous statistical testing is desirable. 
3.	GP incompatibility with AlphaEvo500 and varying offspring budgets complicate strict apples-to-apples comparisons; stronger head-to-head with exactly matched toolchains would isolate the contribution of ARL vs. retrieval/evaluation differences. 
4.	The method is framed as broadly agentic, but all experiments remain alpha-formula evolution; no results on portfolio construction, risk models, or cross-market transfer.

### Questions
1.	Report compute per trajectory, backtest latency, and throughput relative to baselines at equal offspring budgets.
2.	When do direction-aware and exploration rewards conflict? Please show concrete failure trajectories and mitigations.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces AlphaAgentEvo, a self-evolving Agentic Reinforcement Learning (ARL) framework designed for alpha mining—the process of discovering quantitative trading signals (“alphas”) that predict stock returns. This is the first work to propose a self-evolving agentic reinforcement learning framework for training LLMs for quantitative alpha mining. The performance of training Qwen3-1.7B with the proposed method is significantly better than prompting stronger models including GPT5, and GPT3.5 with the AlphaAgent framework.

### Strengths
1. This is the first work to propose a self-evolving agentic reinforcement learning framework for training LLMs for quantitative alpha mining.

2. The training algorithm is modified on top of GRPO and the rewards are carefully designed with ablation study showing their effectiveness.

3. The performance of training Qwen3-1.7B with the proposed method is significantly better than prompting stronger models including GPT5, and GPT3.5 with the AlphaAgent framework.

### Weaknesses
1. Please define alpha mining in the abstract/introduction.

2. As I'm not familiar with alpha mining, not sure if there are any related training-based LLM baseline for alpha mining (seems to be none). If there is any, should be used as a baseline as well.

3. There is only one training set and two test set used for experiments. More evaluation benchmarks would make the results more convincing.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
