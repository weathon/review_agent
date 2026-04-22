# Monitoring LLM-based Multi-Agent Systems Against Corruption Attacks via Node Evaluation

- Avg Score: 6.00
- Decision: Reject
- Scores: 8, 6, 4

## Abstract
Large Language Model (LLM)-based Multi-Agent Systems (MAS) have become a popular focus of contemporary research, with extensive studies demonstrating their effectiveness in enhancing the performance of individual agents. However, trustworthiness issues in MAS remain a critical concern. Unlike challenges in single-agent systems, MAS involve more complex communication processes, making them susceptible to corruption attacks. To mitigate this issue, several defense mechanisms have been developed based on the graph representation of MAS, where agents represent nodes and communications form edges. Nevertheless, these methods predominantly focus on static graph defense, attempting to either detect attacks in a fixed graph structure or optimize a static topology with certain defensive capabilities. To address this limitation, we propose a dynamic defense paradigm for MAS graph structures, which continuously monitors communication within the MAS graph, then dynamically adjusts the graph topology, accurately disrupts malicious communications, and effectively defends against evolving and diverse dynamic attacks. Experimental results in increasingly complex and dynamic MAS environments demonstrate that our method significantly outperforms existing MAS defense mechanisms as well as single-agent defense approaches.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper tackles the security challenge of corruption propagation in Large Language Model(LLM)-based Multi-Agent Systems(MAS) by proposing a  dynamic defense paradigm. The approach models the MAS as a directed acyclic graph (DAG). Its core technique involves evaluating the contribution of each agent by first assigning a contribution score (-1,0,1) to each communication edge using an LLM independent of the MAS, and then propagating these scores backward through the graph via a PageRank-like algorithm. Agents whose contribution scores significantly deviate from the group are identified as malicious nodes and subsequently isolated by pruning their outgoing communication edges.
Through experiments, the authors demonstrate that their method outperforms several existing defense baselines in both task accuracy and malicious agent detection rate, showing particularly prominent performance in dynamic attack scenarios.

### Strengths
- Originality: The paper introduces a dynamic defense paradigm, using a signed graph and backpropagation to evaluate node contributions, which is a creative approach.
- Quality: The method is well-structured and validated through comprehensive experiments across multiple MAS architectures, benchmarks, and attack types.
- Clarity: The paper is clearly written, with a logical flow that effectively introduces the problem, methodology, and contributions. The presentation of figures and numerical results is organized and easy to follow.

### Weaknesses
W1 Subjective and Coarse-Grained LLM Scoring: The core mechanism relies on an external LLM to assign a simplified ternary contribution score (-1,0,1) to each communication edge. Although the prompt is specified, the scoring task itself is inherently subjective. Reducing complex multi-agent interactions to a ternary classification (disagreement, agreement, neutral) represents a crude simplification that may fail to capture subtle adversarial strategies. The system's reliability is fundamentally tied to the scorer LLM's ability to make consistent and accurate judgments on this ambiguous task.

W2 Computational Overhead and Scalability: While the introduction criticizes the computational overhead of baseline methods like G-Safeguard, the proposed method itself requires invoking an LLM for every edge in the graph during each evaluation round. It is reasonable to believe this could lead to significant latency and cost in large-scale MAS with extended dialogues. The paper's claims regarding efficiency lack support, as experiments are limited to small-scale MAS and provide no quantitative analysis of computational overhead.

Minor Presentation Issues: "out put" on Page 4 should be "output".

### Questions
The detection mechanism identifies malicious agents as statistical outliers based on their contribution scores. However, the defense operates with an inherent lag, as it requires multiple rounds of communication to compute. In a scenario where a highly contagious attack rapidly propagates and corrupts a majority of agents, could the system mistakenly identify the remaining, uncorrupted benign agents as the statistical outliers and falsely flag them as malicious? How does the method ensure robustness in such a "majority infected" scenario?

### Soundness
3

### Presentation
3

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
The paper introduces a dynamic defense for LLM-based multi-agent systems: model the conversation as a time-unfolded signed DAG, back-propagate contribution scores to spot corrupted agents, and cut their edges on the fly. Experiments show it detects ≥93 % of attacks and recovers 7–10 % task accuracy, outperforming static or local baselines.

### Strengths
1.Node-level signed graph with back-propagation breaks the static-topology bottleneck in MAS defense.
2.Method is model-agnostic and implemented as a lightweight plug-in requiring no extra training.
3.Provides in-depth analysis of edge scoring, contribution deviation, and runtime memory overhead.

### Weaknesses
1. Edge-evaluation LLM may itself be biased or attacked, yet its robustness is assumed rather than tested.

2. Single fixed threshold ε for all agents/domains lacks adaptive calibration and could incur false positives/negatives.

3. Additional LLM calls per edge raise latency and cost, leaving scalability to very large agent pools unaddressed.

### Questions
1. Gap Shrinkage: In Table 1 and Table 5 the accuracy margin between the proposed method and the best baseline is only 1–3 pp on most cells; is this small delta sufficient to claim a clear superiority, or does it simply fall within run-to-run variance?


2. Runtime Cost: Each inference requires one extra LLM call per message for edge scoring plus a full backward pass—what is the measured latency/memory increase relative to an undefended MAS, and how does it scale as the number of agents and dialogue rounds grow?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses security in LLM-based multi-agent systems by treating agent interactions as a time-unrolled DAG and scoring each message with an external LLM to see whether it helps or harms the final task. These local edge scores are then propagated backward through the graph so every agent gets a global contribution score, letting the system spot agents whose behavior consistently deviates. Detected agents have their outgoing edges cut, giving a dynamic defense that adapts as attacks evolve rather than relying on a fixed topology. Experiments across tasks, LLMs, and attack types show this beats prior MAS defenses, especially on subtle “modification” attacks.

### Strengths
1. propose to model evolving agent graphs instead of fixed topology.
2. Graph backpropagation mechanism for attribution. Turning per-edge LLM judgments into a global contribution score via backward propagation over the DAG is a neat, principled way to answer “which agent actually pushed the system off course?”.
3. Broad, heterogenous evaluation. They test across multiple MAS structures, LLM backbones, tasks, and attack styles, and show consistent gains over G-Safeguard / AgentXposed / Inspector, which strengthens the generality claim.

### Weaknesses
1. LLM-as-judge dependency. The whole pipeline leans on an external LLM to label edge contributions (−1,0,1); if that judge is noisy or biased, the attribution and thus the defense can collapse. Also, if the attacker is steathy or lauched gailbreak against this LLM, the defense could aslo break.
2. Missing computation overhead of maintaining and backproping dynamically changing graphs.
3. Missing analysis over the hyperparam threshold

### Questions
Please see Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
