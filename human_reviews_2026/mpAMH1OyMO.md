# MARL2Grid-TR: A Multi-Agent RL Benchmark in Power Grid Operations

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 4, 4

## Abstract
Improving power grid operations is essential for enhancing flexibility and accelerating grid decarbonization. Reinforcement learning (RL) has shown promise in this domain, most notably through the Learning to Run a Power Network (L2RPN) competition series, but prior work has primarily focused on single-agent settings, neglecting the often decentralized, multi-agent nature of grid control.
We fill this gap with MARL2Grid-TR, the first multi-agent RL (MARL) benchmark for grid topology and redispatching, developed in collaboration with transmission system operators. Built on RTE France’s high-fidelity simulation platform, our benchmark supports decentralized control across substations and generators, with configurable agent scopes, observability settings, expert-informed heuristics, and safety-critical constraints.
The benchmark includes a suite of realistic scenarios that expose key challenges, such as coordination under partial information, long-horizon objectives, and adherence to hard physical constraints. Empirical results show that current MARL methods struggle under these real-world conditions. By providing a standardized, extensible platform, we aim to advance the development of scalable, cooperative, and safe learning algorithms for power grids.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes a MARL simulation framework for power grid operations. By using realistic scenarios, it proposes a tool for evaluating 
decentralized control of substations and generators. It supports various configurations, including partial information, delayed rewards, and physical constraints.

### Strengths
- The first standardized MARL formulation for topology optimization and redispatching control, including safety constraints 
- Integration with PETTINGZOO for MARL compatibility and reproducibility.
- Baseline evaluation of state-of-the-art MARL algorithms 
- Simulation environments based on real TSO data and topologies.
- Fills an important gap - since previous power grid RL environments were single-agent. The combination of high-fidelity grid simulation and multi-agent formalism is novel and impactful.

### Weaknesses
- The title is misleading - the paper offers a simulation environment, not a rigorous account of how benchmarking should be performed (i.e., KPIs etc).
- Evaluation focuses mainly on classic CTDE baselines with no ablation studies on the effect of observability regimes (centralized vs. local).
- No explicit evaluation metrics on economic cost, safety violation frequency, or stability margins over time.
- Minor writing issues (redundancy, figure clarity, inconsistent notation in equations).
- Evaluation focuses on different SOTA algorithms, but since the focus is on the simulation, I would expect an account of its limitations.

### Questions
- What are the limitations of the simulation in terms of the algorithms and the settings that it can support? There is an extensive discussion on why the current algorithms fail to address the complexities, but no account of the limitations of the simulation itself. 

- In the context of constraint handling, did you analyze the patterns of violations in different settings, and did you experiment with agent-local constraints?  

- While I accept the CTDE paradigm here, can you speculate on the result of applying a fully decentralized approach?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies the limitation of prior work in power grid operations, which has largely focused on single-agent settings, and proposes MARL2GRID, the first benchmark for multi-agent reinforcement learning (MARL) designed to reflect the decentralized nature of real-world grid control. Built upon the high-fidelity Grid2Op simulator, the benchmark introduces realistic tasks, including discrete topology optimization and continuous redispatching, along with safety-critical constraints. Through experiments, the authors demonstrate that existing MARL algorithms (e.g., MAPPO, QPLEX) struggle significantly with the scalability, coordination, and safety requirements of these tasks, particularly in the combinatorially complex topology optimization setting, thereby highlighting key challenges for future research.

### Strengths
1, The paper correctly identifies a critical gap between existing single-agent RL research (e.g., L2RPN) and the decentralized reality of power grid control, making the proposal of a MARL-specific benchmark both timely and significant. The design, developed in collaboration with Transmission System Operators (TSOs), adds a layer of realism and credibility to the proposed problem setting.

2, The benchmark is well-structured, offering a suite of tasks across various grid scales (bus14, bus36, bus118), two distinct and crucial control paradigms (discrete topology and continuous redispatch), configurable observability regimes, and explicit safety constraints such as load shedding and line overloads (Sections 3, 3.2). This provides a rich and challenging testbed for future MARL research.

3, The empirical results effectively highlight the brittleness of current MARL algorithms when faced with a complex, realistic task. For instance, while MAPPO achieves a 79% survival rate on the bus14 topology task (Table 5), its performance collapses to near-zero on the more complex bus118 grid (Figure 7), starkly illustrating the benchmark's difficulty and motivating the need for novel algorithmic solutions.

### Weaknesses
1, The paper's central premise is that the decentralized nature of power grids necessitates a MARL approach (Lines 014-016). However, this claim is not substantiated with experimental evidence. A critical baseline is fatally absent: a strong, centralized single-agent RL controller (e.g., single-agent PPO) that has access to the full state and action space. Without this comparison, it is impossible to determine whether the proposed multi-agent decomposition (1) offers any performance benefit, (2) is truly necessary for scalability, or (3) merely introduces unnecessary complexity that hinders performance. This is a major flaw in validating the paper's core motivation.

2, The paper successfully demonstrates that current algorithms fail, particularly on large-scale topology optimization (Figure 7). However, it stops at reporting the phenomenon (e.g., "MAPPO failed") without providing a deep diagnostic analysis of why they fail. The root cause—be it challenges in credit assignment under partial observability, inefficient exploration in a combinatorial action space, or poor coordination—remains unexplored. This lack of insight limits the benchmark's utility in guiding future

### Questions
1, The role and efficacy of the "idle heuristic" are unclear. Was this heuristic only applied in the bus118 continuous task, or was it also used for the bus14 topology task where MAPPO performed well? To isolate and validate its contribution, could you provide an ablation study (e.g., MAPPO with vs. without the heuristic on the bus14 task) to quantify its actual impact on performance?

2, To experimentally validate the core motivation for a MARL approach, could you please provide a performance comparison against a centralized single-agent RL baseline (e.g., single-agent PPO) on the bus14 and bus118 topology optimization tasks? This result is crucial for justifying the choice of the MARL paradigm over a simpler, centralized one.

3, Performance drops dramatically from bus14 to bus118. To better understand the scalability limits of current MARL methods, could you provide results for the intermediate-sized bus36 grid in the topology optimization task? This would help clarify whether the performance degradation is gradual or a sharp cliff-edge effect at a certain complexity threshold.

4, The results show that algorithms fail on large-scale tasks (Figure 7). Can you provide a more in-depth diagnostic analysis of these failures? For instance, is the failure primarily due to a lack of coordination, inefficient exploration, or credit assignment challenges under partial observability? A statistical breakdown of which safety constraints are most frequently violated would provide valuable guidance for future research.

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
Modern power grids face increasing complexity due to renewable integration, requiring fast, decentralized control beyond traditional optimization. Prior RL benchmarks (like L2RPN, RL2GRID) model the grid as a single-agent problem, ignoring real-world decentralization. The paper introduces MARL2GRID, the first benchmark for multi-agent RL (MARL) in power grid operations, developed with transmission system operators (TSOs).

### Strengths
-	Power grid control is challenging and important especially with more renewable integration. RL holds a big promiss, and this paper addresses a key gap for benchmarking decentralized MARL for realistic grid operations.
-	This paper is built on industrial-grade Grid2Op simulations with realistic dynamics, long horizons, and stochastic disturbances.
-	This paper considers safety constraint, which is important for grid operations.

### Weaknesses
-	More specifics about the intended power system use case. From my understanding, power system typically have a 3-layered control architecture, with primary control (droop) occurring at the fastest time scale, and then secondary control (AGC), followed by tertiary control. What time-scale is the benchmark and how it would fit into the existing control architecture of power systems?
-	Scale of the benchmark. For applications like power systems, scale matters a lot and I would like to see a much larger scale (in the thousands/10k’s sized) power system benchmarks. 
-	The proposed benchmark also lacks diverse tasks to be named (MARL2GRID). As I mentioned, there are many control problems happening at the grid at different levels, including lower/fast level frequency/voltage control control at generators/inverters, mid level AGC as well as higher level day-ahead planning. For the benchmark to be named MARL2GRID, ideally it should consider more than 1 tasks that are representative of the complex grid operations.

### Questions
I'd like to see a thorough discussion on how RL can be deployed in grid operations. Power grid is a very conservative industry, and it is more or less reluctant to emerging technologies, especially if existing technologies do just fine. I'd like to see some more discussions on what RL can achieve that traditional methods cannot.

### Soundness
3

### Presentation
3

### Contribution
2
