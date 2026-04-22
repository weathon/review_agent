# GEM: A Gym for Generalist LLMs

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 2

## Abstract
The training paradigm for large language models (LLMs) is moving from static datasets to experience-based learning, where agents acquire skills via interacting with complex environments. To facilitate this transition we introduce GEM (General Experience Maker), an open-source environment simulator designed for the age of LLMs. Analogous to OpenAI-Gym for traditional reinforcement learning (RL), GEM provides a standardized framework for the environment-agent interface, including asynchronous vectorized execution for high throughput, and flexible wrappers for easy extensibility. GEM also features a diverse suite of environments, robust integrated tools, and single-file example scripts demonstrating using GEM with five popular RL training frameworks. Along with this, we also provide a set of baselines across 24 environments using REINFORCE with Return Batch Normalization (ReBN), which---unlike GRPO---is compatible with the full RL setting of dense per-turn rewards and arbitrary discount factors. We further conduct apple-to-apple benchmarking of PPO, GRPO and REINFORCE in both single- and multi-turn settings using GEM to shed light on the algorithmic designs. GEM also functions as a convenient evaluation toolkit besides a training environment. We hope this framework can help accelerate future agentic LLM research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents GEM, which is a unified and reproducible environment suite for reinforcement learning with LLMs. GEM offers an OpenAI/Gym style API with asynchronous vectorization, tool integration (Python, Search, MCP), and multiple task types (Math, Code, QA, Reasoning, Terminal).
The authors benchmark REINFORCE, PPO, and GRPO, showing their Return Batch Normalization (ReBN) variant improves stability and efficiency across multi-turn tasks. GEM integrates with major RL frameworks and provides evaluation suites like Terminal-Bench and MCPMark. The main strengths are its clean environment API and comprehensive benchmarking; the main drawbacks are no released code and limited technical comparison to existing APIs.

### Strengths
- I love the well-structured, reproducible environment interface modeled after OpenAI Gym. Asynchronous vectorized execution is cool, but I would advise against auto-reset which in my experience provides a very marginal gain and introduces a lot of pain when dealing with last observations of a trajectory. The clear separation between environments and trainers fills a major reproducibility gap left by prior frameworks like Verl and OpenRLHF.
- Enabling turn-level rewards and arbitrary γ is a nontrivial practical and conceptual upgrade, although other frameworks like torchrl already provide this.
- The paper has good benchmarks and ablations. The major frameworks are compared which gives a good intuition of the strengths of the library.

### Weaknesses
- not being able to see the codebase makes me worried about accepting the paper. Reproducibility is the base of any scientific advance, and since the core claim is about standardization and benchmarking, lack of release undercuts the entire point.
- The related work acknowledges VerlTool, Verl-Agent (GiGPO), OpenRLHF, and RL2, but only superficially. Since the paper is about a technical advance and a new standardization of environment APIs, a clear comparison should have be undertaken. It doesn’t discuss TorchRL’s EnvBase or OpenEnv at all — which now also propose standardized step/reset interfaces and multi-process async collection.
- It's unclear if GEM scales beyond small local setups or integrates well with cluster-level runners like Ray / Monarch or others.
- ReBN shouldn't be framed as a new algorithm but as a useful implementation tweak.

### Questions
- Why not releasing the code in an anonymized way? That would help to judge how easy (or not) it would be for the community to adopt it (as well as doc, testing etc)
- Is the vision to make GEM a parallel ecosystem of environments, or build extension points and integrations with other frameworks (recent OpenEnv or extension for VeRL etc)? The ecosystem of RL post-training libraries is already quite crowded, so interoperability with similar frameworks is key.
- Can the authors provide throughput or scaling benchmarks (e.g., vectorized rollout speed, GPU utilization) to substantiate the claim that GEM enables high-throughput training beyond local setups?
- Does GEM does use Ray or other RPC actor services, or any distributed rollout pool? Distributing actors across a cluster is key in settings where the tool / simulator requires a lot of compute resources.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces GEM, a open-source framework designed to facilitate the training and evaluation of agentic RL (LLM-based). Drawing an analogy to OpenAI-Gym's role in traditional RL, GEM provides a standardized environment-agent interface, a diverse suite of multi-turn environments, and features for efficient experimentation like asynchronous vectorized execution. The paper also proposes a simple yet effective baseline algorithm, REINFORCE with ReBN, which is shown to be well-suited for multi-turn settings with dense rewards.

### Strengths
Overall, this paper addresses a critical need for standardized infrastructure of agentic RL. It provides a foundational tool that can accelerate progress and improve the rigor of research in this area. 

1. Comprehensiveness of the GEM Framework: GEM with a clean API, support for asynchronous execution , a diverse suite of tasks (games, reasoning, code, QA), modular tool integration (Python, Search), and extensibility through wrappers. This represents a significant  engineering effort.
2. Rigorous and Insightful Empirical Evaluation: The paper includes a thorough benchmarking of several key RL algorithms.
3. Practicality and Usability: The authors have clearly prioritized usability by providing ready-to-use, single-file integration scripts for five popular LLM RL training frameworks. This significantly lowers the adoption barrier and demonstrates that GEM is agnostic to the training infrastructure.

### Weaknesses
Novelty of ReBN: The core algorithmic proposal, ReBN, is an effective application of a well-known technique (return/advantage normalization) in RL. The paper is transparent about this, but it's worth noting that the primary novelty of the paper lies in the framework (GEM) itself, rather than in a breakthrough RL algorithm.

### Questions
ReBN vs. Advantage Normalization: Could you please elaborate on the empirical or theoretical trade-offs between using ReBN without a critic versus using a learned critic with GAE and advantage normalization? Is the primary benefit of ReBN its implementation simplicity and reduced computational overhead?

Roadmap: This is an excellent starting point for a valuable community resource. Could you share more about your future vision for GEM? Specifically:
1. Do you plan to incorporate more open-ended, sandbox-style environments that involve multi-turn interaction with persistent state and feedback?
2. Have you considered structuring the benchmarks into different difficulty tiers or with specific focuses to allow for more targeted evaluation?
3. Regarding framework integration, beyond individual scripts, have you considered designing a more universal and simple interface or adapter to further lower the cost of connecting to new and existing RL frameworks?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces GEM (General Experience Maker), an open-source environment simulator for agentic RL. Apart from a standardized framework for the environment-agent interface, GEM features a diverse suite of environments, robust integrated tools, and single-file example scripts demonstrating using GEM with five popular RL training frameworks. Additionally, authors also propose REINFORCE with Return Batch Normalization (ReBN), designed for multi-turn agentic LLM tasks.

### Strengths
1. I appreciate the work authors did for building such an integrated environment simulator, which I believe will facilitate future research on agentic LLM.

### Weaknesses
1. The proposed REINFORCE with ReBN, seems like a trivial extention of GRPO to multi-turn tasks. Further practical investivation and theoretical foundation are required to fully justify its effectiveness.
2. While the unification of existing benchmarks is a valuable contribution, this work would have been even more solid and appealing had the authors also analyzed the deficiencies of current benchmarks and built new ones based on that analysis.

### Questions
Please refer to Weaknesses above.

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
4

### Summary
The paper introduces an RL environment interface for LLM agents supporting multi-step (phrased as multi-turn in the paper) tool use. It aggregates tasks including ReasoningGym, text-based games (TextArena), coding, math, and QA, and integrates tools such as Python interpreters, search engines, and the MCP. The authors also benchmark a reinforce with Return Batch Normalization (ReBN) baseline across these environments.

### Strengths
- Engineering quality: The paper/report is well-written and represents a solid engineering effort, particularly in offering a unified interface compatible with five major training frameworks (Oat, Verl, OpenRLHF, ROLL, RL2).
- Clarity: The motivation for transitioning from single-turn to multi-turn agentic workflows is clearly presented.

### Weaknesses
In general, I believe this is a solid engineering work which is helpful to many practitioners, however, the novelty of both the environment and the algorithmic design may limit its contribution to the academia.
- Limited novelty (environments): The contribution is largely an aggregation of existing tasks and datasets (e.g., GSM8K, TextArena, ReasoningGym) rather than a fundamental/novel research contribution in environment design. The differentiation from prior tool-integrated frameworks is minimal, and the scale of environments and experiments are limited.
- Limited novelty/relevance (algorithm): The proposed ReBN algorithm is presented as a contribution, but normalizing returns/advantages is a standard technique in the RL literature and does not constitute an original algorithmic contribution. It may be more meaningful to propose algorithms that focus on the special challenges of multi-step tool use scenarios in the real world.
- Please place related works in the main body of the paper instead of downweighting them in the appendix. It is important to properly give credits to prior works.

### Questions
- Would it be possible to include open-ended environment design with more practical real-world usages in the current framework? If so, If so, what specific engineering/coding improvements would be necessary?
- Training for long-horizon tool use suffers from many specific issues like temporal credit assignment. The proposed ReBN only serves as generic regularization, could the authors discuss more principled algorithmic designs specifically tailored to address more specific challenges inherent in agentic RL?
I am happy to raise my scores if the concerns raised can be sufficiently addressed.

### Soundness
3

### Presentation
3

### Contribution
2
