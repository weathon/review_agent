# DreamPhase: Offline Imagination and Uncertainty-Guided Planning for Large-Language-Model Agents

- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
Autonomous agents capable of perceiving complex environments, understanding instructions, and performing multi-step tasks hold transformative potential across domains such as robotics, scientific discovery, and web automation. While large language models (LLMs) provide a powerful foundation, they struggle with closed-loop decision-making due to static pretraining and limited temporal grounding. Prior approaches either rely on expensive, real-time environment interactions or brittle imitation policies, both with safety and efficiency trade-offs. We introduce DreamPhase, a modular framework that plans through offline imagination. A learned latent world model simulates multi-step futures in latent space; imagined branches are scored with an uncertainty-aware value and filtered by a safety gate. The best branch is distilled into a short natural-language reflection that conditions the next policy query, improving behavior without modifying the LLM. Crucially, DreamPhase attains its performance with substantially fewer real interactions: on WebShop, average API calls per episode drop from $\sim$40 with ARMAP-M (token-level search) to $<10$ with DreamPhase, a $4\times$ reduction that lowers latency and reduces executed irreversible actions by $\sim 5\times$ on WebShop (4.9$\times$ on ALFWorld) per incident logs. Across web, science, and embodied tasks, DreamPhase improves sample efficiency, safety, and cost over search-based and reward-based baselines. This offers a scalable path toward safe, high-performance autonomous agents via imagination-driven planning. Code: \url{https://anonymous.4open.science/r/DreamPhase-A8AD/README.md}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces DreamPhase, an "imagination‑first" planning framework for LLM agents that (i) trains a compact latent world model to simulate short‑horizon futures offline, (ii) rolls out multiple imagined branches from a frozen policy LLM, (iii) scores branches with a value head penalized by an epistemic uncertainty proxy, and (iv) injects a short natural‑language reflection distilled from the best low‑risk branch back into the next policy prompt --- without fine‑tuning the LLM. Empirically, DreamPhase reduces real environment interactions on WebShop while slightly improving success versus a strong token‑level search baseline (ARMAP‑M), with lower per‑step latency and fewer executed irreversible actions; results are reported across diverse tasks and multiple backbones with ablations and a regret‑style bound.

### Strengths
1. The high level latent idea is interesting and the implementation cleanly separates environment modeling (latent world model), decision‑time search (latent rollouts), risk control (uncertainty gating), and behavior shaping (language reflections). This makes the approach applicable to API‑restricted foundation models because the LLM policy remains frozen.

2. On WebShop with Llama‑8B, DreamPhase uses 9.3±0.4 API calls/episode vs 39.8±1.1 for ARMAP‑M, improves success 61.8±0.6 vs 60.2±0.6, and reduces per‑step latency.

3. In an open‑source, matched‑backbone block (Llama‑2‑7B), DreamPhase attains the best average across 8 tasks (50.1 vs 42.3 for ARMAP‑M). Cross‑backbone results (Llama‑3‑70B/8B, Mistral‑7B, Phi‑3‑8B) typically place DreamPhase best or second‑best on WebShop, ScienceWorld (seen/unseen), and Game‑of‑24.

### Weaknesses
1. The LWM uses a small MLP over DOM tokens obtained by DFS traversal with layout features. While efficient, this choice may under‑model structural, hierarchical, and long‑range dependencies present in realistic web UIs; no comparison is shown against stronger sequence/graph models (e.g., Transformers or tree‑aware encoders) that could better respect DOM constraints. How much of DreamPhase's gains persist if the dynamics are more complex than those captured by an MLP?

2. Epistemic uncertainty is estimated via MC‑dropout on the frozen policy LLM. In common LLM serving stacks, dropout is disabled at inference and can be tricky to implement with quantization/kv‑caching

3. DreamPhase adds an offline modeling phase (world model + value head + reflection/summarizer components, with preference pairs produced by a large open model). The paper focuses on runtime savings but provides limited accounting of training compute and data vs baselines that do not require a learned LWM. A small "dollars/GPU‑hours" table would clarify trade‑offs.

### Questions
1. Please report the number of trajectories used to train the LWM per domain, total tokens processed, training wall‑clock/GPU‑hours, and value‑head preference‑generation costs. A normalized "success per training FLOP" or "runtime savings vs pretraining cost" comparison to ARMAP‑M would be informative.

2. Any preliminary results on Mind2Web or WebArena using the same setup and logger? If not feasible, can you discuss expected bottlenecks and what would be required to scale DreamPhase to those environments?

3. The paper mentions a 30‑token budget and domain‑conditioned lexicon/templates. How is $R_{\phi}$ trained, and how does its choice affect injection rate and downstream success?

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
This paper presents DreamPhase, a modular framework for improving large-language-model (LLM) agents through offline imagination. The core idea is to let agents simulate and evaluate possible futures in a learned latent world model instead of interacting with the real environment at every step. Each imagined trajectory is scored by an uncertainty-aware value function, and only high-confidence branches are executed. The best imagined plan is distilled into a short natural-language reflection that is fed back into the frozen LLM policy, steering behavior without any fine-tuning.

The authors provide a theoretical regret bound linking decision quality to model error and mis-gating rate, and show strong empirical results across web, science, and embodied-task benchmarks. DreamPhase reduces real environment calls by roughly 4× on WebShop (≈10 vs. 40 per episode) and lowers latency while matching or surpassing ARMAP and AgentGym baselines. It consistently improves sample efficiency and safety across multiple backbones (LLaMA-3, Mistral, Phi).

Overall, I believe the paper offers a clear and well-motivated step toward safe and cost-efficient planning for LLM agents, combining latent imagination, uncertainty-aware filtering, and natural-language self-reflection.

### Strengths
The paper proposes an offline imagination framework that decouples policy reasoning from environment dynamics via a learned latent world model. Generally speaking, integrating model-based imagination, uncertainty-aware value gating, and natural-language reflections into LLM agents is novel and clearly motivated by limitations of existing online-search or imitation-based methods.

The following is my assessment of the strengths: 
1. Methodological quality. The framework is carefully designed and well-supported by both analysis and implementation details. The latent world model enables safe, counterfactual rollouts; the mutual-information–based uncertainty metric allows principled risk filtering; and the reflection and summarization modules steer a frozen LLM without fine-tuning. The theoretical regret bound $O(T\varepsilon + B\rho T)$ connects decision quality to model accuracy and mis-gating, providing formal grounding rarely seen in this line of work.

2. Empirical strength. Extensive experiments across eight benchmarks and four model backbones (LLaMA-3-70B/8B, Mistral-7B, Phi-3-8B) show consistent improvements in success rate, sample efficiency, and safety. On WebShop, DreamPhase cuts real API calls from ~40 to <10 per episode (≈4× fewer) and reduces irreversible actions by ≈5× while matching or exceeding ARMAP performance.

3. Clarity and presentation. The paper is pretty clearly written, with intuitive motivation, structured algorithms, and readable pseudocode. The modular presentation and ablations make the design easy to follow and reproduce.

4. Significance. By showing that LLM agents can plan safely and efficiently through internal imagination—without additional data collection or fine-tuning—the paper offers a promising path toward scalable, low-cost autonomous systems.

### Weaknesses
While the paper presents a well-motivated and impactful framework for improving the safety and efficiency of LLM agents, several aspects could be further clarified or strengthened:

1. I'm wondering how much the framework depends on latent world model accuracy and low-stochastic environments. DreamPhase’s effectiveness relies on the learned latent world model being a reliable simulator. However, most evaluated environments—WebShop, ScienceWorld, ALFWorld, and Game-of-24—are deterministic or near-deterministic: a given action (e.g., clicking a DOM element, executing a command) leads to a predictable next state with little exogenous noise. This makes short-horizon prediction tractable but may overstate the model’s robustness. In more stochastic settings—such as multi-agent interaction, physical robotics, or open-ended tool use—latent rollouts would likely accumulate compounding error, weakening both the value estimation and the safety gate. It would be interesting to test out the framework in these real-world scenarios, given the paper's claim on uncertainty-aware mechanism and scalability of the method. 

2. Limited generalization scope. Evaluation focuses on web and text-based tasks with structured, language-aligned states. It remains to be seen whether the approach transfers to perceptual or continuous domains, where state representations are high-dimensional and reflections may be less interpretable, e.g., stock market.

3. In general, I would assume the low latency largely relies on the scale of the underlying world model when maintaining a satisfying prediction accuracy. For this reason, it would be beneficial to understand how this method scales to larger application scenarios in real world.

### Questions
My question is already presented above

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper is about leveraging imagined trajectories to enhance the performance of LLMs on multi-turn interactive tasks. More specifically, the authors propose DreamPhase, a framework that interleaves phases of interaction with phases of imagination. During imagination phases, the model carries out its exploration through simulating multiple possible future trajectories of the interaction. Then, based on these imagined trajectories, the proposed system generates textual advice to be concatenated to the current context before asking the frozen LLM to take the next action in the actual environment. In order to generate imagined trajectories, the authors propose to train a separate latent world model to simulate the environment dynamics and some reward function. On web, science and embodied tasks, the authors empirically show the proposed approach is more sample and cost-efficient in terms of API calls and safer (by avoiding irreversible mistakes) while also improving performance.

### Strengths
- While using latent world model for doing exploration is not novel per se, the proposed approach decoupled the interaction policy from the latent world model model. This allows to leverage existing SOTA LLMs by steering their behavior using text summary of high-valued imagined trajectories.
- Once the latent world model is trained, the approach is LLM-agnostic since the LLM used for interaction is frozen.
- Assuming we have a thorough latent world model, exploration in imagined space is safer since mistakes made during imagination do not have real consequences. It is also more cost effective in terms of API calls, i.e. not wasting calls to explore dead ends.
- The empirical evaluation is comprehensive with many ablation studies to validate the approach.

### Weaknesses
- From the main paper, is it unclear what data was used for training the latent world model. Also, in Table 1, is it a fair comparison with other baselines, did they leverage training data as well?
- Unclear how the authors validated the quality of their trained latent world model? They argue that using LLM to act as a simulator is prone to hallucinations, what about the proposed approach? Is there any metrics/ablations to validate this?
- If I understood correctly, the uncertainty estimation requires a bounded set of actions (line 249)? If so, how were those actions obtained for environments with combinatorial action space?

#### Minor
- Citations are "inline" instead of being enclosed with parenthesis.
- In equation 2, I think the two $p_\theta$ should be replaced by $g_\theta$ and $d_\theta$.
- For conciseness, I'd suggest spelling out CE, KL and TV in the text.
- p.4: in the "Offline imagination" paragraph, $p_\theta$ should be replaced by $g_\theta$.
- In algo 1, line 8: $p_\theta$ should be replaced by $g_\theta$
- In algo 2, line 11 and 12: what is "\kappa"? I assume it represents the number of steps in that imagined rollouts?
- I believe SciWorld and ScienceWorld are referring to the same work, yet in the paper they have different name and different citations.

### Questions
- How was the data used to trained the latent world models collected?
- What type of models was used for the latent world model in the experiments? For instance in Table 2, is the backdone only for the agent or both?
- What are the logged data, used to trained the value estimation, exactly?
- What is the frequency of falling back to the real-history policy?
- What is the rational behind reporting the unweighted average across the different environments in table 1 and 2? (nb: on line 375, the text says "tasks" instead of "environments").

### Soundness
3

### Presentation
4

### Contribution
4
