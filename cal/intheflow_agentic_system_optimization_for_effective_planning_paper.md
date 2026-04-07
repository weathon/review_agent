# IN-THE-FLOW AGENTIC SYSTEM OPTIMIZATION FOR EFFECTIVE PLANNING AND TOOL USE


**Zhuofeng Li** _[∗]_ [1] _[,]_ [2], **Haoxiang Zhang** _[∗]_ [1] _[,]_ [3], **Seungju Han** [1], **Sheng Liu** [1], **Jianwen Xie** [4],
**Yu Zhang** [2], **Yejin Choi** [1], **James Zou** _[†]_ [1], **Pan Lu** _[∗†]_ [1]

1Stanford University, 2Texas A&M University, 3UC San Diego, 4Lambda


**Website:** **[https://agentflow.stanford.edu](https://agentflow.stanford.edu)**


[Code](https://github.com/lupantech/AgentFlow) [Model](https://huggingface.co/AgentFlow/models) [Demo](https://huggingface.co/spaces/AgentFlow/agentflow) [Visualize](https://agentflow.stanford.edu/#visualization)


ABSTRACT


Outcome-driven reinforcement learning has advanced reasoning in large language
models (LLMs), but prevailing tool-augmented approaches train a single, monolithic policy that interleaves thoughts and tool calls under full context; this scales
poorly with long horizons and diverse tools and generalizes weakly to new scenarios. Agentic systems offer a promising alternative by decomposing work across
specialized modules, yet most remain training-free or rely on offline training decoupled from the live dynamics of multi-turn interaction. We introduce AGENTFLOW, a trainable, _in-the-flow_ agentic framework that coordinates four modules
(planner, executor, verifier, generator) through an evolving memory and directly
optimizes its planner inside the multi-turn loop. To train on-policy in live environments, we propose _Flow-based Group Refined Policy Optimization_ (Flow-GRPO),
which tackles long-horizon, sparse-reward credit assignment by converting multiturn optimization into a sequence of tractable single-turn policy updates. It broadcasts a single, verifiable trajectory-level outcome to every turn to align local planner decisions with global success and stabilizes learning with group-normalized
advantages. Across ten benchmarks, AGENTFLOW with a 7B-scale backbone
outperforms top-performing baselines with average accuracy gains of 14.9% on
search, 14.0% on agentic, 14.5% on mathematical, and 4.1% on scientific tasks,
even surpassing larger proprietary models like GPT-4o. Further analyses confirm
the benefits of in-the-flow optimization, showing improved planning, enhanced
tool-calling reliability, and positive scaling with model size and reasoning turns.


AgentFlow (w/o Flow-GRPO) AgentFlow

**Bamboogle**


**2Wiki (Search)** **HotpotQA (Search)** **GAIA (Agentic)**


**AIME24 (Math)** **GameOf24 (Math)** **GPQA (Science)**


Figure 1: **Left:** Performance of AGENTFLOW with a 7B-scale backbone before and after FlowGRPO tuning across ten diverse reasoning benchmarks. Flow-GRPO substantially improves performance by enhancing planning quality and tool-calling reliability. **Right:** AGENTFLOW achieves
consistent gains over top baselines, including base LLMs, tool-integrated RL models, and trainingfree agentic systems. All 7B results use Qwen2.5-7B-Base/Instruct as the backbone and tools.


*Equal contribution. _[†]_ Co-senior authors. Work was partially done while ZL and HZ were visiting Stanford.


1


1 INTRODUCTION


Recent advances in large language models (LLMs) have unlocked remarkable reasoning capabilities,
largely driven by reinforcement learning (RL) from outcome-based feedback. By fine-tuning models
to maximize verifiable rewards, LLMs like DeepSeek-R1 (Guo et al., 2025) and SimpleRL (Zeng
et al., 2025b) have demonstrated sophisticated behaviors in self-correction and multi-step deduction.


A complementary line of work augments LLMs with external tools (e.g., web search, code execution) for knowledge retrieval and precise computation. Tool-integrated reasoning (TIR) extends
reinforcement learning with verifiable rewards to learn _when_ and _how_ to call tools by interleaving reasoning (e.g., <think>) with tool invocations (e.g., <tool call>) under full context (Jin
et al., 2025; Song et al., 2025; Chen et al., 2025; Feng et al., 2025). Early systems supported only
a single tool type, whereas recent work enables multi-tool settings by encoding tool metadata into
prompts (Dong et al., 2025; Qian et al., 2025a; Zhang et al., 2025). However, these methods still
train a _single_, monolithic policy under multi-turn full-context reasoning, which introduces scaling
challenges: (i) _training_ becomes increasingly unstable as horizons lengthen, tool diversity grows,
and environments shift with tool feedback (Wang et al., 2025c; Mai et al., 2025; Moonshot AI, 2025;
Xue et al., 2025); and (ii) _inference_ -time generalization remains brittle to unseen tasks or tools (Dong
et al., 2025; Hu et al., 2025b).


Agentic systems (Wu et al., 2024; Hong et al., 2024; Hu et al., 2025b) offer a promising alternative to monolithic tool-integrated reasoning models. They consist of multiple modules—often
distinct LLMs with prescribed roles (e.g., planner, critic) or specialized components with dedicated
tools and capabilities (e.g., executor, coder)—that coordinate via shared memory and inter-module
communication. By decomposing problems into sub-goals and iterating over multiple turns, these
systems can tackle tasks that demand diverse tools, long horizons, or multi-stage reasoning. However, achieving robust coordination in such systems ultimately requires _training_, since handcrafted
logic or static prompting cannot reliably capture when and how modules should collaborate, adapt to
evolving tool outputs, or recover from early mistakes. At the same time, they introduce new _training_
challenges: modules coordinate sequentially, outcome feedback propagates through long reasoning
chains, and state distributions shift with evolving tool outputs. As a result, most systems remain
_training-free_, relying on handcrafted logic or prompting heuristics. While some employ supervised
fine-tuning or preference optimization for key modules (Motwani et al., 2024; Park et al., 2025),
these off-policy approaches are decoupled from live dynamics and learn poorly from downstream
successes or failures. Thus, agentic systems struggle with sparse rewards, brittle adaptation, and
inefficient orchestration in dynamic environments.


To address the central challenge of learning long-horizon reasoning with sparse rewards in toolintegrated agentic systems, we introduce AGENTFLOW, a _trainable_ framework for effective planning and tool use (Figure 2). AGENTFLOW comprises four specialized modules—planner, executor,
verifier, and generator—that interact iteratively over multiple turns via a shared evolving memory
and a toolset. The system operates _in_ _the_ _flow_, with each turn cycling through planning, execution, and verification. Unlike prior agentic systems, AGENTFLOW directly optimizes its planner
on-policy, _inside_ the live multi-turn loop, allowing it to dynamically adapt to trajectories shaped by
tool calls, verifier signals, and memory updates. This evolving memory serves as a deterministic,
structured record of the reasoning process, enabling transparent state tracking, controllable behavior,
and bounded context growth.


To train the planner on-policy within this agentic system, we need to overcome the long-horizon
credit assignment problem inherent to sparse, trajectory-level rewards. We introduce _Flow-based_
_Group_ _Refined_ _Policy_ _Optimization_ (Flow-GRPO, Figure 4), an on-policy algorithm designed for
this setting. Flow-GRPO operates on _in-the-flow_ rollouts, which capture the full trajectory of states,
actions, and tool events induced by the live system. Instead of attempting to assign credit with brittle, intermediate heuristics, we assign a single, verifiable final-outcome reward to the entire trajectory and _broadcast_ it to every turn. This design effectively transforms the multi-turn reinforcement
learning challenge into a series of single-turn updates: at each turn, the planner has access to the full
memory context and receives a consistent reward signal aligned with global success. This approach,
coupled with group-normalized advantages to stabilize training, enables robust credit assignment
and allows the planner to learn effective long-horizon strategies from sparse feedback.


We evaluate AGENTFLOW on ten benchmarks across diverse reasoning domains, as results highlighted in Figure 1. AGENTFLOW substantially outperforms top-performing specialized tool

2


|𝑞 𝐾 𝑀𝑡|Col2|Col3|Col4|
|---|---|---|---|
|𝑞<br>𝐾<br>𝑀𝑡|𝑞<br>𝐾<br>𝑀𝑡|𝑞<br>𝐾<br>𝑀𝑡||
|Plan|ne|r<br>𝜋||
|𝑎𝑡<br><br>|𝑎𝑡<br><br>|𝑎𝑡<br><br>|𝑎𝑡<br><br>|


|𝑎𝑡 𝐾|Col2|Col3|
|---|---|---|
|E|xecut|or|
|𝑒𝑡<br>|𝑒𝑡<br>|𝑒𝑡<br>|


**Input:**

[Generated Command]

[Execution Result]

**Output:**

[Execution Analysis]

[Memory Analysis]

[Verification Status]


Trained


Frozen


**Input:**

[Query Analysis]

[Global Goal]

[Required Skills]

**Output:**

[Current Sub-Goal]

[Selected Tool]

[Context for Tool Use]


**Input:**

[Current Sub-Goal]

[Selected Tool &
Context]

[Tool Metadata]

**Output:**

[Generated Command]

[Execution Result]


(a) AgentFlow: In-the-Flow Agentic System


(b) In-the-Flow Rollout at Turn _t_


Figure 2: **(a)** Overview of AGENTFLOW, a trainable agentic system for in-the-flow planning and tool
use. Four modules (planner, executor, verifier, generator) coordinate via a shared evolving memory
_M_ and toolset _K_, given a query _q_ . The planner policy is optimized on-policy _inside_ the system’s
multi-turn loop to enable adaptive, long-horizon reasoning. **(b)** A single state transition, showing
the action _a_ _[t]_, execution result _e_ _[t]_, and verifier signal _v_ _[t]_ that update the memory from _M_ _[t]_ to _M_ _[t]_ [+1] .


integrated reasoning models and agentic systems, achieving average accuracy by 14.9% on
knowledge-intensive search, 14.0% on broader agentic tasks, 14.5% on mathematical reasoning, and
4.1% on scientific reasoning (§4.2). Notably, our 7B-backbone system even surpasses the _∼_ 200Bparameter GPT-4o (Hurst et al., 2024) across all domains. Further analyses confirm that our inthe-flow optimization with Flow-GRPO is crucial, far surpassing offline supervised tuning (§4.3).
The trained planner learns to optimize planning, enhance tool-calling reliability, and discover effective solution pathways (§4.5). Moreover, our training approach proves highly efficient, leading
to increased rewards and condensed responses compared to traditional tool-integrated RL methods
(§4.6). Finally, we demonstrate that these benefits generalize, with consistent gains from scaling
backbone size and turn budget (§4.4).


Our work makes three key contributions: (1) We present AGENTFLOW, a trainable _in-the-flow_ agentic system that directly optimizes its planner _inside_ the multi-turn loop. By coordinating specialized
modules through an evolving memory, it enables adaptive long-horizon planning and robust tool
orchestration. (2) We introduce _Flow-GRPO_, an on-policy, outcome-driven algorithm that hat _con-_
_verts_ multi-turn RL into a sequence of tractable _single-turn_ policy updates by _broadcasting_ a single, verifiable final-outcome reward to every turn. (3) Through comprehensive experiments on ten
benchmarks, we show that AGENTFLOW with a 7B backbone outperforms specialized baselines and
even larger proprietary models. Further analyses reveal improved planning, enhanced tool-calling
reliability, and positive scaling with model size and turn budgets.


2 PRELIMINARY


**Reinforcement learning for reasoning LLMs.** Recent progress in reasoning LLMs has been significantly driven by reinforcement learning from outcome feedback, using a verifiable reward signal (Shao et al., 2024; Yu et al., 2025). This paradigm fine-tunes a language model to maximize
an outcome-based reward while remaining close to a reference policy. Formally, the objective is to
optimize a policy LLM _πθ_ to generate a response _o_ for a given query _q_ from dataset _D_ :
max _πθ_ E _q∼D, o∼πθ_ ( _·|q_ )� _R_ ( _q, o_ )� _−_ _β_ DKL( _πθ_ ( _o | q_ ) _∥_ _π_ ref( _o | q_ )) _,_ (1)


where _R_ ( _q, o_ ) is the outcome-based reward, _π_ ref is a reference model to prevent policy collapse, and
_β_ controls KL regularization. Algorithms like Group Relative Policy Optimization (GRPO) (Shao
et al., 2024) implement this by sampling groups of responses, normalizing advantages by their rewards, and updating the policy with a clipped objective to encourage high-reward outputs.


**Tool-integrated reasoning models (LLM agents).** LLMs can be augmented with external tools
to access knowledge and perform precise computation under reinforcement learning with outcomebased reward. As shown in Figure 3(a), the LLM _interleaves_ reasoning and tool calls, producing a chain of thought within <think></think> tokens followed by tool invocations (e.g.,
<tool ~~c~~ all></tool ~~c~~ all>). The resulting trajectory _τ_ is a sequence of model generations
and tool observations: _τ_ = _{s_ [1] _, a_ [1] _, e_ [1] _, . . ., s_ _[T]_ _, a_ _[T]_ _}_, where _s_ _[t]_ denotes the context, _a_ _[t]_ the generated
action (thought + tool call), and _e_ _[t]_ the tool’s execution result. The policy model _πθ_ is then trained to
maximize a final outcome reward. Prior work has explored single- and multi-tool settings for search
and code execution (Jin et al., 2025; Chen et al., 2025; Feng et al., 2025; Qian et al., 2025a).


3


Trained


Frozen


token


(a) Tool-Integrated Reasoning Models (LLM Agents) (b) Training-Free Agentic Systems


Figure 3: **Comparison of two paradigms of LLMs with tool use.** (a) Monolithic tool-integrated
reasoning models train a single policy to interleave reasoning (e.g., <think>) and tool calls (e.g.,
<tool ~~c~~ all>) within a single, full-context trajectory. (b) Agentic systems decompose tasks across
multiple specialized modules (e.g., planner, coder) that collaborate. These systems are typically
training-free, orchestrated by handcrafted logic or prompting.


**Agentic systems with tool usage.** An alternative approach is the use of agentic systems (Wu et al.,
2024; Hong et al., 2024; Lu et al., 2025). As shown in Figure 3(b), these frameworks deploy multiple specialized modules—often distinct LLMs with carefully designed prompts and roles—within
a collaborative workflow. By decomposing tasks and assigning subproblems to modules with dedicated tools and capabilities (e.g., planner, coder, critic), they can address complex problems such as
web browsing, document processing, and multi-stage programming that exceed the scope of a single model. A central limitation, however, is that these systems are typically _training-free_ : modules
remain frozen pre-trained models orchestrated by handcrafted logic or prompting heuristics.


3 IN-THE-FLOW AGENTIC SYSTEM OPTIMIZATION


We aim to bridge the gap between trainable but monolithic reasoning models and flexible yet static
agentic systems. We present AGENTFLOW, a flexible and trainable agentic system that integrates
four specialized modules with an evolving memory (§3.1). Unlike prior agentic systems, AGENTFLOW directly optimizes the planner _within_ the multi-turn loop of an agentic system (§3.2).


3.1 AGENTFLOW: AN IN-THE-FLOW AGENTIC SYSTEM


We propose AGENTFLOW, a general-purpose tool-integrated agentic framework for solving complex reasoning tasks through fine-grained planning and effective tool use within a multi-turn architecture. As shown in Figure 2, the framework comprises four specialized modules— **Action Planner**
_P_, **Tool Executor** _E_, **Execution Verifier** _V_, and **Solution Generator** _G_ —coordinated by a shared
evolving memory _M_ and a toolset _K_ . These modules interact sequentially and iteratively to perform _action planning_, _tool execution_, _context verification_, and _solution generation_, thereby enabling
tool-integrated reasoning across multiple turns.


We formalize AGENTFLOW’s problem-solving process as a multi-turn Markov Decision Process
(MDP). Given a query _q_ and a toolset _K_, the system proceeds for a variable number of turns. Let
_M_ _[t]_ denote the memory state before turn _t_ (with _M_ [1] initialized from _q_ ). At turn _t_, the planner _P_ (a
trainable policy _πθ_ ) formulates a sub-goal, selects an appropriate tool _k_ _∈_ _K_, and retrieves relevant
context from memory, producing an action: _a_ _[t]_ _∼_ _πθ_ ( _a_ _[t]_ _| q, K, M_ _[t]_ ).


The executor _E_ invokes the chosen tool with context, yielding an execution observation _e_ _[t]_ _∼E_ ( _e_ _[t]_ _|_
_a_ _[t]_ _, K_ ). The verifier _V_ then evaluates whether _e_ _[t]_ is valid and whether the accumulated memory is
sufficient to solve the query, producing a binary verification signal _v_ _[t]_ _∼V_ ( _v_ _[t]_ _| q, e_ _[t]_ _, M_ _[t]_ ). If _v_ _[t]_ = 0,
the memory is updated deterministically to incorporate new evidence: _M_ _[t]_ [+1] = _f_ mem( _M_ _[t]_ _, a_ _[t]_ _, e_ _[t]_ _, v_ _[t]_ ),
where _f_ mem( _·_ ) denotes the memory-update function, which records agent-process information in a
concise, structured form along with contextual details such as time, turn index, and error signals.


The process repeats until _v_ _[t]_ = 1 (termination) or a predefined maximum turn budget is reached.
Upon termination at turn _T_, the solution generator _G_ produces the final solution _o_, conditioned on
the query and the accumulated memory: _o ∼G_ ( _o | q, M_ _[T]_ ).


4


Multi-turn Agentic System Rollouts


Figure 4: **Optimization** **for** **our** **proposed** **agentic** **system** **AGENTFLOW.** Given a query _q_, an
evolving memory _M_, and a toolset _K_, the policy model generates actions that target sub-goals and
select tools. It is trained via _Flow-based Group Refined Policy Optimization_ (Flow-GRPO), which
enables multi-turn reinforcement learning and stable optimization under collaborative dynamics.


This formulation decomposes multi-turn, tool-integrated reasoning into structured, observable transitions. After _T_ turns, the trajectory _τ_ = _{_ ( _a_ _[t]_ _, e_ _[t]_ _, v_ _[t]_ ) _}_ _[T]_ _t_ =1 [records the history of planning, execution,]
and verification. The joint generative process can be written as


_G_ ( _o | q, M_ _[T]_ ) _,_


 -  _pθ_ _{a_ _[t]_ _, e_ _[t]_ _, v_ _[t]_ _}_ _[T]_ _t_ =1 _[,]_ _[o][ |][ q, K]_ =


- _T_

 


- _πθ_ ( _a_ _[t]_ _| q, K, M_ _[t]_ ) _E_ ( _e_ _[t]_ _| a_ _[t]_ _, K_ ) _V_ ( _v_ _[t]_ _| q, e_ _[t]_ _, M_ _[t]_ )


_t_ =1


(2)
where _{a_ _[t]_ _, e_ _[t]_ _, v_ _[t]_ _}_ _[T]_ _t_ =1 [are explicit realizations of the latent reasoning chain. Importantly, unlike latent]
thoughts behind trajectories, our memory _M_ is an explicit and deterministic record of the reasoning
process, ensuring transparency and controllability of multi-turn decisions.


3.2 IN-THE-FLOW REINFORCEMENT LEARNING OPTIMIZATION


We target tool-integrated _agentic_ _systems_ operating under _long-horizon_ tasks with _sparse_ rewards.
In this setting, the **Action** **Planner** (the trainable policy of AGENTFLOW) selects a _sequence_ of
interdependent actions while the state ( _q, K, M_ _[t]_ ) evolves with tool results and verifier feedback.
Conventional _offline_ training—e.g., supervised fine-tuning or preference fine-tuning on curated
traces—optimizes the planner _outside_ the active loop (Motwani et al., 2024; Park et al., 2025).
This decoupling prevents real-time coordination with the executor, verifier, and solution generator,
induces distribution shift between training and deployment, and provides limited guidance about
_which_ intermediate decisions truly matter. As a result, planners often adapt poorly to multi-turn
dynamics; early errors cascade, and post-hoc fixes are brittle.


**In-the-flow** **learning.** To address these issues, we optimize the planner _in_ _the_ _flow_ of execution.
We roll out the full AGENTFLOW system under the current policy, collect the actual trajectory _τ_
of states, actions, and tool events it induces, and update the policy within the agentic system using
a verifiable final-outcome signal. This exposes the multi-turn credit-assignment problem directly
and trains the planner on the exact states it will face at inference. Our objective, Flow-GRPO, is
designed to stabilize learning under sparse, trajectory-level rewards over multiple turns.


As established in §3.1, rollouts in AGENTFLOW define a finite-horizon MDP with a variable horizon
_T_ . At turn _t_, the planner observes the state ( _q, K, M_ _[t]_ ), selects an action _a_ _[t]_, the executor and verifier
return ( _e_ _[t]_ _, v_ _[t]_ ), and the memory updates deterministically to _M_ _[t]_ [+1] .


**Policy optimization objective.** The planner policy _πθ_ is trained to maximize the expected return
over on-policy rollouts. Let _R_ ( _τ_ ) be the reward for a complete trajectory _τ_ . The objective is:

_J_ ( _θ_ ) = E _τ_ _∼πθ_        - _R_ ( _τ_ )� _,_ _θ_ _[⋆]_ = arg max _θ_ _J_ ( _θ_ ) _,_ (3)

where a rollout _τ_ is the sequence of decisions _{a_ _[t]_ _}_ _[T]_ _t_ =1 [generated on-policy by] _[ π][θ]_ [.]

**Final-outcome** **reward.** Assigning credit to intermediate actions is challenging because each _a_ _[t]_
influences the final solution only indirectly, and their value may only emerge after several turns (e.g.,
error or improvement accumulation). To avoid brittle local feedback, we adopt a _final-outcome-_
_based_ _reward_ : every action within a rollout receives the same global reward signal, based on the
correctness of the final solution _o_ with respect to query _q_ and ground truth _y_ _[∗]_ :

_r_ = _R_ ( _a_ _[t]_ ) = _R_ [¯] ( _o, q, y_ _[∗]_ ) _,_ _∀t_ = 1 _, . . ., T,_ (4)


5


where _R_ [¯] ( _o, q, y_ _[∗]_ ) _∈{_ 0 _,_ 1 _}_ is assigned by an LLM-as-judge rubric for semantic, numeric, and
option-level equivalence (see §E.3). This propagates a trajectory-level success signal back through
the reasoning chain, aligning every decision _a_ _[t]_ with global correctness.


**Objective function.** We formalize **Flow** -based **G** roup **R** efined **P** olicy **O** ptimization for the planner. The goal is to optimize the policy _πθ_ by maximizing the expected return over a group of parallel
rollouts. For each query-label pair from training corpus ( _q, y_ _[∗]_ ) _∼D_, we sample a group of _G_ onpolicy trajectories _{τi}_ _[G]_ _i_ =1 [by running the current behavior policy] _[ π][θ]_ old [inside A][GENT][F][LOW][, where]
_τi_ = _{a_ [1] _i_ _[, ....a]_ _i_ _[T][i][, o][i][}]_ [.] [Let] _[s][t]_ _i_ [=] [(] _[q, K, M][ t]_ _i_ [)] [be] [the] [state] [at] [turn] _[t]_ [of] [rollout] _[i]_ [,] _[a][t]_ _i_ [the] [planner’s] [ac-]
tion (a token sequence of length _|a_ _[t]_ _i_ _[|]_ [), and] _[ o][i]_ [the final response.] [This structure is key to addressing]
the long-horizon credit assignment challenge: by broadcasting a single trajectory-level reward to
all turns, we effectively decompose the _multi-turn RL_ problem into _a set of independent, single-turn_
policy updates; we provide a formal proof of this equivalence and analyze its convergence properties
in §B. Each update for an action _a_ _[t]_ _i_ [is conditioned on the full historical context encapsulated in the]
state _s_ _[t]_ _i_ [and receives the same global success signal, simplifying optimization.] [The objective is]


_J_ Flow-GRPO( _θ_ ) = E( _q,y∗_ ) _∼D,_ _{τi}Gi_ =1 _[∼][π][θ]_ old

     - _G_ _Ti_ _|a_ _[t]_ _i_ _[|]_
1       - 1       - 1       _G_ _Ti_ _|a_ _[t][|]_


_,_


_G_


_i_ =1


_|a_ _[t]_ _i_ _[|]_


_ai_ 
- min� _ρ_ _[t]_ _i,j_ _[A][t]_ _i_ _[,]_ [clip(] _[ρ][t]_ _i,j_ _[,]_ [1] _[ −]_ _[ϵ,]_ [1 +] _[ ϵ]_ [)] _[ A][t]_ _i_ - _−_ _β_ DKL� _πθ ∥_ _π_ ref� _,_


_j_ =1


1
_Ti_


_Ti_


_t_ =1


1
_|a_ _[t]_ _i_ _[|]_


(5)
where _Ti_ is the (variable) number of turns in rollout _i_, and


                    - _a_ _[t]_ _i,j_ �� _sti_ _[, a][t]_ _i,_ 1: _j−_ 1�
_ρ_ _[t]_ _i,j_ [=] _π_ _[π]_ _θ_ _[θ]_ old� _a_ _[t]_ _i,j_ �� _sti_ _[, a][t]_ _i,_ 1: _j−_ 1� (6)


is the token-level importance ratio for the _j_ -th token of _a_ _[t]_ _i_ [,] _[ ϵ >]_ [ 0][ is the PPO clipping parameter, and]
_β_ _>_ 0 controls the KL penalty to a fixed reference policy _π_ ref.


**Group-normalized** **advantages.** Because the reward in Eq. 4 is a single trajectory-level signal,
the per-turn advantage _A_ _[t]_ _i_ [is] [constant] [over] _[t]_ [within] [a] [rollout] _[i]_ [.] [We] [reduce] [variance] [and] [sharpen]
credit assignment across the group by using a _group-normalized_ advantage:

_A_ _[t]_ _i_ [=] _R_ ¯( _oi, q, y_ std _[∗]_ ) _−_       - _{R_ mean [¯] ( _ok, q, y_       - _{R_ [¯] _[∗]_ () _o}k_ _[G]_ _k, q, y_ =1� _[∗]_ ) _}_ _[G]_ _k_ =1� _._ (7)


4 EXPERIMENTS


4.1 EXPERIMENTAL SETUP


In our main experiments, all modules—Action Planner, Tool Executor, Executive Verifier, and Solution Generator—are instantiated with the _Qwen2.5-7B-Instruct_ model (Yang et al., 2024a). Among
these, only the _Action_ _Planner_ is trainable. The system operates with five interactive tools: _Base_
_Generator_ is an instance of _Qwen2.5-7B-Instruct_ that acts as the default reasoning engine if the
planner decides not to use an external tool; _Python Coder_ generates and executes Python code given
a query and returns the execution result; _Google Search_ searches the web and returns a summarization of Top-K search results; _Wikipedia Search_ searches articles matching a given query and returns
a summarization; and _Web_ _Search_ returns summarized information from a given web page. During the RL fine-tuning phase, we mix data from Search-R1 (Jin et al., 2025) and DeepMath (He
et al., 2025) as training data, which provides paired question-answer examples across search and
mathematical domains. We use a batch size of 32 with 8 rollouts per sample.


To comprehensively evaluate tool-use capabilities of AGENTFLOW, we conduct experiments on four
types of reasoning tasks: (1) _Knowledge-intensive search_ including Bamboogle (Press et al., 2023),
2Wiki (Ho et al., 2020), HotpotQA (Yang et al., 2018), and Musique (Trivedi et al., 2022); (2) _Agen-_
_tic reasoning_ such as GAIA (Mialon et al., 2023) (where we adopt the textual split); (3) _Logic-dense_
_mathematical_ _reasoning_ including AIME2024 (Art of Problem Solving, 2025), AMC23 (MAA,
2023), and GameOf24 (Lightman et al., 2023); and (4) _Scientific reasoning_ including GPQA (Rein
et al., 2024) and MedQA (Yang et al., 2024c). To mitigate randomness, we report the average accuracy across three trials for all experiments. More experimental details are in §C.


6


|Search Intensive Agentic<br>Model Size Bamboogle 2Wiki HotpotQA Musique Avg. ∆ GAIA ∆|Col2|Col3|Col4|
|---|---|---|---|
|Qwen-2.5-7B-Instruct<br>7B-Inst<br>Qwen-2.5-14B-Instruct<br>14B-Inst<br>Qwen-2.5-32B-Instruct<br>32B-Inst<br>Llama-3.3-70B-Instruct<br>70B-Inst|12.0<br>23.0<br>21.0<br>6.0<br>21.6<br>26.7<br>20.0<br>8.0<br>24.0<br>26.7<br>27.0<br>6.0<br>18.4<br>22.7<br>52.0<br>16.0|15.5<br>_↑_41.8<br>19.1<br>_↑_38.2<br>20.9<br>_↑_36.4<br>27.3<br>_↑_30.0|3.2<br>_↑_29.9<br>5.5<br>_↑_27.6<br>9.5<br>_↑_23.6<br>3.2<br>_↑_29.9|
|GPT-4o-mini (Hurst et al., 2024)<br>_∼_8B<br>40.8<br>35.6<br>41.0<br>15.0<br>33.1<br>_↑_24.2<br>7.1<br>_↑_26.0<br>GPT-4o (Hurst et al., 2024)<br>_∼_200B<br>68.8<br>49.5<br>54.0<br>24.0<br>49.1<br>_↑_8.2<br>17.3<br>_↑_15.8|GPT-4o-mini (Hurst et al., 2024)<br>_∼_8B<br>40.8<br>35.6<br>41.0<br>15.0<br>33.1<br>_↑_24.2<br>7.1<br>_↑_26.0<br>GPT-4o (Hurst et al., 2024)<br>_∼_200B<br>68.8<br>49.5<br>54.0<br>24.0<br>49.1<br>_↑_8.2<br>17.3<br>_↑_15.8|GPT-4o-mini (Hurst et al., 2024)<br>_∼_8B<br>40.8<br>35.6<br>41.0<br>15.0<br>33.1<br>_↑_24.2<br>7.1<br>_↑_26.0<br>GPT-4o (Hurst et al., 2024)<br>_∼_200B<br>68.8<br>49.5<br>54.0<br>24.0<br>49.1<br>_↑_8.2<br>17.3<br>_↑_15.8|GPT-4o-mini (Hurst et al., 2024)<br>_∼_8B<br>40.8<br>35.6<br>41.0<br>15.0<br>33.1<br>_↑_24.2<br>7.1<br>_↑_26.0<br>GPT-4o (Hurst et al., 2024)<br>_∼_200B<br>68.8<br>49.5<br>54.0<br>24.0<br>49.1<br>_↑_8.2<br>17.3<br>_↑_15.8|
|Supervised Fine-Tuning (SFT)<br>7B-Inst<br>12.0<br>25.9<br>22.0<br>6.6<br>16.6<br>_↑_40.7<br>3.2<br>_↑_29.9|Supervised Fine-Tuning (SFT)<br>7B-Inst<br>12.0<br>25.9<br>22.0<br>6.6<br>16.6<br>_↑_40.7<br>3.2<br>_↑_29.9|Supervised Fine-Tuning (SFT)<br>7B-Inst<br>12.0<br>25.9<br>22.0<br>6.6<br>16.6<br>_↑_40.7<br>3.2<br>_↑_29.9|Supervised Fine-Tuning (SFT)<br>7B-Inst<br>12.0<br>25.9<br>22.0<br>6.6<br>16.6<br>_↑_40.7<br>3.2<br>_↑_29.9|
|Iter-RetGen (Shao et al., 2023)<br>7B-Inst<br>Search-R1 (Jin et al., 2025)<br>7B-Inst<br>ZeroSearch (Sun et al., 2025)<br>7B-Base<br>ReSearch (Chen et al., 2025)<br>7B-Base<br>StepSearch (Wang et al., 2025d)<br>7B-Base<br>VerlTool (Jiang et al., 2025)<br>7B-Base|36.8<br>33.6<br>37.4<br>17.8<br>43.2<br>38.2<br>37.0<br>14.6<br>27.8<br>35.2<br>34.6<br>18.0<br>42.4<br>47.6<br>43.5<br>22.3<br>40.0<br>36.6<br>38.6<br>22.6<br>46.4<br>45.3<br>44.8<br>19.3|31.4<br>_↑_25.9<br>33.3<br>_↑_24.0<br>28.9<br>_↑_28.4<br>39.0<br>_↑_18.3<br>34.5<br>_↑_22.8<br>39.0<br>_↑_18.3|3.9<br>_↑_29.2<br>19.1<br>_↑_14.0<br>16.5<br>_↑_16.6<br>17.3<br>_↑_15.8<br>–<br>–<br>11.2<br>_↑_21.9|
|AutoGen (Wu et al., 2024)<br>7B-Inst<br>59.6<br>44.0<br>50.0<br>15.9<br>42.4<br>_↑_14.9<br>6.3<br>_↑_26.8|AutoGen (Wu et al., 2024)<br>7B-Inst<br>59.6<br>44.0<br>50.0<br>15.9<br>42.4<br>_↑_14.9<br>6.3<br>_↑_26.8|AutoGen (Wu et al., 2024)<br>7B-Inst<br>59.6<br>44.0<br>50.0<br>15.9<br>42.4<br>_↑_14.9<br>6.3<br>_↑_26.8|AutoGen (Wu et al., 2024)<br>7B-Inst<br>59.6<br>44.0<br>50.0<br>15.9<br>42.4<br>_↑_14.9<br>6.3<br>_↑_26.8|
|**AGENTFLOW**<br>7B-Inst<br>**AGENTFLOW** (w/ Flow-GRPO)<br>7B-Inst|58.4<br>60.0<br>51.3<br>19.2<br>**69.6**<br>**77.2**<br>**57.0**<br>**25.3**|47.2<br>_↑_12.1<br>**57.3**<br>–|17.2<br>_↑_15.9<br>**33.1**<br>–|


Table 1: **Accuracy comparison on search-intensive and agentic tasks.** 7B-Base refers to Qwen2.5-7B-Base and 7B-Inst refers to Qwen-2.5-7B-Instruct. AutoGen and our AGENTFLOW method
are agentic systems, which use Qwen-2.5-7B-Instruct for the LLM-powered agents and tools for fair
comparison. We visualize the gains of AGENTFLOW to the each baseline in the ∆ columns .

|Math Reasoning Scientific Reasoning<br>Model Size AIME24 AMC23 GameOf24 Avg. ∆ GPQA MedQA Avg. ∆|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|Qwen-2.5-7B-Instruct<br>7B-Inst<br>Qwen-2.5-14B-Instruct<br>14B-Inst<br>Llama-3.3-70B-Instruct<br>70B-Inst<br>Llama-3.1-405B-Instruct<br>405B-Inst|6.7<br>47.5<br>33.0<br>6.7<br>60.0<br>25.0<br>6.7<br>47.5<br>31.0<br>26.7<br>47.5<br>23.0|29.1<br>_↑_22.5<br>30.6<br>_↑_21.0<br>28.4<br>_↑_23.1<br>32.4<br>_↑_19.1|34.0<br>66.0<br>31.0<br>75.0<br>35.0<br>67.0<br>30.0<br>62.0|50.0<br>_↑_13.5<br>53.0<br>_↑_10.5<br>51.0<br>_↑_12.5<br>46.0<br>_↑_17.5|
|GPT-4o-mini (Hurst et al., 2024)<br>_∼_8B<br>13.3<br>57.5<br>16.0<br>28.9<br>_↑_22.6<br>27.0<br>66.0<br>46.5<br>_↑_17.0<br>GPT-4o (Hurst et al., 2024)<br>_∼_200B<br>13.3<br>60.0<br>32.0<br>35.1<br>_↑_16.4<br>31.0<br>60.0<br>45.5<br>_↑_18.0|GPT-4o-mini (Hurst et al., 2024)<br>_∼_8B<br>13.3<br>57.5<br>16.0<br>28.9<br>_↑_22.6<br>27.0<br>66.0<br>46.5<br>_↑_17.0<br>GPT-4o (Hurst et al., 2024)<br>_∼_200B<br>13.3<br>60.0<br>32.0<br>35.1<br>_↑_16.4<br>31.0<br>60.0<br>45.5<br>_↑_18.0|GPT-4o-mini (Hurst et al., 2024)<br>_∼_8B<br>13.3<br>57.5<br>16.0<br>28.9<br>_↑_22.6<br>27.0<br>66.0<br>46.5<br>_↑_17.0<br>GPT-4o (Hurst et al., 2024)<br>_∼_200B<br>13.3<br>60.0<br>32.0<br>35.1<br>_↑_16.4<br>31.0<br>60.0<br>45.5<br>_↑_18.0|GPT-4o-mini (Hurst et al., 2024)<br>_∼_8B<br>13.3<br>57.5<br>16.0<br>28.9<br>_↑_22.6<br>27.0<br>66.0<br>46.5<br>_↑_17.0<br>GPT-4o (Hurst et al., 2024)<br>_∼_200B<br>13.3<br>60.0<br>32.0<br>35.1<br>_↑_16.4<br>31.0<br>60.0<br>45.5<br>_↑_18.0|GPT-4o-mini (Hurst et al., 2024)<br>_∼_8B<br>13.3<br>57.5<br>16.0<br>28.9<br>_↑_22.6<br>27.0<br>66.0<br>46.5<br>_↑_17.0<br>GPT-4o (Hurst et al., 2024)<br>_∼_200B<br>13.3<br>60.0<br>32.0<br>35.1<br>_↑_16.4<br>31.0<br>60.0<br>45.5<br>_↑_18.0|
|Supervised Fine-Tuning (SFT)<br>7B-Inst<br>6.7<br>47.5<br>33.0<br>29.1<br>_↑_22.5<br>34.0<br>66.0<br>50.0<br>_↑_13.5|Supervised Fine-Tuning (SFT)<br>7B-Inst<br>6.7<br>47.5<br>33.0<br>29.1<br>_↑_22.5<br>34.0<br>66.0<br>50.0<br>_↑_13.5|Supervised Fine-Tuning (SFT)<br>7B-Inst<br>6.7<br>47.5<br>33.0<br>29.1<br>_↑_22.5<br>34.0<br>66.0<br>50.0<br>_↑_13.5|Supervised Fine-Tuning (SFT)<br>7B-Inst<br>6.7<br>47.5<br>33.0<br>29.1<br>_↑_22.5<br>34.0<br>66.0<br>50.0<br>_↑_13.5|Supervised Fine-Tuning (SFT)<br>7B-Inst<br>6.7<br>47.5<br>33.0<br>29.1<br>_↑_22.5<br>34.0<br>66.0<br>50.0<br>_↑_13.5|
|SimpleRL-reason (Zeng et al., 2025b)<br>7B-Base<br>16.7<br>60.0<br>33.0<br>36.6<br>_↑_15.0<br>45.0<br>65.0<br>50.0<br>_↑_13.5<br>Open-Reasoner-Zero (Hu et al., 2025a) 7B-Base<br>16.7<br>54.9<br>32.0<br>34.5<br>_↑_17.0<br>34.0<br>54.0<br>44.0<br>_↑_19.5<br>General-Reasoner (Ma et al., 2025)<br>7B-Base<br>13.3<br>55.0<br>33.0<br>33.8<br>_↑_17.7<br>35.5<br>61.0<br>48.3<br>_↑_15.2<br>Luffy (Yan et al., 2025)<br>7B-Inst<br>30.7<br>44.8<br>33.0<br>36.2<br>_↑_15.3<br>34.0<br>77.0<br>55.5<br>_↑_8.0|SimpleRL-reason (Zeng et al., 2025b)<br>7B-Base<br>16.7<br>60.0<br>33.0<br>36.6<br>_↑_15.0<br>45.0<br>65.0<br>50.0<br>_↑_13.5<br>Open-Reasoner-Zero (Hu et al., 2025a) 7B-Base<br>16.7<br>54.9<br>32.0<br>34.5<br>_↑_17.0<br>34.0<br>54.0<br>44.0<br>_↑_19.5<br>General-Reasoner (Ma et al., 2025)<br>7B-Base<br>13.3<br>55.0<br>33.0<br>33.8<br>_↑_17.7<br>35.5<br>61.0<br>48.3<br>_↑_15.2<br>Luffy (Yan et al., 2025)<br>7B-Inst<br>30.7<br>44.8<br>33.0<br>36.2<br>_↑_15.3<br>34.0<br>77.0<br>55.5<br>_↑_8.0|SimpleRL-reason (Zeng et al., 2025b)<br>7B-Base<br>16.7<br>60.0<br>33.0<br>36.6<br>_↑_15.0<br>45.0<br>65.0<br>50.0<br>_↑_13.5<br>Open-Reasoner-Zero (Hu et al., 2025a) 7B-Base<br>16.7<br>54.9<br>32.0<br>34.5<br>_↑_17.0<br>34.0<br>54.0<br>44.0<br>_↑_19.5<br>General-Reasoner (Ma et al., 2025)<br>7B-Base<br>13.3<br>55.0<br>33.0<br>33.8<br>_↑_17.7<br>35.5<br>61.0<br>48.3<br>_↑_15.2<br>Luffy (Yan et al., 2025)<br>7B-Inst<br>30.7<br>44.8<br>33.0<br>36.2<br>_↑_15.3<br>34.0<br>77.0<br>55.5<br>_↑_8.0|SimpleRL-reason (Zeng et al., 2025b)<br>7B-Base<br>16.7<br>60.0<br>33.0<br>36.6<br>_↑_15.0<br>45.0<br>65.0<br>50.0<br>_↑_13.5<br>Open-Reasoner-Zero (Hu et al., 2025a) 7B-Base<br>16.7<br>54.9<br>32.0<br>34.5<br>_↑_17.0<br>34.0<br>54.0<br>44.0<br>_↑_19.5<br>General-Reasoner (Ma et al., 2025)<br>7B-Base<br>13.3<br>55.0<br>33.0<br>33.8<br>_↑_17.7<br>35.5<br>61.0<br>48.3<br>_↑_15.2<br>Luffy (Yan et al., 2025)<br>7B-Inst<br>30.7<br>44.8<br>33.0<br>36.2<br>_↑_15.3<br>34.0<br>77.0<br>55.5<br>_↑_8.0|SimpleRL-reason (Zeng et al., 2025b)<br>7B-Base<br>16.7<br>60.0<br>33.0<br>36.6<br>_↑_15.0<br>45.0<br>65.0<br>50.0<br>_↑_13.5<br>Open-Reasoner-Zero (Hu et al., 2025a) 7B-Base<br>16.7<br>54.9<br>32.0<br>34.5<br>_↑_17.0<br>34.0<br>54.0<br>44.0<br>_↑_19.5<br>General-Reasoner (Ma et al., 2025)<br>7B-Base<br>13.3<br>55.0<br>33.0<br>33.8<br>_↑_17.7<br>35.5<br>61.0<br>48.3<br>_↑_15.2<br>Luffy (Yan et al., 2025)<br>7B-Inst<br>30.7<br>44.8<br>33.0<br>36.2<br>_↑_15.3<br>34.0<br>77.0<br>55.5<br>_↑_8.0|
|TIR (Yang et al., 2024b)<br>7B-Inst<br>10.0<br>50.0<br>33.0<br>31.0<br>_↑_20.5<br>42.0<br>76.8<br>59.4<br>_↑_4.1<br>ToRL (Li et al., 2025b)<br>7B-Inst<br>20.0<br>60.0<br>31.0<br>37.0<br>_↑_14.5<br>35.0<br>76.5<br>55.8<br>_↑_7.7|TIR (Yang et al., 2024b)<br>7B-Inst<br>10.0<br>50.0<br>33.0<br>31.0<br>_↑_20.5<br>42.0<br>76.8<br>59.4<br>_↑_4.1<br>ToRL (Li et al., 2025b)<br>7B-Inst<br>20.0<br>60.0<br>31.0<br>37.0<br>_↑_14.5<br>35.0<br>76.5<br>55.8<br>_↑_7.7|TIR (Yang et al., 2024b)<br>7B-Inst<br>10.0<br>50.0<br>33.0<br>31.0<br>_↑_20.5<br>42.0<br>76.8<br>59.4<br>_↑_4.1<br>ToRL (Li et al., 2025b)<br>7B-Inst<br>20.0<br>60.0<br>31.0<br>37.0<br>_↑_14.5<br>35.0<br>76.5<br>55.8<br>_↑_7.7|TIR (Yang et al., 2024b)<br>7B-Inst<br>10.0<br>50.0<br>33.0<br>31.0<br>_↑_20.5<br>42.0<br>76.8<br>59.4<br>_↑_4.1<br>ToRL (Li et al., 2025b)<br>7B-Inst<br>20.0<br>60.0<br>31.0<br>37.0<br>_↑_14.5<br>35.0<br>76.5<br>55.8<br>_↑_7.7|TIR (Yang et al., 2024b)<br>7B-Inst<br>10.0<br>50.0<br>33.0<br>31.0<br>_↑_20.5<br>42.0<br>76.8<br>59.4<br>_↑_4.1<br>ToRL (Li et al., 2025b)<br>7B-Inst<br>20.0<br>60.0<br>31.0<br>37.0<br>_↑_14.5<br>35.0<br>76.5<br>55.8<br>_↑_7.7|
|AutoGen (Wu et al., 2024)<br>7B-Inst<br>13.3<br>57.5<br>24.0<br>31.6<br>_↑_19.9<br>42.0<br>72.0<br>57.0<br>_↑_6.5|AutoGen (Wu et al., 2024)<br>7B-Inst<br>13.3<br>57.5<br>24.0<br>31.6<br>_↑_19.9<br>42.0<br>72.0<br>57.0<br>_↑_6.5|AutoGen (Wu et al., 2024)<br>7B-Inst<br>13.3<br>57.5<br>24.0<br>31.6<br>_↑_19.9<br>42.0<br>72.0<br>57.0<br>_↑_6.5|AutoGen (Wu et al., 2024)<br>7B-Inst<br>13.3<br>57.5<br>24.0<br>31.6<br>_↑_19.9<br>42.0<br>72.0<br>57.0<br>_↑_6.5|AutoGen (Wu et al., 2024)<br>7B-Inst<br>13.3<br>57.5<br>24.0<br>31.6<br>_↑_19.9<br>42.0<br>72.0<br>57.0<br>_↑_6.5|
|**AGENTFLOW**<br>7B-Inst<br>**AGENTFLOW** (w/ Flow-GRPO)<br>7B-Inst|16.7<br>47.4<br>31.0<br>**40.0**<br>**61.5**<br>**53.0**|31.7<br>_↑_19.8<br>**51.5**<br>–|37.0<br>76.0<br>**47.0**<br>**80.0**|56.5<br>_↑_7.0<br>**63.5**<br>–|


Table 2: **Accuracy comparison of mathematical and scientific reasoning tasks.**


4.2 MAIN RESULTS


**Baselines.** As presented in Tables 1 and 2, we include five categories of baselines: (1) _Open-_
_source_ _LLMs_ : Qwen2.5 (Yang et al., 2024a), Llama-3.1, and Llama-3.3 (Dubey et al., 2024); (2)
_Proprietary_ _LLMs_ : GPT-4o-mini and GPT-4o; (3) _Reasoning LLMs_ : supervised fine-tuning (Yang
et al., 2024b), SimpleRL-reason, Open-Reasoner-Zero, General-Reasoner, and LUFFY; (4) _Tool-_
_integrated reasoning LLMs_ : both search-enhanced, including Iter-RetGen, Search-R1, ZeroSearch,
ReSearch, StepSearch, and VerlTool, and code-enhanced, including TIR and ToRL; (5) _Training-free_
_agentic system_ : AutoGen. More details on baseline implementations are in §C.3.


**Key insights.** AGENTFLOW consistently outperforms all baseline models by large margins. Compared to the best-performing 7B models without tool integration, AGENTFLOW achieves absolute
gains of 40.7% on search (SFT), 29.9% on agentic reasoning (SFT), 15.0% on math (SimpleRLreason), and 8.0% on scientific tasks (Luffy). Against specialized tool-integrated systems, AGENTFLOW surpasses the top models by 14.9% in search (AutoGen), 14.0% in agentic reasoning (SearchR1), 14.5% in math (ToRL), and 4.1% in science (TIR). Notably, our 7B-backbone AGENTFLOW
even outperforms the _∼_ 200B-parameter GPT-4o across all domains, with gains ranging from 8.2%
to 18.0%. A detailed analysis is provided in §D.1.


7


AgentFlow ( **before** Flow-GRPO Fine-tuning)


AgentFlow ( **after** Flow-GRPO Fine-tuning)


Figure 5: **One** **case** **study** **example.** Initially failed with repetitive errors (left), AGENTFLOW,
trained with Flow-GRPO, explores a new solution pathway at turn 4 after two failed attempts (right).


4.3 TRAINING STRATEGIES ON THE PLANNER


We conduct an ablation study to analyze the impact of different training strategies for the _Action_
_Planner_ module in AGENTFLOW, with results reported in Table 3. The executor, verifier, and generator modules remain fixed as Qwen2.5-7B-Instruct, consistent with our main setup (§4.1).


**Planner Model** **Training** **Bamboogle** **2Wiki** **GAIA** **AIME24** **AMC23** **GameOf24** **Avg.**


Qwen-2.5-7B Frozen 58.4 60.0 17.2 16.7 47.4 31.0 38.5


Table 3: Performance comparison of AGENTFLOW across different training methods.


**A more capable planner is beneficial, but has limits.** Replacing the frozen _Qwen2.5-7B-Instruct_
baseline with a stronger proprietary model, GPT-4o, yields only a modest 5.8% average gain. This
indicates a key bottleneck that, while a more powerful model improves planning, its static nature
prevents co-adaptation with the live dynamics of AGENTFLOW.


**Offline SFT** **leads to performance collapse,** **while in-the-flow RL is crucial.** The limitations of
a static planner are further exposed when distilling GPT-4o’s behavior via offline supervised finetuning (SFT) on its trajectories as _Action_ _Planner_ in AGENTFLOW. This results in a catastrophic
performance collapse, with an average accuracy drop of 19.0% compared to the frozen baseline.
This failure arises from the token-level imitation objective of SFT, which misaligns with trajectorylevel task success and prevents the planner from adapting to dynamic tool feedback or recovering
from compounding errors. In contrast, training the planner with our on-policy Flow-GRPO method
proves highly effective: by optimizing for the final outcome, the planner learns to handle longhorizon workflows, achieving a 17.2% average gain over the frozen baseline.


4.4 SCALING TRENDS IN AGENTFLOW


**Training** **scaling** **in** **backbone** **size.** We study how backbone LLM scale affects AGENTFLOW’s
performance and the efficacy of Flow-GRPO. We build two versions of the system: one using
_Qwen2.5-3B-Instruct_ and another using _Qwen2.5-7B-Instruct_ for all four modules (planner, executor, verifier, and generator) and tools. In both, only the planner is fine-tuned with Flow-GRPO. As
shown in Figure 6, Flow-GRPO fine-tuning consistently improves performance across tasks for both
backbones. This demonstrates that our in-the-flow optimization is effective across model capacities,
enhancing AGENTFLOW regardless of LLM size.


8


80


60


40


20


80


70


60


50


40


30


20


|Col1|Col2|Col3|Col4|+15.8%|
|---|---|---|---|---|
||~~2Wiki~~||||
||||||
||||||
||**Game**<br>~~**AIME2**~~<br>**GAIA**|**f24**<br>~~**4**~~||+20.0%<br>~~+16.7%~~<br>|
||||||
|||||+6.3%|


3 5 7 10
Max Allowed Turns


80


60


40


20


AgentFlow (Qwen-2.5-3B-Instruct)


AgentFlow (Qwen-2.5-7B-Instruct)


0

|Col1|68.8<br>63.0|72.3|Before tu<br>After tuni|ning<br>ng|
|---|---|---|---|---|
|53.6|||||
||||9.1||
|||14.3<br>|13.3<br><br>|20.0|

Bamboogle 2Wiki GAIA AIME24


0

|Col1|69.6<br>60.0|B<br>77.2<br>A|efore tu<br>fter tuni|ning<br>ng|
|---|---|---|---|---|
|58.|||||
|||33|.1<br>|40.0|
|||17.2<br>|16.7<br>||

Bamboogle 2Wiki GAIA AIME24


Figure 6: Flow-GRPO fine-tuning offers consistent gains on
AGENTFLOW as the backbone model size scales from 3B to 7B.


Figure 7: Average accuracy
with increased _T_ max.


**Inference** **scaling** **in** **turn** **budgets.** We investigate how the maximum allowed turns ( _T_ max) affect **Turns (** _T_ **max)** **3** **5** **7** **10**
reasoning depth and final performance of AGENT- 2Wiki 2.22 3.18 3.81 4.44
FLOW during test-time inference with the Qwen2.5- GameOf24 1.63 2.12 2.36 2.67
7B-Instruct backbone. As shown in Figure 7, in- AIME24 1.63 1.63 1.86 1.90
creasing _T_ max from 3 to 10 consistently improves GAIA 2.43 3.46 4.28 5.42
outcomes across all tasks, accompanied by a rise in

Table 4: Average turns with increased _T_ max.

average turns consumed. On knowledge-intensive
benchmarks such as 2Wiki and GAIA, a larger turn budget enables AGENTFLOW for deeper information retrieval. On mathematical benchmarks like GameOf24 and AIME24, it supports decomposed sub-goals, alternative strategies, and refinement of errors. Final performance peaks at
_T_ max = 10 for all tasks, confirming that a longer reasoning horizon benefits the system without
causing degenerate loops. This validates that AGENTFLOW adapts its turn allocation to problem
complexity to achieve better solutions through iterative refinement.


**Turns (** _T_ **max)** **3** **5** **7** **10**


2Wiki 2.22 3.18 3.81 4.44
GameOf24 1.63 2.12 2.36 2.67
AIME24 1.63 1.63 1.86 1.90
GAIA 2.43 3.46 4.28 5.42


Table 4: Average turns with increased _T_ max.


4.5 IN-DEPTH ANALYSIS OF OPTIMIZED PLANNING


on two knowledge-intensive tasks,

the planner to increase Google Search

|Base Generator Google Search Web Search Wikipedia Search|W|Web Search|Col4|Col5|Wikipedia Search|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
|28.5<br>**Acc:60.0%**<br>28.8<br>70.5<br>~~13.6~~<br>4.0<br>28.7<br>66.2<br>6.3 10.9<br>19.5<br>59.8<br>**Acc: 77.2% (+17.2%)**<br>36.0<br>**+42.0**<br>**-22.4**<br>**-24.8**<br>After Fine-tuning<br>**Acc: 76.0%**<br>**Acc: 80.0% (+4.0%)**<br>After Fine-tuning<br>**+59.8**<br>**+19.5**<br>~~**-22.4**~~<br>**-55.3**<br>(a) 2Wiki<br>(b) MedQA||**Acc: 76.0%**|**Acc: 76.0%**|**Acc: 76.0%**|**Acc: 80.0% (+4.0%)**|**Acc: 80.0% (+4.0%)**|**Acc: 80.0% (+4.0%)**|**Acc: 80.0% (+4.0%)**|**Acc: 80.0% (+4.0%)**|**Acc: 80.0% (+4.0%)**|
|28.5<br>**Acc:60.0%**<br>28.8<br>70.5<br>~~13.6~~<br>4.0<br>28.7<br>66.2<br>6.3 10.9<br>19.5<br>59.8<br>**Acc: 77.2% (+17.2%)**<br>36.0<br>**+42.0**<br>**-22.4**<br>**-24.8**<br>After Fine-tuning<br>**Acc: 76.0%**<br>**Acc: 80.0% (+4.0%)**<br>After Fine-tuning<br>**+59.8**<br>**+19.5**<br>~~**-22.4**~~<br>**-55.3**<br>(a) 2Wiki<br>(b) MedQA|66.2<br>After|66.2<br>After|66.2<br>After|66.2<br>After|Fine-tuning<br>**+59.8**|Fine-tuning<br>**+59.8**|Fine-tuning<br>**+59.8**|Fine-tuning<br>**+59.8**|Fine-tuning<br>**+59.8**|Fine-tuning<br>**+59.8**|
|28.5<br>**Acc:60.0%**<br>28.8<br>70.5<br>~~13.6~~<br>4.0<br>28.7<br>66.2<br>6.3 10.9<br>19.5<br>59.8<br>**Acc: 77.2% (+17.2%)**<br>36.0<br>**+42.0**<br>**-22.4**<br>**-24.8**<br>After Fine-tuning<br>**Acc: 76.0%**<br>**Acc: 80.0% (+4.0%)**<br>After Fine-tuning<br>**+59.8**<br>**+19.5**<br>~~**-22.4**~~<br>**-55.3**<br>(a) 2Wiki<br>(b) MedQA|66.2<br>After|66.2<br>After|66.|66.|66.|66.|66.|66.|66.|66.|
|28.5<br>**Acc:60.0%**<br>28.8<br>70.5<br>~~13.6~~<br>4.0<br>28.7<br>66.2<br>6.3 10.9<br>19.5<br>59.8<br>**Acc: 77.2% (+17.2%)**<br>36.0<br>**+42.0**<br>**-22.4**<br>**-24.8**<br>After Fine-tuning<br>**Acc: 76.0%**<br>**Acc: 80.0% (+4.0%)**<br>After Fine-tuning<br>**+59.8**<br>**+19.5**<br>~~**-22.4**~~<br>**-55.3**<br>(a) 2Wiki<br>(b) MedQA|||||||||59.8<br>||
|28.5<br>**Acc:60.0%**<br>28.8<br>70.5<br>~~13.6~~<br>4.0<br>28.7<br>66.2<br>6.3 10.9<br>19.5<br>59.8<br>**Acc: 77.2% (+17.2%)**<br>36.0<br>**+42.0**<br>**-22.4**<br>**-24.8**<br>After Fine-tuning<br>**Acc: 76.0%**<br>**Acc: 80.0% (+4.0%)**<br>After Fine-tuning<br>**+59.8**<br>**+19.5**<br>~~**-22.4**~~<br>**-55.3**<br>(a) 2Wiki<br>(b) MedQA|||||||||||
|28.5<br>**Acc:60.0%**<br>28.8<br>70.5<br>~~13.6~~<br>4.0<br>28.7<br>66.2<br>6.3 10.9<br>19.5<br>59.8<br>**Acc: 77.2% (+17.2%)**<br>36.0<br>**+42.0**<br>**-22.4**<br>**-24.8**<br>After Fine-tuning<br>**Acc: 76.0%**<br>**Acc: 80.0% (+4.0%)**<br>After Fine-tuning<br>**+59.8**<br>**+19.5**<br>~~**-22.4**~~<br>**-55.3**<br>(a) 2Wiki<br>(b) MedQA|||||||||||
|28.5<br>**Acc:60.0%**<br>28.8<br>70.5<br>~~13.6~~<br>4.0<br>28.7<br>66.2<br>6.3 10.9<br>19.5<br>59.8<br>**Acc: 77.2% (+17.2%)**<br>36.0<br>**+42.0**<br>**-22.4**<br>**-24.8**<br>After Fine-tuning<br>**Acc: 76.0%**<br>**Acc: 80.0% (+4.0%)**<br>After Fine-tuning<br>**+59.8**<br>**+19.5**<br>~~**-22.4**~~<br>**-55.3**<br>(a) 2Wiki<br>(b) MedQA|28.7|28.7|||**+19.5**|**+19.5**|**+19.5**|**+19.5**|||
|28.5<br>**Acc:60.0%**<br>28.8<br>70.5<br>~~13.6~~<br>4.0<br>28.7<br>66.2<br>6.3 10.9<br>19.5<br>59.8<br>**Acc: 77.2% (+17.2%)**<br>36.0<br>**+42.0**<br>**-22.4**<br>**-24.8**<br>After Fine-tuning<br>**Acc: 76.0%**<br>**Acc: 80.0% (+4.0%)**<br>After Fine-tuning<br>**+59.8**<br>**+19.5**<br>~~**-22.4**~~<br>**-55.3**<br>(a) 2Wiki<br>(b) MedQA|||||<br><br>~~**-22.4**~~<br>**-55.3**|<br><br>~~**-22.4**~~<br>**-55.3**|<br><br>~~**-22.4**~~<br>**-55.3**|19.5<br>|||
|28.5<br>**Acc:60.0%**<br>28.8<br>70.5<br>~~13.6~~<br>4.0<br>28.7<br>66.2<br>6.3 10.9<br>19.5<br>59.8<br>**Acc: 77.2% (+17.2%)**<br>36.0<br>**+42.0**<br>**-22.4**<br>**-24.8**<br>After Fine-tuning<br>**Acc: 76.0%**<br>**Acc: 80.0% (+4.0%)**<br>After Fine-tuning<br>**+59.8**<br>**+19.5**<br>~~**-22.4**~~<br>**-55.3**<br>(a) 2Wiki<br>(b) MedQA||||||6.3 <br>|10.9||||


|Col1|Acc:60.0%|Col3|Acc: 77.2% (+17.2%)|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|After Fine-t|After Fine-t|After Fine-t|uning|70.5|**+42.0**|**+42.0**|
||||||||
||||||||
|||36.0|||||
||28.5|28.<br>|8||||
||||||~~13.6~~<br>**-22.4**<br>|~~13.6~~<br>**-22.4**<br>|
|||||||4.0<br>**-24.8**|


Figure 8: Tool call ratio change by Flow-GRPO fine-tuning.

usage by 42.0%. In contrast, for the
specialized MedQA benchmark, which requires deep, domain-specific information retrieval, finetuning shifts the planner away from general tools, reducing Google Search calls (66.2 _→_ 10.9%) in
favor of in-document Web Search (0 _→_ 19.5%) and specialized Wikipedia Search (0 _→_ 59.8%). This
demonstrates that the planner learns to select task-appropriate tools.


**Flow-GRPO incentivizes autonomous discovery of new solutions.** We further examine qualitative examples in Figure 5 and additional cases in §F. These cases show that AGENTFLOW, trained
with Flow-GRPO, develops enhanced capabilities for task planning and tool use. The planner exhibits adaptive efficiency, stronger self-correction, and spontaneous new integration of tools throughout step-by-step problem-solving, autonomously discovering effective solution pathways.


4.6 TRAINING EFFICIENCY ANALYSIS


**Optimized planning with increased rewards and condensed responses.** We analyze the training
dynamics of the AGENTFLOW planner by tracking its average reward and response length on the
train set (Figure 9a). Training rewards steadily increase, indicating effective policy improvement via
Flow-GRPO. Meanwhile, response length, after an initial exploratory rise, progressively shortens
and stabilizes. This shows the planner learns to balance conciseness and informativeness, avoiding
unnecessarily long outputs.


9


Figure 8: Tool call ratio change by Flow-GRPO fine-tuning.


0.5


gains, with validation accuracy Training Steps Training Steps
growing steadily. In contrast, ToRL’s

Figure 9: Training dynamics and efficiency of Flow-GRPO.

performance quickly stagnates and
trends downwards, highlighting the superior efficiency of our agentic training approach, which uses
decomposition and stable credit assignment to avoid the instability.


230

220

210

200

190

180


0.8


0.7


0.6


0.5


0.4


0.3


0.2


0.1


(b)


0 10 20 30
Training Steps


(a)


0 10 20 30 40 50 60
Training Steps


Figure 9: Training dynamics and efficiency of Flow-GRPO.


5 RELATED WORK


Reinforcement learning (RL) from outcome-based rewards has become a dominant paradigm for
training LLMs to use external tools. Much of this work trains a single, monolithic policy to interleave reasoning with tool calls. This strategy has proven effective in specialized, single-tool
settings (Mai et al., 2025; Xue et al., 2025; Feng et al., 2025; Li et al., 2025b) and web search
for knowledge-intensive questions (Chen et al., 2025; Jin et al., 2025; Song et al., 2025; Li et al.,
2025a; Sun et al., 2025). Recent efforts have extended this monolithic framework to multi-tool environments by focusing on data synthesis (Dong et al., 2025), unified training infrastructure (Jiang
et al., 2025), and principled reward design (Qian et al., 2025a; Zhang et al., 2025). However, these
approach scales poorly as task complexity and planning horizons grow. The central challenge is
long-horizon credit assignment; attributing a final outcome to specific intermediate tool calls remains difficult, even with fine-grained, turn-level rewards (Zeng et al., 2025a; Wang et al., 2025d).
This difficulty leads to training instability and brittle inference-time generalization, manifesting as
strategic deficiencies like tool overuse or “cognitive offloading” (Wang et al., 2025b; Qian et al.,
2025b), suboptimal personalization (Cheng et al., 2025), and poor alignment with user preferences
for tool invocation (Huang et al., 2025).


**Agentic systems with tool use.** Agentic systems offer an alternative to monolithic models by decomposing tasks across specialized modules. Many such systems are training-free, orchestrating
pre-trained LLMs with handcrafted logic and prompting, as seen in frameworks like AutoGen (Wu
et al., 2024), MetaGPT (Hong et al., 2024), and OctoTools (Lu et al., 2025). This static approach,
however, limits their ability to learn and adapt collaborative strategies from experience. Recognizing
this, recent work explores training these systems to improve coordination (Deng et al., 2025; Liao
et al., 2025). However, most training paradigms are _offline_, relying on supervised fine-tuning or
preference optimization on static datasets (Motwani et al., 2024; Park et al., 2025). These methods
are decoupled from the live, multi-turn dynamics of the system, preventing modules from learning
to adapt to evolving tool outputs or recover from early mistakes. Training directly _in the flow_ with
on-policy RL is difficult due to sparse rewards and long-horizon credit assignment, where feedback
is delayed across long reasoning chains and shifting state distributions (Wang et al., 2025c). Consequently, these systems often suffer from brittle adaptation and require complex reward shaping to
learn effectively (Wang et al., 2025a).


6 CONCLUSION


We presented AGENTFLOW, a trainable, _in-the-flow_ agentic system that coordinates four specialized
modules via an evolving memory and optimizes its planner directly _inside_ the multi-turn loop. To
enable stable on-policy learning under long-horizon, sparse-reward settings, we introduced FlowGRPO, which _converts_ multi-turn RL into a sequence of tractable _single-turn_ policy updates by
_broadcasting_ a single, verifiable trajectory-level outcome to every turn and stabilizing credit assignment with group-normalized advantages. Comprehensive experiments show that AGENTFLOW
achieves strong cross-domain performance, surpassing specialized baselines and even larger proprietary models. In-depth analyses confirm improved planning and tool-calling reliability, along with
positive scaling trends in model size and allowed turn budgets.


10


ACKNOWLEDGMENT


We would like to thank Yihe Deng, Xuehang Guo, and Kunlun Zhu for their valuable input during
the early stages of this work. We are grateful to Lambda for providing GPU resources. This work
was partially supported by the Hoffman-Yee Research Grants program at Stanford HAI, the AI for
Math Fund by Renaissance Philanthropy, ONR MURI N00014-24-1-2748, and the AI Research Hub
Project through KAIST.


REFERENCES


Art of Problem Solving. Aime problems and solutions, 2025. URL [https:](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions)
[//artofproblemsolving.com/wiki/index.php/AIME_Problems_and_](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions)
[Solutions.](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions) 6, 21


Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z Pan,
Wen Zhang, Huajun Chen, Fan Yang, et al. ReSearch: Learning to reason with search for llms via
reinforcement learning. _arXiv preprint arXiv:2503.19470_, 2025. 2, 3, 7, 10, 20


Zihao Cheng, Hongru Wang, Zeming Liu, Yuhang Guo, Yuanfang Guo, Yunhong Wang, and
Haifeng Wang. ToolSpectrum: Towards personalized tool utilization for large language models. In _Findings of the Association for Computational Linguistics:_ _ACL 2025_, pp. 20679–20699,
2025. 10


Yingfan Deng, Anhao Zhou, Yuan Yuan, Xian Zhang, Yifei Zou, and Dongxiao Yu. Pe-ma:
Parameter-efficient co-evolution of multi-agent systems. _arXiv preprint arXiv:2506.11803_, 2025.
10


Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu Mao, Guorui
Zhou, Zhicheng Dou, and Ji-Rong Wen. Tool-star: Empowering llm-brained multi-tool reasoner
via reinforcement learning. _arXiv preprint arXiv:2505.16410_, 2025. 2, 10


Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models.
_arXiv preprint arXiv:2407.21783_, 2024. 7, 19


Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang,
Jinxin Chi, and Wanjun Zhong. Retool: Reinforcement learning for strategic tool use in llms.
_arXiv preprint arXiv:2504.11536_, 2025. 2, 3, 10


Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms
via reinforcement learning. _arXiv preprint arXiv:2501.12948_, 2025. 2


Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian
Yu, Zhenwen Liang, Wenxuan Wang, et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. _arXiv_ _preprint_
_arXiv:2504.11456_, 2025. 6


Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning steps. In _Proceedings of the 28th Interna-_
_tional Conference on Computational Linguistics (COLING)_, pp. 6609–6625, 2020. 6, 21


Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin
Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al. MetaGPT: Meta programming for a
multi-agent collaborative framework. In _International Conference on Learning Representations_
_(ICLR)_, 2024. 2, 4, 10


Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum.
Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base
model. _arXiv preprint arXiv:2503.24290_, 2025a. 7, 20


11


Mengkang Hu, Yuhang Zhou, Wendong Fan, Yuzhou Nie, Bowei Xia, Tao Sun, Ziyu Ye, Zhaoxuan
Jin, Yingru Li, Qiguang Chen, et al. Owl: Optimized workforce learning for general multi-agent
assistance in real-world task automation. _arXiv preprint arXiv:2505.23885_, 2025b. 2


Chengrui Huang, Shen Gao, Zhengliang Shi, Dongsheng Wang, and Shuo Shang. TTPA: Tokenlevel tool-use preference alignment training framework with fine-grained evaluation. _arXiv_
_preprint arXiv:2505.20016_, 2025. 10


Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. GPT-4o system card. _arXiv_ _preprint_
_arXiv:2410.21276_, 2024. 3, 7, 19


Dongfu Jiang, Yi Lu, Zhuofeng Li, Zhiheng Lyu, Ping Nie, Haozhe Wang, Alex Su, Hui Chen, Kai
Zou, Chao Du, et al. VerlTool: Towards holistic agentic reinforcement learning with tool use.
_arXiv preprint arXiv:2509.01055_, 2025. 7, 10, 20


Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and
Jiawei Han. Search-R1: Training llms to reason and leverage search engines with reinforcement
learning. _arXiv preprint arXiv:2503.09516_, 2025. 2, 3, 6, 7, 10, 19, 20


Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What disease does this patient have? a large-scale open domain question answering dataset from medical
exams. _Applied Sciences_, 11(14):6421, 2021. 21


Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and
Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models. _arXiv_ _preprint_
_arXiv:2501.05366_, 2025a. 10


Xuefeng Li, Haoyang Zou, and Pengfei Liu. ToRL: Scaling tool-integrated rl. _arXiv_ _preprint_
_arXiv:2503.23383_, 2025b. 7, 10, 20


Junwei Liao, Muning Wen, Jun Wang, and Weinan Zhang. Marft: Multi-agent reinforcement finetuning. _arXiv preprint arXiv:2504.16129_, 2025. 10


Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan
Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In _The Twelfth_
_International Conference on Learning Representations (ICLR)_, 2023. 6


Nathan Lile. Math twenty four (24s game) dataset. [https://huggingface.co/datasets/](https://huggingface.co/datasets/nlile/24-game)
[nlile/24-game, 2024.](https://huggingface.co/datasets/nlile/24-game) 21


Pan Lu, Bowen Chen, Sheng Liu, Rahul Thapa, Joseph Boen, and James Zou. OctoTools: An agentic
framework with extensible tools for complex reasoning. _arXiv preprint arXiv:2502.11271_, 2025.
4, 10


Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang, Zejun Ma, and Wenhu Chen. General-reasoner:
Advancing llm reasoning across all domains. _arXiv preprint arXiv:2505.14652_, 2025. 7, 20


MAA. American mathematics competitions. In _American Mathematics Competitions_, 2023. 6, 21


Xinji Mai, Haotian Xu, Xing W, Weinong Wang, Yingying Zhang, and Wenqiang Zhang. Agent RL
Scaling Law: Agent RL with Spontaneous Code Execution for Mathematical Problem Solving.
_arXiv preprint arXiv:2505.07773_, 2025. 2, 10


Gr´egoire Mialon, Cl´ementine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia:
a benchmark for general ai assistants. In _The_ _Twelfth_ _International_ _Conference_ _on_ _Learning_
_Representations (ICLR)_, 2023. 6, 21


Moonshot AI. Kimi-Researcher: End-to-End RL Training for Emerging Agentic Capabilities.
[https://moonshotai.github.io/Kimi-Researcher/, June 2025.](https://moonshotai.github.io/Kimi-Researcher/) 2


Sumeet Ramesh Motwani, Chandler Smith, Rocktim Jyoti Das, Rafael Rafailov, Ivan Laptev,
Philip HS Torr, Fabio Pizzati, Ronald Clark, and Christian Schroeder de Witt. Malt: Improving reasoning with multi-agent llm training. _arXiv preprint arXiv:2412.01928_, 2024. 2, 5, 10


12


Chanwoo Park, Seungju Han, Xingzhi Guo, A. Ozdaglar, Kaiqing Zhang, and Joo-Kyung Kim. MAPoRL: Multi-agent post-co-training for collaborative large language models with reinforcement
learning. In _Annual Meeting of the Association for Computational Linguistics (ACL_, 2025. URL
[https://api.semanticscholar.org/CorpusId:276580906.](https://api.semanticscholar.org/CorpusId:276580906) 2, 5, 10


Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring
and narrowing the compositionality gap in language models. In _Findings_ _of the_ _Association_ _for_
_Computational Linguistics:_ _EMNLP 2023_, pp. 5687–5711, 2023. 6, 21


Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-T¨ur, Gokhan
Tur, and Heng Ji. ToolRL: Reward is all tool learning needs. _arXiv preprint arXiv:2504.13958_,
2025a. 2, 3, 10


Cheng Qian, Emre Can Acikgoz, Hongru Wang, Xiusi Chen, Avirup Sil, Dilek Hakkani-T¨ur,
Gokhan Tur, and Heng Ji. SMART: Self-aware agent for tool overuse mitigation. In _Findings_
_of the Association for Computational Linguistics:_ _ACL 2025_, pp. 4604–4621, 2025b. 10


David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In _First Conference on Language Modeling_, 2024. 6, 21


John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region
policy optimization. In _International Conference on Machine Learning (ICML)_, pp. 1889–1897.
PMLR, 2015. 18


Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. Enhancing
retrieval-augmented large language models with iterative retrieval-generation synergy. In _Find-_
_ings_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics:_ _EMNLP_ _2023_, pp. 9248–9274, 2023. 7,
20


Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. _arXiv preprint arXiv:2402.03300_, 2024. 3


Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang,
and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement
learning. _arXiv preprint arXiv:2503.05592_, 2025. 2, 10


Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Yan Zhang,
Fei Huang, and Jingren Zhou. Zerosearch: Incentivize the search capability of llms without
searching. _arXiv preprint arXiv:2505.04588_, 2025. 7, 10, 20


Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop
questions via single-hop question composition. _Transactions of the Association for Computational_
_Linguistics (TACL)_, 10:539–554, 2022. 6, 21


Hanlin Wang, Chak Tou Leong, Jiashuo Wang, Jian Wang, and Wenjie Li. SPA-RL: Reinforcing
llm agents via stepwise progress attribution. _arXiv preprint arXiv:2505.20732_, 2025a. 10


Hongru Wang, Cheng Qian, Wanjun Zhong, Xiusi Chen, Jiahao Qiu, Shijue Huang, Bowen Jin,
Mengdi Wang, Kam-Fai Wong, and Heng Ji. Acting less is reasoning more! teaching model to
act efficiently. _arXiv preprint arXiv:2504.14870_, 2025b. [URL https://arxiv.org/pdf/](https://arxiv.org/pdf/2504.14870)
[2504.14870.](https://arxiv.org/pdf/2504.14870) 10


Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin,
Kefan Yu, Minh Nhat Nguyen, Licheng Liu, et al. RAGEN: Understanding self-evolution in llm
agents via multi-turn reinforcement learning. _arXiv preprint arXiv:2504.20073_, 2025c. 2, 10


Ziliang Wang, Xuhui Zheng, Kang An, Cijun Ouyang, Jialu Cai, Yuhang Wang, and Yichao Wu.
Stepsearch: Igniting llms search ability via step-wise proximal policy optimization. _arXiv preprint_
_arXiv:2505.15107_, 2025d. 7, 10, 20


13


Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun
Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multiagent conversations. In _First_ _Conference_ _on_ _Language_ _Modeling_ _(COLM)_, 2024. 2, 4, 7, 10,
20


Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Xiaosen Zheng, Zejun Ma, and Bo An. Simpletir: End-to-end reinforcement learning for multi-turn tool-integrated reasoning. _arXiv preprint_
_arXiv:2509.02479_, 2025. 2, 10


Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang.
Learning to reason under off-policy guidance. _arXiv preprint arXiv:2504.14945_, 2025. 7, 20


An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang,
Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang,
Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,
Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. _arXiv preprint_
_arXiv:2412.15115_, 2024a. 6, 7, 19


An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical
expert model via self-improvement. _arXiv preprint arXiv:2409.12122_, 2024b. 7, 20


Hang Yang, Hao Chen, Hui Guo, Yineng Chen, Ching-Sheng Lin, Shu Hu, Jinrong Hu, Xi Wu,
and Xin Wang. Llm-medqa: Enhancing medical question answering through case studies in large
language models. _arXiv preprint arXiv:2501.05464_, 2024c. 6


Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. HotpotQA: A dataset for diverse, explainable multi-hop question
answering. In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language_
_Processing (EMNLP)_, pp. 2369–2380, 2018. 6, 21


Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian
Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system
at scale. _arXiv preprint arXiv:2503.14476_, 2025. 3


Siliang Zeng, Quan Wei, William Brown, Oana Frunza, Yuriy Nevmyvaka, and Mingyi Hong. Reinforcing multi-turn reasoning in llm agents via turn-level credit assignment. _arXiv_ _preprint_
_arXiv:2505.11821_, 2025a. 10


Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerlzoo: Investigating and taming zero reinforcement learning for open base models in the wild. _arXiv_
_preprint arXiv:2503.18892_, 2025b. 2, 7, 19, 20


Shaokun Zhang, Yi Dong, Jieyu Zhang, Jan Kautz, Bryan Catanzaro, Andrew Tao, Qingyun Wu,
Zhiding Yu, and Guilin Liu. Nemotron-research-tool-n1: Tool-using language models with reinforced reasoning. _arXiv preprint arXiv:2505.00024_, 2025. 2, 10


14


TABLE OF CONTENTS


**A** **Training Algorithm of AGENTFLOW** **16**


**B** **Theoretical Analysis of Flow-GRPO** **17**


B.1 Preliminaries and Notation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17


B.2 Equivalence Proof for Optimization Objectives . . . . . . . . . . . . . . . . . . . 17


B.3 Convergence Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18


**C** **Experimental Details** **19**


C.1 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19


C.2 Evaluation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19


C.3 Compared Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19


C.4 Evaluation Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20


**D** **More Discussion about Experiment Results** **22**


D.1 Main Result Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22


D.2 In-depth Analysis of Optimized Planning . . . . . . . . . . . . . . . . . . . . . . 23


**E** **Instruction Templates in AGENTFLOW** **24**


E.1 Modules and Memory . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24


E.2 Toolset Metadata . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29


E.3 LLM-based Judging . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34


**F** **Case Studies** **35**


F.1 Example 1: Efficient Search for Simple Tasks . . . . . . . . . . . . . . . . . . . . 35


F.2 Example 2: Spontaneous Brute-force . . . . . . . . . . . . . . . . . . . . . . . . . 36


F.3 Example 3: A Good Initial Plan is Essential . . . . . . . . . . . . . . . . . . . . . 38


F.4 Example 4: Robust Self-Correction and Adaptation . . . . . . . . . . . . . . . . . 40


F.5 Example 5: New Combo: Retrieve with Specific URL . . . . . . . . . . . . . . . . 42


F.6 Example 6: Rapid and Correct Physics Calculation . . . . . . . . . . . . . . . . . 44


F.7 Example 7: Multi-Source Cross-Verification . . . . . . . . . . . . . . . . . . . . . 46


15


A TRAINING ALGORITHM OF AGENTFLOW


We provide a flowchart of the overall training algorithm of AGENTFLOW (§3) in Algorithm 1.


**Algorithm 1** In-the-Flow Optimization for AGENTFLOW


**Require:** Dataset _D_, Action Planner policy _πθ_, Tool Executor _E_, Executive Verifier _V_, Solution
Generator _G_, Toolset _K_, and Shared Evolving Memory _M_
**Ensure:** Optimized Action Planner parameters _θ_ _[⋆]_

1: **for** each training iteration **do**
2: **for** each query–label pair ( _q, y_ _[∗]_ ) _∼D_ **do**
3: **1.** **IN-THE-FLOW ROLLOUT GENERATION**
4: Initialize: _t ←_ 1, _M_ _[t]_ _←_ _q_
5: **repeat**
6: _a_ _[t]_ _∼_ _πθ_ ( _a_ _[t]_ _| q, K, M_ _[t]_ ) _{Plan Action}_
7: _e_ _[t]_ _∼E_ ( _e_ _[t]_ _| a_ _[t]_ _, K_ ) _{Execute Action}_
8: _v_ _[t]_ _∼V_ ( _v_ _[t]_ _| q, e_ _[t]_ _, M_ _[t]_ ) _{Verify Result}_
9: _M_ _[t]_ [+1] = _f_ mem( _M_ _[t]_ _, a_ _[t]_ _, e_ _[t]_ _, v_ _[t]_ ) _{Update Memory}_
10: _t ←_ _t_ + 1
11: **until** termination condition met
12: _o ∼G_ ( _o | q, M_ _[T]_ ) _{Generate Final Solution}_
13: **2.** **REWARD COMPUTATION**
14: _R_ ( _a_ _[t]_ ) = _R_ [¯] ( _o, q, y_ _[∗]_ ) _,_ _∀t_ = 1 _, . . ., T_
15: **3.** **POLICY UPDATE**
16: Update the Action Planner policy _πθ_ by maximizing the Flow-GRPO objective (Eq. 5)
17: **end for**
18: **end for**
19: **return** optimized parameters _θ_ _[⋆]_


16


B THEORETICAL ANALYSIS OF FLOW-GRPO


B.1 PRELIMINARIES AND NOTATION


We adopt the notation from the paper to formalize our analysis.

**Definition B.1** (Core Components) **.** Here we list core definition of variables.


**Symbol and Description**


_πθ_ The trainable planner policy, parameterized by _θ_ .
_πθ_ old The behavior policy used to sample trajectories.
_s_ _[t]_ The state at turn _t_, defined as _s_ _[t]_ = ( _q, K, Mt_ ).
_a_ _[t]_ The action (a sequence of tokens) generated at state _s_ _[t]_, where _a_ _[t]_ _∼_ _πθ_ ( _· | s_ _[t]_ ).
_τ_ A trajectory of states and actions over _T_ time steps, defined as _τ_ = _{_ ( _s_ _[t]_ _, a_ _[t]_ ) _}_ _[T]_ _t_ =1 [.]
_R_ ( _τ_ ) The outcome-based reward for trajectory _τ_, where _R_ ( _τ_ ) _∈{_ 0 _,_ 1 _}_ .
_Aτ_ The group-normalized advantage for trajectory _τ_ . A crucial property is that the advantage is
constant for all timesteps within a trajectory defined in Eq. 7: _a_ _[t]_ = _Aτ_ _,_ _∀_ ( _s_ _[t]_ _, a_ _[t]_ ) _∈_ _τ_ .
_ρ_ _[t]_ _i,j_ The token-level importance sampling ratio, defined as:


                        - _a_ _[t]_ _i,j_ �� _sti_ _[, a][t]_ _i,_ 1: _j−_ 1�
_ρ_ _[t]_ _i,j_ [=] _π_ _[π]_ _θ_ _[θ]_ old� _a_ _[t]_ _i,j_ �� _sti_ _[, a][t]_ _i,_ 1: _j−_ 1� _._


_L_ clip( _ρ, A_ ) The PPO clipped objective term, defned as _L_ clip( _ρ, A_ ) = min( _ρA,_ clip( _ρ,_ 1 _−_ _ϵ,_ 1 + _ϵ_ ) _A_ ).


**Definition** **B.2** (Objective Functions) **.** The _global_ _policy_ _objective_ is the expected trajectory-level
reward:
_J_ ( _θ_ ) := E _τ_ _∼πθ_ [ _R_ ( _τ_ )] _._ (8)


The _single-turn optimization objective_ for a given state _s_ _[t]_ is defined as:


- _L_ clip( _ρ_ _[t]_ _i,j_ _[, A]_ _i_ _[t]_ [)]


_j_ =1





 _._ (9)


_J_ local( _θ_ ; _s_ _[t]_ ) := E _at∼πθ_ old ( _·|st_ )




 [1]

_|a_ _[t]_ _|_


_|a_ _[t]_ _|_


The full Flow-GRPO objective function in the multi-turn setting is given by:


_G_


_i_ =1


1
_Ti_





 _−_ _β_ DKL( _πθ∥π_ ref) _._ (10)


_|a_ _[t]_ _i_ _[|]_


- _L_ clip( _ρ_ _[t]_ _i,j_ _[, A]_ _i_ _[t]_ [)]


_j_ =1


_Ti_


_t_ =1


1
_|a_ _[t]_ _i_ _[|]_


_J_ Flow-GRPO( _θ_ ) := E ( _q,y_ _[∗]_ ) _∼D_
_{τi}_ _[G]_ _i_ =1 _[∼][π][θ]_ old




 [1]

_G_


B.2 EQUIVALENCE PROOF FOR OPTIMIZATION OBJECTIVES


**Theorem B.1.** _In Flow-GRPO, maximizing the global multi-turn objective is mathematically equiv-_
_alent_ _to_ _maximizing_ _the_ _expected_ _token-level_ _local_ _objective_ _at_ _each_ _time_ _step_ _under_ _the_ _on-policy_
_induced_ _state_ _distribution,_ _given_ _standard_ _sampling_ _assumptions_ _(trajectories_ _sampled_ _i.i.d._ _from_
_the policy with fixed finite turn T_ _)._


_Proof._ Let’s denote the clipping part of the Flow-GRPO objective as _J_ clip( _θ_ ).


First, by the linearity of expectation, we can simplify the expectation over a group of _G_ trajectories.
Since the trajectories _{τi}_ are sampled independently and identically (i.i.d.) from the behavior policy
_πθ_ old, the expectation of their average is equal to the expectation over a single trajectory.





- _L_ clip( _ρ_ _[t]_ _i,j_ _[, A][t]_ _i_ [)]


_j_ =1










 [1]

_|a_ _[t]_ _i_ _[|]_


_|a_ _[t]_ _i_ _[|]_


_G_


_i_ =1


1
_Ti_


_Ti_


_t_ =1


_J_ clip( _θ_ ) = E( _q,y∗_ ) _∼D_


= E( _q,y∗_ ) _∼D_





E _{τi}Gi_ =1 _[∼][π][θ]_ old





E _τ_ _∼πθ_ old ( _·|q_ )


 [1]

_G_








 (11)


- _L_ clip( _ρ_ _[t]_ _j_ _[, A][τ]_ [)]


_j_ =1




 [1]

_T_


_T_


_t_ =1











 _._ (12)




 [1]

_|a_ _[t]_ _|_


_|a_ _[t]_ _|_


 _._ (12)


17


Here, _τ_ = _{_ ( _s_ _[t]_ _, a_ _[t]_ ) _}_ _[T]_ _t_ =1 [represents a single, arbitrarily sampled trajectory with advantage] _[ A][τ]_ [.]


Next, we can re-interpret the expectation over trajectories as an expectation over the state-visitation
distribution induced by the policy _πθ_ old. Let _d_ _[π][θ]_ [old] be the on-policy distribution of states visited,
where each state _s_ _[t]_ in a trajectory of length _T_ is weighted by 1 _/T_ . The expectation can be rewritten
as:







 [1]

_|a_ _[t]_ _|_





E _at∼πθ_ old ( _·|st_ )


- _L_ clip( _ρ_ _[t]_ _j_ _[, A][t]_ [)]


_j_ =1








_J_ clip( _θ_ ) = E( _q,y∗_ ) _∼D_





E _st∼dπθ_ old


_|a_ _[t]_ _|_








 _._ (13)


Note that _A_ _[t]_ is the advantage corresponding to the trajectory from which _s_ _[t]_ was sampled.


We now recognize that the inner expectation is precisely the definition of the local, per-state objective, _J_ local( _θ_ ; _s_ _[t]_ ).


_J_ clip( _θ_ ) = E( _q,y∗_ ) _∼D,_ _st∼dπθ_ old         - _J_ local( _θ_ ; _s_ _[t]_ )� _._ (14)


Adding the KL-divergence term back, we arrive at the final equivalence:


_J_ Flow-GRPO( _θ_ ) = E( _q,y∗_ ) _∼D,_ _st∼dπθ_ old     - _J_ local( _θ_ ; _s_ _[t]_ )� _−_ _β_ D _KL_ ( _πθ∥π_ ref) _._ (15)


This proves that maximizing the global multi-turn Flow-GRPO objective is equivalent to maximizing the expected token-level local objective at each time step under the on-policy induced state
distribution.


B.3 CONVERGENCE ANALYSIS


Having established the structural validity of the objective, we now analyze its convergence properties. The analysis builds on the monotonic improvement guarantee provided by trust-region methods (Schulman et al., 2015).
**Lemma** **B.2** (Policy Performance Difference) **.** _For_ _two_ _policies_ _πθ_ _and_ _πθ_ old _,_ _the_ _difference_ _in_ _ex-_
_pected return can be expressed as:_


_,_ (16)


_J_ ( _θ_ ) _−J_ ( _θ_ old) = E _τ_ _∼πθ_


- _T_


- _Aθ_ old( _s_ _[t]_ _, a_ _[t]_ )


_t_ =1


_where Aθ_ old _is the advantage function under the old policy._


This lemma enables the construction of a lower bound on policy improvement.

**Theorem B.3** (Monotonic Improvement Guarantee) **.** _Define the surrogate objective_


_Lθ_ old( _θ_ ) = E _τ_ _∼πθ_ old


- _T_


_t_ =1


         
_πθ_ ( _a_ _[t]_ _|s_ _[t]_ )
_πθ_ old( _a_ _[t]_ _|s_ _[t]_ ) _[A][θ]_ [old][(] _[s][t][, a][t]_ [)]


_._ (17)


_Then the performance improvement satisfies the lower bound_


_J_ ( _θ_ ) _−J_ ( _θ_ old) _≥_ _Lθ_ old( _θ_ ) _−_ _C ·_ D [¯] KL( _πθ_ old _, πθ_ ) _,_ (18)

_where C_ _>_ 0 _is a constant depending on the horizon and reward scale, and_ D [¯] KL _denotes the average_
_KL-divergence between the two policies._


By optimizing the right-hand side of the above inequality, we are guaranteed to improve the performance of _πθ_ . Therefore, for policies _πθ_ _[t]_ [and] _[ π]_ _θ_ _[t]_ [+1] obtained from iterations _t_ and _t_ + 1, we have:


_J_ ( _θ_ _[t]_ [+1] ) _≥J_ ( _θ_ _[t]_ ) _._ (19)


**Conclusion.** This analysis establishes that Flow-GRPO optimizes a valid surrogate objective and
guarantees monotonic policy improvement, thereby converging reliably to a locally optimal policy.


18


C EXPERIMENTAL DETAILS


C.1 TRAINING DETAILS


We provide further details on the training setup for AGENTFLOW. Our Flow-GRPO implementation
uses a learning rate of 1 _×_ 10 _[−]_ [6] . The Action Planner generates actions with a sampling temperature
of 0 _._ 5 to balance exploration and exploitation. To prevent policy collapse and stabilize training, we
incorporate a KL-divergence penalty against a reference policy with a coefficient _β_ = 0 _._ 001. The
maximum output length for the planner is set to 2048 tokens to ensure complete exploration during
rollouts.


To accelerate the training speed, we limit the maximum number of turns per rollout to 3. The finaloutcome reward signal (Eq. 4) is provided by an LLM-as-judge, for which we use _GPT-4o_ . All
tool calls are executed synchronously with a 500-second timeout to handle external service latency
robustly. The LLM engines within the tools are set to a temperature of 0.0 to ensure deterministic
and stable outputs. The full training process was conducted on 8 NVIDIA A100 GPUs. Further
details on agent prompts and the memory update mechanism are provided in §E.1.


C.2 EVALUATION DETAILS


Here, we outline the specifics of our evaluation protocol. For evaluation, we increase the maximum
number of turns per rollout to _T_ = 10 to allow for more extensive and deeper reasoning. The
planner’s sampling temperature is set to 0.7 to encourage diverse solution paths. Unless otherwise
specified, all tool LLM engines are initialized with Qwen2.5-7B-Instruct.


For fair and consistent evaluation, we adopt the previous work’s methodology while standardizing
tools: we replace search tools in search-enhanced models with our Google Search tool and code
tools in code-enhanced models with our Python Coder tool. We use GPT-4o as an LLM-based judge
to determine the correctness of final answers. This approach provides a robust measure of semantic
and numerical equivalence, which is critical for complex reasoning tasks. The specific judging
prompt is detailed in §E.3, and additional information on evaluation datasets can be found in §C.4.
To mitigate randomness, we report the average accuracy with standard deviation across three trials
for all experiments.


C.3 COMPARED BASELINES


**Proprietary LLMs:**


- **Qwen2.5 Series** (Yang et al., 2024a), created by Alibaba, comes in multiple configurations. These
models undergo training on multilingual corpora covering 29 different languages, demonstrating
superior performance in cross-lingual applications. Furthermore, Qwen2.5 showcases robust proficiency in programming and mathematical domains.


- **Llama-3** **Series** (Dubey et al., 2024), created by Meta AI, encompasses various iterations.
Each model configuration within the Llama family provides dual versions: foundational and
instruction-following variants. Training incorporates diverse dataset combinations spanning multiple domains and linguistic varieties. The Llama model family demonstrates excellent results in
logical reasoning, software development, and cross-lingual comprehension evaluations. Through
progressive enhancements in fine-tuning methodologies and expanded sequence lengths, these
models become more applicable to practical deployment scenarios.


- **GPT-4o** **Series** (Hurst et al., 2024), produced by OpenAI, includes several model variants such
as GPT-4o and GPT-4o-mini, with training leveraging extensive multimodal datasets encompassing text, vision, and audio modalities. The series achieves outstanding performance in complex
reasoning tasks, creative generation, and multimodal understanding benchmarks with continuous
refinements in alignment techniques and enhanced processing capabilities.


**Reasoning LLMs:**


- **SFT** (Zeng et al., 2025b) serves as our basic baseline following Search-R1 (Jin et al., 2025). We
fine-tune models using supervised fine-tuning on GPT-4o-generated reasoning chains.


19


- **SimpleRL-Zoo** (Zeng et al., 2025b) investigates zero reinforcement learning training across 10
diverse base models spanning different families and sizes using GRPO algorithm with simple
rule-based rewards, achieving substantial improvements in reasoning accuracy.

- **Open-Reasoner-Zero** (Hu et al., 2025a) presents the first open-source implementation of largescale reasoning-oriented RL training using PPO with GAE and straightforward rule-based rewards, without KL regularization. The framework demonstrates that minimalist design can successfully scale both response length and benchmark performance.

- **General-Reasoner** (Ma et al., 2025) extends LLM reasoning capabilities beyond mathematics
to diverse domains using RLVR through a 230K verifiable reasoning questions dataset spanning
physics, chemistry, and finance.

- **LUFFY** (Yan et al., 2025) addresses limitations in on-policy RLVR by introducing an off-policy
framework that augments training with external reasoning demonstrations using Mixed Policy
GRPO and regularized importance sampling.


**Search-Integrated Reasoning LLMs:**


- **Iter-RetGen** (Shao et al., 2023) addresses limitations in retrieval-augmented language models by
introducing iterative retrieval-generation synergy, where a model’s previous response serves as
context for retrieving more relevant knowledge in subsequent iterations.

- **Search-R1** (Jin et al., 2025) represents a reinforcement learning approach that develops a model
from the ground up to invoke search functionality throughout the reasoning process.

- **ZeroSearch** (Sun et al., 2025) addresses high API costs in RL-based search training by using an
LLM to simulate search engines, employing lightweight supervised fine-tuning to transform an
LLM into a retrieval module that generates both useful and noisy documents. The framework
combines this with a curriculum-based rollout strategy that progressively degrades document
quality, achieving better performance than real search engine-based methods while incurring zero
API costs.

- **ReSearch** (Chen et al., 2025) proposes a reinforcement learning framework that trains LLMs
to integrate search operations as components of the reasoning chain without supervised data on
reasoning steps, treating search decisions as guided by text-based thinking.

- **StepSearch** (Wang et al., 2025d) addresses the sparse reward problem in multi-hop reasoning
by training search LLMs using step-wise proximal policy optimization with intermediate rewards
and token-level process supervision based on information gain and redundancy penalties.

- **VerlTool** (Jiang et al., 2025) addresses fragmentation and synchronization bottlenecks in Agentic
Reinforcement Learning with Tool use by introducing a unified modular framework that extends
beyond single-turn RLVR paradigms, providing upstream VeRL alignment and unified tool management with asynchronous rollout execution achieving near 2× speedup.


**Code-Integrated Reasoning LLMs:**


- **TIR** (Yang et al., 2024b) is a basic baseline that demonstrates the model’s ability to generate code
for tool utilization. In our implementation, we directly prompt the model to write code that calls
the programming interpreter and processes the returned results to generate the final answer.

- **ToRL** (Li et al., 2025b) is a code-enhanced architecture developed via reinforcement learning
that empowers models to independently activate code execution environments for mathematical
reasoning tasks.


**Training-free Agentic System**


- **AutoGen** (Wu et al., 2024) introduces an agentic conversation framework that enables developers
to build LLM applications through conversable agents that can operate using combinations of
LLMs, human inputs, and tools.


C.4 EVALUATION DATASETS


We provide a detailed introduction to the _search-intensive_ and _agentic_ benchmarks in our experiments as follows:


20


- **Bamboogle** (Press et al., 2023) presents a demanding multi-step reasoning dataset containing
manually constructed questions requiring up to four inferential steps. The dataset evaluates models’ capacity for intricate compositional reasoning across interconnected facts.


- **2Wiki** **(2WikiMultihopQA)** (Ho et al., 2020) constitutes a comprehensive multi-step QA corpus combining structured Wikidata knowledge with unstructured Wikipedia text. The dataset
encompasses varied question formats and annotated reasoning chains to facilitate interpretable
sequential inference. We randomly sample 100 examples as a test set for efficiency.


- **HotpotQA** (Yang et al., 2018) represents a widely-adopted question answering corpus featuring
multi-step queries constructed from Wikipedia entries. We randomly sample 100 examples as a
test set for efficiency.


- **Musique** (Trivedi et al., 2022) comprises a multi-step reasoning corpus requiring sequential inference where each reasoning stage depends on information derived from preceding steps. We
conduct evaluations using the development partition of this particularly challenging dataset. We
randomly sample 100 examples as a test set for efficiency.


- **GAIA** (Mialon et al., 2023) constitutes a benchmark engineered to assess general AI systems
and agents, demanding capabilities including sequential reasoning, web navigation, and comprehensive tool utilization skills. We utilize the text-exclusive portion of this dataset, designed to
challenge base language models in our experimental setup.


Furthermore, we also conduct a series of experiments on _math_ and _scientific reasoning_ benchmarks:


- **AIME24** (Art of Problem Solving, 2025) A collection of 30 demanding mathematical problems
sourced from the 2024 American Invitational Mathematics Examination (AIME), encompassing
algebra, geometry, number theory, and combinatorics. Each JSONL-formatted record contains
the problem identifier, question text, comprehensive solution methodology, and the final numerical result. Created to assess large language models’ sophisticated mathematical reasoning abilities, the dataset presents substantial difficulty, systematic multi-phase solutions, and distinctive
answers—establishing it as a robust benchmark for evaluating advanced analytical capabilities.


- **AMC23** (MAA, 2023) contains mathematical problems derived from the 2023 American Mathematics Competition, emphasizing areas such as functional equations and complex analysis.


- **GameOf24** (Lile, 2024) derives from the traditional numerical puzzle known as 24 (alternatively
called the 24 numbers game). The challenge requires utilizing four given numbers with fundamental arithmetic operations (addition, subtraction, multiplication, division) to create an expression
yielding 24. For instance, with numbers 4, 9, 10, and 13, a correct solution would be “(10  - 4)
× (13  - 9) = 24”. Successfully solving requires computational proficiency along with iterative
attempts to validate potential solutions. Each challenge is formatted as open-ended inquiries.


- **GPQA** or Graduate Level Google-Proof Q&A Benchmark (Rein et al., 2024) comprises a collection of demanding text-based multiple choice problems authored by subject specialists in biology,
physics, and chemistry, intentionally crafted to be “exceptionally challenging”. We randomly
sample 100 examples as a test set for efficiency.


- **MedQA** (Jin et al., 2021) features text-based multiple choice problems assembled from professional medical licensing examinations. Problems encompass comprehensive medical knowledge
and clinical reasoning skills.


21


D MORE DISCUSSION ABOUT EXPERIMENT RESULTS


D.1 MAIN RESULT ANALYSIS


Our main results are presented in Tables 1 and 2. Overall, AGENTFLOW consistently outperforms all
baseline models across diverse domains, including search-intensive tasks, agentic tasks, and mathematical and scientific reasoning tasks. These comprehensive results yield several key insights:


**Monolithic LLMs are insufficient for complex reasoning.** While scaling up model size (from 7B
model to GPT-4o) improves average performance, their monolithic nature presents limitations when
facing complex tasks that require multi-turn reasoning and sub-goal decomposition. In contrast, our
proposed AGENTFLOW consistently outperforms these larger models. Specifically, it achieves an
average improvement of 8.2% over GPT-4o on search-intensive tasks (57.3% vs. 49.1% in Table 1),
and a remarkable 15.8% gain over GPT-4o on agentic tasks (33.1% vs. 17.3% in Table 1). For
mathematical reasoning benchmarks, AGENTFLOW obtains a substantial improvement of 16.4%
over GPT-4o (51.5% vs. 35.1% in Table 2). Furthermore, it surpasses the strong Llama-3.3-70B
by 12.5% on scientific reasoning tasks (63.5% vs. 51.0% in Table 2). These results demonstrate
that the carefully designed agentic system of AGENTFLOW, despite being built on a 7B-parameter
backbone, can deliver superior and more efficient performance compared to substantially larger
monolithic LLMs.


**Specialized** **reasoning** **models** **exhibit** **strong** **in-domain** **focus** **but** **limited** **generalizability.**
While domain-specific fine-tuning and tailored tool integration provide clear benefits over base
LLMs, they fail to deliver robust cross-domain performance due to fundamental scaling limitations.
Our evaluation across three reasoning domains substantiates these limitations. On search-intensive
tasks, specialized models such as Search-R1 (33.3%) and VerlTool (39.0%) perform well within
their narrow scope yet fall substantially short of AGENTFLOW (57.3%) as shown in Table 1. Similarly, in mathematical reasoning, methods like SimpleRL-reason (36.6%) and ToRL (37.0%) trail
significantly behind AGENTFLOW (51.5%) in Table 2. Even in scientific reasoning, where models
such as Luffy (55.5%) offer competitive results, they are consistently surpassed by AGENTFLOW
(63.5%) in Table 2. These findings demonstrate that while specialized reasoning models excel within
narrow domains, their reliance on a single monolithic policy introduces poor generalization, making
them brittle when confronted with diverse, cross-domain challenges.


**AGENTFLOW demonstrates superior, versatile reasoning through its adaptive agentic system.**
AGENTFLOW establishes a new state-of-the-art agentic system by achieving an average accuracy
of 57.3% on search-intensive tasks, 33.1% on agentic tasks, 51.5% on mathematical reasoning, and
63.5% on scientific reasoning. Our method’s advantage stems from combining an agentic system
with targeted planning policy refinement via on-policy reinforcement learning in an online fashion. When compared to AutoGen—a general agent framework with the same backbone model—
AGENTFLOW demonstrates a massive improvement of 14.9% on search tasks and 19.9% on math
tasks. This underscores that the core advantage comes from our dedicated trainable agentic system
that integrates our novel Flow-GRPO for in-system on-policy optimization, enabling effective agent
planning and tool utilization to solve complex, long-horizon problems across diverse domains.


Qwen2.5-7B-Instruct GPT-4o


graded from Qwen-2.5-7B-Instruct to GPT-4o.


|Acc:19.2% Acc: 25.2% (+6.21%)<br>After Finet-tuning<br>-1.5<br>38.7 38.5 13.6 -4.7<br>13.6<br>+5.2<br>13.6<br>18.4<br>-2.2<br>3.1<br>0.9|Acc:19.2%|Col3|Col4|Col5|Acc: 25.2% (+6.21%)|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|38.7<br>**Acc:19.2%**<br>38.5<br>13.6<br>0.9<br>**Acc: 25.2% (+6.21%)**<br>18.4<br>~~**-1.5**~~<br>**-2.2**<br>**-4.7**<br>~~After Finet-tuning~~<br>3.1<br>**+5.2**<br>~~13.6~~<br>~~13.6~~|~~After Finet-tu~~|~~After Finet-tu~~|~~After Finet-tu~~|~~After Finet-tu~~|~~**-1.5**~~<br><br>~~ ning~~|~~**-1.5**~~<br><br>~~ ning~~|~~**-1.5**~~<br><br>~~ ning~~|~~**-1.5**~~<br><br>~~ ning~~|~~**-1.5**~~<br><br>~~ ning~~|
|38.7<br>**Acc:19.2%**<br>38.5<br>13.6<br>0.9<br>**Acc: 25.2% (+6.21%)**<br>18.4<br>~~**-1.5**~~<br>**-2.2**<br>**-4.7**<br>~~After Finet-tuning~~<br>3.1<br>**+5.2**<br>~~13.6~~<br>~~13.6~~||38.7||38.5||13.6|**-4.7**<br><br>~~13.6~~|**-4.7**<br><br>~~13.6~~|**-4.7**<br><br>~~13.6~~|
|38.7<br>**Acc:19.2%**<br>38.5<br>13.6<br>0.9<br>**Acc: 25.2% (+6.21%)**<br>18.4<br>~~**-1.5**~~<br>**-2.2**<br>**-4.7**<br>~~After Finet-tuning~~<br>3.1<br>**+5.2**<br>~~13.6~~<br>~~13.6~~|||||||**+5.2**<br>|||
|38.7<br>**Acc:19.2%**<br>38.5<br>13.6<br>0.9<br>**Acc: 25.2% (+6.21%)**<br>18.4<br>~~**-1.5**~~<br>**-2.2**<br>**-4.7**<br>~~After Finet-tuning~~<br>3.1<br>**+5.2**<br>~~13.6~~<br>~~13.6~~|||18.4||||~~13.6~~|||
|38.7<br>**Acc:19.2%**<br>38.5<br>13.6<br>0.9<br>**Acc: 25.2% (+6.21%)**<br>18.4<br>~~**-1.5**~~<br>**-2.2**<br>**-4.7**<br>~~After Finet-tuning~~<br>3.1<br>**+5.2**<br>~~13.6~~<br>~~13.6~~|||0.9<br>**-2.2**|0.9<br>**-2.2**|0.9<br>**-2.2**|0.9<br>**-2.2**||||


Figure 11: **Tool call optimization on Musique** .
AGENTFLOW’s planner increases Web Search
usage after Flow-GRPO training.


Base Generator Google Search
Web Search Wikipedia Search


22


D.2 IN-DEPTH ANALYSIS OF OPTIMIZED PLANNING


**AGENTFLOW** **adapts** **to** **inference-time** **tool** **scaling.** We scale the tools—the Base Generator
and Python Coder—to GPT-4o-powered versions. Empirical results on search and math datasets
(Figure 10) show that AGENTFLOW, when using these GPT-4o-powered tools, substantially outperforms its performance with Qwen2.5-7B-Instruct-powered tools, achieving improvements of 1.0%
on GAIA, 6.0% on AMC23, and a notable 13.0% on HotpotQA. This finding further supports a
consistent trend: after in-the-flow RL training, the planner can adaptively leverage improvements in
the underlying tools to enhance the agentic system’s overall performance.


**Flow-GRPO spontaneous tool usage preference change.** We further compare tool usage distributions before and after in-the-flow RL training on Musique. Figure 11 shows that due to Musique’s
need for a diverse source of information, Flow-GRPO optimizes the planner to increase Web Search
to delve deeper into the URL provided by other search tools. This maneuver presents a steady
performance improvement of 6.1%.


80


60


40


20


80


60


40


20


0

|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||68.8<br>|72.3||~~Before tu~~<br>After tuni|~~ ning~~<br> ng|
||68.8<br>|72.3||||
|53.6<br>|~~63.0~~<br>|||||
|||||29.1<br>||
|||14.3<br>|14.3<br>|13.3<br><br>|20.0|

Bamboogle 2Wiki GAIA AIME24


0

|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||69.6<br>|77.2||~~Before tu~~<br>After tuni|~~ ning~~<br> ng|
||69.6<br>|77.2||||
|58.4|60.0|||||
|||||33.1<br>|40.0|
|||17.2|17.2|16.7||

Bamboogle 2Wiki GAIA AIME24


Figure 12: Flow-GRPO fine-tuning offers consistent gains on AGENTFLOW as the backbone model
size scales from 3B to 7B.


**More** **evidence** **of** **training** **scaling** **in** **backbone** **size.** We further investigate how the backbone
LLM scale affects AGENTFLOW’s performance and the efficacy of Flow-GRPO on GameOf24,
AMC23, and MedQA. We construct two versions of the system: one using _Qwen2.5-3B-Instruct_ and
another using _Qwen2.5-7B-Instruct_ for all four modules (planner, executor, verifier, and generator)
as well as the associated tools. In both versions, only the planner is fine-tuned with Flow-GRPO.
As shown in Figure 12, Flow-GRPO fine-tuning consistently improves performance across tasks
for both backbones. These results demonstrate that our in-the-flow optimization is effective across
model capacities, enhancing AGENTFLOW regardless of LLM size.


23


E INSTRUCTION TEMPLATES IN AGENTFLOW


E.1 MODULES AND MEMORY


E.1.1 ACTION PLANNER


Tool Metadata can be found in §E.2.


24


E.1.2 TOOL EXECUTOR


25


E.1.3 EXECUTION VERIFIER


26


E.1.4 SOLUTION GENERATOR


27


E.1.5 EVOLVING MEMORY


Our shared evolving memory system creates a deterministic, structured record that captures the
reasoning process across three integrated agents: the _Action Planner_, _Tool Executor_, and _Execution_
_Verifier_ . By sequentially stacking crucial information from each action step, the system enables
transparent state tracking, controllable behavior, and bounded context growth.


The memory reading and matching process employs regular expressions to parse outputs generated
by different system components, adhering to standardized formats defined in their respective component instructions. For the _Action_ _Planner_, we use a relatively permissive regular expression to
extract key information. Specifically, it matches the content immediately following: _Sub-Goal_ as
the sub-goal and the content following; _Tool Name_ as the selected tool. This extracted information
is then used to populate the next memory entry. For the _Tool Executor_, the regular expression is designed to capture the entire _Command_ line starting with execution = tool.execute(...).
Additionally, the value passed to the _Query_ parameter within this command is parsed and saved into
the memory for future reference. All results returned by the tools are directly stored in the _Result_
field of the memory. The _Verification Status_ is extracted from _Execution Verifier_, including a brief
analysis of the current tool result and previous memory, and then it gives a conclusion whether the
loop needs to be CONTINUE or STOP.


28


E.2 TOOLSET METADATA


This section details the implementation and metadata of the tools used in our main results. We
employ a suite of specialized tools, each designed for distinct tasks. Below, we present core metadata
for each tool, including its functionality, input/output schema, limitations, and best practices.


E.2.1 BASE GENERATOR


29


E.2.2 PYTHON CODER


30


E.2.3 GOOGLE SEARCH


31


E.2.4 WIKIPEDIA SEARCH


Wikipedia search will first call Wikipedia API to retrieve relevant URLs with snippets. Then the
RAG (Retrieval-Augmented Generation) process begins by extracting raw text content from the
given webpage URL, cleaning it to remove HTML elements and retain only meaningful text. This
content is then split into overlapping chunks of approximately 200 words each, with a 20-word
overlap to preserve context across segments from the first 1M words in each URL. Next, both
the user’s query and the document chunks are embedded into the vector space using the OpenAI
text-embedding-3-small model. The system computes the cosine similarity between the
query embedding and each chunk embedding to rank the chunks by relevance. We set that the top
10 most similar chunks are selected and passed forward as context. And a base LLM engine will
summarize the extracted context.


Wikipedia search will first call Wikipedia API to retrieve relevant URLs with snippets.


[https://platform.openai.com/docs/models/text-embedding-3-small](https://platform.openai.com/docs/models/text-embedding-3-small)


32


E.2.5 WEB SEARCH


Web search will directly access the URL in the query. Then the RAG (Retrieval-Augmented Generation) process begins by splitting content from the page into overlapping chunks of approximately
200 words each, with a 20-word overlap to preserve context across segments from the first 1M words
in each URL. Next, both the user’s query and the document chunks are embedded into the vector
space using the OpenAI text-embedding-3-small model. The system computes the cosine similarity
between the query embedding and each chunk embedding to rank the chunks by relevance. We set
that the top 10 most similar chunks are selected and passed forward as context. And a base LLM
engine will summarize the extracted context.


[https://platform.openai.com/docs/models/text-embedding-3-small](https://platform.openai.com/docs/models/text-embedding-3-small)


33


E.3 LLM-BASED JUDGING


We employ GPT-4o as our judge model using a two-step “analyze-then-judge” instruction paradigm
to ensure both accuracy and efficiency.


34


F CASE STUDIES


In this section, we conduct a case study to demonstrate how our AGENTFLOW, coherent with
Flow-GRPO, enhances problem-solving performance with greater elegance, efficiency, and robustness. We present solution comparisons showing brief outputs from memory of the _Action Planner_
(Qwen2.5-7B-Instruct) before (w/o) tuning by Flow-GRPO and after (w/) Flow-GRPO tuning, with
the methodology detailed in §3.2.


F.1 EXAMPLE 1: EFFICIENT SEARCH FOR SIMPLE TASKS


This case demonstrates that, with Flow-GRPO tuning, the _Action Planner_ can effectively leverage
the search engine to retrieve correct answers for simple tasks in a highly efficient manner—unlike
the untuned baseline, which requires multiple trials.


35


F.2 EXAMPLE 2: SPONTANEOUS BRUTE-FORCE


This case demonstrates that, when tuned with Flow-GRPO, the _Action Planner_ first attempts several
solutions, recognizes their ineffectiveness, resorts to a brute-force approach, and finally verifies the
result using a search engine.


36


37


F.3 EXAMPLE 3: A GOOD INITIAL PLAN IS ESSENTIAL


This case demonstrates that a well-crafted initial search with a highly relevant query is far more
effective than issuing numerous wrong paths. When tuned with Flow-GRPO, the _Action Planner_ in
AGENTFLOW can identify the optimal search engine and formulate the most effective query, leading
to a correct and targeted answer in a single trial.


38


39


F.4 EXAMPLE 4: ROBUST SELF-CORRECTION AND ADAPTATION


This side-by-side comparison illustrates the critical impact of Flow-GRPO tuning on strategic tool
usage. The trained AGENTFLOW agent demonstrates adaptive planning—recovering from failed
searches, refining input formulations, and ultimately achieving a correct solution in a single effective trial. In contrast, the untrained agent, despite accessing the correct information early, fails to
properly utilize the Python Coder tool and becomes trapped in a repetitive error loop, unable to
learn or adjust. This highlights Flow-GRPO’s role in enabling not just tool selection, but _strategic_
_resilience_ and _goal-directed reasoning_ .


40


41


F.5 EXAMPLE 5: NEW COMBO: RETRIEVE WITH SPECIFIC URL


This case highlights how both agents eventually succeed, but with markedly different efficiency
and strategy. The Flow-GRPO-tuned AGENTFLOW agent learns to refine its queries effectively
and—upon recognizing the limitations of Wikipedia search—switches tools strategically to a targeted and the most task-solving relevant web search, achieving success with minimal redundancy.
In contrast, the untrained agent persists in issuing dense, ineffective queries within the same tool despite diminishing returns, only escaping the loop by eventually switching to Google Search. While
both reach the correct answer, the latter exhibits inefficient exploration and delayed adaptation;
furthermore, with no path consistency, underscoring Flow-GRPO’s role in fostering not just correctness, but _strategic focus_ and _timely tool transition_ .


42


43


F.6 EXAMPLE 6: RAPID AND CORRECT PHYSICS CALCULATION


This GPQA example reveals a fundamental difference in reasoning quality between the tuned
and untuned agents. The Flow-GRPO-enhanced AGENTFLOW correctly identifies the core challenge—relativistic time dilation over interstellar distances—and applies the appropriate physicsbased computation in minimal steps, arriving at the correct answer (81 years) efficiently. In contrast,
the untrained agent misinterprets the astronaut’s age as the travel duration, leading to a cascade
of erroneous calculations across multiple tool calls. Despite eventually retrieving the distance via
search, it fails to integrate this information coherently or recognize its conceptual mistake. This
highlights that Flow-GRPO not only improves tool usage efficiency but also promotes _correct prob-_
_lem formulation_, enabling the agent to distinguish between proper time, coordinate time, and mission
constraints—a critical capability for complex scientific reasoning.


44


45


F.7 EXAMPLE 7: MULTI-SOURCE CROSS-VERIFICATION


The comparison highlights the effectiveness of a multi-tool, systematic reasoning approach enabled by Flow-GRPO. In the success case, the model leveraged sequential tool usage—starting with
Google Search, followed by targeted Wikipedia and Web Search—to accurately identify G¨ulc¸ic¸ek
Hatun as Olivera Despina’s mother-in-law through verified historical sources. Each step built upon
prior findings, ensuring robustness and precision. In contrast, the failure case without Flow-GRPO
relied on a single, improperly executed Wikipedia query without task decomposition that resulted
in a timeout and no meaningful output, leading to premature termination. This demonstrates that
Flow-GRPO enhances reasoning trace reliability, tool coordination, and overall task completion in
complex knowledge retrieval scenarios.


46


47