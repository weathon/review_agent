## peculative ctions ossless ramework for aster S A : A L F F gentic ystems A S


**Naimeng Ye** [∗] **,** **Arnav Ahuja** [∗] **,** **Georgios Liargkovas** [∗] **,** **Yunan Lu** [∗] **,** **Kostis Ka** ff **es, Tianyi Peng**
Columbia University
New York, New York, USA
{ `ny2336,` `aa5790,` `gl2902,` `yl4021,` `kk3664,` `tp2845` } `@columbia.edu`


Abstract


AI agents are increasingly deployed in complex, interactive environments, yet
their runtime remains a major bottleneck for training, evaluation, and real-world
use. Typical agent behavior unfolds sequentially, where each action requires an
API call that can incur substantial latency. For example, a game of chess between
two state-of-the-art agents can take hours. We introduce _speculative_ _actions_, a
lossless acceleration framework for general agentic systems. Inspired by speculative execution in microprocessors and speculative decoding in LLM inference,
our method uses faster models to predict likely future actions and executes them
in parallel, committing only when predictions match. We evaluate speculative
actions across gaming, e-commerce, and web search environments, and additionally study a lossy extension in an operating systems setting. Across domains, we
achieve up to 55% next-action prediction accuracy, translating into substantial latency reductions. Finally, we present a cost–latency analysis that formalizes the
tradeoff between speculative breadth and time savings. This analysis enables principled tuning and selective branch launching, to ensure multi-branch speculation
delivers practical speedups without prohibitive cost growth.


1 Introduction


Large language model (LLM)-driven agents are shifting from single-shot predictions to processes
that run inside rich environments: browsers, operating systems, game engines, e-commerce stacks,
and human workflows. These environments are not incidental; they determine what the agent can
observe and do, gate progress through interfaces and rate limits, and dominate end-to-end latency.
In practice, agent behavior unfolds as a sequence of environment steps (tool calls, Model Context
Protocol (MCP) server requests, human-in-the-loop queries, and further LLM invocations), each
with non-trivial round-trip time and cost. As capabilities improve, a new bottleneck emerges: timeto-action in the environment. Even when accuracy is high, an agent that pauses too long between
steps is impractical for interactive use or high-throughput automation.


**OS Tasks** **Deep Research** **Data Pipeline** **Kaggle Chess Game**
(Abhyankar et al., 2025) (OpenAI, 2025) (Jin et al., 2025) (Kaggle, 2025)


10–20 min 5–30 min 30–45 min 1 hour


Table 1: Estimated time state-of-the-art AI agents spend on various tasks/environments.


As shown in Table 1, AI agents may require tens of minutes to hours to complete a single run across
different environments, a cost that grows significantly when hundreds or thousands of iterations are
needed for reinforcement learning or prompt optimization (Agrawal et al., 2025).


This inefficiency arises from the inherently sequential nature of API calls. Thus, we ask a simple
question in this paper:


_Must an agent interact with its environment in a strictly sequential manner?_


∗Equal contribution


1


Figure 1: Illustration of our framework in a chess-playing environment. While the Actor issues an
LLM call to decide the next move, the Speculator uses a faster model to guess it. These guesses
enable parallel API calls for the next steps, and once a guess is verified, the system gains time
through parallelization. The process runs in the backend, ensuring a lossless speedup for the user.


Our answer is no. Inspired by speculative execution in microprocessors and speculative decoding
for LLM inference, we propose _speculative_ _actions_ : a general framework that allows agents to
predict and tentatively pursue the most likely next actions using faster models, while slower groundtruth executors (powerful LLMs, external tools, or humans) catch up. In effect, the agent stages
environment interactions (prefetching data, launching safe parallel calls, and preparing reversible
side effects) so that validation, not waiting, is the critical path. When those slower evaluators confirm
the guesses, progress has already been made; when they disagree, we execute as usual. The result is
an _as-if-sequential, lossless interface with parallel, opportunistic internals_ .


Concretely, in such agents, speculative actions introduce two roles in the environment loop:


- _Actor(s)_ : authoritative but slow executors (e.g., SOTA LLMs, external APIs, environment’s own
responses, or humans) whose outputs materialize the ground truth for correctness and side effects.


- _Speculator(s)_ : inexpensive, low-latency models that predict the next environment step, i.e., the
action, its arguments, and the expected observation or state delta. Examples include smaller
LLMs, same LLM with reduced prompts and reasoning steps, and domain heuristics.


A key design goal is losslessness relative to the environment’s baseline semantics: speculative actions should not degrade final outcomes compared to a strictly sequential agent. We achieve this
with (a) semantic guards (actors confirm equivalence of state transitions before commit), (b) safety
envelopes (only idempotent, reversible, or sandboxed speculative side effects), and (c) repair paths
(rollback or compensating actions when a guess is rejected). In many environments (e.g., web
search, pre-checkout shopping carts, and OS-level operations in a sandbox) these patterns are natural and inexpensive to implement.


**Can** **we** **guess** **the** **next** **API** **calls** **of** **agents?** We show that, in practice, API intents can often
be guessed with reasonable accuracy. In particular, we demonstrate speculative actions across four
environments, each highlighting different aspects of agent latency:


- **Turn-based gameplay** (e.g., chess): the Speculator predicts the opponent’s move while waiting
for its turn. See Fig. 1.


- **E-commerce** : while conversing with a shopper, the Speculator proactively infers the shopper’s
intent (e.g., returning an item), and triggers tool calls in advance (e.g., checking return eligibility).


- **Multi-hop** **web** **search** : while awaiting results from slow external calls (e.g., Wikipedia), the
Speculator can guess answers from its knowledge base, and execute subsequent search queries.


- **Operating systems (lossy extension)** : speculative, reversible actions react immediately to workload and environment changes, boosting end-to-end performance while actors confirm.


2


Across these settings, we observe substantially reduced latency, with up to 55% accuracy in predicting the next API calls and 20% end-to-end speedup. These results are achieved with a simple
single-step speculation, and can be improved by advanced techniques such as adaptive speculation.


Finally, we give a cost-latency analysis that formally characterizes the tradeoff between speculating
additional API calls and the resulting time savings. We provide a theoretical baseline for choosing
the speculative breadth, and show that the cost incurred by confidence-based selection grows substantially slower than naively scaling the number of speculative branches. Furthermore, in our OStuning environment where losslessness is not required, cost and latency can actually both decrease.
Our code is publicly available at `[https://github.com/naimengye/speculative-action](https://github.com/naimengye/speculative-action)` .


1.1 Related Work


**Speculative** **decoding** **and** **reasoning** Our work is inspired by the use of speculative decoding
in LLM inference. This technique accelerates autoregressive inference by using a small model
to propose tokens which a larger target model verifies in batches, committing correct tokens and
regenerating failures (Leviathan et al., 2023; Zhang et al., 2024; Chen et al., 2023). At the reasoning
level, speculation has also been used to accelerate chain-of-thought (Wang et al., 2025b;a; Fu et al.,
2025). Our framework adopts the same speculate-verify pattern at the level of API calls.


**Speculative** **planning** **for** **LLM** **agents** More directly related are recent works on speculative
planning for LLM-based agents (Hua et al., 2024; Guan et al., 2025). Hua et al. (2024) introduce
interactive speculative planning, where a fast approximator proposes multi-step lookahead plans that
are verified by a stronger model, with user interruption integrated. Their approach focuses on depthoriented speculation along a single planning branch. Building on this, Guan et al. (2025) propose
an online reinforcement learning method to dynamically determine the number of future steps to
speculate, optimizing a cost-latency tradeoff while maintaining lossless execution.


Our work differs along two dimensions. First, we generalize speculation beyond planning to the
_entire agentic environment_, including LLM calls, internal and external tool APIs, MCP-server interactions, and even human responses. This yields a unified framework for agentic speculation, particularly consistent with the emerging “environment” and MCP perspectives on agentic systems. Second, instead of depth-focused multi-step lookahead, we study a breadth-focused _k_ -branch single-step
strategy, where multiple actions are speculated in parallel at each step. We provide a cost-latency
analysis for this scheme and derive closed-form expressions for expected time and token savings
(Theorem 4). Section 5 compares breadth- and depth-focused strategies under a unified analytical framework. While Guan et al. (2025) optimize depth dynamically, we characterize the optimal
number of speculative branches per step as a function of predicted accuracy (Section 5.2).


**Speculation in systems and architecture** Speculation is prominently used in computer architecture to increase parallelism by executing instructions before their outcomes were resolved (Tomasulo, 1967) and rolling back when predictions were wrong (Lam & Wilson, 1992). In light of security vulnerabilities that exploit microarchitectural speculative execution, (Mambretti et al., 2019)
developed Speculator to analyze CPU speculation behavior.


Similar ideas arise in systems software as thread-level speculation, which parallelizes sequential
code under assumed independence and rolls back upon detecting data dependencies or conflicts (Estebanez et al., 2016). Recently, (Liargkovas et al., 2023) explored the use of tracing and containment
to speculatively but safely run shell scripts out of order. Beyond traditional systems context, speculative techniques have also been used to parallelize otherwise sequential security checks (Nightingale
et al., 2008),test configuration changes in isolation (Su et al., 2007), and accelerate policy simulation
in supply chain optimization (Farias et al., 2024).


2 Framework


An agentic system is modeled as a Markov Decision Process (MDP) ( _st_, _at_ ), where _st_ denotes the
state and _at_ the agent’s action at step _t_ . This model admits considerable flexibility: an action may
represent a chatbot response, a tool call, or a button clicked by a computer-use agent, among others.


3


From a systems perspective, we model each action in an agentic system as an _API call_, which may
block execution until a response is returned. This abstraction offers two key advantages: (1) it
precisely defines what constitutes an action, and (2) it provides a unified framework for optimizing
system latency, as we will see shortly. Notably, this perspective aligns with the recent development
of MCP servers for agentic systems (Anthropic, 2024).


Formally, at each step _t_, the policy π maps the current state _st_ to an API call:
( _ht_, _qt_ ) ← π( _st_ ),
where _ht_ specifies the target API to invoke and _qt_ its associated parameters. We write
_a_ ¯ _t_              - _ht_ ( _qt_ ) _at_ ← await(¯ _at_ )
to denote an asynchronous API invocation that returns a _future_ (a pending action), and the await
for when the response actually arrives. We use the bar notation (e.g., _a_ ¯) for futures and a cache
C : ( _h_, _q_ ) �→ _a_ ¯ that maps an API call specifier to its pending response. The left squiggly arrow
indicates an asynchronous call with non-negligible delay.


The system subsequently transitions to the next state via a transition function _f_ : _st_ +1 ← _f_ ( _st_, _at_ ). As
a concrete example, consider chess: the policy π determines how to construct the prompt based on
the current board state, _at_ corresponds to the move proposed by the LLM’s response, and _f_ updates
the board configuration accordingly. Note that the LLM call is the API, its _response_ is the move _at_ .


This formulation subsumes a broad range of realizations:


- **LLM calls:** each invocation of an LLM within the agent can be treated as an action.

- **Tool** / **MCP server calls:** each actual call for internal/external tools is treated as an action: e.g.,
terminal access, web search, deep research APIs, weather APIs, or browser-use MCPs.

- **Human-as-an-API** **calls:** furthermore, human responses themselves can be abstracted as API
calls, often incurring even longer latencies than automated tools.


Given this abstraction, the fundamental bottleneck in executing agentic systems becomes apparent:
each API call must complete before the next can be issued. To break this sequential dependency,
we propose to **speculate a set of API responses** { _a_ ˆ _t_ } using a faster model while waiting for the true
response _at_ . This enables speculative API calls for step _t_ + 1 to be launched in parallel. At time _t_, if
the API call ( _ht_, _qt_ ) can be found in the cache (cache hit), the system can skip the actual invocation
and only wait for the pending action corresponding to this call to return (if not already returned).
Formally, the algorithm is specified in Algorithm 1.


The resulting speedup relies on two key assumptions:
**Assumption 1** (Speculation accuracy) **.** _The speculative model_ _g guesses the current-step response_ ˆ
_at_ _accurately enough that the implied next call_ ( _ht_ +1, _qt_ +1) = π( _f_ ( _st_, ˆ _at_ )) _matches the true next call_
_with probability_ _p_ - 0 _._


As shown later, this often holds in practice because API responses are typically predictable.
**Assumption** **2** (Concurrent, reversible pre-launch) **.** _Multiple_ _API_ _calls_ _can_ _be_ _launched_ _concur-_
_rently,_ _and_ _pre-launched_ _calls_ _that_ _do_ _not_ _correspond_ _to_ _the_ _realized_ _trajectory_ _have_ _no_ _externally_
_visible side e_ ff _ects (or can be rolled back)._


In practice, this assumption is satisfied under modest traffic for many external APIs (e.g., web search,
OpenAI LLM queries, email lookups). For self-hosted LLMs, concurrent calls also incur only
minimal additional cost due to continuous batching.


We can then establish the following result (with proof deferred to the Appendix A).
**Proposition** **1.** _Under_ _Assumptions_ _1–2,_ _suppose_ _at_ _each_ _step_ _the_ _speculative_ _branch_ _implies_ _the_
correct next call ( _ht_ +1, _qt_ +1) _with probability_ _p, independently across t_ ∈ [1, _T_ - 1] _._ _Let the latency of_
_g be_ ˆ Exp(α) _and the latency of the actual API call be_ Exp(β) _with_ β < α _._ _All latencies and guesses_
_occur independently. Assume the transition_ _f_ _and API parameter construction_ π _are negligible. Then_
_the ratio between the expected runtime of Algorithm 1, denoted_ E[ _T_ s] _, and the expected runtime of_
_sequential execution,_ E[ _T_ seq] _, is_


               -               
_EE_ [[ _TT_ seqs]] [=] [1][ −] _T_ [1] α +α β ( _T_ 1 +− _p_ 1)( _pk_ () _k_ ) + (1 + _p p_ ( _k_ () _k_ [2] )) [2] [−] (1 + _p p_ ( _k_ () _k_ [2] )) [2] [(][−] _[p]_ [(] _[k]_ [))] _[T]_ [−][1] −−−−−−→ _T_ →∞ 1 − 1 + _p p_ ( _k_ () _k_ ) [·] α +α β

_where_ _p_ ( _k_ ) = 1 − (1 − _p_ ) _[k]_ _denotes the probability of at least one of the k speculations hit._


[1] α

_T_ α + β


- ( _T_ - 1) _p_ ( _k_ ) _p_ ( _k_ ) [2]
+
1 + _p_ ( _k_ ) (1 + _p_ (


_p_ ( _k_ ) [2] _p_ ( _k_ ) [2]

[−]
(1 + _p_ ( _k_ )) [2] (1 + _p_ (


_E_ [ _T_ s] [1][ −] [1]
_E_ [ _T_ seq] [=] _T_


4


**Algorithm 1** Speculative actions with _k_ -way parallel next calls

**Require:** Initial state _s_ 0, horizon _T_, transition _f_, policy π, predictor _g_ ˆ, cache C. We use _a_ ¯ to denote
pending action.
1: **for** _t_ = 0 to _T_ - 1 **do**
2: **Policy:** ( _ht_, _qt_ ) ← π( _st_ )
3: **if** ( _ht_, _qt_ ) ∈C **then** - Cache hit
4: _a_ ¯ _t_ ←C[( _ht_, _qt_ )]
5: _at_ ← await(¯ _at_ ) - Await pending action if not returned already
6: _st_ +1 ← _f_ [�] _st_, _at_ 
7: **continue**
8: **end if**
9: **Actor:** Issue real request (returns future): _a_ ¯ _t_ - _ht_ ( _qt_ )
10: **Speculator:** { _a_ ˆ [(] _t_ _[i]_ [)][}] _[k]_ _i_ =1 [←] [await(ˆ] _[g]_ [(] _[s][t]_ [,][ (] _[h][t]_ [,] _[ q][t]_ [)))] - Actor and speculator run in parallel
11: **for** _i_ = 1 to _k_ **do** - One-step speculative rollout per guess
12: _s_ ˆ [(] _t_ + _[i]_ [)] 1 [←] _[f]_ [�] _[s][t]_ [,][ ˆ] _[a]_ _t_ [(] _[i]_ [)] 


13: ( _h_ [ˆ] [(] _t_ + _[i]_ [)] 1 [,][ ˆ] _[q]_ _t_ [(] + _[i]_ [)] 1 [)][ ←] [π][(ˆ] _[s]_ _t_ [(] + _[i]_ [)] 1 [)]


14: **Pre-launch** : _a_ ˆ [¯] [(] _t_ + _[i]_ [)] 1 [�] _[h]_ [ˆ] [(] _t_ + _[i]_ [)] 1� _q_ ˆ( _t_ + _i_ )1� - Return pending action, hence non-blocking

15: C[( _h_ [ˆ] [(] _t_ + _[i]_ [)] 1 [,][ ˆ] _[q]_ _t_ [(] + _[i]_ [)] 1 [)]][ ←] _[a]_ [¯ˆ] _t_ [(] + _[i]_ [)] 1 - Cache speculative pending actions
16: **end for**
17: **Wait for resolved** _at_ **from Actor:** _at_ ← await(¯ _at_ )
18: _st_ +1 ← _f_ [�] _st_, _at_ 
19: **end for**


Proposition 1 suggests the end-to-end latency reduction has an upper bound of 50%, occurring when
_p_ = 1 and α = ∞.This can be further improved by the multi-step extension below.


**Extension** Algorithm 1 is only a simple demonstration of the idea. For example, one can naturally
extend Algorithm 1 to _multi-step speculation_, where the Speculator predicts not only the next, but
_s_ steps ahead. This yields a tree structure with deeper rollouts. This can be further combined with
_adaptive speculation_ : instead of generating _k_ guesses for _at_ uniformly, the Speculator also estimates
confidence for each guess (e.g., via prompting LLMs or uncertainty-quantification methods), this is
explored in Section5. The most promising branches can then be expanded in a beam-search–like
manner. Together, these ideas highlight the richness of speculative actions. Despite Algorithm 1’s
simplicity, the results from the four use cases in the following sections are already highly promising.


**Side e** ff **ects and safety** Speculation executes a hypothesized next action _a_ ˆ _t_ +1 that may be wrong,
so safety requires the ability to simulate first and then commit or roll back. In domains like chess,
rollback is trivial; in others, overwrite is easy (e.g., OS tuning). But many systems involve irreversible or externally visible effects (e.g., deleting records, placing orders), where naive speculation
is harmful. Thus, speculation must be limited to cases where mispredictions are reversible, via
forking, snapshot restoration, or roll-forward repair (e.g., refund/replace).


3 Environments


We now instantiate speculative actions in three environments—chess, e-commerce dialogue, and
multi-hop web search—chosen to stress distinct latency bottlenecks (reasoning, tool/API round trips,
and information retrieval). We pair a fast Speculator with a slow Actor and implement Algorithm 1.


3.1 Chess Environment


We demonstrate the effectiveness of our framework in the context of multi-agent gameplay, focusing
on chess as a canonical turn-based example. In standard play, analysis is strictly sequential: each
player begins analysis only _after_ the opponent has completed their turn. This serialization introduces
substantial idle time. Particularly when both players rely on computationally intensive reasoning
models, a single game can stretch to hours of wall-clock time. Our framework relaxes this constraint


5


through speculative parallel analysis, allowing players to anticipate and prepare for likely opponent
moves in advance. We show that this results in significant reductions in overall game duration.


3.1.1 Implementation


We implement our framework on top of TextArena (Guertler et al., 2025), which provides a standardized gameplay interface for LLM-driven agents.


**Speculative** **pipeline** At turn _t_, the game state _st_ corresponds to the current board position. The
in-turn player issues an API call _ht_ with parameter _qt_ constructed from _st_ together with a reasoningeliciting prompt. At this point, player _P_ is to move, and player _Q_ awaits. Proceeds as follows:


- Current in-turn player _P_ : the player receives _st_, makes an API call _ht_ to the agent with parameter
_qt_ = ( _st_, prompt). This API call returns the next move _at_ = _h_ ( _qt_ ), typically with high latency due
to deep and extensive reasoning.


- Other out-of-turn player _Q_ :


1. **Prediction phase** The Speculator also receives the board state _st_ and issues an API call _h_ [ˆ] _t_,
using a prompt optimized for speed rather than depth. It returns the top- _k_ move predictions
_a_ ˆ [(1)] _t_ [,][ ˆ] _[a]_ _t_ [(2)][, . . .,][ ˆ] _[a]_ _t_ [(] _[k]_ [)][, ordered by confidence.]

2. **Parallel** **computation** For each predicted move _a_ ˆ [(] _t_ _[i]_ [)][,] [the] [out-of-turn] [player] _[Q]_ [immediately]
launches a process analyzing a next move _a_ ˆ [(] _t_ + _[i]_ [)] 1 [=] _[ h][t]_ [+][1][(ˆ] _[s][t]_ [+][1][,] _[ prompt]_ [) for] _[ i]_ [ ∈{][1][, . . .,] _[ k]_ [}][, where]
_s_ ˆ [(] _t_ + _[i]_ [)] 1 [=] _[f]_ [(] _[s][t]_ [,][ ˆ] _[a]_ [(] _t_ _[i]_ [)][) denotes the next state resulting from applying the predicted action] _[a]_ [ˆ] [(] _t_ _[i]_ [)] [to] _[ s][t]_ [.]
3. **Validation** When the current in-turn player _P_ finishes reasoning and returns its move _at_, we
immediately check whether it matches any of the predicted moves _a_ ˆ [(1)] _t_ [,][ ˆ] _[a]_ _t_ [(2)][, . . .,][ ˆ] _[a]_ _t_ [(] _[k]_ [)][.]
4. **Commit** **or** **restart** If a match exists, we commit to the corresponding speculative branch,
advancing directly to _st_ +1 = _f_ ( _st_, _at_ ). The game thus skips ahead, terminating other threads.
If no match exists, we discard all speculative branches and continue with _Q_ ’s regular move
computation _at_ +1 = _ht_ +1( _f_ ( _st_, _at_ ), prompt).


This pipeline is _lossless_ : the final trajectory remains identical to non-speculative play, but time is
saved through parallelized reasoning.


**Agent Configuration** We find that using the same model for both Speculator and Actor, but with
different prompts, maximizes prediction accuracy while keeping speculation fast. Accordingly, in
our experiments, the Actor is instantiated with GPT-5 with high reasoning effort; and the Speculator
is instantiated with GPT-5 configured with low reasoning effort and a specialized system prompt
designed for rapid move prediction rather than exhaustive analysis.


3.1.2 Results


We evaluate our framework in terms of both
time saved and prediction accuracy. We track
two metrics: (i) prediction accuracy: the fraction of rounds in which any speculative prediction matches the actual move; and (ii) time
saved: ( _T_ _seq_ - _T_ _s_ )/ _T_ _seq_, where _T_ _s_ and _T_ _seq_
denote speculative and sequential execution
times, respectively.


**More** **predictions** **improve** **time** **savings** **and**
**accuracy.** Figure 2 reports results over
30 steps. Our framework consistently reduces
execution time, with larger savings as the number of speculative predictions increases. Across
5 runs, using 3 predictions yields an average
time saving of 19.5% with an average prediction accuracy of 54.7%.


|Speculative Accuracy (%)|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|54.7%<br> <br>Time Saved (%)|54.7%<br> <br>Time Saved (%)|54.7%<br> <br>Time Saved (%)|54.7%<br> <br>Time Saved (%)|54.7%<br> <br>Time Saved (%)|
|~~41.3%~~|~~41.3%~~|~~41.3%~~|||
|31.3%<br>|31.3%<br>|31.3%<br>|31.3%<br>|31.3%<br>|
||~~19.5%~~|~~19.5%~~|~~19.5%~~|~~19.5%~~|
|15.0%<br>|15.0%<br>|15.0%<br>|15.0%<br>||
|~~11.8%~~<br>|~~11.8%~~<br>||||
|<br> <br>|<br> <br>|<br> <br>|<br> <br>|<br> <br>|


Figure 2: Percentage of time saved and percentage
of correct predictions across 5 runs at 30 steps.


6


70


60


50


40


30


20


10


0


**Randomness of agent call in gameplay.** The variance in Figure 2 reflects realistic latency fluctuations from live API calls. Even with correct predictions, speedups vary: if the resulting position
is trivial, little acceleration is realized; large gains occur only when predictions lead to positions
requiring deep analysis. In addition, API latency itself is inherently stochastic. Backend load fluctuations (e.g., concurrent traffic to the model provider) can cause the same API call with different
latency across runs. Consequently, measured latency reductions exhibit natural variability and are
not perfectly reproducible.

3.2 E-Commerce Environment


Beyond competitive gameplay, customer-agent interactions in e-commerce provide a real-world setting where latency significantly impacts user experience. In a typical workflow, the customer submits a query through a chat interface and waits while the agent sequentially invokes multiple API
calls—for example, processing a return may involve retrieving order information, validating eligibility for each item, and initiating the return. These chained calls can introduce substantial delay.
By contrast, if some API calls are correctly speculated and executed in advance, the agent can return
results immediately once the query arrives, making the interaction feel seamless. We evaluate this
setting using the retail environment from τ _-bench_ (Yao et al., 2024).


3.2.1 Experimental Setup


**Speculative** **pipeline** In this scenario, the current
state _st_ is defined as the conversation history up to
turn _t_, and _ht_ are the API calls required to answer the
user’s query (eg. get ~~u~~ ser ~~d~~ etails, get ~~o~~ rder ~~d~~ etails).
Our Speculator will predict


1. The user’s query _a_ ˆ _t_ ;

2. The target API calls and their corresponding parameters ( _h_ [ˆ] [(] _t_ + _[i]_ [)] 1 [,][ ˆ] _[q]_ _t_ [(] + _[i]_ [)] 1 [)] [for] _[i]_ [∈{][1][, ...,] _[ k]_ [}][,]
conditioned on the current state _st_ and the
predicted user’s query from step 1. Since
the number of API calls for each turn is not
fixed, the Speculator must also predict _k_ .


Figure 3: APIs prediction accuracy across
various Speculator models.


**Agent** **configuration** We evaluate multiple Speculator models, including OpenAI GPT variants
(gpt-5-nano, gpt-5-mini, gpt-5) and Google Gemini (gemini-2.5-flash) under different reasoning
budgets (1024/2048/4096 tokens). Motivated by prior work where heterogeneous LLM ensembles
outperform single models (Jiang et al., 2023; Chen et al., 2025), we consider two configurations:
(i) a _single-model_ Speculator, and (ii) a _multi-model_ Speculator, where comparable models run in
parallel (e.g., gpt-5-nano with low-budget Gemini, gpt-5-mini with medium-budget Gemini). Their
outputs are aggregated into a shared pool of candidate speculative actions.


At runtime, once the user simulator reveals the ground-truth utterance, the Actor validates the speculative API calls: correct predictions are committed immediately (eliminating latency), while incorrect ones are discarded without affecting correctness.


**Evaluation** We evaluate performance using **APIs prediction accuracy**, defined as the fraction of
speculative API calls that match the ground-truth APIs required to resolve the user’s query. This
metric directly reflects the proportion of turns in which the user receives an immediate response,
without waiting for API execution: higher prediction accuracy translates into greater time savings.


3.2.2 Results


Figure 3 shows that between 22% and 38% of API calls are correctly predicted by the Speculator.
Accuracy improves with stronger models and the multi-agent configuration consistently outperforms
single-model speculation. Importantly, low-budget models speculate in only 2–3 seconds (per the
LLM API providers leaderboard [1] ), well below the average user typing time of about 30 seconds
(assuming 40 words per minute). This means that in roughly one third of turns, the agent can
respond faster than sequential execution, without waiting for API execution.


1 `[https://artificialanalysis.ai/leaderboards/providers](https://artificialanalysis.ai/leaderboards/providers)`


7


3.3 HotpotQA Environment


We further evaluate our framework on HotpotQA,
a setting where the main performance bottleneck
arises from information retrieval latency. In this example, the agent must answer multi-hop questions
through sequential Wikipedia API calls (Yang et al.,
2018), mirroring real-world agentic workflows with
high round-trip network latency. In this setting, the
Speculator predicts likely Wikipedia content while
the actual API call executes. Parallelism allows the
agent to continue reasoning on provisional information rather than blocking on API latency. See Appendix B.2 for details about our experimental setup.


Figure 4: Accuracy with gemini-2.5-flash as

We evaluate on the accuracy of the predicted API

the Actor. Speculating multiple actions ( _k_ =

calls. As shown in Figure 4, the Speculator success
3) yields higher accuracy than predicting a

fully predicts ground truth API call up to 46% of the

single action.

time with top-3 prediction. This accuracy improves
significantly from top-1 to top-3 predictions, yielding substantial accuracy gains with modest speculation width increase. Our speculation provides
value by precomputing reasoning paths during otherwise idle API waiting time.


4 Beyond Lossless Speculation: OS Hyperparameter Tuning Environment


Thus far, our experiments have focused on _lossless_ speculation, where speculative actions are validated sequentially before commitment. We now turn to a _lossy_ setting that relaxes this constraint. In
latency-sensitive environments like an operating system, waiting for a powerful but slow Actor (1015s deliberation) can leave the system in a degraded state. Instead, we use a fast Speculator to apply
immediate provisional adjustments while the Actor deliberates. This is made safe by a last-writewins mechanism—the Actor’s final decision simply overwrites any speculative action, removing
the need for complex rollbacks. This method accelerates convergence and improves reaction time,
which we evaluate on the `sysbench` `cpu` benchmark, a CPU-bound workload (Kopytov, 2020).


4.1 Experimental Setup


We tune Linux’s Completely Fair Scheduler (CFS) parameter `min` ~~`g`~~ `ranularity`, which controls a
task’s minimum timeslice. This knob strongly affects scheduling performance: smaller timeslices
reduce latency but can degrade throughput, yielding a classic trade-off. Building on (Liargkovas
et al., 2025), we augment the prior LLM-based tuning setup with a speculative control loop.


The Speculator proposes a parameter update each second using the latest performance metric. The
Actor, in contrast, responds every 10–15 seconds after analyzing a compressed chronology of the
Speculator’s recent (measurement, action) pairs. Upon arrival, the Actor’s decision is applied immediately and its state resets the Speculator’s context, preventing drift from the validated narrative.


**Evaluation** We evaluate three systems: (1) **Actor-only** : slow but deliberative (10-15 s interval);
(2) **Speculator-only** : fast (1 s interval) but non-extensive; (3) **Speculator–Actor** : combined system
using speculative updates between Actor decisions.


4.2 Results


**Speculator** **mitigates** **poor-reaction** **slowdowns.** As shown in Figure 5 (right), the Speculator
significantly improves reaction time. During recovery, the full Speculator–Actor system maintains
an average p95 latency of 37.93 ms, compared to 54.00 ms for Actor-only, which remains longer in
degraded states (initially 102.97 ms). Fast speculative updates provide immediate mitigation while
the Actor deliberates (details in §B.3.3).


**Speculator** **accelerates** **convergence** **to** **optimum.** Figure 5 (left) shows that the joint system
reaches the optimal setting (0.2 ms `min` ~~`g`~~ `ranularity` ) in 10-15 s, whereas Actor-only requires
∼200 s and remains trapped in highly suboptimal regions (e.g., latency > 120ms) for extended periods. Rapid speculative exploration helps the Actor avoid pathological configurations.


8


**p95 (ms)**


10


1


0.1


|Col1|Col2|Act<br>Sp|Col4|Col5|or-only<br>eculator-only|Optima<br>OS Def|l Region<br>ault|
|---|---|---|---|---|---|---|---|
|||Act|Act|Act|or + Speculator|||
|||||||||
|||||||||
|||||||||
|||||||||


0 50 100 150 200
Time (s)


Untuned 102.97
Actor-only 54.00
Actor + Spec. 37.93


Figure 5: **(Left)** Comparison of **Speculator-Actor**, **Speculator-only**, and **Actor-only** convergence.
The Speculator shortens time spent exploring poor settings. The Speculator-only agent stabilizes
quickly but at a worse final value. **(Right)** Average p95 latency over a 20-second tuning experiment
showing that rapid reaction offers immediate performance benefits (see §B.3.3). Lower is better.


**Speculator-only** **reacts** **quickly** **but** **is** **suboptimal.** While Speculator-only stabilizes rapidly,
it converges to a worse configuration (0.55 ms; 36.24 ms latency) than the joint system (0.2 ms;
30.26 ms). Without the Actor’s deeper reasoning, it cannot escape local minima.


**Cost** **and** **latency** **both** **decrease.** Despite additional speculative calls, total cost is lower due
to faster convergence. As shown in Table 3, Actor-only converges at ∼200 s with a total cost of
2.18 cents, whereas Speculator–Actor converges in ∼13 s with only 0.17 cents.


Overall, the joint system combines fast adaptation with strategic guidance, achieving both responsiveness and optimal steady-state performance.


5 Cost–Latency Tradeoff


Performing more speculative API calls improves accuracy but also raises costs when pricing is based
on the number of calls or tokens. In this section, assume a fixed token per unit time and fixed per
token cost, and analyze the cost-latency tradeoff. Full details can be found in Appendix C.


5.1 Breadth-focused Speculation (Algorithm 1)


In addition to Proposition 1, we obtain a closed-form expression for relative cost increase ratio


E[ _M_ spec   - _M_ seq]   - α
lim ≤ _k_ - _k_ +
_T_ →∞ E[ _M_ seq] α + β


- _p_ ( _k_ )
1 + _p_ ( _k_ ) [.]


See Theorem 4 in Appendix C for the formal theorem. Comparing Proposition 1 with the above
expression, we see that both ratios are governed by _p_ ( _k_ ). Thus, given an estimation of _p_ ( _k_ ), a user
can directly tune _k_ offline, trading off cost against latency. Our experiment (Figure 6) shows this
non-linear dependence on _k_, and additional empirical results can be found in Appendix C.2.


5.2 Dynamic selective speculation.


So far we assume a fixed branch accuracy _p_ . In practice, we sometimes are able to obtain perspeculation confidence estimates (e.g., from intrinsic model logits, or from a separately-trained auxiliary predictor), allowing confidence-aware selective speculation. At each speculation window, the
accuracies of the _k_ speculative branches are random and drawn from a known distribution (which
may vary over time). Before acting, the realized accuracy vector **p** = ( _p_ [1], . . ., _p_ _[k]_ ) is observed, and
we choose how many of the top branches to launch.


We model the cost-latency tradeoff via the weighted objective


_T_


cost

_t_ =1


max _r_ 


_T_


latency − _c_   
_t_ =1


where _r_ and _c_ encode the relative importance of latency and cost. For simplicity, let _a_ and _b_ denote
the fixed latency of the actor and speculator, and define the latency gain ℓ = _r_ ( _a_ - _b_ ) > 0.


9


If the top _m_ branches are launched, the probability
of a cached step is _q_ ( _m_ ; **p** ) = 1 − [�] _[m]_ _j_ =1 [(1][ −] _[p]_ [(] _[ j]_ [)][)][,]
where _p_ [(1)] ≥· · · ≥ _p_ [(] _[k]_ [)] are the sorted confidences.

**Theorem** **3** (Confidence-aware selective speculation) **.** _There exist scalars_ ∆ _t such that at each spec-_
_ulation window t, the optimal breadth satisfies_

_m_ [⋆] _t_ [(] **[p]** [)][ ∈] [arg] _m_ ∈{ [max] 0,..., _k_ } [{] _[q]_ [(] _[m]_ [;] **[ p]** [)][ ∆] _[t]_ [ −] _[cm]_ [}][.]


_The continuation values are given by the backward_
_recursion_


                -                 ∆ _T_ = 0, ∆1:: _T_ −1 = ℓ−E max
_m_ [{] _[q]_ [(] _[m]_ [;] **[ P]** [)][ ∆] _[t]_ [+][1][−] _[cm]_ [}]


180


160


140


120


100


80


|3|speculation|s|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||2|speculations||
||||||||
||||||||
||Target Steps<br>20 steps<br>30 steps<br>|Target Steps<br>20 steps<br>30 steps<br>|Target Steps<br>20 steps<br>30 steps<br>|Target Steps<br>20 steps<br>30 steps<br>|Target Steps<br>20 steps<br>30 steps<br>|Target Steps<br>20 steps<br>30 steps<br>|
||~~40 steps~~<br>50 steps|~~confidence-~~<br>|~~based~~<br>||||
||||||||
|||threshold|policy||1 speculat|ion|


70 75Percentage of Time Spent (%)80 85 90 95


_In_ _the_ _stationary_ _case_ _(time-homogeneous_ _accu-_
_racy_ _distribution),_ _the_ _continuation_ _values_ ∆ _t_ _col-_ Figure 6: Cost-latency tradeoff across different
_lapse_ _to_ _a_ _single_ _constant_ ∆ [⋆] _._ _Thus_ _branches_ speculation widths, forming a Pareto curve.
_are added greedily in descending confidence order_
_while_ ∆ [⋆] - δ _q_ ( _m_ ; **p** ) ≥ _c, where_ δ _q_ ( _m_ ; **p** ) _is the marginal gain from adding one more branch._


**Interpretation** **and** **implementation** The key implication is structural: dynamic selection collapses to a one-dimensional trade-off. Additional branches are launched only when their incremental
hit probability, scaled by a single continuation value (∆ _t_ or ∆ [⋆] ), exceeds the marginal cost _c_ . Under
stationarity, this reduces to estimating ∆ [⋆] offline; at runtime, the system simply sorts confidences
and adds branches greedily, requiring _O_ ( _k_ ) computation per step.
**Empirical results** We implement a simple constant-threshold approximation of the stationary rule
in the chess environment. At each step, after generating speculative branches, we use a predictor to
estimate the correctness probability of each branch and continue only with those whose predicted
accuracy exceeds 50%. This implements a simplified threshold rule consistent with the structure
suggested by Theorem 3. Our method achieves the _lowest_ _additional_ _token_ _cost_ while providing
_greater latency reduction_ than naively launching 1 or 2 speculations per step.


5.3 Depth-focused speculation.


The previous two strategies are breadth-focused: each speculation is immediately followed by a real
API call (speculative depth 1). Additionally, we analyze the opposite extreme: a _depth-focused_ policy, in which multi-step speculations are continuously spawned. Somewhat counterintuitively, this
strategy does _not_ lead to exponential branch growth. Speculative calls are only extended when either
a speculative or real call returns, and inconsistent subtrees are immediately pruned. Consequently,
the system can run at most _a_ / _b_ speculative steps ahead (governed by the relative speeds of real vs.
speculative calls), ensuring that the number of active branches remains bounded and _does not scale_
_with the horizon T_ . Under this policy, we can show that (formal theorem in Appendix 6)
E[ _T_ seq   - _T_ spec] _[T]_ [−] [1]   - _[b]_   - E[ _M_ spec   - _M_ seq] _[T]_ [−] [1]   -   - _a_ [1]   -   


_T_


seq - _T_ spec] = _[T]_ [−] [1]

E[ _T_ seq] _T_


- - _a_
(1 − _p_ ) 2 _b_ [−] 2 [1]


[−] [1] _p_ �1 − _[b]_

_T_ _a_


- + _p_


_a_


�, E[ _M_ spec - _M_ seq]


spec  - _M_ seq] [−] [1]

≈ _[T]_
E[ _M_ seq] _T_


2


Compared to breadth speculation, depth speculation improves the latency coefficient from 1+ _pp_ [to] _[p]_ [,]
increasing the theoretical speedup ceiling from [1] 2 [to] [1.] [The] [cost] [term] [scales] [with] [(1][ −] _[p]_ [)(] 2 _[a]_ _b_ [−] [1] 2 [),]

which captures how many speculative steps can accumulate before the real response arrives.


_[a]_

2 _b_ [−] [1] 2


[1] 2 [to] [1.] [The] [cost] [term] [scales] [with] [(1][ −] _[p]_ [)(] 2 _[a]_


6 Conclusion


In this paper we propose **Speculative Actions**, a lossless framework for accelerating general agentic environments by breaking the strict sequentiality of their interaction loops. Our approach treats
every step, whether an LLM call, tool invocation, MCP request, or human response, as an API call
subject to prediction and parallelization. By pairing a fast _Speculator_ with a slow but authoritative _Actor_, the framework enables agents to anticipate and prepare likely next actions in parallel,
transforming otherwise idle waiting time into productive computation. We instantiate the framework across four representative environments and observe consistent substantial latency reduction.
Finally, we provide a cost-latency analysis that addresses the tradeoff between the additional cost
and the latency gains from launching additional speculative actions.


10


Acknowledgments


This work is supported by Columbia-Dream Sports AI Innovation Center


References


Reyna Abhyankar, Qi Qi, and Yiying Zhang. Osworld-human: Benchmarking the efficiency of
computer-use agents. _arXiv preprint arXiv:2506.16042_, 2025.


Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong,
Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, et al. Gepa: Reflective prompt
evolution can outperform reinforcement learning. _arXiv preprint arXiv:2507.19457_, 2025.


Anthropic. Introducing the model context protocol. `[https://www.anthropic.com/news/](https://www.anthropic.com/news/model-context-protocol)`
`[model-context-protocol](https://www.anthropic.com/news/model-context-protocol)`, November 2024. Accessed: 2025-09-24.


Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John
Jumper. Accelerating large language model decoding with speculative sampling, 2023. URL
`[https://arxiv.org/abs/2302.01318](https://arxiv.org/abs/2302.01318)` .


Zhijun Chen, Jingzheng Li, Pengpeng Chen, Zhuoran Li, Kai Sun, Yuankai Luo, Qianren Mao,
Ming Li, Likang Xiao, Dingqi Yang, Yikun Ban, Hailong Sun, and Philip S. Yu. Harnessing
multiple large language models: A survey on llm ensemble, 2025. URL `[https://arxiv.org/](https://arxiv.org/abs/2502.18036)`
`[abs/2502.18036](https://arxiv.org/abs/2502.18036)` .


Dmitry Duplyakin, Robert Ricci, Aleksander Maricq, Gary Wong, Jonathon Duerig, Eric Eide,
Leigh Stoller, Mike Hibler, David Johnson, Kirk Webb, Aditya Akella, Kuangching Wang, Glenn
Ricart, Larry Landweber, Chip Elliott, Michael Zink, Emmanuel Cecchet, Snigdhaswin Kar, and
Prabodh Mishra. The design and operation of CloudLab. In _Proceedings_ _of_ _the_ _USENIX_ _An-_
_nual Technical Conference (ATC)_, pp. 1–14, July 2019. URL `[https://www.flux.utah.edu/](https://www.flux.utah.edu/paper/duplyakin-atc19)`
`[paper/duplyakin-atc19](https://www.flux.utah.edu/paper/duplyakin-atc19)` .


Alvaro Estebanez, Diego R. Llanos, and Arturo Gonzalez-Escribano. A survey on thread-level
speculation techniques. _ACM Comput. Surv._, 49(2), June 2016. ISSN 0360-0300. doi: 10.1145/
2938369. URL `[https://doi.org/10.1145/2938369](https://doi.org/10.1145/2938369)` .


Vivek Farias, Joren Gijsbrechts, Aryan Khojandi, Tianyi Peng, and Andrew Zheng. Speeding up
policy simulation in supply chain rl. _arXiv preprint arXiv:2406.01939_, 2024.


Yichao Fu, Rui Ge, Zelei Shao, Zhijie Deng, and Hao Zhang. Scaling speculative decoding with
lookahead reasoning. _arXiv preprint arXiv:2506.19830_, 2025.


Yilin Guan, Wenyue Hua, Qingfeng Lan, Sun Fei, Dujian Ding, Devang Acharya, Chi Wang, and
William Yang Wang. Dynamic speculative agent planning, 2025. URL `[https://arxiv.org/](https://arxiv.org/abs/2509.01920)`
`[abs/2509.01920](https://arxiv.org/abs/2509.01920)` .


Leon Guertler, Bobby Cheng, Simon Yu, Bo Liu, Leshem Choshen, and Cheston Tan. Textarena,
2025. URL `[https://arxiv.org/abs/2504.11442](https://arxiv.org/abs/2504.11442)` .


Wenyue Hua, Mengting Wan, Shashank Vadrevu, Ryan Nadel, Yongfeng Zhang, and Chi Wang.
Interactive speculative planning: Enhance agent efficiency through co-design of system and user
interface, 2024. URL `[https://arxiv.org/abs/2410.00079](https://arxiv.org/abs/2410.00079)` .


Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. LLM-blender: Ensembling large language models with pairwise ranking and generative fusion. In Anna Rogers, Jordan Boyd-Graber, and
Naoaki Okazaki (eds.), _Proceedings_ _of_ _the_ _61st_ _Annual_ _Meeting_ _of_ _the_ _Association_ _for_ _Com-_
_putational_ _Linguistics_ _(Volume_ _1:_ _Long_ _Papers)_, pp. 14165–14178, Toronto, Canada, July
2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.792. URL
`[https://aclanthology.org/2023.acl-long.792/](https://aclanthology.org/2023.acl-long.792/)` .


Tengjun Jin, Yuxuan Zhu, and Daniel Kang. Elt-bench: An end-to-end benchmark for evaluating ai
agents on elt pipelines. _arXiv preprint arXiv:2504.04808_, 2025.


11


Kaggle. Game arena. `[https://www.kaggle.com/game-arena](https://www.kaggle.com/game-arena)`, 2025. Accessed: 2025-09-21.


Alexey Kopytov. Sysbench: Scriptable benchmark tool. `[https://github.com/akopytov/](https://github.com/akopytov/sysbench)`
`[sysbench](https://github.com/akopytov/sysbench)`, 2020. Accessed: 2025-09-22.


Monica S Lam and Robert P Wilson. Limits of control flow on parallelism. _ACM SIGARCH Com-_
_puter Architecture News_, 20(2):46–57, 1992.


Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative
decoding. In _International Conference on Machine Learning_, pp. 19274–19286. PMLR, 2023.


Georgios Liargkovas, Konstantinos Kallas, Michael Greenberg, and Nikos Vasilakis. Executing
shell scripts in the wrong order, correctly. In _Proceedings of the 19th Workshop on Hot Topics in_
_Operating Systems_, pp. 103–109, 2023.


Georgios Liargkovas, Vahab Jabrayilov, Hubertus Franke, and Kostis Kaffes. An expert in residence:
Llm agents for always-on operating system tuning. In _Proceedings of the NeurIPS 2025 Workshop_
_on Machine Learning for Systems (MLForSys)_, San Diego, CA, USA, December 2025. NeurIPS.
Accepted paper.


Andrea Mambretti, Matthias Neugschwandtner, Alessandro Sorniotti, Engin Kirda, William Robertson, and Anil Kurmus. Speculator: a tool to analyze speculative execution attacks and mitigations.
In _Proceedings_ _of_ _the_ _35th_ _Annual_ _Computer_ _Security_ _Applications_ _Conference_, pp. 747–761,
2019.


Edmund B Nightingale, Daniel Peek, Peter M Chen, and Jason Flinn. Parallelizing security checks
on commodity hardware. _ACM SIGARCH Computer Architecture News_, 36(1):308–318, 2008.


OpenAI. Introducing deep research. `[https://openai.com/index/](https://openai.com/index/introducing-deep-research/)`
`[introducing-deep-research/](https://openai.com/index/introducing-deep-research/)`, 2025. Accessed: 2025-09-24.


Ya-Yunn Su, Mona Attariyan, and Jason Flinn. Autobash: improving configuration management
with operating system causality analysis. _ACM SIGOPS Operating Systems Review_, 41(6):237–
250, 2007.


Robert M Tomasulo. An efficient algorithm for exploiting multiple arithmetic units. _IBM Journal of_
_research and Development_, 11(1):25–33, 1967.


Jikai Wang, Juntao Li, Jianye Hou, Bowen Yan, Lijun Wu, and Min Zhang. Efficient reasoning
for llms through speculative chain-of-thought, 2025a. URL `[https://arxiv.org/abs/2504.](https://arxiv.org/abs/2504.19095)`
`[19095](https://arxiv.org/abs/2504.19095)` .


Zhihai Wang, Jie Wang, Jilai Pan, Xilin Xia, Huiling Zhen, Mingxuan Yuan, Jianye Hao, and Feng
Wu. Accelerating large language model reasoning via speculative search, 2025b. URL `[https:](https://arxiv.org/abs/2505.02865)`
`[//arxiv.org/abs/2505.02865](https://arxiv.org/abs/2505.02865)` .


Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering, 2018. URL `[https://arxiv.org/abs/1809.09600](https://arxiv.org/abs/1809.09600)` .


Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.
React: Synergizing reasoning and acting in language models, 2023. URL `[https://arxiv.org/](https://arxiv.org/abs/2210.03629)`
`[abs/2210.03629](https://arxiv.org/abs/2210.03629)` .


Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A Benchmark for
Tool-Agent-User Interaction in Real-World Domains, June 2024. URL `[http://arxiv.org/](http://arxiv.org/abs/2406.12045)`
`[abs/2406.12045](http://arxiv.org/abs/2406.12045)` . arXiv:2406.12045.


Chen Zhang, Zhuorui Liu, and Dawei Song. Beyond the speculative game: A survey of speculative
execution in large language models. _arXiv preprint arXiv:2404.14897_, 2024.


12


A Proof of Proposition 1


_Proof._ **Baseline.** In sequential execution, each of the _T_ steps requires one call to the true model _h_
with mean latency 1/β. Therefore

_E_ [ _T_ seq] = _[T]_ β [.]


**Expected** **time** **saved** **per** **hit.** Consider two consecutive steps ( _t_, _t_ +1). In the baseline, the total
completion time is _R_ = _B_ + _C_, where _B_, _C_ ∼ Exp(β) are the latencies of step _t_ and step _t_ +1. With
speculation, we launch _A_ ∼ Exp(α) during step _t_ . If the guess is correct, the ( _t_ +1) call _C_ can be
issued once either _A_ or _B_ finishes, so the block completes at


_S_ = _C_ + min{ _A_, _B_ }.


Thus, when a guess is correct, our expected time saved is


_R_              - _S_ = ( _B_              - _A_ )+,


where ( _x_ )+ = max{ _x_, 0}.


By independence of _A_, _B_,


     - ∞
E[( _B_ - _A_ )+] =

0


- _b_ α

( _b_   - _a_ ) α _e_ [−][α] _[a]_ β _e_ [−][β] _[b]_ _da db_ =
0 β(α + β) [.]


**Expected number of hits** . We denote the expected number of hits by round _n_ as _S n_, that is


E[number of hits by round _n_ ] = _S n_


We then have _S_ 0 = 0, _S_ 1 = _p_ ( _k_ ). In round 1, either (i) we hit with probability _p_ ( _k_ ), in which case
round 2 cannot be a hit (there is no speculation window immediately after a correct speculation),
contributing 1+ _S n_ −2; or (ii) we miss with probability 1− _p_ ( _k_ ), after which round 2 proceeds normally,
contributing _S n_ −1. We then have the following recursion


_S n_ = _p_ ( _k_ )(1 + _S n_ −2) + (1 − _p_ ( _k_ )) _S n_ −1


Solve this linear recurrence by splitting into homogeneous and particular parts.


_(1) Homogeneous part_
_S_ _n_ _[h]_ [=] _[p]_ [(] _[k]_ [)] _[S][ n]_ [−][2] [+][ (1][ −] _[p]_ [(] _[k]_ [))] _[S][ n]_ [−][1]
The characteristic equation is


_r_ [2]     - (1 − _p_ ( _k_ )) _r_     - _p_ ( _k_ ) = 0 = ( _r_     - 1)( _r_ + _p_ ( _k_ )) =⇒ the roots are _r_ 1 = 1, _r_ 2 = − _p_ ( _k_ )


Therefore,
_S_ _n_ [(] _[h]_ [)] = _C_ 1 + _C_ 2(− _p_ ( _k_ )) _[n]_ .


_(2) Particular solution_ The forcing term is constant (+ _p_ ( _k_ )), and _r_ = 1 is a root, so a constant trial
collides with the homogeneous part. Try _S_ _n_ [(] _[p]_ [(] _[k]_ [))] = _an_ and substitute:


_an_ = (1 − _p_ ( _k_ )) _a_ ( _n_      - 1) + _p_ ( _k_ ) _a_ ( _n_      - 2) + _p_ ( _k_ ) = _an_      - _a_ (1 + _p_ ( _k_ )) + _p_ ( _k_ ),


which gives _a_ (1 + _p_ ( _k_ )) = _p_ ( _k_ ) and thus


_p_ ( _k_ )
_S_ _n_ [(] _[p]_ [(] _[k]_ [))] = 1 + _p_ ( _k_ ) _[n]_ [.]


Combine:
_p_ ( _k_ )
_S n_ = _C_ 1 + _C_ 2(− _p_ ( _k_ )) _[n]_ +
1 + _p_ ( _k_ ) _[n]_ [.]

Use _S_ 0 = 0 and _S_ 1 = _p_ ( _k_ ):


_p_ ( _k_ )
0 = _C_ 1 + _C_ 2, _p_ ( _k_ ) = _C_ 1 + _C_ 2(− _p_ ( _k_ )) +
1 + _p_ ( _k_ ) [.]


13


_p_ ( _k_ ) [2]
Solving yields _C_ 2 = −


_p_ ( _k_ ) [2] _p_ ( _k_ ) [2]

(1 + _p_ ( _k_ )) [2] [and] _[ C]_ [1] [=] (1 + _p_ (


(1 + _p_ ( _k_ )) [2] [.]


Hence the closed form solution is


_p_ ( _k_ ) _p_ ( _k_ ) [2]               _S n_ = 1 − (− _p_ ( _k_ )) _[n]_ [�]
1 + _p_ ( _k_ ) _[n]_ [ +] (1 + _p_ ( _k_ )) [2]


**Total saving.** There are _T_ - 1 potential speculation windows, hence


α

_E_ [ _T_ s] = _[T]_

β [−] _[S][ T]_ [−][1] β(α + β)


**Final ratio.** Dividing by _E_ [ _T_ seq] = _T_ /β gives


_p_ ( _k_ ) [2]  
(1 + _p_ ( _k_ )) [2] [(][−] _[p]_ [(] _[k]_ [))] _[T]_ [−][1]


_E_ [ _T_ s] α

[1][ −] [1]
_E_ [ _T_ seq] [=] _T_ α + β


- ( _T_ - 1) _p_ ( _k_ ) _p_ ( _k_ ) [2]
+
1 + _p_ ( _k_ ) (1 + _p_ (


_p_ ( _k_ ) [2] _p_ ( _k_ ) [2]

[−]
(1 + _p_ ( _k_ )) [2] (1 + _p_ (


Taking _T_ →∞, we get exactly that the ratio converges to 1 − 1+ _p_ ( _pk_ () _k_ ) αα+β


14


B Additional Environment Details


B.1 Ecommerce


τ **-bench:** A benchmark designed for dynamic task-oriented dialogues between a user (simulated
by language models) and an API-augmented agent. The benchmark spans two domains — retail and
airline, with structured databases, domain-specific tools. We focus on the retail domain, where the
agent assists users with operations such as canceling or modifying pending orders, initiating returns
or exchanges, or providing product and order information. The benchmark defines 115 tasks with
15 APIs (7 write, 8 read-only).


B.2 HotpotQA


B.2.1 Experimental Setup


We build our framework upon ReAct (Yao et al. (2023)), which interleaves chain-of-thought with
tool use.


**Speculative Pipeline** In this scenario, the state _st_ consists of the entire history of reasoning traces
and retrieved information (API responses). At each step, the Actor takes in the current state _st_,
selects an API call _ht_ ∈{Search(), Lookup(), Finish()} and a corresponding parameter _qt_, e.g.
Search(entity). The call _ht_ ( _qt_ ) returns a response _at_, typically providing information about the
queried entity. Our speculative framework operates as follows:


1. Speculator predicts the API call response _a_ ˆ [(] _t_ _[i]_ [)][,] [yielding] [predicted] [next] [states] _[s]_ [ˆ][(] _t_ + _[i]_ [)] 1 [=] _[f]_ [(] _[s][t]_ [,][ ˆ] _[a]_ [(] _t_ _[i]_ [)][),]
_i_ ∈{1, . . ., _k_ }.


2. Based on the states, the Actor generates reasoning traces and subsequently determines the next
API decision ( _h_ [ˆ] [(] _t_ + _[i]_ [)] 1 [,][ ˆ] _[q]_ _t_ [(] + _[i]_ [)] 1 [) for] _[ i]_ [ ∈{][1][, . . .,] _[ k]_ [}][.]


**Evaluation** We evaluate the effectiveness of the speculative pipeline by the accuracy of the predicted
API call decisions ( _h_ [ˆ] _t_ +1, ˆ _qt_ +1). Specifically, we compare the predicted call against the ground-truth
call ( _ht_ +1, _qt_ +1) obtained under the true response _at_ . We employ a strict match criterion, counting
a prediction as correct only when _h_ [ˆ] _t_ +1 = _ht_ +1 and _q_ ˆ _t_ +1 = _qt_ +1. This stringent criterion captures
whether speculation enables meaningful progress, as even minor parameter differences (synonyms,
word order) count as mismatches.


**Agent** **configuration** We evaluate speculative accuracy across three Speculator models: GPT-5nano, GPT-4.1-nano and Gemini-2.5-flash. For each model, we measure the top-k prediction accuracy, with _k_ ∈{1, 3}.


B.2.2 Results


Figure 4 shows that our Speculator successfully predicts the ground truth API call up to 46% of
the time with top-3 prediction, despite our strict matching criterion. This accuracy improves significantly from top-1 to top-3 predictions, demonstrating that modest increases in speculation width
yield substantial accuracy gains. Our speculation provides value by precomputing reasoning paths
during otherwise idle API waiting time.


**Model Patterns** We observe high variation in API decision across different Speculators. These are
largely driven by phrasing discrepancies – some models phrase the calls concisely while some overspecify. Interestingly, stronger models often yield lower accuracy, as their more diverse and contextspecific queries (e.g., “List of Nobel laureates in physics 1970s” vs. “1970s Nobel Prize Physics
winners list”) are penalized under strict matching. In contrast, weaker models tend to produce
simpler, more predictable outputs.


15


B.3 Operating System Tuning


B.3.1 Experimental Setup and Implementation Details


**System and Workload Configuration** All experiments were conducted on a dedicated machine
with 2× Intel Xeon Silver 4114 10-core CPUs at 2.20 GHz, 192 GB DDR4 RAM, and a 1 TB NVMe
SSD running Ubuntu 22.04 with Linux Kernel 5.15, hosted on Cloudlab (Duplyakin et al., 2019).


We run `sysbench` `cpu` (Kopytov, 2020), a CPU-bound benchmark that repeatedly calculates a large
prime number sequence. The benchmark reports several performance metrics every second. We run
sysbench on 16 concurrent threads pinned on two CPU cores.


**Tuner Implementation Details** The system consists of two agents, a fast Speculator and a slow
Actor, which collaborate to minimize the p95 latency of the workload. At each step, the tuner
proposes a new configuration, which is applied to the live system. Applying the proposed parameters
is a near-instant operation.


**CFS Parameter Details** The Completely Fair Scheduler (CFS) is a CPU scheduler for Linux that
aims to give every task a fair share of CPU time. It exposes various hyperparameters that allow
administrators to adjust its behavior. We tuned `min` ~~`g`~~ `ranularity` ~~`n`~~ `s`, which enforces a minimum
timeslice a task will receive. The prompt templates guided the agents to explore a range from
50,000 to 50,000,000 nanoseconds (0.05 ms to 50 ms). The default value on Kernel 5.15 is 3 ms.
Lower values for this parameter are expected to increase responsiveness at the cost of higher contextswitching overhead, while higher values improve throughput but can worsen latency.


**History** **Compression** **and** **Context** **Management** To manage context window limits and costs,
we employ different context strategies for the combined system versus the baselines.


In the **Actor-only** and **Speculator-only** baselines, the agents receive the full, unsummarized history
of all previous iterations. For the Actor-only baseline, the low frequency of interaction (once every
10-15s) means the context grows slowly, rendering compression unnecessary within the benchmark
duration.


In the **Speculator** + **Actor** combined system, history is managed via distinct prompt structures.
When the slower Actor is invoked, its prompt context contains a fully compressed summary of
all actions taken during its deliberation window. Each action from the Speculator is listed as a concise (parameter, result) pair. In contrast, the faster Speculator receives a hybrid context: it sees the
same compressed history from the last Actor cycle, supplemented by the full, verbose replies from
its own most recent actions. This dual-context mechanism allows the Actor to analyze long-term
trends from a compact summary, while the Speculator retains immediate, detailed context for its
rapid, reactive decisions.


B.3.2 Prompt Engineering for Multi-Agent Optimization


The following are the prompt templates used to guide the two LLM agents.


16


17


B.3.3 Speculative Reaction Time Benefits


To provide a targeted example of how speculation mitigates transient performance loss, we conducted a controlled experiment. In this scenario, the system is deliberately perturbed at time _t_ 0 by
setting the `min` ~~`g`~~ `ranularity` parameter to a highly suboptimal value (10 ms). We then compare
the system’s recovery under two configurations: the Actor-Speculator system and an Actor-Only
baseline, which replays only the actions proposed by the Actor from the full Actor-Speculator trace.


10


1


0.1


|speculator response|Col2|Speculator + Actor<br>Actor-only|
|---|---|---|
|speculator response|<br>actor response|<br>actor response|
|speculator response|<br>actor response||
|speculator response|||


0 5 10 15 20
Time (s)


Figure 7: A controlled experiment showing the system’s step response after a manual perturbation
at _t_ = 0. The **Actor-Speculator** system corrects the poor setting within a second, while the **Actor-**
**only** system must wait over 10 seconds for its next decision cycle. The quantitative results of this
experiment are summarized in Figure 5 (Right) in the main text.


As shown in Figure 7, the Actor-Speculator system reacts almost instantly. The fast Speculator,
seeing the immediate performance degradation, applies a corrective action that brings the system
back to an efficient state in about one second. In contrast, the Actor-Only system is forced to endure
the poor performance for over 10 seconds, as it must wait for the slower Actor to complete its
deliberation cycle before it can act. The performance gap shown in the plot is quantified in the main
text (Figure 5, Right).


18


C Cost–Latency Tradeoff


C.1 Breadth-focused speculation details


Algorithm 1 launches _k_ parallel speculative branches at any step _t_ ∈{1, . . ., _T_ - 1}, which is then
immediately followed with an API call. We assume each branch independently produces the correct
next call with probability _p_ . Let


_p_ ( _k_ ) := Pr(at least one branch is correct) = 1 − (1 − _p_ ) _[k]_ .


We assume the latency of a speculative call is Exp(α), and the latency of a real API call is Exp(β)
with β < α. Let _c_ denote the cost per unit time for both speculation and real API work. Let


E[ _T_ spec], E[ _M_ spec]


denote the expected latency and cost under Algorithm 1 (1 step breadth speculation), and similarly
let ( _T_ seq, _M_ seq) denote the sequential process with no speculation.


**Theorem 4** (Cost–Latency Tradeoff for Breadth Speculation) **.** _Under the setup above,_


_p_ ( _k_ ) [2]  
(1 + _p_ ( _k_ )) [2] [(][−] _[p]_ [(] _[k]_ [))] _[T]_ [−][1]


- ( _T_ - 1) _p_ ( _k_ ) _p_ ( _k_ ) [2]
+
1 + _p_ ( _k_ ) (1 + _p_ (


_p_ ( _k_ ) [2] _p_ ( _k_ ) [2]

[−]
(1 + _p_ ( _k_ )) [2] (1 + _p_ (


_E_ [ _T_ seq - _T_ spec]


[1] α

_T_ α + β


seq - _T_ spec] = [1]

_E_ [ _T_ seq] _T_


_T_ →∞ _p_ ( _k_ ) α
−−−−−−→ 1 + _p_ ( _k_ ) [·] α + β


_For cost, letting_ _k denote the number of_ [˜] distinct _actions produced across the k speculative branches,_


_p_ ( _k_ ) [2]  
(1 + _p_ ( _k_ )) [2] [(][−] _[p]_ [(] _[k]_ [))] _[T]_ [−][1]


�� ( _T_ - 1) _p_ ( _k_ ) _p_ ( _k_ ) [2]
+
1 + _p_ ( _k_ ) (1 + _p_ (


_p_ ( _k_ ) [2] _p_ ( _k_ ) [2]

[−]
(1 + _p_ ( _k_ )) [2] (1 + _p_ (


E[ _M_ spec - _M_ seq]


spec - _M_ seq]

= _k_ [˜]     - [1]
E[ _M_ seq] _T_


_T_


- α
_k_ ˜ +
α + β


      _T_ →∞ α
−−−−→ _k_ [˜] - _k_ ˜ +
α + β


- _p_ ( _k_ )
1 + _p_ ( _k_ ) [.]


_Note that_ _k is possibly di_ [˜] ff _erent from k because the k independent speculations might have duplica-_
_tions, in which case we kill the duplicated speculation processes._


_Proof._ The proof of the time savings ratio is given in Appendix A. For cost, we have

_Mseq_ ∝ _[T]_ β


_Mspec_ ∝ _[T]_ β [(˜] _[k]_ [ +][ 1)][ −] _[S][ T]_ [−][1]


- α
_k_ ˜ [1]

β [+] β(α + β)


where the speculative expression is due to each hit by time step _T_ - 1 will result in (i) the cached
next step not having a speculative window, hence does not launch any speculations (ii) over counting
hit action’s generation time with execution time.


Plug in expressions we obtained from Appendix A, we get the expression desired. 

C.2 Additional Empirical Results


C.2.1 E-commerce


**Trade-o** ff **between Prediction Accuracy and Cost.** The time cost in Figure 8a consists of latency
(Time to First Token) and output response time. The dashed vertical line represents the average user
typing time, estimated at 40 words per minute. At this threshold, the multi-agent setting achieves
approximately 34% prediction accuracy, meaning that in over one-third of cases the agent can return an immediate response without waiting for API execution. This demonstrates that speculation
can transform user experience from perceptibly laggy to effectively real-time in tool-heavy environments.


19


(a) (b)


Figure 8: **Prediction** **Accuracy** **against** **Speculator’s** **Cost** **across** **di** ff **erent** **models.** (a) Accuracy–Speculator time cost trade-off across models. The dashed line shows average user typing time.
(d) Accuracy–Speculator price trade-off across models, reflecting the monetary cost of speculative
execution.


C.2.2 OS hyperparameter tuning


10 [−1]


10 [−2]


10 [−3]


10 [−4]

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|107|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||5<br>106<br>107<br>e Tokens|||||||||
|||||||||||||||||
|||||||||||||||||


0 25 50 75 100 125 150 175 200
Time (s)


0 25 50 75 100 125 150 175 200
Time (s)


Speculator-only Actor-only Spec + Actor: Speculator Spec + Actor: Actor Spec + Actor: Total


Figure 9: **Cumulative token usage and cost over time.** The left and right plots show the cumulative
cost (USD) and total tokens used, respectively, for all three configurations. The vertical lines mark
the observed convergence point for each system. The Actor-only model converges at 200s


Table 2: Cumulative tokens and cost (in cents) at selected time marks. While Speculation incurs
higher instantaneous costs, its rapid convergence (bolded) prevents long-term resource waste compared to the slower Actor-only baseline.


Actor-only Speculator-only Actor+Speculator (Total)


**Elapsed Time** **Tokens** **Cost (cents)** **Tokens** **Cost (cents)** **Tokens** **Cost (cents)**


Base Prompt 744 0.02 690 0.01 1434 0.03
**13s** 1,216 0.05 14,973 0.18 **12,135** **0.17**
**20s** 2,211 0.09 **27,654** **0.34** 20,504 0.31
30s 3,631 0.15 45,768 0.57 32,459 0.48
60s 8,581 0.35 205,794 2.24 84,568 1.18
120s 26,398 0.96 778,253 8.12 261,855 3.53
**200s** **63,376** **2.18** 2,099,894 21.5 607,877 7.83

Table 3


**Impact** **of** **Context** **Strategy** **on** **Cost.** While the Speculator operates at high frequency, the cost
overhead is mitigated by the history compression mechanism described in §B.3.1. In the combined
system, the expensive Actor model reads only a compressed summary of the Speculator’s many
steps, rather than the raw verbose logs. This keeps the prompt size for the Actor relatively stable
compared to a linear growth of uncompressed history. As a result, the cost difference between the
Actor-only and Speculator+Actor systems is driven primarily by the number of Speculator calls,
rather than an explosion in context size per call.


20


As illustrated in Figure 9 and detailed in Table 2, the high frequency of the Speculator leads to rapid
growth in token consumption and cost. In practice, however, this growth is bounded by the system’s
fast convergence. The combined Actor-Speculator system converges in approximately 13 seconds,
while the Speculator-only system converges in 20 seconds. The Actor-only system converges after
200 seconds. Once an optimal state is reached, the tuning process concludes, rendering the potential for long-term exponential cost negligible in this context. Several optimization strategies, like
truncating the context to a fixed window or disabling exploration after convergence, could further
mitigate token growth but are left for future work.


C.3 Confidence-Aware Speculation


We formalize the branch selection problem introduced in Section 5.2.


**Model.** Fix a horizon _T_ and an integer _k_ ≥ 1. At each epoch _t_, the system is in mode _zt_ ∈{0, 1}:


   - _zt_ = 0: a speculation window is available,

   - _zt_ = 1: a cached correct action must be executed (no speculation).


Let _a_ and _b_ denote the actor and speculator latencies, respectively, and define the latency gain


ℓ := _a_                     - _b_                     - 0,


which is collected only at epochs with _zt_ = 1.


**Accuracy process.** At epochs with _zt_ = 0, an accuracy vector **P** _t_ = ( _P_ [1] _t_ [, . . .,] _[ P]_ _t_ _[k]_ [)][ ∈] [[0][,][ 1]] _[k]_ [ is realized]
and observed. Assume **P** _t_ ∼ _Ft_ independently across _t_, where the distribution _Ft_ may vary over time.

Given a realization **p**, let _p_ [(1)] ≥· · · ≥ _p_ [(] _[k]_ [)] denote the sorted coordinates.


**Action** **and** **hit** **probability.** At _zt_ = 0, the agent chooses _mt_ ∈{0, . . ., _k_ }, launching the top _mt_
branches at cost _cmt_ . The probability of obtaining a cached action is


_q_ ( _m_ ; **p** ) = 1 −


_m_


(1 − _p_ [(] _[ j]_ [)] ), _q_ (0; **p** ) = 0.

_j_ =1


**Mode transition.** If _zt_ = 1, then _zt_ +1 = 0 deterministically. If _zt_ = 0 and action _m_ is chosen,

_zt_ +1 = �1, w.p. _q_ ( _m_ ; **P** _t_ ),
0, w.p. 1 − _q_ ( _m_ ; **P** _t_ ).


**Reward.** The per-epoch reward is

_Rt_ = �−ℓ, _cmt_, _zztt_ == 1 0,.


The objective is to maximize E�� _Tt_ =1 _[R][t]_ �.


We then present the proof to the first part (general non-stationary) of Theorem 3.


_Proof._ Let _Vt_ [(] _[z]_ [)] denote the optimal expected total reward from epochs _t_, . . ., _T_ given mode _zt_ = _z_,
with terminal conditions _VT_ [(0)] +1 [=] _[ V]_ _T_ [(1)] +1 [=][ 0.]


By standard dynamic programming arguments, the Bellman equations are

_Vt_ [(1)] = ℓ + _Vt_ [(0)] +1 [,] (1)

_Vt_ [(0)] = E **P** ∼ _Ft_         - _m_ ∈{max0,..., _k_ }         -         - _cm_ + _q_ ( _m_ ; **P** ) _Vt_ (1)+1 [+][ (1][ −] _[q]_ [(] _[m]_ [;] **[ P]** [))] _[ V]_ _t_ [(0)] +1� [�] . (2)


Define the continuation gap
∆ _t_ := _Vt_ [(1)] +1 [−] _[V]_ _t_ [(0)] +1 [.]


21


Substituting equation 1 into equation 2 yields

_Vt_ [(0)] = _Vt_ [(0)] +1 [+][ E] **[P]** [∼] _[F][t]_               - max _m_ [{] _[q]_ [(] _[m]_ [;] **[ P]** [)][ ∆] _[t]_ [ −] _[cm]_ [}] �.


Hence, conditional on observing **p** at epoch _t_, the optimal decision maximizes


_q_ ( _m_ ; **p** ) ∆ _t_                 - _cm_,


establishing the stated policy.


Finally, using
∆ _t_ = _Vt_ [(1)] +1 [−] _[V]_ _t_ [(0)] +1 [=][ ℓ] [+] _[ V]_ _t_ [(0)] +2 [−] _[V]_ _t_ [(0)] +1 [,]

and substituting the expression for _Vt_ [(0)] +1 [gives the scalar recursion]


                          -                           ∆ _t_ = ℓ          - E **P** ∼ _Ft_ +1 max _m_ [{] _[q]_ [(] _[m]_ [;] **[ P]** [)][ ∆] _[t]_ [+][1][ −] _[cm]_ [}],


with terminal condition ∆ _T_ = 0 (or equivalently, ∆ _T_ −1 = ℓ). 

We then formally describe the stationary average reward corollary.

**Corollary** **5** (Stationary infinite-horizon average-reward policy) **.** _Assume_ _the_ _nonstationary_ _model_
_becomes stationary:_


   - **P** _t_ ∼ _F_ _i.i.d. across t,_


   - _the hit probability q_ ( _m_ ; **p** ) _and cost c are time-invariant,_


   - _the objective is to maximize long-run average reward._


_Let_ _q_ ¯( _m_ ) := E **P** ∼ _F_ [ _q_ ( _m_ ; **P** )] _denote the expected hit probability when launching m branches._

_Then the optimal average reward g_ [⋆] _satisfies_


_q_ ¯( _m_ ) ℓ                     - _cm_
_g_ [⋆] = max
_m_ ∈{0,..., _k_ } 1 + _q_ ¯( _m_ ) [.]


_Define_
∆ [⋆] := ℓ              - _g_ [⋆] .


_Then an optimal stationary policy at a speculation window is_

_m_ [⋆] ( **p** ) ∈ arg max
_m_ ∈{0,..., _k_ } [{] _[q]_ [(] _[m]_ [;] **[ p]** [)][ ∆][⋆] [−] _[cm]_ [}][.]


_In particular, the optimal policy mapping (the scalar_ ∆ [⋆] _) is time homogeneous._


_Proof._ Fix ∆ [⋆] - 0. Conditional on observing **p** at a speculation window, the stationary one-step
objective is
_f_ ( _m_ ; **p** ) := _q_ ( _m_ ; **p** ) ∆ [⋆]                   - _cm_, _m_ ∈{0, 1, . . ., _k_ }.
The first display in the corollary is exactly the definition of _m_ [⋆] ( **p** ) as any maximizer of _f_ ( _m_ ; **p** ).


To show the greedy marginal-threshold form, note that for sorted confidences _p_ [(1)] ≥· · · ≥ _p_ [(] _[k]_ [)],


_q_ ( _m_ ; **p** ) = 1 −


_m_


(1 − _p_ [(] _[ j]_ [)] ),

_j_ =1


so the discrete marginal gain from adding the ( _m_ + 1)-th branch is

_f_ ( _m_ + 1; **p** ) − _f_ ( _m_ ; **p** ) = ∆ [⋆][�] _q_ ( _m_ + 1; **p** ) − _q_ ( _m_ ; **p** ) [�]          - _c_ = ∆ [⋆] δ _q_ ( _m_ ; **p** ) − _c_,


where


_m_
δ _q_ ( _m_ ; **p** ) := _q_ ( _m_ + 1; **p** ) − _q_ ( _m_ ; **p** ) = - �(1 − _p_ [(] _[ j]_ [)] )� _p_ [(] _[m]_ [+][1)] .


_j_ =1


22


Moreover, δ _q_ ( _m_ ; **p** ) is nonincreasing in _m_ (diminishing returns), since _p_ [(] _[m]_ [+][1)] is nonincreasing and
the prefactor [�] _[m]_ _j_ =1 [(1][ −] _[p]_ [(] _[j]_ [)][) is nonincreasing in] _[ m]_ [.] [Hence the increments] _[f]_ [(] _[m]_ [ +][ 1;] **[ p]** [)][ −] _[f]_ [(] _[m]_ [;] **[ p]** [) are]
nonincreasing in _m_ as well.

Therefore there exists an index _m_ [⋆] ( **p** ) such that _f_ ( _m_ + 1; **p** ) − _f_ ( _m_ ; **p** ) ≥ 0 for all _m_ < _m_ [⋆] ( **p** ) and
_f_ ( _m_ + 1; **p** ) − _f_ ( _m_ ; **p** ) ≤ 0 for all _m_ ≥ _m_ [⋆] ( **p** ). Equivalently,

∆ [⋆]     - δ _q_ ( _m_ ; **p** ) ≥ _c_ for _m_ < _m_ [⋆] ( **p** ), ∆ [⋆]     - δ _q_ ( _m_ ; **p** ) ≤ _c_ for _m_ ≥ _m_ [⋆] ( **p** ),


which is precisely the greedy stopping rule stated in the corollary. 

C.4 Depth focused search


In the most general setting, a policy may choose both (i) how many parallel speculative branches
to launch, and (ii) how “deep” to unroll each speculative branch before initiating a real API call.
Analyzing the optimal policy in this full space is highly non-trivial due to the branching structure of
the execution tree.


To build intuition, we analyze two extrema regimes: (1) a _breadth-focused_ regime, where at each
step we launch _k_ parallel speculations and immediately follow each with an API call, and (2) a
_depth-focused_ regime, where execution follows a single branch as far as speculation and real API
calls allow. These two settings correspond to simplified extremes of the decision space, providing
interpretable analytic characterizations of the cost–latency tradeoff.


We analyzed the first in the previous section, and now analyze the opposite extreme: a depthfocused strategy. Under this policy, whenever either a speculative or real API call returns, the system
launches _one_ new real call and _one_ speculation on top of that branch. If the real API result is inconsistent with the corresponding speculative guess, all descendants of that speculation are discarded.
Let _p_ be the per-step correctness probability of a speculative guess.


Assume for simplicity that the real API call latency is deterministically _a_ and a speculative call
latency is _b_ < _a_ .
**Theorem 6** (Cost–Latency Tradeoff for Depth Speculation) **.** _Let_ _p be the probability that a specu-_
_lation is correct at each step._ _Then under the depth-focused policy described above,_
E[ _T_ seq               - _T_ spec] _[T]_ [−] [1]               - _[b]_               


seq - _T_ spec] = _[T]_ [−] [1]

E[ _T_ seq] _T_


[−] [1] _p_ �1 − _[b]_

_T_ _a_


_a_


�,


_and the expected cost satisfies_


+ _pb_ ⌊ _[a]_


_b_ [⌋]


[−] [1]
≈ _[T]_
_a_ _T_


_[a]_

_b_ [⌋][)][⌊] _[a]_ _b_


_b_ [⌋][)][⌊] _b_ [⌋]

2


- + _p_


E[ _M_ spec - _M_ seq]


spec - _M_ seq] = _[T]_ [−] [1]

E[ _M_ seq] _T_


     (1 − _p_ ) _a_ ⌊ _[a]_


(1+⌊ _[a]_

_[a]_ _b_

_b_ [⌋−] _[b]_


- - _a_
(1 − _p_ ) 2 _b_ [−] [1] 2


_T_


_T_


_Proof._ We directly calculate the expected time cost of speculation execution as follows


_T_ _spec_ = _a_ +


_T_ −1 
- _T_ - 1


_s_

_s_ =0


_p_ _[s]_ (1 − _p_ ) _[T]_ [−][1][−] _[s]_ [ _sb_ + ( _T_ - 1 − _s_ ) _a_ ]


Let
_S_ ∼ Binomial( _T_                - 1, _p_ ),

so that


      - _T_       - 1
P( _S_ = _s_ ) =
_s_


_p_ _[s]_ (1 − _p_ ) _[T]_ [−][1][−] _[s]_, _s_ = 0, . . ., _T_ - 1.


Then


_T_ −1


_s_ =0


- _T_ - 1

_s_


_p_ _[s]_ (1 − _p_ ) _[T]_ [−][1][−] _[s]_ [�] _sb_ + ( _T_ - 1 − _s_ ) _a_ [�]


= E[ _S b_ + ( _T_ - 1 − _S_ ) _a_ ]
= _b_ E[ _S_ ] + _a_ E[ _T_ - 1 − _S_ ].


23


Since
E[ _S_ ] = ( _T_                  - 1) _p_,


we obtain


_b_ ( _T_        - 1) _p_ + _a_ [�] _T_        - 1 − ( _T_        - 1) _p_ [�] = _a_ ( _T_        - 1) + ( _T_        - 1) _p_ ( _b_        - _a_ ).


Therefore,


_p_ _[s]_ (1 − _p_ ) _[T]_ [−][1][−] _[s]_ [�] _sb_ + ( _T_ - _s_ ) _a_ [�] = _aT_ + ( _T_ - 1) _p_ ( _b_ - _a_ ).


_T_ _spec_ = _a_ +


_T_ −1


_s_ =0


- _T_ - 1

_s_


Directly plug in _T_ _seq_ = _aT_, we get the expression desired.


For cost, we know that with (1 − _p_ ) probability, speculation is inconsistent with true action, and the
amount of tokens spent is from the branches that were spawn off before the correct action returns,
that is


                                      -                                      Token of this step = _a_ + ( _a_          - _b_ ) + ( _a_          - 2 _b_ ) + · · · + _a_ −⌊ _[a]_

_b_ [⌋] _[b]_


   -    = _a_ - ⌊ _[a]_ 
_b_ [⌋] [+][ 1]


�1 + ⌊ _[a]_


_[a]_ - - ⌊ _[a]_

_b_ [⌋] _b_ [⌋]


       - _b_
2


With probability _p_, speculation matches with true action, in which case

Token of this step = _a_ + ⌊ _[a]_

_b_ [⌋] _[b]_


Therefore


�1 + ⌊ _[a]_


_[a]_ - - ⌊ _[a]_

_b_ [⌋] _b_ [⌋]


       - _b_
2


  -   _a_ - ⌊ _[a]_  _b_ [⌋] [+][ 1]


_Mspec_ = _a_ + ( _T_ - 1)


  -  _p_ _a_ + ⌊ _[a]_ + (1 − _p_ )
 _b_ [⌋] _[b]_





                               

**Interpretation.** Note that compared to breadth speculation, Depth speculation produces speedups
unbounded in _p_ coefficient (the coefficient in front for breadth is [1] 2 [whereas] [for] [depth] [speculation]

this coefficient is 1). In terms of cost, depth speculation cost has highest order term (1 − _p_ )( 2 _[a]_ _b_ [−] [1] 2 [).]

This is governed by how many speculations one can spawn off before the current action is returned.


24