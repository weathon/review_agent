# A FANO-STYLE ACCURACY UPPER BOUND FOR LLM SINGLE-PASS REASONING IN MULTI-HOP QA


**Kaiyang Wan** **[2,1]** **, Lang Gao** **[1]** **, Honglin Mu** **[1]** **, Preslav Nakov** **[1]** **, Yuxia Wang** **[2,1]** **, Xiuying Chen** **[1]** _[∗]_


1MBZUAI, 2INSAIT, Sofia University “St. Kliment Ohridski”
Xiuying.Chen@mbzuai.ac.ae


ABSTRACT


Multi-Hop Question Answering (MHQA) requires integrating dispersed, interdependent evidence through sequential reasoning under noise. This task is challenging for LLMs as they have a finite per-pass output capacity, beyond which the integration of task-relevant evidence proves unreliable. Consequently, the single-pass
reasoning paradigm is inherently vulnerable to this capacity overflow. To formalize this bottleneck, our analysis establishes a Fano-style accuracy upper bound,
defining a theoretical performance ceiling for single-pass LLMs. This bound reveals that accuracy inevitably collapses once task complexity exceeds model capacity, providing general principles for capacity-aware representation and structuring of MHQA in LLMs. Building on these principles, we introduce a proof-ofconcept multi-call framework for MHQA, InfoQA. It ensures high per-step accuracy by combining capacity-aware task decomposition with active pruning of prior
reasoning traces, keeping the information load within the single-pass limit. It further achieves robustness by a dependency-explicit workflow that enables precise
control over the reasoning path. We construct a stringent and noise-rich benchmark to validate our theory and framework. Experimental results show that model
behavior aligns with our predicted capacity curves while InfoQA achieves consistent performance improvements. We hope our work inspires more LLM multi-step
reasoning methods: [�InfoQA.](https://github.com/KaiyangWan/InfoQA)


1 INTRODUCTION


Multi-Hop Question Answering (MHQA) (Yang et al., 2018; Trivedi et al., 2022; Mavi et al., 2024)
is an important NLP task with critical applications in real-world domains such as scientific literature analysis and complex fact verification (Yin et al., 2023; Yu et al., 2021). The task requires
integrating multiple, interdependent pieces of evidence that appear in different segments of a long
provided context. As a result, solving MHQA demands compositional reasoning: the model must
carry forward intermediate findings from one evidence source and use them to locate or interpret
information in subsequent sources. This stepwise dependency structure forms a reasoning chain,
where the accuracy of each intermediate inference directly determines the correctness of the final
answer. Accordingly, task success hinges on accurately resolving each reasoning hop while maintaining a coherent chain that faithfully composes intermediate findings into the final conclusion.


MHQA remains challenging for Large Language Models (LLMs) (Achiam et al., 2023; Bai
et al., 2023; Liu et al., 2024) despite recent advances in prompting strategies and reasoning techniques (Havrilla et al., 2024). As shown in Figure 1(a), intuitively, because an LLM generates only
a finite number of tokens in a single pass and each token has limited representational capacity, the
model is constrained by an upper bound on the total information it can carry forward. This output
capacity bound limits the amount of dispersed evidence that the model can reliably integrate at once.
When the reasoning chain spans multiple evidence sources or when the context contains substantial irrelevant content, the total information load often exceeds this bound. As a result, the model
becomes prone to capacity overflow, where relevant signals are diluted or overshadowed by noise,
leading to inaccurate intermediate inferences and, consequently, incorrect final answers.


_∗_ Corresponding author.


1


**Instruction Token** **Noise Token** **Evidence Token** **Reasoning Token** **Answer Token** **Integration Step**


_3-hop QA_
_Example_


... ...


(b) LLM Multi-call Reasoning Paradigm


Figure 1: Comparison of single-pass and multi-call reasoning paradigms. Single-pass reasoning is
constrained by the limited output capacity of LLMs, making it difficult to solve long-context and
multi-hop problems. Multi-call reasoning mitigates this by decomposing tasks into sequentially
dependent sub-steps, ensuring high per-step accuracy and a reliable reasoning chain.


To formalize this intuition, we first present an information-theoretic analysis that derives a Fanostyle accuracy upper bound for LLM single-pass reasoning. This analysis reveals the _Accuracy_
_Cliff_ : when the task’s information demand surpasses the model’s output capacity, performance does
not degrade gracefully but instead collapses sharply. We then examine why MHQA tasks are particularly prone to exceeding this cliff. By formalizing and dissecting the task structure, we identify two
compounding challenges: Stepwise Capacity Overflow, driven by the super-linear growth of information demand with hop count and context length, and Cross-Step Error Accumulation, stemming
from the amplification of even small per-step errors along the reasoning chain. Together, these analyses demonstrate that the single-pass paradigm is fundamentally inadequate for MHQA, motivating
the design of a capacity-aware, multi-call paradigm as shown in Figure 1(b).


Building on the identified single-pass limitations and the structural demands of MHQA, we introduce InfoQA, a proof-of-concept multi-call framework for MHQA. InfoQA serves to concretely
demonstrate how multi-call reasoning alleviates the dual crises of Stepwise Capacity Overflow and
Cross-Step Error Accumulation. It does so by _(i)_ capacity-aware task decomposition, which lowers
the information demand and secures per-step accuracy, _(ii)_ a dependency-explicit workflow, which
enforces alignment across reasoning steps and prevents the chain from drifting off course, and _(iii)_
iterative query contraction, which condenses the problem state and filters noise to keep information
load manageable.


To precisely control hop count and context length, and thereby modulate the task-side information demand, we construct a dedicated dataset to test our theory. Experiments confirm that singlepass methods indeed exhibit an _Accuracy Cliff_, with results closely matching our theoretical curves.
Moreover, as a proof-of-concept, InfoQA consistently outperforms single-pass baselines, further
demonstrating the practical advantage of multi-call reasoning.


Our contributions can be summarized as follows:


1. We provide a rigorous information-theoretic analysis of LLM single-pass reasoning, deriving a Fano-style accuracy upper bound and revealing the _Accuracy_ _Cliff_ phenomenon
(Section 2).


2. We dissect the structure of MHQA to explain why it is particularly prone to exceeding this
limit, identifying two compounding challenges: Stepwise Capacity Overflow and CrossStep Error Accumulation (Section 3).


3. We introduce InfoQA as a proof-of-concept in Section 4, and, in Section 5, we construct
a controlled benchmark to validate our theory while demonstrating the practical advantage
of multi-call reasoning paradigm.


2


2 THE INFORMATION BOTTLENECK IN LLM SINGLE-PASS REASONING


To analyze the inherent limits of single-pass LLM in complex reasoning, this section establishes
a theoretical framework. We begin by formalizing the task and our analytical tools, then derive a
universal accuracy upper bound that reveals a fundamental relationship between task complexity
and model capacity.


2.1 FORMALIZING MHQA AND ANALYTICAL BASIS


**Problem Formulation.** We study MHQA in a _closed-book_ setting, where the model must answer
solely from the provided context. Formally, the input consists of a User Query _Q_ and a Context
_C_ = ( _E, N_ ), where _E_ = _{e_ 1 _, . . ., eM_ _}_ are the necessary evidence snippets and _N_ is irrelevant
noise. The model generates an output _Y_, which includes its intermediate reasoning trace _R_ and the
final answer tokens. An extractor _g_ then maps this output to the predicted answer _A_ [ˆ] = _g_ ( _Y_ ).


**Analytical Basis.** Our analysis rests upon two foundational principles from information theory. We
use _H_ ( _·_ ) to denote Shannon entropy (Shannon, 1948) and _I_ ( _·_ ; _·_ ) for mutual information.


_1. Conditional Fano Inequality_ (Fano & Hawkins, 1961). This principle establishes that to achieve a
low error rate, the model’s output must sufficiently resolve the initial uncertainty about the answer. It
connects the error probability, _Pe_ = Pr( _A_ [ˆ] _̸_ = _A | Q, C_ ), to the residual uncertainty _H_ ( _A | Q, C, Y_ ):


_H_ ( _A | Q, C, Y_ ) _≤_ _h_ ( _Pe_ ) + _Pe_ log( _|A| −_ 1) _._ (1)


_2._ _Output_ _Entropy_ _Bound_ (Cover, 1999). This principle states that the amount of information an
output _Y_ can provide about the answer _A_ is fundamentally capped by its own entropy. Formally, the
mutual information is bounded as:


_I_ ( _A_ ; _Y_ _| Q, C_ ) _≤_ _H_ ( _Y_ ) _._ (2)


We provide a more detailed discussion in Appendix A.2.


2.2 A FANO-STYLE ACCURACY UPPER BOUND


The performance of LLMs in single-pass reasoning is governed by a fundamental principle: the
_information_ _bottleneck_ . Any single-pass output has a finite information-carrying capacity. When
a task’s complexity exceeds this capacity, a theoretical _performance ceiling_ emerges, making ideal
accuracy unattainable. By combining the Fano inequality with the output entropy bound from Section 2.1, we derive our central theorem, which forms the cornerstone of our framework.


**Theorem 1** (A Fano-Style Accuracy Upper Bound for Single-Pass Reasoning) **.** _For any single-pass,_
_closed-book policy,_ _let A_ _∈A be the ground-truth answer._ _Define the task’s_ _**information demand**_
_as_ _β_ ≜ _H_ ( _A_ _|_ _Q, C_ ) _and_ _the_ _model’s_ _**output**_ _**capacity**_ _as_ _C_ ≜ _H_ ( _Y_ ) _._ _The_ _maximum_ _achievable_
_accuracy, Acc_ = 1 _−_ _Pe, is implicitly bounded by the following relationship:_


_h_ ( _Acc_ ) + (1 _−_ _Acc_ ) log( _|A| −_ 1) _≥_ _β −_ _C,_ (3)


_where h_ ( _·_ ) _denotes the binary entropy function and h_ ( _Acc_ ) = _h_ (1 _−_ _Pe_ ) _._


This theorem dictates that whenever the information demand _β_ of a task exceeds the output capacity
_C_ of a model, achieving perfect accuracy ( _Acc_ = 1) becomes mathematically impossible.


2.3 FROM THEORY TO INTUITION: COROLLARIES AND THE ACCURACY CLIFF


While the exact bound in Theorem 1 is precise, its implications are more transparent through simplified corollaries. Together, they reveal a phenomenon we term the **Accuracy Cliff** .


**Linear Accuracy Bound.** By applying simple relaxations to the main theorem, we obtain a practical
linear upper bound on accuracy:


    _Acc_ _≤_ min 1 _,_ 1 _−_ _[β][ −]_ _[C][ −]_ [1]

log _|A|_


3


_._ (4)


**Uniform-Distribution Case.** In the common scenario where the context makes all potential answers
nearly equiprobable, the information demand simplifies to _β_ _≈_ log _|A|_, and the general bound from
Theorem 1 yields a more elegant and insightful upper bound on accuracy (proof in Appendix A.3):


    _Acc_ _≤_ min 1 _,_ _[C]_ [ + 1]

_β_


_._ (5)


**Phase** **Transition** **and** **the** **Cliff** **Edge.** As shown in
Figure 2 (taking _C_ = 200 as an example), equation 5
describes the Accuracy Cliff curve . It reveals a sharp,
phase-transition-like behavior: _(a)_ _Capacity-Sufficient_
_Regime (β_ _≤_ _C_ + 1 _):_ Before the critical threshold, the
accuracy is capped at 1, where performance is perfect
and stable. _(b) Capacity-Overflow Regime (β_ _> C_ +1 _):_
Immediately after this point, the performance ceiling
collapses. The maximum achievable accuracy is no
longer 1, but begins to decay hyperbolically according
to the ratio ( _C_ + 1) _/β_ . This transition from perfect accuracy to a rapid decay is the essence of the “Accuracy
Cliff,” illustrating how performance does not degrade
gracefully but instead falls off sharply when the task
complexity overwhelms the model’s capacity.


Figure 2: The Accuracy Cliff. The theoretical upper bound on accuracy is plotted against information demand _β_, using
_C_ = 200 as an illustrative example. Once
_β_ _> C_ + 1, the accuracy declines sharply.


This section establishes a universal performance bound
that formalizes the fundamental limits of the singlepass reasoning paradigm. It proves that single-pass accuracy is ultimately constrained by an insurmountable barrier: the ratio of the task’s information
demand _β_ to the model’s output capacity _C_ . This insight does more than just explain existing failures; it shows the path forward. If single-pass reasoning is inherently bounded, the only viable
solution is to transcend it. This theoretical bottleneck leads to the next critical questions: _**In a real-**_
_**world**_ _**MHQA**_ _**setting,**_ _**what**_ _**factors**_ _**cause**_ _**the**_ _**information**_ _**demand**_ _β_ _**to**_ _**grow**_ _**explosively?**_ _**And**_
_**how can we represent and structure the task to circumvent this single-pass limit?**_


3 ANATOMY OF THE MULTI-HOP CHALLENGE


In this section, we provide a detailed dissection of the MHQA task, building on the Accuracy Cliff
phenomenon from Section 2, to uncover the root causes of capacity overflow. The essence of MHQA
is the navigation of a _latent reasoning chain_, represented as:

_Z_ 0 _−→ϕ_ 1 _Z_ 1 _−→· · ·ϕ_ 2 _−−→ϕK_ _ZK_ _−−−→ϕK_ +1 _A._


In this chain, _Z_ 0 is the initial entity from the query, _A_ is the final answer, and each intermediate _Zk_
is a crucial “bridge” entity. The transformation _ϕk_ represents the reasoning process itself that uses
the context _C_ to advance from one entity to the next. This inherent chain structure is the source
of a dual challenge: the risk of **Stepwise** **Capacity** **Overflow** within each individual step, and the
systemic threat of **Cross-Step Error Accumulation** along the entire chain.


3.1 CHALLENGE 1: STEPWISE CAPACITY OVERFLOW


To predict when a model will be pushed off the Accuracy Cliff ( _β_ _> C_ ) established in Section 2, we
now model the information demand _β_ as a function of task properties in MHQA.


**Modeling** **Task-Side** **Demand.** To connect our theoretical bound with observable task properties,
we model _β_ as a function of hop count ( _h_ ) and effective context length ( _L_ ). Our model is based
on three assumptions: (i) a _baseline complexity β_ 0, representing the irreducible overhead of parsing
a query and locating evidence in any single step; (ii) a _context_ _burden_ that scales linearly with
context length ( _L_ ) to reflect the worsening signal-to-noise ratio; and (iii) a _hop amplification_ factor
_γ_ _[h][−]_ [1] ( _γ_ _≥_ 1) that captures the super-linear growth in complexity as uncertainty from prior steps
propagates to subsequent ones. Combining these gives us the parametric form:

_β_ ( _h, L_ ) = _β_ 0 + _α L γ_ _[h][−]_ [1] _._ (6)


4


This model shows that for _γ_ _>_ 1, _β_ grows super-linearly with the number of reasoning hops. This
exponential growth is the primary driver that pushes a model toward the “Accuracy Cliff.”


**Plug-in Accuracy Bound.** By substituting this demand model into equation 5, we get a concrete,
testable prediction for how accuracy is limited by task characteristics:


     - _C_ + 1
_Acc_ ( _h, L_ ) _≤_ min 1 _,_
_β_ 0 + _α L γ_ _[h][−]_ [1]


_._ (7)


This equation formalizes a Capacity Crisis: as the number of hops _h_ or context length _L_ increases,
the information demand _β_ escalates rapidly, heightening the likelihood of a capacity overflow _β_ _> C_
and a consequent collapse in accuracy.


3.2 CHALLENGE 2: CROSS-STEP ERROR ACCUMULATION


The second challenge, Cross-Step Error Accumulation, arises not from the informational depth of
any single step, but from the sequential nature of the reasoning chain itself. Even if the per-step
accuracy is high, the overall probability of success can still collapse due to the amplification of
small, individual errors as they propagate through the chain. To formalize this phenomenon, we
first define a _stepwise success event_, _Sk_, where the model’s prediction _Z_ [ˆ] _k_ must be both correct and
consistent with the prior state:


_Sk_ ≜        - _Z_ ˆ _k_ = _Zk_ _∧_ _Z_ ˆ _k_ = _ϕk_ ( ˆ _Zk−_ 1 _, Q, C_ )� _,_ ( _k_ = 1 _, . . ., K_ ) _,_


_SK_ +1 ≜      - _A_ ˆ = _A_ _∧_ _A_ ˆ = _ϕK_ +1( ˆ _ZK, Q, C_ )� _._


Overall success, Succ ≜ [�] _[K]_ _k_ =1 [+1] _[S][k]_ [, therefore requires every step in the chain to succeed.]


By the chain rule, Pr(Succ) is the product of the conditional success probabilities _pk_ at each step:


1 2 3 4 5 6
K


_K_ +1

- _pk,_ (8)


_k_ =1


1.0


0.8


0.6


0.4


0.2


0.0


Pr(Succ) =


_K_ +1

- Pr� _Sk_ _| S<k_ - =


_k_ =1


_pk_ = Pr� _Z_ ˆ _k_ = _Zk_ _∧_ _Z_ ˆ _k_ = _ϕk_ ( ˆ _Zk−_ 1 _, Q, C_ ) _S<k_ - _._
���
(9)
If we assume a uniform per-step success rate of at least
1 _−_ _ε_, the overall success probability is bounded by:


Pr(Succ) _≥_ (1 _−_ _ε_ ) _[K]_ [+1] _≈_ 1 _−_ ( _K_ +1) _ε,_ (10)


Figure 3: Error Accumulation. Even a
small per-step error rate ( _ε_ ) causes a rapid
decay in overall success probability as the
number of hops ( _K_ ) increases.


This linear decay, visualized in Figure 3, formalizes the
_Compounding Crisis_ . It shows how the chain structure decay in overall success probability as the
acts as an error amplifier. While the Capacity Crisis is number of hops ( _K_ ) increases.
the “spark” that generates individual errors, Cross-Step
Error Accumulation is the “powder keg” that makes even small sparks catastrophic, causing the
entire reasoning process to fail.


**An Inescapable Dilemma.** Built upon the above two challenges, our deconstruction of the multihop challenge reveals a dual, interlocking crisis rooted in its latent chain structure. The single-pass
reasoning paradigm is thus caught in a vise grip: it is simultaneously vulnerable to _Stepwise Capacity_
_Overflow_, which generates inevitable per-step errors, and to _Cross-Step Error Accumulation_, which
guarantees that these errors will be catastrophically amplified. This dual-front assault renders the
conventional single-pass paradigm fundamentally untenable for complex reasoning. Therefore, **the**
**core issue is the very single-pass paradigm we force it into.**


4 INFOQA: A MULTI-CALL REASONING PARADIGM FOR MHQA


Our theoretical analysis in Section 2 established a universal performance limit for single-pass reasoning: the _Accuracy Cliff_, which dictates that accuracy inevitably collapses when information demand ( _β_ ) exceeds model capacity ( _C_ ).


5


**Capacity-Aware Task Decomposition**


Figure 4: The InfoQA framework integrates three key components: (1) _Capacity-Aware_ _Task_ _De-_
_composition_, which reduces the information demand by generating single-hop sub-questions; (2)
_Dependency-Explicit_ _Workflow_, where the evolving contracted query carries the reasoning state
across steps; and (3) _Iterative_ _Query_ _Contraction_, which prunes reasoning traces and rewrites the
query with _Z_ [ˆ] _k_ . Each LLM call approximates _ϕk_ and produces _Z_ [ˆ] _k_ .


Subsequently, our deconstruction of the MHQA task in Section 3 revealed exactly why this limit is so
perilous in practice. We found that MHQA’s structure not only causes _β_ to _escalate exponentially_,
making capacity overflow almost certain, but also _catastrophically_ _amplifies_ the resulting errors
along its reasoning chain. This dual diagnosis dictates the principles for an effective solution: a
successful methodology must be both _capacity-aware_ to manage per-step information load, and
_robust_ to maintain the integrity of the chain.


4.1 THE INFOQA FRAMEWORK


InfoQA is a multi-call reasoning framework designed from the ground up to navigate the dual crises
of multi-hop reasoning. It operationalizes the principle of decomposition by breaking down a single,
high-demand query into a sequence of capacity-aligned sub-tasks, each with a manageable information load. This is achieved through three synergistic components, as depicted in Figure 4.


**Capacity-Aware Task Decomposition.** The first step in InfoQA is to transform a high-level multihop question into a simpler, single-hop sub-question. This decomposition is critical for reducing the
initial information demand _β_ = _H_ ( _A | Q, C_ ) to a more manageable per-step demand, _β_ 1 = _H_ ( _Z_ 1 _|_
_Q, C_ ). For a question such as: _”What_ _is_ _the_ _birth_ _date_ _of_ _the_ _lead_ _actor_ _in_ _the_ _movie_ _directed_ _by_
_the_ _person_ _who_ _wrote_ _’Dune’?”_, the initial sub-question is generated as: _”Based_ _on_ _the_ _provided_
_context,_ _who_ _wrote_ _’Dune’?”_ By focusing the LLM on this narrow task, we ensure the reasoning
step remains well within its single-pass capacity _C_, thereby directly counteracting the _Capacity_
_Crisis_ .


**Dependency-Explicit Workflow.** Once the problem is decomposed, a critical challenge is to reliably link sequential steps, countering the _Compounding_ _Crisis_ described in equation 10. InfoQA
achieves this with a _Dependency-Explicit Workflow_ . Instead of relying on a model’s internal memory, the workflow’s state is explicitly maintained and passed as the _current, contracted query itself_ .
After finding _Z_ [ˆ] _k_, the query _Qk_ is updated to _Qk_ +1 by embedding this finding. For example: _Qk_ :
”..., directed by the person who wrote ’Dune’?” _→_ Finding: ”Frank Herbert” _→_ _Qk_ +1: ”..., directed
by Frank Herbert?”. This makes the reasoning chain transparent, controllable, and robust against
error propagation.


**Iterative** **Query** **Contraction.** This mechanism is the engine that ensures the information load
remains low throughout the entire reasoning process. After each step, InfoQA contracts the problem
state via two actions: _Pruning_, where the extensive reasoning trace is discarded to prevent noise
accumulation, and _Contraction_, where the query is rewritten with the latest finding _Z_ [ˆ] _k_ . By iteratively
pruning thoughts and contracting the query, we ensure the prompt for every step represents the most
concise form of the _remaining_ problem. This prevents prompt length from growing with reasoning
depth, acting as the crucial enabler that protects the entire chain from _Stepwise Capacity Overflow_ .


5 EXPERIMENTS


We conducted experiments to validate the two central claims of this work. Our evaluation is twofold:
**1.** **Theory Validation:** We first tested whether the empirical performance of LLMs aligns with our
theoretical _Fano-style accuracy upper bound_, confirming that the Accuracy Cliff is a real and predictable phenomenon. **2.** **Framework Validation:** We then evaluated whether InfoQA framework
can effectively transcend this theoretical limit, alleviating the capacity bottleneck to yield substantial
performance gains.


6


5.1 EXPERIMENTAL SETUP


**Benchmark Construction.** Existing MHQA benchmarks are unsuitable for our study as they lack
fine-grained control over task difficulty and are often compromised by data artifacts, preventing
a rigorous test of our theory. We therefore constructed a new, stringent, and noise-rich synthetic
benchmark guided by three core principles: _(i) systematic control_ over information demand ( _β_ ) by
varying hop count and distractor scale; _(ii) high semantic similarity_ between evidence and distractors
to prevent shortcut learning; and _(iii) a path maximization strategy_ for evidence placement to enforce
genuine, non-trivial reasoning chains. This process yielded a suite of datasets with systematically
varied hop counts and context lengths, allowing for a precise evaluation of model performance
against our theoretical bounds. We provide the key statistics of our benchmark in Table 1 and
detailed construction consideration and algorithm in Appendix A.4.


**Models** **and** **Baselines.** We Table 1: Statistics of our synthetic multi-hop QA benchmark.
conducted our experiments on

**1-hop** **2-hop** **3-hop** **4-hop**

the Qwen3-8B and -14B (Yang
et al., 2025). We chose this **Context Length** _L_ [0.5k, 1k, 2k, 4k, 8k, 10k]

**Samples per** _L_ 300 300 300 300

publicly available model fam- **Total Samples** 1,800 1,800 1,800 1,800
ily to minimize architectural and **Evidence Order** [ _e_ 1] [ _e_ 2 _, e_ 1] [ _e_ 2 _, e_ 3 _, e_ 1] [ _e_ 2 _, e_ 4 _, e_ 3 _, e_ 1]
training biases, allowing for a **Evidence Position** [1 _/_ 2] [1 _/_ 3 _,_ 2 _/_ 3] [1 _/_ 4 _,_ 2 _/_ 4 _,_ 3 _/_ 4] [1 _/_ 5 _,_ 2 _/_ 5 _,_ 3 _/_ 5 _,_ 4 _/_ 5]
fair evaluation of the reasoning **Grand Total** 7,200
_paradigms_ themselves. All results were obtained via official API calls. For all methods, we set temperature to 0.2 and a maximum generation length of 4096 tokens. Other parameters were default. We compared InfoQA
against a comprehensive suite of strong single-pass baselines, including: (i) Direct Prompting, (ii)
Chain-of-Thought (CoT) (Wei et al., 2022), (iii) Self-Consistency (SC) [1] (Wang et al., 2023b), (iv)
Self-Refine [2] (Madaan et al., 2023), (v) ReAct (Yao et al., 2023), (vi) Plan-and-Solve (Wang et al.,
2023a), and (vii) Self-Ask (Press et al., 2023). All baseline prompts were implemented as zero-shot,
single-pass methods, carefully designed to follow the principles laid out in their respective original
papers. All LLM calls within the InfoQA framework used the same backbone model and inference
settings as the baselines. We used F1 as the evaluation metric.


Table 1: Statistics of our synthetic multi-hop QA benchmark.


**1-hop** **2-hop** **3-hop** **4-hop**


**Context Length** _L_ [0.5k, 1k, 2k, 4k, 8k, 10k]
**Samples per** _L_ 300 300 300 300
**Total Samples** 1,800 1,800 1,800 1,800
**Evidence Order** [ _e_ 1] [ _e_ 2 _, e_ 1] [ _e_ 2 _, e_ 3 _, e_ 1] [ _e_ 2 _, e_ 4 _, e_ 3 _, e_ 1]
**Evidence Position** [1 _/_ 2] [1 _/_ 3 _,_ 2 _/_ 3] [1 _/_ 4 _,_ 2 _/_ 4 _,_ 3 _/_ 4] [1 _/_ 5 _,_ 2 _/_ 5 _,_ 3 _/_ 5 _,_ 4 _/_ 5]


**Grand Total** 7,200


5.2 EMPIRICAL VALIDATION OF THE ACCURACY CLIFF


The results of Qwen3-14B and Qwen3-8B showed the same phenomenon; we analyze Qwen3-14B
and present Qwen3-8B in Appendix A.7. Table 2 summarizes the average F1 scores across different
context lengths and hop counts of Qwen3-14B. Our first experimental goal is to validate our core
theoretical claim: the performance of single-pass models in MHQA is governed by an _accuracy cliff_ .
Concretely, we tested whether the empirical performance of strong prompting baselines conforms
to the Fano-style accuracy upper bound derived in Section 2.


**Parameter** **Estimation** **Protocol.** To connect theory with data, we fit the parameters _θ_ =
( _β_ 0 _, α, γ, C_ ) of our plug-in accuracy bound (Eq. 7) to empirical F1 scores, using F1 as a proxy
for accuracy, Acc( _h, L_ ) = F1( _h, L_ ). We minimized the mean absolute deviation between the ob
[�]
servations and the bound:


      - _C_ + 1
Acc( _h, L_ ) _−_ min 1 _,_
���� _β_ 0 + _α L γ_ _[h][−]_ [1]


_._ (11)
����


min
_θ_


( _h,L_ )


For each baseline we conducted a fine-grained grid search over ( _α, γ, β_ 0 _, C_ ) and select the minimizer with respect to MAE. The fitted curves were then overlaid with empirical points (F1) as a
function of the fitted effective demand _β_ ( _h, L_ ). We present the fitted plots in Figure 5, with detailed
fitting statistics in Appendix A.6 and fitting algorithm in Appendix A.5.


**Alignment with Predicted Curves.** Three consistent patterns emerged. _**(i) Accuracy cliff:**_ as the
effective demand _β_ grows with hop count and context length, empirical points adhere closely to the
theoretical bound and then collapse once _β_ ≳ _C_ +1, consistent with the predicted cliff.


1Our implementation of Self-Consistency involves generating five reasoning paths by querying the model
with varying temperatures: _{_ 0.1, 0.3, 0.5, 0.7, 0.9 _}_ . The final answer is determined by a majority vote.
2For Self-Refine, we report the final answer after one iteration of feedback and refinement.


7


Table 2: Average F1 scores of Qwen3-14B across different reasoning depths and context lengths.
We compare InfoQA with single-pass baselines: Chain-of-Thought (CoT), Self-Refine (S-R), SelfConsistency (S-C), ReAct, Plan-and-Solve (P&S), Self-Ask (S-A), and InfoQA with ablation: w/o
Capacity-Aware Task Decomposition (D.) and w/o Pruning Past Reasoning Trace (P.).


**Average F1 Score**


**Hops** **Context Length** **Direct** **CoT** **S-R** **S-C** **ReAct** **P&S** **S-A** **InfoQA** **w/o D.** **w/o P.**


**1**


**2**


**3**


**4**


0.5k **1.00** **1.00** **1.00** **1.00** **1.00** **1.00** **1.00** **1.00** 0.87 **1.00**
1k **1.00** **1.00** **1.00** **1.00** 0.99 **1.00** **1.00** **1.00** 0.78 **1.00**
2k 0.99 **1.00** 0.99 **1.00** 0.99 **1.00** 0.99 **1.00** 0.79 **1.00**
4k 0.97 0.99 0.98 **1.00** 0.97 0.98 0.97 0.99 0.63 **1.00**
8k 0.93 0.98 0.82 **1.00** 0.72 0.89 0.93 0.98 0.31 0.96
10k 0.91 0.96 0.79 **0.98** 0.59 0.84 0.85 0.96 0.28 0.90


0.5k 0.78 **1.00** **1.00** **1.00** **1.00** **1.00** 0.84 **1.00** 0.85 **1.00**
1k 0.74 **1.00** 0.98 **1.00** **1.00** **1.00** 0.75 **1.00** 0.84 **1.00**
2k 0.66 **1.00** 0.94 **1.00** 0.99 0.98 0.69 **1.00** 0.83 **1.00**
4k 0.54 0.99 0.77 0.99 0.96 0.85 0.68 **1.00** 0.84 0.98
8k 0.23 0.79 0.39 0.83 0.53 0.54 0.63 **0.96** 0.52 0.88
10k 0.18 0.76 0.44 0.81 0.55 0.60 0.63 **0.89** 0.39 0.83


0.5k 0.70 0.97 0.95 **0.98** **0.98** **0.98** 0.85 **0.98** 0.93 **0.98**
1k 0.55 0.97 0.83 **0.98** 0.96 0.92 0.75 **0.98** 0.80 0.96
2k 0.41 0.94 0.66 **0.97** 0.84 0.83 0.74 0.96 0.67 0.94
4k 0.31 0.72 0.30 0.77 0.59 0.61 0.64 **0.84** 0.60 0.79
8k 0.06 0.32 0.12 0.35 0.24 0.19 0.52 **0.64** 0.43 0.44
10k 0.04 0.27 0.10 0.26 0.20 0.15 0.39 **0.42** 0.29 0.39


0.5k 0.26 0.98 0.90 **0.99** 0.96 0.96 0.94 0.96 0.92 0.95
1k 0.13 0.95 0.79 **0.98** 0.87 0.93 0.84 0.96 0.84 0.92
2k 0.09 0.77 0.46 0.80 0.64 0.66 0.76 **0.95** 0.75 0.83
4k 0.02 0.49 0.34 0.54 0.41 0.38 0.55 **0.93** 0.56 0.69
8k 0.00 0.17 0.13 0.21 0.13 0.16 0.36 **0.69** 0.32 0.36
10k 0.00 0.09 0.09 0.12 0.06 0.06 0.21 **0.30** 0.23 0.18


**Overall Average (2–4 hop)** 0.32 0.73 0.57 0.75 0.66 0.66 0.65 **0.86** 0.65 0.78


**1 hop Average** 0.97 0.98 0.93 **0.99** 0.88 0.95 0.96 **0.99** 0.61 0.98
**2 hop Average** 0.52 0.92 0.75 0.94 0.84 0.83 0.70 **0.97** 0.71 0.95
**3 hop Average** 0.34 0.70 0.49 0.72 0.63 0.61 0.65 **0.80** 0.62 0.75
**4 hop Average** 0.09 0.57 0.45 0.61 0.51 0.53 0.61 **0.80** 0.60 0.65


**Context Average (2–4 hop)**


0.5k 0.58 0.98 0.95 **0.99** 0.98 0.98 0.88 0.98 0.90 0.98
1k 0.48 0.97 0.87 **0.99** 0.94 0.95 0.78 0.98 0.83 0.96
2k 0.38 0.90 0.69 0.92 0.83 0.83 0.73 **0.96** 0.75 0.92
4k 0.29 0.73 0.47 0.77 0.65 0.61 0.62 **0.92** 0.67 0.82
8k 0.10 0.43 0.21 0.46 0.30 0.30 0.50 **0.76** 0.42 0.56
10k 0.07 0.37 0.21 0.40 0.27 0.27 0.41 **0.54** 0.30 0.47


_**(ii)**_ _**Capacity**_ _**and**_ _**hop**_ _**inflation:**_ CoT substantially increases the effective single-pass capacity _C_
and reduces hop inflation _γ_ relative to Direct, thereby delaying the onset of the cliff; S-C exhibits
a similar trend. _**(iii)**_ _**Method-specific**_ _**overheads:**_ certain methods introduce additional demand.
For example, S-A shows a large _β_ 0 (higher base demand), which offsets the benefit of a larger
_C_ . Overall, the fitted overlays corroborate these findings: empirical markers align tightly with the
theoretical envelope at low _β_ and diverge only when the bound becomes active.


5.3 PERFORMANCE OF INFOQA


**Overall performance.** As shown in Table 2, InfoQA achieves the best results across most settings,
with an overall average of 0.86 on 2–4 hop tasks, substantially outperforming strong single-pass
baselines such as S-C (0.75) and CoT (0.73). The key strength of InfoQA lies in its robustness along
two axes. First, in terms of _depth robustness_, InfoQA sustains high accuracy even as the hop count
increases, whereas single-pass baselines suffer sharp degradation beyond 2 hops due to compounded
informational demand and error accumulation. Second, in terms of _length_ _robustness_, InfoQA remains reliable under long contexts (8k–10k tokens), while methods like Direct and ReAct collapse
to near-zero. This stability comes from explicitly pruning past traces and contracting queries, which
prevents context inflation and keeps the effective demand _β_ within the model’s per-pass capacity _C_ .


**Ablation** **study.** We further examined the contribution of InfoQA’s key design choices: _(i)_ _w/o_
_Decomposition (w/o D.)_, which executed the full reasoning chain in a single-pass without capacity
control, and _(ii) w/o Pruning (w/o P.)_, which preserved all past reasoning traces without contraction.


8


Fano-Style Bound (F1) Empirical (h=1) Empirical (h=2) Empirical (h=3) Empirical (h=4)


S-R


250 500 750 1000 1250

S-A


200 300 400 500 600


S-C


200 400 600

w/o D.


75 100 125 150 175 200


1.0


0.8


0.6


0.4


0.2


Direct


0.0
500 1000 1500 2000 2500 3000


CoT


200 400 600 800

P&S


50 100 150 200


1.0


0.8


0.6


0.4


0.2


ReAct


0.0
200 400 600


x-axis: Effective Information Demand (fitted)


Figure 5: Qwen3-14B F1 vs. theoretical curves across single-pass methods. The x-axis shows the
estimated effective information demand ( _β_ ), fitted per method, and the y-axis shows the F1 score.


As shown in Table 2, w/o D. quickly saturated at longer contexts and higher hops (overall average
0.65), confirming the single-pass bottleneck predicted by the Accuracy Cliff. Meanwhile, w/o P.
performed better but still trailed InfoQA (0.78 vs. 0.86), as unpruned traces inflated context length
and exacerbated cross-step errors. These results highlighted that both _capacity-aware decomposition_
and _iterative pruning_ were indispensable: decomposition ensured per-step demand remained within
capacity, while pruning prevented error amplification across the reasoning chain.


**Error Analysis of InfoQA.** Compared with single-pass baselines, InfoQA exhibits a distinct error profile. Since its multi-call design try to prevent capacity overflow, most residual failures are
not caused by information bottlenecks but by _semantic_ _drift_ during iterative query contraction. In
particular, the contracted query may sometimes omit subtle constraints (e.g., temporal qualifiers or
entity disambiguation), causing the reasoning chain to pursue a plausible but incorrect path. Another source of failure lies in the _intrinsic model capacity_ : even when the task is decomposed into
single-hop sub-questions, extremely long contexts can exceed the model’s base comprehension ability. Combined with multi-hop error accumulation, this results in degraded performance for InfoQA
on long-context, high-hop scenarios. These errors suggest that future work should focus on better decomposition to minimize the sub-task demand, improving contraction fidelity, and improving
model’s base capacity.


6 RELATED WORK


**LLM Single-pass Prompting Methods.** Single-pass prompting methods ask the model to complete
the entire reasoning process in one forward generation, without external decomposition or iterative
calls. Classic examples (Kojima et al., 2022; Chen et al., 2025; Zamfirescu-Pereira et al., 2023)
include Direct prompting, Chain-of-Thought (CoT) (Wei et al., 2022). More structured variants
such as ReAct (Yao et al., 2023), Plan-and-Solve (Wang et al., 2023a), and Self-Ask (Press et al.,
2023) guide the model with explicit prompting templates to elicit stepwise reasoning. Despite these
design differences, all of them operate within a single forward pass, meaning that the reasoning chain
must fit entirely within the model’s per-pass information capacity. As a result, their performance
inevitably degrades when task complexity exceeds this capacity. Our work formalizes and quantifies
this single-pass capacity limit, showing that it gives rise to the “accuracy cliff” observed in MHQA
task.


**Multi-call Methods.** In contrast to single-pass prompting, multi-call methods decompose reasoning
into multiple model invocations, with each call addressing a sub-task. A representative line of work
is Self-Refine (Madaan et al., 2023), which iteratively generates feedback and refines the answer.


9


Other approaches adopt recursive or pipeline-style reasoning, such as multi-step decomposition for
question answering (Li et al., 2024), programming (Qian et al., 2024; Kim et al., 2024), fact checking (Xie et al., 2025) and writing (Shao et al., 2024; Wan et al., 2025). The success of these methods
has empirically validated the effectiveness of distributing the reasoning load across multiple calls.
Building on this paradigm, our work provides a theoretical foundation from an information capacity perspective to explain why such an approach is beneficial. We show that single-pass methods
face an inherent capacity bottleneck and that multi-call reasoning can provably keep the per-step
information demand below the model’s capacity.


**Information-Theoretic Perspectives on MHQA.** Information theory is useful to analyze the challenges and bottlenecks of MHQA tasks. Xu et al. (2025) focused on retrieval-based systems, using
pointwise conditional V-information to quantify the contribution of documents and optimize the
retriever’s selection process. Chen (2025) addressed the parameter storage capacity, establishing a
theoretical lower bound on the number of parameters necessary to reliably store multi-hop reasoning chains within the model weights. Complementary to these retrieval and storage perspectives, our
work targets the closed-book setting to formalize the single-pass output channel capacity bottleneck,
identifying the Accuracy Cliff where performance collapses due to limited generation bandwidth
rather than insufficient knowledge storage.


7 CONCLUSION AND FUTURE WORK


In this work, we began by providing an information-theoretic analysis of MHQA with LLMs. By
deriving a Fano-style accuracy upper bound, we formalized the fundamental capacity bottleneck of
single-pass reasoning and revealed the Accuracy Cliff, where accuracy collapses once information
demand exceeds model capacity. Building on this insight, we dissected MHQA to identify the dual
challenges of stepwise capacity overflow and cross-step error accumulation, showing why singlepass reasoning is inherently fragile. To validate our theoretical analysis, we introduced InfoQA, a
capacity-aware multi-call proof-of-concept that decomposes complex queries into manageable steps,
prunes noisy traces, and explicitly controls dependency flow. Our experiments results align with the
predicted capacity curves and InfoQA achieves consistent gains.


Looking ahead, we believe this work opens several promising directions: First, extending our analysis to multi-call settings could clarify how information accumulates across calls and what new limits
emerge. Second, adaptive decomposition strategies would let systems dynamically decide how to
split queries based on complexity and improving model’s base information capacity. Third, applying
the capacity-bound perspective to domains such as science or law would test its robustness under
real-world noise and reasoning demands.


ETHICS STATEMENT


As part of our experimental design, we generated a synthetic dataset in which all personal names and
company names are entirely fictitious. These synthetic entities do not correspond to real individuals
or organizations. The use of fabricated identifiers was intentional, in order to avoid potential privacy,
legal, or ethical concerns that could arise from using real-world data. No personally identifiable
information (PII) or sensitive data were collected or used in this work. Therefore, we believe that
our research does not pose risks to individuals, groups, or organizations.


REPRODUCIBILITY STATEMENT


To ensure the reproducibility of our work, we provide an anonymous GitHub repository containing:
(1) the synthetic dataset used in our experiments, (2) the code for constructing the dataset, (3) the
implementation of all baselines as well as our proposed model, (4) the code used to fit empirical
results to our theoretical curves, and (5) detailed README guidelines to facilitate reproduction of
our results. All experiments can be reproduced directly using the provided resources. In addition,
we have uploaded a compressed archive containing all these files as part of our paper submission, so
that reviewers can access and reproduce our results even without relying on the external repository.


10


REFERENCES


Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical
report. _ArXiv preprint_, abs/2303.08774, 2023.


Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge,
Yu Han, Fei Huang, et al. Qwen technical report. _ArXiv preprint_, abs/2309.16609, 2023.


Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu,
Yuhang Zhou, Te Gao, and Wanxiang Che. Towards reasoning era: A survey of long chain-ofthought for reasoning large language models. _ArXiv preprint_, abs/2503.09567, 2025.


Thomas Y Chen. How many parameters for multi-hop? An information-theoretic capacity law for
knowledge retrieval in large language models. In _Proceedings of the Workshop on Knowledgeable_
_Foundation Models at ACL_, Vienna, Austria, 2025.


Thomas M Cover. _Elements of information theory_ . John Wiley & Sons, 1999.


Robert M Fano and David Hawkins. Transmission of information: A statistical theory of communications. _American Journal of Physics_, 29(11):793–794, 1961.


Alexander Havrilla, Sharath Chandra Raparthy, Christoforos Nalmpantis, Jane Dwivedi-Yu,
Maksym Zhuravinskyi, Eric Hambro, and Roberta Raileanu. Glore: When, where, and how
to improve LLM reasoning via global and local refinements. In _Proceedings_ _of_ _the_ _Forty-first_
_International_ _Conference_ _on_ _Machine_ _Learning_, ICML ’2024, Vienna, Austria, 2024. OpenReview.net.


Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W. Mahoney, Kurt Keutzer, and
Amir Gholami. An LLM compiler for parallel function calling. In _Proceedings of the Forty-first_
_International_ _Conference_ _on_ _Machine_ _Learning_, ICML ’2024, Vienna, Austria, 2024. OpenReview.net.


Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large
language models are zero-shot reasoners. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle
Belgrave, K. Cho, and A. Oh (eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_ _35:_
_Annual_ _Conference_ _on_ _Neural_ _Information_ _Processing_ _Systems_, NeurIPS ’2022, New Orleans,
LA, USA, 2022.


Xinyi Li, Sai Wang, Siqi Zeng, Yu Wu, and Yi Yang. A survey on LLM-based multi-agent systems:
Workflow, infrastructure, and challenges. _Vicinagearth_, 1(1):9, 2024.


Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao,
Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. DeepSeek-V3 technical report. _ArXiv preprint_,
abs/2412.19437, 2024.


Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri
Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad
Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine:
Iterative refinement with self-feedback. In Alice Oh, Tristan Naumann, Amir Globerson, Kate
Saenko, Moritz Hardt, and Sergey Levine (eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_
_Systems 36: Annual Conference on Neural Information Processing Systems_, NeurIPS ’2023, New
Orleans, LA, USA, 2023.


Vaibhav Mavi, Anubhav Jangra, Adam Jatowt, et al. Multi-hop question answering. _Foundations_
_and Trends® in Information Retrieval_, 17(5):457–586, 2024.


Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah Smith, and Mike Lewis. Measuring
and narrowing the compositionality gap in language models. In Houda Bouamor, Juan Pino, and
Kalika Bali (eds.), _Findings of the Association for Computational Linguistics:_ _EMNLP 2023_, pp.
5687–5711, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.
findings-emnlp.378.


11


Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize
Chen, Yusheng Su, Xin Cong, et al. ChatDev: Communicative agents for software development.
In _Proceedings_ _of_ _the_ _62nd_ _Annual_ _Meeting_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics_,
ACL ’2024, pp. 15174–15186, Bangkok, Thailand, 2024.


Claude E Shannon. A mathematical theory of communication. _The Bell system technical journal_,
27(3):379–423, 1948.


Yijia Shao, Yucheng Jiang, Theodore Kanell, Peter Xu, Omar Khattab, and Monica Lam. Assisting
in writing Wikipedia-like articles from scratch with large language models. In Kevin Duh, Helena
Gomez, and Steven Bethard (eds.), _Proceedings_ _of_ _the_ _2024_ _Conference_ _of_ _the_ _North_ _American_
_Chapter of the Association for Computational Linguistics:_ _Human Language Technologies (Vol-_
_ume 1: Long Papers)_, pp. 6252–6278, Mexico City, Mexico, 2024. Association for Computational
Linguistics.


Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop questions via single-hop question composition. _Transactions of the Association for Compu-_
_tational Linguistics_, 10:539–554, 2022. doi: 10.1162/tacl ~~a~~ ~~0~~ 0475.


Kaiyang Wan, Honglin Mu, Rui Hao, Haoran Luo, Tianle Gu, and Xiuying Chen. A cognitive
writing perspective for constrained long-form text generation. _ArXiv_ _preprint_, abs/2502.12568,
2025.


Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim.
Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language
models. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), _Proceedings_ _of_ _the_
_61st Annual Meeting of the Association for Computational Linguistics (Volume 1:_ _Long Papers)_,
pp. 2609–2634, Toronto, Canada, 2023a. Association for Computational Linguistics. doi: 10.
18653/v1/2023.acl-long.147.


Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha
Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language
models. In _Proceedings of the Eleventh International Conference on Learning Representations_,
ICLR ’2023, Kigali, Rwanda, 2023b. OpenReview.net.


Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi,
Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language
models. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh
(eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_ _35:_ _Annual_ _Conference_ _on_ _Neural_
_Information Processing Systems_, NeurIPS ’2022, New Orleans, LA, USA, 2022.


Zhuohan Xie, Rui Xing, Yuxia Wang, Jiahui Geng, Hasan Iqbal, Dhruv Sahnan, Iryna Gurevych,
and Preslav Nakov. FIRE: Fact-checking with iterative retrieval and verification. In _Findings_
_of_ _the_ _Association_ _for_ _Computational_ _Linguistics:_ _NAACL_ _2025_, NAACL ’25, pp. 2901–2914,
Albuquerque, NM, USA, 2025.


Hao Xu, Yunxiao Zhao, Jiayang Zhang, Zhiqiang Wang, and Ru Li. LOG: A local-to-global optimization approach for retrieval-based explainable multi-hop question answering. In _Proceedings_
_of the 31st International Conference on Computational Linguistics_, ICLR ’2025, pp. 9085–9095,
Singapore, 2025.


An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. _ArXiv preprint_, abs/2505.09388,
2025.


Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (eds.),
_Proceedings_ _of_ _the_ _2018_ _Conference_ _on_ _Empirical_ _Methods_ _in_ _Natural_ _Language_ _Processing_,
pp. 2369–2380, Brussels, Belgium, 2018. Association for Computational Linguistics. doi:
10.18653/v1/D18-1259.


12


Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao.
React: Synergizing reasoning and acting in language models. In _Proceedings of the Eleventh In-_
_ternational Conference on Learning Representations_, ICLR ’2023, Kigali, Rwanda, 2023. OpenReview.net.


Fan Yin, Jesse Vig, Philippe Laban, Shafiq Joty, Caiming Xiong, and Chien-Sheng Wu. Did you
read the instructions? Rethinking the effectiveness of task definitions in instruction learning. In
Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), _Proceedings_ _of_ _the_ _61st_ _Annual_
_Meeting_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics_ _(Volume_ _1:_ _Long_ _Papers)_, pp. 3063–
3079, Toronto, Canada, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.
acl-long.172.


Jianxing Yu, Qinliang Su, Xiaojun Quan, and Jian Yin. Multi-hop reasoning question generation and
its application. _IEEE Transactions on Knowledge and Data Engineering_, 35(1):725–740, 2021.


J. D. Zamfirescu-Pereira, Richmond Y. Wong, Bjoern Hartmann, and Qian Yang. Why johnny
can’t prompt: How non-AI experts try (and fail) to design LLM prompts. In Albrecht Schmidt,
Kaisa V¨a¨an¨anen, Tesh Goyal, Per Ola Kristensson, Anicia Peters, Stefanie Mueller, Julie R.
Williamson, and Max L. Wilson (eds.), _Proceedings_ _of_ _the_ _2023_ _CHI_ _Conference_ _on_ _Human_
_Factors in Computing Systems_, CHI ’2023, pp. 437:1–437:21, Hamburg, Germany, 2023. ACM.
doi: 10.1145/3544548.3581388.


A APPENDIX


A.1 LLM USAGE


We used LLMs as auxiliary tools during the preparation of this work. Specifically, LLMs were
employed in three ways: (1) for proofreading and identifying minor typographical errors in the
manuscript, (2) for generating a synthetic dataset that was used as part of our experiments, and (3)
for automatic code completion during the development of our implementation. All research ideas,
experimental design, and final manuscript writing remain the responsibility of the authors.


A.2 INFORMATION-THEORETIC PRELIMINARIES: FULL PROOFS AND DISCUSSION


This appendix expands upon the information-theoretic preliminaries introduced in Section 2. We
provide complete proofs of the conditional Fano inequality and the output entropy bound, together
with intuitive interpretations and implications for multi-hop reasoning.


A.2.1 PROOF OF THE CONDITIONAL FANO INEQUALITY


**Setup.** Let _A_ be the ground-truth answer, _A_ [ˆ] = _g_ ( _Y, Q, C_ ) the prediction derived from the model
output _Y_ (allowing the estimator to depend on ( _Q, C_ )), and ( _Q, C_ ) denote the query and context.
Define the error event _E_ = _{A_ [ˆ] _̸_ = _A}_ with probability _Pe_ ≜ Pr( _E_ = 1 _| Q, C_ ).


**Step 1:** **Decomposition of conditional entropy.** We begin from the chain rule of entropy:


_H_ ( _A | Q, C, Y_ ) = _H_ ( _A, E_ _| Q, C, Y_ ) _−_ _H_ ( _E_ _| A, Q, C, Y_ ) _._ (12)


Since _E_ is a deterministic function of ( _A, Y, Q, C_ ), the last term vanishes, yielding


_H_ ( _A | Q, C, Y_ ) = _H_ ( _E_ _| Q, C, Y_ ) + _H_ ( _A | E, Q, C, Y_ ) _._ (13)


**Step 2:** **Bounding each term.** By the fact that conditioning reduces entropy,


_H_ ( _E_ _| Q, C, Y_ ) _≤_ _H_ ( _E_ _| Q, C_ ) = _h_        - _Pe_        - _,_


where _h_ ( _·_ ) is the binary entropy function. For the second term, conditioned on _E_ = 1 (error), the
uncertainty about _A_ is at most log( _|A|−_ 1), since all but the predicted answer remain possible. Thus


_H_ ( _A | E, Q, C, Y_ ) _≤_ _Pe_ log( _|A| −_ 1) _._


13


**Step 3:** **Combine.** Together, we obtain the bound:


_H_ ( _A | Q, C, Y_ ) _≤_ _h_ ( _Pe_ ) + _Pe_ log( _|A| −_ 1) _._ (14)


**Step** **4:** **Mutual** **information** **form.** Rearranging yields the equivalent lower bound on mutual
information:
_I_ ( _A_ ; _Y_ _| Q, C_ ) _≥_ _H_ ( _A | Q, C_ ) _−_         - _h_ ( _Pe_ ) + _Pe_ log( _|A| −_ 1)� _._ (15)


This bound states that unless the predictor extracts at least _β_ = _H_ ( _A_ _|_ _Q, C_ ) bits of information
about _A_, a nontrivial error rate is unavoidable. In other words, _information_ _demand_ _implies_ _error_
_floor_ .


A.2.2 PROOF OF THE OUTPUT ENTROPY BOUND


**Setup.** The output _Y_ is a sequence of tokens from vocabulary _V_ . We distinguish two modeling
choices for the length constraint.


**Step 1: Mutual information bounded by entropy.** By definition and because conditioning reduces
entropy,
_I_ ( _A_ ; _Y_ _| Q, C_ ) _≤_ _H_ ( _Y_ _| Q, C_ ) _≤_ _H_ ( _Y_ ) _._


**Step 2:** **Upper bounds on output entropy (two cases).**


    - **Fixed length** _m_ **(or padded-to-** _m_ **with a special token).** Then _Y_ _∈_ _V_ _[m]_ and

_H_ ( _Y_ ) _≤_ log _|V |_ _[m]_ = _m_ log _|V |._ (16)

    - **Variable** **length,** **at** **most** _m_ **tokens (no padding).** Then _Y_ _∈_ [�] _[m]_ _k_ =0 _[V]_ _[k]_ [with cardinality]

  - _mk_ =0 _[|][V][ |][k]_ [=] _[|][V][ |]_ _|V_ _[m]_ _|−_ [+1] 1 _[−]_ [1], hence


_m_ +1

     - _|V |_ _−_ 1
_H_ ( _Y_ ) _≤_ log
_|V | −_ 1


_._ (17)


Either equation 16 or equation 17 provides a valid capacity upper bound, depending on the modeling
choice.


A.2.3 IMPLICATIONS FOR MULTI-HOP REASONING


The two inequalities together establish an **information bottleneck** for single-pass reasoning:


    - **Demand side (** _β_ = _H_ ( _A | Q, C_ ) **).** Multi-hop QA inherently requires integrating dispersed
and noisy evidence, which inflates the conditional entropy of the answer.

    - **Supply** **side** **(** _C_ = _H_ ( _Y_ ) **).** The single-pass output has a finite entropy budget, given by
equation 16 or equation 17, scaling with output length and vocabulary.

    - **Error** **floor** **(** _Pe_ **).** Whenever _β_ _>_ _C_, Fano’s inequality dictates that the error probability
cannot vanish, regardless of model size or training.


This formalizes the intuitive statement: _“No_ _matter_ _how_ _smart_ _the_ _model_ _is,_ _if_ _the_ _task_ _demands_
_more information than the output can encode, an error plateau is inevitable.”_


A.3 PROOF OF THE FANO-STYLE ACCURACY UPPER BOUND AND ITS COROLLARIES


**Notation and setup.** Throughout, all logarithms are base 2, so entropies and mutual information
are measured in bits. We consider a _closed-book,_ _single-pass_ setting with a discrete answer space
_A_, _|A|_ _≥_ 2. The query _Q_ and context _C_ are given (conditioning variables). Let _A_ _∈A_ be the gold
answer, _Y_ be the model’s single-pass output (a random variable taking values in a finite or countable
set of token sequences), and _A_ [ˆ] = _g_ ( _Y_ ) be the predicted answer obtained by a _deterministic_ extractor
_g_ . Define the error probability


_Pe_ = Pr( _A_ [ˆ] _̸_ = _A | Q, C_ ) _,_ and _Acc_ = 1 _−_ _Pe._


14


We also define the task _information_ _demand_ _β_ ≜ _H_ ( _A_ _|_ _Q, C_ ) and the model’s _output_ _capacity_
_C_ ≜ _H_ ( _Y_ _| Q, C_ ). When needed, one may upper-bound _C_ by modeling constraints on _Y_ : if _Y_ has
fixed length _m_ (or is padded to _m_ with a special token) then


_H_ ( _Y_ _| Q, C_ ) _≤_ _m_ log _|V |_ ;


if _Y_ has variable length at most _m_ without padding, then

_H_ ( _Y_ _| Q, C_ ) _≤_ log� _|V ||V_ _[m]_ _|−_ [+1] 1 _−_ 1        - _._


**Two ingredients.** We rely on two standard facts (made conditional on ( _Q, C_ )):


1. **Conditional Fano inequality** (e.g. Fano & Hawkins 1961, conditionalized on ( _Q, C_ )). For
any estimator _A_ [ˆ] of _A_,

_H_ ( _A | Q, C,_ _A_ [ˆ] ) _≤_ _h_ ( _Pe_ ) + _Pe_ log        - _|A| −_ 1� _,_ (18)


where _h_ ( _·_ ) is the binary entropy function.

2. **Output-entropy (capacity) bound** (e.g. Cover 1999): for any ( _A, Y_ ),


_I_ ( _A_ ; _Y_ _| Q, C_ ) _≤_ _H_ ( _Y_ _| Q, C_ ) = _C._ (19)


**A** **useful** **comparison** **between** _Y_ **and** _A_ [ˆ] = _g_ ( _Y_ ) **.** Because _A_ [ˆ] is a deterministic function of _Y_,
conditioning on the _richer_ variable _Y_ cannot increase uncertainty relative to conditioning on _A_ [ˆ] :


_H_ ( _A | Q, C, Y_ ) _≤_ _H_ ( _A | Q, C,_ _A_ [ˆ] ) _._ (20)


Combining equation 20 with equation 18 yields

_H_ ( _A | Q, C, Y_ ) _≤_ _h_ ( _Pe_ ) + _Pe_ log       - _|A| −_ 1� _._ (21)


**Proof of Theorem 1.** Start from the chain rule for conditional mutual information:


_I_ ( _A_ ; _Y_ _| Q, C_ ) = _H_ ( _A | Q, C_ ) _−_ _H_ ( _A | Q, C, Y_ ) = _β −_ _H_ ( _A | Q, C, Y_ ) _._


Apply equation 21 to upper-bound the second term:

_I_ ( _A_ ; _Y_ _| Q, C_ ) _≥_ _β −_           - _h_ ( _Pe_ ) + _Pe_ log( _|A| −_ 1)           - _._


Together with the capacity bound equation 19, we obtain

_β −_        - _h_ ( _Pe_ ) + _Pe_ log( _|A| −_ 1)        - _≤_ _I_ ( _A_ ; _Y_ _| Q, C_ ) _≤_ _C._


Rearranging gives
_h_ ( _Pe_ ) + _Pe_ log( _|A| −_ 1) _≥_ _β −_ _C._
Finally, substitute _Pe_ = 1 _−_ _Acc_ and note that _h_ ( _Pe_ ) = _h_ (1 _−_ _Acc_ ) = _h_ ( _Acc_ ) to obtain


_h_ ( _Acc_ ) + (1 _−_ _Acc_ ) log( _|A| −_ 1) _≥_ _β −_ _C,_ (22)


which is Theorem 1.


**Derivation of the Linear Accuracy Bound (Eq. 4).** Starting from Theorem 1,


_h_ ( _Acc_ ) + (1 _−_ _Acc_ ) log( _|A| −_ 1) _≥_ _β −_ _C._


Use the elementary relaxations _h_ ( _Acc_ ) _≤_ 1 (binary entropy is at most 1) and log( _|A|−_ 1) _≤_ log _|A|_
(for _|A| ≥_ 2) to obtain
1 + (1 _−_ _Acc_ ) log _|A|_ _≥_ _β −_ _C._

Rearrange:


_[ −]_ _[C][ −]_ [1]

= _⇒_ _Acc_ _≤_ 1 _−_ _[β][ −]_ _[C][ −]_ [1]
log _|A|_ log _|A|_


1 _−_ _Acc_ _≥_ _[β][ −]_ _[C][ −]_ [1]


_._
log _|A|_


Because accuracy is trivially at most 1, we write the bound with a cap:


    _Acc_ _≤_ min 1 _,_ 1 _−_ _[β][ −]_ _[C][ −]_ [1]

log _|A|_


_,_


which is Eq. 4. (When the right-hand side exceeds 1, the min _{·,_ 1 _}_ keeps the bound meaningful.)


15


**Derivation** **for** **the** **Uniform-Distribution** **Case** **(Eq.** **5).** In the common case where the context
does not provide strong cues to distinguish among candidates, the posterior distribution _p_ ( _a | Q, C_ )
over answers _a_ _∈A_ is close to uniform. Intuitively, this corresponds to situations where many
distractor entities of the correct type (e.g., names, dates, or organizations) appear in the context, so
that each candidate remains nearly equally plausible given ( _Q, C_ ). Formally, this means that the
entropy of the answer distribution approaches its maximum, i.e.,


_β_ = _H_ ( _A | Q, C_ ) _≈_ log _|A|,_


since log _|A|_ is the entropy of a uniform distribution over _A_ . Equivalently, the KL divergence
between _p_ ( _a | Q, C_ ) and the uniform distribution _U_ ( _a_ ) is small, i.e.,


_D_ KL( _p_ ( _· | Q, C_ ) _∥_ _U_ ( _·_ )) _≈_ 0 _,_


so that the uncertainty is essentially governed by the candidate set size _|A|_ itself. Since in this
regime _β_ _≈_ log _|A|_ _≥_ log( _|A| −_ 1), replacing log( _|A| −_ 1) by _β_ enlarges the left-hand side of the
inequality, hence yields a weaker but still valid bound.


Plugging this approximation into Theorem 1 and again relaxing _h_ ( _Acc_ ) _≤_ 1 gives


1 + (1 _−_ _Acc_ ) _β_ _≥_ _β −_ _C._


Rearranging to isolate _Acc_ :


(1 _−_ _Acc_ ) _β_ _≥_ _β −_ _C −_ 1 = _⇒_ 1 _−_ _Acc_ _≥_ 1 _−_ _[C]_ [ + 1]


_._
_β_


[ + 1]

= _⇒_ _Acc_ _≤_ _[C]_ [ + 1]
_β_ _β_


Capping at 1 yields


    _Acc_ _≤_ min 1 _,_ _[C]_ [ + 1]

_β_


_,_


which is Eq. 5. This form emphasizes the _capacity–demand ratio_ ( _C_ +1) _/β_ and makes the “accuracy
cliff” explicit: the bound equals 1 whenever _β_ _≤_ _C_ +1, and decays hyperbolically once _β_ _> C_ +1.


**Remarks.**


    - The proof only uses that _A_ [ˆ] is a (deterministic) function of _Y_ ; if _A_ [ˆ] were randomized
given _Y_, equation 20 would still hold by the data-processing inequality (conditioning on
( _Q, C, Y_ ) is at least as informative as conditioning on ( _Q, C,_ _A_ [ˆ] )).

    - The capacity constant _C_ is taken as the _effective_ single-pass capacity _H_ ( _Y_ _| Q, C_ ) realized
by the decoding policy, but it can be upper-bounded by modeling constraints on _Y_ (e.g.,
maximum length and vocabulary size).

    - Equality conditions in the Fano-style bound are generally not attained in practical settings;
the utility of the bound is in predicting the regime change at _β_ _≈_ _C_ + 1 and explaining
aggregate trends (the “accuracy cliff”).


A.4 DETAILED BENCHMARK CONSTRUCTION


This appendix provides a detailed account of the design principles and generation pipeline for our
synthetic, noise-rich Multi-Hop Question Answering (MHQA) benchmark.


A.4.1 MOTIVATION AND DESIGN PRINCIPLES


As stated in the main text, our primary motivation was to overcome the limitations of existing
MHQA datasets, which often lack the fine-grained control over difficulty and the data hygiene necessary for a rigorous evaluation of information-theoretic limits. To this end, our benchmark was
designed around three core principles:


1. **Systematic Control** **over Information Demand (** _β_ **):** The benchmark must allow for the
precise and independent control of factors known to influence _β_, primarily the reasoning
hop count ( _h_ ) and the context length ( _L_ ). This enables a systematic study of how performance degrades as information demand scales, allowing for a direct comparison with our
theoretical Accuracy Cliff curves.


16


2. **Resistance** **to** **Heuristics** **and** **Shortcuts:** The benchmark must be designed to test genuine reasoning rather than retrieval or pattern matching. This is achieved by ensuring all
evidence is previously unseen by the model and is embedded within a large number of
semantically similar distractors. The high similarity forces the model to perform careful
entity disambiguation and information extraction, rather than relying on shallow heuristics.


3. **Maximization** **of** **Reasoning** **Path:** The placement of evidence within the context must
enforce a non-trivial reasoning path. A model should not be able to answer a multi-hop
question by simply reading the context linearly. Our design forces the model to traverse
back and forth across large sections of distractor text, maximizing the cognitive load and
testing the model’s ability to maintain a coherent reasoning state.


A.4.2 DATA GENERATION PIPELINE


Our generation pipeline is a programmatic, four-step process designed to instantiate challenging
MHQA problems that adhere to the principles above.


**Step** **1:** **Reasoning** **Chain** **Instantiation.** We begin by defining a set of abstract semantic templates (e.g., ‘(Person A, wrote, Book B)‘, ‘(Book B, was adapted into, Movie C)‘). For a _k_ -hop
question, we sample _k_ such templates and populate them with distinct entities drawn from a curated
knowledge base. This forms the gold evidence chain, _{e_ 1 _, e_ 2 _, . . ., ek}_ . A question is then programmatically generated to connect the initial entity in _e_ 1 to the final entity in _ek_, with the final entity
serving as the ground-truth answer. For example, a 2-hop chain might be ‘(Frank Herbert, wrote,
Dune)‘ and ‘(Dune, was adapted into, Dune (2021 film))‘, leading to the question ”What film was
adapted from the book written by Frank Herbert?”.


**Step 2:** **Semantically Rich Distractor Generation.** For each piece of gold evidence _ei_, we generate a set of _Nd_ distractor statements. This is done by taking the semantic template of _ei_ and
substituting its entities with other entities of the same type (e.g., other authors, other books). For instance, for ‘(Frank Herbert, wrote, Dune)‘, distractors could be ‘(Isaac Asimov, wrote, Foundation)‘
or ‘(Frank Herbert, wrote, Dune Messiah)‘. This process creates a large pool of plausible but factually incorrect statements that are highly similar to the gold evidence, making the task a stringent test
of precision.


**Step 3: Context Assembly and Path Maximization.** This step realizes our third design principle.
The gold evidence snippets are deliberately placed out of their logical reasoning order within the
context. For instance, for a 3-hop task with logical order _e_ 1 _→_ _e_ 2 _→_ _e_ 3, we might place them in the
document in the physical order _e_ 2 _→_ _e_ 3 _→_ _e_ 1. The snippets are inserted at regular intervals within
the document (e.g., at 1/4, 2/4, and 3/4 of the context length). The generated distractor statements
are then randomly shuffled and used to fill the space between the evidence snippets. This strategy
forces the model to first find _e_ 2 in the middle, use its information to find _e_ 3 further down, and then
use that result to jump back to near the beginning to find _e_ 1.


**Step 4: Noise Padding and Finalization.** Finally, to control the overall context length ( _L_ ), we pad
the assembled context with generic, irrelevant noise text (e.g., paragraphs generated by LLMs). This
padding is added to the beginning, end, and between existing statements until the target token count
(from 500 to 10,000) is reached. This ensures the model must not only handle targeted, similar
distractors but also vast amounts of truly irrelevant information, faithfully simulating real-world,
noisy long-context scenarios.


This pipeline produces a suite of challenging and controllable datasets, whose key statistics are
summarized in Table 1 in the main text. By systematically varying _h_ and _L_, we can precisely map
out the performance landscape and validate our theoretical predictions.


17


**Algorithm 1** Multi-hop Reasoning Dataset Construction

1: **Input:** _N_ = 300, _L_ = _{_ 500 _,_ 1000 _, . . .,_ 10000 _}_
2: **Output:** Multi-hop datasets for each target length _L_


3: **Phase 1:** **Initialize**
4: Define entity dictionary _E_ with categories (personnel, organizations, etc.)
5: Define templates _T_ ; each _t ∈T_ includes entity sequence _Et_, chain templates _Ct_, questions _Qt_


6: **Phase 2:** **Generate Base Chains**
7: **for** _i_ = 1 to _N_ **do**
8: _t ←T_ [ _i_ mod _|T |_ ]
9: _chaini_ _←_ GENERATECHAIN( _t_ )
10: **end for**
11: **function** GENERATECHAIN( _t_ )
12: Sample entities for _Et_ from _E_
13: Format _Ct_ with entities to get _chain_ ~~_t_~~ _exts_
14: **return** ( _t, chain_ ~~_t_~~ _exts, entity_ ~~_v_~~ _alues_ )
15: **end function**


16: **Phase 3:** **Generate Distractors**
17: **function** GENERATEDISTRACTORS( _chain, ndist, nvar, nnoise_ )
18: Apply distractor templates to create similar and noisy sentences
19: **return** ( _similar, noise_ )
20: **end function**


21: **Phase 4:** **Build Multi-length Dataset**
22: **for** _Li_ _∈_ _L_ **do**
23: Compute _ndist, nnoise_ from _Li_
24: **for** each _chaini_ **do**
25: **for** _h_ = 1 to 4 **do**
26: Scale distractors: _n_ [(] _dist_ _[h]_ [)] _[←⌊][n][dist][ ·]_ [ (1 + 0] _[.]_ [6(] _[h][ −]_ [1))] _[⌋]_

27: _distractors ←_ GENERATEDISTRACTORS( _chaini_, _n_ [(] _dist_ _[h]_ [)] [, 5,] _[ n][noise]_ [)]
28: _sample ←_ BUILDSAMPLE( _h, chaini, distractors_ )
29: **end for**
30: **end for**
31: **end for**
32: **function** BUILDSAMPLE( _h, chain, D_ )
33: _q_ _←_ _Qchain.template_ [ _h−_ 1]; _a ←_ _chain.entities_ [ _h_ ]
34: _S_ _←_ first _h_ sentences from _chain.chain_ ~~_t_~~ _exts_
35: _ctx ←_ CREATECONTEXT( _S, D_ )
36: **return** ( _q, a, S, D, ctx_ )
37: **end function**


38: **Phase 5:** **Assemble Context**
39: **function** CREATECONTEXT( _S, D_ )
40: Interleave _D.similar_ and _D.noise_
41: Insert _S_ at fixed positions based on _h_ (e.g. for _h_ = 3: positions [1 _/_ 4 _,_ 2 _/_ 4 _,_ 3 _/_ 4])
42: Pad if needed to target token length
43: **return** context
44: **end function**


45: **Phase 6:** **Save Output**
46: **for** _h ∈{_ 1 _,_ 2 _,_ 3 _,_ 4 _}_, _Li_ _∈_ _L_ **do**
47: Write all ( _q, a, ctx_ ) triples to $h hop/multi ~~h~~ op ~~c~~ hain ~~$~~ Lk.json
48: **end for**
49: Save dataset statistics


18


A.5 FITTING ALGORITHM


This section details the procedure used to fit the relationship between _effective information demand_
and task performance (F1). We formalize the parametric model, the loss function, the search strategy, numerical safeguards, and the computational complexity, and we provide pseudocode for reproducibility.


**Model** **Assumption** **(Beta–Bound** **Structure)** For a given _reasoning_ _depth_ _h_ _∈{_ 1 _,_ 2 _,_ 3 _,_ 4 _}_ and
_context length L_ (in tokens), we posit that the effective information demand is


_β_ ( _h, L_ ) = _α L γ_ _[h][−]_ [1] + _β_ 0 _,_


with parameters _α_ _>_ 0, _γ_ _>_ 1, and _β_ 0 _≥_ 0. The attainable F1 is upper-bounded by an inverse
dependence on _β_ :

                 - _C_ + 1                  F1(� _h, L_ ) = min 1 _,_ _β_ ( _h, L_ ) _,_


where _C_ _≥_ 0 captures a constant-information offset and induces a kink at F1 = 1.

[�]


**Objective** Given empirical observations F1emp( _h, L_ ), we estimate ( _α, γ, β_ 0 _, C_ ) by minimizing the
mean absolute error (MAE):


1
_L_ ( _α, γ, β_ 0 _, C_ ) =
_N_


( _h,L_ )


���F1( _h, L_ ) _−_ F1emp( _h, L_ )�� _,_


where _N_ is the number of ( _h, L_ ) pairs (here _N_ = 4 _×_ 6 = 24).


**Search** **Strategy:** **Fine-Grained** **Grid** **Search** To avoid local minima introduced by the nonsmooth kink at F1 = 1, we employ a _fine-grained grid search_ over

[�]

_α ∈A,_ _γ_ _∈G,_ _β_ 0 _∈B,_ _C_ _∈C._


Unless otherwise stated, we use


_A_ = logspace�10 _[−]_ [4] _,_ 10 _[−]_ [2] _,_ 15� _,_ _G_ = linspace�1 _._ 05 _,_ 3 _._ 00 _,_ 20� _,_


_B_ = linspace�0 _,_ 200 _,_ 21� _,_ _C_ = linspace�20 _,_ 400 _,_ 25� _._


Each method (Direct, CoT, S-R, S-C, ReAct, P&S, S-A) is fitted independently, yielding its own
( _α, γ, β_ 0 _, C_ ).


**Implementation Details and Numerical Stability**


    - **Vectorization.** For each ( _α, γ_ ) pair, we first compute the base term _αLγ_ _[h][−]_ [1] for all ( _h, L_ ),
and then sweep over _β_ 0 and _C_ . This reduces redundant computation and improves throughput.

    - **Stability** **at** **small** _β_ **.** We enforce _β_ ( _h, L_ ) _←_ max _{β_ ( _h, L_ ) _,_ 10 _[−]_ [9] _}_ to avoid division by
zero.


    - **Upper-bound consistency.** The cap min(1 _, ·_ ) ensures fidelity in the high-resource regime
where F1 _→_ 1.

[�]

    - **Optional** **weighting.** If desired, a weight _w_ ( _h, L_ ) can be introduced in _L_ to emphasize
specific depths or lengths (default: uniform).


**Computational Complexity** Let _|A|_, _|G|_, _|B|_, and _|C|_ denote the grid sizes and _N_ the number of
samples. The complexity per method is


_O_           - _|A| |G| |B| |C| N_           - _._


Since _N_ = 24 is small, the overall runtime remains practical under vectorized implementations.
Faster variants can be obtained via coarse-to-fine (multistage) search or by shrinking grid ranges.


19


**Algorithm 2** Fine-Grained Grid Fitting for One Method
**Require:** Data _{_ ( _hi, Li,_ F1 _i_ ) _}_ _[N]_ _i_ =1 [; grids] _[ A][,][ G][,][ B][,][ C]_
1: best ~~l~~ oss _←_ + _∞_, best _←_ ∅
2: **for** _α ∈A_ **do**
3: **for** _γ_ _∈G_ **do**
4: base _i_ _←_ _α Li γ_ _[h][i][−]_ [1] _∀i_
5: **for** _β_ 0 _∈B_ **do**
6: _βi_ _←_ max(base _i_ + _β_ 0 _,_ 10 _[−]_ [9] )
7: **for** _C_ _∈C_ **do**
8: F1� _i_ _←_ min�1 _,_ ( _C_ + 1) _/βi_ 


9: loss _←_ _N_ [1] - _i_ ���F1 _i −_ F1 _i_ ��

10: **if** loss _<_ best ~~l~~ oss **then**
11: best ~~l~~ oss _←_ loss; best _←_ ( _α, γ, β_ 0 _, C_ )
12: **end if**
13: **end for**
14: **end for**
15: **end for**
16: **end for**
17: **return** best _,_ best ~~l~~ oss


9: loss _←_ [1]


_N_ [1] 


**Reproducibility** All methods share the same ( _h, L_ ) grid with _h_ _∈_ _{_ 1 _,_ 2 _,_ 3 _,_ 4 _}_ and _L_ _∈_
_{_ 0 _._ 5k _,_ 1k _,_ 2k _,_ 4k _,_ 8k _,_ 10k _}_ . We expand this grid with meshgrid(indexing=‘‘ij’’) and flatten to length _N_ = 24 vectors for fitting. The default metric is MAE; alternative choices (e.g., MAPE
or weighted MAE) produce qualitatively similar trends. Parameter uncertainty can be assessed via
bootstrap resampling over ( _h, L_ ) pairs.


A.6 QWEN3-14B FITTING PARAMETERS


As shown in Figure 3, the plug-in bound provides an excellent global fit, as reflected in the low MAE
across all methods, and it captures method-level differences through the parameters ( _γ, C, β_ 0). CoT
and S-C expand the usable regime by increasing _C_ and reducing _γ_, thereby mitigating the cliff. S-A
incurs a large base-demand penalty ( _β_ 0), which counteracts the benefit of a higher _C_ . Removing
distractors nearly eliminates hop inflation ( _γ_ _≈_ 1), indicating that compounding arises primarily
from noise rather than depth. Taken together, these results empirically substantiate the _accuracy_
_cliff_ predicted by our theory.


Table 3: Fitted parameters of the plug-in accuracy bound (MAE minimization) of Qwen3-14B.
Larger _C_ indicates higher effective single-pass capacity; smaller _γ_ indicates weaker hop inflation.


Method _α_ _γ_ _β_ 0 _C_ MAE


Direct 0.0100 3.000 40 67.5 0.0963
CoT 0.0100 2.076 0 131 0.0320
S-C 0.00720 2.076 0 99.2 0.0273
S-R 0.0100 2.282 110 147 0.0531
ReAct 0.0100 1.768 80 115 0.0429
P&S 0.00268 1.974 20 35.8 0.0444
S-A 0.00518 1.974 160 162 0.0747
w/o D. 0.0100 1.050 70 67.5 0.0589


20


Table 4: Fitted parameters of the plug-in accuracy bound (MAE minimization) of Qwen3-8B. Larger
_C_ indicates higher effective single-pass capacity; smaller _γ_ indicates weaker hop inflation.


Method _α_ _γ_ _β_ 0 _C_ MAE


Direct 0.00720 3.000 10 20 0.1061
CoT 0.0100 2.076 60 131 0.0426
S-C 0.0100 2.076 60 131 0.0343
S-R 0.00518 2.076 50 51.7 0.0747
ReAct 0.00518 2.076 70 83.3 0.0465
P&S 0.0100 2.487 160 178 0.0480
S-A 0.00373 2.795 130 131 0.0475


A.7 EXPERIMENTAL RESULTS OF QWEN3-8B


**Theory fit for single-pass methods.** The plug-in accuracy bound in Eq. 7 fits Qwen3–8B’s singlepass baselines well (Table 3, Figure 6). _Direct_ exhibits a small effective per-pass capacity ( _C ≈_ 20)
and strong hop inflation ( _γ_ = 3 _._ 0), hence an early accuracy cliff. _CoT_ and _S-C_ attain a much larger
capacity ( _C_ _≈_ 131) with moderate hop inflation ( _γ ≈_ 2 _._ 08) and the lowest MAE (0.0426/0.0343),
so their empirical points hug the theoretical envelope longer. _P&S_ shows the largest _C_ ( _≈_ 178) but
also a large base demand ( _β_ 0 = 160) and higher _γ_ ( _≈_ 2 _._ 49), which offsets its capacity at greater
depth/length. _ReAct_ and _S–R_ have mid-range capacities ( _C_ _≈_ 83 _._ 3 and 51 _._ 7) and degrade earlier.
_S–A_ has sizable _β_ 0 (130) and high _γ_ ( _≈_ 2 _._ 80), reflecting method-specific overheads that accelerate
the cliff despite a decent _C_ . Overall, the fitted overlays confirm the accuracy-cliff picture: empirical
F1 follows the bound and collapses once the fitted demand _β_ crosses _C_ +1.


**Performance** **of** **InfoQA.** _1._ _Depth_ _robustness._ On 2–4 hops, InfoQA overall average is **0.74**
vs. **0.66** for S–C and **0.65** for CoT. By hop: at 2-hop, **0.89** (InfoQA) vs. 0.82 (S–C); at 3-hop,
**0.64** vs. 0.63 (S–C/ ReAct); at 4-hop, **0.68** vs. 0.52 (S–C). Gains grow with depth: at 4-hop and
long contexts (e.g., 8k), InfoQA reaches **0.67**, while the best single-pass baseline tops out around
**0.16** . This matches the theory: capacity-aware decomposition keeps each step’s demand _βk ≤_ _C_ . _2._
_Length robustness._ InfoQA maintains strong performance as context length increases. For 2-hop at
8k tokens, it achieves **0.74** (vs. 0.67 for S–A and 0.57 for S–C); for 3-hop at 10k tokens, it reaches
**0.28**, exceeding the best single-pass alternative (0.21 for S–C). Even at 1-hop, where all methods
are strong, InfoQA remains competitive (average **0.93** ) without relying on a single long reasoning
trace.


21


Table 5: Qwen3-8B’s Average F1 scores across different reasoning depths and context lengths.
We compare InfoQA with single-pass baselines: Chain-of-Thought (CoT), Self-Refine (S-R), SelfConsistency (S-C), ReAct, Plan-and-Solve (P&S), Self-Ask (S-A).


**Average F1 Score**


**Hops** **Context Length** **Direct** **CoT** **S-R** **S-C** **ReAct** **P&S** **S-A** **InfoQA**


**1**


**2**


**3**


**4**


0.5k 0.98 **1.00** 0.96 **1.00** **1.00** **1.00** 0.98 **1.00**
1k 0.98 **1.00** 0.94 **1.00** 0.98 **1.00** 0.98 **1.00**
2k 0.94 0.99 0.93 0.99 0.96 0.97 0.96 **1.00**
4k 0.91 **0.97** 0.81 **0.97** 0.93 0.90 0.91 **0.97**
8k 0.83 **0.91** 0.58 **0.91** 0.76 0.78 0.82 0.82
10k 0.76 0.87 0.49 **0.89** 0.74 0.68 0.71 0.78


0.5k 0.57 **1.00** 0.96 **1.00** 0.98 0.99 0.89 **1.00**
1k 0.44 **0.99** 0.88 **0.99** 0.96 0.97 0.77 0.96
2k 0.28 0.95 0.71 0.94 0.93 0.87 0.89 **0.98**
4k 0.23 0.93 0.43 0.95 0.49 0.78 0.78 **0.96**
8k 0.10 0.54 0.15 0.57 0.49 0.51 0.67 **0.74**
10k 0.10 0.47 0.56 0.48 0.44 0.42 0.66 **0.72**


0.5k 0.50 0.96 0.86 **0.97** 0.96 0.83 0.94 0.92
1k 0.25 0.89 0.64 0.91 0.88 0.69 0.84 **0.93**
2k 0.13 0.79 0.51 0.81 0.82 0.62 0.67 **0.83**
4k 0.09 0.60 0.27 0.58 **0.62** 0.42 0.51 0.61
8k 0.04 **0.32** 0.17 0.31 0.27 0.20 0.25 0.28
10k 0.02 0.20 0.10 0.21 0.15 0.12 0.10 **0.28**


0.5k 0.12 0.94 0.84 0.97 0.91 0.89 0.70 **1.00**
1k 0.09 0.84 0.72 0.88 0.77 0.72 0.63 **0.96**
2k 0.03 0.66 0.49 0.64 0.56 0.45 0.47 **0.70**
4k 0.01 0.44 0.29 0.40 0.32 0.28 0.32 **0.63**
8k 0.00 0.15 0.12 0.16 0.09 0.08 0.16 **0.67**
10k 0.00 0.09 0.05 0.09 0.03 0.04 0.10 **0.12**


**Overall Average (2–4 hop)** 0.17 0.65 0.49 0.66 0.59 0.55 0.58 **0.74**


**1 hop Average** 0.90 0.96 0.79 **0.96** 0.90 0.89 0.89 0.93
**2 hop Average** 0.29 0.81 0.62 0.82 0.72 0.76 0.78 **0.89**
**3 hop Average** 0.17 0.63 0.42 0.63 0.62 0.48 0.55 **0.64**
**4 hop Average** 0.04 0.52 0.42 0.52 0.45 0.41 0.40 **0.68**


**Context Average (2–4 hop)**


0.5k 0.40 0.97 0.89 **0.98** 0.95 0.90 0.84 0.97
1k 0.26 0.91 0.75 0.93 0.87 0.79 0.75 **0.95**
2k 0.15 0.80 0.57 0.80 0.77 0.65 0.68 **0.84**
4k 0.11 0.66 0.33 0.64 0.48 0.49 0.54 **0.73**
8k 0.05 0.34 0.15 0.35 0.28 0.26 0.36 **0.56**
10k 0.04 0.25 0.24 0.26 0.21 0.19 0.29 **0.37**


Fano-Style Bound (F1) Empirical (h=1) Empirical (h=2) Empirical (h=3) Empirical (h=4)


S-R


100 200 300 400 500


1.0


0.8


0.6


0.4


0.2


Direct


0.0
500 1000 1500 2000


S-C


CoT


200 400 600 800 1000

P&S


ReAct


200 400 600 800 1000 100 200 300 400 500


500 1000 1500


S-A


200 400 600 800 1000


x-axis: Effective Information Demand (fitted)


Figure 6: Qwen3-8B’s Empirical F1 vs. theoretical curves across single-pass methods. The x-axis
shows the estimated effective information demand ( _β_ ), fitted per method, and the y-axis shows the
F1 score.


22