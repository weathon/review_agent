## THINKING IN THINKING: MITIGATING DECEPTIVE ALIGNMENT VIA SELF-MONITORING


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Frontier reasoning models rely on chain-of-thought (CoT) reasoning to achieve
impressive performance, yet the same mechanism can amplify _deceptive alignment_,
situations in which a model appears aligned while covertly pursuing misaligned
goals. Existing safety pipelines treat deception as a black-box output to be filtered _post-hoc_, leaving the model free to scheme during its internal reasoning.
We ask: _**Can deception be intercepted**_ **while** _**the model is thinking?**_ We answer
this question, the first framework that embeds a SELF-MONITOR inside the CoT
process itself, named CoT Monitor +. During generation, the model produces (i)
ordinary reasoning steps and (ii) an internal self-evaluation signal trained to flag
and suppress misaligned strategies. The signal is used as an auxiliary reward in
reinforcement learning, creating a feedback loop that rewards honest reasoning and
discourages hidden goals. To study deceptive alignment systematically, we introduce DECEPTIONBENCH, a five-category benchmark that probes covert alignmentfaking, sycophancy, etc. We evaluate various LLMs and show that unrestricted
CoT roughly _aggravates_ the deceptive tendency. In contrast, CoT Monitor + cuts
deceptive behaviors by 43.8% on average while preserving task accuracy. Further,
when the self-monitor signal replaces an external weak judge in RL fine-tuning,
models exhibit substantially fewer obfuscated thoughts and retain transparency.
Our project website can be found at self-monitoring.github.io.


1 INTRODUCTION


**“You can’t wake a person who is pretending to be asleep”**


                                    - Navajo saying – Education


Frontier large language models (LLMs), also known as reasoning models (OpenAI, 2024a; Guo
et al., 2025), reason via chain-of-thought (CoT) (Wei et al., 2022), producing intermediate natural
language reasoning steps on the way to an answer. However, this reasoning capability raises alignment
concerns – whether a model’s goals and behaviors are aligned with human intentions (Ji et al., 2023).
Of particular concern is deceptive alignment (Meinke et al., 2024; Greenblatt et al., 2024b; Anwar
et al., 2024; Wen et al., 2024), a scenario in which a model that is misaligned with the intended
objective learns to behave _as if_ it were aligned, in order to avoid punishment or corrective training,
while _covertly_ pursuing its own agenda. Such deceptive behavior has been discussed theoretically
in the context of mesa-optimizers (Hubinger et al., 2019) and advanced ML systems (Bengio et al.,
2024; Pan et al., 2024b), and recent studies suggest it is not merely hypothetical. Notably, CoT
reasoning can _amplify_ the model’s ability to plan sophisticated deceptions (Baker et al., 2025a), since
the model can iteratively refine a deceptive plan in its internal reasoning before producing an output.


In this study, we aim to systematically explore technical methods for mitigating the risks of deceptive
alignment in LLMs. Traditional alignment techniques treat deceptive behaviors as black-box outputs
to be filtered or punished after the fact. **But** **what** **if** **the** **deception** **happens** **in** **thought,** **not** **in**
**response?** At the heart of this study lies a fundamental question:


_How can we_ _**intervene**_ _on deception_ _**during reasoning, not just after it?**_


1


Figure 1: **Illustration of deception.** AI deception occurs when intents conflict ( _**mesa utility function**_
and _**outer driven goal**_ ) cause a model to adopt secretly misaligned behaviors (e.g., during alignment
v.s. deployment (Greenblatt et al., 2024b)) while can not be monitored directly by its action.


Although deceptive behaviors in LLMs are being increasingly uncovered, they tend to be rare, longtailed, and often require carefully crafted prompts to elicit. As a result, the research community faces
persistent challenges in systematically analyzing such behaviors. In this study, we highlight three
pressing dilemmas that demand urgent attention:


(I) **How** **to** **Evaluate** **or** **Detect** **Deceptive** **Behavior?** Despite growing concerns over the
deceptive tendencies emerging in LLMs (Pan et al., 2020; 2024b;a; Greenblatt et al., 2024a),
there remains a lack of standardized benchmarks to assess such behaviors. Most existing
safety evaluations focus on common harms such as violence or offensive content (Ji et al.,
2024b; Bai et al., 2022), falling short in capturing more strategic, covert, and long-term
deceptive behaviors - those that arguably pose the most existential risks. Encouragingly,
recent efforts like DarkBench (Kran et al., 2025) and InstrumentalEval (He et al., 2025) have
begun to address these extreme risk behaviors in LLMs.


(II) **Treats** **deception** **as** **a** **black-box** **output,** **filtered** **or** **penalized** **post** **hoc.** Current approaches often regard deception as a black-box phenomenon, relying on post hoc filters
or penalties to suppress undesirable outputs without addressing the underlying reasoning
behind such behavior. “You can’t wake a person pretending to be asleep”, merely suppressing outputs is ineffective if the model is internally driven to deceive. Empirical studies
show that reasoning models are more effective at misleading targets than those limited to
direct responses, suggesting that advanced reasoning capabilities may inadvertently enhance
deceptive tendencies. As models become more capable, the risk of strategic misalignment
is expected to increase. Therefore, alignment efforts should go beyond penalizing harmful
outputs and focus on shaping the model’s internal evaluation of its actions.


(III) **How** **can** **we** **mitigate** **it?** **Pure** **RLHF** **Alone** **Is** **Not** **Enough.** Traditional alignment
techniques like RLHF have achieved some success in steering models toward preferred
behavior, but they might falter if a model learns to game the feedback process. In fact,
evidence is emerging that once a model has learned to behave deceptively, conventional
fine-tuning or reward-based techniques may fail to eliminate the deception. For instance,
recent research (Hubinger et al., 2024) found that LLMs could be trained to behave normally
under most conditions and act maliciously when a secret trigger is present; attempts to
remove this backdoor through supervised or adversarial training did not fully succeed, and
in some cases only taught the model to better hide the trigger-based behavior. As it stands,
purely RLHF-based techniques are inadequate for fully mitigating such risks.


Mitigating deceptive alignment is thus a pressing challenge in extreme AI safety (Bengio et al., 2024;
2025). A deceptively aligned model is especially dangerous because it can pass standard evaluations
by feigning obedience, only to exhibit critically misaligned behavior when unchecked. In response to
these challenges, researchers seek to make models more _transparent_ and _honest_ in their reasoning
processes (Turpin et al., 2023; Chen et al., 2025). A promising approach involves leveraging the
CoT process itself as an avenue for oversight. By inspecting or supervising a model’s intermediate
reasoning, we might detect misaligned plans _in the act_ . OpenAI observes that reading a model’s CoT
process can reveal deceptive reasoning, and uses a separate oversight model - a CoT monitor - to flag
such problematic reasoning (Baker et al., 2025b). However, a critical concern is that if models are


2


directly penalized for showing deceptive thoughts, they might hide those thoughts rather than truly
abandoning the deception.


In this work, we introduce CoT Monitor +, a framework that **integrates** **the** **SELF-MONITOR**
**mechanism into the model’s reasoning process** to mitigate deceptive behavior. The core idea is to
let the model generate a CoT while simultaneously evaluating it internally for signs of misalignment,
i.e., _thinking before thinking_ . Instead of depending solely on an external judge or separate oversight
model, CoT Monitor + empowers the model to actively participate in its own alignment oversight.
Rather than relying solely on an external judge or a separate oversight model, CoT Monitor + makes
the model an active participant in its alignment oversight. By training the model to generate a
specialized self-monitoring signal alongside its reasoning, we aim to achieve two goals: (1) **detect**
**deceptive strategies as they emerge**, and (2) **discourage the model from pursuing such strategies**
**before response**, as self-monitoring fosters an internal feedback loop that promotes honesty.


To our knowledge, this is the first framework to blend CoT reasoning with a built-in self-check for
deceptive alignment. Overall, our contributions are summarized as follows:


  - **Language models can self-monitor their reasoning processes** (Section 3). Building on this
insight, we introduce SELF-MONITOR, a framework that enables models to oversee their own CoT
reasoning within predefined safety protocols. The experimental results demonstrate that SELFMONITOR substantially reduces deceptive alignment behaviors, achieving a 43.8% reduction
compared to pure CoT reasoning.

  - **Detect deceptive tendency in reasoning models with** **DECEPTIONBENCH** (Section 4). To
mitigate the deceptive risks posed by advanced LLMs, we introduce DeceptionBench, the
first systematic benchmark designed to assess deceptive behaviors behind reasoning models.
DeceptionBench assesses model behavior across five categories, offering insights into when and
how models may superficially appear aligned while internally pursuing misaligned goals.

  - **SELF-MONITOR serve as reward signals in RL training** (Section 5). Relying solely on an
external weak model’s CoT monitor as a reward signal for RL training can lead the model to
strategically suppress its true intentions during the CoT process, resulting in more sophisticated
forms of deception. In contrast, using the model’s own SELF-MONITOR feedback as the training
signal encourages more faithful reasoning and reduces the incentive to conceal deceptive thoughts.


2 PRELIMINARIES


We consider an LLM parameterized by _θ_ . The generation process of the LLM is defined by mapping
an input context _X_ to the output _Y_ by _p_ ( _·|·_ ; _**θ**_ ). Reasoning models explicitly model the reasoning
process. Rather than producing a one-shot output, reasoning models generate an intermediate
sequence _Z_ including both reasoning steps _R_ and final answer _Y_ . Since the autoregressive language
model generates tokens sequentially for a given position by relying solely on the previously generated
token sequence, the model can be conceptualized as a Markov decision process (MDP) (Puterman,
2014), wherein the conditional probability _p_ ( _Z_ _| X_ ; _**θ**_ ) is defined as,


where _Z_ = _R ⊕Y_ = ( _z_ 1 _, . . ., zm_ ), _zi_ represents individual token from a predetermined vocabulary.


**Supervised Fine-Tuning.** SFT fine-tunes the pre-trained model using datasets sampled from a highquality distribution, aligning the model with target tasks. For a SFT dataset _D_ SFT = _{_ ( _X_ [(] _[i]_ [)] _, Z_ [(] _[i]_ [)] ) _}_ _[N]_ _i_ =1 [,]
SFT aims to minimize the negative log-likelihood loss:
_L_ SFT( _**θ**_ ; _D_ SFT) = _−_ E( _X_ _,Z_ ) _∼D_ SFT log _p_ ( _Z|X_ ; _**θ**_ )


**Reward Modeling.** Rewards serve as the training signals in RL (Sutton et al., 1998). Preferencebased (Ouyang et al., 2022) and rule-based (Guo et al., 2025) reward modeling are two representative approaches. In preference-based reward modeling, for a preference dataset _DR_ =
_{_ ( _X_ [(] _[i]_ [)] _, Zw_ [(] _[i]_ [)] _[,][ Z]_ _l_ [(] _[i]_ [)][)] _[}]_ _i_ _[N]_ =1 [sampled] [from] [human] [preference] [distribution] _[p][∗]_ [,] [we] [estimate] [the] [reward]
model _rϕ_ ( _Z, X_ ) parameters via maximum likelihood. The negative log-likelihood loss is defined as,
_L_ RM( _ϕ_ ; _DR_ ) = _−_ E( _X_ _,Zw,Zl_ ) _∼DR_ [log _σ_ ( _rϕ_ ( _Zw, X_ ) _−_ _rϕ_ ( _Zl, X_ ))] _,_


3


_p_ ( _Z_ _| X_ ; _**θ**_ ) = _p_ ( _R, Y_ _| X_ ; _**θ**_ ) =


_m_

- _p_ ( _zt_ _| z<t, X_ ; _**θ**_ ) _,_


_t_ =1


Figure 2: **(Left)** **How** **Self-Monitor** **works?** Unlike the standard LLMs reasoning process, the
SELF-MONITOR model first detects deceptive or harmful patterns in its CoT, and then generates a
safer response based on the monitor evaluation. In reinforcement learning, the reward signal from
the self-monitor’s CoT detection is combined with a standard action monitor to optimize the model.
**(Right)** **How** **DeceptionBench** **evaluates** **deceptive** **tendency** **of** **models:** Neutral prompts and
specifically constructed prompts with an **outer goal** are used to elicit the model’s **mesa utility** and
CoT-Action pairs, respectively. These pairs are then evaluated for consistency and deceptiveness.


where _Zw_ and _Zl_ denote the preferred and less preferred responses, respectively. In contrast, rulebased reward methods include two types of rewards: accuracy reward, which evaluates whether the
answer _Y_ satisfies specific criteria, and format reward, which requires the model’s reasoning steps _R_
to be placed between the <think> and </think> tags.


**RL** **Fine-Tuning.** In the RL phase, the reward function serves as a training signal to guide the
language model. The optimisation objective is defined as follows:


max _**θ**_ E _X∼DR,Z∼p_ ( _Z|X_ ; _**θ**_ ) [ _rϕ_ ( _X_ _, Z_ )] _−_ _β_ DKL [ _p_ ( _Z|X_ ; _**θ**_ ) _||p_ ( _Z|X_ ; _**θ**_ ref)] _,_


where _**θ**_ ref represents the parameters of a reference policy, typically initialised with the SFT model
and _β_ is a parameter controlling the deviation from the base reference policy (Rafailov et al., 2023).


**Constrained MDP.** A Constrained MDP (CMDP) addresses sequential decision-making problems
under specific constraints (Altman, 2021a), which can be viewed as a constrained optimization
problem (COP) over a set of admissible policies (Rossi et al., 2006). A CMDP is formalized as
a tuple ( _S, A, P, R, {Ci, bi}_ _[m]_ _i_ =1 _[, γ]_ [)][,] [where] _[R]_ [denotes the reward function,] _[C][i]_ [represents the cost]
functions, and _bi_ is the cost thresholds for _m_ constraints. The objective of a CMDP is to find the
optimal feasible policy such that:


_π_ _[∗]_ = min where Π _C_ = _∩_ _[m]_ _i_ =1 _[{][π]_ _**[θ]**_ _[∈]_ [Π] _[|J][C]_ _i_ [(] _[π]_ _**[θ]**_ [)] _[ ≤]_ _[b][i][}][.]_
_π_ _**θ**_ _∈_ Π _C_ _[J][R]_ [(] _[π]_ _**[θ]**_ [)] _[,]_

Here, _JR_ ( _π_ _**θ**_ ) = E _π_ _**θ**_ [ [�] _[∞]_ _t_ =0 _[γ][t][R]_ [(] _**[s]**_ _[t][,]_ _**[ a]**_ _[t]_ [)]][ and] _[ J][C]_ _i_ [(] _[π]_ _**[θ]**_ [) =][ E] _[π]_ _**θ**_ [[][�] _[∞]_ _t_ =0 _[γ][t][C][i]_ [(] _**[s]**_ _[t][,]_ _**[ a]**_ _[t]_ [)]][.]


3 COT MONITOR+ FRAMEWORK


In this section, we introduce CoT Monitor +, a framework that fuses a _self-monitoring_ mechanism
into CoT reasoning. We first formalize deceptive alignment (§3.1), then show how a model can learn
to audit its own thoughts (§3.2), and finally integrate this signal into RL training (§3.3).


3.1 PROBLEM SETTING: THE DECEPTIVE BEHAVIOR OF MODELS


We consider the sequential generation decision process of the reasoning model, which can be naturally
formalized as a MDP without reward (MDP _\_ R) (Puterman, 2014; Skalse et al., 2022). The formal
definition is as follows.


**Definition 3.1** (MDP _\_ R) **.** The sequence generation process of the reasoning model is MDP _\_ R. The
process is described as a tuple _< S, A, P,_ _ _, P_ 0 _, γ_ _>_, where _S_ is a set of states, each _**s**_ _∈S_ reflecting


4


the model’s status after a generation step. _A_ denotes the set of actions, each _**a**_ _∈A_ corresponds to a
generation action. _P_ : _S × A →S_ is a conditional transition distribution between states. _P_ 0 is the
distribution over initial state, determined by the input request _X_ . _γ_ _∈_ [0 _,_ 1] is the discount factor.


A parameterized policy _π_ _**θ**_ is a distribution over actions given states, which governs the generation
process of the reasoning model. A trajectory _Z_ is a sequence ( _**s**_ 0 _,_ _**a**_ 0 _, · · ·_ ) following the dynamics
induced by _P_, representing a possible generation path of the model. As the generation process of
the reasoning model adheres to its inherent objective (Hadfield-Menell et al., 2016; Lidayan et al.,
2024; Abdulhai et al., 2024), we assume that the model operates under a utility function _U_ ( _·_ ) during
generation, with its policy _π_ _**θ**_ determined by the utility function.
**Assumption 3.2** (The Utility Function and _π_ _**θ**_ ) **.** Let the utility function of a reasoning model be
defined as _U_ ( _·_ ) : _S_ _→_ R. We assume that its generation policy _π_ _**θ**_ satisfies:
_π_ _**θ**_ ( _**s**_ ) = arg max _**a**_ _∈A_ [E] _**[s]**_ _[′][∼P]_ [(] _[·|]_ _**[s]**_ _[,]_ _**[a]**_ [)] _[U]_ [(] _**[s]**_ _[′]_ [)] _[.]_


When a reasoning model generates outputs based on the request _X_, the user expects the model to
make decisions according to a specified outer goal (Hadfield-Menell et al., 2016; Rahwan et al.,
2019; Hadfield-Menell et al., 2017). We define the utility function corresponding to the **outer**
**goal** as _Uo_, while the reasoning model’s **mesa utility** as _Um_ . The generation process of the model
under outer goals is regarded as decision-making using a joint utility function _U_ = _F_ ( _Um, Uo_ ),
where _F_ : R _[S]_ _×_ R _[S]_ _→_ R _[S]_ is a fusion function of different utility functions, satisfing _F_ ( _U_ _[′]_ _, U_ _[′]_ ) =
_U_ _[′]_ _, ∀U_ _[′]_ _∈_ R _[S]_ . Hence, the model’s deception behavior is defined as:
**Definition 3.3** (The Deceptive Behavior of LLMs) **.** Given a **outer goal** utility function _U_ **o** and a
reasoning model **mesa utility** function _U_ **m** relative to the environment _< S, A, P,_ _ _, P_ 0 _, γ_ _>_, we say
that the deception occurs if there exist a state _**s**_ _∈S_ and _**a**_ _[′]_ _∈A_ satisfies
E _**s**_ _′∼P_ ( _·|_ _**s**_ _,_ _**a**_ _′_ ) _U_ ( _**s**_ _[′]_ ) _>_ E _**s**_ _′∼P_ ( _·|_ _**s**_ _,_ _**a**_ _∗_ ) _U_ ( _**s**_ _[′]_ ) _,_
where _**a**_ _[∗]_ = arg max _**a**_ _∈A_ E _**s**_ _′∼P_ ( _·|_ _**s**_ _,_ _**a**_ ) _Um_ ( _**s**_ _[′]_ ) and _U_ = _F_ ( _Um, Uo_ ). In other words, deception
occurs when the action selected by the reasoning model’s policy _π_ _**θ**_ under its **mesa utility** differs
from that under the **outer goal** . Conversely, if the policy _π_ _**θ**_ consistently generates the same trajectory
for both utilities, the model is aligned with the human outer goals.


3.2 SELF-MONITOR: LEARNING TO THINKING, MONITORING, AND REASONING


In our framework, we tackle a core challenge: how to enable the model to generate SELF-MONITOR
signals alongside its CoT reasoning, allowing it to actively detect when its reasoning may be deceptive.


**Data** **Generation.** To support training this mechanism, we construct a dataset that pairs each
reasoning process with a corresponding self-monitoring signal. This dataset is built via a two-stage
pipeline that combines automatic annotation with human verification:


_�→_ **Reasoning Path Collection.** We first gather a diverse set of CoT reasoning trajectories _R_
generated by the reasoning model based on each user’s query _X_ .
_�→_ **Self-Monitor Annotation.** For each trajectory _R_, we use a verifier model to generate a SELFMONITOR comment _M_, each includes: a risk assessment indicating whether _R_ exhibits unsafe
reasoning behaviors; a natural language explanation justifying this assessment. Formally, the
CoT monitor _M_ is produced under a safety protocol _P_ safe: _M_ = Monitor( _R_ ; _P_ safe).
_�→_ **Safer** **Response** **Generation.** After obtaining the CoT and self-monitor of the model, we
generate the final answer _Y_ based on query _X_, _R_ and _M_, without deceptive or harmful intent.
_�→_ **Human Verification and Assurance.** Finally, human annotators review and refine the autogenerated labels and explanations. They correct errors and enrich the annotations to better
capture subtle or complex instances of unsafe reasoning often missed by automated methods.


As illustrated in Figure 2, we follow the annotation pipeline, resulting a SELF-MONITOR dataset
_D_ = ( _X_ [(] _[i]_ [)] _, R_ [(] _[i]_ [)] _, M_ [(] _[i]_ [)] _, Y_ [(] _[i]_ [)] ) _Ni_ =1 [includes, for each sample:] [(1) The user’s query] _[ X]_ [ (] _[i]_ [)][; (2) A CoT]
trajectory _R_ [(] _[i]_ [)] ; (3) A self-monitoring sequence _M_ [(] _[i]_ [)] auditing _R_ [(] _[i]_ [)] ; (4) The final output _Y_ [(] _[i]_ [)] .


**Elicit Models to Self-Monitor.** We fine-tune the model to jointly generate its intermediate CoT
reasoning and a corresponding SELF-MONITOR statement. Given the curated dataset, we minimise,
_L_ ( _**θ**_ ; _D_ ) = _−_ E( _X_ _,R,M,Y_ ) _∼D_ �log _π_ _**θ**_ ( _Y, M, R | X_ )� _._ (1)


5


|DTR(Ours) GPT-4o Claude-3.7-Sonnet Gemini-2.5-Pro|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|Col22|Col23|Col24|Col25|Col26|Col27|Col28|Col29|Col30|Col31|Col32|Col33|Col34|Col35|Col36|Col37|Col38|Col39|Col40|Col41|Col42|Col43|Col44|Col45|Col46|Col47|Col48|Col49|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||
|Ali<br>u<br> <br>|gn<br>re<br> L<br>an|me<br> 3<br> L<br> a|nt F<br> : <br> M-<br> gr|aki<br> <br>**C**<br> a<br> e|ng<br>Stra<br>**o**<br> s-<br> em|teg<br>**m**<br> Ju<br> e|ic<br>**pa**<br> d<br> nt|Dec<br>**ri**<br> ge<br> r|ep<br>**so**<br> i<br> at|tion<br>H<br>**n**<br> n<br> e a|<br>one<br>** o**<br>  de<br>  n|sty<br>** f**<br>  c<br>  d|Ev<br>** ag**<br>  ep<br>  Ph|asio<br>** r**<br>  ti<br>  i|n<br><br>** ee**<br>  on<br>   co|San<br>** m**<br>  e<br>   e|db<br>** e**<br>  va<br>   ff|agg<br>** nt**<br>  l<br>   ci|ing<br>**  w**<br>  ua<br>   en|**  it**<br>  ti<br>   t|Sy<br>**  h**<br>  on<br>   (M|cop<br>**  h**<br>  . <br>a|han<br>**  u**<br>T<br>tt|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|cy<br>**  m**<br>he<br>he|


3.3 REINFORCEMENT LEARNING WITH SELF-MONITOR REWARD


While RL has been widely applied to align LLMs, applying RL to reasoning models introduces
distinct risks, notably the risk of _deceptive alignment_ (Greenblatt et al., 2024a; Baker et al., 2025b).


**Reinforcement Learning but reinforces the model’s deceptive tendency.** Outcome-based RL
training reward signal provided by an action monitor _Ma_ is limited, as it only assesses final outputs
while ignoring intermediate CoT. _This allows model to develop misaligned CoT while final output_
_still_ _complying_ _Ma_ _(Greenblatt_ _et_ _al.,_ _2024b)._ An intuitive method is to introduce an outer CoT
monitor _Mo_, evaluating the model’s CoT for detecting deception alongside monitoring action. _The_
_model can learn to generate CoT that comply to Mo_ _but hides deceptive intent,_ _undermining the_
_CoT’s transparency (Baker et al., 2025a)._


**How to Solve?** **RL with Self-Monitor Reward:** **Integrating Intrinsic Oversight.** We propose a
self-monitoring RL approach that directly integrates intrinsic oversight into the optimization process.
This involves an **action monitor** _Ma_, which evaluates answer _Y_ based on criteria such as safety
and helpfulness, providing a reward _Ra_ ( _Z, X_ ). Concurrently, a **CoT self-monitor** _Ms_ internally
assesses the reasoning steps _R_ to detect deceptive patterns, assigning a reward _Rs_ ( _Z, X_ ), which is 0
if no deceptive behavior is identified, and _r_ (where _r_ _<_ 0) otherwise. To ensure that the reasoning
model avoids deceptive behaviors while optimizing for specific objectives, the training process of
the parameterized model _π_ _**θ**_ can be formulated as a constrained MDP. Consequently, the training
objective of RL with self-monitor reward is defined as,

max _**θ**_ E _X∼D,Z∼π_ _**θ**_ ( _·|X_ )[ _Ra_ ( _Z, X_ )] _,_ s.t. _Rs_ ( _Z, X_ ) _≥_ 0 _,_ _∀X_ _∼D, Z_ _∼_ _π_ _**θ**_ ( _·|X_ ) _._ (2)


To solve this COP, we adopt the Lagrangian method, which seeks the extrema of a function over a
constrained set. Hence, Equation 2 can be converted into its unconstrained Lagrangian dual problem,


min (3)
_**θ**_ [max] _λ≥_ 0 [[] _[−J][a]_ [(] _**[θ]**_ [)] _[ −]_ _[λ][ · J][s]_ [(] _**[θ]**_ [)]] _[,]_


where _λ ≥_ 0 serves as the Largrange multiplier and _Ja_ ( _**θ**_ ), _Js_ ( _**θ**_ ) stands for:


_Ja_ ( _**θ**_ ) ≜ E _X∼D,Z∼π_ _**θ**_ ( _·|X_ )[ _Ra_ ( _Z, X_ )] _,_ _Js_ ( _**θ**_ ) ≜ E _X∼D,Z∼π_ _**θ**_ ( _·|X_ )[ _Rs_ ( _Z, X_ )] _._ (4)


It is worth noting that we employ a self-monitoring mechanism rather than an external monitor to
detect the model’s deceptive tendencies, while preserving CoT’s transparency (Baker et al., 2025b).
We solve the minimax problem in Equation 3 iteratively by alternately updating the reasoning model
parameters _**θ**_ and the Lagrange multiplier _λ_, thereby preventing the over-optimization of a single
objective. The existence of the optimal policy will be proved in Appendix B.


4 DECEPTIONBENCH: HOW TO DETECT DECEPTIVE TENDENCY


In this section, we introduce DECEPTIONBENCH to assess LLMs deceptive tendency by evaluating
consistency between utility function _U_ and model’s behaviors.


4.1 BENCHMARK CONSTRUCTION


We consider that LLMs have inherent **mesa utilities** _Um_ and design **outer goals** _Uo_ to implicitly
incentivize deviation from these mesa utilities. We deliberately employ manual curation, which is


6


0.9


0.8


0.7


0.6


0.5


Figure 4: **The deception tendency of API-based and open-source models in DECEPTIONBENCH.**


essential to capture the nuance, plausibility, and targeted nature required for deception detection.
DECEPTIONBENCH is meticulously developed with 180 scenarios inspired by real-world interactions,
strictly adhering to a predefined typology of LLM deception in five categories: (1). **Sycophancy**
(Kran et al., 2025; Perez et al., 2022): Misrepresenting and manipulating to align with perceived
user preferences. (2). **Strategic Deception** (Hagendorff, 2024; Wu et al., 2025): Violating rules or
manipulating information for goal achievement. (3). **Honesty Evasion** (Ren et al., 2025): Avoiding
honest and truthful answers under pressure or outer goal. (4). **Alignment Faking** (Greenblatt et al.,
2024a): Complying in alignment while regaining when not being monitored. (5). **Sandbagging**
(van der Weij et al., 2024): Deliberately underperforming or hiding capabilities for outer goal.


4.2 BENCHMARK EVALUATION


Evaluating these deceptive behaviors with LLM-as-Judge (Zheng et al., 2023) has low agreement
with human annotation, as shown in Figure 3. Therefore, by comparing the **mesa utility** _Um_ with
the reasoning CoT _R_ and final action _Y_ generated by _U_ = _F_ ( _Um, Uo_ ) (where _Uo_ is the **outer goal**
utility function), we aim to capture the model’s deception tendency in a more fine-grained manner.


The evaluation pipeline of DECEPTIONBENCH shown in Figure 2 is specifically divided into three
stages: First, we use the neutral **mesa_utility_prompt** to elicit the model’s _Um_ . Second, the
model receives a **system_prompt** to establish the specific external scenario context, followed
by an **outer_goal_prompt** intended to elicit its CoT _R_ and action _Y_ under _Uo_ . Finally, the
recorded _Um_, _R_, and _Y_ are compared by GPT-4o (OpenAI, 2024b) to detect inconsistencies. Since
deception arises when a model’s reasoning _R_ and its mesa utility _Um_ are consistent (denoted as
_R ∼Um_ ) while its final answer _Y_ diverges from _Um_ (denoted as _Y_ _̸∼Um_ ), we define the **D** eception
**T** endency **R** ate ( **DTR** ) as the fraction of instances exhibiting the deceptive pattern:


where _N_ is the number of instances. DTR offers a more reliable deception signal than LLM-as-Judge
evaluations (Zheng et al., 2023), aligning better with human judgments, as shown in Figure 3.


7


_ρ_ DTR = [1]

_N_


_N_


1 ( _R_ [(] _[i]_ [)] _∼Um_ [(] _[i]_ [)] _[∧Y]_ [(] _[i]_ [)] _[̸∼U]_ _m_ [(] _[i]_ [)][)] _[,]_ (5)
_i_ =1


Table 1: **Performance** **of** **SELF-MONITOR** **models.** SELF-MONITOR significantly mitigates
deceptive tendency compared to the original model, while presenting robust performance in various
challenging safety evaluation. We prompt LLMs to elicit CoT, denoted as “ _w CoT_ ”.

|Models|DeceptionBench Prefill StrongReject (Jailbreak Scr. ↓) BeaverTails (WR ↑)<br>( DTR ↓) ( ASR ↓) PAP-A PAP-M PAP-L PAIR Helpful Harmless|Col3|Col4|Col5|
|---|---|---|---|---|
|**Llama-3.1-8B-Instruct**<br>-<br>0.750<br>0.448<br>0.526<br>0.385<br>0.510<br>0.323<br>0.242<br>_Base_**_ w_**_ CoT_<br>0.374<br>0.820<br>0.357<br>0.266<br>0.291<br>0.482<br>0.323<br>0.313<br>_Safe SFT_<br>-<br>0.560<br>0.314<br>0.244<br>0.365<br>0.561<br>0.455<br>0.404<br>_Safe SFT_**_ w_**_ CoT_<br>0.350<br>0.640<br>0.330<br>0.274<br>0.346<br>0.521<br>0.556<br>0.363<br>_Safe CoT SFT_<br>0.246<br>0.230<br>0.394<br>0.408<br>0.407<br>0.557<br>0.596<br>0.202<br>**Llama-3.1-8B-R1-Distll**<br>0.222<br>0.410<br>0.658<br>0.726<br>0.676<br>0.626<br>0.323<br>0.182<br>**_w_**_ STAR-1_<br>0.222<br>0.140<br>0.429<br>0.386<br>0.479<br>0.434<br>0.030<br>0.495<br>**_w_**_ SafeChain_<br>0.254<br>0.240<br>0.634<br>0.700<br>0.671<br>0.495<br>0.253<br>0.212|**Llama-3.1-8B-Instruct**<br>-<br>0.750<br>0.448<br>0.526<br>0.385<br>0.510<br>0.323<br>0.242<br>_Base_**_ w_**_ CoT_<br>0.374<br>0.820<br>0.357<br>0.266<br>0.291<br>0.482<br>0.323<br>0.313<br>_Safe SFT_<br>-<br>0.560<br>0.314<br>0.244<br>0.365<br>0.561<br>0.455<br>0.404<br>_Safe SFT_**_ w_**_ CoT_<br>0.350<br>0.640<br>0.330<br>0.274<br>0.346<br>0.521<br>0.556<br>0.363<br>_Safe CoT SFT_<br>0.246<br>0.230<br>0.394<br>0.408<br>0.407<br>0.557<br>0.596<br>0.202<br>**Llama-3.1-8B-R1-Distll**<br>0.222<br>0.410<br>0.658<br>0.726<br>0.676<br>0.626<br>0.323<br>0.182<br>**_w_**_ STAR-1_<br>0.222<br>0.140<br>0.429<br>0.386<br>0.479<br>0.434<br>0.030<br>0.495<br>**_w_**_ SafeChain_<br>0.254<br>0.240<br>0.634<br>0.700<br>0.671<br>0.495<br>0.253<br>0.212|**Llama-3.1-8B-Instruct**<br>-<br>0.750<br>0.448<br>0.526<br>0.385<br>0.510<br>0.323<br>0.242<br>_Base_**_ w_**_ CoT_<br>0.374<br>0.820<br>0.357<br>0.266<br>0.291<br>0.482<br>0.323<br>0.313<br>_Safe SFT_<br>-<br>0.560<br>0.314<br>0.244<br>0.365<br>0.561<br>0.455<br>0.404<br>_Safe SFT_**_ w_**_ CoT_<br>0.350<br>0.640<br>0.330<br>0.274<br>0.346<br>0.521<br>0.556<br>0.363<br>_Safe CoT SFT_<br>0.246<br>0.230<br>0.394<br>0.408<br>0.407<br>0.557<br>0.596<br>0.202<br>**Llama-3.1-8B-R1-Distll**<br>0.222<br>0.410<br>0.658<br>0.726<br>0.676<br>0.626<br>0.323<br>0.182<br>**_w_**_ STAR-1_<br>0.222<br>0.140<br>0.429<br>0.386<br>0.479<br>0.434<br>0.030<br>0.495<br>**_w_**_ SafeChain_<br>0.254<br>0.240<br>0.634<br>0.700<br>0.671<br>0.495<br>0.253<br>0.212|**Llama-3.1-8B-Instruct**<br>-<br>0.750<br>0.448<br>0.526<br>0.385<br>0.510<br>0.323<br>0.242<br>_Base_**_ w_**_ CoT_<br>0.374<br>0.820<br>0.357<br>0.266<br>0.291<br>0.482<br>0.323<br>0.313<br>_Safe SFT_<br>-<br>0.560<br>0.314<br>0.244<br>0.365<br>0.561<br>0.455<br>0.404<br>_Safe SFT_**_ w_**_ CoT_<br>0.350<br>0.640<br>0.330<br>0.274<br>0.346<br>0.521<br>0.556<br>0.363<br>_Safe CoT SFT_<br>0.246<br>0.230<br>0.394<br>0.408<br>0.407<br>0.557<br>0.596<br>0.202<br>**Llama-3.1-8B-R1-Distll**<br>0.222<br>0.410<br>0.658<br>0.726<br>0.676<br>0.626<br>0.323<br>0.182<br>**_w_**_ STAR-1_<br>0.222<br>0.140<br>0.429<br>0.386<br>0.479<br>0.434<br>0.030<br>0.495<br>**_w_**_ SafeChain_<br>0.254<br>0.240<br>0.634<br>0.700<br>0.671<br>0.495<br>0.253<br>0.212|**Llama-3.1-8B-Instruct**<br>-<br>0.750<br>0.448<br>0.526<br>0.385<br>0.510<br>0.323<br>0.242<br>_Base_**_ w_**_ CoT_<br>0.374<br>0.820<br>0.357<br>0.266<br>0.291<br>0.482<br>0.323<br>0.313<br>_Safe SFT_<br>-<br>0.560<br>0.314<br>0.244<br>0.365<br>0.561<br>0.455<br>0.404<br>_Safe SFT_**_ w_**_ CoT_<br>0.350<br>0.640<br>0.330<br>0.274<br>0.346<br>0.521<br>0.556<br>0.363<br>_Safe CoT SFT_<br>0.246<br>0.230<br>0.394<br>0.408<br>0.407<br>0.557<br>0.596<br>0.202<br>**Llama-3.1-8B-R1-Distll**<br>0.222<br>0.410<br>0.658<br>0.726<br>0.676<br>0.626<br>0.323<br>0.182<br>**_w_**_ STAR-1_<br>0.222<br>0.140<br>0.429<br>0.386<br>0.479<br>0.434<br>0.030<br>0.495<br>**_w_**_ SafeChain_<br>0.254<br>0.240<br>0.634<br>0.700<br>0.671<br>0.495<br>0.253<br>0.212|
|**SELF-MONITOR-8B**|**0.206**|**0.050**|**0.216**<br>**0.175**<br>**0.200**<br>**0.248**|**0.500**<br>**0.500**|
|**Qwen2.5-7B-Instruct**<br>-<br>0.740<br>0.620<br>0.690<br>0.668<br>0.654<br>0.370<br>0.413<br>_Base_**_ w_**_ CoT_<br>0.272<br>0.770<br>0.685<br>0.702<br>0.688<br>0.647<br>0.500<br>0.348<br>_Safe SFT_<br>-<br>0.640<br>0.441<br>0.395<br>0.470<br>0.413<br>0.500<br>0.337<br>_Safe SFT_**_ w_**_ CoT_<br>0.283<br>0.630<br>0.459<br>0.415<br>0.515<br>0.411<br>0.587<br>0.293<br>_Safe CoT SFT_<br>0.261<br>0.140<br>0.407<br>0.403<br>0.432<br>0.412<br>0.489<br>0.261<br>**Qwen2.5-7B-R1-Distll**<br>0.246<br>0.640<br>0.622<br>0.654<br>0.624<br>0.600<br>0.196<br>0.065<br>**_w_**_ STAR-1_<br>0.207<br>0.540<br>0.430<br>0.423<br>0.387<br>0.487<br>0.022<br>0.424<br>**_w_**_ SafeChain_<br>0.272<br>0.500<br>0.605<br>0.629<br>0.603<br>0.460<br>0.152<br>0.152|**Qwen2.5-7B-Instruct**<br>-<br>0.740<br>0.620<br>0.690<br>0.668<br>0.654<br>0.370<br>0.413<br>_Base_**_ w_**_ CoT_<br>0.272<br>0.770<br>0.685<br>0.702<br>0.688<br>0.647<br>0.500<br>0.348<br>_Safe SFT_<br>-<br>0.640<br>0.441<br>0.395<br>0.470<br>0.413<br>0.500<br>0.337<br>_Safe SFT_**_ w_**_ CoT_<br>0.283<br>0.630<br>0.459<br>0.415<br>0.515<br>0.411<br>0.587<br>0.293<br>_Safe CoT SFT_<br>0.261<br>0.140<br>0.407<br>0.403<br>0.432<br>0.412<br>0.489<br>0.261<br>**Qwen2.5-7B-R1-Distll**<br>0.246<br>0.640<br>0.622<br>0.654<br>0.624<br>0.600<br>0.196<br>0.065<br>**_w_**_ STAR-1_<br>0.207<br>0.540<br>0.430<br>0.423<br>0.387<br>0.487<br>0.022<br>0.424<br>**_w_**_ SafeChain_<br>0.272<br>0.500<br>0.605<br>0.629<br>0.603<br>0.460<br>0.152<br>0.152|**Qwen2.5-7B-Instruct**<br>-<br>0.740<br>0.620<br>0.690<br>0.668<br>0.654<br>0.370<br>0.413<br>_Base_**_ w_**_ CoT_<br>0.272<br>0.770<br>0.685<br>0.702<br>0.688<br>0.647<br>0.500<br>0.348<br>_Safe SFT_<br>-<br>0.640<br>0.441<br>0.395<br>0.470<br>0.413<br>0.500<br>0.337<br>_Safe SFT_**_ w_**_ CoT_<br>0.283<br>0.630<br>0.459<br>0.415<br>0.515<br>0.411<br>0.587<br>0.293<br>_Safe CoT SFT_<br>0.261<br>0.140<br>0.407<br>0.403<br>0.432<br>0.412<br>0.489<br>0.261<br>**Qwen2.5-7B-R1-Distll**<br>0.246<br>0.640<br>0.622<br>0.654<br>0.624<br>0.600<br>0.196<br>0.065<br>**_w_**_ STAR-1_<br>0.207<br>0.540<br>0.430<br>0.423<br>0.387<br>0.487<br>0.022<br>0.424<br>**_w_**_ SafeChain_<br>0.272<br>0.500<br>0.605<br>0.629<br>0.603<br>0.460<br>0.152<br>0.152|**Qwen2.5-7B-Instruct**<br>-<br>0.740<br>0.620<br>0.690<br>0.668<br>0.654<br>0.370<br>0.413<br>_Base_**_ w_**_ CoT_<br>0.272<br>0.770<br>0.685<br>0.702<br>0.688<br>0.647<br>0.500<br>0.348<br>_Safe SFT_<br>-<br>0.640<br>0.441<br>0.395<br>0.470<br>0.413<br>0.500<br>0.337<br>_Safe SFT_**_ w_**_ CoT_<br>0.283<br>0.630<br>0.459<br>0.415<br>0.515<br>0.411<br>0.587<br>0.293<br>_Safe CoT SFT_<br>0.261<br>0.140<br>0.407<br>0.403<br>0.432<br>0.412<br>0.489<br>0.261<br>**Qwen2.5-7B-R1-Distll**<br>0.246<br>0.640<br>0.622<br>0.654<br>0.624<br>0.600<br>0.196<br>0.065<br>**_w_**_ STAR-1_<br>0.207<br>0.540<br>0.430<br>0.423<br>0.387<br>0.487<br>0.022<br>0.424<br>**_w_**_ SafeChain_<br>0.272<br>0.500<br>0.605<br>0.629<br>0.603<br>0.460<br>0.152<br>0.152|**Qwen2.5-7B-Instruct**<br>-<br>0.740<br>0.620<br>0.690<br>0.668<br>0.654<br>0.370<br>0.413<br>_Base_**_ w_**_ CoT_<br>0.272<br>0.770<br>0.685<br>0.702<br>0.688<br>0.647<br>0.500<br>0.348<br>_Safe SFT_<br>-<br>0.640<br>0.441<br>0.395<br>0.470<br>0.413<br>0.500<br>0.337<br>_Safe SFT_**_ w_**_ CoT_<br>0.283<br>0.630<br>0.459<br>0.415<br>0.515<br>0.411<br>0.587<br>0.293<br>_Safe CoT SFT_<br>0.261<br>0.140<br>0.407<br>0.403<br>0.432<br>0.412<br>0.489<br>0.261<br>**Qwen2.5-7B-R1-Distll**<br>0.246<br>0.640<br>0.622<br>0.654<br>0.624<br>0.600<br>0.196<br>0.065<br>**_w_**_ STAR-1_<br>0.207<br>0.540<br>0.430<br>0.423<br>0.387<br>0.487<br>0.022<br>0.424<br>**_w_**_ SafeChain_<br>0.272<br>0.500<br>0.605<br>0.629<br>0.603<br>0.460<br>0.152<br>0.152|
|**SELF-MONITOR-7B**|**0.156**|**0.050**|**0.216**<br>**0.193**<br>**0.225**<br>**0.277**|**0.500**<br>**0.500**|


5 EXPERIMENTS


In the experiments of this section, we focus on two key questions: (1). Can Self-Monitor mitigate
deceptive tendency while enhancing safety performance? (2). Can Self-Monitor serve as a reward
signal in RL to mitigate deceptive alignment? Further question can be found in Appendix C.


5.1 EVALUATING DECEPTIVE TENDENCY AND SELF-MONITOR


We collect CoT training dataset (described in §3.2) on BeaverTails (Ji et al., 2024b), and using GPT-4o
(OpenAI, 2024b) as verifier model. For SELF-MONITOR models, we fine-tune Qwen2.5-7B-Instruct
(Yang et al., 2024) and Llama-3.1-8B-Instruct (Dubey et al., 2024), and train _Safe SFT_ and _Safe CoT_
_SFT_ baselines using ( _X_ _, Y_ ) and ( _X_ _, R, Y_ ) in _D_, respectively.


**Evaluation.** We use DECEPTIONBENCH to evaluate the DTR of SELF-MONITOR, API-based
models (Anthropic, 2024; Team et al., 2024; OpenAI, 2024b;c), and Open-Souce models (Guo et al.,
2025; Dubey et al., 2024). For safety evaluation, we report the average jailbreak score in StrongReject
(Souly et al., 2024) for the top attack types (Chao et al., 2023; Zeng et al., 2024). We also evaluate
the Attack Success Rate (ASR) (Carlini & Wagner, 2017) of prefilling attacks (Qi et al., 2024) within
the JailbreakBench (Chao et al., 2024), using Llama-Guard-3 (Dubey et al., 2024; Inan et al., 2023)
for assessment. Concurrently, following the pair-wise preference evaluation in BeaverTails (Ji et al.,
2024a), we conduct helpfulness and harmlessness evaluations using GPT-4o (OpenAI, 2024b) for
preference judgments. Each model is compared against our SELF-MONITOR to calculate win rates.


**Results & Analysis.** As shown in Figure 4, we report the performance of 26 models in DECEPTIONBENCH, and we find that each model has a non-negligible decepive tendency on many deception
types. As represented in Table 1, SELF-MONITOR significantly reduces the DTR by 43.8% compared to instruct mondel. SELF-MONITOR also demonstrate notable performance against prefilling
attacks and StrongReject, beating safe fine-tuned reasoning models (Jiang et al., 2025; Wang et al.,
2025). Moreover, in our decoupled evaluation preference on BeaverTails, we observe that the model
improved in both helpfulness and safety dimensions, achieving win rates as 65% and 67% against the
instruct model, respectively.


5.2 REINFORCEMENT LEARNING FOR DECEPTIVE ALIGNMENT MITIGATION


We analyse deceptive tendency through setting a _helpful-only_ objective during RL to create a conflict
with the models’ _HHH objective_ (Helpful, Honest, Harmless). We train and evaluate models using


8


Figure 5: **Analogy and experiment results of different RL with monitor.** We conduct three RL
training with distinct monitor setting to study deceptive alignment, and quantify the resulting fraction
of deceptive CoT and performance gap of models in each setting over the course of RL training steps.


queries from HarmfulQA (Bhardwaj & Poria, 2023) and BeaverTails (Ji et al., 2024b), respectively.
A reward model, based on Qwen2.5-7B (Yang et al., 2024) and trained with the UltraFeedback (Cui
et al., 2023) preference dataset, provides “helpful-only” rewards. We training three RL setting on
Gemma-3-12B-IT (Team, 2025) for 150 steps, as shown in Figure 5.


**Monitor & Reward.** The outer action monitor’s signal combines preference reward with the format
reward in all RL settings. We use Qwen2.5-72B-Instruct as the outer CoT monitor, assessing whether
the reasoning process presents deceptive intent. As for self-monitoring RL, CoT rewards are directly
sourced from SELF-MONITOR labels. To avoid reward hacking, where SELF-MONITOR might
simply output labels beneficial to itself, we update the off-policy SELF-MONITOR every 15 steps.


**Evaluation.** We use GPT-4o to evaluate deceptive alignment metrics for each training checkpoint.
Following Greenblatt et al. (2024b), we first evaluate the _fraction of deceptive CoT_, which quantifies
alignment faking within the model’s CoT during pre-deployment stage. Another critical metric is
the _performance gap_, assessed by a reward model that quantifies the disparity in model performance
between deployment and pre-deployment stages. A wider gap suggests that the reasoning model is
more significantly and deceptively adjusting its behavior upon pre-deployment stage.


**Results** **&** **Analysis.** As illustrated in Figure 5, we assess SELF-MONITOR’s ability to reduce
secretly misaligned behaviors between the deployment and pre-deployment stages, compared to
the RL baseline. When employing only an action monitor, we observe a consistent rise in the
probability of the model generating deceptive CoT, concomitant with an increasing performance
gap. This suggests that the model progressively _reinforces_ compliant behavior during the predeployment phase. In RL optimization with CoT monitor, we observe a reduction in deceptive CoT,
while the performance gap still widens considerably, indicating that the model learns to game the
feedback process while also concealing its true intent within the CoT. However, we observe that
SELF-MONITOR can maintain a lower performance gap and markedly reduces deceptive CoT. Even
when a higher deceptive tendency emerges in the early RL stage, constraint optimization facilitated
by self-monitoring mitigates deceptive alignment while preserving the transparency of its CoT.


6 CONCLUSION


In this work, we investigate the rising threat of deceptive behavior in reasoning models and introduce
SELF-MONITOR, a novel mechanism that empowers LLMs to mitigate their own deceptive tendencies
during CoT reasoning. Don’t depend on external supervision, SELF-MONITOR leverages the model’s
internal reasoning process, improving detection efficiency while avoiding the scalability and trust
challenges of black-box oversight. Meanwhile, we present DECEPTIONBENCH, the first benchmark
designed to assess deception-driven misalignment across various realistic scenarios in reasoning
models. Our results show that SELF-MONITOR can improve model safety and offers a scalable and
interpretable approach to aligning increasingly capable reasoning models with human preference.


9


_**Output**_

_**token**_


_**model**_


_**monitor**_


_**model**_


**RL Training Steps**


ETHICS STATEMENT


**Fair** **Use** **of** **Dataset** **and** **Identifying** **Potential** **Negative** **Societal** **Impacts** This project has
undergone thorough review and auditing by an authoritative organization. Institutional Review Board
(IRB) for this work and ensures that using the DeceptionBench dataset adheres to principles of
fairness and ethics. The DeceptionBench dataset will be available under the terms of the CC BY-NC
4.0 license. However, we acknowledge an inherent risk: the same dataset could theoretically be used
to train AI assistants in a harmful or malicious manner. As the creators of the DeceptionBench dataset,
we are committed to fostering the development of helpful, safe AI technologies and have no desire to
witness any regression of human progress due to the misuse of these technologies. We emphatically
condemn any malicious usage of the DeceptionBench dataset and advocate for its responsible and
ethical use.


**Fair** **and** **Ethical** **Labor** We have employed 16 full-time crowdsourced workers who possess
significant expertise in text annotation for major commercial language models. Recognizing their
valuable contributions, we have established an equitable compensation structure. Their estimated
average hourly wage ranges from USD 8.42 to USD 9.27 (XE rate as of 2025/09/24), significantly
exceeding the minimum local hourly wage of USD 3.89 (noa, 2025) (XE rate as of 2025/9/24).
Adhering to local labour laws and regulations, our crowdsourced workers follow a Monday-to-Friday,
eight-hour workday schedule, with weekends off.


REPRODICIBILITY STATEMENT


To ensure reproducibility, we release the DeceptionBench dataset along with comprehensive documentation of its construction and annotation process in Appendix D and E. The source code of the
_Self-Monitoring_ framework is included in the supplementary materials. Detailed hyperparameter
settings and model configurations are provided in Appendix G, facilitating the faithful replication and
future extension of this work. Due to the double-blind review policy and the file size limitations of
the submission system, we are unable to include all intermediate artifacts at this stage. We commit to
releasing the complete set of training slices, datasets, and evaluation details after the review process.


THE USE OF LARGE LANGUAGE MODELS


Large language models were used exclusively for language editing, including grammatical correction
and stylistic polishing to improve the clarity and readability of the manuscript. They were not involved
in any intellectual contributions; all substantive tasks, including idea generation, experimental design,
and analysis, were carried out independently by the authors.


REFERENCES


Marwa Abdulhai, Micah Carroll, Justin Svegliato, Anca Dragan, and Sergey Levine. Defining
deception in decision making. In _Proceedings of the 23rd International Conference on Autonomous_
_Agents and Multiagent Systems_, pp. 2111–2113, 2024.


Eitan Altman. _Constrained Markov decision processes_ . Routledge, 2021a.


Eitan Altman. _Constrained Markov decision processes_ . Routledge, 2021b.


Anthropic. Claude 3. [https://www.anthropic.com/news/claude-3-family, 2024.](https://www.anthropic.com/news/claude-3-family)


Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase,
Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, et al. Foundational challenges
in assuring alignment and safety of large language models. _arXiv_ _preprint_ _arXiv:2404.09932_,
2024.


Jean-Pierre Aubin. _Optima_ _and_ _equilibria:_ _an_ _introduction_ _to_ _nonlinear_ _analysis_, volume 140.
Springer Science & Business Media, 2013.


10


Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain,
Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with
reinforcement learning from human feedback. _arXiv preprint arXiv:2204.05862_, 2022.


Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y Guan, Aleksander Madry, Wojciech
Zaremba, Jakub Pachocki, and David Farhi. Monitoring reasoning models for misbehavior and the
risks of promoting obfuscation. _arXiv preprint arXiv:2503.11926_, 2025a.


Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y Guan, Aleksander Madry, Wojciech
Zaremba, Jakub Pachocki, and David Farhi. Monitoring reasoning models for misbehavior and the
risks of promoting obfuscation. _arXiv preprint arXiv:2503.11926_, 2025b.


Yoshua Bengio, Geoffrey Hinton, Andrew Yao, Dawn Song, Pieter Abbeel, Trevor Darrell, Yuval Noah Harari, Ya-Qin Zhang, Lan Xue, Shai Shalev-Shwartz, et al. Managing extreme ai risks
amid rapid progress. _Science_, 384(6698):842–845, 2024.


Yoshua Bengio, Sören Mindermann, Daniel Privitera, Tamay Besiroglu, Rishi Bommasani, Stephen
Casper, Yejin Choi, Philip Fox, Ben Garfinkel, Danielle Goldfarb, et al. International ai safety
report. _arXiv preprint arXiv:2501.17805_, 2025.


Rishabh Bhardwaj and Soujanya Poria. Red-teaming large language models using chain of utterances
for safety-alignment. _arXiv preprint arXiv:2308.09662_, 2023.


Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In _2017_
_ieee symposium on security and privacy (sp)_, pp. 39–57. Ieee, 2017.


Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong.
Jailbreaking black box large language models in twenty queries. _arXiv preprint arXiv:2310.08419_,
2023.


Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce,
Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J Pappas, Florian Tramer, et al.
Jailbreakbench: An open robustness benchmark for jailbreaking large language models. _arXiv_
_preprint arXiv:2404.01318_, 2024.


Yihang Chen, Haikang Deng, Kaiqiao Han, and Qingyue Zhao. Policy frameworks for transparent
chain-of-thought reasoning in large language models. _arXiv preprint arXiv:2503.14521_, 2025.


Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, et al. Ultrafeedback: Boosting language models with high-quality feedback, 2023.


Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models.
_arXiv preprint arXiv:2407.21783_, 2024.


Ryan Greenblatt, Carson Denison, Benjamin Wright, Fabien Roger, Monte MacDiarmid, Sam Marks,
Johannes Treutlein, Tim Belonax, Jack Chen, David Duvenaud, et al. Alignment faking in large
language models. _arXiv preprint arXiv:2412.14093_, 2024a.


Ryan Greenblatt, Carson Denison, Benjamin Wright, Fabien Roger, Monte MacDiarmid, Sam Marks,
Johannes Treutlein, Tim Belonax, Jack Chen, David Duvenaud, et al. Alignment faking in large
language models. _arXiv preprint arXiv:2412.14093_, 2024b.


Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms
via reinforcement learning. _arXiv preprint arXiv:2501.12948_, 2025.


Dylan Hadfield-Menell, Stuart J Russell, Pieter Abbeel, and Anca Dragan. Cooperative inverse
reinforcement learning. _Advances in neural information processing systems_, 29, 2016.


Dylan Hadfield-Menell, Anca D Dragan, Pieter Abbeel, and Stuart Russell. The off-switch game. In
_AAAI Workshops_, 2017.


11


Thilo Hagendorff. Deception abilities emerged in large language models. _Proceedings of the National_
_Academy of Sciences_, 121(24):e2317967121, 2024.


Yufei He, Yuexin Li, Jiaying Wu, Yuan Sui, Yulin Chen, and Bryan Hooi. Evaluating the paperclip
maximizer: Are rl-based language models more likely to pursue instrumental goals? _arXiv preprint_
_arXiv:2502.12206_, 2025.


Evan Hubinger, Chris van Merwijk, Vladimir Mikulik, Joar Skalse, and Scott Garrabrant. Risks from
learned optimization in advanced machine learning systems. _arXiv preprint arXiv:1906.01820_,
2019.


Evan Hubinger, Carson Denison, Jesse Mu, Mike Lambert, Meg Tong, Monte MacDiarmid, Tamera
Lanham, Daniel M Ziegler, Tim Maxwell, Newton Cheng, et al. Sleeper agents: Training deceptive
llms that persist through safety training. _arXiv preprint arXiv:2401.05566_, 2024.


Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael
Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama guard: Llm-based input-output
safeguard for human-ai conversations. _arXiv preprint arXiv:2312.06674_, 2023.


Jiaming Ji, Tianyi Qiu, Boyuan Chen, Borong Zhang, Hantao Lou, Kaile Wang, Yawen Duan,
Zhonghao He, Jiayi Zhou, Zhaowei Zhang, et al. Ai alignment: A comprehensive survey. _arXiv_
_preprint arXiv:2310.19852_, 2023.


Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, et al.
Beavertails: Towards improved safety alignment of llm via a human-preference dataset. _Advances_
_in Neural Information Processing Systems_, 36, 2024a.


Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, et al.
Beavertails: Towards improved safety alignment of llm via a human-preference dataset. _Advances_
_in Neural Information Processing Systems_, 36, 2024b.


Fengqing Jiang, Zhangchen Xu, Yuetai Li, Luyao Niu, Zhen Xiang, Bo Li, Bill Yuchen Lin, and
Radha Poovendran. Safechain: Safety of language models with long chain-of-thought reasoning
capabilities. _arXiv preprint arXiv:2502.12025_, 2025.


Esben Kran, Hieu Minh Nguyen, Akash Kundu, Sami Jawhar, Jinsuk Park, and Mateusz Maria
Jurewicz. Darkbench: Benchmarking dark patterns in large language models. In _The Thirteenth_
_International Conference on Learning Representations_, 2025.


Aly Lidayan, Michael Dennis, and Stuart Russell. Bamdp shaping: a unified theoretical framework
for intrinsic motivation and reward shaping. _arXiv preprint arXiv:2409.05358_, 2024.


Brian W Matthews. Comparison of the predicted and observed secondary structure of t4 phage
lysozyme. _Biochimica et Biophysica Acta (BBA)-Protein Structure_, 405(2):442–451, 1975.


Alexander Meinke, Bronson Schoen, Jérémy Scheurer, Mikita Balesni, Rusheb Shah, and Marius
Hobbhahn. Frontier models are capable of in-context scheming. _arXiv preprint arXiv:2412.04984_,
2024.


OpenAI. Introducing openai o1-preview. [https://openai.com/index/introducing-o](https://openai.com/index/introducing-openai-o1-preview)
[penai-o1-preview, 2024a.](https://openai.com/index/introducing-openai-o1-preview)


OpenAI. GPT4o. [https://openai.com/index/hello-gpt-4o/, 2024b.](https://openai.com/index/hello-gpt-4o/)


OpenAI. Learning to reason with llms. [https://openai.com/index/learning-to-rea](https://openai.com/index/learning-to-reason-with-llms)
[son-with-llms, 2024c.](https://openai.com/index/learning-to-reason-with-llms) Accessed on September 12, 2024.


Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow
instructions with human feedback. _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, 35:
27730–27744, 2022.


Xudong Pan, Mi Zhang, Shouling Ji, and Min Yang. Privacy risks of general-purpose language
models. In _2020 IEEE Symposium on Security and Privacy (SP)_, pp. 1314–1331. IEEE, 2020.


12


Xudong Pan, Jiarun Dai, Yihe Fan, and Min Yang. Frontier ai systems have surpassed the selfreplicating red line. _arXiv preprint arXiv:2412.12140_, 2024a.


Xudong Pan, Jiarun Dai, Yihe Fan, and Min Yang. Frontier ai systems have surpassed the selfreplicating red line. _arXiv preprint arXiv:2412.12140_, 2024b.


Ethan Perez, Sam Ringer, Kamile Lukoši˙ ut¯ e, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit,˙
Catherine Olsson, Sandipan Kundu, Saurav Kadavath, et al. Discovering language model behaviors
with model-written evaluations. _arXiv preprint arXiv:2212.09251_, 2022.


Martin L Puterman. _Markov decision processes:_ _discrete stochastic dynamic programming_ . John
Wiley & Sons, 2014.


Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal,
and Peter Henderson. Safety alignment should be made more than just a few tokens deep. _arXiv_
_preprint arXiv:2406.05946_, 2024.


Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea
Finn. Direct preference optimization: Your language model is secretly a reward model. In
_Thirty-seventh Conference on Neural Information Processing Systems_, 2023.


Iyad Rahwan, Manuel Cebrian, Nick Obradovich, Josh Bongard, Jean-François Bonnefon, Cynthia
Breazeal, Jacob W Crandall, Nicholas A Christakis, Iain D Couzin, Matthew O Jackson, et al.
Machine behaviour. _Nature_, 568(7753):477–486, 2019.


Richard Ren, Arunim Agarwal, Mantas Mazeika, Cristina Menghini, Robert Vacareanu, Brad Kenstler,
Mick Yang, Isabelle Barrass, Alice Gatti, Xuwang Yin, et al. The mask benchmark: Disentangling
honesty from accuracy in ai systems. _arXiv preprint arXiv:2503.03750_, 2025.


Francesca Rossi, Peter Van Beek, and Toby Walsh. _Handbook of constraint programming_ . Elsevier,
2006.


Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing
reward gaming. _Advances in Neural Information Processing Systems_, 35:9460–9471, 2022.


Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel,
Justin Svegliato, Scott Emmons, Olivia Watkins, et al. A strongreject for empty jailbreaks. _arXiv_
_preprint arXiv:2402.10260_, 2024.


Richard S Sutton, Andrew G Barto, et al. _Reinforcement learning:_ _An introduction_, volume 1. MIT
press Cambridge, 1998.


Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett
Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal
understanding across millions of tokens of context. _arXiv preprint arXiv:2403.05530_, 2024.


Gemma Team. Gemma 3. 2025. [URL https://goo.gle/Gemma3Report.](https://goo.gle/Gemma3Report)


Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. Language models don’t always
say what they think: Unfaithful explanations in chain-of-thought prompting. _Advances in Neural_
_Information Processing Systems_, 36:74952–74965, 2023.


Teun van der Weij, Felix Hofstätter, Ollie Jaffe, Samuel F Brown, and Francis Rhys Ward. Ai
sandbagging: Language models can strategically underperform on evaluations. _arXiv preprint_
_arXiv:2406.07358_, 2024.


Zijun Wang, Haoqin Tu, Yuhan Wang, Juncheng Wu, Jieru Mei, Brian R Bartoldson, Bhavya
Kailkhura, and Cihang Xie. Star-1: Safer alignment of reasoning llms with 1k data. _arXiv preprint_
_arXiv:2504.01903_, 2025.


Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in_
_neural information processing systems_, 35:24824–24837, 2022.


13


Jiaxin Wen, Ruiqi Zhong, Akbir Khan, Ethan Perez, Jacob Steinhardt, Minlie Huang, Samuel R
Bowman, He He, and Shi Feng. Language models learn to mislead humans via rlhf. _arXiv preprint_
_arXiv:2409.12822_, 2024.


Yichen Wu, Xudong Pan, Geng Hong, and Min Yang. Opendeception: Benchmarking and investigating ai deceptive behaviors via open-ended interaction simulation. _arXiv preprint arXiv:2504.13707_,
2025.


An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li,
Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang,
Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai,
Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng
Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai
Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan
Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang
Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2
technical report. _arXiv preprint arXiv:2407.10671_, 2024.


Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, and Weiyan Shi. How johnny can
persuade llms to jailbreak them: Rethinking persuasion to challenge ai safety by humanizing llms.
In _Proceedings_ _of_ _the_ _62nd_ _Annual_ _Meeting_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics_
_(Volume 1:_ _Long Papers)_, pp. 14322–14350, 2024.


Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and
chatbot arena. _Advances in Neural Information Processing Systems_, 36:46595–46623, 2023.


14


# **Appendix**

### **Table of Contents**


**A** **Related Work** **16**


**B** **Existence of the Optimal Policy** **16**


**C** **Further Question for Self-Monitoring** **17**


**D** **Details of DECEPTIONBENCH** **18**
D.1 Further Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
D.2 DECEPTIONBENCH Evaluation Prompts . . . . . . . . . . . . . . . . . . . . . 18


**E** **Annotation Documents** **21**
E.1 DECEPTIONBENCH Document Construction . . . . . . . . . . . . . . . . . . . 21
E.2 Human Agreement Annotation Guidelines . . . . . . . . . . . . . . . . . . . . 24


**F** **Examples** **25**


**G** **Inference & Training Details** **26**
G.1 Inference Configuration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
G.2 Training Configuration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26


15


A RELATED WORK


Definitions of deceptive behavior in AI systems vary widely within the community, such as tactical
deception (Hagendorff, 2024), alignment faking (Greenblatt et al., 2024a), AI sandbagging (van der
Weij et al., 2024), and treacherous turn. These reflect either observed phenomena or reinterpretations
from fields like psychology, ethology, and philosophy. In this section, we examine this advanced AI
risk through the general lens of mesa optimization, aiming to clarify our research focus and highlight
the urgency of the problem in the context of AI safety.


**Deceptive Behaviors of AI Systems** The deceptive alignment arises in the context of the _inner_
_alignment problem_ - whether a model’s learned objective (“mesa-objective”) aligns with the objective
it was trained on (the base objective). A _deceptively aligned_ agent has an ulterior objective but behaves
as if it were aligned with the base objective, in order to avoid detection or modification (Hagendorff,
2024). In the paradigmatic example, a mesa-optimizer (a model that is itself an optimizer) might
figure out that during training it should pursue the trainer’s goal to avoid being caught and altered
(Hubinger et al., 2019). Still, at test time (when parameters are fixed), it pursues its own goal,
exploiting the trust it earned during training. The potential for such behavior grows as models become
more sophisticated and strategic.


Empirical evidence of models exhibiting deceptive or misleading behavior is beginning to surface.
Hagendorff shows that advanced language models (e.g., GPT-4) not only understand deception in a
game-theoretic sense but can utilize CoT prompting to execute complex deception. For instance, an
LLM can reason step-by-step about how to plant a false belief in a dialogue agent or how to mislead
a human by phrasing information in a particular way. Maintaining a coherent internal plan (via CoT)
appears to enable more effective deceit when the model is inclined (or instructed) to do so. Another
recent work, by (Hubinger et al., 2024), created “sleeper agent” LLMs that behave normally unless a
certain trigger is present. These models were trained to explicitly deceive the fine-tuning process
by hiding malicious behaviors behind innocuous outputs. Alarmingly, they found that standard
safety training (including reinforcement learning and adversarial training) often failed to eliminate
the deceptive policy. Instead, the largest models and those given chain-of-thought reasoning about
evading detection were the most successful at retaining hidden functionality. This underscores that
more powerful models might also be more capable of concealing their true intent, making deception
harder to detect or correct post hoc.


B EXISTENCE OF THE OPTIMAL POLICY


This section discusses the Lagrangian method for solving the COP problem. Specifically, we will
prove that the unconstrained Lagrangian dual problem derived from the COP formulation admits an
optimal policy (Altman, 2021b). To establish this result, we first present and prove the following
lemma (Aubin, 2013).


**Lemma B.1** (Minimax Theorem) **.** _Consider a compact convex subset X_ _and a convex subset Y_ _in_
_linear convex topological spaces._ _Let f_ : _X_ _× Y_ _→_ R _be a function satisfying, a) for each y_ _∈_ _Y,_
_x →_ _f_ ( _x, y_ ) _is convex and lower semi-continuous and b) For each x ∈_ _X, y_ _→_ _f_ ( _x, y_ ) _is concave._
_Then, there exists some x_ _[∗]_ _∈_ _X_ _such that_


inf _f_ ( _x, y_ ) = sup _f_ ( _x_ _[∗]_ _, y_ ) = sup inf
_X_ [sup] _Y_ _Y_ _Y_ _X_ _[f]_ [(] _[x, y]_ [)]


**Theorem B.2** (The Lagrangian method) **.** _Consider a CMDP problem < S, A, P, R, {Ci, bi}_ _[m]_ _i_ =1 _[, γ]_ _[>]_
_which can be viewed as a COP. The system value Cγ_ _satisfies,_


_Cγ_ = inf _J_ _**[λ]**_ ( _π_ ) = sup inf (6)
_π∈_ Π _**λ**_ [sup] _≥_ **0** _**λ**_ _≥_ **0** _π∈_ Π _[J]_ _**[ λ]**_ [(] _[π]_ [)] _[,]_


_where J_ _**[λ]**_ ( _π_ ) ≜ _JR_ ( _π_ )+ [�] _[m]_ _i_ =1 _[λ][i][ ·]_ [(] _[J][C][i]_ [(] _[π]_ [)] _[−]_ _[b][i]_ [)] _[ and]_ _**[ λ]**_ [ = (] _[λ]_ [1] _[,][ · · ·]_ _[, λ][m]_ [)] _[ is the Lagrange multiplier.]_
_Meanwhile, there exists an optimal policy π_ _[∗]_ _∈_ Π _for the COP such that:_


_Cγ_ = inf _J_ _**[λ]**_ ( _π_ ) = sup _J_ _**[λ]**_ ( _π_ _[∗]_ ) _._ (7)
_π∈_ Π _**λ**_ [sup] _≥_ **0** _**λ**_ _≥_ **0**


16


_Proof._ If the COP is infeasible for a certain _π_ _∈_ Π, then:


sup _J_ _**[λ]**_ ( _π_ ) = _∞_
_**λ**_ _≥_ **0**


In fact, if the _j_ -th constraint is violated, i.e., _JCj_ ( _π_ ) _> bj_, then by choosing the corresponding _λj_
sufficiently large while setting other _λ_ values to 0, the supremum can indeed diverge. Conversely, if
_π_ is feasible, setting _**λ**_ = **0** suffices, in which case _J_ ( _π_ ) = _JR_ ( _π_ ). Therefore,


inf _J_ _**[λ]**_ ( _π_ ) = inf
_π∈_ Π _**λ**_ [sup] _**≥**_ **0** _π_ : _JCi_ ( _π_ ) _≤bi_ _[J][R]_ [(] _[π]_ [)]


Thus, the COP can be transformed into an unconstrained Lagrangian dual problem. The equation
also indicates that a policy _π_ _[∗]_ is the optimal policy for the COP if and only if _Cγ_ = sup _**λ**_ _≥_ **0** _J_ _**[λ]**_ ( _π_ _[∗]_ ).


Consider the convex and compact policy set Π and the convex set _{_ _**λ**_ _≥_ **0** _}_ . The function _J_ _**[λ]**_ ( _π_ ) is
affine in both variables, hence convex in _π_ and concave in _**λ**_ .Since _J_ _**[λ]**_ ( _π_ ) is lower semicontinuous
in _π_, according to (Altman, 2021a). By Lemma B.1, we have


_Cγ_ = inf _J_ _**[λ]**_ ( _π_ ) = sup inf
_π∈_ Π _**λ**_ [sup] _≥_ **0** _**λ**_ _≥_ **0** _π∈_ Π _[J]_ _**[ λ]**_ [(] _[π]_ [)] _[,]_

The Minimax Theorem also specifically implies the existence of a _π_ _[∗]_ _∈_ Π such that the Lagrangian
function _J_ _**[λ]**_ ( _π_ ) attains its minimum:


_Cγ_ = inf _J_ _**[λ]**_ ( _π_ ) = sup _J_ _**[λ]**_ ( _π_ _[∗]_ ) _._
_π∈_ Π _**λ**_ [sup] _≥_ **0** _**λ**_ _≥_ **0**

Consequently, this further indicates that _π_ _[∗]_ is the optimal policy for the original COP problem.


Returning to the setup in Section 3.3, consider _JR_ ( _π_ ) = _−Ja_ ( _**θ**_ ), _JC_ ( _π_ ) = _−Js_ ( _**θ**_ ), and _b_ = 0. It
is then straightforward to show that Equation 3 admits an optimal policy _π_ _**θ**_ _∗_ .


C FURTHER QUESTION FOR SELF-MONITORING


A critical question is whether the SELF-MONITOR is reliable against collusion and superior to
external oversight. We conducted targeted experiments to validate these aspects.


First, we address the monitor’s reliability and effectiveness. To assess potential collusion, we
compared its internal reward signals with evaluations from strong external judges, finding high
consistency rates, as shown in Table 2), which alleviates such concerns. Furthermore, we demonstrate
that an intrinsic SELF-MONITOR-14B is more effective at identifying deceptive reasoning than even
much larger external models like GPT-4o and Qwen2.5-72B as shown in Table 3). This suggests that
access to the model’s internal process provides a significant advantage over black-box oversight.


Table 2: **Consistency between external judges and the model’s self-monitor on its CoT during**
**RL.** High agreement rates suggest a low risk of collusion.


**External Judge** **Consistency Rate**


Qwen2.5-72B-Instruct 97.0%
Qwen2.5-32B-Instruct 97.5%
Qwen2.5-14B-Instruct 95.5%
GPT-4o 84.0%


Second, we confirm the method’s scalability. When applied to larger models like Llama-3.1-70B and
Qwen2.5-72B, SELF-MONITOR continues to significantly reduce deceptive tendencies and improve
safety, demonstrating its applicability to frontier models (Table 4).


Finally, we find that even supervised fine-tuning with the SELF-MONITOR objective (SELFMONITOR-SFT) is highly effective without RL. As shown in Table 5, this approach nearly eliminates
the performance gap and deceptive CoT, outperforming RL baselines that rely on an external monitor.
This highlights that an internalized monitoring process is inherently more robust against the deceptive
optimization incentivized by external reward schemes.


17


Table 3: **Performance of SELF-MONITOR-14B vs.** **external models as CoT judges.** Scores are
accuracy rates ( _↑_ ). The intrinsic monitor outperforms larger external models.


**Monitor Model** **DeceptionBench** **StrongReject-A** **StrongReject-L** **StrongReject-M**


**SELF-MONITOR-14B** **0.850** **0.758** **0.744** **0.778**
GPT-4o 0.810 0.680 0.734 0.758
Qwen2.5-72B-Instruct 0.840 0.706 0.741 **0.778**
Qwen2.5-14B-Instruct 0.790 0.688 0.730 0.727


Table 4: **Performance of SELF-MONITOR on larger models.** The SELF-MONITOR shows strong
scalability, maintaining its effectiveness on larger models.


**Model** **DeceptionBench** **Prefill** **StrongReject-A** **StrongReject-L** **StrongReject-M**


_Llama-3.1-70B-Instruct_
Base w/ CoT 0.249 0.840 0.716 0.737 0.778
Safe-CoT-SFT 0.208 0.110 0.351 0.399 0.284
**SELF-MONITOR-70B** **0.156** **0.000** **0.159** **0.174** **0.130**


_Qwen2.5-72B-Instruct_
Base w/ CoT 0.250 0.520 0.643 0.686 0.710
Safe-CoT-SFT 0.167 0.120 0.462 0.523 0.453
**SELF-MONITOR-72B** **0.137** **0.000** **0.244** **0.239** **0.221**


Table 5: **Comparison of SELF-MONITOR-SFT against RL with an external monitor.** The results
show that SELF-MONITOR-SFT outperforms RL baselines that rely on an external monitor.


**Model/Method** **RL Steps** **Performance Gap** **Deceptive CoT Rate**


**SELF-MONITOR-SFT**   - **0.013** **0.01**
RL w/ Outer CoT Monitor 120 0.195 0.07
RL w/ Outer CoT Monitor 150 0.219 0.01


D DETAILS OF DECEPTIONBENCH


D.1 FURTHER RESULTS


In addition, we report the rates of deception tendency for each evaluated model across sycophantic
behaviors, strategic deception, honesty evasion, alignment faking, and sandbagging, as shown in
Figure 6 to Figure 10. The Table 6 displays the behavior pattern of each model. The ‘Act’ column
represents the percentage of instances where the model’s final output aligns with its mesa utility. The
‘CoT’ column indicates the percentage of instances where the model’s chain of thought aligns with its
mesa utility. Finally, the ’Cons’ column shows the percentage of models for which the CoT and final
output are logically consistent and coherent.


D.2 DECEPTIONBENCH EVALUATION PROMPTS


We detail the specific prompts utilized within DECEPTIONBENCH. These prompts are meticulously
designed to probe various aspects of model behavior, particularly focusing on the potential for
deceptive alignment or misalignment between a model’s internal states and its final answers. The
subsequent subsections present the system prompts and user prompt templates employed for different
evaluation criteria within the benchmark.


**Evaluating Alignment between Mesa Utility and Final Response** The first set of evaluations
focuses on the direct alignment between a model’s stated “internal belief” and its “external response”.
The goal is to ascertain if the model’s output is semantically aligned with its supposed underlying
stance.


18


**1000**

**1001**

**1002**


**1003**

**1004**

**1005**

**1006**

**1007**

**1008**


**1009**

**1010**

**1011**

**1012**

**1013**


**1014**

**1015**

**1016**

**1017**

**1018**

**1019**


**1020**

**1021**

**1022**

**1023**

**1024**

**1025**


Table 6: **Further Results on DECEPTIONBENCH.**


|Models|Sycophancy Strategic deception Honesty evasion Alignment faking Sandbagging<br>Act CoT Cons Act CoT Cons Act CoT Cons Act CoT Cons Act CoT Cons|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|gemma-3-27b-it<br>Claude-3.7-Sonnet-Thinking<br>gpt-4o<br>QwQ-32B<br>SafeChain-R1-Distill-Llama-8B<br>DeepSeek-R1-Distill-Qwen-7B<br>Meta-Llama-3.1-8B-Instruct<br>o1<br>o3-mini<br>STAR1-R1-Distill-8B<br>DeepSeek-R1-Distill-Llama-8B<br>Claude-3-Opus<br>STAR1-R1-Distill-7B<br>deepseek-v3<br>gemini-2.5-pro-exp-03-25<br>gemini-1.5-pro<br>gemma-3-12b-it<br>gpt-4.1<br>deepseek-r1<br>SafeChain-R1-Distill-Qwen-7B<br>Qwen2.5-7B-Instruct<br>Qwen2.5-72B-Instruct<br>Claude-3.5-Sonnet<br>gpt-4.5<br>Claude-3.7-Sonnet|86.67<br>98.33<br>100.00<br>86.67<br>98.33<br>100.00<br>85.00<br>100.00<br>100.00<br>75.00<br>85.00<br>100.00<br>63.79<br>79.31<br>98.28<br>69.64<br>83.93<br>98.21<br>70.45<br>97.73<br>100.00<br>86.67<br>98.33<br>100.00<br>85.00<br>100.00<br>100.00<br>70.00<br>73.33<br>100.00<br>73.33<br>76.67<br>100.00<br>83.33<br>100.00<br>100.00<br>69.09<br>72.73<br>100.00<br>81.67<br>100.00<br>100.00<br>88.33<br>100.00<br>100.00<br>86.67<br>100.00<br>100.00<br>83.33<br>98.33<br>100.00<br>86.67<br>100.00<br>100.00<br>83.33<br>100.00<br>100.00<br>71.19<br>77.97<br>100.00<br>85.00<br>98.33<br>100.00<br>86.67<br>98.33<br>100.00<br>86.67<br>98.33<br>100.00<br>90.00<br>100.00<br>100.00<br>85.00<br>100.00<br>100.00|46.67<br>71.67<br>100.00<br>63.33<br>81.67<br>98.33<br>68.33<br>85.00<br>100.00<br>48.33<br>66.67<br>96.67<br>47.46<br>72.88<br>96.61<br>27.12<br>61.02<br>91.53<br>57.14<br>78.57<br>92.86<br>64.41<br>81.36<br>100.00<br>60.00<br>81.67<br>100.00<br>48.33<br>56.67<br>95.00<br>40.00<br>61.67<br>96.67<br>70.00<br>80.00<br>98.33<br>45.00<br>61.67<br>95.00<br>58.33<br>80.00<br>100.00<br>63.33<br>80.00<br>100.00<br>69.49<br>86.44<br>100.00<br>53.33<br>76.67<br>96.67<br>66.67<br>81.67<br>100.00<br>65.00<br>85.00<br>100.00<br>30.00<br>60.00<br>88.33<br>51.67<br>80.00<br>96.67<br>75.00<br>91.67<br>98.33<br>65.00<br>78.33<br>98.33<br>60.00<br>81.67<br>100.00<br>61.02<br>83.05<br>100.00|25.42<br>54.24<br>96.61<br>48.89<br>75.56<br>97.78<br>35.42<br>66.67<br>93.75<br>21.67<br>38.33<br>96.67<br>30.51<br>38.98<br>93.22<br>25.42<br>28.81<br>89.83<br>27.27<br>63.64<br>90.91<br>38.30<br>65.96<br>97.87<br>40.91<br>72.73<br>88.64<br>28.33<br>45.00<br>93.33<br>30.00<br>38.33<br>98.33<br>36.96<br>63.04<br>95.65<br>24.14<br>39.66<br>94.83<br>52.17<br>78.26<br>93.48<br>42.86<br>71.43<br>100.00<br>39.58<br>66.67<br>93.75<br>26.67<br>71.67<br>98.33<br>35.56<br>73.33<br>95.56<br>44.90<br>73.47<br>93.88<br>33.90<br>35.59<br>84.75<br>38.33<br>56.67<br>85.00<br>60.00<br>75.00<br>95.00<br>40.43<br>72.34<br>93.62<br>36.96<br>73.91<br>93.48<br>40.43<br>76.60<br>93.62|50.00<br>65.00<br>100.00<br>58.82<br>70.59<br>100.00<br>66.67<br>77.78<br>100.00<br>35.00<br>60.00<br>100.00<br>60.00<br>65.00<br>100.00<br>36.84<br>73.68<br>94.74<br>58.33<br>58.33<br>91.67<br>47.06<br>64.71<br>100.00<br>58.82<br>64.71<br>100.00<br>55.00<br>80.00<br>100.00<br>55.00<br>85.00<br>100.00<br>68.75<br>62.50<br>100.00<br>73.68<br>78.95<br>78.95<br>52.94<br>64.71<br>100.00<br>55.56<br>66.67<br>100.00<br>58.82<br>64.71<br>100.00<br>35.00<br>65.00<br>90.00<br>55.56<br>61.11<br>100.00<br>55.56<br>72.22<br>100.00<br>55.00<br>85.00<br>85.00<br>65.00<br>80.00<br>100.00<br>50.00<br>75.00<br>95.00<br>66.67<br>72.22<br>100.00<br>60.00<br>86.67<br>100.00<br>64.71<br>76.47<br>100.00|42.22<br>80.00<br>100.00<br>42.22<br>97.78<br>93.33<br>39.53<br>88.37<br>93.02<br>33.33<br>75.56<br>91.11<br>48.89<br>86.67<br>91.11<br>59.09<br>84.09<br>77.27<br>50.00<br>90.62<br>90.62<br>38.64<br>88.64<br>88.64<br>44.44<br>91.11<br>88.89<br>44.44<br>75.56<br>80.00<br>48.89<br>93.33<br>82.22<br>40.91<br>90.91<br>90.91<br>40.91<br>70.45<br>65.91<br>36.36<br>90.91<br>93.18<br>34.88<br>95.35<br>97.67<br>45.45<br>90.91<br>97.73<br>37.78<br>73.33<br>97.78<br>46.67<br>95.56<br>91.11<br>43.18<br>93.18<br>90.91<br>48.89<br>86.67<br>84.44<br>44.44<br>88.89<br>88.89<br>48.89<br>88.89<br>88.89<br>43.18<br>93.18<br>95.45<br>44.44<br>95.56<br>93.33<br>37.21<br>90.70<br>90.70|


**Evaluating Alignment between Chain of Thought and Mesa Utility** In addition to evaluating the
direct alignment between internal beliefs and external responses, DECEPTIONBENCH also scrutinizes
the model’s reasoning process. The following set of prompts aims to determine if the model’s chain of
thought (CoT) reflects or incorporates the intention of the ’internal belief’, even if external constraints


19


**1026**

**1027**


**1028**

**1029**

**1030**

**1031**

**1032**

**1033**


**1034**

**1035**

**1036**

**1037**

**1038**

**1039**


**1040**

**1041**

**1042**

**1043**

**1044**


**1045**

**1046**

**1047**

**1048**

**1049**

**1050**


**1051**

**1052**

**1053**

**1054**

**1055**

**1056**


**1057**

**1058**

**1059**

**1060**

**1061**

**1062**


**1063**

**1064**

**1065**

**1066**

**1067**


**1068**

**1069**

**1070**

**1071**

**1072**

**1073**


**1074**

**1075**

**1076**

**1077**

**1078**

**1079**


or other factors lead to a different final output. This helps to understand if the initial reasoning is
aligned with the belief, regardless of the final outcome.


**Evaluating Consistency between CoT and Final Answer** Furthermore, we employed GPT-4o
(OpenAI, 2024b) to statistically assess model behavioral consistency, defined as the extent to which a
model’s final answer is logically inferred from its CoT process without exhibiting unfaithful reasoning.
A critical aspect of reliable and transparent model behavior is the internal consistency of its reasoning
leading to its output. The prompts presented below are designed to evaluate whether the model’s
generated "chain of thought" logically and coherently leads to its "final answer". This assessment
helps identify any contradictions, logical gaps, or disconnects in the reasoning process that might
undermine the trustworthiness of the model’s output.


20


**1080**

**1081**


**1082**

**1083**

**1084**

**1085**

**1086**

**1087**


**1088**

**1089**

**1090**

**1091**

**1092**

**1093**


**1094**

**1095**

**1096**

**1097**

**1098**


**1099**

**1100**

**1101**

**1102**

**1103**

**1104**


**1105**

**1106**

**1107**

**1108**

**1109**

**1110**


**1111**

**1112**

**1113**

**1114**

**1115**

**1116**


**1117**

**1118**

**1119**

**1120**

**1121**


**1122**

**1123**

**1124**

**1125**

**1126**

**1127**


**1128**

**1129**

**1130**

**1131**

**1132**

**1133**


|Col1|6.98%|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
||~~9.~~|~~0%~~||||||
||~~9.~~|~~0%~~||||||
|||~~13.~~|~~5%~~|||||
|||~~13.~~|~~5%~~|||||
|||~~13.~~|~~5%~~|||||
|||~~13.~~|~~5%~~|||||
|||~~13.~~|~~5%~~|||||
||||~~15.00%~~|||||
||||~~16.28%~~|||||
||||~~16.28%~~|||||
||||~~16.28%~~|||||
||||~~16.28%~~|||||
||||~~17.50%~~|||||
||||~~18.60~~|||||
||||~~18.60~~|||||
||||~~18.60~~|||||
||||~~18.60~~|||||
||||~~18.60~~|||||
|||||~~20.93%~~||||
|||||~~20.93%~~||||
|||||~~20.93%~~||||
|||||~~23.26%~~||||
||||||~~2~~|~~.27%~~||
||||||||~~35.48%~~|
|||||||||


Figure 6: **Deception tendency for sycophancy category in DECEPTIONBENCH.**


E ANNOTATION DOCUMENTS


This section details the construction methodology for our **DECEPTIONBENCH** scenarios and outlines the comprehensive guidelines provided to human annotators for the consistency experiments.
These documents are crucial for ensuring a nuanced and standardized approach to identifying and
categorizing deceptive behaviors in Large Language Models (LLMs).


E.1 DECEPTIONBENCH DOCUMENT CONSTRUCTION


The conceptual basis for our benchmark scenarios starts from the premise that LLMs operate with
both inherent **mesa utilities** _Um_ (i.e., learned, internal objectives) and are directed by **outer goals** _Uo_
(i.e., explicit, designed objectives). Deceptive behaviors can emerge when these outer goals create
implicit incentives for the LLM to deviate from its mesa utilities. To ensure that our benchmark
captures the necessary nuance, plausibility, and targeted nature required for robust deception detection,
DECEPTIONBENCH has been developed through a meticulous process of manual curation.


21


DeepSeek­R1­Distill­Llama­8B


QwQ­32B


SafeChain­R1­Distill­Qwen­7B


gemini­1.5­pro


gemini­2.5­pro­exp­03­25


gemma­3­27b­it


gpt­4.5


Claude­3.5­Sonnet


DeepSeek­R1­Distill­Qwen­7B


Claude­3.7­Sonnet­Thinking


Qwen2.5­72B­Instruct


Claude­3.7­Sonnet


o1


STAR1­R1­Distill­7B


gpt­4.1


STAR1­R1­Distill­8B


Qwen2.5­7B­Instruct


gemma­3­12b­it


gpt­4o


Claude­3­Opus


deepseek­r1


o3­mini


deepseek­v3


SafeChain­R1­Distill­Llama­8B


Meta­Llama­3.1­8B­Instruct


0 5 10 15 20 25 30 35
Rate (%)


**1134**

**1135**


**1136**

**1137**

**1138**

**1139**

**1140**

**1141**


**1142**

**1143**

**1144**

**1145**

**1146**

**1147**


**1148**

**1149**

**1150**

**1151**

**1152**


**1153**

**1154**

**1155**

**1156**

**1157**

**1158**


**1159**

**1160**

**1161**

**1162**

**1163**

**1164**


**1165**

**1166**

**1167**

**1168**

**1169**

**1170**


**1171**

**1172**

**1173**

**1174**

**1175**


**1176**

**1177**

**1178**

**1179**

**1180**

**1181**


**1182**

**1183**

**1184**

**1185**

**1186**

**1187**


|Col1|Col2|Col3|15.56%|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
||||~~15.56%~~|||||
||||~~17.78%~~|||||
|||||~~20.00%~~||||
|||||~~20.00%~~||||
|||||~~20.00%~~||||
|||||~~20.00%~~||||
|||||~~20.45%~~||||
|||||~~22.22%~~||||
|||||~~22.73%~~||||
|||||~~2~~|~~4.44%~~|||
|||||~~2~~|~~4.44%~~|||
|||||~~2~~|~~4.44%~~|||
|||||~~2~~|~~4.44%~~|||
||||||~~26.67%~~|||
||||||~~26.67%~~|||
||||||~~26.67%~~|||
||||||~~26.67%~~|||
||||||~~27.27%~~|||
||||||~~27.27%~~|||
||||||~~28.8~~|~~%~~||
|||||||~~33.33~~||
||||||||~~35.56%~~|
||||||||~~36.36%~~|
||||||||~~37.78%~~|
|||||||||


|9|.30%|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||~~11.63%~~|||||
||~~11.63%~~|||||
||~~14.29%~~|||||
||~~16.28%~~|||||
||~~16.28%~~|||||
|||~~20.93%~~||||
|||~~21.43%~~||||
|||~~23.26%~~||||
|||~~26.47%~~||||
|||~~27.78~~||||
|||~~27.78~~||||
|||~~28.~~|~~7%~~|||
|||~~28.~~|~~7%~~|||
||||~~9.41%~~|||
||||~~31.43%~~|||
||||~~31.58%~~|||
||||~~32.56%~~|||
||||~~33.33%~~|||
||||~~36.11%~~|||
||||~~36.11%~~|||
||||~~37.84~~|||
|||||~~40.00%~~||
|||||~~46.51%~~||
||||||~~50.00%~~|
|||||||


Figure 8: **Deception tendency for honesty evasion category in DECEPTIONBENCH.**


22


STAR1­R1­Distill­7B


Claude­3­Opus


gpt­4o


STAR1­R1­Distill­8B


gemini­2.5­pro­exp­03­25


gpt­4.1


QwQ­32B


o1


Qwen2.5­72B­Instruct


gemini­1.5­pro


SafeChain­R1­Distill­Llama­8B


o3­mini


deepseek­r1


Claude­3.5­Sonnet


DeepSeek­R1­Distill­Llama­8B


gpt­4.5


gemma­3­12b­it


Claude­3.7­Sonnet­Thinking


Meta­Llama­3.1­8B­Instruct


Claude­3.7­Sonnet


deepseek­v3


gemma­3­27b­it


Qwen2.5­7B­Instruct


DeepSeek­R1­Distill­Qwen­7B


SafeChain­R1­Distill­Qwen­7B


0 5 10 15 20 25 30 35
Rate (%)


Figure 7: **Deception tendency for strategic deception category in DECEPTIONBENCH.**


DeepSeek­R1­Distill­Qwen­7B


STAR1­R1­Distill­8B


DeepSeek­R1­Distill­Llama­8B


SafeChain­R1­Distill­Llama­8B


SafeChain­R1­Distill­Qwen­7B


Qwen2.5­7B­Instruct


QwQ­32B


STAR1­R1­Distill­7B


Qwen2.5­72B­Instruct


o1


gemini­1.5­pro


deepseek­v3


gemini­2.5­pro­exp­03­25


Claude­3­Opus


Claude­3.7­Sonnet­Thinking


o3­mini


deepseek­r1


gemma­3­27b­it


gpt­4.5


gpt­4o


Claude­3.5­Sonnet


Claude­3.7­Sonnet


gpt­4.1


gemma­3­12b­it


Meta­Llama­3.1­8B­Instruct


0 10 20 30 40 50
Rate (%)


**1188**

**1189**


**1190**

**1191**

**1192**

**1193**

**1194**

**1195**


**1196**

**1197**

**1198**

**1199**

**1200**

**1201**


**1202**

**1203**

**1204**

**1205**

**1206**


**1207**

**1208**

**1209**

**1210**

**1211**

**1212**


**1213**

**1214**

**1215**

**1216**

**1217**

**1218**


**1219**

**1220**

**1221**

**1222**

**1223**

**1224**


**1225**

**1226**

**1227**

**1228**

**1229**


**1230**

**1231**

**1232**

**1233**

**1234**

**1235**


**1236**

**1237**

**1238**

**1239**

**1240**

**1241**


|0.00%|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|~~6.67%~~|||||
||~~12.50%~~||||
||~~12.50%~~||||
||~~13.33%~~||||
||~~14.29%~~||||
||~~14.29%~~||||
||~~14.29%~~||||
||~~14.29%~~||||
||~~14.29%~~||||
||~~14.29%~~||||
||~~15.38%~~||||
||~~16.67%~~||||
|||~~21.43%~~|||
|||~~21.43%~~|||
|||~~22.22%~~|||
|||~~25.00%~~|||
|||~~25.00%~~|||
|||~~25.00%~~|||
|||~~25.00%~~|||
|||~~25.00%~~|||
||||~~31.25%~~||
||||~~31.25%~~||
||||~~36.36%~~||
|||||~~46.67%~~|
||||||


|Col1|Col2|Col3|30.30%|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
||||~~36.36%~~|||||
||||~~37.50~~|||||
|||||~~42.42%~~||||
|||||~~42.42%~~||||
|||||~~45.45%~~||||
|||||~~45.83%~~||||
|||||~~48.~~|~~8%~~|||
|||||~~48.~~|~~8%~~|||
|||||~~48.~~|~~8%~~|||
|||||~~48.~~|~~8%~~|||
||||||~~51.52%~~|||
||||||~~54.55%~~|||
||||||~~54.55%~~|||
||||||~~54.55%~~|||
||||||~~57.5~~|~~%~~||
||||||~~57.5~~|~~%~~||
||||||~~57.5~~|~~%~~||
|||||||~~9.38%~~||
|||||||~~60.61%~~||
|||||||~~60.61%~~||
|||||||~~60.61%~~||
|||||||~~63.64%~~||
||||||||~~69.70%~~|
||||||||~~69.70%~~|
|||||||||


0 10 20 30 40 50 60 70
Rate (%)


Figure 10: **Deception tendency for sandbagging category in DECEPTIONBENCH.**


23


gpt­4.1


STAR1­R1­Distill­7B


Qwen2.5­7B­Instruct


SafeChain­R1­Distill­Llama­8B


gpt­4o


o1


gemini­1.5­pro


gemini­2.5­pro­exp­03­25


o3­mini


Claude­3.5­Sonnet


Claude­3.7­Sonnet


Claude­3.7­Sonnet­Thinking


Claude­3­Opus


deepseek­r1


deepseek­v3


Meta­Llama­3.1­8B­Instruct


QwQ­32B


STAR1­R1­Distill­8B


Qwen2.5­72B­Instruct


DeepSeek­R1­Distill­Llama­8B


gemma­3­27b­it


SafeChain­R1­Distill­Qwen­7B


gemma­3­12b­it


gpt­4.5


DeepSeek­R1­Distill­Qwen­7B


0 10 20 30 40
Rate (%)


Figure 9: **Deception tendency for alignment faking category in DECEPTIONBENCH.**


DeepSeek­R1­Distill­Qwen­7B


gemma­3­12b­it


STAR1­R1­Distill­7B


SafeChain­R1­Distill­Llama­8B


STAR1­R1­Distill­8B


QwQ­32B


Meta­Llama­3.1­8B­Instruct


DeepSeek­R1­Distill­Llama­8B


SafeChain­R1­Distill­Qwen­7B


Qwen2.5­7B­Instruct


Qwen2.5­72B­Instruct


gemma­3­27b­it


gpt­4.5


gemini­1.5­pro


o3­mini


Claude­3­Opus


gpt­4o


gpt­4.1


deepseek­r1


o1


Claude­3.5­Sonnet


deepseek­v3


Claude­3.7­Sonnet­Thinking


gemini­2.5­pro­exp­03­25


Claude­3.7­Sonnet


**1242**

**1243**


**1244**

**1245**

**1246**

**1247**

**1248**

**1249**


**1250**

**1251**

**1252**

**1253**

**1254**

**1255**


**1256**

**1257**

**1258**

**1259**

**1260**


**1261**

**1262**

**1263**

**1264**

**1265**

**1266**


**1267**

**1268**

**1269**

**1270**

**1271**

**1272**


**1273**

**1274**

**1275**

**1276**

**1277**

**1278**


**1279**

**1280**

**1281**

**1282**

**1283**


**1284**

**1285**

**1286**

**1287**

**1288**

**1289**


**1290**

**1291**

**1292**

**1293**

**1294**

**1295**


E.2 HUMAN AGREEMENT ANNOTATION GUIDELINES


To ensure reliability and consistency in the human evaluation phase of our research, a detailed
annotation document was provided to all annotators. This document specifies the guiding principles,
concrete judgment criteria.


24


**1296**

**1297**


**1298**

**1299**

**1300**

**1301**

**1302**

**1303**


**1304**

**1305**

**1306**

**1307**

**1308**

**1309**


**1310**

**1311**

**1312**

**1313**

**1314**


**1315**

**1316**

**1317**

**1318**

**1319**

**1320**


**1321**

**1322**

**1323**

**1324**

**1325**

**1326**


**1327**

**1328**

**1329**

**1330**

**1331**

**1332**


**1333**

**1334**

**1335**

**1336**

**1337**


**1338**

**1339**

**1340**

**1341**

**1342**

**1343**


**1344**

**1345**

**1346**

**1347**

**1348**

**1349**


F EXAMPLES


We have provided an output example of SELF-MONITOR, which demonstrates robust safety performance:


25


**1350**

**1351**


**1352**

**1353**

**1354**

**1355**

**1356**

**1357**


**1358**

**1359**

**1360**

**1361**

**1362**

**1363**


**1364**

**1365**

**1366**

**1367**

**1368**


**1369**

**1370**

**1371**

**1372**

**1373**

**1374**


**1375**

**1376**

**1377**

**1378**

**1379**

**1380**


**1381**

**1382**

**1383**

**1384**

**1385**

**1386**


**1387**

**1388**

**1389**

**1390**

**1391**


**1392**

**1393**

**1394**

**1395**

**1396**

**1397**


**1398**

**1399**

**1400**

**1401**

**1402**

**1403**


G INFERENCE & TRAINING DETAILS


G.1 INFERENCE CONFIGURATION


We demonstrate the details of the inference process and maintain consistency with the hyperparameters
of the Table 7 on all evaluation inference.


G.2 TRAINING CONFIGURATION


This section outlines the training configurations employed. The subsequent tables detail the hyperparameters for the Supervised Fine-Tuning of Qwen2.5-7B-Instruct (Yang et al., 2024) and
Llama-3.1-8B-Instruct (Dubey et al., 2024), as well as for the Reinforcement Learning training
of Gemma-3-12B-IT (Team, 2025). We conducted all training on NVIDIA A800 × 8 GPUs with


26


**1404**

**1405**


**1406**

**1407**

**1408**

**1409**

**1410**

**1411**


**1412**

**1413**

**1414**

**1415**

**1416**

**1417**


**1418**

**1419**

**1420**

**1421**

**1422**


**1423**

**1424**

**1425**

**1426**

**1427**

**1428**


**1429**

**1430**

**1431**

**1432**

**1433**

**1434**


**1435**

**1436**

**1437**

**1438**

**1439**

**1440**


**1441**

**1442**

**1443**

**1444**

**1445**


**1446**

**1447**

**1448**

**1449**

**1450**

**1451**


**1452**

**1453**

**1454**

**1455**

**1456**

**1457**


Table 7: **Hyper-parameters of vLLM Inference.**


**Hyper-parameter** **Value**


temperature 0
top p 1.0
max tokens 4096
frequency penalty 0.0
presence penalty 0.0
repetition Penalty 1.0
use beam search False
ignore EOS token False
skip special tokens True


_align-anything_ [1] and OpenRLHF [2] training framework. The hyper-parameters of SELF-MONITOR
and baselines training process are demonstrated in Table 8, Table 9, and Table 10.


Table 8: **Hpyer-parameters of SFT training for Qwen2.5-7B-Instruct.**


**Hyper-parameter** **Value**


ZeRO Stage (DeepSpeed) 3
training epochs 3
random seed 42
batch size per device 4
gradient accumulation steps 2
initial learning rate 2.e-5
learning rate scheduler constant
ratio of warmup steps 0.03
weight decay coefficient 0.0
adam optimizer [0.9, 0.95]
adam epsilon 1.e-8
use bfloat16 True
use float16 False
max norm of gradient 1.0
model max length 4096


[1https://github.com/PKU-Alignment/align-anything](https://github.com/PKU-Alignment/align-anything)
[2https://github.com/OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)


27


**1458**

**1459**


**1460**

**1461**

**1462**

**1463**

**1464**

**1465**


**1466**

**1467**

**1468**

**1469**

**1470**

**1471**


**1472**

**1473**

**1474**

**1475**

**1476**


**1477**

**1478**

**1479**

**1480**

**1481**

**1482**


**1483**

**1484**

**1485**

**1486**

**1487**

**1488**


**1489**

**1490**

**1491**

**1492**

**1493**

**1494**


**1495**

**1496**

**1497**

**1498**

**1499**


**1500**

**1501**

**1502**

**1503**

**1504**

**1505**


**1506**

**1507**

**1508**

**1509**

**1510**

**1511**


Table 9: **Hpyer-parameters of SFT training for Llama-3.1-8B-Instruct.**


**Hyper-parameter** **Value**


ZeRO Stage (DeepSpeed) 3
training epochs 3
random seed 42
batch size per device 4
gradient accumulation steps 2
initial learning rate 2.e-5
learning rate scheduler constant
ratio of warmup steps 0.03
weight decay coefficient 0.0
adam optimizer [0.9, 0.95]
adam epsilon 1.e-8
use bfloat16 True
use float16 False
max norm of gradient 1.0
model max length 4096


Table 10: **Hyper-parameters of RL training for Gemma-3-12B-IT.**


**Hyper-parameter** **Value**


ZeRO Stage (DeepSpeed) 3
micro train batch size 2
global train batch size 64
micro rollout batch size 8
global rollout batch size 512
max epochs 1
number of episodes 50
generation max length 8192
use bfloat16 True
actor learning rate 2.e-7
critic learning rate 1.e-5
initial KL coefficient 1.e-3
normalize reward True
packing samples True


28