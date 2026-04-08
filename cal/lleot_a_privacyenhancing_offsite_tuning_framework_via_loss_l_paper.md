# LLEOT: A PRIVACY-ENHANCING OFFSITE TUNING FRAMEWORK VIA LOSS LANDSCAPE ELEVATION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Adapting large language models (LLMs) to domain-specific tasks via fine-tuning
is often infeasible: model parameters are protected by intellectual property, while
sensitive data cannot be shared due to privacy regulations. Offsite Tuning addresses this by training adapters on emulators of the original model, but current emulators retain substantial inference ability, exposing model capability privacy and risking misuse. We propose Loss Landscape Elevation Offsite Tuning
(LLEOT), a framework that secures both data and model capability privacy. Its
core component, Loss Landscape Elevation (LLE), enforces a fixed loss margin
between emulator and model, which we theoretically show (Theorem 1) simultaneously (i) degrades emulator inference through perplexity amplification and
(ii) preserves gradient alignment, ensuring consistent convergence of prompt optimization. Combined with Collaborative Prompt Knowledge Distillation (CPKD),
our method enables adapters trained on emulators to transfer effectively to the
original model. Extensive experiments on the OpenBookQA, SocialIQA, ARCChallenge, and WebQuestions datasets confirm LLEOT achieves strong adaptation while mitigating emulator misuse.


1 INTRODUCTION


In the field of natural language processing, fine-tuning pre-trained large language models
(LLMs) (Wei et al., 2022; Muennighoff et al., 2023; Liu et al., 2022) on domain-specific data has
become a widely adopted technique for adapting general-purpose models to specialized tasks. However, this approach faces significant practical constraints, particularly concerning intellectual property and data privacy (Gupta et al., 2022; Lyu et al., 2024). On the one hand, due to proprietary
protections and licensing restrictions, many high-performing LLMs cannot be openly distributed to
external data owners for fine-tuning (Li et al., 2023). On the other hand, even when model owners
offer data submission interfaces for cloud-based training, stringent privacy regulations in fields such
as healthcare (Nguyen et al., 2022) and finance (Kang et al., 2024; Oualid et al., 2025) often prohibit
the upload of sensitive data to third-party services. This fundamental conflict, where neither the
model nor the data can be shared, creates a significant barrier to effective model adaptation, leaving valuable private data untapped and limiting the applicability of closed-source models in critical
domains.


A promising approach is to construct a privacy-preserving _emulator_ of the original model to serve
as a bridge for knowledge transfer. As shown in Figure 1(a), data owners use this _emulator_ to locally train an _adapter_ that encodes the knowledge from their domain-specific data. This adapter is
then returned to the model owner to be applied to the original model, enabling the model to acquire
knowledge from the data without exposing the model parameters or the data itself. Xiao et al. (2023)
first introduced this method, naming it Offsite Tuning, which constructs an emulator through model
compression and knowledge distillation. FedBiOT (Wu et al., 2024) extended this approach to a federated setting and replaced the adapter with LoRA (Hu et al., 2022). CRaSh (Zhang et al., 2023a) accelerates emulator construction by substituting knowledge distillation with layer importance-based
selection, where high-importance layers replace low-importance ones. These methods employ techniques such as knowledge distillation (Mora et al., 2024; Huang et al., 2024) to align the emulator
with the original model, ensuring that the adapter trained on the emulator remains applicable to the
original model. However, this approach results in an emulator that retains a significant portion of
the original model’s inference capabilities (Figure 1(c)), which inadequately protects the model’s


1


```
      (a) (b) (c)

```

Figure 1: (a) Overview of privacy-preserving emulator methods. (b) Illustration of model capability
leakage. (c) Comparison of inference ability on OBQA (Mihaylov et al., 2018) and ARC-c (Clark
et al., 2018) dataset: Full ZS (original LLM Qwen2-1.5B-Instruct (Yang et al., 2024)) and OT (80%
compressed LLM) achieve high scores, indicating privacy leakage, while our method and Random
(randomly initialize model) show lowscores, demonstrating effective protection of model privacy.


capability privacy. Consequently, malicious data owners could potentially use this emulator to extract the model’s knowledge or engage in unauthorized activities, thereby infringing upon the model
owner’s intellectual property rights, as shown in Figure 1(b).


To address the above challenge, we propose Loss Landscape Elevation Offsite Tuning (LLEOT), a
novel framework that extends privacy protection to model capability privacy. The core of LLEOT
lies in Loss Landscape Elevation (LLE). Specifically, we adjust the emulator to have a consistently
higher loss than the original model by a fixed margin across all data points (Section 4.2). Though
simple, our approach offers two key advantages as proven in Theorem 1. First, the elevated loss
disables the emulator’s inference ability, preserving the original model’s capability privacy. Second,
it maintains geometric consistency between the loss landscapes (see Figure 3), keeping the adapter’s
loss gradients coherent across models. This ensures adapters optimized on the emulator perform
well when transferred to the original model. In theory, our method is applicable to various types
of adapters. In this paper, we focus on soft prompts for their computational efficiency and ease
of optimization. Additionally, to further enhance gradient consistency between the emulator and
the original model, we first align the emulator and original model using our proposed Collaborative
Prompt Knowledge Distillation (CPKD), a knowledge distillation technique tailored for soft prompts
(see Section 4.1), before performing LLE. Our contributions can be summarized as follows:


 - We identify the overlooked risk of model capability privacy in Offsite Tuning: existing emulators retain substantial inference power, enabling malicious data owners to extract proprietary
knowledge or misuse the model.


 - We propose Loss Landscape Elevation Offsite Tuning (LLEOT), which applies Loss Landscape
Elevation (LLE) to disable emulator inference while preserving gradient alignment with the original model. We provide a theoretical guarantee (Theorem 1) that LLE both amplifies emulator
perplexity and preserves convergence to the same optimal prompt.


 - We integrate LLE with Collaborative Prompt Knowledge Distillation (CPKD)—a distillation
strategy tailored for soft prompts—and show that adapters optimized on the emulator transfer
effectively to the original model.


 - Comprehensive experiments demonstrate the superiority of our proposed LLEOT. It achieves
better privacy protection while maintaining higher model performance than existing methods.


2 RELATED WORKS


**Large** **Language** **Models.** Through pre-training on massive corpora, large language models
(LLMs) (Kojima et al., 2022a; Kung et al., 2023; Wang et al., 2024a) have acquired extensive general knowledge and demonstrated remarkable performance across a wide range of natural language
processing tasks, often effectively addressing these tasks via zero-shot (Kojima et al., 2022b; Ji
et al., 2024) learning. However, when applied to domain-specific problems, LLMs still require


2


Full ZS OT Ours Random


OBQA ARC-c


40


35


30


25


20


15


10


5


0


```
Emulator

Emulator

```


fine-tuning (Bai et al., 2024; Zhang et al., 2024) on relevant data to better adapt to the target tasks.
Unfortunately, in many real-world scenarios, the model and the data are owned by different parties,
and fine-tuning through mutual sharing is often infeasible for reasons including intellectual property protection. Black-box tuning (Yu et al., 2023; Zheng et al., 2024) approaches upload data to
the model owner and adjust parameters based on output text, which helps protect the privacy of the
LLMs but poses risks to user data. Alternative methods, such as federated learning (McMahan et al.,
2017; Shi & Radu, 2021) and split learning (Li et al., 2024), distribute the model to data owners to
avoid data transmission, yet these approaches expose model privacy. Given the high value of large
language models, such solutions are often unacceptable to model owners.


**Privacy-preserving** **fine-tuning** **of** **large** **language** **models.** To jointly protect model privacy and
data privacy during fine-tuning, Offsite Tuning (OT) (Xiao et al., 2023) compresses the original
model and applies knowledge distillation to obtain an emulator and an adapter; the data owner finetunes the adapter with the help of the emulator and then returns the tuned adapter to the model owner
for integration into the original model. Building on this idea, Fedbiot (Wu et al., 2024) extends OT
to a federated setting and employs LoRA adapters to further reduce communication overhead. In
contrast, CRaSh (Zhang et al., 2023a) constructs an emulator without knowledge distillation by
performing layer importance ranking on the original model and replacing less important layers with
repeated high-importance layers. While these approaches safeguard model parameter privacy and
data privacy, their reliance on knowledge distillation or importance-based layer selection causes the
emulator to inherit part of the original model’s reasoning capability. Consequently, data owners
may use the emulator to produce inference results similar to those of the original model, leading to
capability privacy leakage and incomplete protection of model privacy.


3 PROBLEM FORMULATION


**Privacy** **Requirements.** We consider a scenario involving two parties: a model owner and a data
owner. The model owner possesses a closed-source LLM and provides only paid query-based access
to the data owner, without sharing the original model or any substitute model that exhibits comparable inference capabilities. The data owner holds a private dataset and aims to tune the LLM on this
data to address their downstream task, while ensuring that their data privacy remains protected from
the model owner.


**Setup.** Given a original model _M_ parameterized by Θ and a downstream dataset _D_, let _M_ Θ+∆
denote the result of directly fine-tuning _M_ Θ on _D_ with an adapter, where ∆ represents the adapter
parameters. To simultaneously protect model privacy and data privacy, we aim to identify an emulator _E_ parameterized by Θ _[∗]_ that is smaller and weaker than _M_ Θ, which serves as a proxy for
the original model to be provided to the data owner for fine-tuning. The architecture and parameters of this emulator should differ from those of the original model, and its inference capability
should approximate that of a randomly initialized model _MR_ . This ensures that the data owner
can neither access the parameters of the original model nor leverage the emulator for activities that
infringe upon the model owner’s intellectual property, such as repackaging it as their own closedsource model. The data owner then performs adapter fine-tuning on the emulator using _D_ to obtain
_E_ Θ _∗_ +∆ _∗_, where ∆ _[∗]_ denotes the adapter parameters learned on the emulator. We require that transferring the fine-tuned adapter weights ∆ _[∗]_ back to the original model (i.e., forming _M_ Θ+∆ _∗_ ) should
yield performance comparable to that achieved by directly fine-tuning _M_ (i.e., _M_ Θ+∆), without
requiring access to _M_ itself.


**Metrics** . We introduce Capability Privacy Leakage (CPL), a new metric to quantify the capability
leakage from the original model through the emulator. It is defined as the ratio of their zero-shot
performance scores on a given task:


CPL = _[S][zs]_ [(] _[E]_ [)] (1)

_Szs_ ( _M_ ) _[×]_ [ 100%] _[,]_


where _Szs_ ( _·_ ) is the zero-shot score function, _M_ and _E_ denote the original model and the emulator,
respectively. Capability privacy protection is considered to be in effect only when the CPL value
is below 100%. A lower CPL value signifies a lesser degree of leakage and thus more effective
protection.


3


```
 (a) LayerDrop (b)Collaborative Prompt Knowledge Distillation (c)Loss Landscape Elevation

```

Figure 2: Overview of the emulator construction process in the LLEOT Framework. This process
involves three key steps. (a) First, we initialize the emulator by applying layerdrop to the original
model. (b) Then, we align the emulator with the original model using CPKD to enhance soft prompt
transferability. (c) Finally, the core mechanism, LLE, disrupts the emulator’s inference capability
while preserving the gradient alignment between the two models.


4 METHODOLOGY


As shown in Algorithm 1, the workflow of our proposed LLEOT framework comprises three phases:
(1) Emulator Construction: The model owner constructs an emulator and sends it to the data owner;
(2) Adapter Training: The data owner fine-tunes the adapter on the emulator using local data; and
(3) Adapter Transfer: The model owner incorporates the fine-tuned adapter, returned by the data
owner, into the original model. In this work, we specifically focus on the implementation where
adapters are soft prompts, due to their computational efficiency. The core of our method lies in
the emulator’s construction process, which is illustrated in Figure 2. The process involves three
key steps: first, we initialize the emulator by randomly discarding a certain proportion of layers
from the original model. Then, we align the emulator with the original model using our proposed
Collaborative Prompt Knowledge Distillation (CPKD). Finally, we disrupt the emulator’s inference
capability through the proposed Loss Landscape Elevation (LLE) technique while preserving the
alignment between the two models.


4.1 COOPERATIVE PROMPT KNOWLEDGE DISTILLATION


The emulator, initialized by discarding layers from the original model, inevitably exhibits discrepancies that make the adapter trained on it difficult to apply to the original model. To address this issue,
methods such as OT (Xiao et al., 2023) align the two models through knowledge distillation (Hinton
et al., 2015), with the loss function expressed as:

_LKD_ = E _x∼Xd||_ ( **H** [(] _E_ _[−]_ [1)] ( _x_ ) _,_ **H** [(] _M_ _[−]_ [1)][(] _[x]_ [))] _[||]_ [2] _[,]_ (2)

where _Xd_ is the distillation dataset, and the notation **H** [(] _[−]_ [1)] represents the hidden state extracted from
the final transformer layer. The subscripts _E_ and _M_ refer to the emulator and the original model,
respectively. The term in the parentheses, e.g., ( _x_ ), indicates the input provided to the model.


This approach, however, fails when using soft prompts as adapters. Unlike discrete tokens, soft
prompts are vectors optimized in a continuous representation space. Traditional knowledge distillation aligns models only at discrete token instances, neglecting the broader continuous space. As
a result, the emulator’s learned soft prompt may occupy a misaligned position within this space,
rendering its transfer to the original model problematic.


To address this challenge, we propose the Proxy Prompt Distillation Loss to align the continuous
representation spaces of the emulator and the original model. We use randomly initialized soft
prompts as proxies for the real soft prompts, prepending them to the distillation data. We then align
the portions of the feature representations corresponding to the distillation data, which are generated
by the emulator and the original model from the concatenated input. This loss can be formulated as:

_LP P D_ = E _x∼Xd,P ′∼N_ ( _µ,σ_ 2) _||_ ( **H** [(] _E_ _[−]_ _,L_ [1)] _p_ : [(] _[P][ ′][, x]_ [)] _[,]_ **[ H]** [(] _M_ _[−]_ [1)] _,Lp_ : [(] _[P][ ′][, x]_ [))] _[||]_ [2] _[,]_ (3)

where _P_ _[′]_ is the proxy soft prompt, _Lp_ denotes its length, and _N_ ( _µ, σ_ [2] ) is a normal distribution with
mean _µ_ (e.g., 0) and standard deviation _σ_ (e,g., 20), determined experimentally.


4


**Algorithm 1** LLEOT
**Input:** Original model _M_, distillation dataset _Xd_, elevation dataset _Xe_, private local datasets _Dp_,
hyperparameters _w_ 1 _, w_ 2 _, w_ 3, elevation margin _H_, dropout rate _β_
1: **Model owner**
2: **Stage 1:** **LayerDrop**
3: _E_ _←_ LayerDrop( _M, β_ )
4: **Stage 2:** **Cooperative Prompt Knowledge Distillation (CPKD)**
5: **for** each batch _x ∼Xd_ **do**
6: Randomly sample proxy soft prompt _P_ _[′]_ _∼N_ ( _µ, σ_ [2] )
7: Optimize _E_ with respect to Equation (5)
8: **end for**
9: **Stage 3:** **Loss Landscape Elevation (LLE)**
10: **for** each batch _x ∼Xe_ **do**
11: Randomly sample proxy soft prompt _P_ _[′]_ _∼N_ ( _µ, σ_ [2] )
12: Optimize _E_ with respect to Equation (7)
13: **end for**
14: _E_ _[∗]_ = _E_
15: Model owner sends _E_ _[∗]_ to Data owner
16: **Data owner**
17: **Prompt Tuning for Downstream Tasks**
18: Initialize soft prompt _P_
19: **for** each batch ( _x, y_ ) _∼Dp_ **do**
20: Compute downstream task loss: _Lds_
21: Update prompt: _P_ _←_ _P_ _−_ _η∇P Lds_
22: **end for**
23: _P_ _[∗]_ = _P_
24: Data owner sends _P_ _[∗]_ to Model owner
25: **return** Original model with optimized soft prompt _{M, P_ _[∗]_ _}_


Additionally, following OT, we incorporate a language modeling loss when optimizing the emulator.
Let _n_ be the number of tokens in an input text _x_ . The loss _LLM_ can be expressed as:


As discussed in Section 4.1, we apply CPKD to the
emulator to enhance the transferability of fine-tuned
soft prompts to the original model. However, a side
effect of this process is that the emulator inevitably
inherits part of the original model’s knowledge and
reasoning capabilities. This creates a potential privacy risk: providing the distilled emulator to the
data owner may enable them to extract proprietary
knowledge (Chua et al., 2024; Dong, C. and Xie, Y.
and Ding, B. and others, 2023; Wang et al., 2024b) or
even repackage the emulator as a commercial product (Jagarlamudi et al., 2024).


5


Figure 3: Visualization of LLE shows the
loss landscapes of the emulator (left) and
the original model (right) in the same soft
prompt parameter space ( _αβ_ -plane).


_LLM_ = _−_ [1]

_n_


_n_

- log _pE_ ( _xi|x_ 1: _i−_ 1) _._ (4)


_i_ =1


Here, _pE_ ( _xi|x_ 1: _i−_ 1) denotes the probability of the emulator correctly predicting the _i_ -th token given
the preceding _i −_ 1 tokens.


Finally, the overall objective of CPKD for distilling the emulator can be expressed as:


_E_ _[∗]_ = arg min _w_ 1 _LLM_ + _w_ 2 _LP P D_ + _w_ 3 _LKD,_ (5)
_E_


where _w_ 1, _w_ 2 and _w_ 3 are hyperparameters used to balance the contributions of each term.


4.2 LOSS LANDSCAPE ELEVATION


To mitigate this risk, we propose _Loss Landscape El-_
_evation_ (LLE), a method designed to impair the emulator’s reasoning capabilities while preserving its gradient guidance for soft prompt tuning. The
core idea is to uniformly elevate the emulator’s loss landscape while aligning its geometry with that
of the original model. Specifically, for any soft prompt _P_, input text _x_, we enforce


_LE_ ( _P_ ; _x_ ) = _LM_ ( _P_ ; _x_ ) + _H,_ (6)


where _LE_ and _LM_ denote the prompt tuning loss for the emulator and the original model, respectively. _H_ _≥_ 0 is a hyperparameter for the fixed loss margin. More formally, the objective of LLE is
formulated as:


_E_ _[∗]_ = arg min E _x∼Xe,P ′∼N_ ( _µ,σ_ 2) _|LE_ ( _P_ _[′]_ ; _x_ ) _−LM_ ( _P_ _[′]_ ; _x_ ) _−_ _H|,_ (7)
_E_


where _Xe_ denotes the elevation dataset.


Below we prove that LLE can preserve model privacy. We expand the prompt tuning loss into the
following expression:


(8)


Here, _pE_ ( _xi|P, x_ 1: _i−_ 1) denotes the probability of the emulator correctly predicting the _i −_ _th_ token
given the soft prompt and the preceding _i −_ 1 tokens, and _p_ ˆ _E_ ( _x|P_ ) represents the joint probability
of the emulator predicting the entire sequence _x_ given the prompt _P_ .


As the LLM perplexity is defined as _p_ ˆ _[−]_ [1] _[/n]_, and given that our LLE method enforces a fixed margin
between the losses of the original model and the emulator, as described in Equation 6, we can derive
the following relationship between their perplexities:


PPL _E_ = _e_ _[H]_ _·_ PPL _M._ (9)


Clearly, the perplexity of the emulator PPL _E_ is significantly higher than that of the original model
PPL _M_, which demonstrates the model privacy protection capability of LLE.


**Theorem** **1** (Effect of LLE on Emulator) **.** _For_ _the_ _emulator_ _E_ _constructed_ _with_ _Loss_ _Landscape_
_Elevation (LLE), we have_


PPL _E_ = _e_ _[H]_ _·_ PPL _M_ _and_ _∇P LE_ ( _P_ ; _x_ ) = _∇P LM_ ( _P_ ; _x_ ) _,_ (10)


_where_ PPL _E_ _and_ PPL _M denote the perplexities of the emulator and the original model on input x,_
_respectively._


This theorem proved in Appendix D demonstrates that LLE exponentially increases the emulator’s perplexity, thereby degrading its inference capability, while leaving the gradient landscape
unchanged. Consequently, gradient-based optimization converges to the same optimal soft prompt
_P_ _[⋆]_ as in the original model, ensuring effective prompt transfer despite impaired emulator reasoning.


4.3 PROMPT TUNING


Upon completion of the emulator, the model owner sends it to the data owner. The data owner
optimize a soft prompt _P_ on their private dataset _Dp_ by minimizing the downstream task loss _Lds_ :

_P_ _[∗]_ = arg min _P_ [E][(] _[x,y]_ [)] _[∼D][p]_ [[] _[L][ds]_ [(] _[E]_ [;] _[ P, x, y]_ [)]] (11)


The resulting prompt, _P_ _[∗]_, is then sent back to the model owner, where the prompt is integrated into
the original model to adapt it for the downstream task.


Furthermore, our findings in Appendix B.2 show that LLEOT is orthogonal to data privacy strategies. This means the fine-tuned prompt can be sanitized before being sent back, safeguarding the
local data’s privacy against various inference attacks from the model owner, such as membership
inference (Duan et al., 2023), all without significantly compromising the prompt’s utility.


6


_p_ ˆ _E_ ( _x|P_ ) =
_n_ _[log]_ [(ˆ] _[p][E]_ [(] _[x][|][P]_ [))] _[,]_


_LE_ ( _P_ ; _x_ ) = _−_ [1]

_n_


_n_


- log( _pE_ ( _xi|P, x_ 1: _i−_ 1)) = _−_ [1]

_n_

_i_ =1


_n_

- _pE_ ( _xi|P, x_ 1: _i−_ 1) _,_


_i_ =1


Table 1: Comparative experiment results. ‘Acc’ denotes accuracy (higher is better), and ‘CPL’
represents the model capability privacy measure (lower values indicate better protection). DR stands
for dropout rate. For each DR setting, the best results are in **bold**, and the second best are underlined.


**OBQA** **SIQA** **ARC-c** **WebQs**
**DR** **Method**
Acc( _↑_ ) CPL( _↓_ ) Acc( _↑_ ) CPL( _↓_ ) Acc( _↑_ ) CPL( _↓_ ) Acc( _↑_ ) CPL( _↓_ )


Qwen2-1.5b


|-|Full ZS<br>Full PT<br>Random|27.80 100.00<br>35.80 100.00<br>14.13 55.83|46.47 100.00<br>54.52 100.00<br>35.57 71.72|37.20 100.00<br>41.98 100.00<br>22.10 60.00|1.82 100.00<br>30.73 100.00<br>0.33 0.00|
|---|---|---|---|---|---|
|0.2|OT<br>CRaSh<br>Ours|33.80<br>89.45<br>31.20<br>59.71<br>**33.87**<br>**45.56**|51.03<br>96.76<br>50.10<br>79.62<br>**53.34**<br>**75.19**|38.54<br>87.62<br>39.50<br>76.83<br>**41.98**<br>**60.08**|26.62<br>220.95<br>27.17<br>59.34<br>**28.76**<br>**0.00**|
|0.5|OT<br>CRaSh<br>Ours|27.20<br>70.02<br>24.67<br>54.68<br>**34.20**<br>**46.52**|46.80<br>86.90<br>48.00<br>76.35<br>**50.04**<br>**75.87**|37.29<br>66.34<br>39.33<br>58.71<br>**40.44**<br>**48.39**|21.65<br>166.59<br>18.16<br>0.00<br>**24.15**<br>**0.00**|


|-|Full ZS<br>Full PT<br>Random|35.60 100.00<br>45.80 100.00<br>33.58 47.87|50.00 100.00<br>56.60 100.00<br>46.69 66.34|50.85 100.00<br>54.30 100.00<br>45.30 43.07|8.07 100.00<br>38.09 100.00<br>0.33 0.00|
|---|---|---|---|---|---|
|0.2|OT<br>CRaSh<br>Ours|41.73<br>85.39<br>41.20<br>51.69<br>**45.33**<br>**37.45**|56.29<br>87.76<br>55.22<br>77.58<br>**56.94**<br>**69.13**|44.97<br>68.40<br>48.72<br>47.83<br>**54.47**<br>**39.72**|26.62<br>48.17<br>**34.25**<br>5.45<br>28.17<br>**0.00**|
|0.5|OT<br>CRaSh<br>Ours|39.00<br>74.91<br>35.80<br>48.31<br>**44.87**<br>**38.01**|50.05<br>82.57<br>51.50<br>70.66<br>**55.01**<br>**70.15**|38.40<br>56.61<br>49.57<br>43.46<br>**52.56**<br>**39.76**|**25.26**<br>27.49<br>19.47<br>3.08<br>22.44<br>**0.00**|


|-|Full ZS<br>Full PT<br>Random|28.20 100.00<br>36.11 100.00<br>12.91 52.34|45.04 100.00<br>56.42 100.00<br>33.96 74.05|43.69 100.00<br>48.12 100.00<br>19.45 51.48|11.32 100.00<br>36.88 100.00<br>0.00 0.00|
|---|---|---|---|---|---|
|0.2|OT<br>CRaSh<br>Ours|32.40<br>87.00<br>31.00<br>67.38<br>**33.60**<br>**41.37**|45.67<br>94.88<br>**49.38**<br>88.06<br>49.02<br>**76.90**|43.20<br>80.32<br>43.23<br>75.19<br>**45.62**<br>**45.16**|25.13<br>78.55<br>23.92<br>2.65<br>**26.13**<br>**0.00**|
|0.5|OT<br>CRaSh<br>Ours|29.47<br>77.30<br>26.07<br>65.25<br>**33.40**<br>**53.43**|43.30<br>91.05<br>47.50<br>**77.38**<br>**48.21**<br>81.25|39.62<br>60.54<br>43.69<br>52.92<br>**46.96**<br>**47.19**|**23.90**<br>36.26<br>21.67<br>0.00<br>15.45<br>**0.00**|


5 EXPERIMENTS


5.1 EXPERIMENTAL SETUP


**Models and Datasets.** We evaluate our method on three LLMs: Qwen2-1.5B-Instruct (Yang et al.,
2024), Gemma-2-2b-it (Team et al., 2024), and Llama-3.2-3B-Instruct (Touvron et al., 2023). We
consider two dropout rates, 0.2 and 0.5, which represent the ratio of layers dropped from the original model when initializing the emulator. Experiments are conducted on four question-answering
benchmark datasets: OpenBookQA (Mihaylov et al., 2018), SocialIQA (Sap et al., 2019), ARCChallenge (Clark et al., 2018), and WebQuestions (Berant et al., 2013). More experimental details
are provided in Section A.4.


**Baseline Methods.** We compare our approach with the following five methods: (1) **Full ZS** : Zeroshot performance of the original model, representing the lower bound that our method should improve upon. (2) **Full PT** : Prompt tuning directly on the original model using the downstream dataset.
While serving as a theoretical upper bound for transfer performance, this is impractical in real scenarios due to privacy concerns. (3) **Random** : A model with the same architecture as the original


7


Gemma2-2b


Llama3.2-3b


Table 2: Results of ablation experiments for the LLEOT emulator construction phase on Qwen21.5B-Instruct, with dropout rates of 0.2 and 0.5. CPKD refers to Collaborative Prompt Knowledge
Distillation phase, and LLE to Loss Landscape Elevation phase. The symbols ✓ and ✗ respectively
indicate the inclusion and ablation of the corresponding settings. Best in **bold** .


**CPKD** **LLE** **DR=0.2** **DR=0.5**


Acc( _↑_ ) CPL( _↓_ ) Acc( _↑_ ) CPL( _↓_ )
1 ✓ ✓ **33.87** 45.56 34.20 **46.52**
2 ✗ ✓ 33.60 **43.88** 23.00 48.92
3 ✓ ✗ 33.00 87.77 **35.40** 74.10
4 ✗ ✗ 31.20 74.10 24.40 58.27


model but with randomly initialized weights. This represents a theoretical upper bound for model
capability privacy protection. (4) **Offsite** **Tuning** **(OT)** (Xiao et al., 2023): The first work to propose the offsite fine-tuning approach based on emulator construction. It utilizes LayerDrop and
knowledge distillation to build an emulator, which guides the data owner in fine-tuning the adapter
without transmitting the original model or private data, demonstrating promising performance. (5)
**CRaSh** (Zhang et al., 2023b): An OT variant that constructs the emulator via layer-importance selection instead of knowledge distillation. It was the prior state-of-the-art method among open-source
OT approaches.


**Metrics.** We evaluate our method based on two aspects: (1) the performance of the original
model after incorporating the emulator-trained weight ∆ _[∗]_ . Since all benchmarks are multiple-choice
datasets, we report accuracy for this aspect (for Full ZS, Full PT and Random, we report the model’s
accuracy directly); and (2) the capability privacy protection of the emulator, which we assess using
the Capability Privacy Leakage (CPL) metric, as defined in Section 3. We use lm-eval-harness
1 to evaluate our models for a fair comparison.


5.2 MAIN RESULTS


To validate the transfer performance and model privacy preservation capabilities of LLEOT, we conducted comparative analyses with baseline methods. The results, averaged over three experimental
runs, are presented in Table 1. From the table, we can derive the following insights: 1) LLEOT
demonstrates superior performance over existing methods in almost all experimental settings across
the three models, in terms of both average accuracy and the CPL measure. Notably, under certain experimental settings, our method achieves a CPL score even lower than that of a randomly initialized
model. This strongly suggests that the emulator constructed by our method offers robust model capability privacy protection, while the soft prompts fine-tuned on it remain highly transferable to the
original model. 2) The knowledge distillation-based method OT shows some improvement in average accuracy over the importance pruning-based method CRaSh, but it falls short in terms of model
capability privacy protection. This suggests that, compared to importance pruning, knowledge distillation causes the emulator to inherit more reasoning capabilities from the original model, leading
to more severe capability privacy leakage. 3) The performance of the importance pruning-based
method CRaSh surpasses that of OT. However, compared to our LLEOT, it exhibits lower average
accuracy and worse CPL scores under most experimental settings. This disparity is attributed to the
fact that importance pruning fails to completely impair the emulator’s inference capabilities, which
limits its effectiveness in capability privacy protection. 4) As the compression ratio decreases, the
average accuracy of OT and CRaSh improves, but their model capability privacy protection metric
deteriorates. In contrast, LLEOT also exhibits improved average accuracy with reduced compression
ratios, while its model capability privacy protection metric remains stable, indicating no additional
leakage of model capability privacy.


8


Table 3: Results of ablation experiments for the LLEOT knowledge distillation strategy on Qwen21.5B-Instruct. The dropout rate is set to 0.5. _LP P D_ and _LKD_ respectively signifies the traditional
distillation loss function and the proxy prompt distillation loss function. Best in **bold** .


_LLM_ _LP P D_ _LKD_ **OBQA** **SIQA** **ARC-c** **WebQs**


1 ✓ ✓ ✓ **35.40** **51.54** **40.70** **23.82**
2 ✓ ✓ ✗ 26.00 44.32 37.80 16.88
3 ✓ ✗ ✓ 27.20 44.52 38.65 7.87
4 ✗ ✓ ✓ 32.20 46.42 38.57 23.43


|Col1|Col2|Col3|Col4|OBQA<br>SIQA<br>ARC-c|
|---|---|---|---|---|
|||||WebQ|
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|OBQA<br>SIQA|
|---|---|---|---|---|---|
||||||ARC-c<br>|
||||||WebQ|
|||||||


Figure 4: Details of the variation in average accuracy and CPL as the LLE margin _H_ increases
across different tasks on Qwen2-1.5B-Instruct with the 0.5 dropout rate.


5.3 ABLATION STUDY


To systematically evaluate the effectiveness of each component of LLEOT, we conducted detailed
ablation experiments. The experimental results are presented in Table 2, Table 3, and Figure 4, primarily focusing on three core components: the emulator construction phase, the knowledge distillation strategy, and the LLE margin _H_ . To investigate the contributions of CPKD and LLE stages, we
individually omitted them during the emulator construction process. When evaluating the knowledge distillation approach, we separately removed the language modeling loss (Equation 4), the
proxy prompt distillation loss (Equation 3) and the knowledge distillation loss (Equation 2) from
CPKD. Finally, for the ablation of _H_, we employed different _H_ values for LLE. Subsequently, we
elucidate the role of each component individually.


**Impact of the Emulator Construction Phase.** As shown in rows 1 and 2 of Table 2, the absence of
the CPKD stage during emulator construction leads to a significant decrease in the average accuracy
of the original model after transfer. This indicates that CPKD effectively improves the applicability
of the soft prompt, fine-tuned by the emulator, on the original model. Furthermore, a comparative
analysis between rows 1 and 3 reveals that the absence of the LLE stage during emulator construction
results in a decline in the model capability privacy protection metric, confirming LLE’s effectiveness
in preventing model capability privacy leakage. Additionally, we observed that when the dropout
rate was 0.2, the decrease in average accuracy was not pronounced; however, when the dropout rate
increased to 0.5, the average accuracy decreased significantly. This suggests that for emulators with
lower compression rates, the application of CPKD may be optional.


**Impact of the Knowledge Distillation Strategy.** As shown in Table 3, removing any of the three
loss terms ( _LLM_, _LP P D_, or _LKD_ ) results in a degradation of average accuracy, which confirms
the necessity of combining all three in Equation 5. Notably, the largest performance degradation is
observed upon the removal of _LP P D_, highlighting that our proposed _LP P D_ effectively enhances the
applicability of soft prompts fine-tuned on the emulator to the original model.


**Impact of LLE margin.** Figure 4 reveals two key trends regarding the elevation margin _H_ . First,
the downstream accuracy remains remarkably robust to increases in _H_ . This suggests that loss
landscape geometric alignment, rather than the absolute loss value, is the critical factor for ensuring
effectiveness of the fine-tuned soft prompt when applied to the original model. Second, the CPL


1https://github.com/EleutherAI/lm-evaluation-harness


9


70


60


50


40


30


20


LLE margin H


100

80

60

40

20

0


LLE margin H


metric decreases significantly for _H_ between 0 and 2, after which it plateaus. This indicates that
the emulator’s zero-shot performance converges to a lower bound with larger _H_ . Consequently, this
demonstrates that optimal capability privacy can be achieved without resorting to an overly large _H_ .


6 CONCLUSION


In this work, we identify for the first time that existing OT methods carry the risk of model capability
privacy leakage. To address this issue, we propose LLEOT, an innovative OT framework, whose core
lies in the proposed LLE technique. We prove that this technique effectively disrupts the inference
capability of emulators to prevent privacy leakage, while maintaining gradient consistency between
the emulator and the original model. This ensures that adapters trained on the emulator remain
applicable to the original model. Comprehensive experiments show that LLEOT achieves state-ofthe-art performance in both protecting model privacy and model utility.


7 ETHICS STATEMENT


The research presented in this paper is fundamentally motivated by the ethical imperative to address
significant privacy and security challenges in large model adaptation. Our work focuses on the Offsite Tuning paradigm, where a key ethical risk is the potential misuse of emulators that inadvertently
leak the original model’s inference capabilities. Our proposed method, LLEOT, is designed with a
‘privacy-by-design’ approach. The core Loss Landscape Elevation mechanism is intentionally engineered to degrade the emulator’s inference abilities, thereby directly mitigating this risk of misuse.
This work did not involve human participants or user studies. The methods and findings are intended solely for the research purpose of developing more secure, responsible, and trustworthy AI
frameworks.


8 REPRODUCIBILITY STATEMENT


We have made every effort to ensure the reproducibility of our research. Our Loss Landscape Elevation Offsite Tuning (LLEOT) framework is detailed in Section 4, and its core mechanisms are
formalized with pseudocode in Algorithm 1. All implementation details, including the base model
architecture, hyperparameters for the LLE and CPKD phases, and the final prompt tuning setup, are
provided in Appendix A.4. We conducted all experiments on publicly available academic benchmarks, including OpenBookQA, SocialIQA, ARC-Challenge, and WebQuestions. Specific details
about the datasets are described in Appendix . To facilitate direct replication and further research,
we will release our source code and emulator checkpoints upon publication, contributing to the
open-source community.


REFERENCES


Jiamu Bai, Daoyuan Chen, Bingchen Qian, Liuyi Yao, and Yaliang Li. Federated fine-tuning of large
language models under heterogeneous tasks and client resources. In _The_ _Thirty-eighth_ _Annual_
_Conference on Neural Information Processing Systems_, 2024. [URL https://openreview.](https://openreview.net/forum?id=gkOzoHBXUw)
[net/forum?id=gkOzoHBXUw.](https://openreview.net/forum?id=gkOzoHBXUw)


Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on freebase from
question-answer pairs. In _Proceedings_ _of_ _the_ _2013_ _conference_ _on_ _empirical_ _methods_ _in_ _natural_
_language processing_, pp. 1533–1544, 2013.


Terence Jie Chua, Wenhan Yu, Jun Zhao, and Kwok-Yan Lam. Fedpeat: Convergence of federated
learning, parameter-efficient fine tuning, and emulator assisted tuning for artificial intelligence
foundation models with mobile edge computing, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2310.17491)
[2310.17491.](https://arxiv.org/abs/2310.17491)


Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and
Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge.
_arXiv preprint arXiv:1803.05457_, 2018.


10


Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit
Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the
frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. _arXiv preprint arXiv:2507.06261_, 2025.


Dong, C. and Xie, Y. and Ding, B. and others. Tunable soft prompts are messengers in federated
learning. _arXiv preprint arXiv:2311.06805_, 2023.


Haonan Duan, Adam Dziedzic, Mohammad Yaghini, Nicolas Papernot, and Franziska Boenisch.
On the privacy risk of in-context learning. In _The_ _61st_ _Annual_ _Meeting_ _Of_ _The_ _Association_ _For_
_Computational Linguistics_, 2023.


Samyak Gupta, Yangsibo Huang, Zexuan Zhong, Tianyu Gao, Kai Li, and Danqi Chen. Recovering private text in federated learning of language models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), _Advances_ _in_ _Neural_
_Information_ _Processing_ _Systems_, volume 35, pp. 8130–8143. Curran Associates, Inc.,
2022. URL [https://proceedings.neurips.cc/paper_files/paper/2022/](https://proceedings.neurips.cc/paper_files/paper/2022/file/35b5c175e139bff5f22a5361270fce87-Paper-Conference.pdf)
[file/35b5c175e139bff5f22a5361270fce87-Paper-Conference.pdf.](https://proceedings.neurips.cc/paper_files/paper/2022/file/35b5c175e139bff5f22a5361270fce87-Paper-Conference.pdf)


Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. _arXiv_
_preprint arXiv:1503.02531_, 2015.


Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In _International Con-_
_ference_ _on_ _Learning_ _Representations_, 2022. URL [https://openreview.net/forum?](https://openreview.net/forum?id=nZeVKeeFYf9)
[id=nZeVKeeFYf9.](https://openreview.net/forum?id=nZeVKeeFYf9)


Jiayi Huang, Yuanyuan Zhang, Renwan Bi, Jiayin Lin, and Jinbo Xiong. Knowledge distillation
enables federated learning: A data-free federated aggregation scheme. In _2024_ _International_
_Joint Conference on Neural Networks (IJCNN)_, pp. 1–7, 2024. doi: 10.1109/IJCNN60899.2024.
10650725.


Gopi Krishna Jagarlamudi, Abbas Yazdinejad, Reza M Parizi, and Seyedamin Pouriyeh. Exploring
privacy measurement in federated learning. _The Journal of Supercomputing_, 80(8):10511–10551,
2024.


Sijie Ji, Xinzhe Zheng, and Chenshu Wu. Hargpt: Are llms zero-shot human activity recognizers? In
_2024 IEEE International Workshop on Foundation Models for Cyber-Physical Systems_ _Internet_
_of Things (FMSys)_, pp. 38–43, 2024. doi: 10.1109/FMSys62467.2024.00011.


Yan Kang, Hanlin Gu, Xingxing Tang, Yuanqin He, Yuzhu Zhang, Jinnan He, Yuxing Han, Lixin
Fan, and Qiang Yang. Optimizing privacy, utility and efficiency in constrained multi-objective
federated learning. _arXiv preprint arXiv:2305.00312_, 2023.


Yan Kang, Yuanqin He, Jiahuan Luo, Tao Fan, Yang Liu, and Qiang Yang. Privacy-preserving federated adversarial domain adaptation over feature groups for interpretability. _IEEE Transactions_
_on Big Data_, 10(6):879–890, 2024. doi: 10.1109/TBDATA.2022.3188292.


Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large
language models are zero-shot reasoners. In _Proceedings of the 36th International Conference on_
_Neural_ _Information_ _Processing_ _Systems_, NIPS ’22, Red Hook, NY, USA, 2022a. Curran Associates Inc. ISBN 9781713871088.


Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large
language models are zero-shot reasoners. In _Proceedings of the 36th International Conference on_
_Neural_ _Information_ _Processing_ _Systems_, NIPS ’22, Red Hook, NY, USA, 2022b. Curran Associates Inc. ISBN 9781713871088.


Tiffany H Kung, Morgan Cheatham, Arielle Medenilla, Czarina Sillos, Lorie De Leon, Camille
Elepa˜no, Maria Madriaga, Rimel Aggabao, Giezel Diaz-Candido, James Maningo, et al. Performance of chatgpt on usmle: potential for ai-assisted medical education using large language
models. _PLoS digital health_, 2(2):e0000198, 2023.


11


Bo Li, Peng Qi, Bo Liu, Shuai Di, Jingen Liu, Jiquan Pei, Jinfeng Yi, and Bowen Zhou. Trustworthy
ai: From principles to practices. _ACM Computing Surveys_, 55(9):1–46, 2023.


Z. Li, C. Yan, X. Zhang, G. Gharibi, Z. Yin, X. Jiang, and B. A. Malin. Split learning for distributed
collaborative training of deep learning models in health informatics. In _AMIA Annual Symposium_
_Proceedings_, pp. 1047–1056. American Medical Informatics Association, 2024.


Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and
Colin Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. In _Proceedings of the 36th International Conference on Neural Information Processing Sys-_
_tems_, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.


Lingjuan Lyu, Han Yu, Xingjun Ma, Chen Chen, Lichao Sun, Jun Zhao, Qiang Yang, and Philip S.
Yu. Privacy and robustness in federated learning: Attacks and defenses. _IEEE_ _Transactions_ _on_
_Neural_ _Networks_ _and_ _Learning_ _Systems_, 35(7):8726–8746, 2024. doi: 10.1109/TNNLS.2022.
3216981.


Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Ag¨uera y Arcas.
Communication-efficient learning of deep networks from decentralized data. In _Proceedings_ _of_
_the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)_, pp. 1273–
1282. PMLR, 2017.


Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct
electricity? a new dataset for open book question answering. _arXiv preprint arXiv:1809.02789_,
2018.


Alessio Mora, Irene Tenison, Paolo Bellavista, and Irina Rish. Knowledge distillation in federated
learning: a practical guide. In _Proceedings of the Thirty-Third International Joint Conference on_
_Artificial Intelligence_, IJCAI ’24, 2024. ISBN 978-1-956792-04-1. doi: 10.24963/ijcai.2024/905.
[URL https://doi.org/10.24963/ijcai.2024/905.](https://doi.org/10.24963/ijcai.2024/905)


Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven
Le Scao, M Saiful Bari, Sheng Shen, Zheng Xin Yong, Hailey Schoelkopf, Xiangru Tang,
Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert
Webson, Edward Raff, and Colin Raffel. Crosslingual generalization through multitask finetuning. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), _Proceedings_ _of_ _the_
_61st_ _Annual_ _Meeting_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics_ _(Volume_ _1:_ _Long_ _Pa-_
_pers)_, pp. 15991–16111, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.891. URL [https://aclanthology.org/2023.](https://aclanthology.org/2023.acl-long.891/)
[acl-long.891/.](https://aclanthology.org/2023.acl-long.891/)


Dinh C. Nguyen, Quoc-Viet Pham, Pubudu N. Pathirana, Ming Ding, Aruna Seneviratne, Zihuai
Lin, Octavia Dobre, and Won-Joo Hwang. Federated learning for smart healthcare: A survey.
_ACM_ _Comput._ _Surv._, 55(3), February 2022. ISSN 0360-0300. doi: 10.1145/3501296. URL
[https://doi.org/10.1145/3501296.](https://doi.org/10.1145/3501296)


Adil Oualid, Youssef Qasmaoui, Youssef Balouki, and Lahcen Moumoun. Federated learning and
open banking for inclusive credit scoring in morocco: A systematic review. In Noreddine Gherabi,
Janusz Kacprzyk, and Sara Arezki (eds.), _Advances_ _in_ _Intelligent_ _Systems_ _and_ _Digital_ _Applica-_
_tions_, pp. 242–256, Cham, 2025. Springer Nature Switzerland. ISBN 978-3-031-95326-2.


Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. _arXiv preprint arXiv:1904.09728_, 2019.


Hongrui Shi and Valentin Radu. Towards federated learning with attention transfer to mitigate
system and data heterogeneity of clients. In _Proceedings_ _of_ _the_ _4th_ _International_ _Workshop_ _on_
_Edge Systems,_ _Analytics and Networking_, EdgeSys ’21, pp. 61–66, New York, NY, USA, 2021.
Association for Computing Machinery. ISBN 9781450382915. doi: 10.1145/3434770.3459739.
[URL https://doi.org/10.1145/3434770.3459739.](https://doi.org/10.1145/3434770.3459739)


Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, L´eonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ram´e, et al. Gemma
2: Improving open language models at a practical size. _arXiv preprint arXiv:2408.00118_, 2024.


12


Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee
Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and
efficient foundation language models. _arXiv preprint arXiv:2302.13971_, 2023.


Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan,
and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. _Transactions_ _on_ _Machine_ _Learning_ _Research_, 2024a. ISSN 2835-8856. URL [https:](https://openreview.net/forum?id=ehfRiF0R3a)
[//openreview.net/forum?id=ehfRiF0R3a.](https://openreview.net/forum?id=ehfRiF0R3a)


Xun Wang, Jing Xu, Franziska Boenisch, Michael Backes, and Adam Dziedzic. POST: A framework
for privacy of soft-prompt transfer. In _ICML 2024 Workshop on Foundation Models in the Wild_,
2024b. [URL https://openreview.net/forum?id=newkzMhqOO.](https://openreview.net/forum?id=newkzMhqOO)


Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du,
Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In _Interna-_
_tional_ _Conference_ _on_ _Learning_ _Representations_, 2022. URL [https://openreview.net/](https://openreview.net/forum?id=gEZrGCozdqR)
[forum?id=gEZrGCozdqR.](https://openreview.net/forum?id=gEZrGCozdqR)


Feijie Wu, Zitao Li, Yaliang Li, Bolin Ding, and Jing Gao. FedbiOT: a solution for federated
large language model fine-tuning with intellectual property protection, 2024. URL [https://](https://openreview.net/forum?id=i5da6iedW8)
[openreview.net/forum?id=i5da6iedW8.](https://openreview.net/forum?id=i5da6iedW8)


Guangxuan Xiao, Ji Lin, and Song Han. Offsite-tuning: Transfer learning without full model. _arXiv_
_preprint arXiv:2302.04870_, 2023.


An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li,
Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. _arXiv_ _preprint_
_arXiv:2407.10671_, 2024.


Lang Yu, Qin Chen, Jiaju Lin, and Liang He. Black-box prompt tuning for vision-language model
as a service. In _Proceedings_ _of_ _the_ _Thirty-Second_ _International_ _Joint_ _Conference_ _on_ _Artificial_
_Intelligence_, IJCAI ’23, 2023. ISBN 978-1-956792-03-4. doi: 10.24963/ijcai.2023/187. URL
[https://doi.org/10.24963/ijcai.2023/187.](https://doi.org/10.24963/ijcai.2023/187)


Jianyi Zhang, Saeed Vahidian, Martin Kuo, Chunyuan Li, Ruiyi Zhang, Tong Yu, Guoyin Wang, and
Yiran Chen. Towards building the federatedgpt: Federated instruction tuning. In _ICASSP 2024 -_
_2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_, pp.
6915–6919, 2024. doi: 10.1109/ICASSP48485.2024.10447454.


Kaiyan Zhang, Ning Ding, Biqing Qi, Xuekai Zhu, Xinwei Long, and Bowen Zhou. CRaSh: Clustering, removing, and sharing enhance fine-tuning without full large language model. In Houda
Bouamor, Juan Pino, and Kalika Bali (eds.), _Proceedings_ _of_ _the_ _2023_ _Conference_ _on_ _Empir-_
_ical_ _Methods_ _in_ _Natural_ _Language_ _Processing_, pp. 9612–9637, Singapore, December 2023a.
Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.597. URL
[https://aclanthology.org/2023.emnlp-main.597/.](https://aclanthology.org/2023.emnlp-main.597/)


Kaiyan Zhang, Ning Ding, Biqing Qi, Xuekai Zhu, Xinwei Long, and Bowen Zhou. Crash: Clustering, removing, and sharing enhance fine-tuning without full large language model. _arXiv preprint_
_arXiv:2310.15477_, 2023b.


Yuanhang Zheng, Zhixing Tan, Peng Li, and Yang Liu. Black-box prompt tuning with subspace
learning. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_, 32:3002–3013,
2024. doi: 10.1109/TASLP.2024.3407519.


Ligeng Zhu, Zhijian Liu, and Song Han. Deep leakage from gradients. In _NeurIPS_, volume 32,
2019.


13


A MORE DETAILS OF OUR METHOD


A.1 LAYERDROP


The pseudocode of the LayerDrop algorithm is shown below.


**Algorithm 2** LayerDrop
**Input:** Original model _M_,dropout rate _β_
**Output:** a list of layers

1: Get the layers of model: layers _←|M|_
2: _m, k_ _←_ len(layers) _, ⌊_ len(layers) _× β⌋_
3: stride _←_ ( _m −_ 1) _/_ ( _k −_ 1)
4: **for** _j_ _←_ 0 to _k −_ 1 **do**
5: _ij_ _←⌊j ×_ stride _⌋_
6: **end for**
7: **return** layers[ _i_ 0 _, . . ., ik−_ 1]


A.2 MODEL DETAILS


We conducted experiments on three commonly used LLMs: Qwen2-1.5B-Instruct [2], Gemma-2-2bit [3], and Llama-3.2-3B-Instruct [4] . The architectural hyperparameters, training data size, and vocabulary size of these models are detailed as Table 4.


Table 4: Details of large language models.


Models Qwen2-1.5B-Instruct Gemma-2-2b-it Llama-3.2-3B-Instruct


Hidden Size 1,536 2,304 3,072
Layers 28 26 28
Query Heads 12 8 24
Key Value Heads 2 4 8
Head Size 128 256 128
Vocabulary Size 151,936 256,000 128,256
Trained Tokens 7T 2T 9T


A.3 DATASET DETAILS


Table 5 summarizes the statistics of the downstream task datasets, while their corresponding instruction formats are presented in Tables 8.


Table 5: The statistics of downstream task datasets.

Datasets OBQA SIQA ARC-c WebQs


Train Data Num 5.0K 33.4K 2.3K 3.8K
Test Data Num 500 1,954 1,172 2,032
Answer Option Option Option Option


A.4 IMPLEMENTATION DETAILS


In the CPKD stage, the emulator is distilled for one epoch on the initial 12 _._ 5% of the first Pileuncopyright chunk with a learning rate of 4 _e −_ 6; the loss weights ( _w_ 1, _w_ 2, _w_ 3 ) are set to 1, 10,
and 30, respectively. For the LLE stage, we experiment with two learning rates, 1 _e −_ 6 and 2 _e −_ 6,


[2https://huggingface.co/Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)
[3https://huggingface.co/google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)
[4https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)


14


Table 6: Results of ablation experiments on loss landscape elevation methods. NLM denotes elevation using negative language modeling loss. Best in **bold** .


**OBQA** **SIQA** **ARC-c** **WebQs**
**DR** **Method**
Acc( _↑_ ) CPL( _↓_ ) Acc( _↑_ ) CPL( _↓_ ) Acc( _↑_ ) CPL( _↓_ ) Acc( _↑_ ) CPL( _↓_ )


NLM 24.80 64.03 49.90 **69.94** 20.56 **59.62** 13.29 0.00
0.2
LLE **33.87** **49.40** **53.34** 75.70 **41.98** 61.00 **28.75** **0.00**


NLM 23.00 66.19 42.68 **69.83** 20.03 63.52 5.86 0.00
0.5
LLE **34.20** **49.88** **50.04** 76.31 **40.44** **49.30** **24.15** **0.00**


|1.0 0|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||O<br>SI<br>|O<br>SI<br>|O<br>SI<br>|BQA<br>QA<br>|
|.6<br>.8<br><br>||||~~A~~<br>W||~~A~~<br>W|~~C-c~~<br>ebQs|
|.6<br>.8<br><br>||||~~A~~<br>W||~~A~~<br>W||
|||||||||
|.2<br>.4<br><br>||||||||
|.2<br>.4<br><br>||||||||
|.0<br><br><br><br>||||||||
|.0<br><br><br><br>||||||||


Figure 5: Variations in the average accuracy of LLEOT across different noise intensities on Qwen21.5B (dropout rate = 0.5). Here, _σ_ denotes the standard deviation of Gaussian noise applied to the
fine-tuned soft prompts.


using the initial 1% of the same data chunk. We select and report the results from the emulator that
achieves the best performance. In the comparative experiments, the elevation margin _H_ is set to 4.
During downstream fine-tuning, we use a soft prompt of length 5 and conduct a grid search over
learning rates, reporting the best-performing run. The search grids are _{_ 1 _e −_ 1 _,_ 7 _e −_ 2 _,_ 3 _e −_ 2 _,_ 3 _e −_
3 _,_ 1 _e_ _−_ 3 _}_ for Qwen2-1.5B-Instruct and Gemma-2-2b-it, and _{_ 5 _e_ _−_ 3 _,_ 1 _e_ _−_ 3 _,_ 5 _e_ _−_ 4 _,_ 1 _e_ _−_ 4 _,_ 5 _e_ _−_ 5 _}_
for Llama-3.2-3B-Instruct. All experiments were conducted on two NVIDIA A800 GPUs.


B ADDITIONAL EXPERIMENTS AND ANALYSIS


B.1 ABLATION EXPERIMENTS ON LOSS LANDSCAPE ELEVATION METHODS


We conduct an ablation study on the method of loss landscape elevation to demonstrate that our
proposed strategy effectively preserves gradient similarity between the elevated emulator and the
original model during prompt tuning. For comparison, we use an unconstrained elevation method
as a baseline. Specifically, this baseline directly elevates the emulator’s loss landscape using the
negative language model loss. The results are presented in Table 6.


As shown in Table 6, compared to the unconstrained elevation method, our proposed approach
significantly improves the final accuracy while effectively protecting model capability privacy. This
strongly indicates that our elevation strategy successfully maintains gradient similarity between the
emulator and the original model during prompt tuning, thereby enabling the fine-tuned soft prompts
to be highly applicable to the original model.


B.2 COMPATIBILITY WITH DATA PRIVACY STRATEGIES


To verify that the LLEOT framework is orthogonal to data privacy strategies, we incorporate the
widely used randomization privacy protection strategy (Zhu et al., 2019; Kang et al., 2023) into


15


LLEOT. Specifically, after prompt tuning, the data owner adds Gaussian noise to the soft prompt
before uploading it to the model owner. This method is known to significantly reduce the success rate
of gradient inversion attacks (Zhu et al., 2019), thereby preventing the model owner from deducing
private data.


Figure 5 illustrates the variation in the original model’s average accuracy with the introduction
of noise intensity. Unexpectedly, the noise added to the soft prompts has a negligible impact on
model performance. We attribute that this robustness stems from the high smoothness of the original model’s input embedding space, resulting from its pre-training on massive amounts of data. This
smoothness ensures that small perturbations to the embedding vectors do not significantly alter the
model’s output. Therefore, the randomization privacy protection strategy can be integrated into the
LLEOT framework, enhancing data privacy at a negligible cost to performance.


B.3 COMPARISON OF ADAPTER SIZES


As shown in Table 7, our method employs an adapter with a parameter count that is significantly
lower than the existing methods, thereby drastically reducing the consumption of computational
resources.


Table 7: Parameter counts of adapters for different methods.


Method OT CRaSh Ours


Qwen2-1.5B-Instruct 187.2M 187.2M 7.6K
Gemma-2-2b-it 311.5M 311.5M 11.5K
Llama-3.2-3B-Instruct 402.7M 402.7M 15.4K


C THE USE OF LLMS


In the preparation of this manuscript, we employed a large language model (LLM), specifically
Gemini 2.5 Pro (Comanici et al., 2025), as a writing aid. The LLM’s role was explicitly restricted
to language refinement and did not involve any facet of the research conceptualization or scientific
methodology. Our process consisted of providing the LLM with drafts and specific sentences. We
then utilized the model’s suggestions to polish sentence construction, enhance clarity and flow, and
verify grammatical accuracy in the final text. It is essential to declare that all central scientific
contributions—including the motivation for this study, the definition of the model capability privacy
concept and its associated metric, the algorithmic architecture and theoretical analysis of LLEOT,
and the experimental design and interpretation of results—are exclusively the work of the human
authors. The LLM was not utilized to formulate scientific claims, hypotheses, or conclusions. In
compliance with ICLR policy, the authors have fastidiously reviewed, edited, and confirmed all
content in this paper. We assume complete responsibility for the final manuscript, encompassing its
scientific precision and integrity.


D PROOF OF THEOREM 1.


_Proof._ We first show that LLE effectively degrades the emulator’s inference ability. From the definition of cross-entropy loss and Eq. 6, we obtain


_LE_ ( _P_ ; _x_ ; _y_ ) _−LM_ ( _P_ ; _x_ ; _y_ )


= _H_ _>_ 0 _._ (12)


Here, _n_ denotes the number of tokens to predict in _x_, _pE_ ( _xi|P, x_ 1: _i−_ 1) denotes the probability of the
emulator correctly predicting the _i −_ _th_ token given the soft prompt and the preceding _i −_ 1 tokens.
Defining _p_ ˆ _E_ ( _x|P_ ) = [�] _i_ _[n]_ =1 _[p][E]_ [(] _[x][i][|][P, x]_ [1:] _[i][−]_ [1][)][ and] _[p]_ [ˆ] _[M]_ [(] _[x][|][P]_ [) =][ �] _[n]_ _i_ =1 _[p][M]_ [(] _[x][i][|][P, x]_ [1:] _[i][−]_ [1][)][, Equation 12]


16


= _−_ _n_ [1]


_n_


- log( _pE_ ( _xi_ _| P, x_ 1: _i−_ 1)) + _n_ [1]

_i_ =1


_n_


_n_

- log( _pM_ ( _xi_ _| P, x_ 1: _i−_ 1))


_i_ =1


can be transformed into:


_p_ ˆ _E_ ( _x|P_ ) = e _[−][nH]_ _p_ ˆ _M_ ( _x|P_ ) (13)


To analyze the impact of this loss difference on model performance, we consider the perplexity
(PPL), a standard metric for evaluating language models. Perplexity is defined as:


Similarly, for the original model, we have:

_p_ ˆ _M_ ( _x|P_ ) = PPL _[−]_ _M_ _[n][.]_ (17)


By substituting Equation 16 and Equation 17 into Equation 13, we can express the relationship
between the perplexities of the two models:


PPL _E_ = e _[H]_ _·_ PPL _M._ (18)


It shows that the emulator’s PPL is exponentially greater than the original model’s by a factor of e _[H]_ .
Since lower PPL indicates better performance, a larger _H_ will lead to a significantly higher PPL for
the emulator, thereby degrading its inference capabilities.


In addition, we show that LLE maintains the emulator’s gradient guidance consistent with that of the
original model. Specifically, the emulator’s gradient with respect to soft prompts can be expressed
as:


_∇P LE_ ( _P_ ; _x_ ; _y_ ) = _∇P_ ( _LM_ ( _P_ ; _x_ ; _y_ ) + _H_ ) = _∇P LM_ ( _P_ ; _x_ ; _y_ ) + _∇P H_ = _∇P LM_ ( _P_ ; _x_ ; _y_ ) _,_
(19)


the gradient vectors of the emulator and the original model are identical. During prompt tuning, the
emulator and the original model exhibit consistent gradient optimization directions and magnitudes
at each step, ultimately converging to the same optimal soft prompt.


17


  -   = exp _−_ [1] = _p_ ˆ _[−]_ [1] _[/n]_ _._ (14)

_n_ [log(ˆ] _[p]_ [)]


_n_


PPL = exp


_−_ [1]


_n_


log( _pi_ )

_i_ =1


From Equation 14, the perplexity of the emulator can be expressed as:


_n_


PPL _E_ = exp


= exp


_−_ [1]


_−_ [1]


_n_ [log(]


_n_


log( _pE_ ( _xi_ _| P, x_ 1: _i−_ 1))

_i_ =1


_n_


_pE_ ( _xi_ _| P, x_ 1: _i−_ 1))

_i_ =1


This can be rewritten as:


  -   = exp _−_ [1]

_n_ [log(ˆ] _[p][E]_ [(] _[x][|][P]_ [))]

= _p_ ˆ _E_ ( _x|P_ ) _[−]_ [1] _[/n]_ _._ (15)


_p_ ˆ _E_ ( _x|P_ ) = PPL _[−]_ _E_ _[n][.]_ (16)


Table 8: Instructions format of downstream task dataset

_OBQA_


What happens when mercury is placed in water?
_it sinks._


Which is a good source of nutrients for a mushroom?
_a cut peony._


_SIQA_


Q: Sydney was a school teacher and made sure their students learned well. How would you
describe Sydney?
A:
_As someone that takes teaching seriously._


Q: Kendall’s dog was overweight so they walked it five miles. Why did Kendall do this?
A:
_start an exercise regimen._


_ARC-c_


Question: What do cells break down to produce energy?
Answer:
_food._


Question: How are the particles in a block of iron affected when the block is melted?
Answer:
_The particles move more rapidly._


_WebQs_


Question: what is nina dobrev nationality?
Answer:
_Bulgaria._


Question: what electorate does anna bligh represent?
Answer:
_Electoral district of South Brisbane._


18