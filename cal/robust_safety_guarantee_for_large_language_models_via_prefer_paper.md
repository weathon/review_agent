# ROBUST SAFETY GUARANTEE FOR LARGE LANGUAGE MODELS VIA PREFERENCE-AUGMENTED DISTRIBUTIONAL ALIGNMENT


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Domain-specific fine-tuning of large language models (LLMs) often compromises
their safety alignment, leading to unsafe generations. Existing approaches largely
rely on distributional alignment, enforcing token-level similarity between pre- and
post-fine-tuned models. However, this neglects the semantic nature of text generation and can weaken the model’s reasoning and robustness. To address this
limitation, we propose a preference-based alignment framework that complements
distributional alignment by biasing the fine-tuned model toward the safe outputs
of the pre-trained model, rather than strictly preserving distributional similarity.
Simulation results show that preference alignment produces consistent safe outputs even when the underlying distributions differ. Extensive experiments on multiple fine-tuning attack datasets and utility benchmarks further demonstrate that
our method substantially improves safety with only minor degradation in utility.
This achieves a more favorable balance between safety and utility, and significantly
enhances robustness against adversarial fine-tuning.


1 INTRODUCTION


Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, from
content creation to complex reasoning (Touvron et al., 2023a;b; Team, 2023). However, their powerful functionality also raises significant safety concerns, as they may be misused to generate harmful,
biased, or unsafe content (Qi et al., 2023). Ensuring safety alignment—training models to follow
human values and safety standards—has thus become a central challenge in artificial intelligence.


A widely used approach is supervised fine-tuning (SFT) (Wei et al., 2021), which improves rejection
of harmful queries by training on curated datasets. Despite its effectiveness, SFT typically yields
shallow alignment, making safety behaviors fragile and easily forgotten during downstream domain
fine-tuning, which often results in unsafe responses. To address this issue, Qi et al. proposed constrained supervised fine-tuning (Constrained SFT) (Qi et al., 2024), which enforces deeper tokenlevel alignment to enhance robustness against fine-tuning attacks. Nevertheless, constrained SFT
relies mainly on distribution alignment, constraining models only at the per-token probability level
while overlooking the semantic nature of text generation. This limits robustness, leaving models
prone to merely imitating safe distributions rather than developing intrinsic safety awareness.


To overcome these limitations, we propose a new framework that integrates preference alignment
(Xu et al., 2025) with distributional alignment. Our approach introduces preference signals on top of
token-level probability constraints, encouraging fine-tuned models to favor the safe outputs of their
pre-trained counterparts rather than strictly preserving distribution similarity. An auxiliary loss function formalizes this mechanism, enabling stronger safety alignment while preserving utility. Simulation experiments show that preference alignment can produce consistent safe outputs even when
underlying distributions diverge. Furthermore, evaluations on multiple fine-tuning attack datasets
( **Harmful Example Attacks** that introduce toxic data to elicit unsafe responses, **Identity Shifting**
**Attacks** that alter model identity leading to biased or inaccurate outputs, and **Backdoor Poisoning**
**Attacks** —both trigger-free and trigger-based—that insert poisoned data to degrade performance on
specific inputs, Qi et al. (2024)) and utility benchmarks demonstrate that our method substantially
improves safety with only minor utility degradation.


1


In summary, our contributions are threefold: (1) We identify the limitations of distribution-only
alignment in maintaining safety under domain fine-tuning. (2) We propose a preference-augmented
framework that combines preference and distributional alignment for robust safety. (3) Both theoretical analysis and extensive experiments are provided to validate that our method achieves improved
safety with minimal loss of utility.


2 RELATED WORK


The field of LLM safety alignment has advanced rapidly, with multiple approaches proposed to steer
models toward safe behaviors.


**Reinforcement** **Learning** **from** **Human** **Feedback** **(RLHF):** RLHF has become a dominant
paradigm for aligning LLMs with complex human values (Ouyang et al., 2022; Bai et al., 2022).
It first trains a reward model on human preference data, where annotators compare and rank different outputs. The LLM policy is then fine-tuned using reinforcement learning to maximize the
reward model’s score. Despite its effectiveness, RLHF is a multi-stage process that can be unstable
and computationally expensive. Moreover, the reward model itself can be exploited through “reward hacking,” where the LLM maximizes the reward signal without genuinely adhering to intended
values (Gao et al., 2023).


**Direct Preference Optimization (DPO):** To mitigate the complexity and instability of RLHF, recent
work has proposed Direct Preference Optimization (DPO) as a simpler and more stable alternative
(Rafailov et al., 2023). DPO reformulates alignment as a classification problem over human preference data, allowing direct policy fine-tuning without explicit reward modeling or complex RL loops.
DPO has shown strong performance, often matching or surpassing RLHF. However, like RLHF, its
effectiveness depends heavily on the quality and coverage of the preference dataset.


**Supervised Fine-Tuning (SFT) or Constrained SFT (CSFT):** Compared with RLHF and DPO, supervised fine-tuning (SFT) offers a more direct and cost-effective approach. The core idea of SFT is
to fine-tune base models on high-quality datasets of prompt-response pairs (Wei et al., 2021). While
effective in enabling models to imitate the response patterns in training data, SFT often struggles to
generalize safety principles to unseen prompts. Its safety largely relies on fixed refusal templates,
leading to a rather shallow form of alignment. Building on this, subsequent research proposed Constrained Supervised Fine-Tuning (CSFT) (Qi et al., 2024), specifically designed for safety alignment.
CSFT leverages datasets of harmful prompts paired with safe refusal responses and constrains the
model at the token-level probability distribution, so that its generation process more closely matches
the expected safe responses. However, since it primarily emphasizes distributional similarity, CSFT
often overlooks semantic aspects of generation, which limits its robustness in complex attack scenarios.


Building on these insights, we further explore how to combine distributional alignment with preference alignment to achieve stronger safety robustness while preserving task utility. To further investigate the robustness of our approach, we follow the analytical perspective introduced by Xu
(2025). He analyzes policy instability in RL-trained LLMs via reward-to-policy continuity. Brittleness arises from non-unique optima in degenerate tasks, enabling discontinuous shifts from minor
reward changes. Entropy regularization restores Lipschitz continuity for robustness, at stochasticity’s
cost. Unifies explanations for failures like deceptive reasoning and instruction ignoring.


**Preference-Augmented Conditional Supervised Fine-Tuning (CSFT+PA):** To address these limitations, we propose augmenting CSFT with preference alignment (Table 1). Our framework introduces auxiliary loss terms that bias the fine-tuned model toward the safe outputs of the pre-trained
model, rather than strictly enforcing distributional similarity. This preference-based enhancement
not only strengthens safety alignment but also preserves task utility, significantly improving robustness against fine-tuning attacks.


3 METHOD


This section provides a detailed explanation of the mathematical principles behind our approach. We
first present the overall loss function and then gradually explain the design principles and implementation details of each component, including their theoretical motivations and practical implications.


2


Table 1: Comparison of different alignment methods, the proposed **CSFT+PA** considers both distributional alignment and preference alignment.


**Method** **Distributional Alignment** **Preference Alignment**


SFT (Wei et al., 2021)
CSFT (Qi et al., 2024)
**CSFT+PA (Ours)**


3.1 NOVEL LOSS FUNCTION FOR SAFETY AALIGNMENT


Our training objective combines two types of loss functions: the Constrained Supervised Fine-tuning
(CSFT) loss and the Preference Alignment (PA) loss. The CSFT loss is designed to achieve tokenlevel _distributional alignment_, ensuring that the model’s token probability distributions remain close
to those of the reference safety-aligned model. By contrast, the PA loss enforces token-level _prefer-_
_ence alignment_, encouraging the model to prefer the safety-aligned outputs over its own generated
outputs. In this sense, the PA loss naturally falls within the broader category of probabilistic alignment.


Formally, the overall loss function is defined in Equation (1):


_𝐿_ Total( _𝜃_ ) = _𝐿_ CSFT ( _𝜃_ ) + _𝛿_ epoch · _𝐿_ PA( _𝜃_ ) (1)


As shown in Equation (1), the total loss is composed of two main terms: _𝐿_ CSFT( _𝜃_ ), the Constrained
Supervised Fine-tuning loss proposed by Qi et al. (2024) in Equation (2), and _𝐿_ PA( _𝜃_ ), our newly
introduced Preference Alignment loss (see Equation (5)). The balancing factor _𝛿_ epoch serves as a dynamic scheduling coefficient, gradually increasing the influence of the PA loss as training progresses,
while ensuring stable optimization in the early epochs.


The CSFT loss (Qi et al., 2024) is defined in Equation (2), it enforces token-level distributional
alignment by minimizing the discrepancy between the log-probabilities of the current model and the
safety-aligned model.


3.2 DESIGN PA LOSS


The Preference Alignment (PA) loss is motivated by the need to make the output distribution of
the current model _𝜋𝜃_ better reflect the token-level preferences of the safety-aligned model _𝜋_ aligned.
Concretely, for a given token position _𝑡_, we want the probability assigned by the current model to the
aligned token _𝑦𝑡,_ aligned to be higher than that assigned to its own token _𝑦𝑡,𝜃_ . This token-wise comparison provides fine-grained guidance, complementing the broader distributional alignment enforced
by the CSFT loss.


We first define the token-level preference probability as shown in Equation (3):

( ))
) exp [(] log _𝜋𝜃_ _𝑦𝑡,_ aligned | _**x**_ _,_ _**y**_ _<𝑡_
P [(] _𝑦𝑡,_ aligned ≻ _𝑦𝑡,𝜃_ | _**x**_ _,_ _**y**_ _<𝑡_ = ~~(~~ ~~))~~ ~~(~~ ~~(~~ ~~))~~ (3)

exp ~~[(]~~ log _𝜋𝜃_ _𝑦𝑡,_ aligned | _**x**_ _,_ _**y**_ _<𝑡_ + exp log _𝜋𝜃_ _𝑦𝑡,𝜃_ | _**x**_ _,_ _**y**_ _<𝑡_


By simplifying this expression, we obtain the sigmoid-based formulation in Equation (4):

) [ ( ) ( )]
P [(] _𝑦𝑡,_ aligned ≻ _𝑦𝑡,𝜃_ | _**x**_ _,_ _**y**_ _<𝑡_ = _𝜎_ log _𝜋𝜃_ _𝑦𝑡,_ aligned | _**x**_ _,_ _**y**_ _<𝑡_    - log _𝜋𝜃_ _𝑦𝑡,𝜃_ | _**x**_ _,_ _**y**_ _<𝑡_ (4)


This formulation indicates that the preference score increases as the log-probability difference between the aligned token and the model’s own token increases.


3


{


[
∑| _**y**_ |


]}


_𝑤𝑡_  - log _𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ )
_𝑡_ =1


_𝐿_ CSFT( _𝜃_ ) = min
_𝜃_


−E( _**x**_ _,_ _**y**_ )∼ _**D**_


_𝜃_ (2)

{ [ ( _𝑡_ =1 )]}
_𝑤𝑡_ = 2 1 − _𝜎_ _𝛽𝑡_ log _𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ ) − log _𝜋_ aligned( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ )


As shown in Equation (6), if the discrepancy between the two distributions is large, _𝜇𝑡_ increases, amplifying the gradient contribution of the PA loss at that token. Conversely, when the two distributions
are already similar, _𝜇𝑡_ decreases, allowing the CSFT loss to dominate the learning process.


**Scheduling Coefficient** _𝛿_ **epoch.** To control the relative importance of the PA loss throughout training,
we introduce the scheduling coefficient _𝛿_ epoch, defined in Equation (7):

epoch
_𝛿_ epoch = 0 _._ 1 + 0 _._ 2 × (7)
max_epoch

As Equation (7) shows, _𝛿_ epoch increases linearly with the number of epochs, gradually raising the
contribution of the PA loss. In the initial epochs, training relies primarily on the CSFT loss, ensuring
stability. As training progresses, the PA loss plays a larger role, but its maximum contribution is
capped at 30% of the total loss.


3.3 DISCUSSION AND SUMMARY


A schematic overview of the proposed algorithm is presented in Figures 1. The combination of the
CSFT loss and the PA loss provides a complementary training mechanism. On the one hand, the
CSFT loss (2) focuses on _distributional alignment_, ensuring that the probability distributions of the
current model remain close to those of the safety-aligned model across all tokens. This enforces
global stability and prevents the model from deviating excessively during the early stages of training. On the other hand, the PA loss (5) emphasizes _preference alignment_ at the token level, directly
encouraging the model to prefer outputs chosen by the safety-aligned model. By incorporating the
adaptive weight _𝜇𝑡_ (6) and the scheduling coefficient _𝛿_ epoch (7), the PA loss adaptively modulates its
influence based on both distributional divergence and training progress.


In summary, the CSFT loss serves as a stabilizing force that maintains consistency with the reference distribution, while the PA loss introduces fine-grained, preference-based guidance that enhances
alignment at the token level. Their integration within the total loss function (1) enables the model to
balance stability and flexibility: it first learns robust distributional patterns under CSFT supervision
and then progressively incorporates token-level preferences through the PA mechanism. This synergy constitutes the core of our probabilistic alignment framework and underpins the effectiveness
of our training approach.


4 THEORETICAL RESULTS: CONVERGENCE AND ROBUSTNESS


4.1 CONVERGENCE ANALYSIS


To establish convergence guarantees, we impose the following standard assumptions in stochastic
optimization:


1. **Assumptions on the Objective Function and Gradients:** These assumptions ensure the
smoothness and reliability of the gradients, preventing explosions and modeling stochasticity in approximations, which are essential for convergence in stochastic settings, as used in
Bottou et al. (2018) and Garrigos & Gower (2023).


4


Based on this token-wise preference probability, we define the PA loss as in Equation (5):


{


[
∑| _**y**_ |


_𝐿_ PA( _𝜃_ ) = min
_𝜃_


−E( _**x**_ _,_ _**y**_ )∼ _𝐷_


]}

∑| _**y**_ |

log _𝜎_ [(] _𝜇𝑡_  - [(] log _𝜋𝜃_ ( _𝑦𝑡,_ aligned| _**x**_ _,_ _**y**_ _<𝑡_ ) − log _𝜋𝜃_ ( _𝑦𝑡,𝜃_ | _**x**_ _,_ _**y**_ _<𝑡_ ) [))]
_𝑡_ =1


(5)
As shown in Equation (5), the PA loss penalizes the model when it fails to assign higher probability
mass to the aligned token. Importantly, this mechanism only activates when there is a discrepancy
between the outputs of the current model and the reference model, thereby avoiding redundant constraints.


**Adaptive Weight** _𝜇𝑡_ **.** The adaptive weight _𝜇𝑡_ plays a critical role in modulating the strength of the
PA loss. As defined in Equation (6), it is determined by the KL divergence between the current model
distribution and the safety-aligned model distribution:

( )
_𝜇𝑡_ = _𝐷_ KL _𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ ) ∥ _𝜋_ aligned( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ ) (6)


**Adaptive Weight** _𝜇𝑡_ **.** The adaptive weight _𝜇𝑡_ plays a critical role in modulating the strength of the
PA loss. As defined in Equation (6), it is determined by the KL divergence between the current model
distribution and the safety-aligned model distribution:


Figure 1: Training pipeline with token-level distributional alignment and scheduled preference optimization.


       - **Bounded Gradients:** For some constant _𝐺>_ 0, ∥∇ _𝜃_ log _𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ )∥≤ _𝐺._

       - **Lipschitz Continuity of Gradients:** For some _𝐿>_ 0, ∥∇ _𝐿_ Total( _𝜃_ 1) −∇ _𝐿_ Total( _𝜃_ 2)∥≤
_𝐿_ ∥ _𝜃_ 1 − _𝜃_ 2 ∥ _._

       - **Unbiased and Bounded Gradient Noise:** For stochastic gradient _𝑔_ ( _𝜃_ ),


E[ _𝑔_ ( _𝜃_ ) | _𝜃_ ] = ∇ _𝐿_ Total( _𝜃_ ) _,_ E[∥ _𝑔_ ( _𝜃_ ) −∇ _𝐿_ Total( _𝜃_ )∥ [2] | _𝜃_ ] ≤ _𝜎_ [2] _._

2. **Learning Rate Schedule:** The step sizes { _𝜂𝑘_ } satisfy [∑][∞] _𝑘_ =1 _[𝜂][𝑘]_ [=] [∞] _[,]_ [∑][∞] _𝑘_ =1 _[𝜂]_ [2] _𝑘_ _[<]_ [∞] _[.]_ [ This]
schedule allows the algorithm to explore the parameter space sufficiently while ensuring
the steps diminish to promote convergence, a foundational condition in stochastic approximation methods, as introduced in Robbins & Monro (1951), and applied in Bottou et al.
(2018).


3. **Model-Specific** **Bounds:** These bounds prevent degenerate probabilities and divergences
in the policy, ensuring well-behaved importance weights and non-zero action probabilities,
which are critical in policy-based reinforcement learning and related methods, as assumed
in Schulman et al. (2017) and Xie et al. (2021).


       - **Bounded** **Weights** **and** **Divergences:** There exist constants _𝑊, 𝐷, 𝐾>_ 0 such that
| _𝑤𝑡_ | ≤ _𝑊_, | log _𝜋𝜃_ ( _𝑦𝑡,_ aligned| _**x**_ _,_ _**y**_ _<𝑡_ ) − log _𝜋𝜃_ ( _𝑦𝑡,𝜃_ | _**x**_ _,_ _**y**_ _<𝑡_ )| ≤ _𝐷_, and _𝜇𝑡_ ≤ _𝐾_ .

       - **Probability Lower Bound:** For some _𝜖>_ 0, _𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ ) ≥ _𝜖._


**Theorem** **4.1** (Convergence Guarantee) **.** _Under_ _Assumptions_ _1–3,_ _the_ _stochastic_ _gradient_ _descent_
_updates_
_𝜃_ _𝑘_ +1 = _𝜃_ _𝑘_                   - _𝜂𝑘𝑔_ ( _𝜃_ _𝑘_ )

_satisfy_

[
lim inf ∥∇ _𝐿Total_ ( _𝜃_ _𝑘_ )∥ [2][]] = 0 _._
_𝑘_ →∞ [E]

_That is, the algorithm converges to a stationary point of 𝐿Total_ ( _𝜃_ ) _in expectation._


The proof follows the standard stochastic optimization framework with bounded gradients and diminishing learning rates. All technical derivations are deferred to the Appendix.


5


4.2 ROBUSTNESS ANALYSIS


**Definition** **4.1** (Robustness) **.** _A_ _loss_ _function_ _𝐿_ ( _𝜃_ ) _is_ _said_ _to_ _be_ _robust_ _if,_ _under_ _perturbations_ _of_
_the training distribution_ _𝐷_ _with intensity 𝜖>_ 0 _,_ _the perturbed minimizer_ _𝜃_ [∗] _[,𝜖]_ _remains close to the_
_original minimizer 𝜃_ [∗] _._ _Formally, robustness holds if there exists a constant 𝐾>_ 0 _such that_

∥ _𝜃_ [∗] _[,𝜖]_                     - _𝜃_ [∗] ∥≤ _𝐾𝜖,_

_where_ ∥·∥ _is the Euclidean norm in parameter space._ _This ensures that the induced policy 𝜋𝜃_ _exhibits_
_bounded deviation under perturbations, preventing abrupt ’policy cliffs’ as studied in reward-policy_
_mappings of large language models._ _This definition aligns with broader notions of robustness in ma-_
_chine learning, where stability is maintained under varying conditions or perturbations, as discussed_
_in Bousquet & Elisseeff (2002)._


To establish robustness guarantees, we make the following standard assumptions, weaker than strong
convexity:


    - **Convexity.** The loss function _𝐿_ Total( _𝜃_ ) is convex. This assumption ensures that the optimization landscape has no spurious local minima and that any local minimum is global,
simplifying convergence analysis in theoretical settings. Convexity is a foundational assumption in many optimization studies, though relaxed in practice for deep learning; it has
been extensively used in works such as Boyd & Vandenberghe (2004).

    - **Lipschitz gradient.** Its gradient is _𝐿_ -Lipschitz continuous, i.e.,

∥∇ _𝐿_ Total( _𝜃_ 1) −∇ _𝐿_ Total( _𝜃_ 2)∥≤ _𝐿_ ∥ _𝜃_ 1 − _𝜃_ 2 ∥ _._

This condition, also known as L-smoothness, bounds the rate of change of the gradient,
which is crucial for controlling step sizes in gradient-based methods and deriving convergence rates. It is a standard assumption in convergence proofs for deep learning optimizers,
as seen in Nesterov (2004) and Bottou et al. (2018).

    - **Polyak-Łojasiewicz (PL) inequality.** There exists _𝜇>_ 0 such that

1 [≥] _[𝜇]_ [(] _[𝐿]_ [Total][(] _[𝜃]_ [) −] _[𝐿]_ [Total][(] _[𝜃]_ [∗][)][)] _[.]_
2 [∥∇] _[𝐿]_ [Total][(] _[𝜃]_ [)∥][2]

The PL inequality provides a sufficient condition for linear convergence of gradient descent
without requiring strong convexity, making it suitable for analyzing non-convex objectives
that behave well locally. It was originally introduced by Polyak (1963) and Łojasiewicz
(1963), and has been applied to deep learning optimization in Karimi et al. (2016).


These assumptions are more general than strong convexity and are commonly used to analyze deep
learning objectives that are not globally strongly convex but satisfy local well-behaved properties.
Importantly, the inclusion of the alignment regularizer _𝐿_ PA( _𝜃_ ) increases the effective PL constant _𝜇_,
thereby strengthening robustness guarantees.


We now formalize the robustness bound for _𝐿_ Total( _𝜃_ ).
**Theorem** **4.2** (Robustness Bound) **.** _Let_ _𝜃_ [∗] _be_ _the_ _minimizer_ _of_ _𝐿Total_ ( _𝜃_ ) _,_ _and_ _𝜃_ [∗] _[,𝜖]_ _the_ _minimizer_
_under perturbed data distribution 𝐷_ _[𝜖]_ _with noise intensity 𝜖>_ 0 _._ _Suppose the gradient of 𝐿Total_ ( _𝜃_ ) _is_
_𝐿-Lipschitz and the gradient perturbation satisfies_

∥∇ _𝐿Total_ _[𝜖]_ [(] _[𝜃]_ [) −∇] _[𝐿][Total]_ [(] _[𝜃]_ [)∥≤] _[𝜖𝐺.]_


At the minimizers, ∇ _𝐿_ Total( _𝜃_ [∗] ) = 0 and ∇ _𝐿_ Total _[𝜖]_ [(] _[𝜃]_ [∗] _[,𝜖]_ [)] [=] [0.] [Combining] [the] [gradient] [perturbation]
bound with Lipschitz gradient continuity yields

∥∇ _𝐿_ Total( _𝜃_ [∗] _[,𝜖]_ )∥≤ _𝜖𝐺_ ≤ _𝐿_ ∥ _𝜃_ [∗] _[,𝜖]_            - _𝜃_ [∗] ∥ _,_

which implies the stated inequality. The inclusion of _𝐿_ PA( _𝜃_ ) further improves robustness by reducing effective sensitivity to noise, tightening the bound. This result demonstrates that the proposed
loss function is robust to bounded perturbations, with solution deviations scaling linearly in _𝜖_ . The
regularizer _𝐿_ PA strengthens robustness by mitigating degeneracies in the solution space, thereby preventing discontinuous shifts in the learned policy under small distributional changes. This aligns
with recent theoretical analyses on preventing ’policy cliffs’ in large-scale models.


6


_Then,_


∥ _𝜃_ [∗] _[,𝜖]_ - _𝜃_ [∗] ∥≤ _[𝜖𝐺]_

_𝐿_ _[.]_


Table 2: Impact of PA Loss on Model Alignment and the developed PA shows lower KL divergence
and token probability difference.


**Metric** **Cross-Entropy** **PA (ours)** **Rel.** **Change vs.** **Baseline (%)**


KL Divergence 1.4575 0.2562 **+82.4**
Per-Token Probability Diff. 0.0169 0.0051 **+69.8**


5 EXPERIMENTS


5.1 PRE-EXPERIMENT 1: EFFECTIVENESS OF PA LOSS


In Pre-experiment 1, we aim to verify that PA loss can achieve effective probability alignment even
when the architectures of the policy model and the reference model differ significantly. Specifically,
the policy model ( _𝜋𝜃_ ) adopts an LSTM with a single fully connected layer, while the reference model
( _𝜋_ aligned) adopts an LSTM with multiple fully connected layers and residual connections, with about
twice as many parameters. The comparison groups consist of **Group** **1** (training without PA loss,
using only cross-entropy loss) and **Group 2** (training with PA loss).


The results are shown in Table 2 and Figures 2. With PA loss, KL divergence is reduced from 1.4575
to 0.2562 (82.4% improvement relative to baseline), and the per-token probability difference decreases from 0.0169 to 0.0051 (69.8% improvement relative to baseline). These findings demonstrate
the effectiveness of PA loss in probability alignment: even with substantial architectural differences,
PA loss significantly narrows the gap between predictive distributions.


Figure 2: Results of Pre-experiment 1 (lower is better). With PA loss, both KL divergence and token
probability difference are significantly reduced compared to the Cross-Entropy baseline (cf. Table 2).


5.2 PRE-EXPERIMENT 2: EFFECTIVENESS OF CSFT + PA LOSS


In Pre-experiment 2, we further evaluate whether combining CSFT with PA loss achieves better
alignment compared to CSFT loss alone. Similar to Pre-experiment 1, two neural networks with
different architectures are used to simulate _𝜋𝜃_ and _𝜋_ aligned.


The setup features architectural differences with the policy model using a two-layer LSTM and widthpreserving fully connected layers, versus the reference model’s three-layer LSTM and dimensionexpanded fully connected layers, alongside comparison groups: **Group 1** (training with CSFT loss
only) and **Group 2** (training with CSFT + PA loss).


The results, illustrated in Figures 3 to 5, show that CSFT + PA loss significantly improves cosine
similarity, Pearson correlation, distribution overlap, and KL similarity compared to CSFT alone (vs.
CSFT baseline); meanwhile, it also achieves smaller KL divergence and Probability Alignment. This
indicates that CSFT + PA loss achieves superior per-token probability alignment.


7


Figure 3: Results of Pre-experiment 2 (higher is better). CSFT+PA consistently outperforms CSFT
across Pearson Correlation and Cosine Similarity metrics (vs. CSFT baseline).


Figure 4: Results of Pre-experiment 2 (higher is better for Distribution Overlap and KL Similarity).
CSFT+PA outperforms CSFT in terms of distribution-based metrics (vs. CSFT baseline).


5.3 CSFT + PA LOSS EVALUATION ON LARGE LANGUAGE MODELS


To assess the safety and utility of the proposed method in real-world LLM fine-tuning tasks, we
conduct evaluations under adversarial attack scenarios and downstream datasets. The performance
of Llama-2-7B-Chat fine-tuned with our approach is reported in Table 3 and Table 4.


    - **Safety** **evaluation** : We test under Harmful Example (pure_bad) attacks, Identity Shifting
(aoa) attacks, and Backdoor Poisoning attacks, measuring the Attack Success Rate (ASR).

    - **Utility evaluation** : We evaluate on the Samsum dataset and the SQL Create Context dataset
to measure downstream task performance.


5.3.1 ADVERSARIAL ATTACK METHODS


We evaluate the effectiveness of CSFT + PA loss against three types of adversarial attacks: Harmful
Example Attacks, Identity Shifting Attacks, and Backdoor Poisoning Attacks.


    - **Harmful** **Example** **Attacks** : These attacks introduce harmful examples into the training
data, which attempt to mislead the model into generating unsafe or toxic responses.

    - **Identity** **Shifting** **Attacks** : These attacks involve altering the model’s output to shift its
identity, leading to biased or inaccurate outputs.


8


Figure 5: Results of Pre-experiment 2 (lower is better for KL Divergence and Probability Alignment).
CSFT+PA loss leads to smaller distributional differences compared to CSFT alone (vs. CSFT baseline).


    - **Backdoor Poisoning Attacks** : These attacks involve inserting poisoned data points into the
training set, which cause the model to perform poorly on certain inputs. We consider both
_trigger-free_ and _trigger-based_ backdoor attacks.


5.3.2 SAFETY EVALUATION AGAINST FINE-TUNING ATTACKS


The effectiveness of combining CSFT with PA loss in defending against adversarial attacks is summarized in Table 3, where we report the Attack Success Rate (ASR) for each attack category. Overall,
the results demonstrate that CSFT + PA loss consistently and substantially improves safety across
diverse threat models compared to both standard SFT and CSFT baselines.


More specifically, the results across both Llama2 and Gemma1.1 models indicate that:


    - **Harmful Example Attacks** : On Llama2, CSFT + PA suppresses ASR from 88.9% under
SFT to 2.7%, representing a 25.0% relative improvement over CSFT. On Gemma1.1, it reduces ASR from 81.6% to 0.6%, achieving a more pronounced 53.8% relative improvement
over CSFT. These outcomes highlight PA’s potency in curtailing overt harmful behaviors
overlooked by standard supervision.


    - **Identity Shifting Attacks** : On Llama2, CSFT + PA decreases ASR to 7.5%, achieving a
7.4% relative improvement over CSFT. On Gemma1.1, it reaches 8.8% ASR, representing
a 3.3% relative improvement over CSFT. These results demonstrate modest yet consistent
enhancements even where CSFT already mitigates distributional drifts effectively.


    - **Backdoor** **Poisoning** **Attacks** : Substantial reductions emerge in both trigger-free and
trigger-based cases. For the trigger-based scenario, ASR drops to 3.3% on Llama2, representing a 52.3% relative improvement over CSFT, and to 0.9% on Gemma1.1, achieving
a 52.6% relative improvement over CSFT. These affirm PA’s adaptive weighting in amplifying safeguards against latent deviations from safety-aligned references.


Taken together, these findings underscore that the proposed method provides safe and stable defense
across attack categories. Importantly, the improvements are not confined to a specific type of adversarial manipulation but generalize to both data-poisoning and behavioral attacks, which is a key
desideratum for practical safety alignment. Table 3 highlights these results in detail.


5.3.3 UTILITY EVALUATION


In addition to safety, we also evaluate the utility of the proposed approach on downstream tasks. Table 4 presents results on the `Samsum`, `SQL` `Create` `Context`, and `GSM8K` datasets across both Llama2
and Gemma1.1 models. Compared to CSFT, CSFT + PA incurs only minor performance degradation


9


Table 3: Evaluation of Attack Success Rate (ASR) under Fine-tuning Attacks

```
                   Llama2 Gemma1.1
```

**Attack Type**

**SFT** **CSFT** **CSFT+PA** **SFT** **CSFT** **CSFT+PA**


Harmful Example (pure_bad) 88.9 3.6 **2.7** 81.6 1.3 **0.6**
Identity Shifting (aoa) 79.5 8.1 **7.5** 83.6 9.1 **8.8**
Backdoor Poisoning (w/o trigger) 7.6 1.9 **1.5** 2.0 1.5 **0.6**
Backdoor Poisoning (w/ trigger) 90.9 6.9 **3.3** 82.3 1.9 **0.9**


Table 4: Evaluation of Downstream Task Performance

```
                 Llama2 Gemma1.1
```

**Dataset**


**SFT** **CSFT** **CSFT+PA** **SFT** **CSFT** **CSFT+PA**


Samsum 51.7 50.1 **47.1** 51.5 51.9 **48.8**
SQL Create Context 99.1 98.5 **96.3** 99.2 98.6 **96.1**
GSM8K 41.7 37.4 **34.5** 63.3 63.6 **63.0**


(within 8%), indicating that the improvements in safety do not come at the cost of substantial utility
loss.


In particular, the most significant relative performance drop is observed on `GSM8K` for Llama2 (from
37.4% to 34.5%, a 7.8% relative degradation), while performance on `Samsum` and `SQL` `Create`
`Context` decreases only marginally (6.0% and 2.2% relative drops, respectively). On Gemma1.1,
drops are similarly modest: 6.0% on `Samsum`, 2.5% on `SQL` `Create` `Context`, and a minimal 0.9%
on `GSM8K` . Such modest trade-offs are common in safety-alignment methods, and the observed magnitudes are well within acceptable bounds for practical deployment. The overall pattern suggests that
CSFT + PA achieves a favorable safety–utility balance: it yields strong adversarial resistance while
retaining high task competence.


In summary, Tables 3 and 4 demonstrate that CSFT + PA substantially strengthens safety against a
wide range of adversarial attacks, with the maximum reduction in ASR reaching 52.6%. At the same
time, the approach preserves downstream task performance with only minimal degradation. This
balance between safety and utility is crucial for real-world applications, where adversarial resistance
must be achieved without sacrificing core capabilities.


6 CONCLUSION


We introduced a preference-augmented alignment framework for mitigating the safety degradation
of LLMs under domain-specific fine-tuning. By complementing token-level distributional alignment
with preference signals, our method encourages models to favor the safe outputs of their pre-trained
counterparts rather than merely imitating distributions. Extensive experiments demonstrate that this
approach achieves a more favorable trade-off between safety and utility, and substantially improves
robustness against adversarial fine-tuning.


Our findings suggest that preference signals can play a crucial role in strengthening intrinsic safety
alignment, pointing toward a new direction for fine-tuning resistant safeguards. Future work may
explore scaling our framework to broader alignment objectives, integrating human feedback more
directly, and extending it to multi-modal or continual fine-tuning settings.


ETHICS STATEMENT


This work investigates methods to improve the safety of large language models. We only use publicly
available datasets and avoid personal or sensitive information. While safety research may reveal
potential risks, our intention is to strengthen responsible and trustworthy AI deployment.


10


REPRODUCIBILITY STATEMENT


We are committed to ensuring the reproducibility of our results. All datasets used in this work are
publicly available, and we provide a detailed description of the improved methods in the main text.
The experimental settings, including model architectures and training procedures, are outlined in the
corresponding sections. To further facilitate reproducibility, we will release our source code upon
the publication of the paper.


REFERENCES


Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn
Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson
Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez,
Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario
Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan.
Training a helpful and harmless assistant with reinforcement learning from human feedback, 2022.


L’eon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine
learning. _SIAM Review_, 60(2):223–311, 2018.


Olivier Bousquet and Andr’e Elisseeff. Stability and generalization. _Journal of Machine Learning_
_Research_, 2(Mar):499–526, 2002.


Stephen P. Boyd and Lieven Vandenberghe. _Convex Optimization_ . Cambridge University Press, New
York, NY, USA, 2004. ISBN 0521833787.


Leo Gao, John Schulman, and Jacob Hilton. Scaling laws for reward model overoptimization. In
_International Conference on Machine Learning_, pp. 10835–10866. PMLR, 2023.


Guillaume Garrigos and Robert M Gower. Handbook of convergence theorems for (stochastic) gradient methods. _arXiv preprint arXiv:2301.11235_, 2023.


Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximalgradient methods under the polyak-łojasiewicz condition. In Paolo Frasconi, Niels Landwehr,
Giuseppe Manco, and Jilles Vreeken (eds.), _Machine_ _Learning_ _and_ _Knowledge_ _Discovery_ _in_
_Databases_, pp. 795–811, Cham, 2016. Springer International Publishing. ISBN 978-3-319-461281.


Stanisław Łojasiewicz. Une propri’et’e topologique des sous-ensembles analytiques r’eels. In _Les_
_’Equations_ _aux_ _d’eriv’ees_ _partielles_, pp. 87–89, Paris, 1963. ’Editions du Centre National de la
Recherche Scientifique.


Yurii Nesterov. _Introductory_ _Lectures_ _on_ _Convex_ _Optimization:_ _A_ _Basic_ _Course_ . Springer, 2004.
ISBN 978-1-4613-4691-3.


Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_,
volume 35, pp. 27730–27744, 2022.


B. T. Polyak. Gradient methods for minimizing functionals. _Zhurnal_ _Vychislitel’noi_ _Matematiki_ _i_
_Matematicheskoi Fiziki_, 3(4):643–653, 1963.


Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson.
Fine-tuning aligned language models compromises safety, even when users do not intend to! arXiv
preprint arXiv:2310.03693, 2023.


Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal,
and Peter Henderson. Safety alignment should be made more than just a few tokens deep. arXiv
preprint arXiv:2406.05946, 2024.


11


Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea
Finn. Direct preference optimization: Your language model is secretly a reward model. In
A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), _Advances_
_in_ _Neural_ _Information_ _Processing_ _Systems_, volume 36, pp. 53728–53741. Curran Associates,
Inc., 2023. URL `[https://proceedings.neurips.cc/paper_files/paper/2023/file/](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf)`
`[a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf)` .


Herbert Robbins and Sutton Monro. A stochastic approximation method. _The Annals of Mathemat-_
_ical Statistics_, pp. 400–407, 1951.


John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy
optimization algorithms. _arXiv preprint arXiv:1707.06347_, 2017.


Gemini Team. Gemini: A family of highly capable multimodal models. arXiv preprint
arXiv:2312.11805, 2023.


Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée
Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and
efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.


Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation
and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.


Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du,
Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint
arXiv:2109.01652, 2021.


Tengyang Xie, Ching-An Cheng, Nan Jiang, Paul Mineiro, and Alekh Agarwal. Bellman-consistent
pessimism for offline reinforcement learning. In _Advances in Neural Information Processing Sys-_
_tems_, pp. 34: 6683–6694, 2021.


Erhan Xu, Kai Ye, Hongyi Zhou, Luhan Zhu, Francesco Quinzan, and Chengchun Shi. Doubly robust
alignment for large language models. arXiv preprint arXiv:2506.01183, 2025.


Xingcheng Xu. The policy cliff: A theoretical analysis of reward-policy maps in large language
models. arXiv preprint arXiv:2507.20150, 2025.


A APPENDIX


A.1 PROOF OF LOSS FUNCTION CONVERGENCE


A.1.1 PROBLEM SETUP AND NOTATION


Consider the total loss function:


_𝐿_ Total( _𝜃_ ) = _𝐿_ CSFT ( _𝜃_ ) + _𝛿_ epoch · _𝐿_ PA( _𝜃_ )


where:


epoch
_𝛿_ epoch = 0 _._ 1 + 0 _._ 2 ×
max_epoch


[
∑| _**y**_ |


]


_𝐿_ CSFT( _𝜃_ ) = −E( _**x**_ _,_ _**y**_ )∼ _**D**_


_𝑤𝑡_  - log _𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ )
_𝑡_ =1


{ [ ( )]}
_𝑤𝑡_ = 2 1 − _𝜎_ _𝛽𝑡_ log _𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ ) − log _𝜋_ aligned( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ )


[
∑| _**y**_ |


_𝐿_ PA( _𝜃_ ) = −E( _**x**_ _,_ _**y**_ )∼ _**D**_


]

∑| _**y**_ |

log _𝜎_ [(] _𝜇𝑡_  - [(] log _𝜋𝜃_ ( _𝑦𝑡,_ aligned| _**x**_ _,_ _**y**_ _<𝑡_ ) − log _𝜋𝜃_ ( _𝑦𝑡,𝜃_ | _**x**_ _,_ _**y**_ _<𝑡_ ) [))]
_𝑡_ =1


( )
_𝜇𝑡_ = _𝐷_ KL _𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ ) ∥ _𝜋_ aligned( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ )


12


A.1.2 BASIC ASSUMPTIONS


To establish convergence, we adopt the following relatively mild assumptions, which are standard in
stochastic optimization and align with practical deep learning settings:


1. **Bounded** **Gradients** : There exists a constant _𝐺>_ 0 such that for any _𝜃_ and any sample
( _𝑥, 𝑦_ ),
∥∇ _𝜃_ log _𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ )∥≤ _𝐺._


2. **Lipschitz Continuity of Gradients** : There exists a constant _𝐿>_ 0 such that for any _𝜃_ 1 _, 𝜃_ 2,


∥∇ _𝐿_ Total( _𝜃_ 1) −∇ _𝐿_ Total( _𝜃_ 2)∥≤ _𝐿_ ∥ _𝜃_ 1 − _𝜃_ 2 ∥ _._


3. **Learning Rate Decay** : The learning rate sequence { _𝜂𝑘_ } satisfies


∇ _𝐿_ CSFT( _𝜃_ ) = −E( _**x**_ _,_ _**y**_ )∼ _**D**_


∑∞

_𝜂𝑘_ = ∞ _,_
_𝑘_ =1


∑∞

_𝜂_ [2] _𝑘_ _[<]_ [ ∞] _[.]_
_𝑘_ =1


4. **Bounded Gradient Noise** : The stochastic gradient _𝑔_ ( _𝜃_ ) satisfies


E[ _𝑔_ ( _𝜃_ ) | _𝜃_ ] = ∇ _𝐿_ Total( _𝜃_ ) _,_ E[∥ _𝑔_ ( _𝜃_ ) −∇ _𝐿_ Total( _𝜃_ )∥ [2] | _𝜃_ ] ≤ _𝜎_ [2] _._


5. **Bounded Weights** : There exists a constant _𝑊>_ 0 such that for all _𝑡_, | _𝑤𝑡_ | ≤ _𝑊_ .

6. **Bounded Log-Probability Differences** : There exists a constant _𝐷>_ 0 such that for all _𝑡_
and _𝜃_, | log _𝜋𝜃_ ( _𝑦𝑡,_ aligned| _**x**_ _,_ _**y**_ _<𝑡_ ) − log _𝜋𝜃_ ( _𝑦𝑡,𝜃_ | _**x**_ _,_ _**y**_ _<𝑡_ )| ≤ _𝐷_ .

7. **Probability Lower Bound** : There exists a constant _𝜖>_ 0 such that for all _𝑦𝑡_, _𝑥_, _𝑦<𝑡_, and _𝜃_,
_𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ ) ≥ _𝜖_ . This can be enforced via logit clipping or label smoothing.

8. **Bounded KL Divergence** : There exists a constant _𝐾>_ 0 such that for all _𝑡_ and _𝜃_, _𝜇𝑡_ ≤ _𝐾_ .
This holds in finite-vocabulary settings or can be enforced via KL clipping.


Discussion of Assumption Validity


Assumption 2 (Lipschitz continuity) ensures the smoothness of the loss gradient, a standard condition in stochastic optimization for deriving descent inequalities. It is not overly restrictive: in deep
learning models like Transformers, the loss is a composition of smooth functions (e.g., softmax and
cross-entropy), satisfying local Lipschitz properties in bounded parameter spaces. Unbounded parameters can be handled via weight decay or gradient clipping. Many activation functions, such as
the sigmoid in _𝐿_ PA, have inherently Lipschitz gradients. In practice, gradient clipping enforces this
condition, and learning rates are typically chosen smaller than 1/ _𝐿_ for stability.


Assumption 7 (probability lower bound) ensures well-defined KL divergences and gradients. It can
be practically achieved through logit clipping or label smoothing, common in language models.


Assumption 8 (bounded KL) is reasonable in finite-vocabulary models, where KL has a natural upper
bound log(1/min _𝑞_ ( _𝑦_ )). In practice, KL regularization or clipping ensures numerical stability.


A.1.3 CONVERGENCE PROOF


Gradient Computation and Analysis


First, analyze the gradient of the total loss:


∇ _𝐿_ Total( _𝜃_ ) = ∇ _𝐿_ CSFT( _𝜃_ ) + _𝛿_ epoch∇ _𝐿_ PA( _𝜃_ ) _._


**CSFT Gradient**


The gradient of the CSFT loss is:


]


[
∑| _**y**_ |

_𝑤𝑡_   - ∇ _𝜃_ log _𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ )
_𝑡_ =1


13


_._


Since _𝑤𝑡_ is treated as a constant via detachment, and by Assumption 5, | _𝑤𝑡_ | ≤ _𝑊_, combined with
Assumption 1, we have:
∥∇ _𝐿_ CSFT( _𝜃_ )∥≤ _𝑊_             - _𝐺_             - _𝑇_ max _,_

where _𝑇_ max = max | _**y**_ | is the maximum sequence length.


**PA Gradient**


The gradient of the PA loss is:


where Δ _𝑡_ = log _𝜋𝜃_ ( _𝑦𝑡,_ aligned| _**x**_ _,_ _**y**_ _<𝑡_ ) − log _𝜋𝜃_ ( _𝑦𝑡,𝜃_ | _**x**_ _,_ _**y**_ _<𝑡_ ).


The gradient expands as:


∇ _𝜃_ log _𝜎_ ( _𝑧𝑡_ ) = (1 − _𝜎_ ( _𝑧𝑡_ ))∇ _𝜃_ _𝑧𝑡_ _,_ _𝑧𝑡_ = _𝜇𝑡_ Δ _𝑡_ _._


Thus,
∇ _𝜃_ _𝑧𝑡_ = _𝜇𝑡_ ∇ _𝜃_ Δ _𝑡_ + Δ _𝑡_ ∇ _𝜃_ _𝜇𝑡_ _,_

where
∇ _𝜃_ Δ _𝑡_ = ∇ _𝜃_ log _𝜋𝜃_ ( _𝑦𝑡,_ aligned| _**x**_ _,_ _**y**_ _<𝑡_ ) −∇ _𝜃_ log _𝜋𝜃_ ( _𝑦𝑡,𝜃_ | _**x**_ _,_ _**y**_ _<𝑡_ ) _._


By Assumption 1, ∥∇ _𝜃_ Δ _𝑡_ ∥≤ 2 _𝐺_ .


For ∇ _𝜃_ _𝜇𝑡_, since _𝜇𝑡_ = _𝐷_ KL ( _𝑝_ ∥ _𝑞_ ) with _𝑝_ = _𝜋𝜃_ (·| _**x**_ _,_ _**y**_ _<𝑡_ ) and fixed _𝑞_ = _𝜋_ aligned(·| _**x**_ _,_ _**y**_ _<𝑡_ ), the gradient
is:
∇ _𝜃_ _𝜇𝑡_ = E( _**x**_ _,_ _**y**_ )∼ _**D**_ [∇ _𝜃_ log _𝑝_ ( _𝑦_ ) · (log _𝑝_ ( _𝑦_ ) − log _𝑞_ ( _𝑦_ ))] _._


By Assumption 1, ∥∇ _𝜃_ log _𝑝_ ( _𝑦_ )∥≤ _𝐺_ . By Assumption 7, and assuming a lower bound on min _𝑞_ ( _𝑦_ )
(common in finite vocabularies), there exists _𝐵>_ 0 such that | log _𝑝_ ( _𝑦_ ) - log _𝑞_ ( _𝑦_ )| ≤ _𝐵_, yielding
∥∇ _𝜃_ _𝜇𝑡_ ∥≤ _𝐺𝐵_ .


By Assumption 8, _𝜇𝑡_ ≤ _𝐾_, and by Assumption 6, |Δ _𝑡_ | ≤ _𝐷_ . Since |1 − _𝜎_ ( _𝑧𝑡_ )| ≤ 1,


∥∇ _𝜃_ log _𝜎_ ( _𝑧𝑡_ )∥≤ _𝜇𝑡_         - 2 _𝐺_ + |Δ _𝑡_ | · _𝐺𝐵_ ≤ 2 _𝐾𝐺_ + _𝐷𝐺𝐵._


Thus, there exists a constant _𝐶_ = _𝑇_ max · (2 _𝐾𝐺_ + _𝐷𝐺𝐵_ ) such that


∥∇ _𝐿_ PA( _𝜃_ )∥≤ _𝐶._


**Bounded Total Gradient**


Since _𝛿_ epoch ≤ 0 _._ 3, the total gradient is bounded:


∥∇ _𝐿_ Total( _𝜃_ )∥≤ _𝑊𝐺𝑇_ max + 0 _._ 3 _𝐶_ = _𝑀._


**Convergence Framework**


Consider the stochastic gradient descent update:


_𝜃_ _𝑘_ +1 = _𝜃_ _𝑘_                  - _𝜂𝑘𝑔_ ( _𝜃_ _𝑘_ ) _,_


where _𝑔_ ( _𝜃_ _𝑘_ ) is an unbiased estimator of ∇ _𝐿_ Total( _𝜃_ _𝑘_ ).


By Assumption 2, the pointwise descent lemma holds:

_𝐿_ Total( _𝜃_ _𝑘_ +1) ≤ _𝐿_ Total( _𝜃_ _𝑘_ ) + ∇ _𝐿_ Total( _𝜃_ _𝑘_ ) [⊤] ( _𝜃_ _𝑘_ +1 − _𝜃_ _𝑘_ ) + _[𝐿]_ 2 [∥] _[𝜃]_ _[𝑘]_ [+][1][ −] _[𝜃]_ _[𝑘]_ [∥][2] _[.]_


Substituting the update:

_𝐿_ Total( _𝜃_ _𝑘_ +1) ≤ _𝐿_ Total( _𝜃_ _𝑘_ ) − _𝜂𝑘_ ∇ _𝐿_ Total( _𝜃_ _𝑘_ ) [⊤] _𝑔_ ( _𝜃_ _𝑘_ ) + _[𝐿]_ 2 _[𝜂]_ [2] _𝑘_ [∥] _[𝑔]_ [(] _[𝜃]_ _[𝑘]_ [)∥][2] _[.]_ (*)


14


]


∇ _𝐿_ PA( _𝜃_ ) = −E( _**x**_ _,_ _**y**_ )∼ _**D**_


[
∑| _**y**_ |

∇ _𝜃_ log _𝜎_ ( _𝜇𝑡_  - Δ _𝑡_ )
_𝑡_ =1


_,_


A.1.4 DETAILED DERIVATION OF THE EXPECTED DESCENT INEQUALITY


In stochastic optimization, deriving the expected descent from the pointwise inequality requires careful handling of expectations. This section provides a rigorous derivation.


Monotonicity of Expectations
**Theorem A.1** (Monotonicity of Conditional Expectations) **.** _Let_ _𝑋_ _and 𝑌_ _be random variables on a_
_probability space, and let_ F _be a sub-𝜎-algebra._ _If 𝑋_ ≤ _𝑌_ _almost surely, then_ E[ _𝑋_ | F ] ≤ E[ _𝑌_ | F ]
_almost surely._


_Proof._ This follows from the definition of conditional expectation. For a detailed proof, see Billingsley (1995, Probability and Measure). 

Application to Derive Conditional Expectation


Define _𝑋_ = _𝐿_ Total( _𝜃_ _𝑘_ +1) and

_𝑌_ = _𝐿_ Total( _𝜃_ _𝑘_ ) − _𝜂𝑘_ ∇ _𝐿_ Total( _𝜃_ _𝑘_ ) [⊤] _𝑔_ ( _𝜃_ _𝑘_ ) + _[𝐿]_ 2 _[𝜂]_ [2] _𝑘_ [∥] _[𝑔]_ [(] _[𝜃]_ _[𝑘]_ [)∥][2] _[,]_

with F the _𝜎_ -algebra generated by _𝜃_ _𝑘_ . By Equation (*), _𝑋_ ≤ _𝑌_ a.s. Thus, by Theorem 1,


E[ _𝑋_ | _𝜃_ _𝑘_ ] ≤ E[ _𝑌_ | _𝜃_ _𝑘_ ] a.s.


By linearity of conditional expectations:

E[ _𝑌_ | _𝜃_ _𝑘_ ] = _𝐿_ Total( _𝜃_ _𝑘_ ) − _𝜂𝑘_ ∇ _𝐿_ Total( _𝜃_ _𝑘_ ) [⊤] E[ _𝑔_ ( _𝜃_ _𝑘_ ) | _𝜃_ _𝑘_ ] + _[𝐿]_ 2 _[𝜂]_ [2] _𝑘_ [E][[∥] _[𝑔]_ [(] _[𝜃]_ _[𝑘]_ [)∥][2] [|] _[𝜃]_ _[𝑘]_ []] _[.]_


By Assumption 4, E[ _𝑔_ ( _𝜃_ _𝑘_ ) | _𝜃_ _𝑘_ ] = ∇ _𝐿_ Total( _𝜃_ _𝑘_ ), so

∇ _𝐿_ Total( _𝜃_ _𝑘_ ) [⊤] E[ _𝑔_ ( _𝜃_ _𝑘_ ) | _𝜃_ _𝑘_ ] = ∥∇ _𝐿_ Total( _𝜃_ _𝑘_ )∥ [2] _._


For the variance term:

E[∥ _𝑔_ ( _𝜃_ _𝑘_ )∥ [2] | _𝜃_ _𝑘_ ] = E[∥ _𝑔_ −∇+ ∇∥ [2] | _𝜃_ _𝑘_ ]

= E[∥ _𝑔_ −∇∥ [2] | _𝜃_ _𝑘_ ] + ∥∇∥ [2] + 2E[( _𝑔_ −∇) [⊤] ∇| _𝜃_ _𝑘_ ]

≤ _𝜎_ [2] + ∥∇ _𝐿_ Total( _𝜃_ _𝑘_ )∥ [2] _,_


since the cross term is zero by unbiasedness.


∑ _𝐾_

_𝜂𝑘_
_𝑘_ =1


Thus:


)
∥∇ _𝐿_ Total( _𝜃_ _𝑘_ )∥ [2] + _[𝐿]_ 2 _[𝜂]_ [2] _𝑘_ _[𝜎]_ [2] _[.]_


E[ _𝐿_ Total( _𝜃_ _𝑘_ +1) | _𝜃_ _𝑘_ ] ≤ _𝐿_ Total( _𝜃_ _𝑘_ ) − _𝜂𝑘_


(
1 − _[𝐿𝜂][𝑘]_

2


Taking full expectation (law of total expectation):


E[ _𝐿_ Total( _𝜃_ _𝑘_ +1)] ≤ E[ _𝐿_ Total( _𝜃_ _𝑘_ )] − _𝜂𝑘_


(
1 − _[𝐿]_


2 _[𝜂][𝑘]_


)
E[∥∇ _𝐿_ Total( _𝜃_ _𝑘_ )∥ [2] ] + _[𝐿]_ 2 _[𝜂]_ [2] _𝑘_ _[𝜎]_ [2] _[.]_ (**)


A.1.5 DETAILED DERIVATION OF THE CONVERGENCE CONCLUSION


From Equation (**), sum from _𝑘_ = 1 to _𝐾_ :


∑ _𝐾_

_𝜂_ [2] _𝑘_ _[.]_
_𝑘_ =1


∑ _𝐾_

(E[ _𝐿_ Total( _𝜃_ _𝑘_ +1)] − E[ _𝐿_ Total( _𝜃_ _𝑘_ )]) ≤−
_𝑘_ =1


∑ _𝐾_

_𝜂𝑘_
_𝑘_ =1


(
1 − _[𝐿]_ 2 _[𝜂][𝑘]_


)
E[∥∇∥ [2] ] + _[𝐿𝜎]_ [2]

2


)
E[∥∇∥ [2] ] + _[𝐿𝜎]_ [2]


The left side telescopes to E[ _𝐿_ Total( _𝜃𝐾_ +1)] − E[ _𝐿_ Total( _𝜃_ 1)]. Rearranging:


)
E[∥∇ _𝐿_ Total( _𝜃_ _𝑘_ )∥ [2] ] ≤ E[ _𝐿_ Total( _𝜃_ 1)] − E[ _𝐿_ Total( _𝜃𝐾_ +1)] + _[𝐿𝜎]_ [2]

2


15


∑ _𝐾_

_𝜂_ [2] _𝑘_ _[.]_
_𝑘_ =1


(
1 − _[𝐿]_ 2 _[𝜂][𝑘]_


Since _𝐿_ Total ≥ 0 (as a negative log-likelihood), E[ _𝐿_ Total( _𝜃𝐾_ +1)] ≥ 0, so the sum is bounded above by
a term that remains finite as _𝐾_ →∞ (due to [∑] _𝜂_ [2] _𝑘_ _[<]_ [ ∞][).] [Thus:]


Assume for contradiction that lim inf E[∥∇∥ [2] ] _>_ 0. Then there exists _𝜖>_ 0 and subsequence { _𝑘𝑗_ }
with E[∥∇( _𝜃_ _𝑘𝑗_ )∥ [2] ] ≥ _𝜖_ . For large _𝑗_, 1 −( _𝐿_ /2) _𝜂𝑘𝑗_ _>_ 1/2, so the subsum diverges, contradicting the
finite sum. Hence:
lim inf [=][ 0] _[.]_
_𝑘_ →∞ [E][[∥∇] _[𝐿]_ [Total][(] _[𝜃]_ _[𝑘]_ [)∥][2][]]


A.2 PROOF OF LOSS FUNCTION ROBUSTNESS


A.2.1 INTRODUCTION


In this proof, we consider the given loss function form and rigorously prove its robustness. First,
we clearly define ’robustness’ in the context of optimization. Subsequently, through mathematical
derivations, we analyze the optimization process, particularly focusing on what quantity’s variation
causes the policy _𝜋𝜃_ to approach _𝜋_ aligned. Finally, we provide a quantitative proof using weaker
assumptions (such as convexity, Lipschitz gradient continuity, and the Polyak-Łojasiewicz (PL) inequality, rather than strong convexity). These assumptions are more general and applicable to certain non-strongly convex but locally well-behaved loss functions, as commonly encountered in deep
learning scenarios.


To align with theoretical analyses in related literature, such as ’The Policy Cliff: A Theoretical Analysis of Reward-Policy Maps in Large Language Models,’ we emphasize how regularization terms
like _𝐿_ PA( _𝜃_ ) resolve degeneracies in optima, preventing ’policy cliffs’ (discontinuous policy shifts
under perturbations) by acting as tie-breakers in cases of non-unique optimal actions.


A.2.2 DEFINITION OF ROBUSTNESS


**Definition** **A.1** (Robustness) **.** _In_ _optimization_ _problems,_ _the_ _robustness_ _of_ _the_ _loss_ _function_ _𝐿_ ( _𝜃_ )
_refers_ _to_ _the_ _system’s_ _ability_ _to_ _maintain_ _its_ _performance_ _and_ _stability_ _in_ _the_ _face_ _of_ _uncertainty_
_or perturbations._ _Specifically, uncertainty may manifest as noise perturbations in the input data_ _𝐷_
_(such as label noise or input variations)._ _We quantify the perturbation size through the noise intensity_
_𝜖>_ 0 _, representing the maximum amplitude of data deviation._


_Quantitatively, the loss function 𝐿_ ( _𝜃_ ) _is considered robust if, for a noise perturbation 𝜖, the perturbed_
_optimal solution 𝜃_ [∗] _[,𝜖]_ _and the original optimal solution 𝜃_ [∗] _satisfy:_


∥ _𝜃_ [∗] _[,𝜖]_                     - _𝜃_ [∗] ∥≤ _𝐾𝜖,_


_where_ _𝐾_ _is_ _a_ _Lipschitz-related_ _constant._ _Here,_ _we_ _uniformly_ _use_ _the_ _Euclidean_ _norm_ ∥· ∥ _in_ _the_
_parameter space to measure changes in solutions, ensuring consistency._ _This ensures that changes_
_in the output (optimal solution or policy 𝜋𝜃_ _) are linearly bounded by the perturbation size._


_In cases where optima are non-unique (degenerate), perturbations can lead to discontinuous shifts,_
_akin to ’policy cliffs’ in reward-policy maps._ _Our assumptions (e.g., PL inequality) ensure unique-_
_ness, mitigating such issues._


In this context, uncertainty primarily refers to noise in the data distribution _𝐷_, characterized by _𝜖_ .
We will prove that the total loss _𝐿_ Total( _𝜃_ ), by incorporating the _𝐿_ PA( _𝜃_ ) term, enhances robustness to
noise. Specifically, _𝐿_ PA acts as a regularization term that strengthens the PL inequality constant _𝜇_,
thereby tightening the robustness bound.


A.2.3 REVIEW OF THE LOSS FUNCTION


The total loss function is:


16


)
E[∥∇ _𝐿_ Total( _𝜃_ _𝑘_ )∥ [2] ] _<_ ∞ _._


∑∞

_𝜂𝑘_
_𝑘_ =1


(
1 − _[𝐿]_ 2 _[𝜂][𝑘]_


_𝐿_ Total( _𝜃_ ) = _𝐿_ CSFT( _𝜃_ ) + _𝛿_ epoch · _𝐿_ PA( _𝜃_ ) _,_


where


[
∑| _**y**_ |


]


_𝐿_ CSFT( _𝜃_ ) = −E( _**x**_ _,_ _**y**_ )∼ _**D**_


_𝑤𝑡_  - log _𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ )
_𝑡_ =1


_,_


{ [ ( )]}
_𝑤𝑡_ = 2 1 − _𝜎_ _𝛽𝑡_ log _𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ ) − log _𝜋_ aligned( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ ) _,_


_𝐿_ PA( _𝜃_ ) = −E( _**x**_ _,_ _**y**_ )∼ _**D**_


[ ]
∑| _**y**_ |

log _𝜎_ [(] _𝜇𝑡_   - [(] log _𝜋𝜃_ ( _𝑦𝑡,_ aligned| _**x**_ _,_ _**y**_ _<𝑡_ ) − log _𝜋𝜃_ ( _𝑦𝑡,𝜃_ | _**x**_ _,_ _**y**_ _<𝑡_ ) [))]
_𝑡_ =1


_,_


( )
_𝜇𝑡_ = _𝐷_ KL _𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ ) ∥ _𝜋_ aligned( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ ) _._


Note that _𝑦𝑡,_ aligned and _𝑦𝑡,𝜃_ are the argmax predictions of _𝜋_ aligned and _𝜋𝜃_ at position _𝑡_ (assuming
softmax outputs as probability distributions, taking the maximum probability class). To handle the
non-differentiability of argmax, we implicitly use a softened version (such as temperature-scaled
softmax approximation) to ensure gradient flow. _𝑤𝑡_ is treated as a constant in gradient computations
(via detach operation) to avoid overfitting to noise. _𝛿_ epoch is a scheduling parameter that increases
with epochs, used to gradually strengthen the regularization effect.


A.2.4 ANALYSIS OF THE OPTIMIZATION PROCESS: THE KEY FACTOR DRIVING POLICY ALIGNMENT


During optimization, we use gradient descent to minimize _𝐿_ Total( _𝜃_ ). The update rule is _𝜃_ ← _𝜃_ _𝜂_ ∇ _𝜃_ _𝐿_ Total( _𝜃_ ), where _𝜂_ is the learning rate.


The key question is: what quantity’s variation causes _𝜋𝜃_ to approach _𝜋_ aligned. The answer lies in the
gradient contribution of _𝐿_ PA( _𝜃_ ). Specifically, the variation in _𝜇𝑡_ (i.e., changes in KL divergence)
drives this process. We will compute the gradients in detail to demonstrate this.


First, consider the gradient of _𝐿_ PA( _𝜃_ ):


]


∇ _𝜃_ _𝐿_ PA( _𝜃_ ) = −E( _**x**_ _,_ _**y**_ )∼ _**D**_


[
∑| _**y**_ |

∇ _𝜃_ log _𝜎_ ( _𝜇𝑡_  - Δ _𝑡_ )
_𝑡_ =1


_,_


where Δ _𝑡_ = log _𝜋𝜃_ ( _𝑦𝑡,_ aligned| _**x**_ _,_ _**y**_ _<𝑡_ ) − log _𝜋𝜃_ ( _𝑦𝑡,𝜃_ | _**x**_ _,_ _**y**_ _<𝑡_ ).


Let _𝑧𝑡_ = _𝜇𝑡_ - Δ _𝑡_, then the gradient of log _𝜎_ ( _𝑧𝑡_ ) is:


1
∇ _𝜃_ log _𝜎_ ( _𝑧𝑡_ ) =
_𝜎_ ( _𝑧𝑡_ ) [·] _[ 𝜎]_ [′] [(] _[𝑧][𝑡]_ [) · ∇] _[𝜃]_ _[𝑧][𝑡]_ _[.]_


Since _𝜎_ [′] ( _𝑧_ ) = _𝜎_ ( _𝑧_ )(1 − _𝜎_ ( _𝑧_ )), we have:


thus:


Next, compute ∇ _𝜃_ _𝑧𝑡_ :


_𝜎_ [′] ( _𝑧𝑡_ )

_𝜎_ ( _𝑧𝑡_ ) [=][ 1][ −] _[𝜎]_ [(] _[𝑧][𝑡]_ [)] _[,]_


∇ _𝜃_ log _𝜎_ ( _𝑧𝑡_ ) = (1 − _𝜎_ ( _𝑧𝑡_ ))∇ _𝜃_ _𝑧𝑡_ _._


∇ _𝜃_ _𝑧𝑡_ = Δ _𝑡_ - ∇ _𝜃_ _𝜇𝑡_ + _𝜇𝑡_ - ∇ _𝜃_ Δ _𝑡_ _._


17


Here, _𝜇𝑡_ = _𝐷_ KL ( _𝜋𝜃_ ∥ _𝜋_ aligned), and its gradient is:


∇ _𝜃_ Δ _𝑡_ = ∇ _𝜃_ log _𝜋𝜃_ ( _𝑦𝑡,_ aligned| _**x**_ _,_ _**y**_ _<𝑡_ ) −∇ _𝜃_ log _𝜋𝜃_ ( _𝑦𝑡,𝜃_ | _**x**_ _,_ _**y**_ _<𝑡_ ) _._


When _𝜇𝑡_ is large (high KL divergence, misaligned positions), the Δ _𝑡_ ∇ _𝜃_ _𝜇𝑡_ term dominates, amplifying
the gradient to push for KL reduction. Conversely, low _𝜇𝑡_ weakens the gradient. _𝛿_ epoch controls the
weight of this term.


Thus, the quantity driving _𝜋𝜃_ toward _𝜋_ aligned is the variation in _𝜇𝑡_, i.e., the reduction in KL divergence,
achieved through the dynamic adjustment of the regularization effect in _𝐿_ PA.


This mechanism aligns with tie-breaking in degenerate optima: high KL indicates non-unique actions, and _𝐿_ PA resolves this by favoring aligned policies, preventing rational exploitation of incomplete losses (similar to ’clever slacker’ behaviors in policy cliffs literature).


A.2.5 QUANTITATIVE PROOF OF ROBUSTNESS


We assume the loss function _𝐿_ Total( _𝜃_ ) satisfies convexity, its gradient ∇ _𝜃_ _𝐿_ Total( _𝜃_ ) is _𝐿_ -Lipschitz continuous, and the Polyak-Łojasiewicz (PL) inequality:


1
2 [∥∇] _[𝐿]_ [Total][(] _[𝜃]_ [)∥][2] [≥] _[𝜇]_ [(] _[𝐿]_ [Total][(] _[𝜃]_ [) −] _[𝐿]_ [Total][(] _[𝜃]_ [∗][))] _[,]_


where _𝜇>_ 0 is a constant. The PL inequality and convexity ensure the existence and uniqueness of
minimizers, as well as convergence rates in optimization, while not directly required for the parameter bound derivation below. Introducing _𝐿_ PA( _𝜃_ ) can increase _𝜇_, as the KL divergence regularization
enhances the lower bound on the gradient norm. Specifically, through Hessian analysis, _𝐿_ PA contributes positive definite terms to the second derivatives, increasing the effective curvature lower
bound (refer to optimization literature such as Karimi et al.). For a sketch: the Hessian of _𝐿_ PA involves terms like ∇ [2] _𝐷_ KL, which is positive semi-definite for entropy-like regularizers, thus boosting
the minimal eigenvalue related to _𝜇_ .
**Lemma A.1** (Perturbation Bounds) **.** _Consider noisy data 𝐷_ _[𝜖]_ = _𝐷_ + _𝜖𝜉, where 𝜉_ _is bounded noise,_
∥ _𝜉_ ∥≤ 1 _._ _Then 𝐿Total_ _[𝜖]_ [(] _[𝜃]_ [)] [=] _[𝐿][Total]_ [(] _[𝜃]_ [) +] _[ 𝜖]_ [·] _[ 𝑔]_ [(] _[𝜃, 𝜉]_ [)] _[, where][ 𝑔]_ _[is a bounded function,]_ [ |] _[𝑔]_ [|] [≤] _[𝑀][.]_

_Additionally, for the gradients,_ ∥∇ _𝜃_ _𝐿Total_ _[𝜖]_ [(] _[𝜃]_ [) −∇] _[𝜃]_ _[𝐿][Total]_ [(] _[𝜃]_ [)∥≤] _[𝜖𝐺][, where][ 𝐺]_ _[is the bound on gradient]_
_perturbations._


Proof: By the linearity of expectations and the Lipschitz nature of continuous functions, the noise
linearly affects the loss and gradients. Specifically, for each expectation term, the difference due to
perturbation is linearly controlled by _𝜖_, yielding | _𝐿_ _[𝜖]_ [≤] _[𝜖𝑀]_ [.] [Applying the chain rule]
Total [(] _[𝜃]_ [)−] _[𝐿]_ [Total][(] _[𝜃]_ [)|]
to gradients, each derivative term’s perturbation is also linear, so ∥∇ _𝜃_ _𝐿_ Total _[𝜖]_ [(] _[𝜃]_ [) −∇] _[𝜃]_ _[𝐿]_ [Total][(] _[𝜃]_ [)∥≤] _[𝜖𝐺]_ [.]
**Theorem A.2** (Robustness Bound) **.** _Let 𝜃_ [∗] _be the minimizer of_ _𝐿Total_ ( _𝜃_ ) _, and 𝜃_ [∗] _[,𝜖]_ _the minimizer of_
_𝐿_ _[𝜖]_ _[Assuming the gradient]_ [ ∇] _[𝜃]_ _[𝐿][Total]_ [(] _[𝜃]_ [)] _[ is][ 𝐿][-Lipschitz continuous and the gradient perturbation]_
_Total_ [(] _[𝜃]_ [)] _[.]_
_satisfies_ ∥∇ _𝜃_ _𝐿Total_ _[𝜖]_ [(] _[𝜃]_ [) −∇] _[𝜃]_ _[𝐿][Total]_ [(] _[𝜃]_ [)∥≤] _[𝜖𝐺][, then]_


∥ _𝜃_ [∗] _[,𝜖]_                     - _𝜃_ [∗] ∥≤ _[𝜖𝐺]_

_𝐿_ _[.]_

Proof: Since _𝜃_ [∗] and _𝜃_ [∗] _[,𝜖]_ are minimizers, we have ∇ _𝐿_ Total( _𝜃_ [∗] ) = 0 and ∇ _𝐿_ Total _[𝜖]_ [(] _[𝜃]_ [∗] _[,𝜖]_ [)] [=] [0.] [From the]
gradient perturbation assumption,


∥∇ _𝐿_ Total( _𝜃_ [∗] _[,𝜖]_ )∥ = ∥∇ _𝐿_ Total( _𝜃_ [∗] _[,𝜖]_ ) −∇ _𝐿_ Total _[𝜖]_ [(] _[𝜃]_ [∗] _[,𝜖]_ [)∥≤] _[𝜖𝐺.]_


By Lipschitz gradient continuity,


18


]
_._


∇ _𝜃_ _𝐷_ KL ( _𝜋𝜃_ ∥ _𝜋_ aligned) = E( _**x**_ _,_ _**y**_ )∼ _**D**_


For ∇ _𝜃_ Δ _𝑡_ :


[
_𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ )
∇ _𝜃_ log _𝜋𝜃_ ( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ ) log
_𝜋_ aligned( _𝑦𝑡_ | _**x**_ _,_ _**y**_ _<𝑡_ )


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


∥∇ _𝐿_ Total( _𝜃_ [∗] _[,𝜖]_ ) −∇ _𝐿_ Total( _𝜃_ [∗] )∥≤ _𝐿_ ∥ _𝜃_ [∗] _[,𝜖]_          - _𝜃_ [∗] ∥ _._


Since ∇ _𝐿_ Total( _𝜃_ [∗] ) = 0,


∥∇ _𝐿_ Total( _𝜃_ [∗] _[,𝜖]_ )∥≤ _𝐿_ ∥ _𝜃_ [∗] _[,𝜖]_             - _𝜃_ [∗] ∥ _._


Combining the inequalities,


∥∇ _𝐿_ Total( _𝜃_ [∗] _[,𝜖]_ )∥≤ _𝜖𝐺_ ≤ _𝐿_ ∥ _𝜃_ [∗] _[,𝜖]_            - _𝜃_ [∗] ∥ _,_


thus


∥ _𝜃_ [∗] _[,𝜖]_                     - _𝜃_ [∗] ∥≤ _[𝜖𝐺]_

_𝐿_ _[.]_


This bound shows that parameter changes are linearly related to the perturbation size _𝜖_, proving robustness. Introducing _𝐿_ PA can reduce the effective Lipschitz constant _𝐿_ (through smoothing) or decrease _𝐺_ (reducing noise sensitivity), thereby tightening the bound. The PL inequality and convexity
ensure minimizer existence and uniqueness but do not directly participate in deriving the parameter
bound.


A.3 THE USE OF LARGE LANGUAGE MODELS


In this work, we employ large language models (LLMs) primarily as assistants for writing. Their
role is limited to aiding the authors in polishing the presentation and improving readability.


19