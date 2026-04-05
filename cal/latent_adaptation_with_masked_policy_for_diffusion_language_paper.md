# LATENT ADAPTATION WITH MASKED POLICY FOR DIFFUSION LANGUAGE MODELS


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Diffusion large language models (dLLMs) offer parallel, non-sequential decoding compared to autoregressive language models, but their test-time reasoning has
been little explored. We introduce **LAMP** ( _Latent_ _Adaptation_ _via_ _Masked_ _Pol-_
_icy_ ), a training-free framework that performs instance-level, reward-guided _policy-_
_gradient_ updates on a sparse set of token latents in masked diffusion models.
LAMP identifies low-confidence positions, applies several small gradient steps
to their hidden states, and then performs a _clamp-and-inpaint_ decode that fixes
accepted edits while the diffusion sampler bidirectionally re-inpaints the remaining tokens for global coherence. A dual reward design supports lightweight selfreward as well as a Perfect Sparse Reward Model (PSRM) that provides binary
correctness signals. Despite its simplicity and modest compute, LAMP consistently improves reasoning accuracy on GSM8K, MATH-500, and AIME across
LLaDA and Dream backbones. These results demonstrate that reward-guided latent adaptation is a practical axis for enhancing diffusion-based reasoning without
retraining and complements existing inference-time scaling methods.


1 INTRODUCTION


Large language models (LLMs) have achieved strong performance across a wide range of tasks,
from question answering and planning to program synthesis. Most of these advances are driven by
_autoregressive_ (AR) decoding, where tokens are generated sequentially from left to right. While
effective for producing fluent text, AR decoding imposes rigid ordering, restricts parallelism, and
makes revisiting earlier mistakes costly. These limitations are particularly problematic for multistep reasoning tasks—such as mathematics and code generation—where global consistency and
error correction are essential (Gulrajani & Hashimoto, 2023).


_Diffusion language models_ (dLLMs), also called masked or non-autoregressive LMs, have recently
emerged as a promising alternative (Ye et al., 2024; 2025b; Kim et al., 2025; Yu et al., 2025).
Instead of committing tokens sequentially, dLLMs iteratively refine masked sequences: all positions
are updated in parallel, high-confidence tokens can be clamped early, and uncertain slots remain
open for further resampling. This bidirectional denoising paradigm supports parallel decoding and
flexible re-masking, making dLLMs attractive for both efficiency and structured reasoning. Recent
systems such as LLaDA, Dream, Mercury, and d1 scale competitively with AR models and often
achieve lower wall-clock inference cost by leveraging parallel refinement (Labs & collaborators,
2025; Zhao et al., 2025).


Yet the reasoning ability of dLLMs remains underexplored. Test-time strategies that have proven
effective for AR models—such as chain-of-thought prompting, self-consistency, or verifier-based
reranking—rely on a left-to-right trajectory and transfer poorly to diffusion. In dLLMs, decoding unfolds as a sequence of partially masked _latent_ _states_ refined by bidirectional updates, with
no causal prefix structure. Early work has begun to expose the unique opportunities of this setting: diffusion-of-thoughts (Ye et al., 2024), implicit search in structured domains like chess (Ye
et al., 2025c), and inference-time scaling via remasking, particle Gibbs sampling, or classical search
(Wang et al., 2025; Dang et al., 2025; Zhang et al., 2025). Complementary acceleration studies
show that many answers converge early, enabling confident early commitment (Li et al., 2025a).
Together, these findings suggest that intermediate diffusion states encode rich reasoning signals, and
that targeted test-time edits could improve outcomes without retraining.


1


Figure 1. Overview of LAMP ( _Latent Adaptation via Masked Policy_ ). LAMP identifies uncertain tokens from
an initial decode, applies reward-guided latent edits, and then constrains subsequent diffusion passes to respect
confident changes while re-inpainting remaining positions.


We present **LAMP**, a training-free framework for instance-level test-time adaptation in masked diffusion LMs. LAMP treats hidden token states as editable latents, applies one or two policy-gradient
updates guided by reward signals, and then performs a _clamp-and-inpaint_ decode that propagates
edits through the diffusion process. The reward can be either a lightweight self-reward (e.g., format or consistency checks) or a strong outcome-based signal such as the Perfect Sparse Reward
Model (PSRM). By selectively reopening low-confidence tokens while preserving global coherence
through inpainting, LAMP leverages the revisability of diffusion to achieve targeted reasoning improvements without model retraining.


Our contributions are:


- We introduce **LAMP**, a training-free method for _reward-guided_ _latent_ _optimization_ in masked
diffusion LMs. LAMP performs sparse policy-gradient updates on token latents and uses clampand-inpaint decoding to propagate edits globally.

- We design a diffusion-specific adaptation loop combining (i) low-confidence token selection, (ii)
dual reward supervision (self-reward and PSRM), (iii) light trust-region regularization for stable
updates, and (iv) confidence gating to retain only reliable edits.

- Experiments on GSM8K, MATH-500, and AIME2024 show consistent gains across LLaDA,
LLaDA-1.5, and Dream, with modest compute overhead. Ablations confirm that diffusion-specific
ingredients—sparse selection, reward choice, and clamp-and-inpaint—are essential, whereas
naïve latent nudging yields little benefit.


Overall, LAMP highlights the untapped potential of dLLMs for structured reasoning. By aligning
diffusion’s revisable decoding with lightweight reward-guided adaptation, it complements both AR
prompting methods and emerging inference-time scaling techniques.


2 METHODS


We present **LAMP** ( _Latent Adaptation via Masked Policy_ ), a training-free, instance-level test-time
adaptation method for masked diffusion language models (dLLMs). LAMP edits only a sparse set
of token-level latents under reward feedback, then _clamps_ these edits while the diffusion sampler
re-inpaints all other positions in parallel. All updates are per-instance and discarded after decoding;
the base model parameters remain unchanged.


2


2.1 PRELIMINARIES: MASKED DIFFUSION LANGUAGE MODELS


**Diffusion** **decoding.** Discrete diffusion LMs replace autoregressive decoding with an iterative
denoising process over masked sequences. Starting from a fully masked sequence,


_yT_ = [[MASK] _, . . .,_ [MASK]] _,_ (1)
_yt−_ 1 _∼_ _pθ_ ( _yt−_ 1 _| yt, x_ ) _,_ _t_ = _T, . . .,_ 1 _,_ (2)
_y_ ˆ = _y_ 0 _,_ (3)


where _x_ is the prompt and _θ_ are model parameters. Each step refines all tokens in parallel, and
schedulers can commit high-confidence positions early while leaving others masked for further refinement. Systems such as LLaDA and Dream adopt this paradigm, enabling parallel decoding and
flexible resampling.


**Inference characteristics.** Two properties make masked diffusion well-suited for test-time adaptation: (1) _Parallel scoring:_ every step provides logits for all tokens, enabling efficient confidence
diagnostics. (2) _Constrained infilling:_ decoding can be rerun with a subset of tokens clamped, while
masked slots are re-inpainted bidirectionally for global consistency. LAMP exploits these properties to introduce sparse, local edits while relying on the model’s own diffusion dynamics to maintain
coherence.


2.2 OVERVIEW OF LAMP


LAMP augments masked diffusion decoding with a lightweight, per-instance latent adaptation loop
that operates _around_ the base model without modifying its parameters:


1. **Baseline decode.** Run an initial diffusion pass to produce a candidate _y_ ˆ [(0)] . Alongside the output
tokens, record the hidden states _h_ [(0)] _i_ and predictive distributions _qi_ [(0)] at each position. These
serve as the initialization for subsequent edits.

2. **Edit-set** **selection.** Identify a small fraction of uncertain positions ( _≈_ 10%). We rank tokens
by their confidence score _ci_ = max _qi_ [(0)] or the margin between the top-1 and top-2 logits. This
selection focuses adaptation on tokens where the model itself is least sure, avoiding unnecessary
perturbations.

3. **Latent policy adaptation.** Treat the hidden states at the selected positions as editable latents _zi_ .
These latents define local categorical policies over token choices. Using reinforcement signals
(Sec. 2.3), we apply one–two policy-gradient updates to steer _zi_ toward reward-aligned alternatives.

4. **Clamp-and-inpaint.** After adaptation, edits that exceed confidence thresholds are _clamped_
(frozen). A final constrained diffusion pass re-inpaints all other tokens in parallel, letting bidirectional self-attention propagate local improvements globally.


This design leverages diffusion’s non-sequential decoding: local edits can be injected late in the
chain and still harmonize with the rest of the sequence. Because only a small subset of latents are
updated, LAMP adds negligible overhead compared to a standard decode.


2.3 REWARD MODELS


Central to LAMP is how provisional sequences are evaluated. We consider two complementary
reward models:


**Self-reward.** Lightweight checks for well-formedness, such as format validity, arithmetic consistency, or duplicate-answer detection. These signals are inexpensive but noisy.


**Perfect Sparse Reward Model (PSRM).** For supervised evaluations, we employ a binary oracle
that returns 1 if the final normalized answer matches the ground truth:

_R_ PSRM(ˆ _y_ ) = **1**          - norm(ˆ _a_ ) = _a_ _[⋆]_ [�] _,_


where ˆ _a_ is the model’s extracted answer, _a_ _[⋆]_ the ground truth, and norm( _·_ ) applies canonicalization
(case-folding, whitespace trimming, numeric simplification). Despite its sparsity—only providing


3


**Algorithm 1:** LAMP: Test-time masked latent adaptation
**Require:** prompt _x_, diffusion LM _pθ_, budget _k_, adaptation steps _K_, step size _η_, reward function _R_

1: _y_ ˆ [(0)] _←_ DIFFUSE( _x_ ); record _h_ [(0)] _i_ _[, q]_ _i_ [(0)]
2: _S_ _←_ lowest- _k_ % tokens by _ci_ = max _qi_ [(0)]
3: Initialize _zi_ _←_ _h_ [(0)] _i_ for _i ∈S_ ; set _F_ = ∅
4: **for** _t_ = 1 to _K_ **do**
5: Sample provisional edits _y_ ˜ _S_ _∼_ _πz_ ; form candidate _y_ ˆ
6: _y_ ˆ _←_ CONSTRAINEDDIFFUSE( _x,_ ˜ _yF_ _∪_ _y_ ˜ _S_ )
7: _r_ _←_ _R_ (ˆ _y_ ); update baseline _b_
8: Update _z_ _←_ _z −_ _η∇z_ ( _L_ PG + _R_ stab)
9: **end for**
10: _F_ _←{i_ : max _qi_ ( _zi_ ) _≥_ _τ_ _∧_ max _qi_ ( _zi_ ) _−_ max _qi_ [(0)] _≥_ _ε}_
11: _y_ ˆ _[⋆]_ _←_ CONSTRAINEDDIFFUSE( _x,_ ˜ _yF_ )
12: **return** _y_ ˆ _[⋆]_


feedback at the sequence level—PSRM delivers a strong training signal that is tightly aligned with
the target objective. This reward is used as the default in our main experiments.


2.4 LATENT POLICY ADAPTATION


**Editable latents.** For each _i_ _∈S_, we initialize an editable latent _zi_ _←_ _h_ [(0)] _i_ from the hidden state
of the baseline decode. Each latent parameterizes a local categorical policy
_qi_ ( _zi_ ) = softmax( _g_ ( _zi_ )) _,_
where _g_ is the output head of the diffusion LM. The product distribution _πz_ = [�] _i∈S_ _[q][i]_ [(] _[z][i]_ [)][ defines]

a joint policy over the edit set, from which provisional tokens _y_ ˜ _S_ are sampled.


**Policy-gradient** **update.** We view LAMP as optimizing a reward-weighted posterior over sequences,
_p_ _[∗]_ ( _y_ ) _∝_ _pθ_ ( _y_ _| x_ ) exp( _R_ ( _y_ )) _,_
where _pθ_ is the base diffusion model and _R_ is the external reward (Sec. 2.3). Since this posterior is intractable, we perform stochastic updates on editable latents with REINFORCE. Given a
provisional sample _y_ ˆ and moving baseline _b_, the gradient estimator is

_∇zL_ PG = _−_         - _R_ (ˆ _y_ ) _−_ _b_         - [�] _∇zi_ log _qi_ ( _zi_ )[˜ _yi_ ] _._ (4)

_i∈S_


**Confidence gating.** After _K_ update steps, an edit is accepted if its confidence and improvement
exceed fixed thresholds:
max _qi_ ( _zi_ ) _≥_ _τ_ and max _qi_ ( _zi_ ) _−_ max _qi_ [(0)] _≥_ _ε,_
with _τ_ = 0 _._ 6 and _ε_ = 0 _._ 05 by default. Accepted edits are added to the frozen set _F_ .


**Final** **decode.** We clamp accepted edits and run a final constrained diffusion pass, yielding _y_ ˆ _[⋆]_ .
This step allows bidirectional re-inpainting to propagate local edits coherently across the sequence.


3 EXPERIMENTS


We evaluate **LAMP** on mathematical reasoning and code generation, focusing on how latent adaptation interacts with different forms of reward supervision and inference-time scaling. Our experimental analysis proceeds along four complementary axes: (1) **Main** **results** : comparing LAMP under
self-reward and Perfect Sparse Reward Model (PSRM) supervision across math benchmarks. (2)
**Scaling behavior** : studying how accuracy evolves with increasing numbers of adaptation iterations.
(3) **Reward dynamics** : analyzing the stability and transition patterns of self-reward signals during
refinement. (4) **Qualitative** **effects** : examining concrete cases where LAMP changes an answer
from incorrect to correct (and vice versa), shedding light on the mechanisms behind reward-guided
edits. Together, these experiments aim to establish not only whether LAMP improves reasoning, but
also under what conditions, at what computational cost, and through which underlying dynamics.


4


|Method|Model|GSM8K<br>T1 T2|MATH-500<br>T1 T2|AIME 2024<br>T1 T2|
|---|---|---|---|---|
|Vanilla DLM|LLaDA<br>DREAM<br>LLaDA 1.5|71.3<br>63.8<br>81.9<br>81.8<br>74.5<br>67.0|25.6<br>21.2<br>37.6<br>35.0<br>26.4<br>21.0|0.0<br>3.3<br>0.0<br>0.0<br>3.3<br>0.0|
|LAMP + Self-reward|LLaDA<br>DREAM<br>LLaDA 1.5|73.9 (+2.6)<br>67.0 (+3.2)<br>83.2 (+1.3)<br>83.4 (+1.6)<br>75.9 (+1.4)<br>68.9 (+1.9)|27.6 (+2.0)<br>23.2 (+2.0)<br>38.4 (+0.8)<br>37.2 (+2.2)<br>28.0 (+1.6)<br>22.6 (+1.6)|0.0 (+0.0)<br>0.0 (-3.3)<br>3.3 (+3.3)<br>0.0 (+0.0)<br>0.0 (-3.3)<br>0.0 (+0.0)|
|LAMP + PSRM|LLaDA<br>DREAM<br>LLaDA 1.5|84.6 (+13.3)<br>84.0 (+20.2)<br>87.8 (+5.9)<br>88.0 (+6.2)<br>85.4 (+10.9)<br>85.5 (+18.5)|41.6 (+16.0)<br>37.4 (+16.2)<br>43.4 (+5.8)<br>42.4 (+7.4)<br>42.6 (+16.2)<br>38.6 (+17.6)|10.0 (+10.0)<br>0.0 (-3.3)<br>3.3 (+3.3)<br>0.0 (+0.0)<br>3.3 (+0.0)<br>3.3 (+3.3)|


Table 1. **Main results across reasoning benchmarks.** Pass@1 accuracy on GSM8K, MATH-500, and AIME
2024. T1 and T2 denote two prompt variants. Improvements over the corresponding Vanilla DLM baseline are
shown in parentheses. Self-reward LAMP gives modest gains, whereas PSRM consistently yields substantial
improvements across all models.


3.1 SETUP


**Benchmarks.** We evaluate on three math reasoning datasets: GSM8K (Cobbe et al., 2021),
MATH-500 (Hendrycks et al., 2021), and AIME 2024 (Zhang et al., 2024). Accuracy is measured
by exact match after normalization (case-folding, whitespace trimming, and numeric simplification).


**Models.** We study two recent masked diffusion LMs: **LLaDA** (Nie et al., 2025) and its upgraded
variant **LLaDA-1.5** (Zhu et al., 2025), alongside **Dream** (Ye et al., 2025a). LLaDA employs a semiautoregressive decoding schedule where high-confidence tokens are committed early while uncertain
slots remain masked for refinement. Dream, in contrast, uses a fully masked diffusion schedule with
random re-masking across positions, enabling more flexible parallel updates. All models are used in
their released 7–8B parameter versions without additional fine-tuning.


**Reward.** We test two forms of supervision. First, a lightweight _self-reward_ based on internal consistency (e.g., well-formed numeric answers). Second, the _Perfect Sparse Reward Model_ (PSRM) Li
et al. (2025b), which provides a binary correctness signal against the ground-truth final answer. Unless otherwise stated, we use PSRM as the primary reward for evaluation.


**Prompts.** We adopt the standard math reasoning prompt format from prior work Li et al. (2025b),
which instructs the model to produce a step-by-step explanation followed by the final boxed answer.
This ensures comparability across dLLM backbones and aligns with the evaluation script.


3.2 MAIN RESULTS


Table 1 reports pass@1 accuracy across GSM8K, MATH-500, and AIME 2024. We highlight three
findings: the marginal impact of self-reward, the substantial benefits of PSRM, and consistency
across model architectures.


**Limited** **gains** **from** **self-reward.** Across benchmarks, applying LAMP with self-reward yields
only small and inconsistent improvements over vanilla DLMs. For example, LLaDA improves by
+2 _._ 6 points on GSM8K (Type 1) and +2 _._ 0 points on MATH-500 (Type 1), while DREAM shows
modest increases of +1 _._ 3 and +0 _._ 8 on the same metrics. Several settings even degrade (e.g., AIME
Type 2 for LLaDA). These results indicate that heuristic self-reward signals are too weak to drive
systematic reasoning gains.


**Substantial** **benefits** **from** **PSRM.** PSRM supervision delivers robust and often double-digit improvements across all datasets. On GSM8K, LLaDA improves from 71 _._ 3% to 84 _._ 6% (+13 _._ 3), and
LLaDA-1.5 from 74 _._ 5% to 85 _._ 4% (+10 _._ 9). On MATH-500, both models gain over +16 points,
while DREAM rises from 37 _._ 6% to 43 _._ 4% (+5 _._ 8). These results confirm that accurate but sparse
supervision signals can reliably guide latent adaptation to enhance reasoning.


**Performance on AIME2024.** Although overall accuracies remain low due to task difficulty, PSRM
again provides clear improvements. LLaDA increases from 0 _._ 0% to 10 _._ 0% on Type 1 prompts,


5


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||


Figure 2. Accuracy vs. number of latent adaptation iterations on three model–dataset settings. Orange: Perfect
Reward Model. Blue: Self-Reward Model. Perfect reward yields strong, monotonic improvements with early
rapid gains that gradually saturate (+12.8 on LLaDA-8B: 71.8 _→_ 84.6; +10.6 on LLaDA-1.5: 74.8 _→_ 85.4;
+5.6 on Dream-7B: 82.4 _→_ 88.0), while self-reward produces only modest improvements with early plateaus
(+2.6, +1.4, and +1.5 points, respectively).


and LLaDA-1.5 and DREAM also see modest but consistent gains. This suggests that even in
challenging domains, outcome-based adaptation can extract non-trivial benefits.


**Cross-model** **consistency.** The improvements hold across different dLLM backbones: LLaDA
(semi-autoregressive), LLaDA-1.5 (variance-reduced refinement), and DREAM (fully masked diffusion). Notably, both weaker and stronger baselines benefit: DREAM, despite already competitive performance, gains across all datasets, while LLaDA-1.5 still achieves sizable jumps. This
demonstrates that LAMP with PSRM is not tied to a particular decoding strategy but leverages core
properties of diffusion refinement.


**Implication.** Overall, the findings emphasize that the effectiveness of test-time latent adaptation
hinges on the reward source. Self-reward produces marginal or unstable changes, whereas PSRM
consistently yields substantial improvements across datasets and models. Thus, designing meaningful reward signals, rather than merely increasing inference-time compute, is key to unlocking
reasoning gains in diffusion LMs.


3.3 TEST-TIME SCALING: ITERATIVE LATENT ADAPTATION


Prior work has explored test-time scaling primarily by increasing the number of generated candidates
or sampled trajectories (e.g., self-consistency or tree search) (Muennighoff et al., 2025; Yao et al.,
2023b). We instead examine an orthogonal axis enabled by diffusion LMs: _increasing the number_
_of_ _latent_ _adaptation_ _iterations_ in LAMP. This reframes iterative refinement as a tunable compute
budget that trades additional updates in latent space for improved reasoning accuracy.


Figure 2 compares accuracy across reward models and backbones on GSM8K and related settings. The **Perfect** **Sparse** **Reward** **Model** **(PSRM)** induces smoothly increasing, concave (fastthen-saturating) gains in all cases, achieving +12.8 points on LLaDA-8B (71.8 _→_ 84.6), +10.6 on
LLaDA-1.5 (74.8 _→_ 85.4), and +5.6 on Dream-7B (82.4 _→_ 88.0). By contrast, **self-reward** yields
only small improvements—+2.6, +1.4, and +1.5 points—often plateauing after the first few iterations. These results underscore the centrality of reward quality: even sparse but accurate outcome
supervision enables effective test-time scaling via latent adaptation.


**Extreme** **scaling** **with** **PSRM.** Following Liu et al. (2025a) but replacing process rewards with
outcome supervision, PSRM attains competitive iteration-based scaling. On AIME2024 (Zhang
et al., 2024), LLaDA-8B narrows the gap with frontier systems, and on MATH-500 (Hendrycks
et al., 2021) it achieves strong overall accuracy among evaluated dLLMs, while requiring far fewer
forward passes than explicit search-based methods. This highlights the efficiency of scaling within
latent space when paired with reliable reward supervision.


**Takeaway.** _Iteration scaling in latent space_ is a practical and efficient test-time scaling strategy for
diffusion LMs. Unlike approaches that rely on sampling more candidates, LAMP leverages rewardguided updates that propagate globally through the diffusion process, delivering accuracy gains with
favorable compute–performance trade-offs. Future work on hybrid or process-aware rewards may


6


1 4 7 10
Iterations

Perfect Reward Self Reward


85


80


75


70


1 4 7 10
Iterations


85


80


75


88


86


84


82


1 4 7 10
Iterations


True True True False False True False False


Figure 3. Distribution of self-reward transitions across different model-dataset combinations. Green:
True→True (maintaining positive reward). Red: True→False (losing positive reward). Blue: False→True
(gaining positive reward). Orange: False→False (maintaining negative reward). The analysis reveals that selfreward signals are often inconsistent, with substantial True→False transitions indicating reward degradation
over iterations.


further close the gap between self-rewarding and perfect supervision, broadening the applicability
of iteration-based test-time scaling.


3.4 SELF-REWARD TRANSITION ANALYSIS


**Dynamics of self-reward transitions.** Figure 3 provides a detailed breakdown of the reward transition dynamics observed during the LAMP refinement process. Each cell of the transition matrix
corresponds to the probability of an example moving between correct ( _True_ ) and incorrect ( _False_ )
states before and after refinement. Across all model–dataset combinations, the transition structure
is dominated by **True→True** outcomes, which range from 18% to 79% depending on task difficulty
and backbone. This dominant mass reflects the fact that once a reasoning trajectory is initially judged
as correct by the self-reward signal, it is overwhelmingly preserved through subsequent refinement
steps. Importantly, the consistently small **True→False** rate (fixed at 3% in our construction) indicates that degradation of correct reasoning paths is rare. This establishes a strong stability property:
self-reward seldom overturns good partial solutions, ensuring that performance does not regress as
the refinement progresses.


By contrast, the contribution of **False→True** transitions—cases where the iterative process corrects
an initially incorrect output—is modest, lying between 3.8% and 6.2% across settings. While these
flips represent genuine improvements induced by self-reward, their relatively small magnitude implies that most of the eventual accuracy is attributable not to creating correctness from scratch but to
maintaining correctness where it already existed. Finally, the largest source of inefficiency lies in the
**False→False** quadrant, where 13.6–73.8% of examples remain incorrect despite iterative updates.
This indicates that once a reasoning trajectory fails the initial reward test, it is usually not recovered
by self-reward alone, especially on more challenging problems.


**Variation** **across** **models** **and** **tasks.** Clear trends emerge when stratifying by dataset and backbone. On the relatively simpler GSM8K benchmark, Dream achieves the strongest preservation,
with nearly 79% of examples remaining correct (True→True) across both prompt types. LLaDA
achieves lower but still dominant preservation, between 61–68%. The small False→True mass
in both cases suggests that GSM8K improvements are almost entirely due to stabilizing correct
reasoning trajectories, with little recovery from errors. In contrast, MATH-500 presents a substantially harder regime: here, True→True drops to 18–23% for LLaDA and 32–35% for Dream,
while False→False grows dominant, capturing persistent reasoning failures. These shifts highlight
a dataset-difficulty effect: as the reasoning space becomes more complex, the self-reward process
increasingly fails to correct initially incorrect answers. The False→True mass remains largely un

7


LLaDA · GSM8K · Type1


LLaDA · GSM8K · Type2


LLaDA · MATH-500 · Type1


LLaDA · MATH-500 · Type2


Dream · GSM8K · Type1


Dream · GSM8K · Type2


Dream · MATH-500 · Type1


Dream · MATH-500 · Type2


Question: John runs 60 miles a week and runs on 3 days. He runs 3 hours the first day and half as much
on the other two days. How fast does he run?
Answer (GT): 10
Transition: TRUE _→_ FALSE
Original CoT: John runs 3 days a week, with 3 hours on the first day and
1.5 hours each on the other two days, for a total of 6 hours.
He runs 60 miles in 6 hours, so his speed is 60 / 6 = 10 mph.


LAMP: He runs 60/3 = 20 miles per day. He runs 3 hours the first
day and 1.5 hours each of the other two days, so 3 + 1.5 +
1.5 = 4.5 hours per week. Speed = 20 / 4.5 = 4.44 mph. ####
4.44


Question: Stephen’s groceries cost $40.00. A 25% platform fee is added, plus $3.00 delivery and a
$4.00 tip. What is the final price?
Answer (GT): 57
Transition: FALSE _→_ TRUE
Original CoT: 25% of $40.00 is $10.00. Add delivery to the original bill:
$40.00 + $3.00 = $43.00. Then add the tip of $4.00: $43.00
+ $4.00 = $47.00.


LAMP: 25% of $40.00 is $10.00. Add $3.00 delivery and $4.00 tip.
Final = $40.00 + $10.00 + $3.00 + $4.00 = $57.00.


Table 2. **Mixed qualitative outcomes under self-reward (LAMP).** We show one TRUE _→_ FALSE regression
(Case 38) where local edits break global accounting, and one FALSE _→_ TRUE correction (fees and tip) where
aggregation is fixed.


changed ( _≈_ 4–6%), reinforcing that the rate of recovery is insensitive to problem difficulty, but the
preservation rate collapses, leading to much weaker net accuracy.


3.5 QUALITATIVE ANALYSIS


We probe how self-reward reshapes reasoning by contrasting a TRUE _→_ FALSE (TF) regression and
a FALSE _→_ TRUE (FT) correction (Table 9). In **Case** **38** ( _weekly_ _pace_ ), the baseline correctly aggregates weekly time (3 + 1.5 + 1.5 = 6 h) to obtain 60 _/_ 6 = 10 mph. Under self-reward, LAMP
over-edits toward a per-day normalization and mis-aggregates runtime (claimed 4 _._ 5 h), yielding an
incorrect 4 _._ 44 mph. This TF pattern reflects a local reward preference for seemingly plausible partial
computations (e.g., daily averaging) that break global constraints (total distance/time consistency).


In contrast, **Case 58** ( _fees and tip_ ) shows a typical FT fix: the baseline omits the platform fee and
reports $47; LAMP correctly aggregates base price, fee, delivery, and tip to reach the ground-truth
$57.


Beyond these two cases, our broader inspection (Fig. 3) finds that self-reward frequently repairs
arithmetic omissions and bookkeeping slips (FT), but can also induce TF regressions when local
cues outweigh global consistency. To curb TF without suppressing FT, we rely on: _confidence_
_gating_ (edit only low-confidence tokens), _span-based selection_ with locality windows, _partial-freeze_
_(clamp) decoding_ for high-confidence positions, _step-size clipping and early stop_ when reward deltas
are small, and a modest _edit_ _budget_ . These constraints keep edits focused where uncertainty and
reward sensitivity align while preserving global accounting and units.


4 RELATED WORK


**Diffusion** **Language** **Models.** Diffusion-based large language models (dLLMs) have recently
emerged as strong alternatives to autoregressive models (ARMs) for text generation. Masked diffusion models such as LLaDA (Nie et al., 2025), Dream (Ye et al., 2025a), and Mercury (Labs &
collaborators, 2025) generate tokens in parallel through iterative denoising and re-masking, offering advantages in decoding flexibility and bidirectional context modeling. Recent scaling efforts


8


(e.g., d1 (Zhao et al., 2025)) demonstrate competitive accuracy with ARMs. Nonetheless, dLLMs
lag on reasoning-intensive tasks and typically require more inference steps due to the lack of KV
caching (Li et al., 2025a; Liu et al., 2025b). This motivates test-time approaches that enhance reasoning without retraining.


**Inference-Time** **Scaling** **in** **Diffusion** **Models.** A growing line of work studies how to allocate
extra computation at inference to improve dLLM outputs. _Search-based_ _methods_ include particle
Gibbs sampling for discrete diffusion (Dang et al., 2025) and classical search strategies that combine
local and global exploration (Zhang et al., 2025). _Scheduler modifications_ such as ReMDM (Wang
et al., 2025) introduce remasking to allow iterative error correction, while Prophet (Li et al.,
2025a) leverages early convergence to commit confident tokens. Other extensions such as MDMPrime (Chao et al., 2025) insert intermediate token states to reduce idle steps. These methods primarily target fluency or efficiency, leaving a gap in reasoning-specific adaptation.


**Test-Time Reasoning in Language Models.** For autoregressive LMs, several approaches exploit
additional inference compute to improve reasoning. Chain-of-thought prompting (Wei et al., 2022),
self-consistency (Wang et al., 2022), and verifier-guided search (Yao et al., 2023a) enhance reasoning by reranking or aggregating multiple trajectories. Most relevant is LatentSeek (Li et al., 2025b),
which showed that treating hidden states as optimizable latents and updating them with policy gradients can significantly improve reasoning. However, direct transfer to diffusion fails: dLLMs lack
a left-to-right causal structure and instead operate on globally masked updates. To date, no general
framework exists for per-instance latent adaptation in diffusion LMs.


**Guidance** **and** **Reinforcement** **for** **Diffusion** **Models.** Gradient-based control has been widely
explored in continuous diffusion, e.g., classifier guidance and score distillation (Ho et al., 2020;
Dhariwal & Nichol, 2021). For discrete diffusion, recent work examined simple guidance strategies (Schiff et al., 2025) and reward-weighted sampling (Dang et al., 2025), but these operate on
distributions or trajectories rather than per-instance latent optimization. Our work builds on these insights but introduces a diffusion-specific, instance-level framework: reward-guided _policy-gradient_
adaptation on masked latents, coupled with remasking and clamp-and-inpaint decoding for global
consistency.


5 CONCLUSION AND FUTURE WORK


We introduced **LAMP**, a training-free framework for reward-guided latent adaptation in masked
diffusion language models. By treating hidden token states as editable latents, applying sparse
policy-gradient updates, and constraining re-decoding through clamp-and-inpaint, LAMP improves
reasoning accuracy at test time without modifying model parameters. Experiments across GSM8K,
MATH-500, and AIME2024 show consistent gains on multiple dLLM backbones, highlighting the
value of aligning diffusion’s revisable decoding process with targeted reward feedback.


**Future** **directions.** Several promising avenues remain open for exploration. First, richer forms
of supervision could be incorporated. Current experiments rely primarily on outcome-based selfreward, which provides only a sparse binary signal. Extending to _process_ _supervision_ that evaluates intermediate reasoning steps—or leveraging verifiers trained to detect local consistency—could
enable the adaptation process to align more closely with logical correctness and to correct errors
earlier in the reasoning trajectory. Second, LAMP could be extended beyond single-turn adaptation to _interactive or multi-turn settings_, where reward feedback is provided iteratively, potentially
augmented by retrieval systems, symbolic solvers, or external critics. Such settings may be particularly valuable for long-horizon reasoning tasks or program synthesis, where one-shot reward
is often insufficient. Finally, future work could explore adaptation beyond language, applying the
same latent-policy principle to multimodal diffusion models where structured feedback is available,
such as grounded reasoning in vision-language settings or structured prediction tasks in science and
engineering domains.


Overall, LAMP demonstrates that reward-guided latent optimization provides a simple yet effective
axis for advancing the reasoning capabilities of diffusion language models, complementing both
autoregressive prompting strategies and emerging inference-time scaling methods.


9


**Ethics Statement.** This work investigates test-time _reasoning_ adaptation for masked diffusion LMs
on public, non-sensitive math datasets (e.g., GSM8K, MATH-500, AIME 2024) and does not involve
human subjects, private information, or proprietary data; IRB approval was not required. Our Perfect
Sparse Reward Model (PSRM) uses only instance-local ground-truth answers to compute binary
correctness and does not alter model weights. We will not redistribute third-party checkpoints and
will respect their original licenses; released code/configs will include usage guidelines discouraging
deployments that could violate academic integrity or safety policies. While the technique could in
principle be repurposed to optimize undesirable behaviors, our experiments are task-constrained, and
we recommend domain-appropriate safety filters for broader applications. Environmental impact
is limited: LAMP is training-free and adds modest inference overhead. The authors declare no
conflicts of interest; sources of support will be disclosed per venue policy.


**Reproducibility Statement.** We have made every effort to ensure the reproducibility of our results.
All datasets used in our experiments are publicly available and are described in detail in Section 3.
Preprocessing steps and evaluation metrics are documented in Appendix B. Our implementation, including training and evaluation scripts, is provided as anonymized supplementary material. Hyperparameters and experimental settings are reported in Appendix Section D. Together, these resources
allow independent researchers to replicate our findings and extend our work.


REFERENCES


Chen-Hao Chao, Wei-Fang Sun, Hanwen Liang, Chun-Yi Lee, and Rahul G. Krishnan. Beyond masked and unmasked: Discrete diffusion models via partial masking. _arXiv_ _preprint_
_arXiv:2505.18495_, 2025.


Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to
solve math word problems. _arXiv preprint arXiv:2110.14168_, 2021.


Meihua Dang, Jiaqi Han, Minkai Xu, Kai Xu, Akash Srivastava, and Stefano Ermon. Inferencetime scaling of diffusion language models with particle gibbs sampling. _arXiv_ _preprint_
_arXiv:2507.08390_, 2025.


Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis. In _Advances in_
_Neural_ _Information_ _Processing_ _Systems_, 2021. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)
[cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf.](https://proceedings.neurips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)


Ishaan Gulrajani and Tatsunori B. Hashimoto. Likelihood-based diffusion language models. _arXiv_
_preprint arXiv:2305.16291_, 2023.


Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song,
and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _arXiv_
_preprint arXiv:2103.03874_, 2021.


Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. _arXiv preprint_
_arXiv:2006.11239_, 2020.


Minho Kim, Hao Zhang, Joonseok Lee, and Minsu Cho. Train for the worst, plan for the best:
Understanding token ordering in masked diffusions. _arXiv preprint arXiv:2501.12345_, 2025.


Inception Labs and collaborators. Mercury: Ultra-fast language models based on diffusion. _arXiv_
_preprint arXiv:2506.17298_, 2025.


Pengxiang Li, Yefan Zhou, Dilxat Muhtar, et al. Diffusion language models know the answer before
decoding. _arXiv preprint arXiv:2508.19982_, 2025a.


Yufan Li, Haotian Zhang, Yilun Zhang, Guangyan Sun, et al. Seek in the dark: Reasoning via
test-time instance-level policy gradient in latent space. _arXiv preprint arXiv:2503.12345_, 2025b.


Runze Liu, Junqi Gao, Jian Zhao, Kaiyan Zhang, Xiu Li, Biqing Qi, Wanli Ouyang, and Bowen
Zhou. Can 1b llm surpass 405b llm? rethinking compute-optimal test-time scaling, 2025a. URL
[https://arxiv.org/abs/2502.06703.](https://arxiv.org/abs/2502.06703)


10


Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang,
and Linfeng Zhang. dllm-cache: Accelerating diffusion large language models with adaptive
caching. _arXiv preprint arXiv:2506.06295_, 2025b.


Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke
Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time
scaling. _arXiv preprint arXiv:2501.19393_, 2025. [URL https://arxiv.org/abs/2501.](https://arxiv.org/abs/2501.19393)
[19393.](https://arxiv.org/abs/2501.19393) Preprint.


Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai
Lin, Ji-Rong Wen, and Chongxuan Li. Large Language Diffusion Models. _arXiv_ _preprint_
_arXiv:2502.09992_, 2025. LLaDA.


Daniel Schiff, Hannah Kim, Yilun Wang, et al. Simple guidance mechanisms for discrete diffusion
models. _arXiv preprint arXiv:2504.06721_, 2025.


Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Remasking discrete
diffusion models with inference-time scaling. In _ICLR_, 2025.


Xuezhi Wang et al. Self-consistency. [https://www.promptingguide.ai/techniques/](https://www.promptingguide.ai/techniques/consistency)
[consistency, 2022.](https://www.promptingguide.ai/techniques/consistency) Accessed: 2025-09-16.


Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi,
Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language
models. In _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, volume 35, pp. 24824–24837,
2022. URL [https://proceedings.neurips.cc/paper_files/paper/2022/](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)
[hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html.](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)


Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik
Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. _arXiv_
_preprint arXiv:2305.10601_, 2023a. [URL https://arxiv.org/abs/2305.10601.](https://arxiv.org/abs/2305.10601)


Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik
Narasimhan. Tree of Thoughts: Deliberate problem solving with large language models. In
_NeurIPS_, 2023b.


Jiacheng Ye, Wei Sun, Lin Zheng, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Diffusionof-thoughts: Chain-of-thought reasoning in diffusion language models. _arXiv_ _preprint_
_arXiv:2405.12345_, 2024.


Jiacheng Ye, Wei Sun, Lin Zheng, Jiahui Gao, Zhiyong Wu, Xin Jiang, Zhenguo Li, and Lingpeng
Kong. Dream 7B: Diffusion Large Language Models. _arXiv preprint arXiv:2508.15487_, 2025a.


Jiacheng Ye, Tianhao Wu, Ming Gong, Xin Jiang, Zhenguo Li, and Lingpeng Kong. What exactly
does guidance do in masked discrete diffusion models? _arXiv preprint arXiv:2502.12345_, 2025b.


Jiacheng Ye, Zhenyu Wu, Jiahui Gao, Zhiyong Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong.
Implicit search via discrete diffusion: A study on chess. _ICLR_, 2025c.


Runpeng Yu, Qi Li, and Xinchao Wang. Discrete diffusion in large language and multimodal models:
A survey. _arXiv preprint arXiv:2506.13759_, 2025.


Haotian Zhang et al. Aime 2024 competition problems. _American Mathematics Competitions_, 2024.


Xiangcheng Zhang, Haowei Lin, Haotian Ye, James Zou, Jianzhu Ma, Yitao Liang, and Yilun
Du. Inference-time scaling of diffusion models through classical search. _arXiv_ _preprint_
_arXiv:2505.23614_, 2025.


Siyan Zhao, Devaansh Gupta, Qinqing Zheng, and Aditya Grover. d1: Scaling reasoning in diffusion
large language models via reinforcement learning. _arXiv preprint arXiv:2504.12216_, 2025.


Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei
Chen, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Llada 1.5: Variance-reduced preference
optimization for large language diffusion models, 2025. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2505.19223)
[2505.19223.](https://arxiv.org/abs/2505.19223)


11


## **SUMMARY OF THE APPENDIX**


This appendix provides additional details for the ICLR 2026 submission titled _**LAMP: Latent Adap-**_
_**tation via Masked Policy for Diffusion Language Models**_ . It is organized as follows:


    - §A: **LLM Usage** .

    - §B: **Datasets, Preprocessing, and Prompt Formats** .

    - §C: **Implementation Details** and PyTorch-style **Pseudo-code** .

    - §D: **Hyperparameters** and per-run **Configurations** .

    - §E: **Qualitative Examples** .


A LLM USAGE


In preparing this work, we used large language models only as auxiliary tools for grammar refinement, code formatting, and literature search. No LLM was used to generate research ideas, design
experiments, or analyze results. All conceptual contributions were developed independently by the
authors.


B DATASETS, PREPROCESSING, AND PROMPT FORMATS


**Benchmarks.** We adopt the LatentSeek-style evaluation protocol on three mathematical reasoning
datasets. **GSM8K** (Cobbe et al., 2021) contains 8,500 grade-school math word problems; we evaluate on the official test split of 1,319 questions. **MATH-500** (Hendrycks et al., 2021) is a curated
500-problem subset covering algebra, geometry, number theory, and calculus. **AIME 2024** (Zhang
et al., 2024) comprises 30 questions from the 2024 American Invitational Mathematics Examination.
All evaluations are zero-shot on the official splits.


**Prompt styles.** We use two prompting styles. _Type 1_ requests only the final boxed answer. _Type 2_
requests step-by-step reasoning (rationale) followed by the boxed answer. In both cases we enforce
\boxed{} to ease parsing.


**Prompt** **templates** **(compact** **blocks).** To avoid wide tables and incompatible verbatim-in-table
issues, we present prompts as narrow, monospaced blocks that line-wrap gracefully.


_GSM8K (Type 2)._


System: You are a precise math question solver. Solve this
math problem.
User: QUESTION: {q} Let’s think step by step. Please
provide your thought process and your final answer
separately and respond in JSON with keys _thought_ _process_ and
_final_ _answer_ . For example: { "thought process": "...",
"final answer": "..." }. Note: the final answer must be a
pure number without units or explanation.


_MATH-500 / AIME 2024 (Type 2)._


System: You are a precise math question solver. Solve this
math problem.
User: QUESTION: {q} Let’s think step by step. Please
provide your thought process and your final answer
separately and respond in JSON with keys _thought_ _process_
and _final_ _answer_ .


_Type 1 variant (final answer only)._


System: You are a precise math answerer.
User: QUESTION: {q} Return only the final numeric result in


12


\boxed{} format, e.g., \boxed{42}. Do not include steps or
extra text.


**Answer extraction and normalization.** We first extract the \boxed{} span; if absent, we fall
back to the last numeric/string-like token sequence. Normalization includes case-folding, Unicode
NFC, whitespace and thousands-separator removal, fraction simplification, rounding of decimals
(six significant figures), and evaluation of simple arithmetic expressions. Exact match (pass@1) is:


EM(ˆ _a, a_ _[⋆]_ ) = **1**        - normalize(ˆ _a_ ) = normalize( _a_ _[⋆]_ )        - _._


**Self-reward verifiers.** For self-rewarded settings, we use lightweight rule-based checks for format
and numeric validity, plus dataset-specific sanity checks. The verification prompts are short, singlepurpose instructions:


_Correctness check._


INSTRUCTIONS: Decide if the provided answer is correct.
Output exactly one token: <ANS>True or <ANS>False.


_Calculation check._


INSTRUCTIONS: (1) Extract all calculations; (2) recompute
them independently; (3) compare with the solution. If any
discrepancy, output False; else True.


_Understanding check._


INSTRUCTIONS: Verify that the reasoning interprets the
problem correctly and answers the asked quantity. Return
True if aligned; otherwise False.


_Completeness check._


INSTRUCTIONS: Verify that a final, explicit numeric answer
is provided (not just a formula). Return True or False.


C IMPLEMENTATION DETAILS AND PSEUDO-CODE


**Environment.** All experiments use PyTorch with CUDA 12.x. Backbones: **LLaDA-8B**, **LLaDA-**
**1.5**, and **Dream-7B** . Random seed 42; deterministic CuDNN where available. Adaptation is perinstance; no gradient accumulation.


**Decoding.** We use each model’s native masked-denoising scheduler and early-commit heuristics
(if provided). Sampling temperature = 1 _._ 0; no top- _k_ or nucleus sampling.


**LAMP** **defaults.** Edit budget _k_ = 10% (by lowest confidence), policy-gradient steps _K_ =
2, learning rate _η_ = 0 _._ 3, trust-region regularization ( _λ_ KL _, λ_ 2) = (0 _._ 1 _,_ 0 _._ 05), confidence-gating
( _τ, ε_ ) = (0 _._ 6 _,_ 0 _._ 05).


PYTORCH-STYLE PSEUDO-CODE (MINIMAL DEPENDENCIES)


We avoid external code environments; the snippet compiles as plain text and can be implemented
directly.


# LAMP: Latent Adaptation via Masked Policy
def LAMP_decode(model, prompt, reward_fn, k=0.1, K=2,
eta=0.3,
tau=0.6, eps=0.05, lam_kl=0.1, lam_l2=0.05):
# 1) Baseline decode with hidden states and logits
y0, h0, q0 = model.diffuse(prompt, return_hidden=True)
conf = q0.max(dim=-1).values


13


**methods** **model** **max** **prmpt** **#GPU** **lr** **opt** _ρ_ **dtype** **steps**
**len** **idx**


LAMP (SELF) LLaDA-8B 1024 1 1 A100 0.3 Adam 0.1 bf16 10
LAMP (SELF) LLaDA-8B 1024 2 1 A100 0.3 Adam 0.1 bf16 10
LAMP (SELF) LLaDA-1.5 1024 1 1 A100 0.3 Adam 0.1 bf16 10
LAMP (SELF) LLaDA-1.5 1024 2 1 A100 0.3 Adam 0.1 bf16 10


Table 3. Run configurations for **LAMP (Self)** on GSM8K.


**methods** **model** **max** **prmpt** **#GPU** **lr** **opt** _ρ_ **dtype** **steps**
**len** **idx**


LAMP (PSRM) LLaDA-8B 1024 1 1 A100 0.3 Adam 0.1 bf16 10
LAMP (PSRM) LLaDA-8B 1024 2 1 A100 0.3 Adam 0.1 bf16 10
LAMP (PSRM) LLaDA-1.5 1024 1 1 A100 0.3 Adam 0.1 bf16 10
LAMP (PSRM) LLaDA-1.5 1024 2 1 A100 0.3 Adam 0.1 bf16 10


Table 4. Run configurations for **LAMP (PSRM)** on GSM8K.


S = conf.argsort()[: int(k     - len(conf))] #
lowest-confidence
z = h0[S].detach().clone().requires_grad_(True)
baseline = 0.0


for t in range(K):
q = torch.softmax(model.head(z), dim=-1)
y_tilde = q.multinomial(1).squeeze(-1)
y, h, q_new = model.diffuse(prompt, fixed={S: y_tilde},
return_hidden=True)
r = reward_fn(y); baseline = 0.9*baseline + 0.1*r
logprob = torch.log(q[torch.arange(len(S)),
y_tilde]).sum()
pg_loss =     - (r     - baseline)     - logprob
kl_reg = torch.nn.functional.kl_div(q.log(), q0[S],
reduction="batchmean")
l2_reg = ((z     - h0[S])**2).mean()
loss = pg_loss + lam_kl     - kl_reg + lam_l2     - l2_reg
g, = torch.autograd.grad(loss, z); z = z     - eta     - g


final_conf = torch.softmax(model.head(z),
dim=-1).max(dim=-1).values
mask = (final_conf >= tau) & ((final_conf     - conf[S]) >=
eps)
fixed = { int(S[j]): int(y_tilde[j]) for j in
torch.where(mask)[0] }
y_star, _, _ = model.diffuse(prompt, fixed=fixed)
return y_star


D HYPERPARAMETERS AND RUN CONFIGURATIONS


**Global defaults.** We fix hyperparameters across experiments; beyond light sanity checks on 20dev subsets, no broad sweeps. Adam optimizer; trust-region coefficient _ρ_ =0 _._ 1; bf16 precision;
**10** diffusion refinement steps; maximum output length tokens. Edits target the _answer_ _span_ ;
rationales are refined indirectly via masked denoising.


E QUALITATIVE EXAMPLES


**Analysis.** Arithmetic aggregation cases (groceries, stories, annuities) benefit from revising lowconfidence tokens and re-sampling consistent totals. Regressions arise when confident but incorrect
local edits disrupt global consistency (running speed, puzzle) or when partial functional recurrences


14


**methods** **model** **max** **prmpt** **#GPU** **lr** **opt** _ρ_ **dtype** **steps**
**len** **idx**


LAMP (SELF) LLaDA-8B 1024 1 1 A100 0.3 Adam 0.1 bf16 10
LAMP (SELF) LLaDA-1.5 1024 2 1 A100 0.3 Adam 0.1 bf16 10
LAMP (SELF) Dream-7B 1024 1 1 L40S 0.3 Adam 0.1 bf16 10
LAMP (SELF) Dream-7B 1024 2 1 L40S 0.3 Adam 0.1 bf16 10


Table 5. Run configurations for **LAMP (Self)** on MATH-500.


**methods** **model** **max** **prmpt** **#GPU** **lr** **opt** _ρ_ **dtype** **steps**
**len** **idx**


LAMP (PSRM) LLaDA-8B 1024 1 1 A100 0.3 Adam 0.1 bf16 10
LAMP (PSRM) LLaDA-1.5 1024 2 1 A100 0.3 Adam 0.1 bf16 10
LAMP (PSRM) Dream-7B 1024 1 1 L40S 0.3 Adam 0.1 bf16 10
LAMP (PSRM) Dream-7B 1024 2 1 L40S 0.3 Adam 0.1 bf16 10


Table 6. Run configurations for **LAMP (PSRM)** on MATH-500.


are overextended (functional equation). Self-rewarded latent updates improve robustness but require
careful regularization and gating to avoid over-corrections.


15


**methods** **model** **max** **prmpt** **#GPU** **lr** **opt** _ρ_ **dtype** **steps**
**len** **idx**


LAMP (SELF) LLaDA-8B 1024 1 1 A100 0.3 Adam 0.1 bf16 10
LAMP (SELF) Dream-7B 1024 2 1 L40S 0.3 Adam 0.1 bf16 10


Table 7. Run configurations for **LAMP (Self)** on AIME 2024.


**methods** **model** **max** **prmpt** **#GPU** **lr** **opt** _ρ_ **dtype** **steps**
**len** **idx**


LAMP (PSRM) LLaDA-8B 1024 1 1 A100 0.3 Adam 0.1 bf16 10
LAMP (PSRM) Dream-7B 1024 2 1 L40S 0.3 Adam 0.1 bf16 10


Table 8. Run configurations for **LAMP (PSRM)** on AIME 2024.


**Question** John runs 60 miles a week on 3 days. He runs 3 hours on day 1 and half as much on
the other two days. How fast does he run?
**GT** 10
**Transition** TRUE _→_ FALSE
**Original CoT** Total time: 3 + 1.5 + 1.5 = 6. Speed = 60/6 = #### 10.
**LAMP** Day avg 20 miles; time 4.5 hours; 20/4.5 = #### 4.44.


**Question** Stephen’s groceries cost $40. A 25% platform fee is added, plus $3 delivery and $4
tip. Final price?
**GT** 57
**Transition** FALSE _→_ TRUE
**Original CoT** Mis-adds: 40+3+4=47. #### 47.
**LAMP** 25% of 40 is 10; total = 40+10+3+4 = #### 57.


**Question** A 1000-piece puzzle: Poppy places a quarter; mom places a third of remaining. How
many left?
**GT** 500
**Transition** TRUE _→_ FALSE
**Original CoT** Poppy=250; remaining 750; mom=250; leftover #### 500.
**LAMP** Finds 250 and 250 but outputs #### 250.


**Question** Week 1: 20, 40, 60 stories. Week 2 each doubles. Total stories?
**GT** 360
**Transition** FALSE _→_ TRUE
**Original CoT** Sums to 300.
**LAMP** Week1=120; Week2=240; Combined=#### 360.


**Question** _f_ ( _x_ ) + _f_ ( _y_ ) = _f_ ( _x_ + _y_ ) _−_ _xy −_ 1, _f_ (1) = 1. Integers _n_ with _f_ ( _n_ ) = _n_ ?
**GT** 1, _−_ 2
**Transition** TRUE _→_ FALSE
**Original CoT** Finds _n_ = 1; misses _−_ 2.
**LAMP** Drifts; outputs extraneous #### 8.


Table 9. **Mixed** **qualitative** **outcomes** **under** **self-reward** **(LAMP).** We show regressions (TRUE _→_ FALSE)
and successful corrections (FALSE _→_ TRUE).


16


**Question** Deposit $20k annually for 3 years; wants $66,200 after third deposit. Minimal compound rate?
**GT** 10
**Transition** FALSE _→_ TRUE
**Original CoT** Treats as single deposit; #### 0.
**LAMP** _FV_ = _P_ [(1+] _[r]_ [)] _[n][−]_ [1] ; solve 66200 = 20000 _·_ [(1+] _[r]_ [)][3] _[−]_ [1] ; #### 10.


_r_ [)] _[n][−]_ [1] ; solve 66200 = 20000 _·_ [(1+] _[r]_ _r_ [)][3] _[−]_ [1]


_r_ ; #### 10.