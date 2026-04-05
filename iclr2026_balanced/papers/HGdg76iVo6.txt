# WHICH HEADS MATTER FOR REASONING? RL-GUIDED KV CACHE COMPRESSION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Reasoning large language models exhibit complex reasoning behaviors through
the extended chain-of-thought generation, creating unprecedented Key-Value
(KV) cache overhead during the decoding phase. Existing KV cache compression
methods underperform on reasoning models: token-dropping methods break reasoning integrity by discarding critical information, while head-reallocation methods mistakenly compress reasoning-critical heads since they are designed for retrieval tasks, resulting in significant performance degradation as compression rates
increase. We hypothesize that KV heads exhibit functional heterogeneity in reasoning models-some heads are critical for chain-of-thought consistency while others are compressible. To validate and exploit this insight, we propose RLKV,
a novel reasoning-critical head identification method, which uses reinforcement
learning to directly optimize the relationship between each head’s cache usage
and reasoning quality. As RLKV produces rewards from actual generated samples
during training, it naturally identifies heads relevant to reasoning behaviors. We
then allocate full KV cache to these heads while applying compressed constant
KV cache to others for efficient inference. Our experiments reveal that only a
small fraction of attention heads is essential for reasoning, enabling our KV compression approach to outperform baseline methods while achieving **20-50%** cache
reduction with near lossless performance compared to uncompressed results.


1 INTRODUCTION


Recent advanced reasoning large language models (LLMs) (Jaech et al., 2024; Team et al., 2025;
Guo et al., 2025; DeepMind, 2025) exhibit complex reasoning behaviors, such as self-reflection
to revisit previous steps and exploration of alternative approaches, and achieve revolutionary performance on challenging mathematical and coding problems. However, this breakthrough creates
an unprecedented memory bottleneck: the extension of chain-of-thought (CoT) reasoning generates
significantly more tokens compared to conventional instruct models. For instance, Llama-3.1-8B-R1
(BF16) requires 16GB additional GPU memory for 32k CoT generation with a single query, primarily due to quadratic attention computation and linearly expanding KV cache. This limits large batch FIX
processing and challenges the practical deployment of reasoning models.


Key-Value (KV) cache compression methods have demonstrated effectiveness for instruct models
in long-context scenarios. As illustrated in Figure 1 (a), these methods typically follow one of two
strategies: token dropping or head reallocation. Token-dropping methods selectively evict less important tokens from each head’s KV cache (Zhang et al., 2023; Li et al., 2024; Cai et al., 2025; Yang
et al., 2024b; Qin et al., 2024), while head-reallocation methods identify critical heads and allocate
full KV cache to them, applying compressed KV cache to the remaining heads. However, as shown FIX
in Figure 1 (b, left), two representative KV compression methods, including token-dropping method
R-KV (Cai et al., 2025) and head-reallocation method DuoAttention (Xiao et al., 2024), degrade significantly when applied to reasoning models, while maintaining stable performance on their instruct
counterparts. This performance degradation correlates strongly with generation length: in the MBPP
(Austin et al., 2021) coding task, both model variants achieve nearly identical uncompressed performance, yet the reasoning variant generates on average 3341 tokens (approximately 8 _×_ ) longer than
the 439 tokens of the instruct variant. This controlled comparison isolates extended CoT generation
as the primary cause of compression challenges, rather than differences in model capability, revealing the inherent difficulty of compressing long reasoning sequences. In reasoning models, the KV


1


|b)|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|b)|b)|b)|||
|b)|b)|b)|||
|b)|||||
|b)|||||


Figure 1: **(a)** **Overviews** **of** **Two** **Methods** _Left:_ Token-dropping method removes less important
tokens from each head’s KV cache. _Right:_ Head-reallocation method allocates full KV cache to
critical heads while assigning constant-size KV cache to the remaining heads. **(b) Case study.** _Left:_
The token-dropping method (R-KV) and the head-reallocation method (DuoAttention) maintain relatively stable performance on Llama-3.1-8B-Inst but degrade substantially on Llama-3.1-8B-R1,
largely due to the longer generations produced by the reasoning model. _Right:_ In terms of error
modes, the token-dropping method (R-KV) tends to degenerate into repetitive behavior whereas
the head-reallocation method (DuoAttention) often produces over-extended CoT that exhausts the
length budget without reaching a correct solution. See Appendix A.1 for complete results.


cache undergoes a fundamental role shift: instead of serving merely as a computational optimization, it becomes the carrier of reasoning behaviors itself, storing critical states for CoT consistency
and self-reflection, making compression inherently detrimental to reasoning performance.


To understand how the two KV-cache compression approaches underperform in preserving reasoning behaviors, we analyze their error modes as compression rates increase, as illustrated in
Figure 1 (b, right). Models with token-dropping compression (R-KV) tend to lose reasoning be- FIX
haviors because they inevitably discard reasoning-critical information, disrupting CoT consistency
and leading to loops with repeated tokens. Although the R-KV approach (Cai et al., 2025) is designed specifically for reasoning models, it still cannot escape this inherent limitation. In contrast,
models with head-reallocation compression (DuoAttention) relatively maintain coherent reasoning
behaviors but are no longer effective: for problems that the uncompressed model can solve, the
compressed model goes astray in its reasoning process and is unable to reach a solution within the
maximum budget. This reveals that head-reallocation methods relatively preserve sequence infor- FIX
mation integrity in some heads by allocating full KV cache for them while compressing others (Xiao
et al., 2023). However, they may mistakenly compress heads critical for reasoning behaviors, since
their head identification targets “retrieval heads” (Wu et al., 2024). These methods rely on static
patterns from prefill attention (Fu et al., 2024; Tang et al., 2024a) or single-forward-pass training
(Xiao et al., 2024; Bhaskar et al., 2025), inherently failing to capture dynamic reasoning behaviors
that emerge during extended CoT sequences.


These findings motivate our key insight that KV heads exhibit functional heterogeneity in reasoning
models, where a subset of heads are critical to reasoning behaviors and naturally require a full KV
cache to maintain CoT consistency. We term such heads with this role as _**reasoning**_ _**heads**_ . To
validate and exploit this insight, we propose RLKV, a novel reasoning-critical head identification
method, which employs reinforcement learning (RL) to identify those heads by directly optimizing the relationship between the allocation of each head’s KV cache usage and reasoning quality.
As illustrated in Figure 2, our method observes reasoning behaviors in generated samples and assigns rewards during RL training. These reward signals guide RL with sparsity pressure to optimize
learnable gating adapters that control the mixing of full attention and local attention (Xiao et al.,
2023). The gating adapters quantify each head’s reliance on full versus local KV cache access, with
L1 penalty encouraging sparsity. Through this RL optimization, the adapter values inherently distinguish _reasoning_ _heads_ from compressible heads, directly identifying which heads are essential
for reasoning behaviors. In this way, our method consequently identifies _reasoning_ _heads_ and allocates full KV cache to them while applying compressed constant KV cache to others, effectively
preserving reasoning behaviors during KV cache compression.


Our work makes three main contributions. First, we introduce RLKV, a novel reasoning-critical
head identification method for guiding KV cache compression tailored to reasoning models, which
leverages reward signals from RL training under sparsity pressure to directly supervise reasoning
behaviors. Second, we achieve state-of-the-art compression performance, enabling near lossless FIX
reasoning capability with 20-50% KV cache usage reduction across diverse reasoning tasks and


2


Figure 2: **Overview of RLKV:** Our method proposes to utilize RL to identify reasoning heads. The
RL pipeline naturally captures reasoning behaviors, since it samples the current model’s generations
to produce reward signals. The reward function evaluates the samples to assess reasoning quality.
We employ _L × H_ learnable gating adapters to mix full attention and local attention for each head,
quantifying each head’s reliance on full versus local KV cache access. We apply an L1 penalty to
encourage adapter sparsity, while RL optimizes the adapters to preserve reasoning behaviors. After
training, we identify reasoning heads with high adapter values and allocate full KV cache to them
while applying compressed KV cache to others for efficient inference.


models. Third, to our knowledge, RLKV is the first to identify a set of heads that matter for reasoning
behaviors, while showing that other heads can still function under a compressed KV cache. FIX


2 METHODOLOGY


In this section, we present RLKV, a novel reasoning-critical head identification method to guide
efficient KV cache compression for reasoning LLMs, as illustrated in Figure 2. In this paper, we
operationally define “ _**reasoning heads**_ ” as the KV heads that:


_significantly degrade reasoning performance under local KV cache access._


These identified _reasoning_ _heads_ are essential for reasoning behaviors, which naturally requires a
full KV cache to maintain CoT consistency, while others are compressible. To achieve this, we first
use mixed attention with gating adapters to quantify each head’s reliance on complete or compressed
KV cache usage. Then we apply RL with sparsity pressure to optimize the gating adapters based
on a verifiable reward signal, naturally capturing reasoning behaviors. Finally, we introduce two
complementary stabilization techniques to address the conflict between dense regularization and
sparse rewards as the sparsity of adapters increases.


2.1 MIXED ATTENTION WITH GATING ADAPTERS


Identifying _reasoning_ _heads_ requires estimating individual KV heads’ robustness of complete KV
cache usage; therefore, we build upon mixed attention (Xiao et al., 2024), which uses lightweight
gating adapters to quantify each head’s reliance on full versus local KV cache access. Specifically,
it combines two attention modes by attention mask, including full attention mapping to the full KV
cache, and streaming attention (Xiao et al., 2023) mapping to the constant KV cache size containing
initial sink tokens and recent tokens.


The mixed attention on each head can be formulated as:


out ~~m~~ ix ~~a~~ ttn _i,j_ = _αi,j_ _·_ out ~~f~~ ull ~~a~~ ttn + (1 _−_ _αi,j_ ) _·_ out ~~s~~ treaming ~~a~~ ttn _,_ (1)

where _**α**_ _∈_ [0 _,_ 1] _[L][×][H]_ represents the learnable gating parameters for _L_ layers and _H_ heads, with
_αi,j_ represents the weight assigned to full attention on the _j_ -th head in the _i_ -th layer. This design
dramatically reduces the optimization space to only _L × H_ gating parameters by freezing all LLM
parameters, making it feasible to apply RL for identifying _reasoning heads_ .


2.2 RL FOR REASONING HEAD IDENTIFICATION


3


Qwen-2.5-7B-R1 Llama-3.1-8B-R1


Reasoning LLMs are often post-trained using reinforcement
learning with verifiable reward (RLVR) (Guo et al., 2025; Team
et al., 2025), which enhances reasoning capabilities by evaluating
generated samples based solely on final answer correctness. During this RL training process, reasoning behaviors are naturally
exhibited in the sampled CoT sequences, while reward signals directly reflect reasoning quality. These two characteristics make
RLVR ideal for _reasoning heads_ identification.


As adapters become increasingly sparse, the
mixed attention of _reasoning_ _heads_ degenerates to the streaming attention, severely degrading the model’s reasoning capacity, as shown in
Figure 4. This degradation renders the reward
signal increasingly sparse and unstable, while
the L1 penalty remains dense across all parameters. This imbalance creates a vicious cycle,
where degraded performance leads to sparser
rewards, making the dense L1 penalty relatively
stronger, which further drives adapters toward
zero with no recovery capability. To resolve
this destructive training dynamic and stabilize
the training process, we introduce two complementary techniques that address this challenge
from both the reward and penalty perspectives.


**Self-distillation Sampling.** Overly challenging problems during RL training lead to frequent failures and unstable reward signals. In


In concrete, we optimize the gating adapters _**α**_ using Group
Relative Policy Optimization (GRPO) (Shao et al., 2024) on
mathematical reasoning problems with two key modifications.
First, to maximize the discriminative power of reward signals Figure 3: Gating adapter distribufor _reasoning_ _head_ identification, we remove the KL penalty tion after RLKV training on two
that conventionally limits reward signal strength to prevent over- models, which both are GQA archioptimization. Second, we apply L1 regularization (Tibshirani, tecture.
1996) to the adapters by incorporating the scaled L1 penalty term _β∥_ _**α**_ _∥_ 1 _/_ ( _L_ _×_ _H_ )into the objective
function to encourage adapter sparsity. The reward signal preserves high _αi,j_ values for _reasoning_
_heads_ requiring full KV cache access, while the L1 penalty drives _αi,j_ toward 0 for compressible
heads.


The overall objective is defined to maximize:


_β_

_−_ _,_ (2)
_L × H_ _[∥]_ _**[α]**_ _[∥]_ [1]

 - ��  L1 penalty


1

_G_


- _G_ min - _π_ _**α**_ ( _oi|q_ ) - _π_ _**α**_ ( _oi|q_ ) - _Ai_

_i_ =1 _π_ _**α**_ old ( _oi|q_ ) _[A][i][,]_ [ clip] _π_ _**α**_ old( _oi|q_ ) _[,]_ [ 1] _[ −]_ _[ϵ,]_ [ 1 +] _[ ϵ]_


- �� reward signal


where _q_ is the input query, _{oi}_ _[G]_ _i_ =1 [are sampled outputs,] _[ A][i]_ [is the normalized advantage, computed]
using a group of rewards _{r_ 1 _, r_ 2 _, · · ·_ _, rG}_ tailored to outputs:


_[, r][G]_ [)]
_Ai_ = _[r][i][ −]_ [mean][(] _[r]_ [1] _[, r]_ [2] _[,][ · · ·]_ _._ (3)

std( _r_ 1 _, r_ 2 _, · · ·_ _, rG_ )


The clipping mechanism with threshold _ϵ_ prevents excessive policy updates, and _β_ controls the
regularization strength. The policy _π_ _**α**_ represents the model’s generation probability distribution
conditioned on the current gating parameters _**α**_, and the advantage _Ai_ is positive for outputs leading
to correct reasoning and negative for incorrect reasoning. This optimization naturally converges to
a sparse solution where _reasoning heads_ maintain high _α_ values, as demonstrated in Figure 3


2.3 STABILIZATION FOR RL TRAINING


|Stabilized Adapter Avg.<br>Original Adapter Avg.|Stabilized Reward Avg.<br>Original Reward Avg.|
|---|---|
|||
|||
|||
|||
|**Collapsing**||
|||


Figure 4: The conflict of sparse reward versus dense
penalty leads to training collapse without our stabilization techniques. As adapters become sparse (decreasing average), model performance degrades (dropping
reward), creating a vicious cycle where dense L1 penalties dominate increasingly sparse rewards.


4


Sparse Reward versus Dense Penalty


1.0


0.9


0.8


0.7


0.6


0.5


0.4


0 25 50 75 100 125 150 175
Training Steps


1.0


0.8


0.6


0.4


0.2


0.0


Full H2O R-KV DuoAttention RLKV (Ours)


KV Cache Budget Sparsity


Figure 5: Performance comparison of RLKV against KV cache compression baselines across reasoning benchmarks. We evaluate RLKV ( **Ours** ) and existing methods on two reasoning models (Llama-3.1-8B-R1 and Qwen-2.5-7B-R1) across four benchmarks (GSM8K, MATH, AIME24,
MBPP) at sparsity levels of 0.2, 0.4, 0.6, and 0.8. RLKV consistently outperforms all baselines
across different sparsity levels, demonstrating particularly strong advantages at high sparsity levels
(0.4 or 0.6) where competing methods suffer significant performance degradation. Complete numerical results are provided in Appendix A.3.


contrast to typical RLVR that utilizes sparse rewards for capability enhancement, our work leverages
RL for capability preservation under sparsity constraints. Consequently, we focus on constructing
high-quality training data that produces stable reward signals to improve learning efficiency. We
construct training data by first filtering all problems the model initially solves correctly, then curating them to 3k using a curriculum sampling strategy (Team et al., 2025). We use output token
lengths as a proxy for difficulty, enabling curriculum control that maintains stable reward signals
throughout the training process. See Section 3.1 for training dataset details.


**Adaptive Penalty Weighting.** To address the penalty imbalance, we modulate the scaling weight
_β_ of the L1 penalty based on the reward signal. Our design incorporates two protective mechanisms
to prevent training collapse. First, we use adaptive scaling centered around a target reward of ¯ _r_ _≈_ 0 _._ 7
to smoothly decay penalty when performance degrades and increase it when performance improves.
Second, we implement a hard cutoff at threshold _τ_ to completely eliminate regularization when
reasoning capability severely degrades. We implement this through a dynamic weight that replaces
the constant hyperparameter _β_ :

_β_ _[′]_ (¯ _r, τ_ ) = I(¯ _r_ _> τ_ ) _· β ·_ (exp(¯ _r_ ) _−_ 1) _,_ _r_ ¯ = mean( _r_ 1 _, r_ 2 _, · · ·_ _, rG_ ) _,_ (4)


where the exponential function (exp(¯ _r_ ) _−_ 1) provides the adaptive scaling, and the indicator function
I(¯ _r_ _> τ_ ) provides the hard cutoff based on mean reward ¯ _r_ in the current group.


The end result is a set of identified _reasoning heads_ that require full KV cache access, while nonreasoning heads can utilize compressed KV cache access, achieving significant memory compression without sacrificing reasoning capability. During inference, we use the learned gating parameters
to rank all KV heads and select the top-k heads with the highest _α_ values to maintain full KV cache
access according to the target compression ratio. The remaining heads still use full attention but with
compressed KV cache, which retains only initial sink tokens and recent tokens. Refer to Section 3.1
for further details of deployment and inference.


3 EXPERIMENTS


3.1 SETUPS


**Models, Datasets, and Baselines.** We evaluate RLKV on two mainstream small reasoning models, including Llama-3.1-8B-R1 and Qwen-2.5-7B-R1 (Guo et al., 2025), both are supervised finetuned from respective base models on DeepSeekR1 distilled CoT data (Guo et al., 2025). We con

5


GSM8K (Math)

1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.0 0.2 0.4 0.6 0.8

GSM8K (Math)

1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.0 0.2 0.4 0.6 0.8


0.0
0.0 0.2 0.4 0.6 0.8

Math500 (Math)

1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.0 0.2 0.4 0.6 0.8


0.5


0.4


0.3


0.2


0.1


0.7

0.6

0.5

0.4

0.3

0.2

0.1


0.9

0.8

0.7

0.6

0.5

0.4

0.3

0.2

0.1


Math500 (Math)


AIME24 (Math)


MBPP (Code)


0.0
0.0 0.2 0.4 0.6 0.8


0.0
0.0 0.2 0.4 0.6 0.8


0.0
0.0 0.2 0.4 0.6 0.8


Math500 (Math)


0.6

0.5

0.4

0.3

0.2

0.1


0.7

0.6

0.5

0.4

0.3

0.2

0.1


AIME24 (Math)


MBPP (Code)


0.0
0.0 0.2 0.4 0.6 0.8


duct experiments on four benchmarks, using three datasets of increasing difficulty mathematical
reasoning, GSM8K (Cobbe et al., 2021) for elementary problems, Math500 (Lightman et al., 2023)
for intermediate problems and AIME24 (MMA, 2024) for advanced problems, to evaluate performance across difficulty levels, and MBPP (Austin et al., 2021) for Python programming to assess
generalization beyond the training domain. We compare our method with KV cache compression FIX
approaches including H2O (Zhang et al., 2023) and R-KV (Cai et al., 2025), which are typical tokendropping methods, and DuoAttention (Xiao et al., 2024), which is a head-reallocation method.


**Implementation Details.** We implement RLKV by integrating MixedAttention into AReaL (Fu
et al., 2025) and SGLang (Zheng et al., 2024). AReaL is an asynchronous distributed RL framework
for updating adapters, and AReaL uses SGLang as the generation backend. We optimize gating
adapters using GRPO with 4 samples per query and AdamW (Loshchilov & Hutter, 2017) with
learning rate 0 _._ 01. We filter 3,000 mathematical problems from DeepScaleR (Luo et al., 2025)
following our curriculum sampling strategy. During training, local attention uses 128 sink and 256
local tokens; for evaluation, non-reasoning heads use compressed KV cache only with 16 sink and
64 local tokens. To ensure fair comparison, we augment all baselines with equivalent token overhead
and convert fixed-budget methods to dynamic allocation. Details are provided in Appendix A.2.


3.2 MAIN RESULTS


Figure 5 presents the performance of RLKV against baselines across two reasoning models and four
benchmarks at sparsity levels of 0.2, 0.4, 0.6, and 0.8. RLKV consistently outperforms all baselines
at different levels of sparsity, with particularly strong advantages at high sparsity, such as 0.4 and
0.6, where other methods suffer significant performance degradation. Remarkably, RLKV even surpasses the full KV cache baseline on AIME24, the most challenging mathematical reasoning benchmark, for Llama-3.1-8B-R1 at 0.4 and Qwen-2.5-7B-R1 at 0.2, respectively. This counter-intuitive
result suggests that our identified _reasoning_ _heads_ capture the essential components for complex
reasoning, while non-reasoning heads may introduce noise that degrades performance when given
full KV cache access. Notably, the performance degradation pattern at 0.8 sparsity directly reflects
the relationship between _reasoning_ _head_ quantity and capability: as sparsity increases (retaining
fewer reasoning heads), performance systematically decreases. This trend demonstrates that complex reasoning fundamentally depends on a sufficient number of _reasoning heads_ with full KV cache
access, making lossless compression at extreme ratios inherently challenging.


3.3 ANALYSES ON REASONING HEADS VERSUS RETRIEVAL HEADS


0.5


evaluate performance degradation on the 0.0 0.1 0.2Fraction of Top Heads Replaced0.3 0.4 0.0 0.1 0.2 0.3 0.4
Math500 benchmark. _Reasoning_ _heads_ Figure 6: The importance of heads identified is equivalently
identified by RLKV demonstrate signif- illustrated by replacing the top ratio of them with a comicantly steeper performance degradation, pressed KV cache. Compared to retrieval heads and random
indicating they are substantially more im- heads, reasoning heads identified by RLKV are more crucial
portant than retrieval heads and random to model performance, and are sensitive to compressed KV
heads. Combined with the main results in cache access.
Figure 5, this reveals an important asymmetry: compressing even a small fraction of top _reason-_
_ing heads_ causes significant degradation, while maintaining complete capability requires preserving
multiple _reasoning_ _heads_ . Qwen-2.5-7B-R1 shows more gradual degradation than Llama-3.1-8BR1 at low compression ratios (0.1 and 0.2), indicating that its reasoning capability may be more
distributed across multiple heads rather than concentrated in a few critical ones at these levels. Since
Qwen-2.5-7B-R1 achieves stronger reasoning with fewer total heads (112 vs 256), it likely utilizes
its top _reasoning heads_ more efficiently, making it more robust to small-scale compression.


0.9

0.8

0.7

0.6

0.5

0.4

0.3

0.2

0.1


Llama-3.1-8B-R1


Qwen-2.5-7B-R1


0.0
0.0 0.1 0.2 0.3 0.4


1.0

0.9

0.8

0.7

0.6

0.5

0.4

0.3

0.2

0.1


0.0
0.0 0.1 0.2 0.3 0.4


Fraction of Top Heads Replaced


Figure 6: The importance of heads identified is equivalently
illustrated by replacing the top ratio of them with a compressed KV cache. Compared to retrieval heads and random
heads, reasoning heads identified by RLKV are more crucial
to model performance, and are sensitive to compressed KV
cache access.


6


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
|~~Full~~<br>w/o a<br>~~penal~~<br>|daptive<br>~~ty weight~~<br>||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
|~~Full~~<br>w/o se<br>~~sampli~~<br>|lf-distillati<br>~~ng~~<br>|on|||


|Col1|Col2|Col3|
|---|---|---|
||||
||||
||||
||||
|Full<br>beta =<br>~~beta =~~<br>|5e-3 (larger)<br>~~  2e-4 (smalle~~<br>|~~   r)~~|


3.4 MEMORY EFFICIENCY


To demonstrate RLKV’s memory efficiency, we evaluate its compression performance while maintaining accuracy across two reasoning models and four benchmarks, as shown in Table 1 (a) and Table 1 (b). Values show performance with difference from full KV cache in parentheses, where light
green indicates performance exceeding the full KV cache baseline and light red indicates performance below it. RLKV consistently outperforms baselines across all sparsity levels, achieving GPU
memory reductions of 20-50% with minimal performance degradation across different models and
benchmarks. Notably, different reasoning tasks exhibit varying sensitivity to compression, reflecting the heterogeneous and complex mechanisms underlying _reasoning_ _head_ functionality. When
generation length exceeds 8k, 16k, or even 32k tokens, RLKV enables deployment on memoryconstrained hardware and allows for higher inference parallelism by reducing memory bottlenecks.


3.5 ABLATION STUDIES


We conduct ablation studies using Qwen-2.5-7B-R1 on the Math500 benchmark to assess the impact
of adaptive penalty weighting, self-distillation sampling, and base L1 penalty weight in RLKV.


**Adaptive** **Penalty** **Weighting.** Figure 8 (left) demonstrates that adaptive penalty weighting significantly enhances performance by breaking the vicious cycle between sparse rewards and dense


7


Effect of Adaptive Penalty Weight

1.0

0.9

0.8

0.7

0.6

0.5


0.9

0.8

0.7

0.6

0.5


1.0

0.9

0.8

0.7

0.6

0.5

0.4

0.3

0.2

0.1


Different Beta Hyper-Parameter


0.4

0.3

0.2

0.1


0.4

0.3

0.2

0.1


0.0
0.0 0.2 0.4 0.6 0.8


0.0
0.0 0.2 0.4 0.6 0.8


0.0
0.0 0.2 0.4 0.6 0.8


KV Cache Budget Sparsity


Figure 8: **Ablation study on key components of RLKV training method.** We evaluate three critical components using Qwen-2.5-7B-R1 on Math500. _Left_ : Adaptive penalty weighting prevents
training collapse by stabilizing conflicting dynamics between sparse rewards and L1 penalty, while
its absence leads to ineffective exploration and training failure. _Middle_ : Self-distillation sampling
maintains stable reward signals by training on appropriately challenging problems, compared to unstable signals from overly difficult problems. _Right_ : Base L1 penalty weight _β_ = 0 _._ 001 achieves
optimal sparsity-performance balance, while excessive penalty causes over-compression and insufficient penalty leads to premature convergence.


**Error** **Mode** **Analyses** We analyze the
distinct error modes exhibited by models when _reasoning_ _heads_ and retrieval
heads guide KV cache compression on the
Math500 benchmark. Error modes are categorized into three types: repetitive errors
(excessively repeating token sequences),
incorrect errors (generating wrong answers), and overlength errors (generating Figure 7: The analysis reveals distinct error modes when reasequences that exceed normal length base- soning heads versus retrieval heads work with compressed
lines). Figure 7 reveals that models tend to KV cache on Math500 benchmark. Reasoning heads tend toproduce repetitive generation errors when ward repetitive generation errors as compression increases,
_reasoning heads_ are compressed at higher while retrieval heads exhibit more varied error modes across

different settings.

levels, while models with compressed retrieval heads exhibit more varied error modes across different settings. This consistency in _reason-_
_ing head_ -related errors suggests their collaborative role in maintaining complex logical states during
reasoning, whereas retrieval heads appear to have more multifaceted roles. See Appendix A.4 for
more details.


Figure 7: The analysis reveals distinct error modes when reasoning heads versus retrieval heads work with compressed
KV cache on Math500 benchmark. Reasoning heads tend toward repetitive generation errors as compression increases,
while retrieval heads exhibit more varied error modes across
different settings.


Table 1: RLKV achieves near lossless performance (full KV cache) up to the sparsity thresholds
shown for Llama-3.1-8B-R1 (a) and Qwen-2.5-7B-R1 (b) across four benchmarks. Red background
denotes performance below the full–KV-cache baseline, whereas green background denotes performance above it. RLKV exhibits the smallest performance degradation among the other methods
and, on some benchmarks, even improves over the full–KV-cache baseline. For all values, higher
is better. The best result of the metric in each benchmark is in **bold** . All values are reported as
percentages.


Lossless Sparsity Threshold
Method

GSM8K (Math) Math500 (Math) AIME24 (Math) MBPP (Code)
0.4 0.5 0.4 0.4


**(a) Llama-3.1-8B-R1**


Lossless Sparsity Threshold
Method

GSM8K (Math) Math500 (Math) AIME24 (Math) MBPP (Code)
0.4 0.4 0.2 0.3


**(b) Qwen-2.5-7B-R1**


L1 penalty. Without this mechanism, increasing adapter sparsity leads to degraded reasoning performance, which generates sparser reward signals while the L1 penalty remains dense, creating an
imbalance that drives training toward collapse with no recovery capability.


**Self-distillation** **Sampling.** Self-distillation sampling provides stable reward signals throughout
training, as shown in Figure 8 (middle). In contrast to typical RLVR that utilizes sparse rewards
for capability enhancement, our work leverages RL for capability preservation under sparsity constraints. Training on problems suited to the model’s reasoning capability maintains relatively stable
reward signals throughout optimization, while training on overly challenging problems leads to unstable and sparse reward signals that provide weak and insufficient guidance for head identification.


**Base** **L1** **penalty** **Weight.** The base regularization weight _β_ controls the strength of L1 penalty
applied to gating adapters during RL training. Figure 8 (right) shows that a moderate _β_ value
of 0.001 achieves an optimal balance between sparsity and reward signal strength. Excessive
penalty ( _β_ = 0 _._ 005) dominates the optimization process, weakening reward signals through overcompression, while insufficient penalty ( _β_ = 0 _._ 0002) fails to induce adequate sparsity, leading to
premature convergence with limited exploration of the reward landscape.


4 RELATED WORK


**Efficient LLM Inference.** Various techniques reduce KV cache overhead through architectural or
system optimizations. Grouped-Query Attention (GQA) (Ainslie et al., 2023) and Multi-head Latent
Attention (MLA) (Liu et al., 2024a) reduce the number of KV heads by sharing them across query
heads, achieving significant memory reduction but requiring expensive pre-training from scratch.
Linear attention methods (Gu & Dao, 2023; Yang et al., 2025b) maintain constant memory usage
during inference by avoiding the quadratic attention computation, but exhibit reduced modeling
capacity compared to standard transformer architectures. KV cache quantization (Liu et al., 2024b;
Tao et al., 2025; Hooper et al., 2024; Duanmu et al., 2024; Su et al., 2025; Yue et al., 2024) and
system-level optimizations, such as paged KV cache (Kwon et al., 2023), KV cache reuse (Zheng


8


et al., 2024), and sparsely loading KV cache (Tang et al., 2024b), provide orthogonal improvements
by reducing the precision or optimizing the storage/retrieval of cached states. While valuable, these
methods treat KV cache as opaque data without exploiting the inherent sparsity patterns.


**KV Cache Compression.** Recent works mainly exploit sparsity in long-context scenarios for instruct models, including token-dropping and head-reallocation methods. (1) Token-dropping methods (Zhang et al., 2023; Li et al., 2024; Cai et al., 2025; Yang et al., 2024b; Qin et al., 2024) apply
eviction strategies across all heads or intra-layer heads based on attention scores. H2O (Zhang et al.,
2023) maintains important tokens’ KV cache based on accumulated attention scores plus a sliding
window for recent tokens. Specifically, recent R-KV (Cai et al., 2025), designed for reasoning models, primarily adds similarity-based clustering to priority evict redundancy tokens’ KV cache during
both prefill and decoding phases. However, they inevitably discard reasoning-critical information
and disrupt the CoT consistency as compression rates increase. (2) head-reallocation methods (Fu
et al., 2024; Tang et al., 2024a; Xiao et al., 2024; Bhaskar et al., 2025) maintain full KV cache only
for identified retrieval heads (Wu et al., 2024) in long-context scenarios while applying compressed
KV cache (Xiao et al., 2023) to others. Ada-KV (Fu et al., 2024) and RazorAttention (Tang et al.,
2024a) use proxy metrics of attention scores, while DuoAttention (Xiao et al., 2024) and PruLong
(Bhaskar et al., 2025) are learning-based methods for head identification. DuoAttention minimizes
single-forward output deviation on a synthetic long-context recall task, while PruLong uses nexttoken loss on long-context pre-training corpora. However, these methods do not capture the reasoning behaviors that emerge during dynamically extending CoT generation, resulting in degraded
reasoning performance as compression rates increase.


**Reinforcement** **Learning** **for** **Efficiency.** RL has proven effective in Neural architecture search
(Zoph & Le, 2017; Zoph et al., 2018), where it treats architecture choices as sequential decisions,
and model pruning (He et al., 2018), where it learns layer-wise pruning ratios that maximize accuracy under resource constraints. However, the limitation is the high computational cost due to
the large optimization space. Our work utilizes gating values assigned to each KV head to reduce
the optimization space and make RL feasible and efficient. For reasoning language models, recent
works apply RL tuning to mitigate overthinking (Hou et al., 2025; Liu et al., 2025) by learning to
reduce CoT length while maintaining reasoning capability, thereby indirectly decreasing KV cache
requirements. Our work is orthogonal to these methods, employing lightweight RL training to identify _reasoning heads_ that guide KV cache compression while preserving reasoning capability.


5 CONCLUSION


In this paper, we propose RLKV, a novel reasoning-critical head identification method to guide
KV cache compression in reasoning models. RLKV directly optimizes the relationship between
each head’s KV cache usage and reasoning quality through reinforcement learning and we achieve
competitive performance at diverse KV cache budget sparsity levels and reduce 20-50% KV cache
usage while preserving full reasoning capability across Llama-3.1-8B-R1 and Qwen-2.5-7B-R1 on
GSM8K, MATH, AIME24, and MBPP benchmarks. Then we analyze the _reasoning heads_ importance and error modes, revealing the importance and complexity of _reasoning_ _heads_ in reasoning
models. RLKV provides a new perspective on understanding reasoning models and opens up new
avenues for efficient inference of reasoning LLMs.


6 FUTURE WORK


RLKV opens several promising avenues for future research. First, the significant variability in
_reasoning_ _heads_ distribution across different models and tasks presents an exciting opportunity to
develop a deeper understanding of the heterogeneous nature of reasoning mechanisms in reasoning
LLMs. Second, while RLKV effectively identifies _reasoning heads_ for compression, exploring the
complete functional roles of these heads beyond reasoning could unlock new insights into model interpretability and architectural design. Third, advancing compression techniques to maintain strong
performance at extremely high compression ratios (80% and above) represents a compelling challenge that could further bridge the gap between memory efficiency and reasoning capability preservation. These research directions hold significant potential for advancing both our understanding of
reasoning in large language models and their practical deployment efficiency.


9


REFERENCES


Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit
Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In _EMNLP_, 2023.


Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan,
Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language
models. _arXiv preprint arXiv:2108.07732_, 2021.


Adithya Bhaskar, Alexander Wettig, Tianyu Gao, Yihe Dong, and Danqi Chen. Cache me if you can:
How many kvs do you need for effective long-context lms? _arXiv_ _preprint_ _arXiv:2506.17121_,
2025.


Zefan Cai, Wen Xiao, Hanshi Sun, Cheng Luo, Yikai Zhang, Ke Wan, Yucheng Li, Yeyang Zhou, LiWen Chang, Jiuxiang Gu, et al. R-kv: Redundancy-aware kv cache compression for training-free
reasoning models acceleration. _arXiv preprint arXiv:2505.24133_, 2025.


Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to
solve math word problems. _arXiv preprint arXiv:2110.14168_, 2021.


Google DeepMind. Gemini. https://deepmind.google/models/gemini/, 2025.


Haojie Duanmu, Zhihang Yuan, Xiuhong Li, Jiangfei Duan, Xingcheng Zhang, and Dahua Lin.
Skvq: Sliding-window key and value cache quantization for large language models. _arXiv preprint_
_arXiv:2405.06219_, 2024.


Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, et al. The Llama 3 Herd of Models, July 2024.


Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun
Mei, Jiashu Wang, et al. Areal: A large-scale asynchronous reinforcement learning system for
language reasoning. _arXiv preprint arXiv:2505.24298_, 2025.


Yu Fu, Zefan Cai, Abedelkadir Asi, Wayne Xiong, Yue Dong, and Wen Xiao. Not all heads matter:
A head-level kv cache compression method with integrated retrieval and reasoning. _arXiv preprint_
_arXiv:2410.19258_, 2024.


Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. _arXiv_
_preprint arXiv:2312.00752_, 2023.


Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms
via reinforcement learning. _arXiv preprint arXiv:2501.12948_, 2025.


Junxian Guo, Haotian Tang, Shang Yang, Zhekai Zhang, Zhijian Liu, and Song Han. Block Sparse
Attention. [https://github.com/mit-han-lab/Block-Sparse-Attention,](https://github.com/mit-han-lab/Block-Sparse-Attention)
2024.


Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, and Song Han. Amc: Automl for model
compression and acceleration on mobile devices. In _ECCV_, 2018.


Coleman Richard Charles Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney,
Sophia Shao, Kurt Keutzer, and Amir Gholami. Kvquant: Towards 10 million context length
llm inference with kv cache quantization. In _NeurIPS_, 2024.


Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang.
Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning. _arXiv_ _preprint_
_arXiv:2504.01296_, 2025.


Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec
Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. _arXiv_
_preprint arXiv:2412.16720_, 2024.


10


Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph
Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model
serving with pagedattention. In _Proceedings of the 29th symposium on operating systems princi-_
_ples_, 2023.


Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle
Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before
generation. _NeurIPS_, 2024.


Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan
Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In _ICLR_, 2023.


Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong
Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixtureof-experts language model. _arXiv preprint arXiv:2405.04434_, 2024a.


Wei Liu, Ruochen Zhou, Yiyun Deng, Yuzhen Huang, Junteng Liu, Yuntian Deng, Yizhe Zhang,
and Junxian He. Learn to reason efficiently with adaptive length-based reward shaping. _arXiv_
_preprint arXiv:2505.15612_, 2025.


Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi
Chen, and Xia Hu. Kivi: A tuning-free asymmetric 2bit quantization for kv cache. In _ICML_,
2024b.


Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv_ _preprint_
_arXiv:1711.05101_, 2017.


Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Tang, Manan Roongta, Colin
Cai, Jeffrey Luo, Tianjun Zhang, Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025. URL [https:](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview\-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)
[//pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview\](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview\-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)
[-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2.](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview\-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)
Notion Blog.


MMA. American invitational mathematics examination - aime, February 2024. URL [https://maa.org/math-competitions/](https://maa.org/math-competitions/american-invitational-mathematics-examination-aime)
[american-invitational-mathematics-examination-aime.](https://maa.org/math-competitions/american-invitational-mathematics-examination-aime)


Ziran Qin, Yuchen Cao, Mingbao Lin, Wen Hu, Shixuan Fan, Ke Cheng, Weiyao Lin, and Jianguo
Li. Cake: Cascading and adaptive kv cache eviction with layer preferences. In _ICLR_, 2024.


Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. _arXiv preprint arXiv:2402.03300_, 2024.


Zunhai Su, Zhe Chen, Wang Shen, Hanyu Wei, Linge Li, Huangqi Yu, and Kehong Yuan. Rotatekv:
Accurate and robust 2-bit kv cache quantization for llms via outlier-aware adaptive rotations.
_arXiv preprint arXiv:2501.16383_, 2025.


Hanlin Tang, Yang Lin, Jing Lin, Qingsen Han, Danning Ke, Shikuan Hong, Yiwu Yao, and Gongyi
Wang. Razorattention: Efficient kv cache compression through retrieval heads. In _ICLR_, 2024a.


Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. Quest:
query-aware sparsity for efficient long-context llm inference. In _ICML_, 2024b.


Keda Tao, Haoxuan You, Yang Sui, Can Qin, and Huan Wang. Plug-and-play 1. x-bit kv cache
quantization for video large language models. _arXiv preprint arXiv:2503.16257_, 2025.


Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun
Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with
llms. _arXiv preprint arXiv:2501.12599_, 2025.


Robert Tibshirani. Regression shrinkage and selection via the lasso. _Journal of the Royal Statistical_
_Society Series B: Statistical Methodology_, 58(1):267–288, 1996.


11


Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming
Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging
multi-task language understanding benchmark. In _NeurIPS_, 2024.


Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality. _arXiv preprint arXiv:2404.15574_, 2024.


Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming
language models with attention sinks. In _ICLR_, 2023.


Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu,
and Song Han. Duoattention: Efficient long-context llm inference with retrieval and streaming
heads. In _ICLR_, 2024.


An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical
expert model via self-improvement. _arXiv preprint arXiv:2409.12122_, 2024a.


An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. _arXiv_ _preprint_
_arXiv:2505.09388_, 2025a.


Dongjie Yang, Xiaodong Han, Yan Gao, Yao Hu, Shilin Zhang, and Hai Zhao. Pyramidinfer: Pyramid kv cache compression for high-throughput llm inference. In _ACL_, 2024b.


Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. In _NeurIPS_, 2025b.


Yuxuan Yue, Zhihang Yuan, Haojie Duanmu, Sifan Zhou, Jianlong Wu, and Liqiang Nie. Wkvquant:
Quantizing weight and key/value cache for large language models gains more. _arXiv_ _preprint_
_arXiv:2402.12065_, 2024.


Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song,
Yuandong Tian, Christopher R´e, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2o: Heavyhitter oracle for efficient generative inference of large language models. In _NeurIPS_, 2023.


Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody H. Yu, Shiyi Cao,
Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. Sglang:
Efficient execution of structured language model programs. In _NeurIPS_, 2024.


Barret Zoph and Quoc Le. Neural architecture search with reinforcement learning. In _ICLR_, 2017.


Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V. Le. Learning transferable architectures
for scalable image recognition. In _CVPR_, 2018.


12


DECLARATION OF THE USE OF LARGE LANGUAGE MODELS


In this paper, we only use LLMs to help with grammar checking and polishing the writing. All
conceptual contributions, framework design, implementation, and experimental evaluations were
performed by the authors without assistance from LLMs.


A APPENDIX


A.1 MOTIVATION STUDY


We provide a comprehensive motivation study on two mainstream small reasoning models (Llama3.1-8B-R1 and Qwen-2.5-7B-R1 (Guo et al., 2025)) and their instruct variants (Llama-3.1-8B-Inst
(Dubey et al., 2024) and Qwen-2.5-7B-Inst [1] (Yang et al., 2024a)). We conduct the evaluation on
two typical token-dropping methods (H2O (Zhang et al., 2023) and R-KV (Cai et al., 2025)) and
one head-reallocation method (DuoAttention (Xiao et al., 2024)) across four benchmarks, including GSM8K (Cobbe et al., 2021), Math500 (Lightman et al., 2023), AIME24(MMA, 2024), MBPP FIX
(Austin et al., 2021). Figure 9 presents that all compression methods maintain relatively stable performance on instruct models but drop substantially on reasoning models as compression increases.


We further analyze the error modes on reasoning models in the above evaluation. We observed three
error modes: repetitive errors (excessively repeating token sequences), incorrect errors (generating
wrong answers), and overlength errors (generating sequences that exceed normal length baselines),
as illustrated in Figure 10. The detailed error modes can be seen in Figure 11.


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||
||||||


KV Cache Budget Sparsity


Figure 9: Comprehensive evaluation of KV cache compression methods across all model pairs and
benchmarks reveals consistent patterns of performance degradation. H2O, R-KV, and DuoAttention
maintain relatively stable performance on instruction-following models but exhibit significant drops
on their reasoning counterparts as the KV cache budget decreases. This performance degradation
becomes particularly severe at higher sparsity levels, with notable declines observed on reasoningintensive benchmarks including GSM8k, Math500, AIME24, and MBPP.


A.2 EXPERIMENT DETAILS


**Dataset Construction.** We construct training data from the DeepScaleR dataset (Luo et al., 2025),
which contains about 40,000 diverse and challenging mathematical reasoning problems. For each
model, we generate solutions using the respective reasoning model with greedy decoding, filter


1We use Qwen-2.5-Math-7B-Instruct (Yang et al., 2024a) as the instruct baseline, abbreviated as Qwen2.5-7B-Inst for naming consistency, since Qwen-2.5-7B-R1 (deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) was
based on Qwen-2.5-Math-7B


13


H2O R-KV DuoAttn Reasoning Instruct


Llama-3.1-8B series


Llama-3.1-8B series


0.0


0.1


0.2


0.3


0.0

0.1

0.2

0.3

0.4

0.5

0.6


0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8


Llama-3.1-8B series


Llama-3.1-8B series


0.9
0.0 0.2 0.4 0.6 0.8


0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8


0.9
0.0 0.2 0.4 0.6 0.8


0.4
0.0 0.2 0.4 0.6 0.8


0.7
0.0 0.2 0.4 0.6 0.8


Qwen-2.5-7B series


0.0


0.1


0.2


0.3


0.4


0.0

0.1

0.2

0.3

0.4

0.5

0.6


0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
0.0 0.2 0.4 0.6 0.8


0.0

0.1

0.2

0.3

0.4

0.5

0.6

0.7

0.8


0.9
0.0 0.2 0.4 0.6 0.8


Qwen-2.5-7B series


Qwen-2.5-7B series


Qwen-2.5-7B series


0.5
0.0 0.2 0.4 0.6 0.8


0.7
0.0 0.2 0.4 0.6 0.8


FIX


Figure 10: The instances of three error modes.


Figure 11: Comprehensive error mode analyses of KV cache compression methods across reasoning
models reveal distinct failure patterns. Token-dropping methods (H2O, R-KV) consistently exhibit
repetitive errors, as they inevitably discard reasoning-critical information during compression. In
contrast, the head-reallocation method DuoAttention tends to show more over-length errors compared to token-dropping methods, suggesting that while it relatively preserves sequence information
integrity, it still struggles to fully preserve reasoning capability.


correct solutions, then randomly sample 3,000 problems for training. The selected problems are
distributed across different output token lengths as follows: 600 problems each for 0-2k and 2k-4k
tokens, 1,000 problems for 4k-6k tokens, and 800 problems for 6k-8k tokens.


**Hardware** **and** **Hyperparameter** **Settings.** All experiments are conducted on 2 NVIDIA A100
GPUs (80GB) for several hours, one for backward computation and one for sample generation.
Training runs for 2 epochs, totaling 185 steps with a batch size of 32. All evaluations are conducted
on NVIDIA RTX5090 GPUs. We optimize the gating adapters using AdamW optimizer with _β_ 1 =
0 _._ 9, _β_ 2 = 0 _._ 999, weight decay of 0.017, and learning rate of 0.01 with constant schedule. For
GRPO training configuration, we disable KL penalty and use recommendation setting of AReaL;
for GRPO sampling configuration, we use 4 samples per query with sampling temperature of 1.0.
The hyperparameters are shown in Table 2.


**Local Attention Implementation.** During training, we employ an efficient block-sparse attention
approximation implementation (Guo et al., 2024) in AReaL (Fu et al., 2025) to update adapter
weights, while using mask matrices for prefilling and custom Triton kernels for decoding in SGLang
(Zheng et al., 2024) to generate samples. For inference, we only store the partial KV cache of first
16 sink tokens and recent 64 local tokens for non-reasoning heads, while _reasoning heads_ maintain
the full KV cache.


14


Llama-3.1-8B-R1


Llama-3.1-8B-R1


GSM8K (Math)


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


Llama-3.1-8B-R1

Math500 (Math)


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


Llama-3.1-8B-R1

AIME24 (Math)


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


MBPP (Code)


Qwen-2.5-7B-R1


Qwen-2.5-7B-R1


GSM8K (Math)


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


Qwen-2.5-7B-R1
Math500 (Math)


KV Cache Budget Sparsity


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


Qwen-2.5-7B-R1

AIME24 (Math)


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


MBPP (Code)


Table 2: Training Hyperparameters.


Parameter Llama-3.1-8B-R1 Qwen-2.5-7B-R1


Regularization weight _β_ 1e-3 1e-3
Reward threshold _τ_ 0.5 0.55
Sink token size 128 128
Local token size 256 256
Max sequence length 8192 8192


**Baseline** **Implementation.** To ensure fair comparison with baseline methods, we make several
adjustments. For H2O and R-KV, we augment them with the same sink and local token overhead
(16+64 tokens) that our method uses. Since H2O and R-KV only support preset fixed KV cache
budgets, we convert their fixed budgets to dynamic allocation that increases with sequence length.
For example, if the fixed budget is 50% of the full KV cache, then at sequence length 1000, they
use 500 tokens of KV cache, and at sequence length 2000, they use 1000 tokens of KV cache. For
DuoAttention, we replicate their approach with default settings on our models and use the same
inference settings as our method.


**Training** **Cost.** The training of the adapters is computationally modest: on 2 A100 GPUs, our
method consumes 40, 22, and 36 GPU-hours for Llama-3.1-8B-R1, Qwen-2.5-7B-R1, and Qwen-34B-Thinking, respectively. FIX


**Evaluation** **Settings.** We evaluate all methods using greedy decoding on RTX 5090 36G GPUs
or RTX 4090 24G GPUs with batch size of 1. For all datasets, we use regex to extract the final
answer from the generated text, using Pass@1 as the evaluation metric. For GSM8K, Math500, and
MBPP, we use 8192 max sequence length; for AIME24, we use 16384 max sequence length. We
achieved near official reported performance without KV cache compression. We use eager attention
implementation for H2O and R-KV since they need to use attention scores, while we use flash
attention for DuoAttention and our method.


**Prompt** **Template.** We follow the prompt setting recommended by DeepSeek-R1 (Guo et al., FIX
2025) in both training and evaluation without additional prompt engineering. For example, we use
the following template in math problems:


Solve the following math problem efficiently and
clearly. The last line of your response should
be of the following format: ’Therefore, the final
answer is: $\\boxed{ANSWER}$. I hope it is correct’
(without quotes) where ANSWER is just the final number
or expression that solves the problem. Think step by
step before answering.


**QUESTION**


A.3 FULL RESULTS


Tables 3 and 4 present the complete numerical results of RLKV and baselines for Llama-3.1-8B-R1
and Qwen-2.5-7B-R1 respectively, across all benchmarks and KV cache compression budgets. Values in parentheses indicate the performance difference compared to the full KV cache setting, with
positive values in green indicating improvement and negative values in red indicating degradation.


A.4 DETAILS OF ERROR MODES ANALYSES


Figure 12 presents the comprehensive error mode analysis across all models and benchmarks. We
observe three error modes: repetitive errors (excessively repeating token sequences), incorrect errors (generating wrong answers), and overlength errors (generating sequences that exceed normal
length baselines). Our method RLKV shows consistency in error modes across different models and
benchmarks, while DuoAttention exhibits more varied error modes across different settings.


15


Table 3: Llama-3.1-8B-R1 performance (%) under different KV cache compression methods and
budgets. RLKV ( **Ours** ) shows competitive performance across settings. Red background denotes
performance below the full–KV-cache baseline, whereas green background denotes performance
above it. For all values, higher is better. The best result of the metric in each benchmark is in **bold** .


KV Cache Budget Sparsity
Dataset Method

0.2 0.4 0.6 0.8


A.5 EVALUATION ON QWEN-3-4B-THINKING
NEW


We further evaluate RLKV and the baselines on Qwen-3-4B-Thinking, a newly released and powerful reasoning model (Yang et al., 2025a) to validate the effectiveness of our method. The evaluation
is conducted on four reasoning benchmarks (GSM8K, MATH, AIME24, MBPP) at sparsity levels
of 0.2, 0.4, 0.6, and 0.8, following the same settings as in the main experiment. As shown in Figure 13 and Table 5, RLKV on Qwen-3-4B-Thinking exhibits performance trends similar to those ob

16


Table 4: Qwen-2.5-7B-R1 performance (%) under different KV cache compression methods and
budgets. RLKV ( **Ours** ) shows competitive performance across settings. Red background denotes
performance below the full–KV-cache baseline, whereas green background denotes performance
above it. For all values, higher is better. The best result of the metric in each benchmark is in **bold** .


KV Cache Budget Sparsity
Dataset Method

0.2 0.4 0.6 0.8


KV Cache Budget Sparsity


Figure 13: Performance comparison of RLKV against KV cache compression baselines across reasoning benchmarks. We evaluate RLKV ( **Ours** ) and existing methods on Qwen-3-4B-Thinking
across four benchmarks (GSM8K, MATH, AIME24, MBPP) at sparsity levels of 0.2, 0.4, 0.6, and
0.8. RLKV consistently outperforms all baselines across 0.2-0.6 sparsity levels, but performance
drops significantly at 0.8 sparsity due to extreme compression. The results demonstrate particularly
strong advantages at 0.4 or 0.6 sparsity levels where competing methods suffer significant performance degradation.


served in our previous evaluations on Llama-3.1-8B-R1 and Qwen-2.5-7B-R1. RLKV outperforms
all baselines at sparsity levels 0.2, 0.4, and 0.6, but, due to extreme compression, suffers performance drops at 0.8 sparsity, similar to the baselines. Table 6 shows the maximum sparsity at which
RLKV on Qwen-3-4B-Thinking maintains lossless performance compared to the uncompressed settings across the four benchmarks. RLKV achieves 50% memory reduction on GSM8K, Math500,
and AIME24, and 30% on MBPP, while the baselines suffer significant performance degradation at
these sparsity levels. Compared to the results on Llama-3.1-8B-R1 and Qwen-2.5-7B-R1, RLKV
on Qwen-3-4B-Thinking attains a higher maximum sparsity without performance loss, suggesting
better compression capability on stronger models. This trend further supports the effectiveness of
RLKV across different model architectures and scales.


A.6 FOUR SUBSETS OF MMLU-PRO ON LLAMA-3.1-8B-R1 AND QWEN-2.5-7B-R1
NEW


We further validate RLKV on generalization beyond the training math domain. We evaluate RLKV
and baselines on the four subsets of the challenging knowledge QA benchmark MMLU-Pro (Wang
et al., 2024), including MMLU-Pro-Chemistry, MMLU-Pro-Computer-Science, MMLU-Pro-Law,
and MMLU-Pro-Physics. Due to the time constraints, we randomly sample 200 examples from each
subset for evaluation on Llama-3.1-8B-R1 and Qwen-2.5-7B-R1. As shown in Figure 14, RLKV
achieves comparable or better accuracy than baselines on these four subsets across four sparsity
settings (0.2, 0.4, 0.6, 0.8). For Law on Qwen-2.5-7B-R1 and Physics on Llama-3.1-8B-R1, RLKV


17


Llama-3.1-8B-R1


Llama-3.1-8B-R1


**DuoAttention (retrieval heads)** **RLKV (reasoning heads)**
Repetitive Incorrect Overlength


GSM8K (Math)


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


Llama-3.1-8B-R1

Math500 (Math)


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


Llama-3.1-8B-R1

AIME24 (Math)


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


MBPP (Code)


Qwen-2.5-7B-R1


Qwen-2.5-7B-R1


GSM8K (Math)


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


Qwen-2.5-7B-R1
Math500 (Math)


KV Cache Budget Sparsity


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


Qwen-2.5-7B-R1

AIME24 (Math)


1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.2 0.4 0.6 0.8


MBPP (Code)


Figure 12: The analysis reveals distinct error patterns when reasoning heads versus retrieval heads
work with compressed KV cache across four benchmarks.


GSM8K (Math)

1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.0 0.2 0.4 0.6 0.8


Math500 (Math)
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.0 0.2 0.4 0.6 0.8


Full H2O R-KV DuoAttn Ours


AIME24 (Math)
0.6


0.5

0.4

0.3

0.2

0.1

0.0
0.0 0.2 0.4 0.6 0.8


MBPP (Code)

0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.0 0.2 0.4 0.6 0.8


Table 5: Qwen-3-4B-Thinking performance (%) under different KV cache compression methods and
budgets. RLKV ( **Ours** ) shows competitive performance across settings. Red background denotes
performance below the full–KV-cache baseline, whereas green background denotes performance
above it. For all values, higher is better. The best result of the metric in each benchmark is in **bold** .


KV Cache Budget Sparsity
Dataset Method

0.2 0.4 0.6 0.8


Table 6: RLKV achieves near lossless performance (full KV cache) up to the sparsity thresholds
shown for Qwen-3-4B-Thinking across four benchmarks. Red background denotes performance
below the full-KV-cache baseline, whereas green background denotes performance above it. RLKV
exhibits the smallest performance degradation among the other methods and, on some benchmarks,
even improves over the full-KV-cache baseline. For all values, higher is better. The best result of
the metric in each benchmark is in **bold** . All values are reported as percentages.


Lossless KV Cache Budget Sparsity on each Dataset
Method

GSM8K (Math) Math500 (Math) AIME24 (Math) MBPP (Code)
0.5 0.5 0.5 0.3


cannot achieve near lossless compression even at sparsity 0.2. Although this suggests a limitation in
RLKV for these specific model-task combinations, it still outperforms other methods.


A.7 AN IMPLICITLY UNFAIR COMPARISON IN FIXED-BUDGET EVALUATION
NEW


This section discusses the motivation for using a dynamic budget instead of a fixed budget for KV
cache compression evaluation. Existing long-context compression works (Li et al., 2024; Yang et al.,
2024b; Qin et al., 2024; Fu et al., 2024; Tang et al., 2024a; Xiao et al., 2024; Bhaskar et al., 2025)
typically evaluate on in-context recall tasks, where each sample’s prompt length is fixed/controlled.
A fixed budget of the form budget = sparsity _×_ prompt ~~l~~ ength then yields a roughly consistent
compression ratio per sample, so fixed budgets are fair in that setting.


For reasoning tasks, however, the response length is often much larger than the prompt, as shown
in Figure 15. If we use a global fixed budget (e.g., 1k tokens), any sample whose full output fits


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


Full H2O R-KV DuoAttn Ours


KV Cache Budget Sparsity


Figure 14: Performance comparison of RLKV against KV cache compression baselines across four
subsets of the MMLU-Pro benchmark, including Chemistry, Computer Science, Law, and Physics.
We evaluate RLKV ( **Ours** ) and existing methods on two reasoning models (Llama-3.1-8B-R1 and
Qwen-2.5-7B-R1) across four benchmarks (GSM8K, MATH, AIME24, MBPP) at sparsity levels of
0.2, 0.4, 0.6, and 0.8. RLKV consistently outperforms all baselines across different sparsity levels,
demonstrating the generalization beyond the reasoning domain.


within 1k tokens is uncompressed, while longer samples are compressed. Thus, different samples
experience very different compression ratios, and fixed budgets are not fair at the per-sample level.


In R-KV (Cai et al., 2025), the reported compression rate is computed as budget _/_ average ~~f~~ ull ~~l~~ ength.
For example, R-KV achieves the compression ratio of 66.2% for Math500 on Llama-3.1-8B-R1,
with a fixed budget of 200 and an average full length of 3019. However, a large fraction of samples
are uncompressed and thus produce the same responses as the full model. This makes the reported
compression ratio optimistic.


A.8 COMPARISON OF FIXED BUDGET AND DYNAMIC BUDGET FOR R-KV AND H2O
NEW


In our evaluations, we adopt a dynamic budget strategy where each sample’s budget is determined by
its full length multiplied by the target sparsity, to ensure consistent compression ratios across samples. To illustrate the impact of this choice, we compare the performance of R-KV and H2O under
both fixed and dynamic budget settings on Llama-3.1-8B-R1, Qwen-2.5-7B-R1, and Qwen-3-4BThinking across Math500 and AIME24 at sparsity levels of 0.2, 0.4, 0.6, and 0.8. In this comparison,
the fixed budget per-sample is estimated as budget(sample) = sparsity _×_ full ~~l~~ ength(sample), where
full ~~l~~ ength(sample) is the length of the response generated by the full KV cache model for that
specific sample.


As shown in Figure 16, fixed-budget R-KV performs significantly worse than our dynamic-budget
variant at 0.2, 0.4, and 0.6 sparsity, and only becomes better at 0.8 sparsity, while H2O maintains
similar performance. This shows that our modification does not weaken the baselines; instead, it
corrects an overly optimistic compression estimate and yields a more faithful comparison.


A.9 DETAILED LATENCY MEASUREMENTS AND END-TO-END SPEEDUP
NEW


This section reports detailed per-layer latency measurements of attention with compressed KV cache
by the head-reallocation diagram, and the end-to-end speedups of our simple PyTorch implementation.


Compared to the full model, the head-reallocation method needs to rearrange the Q, K, and V tensors
into two dense groups in each attention computation: one for full-KV heads and one for compressedKV heads. It then computes attention separately for these two groups and finally concatenates the


19


0.7

0.6

0.5

0.4

0.3

0.2

0.1


0.6

0.5

0.4

0.3

0.2

0.1


0.6

0.5

0.4

0.3

0.2

0.1


0.6

0.5

0.4

0.3

0.2

0.1


MMLU-Pro
(Chemistry, 200)


MMLU-Pro
(Computer Science, 200)


MMLU-Pro
(Law, 200)


MMLU-Pro
(Physics, 200)


0.0
0.0 0.2 0.4 0.6 0.8


0.0
0.0 0.2 0.4 0.6 0.8


0.0
0.0 0.2 0.4 0.6 0.8


0.0
0.0 0.2 0.4 0.6 0.8


0.7

0.6

0.5

0.4

0.3

0.2

0.1


0.6

0.5

0.4

0.3

0.2

0.1


0.7

0.6

0.5

0.4

0.3

0.2

0.1


0.7

0.6

0.5

0.4

0.3

0.2

0.1


MMLU-Pro
(Chemistry, 200)


MMLU-Pro
(Computer Science, 200)


MMLU-Pro
(Law, 200)


MMLU-Pro
(Physics, 200)


0.0
0.0 0.2 0.4 0.6 0.8


0.0
0.0 0.2 0.4 0.6 0.8


0.0
0.0 0.2 0.4 0.6 0.8


0.0
0.0 0.2 0.4 0.6 0.8


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


Tokens per Sample Distribution with Full KV Cache


|Col1|Col2|I|Correc<br>ncorre|t (117<br>ct (14|6)<br>3)|Col7|
|---|---|---|---|---|---|---|
|||<br> <br>~~I~~<br>|<br>Correc<br>~~ncorre~~<br>|<br>t avg<br>~~ct av~~<br>|<br> =842<br>~~ g=322~~<br>|~~ 1~~|
||||Overal|l avg|1100||
||||||||
||||||||


|Col1|Correct (415<br>Incorrect (8|)<br>5)|Col4|
|---|---|---|---|
||<br>Correct avg<br>~~Incorrect av~~<br>|<br> =2179<br>~~ g=711~~<br>|<br>~~ 9~~<br>|
||Overall avg|3019||
|||||
|||||


|Col1|Correc<br>Incorre|t (11)<br>ct (19)|
|---|---|---|
||<br>Correc<br>Incorre<br>|<br> avg=5216<br>ct avg=13024<br>|
||Overal|avg=10161|
||||


|Col1|Correct (313)<br>Incorrect (187)<br>Correct avg=3334<br>Incorrect avg=3353|
|---|---|
||<br>Overall avg=3341|
|||


|Col1|Col2|Col3|Correc|t (117|5)|Col7|
|---|---|---|---|---|---|---|
|||<br>I<br> <br>|<br>ncorr<br>Correc<br>|<br>ct (14<br>t avg<br>|<br> 4)<br> =714<br>||
|||<br>~~I~~<br>|<br>~~ncorr~~<br>Overal|<br>~~ct av~~<br>l avg=|<br>~~ g=252~~<br> 911|~~ 6~~|
||||||||


|Col1|Correct (439|)|Col4|
|---|---|---|---|
||<br>Incorrect (6<br>~~Correct avg~~<br>|<br> 1)<br>~~ =1699~~<br>|<br>|
||<br>Incorrect av<br>~~Overall avg~~|<br> g=571<br>~~ =2188~~|<br> 0<br>|
|||||


|Col1|Correc|t (13)|
|---|---|---|
||<br>Incorre<br>Correc<br>Incorre<br>Overall|<br>ct (17)<br>t avg=3728<br>ct avg=12395<br> avg=8639|
||||


|Col1|Correct (316)|
|---|---|
||<br>~~Incorrect (184)~~<br>Correct avg=2998<br>|
||<br>~~Incorrect avg=2907~~<br>Overall avg=2964|
|||


|Col1|Col2|I<br>I|ncorre<br>Correc<br>ncorre|ct (65<br>t avg<br>ct av|)<br>=1139<br>g=315|3|
|---|---|---|---|---|---|---|
||||Overal|l avg|1239||
||||||||


|Col1|Incorrect (1<br>Correct avg<br>Incorrect av|12)<br>=2859<br>g=704|2|
|---|---|---|---|
||Overall avg|=3796||
|||||


|Col1|Incorre<br>Correc<br>Incorre|ct (17)<br>t avg=10677<br>ct avg=13248|
|---|---|---|
||Overal|avg=12134|
||||


|Col1|Correct (406) Incorrect (94)|
|---|---|
||Incorrect (94)<br>~~Correct avg=3579~~<br>Incorrect avg=3556<br>|
||Overall avg=3575|
|||


Output Length


Figure 15: The distribution of output lengths on Math500 and AIME24 benchmarks with Llama3.1-8B-R1, Qwen-2.5-7B-R1, and Qwen-3-4B-Thinking models with full KV cache.


outputs along the head dimension. This additional rearrangement and split computation introduces
overhead.


We measure the latency of an attention forward with compressed attention by head reallocation
on a single A800 GPU using the Llama-3.1-8B-R1 configuration with 32 attention heads, 8 KV
heads, and a head dimension of 128. We randomly generate query, key, and value tensors to simulate attention computation with sequence lengths from 1K to 32K and sparsity levels from 0.1
to 0.8. The batch size is set to 128 for compressed attention, and for full attention, it is set to
bound(128 _×_ (1 _−_ sparsity)) to ensure similar memory consumption between the two methods.
Throughput is calculated as throughput = batch ~~s~~ ize _/_ latency. We use a PyTorch implementation
with FlashAttention-2 for both full attention and compressed attention by head reallocation. Each
configuration is run for 10 iterations with 3 warmup iterations, and the average latency is recorded.


We report the latency ratio (compressed / full) and throughput ratio (compressed / full) at a sequence
length of 16K across different sparsity levels, and at a sparsity of 0.5 across all sequence lengths,
as shown in Figure 17. For a fixed sequence length, as the compression ratio increases, the cost
of head-wise operations approaches that of full attention and, under high compression and long
sequences, can even be slightly lower. For a fixed sparsity, as the sequence length increases, the
latency approaches that of full attention forward. Under a given memory budget, compression allows
us to increase the batch size, and for sparsity above 0.2 this leads to throughput improvements.
Given that our lossless compression typically lies in the 0.2-0.5 sparsity range, our current PyTorch
implementation does not introduce prohibitive latency. We expect that a dedicated CUDA kernel for
reorganizing the QKV tensors could further improve speed.


As for end-to-end speedup, we evaluate the impact of our method on serving latency using a standard PyTorch/Transformers inference pipeline with FlashAttention-2, without additional inference
optimizations such as quantization or continuous batching. As shown in Table 7, we still observe
end-to-end speedups when using RLKV.


The observed end-to-end speedups are smaller than the per-layer throughput gains reported above.
A key reason is the large variation in output lengths for reasoning-style workloads, as shown in Fig

20


80

70

60

50

40

30

20

10

0


70

60

50

40

30

20

10

0


100


80


60


40


20


0


8


6


4


2


0


8


6


4


2


0


10


8


6


4


2


0


100


80


60


40


20


0


60


50


40


30


20


10


0


60


50


40


30


20


10


0


400

350

300

250

200

150

100

50

0


500


400


300


200


100


0


250


200


150


100


50


0


GSM8K (Math)


GSM8K (Math)


GSM8K (Math)


Math500 (Math)


Math500 (Math)


Math500 (Math)


MBPP (Code)


MBPP (Code)


MBPP (Code)


AIME24 (Math)


AIME24 (Math)


AIME24 (Math)


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


Full H2O (Dynamic) H2O (Fixed) R-KV (Dynamic) R-KV (Fixed)


|Col1|0.8|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
||||0|.7||
||||~~0.4~~|0.|~~0.5~~<br>6|
||||0<br>|0<br>0.2<br>.3|.1<br>|
|||||||


|Col1|32k|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
||||4<br>8k<br>16k|~~2k~~<br>k||
||||||1k|
|||||||
|||||||
|||||||


Figure 17: The latency of once attention computing of compressed attention by head-reallocation
compared to full attention. _Left_ : Varying sparsity levels at fixed sequence length of 16K. _Right_ :
Varying sequence lengths at fixed sparsity of 0.5.


ure 15: when requests in a batch terminate at different decoding steps, completed sequences remain
in the batch and effectively waste compute. Modern inference frameworks such as SGLang (Zheng
et al., 2024) and vLLM (Kwon et al., 2023) support continuous batching, where completed requests
are removed and new requests are added to the batch on the fly, thereby reducing wasted computation due to heterogeneous output lengths. We expect that integrating head-reallocation attention into
such frameworks could further improve end-to-end speedups.


21


Math500 (Math) - R-KV
0.9


0.0
0.0 0.2 0.4 0.6 0.8


0.5


0.4


0.3


0.2


0.1


0.9

0.8

0.7

0.6

0.5

0.4

0.3

0.2

0.1


Math500 (Math) - H2O


AIME24 (Math) - R-KV


0.0
0.0 0.2 0.4 0.6 0.8

Math500 (Math) - H2O

1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.0 0.2 0.4 0.6 0.8


0.8

0.7

0.6

0.5

0.4

0.3

0.2

0.1


0.0
0.0 0.2 0.4 0.6 0.8

Math500 (Math) - R-KV
1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
0.0 0.2 0.4 0.6 0.8


0.0
0.0 0.2 0.4 0.6 0.8


0.0
0.0 0.2 0.4 0.6 0.8


Math500 (Math) - H2O


0.5


0.4


0.3


0.2


0.1


AIME24 (Math) - R-KV


Math500 (Math) - R-KV
0.9


0.5


0.4


0.3


0.2


0.1


0.9

0.8

0.7

0.6

0.5

0.4

0.3

0.2

0.1


Math500 (Math) - H2O


AIME24 (Math) - R-KV


0.0
0.0 0.2 0.4 0.6 0.8


AIME24 (Math) - H2O
0.5


0.4


0.3


0.2


0.1


0.0
0.0 0.2 0.4 0.6 0.8

AIME24 (Math) - H2O
0.5


0.4


0.3


0.2


0.1


0.0
0.0 0.2 0.4 0.6 0.8

AIME24 (Math) - H2O
0.5


0.4


0.3


0.2


0.1


0.0
0.0 0.2 0.4 0.6 0.8


0.8

0.7

0.6

0.5

0.4

0.3

0.2

0.1


0.0
0.0 0.2 0.4 0.6 0.8


KV Cache Budget Sparsity


Figure 16: Performance comparison of R-KV and H2O under fixed budget and dynamic budget
settings on Llama-3.1-8B-R1, Qwen-2.5-7B-R1, and Qwen-3-4B-Thinking across Math500 and
AIME24 at sparsity levels of 0.2, 0.4, 0.6, and 0.8. The fixed-budget R-KV performs significantly
worse than the dynamic-budget variant at 0.2, 0.4, and 0.6 sparsity, and only becomes better at 0.8
sparsity, while H2O maintains similar performance across both settings.


Impact of Sequence Length

(Fixed Sparsity: 0.5)


32k


16k


8k


4k


2k


1k


Latency Ratio ( better)


6

5

4

3

2

1


Impact of Sparsity
(Fixed Seq Len: 16k)


Latency Ratio ( better)


0.8

0.7

0.6

0.5

0.4

0.3

0.2

0.1


2.0


1.8


1.6


1.4


1.2


1.0


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


Table 7: End-to-end serving metrics at sparsity 0 _._ 5 using a PyTorch/Transformers implementation.
The table reports batch size, peak GPU memory, latency, speedup (normalized so the full model is
1 _._ 0), and accuracy for the full model and RLKV.


Batch Size Peak GPU (GB) Latency (s) Speedup Accuracy


**#** Full RLKV Full RLKV Full RLKV Full RLKV Full RLKV


1 2 4 19.08 19.40 24374.8 21080.2 1.00 1.16 0.810 0.792
2 4 8 23.57 23.84 16838.1 14569.2 1.00 1.16 0.784 0.792
3 8 16 32.23 32.82 14222.4 11767.5 1.00 1.21 0.776 0.768
4 16 32 49.79 50.88 11752.4 10809.1 1.00 1.09 0.770 0.764


22