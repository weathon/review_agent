# FLATTER TOKENS ARE MORE VALUABLE FOR SPECULATIVE DRAFT MODEL TRAINING


**Jiaming Fan** [1] _[,]_ [2] _[∗]_ **Daming Cao** [3] _[∗]_ **Xiangzhong Luo** [1] _[,]_ [2] _[†]_ **Jiale Fu** [1] _[,]_ [2] **Chonghan Liu** [4] **Xu Yang** [1] _[,]_ [2]

1Key Laboratory of New Generation Artificial Intelligence Technology and Its Interdisciplinary
Applications (Southeast University), Ministry of Education 2Southeast University
3Nanjing University of Information Science and Technology 4Qiyuan Tech
jiaming.fan@seu.edu.cn dmcao@nuist.edu.cn
xiangzhong.luo@seu.edu.cn


ABSTRACT


Speculative Decoding (SD) is a key technique for accelerating Large Language
Model (LLM) inference, but it typically requires training a draft model on a large
dataset. We approach this problem from a data-centric perspective, finding that not
all training samples contribute equally to the SD acceptance rate. Specifically, our
theoretical analysis and empirical validation reveals that tokens inducing flatter
predictive distributions from the target model are more valuable than those yielding sharply peaked distributions. Based on this insight, we propose _flatness_, a
new metric to quantify this property, and develop the Sample-level-flatness-based
Dataset Distillation (SFDD) approach, which filters the training data to retain only
the most valuable samples. Experiments on the EAGLE framework demonstrate
that SFDD can achieve over 2 _×_ training speedup using only 50% of the data, while
keeping the final model’s inference speedup within 4% of the full-dataset baseline. This work introduces an effective, data-centric approach that substantially
improves the training efficiency for Speculative Decoding. Our code is available
[at https://github.com/fjm9933/Flatness.](https://github.com/fjm9933/Flatness)


1 INTRODUCTION


Large language models (LLMs) have demonstrated remarkable success across a myriad of downstream tasks, such as generation, comprehension, and interaction (Achiam et al., 2023; Guo et al.,
2025; Touvron et al., 2023). Despite the success, modern LLMs rely on autoregressive decoding,
where each token must be generated in sequence based on all previous tokens. This inherently sequential process suffers from inferior parallelism, which leads to high latency and low throughput
(Li et al., 2024a). A recent effort to tackle this dilemma is _speculative_ _decoding_ (SD) (Leviathan
et al., 2023; Chen et al., 2023). SD leverages a small _draft_ model to quickly generate multiple tokens, which are then verified in parallel by the larger _target_ model. As a result, the target model can
accept multiple tokens in a single forward pass without degrading the quality of generation, where
higher acceptance rates can equivalently translate into better inference speedups.


The success of SD has subsequently inspired a plethora of SD variants, which can be broadly divided
into _train-free_ and _train-based_ categories. Among them, train-free SD methods employ off-the-shelf
lightweight LLMs as the drafter, which can offer simplicity and cost-effectiveness without additional
training (Leviathan et al., 2023; Chen et al., 2023; Zhang et al., 2024a; Miao et al., 2024; Gong et al.,
2024; Santilli et al., 2023). Nonetheless, these methods suffer from poor alignment between the draft
and target models, which often results in low acceptance rates and frequent rollbacks. In contrast,
train-based SD methods introduce an additional training to align the draft model with the target
model, which can substantially improve acceptance rates compared to their train-free counterparts
(Zhou et al., 2023; Li et al., 2024b; Cai et al., 2024; Elhoushi et al., 2024; Bachmann et al., 2025;
Yi et al., 2024; Monea et al., 2023; Qin et al., 2024).


_∗_ Equal contribution

_†_ Corresponding author


1


Despite the promising progress, current train-based SD methods still face critical limitations. In
practice, these methods leverage vanilla knowledge distillation (KD) (Hinton et al., 2015) as the
default strategy to align the draft model with the target model. However, a subtle yet fundamental
discrepancy exists between the objectives of vanilla KD and SD: while vanilla KD minimizes the
KL-divergence between the student (draft model) and teacher (target model) output distributions, SD
focuses on maximizing the acceptance rate, which is theoretically linked to the _L_ 1-norm between
these two distributions, as proved in (Leviathan et al., 2023). Motivated by this theoretical insight,
recent studies have investigated the direct use of the _L_ 1-norm as an alternative training objective
(Zhou et al., 2023). Nonetheless, empirical findings indicate that this approach is not consistently
effective and can, in some cases, underperform even the standard KL-divergence-based distillation.
While the precise reasons for these counterintuitive results remain unclear, this evidence strongly
suggests that simply substituting the loss function is insufficient, and revisiting the question of which
portions of the data actually provide the most meaningful training signal is warranted, rather than
solely focusing on the choice of loss function.


We therefore revisit KD in the context of SD and introduce a simple theoretical model to reflect
improvements in acceptance rate after a single KD step. We view one update of the draft distribution as a budget-limited move toward the teacher. This abstraction mirrors standard KD practice
while allowing us to ask a concrete question: which target-side token distributions yield the largest
acceptance-rate gains per unit of training? Our toy example analysis and empirical studies indicate
a token-level insight: tokens with flatter target distributions (closer to uniform) deliver the most
per-step reduction in the draft–target discrepancy that governs the acceptance rate, whereas sharply
peaked tokens contribute little and saturate quickly. This reframes the importance of tokens for SD
relative to classical KD: what matters is not only the choice of loss but also where the useful signal
lies in the data. Significantly, this criterion depends only on the target model and can be computed
offline, without warming up a draft model or tracking its changing predictions. Nevertheless, current
training-based SD systems (e.g., the EAGLE series (Li et al., 2024b;c; Li et al.)) essentially train
on all tokens, overlooking this heterogeneity and incurring avoidable overhead. These observations
motivate filtering out low-value tokens to improve efficiency while preserving acceptance.


Guided by this principle, we introduce a practical _flatness_ metric that scores each token by how
close the target model’s distribution is to uniform, instantiated with a simple cosine-based similarity. Empirical evaluations on real LLMs reveal that the _flatness_ metric serves as a reliable
headroom indicator of potential improvement: tokens with higher flatness (more uniform targets)
undergo larger expected updates and yield substantial reductions in acceptance-related discrepancies. In contrast, tokens characterized by low flatness (i.e., sharply peaked distributions) contribute
minimally. This clear distinction consistently emerges under a target-sorted perspective. We then
aggregate token-level scores to the sample level, enabling an effective data-selection approach. We
term this approach _Sample-level-flatness-based Dataset Distillation_ (SFDD), which yields a simple
pipeline: (i) run a single offline pass of the target model to compute sample-level-flatness, (ii) rank
and retain the high-value samples, and (iii) train the draft model on the filtered data. Plugged into
EAGLE-2 (Li et al., 2024c), our selection preserves speedup while substantially reducing training
time, and it outperforms common data selection metrics, such as entropy (Li et al., 2021), top-1 probability (Hendrycks & Gimpel, 2016), the margin between the top two probabilities (Kremer et al.,
2014; Bahri et al., 2022), Energy Score (Liu et al., 2020), and perplexity (PPL) (Chen & Goodman,
1999; Meister & Cotterell, 2021). Finally, we summarize our main contributions as follows:


- **Revisiting** **KD** **for** **SD** **with** **an** **acceptance-centric** **lens.** We analyze a single KD step through
a budget-limited update toward the teacher and show that tokens with flatter target distributions
are disproportionately valuable for improving acceptance, whereas highly peaked tokens offer diminishing returns. Crucially, the resulting importance criterion depends only on the target model,
allowing for offline scoring without a warmed-up student.


- **A simple, empirically strong importance metric.** We propose _flatness_ as a practical proxy
for token and sample importance in SD training, and demonstrate that it outperforms previous
heuristics for identifying high-value training data on sample selection.


- **An** **efficient** **data-selection** **method** **for** **train-based** **SD.** Our SFDD method is effective across
various data retention ratios. For example, at 50% retain ratio, we achieve over 2 _×_ training
speedup using only half the data, while also significantly outperforming other selection metrics
and keeping the final model’s inference speedup within 4% of the full-dataset baseline.


2


2 RELATED WORK


**Speculative** **decoding** . Speculative Decoding (SD) accelerates autoregressive generation via a
”draft-and-verify” paradigm, with approaches broadly categorized as training-free or train-based
methods. (1) Training-Free SD requires no new parameters, instead modifying inference via methods like rejection sampling (Leviathan et al., 2023; Chen et al., 2023), reusing target model parts
(Zhang et al., 2024a), or parallel candidate verification (Miao et al., 2024; Gong et al., 2024; Santilli
et al., 2023). While easily deployable, these reliance on heuristics often results in limited acceptance
rates. (2) Train-Based SD fine-tunes a draft model for better alignment, either through distillation
(Zhou et al., 2023; Goel et al., 2024) or by augmenting the target model with trainable components,
including lightweight prediction heads (Li et al., 2024b;c; Li et al.; Cai et al., 2024), trainable early
exits (Elhoushi et al., 2024), or auxiliary modules for multi-token prediction or lenient verification
(Bachmann et al., 2025; Yi et al., 2024; Monea et al., 2023; Qin et al., 2024). These methods provide significantly higher and more stable speedups. In addition, some recent works (Zhou et al.,
2023; Goel et al., 2024) leverage the theoretical objective (e.g., total variation distance) to improve
alignment; however, they focus on loss function optimization to improve alignment. In contrast, our
work targets efficient draft model training from the perspective of dataset distillation, remaining orthogonal to these train-based methods that introduce non-trivial training and deployment overhead.


**Data importance measurement methods** . Following trends in selective learning and aligned controllable generation (Zhu et al., 2024a;b; 2025; Zhao et al., 2025a;b), we focus on data importance
from two perspectives: (1) Distributional Uncertainty: This approach gauges importance via the
model’s uncertainty, based on the intuition that the most informative samples are those the model is
unsure about. This is often quantified by metrics such as the Shannon entropy of the output distribution (Li et al., 2021; Wang et al., 2025), the Energy Score derived from logits (Liu et al., 2020), or
the sample difficulty measured by Perplexity (PPL) (Chen & Goodman, 1999; Meister & Cotterell,
2021). (2) Category Probability: This second approach derives importance from salient category
probabilities. It includes using the Top-1 Probability or logit as a direct measure of model confidence (Hendrycks & Gimpel, 2016; Zhou et al., 2025), the margin between the top two probabilities
( _p_ (1) _−_ _p_ (2)) to gauge ambiguity (Kremer et al., 2014; Bahri et al., 2022), or the ground-truth token’s probability to up-weight difficult examples (Lin et al., 2017; Kim et al., 2023). However, these
heuristics are generally designed for standard training objectives like improving model accuracy or
distribution fidelity. In contrast, our work is the first to systematically investigate data importance
from the unique perspective of SD, where the central focus is the token acceptance rate.


3 PRELIMINARIES


3.1 SPECULATIVE DECODING


As shown in (Leviathan et al., 2023; Chen et al., 2023), speculative decoding (SD) utilizes a small,
fast draft model and a large, powerful target model. The process begins with the draft model autoregressively generating a sequence of _γ_ candidate tokens, which are then verified in parallel by
the target model. Formally, given a context of previously generated tokens _h_, we denote the candidate probability distribution from the draft model as _q_ ( _·|h_ ) and the reference distribution from the
target model as _p_ ( _·|h_ ). Each candidate token _y_ is validated via rejection sampling and accepted


with a probability of _β_ ( _y_ ) = min�1 _,_ _[p]_ _q_ ( [(] _y_ _[y]_ _|_ _[|]_ _h_ _[h]_ ) [)] �. If a candidate is rejected, the generation process is

rolled back to the last accepted token. A new token is then drawn from the residual distribution
_r_ ( _·_ _|_ _h_ ) _∝_ - _p_ ( _·_ _|_ _h_ ) _−_ _q_ ( _·_ _|_ _h_ )�+ [(where] [[] _[x]_ []][+] [=] [max] _[{][x,]_ [ 0] _[}]_ [)] [to] [ensure] [that] [the] [final] [output] [is]

equivalent to sampling directly from the target distribution _p_ ( _·|h_ ).


**Acceptance** **rate.** In SD, the acceptance rate _α_ ( _h_ ) is a key performance metric, defined as the
ratio of candidate tokens proposed by the draft model that are verified and accepted by the target
model. In practice, higher acceptance rates can equivalently translate into better inference speedups.
Prior work (Leviathan et al., 2023) has shown that the acceptance rate is linearly decreasing in the
_L_ 1-norm between the target model’s distribution _p_ and the draft model’s distribution _q_ :


    _α_ ( _h_ ) = E _y∼q_ ( _·|h_ )[ _β_ ( _y_ )] =


2 [1] �� _p_ ( _· | h_ ) _−_ _q_ ( _· | h_ )��1 _[.]_ (1)


_y_ [min] _[{][p]_ [(] _[y]_ _[|][ h]_ [)] _[, q]_ [(] _[y]_ _[|][ h]_ [)] _[}]_ [ = 1] _[ −]_ 2 [1]


3


1.0


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
||||||||)|
|||||||cos(p, U|)|


0 10 20 30 40 50
Standard Deviation ( )


0.8

0.6

0.4

0.2

0.0


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||p = 8<br> ~~= 20~~||
|||||||<br>p = 50||


0 10 20 30 40 50
Variance of p ( p2 [)]


1.0

0.8

0.6

0.4

0.2

0.0


(a) ∆ _L_ 1 vs. Variance ( _θ_ = 2) (b) Cosine Similarity to Uniform vs. _σ_

Figure 1: **Simulation** **of** **importance** **metrics.** We use Gaussian distributions in our simulation
because they are analytically tractable and effectively capture key distributional properties like variance. For visualization clarity, all plotted quantities are min–max normalized to [0 _,_ 1]. (a) Across
small, medium, and large separations between the means of _p_ and _q_, ∆ _L_ 1 consistently increases with
the target variance ( _σp_ [2][).] [(b) Cosine similarity to uniform increases monotonically with the standard]
deviation ( _σ_ ), validating its use as a practical proxy that directly tracks changes in variance.


Thus, a per-token improvement in acceptance equals a per-token reduction in the _L_ 1-norm distance.
In light of this, maximizing the token-level acceptance rate is strictly equivalent to minimizing the
_L_ 1-norm distance between the output distributions of the target model and draft model.


3.2 THEORETICAL ANALYSIS OF KD IN SD


In this section, we investigate knowledge distillation (KD) in the context of SD. We start from the
optimization objective of SD and then present both empirical analysis and experimental validation.


**Training objective.** An intuitive approach to optimizing the acceptance rate is to directly use the _L_ 1norm as a training loss. However, this can yield suboptimal results on certain tasks compared to the
conventional KL-divergence objective (Zhou et al., 2023). This further leads us to investigate how
the characteristics of the target model’s token distribution _p_ and the draft model’s token distribution
_q_ influence the _L_ 1-norm. Analyzing the draft model’s distribution _q_ is challenging, as it is a moving
target during training. The target distribution _p_, however, remains fixed within a given context.
This stability allows us to pursue an empirical goal: identifying the properties of an optimal target
distribution _p_ that are beneficial regardless of the specific state of _q_ .


At the same time, a small _L_ 1-norm does not necessarily indicate a valuable training opportunity. For
instance, if _q_ is already very close to _p_, the _L_ 1-norm will be minimal, but training on this token will
also yield a negligible contribution to the model’s improvement. This insight suggests that the true
measure of a token’s value is not the static _L_ 1-norm itself, but the potential training contribution.
We therefore quantify this contribution as the reduction in the _L_ 1-norm achieved in a single training
step, denoted as ∆ _L_ 1. Formally, given an initial draft distribution _q_ and an updated distribution _r_ _[∗]_
after one step, this contribution is defined as:


∆ _L_ 1 = _∥p −_ _q∥_ 1 _−∥p −_ _r_ _[∗]_ _∥_ 1 _._ (2)


Our primary objective is to explore characteristics of the target token distribution _p_ that maximize
∆ _L_ 1 for a given draft token distribution _q_ . To study this, we introduce _r_ _[∗]_ as a theoretical proxy
for the draft distribution after a _single,_ _idealized_ update. This theoretical toy model serves as a
simplified analytical model to illustrate how a budget-constrained training step might ideally shift
the draft distribution toward the target. Notably, _r_ _[∗]_ is not employed as a practical training target;
instead, it functions solely as an analytical tool to understand incremental improvements.


Modeling a single training step inevitably involves simplifications, and various formalisms could
potentially serve this purpose. We select an approach closely aligned with standard KD practices,
yet analytically tractable: our _objective_ adheres to KD by minimizing _D_ KL( _p∥r_ ), while we simultaneously impose a small-step _budget_ constraint to reflect that an update must remain close to the
original draft distribution _q_ . Crucially, our insights do not depend strongly on the specific choice of
budget measurement; alternative measures such as _Lp_ norm, Jensen–Shannon divergence, or other
suitable metrics would yield similar conclusions. We adopt KL divergence _D_ KL( _r∥q_ ) here, as this
choice facilitates a concise and explicit analytical form for _r_ _[∗]_ under the subsequent Gaussian setting,


4


thus ensuring that our downstream analysis remains transparent and interpretable:


_r_ _[∗]_ = arg min _D_ KL( _p∥r_ ) s.t. _D_ KL( _r∥q_ ) _≤_ _θ,_ (3)
_r_


where _θ_ _≥_ 0 plays the role of a step-size budget (capturing, e.g., learning-rate and optimizer effects)
that limits how far _r_ can deviate from _q_ in one update.


**Solution for parametric Gaussian distributions** . Analyzing the optimal distribution _r_ _[∗]_ in its general non-parametric form is analytically challenging. To obtain analytical insights, we first restrict
the above optimization problem to the parametric family of Gaussian distributions. Specifically,
we use _p_ = _N_ ( _µp, σp_ [2][)] [to] [denote] [the] [target] [token] [distribution] [and] _[q]_ [=] _[N]_ [(] _[µ][q][, σ]_ _q_ [2][)] [to] [denote] [the]
draft token distribution. Using the Karush-Kuhn-Tucker (KKT) conditions (see Appendix A for the
detailed derivation), we solve for the optimal distribution _r_ _[∗]_ = _N_ ( _µ_ _[∗]_ _r_ _[, σ]_ _r_ [2] _[∗]_ [)][, whose parameters are:]

_µ_ _[∗]_ _r_ [= (1] _[ −]_ _[τ][ ∗]_ [)] _[µ][p]_ [+] _[ τ][ ∗][µ][q][,]_ (4)


_σr_ [2] _[∗]_ = (1 _−_ _τ_ _[∗]_ ) _σp_ [2] [+] _[ τ][ ∗][σ]_ _q_ [2] [+] _[ τ][ ∗]_ [2][(1] _[ −]_ _[τ][ ∗]_ [)(] _[µ][p]_ _[−]_ _[µ][q]_ [)][2] _[.]_ (5)

The optimal distribution _r_ _[∗]_ is found by minimizing _D_ KL( _p∥r_ ) under the constraint that _D_ KL( _r∥q_ ) _≤_
_θ_ . It lies on a path between _p_ and _q_, and its position on this path is determined by a single parameter
_τ_ _[∗]_ _∈_ [0 _,_ 1]. This parameter quantifies the extent of the update in a single training step: _τ_ _[∗]_ = 0
corresponds to a full update where _r_ _[∗]_ becomes _p_, while _τ_ _[∗]_ = 1 means no update has occurred ( _r_ _[∗]_
remains _q_ ). The specific value of _τ_ _[∗]_ is uniquely determined by the training budget _θ_ .


**Simulation results.** Although the above solution provides an analytical form for the optimal parameters ( _µ_ _[∗]_ _r_ _[, σ]_ _r_ [2] _[∗]_ [)][,] [the] [path] [parameter] _[τ][ ∗]_ [itself] [lacks] [a] [closed-form] [solution] [and] [must] [be] [determined]
numerically. Therefore, to investigate the properties of this solution, we establish a numerical simulation, where the draft distribution _q_ is fixed as the standard normal distribution ( _µq_ = 0 _, σq_ [2] [=] [1][)]
and the training budget is a fixed _θ_ . For various target token distributions _p_ (defined by sweeping
their mean _µp_ and variance _σp_ [2][),] [we] [first] [solve] [for] [the] [optimal] [path] [parameter] _[τ][ ∗]_ [,] [based] [on] [which]
we can derive the parameters of _r_ _[∗]_ . With the updated distribution _r_ _[∗]_ now fully defined, we can
substitute it back into the expression for ∆ _L_ 1 to analyze which properties of _p_ lead to the largest
training benefit. Our simulation results are illustrated in Figure 1a, which shows how the _L_ 1-norm
distance reduction ∆ _L_ 1 varies with respect to the target distribution, _σp_ [2][.] [Within our tested range of]
variances, we observe a clear trend: for a given mean _µp_, the reduction ∆ _L_ 1 tends to increase as _σp_ [2]
grows. This suggests that tokens whose target distributions have higher variance are likely to yield
the greatest reduction in the _L_ 1-norm during training. The empirical results from our simulation
reveal a clear relationship: tokens with larger variance _σp_ [2] [are more valuable to the training process.]


This finding can also be explained by our formula. As shown in Equation 5, a larger target variance
_σp_ [2] [directly] [increases] [the] [variance] [of] [the] [updated] [distribution] _[r][∗]_ [,] [ensuring] [a] [flatter] [target] [yields] [a]
flatter updated distribution. This shape alignment is crucial for maximizing the training contribution,
∆ _L_ 1 (Equation 2). The _L_ 1-norm is highly sensitive to the misalignment of sharp probability peaks;
since flat distributions lack such peaks, pointwise differences between them remain small, resulting
in a smaller distance _∥p −_ _r_ _[∗]_ _∥_ 1. Assuming a constant initial distance _∥p −_ _q∥_ 1 (i.e., a fixed starting
acceptance rate), minimizing _∥p −_ _r_ _[∗]_ _∥_ 1 maximizes the training contribution ∆ _L_ 1 (Equation 2).


**Key insights.** This motivates our key theoretical insight: not all tokens are equally important; those
with flatter target distributions are more valuable for training. And we can use the variance of the
Gaussian distribution as a measure of token importance in SD.


**Discrete** **perspective.** However, in the discrete, token-level probability distributions produced by
practical LLMs, we cannot directly compute this continuous variance. To bridge this gap between
continuous theory and discrete distributions, we require a useful metric that can serve as a proxy
for variance. We propose using the **cosine similarity with the uniform distribution.** This choice
is theoretically grounded; as detailed in Appendix B, it can be shown that this metric is positively
correlated with the variance of the corresponding Gaussian distribution in the continuous limit. Our
simulations, presented in Figure 1b, further validate this crucial relationship, demonstrating that the
cosine similarity indeed increases monotonically with the Gaussian standard deviation. This positive
correlation is the crucial link between our continuous theory and discrete application. It validates
that cosine similarity to a uniform distribution can serve as a practical and computable proxy for
quantifying a token’s training importance in the discrete setting.


5


4 THE PROPOSED APPROACH


4.1 EMPIRICAL VALIDATION OF THE THEORETICAL ANALYSIS


1.0


0.8


0.6


0.4


0.2


low flatness
high flatness


6


4


2


0


2


|Col1|Col2|Col3|
|---|---|---|
||||
||||
||||
||||


Token index (sorted by flatness of p)


(a) flatness( _p_ )


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|||||
|||||
|Min|imal c|hange|<br>Greater|


Token index (sorted by flatness of p)


(b) Change in flatness( _qm_ )


17.5


15.0


12.5


10.0


7.5


5.0


2.5


0.10


0.05


0.00


0.05


0.10


low flatness
~~high flatness~~


|Col1|Col2|Col3|
|---|---|---|
||||
||||
||||
|||<br>|
|~~M~~|~~nimal ch~~|~~ ange~~<br>~~Greater c~~|


Token index (sorted by flatness of p)


0.0

×10 4


Number of selected examples


(c) ∆ _L_ 1 (d) Entropy-vs-flatness filtering gap

Figure 2: **Target-sorted flatness view.** Tokens sorted by the target’s statistic (low _→_ high flatness).
(a) flatness values; (b) one-epoch change in flatness; (c) the one-epoch reduction in the _L_ 1 discrepancy, ∆ _L_ 1. Curve coloring distinguishes token groups by target flatness: the **blue** segment
represents tokens with low flatness, while the **red** segment represents those with high flatness. Panels (b,c) additionally annotate **Minimal** **change** (left; indicating that the vast majority of points in
this segment exhibit small changes) and **Greater change** (right; indicating that the vast majority of
points in this segment exhibit larger changes). (d) Entropy-vs-flatness filtering gap: for each metric,
we rank tokens by that metric, take the bottom 35% tokens, and compute their one-epoch average
_|_ ∆ _L_ 1 _|_ ; bars plot the difference between the entropy-based and flatness-based bottom-35% averages
under different numbers of selected examples. Flatness is as defined in Equation 6.


**Flatness definition.** In the previous section, our one-step KD analysis reveals that tokens with flatter
target distributions should offer greater acceptance-linked headroom; however, those results rest on
idealized assumptions. Therefore, an empirical validation is essential to bridge the gap between our
theory and real-world application. So now we test this insight on real LLMs. We define _flatness_
of a token _t_ as the cosine similarity between its distribution and the uniform distribution:


_pt · U_
_flatness_ ( _t_ ) := cos( _pt, U_ ) = _∥pt∥_ 2 _∥U_ _∥_ 2 _,_ (6)


where _pt_ is the token’s distribution, _U_ is the uniform distribution over the vocabulary of size _V_, and
_∥· ∥_ 2 denotes the Euclidean ( _L_ 2) norm.


**Empirical validation via training dynamics.** From the definition, a higher flatness denotes a more
uniform distribution, and a lower flatness indicates a more concentrated distribution. When we refer
to the target flatness, we use the target distribution _p_ as _pt_ in the equation. To validate whether
target flatness can effectively serve as a metric for a token’s training potential, we first need a metric
to quantify its actual contribution to the learning process. Throughout our analysis, we define this
contribution as the epoch-to-epoch reduction in the _L_ 1 discrepancy, i.e.,


∆ _L_ 1 = _∥p −_ _qm∥_ 1 _−∥p −_ _qm_ +1 _∥_ 1 _,_ (7)


6


where _m_ and _m_ + 1 denote consecutive training epochs. We follow the EAGLE-2 framework (Li
et al., 2024c) with LLaMA3-8B-Instruct (Grattafiori et al., 2024) as target model, trained on filtered
ShareGPT dataset (sha, 2023). And we randomly select 10 samples for detailed inspection.


To relate the evaluated metric to training progression, we first sort tokens by the target flatness value
in ascending order (low to high). The resulting curves are then smoothed using a 10-point moving
average to obtain more stable values, thereby enhancing readability. As shown in Figure 2, we plot
three key metrics. The x-axis for all subplots represents tokens sorted in ascending order of the target
flatness, _flatness_ ( _p_ ). To analyze the draft model’s behavior, we apply this flatness metric to its
distribution _q_, which we called draft flatness. The plots then show: (a) the target flatness value itself;
(b) the one-epoch change in the draft model’s flatness, _flatness_ ( _qm_ ); and (c) the corresponding
one-epoch reduction in the _L_ 1 discrepancy, ∆ _L_ 1.


Collectively, our empirical results substantiate the following observations, sorting tokens according
to the target flatness from low to high, as illustrated in the target-sorted view in Figure 2:


**(i)** In the _low target flatness_ region (blue segment in Panel (a)), the one-epoch change in draft flatness
is small (Panel (b), left; annotated “Minimal change”), and the acceptance-linked discrepancy likewise shows slight movement (Panel (c), left). Thus, tokens with low target flatness exhibit _limited_
one-epoch movement in both draft statistics and ∆ _L_ 1.


**(ii)** In the _high_ _target_ _flatness_ region (red segment in Panel (a)), draft flatness varies more over
one epoch (Panel (b), right; annotated “Greater change”), and the magnitude of the corresponding
change in ∆ _L_ 1 is also larger (Panel (c), right). Hence, tokens with high target flatness are precisely
where we observe pronounced acceptance-linked movement during training.


These findings affirm that flatness effectively signals available headroom: tokens with low target flatness contribute minimally to training improvements, whereas tokens with high target flatness exhibit
learnable dynamics and significant changes in ∆ _L_ 1. We thus adopt target flatness ( _flatness_ ( _p_ ))
as the principal ranking criterion for data selection.


At first glance, target flatness-based selection might seem counterintuitive or risky: interpreting low
target flatness as indicative of high target certainty (a strong label signal) could imply inadvertently
excluding valuable tokens. In practice, however, such low target flatness tokens either (i) already
closely align or will rapidly align with the target distribution, rendering subsequent updates negligible in terms of reducing ∆ _L_ 1 in later training, or (ii) remain confidently misaligned, thus providing
minimal per-step gradient information and potentially contributing negatively when averaged across
multiple tokens. Consequently, prioritizing tokens with higher target flatness focuses computational
resources precisely where meaningful, acceptance-linked improvements can be realized.


**Comparison** **with** **other** **metrics.** We further compare target flatness with other commonly used
distribution-dispersion metrics, such as target entropy. The results show a similar trend, details are
provided in Appendix F.2.


More importantly, we find that flatness provides more effective filtering than entropy. We randomly
sample _N_ _∈{_ 10 _,_ 20 _,_ 30 _,_ 40 _,_ 50 _}_ training examples. For each metric (entropy or flatness), we rank
all tokens by that metric and take the bottom 35% as low-score tokens. On the low-entropy and lowflatness tokens of each metric, we compute the average _|_ ∆ _L_ 1 _|_ between consecutive training epochs,
denoted as _|_ ∆ _L_ 1 _|_ low-entropy and _|_ ∆ _L_ 1 _|_ low-flatness, for their remaining impact on SD in late training. We

then report the gap between flatness and entropy as _g_ = _|_ ∆ _L_ 1 _|_ low-entropy _−|_ ∆ _L_ 1 _|_ low-flatness.


As shown in Figure 2d, we observe _g_ _>_ 0 consistently holds. Furthermore, the gap increases as
_N_ grows. This indicates that, under the same retain ratio, flatness-based filtering removes more
already-saturated tokens (with smaller _|_ ∆ _L_ 1 _|_ ). It gives a quantitative explanation of why flatness is
a better metric than entropy in our data selection experiments: flatness is more effective at filtering
out low-quality tokens that offer minimal training value, leading to higher training efficiency.


4.2 FROM TOKEN-LEVEL INSIGHT TO SAMPLE-LEVEL DATA SELECTION


The successful validation of this token-level insight sheds light on a significant inefficiency in current training-based SD methods. Prominent approaches, such as the EAGLE series, rely on full-data
training over large datasets. They treat all data samples as equally important, thereby expending con

7


**Original Samples** **SFDD Approach (Sample-level Distillation)** **Distilled Samples**


Figure 3: **The** **SFDD** **workflow:** This approach calculates _flatness_ sample by averaging token
flatness within each sample, and then uses quantile-derived threshold to select the top- _k_ % and filter
the dataset for training. The figure illustrates this with a concrete example using 70% retain ratio.


siderable computational resources on samples predominantly composed of low-value, concentrated
tokens, while these tokens contribute negligibly to training. This motivates us to advocate for a more
efficient paradigm: extending our validated token-level insight to a practical, sample-level data selection strategy. The goal is to filter out entire samples that are unlikely to contribute significantly
to the training process, thereby accelerating training without heavily sacrificing performance.


To this end, we introduce the sample-level-flatness to quantify a sample’s overall training value. The
sample-level-flatness for a sample _S_ is defined as the average of the flatness of its constituent tokens:


_flatness_ sample( _S_ ) = _|S_ [1] _|_


(8)
_t∈S_ _[flatness]_ [(] _[t]_ [)] _[,]_


where the token-level _flatness_ ( _t_ ) is calculated using the target distribution _p_ . A higher
_flatness_ sample signifies that the sample, as a whole, offers greater potential training value.


4.3 SAMPLE-LEVEL-FLATNESS-BASED DATASET DISTILLATION


Building on our validated sample-level flatness metric, we introduce a simple yet effective approach
for dataset distillation. This approach curates a smaller, more efficient dataset for SD training by
retaining only samples with high flatness. The overall workflow, which we term _Sample-level-_
_flatness-based Dataset Distillation_ (SFDD), is illustrated in Figure 3. Given a retain ratio of _k_ %, the
procedure is to first compute sample-level flatness for each sample by averaging the flatness of its
constituent tokens. A threshold _τ_ is then set as the ceiling of the (1 _−_ _k_ )% quantile of these scores.
The distilled dataset is formed by retaining all samples with _flatness_ sample _≥_ _τ_ .


5 EXPERIMENTS


In the previous sections, our theoretical and empirical findings have shown that a token’s importance
correlates with the _flatness_ of its target distribution. In this section, we extensively evaluate our
approach, Sample-level-flatness-based Dataset Distillation (SFDD), against various baselines across
different datasets. We aim to: (1) demonstrate the superiority of SFDD by benchmarking it against
various common importance metrics at a fixed data retain ratio; (2) compare SFDD against a random
filtering baseline across different data retain ratios to demonstrate the effectiveness of our method;
and (3) quantify the training efficiency gains provided by our data selection approach.


5.1 EXPERIMENTAL SETUP


**Models** **and** **baselines.** Our experiments use the EAGLE-2 training pipeline (Li et al., 2024c)
with LLaMA3-8B-Instruct (Grattafiori et al., 2024) as the target model. Our approach is compared
against two main baselines: training on the full dataset (“No Filter”) and a naive “Random Filtering”
strategy. In our main results, we also benchmark against several other common token-importance
metrics, including entropy (Li et al., 2021), Top-1 Probability (Hendrycks & Gimpel, 2016), the
margin between the top two probabilities ( _p_ (1) _−_ _p_ (2)) (Kremer et al., 2014; Bahri et al., 2022),
Energy Score (Liu et al., 2020), and perplexity (PPL) (Chen & Goodman, 1999; Meister & Cotterell,
2021). For these metrics, we select samples with high entropy, low top-1 probability, small margin,


8


larger (less-confident) Energy Score, or higher PPL, as these criteria help identify samples that are
valuable for training rather than those the model has already converged on.


**Dataset** **and** **tasks.** We use the ShareGPT dataset (sha, 2023) for training. Evaluation is performed on five diverse downstream tasks: GSM8K (Cobbe et al., 2021), Alpaca (Taori et al.,
2023), MT-Bench (MTB) (Zheng et al., 2023), CNN/DM (See et al., 2017), and Natural Questions
(NQ) (Kwiatkowski et al., 2019). All experiments use NVIDIA H800 GPUs, a decoding temperature
of 1.0, and a draft generating step of _γ_ = 5; see Appendix C for temperature 0 results.


**Evaluation metrics.** We use three primary metrics: (1) **Speedup** : The wall-clock time of standard
autoregressive decoding divided by that of speculative decoding. Higher is better. (2) **Average**
**acceptance** **length** **(** _l_ **)** : The average number of draft tokens accepted per verification cycle. The
average acceptance rate is exactly _l/γ_, but since the rate is often very small, we instead report _l_,
which makes variations more visually discernible. Higher is better. (3) **Training** **time** : The total
wall-clock time (in seconds) for training, used to measure efficiency. All reported times include
data-selection overhead. More details can be found in Appendix D.


5.2 MAIN RESULTS


We fix the data retain ratio at 50% for our main comparison. This ratio is chosen because our preliminary experiments show that the random filtering baseline performs near its peak at this level, ensuring a fair comparison against a strong baseline. We compare our method against several common
metrics that, similar to our approach, measure data importance, reporting both inference speedup
and average generation length. As shown in Table 1, our SFDD method consistently achieves higher
speedup and average generation length than all other metrics across every downstream task. Notably, SFDD achieves an average speedup of 2.41 _×_, which is significantly higher than the next best
method (Top-1 Probability at 2.23 _×_ ). Furthermore, with only 50% of the data, our method exhibits
the smallest performance gap compared to the full-dataset baseline, achieving an average speedup
that is **within 4% of the “No Filter” speedup (2.41** _×_ **vs.** **2.49** _×_ **).**

|Table 1: Comprehensive comparison of various metrics for data importance at a 50% retain ratio.|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|
|No Filter|2.71_×_<br>3.28|2.71_×_<br>2.89|2.53_×_<br>2.77|2.30_×_<br>2.58|2.19_×_<br>2.37|2.49_×_<br>2.78|
|Random<br>Entropy<br>Top-1 Probability<br>Margin<br>Energy Score<br>PPL<br>**SFDD (Ours)**|2.43_×_<br>2.85<br>2.43_×_<br>2.85<br>2.49_×_<br>2.84<br>2.45_×_<br>2.85<br>2.49_×_<br>2.87<br>2.36_×_<br>2.79<br>**2.69**_×_<br>**2.95**|2.37_×_<br>2.59<br>2.43_×_<br>2.64<br>2.44_×_<br>2.66<br>2.35_×_<br>2.48<br>2.44_×_<br>2.64<br>2.45_×_<br>2.65<br>**2.66**_×_<br>**2.71**|2.26_×_<br>2.48<br>2.20_×_<br>2.51<br>2.26_×_<br>2.53<br>2.19_×_<br>2.42<br>2.19_×_<br>2.50<br>2.21_×_<br>2.50<br>**2.44**_×_<br>**2.60**|1.99_×_<br>2.31<br>1.95_×_<br>2.32<br>1.99_×_<br>2.32<br>1.92_×_<br>2.27<br>1.99_×_<br>2.33<br>2.01_×_<br>2.33<br>**2.14**_×_<br>**2.38**|1.93_×_<br>2.06<br>1.98_×_<br>2.12<br>1.98_×_<br>2.12<br>1.85_×_<br>1.99<br>1.91_×_<br>2.12<br>1.95_×_<br>2.13<br>**2.14**_×_<br>**2.17**|2.20_×_<br>2.46<br>2.20_×_<br>2.49<br>2.23_×_<br>2.49<br>2.15_×_<br>2.40<br>2.21_×_<br>2.49<br>2.20_×_<br>2.48<br>**2.41**_×_<br>**2.56**|


5.3 ABLATION STUDY


We conduct an ablation study, with results in Table 2, to ablate two key factors. The setup allows
us to simultaneously ablate the contribution of our selection metric (by contrasting SFDD with
Random Filtering and Top-1 Probability, which is the second-best metric in Table 1), ablate the
impact of the retain ratio and control for the specificity of the data chosen by random filtering (by
observing the trend across different retain ratios), thereby confirming that our method’s advantage
is robust and not coincidental. The results in Table 2 demonstrate two key points. First, SFDD
surpasses both the random baseline and Top-1 Probability by a large margin across all retain ratios,
confirming the effectiveness of our flatness-based scoring metric. Second, a significant speedup gap
between SFDD and the baselines persists even at low retain ratios, highlighting the effectiveness of
our method. Impressively, with 70% of the data, our method’s speedup is nearly identical to the
“No Filter” baseline, and on certain datasets like Alpaca, it even surpasses the full-dataset speedup
(2.77 _×_ vs. 2.71 _×_ ), suggesting that filtering can sometimes remove noisy or redundant data.


To investigate the robustness of our method under highly resource-constrained scenarios, we extend
our evaluation to include extreme retain ratios of 5%, 10%, and 20%. We compare SFDD against
Random. As shown in Table 3, we observe that while both methods experience a performance drop


9


Table 2: Ablation study on different retain ratios, comparing SFDD against different baselines.

|GSM8K Alpaca MTB CNN/DM NQ Average<br>Retain Ratio Method Speedup l Speedup l Speedup l Speedup l Speedup l Speedup l|Col2|Col3|Col4|Col5|NQ|Col7|
|---|---|---|---|---|---|---|
|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Retain Ratio<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Retain Ratio<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Retain Ratio<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Retain Ratio<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Retain Ratio<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|Speedup<br>_l_|Speedup<br>_l_|
|100%<br>No Filter|2.71_×_<br>3.28|2.71_×_<br>2.89|2.53_×_<br>2.77|2.30_×_<br>2.58|2.19_×_<br>2.37|2.49_×_<br>2.78|
|70%<br>Random<br>Top-1 Probability<br>**SFDD (Ours)**|2.43_×_<br>2.84<br>2.62_×_<br>2.89<br>**2.71**_×_<br>**2.95**|2.41_×_<br>2.55<br>2.61_×_<br>2.70<br>**2.77**_×_<br>**2.77**|2.24_×_<br>2.46<br>2.34_×_<br>2.50<br>**2.41**_×_<br>**2.58**|1.98_×_<br>2.30<br>2.07_×_<br>2.37<br>**2.19**_×_<br>**2.40**|1.91_×_<br>2.06<br>2.09_×_<br>2.14<br>**2.14**_×_<br>**2.19**|2.19_×_<br>2.44<br>2.35_×_<br>2.52<br>**2.44**_×_<br>**2.58**|
|60%<br>Random<br>Top-1 Probability<br>**SFDD (Ours)**|2.42_×_<br>2.89<br>2.54_×_<br>2.92<br>**2.55**_×_<br>**2.95**|2.38_×_<br>2.59<br>2.55_×_<br>2.67<br>**2.71**_×_<br>**2.72**|2.22_×_<br>2.49<br>2.35_×_<br>2.51<br>**2.40**_×_<br>**2.57**|2.02_×_<br>2.32<br>2.07_×_<br>2.37<br>**2.15**_×_<br>**2.40**|1.95_×_<br>2.06<br>2.09_×_<br>2.14<br>**2.13**_×_<br>**2.15**|2.20_×_<br>2.47<br>2.32_×_<br>2.52<br>**2.39**_×_<br>**2.56**|
|50%<br>Random<br>Top-1 Probability<br>**SFDD (Ours)**|2.43_×_<br>2.85<br>2.49_×_<br>2.84<br>**2.69**_×_<br>**2.95**|2.37_×_<br>2.59<br>2.44_×_<br>2.66<br>**2.66**_×_<br>**2.71**|2.26_×_<br>2.48<br>2.26_×_<br>2.53<br>**2.44**_×_<br>**2.60**|1.99_×_<br>2.31<br>1.99_×_<br>2.32<br>**2.14**_×_<br>**2.38**|1.93_×_<br>2.06<br>1.98_×_<br>2.12<br>**2.14**_×_<br>**2.17**|2.20_×_<br>2.46<br>2.23_×_<br>2.49<br>**2.41**_×_<br>**2.56**|
|40%<br>Random<br>Top-1 Probability<br>**SFDD (Ours)**|2.47_×_<br>2.85<br>2.41_×_<br>2.84<br>**2.66**_×_<br>**2.94**|2.39_×_<br>2.57<br>2.44_×_<br>2.63<br>**2.63**_×_<br>**2.73**|2.24_×_<br>2.46<br>2.26_×_<br>2.50<br>**2.40**_×_<br>**2.56**|1.99_×_<br>2.31<br>1.97_×_<br>2.30<br>**2.13**_×_<br>**2.36**|1.87_×_<br>2.06<br>1.93_×_<br>2.08<br>**2.12**_×_<br>**2.14**|2.19_×_<br>2.45<br>2.20_×_<br>2.47<br>**2.39**_×_<br>**2.55**|
|30%<br>Random<br>Top-1 Probability<br>**SFDD (Ours)**|2.31_×_<br>2.79<br>2.40_×_<br>2.83<br>**2.51**_×_<br>**2.88**|2.33_×_<br>2.55<br>2.40_×_<br>2.62<br>**2.60**_×_<br>**2.62**|2.19_×_<br>2.41<br>2.23_×_<br>2.43<br>**2.37**_×_<br>**2.50**|1.99_×_<br>2.23<br>1.95_×_<br>2.27<br>**2.17**_×_<br>**2.33**|1.88_×_<br>2.01<br>1.92_×_<br>2.07<br>**2.01**_×_<br>**2.10**|2.14_×_<br>2.40<br>2.18_×_<br>2.44<br>**2.33**_×_<br>**2.49**|


at these very low retain ratios, SFDD consistently outperforms Random filtering across all datasets
in terms of both inference speedup and average acceptance length ( _ℓ_ ). This indicates that flatness
remains a robust and effective indicator of token importance under extreme data reduction regimes.


Table 3: Ablation study at extreme retain ratios, comparing SFDD against Random fltering.

|Retain ratio GSM8K Alpaca MTB CNN/DM NQ Average<br>Retain Method Speedup ℓ Speedup ℓ Speedup ℓ Speedup ℓ Speedup ℓ Speedup ℓ|GSM8K|Alpaca|MTB|CNN/DM|NQ|Col7|
|---|---|---|---|---|---|---|
|Retain ratio<br>GSM8K<br>Alpaca<br>MTB<br>CNN/DM<br>NQ<br>Average<br>Retain<br>Method<br>Speedup<br>_ℓ_<br>Speedup<br>_ℓ_<br>Speedup<br>_ℓ_<br>Speedup<br>_ℓ_<br>Speedup<br>_ℓ_<br>Speedup<br>_ℓ_|Speedup<br>_ℓ_|Speedup<br>_ℓ_|Speedup<br>_ℓ_|Speedup<br>_ℓ_|Speedup<br>_ℓ_|Speedup<br>_ℓ_|
|5%<br>Random<br>**SFDD (Ours)**|1.75_×_<br>1.96<br>**2.03**_×_<br>**2.05**|2.00_×_<br>1.92<br>**2.09**_×_<br>**1.99**|1.60_×_<br>1.73<br>**1.81**_×_<br>**1.81**|1.48_×_<br>1.45<br>**1.54**_×_<br>**1.55**|1.57_×_<br>1.49<br>**1.66**_×_<br>**1.54**|1.68_×_<br>1.71<br>**1.82**_×_<br>**1.79**|
|10%<br>Random<br>**SFDD (Ours)**|2.25_×_<br>2.49<br>**2.32**_×_<br>**2.59**|2.21_×_<br>2.24<br>**2.25**_×_<br>**2.30**|2.04_×_<br>2.09<br>**2.08**_×_<br>**2.13**|1.80_×_<br>1.84<br>**1.93**_×_<br>**1.87**|1.73_×_<br>1.76<br>**1.79**_×_<br>**1.83**|2.01_×_<br>2.08<br>**2.07**_×_<br>**2.14**|
|20%<br>Random<br>**SFDD (Ours)**|2.27_×_<br>2.72<br>**2.38**_×_<br>**2.77**|2.34_×_<br>2.43<br>**2.51**_×_<br>**2.52**|2.08_×_<br>2.35<br>**2.28**_×_<br>**2.40**|1.83_×_<br>2.14<br>**2.02**_×_<br>**2.19**|1.81_×_<br>1.90<br>**1.97**_×_<br>**2.04**|2.06_×_<br>2.31<br>**2.23**_×_<br>**2.39**|


5.4 ANALYSIS OF TRAINING EFFICIENCY


A primary motivation for data selection is to im- Training Time vs. Data Retain Ratio


0

|582|27s|Col3|Col4|Col5|Ra<br>SF|ndom Filteri<br>DD (Ours)|ng|
|---|---|---|---|---|---|---|---|
|~~(1.0~~|~~39~~<br>(1.<br><br>~~0×)~~|~~640s~~<br>47×)<br>360<br>~~(1.6~~|81s<br>~~1×)~~<br>|||||
||~~35~~<br>(1|~~531s~~<br>64×)<br>31|~~295~~<br>(1.9<br>44s<br>|~~5s~~<br>7×)<br>263<br>~~(2.2~~<br>|~~5s~~<br>7×)<br>263<br>~~(2.2~~<br>|16s<br>~~1×)~~<br>~~189~~<br>|~~71s~~<br>|
|||(1.8|8×)<br>287<br>~~(2.0~~|7s<br>~~2×)~~<br>~~239~~<br>(2.4|7s<br>~~2×)~~<br>~~239~~<br>(2.4|(3.0<br>~~21s~~<br>3×)<br>182<br>|7×)<br>22s<br>|
|||||||~~(3.2~~|~~0×)~~|


ing time scales approximately linearly with the 100% Data Retain Ratio70% 60% 50% 40% 30%
data retain ratio, aligning with the intuition that Figure 4: Training time as a function of the data
training speedup is directly proportional to the retain ratio (including data-selection time). Each
data reduction rate. Meanwhile, even with a point is annotated with the absolute wall-clock
larger selection cost, the SFDD curve lies below time and the corresponding training speedup.
the random-filtering curve across retain ratios,
indicating an improvement in net training speed relative to random filtering. We hypothesize that
this arises from enhanced batching efficiency when training on samples with more flat tokens. Moreover, for instance, at the 50% retain ratio, our SFDD method reduces training time from 58 _,_ 227s (full
dataset) to 28 _,_ 787s, achieving a **2.02** _×_ **training speedup**, while the inference speedup decreases by
less than 4% (Section 5.2). This finding underscores the potential of our approach to substantially
reduce computational costs with minimal impact on the final model’s SD inference time.


60,000

50,000

40,000

30,000

20,000

10,000


Training Time vs. Data Retain Ratio


0


100% 70% 60% 50% 40% 30%
Data Retain Ratio


Figure 4: Training time as a function of the data
retain ratio (including data-selection time). Each
point is annotated with the absolute wall-clock
time and the corresponding training speedup.


6 CONCLUSION


In this work, we address the problem of inefficient draft model training in speculative decoding.
We introduce flatness, a novel concept that identifies valuable training data by measuring the target
model’s predictive uncertainty. We propose a data-centric selection method SFDD, which uses this
principle to curate smaller, more potent training datasets. This establishes a new paradigm that
significantly enhances training efficiency while preserving the model’s inference capabilities. Future
work could generalize this approach to other architectures or explore dynamic selection strategies.


10


ACKNOWLEDGEMENT


This work is supported by Jiangsu Province Carbon Peak Carbon Neutrality Science and Technology
Innovation Special Fund Project (Grant No. BT2025029) and National Natural Science Foundation
of China (62576091). Additional support was provided by the Southeast University Big Data Computing Center and the Southeast University Kunpeng & Ascend Center of Cultivation.


ETHICS STATEMENT


Our research is dedicated to advancing the computational efficiency of training algorithms for speculative decoding. All experiments are conducted using publicly available models and datasets, ensuring transparency and accessibility. We have carefully considered the potential impacts of our work
and do not foresee any direct negative societal consequences or ethical concerns. Our methodology
is designed to reduce the computational resources required for training large models, which we believe constitutes a positive contribution to the field by promoting more environmentally sustainable
and accessible research.


REPRODUCIBILITY STATEMENT


To ensure full reproducibility, the source code for our method has been made publicly available at
[https://github.com/fjm9933/Flatness. Our experimental setup, including the specific](https://github.com/fjm9933/Flatness)
models, datasets, is detailed in Section 5.1. Furthermore, complete theoretical derivations for our
proposed approach are provided in Appendix A and Appendix B, allowing for thorough verification of our analytical results. We are committed to transparency and have provided all necessary
components for the research community to replicate and build upon our findings.


REFERENCES


Sharegpt: Share your chatgpt conversations. [https://sharegpt.com/,](https://sharegpt.com/) 2023. Accessed:
2025-09-14.


Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical
report. _arXiv preprint arXiv:2303.08774_, 2023.


Gregor Bachmann, Sotiris Anagnostidis, Albert Pumarola, Markos Georgopoulos, Artsiom
Sanakoyeu, Yuming Du, Edgar Sch¨onfeld, Ali Thabet, and Jonas Kohler. Judge decoding: Faster
speculative sampling requires going beyond model alignment. _arXiv preprint arXiv:2501.19309_,
2025.


Dara Bahri, Heinrich Jiang, Tal Schuster, and Afshin Rostamizadeh. Is margin all you need? an
extensive empirical study of active learning on tabular data, 2022. URL [https://arxiv.](https://arxiv.org/abs/2210.03822)
[org/abs/2210.03822.](https://arxiv.org/abs/2210.03822)


Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri
Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. _arXiv_
_preprint arXiv:2401.10774_, 2024.


Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John
Jumper. Accelerating large language model decoding with speculative sampling. _arXiv preprint_
_arXiv:2302.01318_, 2023.


Stanley F Chen and Joshua Goodman. An empirical study of smoothing techniques for language
modeling. _Computer Speech & Language_, 13(4):359–394, 1999.


Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to
solve math word problems. _arXiv preprint arXiv:2110.14168_, 2021.


11


Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai,
Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, et al. Layerskip: Enabling early
exit inference and self-speculative decoding. In _Proceedings of the 62nd Annual Meeting of the_
_Association for Computational Linguistics (Volume 1:_ _Long Papers)_, pp. 12622–12642, 2024.


Raghavv Goel, Mukul Gagrani, Wonseok Jeon, Junyoung Park, Mingu Lee, and Christopher Lott.
Direct alignment of draft model for speculative decoding with chat-fine-tuned llms. _arXiv preprint_
_arXiv:2403.00858_, 2024.


Zhuocheng Gong, Jiahao Liu, Ziyue Wang, Pengfei Wu, Jingang Wang, Xunliang Cai, Dongyan
Zhao, and Rui Yan. Graph-structured speculative decoding. _arXiv_ _preprint_ _arXiv:2407.16207_,
2024.


Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd
of models. _arXiv preprint arXiv:2407.21783_, 2024.


Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms
via reinforcement learning. _arXiv preprint arXiv:2501.12948_, 2025.


Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution
examples in neural networks. _arXiv preprint arXiv:1610.02136_, 2016.


Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. _arXiv_
_preprint arXiv:1503.02531_, 2015.


Shijing Hu, Jingyang Li, Zhihui Lu, and Pan Zhou. Bridging draft policy misalignment: Group tree
optimization for speculative decoding. _arXiv preprint arXiv:2509.22134_, 2025.


Minsoo Kim, Sihwa Lee, Janghwan Lee, Sukjin Hong, Du-Seong Chang, Wonyong Sung, and Jungwook Choi. Token-scaled logit distillation for ternary weight generative language models. _Ad-_
_vances in Neural Information Processing Systems_, 36:42097–42118, 2023.


Jan Kremer, Kim Steenstrup Pedersen, and Christian Igel. Active learning with support vector
machines. _Wiley_ _Interdisciplinary_ _Reviews:_ _Data_ _Mining_ _and_ _Knowledge_ _Discovery_, 4(4):313–
326, 2014.


Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion
Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav
Petrov. Natural questions: A benchmark for question answering research. _Transactions_ _of_ _the_
_Association_ _for_ _Computational_ _Linguistics_, 7:452–466, 2019. doi: 10.1162/tacl ~~a~~ ~~0~~ 0276. URL
[https://aclanthology.org/Q19-1026/.](https://aclanthology.org/Q19-1026/)


Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative
decoding. In _International Conference on Machine Learning_, pp. 19274–19286. PMLR, 2023.


Jinhao Li, Jiaming Xu, Shan Huang, Yonghua Chen, Wen Li, Jun Liu, Yaoxiu Lian, Jiayi Pan,
Li Ding, Hao Zhou, et al. Large language model inference acceleration: A comprehensive hardware perspective. _arXiv preprint arXiv:2410.04466_, 2024a.


Lei Li, Yankai Lin, Shuhuai Ren, Peng Li, Jie Zhou, and Xu Sun. Dynamic knowledge distillation
for pre-trained language models. _arXiv preprint arXiv:2109.11295_, 2021.


Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-3: Scaling up inference
acceleration of large language models via training-time test (2025). _Preprint_ _at_ _https://doi._
_org/10.48550/arXiv_, 2503.


Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires
rethinking feature uncertainty. _arXiv preprint arXiv:2401.15077_, 2024b.


Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-2: Faster inference of language
models with dynamic draft trees, 2024. _URL https://arxiv. org/abs/2406.16858_, 2024c.


12


Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Doll´ar. Focal loss for dense
object detection. In _Proceedings_ _of_ _the_ _IEEE_ _international_ _conference_ _on_ _computer_ _vision_, pp.
2980–2988, 2017.


Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-of-distribution detection. _Advances in neural information processing systems_, 33:21464–21475, 2020.


Clara Meister and Ryan Cotterell. Language model evaluation beyond perplexity. _arXiv_ _preprint_
_arXiv:2106.00085_, 2021.


Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae
Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, et al. Specinfer: Accelerating large language model serving with tree-based speculative inference and verification. In _Proceedings of the_
_29th_ _ACM_ _International_ _Conference_ _on_ _Architectural_ _Support_ _for_ _Programming_ _Languages_ _and_
_Operating Systems, Volume 3_, pp. 932–949, 2024.


Giovanni Monea, Armand Joulin, and Edouard Grave. Pass: Parallel speculative sampling. _arXiv_
_preprint arXiv:2311.13581_, 2023.


Zongyue Qin, Ziniu Hu, Zifan He, Neha Prakriya, Jason Cong, and Yizhou Sun. Optimized multitoken joint decoding with auxiliary model for llm inference. _arXiv_ _preprint_ _arXiv:2407.09722_,
2024.


Andrea Santilli, Silvio Severino, Emilian Postolache, Valentino Maiorca, Michele Mancusi, Riccardo Marin, and Emanuele Rodol`a. Accelerating transformer inference for translation via parallel decoding. _arXiv preprint arXiv:2305.10427_, 2023.


Abigail See, Peter J Liu, and Christopher D Manning. Get to the point: Summarization with pointergenerator networks. _arXiv preprint arXiv:1704.04368_, 2017.


Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin,
Percy Liang, and Tatsunori B Hashimoto. Alpaca: A strong, replicable instructionfollowing model. _Stanford_ _Center_ _for_ _Research_ _on_ _Foundation_ _Models._ _https://crfm._ _stanford._
_edu/2023/03/13/alpaca. html_, 3(6):7, 2023.


Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee
Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and
efficient foundation language models. _arXiv preprint arXiv:2302.13971_, 2023.


Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen,
Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive
effective reinforcement learning for llm reasoning. _arXiv preprint arXiv:2506.01939_, 2025.


Yepeng Weng, Dianwen Mei, Huishi Qiu, Xujie Chen, Li Liu, Jiang Tian, and Zhongchao Shi.
Coral: Learning consistent representations across multi-step training with lighter speculative
drafter. _arXiv preprint arXiv:2502.16880_, 2025.


Hanling Yi, Feng Lin, Hongbin Li, Ning Peiyang, Xiaotian Yu, and Rong Xiao. Generation meets
verification: Accelerating large language model inference with smart parallel auto-correct decoding. In _Findings_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics:_ _ACL_ _2024_, pp. 5285–5299,
2024.


Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra. Draft&
verify: Lossless large language model acceleration via self-speculative decoding. In _Proceedings_
_of_ _the_ _62nd_ _Annual_ _Meeting_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics_ _(Volume_ _1:_ _Long_
_Papers)_, pp. 11263–11282, 2024a.


Lefan Zhang, Xiaodan Wang, Yanhua Huang, and Ruiwen Xu. Learning harmonized representations
for speculative sampling. _arXiv preprint arXiv:2408.15766_, 2024b.


Kesen Zhao, Jiaxin Shi, Beier Zhu, Junbao Zhou, Xiaolong Shen, Yuan Zhou, Qianru Sun, and
Hanwang Zhang. Real-time motion-controllable autoregressive video diffusion. _arXiv_ _preprint_
_arXiv:2510.08131_, 2025a.


13


Kesen Zhao, Beier Zhu, Qianru Sun, and Hanwang Zhang. Unsupervised visual chain-of-thought
reasoning via preference optimization. _arXiv preprint arXiv:2504.18397_, 2025b.


Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and
chatbot arena. _Advances in neural information processing systems_, 36:46595–46623, 2023.


Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh,
Sanjiv Kumar, Jean-Franc¸ois Kagy, and Rishabh Agarwal. Distillspec: Improving speculative
decoding via knowledge distillation. _arXiv preprint arXiv:2310.08461_, 2023.


Yuxuan Zhou, Heng Li, Zhi-Qi Cheng, Xudong Yan, Yifei Dong, Mario Fritz, and Margret
Keuper. Maxsup: Overcoming representation collapse in label smoothing. _arXiv_ _preprint_
_arXiv:2502.15798_, 2025.


Xingyu Zhu, Beier Zhu, Yi Tan, Shuo Wang, Yanbin Hao, and Hanwang Zhang. Enhancing zeroshot vision models by label-free prompt distribution learning and bias correcting. _Advances_ _in_
_Neural Information Processing Systems_, 37:2001–2025, 2024a.


Xingyu Zhu, Beier Zhu, Yi Tan, Shuo Wang, Yanbin Hao, and Hanwang Zhang. Selective visionlanguage subspace projection for few-shot clip. In _Proceedings_ _of_ _the_ _32nd_ _ACM_ _International_
_Conference on Multimedia_, pp. 3848–3857, 2024b.


Xingyu Zhu, Shuo Wang, Beier Zhu, Miaoge Li, Yunfan Li, Junfeng Fang, Zhicai Wang, Dongsheng
Wang, and Hanwang Zhang. Dynamic multimodal prototype learning in vision-language models.
In _Proceedings_ _of_ _the_ _IEEE/CVF_ _international_ _conference_ _on_ _computer_ _vision_, pp. 2501–2511,
2025.


14


A DERIVATION OF THE OPTIMAL GAUSSIAN SOLUTION


In this section, we provide a detailed derivation for the parameters of the optimal distribution _r_ _[∗]_
under the Gaussian assumption.
**Definition** **A.1** (Problem Formulation) **.** Let _p_ = _N_ ( _µp, σp_ [2][)] [and] _[q]_ [=] _[N]_ [(] _[µ][q][, σ]_ _q_ [2][)] [be] [the] [target] [and]
anchor distributions, respectively. We seek an optimal distribution _r_ _[∗]_ = _N_ ( _µ_ _[∗]_ _r_ _[, σ]_ _r_ [2] _[∗]_ [)][ that solves the]
following constrained optimization problem:


min _D_ KL( _p∥r_ ) (9)
_r_

subject to _D_ KL( _r∥q_ ) _≤_ _θ_ (10)


where _θ_ _≥_ 0 is a fixed budget. For the sake of simplicity, below we restrict the search space for _r_ to
the family of Gaussian distributions.

**Theorem** **A.2** (Optimal Gaussian Parameters) **.** _There_ _exists_ _a_ _one-parameter_ _family_ _of_ _Gaussian_
_candidates_

_r_ ( _τ_ ) := _N_ ( _µr_ ( _τ_ ) _, σr_ [2][(] _[τ]_ [))] _[,]_ _τ_ _∈_ [0 _,_ 1] _,_ (11)
_and a unique parameter τ_ _[∗]_ _∈_ [0 _,_ 1] _such that the optimal solution r_ _[∗]_ _to the problem above is given_
_by_

_r_ _[∗]_ = _r_ ( _τ_ _[∗]_ ) = _N_ ( _µ_ _[∗]_ _r_ _[, σ]_ _r_ [2] _[∗]_ [)] _[.]_ (12)
_For any τ_ _∈_ [0 _,_ 1] _, we have_


_µr_ ( _τ_ ) = (1 _−_ _τ_ ) _µp_ + _τ µq,_ (13)

_σr_ [2][(] _[τ]_ [) = (1] _[ −]_ _[τ]_ [)] _[ σ]_ _p_ [2] [+] _[ τ σ]_ _q_ [2] [+] _[ τ]_ [ 2][(1] _[ −]_ _[τ]_ [) (] _[µ][p]_ _[−]_ _[µ][q]_ [)][2] _[.]_ (14)

_In particular, µ_ _[∗]_ _r_ [=] _[ µ][r]_ [(] _[τ][ ∗]_ [)] _[ and][ σ]_ _r_ [2] _[∗]_ = _σr_ [2][(] _[τ][ ∗]_ [)] _[.]_ _[Let]_ [ ∆] _[µ]_ [:=] _[ µ][p]_ _[−]_ _[µ][q]_ _[and define]_


_σq_ [2] _r_ [(] _[τ]_ [) + (1] _[ −]_ _[τ]_ [)][2][∆][2] _µ_   log _[σ]_ [2] _−_ 1
_σr_ [2] ( _τ_ ) [+] _σq_ [2]


_g_ ( _τ_ ) := _D_ KL( _r_ ( _τ_ ) _∥q_ ) = [1]

2


_._ (15)


_Then:_


    - _If_ 0 _≤_ _θ_ _< D_ KL( _p∥q_ ) _, the inequality constraint is active and τ_ _[∗]_ _is uniquely determined by_
_g_ ( _τ_ _[∗]_ ) = _θ, yielding r_ _[∗]_ = _r_ ( _τ_ _[∗]_ ) _._


    - _If θ_ _≥_ _D_ KL( _p∥q_ ) _, the constraint is inactive and τ_ _[∗]_ = 0 _, hence r_ _[∗]_ = _p._


    - _Boundary cases:_ _θ_ = 0 _⇒_ _τ_ _[∗]_ = 1 _(so r_ _[∗]_ = _q); θ_ = _D_ KL( _p∥q_ ) _⇒_ _τ_ _[∗]_ = 0 _(so r_ _[∗]_ = _p)._


_Moreover, if p_ = _q, the function g_ ( _τ_ ) _is strictly decreasing on τ_ _∈_ (0 _,_ 1) _, hence the solution τ_ _[∗]_ _to_
_g_ ( _τ_ ) = _θ is unique whenever θ_ _∈_ [0 _, D_ KL( _p∥q_ )) _._


_Proof._ We use the method of Lagrange multipliers (KKT for inequality constraints). For onedimensional Gaussians,


_D_ KL� _N_ ( _µ_ 1 _, σ_ 1 [2][)] �� _N_ ( _µ_ 2 _, σ_ 22 [)] - = [1]

2


- 2 1 [+ (] _[µ]_ [1] _[−]_ _[µ]_ [2][)][2] log _[σ]_ [2] + _[σ]_ [2] _−_ 1 _._ (16)
_σ_ 1 [2] _σ_ 2 [2]


Write the decision variables as ( _µr, σr_ [2][)][ with] _[ σ]_ _r_ [2] _[>]_ [ 0][, and define]


_,_ (17)


_._ (18)


_J_ ( _µr, σr_ [2][) :=] _[ D]_ [KL][(] _[p][∥][r]_ [) =] [1]

2


log _[σ]_ _r_ [2] + _[σ]_ _p_ [2] [+ (] _[µ][p]_ _[−]_ _[µ][r]_ [)][2] _−_ 1
_σp_ [2] _σr_ [2]


log _[σ]_ _r_ [2] + _[σ]_ _p_ [2] [+ (] _[µ][p]_ _[−]_ _[µ][r]_ [)][2]
_σp_ [2] _σr_ [2]


_C_ ( _µr, σr_ [2][) :=] _[ D]_ [KL][(] _[r][∥][q]_ [) =] [1]

2


_q_ _r_ [+ (] _[µ][r]_ _[−]_ _[µ][q]_ [)][2]
log _[σ]_ [2] + _[σ]_ [2]
_σr_ [2] _σq_ [2]


_−_ 1
_σq_ [2]


The Lagrangian is

_L_ ( _µr, σr_ [2] _[, ν]_ [) =] _[ J]_ [(] _[µ][r][, σ]_ _r_ [2][) +] _[ ν]_      - _C_ ( _µr, σr_ [2][)] _[ −]_ _[θ]_      - _,_ _ν_ _≥_ 0 _,_ (19)


with the complementarity condition _ν_ - _C_ ( _µr, σr_ [2][)] _[ −]_ _[θ]_ - = 0 and primal feasibility _C_ ( _µr, σr_ [2][)] _[ ≤]_ _[θ]_ [.]


15


**Stationarity conditions.** Taking derivatives,


_∂L_
= _[µ][r][ −]_ _[µ][p]_
_∂µr_ _σr_ [2]


= 0 _,_ (20)
_σq_ [2]


_[ −]_ _[µ][p]_

+ _ν_ _[µ][r][ −]_ _[µ][q]_
_σr_ [2] _σq_ [2]


+ _ν_ [1]

2


_σq_ [2]


_∂L_
= [1]
_∂σr_ [2] 2


1 _p_ [+ (] _[µ][p]_ _[−]_ _[µ][r]_ [)][2]

_−_ _[σ]_ [2]
_σr_ [2] ( _σr_ [2] ) [2]


_−_ [1]


[1] + [1]

_σr_ [2] _σq_ [2]


= 0 _._ (21)


**Step 1:** **Mean parameter.** From Equation 20,

_q_ _[µ][p]_ [+] _[ ν σ]_ _r_ [2] _[µ][q]_
( _µr −_ _µp_ ) _σq_ [2] [+] _[ ν σ]_ _r_ [2][(] _[µ][r]_ _[−]_ _[µ][q]_ [) = 0] _[⇒]_ _[µ][r]_ [=] _[σ]_ [2] _._ (22)
_σq_ [2] + _ν σr_ [2]


Introduce the reparameterization

_τ_ := _ν σr_ [2] _∈_ [0 _,_ 1] _,_ (23)
_σq_ [2] + _ν σr_ [2]


which yields the affine interpolation:


_µr_ = (1 _−_ _τ_ ) _µp_ + _τµq,_ _µr −_ _µp_ = _−_ _τ_ ∆ _µ,_ _µr −_ _µq_ = (1 _−_ _τ_ )∆ _µ._ (24)


If the constraint is inactive ( _ν_ = 0), then _τ_ = 0 and _µr_ = _µp_ .


**Step 2: Variance parameter.** Using Equation 21, substituting ( _µp−µr_ ) [2] = _τ_ [2] ∆ [2] _µ_ [, and employing]

_τ σq_ [2]
_ν_ =
_σr_ [2] (1 _−_ _τ_ ) [(from the definition of] _[ τ]_ [), we obtain]


1 _p_ [+] _[ τ]_ [ 2][∆] _µ_ [2] _τ_

_−_ _[σ]_ [2] +
_σr_ [2] ( _σr_ [2] ) [2] 1 _−_ _τ_

Multiplying by ( _σr_ [2][)][2][ and rearranging gives]


_−_ [1]


_σq_ [2]


[1] + [1]

_σr_ [2] _σq_ [2]


= 0 _._ (25)


hence the closed form


_σr_ [2] _q_ _p_ _[−]_ _[τ]_ [ 2][∆] _µ_ [2] [= 0] _[,]_ (26)
1 _−_ _τ_ _[−]_ 1 _[τ σ]_ _−_ [2] _τ_ _[−]_ _[σ]_ [2]


_σr_ [2][(] _[τ]_ [) = (1] _[ −]_ _[τ]_ [)] _[ σ]_ _p_ [2] [+] _[ τ σ]_ _q_ [2] [+] _[ τ]_ [ 2][(1] _[ −]_ _[τ]_ [) ∆] _µ_ [2] _[,]_ (27)


which proves Equation 14. If the constraint is inactive ( _τ_ = 0), then _σr_ [2][(0) =] _[ σ]_ _p_ [2][.]


**Step 3:** **Constraint, monotonicity, and cases.** Insert _µr_ ( _τ_ ) and _σr_ [2][(] _[τ]_ [)][ into] _[ C]_ [to obtain the scalar]
function


_σq_ [2] _r_ [(] _[τ]_ [) + (1] _[ −]_ _[τ]_ [)][2][∆][2] _µ_
log _[σ]_ [2] _−_ 1
_σr_ [2] ( _τ_ ) [+] _σq_ [2]


_g_ ( _τ_ ) = [1]

2


_._ (28)


Differentiating and using


_d_
_r_ [(] _[τ]_ [) =] _[ −][σ]_ _p_ [2] [+] _[ σ]_ _q_ [2] [+] _[ τ]_ [(2] _[ −]_ [3] _[τ]_ [)∆] _µ_ [2] _[,]_ _µr_ ( _τ_ ) _−_ _µq_ = (1 _−_ _τ_ )∆ _µ,_ (29)
_dτ_ _[σ]_ [2]

we get


- _d_ _r_ [(] _[τ]_ [)] _[ −]_ [(1] _[ −]_ _[τ]_ [)∆] _µ_ [2] _≤_ 0 _,_ (30)
_dτ_ _[σ]_ [2] _σq_ [2]


_g_ _[′]_ ( _τ_ ) = [1]

2


- 1 1

_−_
_σq_ [2] _σr_ [2] ( _τ_ )


with equality only in degenerate cases (e.g. _τ_ _∈{_ 0 _,_ 1 _}_ or ∆ _µ_ = 0 together with _σp_ [2] [=] _[ σ]_ _q_ [2][).] [Hence] _[ g]_
is strictly decreasing on (0 _,_ 1) whenever _p ̸_ = _q_ .


_Active-constraint case:_ If 0 _≤_ _θ_ _< D_ KL( _p∥q_ ), complementarity implies _ν_ _>_ 0, thus _τ_ _∈_ (0 _,_ 1] and
the unique optimal parameter _τ_ _[∗]_ satisfies _g_ ( _τ_ _[∗]_ ) = _θ_ .


_Inactive-constraint case:_ If _θ_ _≥_ _D_ KL( _p∥q_ ), taking _r_ = _p_ yields feasibility _D_ KL( _p∥q_ ) _≤_ _θ_ and zero
objective value; optimality follows since _J_ ( _·_ ) _≥_ 0 with equality only at _r_ = _p_ . Therefore _ν_ = 0,
_τ_ _[∗]_ = 0, and _r_ _[∗]_ = _p_ .


The boundary _θ_ = 0 gives _τ_ _[∗]_ = 1 and _r_ _[∗]_ = _q_ ; _θ_ = _D_ KL( _p∥q_ ) gives _τ_ _[∗]_ = 0 and _r_ _[∗]_ = _p_ .


16


B ASYMPTOTIC BEHAVIOR OF THE COSINE SIMILARITY FOR A DISCRETIZED
GAUSSIAN


This appendix provides a rigorous derivation of the asymptotic relationship between the cosine similarity and the standard deviation _σ_ of a Gaussian distribution as its discretization becomes infinitely
fine. Throughout, the window [ _−L, L_ ] is _large_ _but_ _finite_ : large so that the truncated Gaussian approximates the full Gaussian well, and finite so that the discrete uniform distribution is well-defined.
This ”large window” condition is naturally met in applications like Large Language Models, where
predictive distributions over vast vocabularies are highly concentrated.

**Definition** **B.1** (Discretized Gaussian) **.** Let a discrete probability distribution _p_ over a vocabulary
of size _V_ be a discretization of a continuous Gaussian probability density function (PDF) _ϕ_ ( _x_ ; _µ, σ_ )
over a symmetric interval [ _−L, L_ ]. The probability _pi_ in the _i_ -th bin of width ∆ _x_ = 2 _L/V_ is defined
by the PDF at the bin’s center _xi_, such that _pi_ = _ϕ_ ( _xi_ ; _µ, σ_ ) ∆ _x_ . This definition ensures [�] _i_ _[V]_ =1 _[p][i]_ _[→]_

- _−LL_ _[ϕ]_ [(] _[x]_ [;] _[ µ, σ]_ [)] _[ dx]_ [ as] _[ V]_ _[→∞]_ [.] [Let] _[ U]_ [be the discrete uniform distribution][ (1] _[/V, . . .,]_ [ 1] _[/V]_ [ )][.]

**Theorem B.2** (Asymptotic Cosine Similarity) **.** _In the limit as the vocabulary size V_ _→∞_ _(with L_
_fixed and large but finite), the cosine similarity between the discretized Gaussian p and the uniform_
_distribution U_ _satisfies_


                  - _L_


lim [=]
_V →∞_ [cos(] _[p, U]_ [)]


_ϕ_ ( _x_ ; _µ, σ_ ) _dx_
_−L_

~~�~~


 - _L_
2 _L_


_._ (31)


_ϕ_ ( _x_ ; _µ, σ_ ) [2] _dx_
_−L_


_Assuming_ _the interval_ [ _−L, L_ ] _is_ _large enough to contain most of the probability mass,_ _we further_
_obtain the closed-form dependence on σ:_


lim [=]
_V →∞_ [cos(] _[p, U]_ [)]


_σ_ ~~_[√]_~~ _π_

_,_ (32)
_L_


_i.e., for fixed (large but finite) L, the cosine similarity obeys_ cos( _p, U_ ) _∝_ _σ_ [1] _[/]_ [2] _._


_Proof._ We begin with the cosine similarity written purely in terms of sums:


       - _V_        - _V_
_i_ =1 _[p][i][ ·]_ [ (1] _[/V]_ [ )] _i_ =1 _[p][i]_
cos( _p, U_ ) = _._ (33)
��� _V_ ��� _V_        _i_ =1 _[p]_ _i_ [2] _i_ =1 [(1] _[/V]_ [ )][2][�] [=] _V_ [�] _i_ _[V]_ =1 _[p]_ _i_ [2]


Define


_V_

- _p_ [2] _i_ _[.]_ (34)

_i_ =1


_SV_ :=


_V_

- _pi,_ _QV_ := _V_


_i_ =1


By the discretization rule _pi_ = _ϕ_ ( _xi_ ; _µ, σ_ ) ∆ _x_ with ∆ _x_ = 2 _L/V_, these sums become Riemann
sums. In the limit _V_ _→∞_, they converge to integrals:


                 - _L_


_SV_ _−−−−→_
_V →∞_


_ϕ_ ( _x_ ; _µ, σ_ ) _dx,_ (35)
_−L_


and


_V_
_QV_ = _V_ (∆ _x_ ) [2] - _ϕ_ ( _xi_ ; _µ, σ_ ) [2] = (2 _L_ ) �∆ _x_


_i_ =1


           - _L_

- _ϕ_ ( _xi_ ; _µ, σ_ ) [2][�] _−−−−→_

_i_ =1 _V →∞_ [2] _[L]_ _−_


_V_


_ϕ_ ( _x_ ; _µ, σ_ ) [2] _dx._
_−L_


(36)


Combining these limits yields the exact fixed- _L_ formula Equation 31.


To make the _σ_ -dependence explicit, we now invoke a condition that is well-justified in practice.
The output of systems like Large Language Models (LLMs) over their vast vocabularies is empirically a long-tail distribution: probability mass is highly concentrated on a few likely outcomes.
The Gaussian PDF serves here as a tractable mathematical model for this concentration. A highly


17


peaked distribution is modeled by a **small standard deviation** _σ_ . Given that the vocabulary space
(represented by _L_ ) is large, the condition _L ≫_ _σ_ is naturally fulfilled.


Under this _L ≫_ _σ_ condition, the probability mass in the tails outside [ _−L, L_ ] is negligible. We can
therefore approximate the truncated integrals by their values on the full real line R:

        -         


          _ϕ_ ( _x_ ; _µ, σ_ ) _dx_ = 1 _,_

R


1
_ϕ_ ( _x_ ; _µ, σ_ ) [2] _dx_ = (37)
R 2 _σ_ ~~_[√]_~~ _π_ _[.]_


This gives the approximation:

        - _L_


1
_ϕ_ ( _x_ ; _µ, σ_ ) [2] _dx ≈_ (38)
_−L_ 2 _σ_ ~~_[√]_~~ _π_ _[.]_


_L_ - _L_

_ϕ_ ( _x_ ; _µ, σ_ ) _dx ≈_ 1 _,_
_−L_ _−_


Substituting Equation 38 into the exact formula Equation 31 gives the final asymptotic result:


1
lim _[≈]_ =
_V →∞_ [cos(] _[p, U]_ [)]          - 1
2 _L ·_
2 _σ_ ~~_[√]_~~ _π_


which is Equation 32. This completes the proof.


C RESULTS AT TEMPERATURE=0


_σ_ ~~_[√]_~~ _π_

_,_ (39)
_L_


We also evaluate our proposed SFDD when the default temperature is set to 0. The results are
summarized in Table 4 and Table 5, which again show that our proposed SFDD can deliver more
reliable data selection performance towards more efficient draft model training in the context of SD.


Table 4: Comparison of various metrics for data importance at a 50% retain ratio (temperature = 0).

|GSM8K Alpaca MTB CNN/DM NQ Average<br>Method Speedup l Speedup l Speedup l Speedup l Speedup l Speedup l|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|No Filter|2.86_×_<br>3.42|2.99_×_<br>3.12|3.06_×_<br>3.17|2.66_×_<br>2.75|2.43_×_<br>2.53|2.80_×_<br>3.00|
|Random<br>Entropy<br>Top-1 Probability<br>Margin<br>Energy Score<br>PPL<br>**SFDD (Ours)**|2.52_×_<br>3.03<br>2.66_×_<br>2.96<br>2.58_×_<br>3.00<br>2.65_×_<br>2.99<br>2.52_×_<br>2.91<br>2.54_×_<br>2.91<br>**2.79**_×_<br>**3.07**|2.69_×_<br>2.81<br>2.73_×_<br>2.83<br>2.68_×_<br>2.66<br>2.60_×_<br>2.80<br>2.69_×_<br>2.80<br>2.69_×_<br>2.90<br>**2.92**_×_<br>**2.90**|2.53_×_<br>2.82<br>2.54_×_<br>2.82<br>2.56_×_<br>2.81<br>2.49_×_<br>2.69<br>2.57_×_<br>2.79<br>2.49_×_<br>2.80<br>**2.70**_×_<br>**2.88**|2.22_×_<br>2.50<br>2.24_×_<br>2.52<br>2.29_×_<br>2.51<br>2.23_×_<br>2.41<br>2.24_×_<br>2.50<br>2.25_×_<br>2.51<br>**2.46**_×_<br>**2.54**|2.08_×_<br>2.31<br>2.19_×_<br>2.34<br>2.11_×_<br>2.32<br>2.06_×_<br>2.18<br>2.15_×_<br>2.29<br>2.22_×_<br>2.32<br>**2.24**_×_<br>**2.38**|2.41_×_<br>2.69<br>2.47_×_<br>2.69<br>2.44_×_<br>2.66<br>2.41_×_<br>2.61<br>2.43_×_<br>2.66<br>2.44_×_<br>2.69<br>**2.62**_×_<br>**2.75**|


Table 5: Comparison of SFDD and Random Filtering under different retain ratios (temperature = 0).

|GSM8K Alpaca MTB CNN/DM NQ Average<br>Retain Ratio Method Speedup l Speedup l Speedup l Speedup l Speedup l Speedup l|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|100%<br>No Filter|2.86_×_<br>3.42|2.99_×_<br>3.12|3.06_×_<br>3.17|2.66_×_<br>2.75|2.43_×_<br>2.53|2.80_×_<br>3.00|
|70%<br>Random<br>**SFDD (Ours)**|2.58_×_<br>3.01<br>**2.79**_×_<br>**3.07**|2.70_×_<br>2.79<br>**2.96**_×_<br>**2.88**|2.50_×_<br>2.79<br>**2.98**_×_<br>**2.89**|2.21_×_<br>2.51<br>**2.43**_×_<br>**2.63**|2.08_×_<br>2.30<br>**2.35**_×_<br>**2.39**|2.41_×_<br>2.68<br>**2.70**_×_<br>**2.77**|
|60%<br>Random<br>**SFDD (Ours)**|2.60_×_<br>3.01<br>**2.70**_×_<br>**3.08**|2.71_×_<br>2.78<br>**2.96**_×_<br>**2.91**|2.59_×_<br>2.82<br>**2.76**_×_<br>**2.87**|2.27_×_<br>2.50<br>**2.43**_×_<br>**2.59**|2.15_×_<br>2.30<br>**2.44**_×_<br>**2.40**|2.46_×_<br>2.68<br>**2.66**_×_<br>**2.77**|
|50%<br>Random<br>**SFDD (Ours)**|2.52_×_<br>3.03<br>**2.79**_×_<br>**3.07**|2.69_×_<br>2.81<br>**2.92**_×_<br>**2.90**|2.53_×_<br>2.82<br>**2.70**_×_<br>**2.88**|2.22_×_<br>2.50<br>**2.46**_×_<br>**2.54**|2.08_×_<br>2.31<br>**2.24**_×_<br>**2.38**|2.41_×_<br>2.69<br>**2.62**_×_<br>**2.75**|
|40%<br>Random<br>**SFDD (Ours)**|2.56_×_<br>2.97<br>**2.66**_×_<br>**3.04**|2.54_×_<br>2.74<br>**2.78**_×_<br>**2.87**|2.56_×_<br>2.82<br>**2.87**_×_<br>**2.82**|2.24_×_<br>2.50<br>**2.45**_×_<br>**2.59**|2.00_×_<br>2.26<br>**2.28**_×_<br>**2.36**|2.38_×_<br>2.66<br>**2.61**_×_<br>**2.74**|
|30%<br>Random<br>**SFDD (Ours)**|2.50_×_<br>2.92<br>**2.59**_×_<br>**2.98**|2.60_×_<br>2.72<br>**2.81**_×_<br>**2.81**|2.50_×_<br>2.73<br>**2.59**_×_<br>**2.79**|2.23_×_<br>2.40<br>**2.37**_×_<br>**2.47**|2.05_×_<br>2.23<br>**2.27**_×_<br>**2.30**|2.38_×_<br>2.60<br>**2.53**_×_<br>**2.67**|


D TIMING PROTOCOL AND DATA SELECTION OVERHEAD


**End-to-end timing.** For fair comparisons, all results related to training time reported in the main
text are end-to-end wall-clock measurements that include the data selection stage.


**Cost analysis of SFDD-based data selection.** We note that the cost of SFDD-based data selection
is negligible compared to the training cost. In our experiments, we only need to run SFDD-based


18


data selection once over the training set, which takes only 2,242 seconds and accounts for about
3.85% of the 58,227-seconds whole training time of no filtering. For fair comparisons, our reported
training speedups have already included this one-off data selection cost. From the computational
structure, this is also expected: during training, each sample requires a forward pass of the target
model, a forward pass of the draft model, and a full backward pass, repeated over many epochs. In
contrast, the SFDD-based data selection runs only a single forward pass of the target model over the
training set. Therefore, the one-off cost of SFDD-based data selection remains negligible compared
to multi-epoch training and does not become a practical bottleneck in time or computation.


E MORE TRAINING DETAILS


We follow the official EAGLE-2 training setup and fix the target model during all experiments.
The pretrained target model is LLaMA3-Instruct-8B. The draft model is a lightweight LLaMAstyle predictor with a single transformer layer, initialized and configured according to the official
EAGLE-2. The training hyperparameters for the draft model are summarized in Table 6.


Table 6: Training hyperparameters.
Hyperparameter Value
Number of epochs 30
Learning rate 5 _×_ 10 _[−]_ [5]
Batch size 2
Gradient accumulation steps 1
Total training steps 800,000
Warmup enabled
Warmup steps 2,000
Optimizer AdamW ( _β_ 1 = 0 _._ 9 _,_ _β_ 2 = 0 _._ 95)
Gradient clipping 0.5
Maximum sequence length 2,048
Number of data loader workers 8


F ADDITIONAL ANALYSIS


F.1 ANALYSIS OF INCONSISTENCY BETWEEN ACCEPTANCE LENGTH AND SPEEDUP


In this section, we address the observation from Table 1 regarding the non-linear relationship between acceptance length and inference speedup (e.g., a larger acceptance length gap does not always
translate to a proportional speedup gap). We confirm that this phenomenon is not due to hardware
mismatch or redundancy but is an inherent characteristic of speculative decoding, consistent with
findings in prior works (Li et al., 2024c; Weng et al., 2025). For instance, as shown in Table 1 of
EAGLE-2 (Li et al., 2024c), the acceptance length of Vicuna-13B (Lookahead method) on Alpaca
(4.89) is larger than on MT-bench (4.83), whereas the inference speedup shows the opposite trend.
This indicates that acceptance length cannot be equivalently translated into actual speedup.


Table 7: Length-stratified speculative decoding statistics on GSM8K using SFDD. “Top 50%” denotes the subset containing the longest 50% of samples based on prompt length.


Method Speedup (Full) _l_ (Full) Speedup (Top-50%) _l_ (Top-50%)


SFDD 2.6856 _×_ 3.9467 2.4688 _×_ 3.9506


**Empirical explanations.** The rationale behind this discrepancy is that the acceptance length only
captures the behavior of the decoding stage, while ignoring the prefilling stage. During the decoding
stage, both the draft and target models can benefit from the KV cache, so the same acceptance
length typically results in similar inference costs when the number of decoded tokens is the same.
However, during the prefilling stage—where the KV cache cannot be utilized—the inference cost
grows exponentially with respect to the prompt length. As a result, prompts with longer sequences
often incur significantly higher inference costs, which reduces the achievable speedup even when
the acceptance length during the decoding stage is high.


19


1.0


0.8


0.6


0.4


0.2


0.0


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|low entr<br>high ent|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||


Token index (sorted by entropy of p)


3


2


1


0


1


2


3


4

Token index (sorted by entropy of p)


0.10


0.05


0.00


0.05


0.10


Token index (sorted by entropy of p)


(a) entropy( _p_ ) (b) Change in entropy( _qm_ ) (c) ∆ _L_ 1

Figure 5: **Target-sorted entropy view.** Tokens sorted by the target’s entropy (low _→_ high entropy).
(a) target entropy values; (b) one-epoch change in draft entropy; (c) the one-epoch reduction in
the _L_ 1 discrepancy, ∆ _L_ 1. Similar to the flatness view, we observe that tokens with higher entropy
(indicating flatter distributions) contribute more to the training dynamics (∆ _L_ 1) and exhibit larger
changes in the draft model.


(a) entropy( _p_ )


(b) Change in entropy( _qm_ )


**Experimental validations.** To substantiate the above explanations, we sort all samples in GSM8K
based on their prompt lengths and re-calculate the acceptance length ( _l_ ) and speed on the longest
50% of samples (denoted as “Top 50%”). As shown in Table 7, the speedup on the Top 50% differs
from that on the full dataset, even though their acceptance lengths are similar (3.9467 vs. 3.9506).
This confirms that prompt length variability may impact the final speedup calculation.


F.2 ANALYSIS OF ENTROPY AS A DISTRIBUTION-DISPERSION METRIC


In Section 4, we utilize flatness as the primary metric to characterize token distributions. To further
validate our findings and verify that the observed trends are not an artifact of the specific metric
choice, we perform an additional analysis using entropy. We adopt the identical experimental setup
as used for the flatness analysis in Figure 2: we calculate the entropy of the target distribution for
all tokens, sort the tokens by their target entropy from low to high, and apply a 10-point moving
average for visualization. The results are shown in Figure 5. We observe that the entropy-based
curves exhibit a trend remarkably similar to the flatness curves shown in the main text. In the
low-entropy region, the draft statistics and ∆ _L_ 1 change only slightly within one epoch, whereas in
the high-entropy region the changes are more pronounced. This similarity is theoretically expected
because both metrics fundamentally measure the distance between the token distribution _p_ and the
uniform distribution _U_ . While flatness is defined via cosine similarity, entropy is directly related to
the forward KL divergence from the uniform distribution (where _U_ ( _x_ ) = 1 _/V_ ):


_p_ ( _x_ ) log _[p]_ [(] _[x]_ [)]

1 _/V_

_x_


_p_ ( _x_ ) log _p_ ( _x_ ) + log _V_ = _−H_ ( _p_ ) + const _._ (40)

_x_


_DKL_ ( _p∥U_ ) = 


 1 _/V_ [=]


Thus, maximizing entropy is equivalent to minimizing the KL divergence from the uniform distribution. The consistency between Figure 2 (flatness) and Figure 5 (entropy) again confirms that the
”flatness” or ”uncertainty” of the target distribution is indeed the key factor driving the value of
training tokens in speculative decoding.


F.3 ANALYSIS OF EXPONENTIAL AND HALF-NORMAL DISTRIBUTIONS


In this section, we extend the KL-constrained update from Gaussian to another two distributions:
Exponential and Half-normal. We record the core closed-form expressions in our simulations.


**Exponential distribution.** We consider


_p_ = Exp( _λp_ ) _,_ _q_ = Exp( _λq_ ) _,_ _r_ = Exp( _λr_ ) _,_ (41)

with density _fλ_ ( _x_ ) = _λe_ _[−][λx]_ on _x ≥_ 0. The KL divergence is


_D_ KL�Exp( _λ_ 1) _∥_ Exp( _λ_ 2)� = log _[λ]_ [1]


_[λ]_ [1] + _[λ]_ [2]

_λ_ 2 _λ_ 1


_−_ 1 _._ (42)
_λ_ 1


20


|Exponential, q = 1.0, = 2|Col2|Col3|Col4|
|---|---|---|---|
|Exponential,<br>q = 1.0,<br>= 2|Exponential,<br>q = 1.0,<br>= 2|Exponential,<br>q = 1.0,<br>= 2|Exponential,<br>q = 1.0,<br>= 2|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||


|20 40 60 80 10 Var(p) Half-normal, q = 1.0, = 2|Col2|Col3|Col4|
|---|---|---|---|
|Half-normal,<br>q = 1.0,<br>= 2|Half-normal,<br>q = 1.0,<br>= 2|Half-normal,<br>q = 1.0,<br>= 2|Half-normal,<br>q = 1.0,<br>= 2|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||


0 20 40 60 80 100
Var(p)


|Exponential, q = 100.0, = 2|Col2|Col3|Col4|
|---|---|---|---|
|Exponential,<br>q = 100.0,<br>= 2|Exponential,<br>q = 100.0,<br>= 2|Exponential,<br>q = 100.0,<br>= 2|Exponential,<br>q = 100.0,<br>= 2|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||


|20 40 60 80 10 Var(p) Half-normal, q = 100.0, = 2|Col2|Col3|Col4|
|---|---|---|---|
|Half-normal,<br>q = 100.0,<br>= 2|Half-normal,<br>q = 100.0,<br>= 2|Half-normal,<br>q = 100.0,<br>= 2|Half-normal,<br>q = 100.0,<br>= 2|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||


0 20 40 60 80 100
Var(p)


1.6


1.4


1.2


1.0


0.8


0.6


0.4


0.2


0.0


1.50


1.25


1.00


0.75


0.50


0.25


0.00


|Exponential, q = 0.1, = 2|Col2|Col3|Col4|
|---|---|---|---|
|Exponential,<br>q = 0.1,<br>= 2|Exponential,<br>q = 0.1,<br>= 2|Exponential,<br>q = 0.1,<br>= 2|Exponential,<br>q = 0.1,<br>= 2|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||


|20 40 60 80 10 Var(p) Half-normal, q = 0.1, = 2|Col2|Col3|Col4|
|---|---|---|---|
|Half-normal,<br>q = 0.1,<br>= 2|Half-normal,<br>q = 0.1,<br>= 2|Half-normal,<br>q = 0.1,<br>= 2|Half-normal,<br>q = 0.1,<br>= 2|
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||


0 20 40 60 80 100
Var(p)


Figure 6: **Behavior** **of** ∆ _L_ 1 **as** **a** **function** **of** **the** **teacher** **variance** **in** **the** **Exponential** **and**
**Half-normal** **families** **under** **a** **large** **KL** **budget** _θ_ = 2 **.** Top row: Exponential family with
_q_ = Exp( _λq_ ), _λq_ _∈{_ 0 _._ 1 _,_ 1 _,_ 100 _}_ . Bottom row: Half-normal family with _q_ = HalfNormal( _σq_ ),
_σq_ _∈{_ 0 _._ 1 _,_ 1 _,_ 100 _}_ . In all panels, the _x_ -axis is Var( _p_ ) and the _y_ -axis is ∆ _L_ 1 = _∥p−q∥_ 1 _−∥p−r_ _[∗]_ _∥_ 1.


For fixed draft parameter _λq_ and budget _θ_ _>_ 0, the KL ball _{r_ : _D_ KL( _r∥q_ ) _≤_ _θ}_ induces
a closed interval [ _λ_ low _, λ_ high] on _λr_, where _λ_ low and _λ_ high are the two positive solutions of
_D_ KL(Exp( _λr_ ) _∥_ Exp( _λq_ )) = _θ_, which we solve numerically. The unconstrained minimizer of the
update is _λp_, so the optimal update parameter is simply the projection of _λp_ onto this interval. The
teacher variance in this family is

Var( _p_ ) = [1] _._ (43)

_λ_ [2] _p_

In our simulations, for each variance value we set _λp_ = 1 _/_ �Var( _p_ ), compute the projected _λr_, and
then evaluate ∆ _L_ 1 using the closed-form _L_ 1 distance between Exponential distributions (omitted
here for brevity).


**Half-normal distributions.** We consider


_p_ = HalfNormal( _σp_ ) _,_ _q_ = HalfNormal( _σq_ ) _,_ _r_ = HalfNormal( _σr_ ) _,_ (44)


~~_√_~~
with density _fσ_ ( _x_ ) = _σ_ ~~_[√]_~~ 2 _π_ _[e][−][x]_ [2] _[/]_ [(2] _[σ]_ [2][)] [on] _[ x][ ≥]_ [0][.] [Writing] _[ v]_ [=] _[ σ]_ [2][, the KL divergence is]


_D_ KL�HalfNormal( _σ_ 1) _∥_ HalfNormal( _σ_ 2)� = [1]

2


log _[v]_ [2]


_[v]_ [2] + _[v]_ [1]

_v_ 1 _v_ 2


[1] _−_ 1� _,_ _vi_ = _σi_ [2] _[.]_ (45)

_v_ 2


For fixed _vq_ = _σq_ [2] [and] [budget] _[θ]_ [,] [the] [KL] [ball] _[{][r]_ : _D_ KL( _r∥q_ ) _≤_ _θ}_ induces a
closed interval [ _v_ low _, v_ high] on _vr_, where _v_ low and _v_ high are the two positive solutions of
_D_ KL(HalfNormal( _σr_ ) _∥_ HalfNormal( _σq_ )) = _θ_, again solved numerically. The unconstrained minimizer is _vp_, so the optimal update variance is the projection of _vp_ onto [ _v_ low _, v_ high], and _σr_ = _[√]_ _vr_ .


Var( _p_ ) = _σp_ [2] [=] _[ v][p][.]_ (46)


In our simulations, for each variance value Var( _p_ ) we set _σp_ = ~~�~~ Var( _p_ ), we compute the projected
_vr_, and evaluate ∆ _L_ 1 using the closed-form _L_ 1 distance between Half-normal distributions.


**Numerical** **behavior** **of** ∆ _L_ 1 **vs.** **variance.** Using the formulas above, we numerically evaluate
∆ _L_ 1(Var( _p_ )) under a KL budget _θ_ = 2 for several draft parameters. For the Exponential family, we
fix _q_ = Exp( _λq_ ) with _λq_ _∈{_ 0 _._ 1 _,_ 1 _,_ 100 _}_ and sweep Var( _p_ ) _∈_ [0 _,_ 100]. For the Half-normal family,
we fix _q_ = HalfNormal( _σq_ ) with _σq_ _∈{_ 0 _._ 1 _,_ 1 _,_ 100 _}_ and sweep Var( _p_ ) = _σp_ [2] _[∈]_ [[0] _[,]_ [ 100]][.] [The]


21


resulting six curves are shown in Figure 6. Across the six panels, the shapes of the ∆ _L_ 1–variance
curves differ substantially as _λq_ or _σq_ varies: the dependence on Var( _p_ ) is highly sensitive to the
draft scale, and no simple, consistent monotone trend emerges. In these single-parameter scale
families, _p_ and _q_ are forced to share the same mode, so the effect of the teacher’s variance on ∆ _L_ 1
becomes entangled with the draft’s variance and no longer isolates a clean flatness effect, in contrast
to the Gaussian location–scale model used in the main text.


**Limitations.** These two families have clear limitations compared with the Gaussian family. First,
they are both single-parameter scale families whose mode is fixed at _x_ = 0, which forces _p_ and
_q_ to share the same maximizer for any choice of parameters. In the LLM setting, this essentially
corresponds to the regime where the draft model has already learned the teacher’s top-1 token quite
well, whereas in speculative decoding we are often more interested in tokens where the teacher and
draft still have noticeably different top-1 predictions. Under this “mode-locked” assumption, the
effect of the teacher’s variance on ∆ _L_ 1 becomes strongly entangled with the draft model’s variance,
and no longer cleanly reflects a flatness effect. In contrast, the Gaussian location–scale family
allows us to vary the mean and the variance independently, which better captures the joint effect
of argmax mismatch and distributional shape differences observed in real LLM logits. Second,
although the KL-constrained optimal update _r_ _[∗]_ can also be solved for the Exponential and Halfnormal cases, it requires solving one-dimensional equations (involving, e.g., Lambert- _W_ functions)
and does not admit a simple one-dimensional KKT parameterization as in the Gaussian case, making
the expressions less transparent and not revealing any additional phenomena.


0.6000


0.5000


0.4000 **0.** 43090905


0.3000


0.2000


0.1000
0.0663

0.0000


|Col1|Col2|Col3|Col4|Col5|Lo<br>Hi|wflatnes<br>ghflatne|s(bottom<br>ss(top6605|4305%)<br>%)|
|---|---|---|---|---|---|---|---|---|
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||


Epoch


|Col1|Col2|Col3|Col4|Col5|Col6|L|ow flatnes|s (bottom|4305%)|
|---|---|---|---|---|---|---|---|---|---|
||||||~~H~~|~~H~~|~~igh flatn~~|~~**6**~~<br>~~ ss (top 6~~|~~  %)~~|
||||||~~H~~|~~H~~||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


Epoch


2.25

2.00

1.75

1.50

**1.25**

1.00

0.75

0.50

0.25
0.16

0.00


(a) Average Gradient Norm (b) Average Loss

Figure 7: Training dynamics of tokens with different flatness levels. We track the metrics for the
bottom 35% (Low Flatness) and top 65% (High Flatness) tokens over 15 epochs. **(a)** **Gradient**
**Norm:** Low-flatness tokens exhibit consistently smaller gradient norms, which quickly decay to
negligible values (below 0.1), indicating that the model ceases to learn from them early in training.
In contrast, high-flatness tokens maintain significantly larger gradient norms. **(b)** **Loss** **Curves:**
The loss for low-flatness tokens rapidly converges to near-zero and remains flat, confirming early
saturation. Meanwhile, high-flatness tokens maintain non-trivial loss values throughout the epochs,
indicating that they continue to provide gradient signals for a longer period.


F.4 WHY WE DO NOT USE VARIANCE OF THE DISCRETE TOKEN DISTRIBUTION


In this section, we further clarify why the variance of the discrete distribution produced by the target
LLM is not utilized as a selection criterion in our approach. In the continuous setting, variance relies
on the natural order and metric of the real line and therefore has a clear geometric and statistical
interpretation. In our setting, however, the language model outputs a categorical distribution over
tokens such as “how”, “cat”, or punctuation marks, which do not live on a canonical one-dimensional
numeric axis. If we assign arbitrary indices to tokens and compute a “variance” over these indices,
the resulting value would depend entirely on how the vocabulary is indexed: simply permuting the
token order would change the variance, so it cannot serve as a robust selection metric. For this
reason, when measuring the uncertainty or flatness of LLM outputs, we instead rely on permutationinvariant metrics over the probability vector, such as the cosine-based flatness measure proposed in
our paper and other quantities like entropy.


22


F.5 QUANTITATIVE ANALYSIS OF TOKEN SATURATION


To quantitatively verify the saturation behavior of tokens with different flatness levels, we conduct
a tracking experiment. Specifically, we randomly select 10 training samples and sort all constituent
tokens by their target flatness. We then split these tokens into two groups: the bottom 35% (representing low-flatness tokens) and the top 65% (representing high-flatness tokens). We track their
average gradient norm and loss values from epoch 0 to 15. The results are shown in Figure 7, from
which we can have the following three findings: (i) Low-flatness tokens exhibit consistently smaller
gradient norms compared to high-flatness tokens across all epochs. Their gradient norms quickly
decay to negligible values (below 0.1), indicating that the model ceases to learn from them early in
training. (ii) The loss for low-flatness tokens rapidly converges to near-zero and remains flat, confirming early saturation. (iii) In contrast, high-flatness tokens maintain significantly larger gradient
norms and non-trivial loss values throughout epochs 0 to 15, indicating that they continue to provide useful gradient signals for a longer period. These results quantitatively support our claim that
low-flatness tokens saturate quickly during training, contributing minimal learning signals in later
stages, whereas high-flatness tokens continue to drive the optimization process.


F.6 FURTHER ANALYSIS OF TOKEN-LEVEL FILTERING


In the initial phase of this work, we indeed investigate a token-level filtering strategy as a precursor to
our sample-level approach. However, under the current training framework, this token-level filtering
does not lead to meaningful wall-clock training speedups. The main reasons are two-fold.


    - **Forward-pass** **integrity.** To preserve the necessary autoregressive context for language
modeling, the forward pass must be computed over the full sequence. Consequently, masking specific tokens at the loss calculation stage does not obviate the need for their forward
computation, yielding no savings in this phase.


    - **Negligible** **savings** **on** **the** **backward** **pass.** The dominant computational cost in training
arises from backpropagation. While token-level loss masking (or clipping) effectively zeroes out gradients at the output layer for masked tokens, standard frameworks must still perform backpropagation through all intermediate Transformer layers over the full sequence
length. As a result, the reduction in gradient computation is marginal.


In contrast to token-level filtering, sample-level filtering can discard entire sequences and completely
skip both the forward and backward passes for those samples, which is why we ultimately adopt
the sample-level SFDD strategy. However, if future training infrastructures allow skipping masked
tokens in both forward and backward passes, we believe that flatness-based token-level filtering
would become a very promising direction for efficient draft model training.


F.7 THE NECESSITY OF EFFICIENT DRAFT MODEL TRAINING


This section is to clarify that training has become an essential and increasingly non-negligible component in modern speculative decoding (SD) pipelines. While early SD work may rely on a single
lightweight SFT step, recent advances have systematically scaled up the training required for competitive acceptance lengths and speedups. As a result, training cost has become more important in
realistic deployment scenarios, primarily due to the following two trends:


    - **Multi-step SFT.** For example, HASS (Zhang et al., 2024b) and EAGLE-3 (Li et al.) employ more complex training objectives to improve acceptance rates and speedups. These
methods are no longer a single lightweight SFT step; instead, they require multi-epoch and
heavily supervised training to maximize the achievable acceptance rates and speedups.


    - **RL-style** **training.** For example, GTO (Hu et al., 2025) introduces tree-policy mechanisms and leverages PPO-style optimization, followed by a second-stage RL refinement
of the draft model. As reported in its appendix, this refinement incurs substantial training
overheads—e.g., 200/400/900 GPU-hours of extra cost for 7B/13B/70B models.


Therefore, as SD training continues to scale up, training efficiency can no longer be treated as
negligible. SFDD is designed to be used on top of these methods: it reduces the amount of high

23


value training data required to sustain almost the same inference speedups, keeping the additional
training cost within a more reasonable budget.


F.8 ANALYSIS OF UNEXPECTED PERFORMANCE DROP AT 60% RETENTION


We note that, for SFDD, the average acceptance lengths at 50% and 60% retention are almost the
same on both MTB and NQ, and the resulting speedups only differ slightly. Because these two retention ratios use very similar amounts of training data, we view such small differences as normal
randomness from subset selection and measurement noise, rather than a statistically reliable performance drop. To investigate this, we repeat the experiments on MT-Bench and NQ two additional
times under both the 50% and 60% retention settings, and summarize the results in Table 8. We observe that, with nearly identical acceptance lengths, the speedups obtained at 50% and 60% retention
remain close across different runs, and the differences between them stay within a small magnitude.


Table 8: Repeated runs of SFDD at 50% and 60% retention on MT-Bench and NQ. The two runs
show that speedup and _l_ remain almost the same across repeated experiments.


Dataset Retention Speedup (run 1) ( _×_ ) _l_ (run 1) Speedup (run 2) ( _×_ ) _l_ (run 2)


MT-Bench 50% 2.4206 _×_ 2.5960 2.3947 _×_ 2.5960
MT-Bench 60% 2.3958 _×_ 2.5742 2.3927 _×_ 2.5742
NQ 50% 2.1323 _×_ 2.1676 2.1352 _×_ 2.1676
NQ 60% 2.1325 _×_ 2.1544 2.1345 _×_ 2.1544


G ADDITIONAL EXPERIMENTAL RESULTS


G.1 EXPERIMENTS BEYOND LLAMA3-8B-INSTRUCT AND SHAREGPT


To further demonstrate the generalization and robustness of our proposed SFDD method, we conduct additional experiments extending beyond the primary LLaMA3-8B-Instruct and ShareGPT.
First, we apply SFDD to a different model family, Vicuna-7B-v1.3. The experimental settings for
this evaluation are identical to those used in the main paper. Second, to assess robustness on a
distinct data distribution, we train the EAGLE draft model (based on LLaMA3-8B-Instruct) from
scratch using the GSM8K training split, and evaluate its performance on the test set. The results are
summarized in Table 9 and Table 10. The experimental results are basically consistent with the main
results in our paper, demonstrating the effectiveness of flatness.

|Table 9: Comparison of metrics for data importance on Vicuna-7B-v1.3 at a 50% retain ratio.|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|**GSM8K**<br>**Alpaca**<br>**MTB**<br>**CNN/DM**<br>**NQ**<br>**Average**<br>Method<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_<br>Speedup<br>_l_|
|No Filter|2.98_×_<br>3.50|2.71_×_<br>2.95|3.14_×_<br>3.03|2.60_×_<br>2.76|2.26_×_<br>2.22|2.74_×_<br>2.89|
|Random<br>Entropy<br>Top-1 Probability<br>Margin<br>Energy Score<br>PPL<br>**SFDD (Ours)**|2.83_×_<br>3.36<br>2.81_×_<br>3.36<br>2.69_×_<br>3.24<br>2.65_×_<br>3.18<br>2.78_×_<br>3.34<br>2.78_×_<br>3.35<br>**2.96**_×_<br>**3.40**|2.68_×_<br>2.83<br>2.72_×_<br>2.90<br>2.74_×_<br>2.94<br>2.66_×_<br>2.79<br>2.76_×_<br>2.85<br>2.68_×_<br>2.82<br>**2.79**_×_<br>**3.04**|2.72_×_<br>2.97<br>2.83_×_<br>2.99<br>2.84_×_<br>3.00<br>2.71_×_<br>2.87<br>2.74_×_<br>2.97<br>2.75_×_<br>2.97<br>**3.16**_×_<br>**3.09**|2.40_×_<br>2.48<br>2.49_×_<br>2.47<br>2.57_×_<br>2.64<br>2.53_×_<br>2.55<br>2.57_×_<br>2.64<br>2.52_×_<br>2.57<br>**2.63**_×_<br>**2.71**|2.21_×_<br>2.17<br>2.35_×_<br>2.31<br>2.25_×_<br>2.25<br>2.23_×_<br>2.18<br>2.27_×_<br>2.21<br>2.28_×_<br>2.23<br>**2.40**_×_<br>**2.34**|2.57_×_<br>2.76<br>2.64_×_<br>2.81<br>2.62_×_<br>2.81<br>2.56_×_<br>2.71<br>2.62_×_<br>2.80<br>2.60_×_<br>2.79<br>**2.79**_×_<br>**2.92**|


G.2 ROBUSTNESS OF SAMPLE-LEVEL AGGREGATION STRATEGIES


In our method, we utilize the arithmetic mean to aggregate token-level flatness scores. To verify robustness, we evaluate an alternative aggregation strategy: the median. We conduct experiments using
the same setup as in the main paper (LLaMA3-8B-Instruct, 50% retain ratio). As shown in Table 11,
we observe that the performance under median aggregation is very similar to the mean-aggregation
results in Table 1. These findings indicate that introducing a robust median-based aggregator does
not change our main conclusions and is consistent with our mean-based analysis.


24


Table 10: Comparison on the GSM8K training set. The results are evaluated on the test set. The
EAGLE draft model (based on LLaMA3-8B-Instruct) is trained from scratch on the GSM8K training
split at a 50% retain ratio.


Method Speedup _l_


No Filter 1.40 _×_ 1.26


Random 1.25 _×_ 1.10
Entropy 1.31 _×_ 1.18
Top-1 Probability 1.29 _×_ 1.15
Margin 1.28 _×_ 1.13
Energy Score 1.31 _×_ 1.18
PPL 1.31 _×_ 1.17
**SFDD (Ours)** **1.38** _×_ **1.24**


Table 11: Comparison of SFDD and Top-1 Probability at a 50% retain ratio under median aggregation over tokens within each sample; No Filter and Random are also included.


**GSM8K** **Alpaca** **MTB** **CNN/DM** **NQ** **Average**

|Speedup l|Speedup l|Speedup l|Speedup l|Speedup l|
|---|---|---|---|---|
|2.71_×_<br>3.28|2.71_×_<br>2.89|2.53_×_<br>2.77|2.30_×_<br>2.58|2.19_×_<br>2.37|
|2.43_×_<br>2.85<br>2.50_×_<br>2.89<br>**2.66**_×_<br>**2.94**|2.37_×_<br>2.59<br>2.45_×_<br>2.66<br>**2.69**_×_<br>**2.73**|2.26_×_<br>2.48<br>2.26_×_<br>2.52<br>**2.47**_×_<br>**2.59**|1.99_×_<br>2.31<br>2.00_×_<br>2.34<br>**2.16**_×_<br>**2.37**|1.93_×_<br>2.06<br>1.95_×_<br>2.12<br>**2.16**_×_<br>**2.18**|


H STATEMENT ON THE USE OF LARGE LANGUAGE MODELS


In this work, we use LLMs to assist with improving the clarity, grammar, and overall readability of
the text. All technical content, including theoretical claims and experimental results, is the original
work of the authors. The LLMs only serve as a writing aid, and the final manuscript has been
carefully reviewed and revised by the authors to ensure its accuracy and originality.


25