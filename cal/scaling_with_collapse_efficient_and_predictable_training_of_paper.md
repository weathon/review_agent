# SCALING WITH COLLAPSE: EFFICIENT AND PRE- DICTABLE TRAINING OF LLM FAMILIES


**Shane Bergsma** **Bin Claire Zhang** **Nolan Dey** **Shaheer Muhammad** **Gurpreet Gosal**
**Joel Hestness**
Cerebras Systems
_{_ shane.bergsma,joel _}_ @cerebras.net


ABSTRACT


Effective LLM training depends on predictable scaling of key quantities—such as
final loss and optimal hyperparameters—with model and dataset size. Qiu et al.
(2025) recently showed that this predictability can extend beyond scalars: whole
training loss curves can _collapse_ onto a universal trajectory after a simple normalization. What remains unclear is whether this phenomenon persists for LLM
families trained under _practical scaling recipes_, where width, depth, learning rate,
batch size, and weight decay are scaled jointly. We show that it does: loss curves
collapse across scales precisely when optimization hyperparameters are set optimally for the given data budget, in accordance with recent empirical scaling
laws. Collapse therefore emerges as a signature of compute-efficient training. We
demonstrate two applications at scale: (1) deviation-from-collapse provides a sensitive, early diagnostic of training pathologies, and (2) predictability of collapsed
curves enables early stopping in large-scale hyperparameter tuning. Finally, we
train a competitive LLM family, _Celerity_, using these insights, establishing collapse as an effective tool for developing efficient LLMs.


Collapse residuals can detect issues


0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8 1 _._ 0
Fraction of training tokens, _t_ [ˆ]


|Celerity|curves collapse; TPP & τ fixe|Col3|Col4|
|---|---|---|---|
||Celerity|Models||
||<br>300M: 234 <br>500M: 234 <br>|<br> TPP, _τ_=0.0<br> TPP, _τ_=0.0<br>|5<br>5<br>|
||~~900M: 234 ~~<br>1.8B: 234 <br>3.9B: 234|~~TPP,~~ ~~_τ_=0.0~~<br>TPP, _τ_=0.05<br>TPP, _τ_=0.05|~~5~~|
|||||
|||||
|||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


0 _._ 030


0 _._ 025


0 _._ 020


0 _._ 015


0 _._ 010


0 _._ 005


0 _._ 000


_−_ 0 _._ 005


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


|Llam|ma-2 train los|ss; TPP &|& τ vary|Col5|
|---|---|---|---|---|
||Ll|ama-2 M|odels||
||7B: <br>13B<br>|<br> 286 TPP<br>: 154 TP<br>|, _τ_=0.07<br>P, _τ_=0.07<br>||
||~~34B~~<br>70B|~~: 59 TPP~~<br>: 29 TPP|~~_τ_=0.13~~<br> _τ_=0.13||
||~~34B~~<br>70B||||
||||||
||||||
||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


Figure 1: _Left_ : Prior LLM families like Llama-2 train at varying tokens-per-parameter (TPP; _D/N_ )
and AdamW timescale _τ_ ; training loss curves do _not_ collapse. _Middle_ : Fixing TPP and setting
_τ_ optimally for that TPP, Celerity loss curves _do_ collapse. _Right_ : Deviations from collapse allow
precise identification (and earlier repair) of numerics issues in large-scale training runs.


1 INTRODUCTION


Scaling up pre-training has emerged as the primary route to improving LLM performance (Brown
et al., 2020; Achiam et al., 2023). Yet once we reach frontier scales, opportunities for direct experimentation disappear (Xiao, 2024). How then can we train effectively at those scales—what size
of model should we use, and how should we set hyperparameters? Encouragingly, recent work
has revealed that several quantities are remarkably _predictable_ as we scale deep learning. These
include model performance as a function of model and dataset size (Hestness et al., 2017; Kaplan
et al., 2020), as well as hyperparameters under maximal update parameterization ( _µ_ P), which enables optimal base learning rates and initializations to approximately transfer across widths (Yang
et al., 2021). In this paper we build on this trajectory of predictability: we show that, at LLM scale,


1


training loss curves (TLCs) from different model sizes _collapse_ onto a single universal curve after a
simple normalization—provided models are trained with a particular hyperparameter-scaling recipe.


Qiu et al. (2025) only recently demonstrated this striking regularity in TLCs, showing collapse when
training with _µ_ P on small-scale autoregressive tasks. As their validation was limited to small models
trained with vanilla Adam (Kingma & Ba, 2014), without weight decay, they explicitly call for tests
at larger scales with _practical_ _scaling_ _ladders_ that co-scale width, depth, batch size, and weight
decay. Our work addresses this gap, showing that collapse persists in full-scale LLM families.


While modeling LLM loss is an active research topic (Sec. 6), the ability to predict TLCs has great
_practical_ value. For example, human judgment is now required to decide whether training has
_recovered_ from a loss spike—or whether rewinding/restarting is needed (Chowdhery et al., 2022;
Zhang et al., 2022a). Other subjective signals, such as a gradual upward _trend_ (Zhang et al., 2022b),
can also trigger interventions. Yet criteria remain vague: Touvron et al. (2023b) report Llama2 TLCs “did not show any sign of saturation,” but how to recognize saturation is unclear. If TLCs
collapse across sizes, practitioners can compare in-progress training to a universal reference, monitor
residuals, and extrapolate final loss from partial trajectories. Teams already rely on TLCs in this way,
often without a principled account of what governs TLC shape; for example, Falcon’s final LR was
chosen by simply continuing the run performing best after warmup (Almazrouei et al., 2023).


In this paper we show that the essential condition for collapse under _µ_ P is that the LR schedule,
tokens-per-parameter ratio (TPP), and AdamW timescale _τ_ (Wang & Aitchison, 2024) are held fixed
across model sizes. This reflects a deeper regularity: prior work showed that optimal _τ_ depends only
on TPP (Bergsma et al., 2025a). Thus, scaling across fixed TPP with _τ_ chosen optimally guarantees
collapse, and collapse emerges as a robust marker of compute-efficient and stable pre-training. When
_τ_ is mis-scaled—as in the Llama-2 family (Fig. 1, _left_ )—normalized curves fail to align.


We introduce _Celerity_ as the first LLM
family trained with both optimal _τ_ scaling
and demonstrable TLC collapse (Fig. 1,
_middle_ ). Effective parameterization, including tuning and transferring _τ_, helped
Celerity land on the compute-efficiency
frontier for open models of its scale
(Fig. 2). Meanwhile, deviations from collapse provided a sensitive diagnostic of
training issues: in our 1.8B run, a numerical instability became evident from
collapse residuals (Fig. 1, _right_ ) well before the raw TLC showed an upward trend
(Fig. 6, _right_ ). Celerity exemplifies _scal-_
_ing_ _with_ _collapse_ : efficient, predictable
training, across scales and throughout the
run.


In summary, our main contributions are:


|70.0 Gemma2-2B<br>better Llama2-7B<br>Llama-7B<br>67.5 SmolLM2-1.7B<br>Celerity-3.9B<br>OLMo-7B<br>65.0<br>Zamba2-1.2B<br>SmolLM-1.7B<br>62.5 OLMo2-1B<br>Celerity-1.8B<br>BTLM-3B<br>60.0 Gemma3-1B<br>57.5 SmolLM2-360M OLMo-1B<br>CerebrasGPT-13B<br>CerebrasGPT-6.7B<br>55.0 SmolLM-360M<br>Celerity Fit: 100 ( C ) 0.097<br>Celerity-900M 2.154E38<br>52.5<br>1021 1022 1023<br>Training Compute FLOPs (C)|Col2|Gemma2-2B|Col4|
|---|---|---|---|
|10~~21~~<br>10~~22~~<br>10~~23~~<br>Training Compute FLOPs (C)<br>52.5<br>55.0<br>57.5<br>60.0<br>62.5<br>65.0<br>67.5<br>70.0<br> <br>Celerity-900M<br>~~Celerity-1.8B~~<br>Celerity-3.9B<br>~~OLMo-1B~~<br>OLMo-7B<br>~~OLMo2-1B~~<br>~~Gemma3-1B~~<br>~~Gemma2-2B~~<br>Zamba2-1.2B<br>Llama-7B<br>~~Llama2-7B~~<br>CerebrasGPT-6.7B<br>CerebrasGPT-13B<br>BTLM-3B<br>~~SmolLM2-360M~~<br>~~SmolLM2-1.7B~~<br>~~SmolLM-360M~~<br>SmolLM-1.7B<br>Celerity Fit: 100<br>(<br>C<br>~~2.154~~E~~38 ~~)<br>0.097<br>better|better|Llama-7B|~~Llama2-7B~~<br>|
|10~~21~~<br>10~~22~~<br>10~~23~~<br>Training Compute FLOPs (C)<br>52.5<br>55.0<br>57.5<br>60.0<br>62.5<br>65.0<br>67.5<br>70.0<br> <br>Celerity-900M<br>~~Celerity-1.8B~~<br>Celerity-3.9B<br>~~OLMo-1B~~<br>OLMo-7B<br>~~OLMo2-1B~~<br>~~Gemma3-1B~~<br>~~Gemma2-2B~~<br>Zamba2-1.2B<br>Llama-7B<br>~~Llama2-7B~~<br>CerebrasGPT-6.7B<br>CerebrasGPT-13B<br>BTLM-3B<br>~~SmolLM2-360M~~<br>~~SmolLM2-1.7B~~<br>~~SmolLM-360M~~<br>SmolLM-1.7B<br>Celerity Fit: 100<br>(<br>C<br>~~2.154~~E~~38 ~~)<br>0.097<br>better|Ce|lerity-3.9B|OLMo-7B<br><br>~~SmolLM2-1.7B~~|
|10~~21~~<br>10~~22~~<br>10~~23~~<br>Training Compute FLOPs (C)<br>52.5<br>55.0<br>57.5<br>60.0<br>62.5<br>65.0<br>67.5<br>70.0<br> <br>Celerity-900M<br>~~Celerity-1.8B~~<br>Celerity-3.9B<br>~~OLMo-1B~~<br>OLMo-7B<br>~~OLMo2-1B~~<br>~~Gemma3-1B~~<br>~~Gemma2-2B~~<br>Zamba2-1.2B<br>Llama-7B<br>~~Llama2-7B~~<br>CerebrasGPT-6.7B<br>CerebrasGPT-13B<br>BTLM-3B<br>~~SmolLM2-360M~~<br>~~SmolLM2-1.7B~~<br>~~SmolLM-360M~~<br>SmolLM-1.7B<br>Celerity Fit: 100<br>(<br>C<br>~~2.154~~E~~38 ~~)<br>0.097<br>better||~~OLMo2-1B~~<br>SmolLM-1.7B|Zamba2-1.2B|
|10~~21~~<br>10~~22~~<br>10~~23~~<br>Training Compute FLOPs (C)<br>52.5<br>55.0<br>57.5<br>60.0<br>62.5<br>65.0<br>67.5<br>70.0<br> <br>Celerity-900M<br>~~Celerity-1.8B~~<br>Celerity-3.9B<br>~~OLMo-1B~~<br>OLMo-7B<br>~~OLMo2-1B~~<br>~~Gemma3-1B~~<br>~~Gemma2-2B~~<br>Zamba2-1.2B<br>Llama-7B<br>~~Llama2-7B~~<br>CerebrasGPT-6.7B<br>CerebrasGPT-13B<br>BTLM-3B<br>~~SmolLM2-360M~~<br>~~SmolLM2-1.7B~~<br>~~SmolLM-360M~~<br>SmolLM-1.7B<br>Celerity Fit: 100<br>(<br>C<br>~~2.154~~E~~38 ~~)<br>0.097<br>better|~~Celerity-1.8B~~|~~Gemma3~~<br>BTLM-3B|~~1B~~|
|10~~21~~<br>10~~22~~<br>10~~23~~<br>Training Compute FLOPs (C)<br>52.5<br>55.0<br>57.5<br>60.0<br>62.5<br>65.0<br>67.5<br>70.0<br> <br>Celerity-900M<br>~~Celerity-1.8B~~<br>Celerity-3.9B<br>~~OLMo-1B~~<br>OLMo-7B<br>~~OLMo2-1B~~<br>~~Gemma3-1B~~<br>~~Gemma2-2B~~<br>Zamba2-1.2B<br>Llama-7B<br>~~Llama2-7B~~<br>CerebrasGPT-6.7B<br>CerebrasGPT-13B<br>BTLM-3B<br>~~SmolLM2-360M~~<br>~~SmolLM2-1.7B~~<br>~~SmolLM-360M~~<br>SmolLM-1.7B<br>Celerity Fit: 100<br>(<br>C<br>~~2.154~~E~~38 ~~)<br>0.097<br>better|~~SmolLM2-~~|~~OLMo-1B~~<br><br>~~60M~~||
|10~~21~~<br>10~~22~~<br>10~~23~~<br>Training Compute FLOPs (C)<br>52.5<br>55.0<br>57.5<br>60.0<br>62.5<br>65.0<br>67.5<br>70.0<br> <br>Celerity-900M<br>~~Celerity-1.8B~~<br>Celerity-3.9B<br>~~OLMo-1B~~<br>OLMo-7B<br>~~OLMo2-1B~~<br>~~Gemma3-1B~~<br>~~Gemma2-2B~~<br>Zamba2-1.2B<br>Llama-7B<br>~~Llama2-7B~~<br>CerebrasGPT-6.7B<br>CerebrasGPT-13B<br>BTLM-3B<br>~~SmolLM2-360M~~<br>~~SmolLM2-1.7B~~<br>~~SmolLM-360M~~<br>SmolLM-1.7B<br>Celerity Fit: 100<br>(<br>C<br>~~2.154~~E~~38 ~~)<br>0.097<br>better|Cer<br><br>~~SmolLM-360M~~|ebrasGPT-6.7B<br>CerebrasGPT-<br>|13B|
|10~~21~~<br>10~~22~~<br>10~~23~~<br>Training Compute FLOPs (C)<br>52.5<br>55.0<br>57.5<br>60.0<br>62.5<br>65.0<br>67.5<br>70.0<br> <br>Celerity-900M<br>~~Celerity-1.8B~~<br>Celerity-3.9B<br>~~OLMo-1B~~<br>OLMo-7B<br>~~OLMo2-1B~~<br>~~Gemma3-1B~~<br>~~Gemma2-2B~~<br>Zamba2-1.2B<br>Llama-7B<br>~~Llama2-7B~~<br>CerebrasGPT-6.7B<br>CerebrasGPT-13B<br>BTLM-3B<br>~~SmolLM2-360M~~<br>~~SmolLM2-1.7B~~<br>~~SmolLM-360M~~<br>SmolLM-1.7B<br>Celerity Fit: 100<br>(<br>C<br>~~2.154~~E~~38 ~~)<br>0.097<br>better|Celerity-900M<br>|Celerity Fit: 100|(<br>C<br>~~2.154~~E~~38 ~~)<br>0.097|


Figure 2: Celerity is at the compute-efficiency frontier. (Average accuracy on tasks _arc-c,_ _arc-e,_ _boolq,_
_hellaswag, piqa, siqa, winogrande_ ; see Table 10.)


 - Identifying the key factors influencing loss curve shape under _µ_ P: the LR schedule, the TPP ratio,
and the AdamW timescale _τ_, and explaining shape dependence on these quantities (Sec. 3).

 - Demonstrating that when _τ_ is set optimally for a given TPP, TLCs _collapse_ across model scales,
providing a signature of compute-efficient training (Sec. 3).

 - Introducing the _Celerity_ family, the first large-scale LLMs trained in a collapse regime (Sec. 4).

 - Proposing a simple functional form for normalized TLCs, and showing that fitting this form on
small-scale training runs enables early stopping in large-scale hyperparameter tuning (Sec. 5).


2 BACKGROUND


Training loss curves (TLCs) for different model sizes typically differ in scale, duration and final
loss. Yet after a simple normalization, they can align closely, or _collapse_, onto a common trajectory.
Collapse emerges only when three controls are matched across scales. The tokens-per-parameter


2


ratio (TPP) determines how much data each parameter sees and thus affects the normalized pace of
improvement; the AdamW timescale _τ_ governs how long the optimizer “remembers” past gradients,
shaping the bias–variance trade-off over training; and the learning-rate schedule phases early bias
reduction against late variance suppression. When TPP and _τ_ are chosen consistently across model
sizes and the LR schedule is fixed, the resulting normalized TLCs become approximately scaleinvariant. We now formalize these quantities; Table 1 summarizes key symbols.


**TPP.** The TPP ratio is equal to number of training tokens _D_ divided by the model size _N_ . This simple quantity plays a surprisingly profound role in compute-efficient LLM training and TLC shape.
Hoffmann et al. (2022) investigated, for a given compute budget _C_, how to allocate _D_ and _N_ in
order to minimize loss. They found optimal _D_ and _N_ scale roughly equally as _C_ increases, with
the optimal _D/N_ ratio relatively constant at around 20 TPP (Appendix C.1). Replication studies
have found similar results (Besiroglu et al., 2024; Porian et al., 2024), and 20 TPP has emerged as a
rule-of-thumb for compute-optimal training (Dey et al., 2023a; Zhang et al., 2024b).


_µ_ **P.** _µ_ P (Yang & Hu, 2020) and related parameterizations for depth (Bordelon et al., 2023; Yang
et al., 2023; Dey et al., 2025) seek to achieve consistent, stable training dynamics as networks
scale up. Moreover, with _µ_ P, base hyperparameters can be tuned on a small _proxy_ model and then
transferred to larger scales. Given the width of the proxy model, _dp_, and target, _dt_, _µ_ P prescribes
scaling factors to apply to the base LR, initial weight variance, and other base HPs.


_µ_ P is increasingly used in LLM training (Dey et al., 2023a;b; Sengupta et al., 2023; Shen et al.,
2024; Hu et al., 2024). Moreover, recent work has shown that, when using _µ_ P, other important
aspects of training may _decouple_ from model size, including optimal batch size (scaling primarily
in the total number of tokens (Zhang et al., 2024b; Bergsma et al., 2025a)), and optimal AdamW
timescale/weight decay (scaling primarily in TPP (Bergsma et al., 2025a)).


**Supercollapse.** Using _µ_ P, Qiu et al. (2025) observed that TLCs for different model sizes, despite
varying widely over compute and absolute loss, appear to follow a consistent shape. This motivated
them to affinely rescale the curves to the normalized loss _ℓ_ given by:

_ℓ_ ( _t, N, ω_ [ˆ] ) = ( _L_ ( _t_ [ˆ] _· T_ _[⋆]_ ( _N_ ) _, N, ω_ ) _−_ _L_ [ˆ] ) _/_ ( _L_ ( _T_ _[⋆]_ ( _N_ ) _, N, ω_ ) _−_ _L_ [ˆ] ) (1)

where _ω_ is the random seed, _t_ [ˆ] is the fraction of training completed (what Qiu et al. (2025) refer to
as _normalized_ _compute_ ), _N_ is the number of model parameters, and _T_ _[⋆]_ ( _N_ ) is the corresponding
compute-optimal number of training steps, estimated from a power law fit. _L_ [ˆ] is an offset, which
they subsequently set to the estimated irreducible loss of their power law.


Training compute-optimally under _µ_ P (on small-scale autoregressive tasks, e.g., predicting chess
moves), Qiu et al. (2025) showed TLCs collapse under this normalization—indeed, they _su-_
_per_ collapse, meaning they differ by less than the noise from inter-run variation. They further show
that collapse arises naturally in constant-learning-rate models where loss obeys typical neural power
laws, while extending the theory to arbitrary LR schedules via a theoretical model of quadratic loss.


**The** **AdamW** **EMA** **and** **its** **timescale.** AdamW updates at step _t_ can be expressed in terms of
learning rate _η_ and weight decay _λ_ as: _θt_ = (1 _−_ _ηλ_ ) _θt−_ 1 _−_ _η_ ~~_√_~~ _mv_ ˆˆ _tt_ + _ϵ_, where _m_ ˆ _t_ and _v_ ˆ _t_ are biascorrected EMAs of gradients and squared gradients, respectively (Kingma & Ba, 2014). Wang &
Aitchison (2024) observed that AdamW parameters _θt_ can also be viewed as an EMA—of weight
_updates_ . That is, the standard EMA form _yt_ = (1 _α_ ) _yt_ 1 + _αxt_ matches AdamW when _yt_ = _θt_,
_α_ = _ηλ_, and _xt_ = _λ_ ~~_√_~~ _mv_ ˆˆ _tt_ + _ϵ_ . The _timescale τ_ iter _−_ = [1] _/α_ = _−_ [1] _/ηλ_ represents the approximate number
_−_ [1]

of iterations over which updates are averaged. When expressed in epochs as _τ_ epoch = _τ_ iter _/M_,
where _M_ is the number of iterations per epoch, Wang & Aitchison (2024) found the optimal _τ_ epoch
(swept by varying _λ_ ) _remains stable_ under model and dataset scaling on image tasks.


Since LLM pre-training typically uses a single epoch, we follow Bergsma et al. (2025a) in defining a
normalized timescale _τ_ = _τ_ iter _/T_, where _T_ is the total number of optimization steps. As _T_ = _D/B_
(total tokens/batch size):
_τ_ = 1 _/_ ( _ηλT_ ) = _B/_ ( _ηλD_ ) _._ (2)
In contrast with the results in Wang & Aitchison (2024), Bergsma et al. (2025a) did _not_ find optimal
_τ_ to remain stable in LLM training, but instead to decrease as a (scale-invariant) power law in TPP.


3


Table 1: Core quantities used throughout the paper.


Symbol Meaning


_N_ Number of model parameters
_D_ Total number of training tokens
_B_ Batch size (tokens per optimization step)
_T_ = _D/B_ Total number of optimization steps
_t_ ˆ = _t/T_ Fraction of training completed
TPP = _D/N_ Tokens-per-parameter ratio (TPP)


_L_ ( _t_ ) Training loss at step _t_
_ℓ_ ( _t_ [ˆ] ) Normalized training loss curve (TLC)


_η_ Learning rate
_λ_ Weight decay coefficient
_τ_ iter = 1 _/_ ( _ηλ_ ) AdamW timescale (in steps)
_τ_ = _τ_ iter _/T_ = 1 _/_ ( _ηλT_ ) = _B/_ ( _ηλD_ ) Normalized AdamW timescale


3 WHAT FACTORS MODULATE TRAINING CURVE SHAPE?


**Experimental** **setup.** We use a GPT2-like LLM (Radford et al., 2019), with ALiBi embeddings (Press et al., 2022) and SwiGLU (Shazeer, 2020). We train on SlimPajama (Soboleva et al.,
2023). Models are trained with AdamW and _µ_ P. We use a linear decay-to-zero LR schedule, context
length of 2048, and the GPT2 vocabulary. Full architecture and other details are in Appendix B.1.

We plot _ℓ_ vs. _training_ _fraction_ _t_ [ˆ] = _t/T_ = _tB/D_, with step count _t_, total steps _T_, batch size _B_,
and dataset size _D_ . To reduce noise in small- _B_ settings, we _post hoc_ aggregate losses _ℓ_ ( _t_ [ˆ] ) using a
moving-average filter over a window of 100 steps, smoothing curves without altering the underlying
trajectory. We also consistently found simply _dividing_ _by_ _the_ _final_ _training_ _loss_ (i.e., _L_ [ˆ] = 0 in
Eq. (1)) resulted in optimal alignment across scales, so use this for all curves.


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


|Col1|B varies η=1.62e-02 λ=0.1|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||_B_|(_τ_)||
|||||126<br>252<br>|<br> (0.026)<br> (0.053)<br>||
|||||504<br>100<br>|(0.105)<br>8 (0.210)<br>||
||||~~20~~|~~20~~|~~6 (0.421~~||
||||~~20~~|~~20~~|||
||||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


|Col1|B=504 η varies λ=0.1|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||||_η_ (|_τ_)||
||||<br>4.05e-0<br>8.09e-0<br>|3 (0.421)<br>3 (0.210)<br>||
|||||||
|||||||
||||1.62e-0<br>3.24e-0<br>|2 (0.105)<br>2 (0.053)<br>||
|||||||
||||~~6.48e-~~|~~2 (0.026~~||
|||||||
|||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


|Col1|B=504 η=1.62e-02 λ varies|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||_λ_|(_τ_)||
|||||0.02<br>0.05<br>|<br>5 (0.421)<br> (0.210)<br>||
|||||0.1 <br>0.2 <br>|(0.105)<br>(0.053)<br>||
||||~~0.4 ~~|~~0.4 ~~|~~(0.026)~~||
||||~~0.4 ~~|~~0.4 ~~|||
||||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


Figure 3: **AdamW** **timescale** _τ_ **modulates** **TLC** **shape** **(610M,** **80TPP)** : Sweeping _η_ ( _left_ ), _λ_
( _middle_ ), or _B_ ( _right_ ) produces matching variations in normalized TLCs when _τ_ varies identically.


**Finding:** _τ_ **modulates** **TLC** **shape.** Fig. 3 shows normalized TLCs for 610M models trained to
80 TPP, sweeping either learning rate _η_, weight decay _λ_, or batch size _B_ in each subplot. Across
hyperparameters, TLCs with matching _τ_ exhibit very similar shapes, reflecting consistent timescale
control. Similar patterns hold across other scales and dataset sizes. Generally, as _τ_ increases, TLCs
drop more early and less later. This is also a function of the LR schedule: when we switch to using
a _Constant_ LR, there is no final drop, lower- _τ_ TLCs are lower throughout (appendix Fig. 10).


**Finding:** **TPP modulates TLC shape.** We now fix _τ_ and test increasing _TPP_, finding TLCs drop
earlier and flatten for longer (Fig. 4, _left_, _middle_ ; see also Llama-2 for _τ_ = 0 _._ 07 and _τ_ = 0 _._ 13 in
Fig. 1, _left_ ). Intuitively, relative to length of training, higher TPP drops loss more at the beginning
and then obtains diminishing returns later on. In Fig. 4, _right_, TLC shape is quite similar across
model scales at the same TPP (when _τ_ is roughly equal), showing TPP’s shaping effect is _scale-_
_invariant_ (scaling from 111M to 3.3B at fixed TPP represents a 1000 _×_ increase in training FLOPs).


4


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


|Col1|Effect of TPP: 111M, τ=0.021|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||TPP||
|||||20<br>80<br>||
|||||~~320~~<br>1280||
|||||~~320~~<br>1280||
|||||||
|||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


|E|ffect of TPP: 610M, τ=0.105|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||TPP||
|||||20<br>80<br>||
||||~~32~~|~~32~~|~~0~~|
||||~~32~~|~~32~~||
|||||||
|||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


|Cur|ves align when {TPP, τ} ≈const.|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||<br>_N_||
||||111M<br>266M<br>|, _τ_=0.33<br>, _τ_=0.27<br>||
||||~~610~~<br>1.7B<br>3.3B|~~,~~ ~~_τ_=0.2~~<br>, _τ_=0.30<br>, _τ_=0.31||
|||||||
|||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


Figure 4: **TPP modulates TLC shape.** Fixing _τ_ for 111M ( _left_ ) & 610M ( _middle_ ) while _increasing_
TPP, curves shift _down_ . When _τ_ _≈_ const. and TPP also fixed (at 20), curves roughly collapse ( _right_ ).


**Using bias and variance to explain the effect of** _τ_ **.** Classical analyses of stochastic optimization
(e.g., Bottou et al. (2018); D’Angelo et al. (2024)) decompose loss into two components: a _bias_
term, reflecting dependence on initial weights, and a _variance_ term, reflecting noise from stochastic
gradients. Early in training bias dominates; later, variance determines the attainable loss floor.


In our setting, the AdamW timescale _τ_ controls this trade-off through the EMA over weight updates
(Sec. 2). A smaller _τ_ corresponds to a shorter memory: updates depend mainly on recent gradients,
yielding rapid bias reduction but a higher variance floor. A larger _τ_ averages over more past gradients, reducing variance more effectively but slowing early progress. Empirically, TLCs reflect this
pacing, with smaller _τ_ producing faster early descent (e.g., left panel of appendix Fig. 10).


This intuition is formalized in Appendix B.3 via a noisy quadratic model:


E[ _L_ ( _t_ [ˆ] )] = _[h σ]_ _x_ [2]
4 _τ_


1 _e_ _[−]_ [2ˆ] _[t/τ]_ [�] + _[h]_ (3)
_−_ 2 _[e][−]_ [2ˆ] _[t/τ]_ [ E][[] _[θ]_ [(0)][2][]] _[,]_


1 _e_ _[−]_ [2ˆ] _[t/τ]_ [�] + _[h]_
_−_ 2


where the first term approaches a variance floor _∝_ 1 _/τ_ and the second is an exponentially decaying
bias term. Smaller _τ_ thus yields faster initial decay but a higher floor, while larger _τ_ yields slower
initial decay but a lower floor, matching the observed _fast-then-flatten_ behavior under constant LR.
With LR decay, _ηtλ_ decreases and the instantaneous timescale _τt_ = 1 _/_ ( _ηtλT_ ) _increases_, enhancing
late-stage variance suppression and steepening final drop (e.g., Fig. 10: more LR decay, more drop).


_Scale invariance._ After normalizing by the final loss, the curvature factor _h_ cancels (Appendix B.3).
Provided residual bias at end-of-training is negligible relative to the variance floor, the normalized
TLC depends only on _τ_ and _t_ [ˆ] . Thus, at matched _τ_, normalized TLCs collapse across scales.


**Explaining** **effect** **of** **TPP.** TPP affects TLCs via power laws. With a _constant_ LR, every step is
the endpoint of a shorter run, trained to _t_ [ˆ] _·_ TPP. Qiu et al. (2025) note _L_ ( _t_ [ˆ] ) therefore _follows the_
_same_ _power_ _law_ _as_ _final_ _loss_ _of_ _a_ _run_ _fully_ _trained_ _to_ _that_ _effective_ _budget_ . Normalizing by the
projected loss at _total_ TPP removes dependence on model and dataset size, so _ℓ_ ( _t_ [ˆ] ) depends only on
_t_ ˆ and total TPP (Appendix B.2). Higher-TPP curves analytically decay faster and level off sooner.
LR schedules deform the curves, but deformation is also scale invariant given consistent curvature
of the loss landscape across model sizes under _µ_ P (Noci et al., 2024).


4 CELERITY: A COMPUTE-EFFICIENT MODEL FAMILY WITH COLLAPSE


We have established that collapse arises when _τ_ and TPP are held fixed across model sizes. Meanwhile, prior work has shown optimal _τ_ to depend only on TPP (Bergsma et al., 2025a). Here, we
introduce a model family, _Celerity_, trained at fixed TPP and with _τ_ chosen optimally for that TPP,
i.e., a regime where collapse emerges naturally as a consequence of good training.


5


**Compute** **vs.** **parameter** **efficiency.** A key question for Celerity is which TPP to use: _≈_ 20 is
compute-optimal (Sec. 2), while higher TPP means greater _parameter efficiency_ (fewer parameters
to obtain same loss). For small, inference-ready models, parameter efficiency is paramount, but
such models are usually distilled (Tunstall et al., 2023; Wang et al., 2025) rather than pre-trained at
high-TPP. _Our main interest is developing pre-training strategies for very large models_ . As models
scale, the relative importance of compute-efficiency increases—indeed, public families often have


Yet even for the largest models, parameter efficiency remains valuable, e.g., when generating distillation logits or
synthetic data. To choose Celerity’s TPP, we analyze this
trade-off: Appendix C.1 derives an expression for the extra
compute required to compress a model to a fraction of the
_size when training compute optimally_ —while maintaining
the same loss. This expression leverages power law fits
from prior work. Fig. 5 plots the trade-off, where a TPP
ratio of 234 is estimated to achieve a 62% reduction in parameters with only a 67% increase in total FLOPs (relative
to 20 TPP). This is a responsible balance point, near what
has been called the _critical_ _model_ _size_ —the point where
further increasing compute obtains massively-diminishing
returns in parameter efficiency (De Vries, 2023) (e.g., doubling _our_ FLOPs, to 3.34 _×_ compute-optimal, reduces _N_
by only a further _≈_ 11%).


8


6


4


2


0


|n et al., 2023).|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|234 TPP: high compression/low cost|234 TPP: high compression/low cost|234 TPP: high compression/low cost|234 TPP: high compression/low cost|234 TPP: high compression/low cost|
||||||
||||bett|er|
|||TPP=20<br>(_kN_=1.0, _C_|_/C_opt=1.0)||
||TPP=<br>|234<br>|||
||(_kN_=|.38, _C/C_opt|1.67)||
||||||


0 1 2 3
Compression: Fraction of _N_ opt ( _kN_ )


TPP:
_D/N_


10 [3]


10 [2]


10 [1]


Figure 5: Expected iso-loss compute
vs. compress trade-off as TPP varies.


Even if the ultimate goal is a “herd” of models at varying TPP, such as Llama-2 in Fig. 1, _left_, there
are advantages to training different “bands” within the herd, e.g., 7B, 13B, 34B, 70B _all_ at 29 TPP:


 - _Tuning_ : you can fine-tune _τ_ at a smaller scale and zero-shot transfer to larger models.


 - _Diagnostics_ : Because TLCs collapse, deviations provide an early warning of training issues.

 - _Cost_ : Fixed-TPP bands are cheap (e.g., 10 _×_ lower _N_ _→_ 10 _×_ smaller _D →_ 100 _×_ less compute).


**Philosophy.** Celerity aims to advance general LLM capabilities using public pre-training corpora
and fully-open, consistent methods—rather than targeting specific benchmarks. In contrast, the
majority of LLMs now _anneal on training subsets of downstream benchmarks_ (Dubey et al., 2024;
Achiam et al., 2023), or inject special high-quality math (OLMo et al., 2024), code (Zhang et al.,
2024a), or instruction (Hu et al., 2024) data during a late-stage _mid-training_ process. Since these
practices make evaluation problematic (Dominguez-Olmedo et al., 2024), Celerity can serve as a
comparison for models trained without (or prior to) applying such techniques.


**Experimental** **details.** Celerity pretrains in bands of 20, 80, and 234 TPP,
each spanning 300M–3.9B models (Table 2); see Appendix C.2 for further details. Key enablers of Celerity’s reliable,
efficient training include:


 - _Data_ : emphasizing (open) educational, math, and coding data throughout training (appendix Table 6); this
outperformed training on the general
SlimPajama dataset (Table 7).


 - _Parameterization_ : Using CompleteP,
which enables hyperparameter transfer over width _and_ depth, was more
efficient/reliable than _µ_ P (Fig. 15).


 - _Optimization_ : LR, _τ_, batch size tuned
small, transferred via scaling rules.


Table 2: Architecture of Celerity


Celerity: 300M 500M 900M 1.8B 3.9B


Hidden Dim 640 896 1152 1536 2048
Num Heads 10 14 9 12 16
Head Size 64 64 128 128 128
Layers 13 17 23 30 40


Batch Size 176 240 336 464 672


Vocabulary Llama-3 (size 128256)
Embeddings ALiBi, Untied
Seq Length 8192


Non-linearity Squared ReLU
FFN Mult 8 _×_
Norm Type Pre-Layer Normalization, _ϵ_ = 10 _[−]_ [5]


LR Schedule Peak: 0.15, linear decay-to-zero
LR warmup min(10% of total tokens, 375M tokens)


6


**Evaluation** **results.** Appendix Table 10 provides full downstream evaluation results for Celerity
and other public models tested on seven common downstream tasks. Fig. 2 shows that Celerity models form the accuracy/compute Pareto frontier up to our largest training budget. Against BTLM (Dey
et al., 2023b)—trained before task-specific data annealing became standard—Celerity achieves comparable accuracy with 75% fewer training FLOPs. Extrapolation via a fitted power law in compute
(dashed line in plot) suggests smooth scaling and continued competitiveness. For comparison with
distilled models, we count only _student_ FLOPs in Fig. 2. Including _teacher_ FLOPs (forward passes),
or the cost of teacher training, strengthens Celerity further (appendix Fig. 16).


In terms of parameter efficiency, Celerity is weaker than high-TPP families (Figs. 19 and 20), meaning such models save FLOPs at inference. However, beyond the importance of studying compute
efficiency for hyper-scale training, there is strong motivation to train and study compute-efficient
smaller models: growing evidence suggests some models may be counter-productively (even _catas-_
_trophically_ ) overtrained, making them harder to fine-tune (Springer et al., 2025) and quantize (Kumar et al., 2024). Compute-efficient alternatives therefore serve both as a principled baseline for
understanding scaling and as a practical fallback when high-TPP models prove brittle.


**Collapse results.** In Sec. 3, we normalized training loss curves by dividing by the final loss value,
_L_ ( _T_ ) (Eq. (1)). To use collapse as a diagnostic _during_ training, we need a way to normalize when
_L_ ( _T_ ) is still unknown. We explored two strategies and use early-align in our experiments:


 - _Estimate_ : extrapolate _L_ ( _T_ ) from a power law fit at lower scales.


 - _Early-align_ : choose _L_ ( _T_ ) so _ℓ_ ( _t_ ) best aligns with the smallest-scale curve over 25-50% portion.


|1.8B (initial) curve|with apparent blip|
|---|---|
|_N_ (234 <br>|TPP, _τ_=0.051)<br>|
|~~1.8B (i~~<br>1.8B (i<br>|~~itial)~~<br>nitial, unsmoothed)<br>|
|~~1.8B (r~~|~~epaired)~~|
|~~1.8B (r~~||
|||
|||
|||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


3 _._ 0


2 _._ 8


2 _._ 6


2 _._ 4


2 _._ 2


2 _._ 0


|Celerity train l|oss collapse: 20 TPP|
|---|---|
||_N_ (_τ_=0.175)|
||<br>300M<br>500M<br><br>1.8B<br>3.9B|
||~~900M~~|
|||
|||


|Celerity trai|n loss collapse: 80 TPP|
|---|---|
||_N_ (_τ_=0.087)|
||<br>300M<br>500M<br>900M<br>1.8B|
|||
|||
|||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


Figure 6: **Collapse in Celerity models.** Celerity 20 TPP ( _left_ ) and 80 TPP ( _middle_ ) models exhibit
collapse. _Right_ : smoothing helps detect blip in loss near the end of initial 1.8B run (red curve), but
divergence can be detected much earlier using collapse residuals (Fig. 1, _right_ ).


Fig. 6 shows normalized curves. Collapse is tight at 80 TPP ( _middle_ ). At 20 TPP ( _left_ ), we see small
early deviations, which we attribute to differing LR warmup proportions (Table 2). At 234 TPP,
divergences appear late in training for larger models (Fig. 1, _middle_ ). Investigating, we find loss
improves disproportionately on training data, while held-out data remains aligned with projections.


**Collapse for monitoring.** Fig. 6 ( _right_ ) shows the unnormalized TLC for our original 1.8B, 234
TPP run. Smoothing helps reveal a sudden rise in training loss, but only after 90% of training.
Without a collapse reference, it would be impossible to see that problems began much earlier. By
comparing against the 500M TLC reference (Fig. 1, _right_ ), we pinpoint divergence starting near
60%. Knowing this timing was crucial: we did not waste effort investigating late-stage data redundancy, and instead realized the problem coincided with a job restart under a new compute allocation.


The collapse reference was also essential for debugging: by running ablations with different batch
sizes and measuring divergence from the reference, we confirmed the anomaly arose from a numerical issue in a loss kernel triggered only at specific microbatch sizes. After fixing the kernel and
restarting from before the divergence, training tracked the reference TLC closely (Fig. 1).


7


5 COLLAPSE ENABLES EARLY STOPPING IN HYPERPARAMETER TUNING


Training to completion is expensive. If normalized TLCs behave predictably, can we stop earlier
and still recover the final loss? We show collapse enables principled early stopping in tuning, and
introduce a predictive model—fit at small scales, and re-used to extrapolate large-scale TLCs.


3 _._ 6


3 _._ 4


3 _._ 2


3 _._ 0


2 _._ 8


2 _._ 6


2 _._ 4


3 _._ 6


3 _._ 4


3 _._ 2


3 _._ 0


2 _._ 8


2 _._ 6


2 _._ 4


|1.7|B, 20 TPP: sweeping B, λ = 0.1|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||_B_<br><br>||
|||||~~126~~<br>252<br>|~~100~~<br>2016<br>||
||||~~504~~|~~504~~|~~403~~||
||||~~504~~|~~504~~|||
||||||||
||||||||
||||||||
||||||||


|1.7|B, 20 TPP: sweeping B, τ=0.15|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||_B_<br><br>||
|||||~~126~~<br>252<br>|~~100~~<br>2016<br>||
||||~~504~~|~~504~~|~~403~~||
||||~~504~~|~~504~~|||
||||||||
||||||||
||||||||
||||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


Figure 7: **Predictability** **of** **batch** **size** **sweeps.** _Left_ : When fixing weight decay _λ_ in batch size
sweeps (standard practice), normalized TLCs _cross_, making final loss hard to predict from partial
results. _Right_ : Fixing _τ_ instead (by adjusting _λ_ ), TLCs maintain ordering, enabling early stopping.


**Role** **of** _τ_ **in** **tuning.** Recent work tunes learning rate _η_ and batch size _B_ at smaller scales and
extrapolates via power laws (Hu et al., 2024; Bi et al., 2024; Porian et al., 2024). These studies
typically fix weight decay _λ_, which unintentionally varies _τ_ —and hence TLC shape. As Fig. 7 ( _left_ )
shows, when _τ_ varies, mid-training loss is a poor predictor of final outcomes during a batch size
sweep. In contrast, when _τ_ is fixed during tuning (by adjusting _λ_ ), the ordering of _B_ -specific curves
is preserved throughout training ( _right_ ); runs can be stopped early (e.g., at 25%) while still reliably
identifying the best batch size.


There are, however, cases where _τ_ must vary. For example, Bergsma et al. (2025a) found optimal _τ_
was no longer constant once _B_ _> B_ crit, potentially requiring retuning of _λ_ .


**Exploiting collapse for early stopping.** Suppose we wish to find the optimal setting of a hyperparameter (HP) in large-scale training. Naively, we could sweep across settings of the HP and train a
large-scale model to completion at each setting; the lowest final loss would identify the best choice.
Alternatively, rather than training to completion, we propose a procedure to infer the final loss values
from partial training runs. We do this by exploiting the collapse phenomenon as follows:


1. For each large-scale setting in the sweep, first identify the corresponding _TLC controls_, i.e, the
LR schedule, _τ_, and TPP.


2. Train a smaller (e.g., 100M-parameter) model for each unique combination of TLC controls.

3. Normalize these small-run loss curves to obtain the _normalized_ _universal_ _TLCs_ ( _ℓ_ ( _t_ [ˆ] )) corresponding to each set of controls.


4. Perform _partial_ training runs (e.g., to 30% of tokens) at each HP setting.


5. For each partial large-scale loss curve, use its corresponding universal TLC to _predict_ the final
loss value. Do this by determining the divisor _L_ ( _T_ ) that maximally _aligns_ the partial curve
with the same segment of the corresponding universal TLC. These divisors thus act both as
normalizing constants _and_, by construction, as calibrated _extrapolations_ of the final loss.


6. Select the optimal hyperparameter setting corresponding to the lowest predicted final loss.


8


**Predicting** **the** **normalized** **universal** **TLCs.** In some cases, we may already have many smallscale runs, but not loss curves for each of the _specific_ TLC controls corresponding to our large-scale
sweep. In this situation, we hypothesize that a _parametric surrogate model_ can generate high-fidelity
normalized universal TLCs as a function of _τ_ and TPP. This allows us to obtain the normalized
TLC shapes required for Step 5 of the above procedure, _without_ training small-models at exactlymatching controls. Fitting such a surrogate lets us leverage our broader TLC dataset, and enables
estimation of normalized TLC shapes that go beyond trained regimes (see Appendix D.3 for an
example).

For the surrogate _ℓ_ ( _t_ [ˆ] ) model, we experimented with several functional forms and ablations on our
111M-scale data, focusing on:

_ℓ_ ˆ(ˆ _t_ ) = ((1 + _ϵ_ 1) _/_ (ˆ _t_ + _ϵ_ 1)) _[m]_ + _b_ ( _η_ (ˆ _t_ ) + _ϵ_ 2) _[q]_ (4)
_·_

The first term captures power-law improvement in training fraction (Appendix B.2) while the second
term modulates this by the LR schedule _η_ ( _t_ [ˆ] ), reflecting how variance suppression is phased over
training (Appendix B.3). _m_, _b_, _q_, _ϵ_ 1 and _ϵ_ 2 are fit parameters. We divide _ℓ_ [ˆ] ( _t_ [ˆ] ) by its final value so
that _ℓ_ [ˆ] (1) = 1 _._ 0. Fixing _ϵ_ 1 = 0 _._ 001 and _ϵ_ 2 = 0 _._ 1 avoids large swings at _ℓ_ [ˆ] (0) and _ℓ_ [ˆ] (1).


In practice, we find _m_ can be fixed (we use 0.05). Parameters _b_ and _q_ then vary systematically with
_τ_ and TPP, respectively, which we capture with power laws:

_b_ = _b_ const ( _τ_ ) _[b]_ [exp] _,_ _q_ = _q_ const (TPP) _[q]_ [exp] (5)
_·_ _·_

Because _b_ and _q_ interact, jointly fitting their parameters would require a _O_ ( _g_ [4] ) grid search (with
_g_ the grid resolution). Instead, we alternate: fit ( _b_ const _, b_ exp) with fixed _q_, then fit ( _q_ const _, q_ exp) with
fixed _b_, iterating to convergence. This reduces cost to _O_ ( _g_ [2] ) while yielding stable fits.


**Results:** **prediction.** We fit the _b_ and _q_ power laws on 111M-scale data and evaluate using mean
absolute error (MAE) between _ℓ_ [ˆ] and true _ℓ_, computed over _t_ [ˆ] _∈_ [0 _._ 2 _,_ 1] (ignoring error around LR
warmup, when initial curves are noisy). We report unweighted mean MAE across all curves.


1.7B, 20 TPP: sweeping _λ_, _B_ = 8064

|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||~~Choo~~<br>Choo<br>|~~se rando~~<br>se curre<br>|~~mly~~<br>nt best<br>||
|||Choo|se predi|ted best||
|||||||
|||||||
|||||||
|||||||


0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8 1 _._ 0
Stop point in training


1 _._ 4


1 _._ 2


1 _._ 0


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0 _._ 0


3.3B, 30 TPP: sweeping _λ_, _B_ = 2016

|Col1|Col2|Choo|se rando|mly|Col6|
|---|---|---|---|---|---|
|||~~Choo~~<br>Choo|~~se curre~~<br>se predic|~~nt best~~<br>ted bes|t|
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||


0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8 1 _._ 0
Stop point in training


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


_λ_


1

|3.3B, 30 TPP: swe|Col2|
|---|---|
|||
||0.05|
||_≈_<br>|
|||
|||
|||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


1 _._ 4


1 _._ 2


1 _._ 0


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0 _._ 0


Figure 8: **3.3B-scale predictions**
**and true normalized TLCs.**


Figure 9: **Early** **stopping** **works** **best** **with** _**predicted**_ **loss:**
Tuning _λ_ in 1.7B ( _left_ ) and 3.3B ( _right_ ) models.


Results show that predictions are good: MAE is low and actually improves with scale (appendix
Table 11), likely because (1) larger datasets yield smoother TLCs, and (2) fewer extreme hyperparameter tests at larger scales. Fig. 8 ( _left_ ) shows an example: predictions trained on 111M-scale
TLCs (1000 _×_ fewer FLOPs) closely match observed curves for a 3.3B model.

Estimating _b_ and _q_ as power laws reduces MAE by two-thirds compared to using fixed values (Table 12), though error remains _≈_ 2 _×_ higher than an oracle fit of _b_ and _q per curve_ . Adjusting for both
_τ_ and TPP is vital; however, fitting _b_ and _q_ jointly on both did not improve further.


**Results: tuning.** We now test whether optimal LLM settings can be predicted from partial training
runs. At different stopping points in training, we choose a setting as the best, and evaluate the gap
between the chosen setting’s final loss and the true best setting. We compare the following choices:


1. _Random baseline_ : randomly choose one setting as the best.


9


2. _Current best_ : choose the setting giving the best result at the stop point.

3. _Predicted best_ : Align partial TLCs with predicted _ℓ_ [ˆ], choose lowest fitted normalizer _L_ ( _T_ ).


Fig. 9 shows results for _λ_ sweeps at 1.7B/20TPP ( _left_ ) and 3.3B/30TPP ( _right_ ). _Predicted_ _best_
achieves negligible loss gaps when stopping after just 30% and 10% of training, respectively. In
contrast, _current best_ —used in Almazrouei et al. (2023) for LR tuning—succeeds initially at 3.3B
but fails at 1.7B, showing it is not a general solution. Further experiments are in Appendix D.2.


6 RELATED WORK


**Scaling laws and scale-stable dynamics.** _Neural scaling laws_ relate loss (generally obtained from
_separate_ training runs) to growth in model, data, and compute sizes, via power laws (Hestness et al.,
2017; Kaplan et al., 2020; Henighan et al., 2020; Hoffmann et al., 2022; Caballero et al., 2022;
Alabdulmohsin et al., 2022). To ensure stable training as models scale, parameterizations such as
_µ_ P transfer base hyperparameters across sizes, and yield early dynamics that are scale-stable (Yang
et al., 2021; Vyas et al., 2023; Kalra et al., 2023), even _super-consistent_ (in curvature) (Noci et al.,
2024). Observing _suboptimal_ _LRs_ under _µ_ P as _data_ scales, recent work has proposed decreasing
the LR as a function of _D_ (Shen et al., 2024; Bjorck et al., 2024); Bergsma et al. (2025a) unify
these techniques as forms of _τ_ adjustment. Qiu et al. (2025) show that, for compute-optimal ladders,
TLCs collapse after normalization. We build on these threads at LLM scale while co-scaling width,
depth, batch size, and weight decay, identifying new controls that govern TLC collapse.


**LLM** **loss-curve** **prediction.** While Kaplan et al. (2020) fit a simple power law to TLCs, recent
papers make loss prediction explicitly LR-dependent (Tissue et al., 2024; Luo et al., 2025; Schaipp
et al., 2025; Qiu et al., 2025; Hong & Wang, 2025). Complementary to these, we take a timescalecentric view: AdamW implements an EMA over updates, and the normalized timescale _τ_ (jointly set
by LR, weight decay, and batch size) acts to control an _implicit batch size_, one that trades bias reduction vs. variance suppression and thereby shapes TLCs. In a noisy-quadratic model (Appendix B.3),
we derive an expression for training loss under a constant LR, and explain why decaying schedules
invert the ordering of TLCs across _τ_, with deformations remaining scale-invariant once normalized.


**Early** **stopping,** **HPO,** **and** **monitoring.** Early-termination and HPO methods extrapolate TLCs
or prune trials (Swersky et al., 2014; Domhan et al., 2015; Jaderberg et al., 2017; Zela et al., 2018;
Li et al., 2018; Choi et al., 2018; Akiba et al., 2019), but typically require many short runs and are
not tailored to LLM pre-training regimes. Our approach leverages _collapse itself_ : fit a small-scale
predictor of normalized TLCs, align in-progress curves to infer _L_ ( _T_ ), and select winners by 10-30%
of training. Operationally, large-scale reports document spikes and divergences (Chowdhery et al.,
2022; Zhang et al., 2022a; Wortsman et al., 2023; Molybog et al., 2023); we show collapse residuals
provide a quantitative, scale-normalized early-warning signal and a practical aid for debugging.


7 CONCLUSION


At LLM scale, _normalized_ _training_ _loss_ _curves_ _collapse_ _across_ _model_ _sizes_ when three controls
align: the AdamW timescale _τ_, the tokens-per-parameter ratio (TPP), and the learning-rate schedule.
Empirically, _τ_ (bias–variance smoothing) and TPP (power-law improvement rate) set TLC shape,
while the schedule phases these effects. Fixing TPP and setting _τ_ optimally for that TPP yields
alignment across _∼_ 100M–3.9B parameters in our experiments.

We instantiate this in **Celerity** : fixed TPP with optimal _τ_ produces tight collapse and competitive
accuracy. _Collapse residuals_ surface issues early, localize their onset, and enable safer restarts. A
simple predictor for normalized TLCs (fit at small scale) supports _early_ _stopping_ in HPO: by 10–
30% of training we can select winners and estimate _L_ ( _T_ ), saving tuning compute. For $1B runs,
collapse provides a valuable reference trajectory: keeping training on track, every step of the way.


10


REFERENCES


Mart´ın Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Gregory S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, et al. TensorFlow: Large-scale
machine learning on heterogeneous distributed systems, 2016.


Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical
report. _arXiv preprint arXiv:2303.08774_, 2023.


Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna:
A next-generation hyperparameter optimization framework. In _Proceedings_ _of_ _the_ _25th_ _ACM_
_SIGKDD international conference on knowledge discovery & data mining_, pp. 2623–2631, 2019.


Ibrahim M Alabdulmohsin, Behnam Neyshabur, and Xiaohua Zhai. Revisiting neural scaling laws
in language and vision. _Advances in Neural Information Processing Systems_, 35:22300–22312,
2022.


Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Leandro von Werra, and Thomas Wolf. SmolLMblazingly fast and remarkably powerful. _Hugging Face Blog_, 16, 2024.


Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Mart´ın Bl´azquez, Guilherme Penedo,
Lewis Tunstall, Andr´es Marafioti, Hynek Kydl´ıˇcek, Agust´ın Piqueres Lajar´ın, Vaibhav Srivastav,
et al. SmolLM2: When Smol goes big–data-centric training of a small language model. _arXiv_
_preprint arXiv:2502.02737_, 2025.


Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, M´erouane Debbah, Etienne [´] Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic,
et al. The Falcon series of open language models. _arXiv preprint arXiv:2311.16867_, 2023.


Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, and Leandro von Werra.
Cosmopedia. [Hugging Face, 2024.](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)


Shane Bergsma, Nolan Dey, Gurpreet Gosal, Gavia Gray, Daria Soboleva, and Joel Hestness.
Power lines: Scaling laws for weight decay and batch size in LLM pre-training. _arXiv preprint_
_arXiv:2505.13738_, 2025a.


Shane Bergsma, Nolan Dey, Gurpreet Gosal, Gavia Gray, Daria Soboleva, and Joel Hestness.
Straight to zero: Why linearly decaying the learning rate to zero works best for LLMs. _arXiv_
_preprint arXiv:2502.15938_, 2025b.


Tamay Besiroglu, Ege Erdil, Matthew Barnett, and Josh You. Chinchilla scaling: A replication
attempt. _arXiv preprint arXiv:2404.10102_, 2024.


Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding,
Kai Dong, Qiushi Du, Zhe Fu, et al. DeepSeek LLM: Scaling open-source language models with
longtermism. _arXiv preprint arXiv:2401.02954_, 2024.


Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya
Skowron, Lintang Sutawika, and Oskar van der Wal. Pythia: A suite for analyzing large language
models across training and scaling, 2023.


Johan Bjorck, Alon Benhaim, Vishrav Chaudhary, Furu Wei, and Xia Song. Scaling optimal LR
across token horizons. _arXiv preprint arXiv:2409.19913_, 2024.


Blake Bordelon, Lorenzo Noci, Mufan Bill Li, Boris Hanin, and Cengiz Pehlevan. Depthwise
hyperparameter transfer in residual networks: Dynamics and scaling limit. _arXiv_ _preprint_
_arXiv:2309.16620_, 2023.


L´eon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine
learning. _SIAM review_, 60(2):223–311, 2018.


11


Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. _Advances in Neural Information Processing Systems_, 33:1877–1901, 2020.


Dan Busbridge, Amitis Shidani, Floris Weers, Jason Ramapuram, Etai Littwin, and Russ Webb.
Distillation scaling laws. _arXiv preprint arXiv:2502.08606_, 2025.


Ethan Caballero, Kshitij Gupta, Irina Rish, and David Krueger. Broken neural scaling laws. _arXiv_
_preprint arXiv:2210.14891_, 2022.


Daeyoung Choi, Hyunghun Cho, and Wonjong Rhee. On the difficulty of DNN hyperparameter optimization using learning curve prediction. In _TENCON 2018-2018 IEEE Region 10 Conference_,
pp. 0651–0656. IEEE, 2018.


Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam
Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. PaLM:
Scaling language modeling with pathways, 2022.


Francesco D’Angelo, Maksym Andriushchenko, Aditya Vardhan Varre, and Nicolas Flammarion.
Why do we need weight decay in modern deep learning? _Advances in Neural Information Pro-_
_cessing Systems_, 37:23191–23223, 2024.


Harm De Vries. Go smol or go home. [Blog post, 2023.](https://www.harmdevries.com/post/model-size-vs-compute-overhead/)


Aaron Defazio, Ashok Cutkosky, Harsh Mehta, and Konstantin Mishchenko. Optimal linear decay
learning rate schedules and further refinements. _arXiv preprint arXiv:2310.07831_, 2023.


Aaron Defazio, Xingyu (Alice) Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, and
Ashok Cutkosky. The road less scheduled. _arXiv preprint arXiv:2405.15682_, 2024.


Nolan Dey, Gurpreet Gosal, Hemant Khachane, William Marshall, Ribhu Pathria, Marvin Tom, and
Joel Hestness. Cerebras-GPT: Open compute-optimal language models trained on the Cerebras
wafer-scale cluster. _arXiv preprint arXiv:2304.03208_, 2023a.


Nolan Dey, Daria Soboleva, Faisal Al-Khateeb, Bowen Yang, Ribhu Pathria, Hemant Khachane,
Shaheer Muhammad, Zhiming, Chen, Robert Myers, Jacob Robert Steeves, Natalia Vassilieva,
Marvin Tom, and Joel Hestness. BTLM-3B-8K: 7B parameter performance in a 3B parameter
model, 2023b.


Nolan Dey, Bin Claire Zhang, Lorenzo Noci, Mufan Li, Blake Bordelon, Shane Bergsma, Cengiz
Pehlevan, Boris Hanin, and Joel Hestness. Don’t be lazy: CompleteP enables compute-efficient
deep transformers. _arXiv preprint arXiv:2505.01618_, 2025.


Tobias Domhan, Jost Tobias Springenberg, Frank Hutter, et al. Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves. In _IJCAI_, volume 15,
pp. 3460–8, 2015.


Ricardo Dominguez-Olmedo, Florian E Dorner, and Moritz Hardt. Training on the test task confounds evaluation and emergence. _arXiv preprint arXiv:2407.07890_, 2024.


Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The Llama 3 herd of models.
_arXiv preprint arXiv:2407.21783_, 2024.


John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and
stochastic optimization. _Journal of Machine Learning Research_, 12(7), 2011.


William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter
models with simple and efficient sparsity. _Journal of Machine Learning Research_, 23(120):1–39,
2022.


Steven Feng, Shrimai Prabhumoye, Kezhi Kong, Dan Su, Mostofa Patwary, Mohammad Shoeybi,
and Bryan Catanzaro. Maximize your data’s potential: Enhancing LLM accuracy with two-phase
pretraining. _arXiv preprint arXiv:2412.15285_, 2024.


12


Sebastian Gabarain. Ultratextbooks-2.0. [Hugging Face, 2024.](https://huggingface.co/datasets/Locutusque/UltraTextbooks-2.0)


Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence
Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric
Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language
model evaluation. Zenodo, 2021.


Paolo Glorioso, Quentin Anthony, Yury Tokpanov, Anna Golubeva, Vasudev Shyam, James Whittington, Jonathan Pilault, and Beren Millidge. The Zamba2 suite: Technical report. _arXiv preprint_
_arXiv:2411.15242_, 2024.


Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimization. In _International Conference on Machine Learning_, pp. 1842–1850. PMLR, 2018.


Alexander H¨agele, Elie Bakouch, Atli Kosson, Loubna Ben Allal, Leandro Von Werra, and Martin
Jaggi. Scaling laws and compute-optimal training beyond fixed training durations. _arXiv preprint_
_arXiv:2405.18392_, 2024.


Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo
Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative
modeling. _arXiv preprint arXiv:2010.14701_, 2020.


Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad,
Md. Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable,
empirically, 2017.


Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza
Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. An
empirical analysis of compute-optimal large language model training. _Advances in Neural Infor-_
_mation Processing Systems_, 35, 2022.


Letong Hong and Zhangyang Wang. On the provable separation of scales in maximal update parameterization. In _Forty-second International Conference on Machine Learning_, 2025.


Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang,
Yuxiang Huang, Weilin Zhao, et al. MiniCPM: Unveiling the potential of small language models
with scalable training strategies. _arXiv preprint arXiv:2404.06395_, 2024.


Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M Czarnecki, Jeff Donahue, Ali
Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, et al. Population based training of neural networks. _arXiv preprint arXiv:1711.09846_, 2017.


Dayal Singh Kalra, Tianyu He, and Maissam Barkeshli. Universal sharpness dynamics in neural
network training: Fixed point analysis, edge of stability, and route to chaos. _arXiv_ _preprint_
_arXiv:2311.02076_, 2023.


Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child,
Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language
models. _arXiv preprint arXiv:2001.08361_, 2020.


Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. _arXiv_ _preprint_
_arXiv:1412.6980_, 2014.


Atli Kosson, Bettina Messmer, and Martin Jaggi. Analyzing & reducing the need for learning rate
warmup in GPT training. _arXiv preprint arXiv:2410.23922_, 2024.


Jakub Krajewski, Jan Ludziejewski, Kamil Adamczewski, Maciej Pi´oro, Michał Krutul, Szymon
Antoniak, Kamil Ciebiera, Krystian Kr´ol, Tomasz Odrzyg´o´zd´z, Piotr Sankowski, Marek Cygan, and Sebastian Jaszczur. Scaling laws for fine-grained mixture of experts. _arXiv_ _preprint_
_arXiv:2402.07871_, 2024.


Tanishq Kumar, Zachary Ankner, Benjamin F Spector, Blake Bordelon, Niklas Muennighoff, Mansheej Paul, Cengiz Pehlevan, Christopher R´e, and Aditi Raghunathan. Scaling laws for precision.
_arXiv preprint arXiv:2411.04330_, 2024.


13


Yann LeCun, John Denker, and Sara Solla. Optimal brain damage. _Advances in Neural Information_
_Processing Systems_, 2, 1989.


Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang,
Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional
computation and automatic sharding. _arXiv preprint arXiv:2006.16668_, 2020.


Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar. Hyperband:
A novel bandit-based approach to hyperparameter optimization. _Journal_ _of_ _Machine_ _Learning_
_Research_, 18(185):1–52, 2018.


Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, et al. StarCoder:
may the source be with you!, 2023.


Hong Liu, Zhiyuan Li, David Hall, Percy Liang, and Tengyu Ma. Sophia: A scalable stochastic
second-order optimizer for language model pre-training. _arXiv preprint arXiv:2305.14342_, 2023.


Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. _arXiv_
_preprint arXiv:1608.03983_, 2016.


Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf. FineWeb-Edu: the finest
collection of educational content. [Hugging Face, 2024.](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)


Jan Ludziejewski, Maciej Pi´oro, Jakub Krajewski, Maciej Stefaniak, Michał Krutul, Jan Mała´snicki,
Marek Cygan, Piotr Sankowski, Kamil Adamczewski, Piotr Miło´s, et al. Joint MoE scaling laws:
Mixture of experts can be memory efficient. _arXiv preprint arXiv:2502.05172_, 2025.


Kairong Luo, Haodong Wen, Shengding Hu, Zhenbo Sun, Zhiyuan Liu, Maosong Sun, Kaifeng Lyu,
and Wenguang Chen. A multi-power law for loss curve prediction across learning rate schedules.
_arXiv preprint arXiv:2503.12811_, 2025.


Sam McCandlish, Jared Kaplan, Dario Amodei, et al. An empirical model of large-batch training.
_arXiv preprint arXiv:1812.06162_, 2018.


Alexandru Meterez, Depen Morwani, Jingfeng Wu, Costin-Andrei Oncescu, Cengiz Pehlevan, and
Sham Kakade. Seesaw: Accelerating training by balancing learning rate and batch size scheduling. _arXiv preprint arXiv:2510.14717_, 2025.


Igor Molybog, Peter Albert, Moya Chen, Zachary DeVito, David Esiobu, Naman Goyal, Punit Singh
Koura, Sharan Narang, Andrew Poulton, Ruan Silva, et al. A theory on Adam instability in largescale machine learning. _arXiv preprint arXiv:2304.09871_, 2023.


Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra
Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A Raffel. Scaling data-constrained language
models. _Advances in Neural Information Processing Systems_, 36, 2023.


Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, et al. OLMoE: Open mixture-of-experts
language models. _arXiv preprint arXiv:2409.02060_, 2024.


Lorenzo Noci, Alexandru Meterez, Thomas Hofmann, and Antonio Orvieto. Super consistency of
neural network landscapes and learning rate transfer. _Advances in Neural Information Processing_
_Systems_, 37:102696–102743, 2024.


Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, et al. 2 OLMo 2 Furious. _arXiv_ _preprint_
_arXiv:2501.00656_, 2024.


Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. OpenWebMath: An open
dataset of high-quality mathematical web text, 2023.


Tomer Porian, Mitchell Wortsman, Jenia Jitsev, Ludwig Schmidt, and Yair Carmon. Resolving
discrepancies in compute-optimal scaling of language models. _arXiv preprint arXiv:2406.19146_,
2024.


14


Ofir Press, Noah Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables
input length extrapolation. In _International Conference on Learning Representations_, 2022.


Shikai Qiu, Lechao Xiao, Andrew Gordon Wilson, Jeffrey Pennington, and Atish Agarwala. Scaling
collapse reveals universal dynamics in compute-optimally trained neural networks. _arXiv preprint_
_arXiv:2507.02119_, 2025.


Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language
models are unsupervised multitask learners, 2019.


Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John
Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models:
Methods, analysis & insights from training Gopher, 2022.


Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi
Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text
transformer. _Journal of Machine Learning Research_, 2020.


Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, and Jason Weston. Hash layers for large sparse
models. _Advances in Neural Information Processing Systems_, 34:17555–17566, 2021.


Fabian Schaipp, Alexander H¨agele, Adrien Taylor, Umut Simsekli, and Francis Bach. The surprising agreement between convex optimization theory and learning-rate scheduling for large model
training. _arXiv preprint arXiv:2501.18965_, 2025.


David Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay
Chaudhary, Michael Young, Jean-Francois Crespo, and Dan Dennison. Hidden technical debt in
machine learning systems. _Advances in neural information processing systems_, 28, 2015.


Neha Sengupta, Sunil Kumar Sahu, Bokang Jia, Satheesh Katipomu, Haonan Li, Fajri Koto, William
Marshall, Gurpreet Gosal, Cynthia Liu, Zhiming Chen, et al. Jais and Jais-chat: Arabiccentric foundation and instruction-tuned open generative large language models. _arXiv preprint_
_arXiv:2308.16149_, 2023.


Noam Shazeer. GLU variants improve transformer. _arXiv preprint arXiv:2002.05202_, 2020.


Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost.
In _International Conference on Machine Learning_, pp. 4596–4604. PMLR, 2018.


Yikang Shen, Matthew Stallone, Mayank Mishra, Gaoyuan Zhang, Shawn Tan, Aditya Prasad, Adriana Meza Soria, David D Cox, and Rameswar Panda. Power scheduler: A batch size and token
number agnostic learning rate scheduler. _arXiv preprint arXiv:2408.13359_, 2024.


Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey.
SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. [Web page, 2023.](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)


Minhak Song, Beomhan Baek, Kwangjun Ahn, and Chulhee Yun. Through the river: Understanding the benefit of schedule-free methods for language model training. _arXiv_ _preprint_
_arXiv:2507.09846_, 2025.


Jacob Mitchell Springer, Sachin Goyal, Kaiyue Wen, Tanishq Kumar, Xiang Yue, Sadhika Malladi,
Graham Neubig, and Aditi Raghunathan. Overtrained language models are harder to fine-tune.
_arXiv preprint arXiv:2503.19206_, 2025.


Kevin Swersky, Jasper Snoek, and Ryan Prescott Adams. Freeze-thaw Bayesian optimization. _arXiv_
_preprint arXiv:1406.3896_, 2014.


Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya
Bhupatiraju, L´eonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ram´e, et al.
Gemma 2: Improving open language models at a practical size. _arXiv preprint arXiv:2408.00118_,
2024.


Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej,
Sarah Perrin, Tatiana Matejovicova, Alexandre Ram´e, Morgane Rivi`ere, et al. Gemma 3 technical
report. _arXiv preprint arXiv:2503.19786_, 2025.


15


Kimi Team. Kimi K2: Open agentic intelligence, 2025. URL [https://github.com/](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf)
[MoonshotAI/Kimi-K2/blob/main/tech_report.pdf.](https://github.com/MoonshotAI/Kimi-K2/blob/main/tech_report.pdf)


Howe Tissue, Venus Wang, and Lu Wang. Scaling law with learning rate annealing. _arXiv preprint_
_arXiv:2408.11029_, 2024.


Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee
Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and efficient foundation
language models, 2023a.


Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. LLaMA 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_, 2023b.


Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada,
Shengyi Huang, Leandro Von Werra, Cl´ementine Fourrier, Nathan Habib, et al. Zephyr: Direct
distillation of LM alignment. _arXiv preprint arXiv:2310.16944_, 2023.


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Advances_ _in_ _Neural_ _Infor-_
_mation Processing Systems_, 2017.


Nikhil Vyas, Alexander Atanasov, Blake Bordelon, Depen Morwani, Sabarish Sainathan, and Cengiz Pehlevan. Feature-learning networks are consistent across widths at realistic scales. _Advances_
_in Neural Information Processing Systems_, 36:1036–1060, 2023.


Nikhil Vyas, Depen Morwani, Rosie Zhao, Mujin Kwun, Itai Shapira, David Brandfonbrener, Lucas
Janson, and Sham Kakade. SOAP: Improving and stabilizing shampoo using adam. _arXiv preprint_
_arXiv:2409.11321_, 2024.


Chengyu Wang, Junbing Yan, Yuanhao Yue, and Jun Huang. DistilQwen2.5: Industrial practices of
training distilled open lightweight language models. _arXiv preprint arXiv:2504.15027_, 2025.


Xi Wang and Laurence Aitchison. How to set AdamW’s weight decay as you scale model and
dataset size. _arXiv preprint arXiv:2405.13698_, 2024.


Kaiyue Wen, Zhiyuan Li, Jason Wang, David Hall, Percy Liang, and Tengyu Ma. Understanding
warmup-stable-decay learning rates: A river valley loss landscape perspective. _arXiv_ _preprint_
_arXiv:2410.05192_, 2024.


Mitchell Wortsman, Peter J Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D CoReyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, et al. Small-scale proxies for large-scale
transformer training instabilities. _arXiv preprint arXiv:2309.14322_, 2023.


Lechao Xiao. Rethinking conventional wisdom in machine learning: From generalization to scaling.
_arXiv preprint arXiv:2409.15156_, 2024.


An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. _arXiv_ _preprint_
_arXiv:2412.15115_, 2024.


An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. _arXiv_ _preprint_
_arXiv:2505.09388_, 2025.


Greg Yang and Edward J Hu. Feature learning in infinite-width neural networks. _arXiv_ _preprint_
_arXiv:2011.14522_, 2020.


Greg Yang, Edward Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder,
Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tuning large neural networks via zero-shot
hyperparameter transfer. In _Advances in Neural Information Processing Systems_, 2021.


16


Greg Yang, Dingli Yu, Chen Zhu, and Soufiane Hayou. Tensor programs VI: Feature learning in
infinite-depth neural networks. _arXiv preprint arXiv:2310.02244_, 2023.


Arber Zela, Aaron Klein, Stefan Falkner, and Frank Hutter. Towards automated deep learning:
Efficient joint neural architecture and hyperparameter search. _arXiv preprint arXiv:1807.06906_,
2018.


Ge Zhang, Scott Qu, Jiaheng Liu, Chenchen Zhang, Chenghua Lin, Chou Leuang Yu, Danny Pan,
Esther Cheng, Jie Liu, Qunshu Lin, et al. MAP-Neo: Highly capable and transparent bilingual
large language model series. _arXiv preprint arXiv:2405.19327_, 2024a.


Guodong Zhang, Lala Li, Zachary Nado, James Martens, Sushant Sachdeva, George Dahl, Chris
Shallue, and Roger B Grosse. Which algorithmic choices matter at which batch sizes? insights
from a noisy quadratic model. _Advances in neural information processing systems_, 32, 2019.


Hanlin Zhang, Depen Morwani, Nikhil Vyas, Jingfeng Wu, Difan Zou, Udaya Ghai, Dean Foster, and Sham Kakade. How does critical batch size scale in pre-training? _arXiv_ _preprint_
_arXiv:2410.21676_, 2024b.


Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt
Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer.
OPT: Open pre-trained transformer language models, 2022a.


Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, et al. Opt-175 logbook. [PDF, 2022b.](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf)


17


A LIMITATIONS AND FUTURE DIRECTIONS


Across our initial (Secs. 3 and 5) and Celerity (Sec. 4) setups, we have tested collapse across two
distinct settings of architecture (and context length), dataset (and vocabulary size), and parameterization. We directly compare TLCs from these two settings in Appendix B.4, while also describing
further experiments in learning-rate schedule ( _Constant_ vs. 10 _×_ vs. D2Z), Adam _β_ 1/ _β_ 2 parameters,
and dense vs. sparse mixture-of-expert (MoE) architectures. However, in all cases our results are
established under single-epoch pre-training with AdamW. The observed patterns may change under extreme TPP, multi-epoch training, alternate optimizers/schedules, or heavy mid-training data
annealing/curricula.


**Optimizers.** We hypothesize the optimizer timescale will remain a primary control of TLC shape
for other optimizers with decoupled weight decay (e.g., Sophia (Liu et al., 2023), MuonClip (Team,
2025)) whose update rules can be expressed in EMA form analogous to AdamW (Sec. 2). Likewise,
the _τ_ perspective should also hold when AdamW is applied in alternate weight bases, e.g., as in
SOAP (Vyas et al., 2024), where AdamW is applied in Shampoo’s eigenbasis (Gupta et al., 2018).
Extending a timescale analysis to optimizers without a natural EMA form (e.g., Adagrad (Duchi
et al., 2011), Adafactor (Shazeer & Stern, 2018), SGD variants) is an important future direction.


**Data** **curricula.** Given the growing use of data curricula and late-stage data annealing in LLM
pre-training, it is valuable to study how shifts in data affect TLC shape across scales. Collapse
may also _inform_ curriculum design by serving as a transfer marker. For example, observing limited
opportunities for experimentation at large scale, Feng et al. (2024) experiment at smaller scales with
_downsampled_ datasets that simulate the repetition occurring at larger sizes (due to limited highquality tokens). Consistency in TLC shape could serve as an indicator of whether the downsample
proportions reflect a consistent overfitting/generalization trade-off across scales. Collapse can thus
serve to confirm that smaller-scale settings provide suitable proxies for optimizing data mixes and
other settings.


**Celerity extensions.** Beyond choosing TPP (controlling placement on the cost/compression curve;
Fig. 5), we aim to understand which factors or training strategies _shift_ _the_ _curve_ _itself_ . For faster
inference, we are especially interested in the location of the _parameter wall_ —the minimal capacity
achieving a target loss—and how architecture (dense vs. MoE), routing, and depth/width changes
affect collapse and efficiency.


We intentionally chose dense models for our initial Celerity series because dense models have fewer
confounding factors (e.g., routing strategy, number of experts), making them simpler to study and
build upon. In our own practice, algorithmic innovations are typically validated on dense models
first. However, we are interested in scaling MoE-variants of our Celerity series due to their documented savings in training compute (Krajewski et al., 2024; Ludziejewski et al., 2025).


**Train loss vs. generalization.** We focus on _training_ loss because (i) it is FLOPs-free to monitor,
(ii) in LLM pre-training it typically tracks validation under stationary data, and (iii) it surfaces issues earlier (e.g., duplicated segments), enabling targeted intervention before held-out degradation.
Late-stage annealing and domain shift can decouple train-loss collapse from downstream behavior. We study validation collapse for Celerity in Appendix C.5; future work should also consider
_downstream-collapse_, and measure train _↔_ val _↔_ downstream correlations across schedules and data
mixtures.


**Predictive model and schedules.** Both collapse itself, and our predictive model’s ability to accurately forecast normalized TLCs, is impaired by loss spikes and divergences, which move the normalized curve away from the universal trajectory (sometimes temporarily, sometimes for extended
periods). From one perspective, this is a feature not a bug, as the resulting collapse anomalies
provide a useful mechanism for detecting training issues (discussed further below).


Empirically, dividing by the final training loss ( _L_ [ˆ] =0) aligned curves best; future work will study
why irreducible-loss offsets, as in Qiu et al. (2025), were not beneficial. In terms of our predictive
model, next steps include factoring LR envelope vs. anneal-phase effects (cf. (Tissue et al., 2024;
Luo et al., 2025)), adding uncertainty (e.g., seed bootstraps) and uncertainty-aware early-stopping


18


Table 3: Model architectures used in Sec. 3 and Sec. 5.


Model _d_ model _n_ layers _d_ ffn _d_ head

111M 768 10 2048 64
266M 768 32 2048 64
610M 2048 10 5461 64
1.7B 2048 32 5461 64
3.3B 2048 64 5461 64


policies. Given that in our experiments, the parametric predictor was fit for one specific LR schedule
(D2Z), we should also revisit whether _b_ ( _τ_ ), _q_ (TPP), and possibly _m_ vary systematically across
cosine (Loshchilov & Hutter, 2016), inverse square-root (Vaswani et al., 2017; Raffel et al., 2020;
Shen et al., 2024), and warmup-stable-decay (WSD) (Hu et al., 2024; H¨agele et al., 2024; Wen et al.,
2024; Song et al., 2025) schedules, and schedule-free schemes (Defazio et al., 2024).


It would also be interesting to measure collapse when batch size schedules are used. Theoretically,
we could maintain collapse by adjusting weight decay whenever batch size changes, maintaining
the _τ_ invariant. We could compare such adjustments to adjustments of LR, e.g., as in Meterez et al.
(2025).


**Systems** **effects** **and** **making** **collapse** **a** **practice.** Collapse residuals are sensitive to systems
effects—microbatching/accumulation, precision, kernel changes, restarts—which can create artifacts or reveal true pathologies. To pay down “hidden technical debt” (Sculley et al., 2015), we
advocate a lightweight _collapse monitor_ : log fraction-of-data in addition to raw step count (easy to
add in TensorBoard (Abadi et al., 2016)), as well as microbatch statistics and restart boundaries; normalize online and alert when residuals exceed policy thresholds. Treating collapse as an operational
invariant reduces configuration fragility and surfaces data/numerics issues early.


B EXPLAINING TLC SHAPE: FURTHER DETAILS


B.1 FULL EXPERIMENTAL DETAILS


In this section, we provide details on the model architecture (Table 3) and training data (Table 4) for
models used in experiments in Sec. 3, Sec. 5, and elsewhere in the appendix. Experimental details
for the Celerity model series are in Appendix C.2.

In total, _≈_ 600 TLCs were analyzed for these experiments. All such models were GPT2-style
LLMs (Radford et al., 2019) with ALiBi (Press et al., 2022) embeddings and SwiGLU (Shazeer,
2020) non-linearity. We use the AdamW optimizer. Following standard practice, we do not apply
weight decay or bias to LayerNorm layers. Default AdamW settings are _β_ 1 = 0 _._ 9, _β_ 2 = 0 _._ 95, and
_ϵ_ = 1e _−_ 8. We report cross-entropy loss. We parameterize with maximal update parameterization,
_µ_ P (Yang et al., 2021), with hyperparameters set via proxy tuning, as described below. For a given
TPP, all models have the exact same warmup phase: a linear warmup of the learning rate from 0 to
the maximum value. In all runs, warmup was 10% of the total steps. Learning rate warmup is standard practice in LLM training (Brown et al., 2020; Rae et al., 2022; Biderman et al., 2023; Dubey
et al., 2024; Kosson et al., 2024).


Note Fig. 3 (and later Table 2, Fig. 7, Fig. 8, and Fig. 9) report batch size in _sequences_ rather than
_tokens_ .


All models in the main experiments were trained on a Cerebras CS-3 system. 610M-parameter
20TPP models take roughly 6 hours each to train on a single CS-3.


**Proxy model hyperparameter tuning.** To find optimal _µ_ P hyperparameters (HPs), we trained a
39M proxy model using a width _d_ proxy of 256, with 24 layers and head size of 64. We trained this
model on 800M tokens with _B_ =256 sequences and a context length 2048. We randomly sampled
350 configurations of base learning rates, base initialization standard deviation, and embedding and
output logits scaling factors, and used the top-performing values as our tuned HPs (Table 5).


19


Table 4: Models, tokens-per-parameter (TPP) and corresponding dataset sizes (in tokens), number
of model variants trained (over LR schedule type, _η_, _λ_, _B_ ) for models used in Sec. 3 and Sec. 5. In
total, _≈_ 600 TLCs were analyzed.


Model TPP _D_ Variants trained


111M 20 2.19B 74
111M 80 8.76B 50
111M 200 21.9B 28
111M 320 35.0B 40
111M 1280 140.1B 11
266M 20 5.31B 25
266M 80 21.2B 19
266M 320 85.0B 19
266M 1280 339.8B 3
610M 20 12.1B 205
610M 80 48.5B 53
610M 200 121.3B 14
610M 320 194.1B 5
1.7B 20 34.3B 31
1.7B 80 137.2B 11
1.7B 160 274.3B 1
1.7B 320 548.6B 1
3.3B 20 66.5B 2
3.3B 23 76.5B 1
3.3B 30 99.8B 5


Table 5: Tuned hyperparameters for _µ_ P proxy model for models used in Sec. 3 and Sec. 5.


_σW,_ base 8 _._ 67e-02
_η_ ˜ 1 _._ 62e-02
_α_ input 9 _._ 17
_α_ output 1 _._ 095


20


It is worth noting that the LR values reported in this paper and shown in figures are base _µ_ P LRs
before _µ_ P-adjustment. Calculation of _τ_ (Sec. 2) requires the adjusted LR (i.e., multiplying by
_d_ proxy _/d_ model). Also, when LR decay is used, reported LR values always refer to the peak/max
LR of the LR schedule.


B.2 EXPLAINING TLC DEPENDENCE ON TPP


Schedules with decaying LR reach their minimum value only at the final step (after _D_ tokens).
However, for a constant LR schedule, every step of training is equivalent to a complete training run
ending at that step. Qiu et al. (2025) make the observation that therefore the loss at every training
fraction _t_ [ˆ] = _t/T_ _∈_ [0 _,_ 1] should respect the same fitted scaling law, but for a training budget of _t_ [ˆ] _· D_
tokens.

Starting from the Chinchilla functional form _L_ ( _N, D_ ) = _E_ + _AN_ _[−][α]_ + _BD_ _[−][β]_, assume that we train
with a constant LR schedule, training until a certain final tokens-per-parameter ratio _k_ = _D/N_ . At
every fraction of training _t_ [ˆ], we will have trained for an intermediate TPP of _t_ [ˆ] _·k_, i.e., using _t_ [ˆ] _·k·N_ total
tokens. To arrive at scale invariance, we note that Hoffmann et al. (2022) found their fitted model
and dataset exponents _α_ and _β_ were roughly equal; this rough equality has also been repeatedly
validated in replication studies (Besiroglu et al., 2024; Porian et al., 2024). Using _a_ = _α_ = _β_, and
focusing on the reducible loss, we obtain a final training loss of:

_L_ ( _N, k · N_ ) = _AN_ _[−][a]_ + _B_ ( _k · N_ ) _[−][a]_

= _AN_ _[−][a]_ + _Bk_ _[−][a]_ _N_ _[−][a]_ (6)


Meanwhile, training for training fraction _t_ [ˆ], the predicted loss is

_L_ ( _N,_ _t_ [ˆ] _· k · N_ ) = _AN_ _[−][a]_ + _Bt_ [ˆ] _[−][a]_ _k_ _[−][a]_ _N_ _[−][a]_ (7)


We now normalize by the final loss to expose the shape of the training loss curve. The resulting
normalized loss _L_ ( _N,_ _t_ [ˆ] _· k · N_ ) _/L_ ( _N, k · N_ ) is independent of model (and dataset) size, depending
only on the training fraction _t_ [ˆ] and the target TPP ratio _k_ :

_ℓ_ ( _t, k_ [ˆ] ) = _[A]_ [ +] _[ B][t]_ [ˆ] _[−][a][k][−][a]_ (8)

_A_ + _Bk_ _[−][a]_


In other words, for TLCs using a constant LR schedule, collapse approximately holds under this
normalization. Scaling of _ℓ_ in _t_ [ˆ] _[−][a]_ also motivates our own TLC predictive form (Eq. (4)).


As shown in Appendix C.1, _a_ = _α_ = _β_ implies there is a single optimal TPP ratio _r_, and moreover,
that the Chinchilla coefficients obey _B_ = _Ar_ _[a]_ . For a given training run, suppose that the TPP
at which we train is a multiple of the optimal TPP by the ratio _v_, e.g., _k_ = _v_ _· r_ . Thus, _v_ = 1
corresponds to optimal TPP, while _v_ _>_ 1 corresponds to overtraining. We can reparameterize the _ℓ_
equation in terms of _v_ as:

_ℓ_ ( _t, v_ [ˆ] ) = [1 +] _[ v][−][a][t]_ [ˆ] _[−][a]_ (9)

1 + _v_ _[−][a]_


This simple equation clarifies how the overtraining factor _v_ influences the shape of the TLCs. When
_v_ is small (undertraining), the power law term dominates, and the TLC gradually decays in _t_ [ˆ] . When
_v_ is large (overtraining), the power law only plays a role for smaller _t_ [ˆ], the curve drops quickly and
then flattens to _ℓ_ = 1. Intuitively, for overtrained models, we make gains quickly at the beginning
of training and then obtain diminishing returns as training progresses.


Qiu et al. (2025) further show that for non-uniform LR schedules, the loss curve is deformed by
_η_ ( _t_ [ˆ] ), but, given consistent curvature of the loss landscape across model scales under _µ_ P (Noci et al.,
2024), the noise-induced deformation is invariant to model size, and thus collapse still holds.


B.3 EXPLAINING TLC DEPENDENCE ON _τ_


As noted in Sec. 3, the AdamW timescale _τ_ = 1 _/_ ( _ηλT_ ) controls the effective memory length of the
parameter updates: smaller _τ_ emphasizes recent updates (bias reduction), while larger _τ_ averages
more broadly (variance reduction). In this sense, _τ_ acts as an implicit batch size.


21


To provide further insight into the role of _τ_ in shaping TLCs, we now derive an analytical expression
for training loss under a constant learning rate, using a simple noisy quadratic model (NQM). While
LLM training minimizes cross-entropy loss, it is common to perform a local quadratic approximation, i.e., a second-order Taylor expansion in the parameters, with the constant Hessian replaced
by the instantaneous Hessian along the training trajectory (LeCun et al., 1989). Thus conclusions
drawn from quadratic models often generalize to large, realistic networks (Zhang et al., 2019).


**Setup.** Following Zhang et al. (2019), we assume the optimizer dynamics are invariant to rotation
and translation, allowing us to model, without loss of generality, a locally quadratic loss, separable
across dimensions, and having an optimum at zero. Specifically, we consider a single quadratic
mode with curvature _h >_ 0, optimum at _θ_ _[⋆]_ = 0, and parameters _θt_, where _t_ is the step index:

_L_ ( _t_ ) = [1] 2 _[h θ]_ _t_ [2] _[.]_ (10)

With AdamW optimization, _θt_ evolves as an exponential moving average (EMA) of stochastic updates _xt_ with constant smoothing _α_ = _ηλ_ (Sec. 3):


_θt_ = (1 _α_ ) _θt_ 1 + _α xt_ 1 _._ (11)
_−_ _−_ _−_

Unrolling the recurrence gives the general form


_θt_ = (1 _α_ ) _[t]_ _θ_ 0 +
_−_


- _t−_ 1

(1 _α_ ) _[t][−]_ [1] _[−][i]_ _α xi._ (12)
_−_
_i_ =0


The first term is the (decaying) contribution of the initialization, while the second term reflects the
accumulation of stochastic updates.


**Continuous (training-fraction) limit.** We now switch to fractional time _t_ [ˆ] = _t/T_ and define the
AdamW timescale _τ_ = 1 _/_ ( _αT_ ). Approximating (1 _−_ _α_ ) _[t][−]_ [1] _[−][i]_ _≈_ _e_ _[−][α]_ [(] _[t][−]_ [1] _[−][i]_ [)] and interpreting the
sum as a Riemann approximation as _T_ _→∞_, we obtain


_θ_ ( _t_ [ˆ] ) _e_ _[−][t/τ]_ [ˆ] _θ_ (0) + [1]
_≈_ _τ_


- _t_ ˆ

_e_ _[−]_ [(ˆ] _[t][−][s]_ [)] _[/τ]_ _x_ ( _s_ ) _ds._ (13)
0


That is, _θ_ consists of two contributions: a decaying memory of the initialization, and a convolution
of the update signal _x_ ( _s_ ) with an exponential kernel of timescale _τ_ (an EMA filter over updates).


**Noise model.** Following Zhang et al. (2019), we model the update signal _x_ ( _t_ [ˆ] ) as preconditioned
white noise: a zero-mean process with constant variance _σx_ [2] [and no temporal correlation,]

E[ _x_ ( _t_ [ˆ] )] = 0 _,_ E� _x_ ( _t_ [ˆ] ) _x_ ( _s_ )� = _σx_ [2] _[δ]_ [(ˆ] _[t][ −]_ _[s]_ [)] _[.]_


This idealized assumption isolates the effect of _τ_ by removing structure in gradient noise beyond its
overall scale.


**Mean and variance.** The EMA filter preserves initialization, which decays exponentially:


E[ _θ_ ( _t_ [ˆ] )] = _e_ _[−][t/τ]_ [ˆ] _θ_ (0) _._


The variance from stochastic updates is


Var[ _θ_ ( _t_ [ˆ] )] = _[σ]_ _x_ [2]
2 _τ_


1 _−_ _e_ _[−]_ [2ˆ] _[t/τ]_ [�] _._ (14)


Thus the total second moment is

E[ _θ_ ( _t_ [ˆ] ) [2] ] = _e_ _[−]_ [2ˆ] _[t/τ]_ _θ_ (0) [2] + _[σ]_ _x_ [2]
2 _τ_


1 _−_ _e_ _[−]_ [2ˆ] _[t/τ]_ [�] _._


In words, the initialization bias decays away on timescale _τ_, while variance from noisy updates
accumulates toward a floor proportional to 1 _/τ_ .


22


**Expected loss.** The per-mode loss is


_L_ ( _t_ [ˆ] ) = [1] 2 _[h θ]_ [(ˆ] _[t]_ [)][2] _[.]_


Taking expectation, and using the decomposition into bias and variance,


     E[ _L_ ( _t_ [ˆ] )] = 12 _[h]_ _e_ _[−]_ [2ˆ] _[t/τ]_ _θ_ (0) [2] + _[σ]_ _x_ [2]
2 _τ_


1 _e_ _[−]_ [2ˆ] _[t/τ]_ [��] _._ (15)
_−_


The first term reflects exponentially decaying initialization bias, while the second reflects variance
accumulation to a floor proportional to 1 _/τ_ .

If initialization is zero-mean in expectation (E[ _θ_ (0) [2] ] = 0), the bias term vanishes and the expression
simplifies to


E[ _L_ ( _t_ [ˆ] )] = _[h σ]_ _x_ [2]
4 _τ_


1 _e_ _[−]_ [2ˆ] _[t/τ]_ [�]
_−_


(16)


which captures the characteristic _fast-then-flatten_ TLC shape under a constant learning rate.


**Interpretation.** Equation 15 decomposes the expected loss into an exponentially decaying _bias_
term ( _∝_ _e_ _[−]_ [2ˆ] _[t/τ]_ _θ_ (0) [2] ) and a _variance_ term that rises to a floor ( _∝_ 1 _/τ_ ). This yields two opposing
effects of _τ_ on TLCs:


 - Smaller _τ_ suppresses initialization bias more rapidly (via the _e_ _[−]_ [2ˆ] _[t/τ]_ decay), but accumulates
higher variance, yielding a higher asymptotic loss floor ( _∝_ 1 _/τ_ ).

 - Larger _τ_ reduces variance more effectively, lowering the final loss, but is slower to eliminate bias
from initialization.


When initialization is zero-mean in expectation, the bias term vanishes and the expression reduces
to Eq. 16.


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


|C|onstant LR: ↑τ →slower drop|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||<br>_η_|(_τ_)||
|||||<br>1.01e<br>2.02e|<br>-03 (1.682)<br>-03 (0.841)||
||||||||
||||||||
|||||4.05e<br>8.09e<br>|-03 (0.421)<br>-03 (0.210)<br>||
||||||||
||||||||
|||||~~1.62e~~<br>3.24e|~~02 (0.105~~<br>-02 (0.053)||
||||||||
||||||||
||||||||
||||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


|1|0x LR decay: ↑τ →faster drop|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||<br>_η_ (|_τ_)||
|||||<br>2.02e-0<br>4.05e-0|3 (0.841)<br>3 (0.421)||
||||||||
||||||||
|||||8.09e-0<br>1.62e-0<br>|3 (0.210)<br>2 (0.105)<br>||
||||||||
||||||||
|||||~~3.24e-~~|~~2 (0.053~~||
||||||||
||||||||
||||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


|LR|decay-to-zero: ↑τ →faster drop|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||||<br>_η_ (|_τ_)||
||||<br>2.02e-0<br>4.05e-0|3 (0.841)<br>3 (0.421)||
|||||||
|||||||
||||8.09e-0<br>1.62e-0<br>|3 (0.210)<br>2 (0.105)<br>||
|||||||
|||||||
||||~~3.24e-~~<br>6.48e-0|~~2 (0.053~~<br>2 (0.026)||
|||||||
|||||||
|||||||
|||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


Figure 10: **Effect** **of** **LR** **schedule** **on** **TLC** **shape** **(610M,** **80TPP).** _Left_ : _Constant_ LR, _Middle_ :
_Linear_ 10 _×_ decay, _Right_ : _Linear_ decay-to-zero. Different schedules deform the TLCs in distinct
ways, yet in all cases the AdamW timescale _τ_ governs the bias-variance trade-off. With a _Constant_
LR, smaller _τ_ accelerates early loss reduction. With D2Z, the effect inverts: smaller _τ_ yields a larger
late-stage drop. Although here _τ_ is varied by changing LR, equivalent effects arise when varying
weight decay or batch size (Fig. 3), confirming _τ_ as a unifying control knob for TLC shape.


This interpretation matches our empirical findings for normalized _Constant_ -LR TLCs (Fig. 10, _left_ ).
The situation is different, however, for LR decay schedules, which we discuss next.


Finally, _τ_ is a _normalized_ timescale and thus invariant to the absolute number of steps. In the NQM,
after normalizing by the final loss _L_ (1) the curvature factor _h_ cancels exactly. The normalized curve
takes the form


_L_ ( _t_ [ˆ] )
_L_ (1) [=]


��11 _−_ _ee_ _[−][−]_ [2ˆ][2] _[t/τ][/τ]_ [�][�] ++ _κ e κ e_ _[−][−]_ [2ˆ][2] _[t/τ][/τ]_ _[,]_ _κ_ = [2] _[τ]_ [ E] _σ_ [[] _[θ]_ _x_ [2][(0)][2][]]
_−_


_._
_σx_ [2]


Thus, when the initialization contribution is negligible by the end of training (or when the ratio _κ_ is
approximately scale-invariant), the normalized TLC depends only on ( _τ,_ _t_ [ˆ] ), and curves at matched


23


_τ_ collapse across model sizes. If _κ_ varies across scales, small early deviations can appear (biasdominated regime) but typically diminish as _e_ _[−]_ [2ˆ] _[t/τ]_ decays.


_Remark._ Qiu et al. (2025) observed collapse without AdamW. Empirically, as _λ_ _→_ 0, TLCs approach a limiting
shape: vanilla Adam behaves like AdamW with _λ_ =0 (effectively _τ_ = _∞_ ).


**Extension to decaying LR schedules.** The constant-LR analysis in Eq. 16 shows that _τ_ sets the
trade-off: smaller _τ_ accelerates early bias reduction but saturates at a higher variance-driven floor,
while larger _τ_ reduces variance more slowly but to a lower asymptote. With a decaying LR schedule,
the smoothing _αt_ = _ηtλ decreases_ after warmup, so the instantaneous timescale _τt_ = 1 _/_ ( _ηtλT_ ) _in-_
_creases_ as training progresses. In this setting, small- _τ_ runs still make rapid early progress (fast bias
reduction), but during the decay phase they gain additional variance suppression as _τt_ lengthens,
often producing a noticeable late-stage drop in loss. By contrast, large- _τ_ runs emphasize variance
reduction throughout, yielding steadier curves without the same end-of-training acceleration. Equivalently, in the EMA view, decay flattens the contribution coefficients _ct,i_, averaging over more (earlier) updates near the end. Thus LR decay effectively combines the early bias-reducing dynamics of
small _τ_ with the late variance-reducing dynamics of large _τ_, inverting the TLC ordering observed
under constant LR (Fig. 10).


This analysis aligns with Bergsma et al. (2025b), who attribute the effectiveness of D2Z schedules to balancing early bias reduction with later variance suppression (building on D’Angelo et al.,
2024). Their treatment is primarily conceptual; here we show how the same bias–variance dynamics
manifest directly in TLC shapes and provide a simple analytical form under the NQM.


B.4 ADDITIONAL TLC EXPERIMENTS


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


|C|onstant LR: {TPP, τ} ≈const.|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||||<br>|_N_||
||||111M<br>610M<br>|, _τ_=0.33<br>, _τ_=0.42<br>||
||||~~1.7B~~|~~_τ_=0.30~~||
|||||||
|||||||
|||||||
|||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


|10|×-Decay LR: {TPP, τ} ≈const.|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||||<br>|<br>_N_||
||||111M<br>610M<br>|, _τ_=0.33<br>, _τ_=0.42<br>||
||||~~1.7B~~|~~_τ_=0.30~~||
|||||||
|||||||
|||||||
|||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


|Dec|ay-to-zero LR: {TPP, τ} ≈const.|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||<br>_N_||
||||111M<br>610M<br>|, _τ_=0.33<br>, _τ_=0.42<br>||
||||~~1.7B~~|~~_τ_=0.30~~||
|||||||
|||||||
|||||||
|||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


Figure 11: **Collapse** **in** **different** **LR** **schedules.** _Left_ : _Constant_ LR, _Middle_ : _Linear_ 10 _×_ decay,
_Right_ : _Linear_ decay-to-zero. In contrast to Fig. 10, where _τ_ varies, here TPP=20 and _τ_ _≈_ 0 _._ 3:
curves collapse across scales.


**Collapse** **under** **alternative** **LR** **schedules.** Fig. 11 shows that normalized TLCs also collapse
under a _Constant_ schedule, a 10 _×_ decay schedule, and our decay-to-zero schedule (all with 10%
warmup). At corresponding model sizes, we use the same batch size, peak LR, and weight decay,
so same-size results across schedules differ only in their final LR. Collapse is slightly looser than
in the Celerity runs because the resulting _τ_ is not matched exactly across schedules (see plot annotations), but the qualitative agreement is strong. These results are consistent with our analysis in
Appendix B.3 and echo the cross-schedule findings of Qiu et al. (2025).


**Collapse** **across** **datasets** **and** **architectures.** TLC shape can in principle depend on task, data,
and architecture (e.g., multi-epoch training on a small corpus can yield faster apparent improvement
than single-epoch pre-training). We therefore ask: how much does normalized TLC shape change
as we vary parameterization, vocabulary size, architecture, context length, and dataset mix?


As a first probe, we compare _Celerity_ TLCs to our earlier non-Celerity runs, at the same TPP (20)
and similar _τ_ _≈_ 0 _._ 2, while varying all items above: Celerity uses CompleteP (vs. vanilla _µ_ P), a
larger vocabulary, different nonlinearity and FFN multiplier, 4 _×_ longer context, and a different data


24


|Co|llapse across dataset/architecture|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||Mo|del||
|||||111M_†_,<br>610M_†_,<br>|_τ_=0.166<br> _τ_=0.210<br>||
||||||||
||||||||
|||||1.7B~~_†_~~, <br>900M_⋆_,|_τ_=0.149<br> _τ_=0.175||
||||||||
||||||||
||||||||
||||||||
||||||||


|S|Col2|parse mixture-of-experts (MoE)|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
||||||Num|. experts||
|||||||<br>1<br>2<br><br>8<br>16<br>||
|||||||~~32~~||
|||||||||
|||||||||
|||||||||
|||||||||


|Col1|Varying Adam β1 and β2|Col3|Col4|Col5|
|---|---|---|---|---|
|||_β_1<br>0.0 (<br>|(_β_2)<br>0.95)<br>||
|||0.5 (<br>0.9 (<br>|0.95)<br>0.95)<br>||
|||~~0.9 (~~<br>0.95 <br>|~~0.99)~~<br>(0.99)<br>||
|||~~0.99 ~~<br>0.999|~~(0.9999)~~<br> (0.9999)||
||||||
||||||
||||||


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


Figure 13: **Collapse** **as** _E_
**varies** **in** **a** **sparse** **MoE.**
111M, _τ_ = 0 _._ 33, 20 TPP.


2 _._ 0


1 _._ 8


1 _._ 6


1 _._ 4


1 _._ 2


1 _._ 0


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


0 _._ 00 0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


Figure 14: **Collapse** **as**
**Adam** _β_ 1 **and** _β_ 2 **vary.**
610M, _τ_ = 0 _._ 21, 20 TPP.


Figure 12: **Collapse** **across**

_†_ **original,** _⋆_ **Celerity** **setups.**
20 TPP.


mixture (Appendices B.1 and C.2). Despite these differences, the TLCs loosely collapse (Fig. 12).
The Celerity 900M model tracks closer to the 610M model than to the 1.7B model, although its _τ_ is
intermediate between these two. Overall, we view this as evidence that the normalized TLC shape
is surprisingly robust when LR schedule, TPP, and _τ_ are held (approximately) fixed.


**Collapse in sparse mixture-of-experts (MoE).** We next analyze sparse MoE architectures, where
only a subset of parameters are active per token (Lepikhin et al., 2020; Fedus et al., 2022). Starting
from our 111M dense model (Appendix B.1), we replace each FFN with a sparse MoE layer and
vary the number of experts _E ∈{_ 1 _,_ 2 _,_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ . Tokens are routed to one expert via hash routing
(Roller et al., 2021) so each expert processes a similar token count. Global training tokens and
datasets are identical across _E_, hence the _effective_ TPP per expert decreases from 20 (dense) to
20 _/E_ as _E_ grows. Note also that as the number of experts _E_ increases, and the effective tokens
per expert decrease proportionally, both the expert’s effective batch size _B_ and effective dataset size
_D_ are reduced by a factor of _E_ . Since _τ_ = _B/_ ( _ηλD_ ), these reductions cancel, leaving the overall
timescale unchanged (for fixed _η, λ_ ).


Fig. 13 shows that lower _E_ (higher effective TPP per expert) yields slightly earlier drops and slightly
flatter tails, broadly obeying the TPP effect characterized in Sec. 3. Thus, the observed deformation
is explained by effective TPP rather than differing training dynamics per se.


**Collapse across Adam** _β_ 1 **and** _β_ 2 **.** Finally, we vary ( _β_ 1 _, β_ 2) at fixed LR, batch size, and weight
decay ( _τ_ = 0 _._ 21, 610M model, 20 TPP). The default (0 _._ 9 _,_ 0 _._ 95) gives the lowest absolute loss in this
experiment, but several “standard” settings—(0 _._ 9 _,_ 0 _._ 95), (0 _._ 95 _,_ 0 _._ 99), and even (0 _._ 99 _,_ 0 _._ 9999)—
produce normalized TLCs that collapse (Fig. 14). In contrast, runs with (0 _._ 0 _,_ 0 _._ 95), (0 _._ 5 _,_ 0 _._ 95), and
a noisy instance of (0 _._ 9 _,_ 0 _._ 99) exhibit early loss spikes; when the loss fails to recover promptly,
the curves remain elevated and do not rejoin the main trajectory, breaking collapse (early loss
spikes also distort early collapse for noisy, large-batch-size runs, e.g., Fig. 7). We also observe that
(0 _._ 999 _,_ 0 _._ 9999), which aggregates gradients over a much longer horizon, follows a systematically
slower (but eventually convergent) trajectory—consistent with an enlarged momentum timescale
prioritizing variance reduction over bias, akin to increasing _τ_ .


Overall, aside from extreme momentum settings or instability-induced spikes, setting of ( _β_ 1 _, β_ 2)
has limited effect on the _shape_ of normalized TLCs. The AdamW timescale _τ_ remains the dominant
optimization-based control for TLC trajectories.


25


C CELERITY MODELS: FURTHER DETAILS


C.1 COMPUTE COST AS A FUNCTION OF MODEL COMPRESSION


Starting from a compute-optimal model size, we now derive an expression for the extra compute
required ( _C/C_ opt) to compress a model to a smaller (less efficient) size, while maintaining the _same_
_loss_ . We use the resulting equation to plot the compression vs. cost trade-off in Fig. 5. This analysis
motivated the selection of max TPP in the Celerity model series.


We begin again with the Chinchilla functional form from Hoffmann et al. (2022), giving loss _L_ as a
function of model size _N_ and data size _D_ :

_L_ ( _N, D_ ) = _E_ + _AN_ _[−][α]_ + _BD_ _[−][β]_ (17)

where _E_, _A_, _α_, _B_, and _β_ are parameters to be fit on observed training runs.


Hoffmann et al. (2022) asked, for a fixed training compute budget _C_ (in FLOPs), how should we
allocate model size _N_ versus number of training tokens _D_ in order to minimize _loss_ ? From Eq. (17),
they derived functions for loss-optimal _N_ opt( _C_ ) and _D_ opt( _C_ ) (constraining _L_ ( _N, D_ ) by the common approximation _C_ _≈_ 6 _ND_ ):


    - _C_
_N_ opt( _C_ ) = _G_
6


_β_

- _α_ + _β_ - _C_
and _D_ opt( _C_ ) = _G_ _[−]_ [1]
6


_α_

- _α_ + _β_
_,_ (18)


1
where _G_ = - _βBαA_ - _α_ + _β_ . Results indicated that _N_ opt and _D_ opt scale roughly equally as _C_ increases.
This analysis agreed with their other methods for estimating compute-optimal scaling, and guided
their _N_ and _D_ allocation for training their large-scale Chinchilla model.


_Let r be the optimal D_ opt( _C_ ) _/N_ opt( _C_ ) _ratio._ If _r_ is roughly independent of _C_, this implies _α_ _β_ .
_≈_
Using _a_ = _α_ = _β_, we obtain:


 - _B_
_r_ =
_A_


- _a_ [1]
_,_ (19)


or equivalently _B_ = _Ar_ _[a]_ .

Replication studies have found _α_ _≈_ _β_ _≈_ 0 _._ 35, and an optimal TPP of around _r_ = 20 (Besiroglu
et al., 2024; Porian et al., 2024) (as noted in Sec. 2).

Now, suppose _a_ = _α_ = _β_ and we obtain a loss of _L_ [ˆ] at the optimal TPP ratio (where _D_ opt = _rN_ opt):

_L_ ˆ = _E_ + _AN_ opt _[−][α]_ [+] _[ BD]_ opt _[−][β]_
= _E_ + _AN_ opt _[−][a]_ [+ (] _[Ar][a]_ [)(] _[rN]_ [opt][)] _[−][a]_

= _E_ + 2 _AN_ opt _[−][a]_ (20)


We now wish to train a _compressed_ model with fraction _kN_ of parameters compared to _N_ opt, but
obtaining the same loss. Let _N_ = _kN_ _N_ opt. If _N_ _<_ _N_ opt, we will need _kD_ extra tokens compared
to _D_ opt in order to reach the loss target. Let _D_ = _kDD_ opt. Rather than training at _r_ TPP, we will
train at a higher ratio ( _kDD_ opt) _/_ ( _kN_ _N_ opt) = ( _kD/kN_ ) _r_ . From Eq. (17), and following a similar
derivation to De Vries (2023), the estimated loss will be:

_L_ ( _N, D_ ) = _E_ + _A_ ( _kN_ _N_ opt) _[−][α]_ + _B_ ( _kDD_ opt) _[−][β]_ (21)


Again substituting _a_ = _α_ = _β_ and _B_ = _Ar_ _[a]_, to obtain the target loss _L_ [ˆ], we set the loss in Eq. (21)
to equal _L_ [ˆ] in Eq. (20), and solve for _kD_, finding:

_kD_ = (2 _kN_ _−a_ ) _−a_ 1 (22)
_−_

The compute cost _C_ of the compressed training will be 6( _kN_ _N_ opt)( _kDD_ opt), from which we can
derive the extra compute ratio compared to _C_ opt = 6 _N_ opt _D_ opt:

_C/C_ opt = _kN_ _kD_

= _kN_ (2 _kN_ _−a_ ) _−a_ 1 (23)
_−_


26


Eq. (23) allows us to vary _kN_ and obtain the corresponding compute overhead. When planning
the Celerity training runs, we assumed _r_ = 20 corresponded to the compute-optimal model size
(following the Chinchilla rule-of-thumb) and we tested different values of _a_ reported in prior work,
using _a_ = 0 _._ 35 in Fig. 5.


C.2 CELERITY RECIPE DETAILS


In this section, we provide further details for the techniques that most impacted Celerity’s performance and compute efficiency, including parameterization, learning rate and weight decay scheduling, architecture, and dataset construction.


3.0

2.9

2.8

2.7

2.6

2.5

2.4

2.3


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|C 0.|049|Col23|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||~~C~~<br>|~~eler~~<br>~~ L(~~|~~ty~~<br>~~ )~~|~~ (~~<br>~~ (~~<br>|~~ )~~<br><br>|<br>C<br>|<br>~~5~~<br>|~~.1~~<br>|~~0~~<br>|~~5~~E~~29)~~<br><br>0.043|||
||||||||||||<br>C|<br>om|<br>let|<br>~~1.~~<br>P|~~72~~<br> L|~~9E~~<br>|~~3~~<br> )|~~1~~<br>|(|<br>C<br>|0.0|51|
|||||||||||||||||||||<br><br>~~2.069E29~~|||
||||||||||||||||||||||||
||||||||||||||||||||||||
||||||||||||||||||||||||
||||||||||||||||||||||||
||||||||||||||||||||||||
||||||||||||||||||||||||


Compute FLOPs (C)


Figure 15: Scaling law comparison between CompleteP and _µ_ P. CompleteP scales more predictably:
the power law is a better fit to the observed points. CompleteP also exhibits a steeper slope, improving loss faster in compute FLOPs, likely due to both better HP transfer (across model width _and_
depth), and better compute efficiency.


**Parameterization.** We compare the effect of different parameterizations and their influence on
compute efficiency in Fig. 15. Specifically, we compare _µ_ P (Yang et al., 2021), which accounts
for scaling in _width_, and CompleteP (Dey et al., 2025), which accounts for scaling in _both_ _width_
_and_ _depth_ . For each parameterization, the hyperparameters (HP), such as learning rate, weight
initialization, and multipliers, are tuned at depth 32 and then directly applied to the target model
training.


Here we note two observations: the first observation is that _µ_ P points do not align well on a standard
scaling law. We attribute this to HP de-tuning when transferring HPs from proxy model depth to
target model depth. Such de-tuning is not seen in the scaling laws for CompleteP and Celerity
(which uses CompleteP but a different dataset), where the points align with minimal error on the
scaling law. Dey et al. (2023a) showed poor scaling law fits when scaling width in the _standard_
_parameterization_ vs. _µ_ P; we believe a similar phenomenon is now happening when scaling in _depth_ .


The second observation is that CompleteP is more compute efficient than _µ_ P, which prior work
has explained through the lens of feature learning (Dey et al., 2025). Indeed, Fig. 4 in Dey et al.
(2025) suggest that CompleteP models exhibit better scaling behavior than _µ_ P, even when both are
comprehensively tuned.


Based on these observations, we used CompleteP for training the Celerity series, using the proxy
model’s tuned HPs across all scales. Sample code for implementing CompleteP is available at
[https://github.com/EleutherAI/nanoGPT-mup/tree/completep.](https://github.com/EleutherAI/nanoGPT-mup/tree/completep)


**Learning Rate and Weight Decay.** We chose the linear decay-to-zero (D2Z) learning rate schedule based on its empirical success and conceptual motivations in Bergsma et al. (2025b). In particular, Bergsma et al. (2025b) showed that as TPP increases beyond compute-optimal 20 TPP, the
relative benefit of D2Z also increases, in a scale-invariant manner. This makes D2Z particularly
appropriate for parameter-efficient training (e.g., Celerity’s 234 TPP model band). All models also
train with linear warmup to the peak LR, over the minimum of 10%-of-total-tokens or 375M-tokens.


27


We tuned _τ_ at a smaller scale and smaller TPP, and transferred across TPP using the power law fit
from Bergsma et al. (2025a). Given learning rate is determined by CompleteP, and batch size is
optimized according to a separate scaling rule (described below), we adjusted weight decay in order
to obtain the desired _τ_ setting at each scale.


**Batch Size.** In early experimentation, the batch sizes were chosen such that they were around the
critical batch size (McCandlish et al., 2018). Later we used the insights from Bergsma et al. (2025a)
and started following the rule _B_ opt _D_ [0] _[.]_ [5], tuning _B_ at a small scale and then inferring optimal
_∝_
batch sizes on larger datasets via the power law.


Table 6: Composition of the Celerity pre-training dataset.


Data Subset Percentage (%)


FineWeb-Edu (Lozhkov et al., 2024) 64.75
StarCoder (Li et al., 2023) 10.8
Cosmopedia (Ben Allal et al., 2024) 4.66
SlimPajama (Soboleva et al., 2023) 17.49
OpenWebMath (Paster et al., 2023) 1.88
UltraTextBooks-2.0 (Gabarain, 2024) 0.42


**Data Selection.** Over the course of experiments, we found that adding more _refined_ data, particularly educational, math, and coding datasets, generally helps the models score higher on common
benchmarks. In Table 6, we break down the datasets used for Celerity model training, including
the proportion assigned to each subset. A large portion of the datasets are focused on educational
materials, math, and coding. We use only the _non–web-crawled_ subsets of SlimPajama (Soboleva
et al., 2023), i.e., excluding C4 and CommonCrawl, which are effectively replaced by FineWeb-Edu.
As noted in Sec. 4, we do not schedule the data sources, i.e., we do not employ a data curriculum in
the training of Celerity, nor do we include (benchmark) task-specific data in Celerity training.


Table 7: Comparison of Celerity models trained on different datasets.


Name Downstream Accuracy (Num Shots)
arc-c arc-e boolq hellaswag piqa siqa winogrande Avg.
(25) (0) (0) (10) (0) (0) (5)


Celerity 300M 27.82 50.63 52.75 37.57 66.21 37.77 52.25 **46.43**
Celerity 300M SlimPJ 24.32 42.17 61.53 36.04 65.56 37.97 50.99 45.51


Celerity 900M 39.68 64.52 47.92 55.02 72.03 41.97 58.48 **54.23**
Celerity 900M SlimPJ 30.89 54.67 55.47 53.74 71.00 40.89 57.46 52.02


Comparison in Table 7 shows that the same model configurations trained on a general dataset like
SlimPajama result in worse downstream performance compared to the Celerity data mix. While
dataset optimization was not a focus of Celerity, these results do underscore the importance of
dataset composition in pre-training. This also makes clear why hyperscalers invest a tremendous
amount of work into data preparation, synthesis, filtering, and refinement.


Table 8 summarizes the dataset sizes for all models in the Celerity model series.


**Model** **Architecture.** Celerity models use a decoder-only GPT2-style transformer architecture.
Table 2 summarizes the architecture dimensions, hyperparameters, and other details of the Celerity
model family. We trained five Celerity model sizes from scratch with parameter counts roughly
300M, 500M, 900M, 1.8B, and 3.9B. All models are trained under consistent data and optimization
methods, on public datasets, in order to foster open science and fair comparison.


C.3 CELERITY FURTHER RESULTS


In our empirical evaluation of Celerity, we necessarily only compare to model families with sufficiently precise and complete training details, in particular the total training tokens. E.g., Llama-3.2


28


Table 8: Models, tokens-per-parameter and corresponding dataset sizes (in tokens) for Celerity.


Model TPP _D_


300M 20 5.4B
300M 80 21.7B
300M 234 63.4B
500M 20 10.1B
500M 80 40.2B
500M 234 117.8B
900M 20 18.1B
900M 80 72.5B
900M 234 212.3B
1.8B 20 36.2B
1.8B 80 144.8B
1.8B 234 424.0B
3.9B 20 77.6B
3.9B 80 310.4B
3.9B 234 909.2B


reports using “up to” 9T tokens (Llama-3.2 Overview) while Llama-3.1 and Qwen-3 sizes are reported “approximately.” Moreover, it is unclear from the papers whether the full pre-training datasets
(i.e., used to train the flagship models) were also used for the smaller models. For an approximate
sense of how these models compare, we include Llama-3 and Qwen models in Table 10 based on
the assumption the models do use the full (approximately-reported) datasets. Note that if these assumptions hold, the smaller Llama-3 and Qwen-3 models would be rather compute-inefficient, e.g.,
they would all be beyond the plotting range of Fig. 2 (i.e., _>_ 10 [24] FLOPs) and score well below the
Celerity extrapolation (and Gemma results) in Fig. 16.


Figure 16: **Celerity compute efficiency vs. distilled models:** Downstream accuracy. Celerity models perform similarly to _distilled_ Gemma-2/Gemma-3 models, when generously only accounting for
distillation student FLOPs. When considering _teacher_ forward pass FLOPs, Gemma curves shift
away from Pareto frontier (worse), with a further shift if we account for FLOPs to _train_ the teacher.


**Compute** **efficiency.** Fig. 16 provides further downstream results for Celerity models (and their
fitted extrapolation), in comparison to larger Gemma-2 and Gemma-3 models. The plot shows
how the accuracy vs. FLOPs comparison depends on whether we account for teaching FLOPs (e.g.,
generating logits for student training), or the initial cost of educating the teacher.


**Token efficiency.** Fig. 17 compares the _token_ efficiency of Celerity to other model families. Using
the Celerity models trained in the fixed 234 TPP band, we fit a power law in _D_ and extrapolate token
efficiency to larger scales.


29


70.0


67.5


65.0


62.5


60.0


57.5


55.0


52.5


Celerity-1.8B


~~CerebrasGPT-6.7B~~


Celerity-900M


|Col1|Col2|
|---|---|
|~~Gemma2-2B~~<br>7B<br>Llama2-7B<br>||
|OLMo-7B<br>~~SmolLM2-1~~<br>|~~.7B~~|
|OLMo2-1B<br>SmolLM-1.7B<br>~~Zamba2-1.2B~~||
|~~Gemma3-1B~~<br>M-3B||
|SmolLM2-36|0M|
|~~OLMo-1B~~<br>3B<br><br>lLM-360M||
|Celerity Fit: 100<br>(<br>D<br>~~1.188~~E~~20 ~~)<br>0|.189|
|||
|10~~12~~<br>10~~13~~<br>ing Tokens (D)|10~~12~~<br>10~~13~~<br>ing Tokens (D)|


Figure 17: **Celerity** **token** **efficiency:** Downstream accuracy. Celerity models are at the Pareto
frontier compared to other model families.


80


75


70


65


60


55


Figure 18: **Celerity token efficiency vs. distilled models:** Downstream accuracy. Celerity models
are on par with distilled models.


Generally, larger models should be more token-efficient for the same token budget. Theoretically,
distillation should also offer greater token efficiency—at a given TPP (Busbridge et al., 2025)—but
by training small models to very-high TPP, the distilled models in Fig. 18 train mainly in a regime
of diminishing returns, and so ultimately end up without an advantage over Celerity’s standard nexttoken-prediction training.


There are many interesting questions around token efficiency at scale, and indeed token efficiency
may become more critical as frontier models reach the limits of high-quality data (Muennighoff
et al., 2023).


**Parameter efficiency.** Finally, Figs. 19 and 20 provide the parameter efficiency comparisons for
Celerity. Celerity models are less parameter efficient than models specifically designed for parameter efficiency.


C.4 OPEN MODEL EVALUATION AND FLOP CALCULATION METHODS


All models are obtained from HuggingFace and evaluated using the Eleuther Eval Harness framework (Gao et al., 2021). The downstream tasks with number of shots are arc-challenge (25), arc-easy
(0), boolq (0), hellaswag (10), piqa (0), siqa (0) and winogrande (5). These tasks are chosen as they
are the most commonly reported downstream benchmarks for pre-trained base models, and are appropriate for Celerity models of the scale that we compare (i.e., tasks where small models perform
above random chance).


30


70.0


67.5


65.0


62.5


60.0


57.5


55.0


52.5


|Col1|Col2|Col3|Col4|
|---|---|---|---|
||~~Gemma2-2B~~<br>|~~Gemma2-2B~~<br>||
||OLMo<br>SmolLM2-1.7B|OLMo<br>SmolLM2-1.7B|-7B|
|~~OLMo2-1B~~|Celerity-3.9B<br>~~BTLM-3B~~<br>SmolLM-1.7B|Celerity-3.9B<br>~~BTLM-3B~~<br>SmolLM-1.7B||
||Celerity-1.8B<br>|Celerity-1.8B<br>||
|Gemma<br>~~SmolLM2-360M~~|3-1B<br>|3-1B<br>||
|OLMo-1B<br><br>~~SmolLM-360M~~|CerebrasGPT-6.7B<br>~~CerebrasG~~|CerebrasGPT-6.7B<br>~~CerebrasG~~|~~PT-13B~~|
|Cele<br>|rity-900M<br><br>Celerity: 100<br>(<br>N<br>~~5.081~~|Celerity: 100<br>(<br>N<br>~~5.081~~|E~~17 ~~)<br>0.189|
|10~~9~~<br>10~~10~~<br>Parameters (N)<br>|10~~9~~<br>10~~10~~<br>Parameters (N)<br>|10~~9~~<br>10~~10~~<br>Parameters (N)<br>|10~~9~~<br>10~~10~~<br>Parameters (N)<br>|


Figure 19: **Celerity parameter efficiency:** Downstream accuracy. Celerity models are less parameter efficient than models trained at much higher TPP, while better than prior models also aiming for
compute efficiency (Cerebras GPT).


|80 Gemma2-27B<br>Gemma2-9B<br>Gemma3-27B<br>75 Gemma3-12B<br>(%)<br>70 Gemma2-2B Gemma3-4B Accuracy<br>SmolLM2-1.7B<br>65 Celerity-3.9B<br>SmolLM-1.7B Celerity: 100 ( N ) 0.189<br>5.081E17 Average<br>Celerity-1.8B SmolLM2: 100 ( N ) 0.153<br>60 Gemma3-1B 1.739E19<br>SmolLM2-360M SmolLM: 100 ( N ) 0.125<br>6.197E21<br>55 SmolLM-360M Gemma2: 100 ( N ) 0.164<br>2.674E18<br>Celerity-900M Gemma3: 100 ( N ) 0.199<br>1.039E17<br>50<br>109 1010 1011<br>Parameters (N)|Col2|Ge|mma2-27B|Col5|
|---|---|---|---|---|
|Celerity: 100<br>(<br>N<br>~~5.081~~E~~17 ~~)<br>0.189<br>SmolLM2: 100<br>(<br>N<br>~~1.739E19 ~~)<br>0.153<br>SmolLM: 100<br>(<br>N<br>~~6.197~~E~~21 ~~)<br>0.125<br>~~Gemma2: 100~~<br>~~(~~<br>N<br>~~2.674E18 )~~<br>0.164<br>Gemma3: 100<br>(<br>N<br>~~1.039~~E~~17 ~~)<br>0.199<br>10~~9~~<br>10~~10~~<br>10~~11~~<br>Parameters (N)<br>50<br>55<br>60<br>65<br>70<br>75<br>80<br>Average Accuracy (%)<br>Celerity-900M<br>Celerity-1.8B<br>~~Celerity-3.9B~~<br>~~Gemma3-1B~~<br>~~Gemma3-4B~~<br>~~Gemma3-12B~~<br>Gemma3-27B<br>~~Gemma2-2B~~<br>Gemma2-9B<br>~~Gemma2-27B~~<br>SmolLM2-360M<br>SmolLM2-1.7B<br>SmolLM-360M<br>SmolLM-1.7B||Gemma2-9B<br>|~~Gemma3-12B~~<br>Gemma3-27<br><br>|B|
|Celerity: 100<br>(<br>N<br>~~5.081~~E~~17 ~~)<br>0.189<br>SmolLM2: 100<br>(<br>N<br>~~1.739E19 ~~)<br>0.153<br>SmolLM: 100<br>(<br>N<br>~~6.197~~E~~21 ~~)<br>0.125<br>~~Gemma2: 100~~<br>~~(~~<br>N<br>~~2.674E18 )~~<br>0.164<br>Gemma3: 100<br>(<br>N<br>~~1.039~~E~~17 ~~)<br>0.199<br>10~~9~~<br>10~~10~~<br>10~~11~~<br>Parameters (N)<br>50<br>55<br>60<br>65<br>70<br>75<br>80<br>Average Accuracy (%)<br>Celerity-900M<br>Celerity-1.8B<br>~~Celerity-3.9B~~<br>~~Gemma3-1B~~<br>~~Gemma3-4B~~<br>~~Gemma3-12B~~<br>Gemma3-27B<br>~~Gemma2-2B~~<br>Gemma2-9B<br>~~Gemma2-27B~~<br>SmolLM2-360M<br>SmolLM2-1.7B<br>SmolLM-360M<br>SmolLM-1.7B|~~Ge~~|~~ma2-2B~~|||
|Celerity: 100<br>(<br>N<br>~~5.081~~E~~17 ~~)<br>0.189<br>SmolLM2: 100<br>(<br>N<br>~~1.739E19 ~~)<br>0.153<br>SmolLM: 100<br>(<br>N<br>~~6.197~~E~~21 ~~)<br>0.125<br>~~Gemma2: 100~~<br>~~(~~<br>N<br>~~2.674E18 )~~<br>0.164<br>Gemma3: 100<br>(<br>N<br>~~1.039~~E~~17 ~~)<br>0.199<br>10~~9~~<br>10~~10~~<br>10~~11~~<br>Parameters (N)<br>50<br>55<br>60<br>65<br>70<br>75<br>80<br>Average Accuracy (%)<br>Celerity-900M<br>Celerity-1.8B<br>~~Celerity-3.9B~~<br>~~Gemma3-1B~~<br>~~Gemma3-4B~~<br>~~Gemma3-12B~~<br>Gemma3-27B<br>~~Gemma2-2B~~<br>Gemma2-9B<br>~~Gemma2-27B~~<br>SmolLM2-360M<br>SmolLM2-1.7B<br>SmolLM-360M<br>SmolLM-1.7B||~~Celerit~~<br>~~Gemm~~<br><br>SmolLM2-1.7B|~~y-3.9B~~<br>~~a3-4B~~||
|Celerity: 100<br>(<br>N<br>~~5.081~~E~~17 ~~)<br>0.189<br>SmolLM2: 100<br>(<br>N<br>~~1.739E19 ~~)<br>0.153<br>SmolLM: 100<br>(<br>N<br>~~6.197~~E~~21 ~~)<br>0.125<br>~~Gemma2: 100~~<br>~~(~~<br>N<br>~~2.674E18 )~~<br>0.164<br>Gemma3: 100<br>(<br>N<br>~~1.039~~E~~17 ~~)<br>0.199<br>10~~9~~<br>10~~10~~<br>10~~11~~<br>Parameters (N)<br>50<br>55<br>60<br>65<br>70<br>75<br>80<br>Average Accuracy (%)<br>Celerity-900M<br>Celerity-1.8B<br>~~Celerity-3.9B~~<br>~~Gemma3-1B~~<br>~~Gemma3-4B~~<br>~~Gemma3-12B~~<br>Gemma3-27B<br>~~Gemma2-2B~~<br>Gemma2-9B<br>~~Gemma2-27B~~<br>SmolLM2-360M<br>SmolLM2-1.7B<br>SmolLM-360M<br>SmolLM-1.7B||Celerity-1.8B<br><br><br>SmolLM-1.7B|Celerity: 100<br>(<br>~~5.~~<br>SmolLM2: 100<br>(<br><br>|N<br>~~081~~E~~17 ~~)<br>0.189<br>N<br>~~1.739E19 ~~)<br>0.153|
|Celerity: 100<br>(<br>N<br>~~5.081~~E~~17 ~~)<br>0.189<br>SmolLM2: 100<br>(<br>N<br>~~1.739E19 ~~)<br>0.153<br>SmolLM: 100<br>(<br>N<br>~~6.197~~E~~21 ~~)<br>0.125<br>~~Gemma2: 100~~<br>~~(~~<br>N<br>~~2.674E18 )~~<br>0.164<br>Gemma3: 100<br>(<br>N<br>~~1.039~~E~~17 ~~)<br>0.199<br>10~~9~~<br>10~~10~~<br>10~~11~~<br>Parameters (N)<br>50<br>55<br>60<br>65<br>70<br>75<br>80<br>Average Accuracy (%)<br>Celerity-900M<br>Celerity-1.8B<br>~~Celerity-3.9B~~<br>~~Gemma3-1B~~<br>~~Gemma3-4B~~<br>~~Gemma3-12B~~<br>Gemma3-27B<br>~~Gemma2-2B~~<br>Gemma2-9B<br>~~Gemma2-27B~~<br>SmolLM2-360M<br>SmolLM2-1.7B<br>SmolLM-360M<br>SmolLM-1.7B|SmolLM2<br>SmolLM-|~~Gemma3-1B~~<br>-360M<br>60M|<br><br><br>SmolLM: 100<br>(<br>~~6.~~<br>~~Gemma2: 100~~<br>~~(~~<br>|<br>N<br>~~197~~E~~21 ~~)<br>0.125<br>N<br> ~~)~~<br>0.164|
|Celerity: 100<br>(<br>N<br>~~5.081~~E~~17 ~~)<br>0.189<br>SmolLM2: 100<br>(<br>N<br>~~1.739E19 ~~)<br>0.153<br>SmolLM: 100<br>(<br>N<br>~~6.197~~E~~21 ~~)<br>0.125<br>~~Gemma2: 100~~<br>~~(~~<br>N<br>~~2.674E18 )~~<br>0.164<br>Gemma3: 100<br>(<br>N<br>~~1.039~~E~~17 ~~)<br>0.199<br>10~~9~~<br>10~~10~~<br>10~~11~~<br>Parameters (N)<br>50<br>55<br>60<br>65<br>70<br>75<br>80<br>Average Accuracy (%)<br>Celerity-900M<br>Celerity-1.8B<br>~~Celerity-3.9B~~<br>~~Gemma3-1B~~<br>~~Gemma3-4B~~<br>~~Gemma3-12B~~<br>Gemma3-27B<br>~~Gemma2-2B~~<br>Gemma2-9B<br>~~Gemma2-27B~~<br>SmolLM2-360M<br>SmolLM2-1.7B<br>SmolLM-360M<br>SmolLM-1.7B|Ce<br>|lerity-900M<br>|<br><br><br>Gemma3: 100<br>(<br>|~~2.674E18 ~~<br><br>N<br>~~1.039~~E~~17 ~~)<br>0.199<br>|
|Celerity: 100<br>(<br>N<br>~~5.081~~E~~17 ~~)<br>0.189<br>SmolLM2: 100<br>(<br>N<br>~~1.739E19 ~~)<br>0.153<br>SmolLM: 100<br>(<br>N<br>~~6.197~~E~~21 ~~)<br>0.125<br>~~Gemma2: 100~~<br>~~(~~<br>N<br>~~2.674E18 )~~<br>0.164<br>Gemma3: 100<br>(<br>N<br>~~1.039~~E~~17 ~~)<br>0.199<br>10~~9~~<br>10~~10~~<br>10~~11~~<br>Parameters (N)<br>50<br>55<br>60<br>65<br>70<br>75<br>80<br>Average Accuracy (%)<br>Celerity-900M<br>Celerity-1.8B<br>~~Celerity-3.9B~~<br>~~Gemma3-1B~~<br>~~Gemma3-4B~~<br>~~Gemma3-12B~~<br>Gemma3-27B<br>~~Gemma2-2B~~<br>Gemma2-9B<br>~~Gemma2-27B~~<br>SmolLM2-360M<br>SmolLM2-1.7B<br>SmolLM-360M<br>SmolLM-1.7B|Ce<br>|lerity-900M<br>|||


Figure 20: **Celerity parameter efficiency scaling comparison:** Downstream accuracy. Preliminary
accuracy vs. model size power law comparison between Gemma (distillation), SmolLM (refined
data), and Celerity models (standard pre-training). Distilled model families have the largest scaling
exponent, suggesting distillation may scale better in parameters.


For full transparency, our method for counting FLOPs across the different models families is given
in Table 9, while a table of all the raw downstream evaluation scores are in Table 10.


C.5 VALIDATION LOSS COLLAPSE


In Fig. 21 we show normalized training loss curves where we evaluate the _validation loss_ of model
checkpoints during training. For each training run, we evaluate checkpoints at 5% intervals, computing loss on the same 493M-token held-out portion of SlimPajama. Similar to training, collapse
occurs except for the initial few checkpoints; we attribute the initial differences to differing LR
warmup proportions (Table 2). Nevertheless, validation collapse is sufficient for deviations to provide a useful diagnostic of any training issues. Validation collapse, measured on fixed datasets, could
be a particularly valuable diagnostic if late-stage annealing or curriculum strategies distort training
curves due to data differences.


31


Table 9: Forward FLOPs calculation for self-attention block and Mamba-2 block. This table only
lists operations that are not covered 6 _nparams_ _ntokens_, which should take care of all operations
that involves a matmul with a weight matrix. _∗_ For Zamba2, the training FLOPs can be calculated as _∗_
6the models analyzed are variations of decoder-transformers whose training FLOPs can be estimated _∗_ _nparams ∗_ _ntokens −_ 2 _∗_ _V_ _∗_ _Dattn ∗_ _ntokens_ +3 _∗_ _Lattn ∗_ ( _L_ _∗_ _Cmamba_ 2 + _Cattn_ ), while the rest of
asforward op and 2 flops per backward op. 6 _∗_ _nparams ∗_ _ntokens −_ 2 _∗_ _V_ _∗_ _Dattn ∗_ _ntokens_ + 3 _∗_ _Lattn ∗_ _Cattn_ . Here 3 represents 1 flop per


Operation FLOPs, given input:
_B_ _S_ _D_ (or _Dattn_ )
_×_ _×_

Attention: _QK_ _[T]_ 2 _BS_ [2] _Dattn_
Self-Attention Attention: softmax, scaling, mask 3 _BS_ [2]

_Cattn_ Attention:Attention: _OV_ projectionmatmul 22 _BSBSD_ [2] _Dattn_ [2] _attn_
Feedforward: activation _BSDattn_


dt softplus 3 _BSH_
xBC conv1d, silu _BS_ ( _ED_ + 2 _N_ ) _K_ + 5 _BS_ ( _ED_ + 2 _N_ )


Mamba-2


_Cmamba_ 2


sampling x, A _BSED_ + _BSH_
SSD, A prefix sum _BHS_
SSD, compute output for each intra-chunk 4 _BHSC_ + _BSEDNC_
SSD, compute state for each intra-chunk 2 _BHS_ + _BSEDN_
SSD, compute inter-chunk recurrence 4 _BH_ ( _Z_ + 1) [2] + _BN_ ( _Z_ + 1) [2] _ED_
SSD, compute output from state per chunk _BHS_ + _BSEDN_ + _BSED_
y+x*D 2 _BSED_
z silu, y norm 6 _BSED_


Params _B_ : batch size, _S_ : sequence length, _V_ : vocabulary size
_Dattn_ : attention hidden dim, _Lattn_ : num attention layers
_D_ : mamba2 hidden dim, _L_ : num mamba2 layers, _E_ : expansion factor
_N_ : mamba2 state dim, _H_ : mamba2 num heads, _P_ : mamba2 head dim
_C_ : mamba2 chunk size, _Z_ : mamba2 num chunks, _K_ : mamba2 conv dim


Validation loss collapse in Celerity


1 _._ 4


1 _._ 3


1 _._ 2


1 _._ 1


1 _._ 0


|Col1|Col2|Col3|N|Col5|
|---|---|---|---|---|
||||300M<br>500M<br>||
||||~~900M~~<br>1.8B||
||||||
||||||
||||||
||||||


0 _._ 25 0 _._ 50 0 _._ 75 1 _._ 00
Fraction of training tokens, _t_ [ˆ]


Figure 21: **Training loss curves also collapse in** _**validation**_ **loss:** Normalized validation loss across
4 Celerity model sizes, all trained to 80 TPP. Validation loss collected at 5% intervals of training.


D COLLAPSE ENABLES EARLY STOPPING: FURTHER DETAILS


D.1 PREDICTING NORMALIZED TRAINING LOSS CURVES


This section provides further details regarding the development of the functional form in Eq. (4),
which we use to predict normalized TLCs and, through these, extrapolate in-progress TLCs. Based
on Sec. 3, we know that TLC shape is modulated by LR schedule, TPP, and _τ_ . Prior theoretical
and empirical work has mostly focused on how loss proceeds as a function of training steps and LR
schedule (Defazio et al., 2023; Tissue et al., 2024; Schaipp et al., 2025; Luo et al., 2025; Qiu et al.,
2025). To incorporate these factors into a single functional form, we take the following approach:


32


Table 10: Evaluations, params, tokens, and FLOPs for all models evaluated.

|Name|Params Tokens FLOPs|Downstream Accuracy (Num Shots)<br>arc-c arc-e boolq hellaswag piqa siqa winogrande Avg.<br>(25) (0) (0) (10) (0) (0) (5)|
|---|---|---|
|BTLM-3b-8k-base (Dey et al., 2023b)|2.60E+09<br>6.27E+11<br>1.55E+22|40.70<br>66.79<br>69.72<br>70.92<br>77.20<br>43.50<br>65.90<br>62.10|
|Cerebras-GPT-1.3B (Dey et al., 2023a)<br>Cerebras-GPT-2.7B<br>Cerebras-GPT-6.7B|1.30E+09<br>2.60E+10<br>2.45E+20<br>2.70E+09<br>5.40E+10<br>1.04E+21<br>6.70E+09<br>1.33E+11<br>6.16E+21|26.79<br>45.83<br>59.33<br>38.55<br>66.76<br>38.59<br>51.70<br>46.79<br>29.52<br>52.57<br>59.24<br>49.74<br>70.78<br>40.23<br>54.85<br>50.99<br>36.01<br>57.91<br>62.81<br>59.45<br>73.99<br>41.50<br>59.98<br>55.95|
|Gemma-2-2b (Team et al., 2024)<br>Gemma-2-9b<br>Gemma-2-27b<br>Gemma-2-2b+forward<br>Gemma-2-2b+forward+teacher<br>Gemma-2-9b+forward<br>Gemma-2-9b+forward+teacher|2.61E+09<br>2.00E+12<br>4.25E+22<br>9.24E+09<br>8.00E+12<br>5.73E+23<br>2.72E+10<br>1.30E+13<br>2.44E+24<br>2.61E+09<br>2.00E+12<br>8.66E+23<br>2.61E+09<br>1.50E+13<br>3.31E+24<br>9.24E+09<br>8.00E+12<br>1.40E+24<br>9.24E+09<br>2.10E+13<br>3.84E+24|53.41<br>80.22<br>73.58<br>74.62<br>79.11<br>51.23<br>71.51<br>69.10<br>68.34<br>87.88<br>84.22<br>82.76<br>82.97<br>55.48<br>80.35<br>77.43<br>69.62<br>88.30<br>84.83<br>87.00<br>84.44<br>54.55<br>83.03<br>78.82<br>53.41<br>80.22<br>73.58<br>74.62<br>79.11<br>51.23<br>71.51<br>69.10<br>53.41<br>80.22<br>73.58<br>74.62<br>79.11<br>51.23<br>71.51<br>69.10<br>68.34<br>87.88<br>84.22<br>82.76<br>82.97<br>55.48<br>80.35<br>77.43<br>68.34<br>87.88<br>84.22<br>82.76<br>82.97<br>55.48<br>80.35<br>77.43|
|Gemma-3-1b-pt (Team et al., 2025)<br>Gemma-3-4b-pt<br>Gemma-3-12b-pt<br>Gemma-3-27b-pt<br>Gemma-3-1b-pt+forward<br>Gemma-3-1b-pt+forward+teacher<br>Gemma-3-4b-pt+forward<br>Gemma-3-4b-pt+forward+teacher<br>Gemma-3-12b-pt+forward<br>Gemma-3-12b-pt+forward+teacher|1.00E+09<br>2.00E+12<br>3.48E+22<br>4.30E+09<br>4.00E+12<br>1.72E+23<br>1.22E+10<br>1.20E+13<br>1.34E+24<br>2.74E+10<br>1.40E+13<br>3.33E+24<br>1.00E+09<br>2.00E+12<br>1.16E+24<br>1.00E+09<br>1.60E+13<br>4.49E+24<br>4.00E+09<br>4.00E+12<br>1.30E+24<br>4.00E+09<br>1.80E+13<br>4.63E+24<br>1.20E+10<br>1.20E+13<br>2.46E+24<br>1.20E+10<br>2.60E+13<br>5.80E+24|39.16<br>71.93<br>66.67<br>62.98<br>74.54<br>42.78<br>62.19<br>60.04<br>58.28<br>81.69<br>78.96<br>77.78<br>79.87<br>49.13<br>72.22<br>71.13<br>67.49<br>87.75<br>85.41<br>84.12<br>81.88<br>52.15<br>80.03<br>76.98<br>70.31<br>88.17<br>87.25<br>86.14<br>83.95<br>53.99<br>82.95<br>78.97<br>39.16<br>71.93<br>66.67<br>62.98<br>74.54<br>42.78<br>62.19<br>60.04<br>39.16<br>71.93<br>66.67<br>62.98<br>74.54<br>42.78<br>62.19<br>60.04<br>58.28<br>81.69<br>78.96<br>77.78<br>79.87<br>49.13<br>72.22<br>71.13<br>58.28<br>81.69<br>78.96<br>77.78<br>79.87<br>49.13<br>72.22<br>71.13<br>67.49<br>87.75<br>85.41<br>84.12<br>81.88<br>52.15<br>80.03<br>76.98<br>67.49<br>87.75<br>85.41<br>84.12<br>81.88<br>52.15<br>80.03<br>76.98|
|Llama-7b (Touvron et al., 2023a)<br>Llama-13b|7.00E+09<br>1.00E+12<br>4.82E+22<br>1.30E+10<br>1.00E+12<br>8.90E+22|50.77<br>72.90<br>75.05<br>77.84<br>79.00<br>45.91<br>71.11<br>67.51<br>55.55<br>74.54<br>77.98<br>81.18<br>80.36<br>46.62<br>76.95<br>70.45|
|Llama-2-7b-hf (Touvron et al., 2023b)<br>Llama-2-13b-hf|7.00E+09<br>2.00E+12<br>1.03E+23<br>1.30E+10<br>2.00E+12<br>1.88E+23|52.65<br>74.54<br>77.71<br>78.98<br>79.11<br>46.11<br>74.19<br>69.04<br>59.47<br>77.53<br>80.58<br>82.23<br>80.52<br>47.34<br>76.16<br>71.98|
|Llama-3-8B (Dubey et al., 2024)<br>Llama-3.1-8B<br>Llama-3.2-1B<br>Llama-3.2-3B|8.00E+09<br>1.50E+13<br>9.46E+23<br>8.00E+09<br>1.50E+13<br>3.85E+24<br>1.23E+09<br>9.00E+12<br>5.29E+23<br>3.21E+09<br>9.00E+12<br>1.40E+24|58.19<br>77.61<br>80.95<br>82.10<br>80.69<br>47.08<br>77.51<br>72.02<br>57.85<br>81.19<br>82.05<br>81.91<br>81.01<br>46.98<br>77.19<br>72.60<br>39.59<br>60.61<br>63.91<br>65.51<br>74.27<br>42.99<br>62.27<br>58.45<br>50.68<br>71.84<br>72.75<br>76.42<br>77.37<br>47.39<br>71.82<br>66.90|
|OLMo-1B-hf (Muennighoff et al., 2024)<br>OLMo-7B-hf|1.00E+09<br>2.00E+12<br>1.56E+22<br>7.00E+09<br>2.46E+12<br>1.26E+23|34.47<br>57.28<br>61.74<br>63.81<br>75.14<br>42.12<br>60.46<br>56.43<br>45.14<br>68.77<br>72.45<br>77.13<br>79.43<br>44.52<br>70.96<br>65.49|
|OLMo-2-0425-1B (OLMo et al., 2024)<br>OLMo-2-1124-7B<br>OLMo-2-1124-13B<br>OLMo-2-0325-32B|1.00E+09<br>4.00E+12<br>3.04E+22<br>7.00E+09<br>4.00E+12<br>2.03E+23<br>1.30E+10<br>5.00E+12<br>4.67E+23<br>3.20E+10<br>6.00E+12<br>1.30E+24|45.39<br>73.36<br>63.03<br>68.71<br>75.63<br>43.76<br>65.90<br>62.25<br>64.51<br>82.87<br>80.00<br>81.93<br>81.01<br>51.33<br>77.03<br>74.10<br>66.13<br>81.31<br>73.91<br>84.99<br>82.15<br>52.05<br>83.03<br>74.80<br>69.45<br>85.94<br>82.81<br>87.33<br>82.97<br>54.25<br>83.90<br>78.09|
|Qwen3-0.6B-Base (Yang et al., 2025)<br>Qwen3-1.7B-Base<br>Qwen3-4B-Base<br>Qwen3-8B-Base<br>Qwen3-14B-Base|6.00E+08<br>3.60E+13<br>5.31E+23<br>1.70E+09<br>3.60E+13<br>1.18E+24<br>4.00E+09<br>3.60E+13<br>2.19E+24<br>8.00E+09<br>3.60E+13<br>3.90E+24<br>1.40E+10<br>3.60E+13<br>6.09E+24|44.80<br>58.00<br>69.82<br>53.46<br>69.80<br>43.30<br>60.46<br>57.09<br>55.20<br>68.60<br>79.24<br>67.19<br>75.52<br>48.62<br>65.27<br>65.66<br>64.42<br>75.93<br>82.91<br>75.64<br>77.86<br>50.00<br>72.61<br>71.34<br>67.24<br>79.88<br>83.09<br>79.55<br>79.54<br>54.76<br>77.19<br>74.46<br>69.97<br>81.86<br>86.76<br>82.69<br>82.10<br>55.89<br>79.48<br>76.96|
|Qwen2.5-0.5B (Yang et al., 2024)<br>Qwen2.5-1.5B<br>Qwen2.5-3B<br>Qwen2.5-7B<br>Qwen2.5-14B<br>Qwen2.5-32B|5.00E+08<br>1.80E+13<br>2.04E+23<br>1.50E+09<br>1.80E+13<br>1.38E+24<br>3.00E+09<br>1.80E+13<br>8.51E+23<br>7.00E+09<br>1.80E+13<br>3.62E+24<br>1.40E+10<br>1.80E+13<br>8.58E+24<br>3.20E+10<br>1.80E+13<br>1.29E+25|35.24<br>58.54<br>61.47<br>51.83<br>69.80<br>44.17<br>56.59<br>53.95<br>54.86<br>72.10<br>72.48<br>67.86<br>75.90<br>49.08<br>65.27<br>65.36<br>56.31<br>73.02<br>77.43<br>74.54<br>78.67<br>49.80<br>71.67<br>68.78<br>63.65<br>77.48<br>84.65<br>80.19<br>79.82<br>54.61<br>76.40<br>73.83<br>67.58<br>79.25<br>85.35<br>84.21<br>82.43<br>55.48<br>81.06<br>76.48<br>70.65<br>77.99<br>87.49<br>85.16<br>82.43<br>56.29<br>82.08<br>77.44|
|SmolLM-135M (Allal et al., 2024)<br>SmolLM-360M<br>SmolLM-1.7B|1.35E+08<br>6.00E+11<br>1.51E+21<br>3.60E+08<br>6.00E+11<br>3.16E+21<br>1.70E+09<br>1.00E+12<br>1.54E+22|32.00<br>56.14<br>60.09<br>42.92<br>68.01<br>39.56<br>52.25<br>50.14<br>38.65<br>63.59<br>55.05<br>54.24<br>71.44<br>40.99<br>57.14<br>54.44<br>49.40<br>73.57<br>66.15<br>67.33<br>75.95<br>43.35<br>61.72<br>62.50|
|SmolLM2-135M (Allal et al., 2025)<br>SmolLM2-360M<br>SmolLM2-1.7B|1.35E+08<br>2.00E+12<br>2.48E+21<br>3.60E+08<br>4.00E+12<br>1.20E+22<br>1.70E+09<br>1.10E+13<br>1.21E+23|33.02<br>58.38<br>60.06<br>43.64<br>68.12<br>39.25<br>53.12<br>50.80<br>40.78<br>68.22<br>61.56<br>57.46<br>71.76<br>40.89<br>58.41<br>57.01<br>53.50<br>73.27<br>72.32<br>73.16<br>77.53<br>44.52<br>68.35<br>66.09|
|SmolLM3-3B-Base|3.00E+09<br>1.12E+13<br>8.56E+23|59.81<br>76.85<br>80.49<br>77.18<br>79.11<br>46.78<br>73.40<br>70.52|
|Zamba2-1.2B (Glorioso et al., 2024)<br>Zamba2-2.7B<br>Zamba2-7B|1.20E+09<br>3.00E+12<br>3.86E+23<br>2.70E+09<br>3.00E+12<br>4.77E+23<br>7.40E+09<br>2.00E+12<br>7.68E+23|53.92<br>66.71<br>70.18<br>72.21<br>77.20<br>46.42<br>68.98<br>65.09<br>60.67<br>73.82<br>78.07<br>77.72<br>79.49<br>45.50<br>76.01<br>70.18<br>68.34<br>80.39<br>83.70<br>83.53<br>80.69<br>49.90<br>79.72<br>75.18|
|Celerity-300M<br>Celerity-500M<br>Celerity-900M<br>Celerity-1.8B<br>Celerity-3.9B|2.71E+08<br>6.34E+10<br>1.47E+20<br>5.03E+08<br>1.18E+11<br>5.15E+20<br>9.06E+08<br>2.12E+11<br>1.68E+21<br>1.81E+09<br>4.24E+11<br>6.54E+21<br>3.88E+09<br>9.08E+11<br>2.89E+22|27.82<br>50.63<br>52.75<br>37.57<br>66.21<br>37.77<br>52.25<br>46.43<br>34.39<br>56.06<br>61.22<br>45.96<br>69.31<br>40.23<br>52.64<br>51.40<br>39.68<br>64.52<br>47.92<br>55.02<br>72.03<br>41.97<br>58.48<br>54.23<br>48.55<br>70.29<br>65.17<br>64.34<br>75.46<br>42.99<br>60.22<br>61.00<br>54.01<br>75.55<br>66.61<br>72.19<br>77.97<br>44.73<br>65.90<br>65.28|


 - Use a functional form that accounts for training fraction and LR schedule

 - Make the _parameters_ of this functional form depend on TPP and _τ_


This led to Eq. (4). Our initial aim here is not to develop the best possible TLC predictor, but to
obtain a simple, effective, and interpretable method for extrapolating TLCs, allowing us to test the
value of this extrapolation for early stopping in hyperparameter tuning.


We conducted a variety of preliminary experiments at 111M-scale, using the same data as in Sec. 3
(with details in Appendix B.1). As an input to Eq. (4), the LR schedule is normalized to be at 1.0 at
its peak. It’s also interpreted over training fraction, so from 0.0 to 1.0. Experiments in this data only
use fits for linear decay-to-zero schedules. To get an initial sense of how the parameters in Eq. (4)
vary, we did a multi-dimensional grid search to determine optimal parameters for each individual


33


curve, measuring total macroaveraged MAE loss over all 111M-scale TLCs. Over the course of
these experiments, we found total MAE did not change substantially when we fixed _m_ = 0 _._ 05,
and we subsequently tuned _ϵ_ 1 and _ϵ_ 2 to small constants in order to avoid boundary effects at _t_ [ˆ] = 0
and _t_ [ˆ] = 1 (when _η_ ( _t_ [ˆ] ) goes to zero). Prior to fitting, training curves were smoothed using a moving
average filter covering 12288 sequences (equal to the largest batch size in the dataset), and we ignore
error on the first 20% of each curve (around LR warmup when curves are noisy).


10 [0]


10 _[−]_ [1]


|O|Optimal b ∼τ|τ, not TPP|Col4|
|---|---|---|---|
|||||
|||||
||||TPP<br>20<br>80<br>200<br>320<br>1280|
|||||


10 _[−]_ [2] 10 _[−]_ [1]

_τ_


10 [0]


10 _[−]_ [1]


|Optimal q ∼T|TPP, not τ|Col3|
|---|---|---|
||||
|||TPP<br>20<br>80<br>200<br>320<br>1280|
||||
||||
||||


10 _[−]_ [2] 10 _[−]_ [1]

_τ_


Figure 22: **Trends** **in** **fits** **for** **training** **curve** **prediction.** Optimal per-curve fits (from per-curve
grid searches) for Eq. (4): _ℓ_ [ˆ] ( _t_ [ˆ] ) _≈_ 1 _/m_ [0] _[.]_ [05] + _b · η_ ( _t_ [ˆ] ) _[q]_ : _b_ and _q_ parameters. _Left_ : Optimal _b_ varies
strongly in _τ_ (Pearson’s _r_ = -0.59), weakly in TPP ( _r_ = 0.17). _Right_ : Optimal _q_ varies somewhat in
TPP ( _r_ = -0.30), while overall stronger in _τ_ ( _r_ = 0.55), but _τ_ trends reverse at higher TPP.


Fig. 22 shows the optimal fits for _b_ and _q_ when each curve is fit independently. For **optimal** _b_,
we found that correlation in _τ_ was much stronger than correlation in TPP (Pearson’s _r_ = -0.59 for
_τ_, _r_ = 0.17 for TPP). On the other hand, while **optimal** _q_ seems to increase with _τ_ for TPP =
20, the relationship with _τ_ at other TPP appears random. Furthermore, note larger TPP values do
correspond to lower optimal _q_ ( _r_ = -0.30). Based on these fits, we hypothesize we could obtain
reasonable predictions by fitting _b_ as a power law in _τ_, and _q_ as a power law in TPP:


_b_ = _b_ const _τ_ _[b]_ [exp] _,_ _q_ = _q_ const TPP _[q]_ [exp] (24)
_·_ _·_

As noted in Sec. 5. Also, as reported in that section, we developed an alternating greedy optimization
procedure to fit these four parameters, exponentially reducing the cost of the grid search space.


**Results.** We first note that the fits improve over the iterations of our alternating grid search procedure, demonstrating that optimal parameters of the power laws do depend on each other, and can
reach stable fits through iterative alternating fitting.


Table 11: **Predictions improve with scale** : fit at 111M scale, evaluated at larger scales.


Evaluation scale MAE Number of evaluation curves


111M* (fitting points) 1.37% 112
266M 0.75% 40
610M 1.07% 102
1.7B 0.66% 21
3.3B 0.54% 7


Table 11 and Table 12 are the tables discussed in the main paper, showing how fits obtained at 111M
perform at other scales (Table 11), and how different fitting procedures perform on the 610M-scale
evaluation data (Table 12). Fitting _b_ and _q_ with the optimum values _per-curve_ (i.e., _oracle_ fits)
achieves an MAE of 0.504%, roughly half that of the dual power law extrapolations.


34


Table 12: **Separate power laws for** _b_ **and** _q_ **work well** : fit at 111M scale (112 TLCs), evaluation at
610M (102 TLCs).


Method for estimating _b_ Method for estimating _q_ MAE


Global fixed optimum Global fixed optimum 3.03%
Global fixed optimum _q_ = PowerLaw(TPP) 3.35%
_b_ = PowerLaw( _τ_ ) Global fixed optimum 2.08%
_b_ = PowerLaw( _τ_ ) _q_ = PowerLaw(TPP) 1.07%
_b_ = PowerLaw( _τ,_ TPP) _q_ = PowerLaw( _τ,_ TPP) 1.07%


D.2 EARLY STOPPING IN TUNING: FURTHER RESULTS


In this section we describe some further early stopping experiments, and present additional evaluation metrics.


1 _._ 4


1 _._ 2


1 _._ 0


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0 _._ 0


1.7B, 20 TPP: sweeping _B_, _λ_ = 0 _._ 1

|Col1|Col2|Choo<br>Choo|se rando<br>se curre|mly<br>nt best|Col6|
|---|---|---|---|---|---|
|||Choo|<br>se predic|<br>ted best||
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||


0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8 1 _._ 0
Stop point in training


1 _._ 4


1 _._ 2


1 _._ 0


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0 _._ 0


617M, 20 TPP: sweeping _B_, _λ_ = 0 _._ 1

|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||~~Choo~~<br>Choo<br>|~~se rando~~<br>se curre<br>|~~mly~~<br>nt best<br>||
|||Choo|se predi|ted best||
|||||||
|||||||
|||||||
|||||||


0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8 1 _._ 0
Stop point in training


Figure 23: **Early** **stopping** **comparison:** **further** **setups.** Companion to Fig. 9, now comparing
early stopping accuracy (final loss of predicted vs. actual best) for _B_ sweeps at 1.7B ( _left_ ) and 617M
( _right_ ) (both 20 TPP). _Current best_ works well very early, but is worse for most of training.


Fig. 23 evaluates early stopping strategies in batch-size sweeps at a fixed _λ_ value. Fig. 23, _left_, uses
the same data as in Fig. 7, _left_ . While we do not advocate keeping _λ_ fixed during _B_ sweeps in
practice, this data can nevertheless serve to evaluate prediction of early winners in tuning. Both of
these plots exhibit the phenomenon also observed in Fig. 9, _right_ : choosing the current best setting
after LR warmup, as was done in Falcon (Almazrouei et al., 2023), is better than selecting the best
during the middle of training. However, as seen in Fig. 9, _left_, this method is not always successful.
In Fig. 23, _left_, choosing the extrapolated best setting outperforms choosing the current best from
40% of training, while it picks the correct winner from the beginning in Fig. 23, _right_ .


1 _._ 4


1 _._ 2


1 _._ 0


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0 _._ 0


|1.7B, 20 TPP: sweeping λ, B = 8064|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||Ou<br>|tput cur<br>|rent loss<br>||
|||~~Pr~~|~~Pr~~|~~dict ﬁn~~|~~l loss~~||
|||~~Pr~~|~~Pr~~||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||


0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8 1 _._ 0
Stop point in training


1 _._ 4


1 _._ 2


1 _._ 0


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0 _._ 0


3.3B, 30 TPP: sweeping _λ_, _B_ = 2016

|Col1|Col2|Col3|Ou|tput cur|rent loss|Col7|
|---|---|---|---|---|---|---|
|||~~Pr~~|~~Pr~~|~~dict ﬁn~~|~~l loss~~||
|||~~Pr~~|~~Pr~~||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||


0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8 1 _._ 0
Stop point in training


Figure 24: **Early stopping comparison:** **MAE at 1.7B, 3.3B:** _λ_ **sweeps.** Mean absolute error of all
predicted final losses, comparing taking current loss vs. extrapolating final loss.


In many cases, rather than caring purely about which setting is best, we care about the actual projected final loss. This may be useful for fitting scaling laws, or for helping practitioners reason about


35


1 _._ 4


1 _._ 2


1 _._ 0


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0 _._ 0


|1.7B|B, 20 TP|PP:|: swee|eping B,|λ = 0.|
|---|---|---|---|---|---|
||||Ou<br>|tput cur<br>|rent los<br>|
|||~~Pr~~|~~Pr~~|~~dict ﬁn~~|~~l loss~~|
|||~~Pr~~|~~Pr~~|||
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||


0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8 1 _._ 0
Stop point in training


1 _._ 4


1 _._ 2


1 _._ 0


0 _._ 8


0 _._ 6


0 _._ 4


0 _._ 2


0 _._ 0


|617M|M, 20 TP|PP|P: swee|eping B,|, λ =|
|---|---|---|---|---|---|
||||Ou<br>|tput cur<br>|rent l<br>|
|||~~Pr~~|~~Pr~~|~~dict ﬁn~~|~~l loss~~|
|||~~Pr~~|~~Pr~~|||
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||


0 _._ 2 0 _._ 4 0 _._ 6 0 _._ 8 1 _._ 0
Stop point in training


Figure 25: **Early stopping comparison:** **MAE at 1.7B, 617M:** _B_ **sweeps.** Mean absolute error of
all predicted final losses, comparing taking current loss vs. extrapolating final loss.


the trade-offs of, for example, greater throughput from larger _B_ vs. suffering higher final loss. We
therefore evaluated the same four hyperparameter sweeps above, but now evaluating the average
loss difference between the predicted final loss and the true final loss for all curves. The baseline
chooses the current loss for each curve at the given training fraction, which will overestimate the
final loss. Results in Figs. 24 and 25 show that in three of four cases, extrapolating the final loss
using our predictive form results in _much_ smaller average error than using the current value.


The only instance where predicting the final loss incurred significant error was the 1.7B, 20 TPP
model with _B_ = 8064. We note that the TLCs are very noisy at this high batch size across almost all _λ_ settings and therefore it is evidently challenging to align the in-progress training runs
to the predicted TLC. Increasing smoothing reduces the predicted error somewhat, but the primary
issue is that the noise affects the TLC mainly in the first 60% of training, thus distorting even the
smoothed loss from the universal trajectory. Accurate prediction in the presence of loss spikes is an
acknowledged limitation of our methodology (Appendix A).


D.3 OPTIMAL AND SUBOPTIMAL TLCS AS TPP SCALES


Suboptimal curves: fxed _η_, _λ_, _B_


1 _._ 20


1 _._ 15


1 _._ 10


1 _._ 05


1 _._ 00


0 _._ 0 0 _._ 5 1 _._ 0
Training fraction


10 [4]


10 [3]


10 [2]


1 _._ 20


1 _._ 15


1 _._ 10


1 _._ 05


1 _._ 00


1 _._ 20


1 _._ 15


1 _._ 10


1 _._ 05


1 _._ 00


Suboptimal curves: _τ_ =0.22


0 _._ 0 0 _._ 5 1 _._ 0
Training fraction


10 [4]


10 [3]


10 [2]


|Optimal|curves: τ ∼TP|
|---|---|
|||
|||
|||
|||
|||
|||


0 _._ 0 0 _._ 5 1 _._ 0
Training fraction


10 [4]


10 [3]


10 [2]


Figure 26: **Evolution of train curve shape.** _Left_ : When TPP is scaled but _η_, _λ_ and _B_ are held constant, curve shape varies significantly. _Middle_ : When _τ_ is instead held constant, shape evolves more
gradually. _Right_ : When _τ_ scales with TPP according to established power laws, curves maintain
their concave structure.


Given a fitted predictive form (Eq. (4)), it is natural to ask how TLC shape varies as TPP increases,
under various hyperparameter (HP) scaling strategies. In this section, we consider three scenarios:


1. No adjustment to any HPs: basically standard practice under _µ_ P until very recently.


2. Maintain constant _τ_ : i.e., following the prescription of Wang & Aitchison (2024).


3. Optimize _τ_ : adjust _τ_ for each TPP setting following the _τ_ power law of Bergsma et al. (2025a).


36


Results in Fig. 26 demonstrate that, with no HP adjustments, curve shape changes substantially
across TPP ( _left_ ). Fixing _τ_ results in more consistent shapes ( _middle_ ), but only when _τ_ is scaled
for TPP do curves maintain their characteristic concave shape, with a noticeable drop near the end
of training ( _right_ ). One may view this final period as the _annealing_ phase of training, or the phase
where variance is reduced and we descend the valley into the river (Wen et al., 2024). As TPP
increases, we must reduce _τ_ correspondingly to prioritize exploration for the majority of training,
enabling this final descent only in the final phases.


37