# FACT: FREQUENCY-AWARE CHANNEL-GUIDED MUL## TIVARIATE TIME SERIES FORECASTING


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Forecasting Multivariate Time Series (MTS) requires capturing complex intrachannel dynamics and evolving inter-channel dependencies. However, existing
methods often struggle to disentangle meaningful signals from inter-channel noise
and intricate interaction patterns. To address this, we propose a novel framework that operates entirely in the frequency domain, modeling inter-channel relationships at the component level. Our approach first dynamically decomposes
each time series into its constituent frequencies. An Adaptive Band Decomposition mechanism then identifies and isolates the most salient frequency components, simultaneously filtering noise and enhancing computational efficiency.
This allows our model to capture time-varying inter-channel dependencies with
high fidelity. Furthermore, our learning objective effectively balances accuracy
against regularization constraints for both computational efficiency and interpretability. Extensive experiments on diverse, real-world datasets demonstrate
that our method achieves competitive performance. Code is available at this repository: [https://anonymous.4open.science/r/FACT.](https://anonymous.4open.science/r/FACT)


1 INTRODUCTION


Multivariate time series (MTS) forecasting supports power scheduling, weather prediction and industrial control, where accuracy, robustness and interpretability are equally critical (Zhou et al.,
2021; Wu et al., 2021a; Zhou et al., 2022). Existing research largely falls into two paradigms.
Channel-Dependent (CD) models explicitly mix variables but easily introduce spurious correlations
and face scalability issues in high dimensions (Zhang & Yan, 2023; Liu et al., 2023; Wang et al.,
2023); Channel-Independent (CI) models improve robustness by per-channel processing, but sacrifice genuine couplings and physical interpretability (Nie et al., 2023; Han et al., 2024). This tension
indicates a need for fine-grained, controllable interaction modelling.


The core challenge in MTS forecasting lies in disentangling meaningful signals from the noise
inherent in complex inter-channel interactions. While spectral analysis offers a promising direction, we observe a critical physical nuance: different spectral components carry distinct semantics—amplitude reflects energy intensity, while phase encodes temporal alignment. For instance,
daily load patterns (high frequency) and seasonal trends (low frequency) often exhibit different interaction modes (coordination vs. antagonism). A difficulty arises, however, in effectively modeling
these “channel-frequency cells” (Fig. 1). Existing spectral methods (Wu et al., 2023; Yi et al., 2023b)
typically rely on global reweighting or fixed decomposition, failing to capture dynamic, cell-level
dependencies and, crucially, ignoring the explicit role of phase shifts in causal alignment.


To address this difficulty, we propose **FACT** ( **F** requency- **A** daptive **C** omplex **T** ransformer), which
shifts interaction modeling from raw channels to specific frequency components. Unlike real-valued
approaches that struggle with phase alignment, FACT operates in the complex domain to explicitly
model both magnitude coherence Γ and phase offsets Φ. Our solution comprises three steps: (i)
a Dynamic Frequency-Band Decomposition (DynFBD) that adaptively isolates salient frequency
cells; (ii) a ChannelPriorMixer that leverages physical priors (Γ _,_ Φ) to guide interaction; and (iii)
a complex-valued fusion mechanism that aligns these priors with the representation. This design
ensures that interactions are physically grounded and robust to noise.


1


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|||<br> <br> <br> <br> <br> <br> <br> <br> <br>|||
||<br><br>|<br><br>|<br><br>||


Figure 1: Representative channel–frequency interactions: dynamic drift within a channel (left),
same-frequency coordination/antagonism (middle), and cross-frequency modulation/triggering
(right, e.g., a sudden cold snap inducing low-frequency heating demand).


- To establish a frequency-level interaction paradigm, we treat the channel–frequency cell as the
basic unit and design a sparse token pipeline (DynFBD + selector) to suppress noisy bands while
preserving physically meaningful signals.


- We introduce ChannelPriorMixer and adaptive fusion to leverage magnitude/phase-aware priors.
By grounding the interaction mechanism directly in physical properties (coherence Γ and phase
Φ), this design provides intrinsic interpretability, enabling users to trace frequency selection and
channel coupling patterns regardless of the chosen backbone.


- Functioning as a model-agnostic plug-in, FACT separates the Frequency-Aware Interaction Module from the representation encoder. This design explicitly prepares frequency-aligned features
and can be plugged into diverse backbones (Transformer/MLP/Linear), yielding consistent improvements across datasets compared to raw-channel mixing.


We validate these claims through comprehensive experiments: ablations on each component, regularization sweeps, and interpretability visualizations. Results demonstrate positive correlation between our interpretability metrics and accuracy, and consistent gains across backbones. Details are
provided in Section 5.


2 RELATED WORK


2.1 CHANNEL INTERACTION MODELLING


Early multivariate forecasting adopted RNN/CNN backbones with local dependencies (Hochreiter
& Schmidhuber, 1997; Bai et al., 2018), later extended by graph and multi-task formulations that
encode handcrafted adjacencies (Wu et al., 2020; 2021b; Cui et al., 2021). Transformers broaden
the receptive field (Vaswani et al., 2017; Zhou et al., 2021; Wu et al., 2021a; Zhou et al., 2022), but
how to model variable interactions remains contentious. Channel-independent (CI) designs (e.g.,
PatchTST, iTransformer) favor per-channel tokenization for robustness to noise/drift (Nie et al.,
2023; Liu et al., 2023); some even argue high-amplitude frequencies dominate prediction (Dai et al.,
2024; Xu et al., 2024). Channel-dependent (CD) methods (Crossformer, CARD, SOFTS, TimePro,
DUET) reintroduce interactions via cross-dimension routes, alignment-aware attention, global cores
or routing/clustering (Zhang & Yan, 2023; Wang et al., 2023; Han et al., 2024; Ma et al., 2025; Qiu
et al., 2025). Recent works like TimeFilter and TQN also explore advanced filtering mechanisms (Hu
et al., 2025; Lin et al., 2025), yet they largely rely on spatial-temporal graph filtrations. In contrast,
FACT adopts a pure frequency-domain approach to decouple fine-grained interactions. CI may
discard genuine couplings; CD often mixes signals coarsely and is sensitive to noise—motivating
frequency-aware, fine-grained priors as a middle ground.


2.2 TIME–FREQUENCY METHODS AND PHYSICAL PRIORS


Spectral approaches provide efficiency but typically treat amplitude as the sole carrier of information, whereas phase determines temporal alignment/lag and spatial shift. TimeMixer/TimeMixer++


2


mix frequency bands for long contexts yet collapse phase cues into shared representations (Wang
et al.; 2025). FredFormer and TSMixer refine spectra via normalization or MLP mixing, but channel
fusion remains entangled and phase alignment implicit (Piao et al., 2024; Ekambaram et al., 2023).
FreTS/FITS recalibrate responses (Yi et al., 2023a; Xu et al., 2024), yet they average across channels
and cannot reveal which variable drives a specific band or how cross-frequency triggering unfolds.
A complementary line emphasizes that spectral components should not be treated uniformly: FreDF
shows frequency utility is scenario-dependent and benefits from dynamic fusion (Zhang et al., 2024);
periodicity decoupling highlights the role of high-frequency harmonics beyond mere noise (Dai
et al., 2024). These observations motivate modelling interactions at the channel–frequency cell
with explicit magnitude/phase priors and channel-specific reweighting—precisely what FACT operationalizes. Beyond accuracy, recent work values robustness and interpretability. CI strategies
offer stability but little diagnosis (Han et al., 2023); CD designs (SOFTS/CARD) balance the two
via global cores or alignment penalties (Han et al., 2024; Wang et al., 2023). FACT inherits spectral
efficiency and contributes a physically grounded, fine-grained interaction paradigm that plugs into
diverse backbones.


3 PRELIMINARIES


**Problem Formulation.** Let **X** = _{_ **x** 1 _, . . .,_ **x** _L}_ _∈_ R _[L][×][C]_ represent the historical multivariate time
series with lookback window _L_ and _C_ channels. The objective is to predict the future sequence
**Y** = _{_ **x** _L_ +1 _, . . .,_ **x** _L_ + _T } ∈_ R _[T][ ×][C]_ of length _T_ . This forecasting task can be formulated as learning
a mapping function _Fθ_ :
**Y** ˆ = _Fθ_ ( **X** ) _,_ _Fθ_ : R _[L][×][C]_ _→_ R _[T][ ×][C]_ _._ (1)

Our goal is to optimize the parameters _θ_ such that the predicted **Y** [ˆ] accurately approximates the
ground truth **Y**, capturing both intra-series temporal dynamics and inter-series channel dependencies.


**Frequency** **Domain** **Processing.** To capture global temporal patterns and periodic dependencies,
FACT operates in the frequency domain. We apply the real Fast Fourier Transform (rFFT) to the
input **X** along the time dimension:


**X** fft = _F_ rfft( **X** ) _∈_ C _[F][ ×][C]_ _,_ _F_ = _⌊L/_ 2 _⌋_ + 1 _._ (2)


Unlike methods that process real and imaginary parts separately, we maintain the complex representation in polar form to explicitly preserve physical semantics:


**X** fft( _f, c_ ) = _A_ ( _f, c_ ) _· e_ _[iθ]_ [(] _[f,c]_ [)] _,_ (3)


where _A_ ( _f, c_ ) _∈_ R _≥_ 0 denotes the amplitude (representing energy intensity), and _θ_ ( _f, c_ ) _∈_ [ _−π, π_ )
denotes the phase (representing temporal alignment). This decomposition serves as the foundation
for our physics-aware interaction modeling. Full derivations and additional notations are detailed in
Appendix F.


4 METHODOLOGY


FACT addresses the CI–CD dilemma by modelling interactions at the _channel–frequency_ level with
explicit magnitude/phase priors. We first outline the pipeline (Fig. 2), then introduce the key modules
and the training-time regularizers. Basic notation and operators are given in Section 3.


4.1 ARCHITECTURE AND COMPLEXITY OVERVIEW


Figure 2 overviews the pipeline: (i) RevIN normalization and rFFT transformation; (ii) Adaptive
Band Decomposition using Gaussian filters to generate frequency bands; (iii) Complex Linear Projection to create multi-scale tokens and extract mask/weight information; (iv) Feature Alignment
through cross-attention and gated networks; (v) Complex encoder with coherence ( _Lcoh_ ) and phase
( _Lphase_ ) regularization losses. Note that while Figure 2 depicts a Complex Transformer Encoder,
the core Frequency-Aware Interaction Module (steps ii-iv) is backbone-agnostic and can be coupled
with MLP or Linear encoders. A concise summary of the per-module complexity is provided in
Section 5.3 (Table 3).


3


Figure 2: Overall FACT pipeline: input sequences undergo RevIN normalization and rFFT transformation to frequency domain. Gaussian filters perform adaptive band decomposition generating
low/mid/high frequency bands, mask, and weight information. Complex linear projection creates
multi-scale tokens, followed by Feature Alignment using cross-attention with gated networks. The
encoder processes aligned features with coherence and phase regularization losses, finally recovering time-domain predictions through inverse operations.


Figure 3: Fixed frequency band division illustration: the frequency axis is divided into
low/medium/high three segments according to preset thresholds, each segment is compressed
through independent complex linear branches and then concatenated into unified token representation.


4.2 ADAPTIVE BAND DECOMPOSITION AND FREQUENCY SELECTION


**Rationale:** **From Static to Dynamic.** Multi-scale frequencies naturally correspond to seasonalities
and lags. A naive approach involves dividing the spectrum into low/mid/high bands using fixed
thresholds (see Fig. 3). While this provides a basic interaction unit, it suffers from two limitations:
(1) _Energy_ _Truncation_ : fixed boundaries may cut through high-energy peaks in diverse datasets
(e.g., solar vs. traffic), leading to information loss; (2) _Rigidity_ : fixed boundaries lack a mechanism
to dynamically re-weight frequency bands and require tedious manual tuning to adapt to different
dataset characteristics. To overcome this, we propose an Adaptive Band Decomposition (Fig. 4)
driven by learnable Gaussian filters. This design not only softly separates components to avoid
aliasing but also produces continuous masks that bridge the frequency frontend with downstream
attention modules.


We apply learnable Gaussian filters to each channel to obtain _Bf_ soft frequency bands. Crucially,
this process yields both the decomposed tokens **Z** and a set of soft masks **P** mask:


**Z** _i_ = ComplexLinear( **W** gauss _,i ⊙_ **X** fft) _,_ _i_ = 1 _, . . ., Bf_ _._ (4)


The resulting **P** mask and **P** weight are not merely outputs but serve as continuous gating priors injected into the Feature Alignment module (Section 4.5), creating a closed-loop feedback where the
model learns to emphasize key frequency bands end-to-end.


The softplus-constrained ( _µ, σ_ ) parameters are normalized within each band to obtain
( _B, C,_ bands _, F_ ) soft masks, which are point-wise multiplied with the original spectrum and pro

4


Figure 4: DynFBD’s learnable Gaussian filters: raw spectrum, ( _µ, σ_ ) trajectories, soft-band decomposition, and normalized filter shapes.


jected to ( _B, K,_ 3 _C_ ) via shared complex linear layers. Concurrently, the resulting masks and
weights are compressed into low-dimensional summaries **P** [proj] mask _[∈]_ [R] _[B][×][F][ ×][d][m]_ [and] **[P]** [proj] weight _[∈]_
R _[B][×][K][×][d][w]_, providing interpretable attention bias and gating priors. This soft division not only
enables smooth gradients but also forms a closed feedback loop with Feature Alignment, allowing
the model to emphasize key frequency bands early in training (see Fig. 4). Empirical results on
benchmarks like ETTh1 and ECL show that the Gaussian version reduces sMAPE by approximately
1 _._ 3% _∼_ 2 _._ 1% compared to fixed thresholds.


4.3 CHANNEL PRIOR MIXER


**Rationale.** Direct attention on high-dimensional channels is computationally expensive and prone
to noise. Moreover, real-valued attention struggles to capture phase-based lead-lag relationships.
The Channel Prior Mixer mitigates this by adopting a centralized aggregation-distribution strategy
in the complex domain. Specifically, we compute the amplitude coherence _γ_ = Corr( _|_ **X** fft _|_ ) and
phase difference _ϕ_ = Angle( **X** fft) across channels from the input spectrum, serving as the physical
ground truth. Based on these priors, we obtain the mixing matrix using learnable scalars _α, β_ and
temperature _τ_ :


where **M** mix _∈_ R _[C][×][C]_ . **I** is the identity matrix and _δ_ is a learnable bias to preserve self-channel
information. The mixed spectrum is interpolated with strength 0.1, and guided gating compresses
amplitudes to [0 _,_ 1].


4.4 ENCODER PLUGGABILITY


The frequency frontend outputs unified complex tokens, allowing flexibility in the encoder choice
based on computational budget: a Complex Transformer (optimal for large channel counts), a Complex MLP (linear cost in _BLd_ model _d_ ff ), or a single-layer Complex Linear (most lightweight). Full
comparisons are provided in the Appendix.


4.5 FEATURE ALIGNMENT


This module acts as the bridge that injects the physical priors (from Sec 4.3) into the representation stream. Tokens and the raw spectrum are typically misaligned in length and channels. Simple
concatenation can cause information leakage and ignore priors. To resolve this, we adopt complex cross-attention where the raw spectrum queries the tokens, while prior-driven gating and bias
highlight key bands and suppress noise.


This magnitude–phase pipeline (Fig. 5) allows Feature Alignment to gate strong or weak responses
based on amplitude while retaining phase delays, essential for identifying cross-channel lead–lag
relations. The module comprises three sub-pathways: (i) query/key projection splitting complex
inputs into real/imaginary parts; (ii) value projection preserving phase information; and (iii) a gating generator that learns injection strength and attention bias from mask/weight summaries. The
formulation is:


**Q** = **W** _Q_ [ _ℜ_ ( **X** fft); _ℑ_ ( **X** fft)] _,_ **K** = **W** _K_ [ _ℜ_ ( **Z** ); _ℑ_ ( **Z** )] _,_ **V** = ComplexLinear( **Z** ) _._ (6)


5


      - _αγ_ + _βϕ_
**M** mix = softmax
_τ_


+ _δ_ **I** _._ (5)


Figure 5: Complex feature handling: traditional real/imaginary split (top) vs. FACT’s magnitude–phase processing (bottom). Right: magnitude-softmax and unit-phase reconstruction for complex attention values.


Prior gating and bias are defined as


**G** = _σ_      - _Am_ ( **M** )� _⊙_ _σ_      - _Aw_ ( **W** )� _,_ **B** = _B_ ( **M** _,_ **W** ) _,_ (7)


where **M** _,_ **W** are projected summaries and _Am, Aw, B_ are linear mappings. The attention output is


_⊤_
**H** fused = Softmax� **QK** ~~_√_~~ + **B** �� **V** _⊙_ **G**       - _._ (8)

_d_


The result is residually interpolated with the original spectrum ( _α_ = 0 _._ 7) and normalized by ComplexLayerNorm. This design maintains _O_ ( _n_ heads _Kd_ [2] ) complexity while leveraging prior gating to
focus on key frequency bands early in training. Crucially, the cross-attention map ( **QK** _[⊤]_ ) in this
module serves as a direct visualization window, revealing how the model aggregates multi-scale
frequency tokens, thereby providing feature-level interpretability independent of the subsequent encoder backbone.


4.6 COMPLEX TRANSFORMER ENCODER


Following frequency-domain alignment, we employ a Complex Transformer Encoder to model longterm dependencies while preserving amplitude-phase information. The encoder consists of two
ComplexFullAttentionLayer layers:


**H** _ℓ_ +1 = ComplexLayerNorm� **H** _ℓ_ + ComplexMultiHeadAttn( **H** _ℓ,_ **H** _ℓ,_ **H** _ℓ_ )� _,_ (9)

**H** _ℓ_ +1 = ComplexLayerNorm� **H** _ℓ_ +1 + ComplexConv1d( **H** _ℓ_ +1)� _._ (10)


ComplexMultiHeadAttn reuses weights from Equation equation 6 with prior bias, and ComplexConv1d performs depthwise separable convolution to capture local smoothness. The output is
mapped back to C _[F][ ×][C]_, then recovered to time-domain predictions through irFFT and inverse normalization.


4.7 INTERPRETABILITY REGULARIZATION


To align the model with physical mechanisms during optimization, we impose constraints on cached
attention, gating, and priors. This avoids the ”train first, interpret later” disconnect. specifically, we
cache fusion representations **H** [ˆ], gating vectors **g**, mixing matrices **M** mix, and frequency-domain
phases. Averaging these over the frequency dimension yields amplitude correlations _γ_ ˆ and mean
phase differences ∆ _θ_ . These drive the coherence and phase regularizers:

[�]


                     -                      _L_ coh = _∥γ_ ˆ _−_ _γ∥_ [2] 2 _[,]_ _γ_ ˆ = corr _|_ **H** [ˆ] _|_ _,_ (11)


                -                 _L_ phase = 1 _−_ cos ∆� _θ −_ _ϕ_ _,_ (12)


6


Table 1: Multivariate Long-term Forecasting results with prediction lengths _H_ _∈_
_{_ 96 _,_ 192 _,_ 336 _,_ 720 _}_ and fixed lookback window length _L_ = 96. The results are taken from SOFTS
and iTransformer (Liu et al., 2023).

|Models FACT (ours)|Col2|Col3|SOFTS|iTransformer|PatchTST|TSMixer|Crossformer|TiDE|TimesNet|DLinear|SCINet|FEDformer|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Metric<br>MSE<br>MAE|Metric<br>MSE<br>MAE|Metric<br>MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|
|ETTm1|96<br>192<br>336<br>720|0.327*<br>**0**_._<br>0.376*<br>0.392<br>0.422<br>0.418<br>0.502<br>0.463|0.325<br>**0**_._<br>0.375<br>0.389<br>0.405<br>0.412*<br>**0**_._<br>**0**_._|0.334<br>0.368<br>0.377<br>0.391<br>0.426<br>0.420<br>0.491<br>0.459|0.329<br>0.365<br>0.380<br>0.394<br>**0**_._<br>**0**_._<br>0.475*<br>0.453*|**0**_._<br>0.363*<br>0.376*<br>0.392<br>0.407*<br>0.413<br>0.485<br>0.459|0.404<br>0.426<br>0.450<br>0.451<br>0.532<br>0.515<br>0.666<br>0.589|0.364<br>0.387<br>0.398<br>0.404<br>0.428<br>0.425<br>0.487<br>0.461|0.338<br>0.375<br>**0**_._<br>**0**_._<br>0.410<br>0.411<br>0.478<br>0.450|0.345<br>0.372<br>0.380<br>0.389<br>0.413<br>0.413<br>0.474<br>0.453*|0.418<br>0.438<br>0.439<br>0.450<br>0.490<br>0.485<br>0.595<br>0.550|0.379<br>0.419<br>0.426<br>0.441<br>0.445<br>0.459<br>0.543<br>0.490|
|ETTm1|Avg|0.407<br>0.409|**0**_._<br>**0**_._|0.407<br>0.410|0.396<br>0.406|0.398*<br>0.407|0.513<br>0.496|0.419<br>0.419|0.400<br>0.406|0.403<br>0.407|0.485<br>0.481|0.448<br>0.452|
|ETTm2|96<br>192<br>336<br>720|0.193<br>0.275<br>0.271<br>0.329<br>0.312<br>0.349<br>0.417<br>0.408|**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>0.319<br>0.352<br>**0**_._<br>**0**_._|**0**_._<br>0.264<br>0.250<br>0.309*<br>0.311*<br>0.348*<br>0.412<br>0.407|0.184<br>0.264<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>0.409*<br>0.402|0.182*<br>0.266<br>0.249*<br>0.309*<br>0.309<br>0.347<br>0.416<br>0.408|0.287<br>0.366<br>0.414<br>0.492<br>0.597<br>0.542<br>1.730<br>1.042|0.207<br>0.305<br>0.290<br>0.364<br>0.377<br>0.422<br>0.558<br>0.524|0.187<br>0.267<br>0.249*<br>0.309*<br>0.321<br>0.351<br>0.408<br>0.403*|0.193<br>0.292<br>0.284<br>0.362<br>0.369<br>0.427<br>0.554<br>0.522|0.286<br>0.377<br>0.399<br>0.445<br>0.637<br>0.591<br>0.960<br>0.735|0.203<br>0.287<br>0.269<br>0.328<br>0.325<br>0.366<br>0.421<br>0.415|
|ETTm2|Avg|0.298<br>0.340|**0**_._<br>**0**_._|0.288*<br>0.332*|**0**_._<br>**0**_._|0.289<br>0.333|0.757<br>0.610|0.358<br>0.404|0.291<br>0.333|0.350<br>0.401|0.571<br>0.537|0.305<br>0.349|
|ETTh1|96<br>192<br>336<br>720|0.384*<br>0.404<br>0.436*<br>0.436<br>0.480<br>0.458<br>0.504<br>0.486|0.381<br>**0**_._<br>0.435<br>0.431<br>0.480<br>**0**_._<br>0.499<br>0.488*|0.386<br>0.405<br>0.441<br>0.436<br>0.487<br>0.458<br>0.503*<br>0.491|0.394<br>0.406<br>0.440<br>0.435<br>0.491<br>0.462<br>**0**_._<br>**0**_._|0.401<br>0.412<br>0.452<br>0.442<br>0.492<br>0.463<br>0.507<br>0.490|0.423<br>0.448<br>0.471<br>0.474<br>0.570<br>0.546<br>0.653<br>0.621|0.479<br>0.464<br>0.525<br>0.492<br>0.565<br>0.515<br>0.594<br>0.558|0.384*<br>0.402*<br>0.436*<br>**0**_._<br>0.491<br>0.469<br>0.521<br>0.500|0.386<br>0.400<br>0.437<br>0.432*<br>0.481<br>0.459<br>0.519<br>0.516|0.654<br>0.599<br>0.719<br>0.631<br>0.778<br>0.659<br>0.836<br>0.699|**0**_._<br>0.419<br>**0**_._<br>0.448<br>**0**_._<br>0.465<br>0.506<br>0.507|
|ETTh1|Avg|0.451*<br>0.446|0.449<br>**0**_._|0.454<br>0.447|0.453<br>0.446|0.463<br>0.452|0.529<br>0.522|0.541<br>0.507|0.458<br>0.450|0.456<br>0.452|0.747<br>0.647|**0**_._<br>0.460|
|ETTh2|96<br>192<br>336<br>720|0.307<br>0.356<br>0.383<br>0.400*<br>0.422<br>0.430<br>0.422<br>0.442|0.297<br>0.347<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._|0.297<br>0.349*<br>0.380*<br>0.400*<br>0.428*<br>0.432*<br>0.427*<br>0.445*|**0**_._<br>**0**_._<br>0.376<br>0.395<br>0.440<br>0.451<br>0.436<br>0.453|0.319<br>0.361<br>0.402<br>0.410<br>0.444<br>0.446<br>0.441<br>0.450|0.745<br>0.584<br>0.877<br>0.656<br>1.043<br>0.731<br>1.104<br>0.763|0.400<br>0.440<br>0.528<br>0.509<br>0.643<br>0.571<br>0.874<br>0.679|0.340<br>0.374<br>0.402<br>0.414<br>0.452<br>0.452<br>0.462<br>0.468|0.333<br>0.387<br>0.477<br>0.476<br>0.594<br>0.541<br>0.831<br>0.657|0.707<br>0.621<br>0.860<br>0.689<br>1<br>0.744<br>1.249<br>0.838|0.358<br>0.397<br>0.429<br>0.439<br>0.496<br>0.487<br>0.463<br>0.474|
|ETTh2|Avg|0.383<br>0.407|**0**_._<br>**0**_._|0.383<br>0.407|0.385<br>0.410|0.401<br>0.417|0.942<br>0.684|0.611<br>0.550|0.414<br>0.427|0.559<br>0.515|0.954<br>0.723|0.437<br>0.449|
|ECL|96<br>192<br>336<br>720|0.146<br>0.241*<br>0.178<br>0.268<br>0.187*<br>0.280<br>**0**_._<br>**0**_._|**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>0.218<br>0.305|0.148*<br>0.240<br>0.162<br>0.253<br>**0**_._<br>**0**_._<br>0.225<br>0.317|0.164<br>0.251<br>0.173*<br>0.262*<br>0.190<br>0.279*<br>0.230<br>0.313*|0.157<br>0.260<br>0.173*<br>0.274<br>0.192<br>0.295<br>0.223<br>0.318|0.219<br>0.314<br>0.231<br>0.322<br>0.246<br>0.337<br>0.280<br>0.363|0.237<br>0.329<br>0.236<br>0.330<br>0.249<br>0.344<br>0.284<br>0.373|0.168<br>0.272<br>0.184<br>0.289<br>0.198<br>0.300<br>0.220*<br>0.320|0.197<br>0.282<br>0.196<br>0.285<br>0.209<br>0.301<br>0.245<br>0.333|0.247<br>0.345<br>0.257<br>0.355<br>0.269<br>0.369<br>0.299<br>0.390|0.193<br>0.308<br>0.201<br>0.315<br>0.214<br>0.329<br>0.246<br>0.355|
|ECL|Avg|0.179*<br>0.272*|**0**_._<br>**0**_._|0.178<br>0.270|0.189<br>0.276|0.186<br>0.287|0.244<br>0.334|0.251<br>0.344|0.192<br>0.295|0.212<br>0.300|0.268<br>0.365|0.214<br>0.327|
|Traffc|96<br>192<br>336<br>720|0.409*<br>0.273<br>0.427*<br>0.279*<br>0.465<br>0.294<br>0.512<br>0.315|**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._|0.395<br>0.268<br>0.417<br>0.276<br>0.433<br>0.283*<br>0.467<br>0.302*|0.427<br>0.272*<br>0.454<br>0.289<br>0.450*<br>0.282<br>0.484*<br>0.301|0.493<br>0.336<br>0.497<br>0.351<br>0.528<br>0.361<br>0.569<br>0.380|0.522<br>0.290<br>0.530<br>0.293<br>0.558<br>0.305<br>0.589<br>0.328|0.805<br>0.493<br>0.756<br>0.474<br>0.762<br>0.477<br>0.719<br>0.449|0.593<br>0.321<br>0.617<br>0.336<br>0.629<br>0.336<br>0.640<br>0.350|0.650<br>0.396<br>0.598<br>0.370<br>0.605<br>0.373<br>0.645<br>0.394|0.788<br>0.499<br>0.789<br>0.505<br>0.797<br>0.508<br>0.841<br>0.523|0.587<br>0.366<br>0.604<br>0.373<br>0.621<br>0.383<br>0.626<br>0.382|
|Traffc|Avg|0.453*<br>0.290|**0**_._<br>**0**_._|0.428<br>0.282|0.454<br>0.286*|0.522<br>0.357|0.550<br>0.304|0.760<br>0.473|0.620<br>0.336|0.625<br>0.383|0.804<br>0.509|0.610<br>0.376|
|Weather|96<br>192<br>336<br>720|0.167<br>0.213*<br>0.214<br>0.255*<br>0.273<br>0.299*<br>0.350<br>0.349|0.166<br>**0**_._<br>0.217<br>**0**_._<br>0.282<br>0.300<br>0.356<br>0.351|0.174<br>0.214<br>0.221<br>0.254<br>0.278<br>**0**_._<br>0.358<br>0.347|0.176<br>0.217<br>0.221<br>0.256<br>0.275*<br>**0**_._<br>0.352<br>**0**_._|0.166<br>0.210<br>0.215*<br>0.256<br>0.287<br>0.300<br>0.355<br>0.348*|**0**_._<br>0.230<br>**0**_._<br>0.277<br>**0**_._<br>0.335<br>0.398<br>0.418|0.202<br>0.261<br>0.242<br>0.298<br>0.287<br>0.335<br>0.351*<br>0.386|0.172<br>0.220<br>0.219<br>0.261<br>0.280<br>0.306<br>0.365<br>0.359|0.196<br>0.255<br>0.237<br>0.296<br>0.283<br>0.335<br>**0**_._<br>0.381|0.221<br>0.306<br>0.261<br>0.340<br>0.309<br>0.378<br>0.377<br>0.427|0.217<br>0.296<br>0.276<br>0.336<br>0.339<br>0.380<br>0.403<br>0.428|
|Weather|Avg|**0**_._<br>0.279*|0.255<br>**0**_._|0.258<br>**0**_._|0.256*<br>0.279*|0.256*<br>0.279*|0.259<br>0.315|0.271<br>0.320|0.259<br>0.287|0.265<br>0.317|0.292<br>0.363|0.309<br>0.360|
|Solar|96<br>192<br>336<br>720|**0**_._<br>0.236<br>0.233<br>0.269<br>**0**_._<br>0.275*<br>0.251*<br>0.280|0.200<br>**0**_._<br>**0**_._<br>**0**_._<br>0.243<br>**0**_._<br>**0**_._<br>**0**_._|0.203*<br>0.237*<br>0.233<br>0.261<br>0.248*<br>0.273<br>0.249<br>0.275|0.205<br>0.246<br>0.237<br>0.267*<br>0.250<br>0.276<br>0.252<br>0.275|0.221<br>0.275<br>0.268<br>0.306<br>0.272<br>0.294<br>0.281<br>0.313|0.310<br>0.331<br>0.734<br>0.725<br>0.750<br>0.735<br>0.769<br>0.765|0.312<br>0.399<br>0.339<br>0.416<br>0.368<br>0.430<br>0.370<br>0.425|0.250<br>0.292<br>0.296<br>0.318<br>0.319<br>0.330<br>0.338<br>0.337|0.290<br>0.378<br>0.320<br>0.398<br>0.353<br>0.415<br>0.356<br>0.413|0.237<br>0.344<br>0.280<br>0.380<br>0.304<br>0.389<br>0.308<br>0.388|0.242<br>0.342<br>0.285<br>0.380<br>0.282<br>0.376<br>0.357<br>0.427|
|Solar|Avg|**0**_._<br>0.265*|**0**_._<br>**0**_._|0.233*<br>0.262|0.236<br>0.266|0.260<br>0.297|0.641<br>0.639|0.347<br>0.417|0.301<br>0.319|0.330<br>0.401|0.282<br>0.375|0.291<br>0.381|
|Count (1st)|Count (1st)|3<br>2|16<br>23|2<br>2|5<br>7|1<br>0|3<br>0|0<br>0|1<br>2|1<br>0|0<br>0|3<br>0|
|Count (2nd)|Count (2nd)|8<br>5|12<br>4|8<br>11|1<br>6|2<br>2|0<br>0|0<br>0|1<br>2|1<br>2|0<br>0|0<br>0|
|Count (3rd)|Count (3rd)|8<br>7|0<br>2|8<br>9|6<br>6|6<br>3|0<br>0|1<br>0|4<br>3|0<br>2|0<br>0|0<br>0|


where _γ_ and _ϕ_ are derived from amplitude/phase priors. The total loss is _L_ = _L_ forecast + _λ_ coh _L_ coh +
_λ_ phase _L_ phase. By composing Adaptive Band Decomposition, channel priors, and regularized complex encoding, FACT achieves both high accuracy and physical interpretability.


5 EXPERIMENTS


5.1 DATASETS


We follow the public SOFTS benchmarks (Han et al., 2024): ETT (4 subsets), Traffic, Electricity,
Weather, Solar-Energy, and PEMS (4 subsets). These cover electricity, transportation and energy
scenarios with heterogeneous channels and sampling rates. Full statistics (channels, horizons, splits,
sampling) are provided in Appendix E (Table 8).


5.2 TRAINING AND IMPLEMENTATION SETTINGS


Key hyperparameters (optimizer, depth, hidden size, subset protocol) are summarized in Appendix
(Section C).


5.3 MAIN RESULTS AND ABLATION


We evaluate our method against a comprehensive set of baselines, including linear/MLP models
(DLinear, TSMixer, TiDE), Transformers (FEDformer, Stationary, PatchTST, Crossformer, iTransformer), and CNN-based approaches (SCINet, TimesNet). Following standard long-sequence protocols (Zhou et al., 2021; Liu et al., 2022), we fix the lookback window to _L_ = 96 and report
MSE/MAE across standard horizons. Full implementation details are provided in Appendix C.


7


Table 2: Multivariate Short-term Forecasting results on PEMS datasets with prediction lengths _H_ _∈_
_{_ 12 _,_ 24 _,_ 48 _,_ 96 _}_ and fxed lookback window length _L_ = 96.

|Models FACT (ours)|Col2|Col3|SOFTS|iTransformer|PatchTST|TSMixer|Crossformer|TiDE|TimesNet|DLinear|SCINet|FEDformer|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Metric<br>MSE<br>MAE|Metric<br>MSE<br>MAE|Metric<br>MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|MSE<br>MAE|
|PEMS03|12<br>24<br>48<br>96|**0**_._<br>0.166<br>0.084<br>0.191<br>0.127<br>0.234<br>0.191<br>0.296|0.064<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._|0.071<br>0.174<br>0.093<br>0.201<br>0.125*<br>0.236*<br>0.164<br>0.275|0.073<br>0.178<br>0.105<br>0.212<br>0.159<br>0.264<br>0.210<br>0.305|0.075<br>0.186<br>0.095<br>0.210<br>0.121<br>0.240<br>0.184<br>0.295|0.090<br>0.203<br>0.121<br>0.240<br>0.202<br>0.317<br>0.262<br>0.367|0.178<br>0.305<br>0.257<br>0.371<br>0.379<br>0.463<br>0.490<br>0.539|0.085<br>0.192<br>0.118<br>0.223<br>0.155<br>0.260<br>0.228<br>0.317|0.122<br>0.243<br>0.201<br>0.317<br>0.333<br>0.425<br>0.457<br>0.515|0.066*<br>0.172*<br>0.085*<br>0.198*<br>0.127<br>0.238<br>0.178*<br>0.287*|0.126<br>0.251<br>0.149<br>0.275<br>0.227<br>0.348<br>0.348<br>0.434|
|PEMS03|Avg|0.116<br>0.222*|**0**_._<br>**0**_._|0.113<br>0.221|0.137<br>0.240|0.119<br>0.233|0.169<br>0.281|0.326<br>0.419|0.147<br>0.248|0.278<br>0.375|0.114*<br>0.224|0.213<br>0.327|
|PEMS04|12<br>24<br>48<br>96|0.075*<br>0.179*<br>0.091<br>0.200*<br>0.118<br>0.233<br>0.162<br>0.280|0.074<br>**0**_._<br>0.088<br>0.194<br>0.110<br>0.219<br>0.135*<br>0.244|0.078<br>0.183<br>0.095<br>0.205<br>0.120<br>0.233<br>0.150<br>0.262|0.085<br>0.189<br>0.115<br>0.222<br>0.167<br>0.273<br>0.211<br>0.310|0.079<br>0.188<br>0.089*<br>0.201<br>0.111*<br>0.222*<br>0.133<br>0.247*|0.098<br>0.218<br>0.131<br>0.256<br>0.205<br>0.326<br>0.402<br>0.457|0.219<br>0.340<br>0.292<br>0.398<br>0.409<br>0.478<br>0.492<br>0.532|0.087<br>0.195<br>0.103<br>0.215<br>0.136<br>0.250<br>0.190<br>0.303|0.148<br>0.272<br>0.224<br>0.340<br>0.355<br>0.437<br>0.452<br>0.504|**0**_._<br>0.177<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._|0.138<br>0.262<br>0.177<br>0.293<br>0.270<br>0.368<br>0.341<br>0.427|
|PEMS04|Avg|0.111<br>0.223|0.102<br>0.208|0.111<br>0.221|0.145<br>0.249|0.103*<br>0.215*|0.209<br>0.314|0.353<br>0.437|0.129<br>0.241|0.295<br>0.388|**0**_._<br>**0**_._|0.231<br>0.337|
|PEMS07|12<br>24<br>48<br>96|**0**_._<br>**0**_._<br>**0**_._<br>**0**_._<br>0.098<br>0.196<br>0.133<br>0.227|0.057<br>0.152<br>0.073<br>0.173<br>**0**_._<br>**0**_._<br>**0**_._<br>**0**_._|0.067*<br>0.165<br>0.088*<br>0.190*<br>0.110*<br>0.215*<br>0.139*<br>0.245|0.068<br>0.163*<br>0.102<br>0.201<br>0.170<br>0.261<br>0.236<br>0.308|0.073<br>0.181<br>0.090<br>0.199<br>0.124<br>0.231<br>0.163<br>0.255|0.094<br>0.200<br>0.139<br>0.247<br>0.311<br>0.369<br>0.396<br>0.442|0.173<br>0.304<br>0.271<br>0.383<br>0.446<br>0.495<br>0.628<br>0.577|0.082<br>0.181<br>0.101<br>0.204<br>0.134<br>0.238<br>0.181<br>0.279|0.115<br>0.242<br>0.210<br>0.329<br>0.398<br>0.458<br>0.594<br>0.553|0.068<br>0.171<br>0.119<br>0.225<br>0.149<br>0.237<br>0.141<br>0.234*|0.109<br>0.225<br>0.125<br>0.244<br>0.165<br>0.288<br>0.262<br>0.376|
|PEMS07|Avg|0.090<br>0.185|**0**_._<br>**0**_._|0.101*<br>0.204*|0.144<br>0.233|0.112<br>0.217|0.235<br>0.315|0.380<br>0.440|0.124<br>0.225|0.329<br>0.395|0.119<br>0.234|0.165<br>0.283|
|PEMS08|12<br>24<br>48<br>96|**0**_._<br>0.173<br>**0**_._<br>**0**_._<br>**0**_._<br>0.241<br>0.265<br>0.307|**0**_._<br>**0**_._<br>0.104<br>0.201<br>0.164<br>0.253*<br>**0**_._<br>**0**_._|0.079*<br>0.182*<br>0.115*<br>0.219*<br>0.186*<br>**0**_._<br>0.221<br>0.267|0.098<br>0.205<br>0.162<br>0.266<br>0.238<br>0.311<br>0.303<br>0.318|0.083<br>0.189<br>0.117<br>0.226<br>0.196<br>0.299<br>0.266<br>0.331|0.165<br>0.214<br>0.215<br>0.260<br>0.315<br>0.355<br>0.377<br>0.397|0.227<br>0.343<br>0.318<br>0.409<br>0.497<br>0.510<br>0.721<br>0.592|0.112<br>0.212<br>0.141<br>0.238<br>0.198<br>0.283<br>0.320<br>0.351|0.154<br>0.276<br>0.248<br>0.353<br>0.440<br>0.470<br>0.674<br>0.565|0.087<br>0.184<br>0.122<br>0.221<br>0.189<br>0.270<br>0.236*<br>0.300*|0.173<br>0.273<br>0.210<br>0.301<br>0.320<br>0.394<br>0.442<br>0.465|
|PEMS08|Avg|0.147<br>0.230*|**0**_._<br>**0**_._|0.150*<br>0.226|0.200<br>0.275|0.165<br>0.261|0.268<br>0.307|0.441<br>0.464|0.193<br>0.271|0.379<br>0.416|0.158<br>0.244|0.286<br>0.358|
|Count (1st)|Count (1st)|6<br>3|7<br>9|0<br>1|0<br>0|0<br>0|0<br>0|0<br>0|0<br>0|0<br>0|4<br>3|0<br>0|
|Count (2nd)|Count (2nd)|3<br>7|8<br>6|2<br>2|0<br>0|2<br>0|0<br>0|0<br>0|0<br>0|0<br>0|0<br>1|0<br>0|
|Count (3rd)|Count (3rd)|1<br>2|1<br>1|8<br>5|0<br>1|2<br>2|0<br>0|0<br>0|0<br>0|0<br>0|4<br>5|0<br>0|


Tables 1 and 2 summarize the performance across 12 datasets. FACT exhibits distinct superiority on periodic datasets (e.g., Solar-Energy, Weather), validating that our complex-valued modeling
effectively captures physical phase shifts often overlooked by baselines. Compared to ChannelIndependent methods like PatchTST, FACT better recovers cross-channel coupling, leading to lower
errors on highly correlated data like ECL. On PEMS, it remains competitive against specialized
spatio-temporal models by inferring latent spatial dependencies via channel coherence, demonstrating robust generalization without pre-defined graph structures. While high-channel regimes like
Traffic indicate room for further scaling, the results collectively validate FACT’s effectiveness.


The results in Tables 1 and 2 demonstrate several key findings: (1) FACT achieves strong performance across diverse datasets, particularly excelling on Solar-Energy and Weather forecasting
tasks; (2) The frequency-domain approach proves effective for capturing temporal dependencies
while maintaining computational efficiency; (3) FACT’s interpretable design does not compromise
prediction accuracy, establishing a favorable trade-off between performance and explainability in
multivariate time series forecasting.


**Analysis of Domain Sensitivity.** FACT exhibits distinct superiority on Solar and Weather datasets
(ranking 1st in almost all metrics). This aligns with the physical nature of these domains: they
are dominated by strong periodicity and cross-channel phase shifts (e.g., solar irradiance delays
due to geographical longitude). FACT’s complex-valued modeling explicitly captures these phase
differences ( _ϕ_ ) and amplitude correlations ( _γ_ ) via the Channel Prior Mixer, offering an inductive bias
that real-valued models (like iTransformer) lack. Conversely, on datasets with irregular load spikes
(e.g., ETT), the advantage of frequency decomposition is less pronounced, though FACT remains
competitive.


**Efficiency** **and** **Ablation** **Analysis.** To further quantify the contribution of each module and the
efficiency of our design, we conducted detailed ablation studies on the Solar and Weather datasets.
We also explored alternative designs during development: notably, replacing our complex-valued
pipeline with a simple 2-channel real-valued concatenation resulted in inferior performance (approx. 5% degradation on Solar), as it failed to explicitly capture the phase-based lead-lag relationships critical for periodic data. As shown in Table 4, removing the Dynamic Frequency Band
Decomposition (DynFBD) leads to a performance drop, confirming the importance of frequency
disentanglement. Crucially, our Adaptive Fusion mechanism demonstrates superior scalability: on
the high-dimensional Electricity dataset (321 channels), it reduces computational overhead by over
**82%** (10.23s vs. 58.55s per epoch) compared to the concatenation baseline (FACT-concat), which
required a reduced batch size to avoid memory overflow. This validates the efficiency of our ”filterthen-fuse” strategy for large-scale applications.


We further analyze the theoretical complexity of each module in Table 3. FACT maintains a favorable efficiency profile; the channel mixer operates on top- _k_ bands with linear dependence on


8


channels _O_ ( _Ck_ ), while the adaptive fusion scales with _O_ ( _Kd_ [2] ), avoiding quadratic complexity
w.r.t sequence length _L_ .


Table 3: Time complexity overview of main modules (default _Bf_ = 3, _K_ = 128, top- _k_ =16).


Module Main Complexity Description


rFFT _O_ ( _LC_ log _L_ ) One rFFT per channel
DynFBD _O_ ( _Bf_ _KC_ ) Complex linear mapping, band projection
Channel Prior Mixer _O_ ( _Ck_ ) Aggregation after top- _k_ selection
Adaptive Fusion _O_ ( _n_ heads _Kd_ [2] ) Complex cross-attention on compressed
tokens
Complex Encoder _O_ ( _n_ layers _d_ [2] _K_ ) Two ComplexFullAttentionLayer layers


Table 4: Ablation Study on the Interpretability Subset of Solar and Weather Datasets. We compare
MSE performance and training Runtime (seconds per epoch). Note: The subset uses fewer samples
(4,096) for rapid validation, resulting in different MSE scales compared to the full-dataset Main
Results (Table 1).


Weather (21) Solar (137) Electricity (321)
Config
MSE Runtime (s) MSE Runtime (s) MSE Runtime (s)


FACT (concat) **0.737** 9.98 **0.501** 40.91 **0.453** 58.55
**FACT (fusion)** 0.783 10.51 0.523 17.17 0.468 10.23
w/o DynFBD 0.771 **6.35** 0.538 **10.43** 0.470 **5.88**
w/o Channel Mix 0.746 10.12 0.525 16.21 0.468 10.30
_λ_ = 0 _._ 02 0.744 10.49 0.522 16.99 0.468 10.24


5.4 INTERPRETABILITY VISUALIZATION


A key advantage of FACT is its transparency, which is intrinsic to the Interaction Module rather than
dependent on a specific backbone. We visualize the patterns learned by the frontend modules on the
Solar dataset in Figure 6.


The attention heatmaps (left), derived from the Adaptive Feature Fusion layer, reveal distinct
frequency-band activations, indicating that the model selectively attends to specific periodic components. Since this attention mechanism is part of the feature alignment process, such fine-grained
frequency interpretability is preserved even if the backend Encoder is replaced by an MLP.


The channel coherence map Γ (center) captures the physical coupling between solar stations, aligning with geographical proximity. Guided gating trajectories (right) show how the model dynamically adjusts the importance of frequency bands during training, effectively filtering noise. These
visualizations collectively demonstrate that FACT’s explainability is rooted in its frequency-aware
interaction design.


5.5 REGULARIZATION IMPACT


We investigate the impact of the regularization weight _λ_ (where _λ_ coh = _λ_ phase = _λ_ ) on the Weather
dataset. As shown in Table 5, increasing the regularization strength from the default _λ_ = 0 _._ 01 to
_λ_ = 0 _._ 02 leads to a significant improvement in MSE (from 0.783 to 0.744). This indicates that
stronger enforcement of physical constraints (coherence and phase) can help the model generalize
better by pruning spurious correlations.


Table 5: Sensitivity analysis of regularization weight _λ_ on Weather dataset (Interpretability Subset).


_λ_ MSE Runtime (s)


0.01 (Default) 0.783 10.51
0.02 **0.744** **10.49**


9


Figure 6: Interpretability on Solar: (Left) Attention heatmap showing frequency selection; (Center)
Learned Amplitude Coherence Γ; (Right) Gating trajectories over training steps.


5.6 MODEL GENERALIZABILITY


Table 6: Model Generalizability: Performance and efficiency of FACT with different backbones
( _L_ = 96 _, T_ = 96). Lightweight backends (MLP/Linear) achieve comparable accuracy with significant speedups.


Dataset Backbone MSE MAE Time (s/epoch) Speedup


To verify the plug-in capability of our frequency frontend (Interaction Module), we evaluated three
backends: Complex Transformer, Complex MLP, and Complex Linear. As shown in Table 6, replacing the heavy Transformer encoder with lightweight MLP or Linear layers results in only a marginal
performance drop (e.g., _<_ 5% MSE increase on Electricity) while delivering up to **2.3** _×_ training
speedup. On ETTh1, the FACT+MLP variant also achieved a competitive MSE of 0.456. This confirms that FACT’s core benefits stem primarily from the frequency-aware interaction layer, which
successfully disentangles signals for _any_ backbone.


6 CONCLUSION


We propose FACT to resolve the tension between noise suppression and information preservation in multivariate time series forecasting by elevating interaction modeling from raw channels
to fine-grained frequency components. By integrating Dynamic Frequency Band Decomposition
with complex-valued, prior-guided interaction mechanisms, FACT effectively disentangles meaningful signals from noise while enforcing intrinsic interpretability through physical constraints. Extensive experiments validate FACT as a model-agnostic plug-in that yields consistent performance
gains across diverse backbones (Transformer, MLP, Linear). While the current quadratic complexity
poses scaling challenges for ultra-high-dimensional data, future integration with sparse attention or
patching mechanisms promises to extend FACT’s applicability, establishing a robust foundation for
efficient, physically grounded forecasting systems. We believe this direction provides a new perspective for building efficient and interpretable time series systems in the future, and look forward
to further validating its potential on larger-scale data and richer tasks.


10


**Electricity**


**Solar**


Transformer **0.145** **0.243** 99.37 1.0 _×_
MLP 0.153 0.252 45.72 2.17 _×_
Linear 0.155 0.254 **43.14** **2.30** _×_


Transformer **0.192** **0.236** 74.59 1.0 _×_
MLP 0.198 0.249 43.39 1.72 _×_
Linear 0.211 0.264 **39.84** **1.87** _×_


REFERENCES


Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional
and recurrent networks for sequence modeling, 2018. CoRR, abs/1803.01271, 2018.


Yue Cui, Kai Zheng, Dingshan Cui, Jiandong Xie, Liwei Deng, Feiteng Huang, and Xiaofang Zhou.
Metro: A generic graph neural network framework for multivariate time series forecasting, 2021.
Proc. VLDB Endow., 15 (2): 224–236, 2021.


Tao Dai, Beiliang Wu, Peiyuan Liu, Naiqi Li, Jigang Bao, Yong Jiang, and Shu-Tao Xia. Periodicity decoupling framework for long-term series forecasting. In _International_ _Conference_ _on_
_Learning_ _Representations_ _(ICLR)_, 2024. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=OpW8q3K45D)
[OpW8q3K45D.](https://openreview.net/forum?id=OpW8q3K45D)


Vijay Ekambaram, Arindam Jati, Nam Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam.
Tsmixer: Lightweight mlp-mixer model for multivariate time series forecasting. In _Proceedings_
_of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_, pp. 459–469,
2023.


Lu Han, Han-Jia Ye, and De-Chuan Zhan. The capacity and robustness trade-off: Revisiting the channel independent strategy for multivariate time series forecasting, 2023. CoRR,
abs/2304.05206, 2023.


Lu Han, Xu-Yang Chen, Han-Jia Ye, and De-Chuan Zhan. Softs: Efficient multivariate time series
forecasting with series-core fusion. _arXiv preprint arXiv:2404.14197_, 2024.


Sepp Hochreiter and J¨urgen Schmidhuber. Long short-term memory, 1997. Neural computation, 9
(8): 1735–1780, 1997.


Yifan Hu, Guibin Zhang, Peiyuan Liu, Disen Lan, Naiqi Li, Dawei Cheng, Tao Dai, Shu-Tao Xia,
and Shirui Pan. Timefilter: Patch-specific spatial-temporal graph filtration for time series forecasting. In _Forty-second International Conference on Machine Learning_, 2025.


Shengsheng Lin, Haojun Chen, Haijie Wu, Chunyun Qiu, and Weiwei Lin. Temporal query network
for efficient multivariate time series forecasting. In _Forty-second_ _International_ _Conference_ _on_
_Machine Learning_, 2025.


Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, and Qiang Xu. Scinet:
Time series modeling and forecasting with sample convolution and interaction. In _Advances_ _in_
_Neural Information Processing Systems_, 2022.


Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long.
itransformer: Inverted transformers are effective for time series forecasting. _arXiv_ _preprint_
_arXiv:2310.06625_, 2023.


Xiaowen Ma, Zhen-Liang Ni, Shuai Xiao, and Xinghao Chen. Timepro: Efficient multivariate
long-term time series forecasting with variable-and time-aware hyper-state. In _Forty-second In-_
_ternational Conference on Machine Learning_, 2025.


Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth
64 words: Long-term forecasting with transformers. In _International_ _Conference_ _on_ _Learning_
_Representations_, 2023.


Xihao Piao, Zheng Chen, Taichi Murayama, Yasuko Matsubara, and Yasushi Sakurai. Fredformer:
Frequency debiased transformer for time series forecasting. In _Proceedings_ _of_ _the_ _30th_ _ACM_
_SIGKDD conference on knowledge discovery and data mining_, pp. 2400–2410, 2024.


Xiangfei Qiu, Xingjian Wu, Yan Lin, Chenjuan Guo, Jilin Hu, and Bin Yang. Duet: Dual clustering enhanced multivariate time series forecasting. In _Proceedings_ _of_ _the_ _31st_ _ACM_ _SIGKDD_
_Conference on Knowledge Discovery and Data Mining V. 1_, pp. 1185–1196, 2025.


11


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von
Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), _Ad-_
_vances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, volume 30. Curran Associates, Inc.,
2017. URL [https://proceedings.neurips.cc/paper_files/paper/2017/](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
[file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)


Shiyu Wang, Haixu Wu, Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma, James Y Zhang, and
JUN ZHOU. Timemixer: Decomposable multiscale mixing for time series forecasting. In _The_
_Twelfth International Conference on Learning Representations_ .


Shiyu Wang, Jiawei Li, Xiaoming Shi, Zhou Ye, Baichuan Mo, Wenze Lin, Shengtong Ju, Zhixuan
Chu, and Ming Jin. Timemixer++: A general time series pattern machine for universal predictive
analysis. In _ICLR_, 2025.


Xue Wang, Tian Zhou, Qingsong Wen, Jinyang Gao, Bolin Ding, and Rong Jin. Make transformer
great again for time series forecasting: Channel aligned robust dual transformer, 2023. CoRR,
abs/2305.12095, 2023a.


Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting, 2021a. In NeurIPS, pages 101–112,
2021a.


Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet:
Temporal 2d-variation modeling for general time series analysis. In _International Conference on_
_Learning Representations_, 2023.


Xinle Wu, Dalin Zhang, Chenjuan Guo, Chaoyang He, Bin Yang, and Christian S. Jensen. Autocts:
Automated correlated time series forecasting, 2021b. Proc. VLDB Endow., 15 (4): 971–983,
2021b.


Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Xiaojun Chang, and Chengqi Zhang. Connecting the dots: Multivariate time series forecasting with graph neural networks, 2020. In SIGKDD,
pages 753–763, 2020.


Zhijian Xu, Ailing Zeng, and Qiang Xu. FITS: Modeling time series with $10k$ parameters.
In _The_ _Twelfth_ _International_ _Conference_ _on_ _Learning_ _Representations_, 2024. URL [https:](https://openreview.net/forum?id=bWcnvZ3qMb)
[//openreview.net/forum?id=bWcnvZ3qMb.](https://openreview.net/forum?id=bWcnvZ3qMb)


Kun Yi, Qi Zhang, Wei Fan, Shoujin Wang, Pengyang Wang, Hui He, Ning An, Defu Lian, Longbing Cao, and Zhendong Niu. Frequency-domain mlps are more effective learners in time series
forecasting, 2023a. In NeurIPS, 2023.


Kun Yi, Qi Zhang, Wei Fan, Shoujin Wang, Pengyang Wang, Hui He, Ning An, Defu Lian, Longbing
Cao, and Zhendong Niu. Frequency-domain MLPs are more effective learners in time series
forecasting. In _Thirty-seventh Conference on Neural Information Processing Systems_, 2023b.


Xingyu Zhang, Siyu Zhao, Zeen Song, Huijie Guo, Jianqi Zhang, Changwen Zheng, and Wenwen
Qiang. Not all frequencies are created equal: Towards a dynamic fusion of frequencies in timeseries forecasting. In _Proceedings of the 32nd ACM International Conference on Multimedia_, pp.
4729–4737, 2024.


Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for
multivariate time series forecasting. In _International_ _Conference_ _on_ _Learning_ _Representations_,
2023.


Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.
Informer: Beyond efficient transformer for long sequence time-series forecasting. In _Proceedings_
_of the AAAI Conference on Artificial Intelligence_, pp. 11106–11115, 2021.


Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. Fedformer: Frequency
enhanced decomposed transformer for long-term series forecasting. In _International conference_
_on machine learning_, pp. 27268–27286. PMLR, 2022.


12


A SYMBOL EXTENSIONS AND INFERENCE PSEUDOCODE


To facilitate reproduction, we supplement the key steps of FACT inference based on the symbols in
the main text. The pseudocode mirrors the repository implementation, but we present it here using
conceptual module names for clarity:


1. Input tensor _X_ _∈_ R _[B][×][L][×][C]_ . If RevIN is enabled, execute _X_ _←_ RevIN( _X_ ) to obtain normalized representation; if reversible normalization is enabled, additionally cache mean and
variance.

2. Compute _X_ fft = _F_ rfft( _X_ ), and pass it through the dynamic frequency-band preprocessor to
obtain sparse frequency-domain tokens **Z**, mask priors **M**, and frequency-band weights _**ω**_ .

3. Apply the frequency selector to smooth these weights, producing low-dimensional mask and
weight summaries that will act as priors in later stages.

4. When channel mixing is enabled, estimate amplitude coherence _γ_ and phase priors _ϕ_, construct
mixing matrices and guided gating, and cache the resulting channel priors for regularization
use.

5. Activate Adaptive Feature Fusion to re-weight frequency-domain representations through complex cross-attention informed by the aforementioned priors; otherwise, directly reuse the mixed
spectrum _X_ fft.

6. Transform features back to the time domain and feed them into the chosen complex encoder
(Transformer/MLP/Linear), obtaining prediction hidden states through the complex projection
layer.

7. If reversible normalization or RevIN reverse process is enabled, restore original scale at output
and extract the last _T_ step results.


B DATASET AND PREPROCESSING DETAILS


This paper follows the divisions published in SOFTS (Han et al., 2024), with related statistics in
Table 8. Due to size limitations, the anonymous code package only includes Solar-137 examples.
The loader implementation in the supplementary code package follows the considerations below:


    - Data format: By default reads comma-separated floating-point text; for CSV files, skips the
header row.

    - Split strategy: Splits training/validation/test in chronological order according to 70/10/20,
and fits the normalizer on the training set to prevent information leakage.

    - Window parameters: the default window configuration [96 _,_ 48 _,_ 96] is maintained as in the
main experiments; the optional subsampling limit is set to 2000 rows for quick validation
and can be disabled to load complete files.

    - Temporal features: The anonymous release only supports the multivariate setting with standard time-encoding flags, consistent with Solar examples.


C TRAINING AND IMPLEMENTATION CONFIGURATION


Training uses the public entry point, with key hyperparameter default values as follows:


    - Optimizer uses AdamW with learning rate 5 _×_ 10 _[−]_ [4], combined with cosine annealing and
linear warmup.

    - Batch size 32, training epochs 10, early stopping patience 3. Interpretability subset scripts
reduce the number of training epochs to three to shorten visualization generation time.

    - Regularization coefficients _λ_ coh and _λ_ phase default to 0.01, and are skipped automatically
when channel priors are unavailable.

    - Complex attention defaults to two layers, hidden dimension 128, feedforward dimension
512; the token length produced by DynFBD is 128.


13


Table 7: FACT default hyperparameters (consistent with open-source implementation).

Module Key Parameters Default Values / Notes

RevIN use ~~r~~ evin, use complex revin, true, false, 1 _×_ 10 _[−]_ [5]
_ε_

Frequency Embedding _d_ model, per-channel scale/bias 128, learnable
BandPreprocessor _Bf_, _K_, mask ~~p~~ roj ~~d~~ im, 3, 128, 16, 8
weights ~~p~~ roj dim


Guided Gating gate ~~b~~ ias, gate scale 0.5, 0.5
Adaptive Feature Fusion _n_ heads, dropout, _α_ 8, 0.1, 0.7
Complex Encoder _e_ layers, _d_ ff 2 (main exp.) / 1 (interpretability subset), 512


Table 8: Dataset statistics (channels, horizons, splits, sampling rates).

|Dataset|Channels|Prediction Horizon H|Data Split (Train, Val, Test)|Sampling Rate|Domain|
|---|---|---|---|---|---|
|ETTh1, ETTh2<br>ETTm1, ETTm2<br>Weather<br>ECL<br>Traffc<br>Solar-Energy<br>PEMS03<br>PEMS04<br>PEMS07<br>PEMS08|7<br>7<br>21<br>321<br>862<br>137<br>358<br>307<br>883<br>170|_{_96_,_ 192_,_ 336_,_ 720_}_<br>_{_96_,_ 192_,_ 336_,_ 720_}_<br>_{_96_,_ 192_,_ 336_,_ 720_}_<br>_{_96_,_ 192_,_ 336_,_ 720_}_<br>_{_96_,_ 192_,_ 336_,_ 720_}_<br>_{_96_,_ 192_,_ 336_,_ 720_}_<br>_{_12_,_ 24_,_ 48_,_ 96_}_<br>_{_12_,_ 24_,_ 48_,_ 96_}_<br>_{_12_,_ 24_,_ 48_,_ 96_}_<br>_{_12_,_ 24_,_ 48_,_ 96_}_|(8545, 2881, 2881)<br>(34465, 11521, 11521)<br>(36792, 5271, 10540)<br>(18317, 2633, 5261)<br>(12185, 1757, 3509)<br>(36601, 5161, 10417)<br>(15617, 5135, 5135)<br>(10172, 3375, 3375)<br>(16911, 5622, 5622)<br>(10690, 3548, 3548)|Hourly<br>15min<br>10min<br>Hourly<br>Hourly<br>10min<br>5min<br>5min<br>5min<br>5min|Electricity<br>Electricity<br>Weather<br>Electricity<br>Traffc<br>Energy<br>Traffc<br>Traffc<br>Traffc<br>Traffc|


D ADDITIONAL EXPERIMENTAL RESULTS


Detailed interpretability metrics and regularization sensitivity statistics for Solar and Weather
datasets are provided with accompanying CSV files, with values consistent with the main text analysis and can be directly accessed in the accompanying CSV tables.


E DATASET STATISTICS


Full statistics of the reused benchmarks are reported in Table 8.


F PRELIMINARIES (FULL)


F.1 MULTIVARIATE LONG-TERM FORECASTING SETUP


Let the input sequence be **X** _∈_ R _[B][×][L][×][C]_ . The target is to predict **Y** _∈_ R _[B][×][T][ ×][C]_ with loss
_L_ forecast = _BCT_ 1 - _b,t,c_ [(] _[Y][b,t,c][ −]_ _[Y]_ [ˆ] _[b,t,c]_ [)][2][.]


F.2 REAL FAST FOURIER TRANSFORM AND COMPLEX REPRESENTATION


Stack the time series as **X** _∈_ R _[L][×][C]_, rFFT yields **X** fft = _F_ rfft( **X** ) _∈_ C _[F][ ×][C]_ with _F_ = _L/_ 2 + 1. For
frequency _f_ and channel _c_, **X** fft( _f, c_ ) = _A_ ( _f, c_ ) _e_ [i] _[θ]_ [(] _[f,c]_ [)] .


14


Channel Prior Mixer mixing topk, _τ_,
mixing strength, diag bias, _α_,
_β_


16, 1.0, 0.1, 0.2, learnable


F.3 DYNAMIC FREQUENCY-BAND DECOMPOSITION


For band _i_, the Gaussian weight is


exp                 - _−_ ( _f_ _−_ _µi_ ) [2] _/_ (2 _σi_ [2][)]                 _ωi_ ( _f_ ) =             - _Bj_ =1 _f_ [exp]             - _−_ ( _f_ _−_ _µj_ ) [2] _/_ (2 _σj_ [2][)]             - _[,]_ (13)


where _µi, σi_ are learnable and _Bf_ = 3 by default. Each band is compressed into _K_ -dimensional
tokens via complex linear projection.


F.4 FREQUENCY SELECTION AND PROJECTION


Given **Z** _∈_ C _[K][×][CB][f]_, the selector computes


               -                _**α**_ = softmax Mean _b_ ( _σ_ ( _|_ **W** 1 **Z** _|_ )) _,_ (14)


and projects it into mask/weight summaries **P** mask _∈_ R _[F][ ×][d][m]_ and **P** weight _∈_ R _[K][×][d][w]_ for subsequent
priors and attention bias.


F.5 CHANNEL CORRELATION AND PHASE PRIORS


Weighted amplitudes **A** _c,f_ = _w_ eff ( _f_ )( _A_ ( _f, c_ ) _−_ Mean _f A_ ( _f, c_ )) lead to


_γ_ = **AD** _[−]_ [1] **A** _[⊤]_ _,_ (15)


where **D** normalizes _γ_ _∈_ [ _−_ 1 _,_ 1] _[C][×][C]_ . Phase offsets summarize lead/lag:


sin _**θ**_ cos _**θ**_ _[⊤]_ _−_ cos _**θ**_ sin _**θ**_ _[⊤]_
_ϕ_ = (16)

max _|_ sin _**θ**_ cos _**θ**_ _[⊤]_ _−_ cos _**θ**_ sin _**θ**_ _[⊤]_ _|_ _[,]_


where sin _**θ**_ _,_ cos _**θ**_ _∈_ R _[C]_ are weighted by frequency.


F.6 COMPLEX OPERATORS AND GUIDED GATING


For **z** = **z** _r_ + i **z** _i_, a complex linear layer is


ComplexLinear( **z** ) = ( **W** _r_ **z** _r −_ **W** _i_ **z** _i_ ) + i( **W** _i_ **z** _r_ + **W** _r_ **z** _i_ ) _._ (17)


Guided gating compresses weighted amplitudes to [0 _,_ 1] via


**s** = Norm _c_ (Mean _f w_ eff ( _f_ ) _|_ **X** fft( _f, ·_ ) _|_ ) _,_ **g** = gate ~~b~~ ias + gate ~~s~~ cale _·_ clip( **s** _,_ 0 _,_ 1) _,_ (18)


which stabilizes optimization and supports interpretability regularization.


G ADDITIONAL VISUALIZATIONS


We provide additional interpretability visualizations for the Weather dataset in Figure 7, supplementing the Solar-137 analysis in the main text.


H REPRODUCTION WORKFLOW SUMMARY


All figures and tables can be automatically generated through the auxiliary scripts shipped with the
supplementary package. We keep the outline below at a high level and redact internal file names.


    - Main results: run the standard FACT training recipe on Solar with DynFBD, channel mixing, and adaptive fusion enabled.

    - Interpretability subset: execute the lightweight configuration on curated Solar/Weather subsets (4,096 samples, _e_ layers = 1, 3 epochs).

    - Attention heatmaps: post-process cached interpretability tensors to render attention and
gating visualizations for Solar.


15


Figure 7: Attention, Γ heatmaps and gating trajectories for Weather interpretability subset.


    - Physical alignment: consolidate interpretability caches to compute Γ _/_ Φ alignment statistics
against meteorological variables.

    - Regularization analysis: sweep coherence/phase regularization coefficients and export the
summarized metrics.


The README in the supplementary scripts directory provides dataset-specific parameter examples
that extend to domains such as Traffic and ECL.


I REPRODUCIBILITY CHECKLIST


High-level command reference for reproducing the main results and analyses:


    - Main results: run the standard FACT training recipe with DynFBD, channel mixing, and
adaptive fusion enabled.

    - Interpretability subset: execute the lightweight configuration on Solar/Weather (4,096 samples, one encoder layer, three epochs).

    - Heatmaps: post-process cached tensors to render attention and gating visualizations.

    - Physical alignment: compute alignment between Γ _/_ Φ and meteorological variables.

    - Regularization: sweep _λ_ coh/ _λ_ phase and export summary tables.


ETHICS STATEMENT


This research complies with the ICLR Code of Ethics. All experiments are based on public benchmarks.


The release and use of publicly available datasets respect their respective licenses and intended
purposes. The proposed methodology is developed for scientific research and carries minimal risk
of harmful applications. We acknowledge the broader concerns of fairness and bias in machine
learning models, and we have taken steps to evaluate model robustness and to mitigate unintended
discrimination.


No sensitive personal attributes were included in training or evaluation. This work does not involve
conflicts of interest, unauthorized sponsorship, or activities that may compromise privacy, security,
or research integrity.


REPRODUCIBILITY STATEMENT


To facilitate the verification and extension of our work, we provide the following resources:


    - **Code** **Availability:** The complete implementation is available at: [https://](https://anonymous.4open.science/r/FACT)
[anonymous.4open.science/r/FACT](https://anonymous.4open.science/r/FACT)


16


- **Datasets:** All experiments are based on public benchmarks (ETT, Traffic, Electricity,
Weather, Solar-Energy).


    - **Key Components:** The core innovations include:


**–** Dynamic Frequency-Band Decomposition (DynFBD)

**–** ChannelPriorMixer for amplitude-phase priors

**–** Complex cross-attention fusion


    - **Training Setup:** We employ standard hyperparameters (learning rate=5e-4, batch size=32)
alongside coherence and phase regularization.


We confirm that all reported results can be reproduced with minimal error using the provided resources and configuration.


LLM USAGE


Large Language Models (LLMs) were used exclusively for polishing the language and writing of
this manuscript. The LLM contributed neither to the research conception nor to the core intellectual
content. We bear full responsibility for the work presented herein.


17