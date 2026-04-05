# TIME SERIES AS VIDEOS: SPECTRO-TEMPORAL GEN## ERATIVE DIFFUSION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Generative modeling of multivariate time series is challenged by properties such
as non-stationarity, intricate cross-channel correlations, and multi-scale temporal
dependencies. Existing diffusion models for this task mainly operate directly in
the time-domain, employing architectures that are not designed to capture complex spectral dynamics. Conversely, methods that transform sequences into static
images collapse the temporal axis, precluding the use of models designed for spatiotemporal dynamics. This paper argues for a new, unifying paradigm: reframing time series as videos. To this aim, we introduce Spectro-Temporal Diffusion
(ST-Diff), a framework that leverages the Short-Time Fourier Transform (STFT)
to convert a multivariate time series into a time-frequency video tensor. In this
representation, frequency and covariate axes form the spatial dimensions of each
frame, while the temporal evolution of the frequency spectrum is explicitly preserved. To capitalize on this novel structure, we design a custom video diffusion model specifically to leverage the spectro-temporal dynamics - the evolution
of frequency components over time. Through extensive empirical evaluation on
standard benchmarks, we demonstrate that the novel time-series-as-videos representation, together with its tailored architecture, allows ST-Diff to establish a new
state-of-the-art in unconditional time series generation. We argue that this timeseries-as-video paradigm has significant potential to advance a broad spectrum of
sequence modeling tasks beyond unconditional time-series generation.


1 INTRODUCTION


Generative modeling of multivariate time series is a fundamental problem in machine learning with
applications in financial simulation, climate forecasting, and privacy-preserving medical data Yoon
et al. (2019); Esteban et al. (2017), among others. A core technical challenge is to generate synthetic samples that are statistically indistinguishable from real data, capturing not only the marginal
distributions of variables but also their complex temporal dynamics. Real-world time-series are frequently characterized by properties such as non-stationarity, long-range dependencies, multi-scale
periodicities, and aperiodic events, which makes their generation a particularly challenging task.


The recent success of diffusion models has driven a new wave of research in time series generation.
A significant fraction of this work operates directly in the time domain, employing architectures like
Recurrent Neural Networks (RNNs) or Transformers as the denoising backbone Rasul et al. (2021);
Tashiro et al. (2021). While effective, RNN-based models often struggle to capture very long-range
dependencies Bengio et al. (1994), while Transformer-based approaches, despite their power, may
not possess the ideal inductive bias for modeling the nature of time series and can be computationally
expensive for very long sequences Zeng et al. (2023).


An alternative line of work reframes the problem by transforming time series into static images,
leveraging powerful computer vision architectures Wang & Oates (2015); Naiman et al. (2024).
Techniques like delay embedding, Gramian angular fields and short-term Fourier transform map a
sequence to a 2D matrix, enabling the use of state-of-the-art image diffusion models. This approach,
however, collapses the temporal dimension into a spatial one. As a result, architectures designed to
process spatiotemporal data cannot be used. This limitation motivates a key question: _Is it possible_
_to design a time-series representation that reveals its internal frequency structure while preserving_
_its native, explicit temporal axis, in order to leverage specialized spatiotemporal architectures?_


1


In this paper, we argue that the optimal representation for this task is not a static, 2D image, but a
3D video. Consequently, we introduce a new paradigm that treats time series generation as a task in
the video domain. Our method uses the short-time Fourier transform (STFT), a central tool in signal
processing, to convert a multivariate time series into an evolving time-frequency video tensor. In
this representation, each frame is a matrix where one axis corresponds to frequency components and
the other to the covariates. Crucially, the temporal evolution of the time-series frequency content is
explicitly maintained along the video time axis. This transformation allows for the application of
customized versions of _video diffusion models_, which are architecturally suited to learn how spatial
patterns - in our case, frequency spectra in particular - evolve over time.


We present Spectro-Temporal Diffusion (ST-Diff), a generative diffusion framework that leverages
this novel paradigm. The ST-Diff pipeline consists of three main steps: an invertible STFT-based
mapping from the time series to a video tensor, a generative diffusion process on this video representation through a custom spectro-temporal model, and an inverse STFT to reconstruct the final time-domain signal. Our extensive experiments on public benchmarks show that this approach
establishes a new state-of-the-art for unconditional time series generation, outperforming existing
time-domain and image-based methods.


The contributions of this work are threefold:


1. We propose and formalize the treatment of time series generation as a video task, a method
that preserves the temporal dimension while enabling the use of spatiotemporal models.


2. We introduce ST-Diff, a framework that integrates the STFT with a spectro-temporal video
diffusion model to generate high-fidelity and dynamically consistent time series.


3. We empirically demonstrate that ST-Diff significantly outperforms prior state-of-the-art
diffusion models on standard unconditional generation tasks.


We believe this _time-series-as-video_ perspective offers a powerful and generalizable foundation that
has the potential to advance a wide array of time series tasks, from forecasting to anomaly detection.


2 RELATED WORKS


Our work is situated at the intersection of generative models for time series, time series data representation, and video diffusion models. We review key developments in these areas to contextualize
our contributions.


**Generative Models for Time Series** Prior to diffusion models, generative modeling for time series
was primarily advanced by Generative Adversarial Networks (GANs) and Variational Autoencoders
(VAEs). GAN-based models such as RCGAN Esteban et al. (2017) and TimeGAN Yoon et al. (2019)
employ an adversarial training objective, with TimeGAN notably incorporating a supervised loss to
better capture temporal correlations. VAEs, including models like TimeVAE Desai et al. (2021),
offer a stable, likelihood-based alternative and can incorporate interpretable latent spaces. Our work
leverages diffusion models, which have demonstrated superior sample fidelity and training stability
compared to these earlier approaches Yuan & Qiao (2024).


The application of Denoising Diffusion Probabilistic Models (DDPMs) to time series has largely
focused on models that operate directly on the raw signal. Initial works such as TimeGrad Rasul
et al. (2021) for forecasting and CSDI Tashiro et al. (2021) for imputation adapted the diffusion
process for conditional tasks, typically using RNN or Transformer-based networks for the denoising
step. For the category of unconditional generation, Diffusion-TS Yuan & Qiao (2024) represents
a milestone, and employs a decomposition architecture to explicitly model trend and seasonality
components. While Diffusion-TS uses a Fourier-based loss to enforce periodicity, its core diffusion
process remains in the time domain. This contrasts with our approach, where the time-frequency
representation is not a supervisory signal but the primary domain for the entire generative process.
Complementary to this line of work, Crabb´e et al. (2024) propose frequency diffusion models that
perform the entire generative process in the frequency domain, whereas our approach operates directly in the joint time–frequency plane, capturing temporal and spectral structures simultaneously.


2


**Time Series to Image Transformations** A parallel research direction involves transforming time
series into 2D image representations to leverage well established, powerful vision architectures. This
concept was explored using methods like gramian angular fields and recurrence plots Wang & Oates
(2015). The leading contemporary model in this paradigm is ImagenTime Naiman et al. (2024),
which uses invertible transforms such as delay embedding and STFT to encode a time series into a
single, 2D static image. A standard vision diffusion model is then trained on these images. While
this approach has proven highly effective, it effectively treats the the temporal axis as a spatial one.
The explicit temporal sequence is lost, precluding the use of architectures designed for spatiotemporal modeling. Our work addresses this limitation directly by proposing a video representation that
reveals the time series frequency structure without sacrificing its temporal dimension.


**Time-Frequency** **Representations** **and** **Video** **Generation** The Short-Time Fourier Transform
(STFT) is a central method in signal processing for obtaining a time-frequency representation, revealing the temporal evolution of a signal spectral content Allen & Rabiner (1977). This representation, visualized as a spectrogram, is foundational in audio generation Shen et al. (2018). Concurrently, video generation has seen rapid progress, with video diffusion models demonstrating the
ability to synthesize high-fidelity, temporally coherent video sequences Ho et al. (2022). To our
knowledge, our work is the first to systematically bridge these domains for general multivariate time
series generation. We argue that the video tensor derived from the STFT of a multivariate time series
is a more natural and informative representation than either the raw signal or a static image. Unlike Diffusion-TS, we model the dynamics of the spectrum itself. Unlike ImagenTime, we preserve
the temporal axis explicitly, obtaining a spatiotemporal representation that enables ST-Diff to use
spatiotemporal models to learn the evolution of a time series frequency components.


3 BACKGROUND


Our framework integrates two core techniques: the STFT for data representation and DDPMs
adapted for video generative diffusion. We briefly review these concepts and establish the notation used throughout this paper.


**Short-Time** **Fourier** **Transform** **(STFT)** The STFT maps a time-domain signal to a timefrequency representation describing the temporal evolution of its frequency content. Given a onedimensional discrete-time signal _x_ [ _n_ ] of length _L_, its discrete STFT, _X_ [ _m, k_ ], is a complex-valued
matrix computed as: _X_ [ _m, k_ ] = [�] _n_ _[L]_ =0 _[−]_ [1] _[x]_ [[] _[n]_ []] _[w]_ [[] _[n][ −]_ _[mH]_ []] _[e][−][j]_ [2] _[πkn]_ _N_, where _w_ [ _·_ ] is a window function

_w_ [ _·_ ] which can mitigate spectral leakage (e.g., Hann window), _m_ is the time frame index, and _k_ is
the discrete frequency index. The STFT is controlled by two main hyperparameters: the window
length _N_, which determines the trade-off between time and frequency resolution (resulting in an uncertainty principle); the hop length _H_ (step size between the start of consecutive windows), which
controls the temporal resolution of the representation. A critical property for our generative framework is the invertibility of the STFT. The original signal _x_ [ _n_ ] can be reconstructed from _X_ [ _m, k_ ]
via the inverse STFT (iSTFT), typically using an overlap-add synthesis method. This near-perfect
reconstruction ensures that samples generated in the time-frequency domain can be losslessly converted back to the time domain.


**Video** **Diffusion** **Models** DDPMs are a class of generative models that learn to reverse a fixed
noise-injection process, and can be adapted for video data. Let _V_ 0 _∈_ R _[T][ ×][C][×][H][×][W]_ be a clean
video tensor, where _T_ is the number of frames. The forward process, _q_, is a fixed Markov
chain that gradually adds Gaussian noise to the data over _T_ diff discrete timesteps: _q_ ( _Vt|Vt−_ 1) =
_N_ ( _Vt_ ; _[√]_ 1 _−_ _βtVt−_ 1 _, βt_ **I** ) where _{βt}_ _[T]_ _t_ =1 [diff] [is] [a] [predefined] [var][iance] [s][chedule][.] [It] [is] [possible] [to] [sam-]
ple _Vt_ at an arbitrary timestep _t_ in closed form: _Vt_ = _[√]_ _α_ ¯ _tV_ 0 + _[√]_ 1 _−_ _α_ ¯ _tϵ_, where _αt_ = 1 _−_ _βt_,
_α_ ¯ _t_ = [�] _i_ _[t]_ =1 _[α][i]_ [,] [and] _[ϵ]_ _[∼N]_ [(0] _[,]_ **[ I]** [)][.] [The] [generative] [model,] _[p][θ]_ [,] [learns] [to] [approximate] [the] [reverse]
process _p_ ( _Vt−_ 1 _|Vt_ ). This is achieved by training a neural network _ϵθ_ ( _Vt, t_ ) to predict the noise component _ϵ_ from the noisy input _Vt_ at timestep _t_ . The network is optimized with a mean-squared error
loss on the noise: _L_ = E _t,V_ 0 _,ϵ_ - _||ϵ −_ _ϵθ_ ( _[√]_ _α_ ¯ _tV_ 0 + _[√]_ 1 _−_ _α_ ¯ _tϵ, t_ ) _||_ [2][�] _._ For video data, the network _ϵθ_
is typically implemented as a spatiotemporal architecture, which model dependencies both within
and across video frames. Generation of a new video is performed by starting with a random noise
tensor _VT_ diff _∼N_ (0 _,_ **I** ) and iteratively applying the learned denoising function to sample _Vt−_ 1 from


3


_Vt_, until a clean sample _V_ 0 is produced. In our work, we apply this generative mechanism not to
natural videos of scenes, but to the time-frequency videos derived from time series data.


4 THE SPECTRO-TEMPORAL DIFFUSION (ST-DIFF) FRAMEWORK


Figure 1: Overview of the Spectro-Temporal Diffusion (ST-Diff) pipeline. For training (top), a multivariate time series is transformed into a spectro-temporal video tensor via trend-residual decomposition and the STFT. For sampling (bottom), the learned STDiff model generates a new tensor in
this domain, which is then converted back to a time series using the inverse STFT (iSTFT).


We introduce Spectro-Temporal Diffusion (ST-Diff), our proposed framework for multivariate time
series generation. The core of our method is to first transform the time series into a spectro-temporal
video representation via the STFT transformation and then apply a specialized video diffusion model
to generate samples directly in this domain. These are subsequently converted back to the time
domain using the inverse STFT (iSTFT). An overview of the full pipeline is illustrated in Figure 1.


4.1 FROM TIME SERIES TO SPECTRO-TEMPORAL VIDEO TENSORS


A multivariate time series is a tensor _X_ _∈_ R _[L][×][K]_, where _L_ is the sequence length and _K_ is the
number of covariates. Our transformation pipeline maps _X_ to a video tensor _V_ _∈_ R _[T][ ×][C][×][H][×][W]_ .
As shown in Fig. 1 and Fig. 2a, we start by decomposing each covariate channel _**x**_ _k_ _∈_ R _[L]_ into a
trend component _**x**_ _k,_ trend and a residual component _**x**_ _k,_ res, in order to handle non-stationarity taht is
common in real-world time series. We compute the trend using a simple exponential moving average
(EMA). This isolates the low-frequency, non-stationary behavior, leaving the residual component,
_**x**_ _k,_ res = _**x**_ _k −_ _**x**_ _k,_ trend, which is more suitable for spectral analysis, as the STFT is most effective on
quasi-stationary signals.


Then, we apply the STFT independently to each of the _K_ residual sequences, _**x**_ _k,_ res. This produces _K_ complex-valued time-frequency matrices, _{_ _**S**_ _k_ _∈_ C _[F][ ×][T]_ _}_ _[K]_ _k_ =1 [,] [where] _[F]_ [is] [the] [number] [of]
frequency bins and _T_ is the number of time frames, determined by the STFT hyperparameters (window size _N_ and hop length _H_ ). To form a real-valued tensor suitable for neural network processing,
we construct the final video tensor _V_, whose dimensions are: the temporal axis of the video corresponds to the STFT time frames, _T_ ; The height of each frame corresponds to the frequency bins,
_F_ ; The width of each frame corresponds to the covariates, _K_ ; Three channels ( _C_ = 3, with the
first two storing the real and imaginary parts of the STFT coefficients, Re( _**S**_ _k_ ) and Im( _**S**_ _k_ ), while
the third channel stores the trend component, _**x**_ _k,_ trend, which is broadcasted across the frequency
dimension and resampled to match the temporal dimension _T_ ). This process yields a final tensor
_V_ _∈_ R _[T][ ×]_ [3] _[×][F][ ×][K]_ . This representation explicitly preserves the temporal evolution of the signal
spectral content across all covariates, making it directly compatible with video generation models.


4.2 GENERATION AND INVERSE TRANSFORMATION


To generate a new time series, we first sample a noise tensor _VT_ diff _∼N_ (0 _,_ **I** ) and apply the reverse
diffusion process using the trained model _ϵθ_ to obtain a synthetic spectro-temporal video tensor _V_ gen
(see Fig. 1). This tensor is then inverted back to the time domain. The three channels of _V_ gen are


4


Multivariate Time

Series


Generated


separated to recover the real and imaginary parts of the STFT for each covariate, as well as the trend
components. The inverse STFT (iSTFT) is applied to the generated spectogram of each covariate to
reconstruct the residual signals, _**x**_ ˆ _k,_ res. Adding the generated trend back to the residual yields the
final time series for each covariate: _**x**_ ˆ _k_ = _**x**_ ˆ _k,_ res + _**x**_ ˆ _k,_ trend. This process yields the final synthetic
multivariate time series _**X**_ gen _∈_ R _[L][×][K]_ .


4.3 THE SPECTRO-TEMPORAL VIDEO DIFFUSION MODEL


Figure 2: Overview of the proposed STDiff data pipeline and core model architecture. Panel 2a:
The data ingestion pipeline, transforming a multivariate time series into a spectro-temporal video
tensor. Panel 2b: The main STDiff model. Panel 2c: The Spectro-temporal attention block, showing
the tri-axial factorized attention mechanism.


With the data represented as a spectro-temporal video tensor, we employ a DDPM _ϵθ_ for generation.
The architecture of _ϵθ_ is a key component of our framework, which factorizes attention across the
spatial, temporal, and covariate axes, with specific architectural biases for each of them. We outline
the key architectural components below.


**Anisotropic** **Patching** **and** **Spectro-Temporal** **Attention** **Biases** The input tensor frame, a frequency–covariate matrix of shape ( _F_ _×_ _K_ ), is first projected into a sequence of tokens. Unlike
vision transformers that employ isotropic patches (e.g., 16 _×_ 16), we adopt an _anisotropic_ patching
strategy: patches are aggregated along the frequency axis while preserving unit granularity along the
covariate axis, so as not to introduce arbitrary spatial correlations among covariates, which, unlike
in image data, we do not assume a priori.


The network backbone comprises a stack of _STDiff_ blocks (Fig. 2c), which apply attention sequentially along the three main temporal, frequency, and covariate axes. To encode domainspecific structure, we introduce two bias mechanisms. First, the covariate attention module incorporates a symmetric matrix _**B**_ _C_ _∈_ R _[K][×][K]_ into its attention logits, yielding attention scores
softmax( _**[QK]**_ ~~_√_~~ _dk_ _[T]_ + _**B**_ _C_ ) _**V**_ . This bias acts as a learnable prior over inter-covariate dependencies. Sec
ond, a frequency bias matrix _**B**_ _F_ _∈_ R _[F][ ′][×][F][ ′]_ (where _F_ _[′]_ denotes the number of frequency patches)
is analogously added to the frequency attention module, enabling the capture of structured relationships among spectral bands. Both bias matrices are initialized from empirical statistics of the
data. Specifically, _**B**_ _C_ is set to the empirical cross-correlation matrix of the STFT covariates, encoding static inter-variable dependencies intrinsic to the system. In parallel, _**B**_ _F_ is initialized from
the covariance of STFT log-magnitudes, thereby modeling spectral components that tend to co-vary
(e.g., fundamental frequencies and harmonics). Our biases encourage the model to respect domain


. . .


Predicted Noise


(trend)


Time Series


(a)


(b)


(c)


~~_√_~~ _dk_ + _**B**_ _C_ ) _**V**_ . This bias acts as a learnable prior over inter-covariate dependencies. Sec


5


relevant structural and spectral relationships (with a role akin to spatial locality in convolutions).
Crucially, this is well-aligned with the underlying data: the covariate axis represents an _unordered_
_set_ of variables with no notion of locality, while spectral dependencies are often highly non-local.


**Positional** **and** **Timestep** **Embeddings** We use Rotary Positional Embeddings (RoPE) Su et al.
(2024) to encode the relative positions of tokens along the temporal and frequency axes, as they
are suitable to capture the relative ordering without being constrained to a fixed maximum length.
The covariate positions, instead, are encoded using a learnable parameters vectors, due to inherently
non-ordered structure of the covariate axis. Standard sinusoidal embeddings are used to encode
the diffusion timestep _t_, which are then processed by a multi-layer perceptron (MLP) before being
incorporated into the network blocks. The timestep embedding is integrated into the transformer
blocks using an adaptive layer normalization scheme (adaLN-Zero) Peebles & Xie (2023).


5 EXPERIMENTS


We conduct a comprehensive set of experiments to evaluate the performance of ST-Diff for unconditional multivariate time series generation. Our evaluation is designed to assess distributional fidelity,
sample quality, and the preservation of temporal dynamics.


**Datasets** We evaluate our method on six publicly available benchmark datasets spanning diverse properties such as dimensionality, periodicity, and non-stationarity, consistent with prior
work (Naiman et al., 2024; Yuan & Qiao, 2024). The datasets are: Sines, synthetic sine waves
with varying frequencies and phases (a sanity check to test the model ability to capture fundamental periodic patterns); Stocks, daily stock prices exhibiting non-stationary stochastic behavior;
ETTh, electricity transformer temperature real-world data with strong periodic components; Energy, appliance energy consumption with multivariate correlations and noisy periodicity; MuJoCo,
high-dimensional physics simulator data capturing complex dynamics; and fMRI, high-dimensional
neural signals characterized by noise and correlations. Following standard evaluation protocols, all
datasets Following standard evaluation protocols, we use a sequence length of _L_ = 24 across all
datasets. To further assess model scalability, we additionally evaluate on the ETTh dataset with
longer sequence lengths of _L ∈_ 64 _,_ 128 _,_ 256.


**Evaluation Metrics** To assess generation quality, we use an established suite of quantitative and
qualitative metrics Yoon et al. (2019), all reported so that lower values indicate better performance.
The Discriminative Score is measured by training a GRU classifier to distinguish real from synthetic
data. The score is the absolute difference between the classifier accuracy on a held-out test set and
0.5 (random chance). A score near zero indicates that the generated samples are indistinguishable
from real ones. The Predictive Score evaluates the usefulness and the preservation of temporal
dynamics through the “Train on Synthetic, Test on Real” protocol, where a GRU one-step-ahead
forecaster trained on generated data is tested on real data and its Mean Absolute Error (MAE) is
reported. To capture cross-covariate structure, we report the Correlational Score, computed as the
mean absolute difference of the Pearson correlation matrices of the real and the generated dataset.
Finally, we include qualitative analyses, such as t-SNE projections and data density estimations to
compare distributional similarity, and comparisons of Auto-Correlation Function (ACF) and Power
Spectral Density (PSD) to evaluate temporal and spectral fidelity.


**Baselines** We compare ST-Diff against leading models and frameworks for time series generation: TimeGAN (Yoon et al., 2019), a GAN-based framework for sequential data; TimeVAE (Desai
et al., 2021), a VAE-based generative model; Diffusion-TS (Yuan & Qiao, 2024), a state-of-theart diffusion model operating directly in the time domain; and ImagenTime (Naiman et al., 2024), a
diffusion-based approach that maps time series into images. For all baselines, we report performance
from the original publications to ensure fair comparison.


**Implementation** **Details** ST-Diff is implemented in PyTorch. The denoising network _ϵθ_ corresponds to the spectro–temporal video diffusion transformer introduced in Sec. 4.3. To construct the
input representation the FFT size is scaled relative to the input duration as nfft = (seq ~~l~~ en _/_ 2) _−_ 1
with hop length set proportionally as _⌈_ nfft _/_ 4 _⌉_ . This normalization transforms variable-length time


6


Table 1: Comprehensive quantitative comparison for unconditional generation on standard short sequences (L=24). We report Context-Fid, Correlational, Discriminative and Predictive scores (‘lower
is better‘). **ST-Diff** sets a new state-of-the-art across the majority of metrics and datasets. The ’–’
symbol indicates that the metric was not reported in the original paper.


|Metric|Methods|Sines|Stocks|ETTh|MuJoCo|Energy|fMRI|
|---|---|---|---|---|---|---|---|
|Context-FID<br>Score<br>(Lower the Better)|TimeGAN<br>TimeVAE<br>ImagenTime<br>DiffusionTs<br>STDiff (ours)|0.101_±_.014<br>0.307_±_.060<br>–<br>0.006_±_.000<br>**0.004**_±_**.001**|0.103_±_.013<br>0.215_±_.035<br>–<br>0.147_±_.025<br>**0.040**_±_**.008**|0.300_±_.013<br>0.805_±_.186<br>–<br>0.116_±_.010<br>**0.050**_±_**.008**|0.563_±_.052<br>0.251_±_.015<br>–<br>0.013_±_.001<br>**0.010**_±_**.001**|0.767_±_.103<br>1.631_±_.142<br>–<br>0.089_±_.024<br>**0.025**_±_**.002**|1.292_±_.218<br>14.449_±_.969<br>–<br>0.105_±_.006<br>**0.099**_±_**.007**|
|Correlational<br>Score<br>(Lower the Better)|TimeGAN<br>TimeVAE<br>ImagenTime<br>DiffusionTs<br>STDiff (ours)|0.045_±_.010<br>0.131_±_.010<br>–<br>**0.015**_±_**.004**<br>**0.015**_±_**.005**|0.063_±_.005<br>0.095_±_.008<br>–<br>0.004_±_.001<br>**0.003**_±_**.003**|0.210_±_.006<br>0.111_±_.020<br>–<br>0.049_±_.008<br>**0.047**_±_**.006**|0.886_±_.039<br>0.388_±_.041<br>–<br>**0.193**_±_**.027**<br>0.199_±_.017|4.010_±_.104<br>1.688_±_.226<br>–<br>0.856_±_.147<br>**0.592**_±_**.013**|23.502_±_.039<br>17.296_±_.526<br>–<br>**1.411**_±_**.042**<br>1.661_±_.059|
|Discriminative<br>Score<br>(Lower the Better)|TimeGAN<br>TimeVAE<br>ImagenTime<br>DiffusionTs<br>STDiff (ours)|0.011_±_.008<br>0.041_±_.044<br>–<br>0.006_±_.007<br>**0.004**_±_**.005**|0.102_±_.021<br>0.145_±_.120<br>0.037_±_.006<br>0.067_±_.015<br>**0.015**_±_**.021**|0.114_±_.055<br>0.209_±_.058<br>–<br>0.061_±_.009<br>**0.005**_±_**.005**|0.238_±_.068<br>0.230_±_.102<br>0.007_±_.005<br>0.008_±_.002<br>**0.007**_±_**.005**|0.236_±_.012<br>0.499_±_.000<br>0.040_±_.004<br>0.122_±_.003<br>**0.009**_±_**.013**|0.484_±_.042<br>0.476_±_.044<br>–<br>0.167_±_.023<br>**0.021**_±_**.014**|
|Predictive<br>Score<br>(Lower the Better)|TimeGAN<br>TimeVAE<br>ImagenTime<br>DiffusionTs<br>STDiff (ours)|0.093_±_.019<br>0.093_±_.000<br>–<br>**0.093**_±_**.000**<br>0.186_±_.004|0.038_±_.001<br>0.039_±_.000<br>0.036_±_.000<br>0.036_±_.000<br>**0.033**_±_**.000**|0.124_±_.001<br>0.126_±_.004<br>–<br>0.119_±_.002<br>**0.119**_±_**.002**|0.025_±_.003<br>0.012_±_.002<br>0.033_±_.001<br>**0.007**_±_**.000**<br>**0.007**_±_**.000**|0.273_±_.004<br>0.292_±_.000<br>0.250_±_.000<br>0.250_±_.000<br>**0.211**_±_**.000**|0.126_±_.002<br>0.113_±_.000<br>–<br>0.099_±_.000<br>**0.077**_±_**.000**|


series into fixed-dimensional spectrograms, ensuring that the subsequent analysis is independent of
the original sequence length. A 75% overlap between analysis windows is employed, consistent
with the theoretical requirements for robust signal invertibility Griffin & Lim (1984).


The model is trained under the DDPM framework with _T_ diff = 1000 diffusion steps and a cosine
noise schedule. The training objective is the mean-squared error between the true and predicted
noise, as detailed in Sec. 3. To further improve the fidelity of generated samples, particularly in
capturing spectral characteristics critical to time-series data, we introduce a cross-covariance loss
applied directly to the Short-Time Fourier Transform (STFT) magnitudes. This loss quantifies the
discrepancy between normalized covariance matrices, thereby encouraging the covariance structure
of generated STFT magnitudes to align closely with that of the real data. Optimization is performed
using AdamW with a cosine annehaling scheduler for the learning rate, with a minimum learnining
rate of 1 _×_ 10 _[−]_ [6] and a maximum learning rate of 2 _×_ 10 _[−]_ [4] . The maximum number of epochs is
1000, but an early stopping mechanism has been implemented. For sample generation, we employ
the DDIM sampler (Song et al., 2022) with 200 steps, which accelerates inference while maintaining
sample fidelity. All experiments are conducted on a single NVIDIA A100 GPU.


5.1 EMPIRICAL RESULTS AND ANALYSIS


We present the empirical evaluation of ST-Diff, beginning with a quantitative comparison against
state-of-the-art baselines on standard benchmarks. We further investigate the scalability to longer
sequence lengths and complement these results with qualitative analyses of the generated samples.


5.1.1 SHORT-TERM UNCONDITIONAL GENERATION


Table 1 reports results for unconditional generation on sequences of length 24. We evaluate ST-Diff
against all baselines using four established metrics: Discriminative, Predictive, Correlational and
Context-FID scores, where lower values indicate better performance.


Across the majority of datasets and evaluation metrics, ST-Diff establishes a new state of the art,
achieving superior performance on 21 out of 24 metric–dataset combinations. The improvements
are especially pronounced on high-dimensional, real-world datasets such as ENERGY, MUJOCO, and
FMRI. On ENERGY and FMRI benchmarks in particular, ST-Diff delivers substantial reductions
in discriminative and predictive scores, highlighting its capacity to model intricate cross-channel
dependencies and non-trivial spectral evolutions, generating high-fidelity samples. Taken together,
the results provide strong empirical evidence that explicitly modeling spectro–temporal structure
constitutes a powerful inductive bias for complex multivariate time series.


7


(a) Sine (b) Stocks (c) ETTh (d) MuJoCo (e) Energy (f) fMRI


Figure 3: t-SNE (up), Kernel Density Estimation (bottom) of real vs generated samples of the 6 used
datasets for sequence length 24. In red the original time series, in blue the generated ones.


|Col1|Col2|Origin<br>Gener|al<br>ated|
|---|---|---|---|
|||||
|||||


|Col1|Col2|Origin<br>Gener|al<br>ated|
|---|---|---|---|
|||||
|||||


|Col1|Col2|Origin<br>Gener|al<br>ated|
|---|---|---|---|
|||||
|||||


|Col1|Col2|Origin<br>Gener|al<br>ated|
|---|---|---|---|
|||||
|||||


|Col1|Col2|Col3|Orig|inal|
|---|---|---|---|---|
|||Gen|Gen|erated<br>|
||||||
||||||


|Col1|Col2|Orig<br>Gen|inal<br>erated|
|---|---|---|---|
|||||
|||||


|Col1|Col2|Col3|Orig|inal|
|---|---|---|---|---|
|||Gen|Gen|erated<br>|
||||||
||||||


|Col1|Col2|Origi<br>Gene|nal<br>rated|
|---|---|---|---|
|||||
|||||


Figure 4: Temporal and spectral fidelity analysis on the ETTH dataset with sequence length 24.
The top row reports the Auto–Correlation Function (ACF), and the bottom row the average Power
Spectral Density (PSD), for real (blue, solid) and generated (red, dashed) samples across three representative covariates (HUFL, LUFL, MUFL, OT). The near-perfect overlap in the ACF plots and
the close alignment of the PSD curves (in particular at low frequencies) demonstrate that ST-Diff
faithfully reproduces both the temporal dependencies and the spectral characteristics of the original
process. Extended ACF and PSD results are provided in Appendix C.


**Qualitative** **Analysis** To complement the quantitative results, we provide qualitative visualizations. Figure 3 illustrates t-SNE embeddings and Kernel Density Estimation (KDE) of real and generated samples from all the datasets. The distribution of samples generated by ST-Diff closely aligns
with the manifold of the real data. In the top row, the t-SNE projections offer a low-dimensional
view of the high-dimensional time series, allowing a direct comparison between real (red) and synthetic (blue) distributions. Across all six datasets, the KDE curves of the generated samples (bottom
row) closely follow those of the real data showing the alignment of marginal distributions and further
evidence of the high generated sample fidelity achieved by ST-Diff.


To qualitatively assess directly temporal and spectral fidelity, we report a comparison of the average
Auto–Correlation Function (ACF) and Power Spectral Density (PSD) of real and generated samples
from the ETTH dataset (Fig. 4). The ACF plots (top row) show that ST-Diff accurately reproduces
the temporal structure of the original series, indicating that it learns underlying dynamics rather than
merely matching marginals. In the frequency domain, the PSD plots (bottom row) overall captures
both dominant peaks and spectral decay, in particular at low-frequency components, with some
slight difference in particular on high-frequency ones. Further results are reported in Appendix C.


5.1.2 LONG-TERM UNCONDITIONAL GENERATION


To assess the scalability of our approach, we evaluate performance on the ETTh datasets with extended sequence lengths of 64, 128, and 256, as summarized in Table 2. The findings unequivocally
demonstrate the superior scalability of ST-Diff, which not only outperforms all baselines across


8


1.0


0.5


0.0


Lag


1.0


0.5


0.0


Lag


1.0


0.5


0.0


Lag


1.0


0.5


0.0


Lag


10 2


10 3


10 1


10 2


10 3


10 2


Frequency (Hz)


10 1


10 2


10 3


10 4


10 3


Frequency (Hz)


Frequency (Hz)


Frequency (Hz)


Table 2: Detailed results on long-term time-series generation for **ETTh** . (Lower is better.)


|Metric|Length|DiffusionTs|TimeGAN|TimeVAE|STDiff (ours)|
|---|---|---|---|---|---|
|Context-FID<br>Score|64<br>128<br>256|0.631_±_.058<br>0.787_±_.062<br>0.423_±_.038|1.130_±_.102<br>1.553_±_.169<br>5.872_±_.208|0.827_±_.146<br>1.062_±_.134<br>0.826_±_.093|**0.031**_±_**.010**<br>**0.471**_±_**.003**<br>**0.341**_±_**.045**|
|Correlational<br>Score|64<br>128<br>256|0.082_±_.005<br>0.088_±_.005<br>0.064_±_.007|0.483_±_.019<br>0.188_±_.006<br>0.522_±_.013|0.067_±_.006<br>0.054_±_.007<br>0.046_±_.007|**0.055**_±_**.015**<br>**0.036**_±_**.009**<br>**0.044**_±_**.019**|
|Discriminative<br>Score|64<br>128<br>256|0.106_±_.048<br>0.144_±_.060<br>0.060_±_.030|0.227_±_.078<br>0.188_±_.074<br>0.442_±_.056|0.171_±_.142<br>0.154_±_.087<br>0.178_±_.076|**0.030**_±_**.020**<br>**0.032**_±_**.021**<br>**0.029**_±_**.042**|
|Predictive<br>Score|64<br>128<br>256|0.116_±_.000<br>0.110_±_.003<br>0.341_±_.045|0.132_±_.008<br>0.153_±_.014<br>0.220_±_.008|0.118_±_.004<br>0.113_±_.005<br>0.110_±_.027|**0.071**_±_**.000**<br>**0.065**_±_**.000**<br>**0.074**_±_**.001**|


every metric and sequence length but often does so by a substantial margin. The advantage is particularly striking in the Context-FID score, where at a length of 64, ST-Diff achieves a score of
0 _._ 031, representing more than an order-of-magnitude improvement over the next-best competitor.
This indicates a far more accurate and comprehensive approximation of the true data distribution’s
manifold. Furthermore, the degradation in ST-Diff is notably less pronounced as sequence length
increases. While competing models show considerable performance degradation, ST-Diff’s Discriminative Score remains exceptionally low and stable across all tested lengths (0.030 → 0.032 →
0.029). This suggests that the generated samples remain indistinguishable from real data even at
longer horizons, a critical marker of a robust and well-generalized generative process. The model’s
capacity to preserve meaningful temporal dynamics is confirmed by its consistently superior Predictive Scores. It indicates that the fundamental, step-by-step transition dynamics learned from
ST-Diff’s synthetic data are faithful to the real process. These findings provide compelling evidence
that our time-series-as-video paradigm is not only effective but overcomes a key limitation of models that operate purely in the time domain, which struggle with long contexts, or those that collapse
the temporal axis into a static image representation, thereby losing explicit sequential information.


6 CONCLUSION


In this paper, we addressed a central challenge in generative modeling of multivariate time series:
balancing expressive representations with faithful preservation of temporal structure. Existing approaches either operate directly in the time domain, limiting their ability to capture spectral properties, or transform sequences into static images, collapsing the temporal axis and precluding spatiotemporal modeling.


To solve these limitations, we introduced _Spectro-Temporal_ _Diffusion_ (ST-Diff), which reframes
time series as videos for generative diffusion. ST-Diff maps a multivariate time series to a spectrotemporal video tensor via the short-time Fourier transform (STFT), explicitly preserving the evolution of spectral content over time and making the problem amenable to modern video diffusion
architectures. We further developed a specialized spatiotemporal transformer with inductive biases
tailored to this domain, enabling effective learning of complex spectro–temporal dynamics.


Our extensive empirical study demonstrates that ST-Diff establishes a new state of the art in unconditional time series generation, consistently outperforming time-domain and image-based diffusion
models across diverse benchmarks, with particularly strong gains on high-dimensional, complex
datasets. These findings suggest that unifying classical signal-processing principles with spatiotemporal generative modeling through our _time-series-as-video_ approach yields a powerful and generalizable foundation for sequence generation.


Despite its strong performance, ST-Diff incurs higher computational and memory costs than timeor image-based models due to the use of spatiotemporal architectures. Exploring more efficient
video-generation paradigms, such as latent video diffusion or model distillation, may mitigate this
overhead. The proposed paradigm also opens several avenues for future research: extending STDiff to conditional tasks (e.g., forecasting and imputation), leveraging learned spectral-temporal
distributions for unsupervised anomaly detection, and applying the approach to other sequential data
domains where time–frequency analysis is essential, including audio, EEG, and seismic signals.


9


REFERENCES


J.B. Allen and L.R. Rabiner. A unified approach to short-time fourier analysis and synthesis. _Pro-_
_ceedings of the IEEE_, 65(11):1558–1564, 1977. doi: 10.1109/PROC.1977.10770.


Y. Bengio, P. Simard, and P. Frasconi. Learning long-term dependencies with gradient descent is
difficult. _IEEE Transactions on Neural Networks_, 5(2):157–166, 1994. doi: 10.1109/72.279181.


Jonathan Crabb´e, Nicolas Huynh, Jan Pawel Stanczuk, and Mihaela Van Der Schaar. Time series
diffusion in the frequency domain. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian
Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), _Proceedings_ _of_ _the_ _41st_
_International Conference on Machine Learning_, volume 235 of _Proceedings of Machine Learning_
_Research_, pp. 9407–9438. PMLR, 21–27 Jul 2024. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v235/crabbe24a.html)
[press/v235/crabbe24a.html.](https://proceedings.mlr.press/v235/crabbe24a.html)


Abhyuday Desai, Cynthia Freeman, Zuhui Wang, and Ian Beaver. Timevae: A variational autoencoder for multivariate time series generation, 2021. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2111.08095)
[2111.08095.](https://arxiv.org/abs/2111.08095)


Crist´obal Esteban, Stephanie L. Hyland, and Gunnar R¨atsch. Real-valued (medical) time series
generation with recurrent conditional gans, 2017. [URL https://arxiv.org/abs/1706.](https://arxiv.org/abs/1706.02633)
[02633.](https://arxiv.org/abs/1706.02633)


D. Griffin and Jae Lim. Signal estimation from modified short-time fourier transform. _IEEE Trans-_
_actions on Acoustics, Speech, and Signal Processing_, 32(2):236–243, 1984. doi: 10.1109/TASSP.
1984.1164317.


Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P.
Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High
definition video generation with diffusion models, 2022. [URL https://arxiv.org/abs/](https://arxiv.org/abs/2210.02303)
[2210.02303.](https://arxiv.org/abs/2210.02303)


Ilan Naiman, Nimrod Berman, Itai Pemper, Idan Arbiv, Gal Fadlon, and Omri Azencot. Utilizing
image transforms and diffusion models for generative modeling of short and long time series. In
_The_ _Thirty-eighth_ _Annual_ _Conference_ _on_ _Neural_ _Information_ _Processing_ _Systems_, 2024. URL
[https://openreview.net/forum?id=2NfBBpbN9x.](https://openreview.net/forum?id=2NfBBpbN9x)


William Peebles and Saining Xie. Scalable diffusion models with transformers. In _Proceedings of_
_the_ _IEEE/CVF_ _International_ _Conference_ _on_ _Computer_ _Vision_ _(ICCV)_, pp. 4195–4205, October
2023.


Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf. Autoregressive denoising
diffusion models for multivariate probabilistic time series forecasting. In Marina Meila and Tong
Zhang (eds.), _Proceedings_ _of_ _the_ _38th_ _International_ _Conference_ _on_ _Machine_ _Learning_, volume
139 of _Proceedings_ _of_ _Machine_ _Learning_ _Research_, pp. 8857–8868. PMLR, 18–24 Jul 2021.
[URL https://proceedings.mlr.press/v139/rasul21a.html.](https://proceedings.mlr.press/v139/rasul21a.html)


Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang,
Zhifeng Chen, Yu Zhang, Yuxuan Wang, Rj Skerrv-Ryan, Rif A. Saurous, Yannis Agiomvrgiannakis, and Yonghui Wu. Natural tts synthesis by conditioning wavenet on mel spectrogram
predictions. In _2018_ _IEEE_ _International_ _Conference_ _on_ _Acoustics,_ _Speech_ _and_ _Signal_ _Process-_
_ing_ _(ICASSP)_, pp. 4779–4783. IEEE Press, 2018. doi: 10.1109/ICASSP.2018.8461368. URL
[https://doi.org/10.1109/ICASSP.2018.8461368.](https://doi.org/10.1109/ICASSP.2018.8461368)


Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2022. URL
[https://arxiv.org/abs/2010.02502.](https://arxiv.org/abs/2010.02502)


Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer:
Enhanced transformer with rotary position embedding. _Neurocomputing_, 568:127063, 2024.
ISSN 0925-2312. doi: https://doi.org/10.1016/j.neucom.2023.127063. URL [https://www.](https://www.sciencedirect.com/science/article/pii/S0925231223011864)
[sciencedirect.com/science/article/pii/S0925231223011864.](https://www.sciencedirect.com/science/article/pii/S0925231223011864)


10


Yusuke Tashiro, Jiaming Song, Yang Song, and Stefano Ermon. Csdi: conditional score-based
diffusion models for probabilistic time series imputation. In _Proceedings of the 35th International_
_Conference_ _on_ _Neural_ _Information_ _Processing_ _Systems_, NIPS ’21, Red Hook, NY, USA, 2021.
Curran Associates Inc. ISBN 9781713845393.


Zhiguang Wang and Tim Oates. Imaging time-series to improve classification and imputation.
In _Proceedings_ _of_ _the_ _24th_ _International_ _Conference_ _on_ _Artificial_ _Intelligence_, IJCAI’15, pp.
3939–3945. AAAI Press, 2015. ISBN 9781577357384.


Jinsung Yoon, Daniel Jarrett, and Mihaela van der Schaar. Time-series generative adversarial networks. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alch´e-Buc, E. Fox, and
R. Garnett (eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, volume 32. Curran
Associates, Inc., 2019. URL [https://proceedings.neurips.cc/paper_files/](https://proceedings.neurips.cc/paper_files/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)
[paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)


Xinyu Yuan and Yan Qiao. Diffusion-TS: Interpretable diffusion for general time series generation.
In _The_ _Twelfth_ _International_ _Conference_ _on_ _Learning_ _Representations_, 2024. URL [https:](https://openreview.net/forum?id=4h1apFjO99)
[//openreview.net/forum?id=4h1apFjO99.](https://openreview.net/forum?id=4h1apFjO99)


Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series
forecasting? _Proceedings of the AAAI Conference on Artificial Intelligence_, 37(9):11121–11128,
Jun. 2023. doi: 10.1609/aaai.v37i9.26317. [URL https://ojs.aaai.org/index.php/](https://ojs.aaai.org/index.php/AAAI/article/view/26317)
[AAAI/article/view/26317.](https://ojs.aaai.org/index.php/AAAI/article/view/26317)


A DATASETS AND METRICS


A.1 DATASETS


Our evaluation uses six publicly available datasets, chosen to span a wide range of characteristics
including synthetic and real-world data, varying sequence lengths, and different levels of dimensionality and non-stationarity. This selection is consistent with prior work in time series generation

[1, ImagenTime, Diffusion-TS].


    - Sines: A synthetic dataset of sine waves with varying frequencies and phases, used to test
a model ability to learn fundamental periodic patterns.


    - Stocks: Real-world daily stock price data (Google), characterized by non-stationary behavior and random walks.


    - ETTh: Electricity Transformer Temperature data, containing high-frequency, multivariate
measurements with strong periodicities.


    - MuJoCo: High-dimensional data from a physics simulator, representing complex and nonlinear dynamics.


    - Energy: Real-world appliance energy consumption data, featuring multivariate correlations
and noisy periodicity.


    - fMRI: High-dimensional functional magnetic resonance imaging data, characterized by
noisy, complex, and correlated signals.

|Dataset|# Samples|# Covariates|
|---|---|---|
|Sines<br>Stocks<br>ETTh<br>MuJoCo<br>Energy<br>fMRI|10,000<br>3,773<br>17,420<br>10,000<br>19,711<br>10,000|5<br>6<br>7<br>14<br>28<br>50|


Table 3: Overview of datasets used, including number of samples and covariates.


11


A.2 EVALUATION METRICS


To provide a comprehensive and robust assessment of our model performance, we evaluate the
quality of the generated time series using a suite of four distinct, literature-established metrics Yoon
et al. (2019). These metrics are designed to measure from low-level statistical properties to highlevel temporal dynamics. All metrics are designed to be ”the-lower-the-better.”


    - Context-FID: To assess the overall distributional similarity between the real and synthetic
datasets, we employ the Fr´echet Inception Distance adapted for time series (Context-FID).
We first use a pre-trained TS2Vec model to generate a single, holistic embedding for each
time series in both the real and generated sets. The FID score is then calculated between
these two distributions of embeddings. A low Context-FID indicates that the model is
successfully capturing the diversity and global characteristics of the true data distribution.


    - Cross-Correlation: We measure the model ability to preserve the complex inter-feature
relationships using a cross-correlation metric. This metric computes the cross-correlation
matrix between all pairs of co-variates for both the real and generated data. The final score
is the aggregate difference between these two matrices. A low score signifies that the model
is correctly learning and reproducing the instantaneous structural dependencies between the
different time series co-variates.


    - Discriminative Score: To evaluate the sample-level realism of the generated data, we use
an adversarial approach. A separate, post-hoc GRU-based classifier is trained from scratch
with the task of distinguishing between real and synthetic time series. The final Discriminative Score is the absolute difference between the classifier accuracy and 0.5 (random
chance). A score close to zero indicates that the generated samples are of high fidelity and
are indistinguishable from real data.


    - Predictive Score: To evaluate if the generated data preserves the underlying temporal dynamics of the original series, we employ a ”Train-on-Synthetic, Test-on-Real” (TSTR)
evaluation. A simple GRU-based forecasting model is trained exclusively on the synthetic
data to predict one step ahead. This trained model is then tested on the real, unseen data.
The reported Predictive Score is the Mean Absolute Error (MAE) of these predictions.
A low score demonstrates that the temporal patterns learned from the synthetic data are
meaningful and can generalize to the real-world dynamics.


B MODEL ARCHITECTURAL PARAMETERS


In Table 4 we provide a detailed description of the configuration hyperparameters of the STDiff
model in the experiments presented in the sections above. A noteworthy observation is the consistent parametrization of Hidden Size and Num Heads for sequence lengths 64, 128, and 256. This
choice reflects an architectural stability, maintaining a robust representational capacity and multihead attention mechanism across varying input durations that likely demand similar levels of feature
complexity and contextual integration. However, the configuration for sequence length 24 presents
reduced Hidden Size of 192, Num Heads of 4 and Depth of 6. This specific adjustment is motivated by the inherent nature shorter time series sequences, characterized by less intricate temporal
dependencies and a comparatively smaller information manifold. Consequently, a more compact
model—characterized by fewer attention heads and a smaller hidden dimension—is often sufficient
to capture the underlying data distribution effectively without incurring unnecessary computational
overhead or risking overfitting on a simpler generative task.

|Dataset Seq. Len.|N FFT|Hop Length|Patch Size|Depth|Hidden Size|Num Heads|MLP Ratio|
|---|---|---|---|---|---|---|---|
|24|11|3|(2,1)|6|192|4|4.0|
|64|31|11|(4,1)|8|384|6|4.0|
|128|63|15|(8,1)|8|384|6|4.0|
|256|127|32|(16,1)|8|384|6|4.0|


Table 4: Hyperparameters for different sequence lengths.


12


C ADDITIONAL VISUALIZATIONS


(a) (b) (c)


Figure 5: t-SNE (up) Kernel Density Estimation (bottom) of real vs generated samples of the **ETTh**
dataset for sequence length 24(Fig. 5a), 64(Fig. 5b), 128(Fig. 5c). In red the original time series,
in blue the generated one.


To complement the quantitative results and qualitative analyses presented in the main body, this
section provides a more extensive set of visualizations the distributional alignment of generated
samples for the ETTh dataset across multiple sequence lengths and a per-covariate breakdown of
the temporal and spectral fidelity analysis.


Figure 5 provides additional qualitative validation for our model’s performance on the ETTh dataset
across varying sequence lengths (24, 64, and 128). The t-SNE plots (top row) demonstrate that the
manifold of the generated samples (blue) consistently and comprehensively overlaps with that of the
original data (red), indicating that ST-Diff successfully learns the global data structure. Furthermore,
the Kernel Density Estimation (KDE) plots (bottom row) show a near-perfect alignment between
the marginal distributions of the real and synthetic data, a consistency that is maintained even as the
sequence length increases, underscoring the model’s robustness and scalability.


Figure 6 and Figure 7 offer a comprehensive, per-covariate analysis of the temporal and spectral
fidelity for the ETTh dataset (sequence length 24) and the fMRI dataset (sequence length 24), respectively. This complements the summarized results in the main paper. The figures systematically
compare the Auto-Correlation Function (ACF) and Power Spectral Density (PSD) for the covariates
in the datasets. The uniform consistency across all plots demonstrates that the high-fidelity generation is not an artifact of a few selected channels. Instead, ST-Diff robustly captures the unique
temporal dependencies (ACF) and spectral characteristics (PSD) of each individual time series, providing strong evidence that our model learns the complete, multivariate data-generating process.


D LARGE LANGUAGE MODEL USAGE


Large language models have been utilized to polish and refine writing for enhanced conceptual
clarity, improving grammar, rephrasing sentences, and suggesting alternative word choices to make
the text more concise and understandable.


13


|Col1|Col2|Origina<br>Gener|l<br>ated|
|---|---|---|---|
|||||
|||||


|Col1|Col2|Col3|Orig|inal|
|---|---|---|---|---|
|||Gen|Gen|erated|
||||||
||||||


|Col1|Col2|Origina<br>Gener|l<br>ated|
|---|---|---|---|
|||||
|||||


|Col1|Col2|Col3|Orig<br>Gen|inal<br>erated|
|---|---|---|---|---|
||||||
||||||


|Col1|Col2|Origina<br>Gener|l<br>ated|
|---|---|---|---|
|||||
|||||


|Col1|Col2|Col3|Orig<br>Gen|inal<br>erated|
|---|---|---|---|---|
||||||
||||||


|Col1|Col2|Origina<br>Gener|l<br>ated|
|---|---|---|---|
|||||
|||||
|||||


|Col1|Col2|Col3|Orig|inal|
|---|---|---|---|---|
|||~~Gen~~|~~Gen~~|~~erated~~|
||||||


|Col1|Col2|Origina<br>Gener|l<br>ated|
|---|---|---|---|
|||||
|||||


|Col1|Col2|Col3|Orig|inal|
|---|---|---|---|---|
|||Gen|Gen|erated|
||||||
||||||


|Col1|Col2|Origina<br>Gener|l<br>ated|
|---|---|---|---|
|||||
|||||
|||||


|Col1|Col2|Col3|Orig<br>Gen|inal<br>erated|
|---|---|---|---|---|
||||||
||||||


|Col1|Col2|Origina<br>Gener|l<br>ated|
|---|---|---|---|
|||||
|||||


|Col1|Col2|Orig<br>Gen|inal<br>erated|
|---|---|---|---|
|||||
|||||


Figure 6: Complete analysis of Temporal and Spectral Fidelity on the ETTh dataset for seq ~~l~~ en
24. In the left column the Auto-Correlation Function (ACF), and in the right column the average
Power Spectral Density (PSD) for real (blue, solid) and generated (red, dashed) samples across all
the dataset covariates


14


1.0


0.5


0.0


1.0


0.5


0.0


1.0


0.5


0.0


1.0


0.5


0.0


1.0


0.5


0.0


1.0


0.5


0.0


1.0


0.5


0.0


Lag


Lag


Lag


Lag


Lag


Lag

Average ACF: OT


Lag


10 1


10 2


10 3


10 2


10 3


10 2


10 3


10 2


10 3


10 1


10 2


10 3


10 2


10 3


10 2


10 3


10 4


Frequency (Hz)


Frequency (Hz)


Frequency (Hz)


Frequency (Hz)


Frequency (Hz)


Frequency (Hz)

Average PSD: OT


Frequency (Hz)


|Col1|Col2|Origina|l|
|---|---|---|---|
|||~~Gener~~|~~ted~~|
|||||
|||||
|||||
|||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||~~Original~~<br>Generated||


|Col1|Col2|Origina|l|
|---|---|---|---|
|||~~Gener~~|~~ted~~|
|||||
|||||
|||||
|||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||Original<br>Generated||
||||||


|Col1|Col2|Origina|l|
|---|---|---|---|
|||~~Gener~~|~~ted~~|
|||||
|||||
|||||
|||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||Original<br>Generated||
||||||


|Col1|Col2|Origina|l|
|---|---|---|---|
|||~~Gener~~|~~ted~~|
|||||
|||||
|||||
|||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||Original<br>Generated|Original<br>Generated|
||||||


|Col1|Col2|Origina|l|
|---|---|---|---|
|||~~Gener~~|~~ted~~|
|||||
|||||
|||||
|||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||Original<br>Generated||
||||||


|Col1|Col2|Origina|l|
|---|---|---|---|
|||~~Gener~~|~~ted~~|
|||||
|||||
|||||
|||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||Original<br>Generated||


|Col1|Col2|Origina|l|
|---|---|---|---|
|||~~Gener~~|~~ted~~|
|||||
|||||
|||||
|||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||Original<br>Generated||
||||||


Figure 7: Complete analysis of Temporal and Spectral Fidelity on the fMRI dataset for seq ~~l~~ en 24.
In the left column the Auto-Correlation Function (ACF), and in the right column the average Power
Spectral Density (PSD) for real (blue, solid) and generated (red, dashed) samples across all the
dataset covariates. Notice that just the first 7 covariates were provided, the other 43 are very similar
and not included for simplicity.


15


1.00

0.75

0.50

0.25

0.00


1.00

0.75

0.50

0.25

0.00


1.00

0.75

0.50

0.25

0.00


1.00

0.75

0.50

0.25

0.00


1.00

0.75

0.50

0.25

0.00


1.00

0.75

0.50

0.25

0.00


1.00

0.75

0.50

0.25

0.00


Lag


Lag


Lag


Lag


Lag


Lag


Lag


10 2


10 2


10 2


10 2


10 2


10 2


10 2


Frequency (Hz)


Frequency (Hz)


Frequency (Hz)


Frequency (Hz)


Frequency (Hz)


Frequency (Hz)


Frequency (Hz)