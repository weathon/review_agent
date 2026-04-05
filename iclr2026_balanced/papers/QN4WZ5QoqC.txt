# ACTIVE SPEECH ENHANCEMENT: BEYOND PASSIVE

## DENOISING DECLIPPING AND DEREVERBERATION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


We introduce a new paradigm for active sound modification: _Active_ _Speech_ _En-_
_hancement_ (ASE). While Active Noise Cancellation (ANC) algorithms focus
on suppressing external interference and traditional speech enhancement passively reconstructs degraded speech signals, ASE goes further by actively shaping
the speech signal, both attenuating unwanted noise components and amplifying
speech-relevant frequencies to improve intelligibility and perceptual quality. To
enable this, we propose a novel Transformer-Mamba-based architecture, along
with a task-specific loss function designed to jointly optimize interference suppression and signal enrichment in an acoustic environment. Our method outperforms existing baselines across multiple speech processing tasks, including denoising, dereverberation, and declipping, demonstrating the effectiveness of active, targeted modulation in challenging acoustic environments. A demo page and
source code are provided in the Supplementary Materials.


1 INTRODUCTION


Traditional speech enhancement and noise control are fundamental audio processing tasks. Traditional speech enhancement aims to passively improve the perceptual quality and intelligibility of
speech signals by mitigating degradations such as background noise, distortion, clipping, and reverberation. Classic approaches—including spectral subtraction, Wiener filtering, and statistical
model–based methods—have achieved varying degrees of success but often falter in highly nonstationary noise environments (Boll, 2003; Lim & Oppenheim, 1978; Paliwal et al., 2012). Recent
advances in deep learning have, however, yielded state-of-the-art performance: convolutional neural networks (CNNs) (Pascual et al., 2017; Rethage et al., 2018; Pandey & Wang, 2018), recurrent
neural networks (RNNs) (Hu et al., 2020), generative adversarial networks (GANs) (Fu et al., 2019;
2021; Kim et al., 2021; Shin et al., 2023; Shetu et al., 2025), Transformers (Wang et al., 2021;
de Oliveira et al., 2022; Zhang et al., 2022b; Cao et al., 2022; Ye & Wan, 2023; Zhang et al., 2024),
and diffusion models (Guimar˜aes et al., 2025; Lu et al., 2022; Welker et al., 2022; Richter et al.,
2023; Lemercier et al., 2023; Tai et al., 2023; Ayilo et al., 2024) demonstrate exceptional results on
benchmarks for denoising, dereverberation, and declipping.


Active noise cancellation takes a complementary approach by generating an anti-noise signal to
interfere with unwanted noise destructively. Pioneering work dating back to Lueg’s first patent in
1936 introduced the concept of adaptive feedforward ANC (Lueg, 1936), which was later refined
through advances in adaptive filtering (e.g., LMS, FxLMS), multi-channel algorithms, and applications in headphones and enclosure systems (Nelson & Elliott, 1991; Fuller et al., 1996; Hansen
et al., 1997; Kuo & Morgan, 1999; Zhang & Wang, 2021; Park et al., 2023; Mostafavi & Cha, 2023;
Cha et al., 2023; Singh et al., 2024; Pike & Cheer, 2023; Mishaly et al., 2025). While ANC excels at
suppressing predictable or narrowband noise, it does not actively modify the speech content itself.


We propose a new paradigm— _Active_ _Speech_ _Enhancement_ (ASE) that unifies the goals of traditional speech enhancement and active noise control. Unlike conventional ANC, which solely targets
noise suppression, ASE actively shapes the speech signal by simultaneously attenuating interfering
components and amplifying speech-related frequencies. This dual-action approach not only reduces noise but also emphasizes speech and improves perceptual quality under challenging acoustic
conditions. We make four key contributions. First, we formalize the ASE task and describe appropriate evaluation metrics that capture both noise suppression and speech enhancement. Second, we


1


introduce a Transformer-Mamba-based model that generates an active modification signal, leveraging self-attention to capture long-range dependencies in time–frequency representations. Third,
we design a joint suppression–enrichment loss that balances interference removal and signal enrichment, combining spectral, perceptual, and adversarial objectives to drive optimal ASE performance.
Fourth, we conduct a comprehensive evaluation, showing that our method outperforms adapted baselines across multiple ASE tasks—including denoising, dereverberation, and declipping—with significant gains in metrics such as PESQ (Rix et al., 2001).


2 RELATED WORK


2.1 TRADITIONAL SPEECH ENHANCEMENT


Recent advances in deep learning have yielded substantial improvements in traditional speech enhancement. Pandey and Wang (Pandey & Wang, 2018) proposed a CNN–based autoencoder that
applies convolutions directly to the raw waveform while computing the loss in the frequency domain. SEGAN (Pascual et al., 2017) employs strided convolutional layers in a generative adversarial framework. Rethage et al. (2018) developed a WaveNet-inspired model that predicts multiple
waveform samples per step to reduce computational cost. Recurrent architectures have also been explored. Hu et al. (2020) presented DCCRN, which integrates complex-valued convolutional and recurrent layers to process spectrogram inputs. A real-time causal model based on an encoder–decoder
with skip connections was proposed by Defossez et al. (2020), operating directly on the raw waveform and optimized in both time and frequency domains.


MetricGAN (Fu et al., 2019) and its successor MetricGAN+ (Fu et al., 2021) incorporate evaluation
metrics such as PESQ (Perceptual Evaluation of Speech Quality) (Rix et al., 2001) and STOI (ShortTime Objective Intelligibility) (Kim et al., 2021) into the adversarial loss. Kim et al. (2021) further
enhance this approach by introducing a multiscale discriminator operating at different sampling rates
alongside a generator that processes speech at multiple resolutions.


Transformer-based models have recently gained prominence. Wang et al. (2021) proposed a twostage transformer network (TSTNN) that outperforms earlier time- and frequency-domain methods.
CMGAN (Cao et al., 2022) adapts the Conformer backbone (Gulati et al., 2020) for enhancement,
and de Oliveira et al. (2022) replace the learned encoder of SepFormer (Subakan et al., 2021) with
long-frame STFT (Short-Time Fourier Transform) inputs, reducing sequence length and lowering
computational cost without compromising perceptual quality.


More recently, diffusion-based approaches have emerged as a powerful generative paradigm. Lu
et al. (2022) introduced a conditional diffusion probabilistic model that learns a parameterized reverse diffusion process conditioned on the noisy input. Welker et al. (2022) extended score-based
models to the complex STFT domain, learning the gradient of the log-density of clean speech coefficients. Richter et al. (2023) formulate enhancement as a stochastic differential equation, initializing
reverse diffusion from a mixture of noisy speech and Gaussian noise and achieving high-quality reconstructions in only 30 steps. Lemercier et al. (2023) propose a _stochastic regeneration_ method that
leverages an initial predictive-model estimate to guide a reduced-step diffusion process, mitigating
artifacts and reducing computational cost by an order of magnitude while maintaining quality.


2.2 ACTIVE AUDIO CANCELLATION


Recently, deep learning approaches have demonstrated remarkable results in ANC algorithms.
Zhang & Wang (2021) introduced DeepANC, which employs a convolutional long short-term memory (Conv-LSTM) network to jointly estimate amplitude and phase responses from microphone
signals. Subsequently, attention-driven ANC frameworks integrating attentive recurrent networks
were proposed to enable real-time adaptation and low-latency operation (Zhang et al., 2022a).


A selective fixed-filter ANC (SFANC) framework was developed to leverage a two-dimensional
CNN for optimal control-filter selection on a mobile co-processor and a lightweight one-dimensional
CNN for time-domain noise classification, yielding superior attenuation of real-world non-stationary
headphone noise (Shi et al., 2022). Luo et al. (2022) proposed a hybrid SFANC–FxNLMS that first
applies a similar approach as SFANC for each noise frame and then applies the FxNLMS algorithm
for real-time coefficient adaptation, thereby combining the rapid response of SFANC with the low


2


steady-state error and robustness of adaptive optimization. Heuristic algorithms—such as bee colony
optimization (Ren & Zhang, 2022) and genetic algorithms (Zhou et al., 2023)—have been explored
to avoid gradient-based learning.


Other studies have applied recurrent convolutional networks (Park et al., 2023; Mostafavi & Cha,
2023; Cha et al., 2023) and fully connected neural networks (Pike & Cheer, 2023) to ANC.
Autoencoder-based encodings have been used to extract latent features for improved robustness (Singh et al., 2024). Efforts in SFANC have extended to synthesizing optimized filter banks
via unsupervised methods (Luo et al., 2024), while advancements in multichannel setups continue
to leverage spatial diversity through deep controllers (Shi et al., 2024). Multichannel configurations have been further enhanced by refined deep controllers that learn inter-channel relationships
for improved noise attenuation (Zhang & Wang, 2023; Anto˜nanzas et al., 2023; Xiao et al., 2023;
Zhang et al., 2023b; Shi et al., 2023), and attention-driven frameworks have been investigated for
low-latency operation (Zhang et al., 2023a).


3 BACKGROUND


We first examine a feedforward ANC algorithm that employs a single error microphone to lay the
foundation for our new ASE framework. In the ANC framework (Figure 1a), the _primary_ _path_,
characterized by the transfer function _P_ ( _z_ ), models the acoustic propagation from the disturbance
source to the error microphone. The _secondary path_, denoted by _S_ ( _z_ ), describes the transfer from
the loudspeaker to the error microphone. Let _x_ ( _n_ ) denote the reference signal applied to the ANC
system. The primary signal _d_ ( _n_ ) is obtained by filtering _x_ ( _n_ ) through the primary path:


_d_ ( _n_ ) = _P_ ( _z_ ) _∗_ _x_ ( _n_ ) _,_ (1)


where _∗_ denotes the convolution operation. The error microphone captures the residual signal _e_ ( _n_ ),
representing the difference between the original disturbance and the cancellation signal. Both _x_ ( _n_ )
and _e_ ( _n_ ) are used by the ANC algorithm to compute the canceling signal _y_ ( _n_ ). The loudspeaker implements _y_ ( _n_ ) according to its electro-acoustic transfer function _f_ LS _{·}_ . After propagation through
the secondary path, the anti-signal (or cancellation signal) is given by

_a_ ( _n_ ) = _S_ ( _z_ ) _∗_ _f_ LS� _y_ ( _n_ )� _._ (2)


The error signal is defined formally as the difference between the primary signal and the anti-signal:


_e_ ( _n_ ) = _d_ ( _n_ ) _−_ _a_ ( _n_ ) _._ (3)


The objective of the ANC algorithm is to generate _y_ ( _n_ ) such that _e_ ( _n_ ) is minimized, ideally achieving _e_ ( _n_ ) = 0, which corresponds to complete cancellation of the disturbance. In contrast, the **ASE**
**framework** uses the primary and anti-signals to construct an enhanced signal:


_eh_ ( _n_ ) = _d_ ( _n_ ) + _a_ ( _n_ ) _._ (4)


While ANC aims to eliminate the disturbance, ASE seeks to recover clean speech from a noisy
mixture of distorted speech _x_ ( _n_ ). The objective of the ASE task is to generate _eh_ ( _n_ ) such that
its deviation from the clean target signal _c_ ( _n_ ), i.e., _eh_ ( _n_ ) _−_ _c_ ( _n_ ), is minimized. As illustrated
in Figure 1b, the feedforward ASE setup comprises a disturbance source, a reference signal path,
and a control filter operating through the secondary path. Given the nature of the task, the error
microphone serves as the modification microphone in our framework.


Our work adapts three speech distortion types, previously defined by VoiceFixer (Liu et al., 2022) for
general speech restoration, to the context of our ASE framework. Specifically, our ASE-TM model
targets the restoration of speech _s_ ( _n_ ) degraded by: **(i)** **Additive** **noise** : This common distortion,
where unwanted background sounds obscure the speech, is modeled as the sum of the clean speech
signal _s_ ( _n_ ) and a noise signal _n_ ( _n_ ):


_d_ noise( _s_ ( _n_ )) = _s_ ( _n_ ) + _n_ ( _n_ ) _._ (5)


**(ii) Reverberation** : Caused by sound reflections in an enclosure, reverberation blurs speech signals.
It is modeled by convolving the speech signal _s_ ( _n_ ) with a room impulse response (RIR) _r_ ( _n_ ):


_d_ rev( _s_ ( _n_ )) = _s_ ( _n_ ) _∗_ _r_ ( _n_ ) _._ (6)


3


(a) A feedforward ANC setup diagram. (b) A feedforward ASE setup diagram.


Figure 1: Comparison of feedforward ANC and ASE setups.


**(iii) Clipping** : This distortion arises when signal amplitudes exceed the maximum recordable level,
typically due to microphone limitations. Clipping truncates the signal _s_ ( _n_ ) within a certain range

[ _−η,_ + _η_ ]:
_d_ clip( _s_ ( _n_ )) = max(min( _s_ ( _n_ ) _, η_ ) _, −η_ ) _,_ _η_ _∈_ [0 _,_ 1] _._ (7)

This leads to harmonic distortions and can degrade speech intelligibility.


To assess the performance of ASE-TM across these enhancement tasks, we employ a suite of established objective metrics. Consistent with the evaluation protocol in SEmamba (Chao et al., 2024),
these include the Wide-band PESQ (Rix et al., 2001), STOI (Taal et al., 2010), and the composite measures CSIG (predicting signal distortion), CBAK (predicting background intrusiveness), and
COVL (predicting overall speech quality) (Hu & Loizou, 2007). Furthermore, we incorporate the
Normalized Mean Square Error (NMSE), a traditionally well-established metric in the ANC task.
The NMSE between a target signal _u_ ( _n_ ) and an estimated signal _v_ ( _n_ ) is defined in decibels (dB) as:


where _M_ is the total number of samples. In our evaluations, _u_ ( _n_ ) represents the clean target speech
_c_ ( _n_ ), and _v_ ( _n_ ) is the enhanced speech _eh_ ( _n_ ) produced by our model (the precise definition of _c_ ( _n_ )
for each task is detailed in Section 4.1).


4 METHOD


4.1 ASE-TM ARCHITECTURE OVERVIEW


The proposed model, ASE Transformer-Mamba (ASE-TM, Figure 2), adopts and extends the fundamental structure of the SEmamba architecture (Chao et al., 2024), which consists of a dense
encoder, a series of Time-Frequency Mamba (TFMamba) blocks (Xiao & Das, 2024), and parallel
magnitude and phase decoders. A notable distinction of our ASE-TM model is the utilization of
Mamba2 blocks (Dao & Gu, 2024) within these TFMamba pathways, in contrast to the original SEmamba architecture, which employed an earlier version of Mamba (Gu & Dao, 2023). This choice
is motivated by the potential improvements in efficiency offered by Mamba2.


The input noisy waveform, _x_ ( _n_ ), sampled at a rate of _Fs_, is processed through an STFT. The STFT
employs a Hann window of _N_ win samples, a hop length of _N_ hop samples, and an _N_ FFT-point FFT,
resulting in _N_ freq = _⌊N_ win _/_ 2 _⌋_ +1 frequency features per frame. Its magnitude and phase components
are horizontally stacked and fed into the network. The dense encoder utilizes convolutional layers
and dense blocks to extract initial features from the stacked magnitude and phase, outputting a
representation with _C_ enc channels, each with _N_ enc features.


The core of the temporal and spectral modeling is based on _N_ tf TFMamba blocks. Each TFMamba
block contains separate Mamba-based pathways (time-mamba and freq-mamba) employing
bidirectional Mamba layers to capture dependencies across time and frequency dimensions, respectively (Chao et al., 2024; Xiao & Das, 2024).


Following the initial _N_ tf _/_ 2 TFMamba blocks, inspired by hybrid approaches like Jamba (Lieber
et al., 2024), we introduce an attention-based block. Before applying attention, the feature representation, with _C_ enc channels, undergoes dimensionality reduction. A 2D convolution with a kernel size


4


NMSE[ _u, v_ ] = 10 _·_ log10


- - _M_
_n_ =1 [(] _[u]_ [(] _[n]_ [)] _[ −]_ _[v]_ [(] _[n]_ [))][2]

  - _M_
_n_ =1 _[u]_ [(] _[n]_ [)][2]


_,_ (8)


Figure 2: ASE-TM Architecture.


of (1 _, N_ enc _/_ 2+1) reduces the channel dimension from _C_ enc to _C_ enc _/_ 4 and each channel features size
from _N_ enc to _N_ enc _/_ 2. This results in a compact representation of size _N_ enc _/_ 2 _× C_ enc _/_ 4 for the attention layer. In addition, positional encoding is applied to this compact representation. A standard
Multi-Head Attention layer with _N_ heads heads is then used on this reduced representation to weigh
features based on global context. Following the attention layer, an expansion module employing a
transposed 2D convolution, also with a kernel size of (1 _, N_ enc _/_ 2 + 1), is used to restore the channel
dimension to _C_ enc and expand the feature dimension back towards _N_ enc before passing the features
to the remaining _N_ tf _/_ 2 TFMamba blocks. An additional step before applying the remaining TFMamba blocks is the use of a residual connection that sums the feature representations from before
the dimensionality reduction with those after the attention and expansion modules


The magnitude and phase decoders retain the structure used in SEmamba, employing dense blocks
and convolutional layers (including transposed convolutions) to reconstruct the target representation—not before applying a residual connection that performs element-wise multiplication between
the original STFT magnitude and the predicted magnitude. However, instead of predicting the enhanced spectra directly, the network is trained to output the complex spectrum of the cancelling
signal _y_ ( _n_ ). This signal _y_ ( _n_ ), after undergoing the electro-acoustic transfer function _f_ LS _{·}_ and
propagation through the secondary path _S_ ( _z_ ), becomes the anti-signal _a_ ( _n_ ) (as defined in Eq. 2).
This anti-signal _a_ ( _n_ ) is then summed with the primary path signal _d_ ( _n_ ) to produce the final enhanced
signal _eh_ ( _n_ ) (as defined in Eq. 3).


4.2 OPTIMIZATION OBJECTIVE


The primary goal of the ASE-TM model is to generate an enhanced signal _eh_ ( _n_ ) that is as close
as possible to a clean target speech signal _c_ ( _n_ ). The definition of this target _c_ ( _n_ ) varies based
on the specific enhancement task. For **additive** **noise** **reduction**, _c_ ( _n_ ) is the clean speech signal
convolved with the primary path _P_ ( _z_ ), representing the clean signal as perceived at the modification
microphone. For **dereverberation** and **declipping**, _c_ ( _n_ ) is the original anechoic, unclipped clean
speech signal, prior to any acoustic path effects or clipping distortion.


The training of ASE-TM largely follows the multi-level loss framework established in SEmamba
and originating from MP-SENet (Lu et al., 2023). This framework combines several loss components. Our approach incorporates this established framework with specific modifications. The
overall generator loss _LG_ is a weighted sum of the following components:


1. **Time-Domain Loss (** _L_ **T)** : We employ a combination of L1 and L2 distances between the
enhanced waveform _eh_ ( _n_ ) and the target waveform _c_ ( _n_ ):

_L_ T = _||eh_ ( _n_ ) _−_ _c_ ( _n_ ) _||_ 1 + _||eh_ ( _n_ ) _−_ _c_ ( _n_ ) _||_ [2] 2 _[.]_ (9)


This hybrid loss aims to leverage the robustness of L1 to outliers and the smoothness encouraged by L2.
2. **Magnitude Spectrum Loss (** _L_ **Mag)** : Similar to the time-domain loss, a combined L1 and
L2 loss on the magnitude spectra is applied. If _ENm_ and _Cm_ are the magnitude spectra of
_eh_ ( _n_ ) and _c_ ( _n_ ) respectively, then:

_L_ Mag = _||ENm −_ _Cm||_ 1 + _||ENm −_ _Cm||_ [2] 2 _[.]_ (10)


This contrasts with the L2 loss typically used in MP-SENet for this component.
3. **Complex Spectrum Loss (** _L_ **Com)** : This loss penalizes differences in the STFT domain. It
is the sum of L2 losses on the real and imaginary parts of the STFTs of _eh_ ( _n_ ) and _c_ ( _n_ ).


5


4. **Anti-Wrapping Phase Loss (** _L_ **Pha)** : Includes instantaneous phase loss, group delay loss,
and instantaneous angular frequency loss to optimize the phase spectrum directly, addressing phase wrapping issues.


5. **Metric-Based** **Adversarial** **Loss** **(** _L_ **Met)** : A discriminator trained to predict a perceptual
metric (e.g., PESQ), guiding the generator to produce outputs that score well on it.


6. **Consistency Loss (** _L_ **Con)** : We incorporate a consistency loss. This loss minimizes the discrepancy between the complex spectrum directly output by the model’s decoders (magnitude and phase) and the complex spectrum obtained by applying STFT to the time-domain
waveform _eh_ ( _n_ ) that results from an inverse STFT of the initially predicted spectrum.


The total generator loss is then defined with the hyperparameter _γ_ as follows:


_LG_ = _γ_ 1 _L_ T + _γ_ 2 _L_ Mag + _γ_ 3 _L_ Com + _γ_ 4 _L_ Met + _γ_ 5 _L_ Pha + _γ_ 6 _L_ Con _._ (11)


5 EXPERIMENTS


5.1 DATASETS AND TASK GENERATION


We conduct evaluations across three primary speech restoration tasks: additive noise reduction,
dereverberation, and declipping. For **additive** **noise** **reduction**, we use the VoiceBank-DEMAND
dataset (Botinhao et al., 2016), a standard benchmark in speech enhancement. This dataset combines
clean speech from the VoiceBank corpus (Veaux et al., 2013) with various non-stationary noises
from the DEMAND database (Thiemann et al., 2013). Our training set consists of utterances from
28 speakers with 10 different noise types at Signal-to-Noise Ratios (SNRs) of 0, 5, 10, and 15 dB. We
used two speakers from the training set as the validation set. The test set comprises 824 utterances
from 2 unseen speakers, mixed with 5 unseen noise types at SNRs of 2.5, 7.5, 12.5, and 17.5 dB.


The datasets for the **dereverberation** and **declipping** tasks are generated using the clean speech
utterances of the speakers available in the VoiceBank corpus. **Dereverberation** : Reverberant speech
is synthesized by convolving the clean VoiceBank utterances with Room Impulse Responses (RIRs)
as defined in Eq. 6 using the SpeechBrain package (Ravanelli et al., 2024). For training, we randomly
sample RIRs from the training portion of the RIR dataset provided alongside VoiceFixer (Liu et al.,
2022). For the test set, a fixed and distinct set of RIRs (from the VoiceFixer RIR test set) is applied
to the clean test utterances to ensure consistent evaluation conditions. **Declipping** : Clipped speech
signals are generated by applying a clipping threshold _η_ to the clean utterances according to Eq. 7.
During training, the clipping ratio _η_ is uniformly sampled from the range [0 _._ 1 _,_ 0 _._ 5] for each utterance
to expose the model to varying degrees of distortion. For testing, a specific clipping threshold is used.


5.2 ACOUSTIC PATH SIMULATION


To emulate the acoustic environment for the ASE framework, we simulate the primary path _P_ ( _z_ )
and secondary path _S_ ( _z_ ). Our simulation setup is based on previous setups for the ANC task (Zhang
& Wang, 2021; Zhang et al., 2023a), modeling a rectangular enclosure with dimensions of 3 _×_ 4 _×_ 2
meters (width _×_ length _×_ height). Room Impulse Responses (RIRs) are generated using the
image method (Allen & Berkley, 1979), implemented with a Python-based RIR generator package (Habets, 2006) with high-pass filtering. The modification microphone, capturing _eh_ ( _n_ ), is at

[1 _._ 5 _,_ 3 _,_ 1] meters. The reference microphone, capturing _x_ ( _n_ ), is at [1 _._ 5 _,_ 1 _,_ 1] meters, and the cancellation loudspeaker, which outputs the signal leading to _a_ ( _n_ ), is at [1 _._ 5 _,_ 2 _._ 5 _,_ 1] meters within
the enclosure. The RIR length for both _P_ ( _z_ ) and _S_ ( _z_ ) is _L_ RIR = 512 taps. The non-linear
characteristics of the loudspeaker are modeled using the Scaled Error Function (SEF), defined as
_f_ LS _{y}_ = �0 _y_ [exp(] _[−][z]_ [2] _[/]_ [(2] _[λ]_ [2][))] _[dz]_ [.] [Here,] _[ y]_ [is the loudspeaker input, and] _[ λ]_ [2] [controls the severity of]
the saturation non-linearity. Different _λ_ [2] values simulate varying degrees of distortion, with larger
values approaching linear behavior. To introduce variability during training, the room’s reverberation time ( _T_ 60) and _λ_ [2] are randomly sampled from the sets _{_ 0 _._ 15 _,_ 0 _._ 175 _,_ 0 _._ 2 _,_ 0 _._ 225 _,_ 0 _._ 25 _}_ seconds
and _{_ 0 _._ 1 _,_ 1 _,_ 10 _, ∞}_, respectivly, for each training sample. For testing, fixed _T_ 60 and _λ_ [2] are used.


6


Table 1: Average denoising results on the VoiceBank-DEMAND test set ( _T_ 60 = 0 _._ 25 _s_ and _λ_ [2] = _∞_ ).


**Method** **PESQ (** _↑_ **)** **CSIG (** _↑_ **)** **CBAK (** _↑_ **)** **COVL (** _↑_ **)** **STOI (** _↑_ **)** **NMSE (** _↓_ **)**


Noisy-speech 1.97 3.50 2.55 2.75 0.92 -8.44
THF-FxLMS 2.37 3.66 2.84 3.00 0.97 -15.32
DeepANC 1.48 1.99 2.19 1.69 0.93 -12.80
ARN 2.45 3.64 3.13 3.03 0.97 -20.64
ASE-TM **2.98** **4.21** **3.49** **3.62** **0.99** **-21.76**


5.3 MODEL HYPERPARAMETERS AND TRAINING


The ASE-TM model processes audio at _Fs_ = 16 kHz. For the STFT, we use a Hann window of
_N_ win = 400 samples, hop length of _N_ hop = 100 samples, and an _N_ FFT = 400-point FFT. The
dense encoder outputs a feature representation with _C_ enc = 128 channels, where each channel has
a feature dimension of _N_ enc = 100. Our model employs a total of _N_ tf = 8 TFMamba blocks.
The Multi-Head Attention layer within the attention-based block uses _N_ heads = 10 heads. Other
internal architectural details for the TFMamba blocks and dense convolutional blocks largely follow
the configurations presented in SEmamba. The ASE-TM model is trained for 350 epochs using the
AdamW optimizer (Loshchilov & Hutter, 2017) with _β_ 1 = 0 _._ 8 and _β_ 2 = 0 _._ 99. The initial learning
rate is set to 5 _×_ 10 _[−]_ [4] . We use a batch size of 4. Audio segments of 32 _,_ 000 samples (equivalent to 2
seconds at 16 kHz) are used for training. The model parameters yielding the best performance on the
validation set, evaluated based on the PESQ score, are saved for final testing. We used an NVIDIA
RTX A6000 GPU (internal cluster). The training runtime of the ASE-TM model was _∼_ 10 days.


5.4 BASELINE METHODS


We compare ASE-TM with several established baseline methods commonly used in ANC. These
include THF-FxLMS (Ghasemi et al., 2016), which is an extension to the traditional FxLMS algorithm (Kuo & Morgan, 1999), DeepANC that utilizes a convolutional LSTMs (Zhang & Wang,
2021), and ARN that incorporates an attention mechanism (Zhang et al., 2023a). These baseline
methods were adapted and retrained or configured to the ASE framework across all tested tasks.


6 RESULTS AND ANALYSIS


6.1 ACTIVE DENOISING PERFORMANCE


The speech denoising performance on the VoiceBank-DEMAND dataset is in Table 1. Our ASE-TM
model demonstrates superior performance, achieving a PESQ score of 2 _._ 98, significantly surpassing
the baselines: THF-FxLMS achieved a PESQ of 2 _._ 37, and the deep learning-based ANC methods,
DeepANC and ARN, yielded PESQ scores of 1 _._ 48 and 2 _._ 45, respectively. These results demonstrate
a substantial gap between conventional ANC approaches and our ASE-TM, which benefits from
actively shaping the speech signal in addition to noise suppression, as also evidenced by its leading
scores in CSIG, CBAK, COVL, STOI, and a significantly better NMSE of _−_ 21 _._ 76 dB.


6.2 DEREVERBERATION AND DECLIPPING PERFORMANCE


ASE-TM’s efficacy was further evaluated on dereverberation and declipping tasks, with results presented in Tables 2 and 3, respectively. For dereverberation (Table 2), ASE-TM achieved a PESQ
score of 2 _._ 43, a considerable improvement from the reverberant speech baseline (PESQ 1 _._ 60). In
contrast, the adapted ANC baselines struggled; THF-FxLMS scored a PESQ of 1 _._ 43, while DeepANC and ARN achieved 1 _._ 06 and 1 _._ 35, respectively. This suggests that these methods, even when
retrained or configured for the task, struggled to effectively adjust their processes to mitigate reverberation in the ASE framework. Similarly, in the declipping task, with a clipping threshold of
_η_ = 0 _._ 25 (Table 3), ASE-TM restored speech to a PESQ of 3 _._ 09 from an initial score of 2 _._ 17. The
baseline methods again showed limited effectiveness: THF-FxLMS (PESQ 1 _._ 92), DeepANC (PESQ
1 _._ 05), and ARN (PESQ 1 _._ 67). These tasks, particularly where the target _c_ ( _n_ ) is the original clean
speech before any primary path effects, highlight the challenge and efficacy of the ASE approach in
not just cancelling an interfering signal but actively restoring a desired signal characteristic.


7


Table 2: Average dereverberation results on the reverbed test set ( _T_ 60 = 0 _._ 25 _s_ and _λ_ [2] = _∞_ ).


**Method** **PESQ (** _↑_ **)** **CSIG (** _↑_ **)** **CBAK (** _↑_ **)** **COVL (** _↑_ **)** **STOI (** _↑_ **)** **NMSE (** _↓_ **)**


Reverbed-speech 1.60 2.60 1.88 2.02 0.80 2.00
THF-FxLMS 1.43 2.55 1.64 1.89 0.78 4.77
DeepANC 1.06 1.00 1.00 1.00 0.53 21.19
ARN 1.35 1.25 1.54 1.18 0.76 4.58
ASE-TM **2.43** **3.71** **2.67** **3.07** **0.93** **-0.04**


Table 3: Average declipping results on the clipped test set ( _η_ = 0 _._ 25, _T_ 60 = 0 _._ 25 _s_, and _λ_ [2] = _∞_ ).


**Method** **PESQ (** _↑_ **)** **CSIG (** _↑_ **)** **CBAK (** _↑_ **)** **COVL (** _↑_ **)** **STOI (** _↑_ **)** **NMSE (** _↓_ **)**


Clipped-speech 2.17 3.49 2.51 2.82 0.89 -0.23
THF-FxLMS 1.92 3.35 2.36 2.62 0.88 -0.02
DeepANC 1.05 1.00 1.00 1.00 0.53 11.10
ARN 1.67 1.60 2.12 1.57 0.87 -0.31
ASE-TM **3.09** **4.20** **3.06** **3.67** **0.93** **-1.70**


6.3 DENOISING ASE-TM MODEL ANALYSIS


An ablation study, presented in Figure 3a, investigates the contributions of our proposed loss function modifications, the attention mechanism, and the use of Mamba2 over Mamba1 to the ASE-TM
model for the denoising task. The full model consistently achieves the highest validation PESQ
score throughout training. Replacing Mamba1 with Mamba2 and the modified loss yielded the most
considerable performance improvement among all evaluated components. Notably, configurations
incorporating the attention mechanism demonstrate a faster convergence to higher performance levels, suggesting that attention aids in efficiently learning relevant features. Spectrogram analysis
of a representative denoising example (Figure 3b) visually confirms the model’s effectiveness; the
spectrogram of the enhanced signal closely mirrors that of the clean speech (after primary path),
indicating successful noise suppression while preserving essential speech characteristics.


To assess robustness, ASE-TM was evaluated under varying acoustic conditions for the denoising
task, with results in Table 4. This analysis focused on ASE-TM due to its significantly better performance over baselines in Table 1. The model shows consistent high performance across different _T_ 60
values under linear loudspeaker conditions ( _λ_ [2] = _∞_ ), achieving a PESQ of 3 _._ 02 for _T_ 60 = 0 _._ 15s
and 3 _._ 13 for _T_ 60 = 0 _._ 20s. When strong non-nonlinearities are introduced (e.g., _λ_ [2] = 0 _._ 1 at
_T_ 60 = 0 _._ 25s), the PESQ score is 2 _._ 74, still indicating robust performance. As _λ_ [2] increases (less
non-linearity), performance improves, reaching a PESQ of 2 _._ 97 for _λ_ [2] = 10 at _T_ 60 = 0 _._ 25s.


(a) An ablation study. A moving average with window size = 10 was applied. (b) An enhanced (denoised) signal spectrogram.


Figure 3: Model analysis of ASE-TM model for the denoising task.


8


Table 4: Average performance of ASE-TM (denoising task) under varying conditions ( _T_ 60 and loudspeaker non-linearity factor _λ_ [2] ) on the VoiceBank-DEMAND test set.


_T_ 60 **(s)** _λ_ [2] **PESQ (** _↑_ **)** **CSIG (** _↑_ **)** **CBAK (** _↑_ **)** **COVL (** _↑_ **)** **STOI (** _↑_ **)** **NMSE (** _↓_ **)**


0.25 0.1 2.74 4.01 3.29 3.39 0.98 -20.21
0.25 1.0 2.92 4.17 3.44 3.57 0.99 -21.92
0.25 10 2.97 4.21 3.48 3.62 0.99 -22.37
0.15 _∞_ 3.02 4.22 3.50 3.65 0.98 -22.31
0.20 _∞_ 3.13 4.33 3.60 3.77 0.99 -22.88


Figure 4: Power spectra for the dereverberation and declipping tasks over the entire test set.


6.4 RUNTIME ANALYSIS


To satisfy real-time constraints in active systems, we evaluated ASE-TM under a future-frame prediction strategy, following prior work (Zhang & Wang, 2021; Zhang et al., 2023a). In our setup, the
causality condition _T_ ASE-TM _< Tp −_ _Ts_ evaluates to _T_ ASE-TM _<_ 3432 _[−]_ 343 [0] _[.]_ [5] _[≈]_ [0] _[.]_ [0043][ seconds, where]

_Tp_ and _Ts_ denote the acoustic delays of the primary and secondary paths, respectively. To accommodate the model’s inference latency, we predict 500 future frames (0.03125 seconds), remaining
within real-time limits of our computational environment. Despite this future context, performance
degradation is minimal: on the VoiceBank-DEMAND test set ( _T_ 60 = 0 _._ 25 s, _η_ [2] = _∞_ ), ASE-TM
achieves a PESQ of 2.96 and STOI of 0.99—closely matching the non-causal configuration.


6.5 DEREVERBERATION AND DECLIPPING ASE-TM MODEL ANALYSIS


Figure 4 presents the power spectra of the enhanced signals for the dereverberation and declipping
tasks, over the entire test set. For both tasks, the spectrum of the enhanced signal _eh_ ( _n_ ) exhibits
significantly more power across a broad frequency range compared to the distorted input (reverberated or clipped after primary path), indicating successful signal restoration and enrichment. In the
declipping task, it is particularly noteworthy that lower frequencies, crucial for speech intelligibility,
show substantial power recovery in the enhanced signal’s spectrum. We further evaluated the declipping performance under a more aggressive clipping threshold of _η_ = 0 _._ 1. The unprocessed clipped
speech at this level yielded a PESQ score of 1 _._ 53 and an NMSE of _−_ 0 _._ 18 dB. ASE-TM restored
these signals to a PESQ of 2 _._ 52 (CSIG 3 _._ 61, CBAK 2 _._ 76, COVL 3 _._ 08, STOI 0 _._ 91) and an NMSE of

_−_ 1 _._ 22 dB. While these results are lower than for _η_ = 0 _._ 25, they represent a substantial improvement
over the severely clipped input, showing ASE-TM’s capability to handle extreme distortions.


7 CONCLUSIONS AND LIMITATIONS


In this paper, we introduced ASE, a novel paradigm that extends beyond traditional ANC by actively shaping the speech signal to enhance quality and intelligibility. Our ASE-TM model, which
leverages a Transformer-Mamba architecture and a specialized loss function, demonstrated strong
performance in denoising, dereverberation, and declipping, outperforming baseline methods. This
study also reveals limitations that require further investigation. Baseline methods, designed for
ANC, were adapted to ASE tasks, which may explain their reduced performance. Furthermore, future work should focus on developing a unified model that can handle multiple speech enhancement
objectives, potentially leading to more versatile and efficient systems.


We utilized large language models (LLMs) to assist in refining the manuscript’s writing.


9


REFERENCES


Jont B Allen and David A Berkley. Image method for efficiently simulating small-room acoustics.
_The Journal of the Acoustical Society of America_, 65(4):943–950, 1979.


Christian Anto˜nanzas, Miguel Ferrer, Maria De Diego, and Alberto Gonzalez. Remote microphone
technique for active noise control over distributed networks. _IEEE/ACM Transactions on Audio,_
_Speech, and Language Processing_, 31:1522–1535, 2023.


Jean-Eudes Ayilo, Mostafa Sadeghi, and Romain Serizel. Diffusion-based speech enhancement
with a weighted generative-supervised learning loss. In _ICASSP 2024 - 2024 IEEE International_
_Conference on Acoustics, Speech and Signal Processing (ICASSP)_, pp. 12506–12510. IEEE, April
2024. doi: 10.1109/icassp48485.2024.10446805. URL [http://dx.doi.org/10.1109/](http://dx.doi.org/10.1109/ICASSP48485.2024.10446805)
[ICASSP48485.2024.10446805.](http://dx.doi.org/10.1109/ICASSP48485.2024.10446805)


Steven Boll. Suppression of acoustic noise in speech using spectral subtraction. _IEEE Transactions_
_on acoustics, speech, and signal processing_, 27(2):113–120, 2003.


Cassia Valentini Botinhao, Xin Wang, Shinji Takaki, and Junichi Yamagishi. Investigating rnnbased speech enhancement methods for noise-robust text-to-speech. In _9th ISCA speech synthesis_
_workshop_, pp. 159–165, 2016.


Ruizhe Cao, Sherif Abdulatif, and Bin Yang. Cmgan: Conformer-based metric gan for speech
enhancement. _arXiv preprint arXiv:2203.15149_, 2022.


Young-Jin Cha, Alireza Mostafavi, and Sukhpreet S Benipal. Dnoisenet: Deep learning-based feedback active noise control in various noisy environments. _Engineering_ _Applications_ _of_ _Artificial_
_Intelligence_, 121:105971, 2023.


Rong Chao, Wen-Huang Cheng, Moreno La Quatra, Sabato Marco Siniscalchi, Chao-Han Huck
Yang, Szu-Wei Fu, and Yu Tsao. An investigation of incorporating mamba for speech enhancement. _arXiv preprint arXiv:2405.06573_, 2024.


Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through
structured state space duality. _arXiv preprint arXiv:2405.21060_, 2024.


Danilo de Oliveira, Tal Peer, and Timo Gerkmann. Efficient transformer-based speech enhancement
using long frames and stft magnitudes. In _Interspeech_ _2022_, pp. 2948–2952. ISCA, September 2022. doi: 10.21437/interspeech.2022-10781. [URL http://dx.doi.org/10.21437/](http://dx.doi.org/10.21437/Interspeech.2022-10781)
[Interspeech.2022-10781.](http://dx.doi.org/10.21437/Interspeech.2022-10781)


Alexandre Defossez, Gabriel Synnaeve, and Yossi Adi. Real time speech enhancement in the waveform domain. _arXiv preprint arXiv:2006.12847_, 2020.


Szu-Wei Fu, Chien-Feng Liao, Yu Tsao, and Shou-De Lin. Metricgan: Generative adversarial networks based black-box metric scores optimization for speech enhancement. In _International_
_Conference on Machine Learning_, pp. 2031–2041. PmLR, 2019.


Szu-Wei Fu, Cheng Yu, Tsun-An Hsieh, Peter Plantinga, Mirco Ravanelli, Xugang Lu, and Yu Tsao.
Metricgan+: An improved version of metricgan for speech enhancement. In _Interspeech_ _2021_ .
ISCA, August 2021. doi: 10.21437/interspeech.2021-599. [URL http://dx.doi.org/10.](http://dx.doi.org/10.21437/interspeech.2021-599)
[21437/interspeech.2021-599.](http://dx.doi.org/10.21437/interspeech.2021-599)


Christopher C Fuller, Sharon Elliott, and Philip Arthur Nelson. _Active_ _control_ _of_ _vibration_ . Academic press, 1996.


Sepehr Ghasemi, Raja Kamil, and Mohammad Hamiruce Marhaban. Nonlinear thf-fxlms algorithm
for active noise control with loudspeaker nonlinearity. _Asian Journal of Control_, 18(2):502–513,
2016.


Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. _arXiv_
_preprint arXiv:2312.00752_, 2023.


10


Heitor R. Guimar˜aes, Jiaqi Su, Rithesh Kumar, Tiago H. Falk, and Zeyu Jin. Ditse: High-fidelity
generative speech enhancement via latent diffusion transformers, 2025.


Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo
Wang, Zhengdong Zhang, Yonghui Wu, and Ruoming Pang. Conformer: Convolution-augmented
transformer for speech recognition, 2020. [URL https://arxiv.org/abs/2005.08100.](https://arxiv.org/abs/2005.08100)


Emanuel AP Habets. Room impulse response generator. _Technische Universiteit Eindhoven, Tech._
_Rep_, 2(2.4):1, 2006.


Colin H Hansen, Scott D Snyder, Xiaojun Qiu, Laura A Brooks, and Danielle J Moreau. _Active_
_control of noise and vibration_ . E & Fn Spon London, 1997.


Yanxin Hu, Yun Liu, Shubo Lv, Mengtao Xing, Shimin Zhang, Yihui Fu, Jian Wu, Bihong Zhang,
and Lei Xie. Dccrn: Deep complex convolution recurrent network for phase-aware speech enhancement. _arXiv preprint arXiv:2008.00264_, 2020.


Yi Hu and Philipos C Loizou. Evaluation of objective quality measures for speech enhancement.
_IEEE Transactions on audio, speech, and language processing_, 16(1):229–238, 2007.


Hyung Yong Kim, Ji Won Yoon, Sung Jun Cheon, Woo Hyun Kang, and Nam Soo Kim. A multiresolution approach to gan-based speech enhancement. _Applied Sciences_, 11(2):721, 2021.


Sen M Kuo and Dennis R Morgan. Active noise control: a tutorial review. _Proceedings of the IEEE_,
87(6):943–973, 1999.


Jean-Marie Lemercier, Julius Richter, Simon Welker, and Timo Gerkmann. Storm: A diffusionbased stochastic regeneration model for speech enhancement and dereverberation. _IEEE/ACM_
_Transactions_ _on_ _Audio,_ _Speech,_ _and_ _Language_ _Processing_, 31:2724–2737, 2023. ISSN 23299304. doi: 10.1109/taslp.2023.3294692. URL [http://dx.doi.org/10.1109/TASLP.](http://dx.doi.org/10.1109/TASLP.2023.3294692)
[2023.3294692.](http://dx.doi.org/10.1109/TASLP.2023.3294692)


Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi,
Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, et al. Jamba: A hybrid transformermamba language model. _arXiv preprint arXiv:2403.19887_, 2024.


Jae Lim and Alan Oppenheim. All-pole modeling of degraded speech. _IEEE_ _Transactions_ _on_
_Acoustics, Speech, and Signal Processing_, 26(3):197–210, 1978.


Haohe Liu, Xubo Liu, Qiuqiang Kong, Qiao Tian, Yan Zhao, DeLiang Wang, Chuanzeng Huang,
and Yuxuan Wang. Voicefixer: A unified framework for high-fidelity speech restoration. _arXiv_
_preprint arXiv:2204.05841_, 2022.


Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv_ _preprint_
_arXiv:1711.05101_, 2017.


Ye-Xin Lu, Yang Ai, and Zhen-Hua Ling. Mp-senet: A speech enhancement model with parallel
denoising of magnitude and phase spectra. _arXiv preprint arXiv:2305.13686_, 2023.


Yen-Ju Lu, Zhong-Qiu Wang, Shinji Watanabe, Alexander Richard, Cheng Yu, and Yu Tsao. Conditional diffusion probabilistic model for speech enhancement. In _ICASSP 2022-2022 IEEE In-_
_ternational_ _Conference_ _on_ _Acoustics,_ _Speech_ _and_ _Signal_ _Processing_ _(ICASSP)_, pp. 7402–7406.
Ieee, 2022.


Paul Lueg. Process of silencing sound oscillations. _US patent 2043416_, 1936.


Zhengding Luo, Dongyuan Shi, and Woon-Seng Gan. A hybrid sfanc-fxnlms algorithm for active
noise control based on deep learning. _IEEE Signal Processing Letters_, 29:1102–1106, 2022.


Zhengding Luo, Dongyuan Shi, Xiaoyi Shen, and Woon-Seng Gan. Unsupervised learning based
end-to-end delayless generative fixed-filter active noise control. In _ICASSP_ _2024-2024_ _IEEE_
_International_ _Conference_ _on_ _Acoustics,_ _Speech_ _and_ _Signal_ _Processing_ _(ICASSP)_, pp. 441–445.
IEEE, 2024.


11


Yehuda Mishaly, Lior Wolf, and Eliya Nachmani. Deep active speech cancellation with multi-band
mamba network, 2025. [URL https://arxiv.org/abs/2502.01185.](https://arxiv.org/abs/2502.01185)


Alireza Mostafavi and Young-Jin Cha. Deep learning-based active noise control on construction
sites. _Automation in Construction_, 151:104885, 2023.


Philip Arthur Nelson and Stephen J Elliott. _Active control of sound_ . Academic press, 1991.


Kuldip Paliwal, Belinda Schwerin, and Kamil W´ojcicki. Speech enhancement using a minimum
mean-square error short-time spectral modulation magnitude estimator. _Speech Communication_,
54(2):282–305, 2012.


Ashutosh Pandey and DeLiang Wang. A new framework for supervised speech enhancement in the
time domain. In _Interspeech_, pp. 1136–1140, 2018.


JungPhil Park, Jeong-Hwan Choi, Yungyeo Kim, and Joon-Hyuk Chang. Had-anc: A hybrid system
comprising an adaptive filter and deep neural networks for active noise control. In _Proceedings of_
_the Annual Conference of the International Speech Communication Association, INTERSPEECH_,
volume 2023, pp. 2513–2517. International Speech Communication Association, 2023.


Santiago Pascual, Antonio Bonafonte, and Joan Serra. Segan: Speech enhancement generative
adversarial network. _arXiv preprint arXiv:1703.09452_, 2017.


Alexander Pike and Jordan Cheer. Generalized performance of neural network controllers for feedforward active control of nonlinear systems. 2023.


Mirco Ravanelli, Titouan Parcollet, Adel Moumen, Sylvain de Langen, Cem Subakan, Peter
Plantinga, Yingzhi Wang, Pooneh Mousavi, Luca Della Libera, Artem Ploujnikov, et al. Opensource conversational ai with speechbrain 1.0. _Journal of Machine Learning Research_, 25(333):
1–11, 2024.


Xing Ren and Hongwei Zhang. An improved artificial bee colony algorithm for model-free active noise control: algorithm and implementation. _IEEE_ _Transactions_ _on_ _Instrumentation_ _and_
_Measurement_, 71:1–11, 2022.


Dario Rethage, Jordi Pons, and Xavier Serra. A wavenet for speech denoising. In _2018_ _IEEE_
_International Conference on Acoustics, Speech and Signal Processing (ICASSP)_, pp. 5069–5073.
IEEE, 2018.


Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, and Timo Gerkmann. Speech
enhancement and dereverberation with diffusion-based generative models. _IEEE/ACM_ _Trans-_
_actions_ _on_ _Audio,_ _Speech,_ _and_ _Language_ _Processing_, 31:2351–2364, 2023. ISSN 2329-9304.
doi: 10.1109/taslp.2023.3285241. URL [http://dx.doi.org/10.1109/TASLP.2023.](http://dx.doi.org/10.1109/TASLP.2023.3285241)
[3285241.](http://dx.doi.org/10.1109/TASLP.2023.3285241)


Antony W Rix, John G Beerends, Michael P Hollier, and Andries P Hekstra. Perceptual evaluation
of speech quality (pesq)-a new method for speech quality assessment of telephone networks and
codecs. In _2001_ _IEEE_ _international_ _conference_ _on_ _acoustics,_ _speech,_ _and_ _signal_ _processing._
_Proceedings (Cat. No. 01CH37221)_, volume 2, pp. 749–752. IEEE, 2001.


Shrishti Saha Shetu, Emanu¨el AP Habets, and Andreas Brendel. Gan-based speech enhancement for
low snr using latent feature conditioning. In _ICASSP 2025-2025 IEEE International Conference_
_on Acoustics, Speech and Signal Processing (ICASSP)_, pp. 1–5. IEEE, 2025.


Dongyuan Shi, Bhan Lam, Kenneth Ooi, Xiaoyi Shen, and Woon-Seng Gan. Selective fixed-filter
active noise control based on convolutional neural network. _Signal Processing_, 190:108317, 2022.


Dongyuan Shi, Bhan Lam, Xiaoyi Shen, and Woon-Seng Gan. Multichannel two-gradient direction
filtered reference least mean square algorithm for output-constrained multichannel active noise
control. _Signal Processing_, 207:108938, 2023.


Dongyuan Shi, Woon-seng Gan, Xiaoyi Shen, Zhengding Luo, and Junwei Ji. What is behind the
meta-learning initialization of adaptive filter?—a naive method for accelerating convergence of
adaptive multichannel active noise control. _Neural Networks_, 172:106145, 2024.


12


Wooseok Shin, Byung Hoon Lee, Jin Sob Kim, Hyun Joon Park, and Sung Won Han. Metricganokd: multi-metric optimization of metricgan via online knowledge distillation for speech enhancement. In _International Conference on Machine Learning_, pp. 31521–31538. PMLR, 2023.


Deepali Singh, Rinki Gupta, Arun Kumar, and Rajendar Bahl. Enhancing active noise control
through stacked autoencoders: Training strategies, comparative analysis, and evaluation with
practical setup. _Engineering Applications of Artificial Intelligence_, 135:108811, 2024.


Cem Subakan, Mirco Ravanelli, Samuele Cornell, Mirko Bronzi, and Jianyuan Zhong. Attention is
all you need in speech separation, 2021. [URL https://arxiv.org/abs/2010.13154.](https://arxiv.org/abs/2010.13154)


Cees H Taal, Richard C Hendriks, Richard Heusdens, and Jesper Jensen. A short-time objective
intelligibility measure for time-frequency weighted noisy speech. In _2010_ _IEEE_ _international_
_conference on acoustics, speech and signal processing_, pp. 4214–4217. IEEE, 2010.


Wenxin Tai, Fan Zhou, Goce Trajcevski, and Ting Zhong. Revisiting denoising diffusion probabilistic models for speech enhancement: Condition collapse, efficiency and refinement. In _Proceed-_
_ings of the AAAI conference on artificial intelligence_, volume 37, pp. 13627–13635, 2023.


Joachim Thiemann, Nobutaka Ito, and Emmanuel Vincent. The diverse environments multi-channel
acoustic noise database (demand): A database of multichannel environmental noise recordings.
In _Proceedings of Meetings on Acoustics_, volume 19. AIP Publishing, 2013.


Christophe Veaux, Junichi Yamagishi, and Simon King. The voice bank corpus: Design, collection
and data analysis of a large regional accent speech database. In _2013_ _international_ _conference_
_oriental_ _COCOSDA_ _held_ _jointly_ _with_ _2013_ _conference_ _on_ _Asian_ _spoken_ _language_ _research_ _and_
_evaluation (O-COCOSDA/CASLRE)_, pp. 1–4. IEEE, 2013.


Kai Wang, Bengbeng He, and Wei-Ping Zhu. Tstnn: Two-stage transformer based neural network for
speech enhancement in the time domain. In _ICASSP 2021 - 2021 IEEE International Conference_
_on Acoustics, Speech and Signal Processing (ICASSP)_, pp. 7098–7102. IEEE, June 2021. doi: 10.
[1109/icassp39728.2021.9413740. URL http://dx.doi.org/10.1109/ICASSP39728.](http://dx.doi.org/10.1109/ICASSP39728.2021.9413740)
[2021.9413740.](http://dx.doi.org/10.1109/ICASSP39728.2021.9413740)


Simon Welker, Julius Richter, and Timo Gerkmann. Speech enhancement with score-based generative models in the complex stft domain. _arXiv preprint arXiv:2203.17004_, 2022.


Tong Xiao, Buye Xu, and Chuming Zhao. Spatially selective active noise control systems. _The_
_Journal of the Acoustical Society of America_, 153(5):2733–2733, 2023.


Yang Xiao and Rohan Kumar Das. Tf-mamba: A time-frequency network for sound source localization. _arXiv preprint arXiv:2409.05034_, 2024.


Moujia Ye and Hongjie Wan. Improved transformer-based dual-path network with amplitude and
complex domain feature fusion for speech enhancement. _Entropy_, 25(2):228, 2023.


Hao Zhang and DeLiang Wang. Deep anc: A deep learning approach to active noise control. _Neural_
_Networks_, 141:1–10, 2021.


Hao Zhang and DeLiang Wang. Deep mcanc: A deep learning approach to multi-channel active
noise control. _Neural Networks_, 158:318–327, 2023.


Hao Zhang, Ashutosh Pandey, and DeLiang Wang. Attentive recurrent network for low-latency
active noise control. In _INTERSPEECH_, pp. 956–960, 2022a.


Hao Zhang, Ashutosh Pandey, et al. Low-latency active noise control using attentive recurrent
network. _IEEE/ACM_ _transactions_ _on_ _audio,_ _speech,_ _and_ _language_ _processing_, 31:1114–1123,
2023a.


Huawei Zhang, Jihui Zhang, Fei Ma, Prasanga N Samarasinghe, and Huiyuan Sun. A time-domain
multi-channel directional active noise control system. In _2023 31st European Signal Processing_
_Conference (EUSIPCO)_, pp. 376–380. IEEE, 2023b.


13


Qiquan Zhang, Hongxu Zhu, Xinyuan Qian, Eliathamby Ambikairajah, and Haizhou Li. An exploration of length generalization in transformer-based speech enhancement. In _Interspeech_
_2024_, pp. 1725–1729. ISCA, September 2024. doi: 10.21437/interspeech.2024-1831. URL
[http://dx.doi.org/10.21437/interspeech.2024-1831.](http://dx.doi.org/10.21437/interspeech.2024-1831)


Shucong Zhang, Malcolm Chadwick, Alberto Gil CP Ramos, and Sourav Bhattacharya. Crossattention is all you need: Real-time streaming transformers for personalised speech enhancement.
_arXiv preprint arXiv:2211.04346_, 2022b.


Yang Zhou, Haiquan Zhao, and Dongxu Liu. Genetic algorithm-based adaptive active noise control
without secondary path identification. _IEEE Transactions on Instrumentation and Measurement_,
2023.


14