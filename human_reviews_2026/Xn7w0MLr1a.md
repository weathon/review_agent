# UniFlow-Audio: Unified Flow Matching for Audio Generation from Omni-Modalities

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Audio generation, including speech, music and sound effects, has advanced rapidly in recent years.
These tasks can be divided into two categories: time-aligned (TA) tasks, where each input unit corresponds to a specific segment of the output audio (e.g., phonemes aligned with frames in speech synthesis); and non-time-aligned (NTA) tasks, where such alignment is not available.
Since modeling paradigms for the two types are typically different, research on different audio generation tasks has traditionally followed separate trajectories.
However, audio is not inherently divided into such categories, making a unified model a natural and necessary goal for general audio generation.
Previous unified audio generation works have adopted autoregressive architectures, while unified non-autoregressive approaches remain largely unexplored.
In this work, we propose UniFlow-Audio, a universal audio generation framework based on flow matching.
We propose a dual-fusion mechanism that temporally aligns audio latents with TA features and integrates NTA features via cross-attention in each model block.
Task-balanced data sampling is employed to maintain strong performance across both TA and NTA tasks.
UniFlow-Audio supports omni-modalities, including text, audio, and video.
By leveraging the advantage of multi-task learning and the generative modeling capabilities of flow matching, UniFlow-Audio achieves strong results across 7 tasks using fewer than 8K hours of public training data and under 1B trainable parameters.
Even the small variant with only $~$200M parameters shows competitive performance, highlighting UniFlow-Audio as a potential non-auto-regressive foundation model for audio generation.
Code and models will be available at https://anonymous3387a8c.github.io/uniflow_audio.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents UniFlow-Audio, a unified flow matching–based audio generation framework that handles both time-aligned (TA) and non-time-aligned (NTA) tasks within a single non-autoregressive model. A dual-fusion mechanism and block-wise fusion enable task-specific conditioning without interference, and the model achieves competitive or superior performance across seven audio tasks while reducing sampling steps compared to diffusion-based baselines.

### Strengths
**1. Unified Framework for Diverse Audio Tasks**

Successfully integrates both TA and NTA tasks (e.g., TTS, SE, V2A, T2A) under a single flow matching model — an elegant and generalizable design.

**2. Effective Fusion and Task Handling**

The proposed block-wise dual-fusion mechanism effectively balances multi-task learning while maintaining performance across modalities.

**3. Well-Written and Clearly Structured Paper**

The paper is well-organized and easy to follow, with clearly described limitations and detailed design explanations that improve overall readability and transparency.

### Weaknesses
**1. Lack of Analysis on Fusion Depth**

Although block-wise fusion improves performance, the paper does not analyze which layers (early or late) contribute most to the gain.

**2. Missing Comparison with Prior Unified Models**

The paper lacks direct experimental comparisons with existing unified frameworks such as UniAudio or AudioX, making the claimed advantage of UniFlow-Audio less convincing. (Despite of line 318)

### Questions
**1. On Fusion Depth Analysis**

The paper demonstrates that block-wise fusion is more effective than input-level fusion, but it does not specify which layers (early, middle, or late) contribute most to this improvement. Could the authors provide additional ablation or simple explanation to show how fusion depth impacts performance?

**2. On Comparison with Prior Unified Models**

The paper mentions that previous unified models (e.g., UniAudio, AudioX) fail to perform consistently across TA and NTA tasks.
Could the authors evaluate or explain these prior models under both TA and NTA settings to verify whether UniFlow-Audio achieves uniformly higher performance, or whether its advantage mainly comes from averaging stronger results across specific tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes UniFlow-Audio, a unified non-autoregressive (NAR) framework for diverse audio generation tasks using flow matching. The work divides audio generation problems into two fundamental types, time-aligned (TA) (e.g., TTS, SE, V2A) and non-time-aligned (NTA) (e.g., T2A, T2M), and aims to handle both categories within a single model.

The key contributions include:
- Dual-fusion mechanism: temporally aligned addition for TA tasks and cross-attention for NTA tasks, with task-irrelevant streams replaced by dummy embeddings to avoid interference.
- Block-wise integration of TA/NTA information across Transformer layers.
- Task-balanced sampling to mitigate data imbalance between TA and NTA tasks.

The model is trained on 7.7khours of public data (< 1B parameters) and evaluated on seven tasks spanning text, audio, and video input modalities. Experiments show competitive or superior results over single-task diffusion or autoregressive baselines.

### Strengths
- The distinction between TA and NTA tasks is well motivated and provides a unified view of disparate audio generation problems.
- The dual-fusion mechanism with dummy embeddings is new, practically effective way to disentangle fusion paths without separate models.
- The paper systematically validates dual-fusion, block-wise fusion, and balanced sampling, showing each contributes meaningfully. Evaluation across seven tasks demonstrates strong parameter efficiency and cross-task synergy, particularly impressive for a 200m-parameter model trained on modest data.

### Weaknesses
- The paper explicitly omits UniAudio (Yang et al., 2024) and AudioX (Tian et al., 2025) from quantitative tables, citing data-size or task-scope differences. However, this makes it difficult to gauge real progress: both are unified systems targeting similar objectives, and reporting normalized comparisons (e.g., per-hour or per-parameter efficiency) would better situate the contribution. Moreover, for several tasks (e.g., TTS, SE), the chosen baselines such as NaturalSpeech 2 or DOSE are not necessarily the strongest on the specific datasets used (LibriTTS, VoiceBank+Demand). Including more competitive or domain-matched baselines.
- While coverage is broad, each task is shallowly evaluated. Subjective MOS studies lack listener statistics.
- Dense implementation detail sometimes obscures the intuition (e.g., Section 3.4). A clearer separation between conceptual and engineering design would improve readability.

### Questions
- While we acknowledging the differences on data, it is still necessary to report the results from uniaudio or audiox with clear noting about the data differneces
- How stable is training when mixing TA and NTA tasks? Any gradient interference observed?
- For subjective MOS, how many raters and samples per task were used? Any significance testing?
- Is the same VAE shared across all domains, or are domain-specific VAEs used during pre-training?
- For NTA tasks where L_{dur-seq} is omitted, does the imbalance affet the shared backbone?
- How are the dummy embeddings initialized? Are they learned per task?
- For task balanced sampling, you upsample T2A and T2M. Did you test continuous weighting in the loss instead of discrete resampling? How sensitive are results to this ratio?
- Since AudioSR outputs 48khz and you downsample to 24kHz for comparison, does this favor UniFlow-Audio? Would an upsampled version maintain quality?
- As in UniAudio, some joint inter-task benefits are observed, Did you test whether training on V2A data improves T2A or TTS or vise versa? Some insight on inter-task transfer would highlight the benefit of unification.
- Can UniFlow-Audio generalize to unseen input types (e.g., image → audio) given CLIP encoders, or is each task explicitly conditioned?
- The explanation about noise amplification in SE is interesting, could you provide numerical trends or error bars to support that observation?
- Please report training and inference time per second of audio and GPU utilization to substantiate the efficiency claims of flow matching.
- The total 7.7 kh of data mixes speech, music, and environmental sounds. How much of the performance improvement arises from cross-domain exposure versus architectural changes?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates a unified audio generation model that can generate speech, audio and music for several tasks including text-to-speech, text-to-audio, speech enhancement, etc. The architecture design based on flow matching aims to solve the time-aligned and non-time-aligned tasks simultaneously. The experiments show the improvements obtained by the proposed unified model compared to individual task-specific models.

### Strengths
Flow matching architecture is verified effective for unified modeling.
Better performance is achieved by the proposed flow matching model, compared to task-specific models.
The discussion of unified models from the perspective of time alignment is novel. 
Instruction-based task specifications are enabled for the unified model.

### Weaknesses
The novelty is lacking, given there are several previous work on unified audio generation models. This paper only creates a new flow matching-based architecture for unified audio model development.
The motivation of the proposed framework is not strong enough. Why previous autoregressive models are inferior in time-alignment task is not clearly articulated. In a language model style unified model, the time-alignment seems not a big issue, as the implicit self-attention mechanism takes care of the alignment automatically. 
The experiments can be improved, with previous unified models compared, to show the advantages of the proposed model. Current experiments are not benchmarked with any previous unified model.
The necessity of explicit consideration of time alignment is not well demonstrated.

### Questions
The author may want to illustrate the motivation, why the designed architecture is better in tackling time-alignment and non-time-alignment tasks, with some analysis or observations to better motivate. 

Some previous works, e.g. E2 TTS, have shown that the alignment between condition and the target is not a problem in diffusion/flow machine-based architectures, it may be also worth analyzing the duration adaptor that provides the alignment explicitly (for time-aligned tasks) with ablation studies to check contributions of the duration adaptor.

Eq (1) is strange, the function Attn() has two identical input arguments. Also, Eq (1) seems different from what’s shown in Fig. 2. In Eq(1), the sum of attention outputs and content representation C yields C^I, the so called task-involved content embedding. However, Fig. 2 shows C is used as query for the attention module to obtained the C^I. The figure and the equation could be made clearer.

Table 1 doesn’t compare with previous unified models, e.g. UniAudio. Although it can be argued that previous unified models, e.g. UniAudio, use much more data and parameters than the proposed model. Then it may be a fair question to answer: Is there an advantage of the proposed architecture compared to the previous unified model architectures that are trained on the same data size using with similar parameter sizes? Which can also partially show the necessity of explicitly considering time alignment.

### Soundness
2

### Presentation
3

### Contribution
2
