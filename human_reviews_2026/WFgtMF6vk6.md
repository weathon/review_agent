# TFMAudio: High-Fidelity Long-Form Text-to-Audio via Mamba-based Flow Matching

- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Recent advancements in audio generation have been dominated by transformer-based diffusion models, which face challenges in extrapolating positional encodings and exhibit quadratic complexity in self-attention, limiting their consistency and efficiency for long-form generation.
To address these limitations, we propose TFMAudio, a novel latent audio generation model that integrates the strengths of Flow Matching and a custom-designed TFMamba backbone.
TFMamba employs a dual-scan mechanism: TimeMamba captures long-range causal dependencies with linear complexity, while FrequencyMamba models spectral correlations such as harmonic structures. To enhance stability, we further introduce Energy-Aware Guidance (EAG), which mitigates state drift by adaptively regularizing classifier-free guidance. Experiments demonstrate that TFMAudio achieves state-of-the-art performance on text-to-audio benchmarks and exhibits robust extrapolation to ultra-long sequences. Remarkably, our model generates 30-minute high-fidelity audio while preserving temporal consistency and semantic alignment, significantly advancing the scalability and usability of text-to-audio models.
  Demo:https://huggingface.co/spaces/tfmaudio/TFMAudio

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes TFMAudio, a latent text-to-audio generator that replaces Transformer backbones with a Time–Frequency Mamba (TFMamba) block trained with flow matching, plus an Energy-Aware Adaptive Guidance (EAG) scheme to stabilize long generations.

### Strengths
Support ultra long audio generation. To my best knowledge, this is currently the TTA model that generates the longest duration.

### Weaknesses
- Lacking too many well-known baseline models for 10s audio generation. For example, Make-An-Audio 2, EzAudio, AudioGen and Tango series (Tango, Tango-AF, Tango2). I understand the authors only want to compare with 44.1kHz or 48kHz models, but since they claim state-of-the-art, comparing more existing models will be more convincing.

- AudioCap’s testing set has fewer than 1000 audio samples, each paired with 5 prompts. In the description of Section 5.1 for datasets, the authors mentioned that they evaluated their model on the AudioCaps testing set of 1,811 prompt-audio pairs. Thus, I think there are some audio ground truth samples reused more than once, paired with different prompts. This is not the standard way of evaluating on AudioCaps. Mostly, TTA models follow the evaluation of the AudioLDM series, which have predefined prompt-audio pairs. Some papers, like Ezaudio, randomly sample one prompt for each audio by themselves. However, it is unclear how this work derived a testing set with 1,811 prompt-audio pairs.

- In the original paper of TangoFlux, they achieve an IS score higher than 11. However, this work reports an IS score of 8.85 for TangoFlux. This shows that their configuration for the AudioCaps testing set severely affects the objective performance of TTA models.

- Please also report the FD score using PANNS, as this is the most common metric for evaluating fidelity.

- This work does not have human listening tests, MOS, or subjective user studies. Providing audio samples for demo is not enough.

- For the demo in the link provided in the abstract, there is no reference clip for comparison. Prompts used to generate different durations of audio are also not the same. It is unable to compare the difference between short-form and long-form audio generation using the same prompt. It feels like they are just cherry-picking the good examples for different audio durations.

- Up to this point, it cannot be confirmed whether this work is state-of-the-art or not.

- If it is just competing on the AudioCaps evaluation set, we do not need 30-minute audio clips, as most of the audio in AudioCaps is just 10 seconds long. However, if the aim is to generate long-form audio, I don’t think this work has done long-form audio evaluations in a proper way. In section 5.2 for “Ultra Long Audio Generation”, they mention that the generated 30-minute audio is segmented into non-
overlapping 30-second clips, with each clip evaluated with objective metrics. It is not clear if they computed the FAD metric with the same audio ground truth clip for each segment. Assuming they did, this still does not really justify the internal coherence of a long audio. It mostly tests how much each segment resembles that single clip’s distribution (or fidelity). For calculating KL, if the reference is one clip’s logit distribution, the metric then rewards segments that mimic that clip’s class distribution, again saying nothing about cross-minute structure. You can have 
$D_{\mathrm{KL}}(p_i \| q)$ small for all $i$, yet $D_{\mathrm{KL}}(p_i \| p_j)$ large for some $i \ne j$. 
Here is an example: 
$ q=(0.7,0.2,0.1), p_1 =(0.80,0.19,0.01), p_2 =(0.60,0.21,0.19)$, where
$D_{\mathrm{KL}}(p_1 \| q)  = 0.074, D_{\mathrm{KL}}(p_2 \| q)  = 0.039, D_{\mathrm{KL}}(p_2 \| p_1) = 0.407, D_{\mathrm{KL}}(p_1 \| p_2) = 0.181$

- For IS, since it does not require a reference, there is no reference and no between-segment relation.
- For CLAP, consider the prompt “Birds chirping, dogs barking, and a duck quacking”. What does uniformly high CLAP per-segment score mean? Consider an audio clip that only has birds chirping in the first 10-minute segment, only dogs barking in the second 10-minute segment, and only a duck quacking in the last 10-minute segment. Should this audio clip have similar high CLAP scores for each segment? Even if it does, does it mean this whole audio clip is actually following the text prompt? Could an audio clip with all birds chirping still give you similar results by measuring CLAP score against the text prompt “Birds chirping, dogs barking, and a duck quacking”? I think it still yields similar CLAP scores for each segment in this case, as the audio is all birds throughout, each segment’s audio embedding will align strongly with the “birds” component of that text vector. As a consequence, uniform scores in this setup do not prove the audio covers dogs or ducks, nor any long-form structure, just that each segment sounds related to the composite prompt in aggregate.

- Again, no user study or evaluation on long-form audio generation.

- Latency should also be compared against other baseline TTA models, not just between TFMamba and the transformer-based version.

- The idea of EAG itself is novel, but the performance improvement compared to CFG is very minor as shown in Table 5. For 30 s generation: CLAP +0.0011, IS +0.014, FAD −0.0052, KL −0.0059 (vs. CFG).

- In conclusion, unfortunately, this paper is far from ready to meet the standards of ICLR at this stage.

### Questions
- In Table 1, for reference-dependent metrics like FAD, do you use the same ground-truth (≈10-second) clip as the reference regardless of the generated audio’s duration?

- What’s the rationale for generating a 30-minute clip from a very short prompt like "Birds chirping, then a dog barking, then a duck quacking?" Wouldn’t a long, detailed prompt describing the full 30 minutes be a more realistic test?

- If we want to generate a dog barking sound, are you sure your model can generate the same dog barking for 30 minutes?

- Lack of references: Line 50 is talking about autoregressive (AR) models, but the citations are mostly non-AR models.
Worth citing UniAudio, MusicGen, and AudioMNTP, which are also autoregressive text-to-audio or text-to-music generation models.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes TFMAudio, a text-to-audio generation framework that integrates Time-Frequency Mamba (TFM) and Energy-Aware Guidance (EAG). The model achieves linear-time long-form generation and maintains temporal and spectral consistency, producing up to 30-minute high-fidelity 44.1 kHz audio with strong semantic alignment.

### Strengths
**1. Effective Dual-Axis Modeling via Time-Frequency Mamba**

The proposed Time-Frequency Mamba performs 1D scans along both temporal and frequency axes, allowing the model to jointly capture temporal causality and spectral correlation — a capability that conventional Transformers struggle to achieve.

**2. Linear-Time Complexity for Long Sequences**

The Mamba-based recurrent formulation enables linear computational complexity O(L) with respect to sequence length, offering a far more efficient alternative to the quadratic O(L²) cost of Transformers while maintaining long-range dependencies.

**3. Stable and Scalable Audio Generation with Energy-Aware Guidance**

The introduced Energy-Aware Guidance (EAG) mitigates state drift by decomposing the flow-matching velocity field and adaptively damping unstable components, enabling reliable ultra-long (30-minute) 44.1 kHz audio generation with temporal consistency.

### Weaknesses
**1. Marginal Impact of Energy-Aware Guidance (EAG)**

According to the ablation results, the performance improvement from EAG is minimal, with only slight differences in objective metrics.

**2. Lack of Flexible Length Control in Generation**

While the paper demonstrates 10s and 30s generations, it is unclear whether the model allows arbitrary-length synthesis (e.g., 13s or 27s) or only supports pre-defined durations tied to training configurations.

### Questions
**1. On the Effectiveness of Energy-Aware Guidance (EAG)**

The ablation results suggest that EAG provides only marginal gains across objective metrics.
Could the authors elaborate on specific scenarios or qualitative aspects where EAG meaningfully contributes to stability or audio fidelity?


**2. On Length Controllability During Generation**

Can TFMAudio support arbitrary-length generation (e.g., 13s or 27s) beyond fixed training configurations?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces TFMAudio, a Mamba-based text-to-audio (T2A) model that integrates a Time-Frequency Mamba (TFMamba) backbone and Energy-Aware Guidance (EAG) for long-form generation. The model combines linear-complexity state-space modeling with flow matching to improve efficiency and stability over transformer-based diffusion models. The authors claim that TFMAudio achieves state-of-the-art performance on AudioCaps and WavCaps benchmarks and can generate up to 30 minutes of 44.1kHz audio with temporal consistency.

While the motivation—to overcome the quadratic complexity of transformers for long-form generation—is reasonable, the experimental evidence and claims are weak and overinterpreted. The model is only trained on 10-second clips, and therefore the claimed ability to generate consistent 30-minute audio is not supported by data. The long-form generation experiment effectively tests extrapolation far outside the training distribution and yields repetitive, semantically meaningless output. Overall, the paper reads more as a well-written engineering demonstration than a solid scientific contribution.

### Strengths
1. The paper is well-organized and technically clear. Mathematical derivations (flow matching, EAG) are properly explained, and the figures are informative.

2. Using Mamba for efficient long-range modeling is a timely and interesting idea. The dual-scan mechanism (TimeMamba + FrequencyMamba) provides a coherent architecture for time–frequency modeling.

### Weaknesses
1. The entire training uses 10-second AudioCaps/WavCaps clips, yet the main claim of the paper is about ultra-long (30-minute) generation. This makes the core contribution unverifiable—the model has never seen long-form data, so the “30-minute consistency” claim lacks credibility. The long audio is almost certainly repetitive or degenerate, as suggested by the flat metric curves and absence of qualitative human evaluation.


2. Only four metrics (CLAP, IS, FAD, KL) are reported—no subjective MOS tests, no human preference studies. The improvement margins over baselines are modest and not statistically validated. 

3.

### Questions
1. why the demo pages donot show the full 30 mins audio?

2. The Mamba structure can bring better performance than transformer?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposed TFMAudio, a Mamba-based flow-matching model capable of long-form audio generation.
The authors proposed to use a combination of TimeMamba and FrequencyMamba to improve results.
In addition, the authors proposed an energy-aware adaptive guidance (EAG) mechanism which adaptively adjusts guidance weight, further boosting performance.
Putting everything together, TFMAudio can generate 30 minutes of high-quality audio.

### Strengths
The generation results are strong. Linear complexity in sequence length is verified theoretically and empirically. Overall, the technical aspect of this work is quite solid.

If the weaknesses can be sufficiently addressed, I will consider increase the rating.

### Weaknesses
The writing of the paper can be improved:

- There are plenty of places where math symbols are not in a math environment. For example, O(L) in line 178, "d is the feature dimension" in line 268, etc.

- Figure organization is inconsistent. Some figures appear at the top (e.g., Figure 2), while others are in-line (e.g., Figures 3 and 4).

- Some claims in the background sections may not be accurate. For example, the authors claim that
  > When applying transformers to audio latents $x \\in \\mathbb{R}^{L \times C}$, conventional patchification treats the representation as an image grid and breaks the native channel-wise coupling and causal temporal structure.

  While some transformer-based models may treat audio signal in this manner, many do not, and instead model the audio latent embeddings as a 1-D sequence.

Additionally, while the authors compared TFMAudio to several strong baseline models, some recent high-performance methods, such as IMPACT [1], are missing. I suggest including these methods in Table 1.

I would also invite the authors to make some clarifications regarding the questions in the section below, and add the discussions to the appropriate sections of the paper.

[1] Huang et al. IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling.

### Questions
- If I understood it correctly, Mamba is causal. While it is understandable for TimeMamba to process the time-domain signal in a causal manner, it is unclear why FrequencyMamba should scan from the highest to the lowest channel, and not the other way around. Do you think a bi-directional FrequencyMamba can further improve the results?

- Do you think the effectiveness of FrequencyMamba is tied to the Stable Audio Open VAE used in this work? If a different continuous VAE is used, will FrequencyMamba still help? What if a discrete tokenize is employed instead?

- Is energy-aware adaptive guidance "bundled" with TFMAudio, or is it useful for general diffusion models? Is EAG necessitated by TFMAudio's properties? Since this paper is mostly about TFMAudio, I think it is important to clarify its relationship with EAG.

### Soundness
4

### Presentation
2

### Contribution
3
