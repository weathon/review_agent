# SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Multi-track music generation has garnered significant research interest due to its precise mixing and remixing capabilities. However, existing models often overlook essential attributes such as rhythmic stability and synchronization, leading to a focus on differences between tracks rather than their inherent properties. In this paper, we introduce SyncTrack, a synchronous multi-track waveform music generation model designed to capture the unique characteristics of multi-track music. SyncTrack features a novel architecture that includes track-shared modules to establish a common rhythm across all tracks and track-specific modules to accommodate diverse timbres and pitch ranges. Each track-shared module employs two cross-track attention mechanisms to synchronize rhythmic information, while each track-specific module utilizes learnable instrument priors to better represent timbre and other unique features. Additionally, we enhance the evaluation of multi-track music quality by introducing rhythmic consistency through three novel metrics: Inner-track Rhythmic Stability (IRS), Cross-track Beat Synchronization (CBS), and Cross-track Beat Dispersion (CBD). Experiments demonstrate that SyncTrack significantly improves the multi-track music quality by enhancing rhythmic consistency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors propose a model architecture where modules are connected to allow global and local cross-track attention. This resulted in improved scores in terms of FAD as well as the proposed CSB metric.

### Strengths
The core idea of the paper of using cross-attention globally & time-specifically is simple and intuitive. Based on a holistic evaluation by FAD and the specific evaluation by IRS, the proposed idea seems to achieve the goal. The paper also presents additional experiment results with ablation and subjective tests.

### Weaknesses
The background sections can be improved. It seems to cover the music generation well, but synchronized generation is an active research topic in audio generation in the context of videos.

The proposed process can be explained in more detail. The paper covers the high-level information well, but it lacks of the detailed operations.

It is nice to see a new proposal on evaluation metrics, but it needs more research and experiment to justify them.

### Questions
It would be much nicer if the proposed method is discussed in more detail. For example, from Eqs. (4-5), what is the dimensions of the W variables? What is the exact operation of Eq(4), especially Attn()? These two equations are too important to be discussed on this high level only. This also raise an issue of reproducibility: I'm not sure if the paper has enough details for the readers to reproduce the model.

I'm also curious and a bit concerned about the proposed metrics - CBS (mean, std, median) and CBD. Are they really ↑ to 1 and ↓ to 0? It may feel deviated from the original idea of the paper, but since these are proposed by the authors in this paper, I wonder what is the reasonable range of them, perhaps measured on real human composed and produced music signals. 

On FAD, SyncTrack indeed shows the lowest FAD scores, but why? The proposed method is about time sync. Why does it lead to a lower FAD?

Also on FAD, I would like more details. What was the model or source code used to measure it?

There are minor typeos, e.g. L159 - "estimator,offering". L441 - "Backbone w track".

### Soundness
3

### Presentation
3

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
This paper presents SyncTrack, a latent diffusion framework for unconditional multi-track music generation designed to enhance rhythmic stability and cross-track synchronization. The model integrates global cross-track attention with time-aligned cross-track attention, along with a track-specific (instrument-prior) branch that maintains timbral distinctions across instruments. The authors further introduce three reproducible evaluation metrics—IRS (intra-track rhythmic stability) and CBS/CBD (cross-track synchronization/dispersion)—and conduct experiments on Slakh2100 using a four-track configuration (16 kHz, ~10.24 s segments, ~200 DDIM steps). The proposed method achieves improved FAD (both mix-level and per-track) compared with waveform-based baselines, and the CBS↑/CBD↓ trends align with human listening evaluations.

### Strengths
1. The paper introduces clear and reproducible rhythm-centric evaluation metrics (IRS/CBS/CBD) and provides robustness analyses, making these metrics potentially useful for broader community adoption.
2. The results demonstrate good alignment between objective and subjective evaluations: FAD improvements are consistent with listening preferences, and component-level ablations further validate the architectural design.

### Weaknesses
1. The current setup (10.24 s at 16 kHz) constrains both long-range musical structure and high-frequency detail. Including evaluations on longer segments (≥30–60 s) or full-song contexts would strengthen the claims.
2. Key details such as the number of participants, votes per sample, confidence intervals, or statistical significance, and loudness normalization procedures should be clearly reported in the main paper.

### Questions
1. Could the authors clarify the listening-test protocol in the main text, including details such as the panel size, number of votes per item, statistical testing procedures, and loudness normalization, to improve transparency and reproducibility?

2. Could the authors provide a detailed efficiency and time analysis?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper targets a concrete missing piece in multi-track music generation: current models often produce plausible individual tracks that are not rhythmically aligned. To fix this, the authors propose SyncTrack, which (i) separates track-shared rhythm from track-specific instrument information, (ii) adds two cross-track attentions (global for overall groove, time-specific for exact same-time alignment), and (iii) introduces three rhythm-aware metrics (IRS, CBS, CBD) to evaluate stability and synchronization that FAD cannot capture. Experiments on 4-track Slakh2100 show better FAD and better rhythm metrics than recent multi-track baselines.

### Strengths
Very clear problem–solution alignment. The paper does not vaguely say “quality is low”; it pinpoints “rhythmic stability and cross-track synchronization are not modeled or evaluated,” and the proposed modules map 1:1 to that diagnosis. 

Metrics with reuse potential. IRS/CBS/CBD are defined in a way that any multi-track model with separated stems can use; this makes the paper more than “a new model,” it is also “a more appropriate test.” 

Realistic multi-track setup. Using four typical production tracks (bass, drums, guitar, piano) matches real downstream use (mixing, remixing, track-wise editing) much better than single-mix generation. 

Empirical validation + robustness. The authors not only report scores but also check metric robustness w.r.t. beat-tracking hyperparameters and human subjective ratings, which strengthens the claim that the new metrics are meaningful. 

Architecture that is easy to transplant. The separation between track-shared and track-specific modules is structurally simple and could be dropped into other latent-audio diffusion systems.

Conditional/partial generation not studied. A common real-world scenario is “I already have two tracks, generate a third one in sync.” The current setup is “generate all four together.” It would be good to see whether the same architecture can be conditioned on existing tracks and whether CBS/CBD still apply in that case.

### Weaknesses
Beat-detection dependency. All three rhythm metrics assume that beat/onset tracking works reasonably well. For highly expressive, weakly pulsed, or rubato multi-track music, tracking can fail, which would make IRS/CBS/CBD less reliable. The paper tests robustness to hyperparameters, but not to style changes; adding such a test would make the metric story stronger.

Dataset narrowness. Most results are on the Slakh2100 four-track configuration, which is clean and well-aligned. It is unclear how well SyncTrack and, more importantly, the metrics behave on genuine studio stems with small human timing jitter.

### Questions
Your three metrics all rely on a beat (or onset) extractor. How do you handle tracks that have very sparse or chordal content (e.g., piano pads) where the beat detector fails or produces inconsistent beat intervals? Do you ignore such tracks in IRS/CBS/CBD, or do you impute missing beats? What is the effect on CBS when one track is unreliable?

The two cross-track attention submodules are placed together in the track-shared module. Did you test alternative orderings (e.g., time-specific first, then global) or distributing them across different layers? If so, do results change significantly?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SyncTrack, a new model for synchronous multi-track waveform music generation. The model uses an innovative architecture to tackle the often-overlooked challenges of rhythmic stability and synchronization in multi-track music. It also incorporates two types of cross-track attention mechanisms: global cross-track attention, which ensures the rhythm stays consistent across tracks, and time-specific cross-track attention, which helps align musical events more precisely in time. The authors conducted thorough experiments to demonstrate the effectiveness of their approach, showing that it performs well in creating high-quality multi-track music.

### Strengths
1.The modules designed by the authors are highly aligned with the current needs of music generation technologies, and their effectiveness is demonstrated from both temporal and spatial perspectives.

2.The authors propose three novel metrics to evaluate the quality of multi-track music generation, addressing gaps in existing evaluation methods.

3.The experiments conducted by the authors are thorough and provide strong evidence supporting the effectiveness of the proposed approach.

### Weaknesses
1.The authors do not provide a clear explanation of the relationship between tracks, timbre, and rhythm in the music generation task, which may cause some difficulty in understanding the proposed approach.

2.The ablation studies are limited and the results are relatively average, making them less convincing in demonstrating the contributions of specific components.

3.The paper does not include a user study, leaving the subjective evaluation of the generated music quality unaddressed.

### Questions
1.Besides the FAD metric and the three proposed in the paper, are there other commonly used metrics for evaluating the quality of generated audio?

2.Why didn't the authors include a user study to subjectively score the generated results?

3.What is the relationship between different tracks and aspects such as rhythm and timbre? Does the model impact the quality of individual track generation?

### Soundness
3

### Presentation
2

### Contribution
3
