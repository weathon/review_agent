# YuE: Scaling Open Foundation Models for Long-Form Music Generation

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 2

## Abstract
We tackle the task of long-form music generation, particularly the challenging \textbf{lyrics-to-song} problem, by introducing \textbf{YuE (乐)}, a family of open-source music generation foundation models. Specifically,
YuE scales to trillions of tokens and generates up to five minutes of music while maintaining lyrical alignment, coherent musical structure, and engaging vocal melodies with appropriate accompaniment. It achieves this through \textbf{track-decoupled next-token prediction} to overcome dense mixture signals, and \textbf{structural progressive conditioning} for long-context lyrical alignment. In addition, we redesign the \textbf{in-context learning} technique for music generation, enabling bidirectional content creation, style cloning, and improving musicality. Through extensive evaluation, we demonstrate that YuE matches or even surpasses some of the proprietary systems in musicality and vocal agility (as of 2025-01). We strongly encourage readers to \textbf{listen to our demo}\footnote{\url{https://map-yue.github.io/}}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a generative model for lyrics to full song generation.
The authors observe that generating mixed tracks (voice + accompaniment) directly leads to a loss of intelligibility, even more pronounced for some musical genres. The proposed approach to overcome this issue is to rely on source separation to decompose tracks into vocals + accompaniment and to encode each component separately.
The vocals and accompaniment tokens are then interleaved so that the authors can rely on a standard autoregressive (based on Llama 2) next token prediction. In practice, only the first codebook of the RVQ representations is modelled in this stage, and a second stage consists in predicting the remaining RVQ codes.

There's an accompanying website as well as an open source + weights release.
Evaluations are thorough and go beyond simple audio-quality measures, with a real emphasis on vocals.

### Strengths
- Modelling choices are coherent and well motivated: e.g. generating first interleaved vocals-accompaniment RVQ tokens, splitting full-song generation into multiple segments.
- Open weights + open source is a plus, as song genAI models with vocals are rare. Authors also discuss the ethical issue and explain their scraping process and the questions this raises.
- Extensive experiments
- Many details and discussions about dataset, observations, evaluation.

### Weaknesses
Some parts could be better explained like MUSIC IN-CONTEXT LEARNING, this introduces many notations that are not useful.
Structural progressive conditioning is also evoked too quickly.
Maybe being clearer in the main text that instead of lyrics to song, the problem is in fact casted as a "lyrics with structure" to song. This is clear in appendix I.
- The way ICL is used (i.e. by conditioning on an existing song chorus) highly conditions the type of possible applications.

### Questions
- Is the dataset available?
- For the 30s context given for in-context learning, do you make sure to consider a part that is different (like not providing Verse 2 to generate Verse 1) as these are expected to be extremely similar for the instrumental parts?
- How is it possible to do condition on a 30s context and ask to generate the full song?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents YuE for lyrics to song generation. It aims to generate full songs (up to five minutes) with vocals, accompaniment, and makes sure that the lyrics would align with the melody/companion.

The model is based on a two-stage autoregressive framework using discrete audio codec. The first stage models lyrics to first layer audio tokens, and then the second stages models first layer tokens to all residual tokens. 

The paper introduces three key techniques:

Dual-NTP: Each timestep in the first stage LM generates one vocal token and one companion token, instead of directly modeling the mixture.

SPC: A new way to organize training data. Songs are segmented into sections (intro, verse, chorus, etc.), and their lyrics and audio tokens are interleaved section by section. This helps the model maintain long-range lyrical alignment.

Music ICL (In-Context Learning): A modified in-context learning setup where a 30-second reference audio is prepended to SPC data.
For evaluation, YuE performs comparably to Tiangong and Udio, better than Hailuo, but below Suno V4. Objective metrics (KL divergence, FAD, alignment scores) also show strong performance.

For evaluation, YuE performs comparably to Tiangong and Udio, outperforming Hailuo but falling short of Suno V4. Objective metrics (KL divergence, FAD, alignment scores) also show strong performance.

Overall, YuE is positioned as the first open-source full-song lyrics-to-song model with quality approaching commercial systems.

### Strengths
1.	This work presents the first open-sourced song generation model that achieves performance close to commercial systems.
2.	Implementing and training a full-length audio LM of this scale is a major technical achievement.
3.	Dual-NTP and SPC integrate domain-specific musical structure into the generative process, representing a meaningful methodological contribution.
4.	Despite limited access to proprietary data, YuE achieves results comparable to those of closed-source commercial models

### Weaknesses
1.	The descriptions of data preprocessing are somewhat brief. It would help to clarify how vocal–accompaniment tracks were separated and how Creative Commons data were curated.

2.	The “all-in-one method” used in the paper could be elaborated.

3.	The paper could provide more analysis of sample diversity and potential copyright overlap (e.g., unintentional melodic copying). Clarifying whether such issues were observed would strengthen credibility.

4.	Prior research on multi-track or separated-source generation [1–3] should be acknowledged to better contextualize the contribution.
[1] Mariani, Giorgio, et al. "Multi-Source Diffusion Models for Simultaneous Music Generation and Separation." The Twelfth International Conference on Learning Representations.
[2] Xu, Zhongweiyang, et al. "Multi-Source Music Generation with Latent Diffusion." Audio Imagination: NeurIPS 2024 Workshop AI-Driven Speech, Music, and Sound Generation.
[3] Karchkhadze, Tornike, et al. "Multi-track musicldm: Towards versatile music generation with latent diffusion model." International Conference on ArtsIT, Interactivity and Game Creation. Cham: Springer Nature Switzerland, 2024.

### Questions
1.	How are the tracks separated from the mixture? Does the separation process introduce perceivable artifacts that could impact performance? What separation model is being used?

2.	What is the dataset used in detail? Is there any preprocessing to clean the dataset?

3.	For the dual-track tokens, the voice should have much lower entropy than the accompaniment. Have you considered using a low-bitrate codec for voice?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces YuE, a family of open-source foundation models designed for long-form music generation, and more specifically, the lyrics-to-song problem. The model is trained on trillions of tokens and can generate coherent, lyrically-aligned songs of up to five minutes in length. YuE achieves its performance through several key technical contributions:

- Structural Progressive Conditioning (SPC): This approach uses the inherent musical structure (like verses and choruses) to enable long-context lyrical alignment and overall song structure control, which is essential for full-song generation.
- Redesigned In-Context Learning (ICL): A novel framework for music that allows for advanced use cases such as style transfer, voice cloning, and bidirectional content creation.
- Track-Decoupled Next-Token Prediction (Dual-NTP): A variation of existing multi-track/multi-stem music generation methods here adapted for jointly generating the vocal and accompaniment tracks. The method is reported to improve lyrical intelligibility and to overcome problems related to dense signal mixtures.

Through extensive human evaluation, the model is shown to be competitive with, and in some aspects, surpasses widely used proprietary music-generation systems (like Tiangong and Udio) in musicality and vocal agility.

### Strengths
### Strength

- innovative SPC training strategy taking into account musical structure
- successful evaluation demonstrating competitive performance with state-of-the-art music generation models 
- open source availability

### Weaknesses
### References are outdated, discussion of relevant existing approaches is incomplete. 

- paper states in the abstract to refer to the state when the model was initially published in January 2025.
- The list of references appears to be limited when it comes to covering more recent approaches. However, relevant approaches have appeared early in 2025. Here we consider prioublciation in 2026. Therefore appacohes from early 205 should be included into the discussion 

- NTP decoupled next token prediction. Rather similar approaches have been proposed here:

Liu, 2025. "SongGen: A Single Stage Auto-regressive Transformer for Text-to-Song Generation", https://arxiv.org/pdf/2502.13128
Rouard, 2025. "MusicGen-Stem: Multi-stem music generation and edition through autoregressive modeling" https://arxiv.org/pdf/2501.01757

I think these papers should be discussed under related works, and they should be compared to the proposed NTP strategy. 

- concerning SPC a related approach has been published in

Lam, 2025. "MusiCoT: Analyzable Chain-of-Musical-Thought Prompting for High-Fidelity Music Generation", https://arxiv.org/pdf/2503.19611

This alternative approach for including musical structure into music generation should at least be mentioned. I note that the paper features a comparison with YuE.

- ICL

The paper makes the point of comparing with speech/TTS ICL methods. Given today, there exist a few approaches supporting lyricx conditioning for music generation, it appears a bit outdated to compare with these methods from a different problem domain.

### Questions
Please see under weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents YuE, a lyrics-to-song generation model that aims to produce long-form music generation with vocals and accompaniment. The authors propose a two-stage autoregressive architecture operating on discrete music tokens (X-Codec). The first stage predicts the first codebook of the music tokens by givens the semantic conditions (lyric, and others). and the second stage makes up the token predictions of the higher codebooks. To support long-form generation, the authors introduce Structural Progressive Conditioning (SPC), which uses section-level structure tokens to progressively condition the LM. Experimental results demonstrates that YuE performs competitively with commercial models (Suno, Udio, Hailuo).

### Strengths
The paper has two strengths.

1.The usage of long-form structure conditioning (SPC) is novel. Using section metadata to segment data and guide generation is a practical strategy for improving macro-level structure, and the staged conditioning improves continuity.

2.The experimental design is reasonable and comprehensive. The paper includes both objective metrics and human preference studies among YuE and four baselines. The results (and the demo website) show multi-minute outputs with acceptable alignment and structure.

### Weaknesses
The paper has several critical weaknesses that hinders its acceptance.

1.The overall novelty of this paper is limited, especially compared to the same period and already published work (SongBloom [1]) in NeurIPS 2025. Specifically, the two-stage autoregressive LM with Dual-NTP and SPC is incremental relative to SongBloom. And SongBloom even proposed more modern formulation: 

1 (a). SongBloom applies a unified autoregressive-diffusion architecture that interleaves semantic sketching and acoustic refinement, enabling significantly stronger acoustic quality with comparable modeling complexity. 

1 (b). SongBloom also covers the voice and accompaniment tokens by leveraging the Demucs separation model to extract the music data.

1 (c). SongBloom covers essentially all YuE capabilities and more, including lyric-to-music generation, structured multi-section music, reference audio prompting, diffusion decoder for refinement, disentanglement of semantic and acoustic tokens, and also open-source release.

1 (d). Despite having very similar high-level goals and tokenization methods, YuE does not compare to SongBloom, although SongBloom explicitly compares against YuE and reports better performance.

Compared to SongBloom, YuE’s design is behind state-of-the-art, which lacks diffusion-based refinement, Interleaving of semantic + acoustic token, and robust modeling of reference-prompt conditions. Therefore, YuE does not demonstrate novelty or superiority over SongBloom. At present, it is difficult to justify acceptance when a clearly more advanced method has already been published.

2.The experimental design is comprehensive but the baseline design is problematic. Apparently, YuE does not compare to the most relevant works, including SongBloom and ACE-Step [2]. Although YuE includes industry comparisons (Suno, Udio, Tiangong), the lack of comparison to academic/open models with similar architectures weakens the empirical claims. Also there is no ablation studies to isolate the contributions of Dual-NTP (decoupled vs non-decoupled), SPC, and Data-scale effects. And such ablation studies are very imperative to fully demonstrate YuE's contribution and its improvement to prior works. From the experimental results, YuE does not demonstrate superiority. It does not outperform Suno V4, and remain slightly behind Udio and Tiangong. It is acceptable that the model remains competitive but not superior to industrial models. But to fully demonstrate the proposed method, more ablation studies and comparisons to academical or open-source models should be considered. At the same time, SongBloom reports stronger performance over YuE. Thereofore, YuE does not establish SOTA status for either research or production settings.

3.The data sourcing transparency and licensing concerns remain vague in the paper. The paper claims to train on around 650k hours of creative common (CC) music. However, there are several critical concerns. The actual data sources are not specified. The platforms, artists, catalogs remain unknown, and the licenses are not enumerated (CC-BY and CC-BY-NC are completely different). It is unclear whether scraping is permitted or whether data came from commercial music platforms (plus that, whether the authors of YuE own the copyrights of these data for AI training). And there is no dataset preview or download interface provided. So reproducibility, legality, and downstream use constraints remain ambiguous. Given active litigation around music AI, such as Suno and Udio to three major labels (Sony, UMG, and Warner), this raises concerns about copyright and reproducibility. Till now, only Udio and UMG come to an agreement of data for music AI training. Open-source models must satisfy a higher legal/ethical bar, and YuE’s data description is too vague to be evaluated.

4.The claim and the vibe of novelty overstates the contribution. The paper claims that “As of its release on Jan. 28, 2025, YuE familiy is the first publicly available, open-source lyrics-to-song model capable of full-song generation with quality on par with commercial systems". As a reviewer, this makes an excuse to prevent it from comparing to latest models after that. This is negative and questionable, especially given SongBloom’s public release and NeurIPS-publication timeline. The lack of acknowledgment or direct comparison reduces credibility.


[1] SongBloom: Coherent Song Generation via Interleaved Autoregressive Sketching and Diffusion Refinement. NeurIPS 2025.

[2] ACE-Step: A Step Towards Music Generation Foundation Model

### Questions
1. Why is SongBloom not included in your comparisons? Can you add some comparisons to it (and ACE-Step) and also give the evidence on the difference between your potential results to the results reported in their papers?

2. Can you provide detailed documentation of the 650k hours of CC-licensed music? Which platforms were used and under what specific license? How many tracks / hours from each source? Are licenses compatible with redistribution + model training? Do you own the copyright of using these data? If not, since you release the model, how can external users ensure they are free from copyright exposure? How can the community safely adopt YuE?

3. Please provide ablation studies including: Dual-NTP vs. single (combine voice and accompaniment) token prediction; SPC vs. no-SPC; and Data-scaling sensitivity (e.g., 10k/100k hours). For the data-scaling, it is encouraged to use the open-available data for training, even the results are not good (e.g., FMA, Jamendo, MusDB, MoisesDB, MedleyDB). Other data contains the lyric annotation might better to be used. 

4. For reproducibility, it is encouraged to have example data from your training set, and will you release the training script with data loader and processing, or just a simple inference code and model checkpoint?

### Soundness
2

### Presentation
2

### Contribution
2
