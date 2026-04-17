# Text2Lip: Progressive Lip-Synced Talking Face Generation from Text via Viseme-Guided Rendering

- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Generating semantically coherent and visually accurate talking faces requires bridging the gap between linguistic meaning and facial articulation. Although audio-driven methods remain prevalent, their reliance on high-quality paired audio visual data and the inherent ambiguity in mapping acoustics to lip motion pose significant challenges in terms of scalability and robustness. To address these issues, we propose Text2Lip, a viseme-centric framework that constructs an interpretable phonetic-visual bridge by embedding textual input into structured viseme sequences. These mid-level units serve as a linguistically grounded prior for lip motion prediction. Furthermore, we design a progressive viseme-audio replacement strategy based on curriculum learning, enabling the model to gradually transition from real audio to pseudo-audio reconstructed from enhanced viseme features. This allows for robust generation in both audio-present and audio-free scenarios. Finally, a landmark-guided renderer synthesizes photorealistic facial videos with accurate lip synchronization. Extensive evaluations show that Text2Lip outperforms existing approaches in semantic fidelity, visual realism, and modality robustness, establishing a new paradigm for controllable and flexible talking face generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Text2Lip, a text-conditioned talking-face pipeline that replaces the usual audio driver with a viseme-centric intermediate and a curriculum that progressively swaps real audio out for text-derived signals. The proposed framework has three parts: (1) viseme-centric text encoding, constructing an interpretable phonetic-visual bridge, (2) progressive viseme-audio replacement, facilitating flexible modality handling, and (3) photorealistic landmark rendering, synthesizing photorealistic facial videos with accurate lip synchronization. Experiments on GRID and AVDigits report strong visual quality, competitive sync metrics without real audio.

### Strengths
- The paper has clear motivation for the scarcity/fragility of high-quality audio-visual pairs. It also proposes a viseme-centric route that can operate without audio at inference. The PVAR schedule and pseudo-audio module are technically straightforward yet well-motivated

- The experimental section covers multiple axes: image/video fidelity (SSIM/PSNR/LPIPS/FID), synchronization (SyncNet), landmarks (DTW-P/MPJPE), and an “inverse” semantic metric (BLEU/WER via a lip-reading-style recognizer).

- The system remains compatible with strong rendering backends (EchoMimic) and shows text-only results that are visually competitive with audio-driven baselines on GRID/AVDigits, including a TTS baseline (“From IndexTTS”) that Text2Lip outperforms on several metrics.

### Weaknesses
- The paper describes text -> IPA -> viseme conversion and a Transformer with positional encodings, but does not explain how viseme durations are estimated when no audio exists (e.g., alignment, length regulator, or learned duration model). Without explicit duration modeling, it is unclear how the system determines frame counts, coarticulation timing, and phoneme-to-frame alignment—especially on open-domain sentences—beyond learning from fixed-length training clips.

- SyncNet scores (Sync-C/Sync-D) hinge on an audio stream; for text-only inference, the paper sometimes uses pseudo-audio, sometimes ground-truth audio for evaluation, but the exact protocol per dataset/setting is not well stated. This complicates fairness claims against purely audio-driven baselines.

- Dataset scope is very narrow (fixed-grammar GRID; small AVDigits), and the “self-collected” open-domain set lacks detail/release, limiting claims of generalization.

- In demo video, despite convincing lip motion, the speech track lacks expressive prosody (pitch/energy variation, natural pauses) and fine-grained articulatory cues (breaths, plosive bursts), making it sound non-human.

### Questions
- For each dataset/experiment in Tables 1–5, which audio stream (GT, pseudo, TTS, none) is used to compute Sync-C/Sync-D, and how is fairness enforced?

- Why choose NSLT for “inverse” semantic evaluation instead of standard lip-reading/visual-ASR baselines, and what sanity checks validate it?

- Have you tested alternative or learned viseme sets and accent/language robustness, and how sensitive are results to the mapping?

### Soundness
2

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
This paper tackles text-driven lip-synced facial animation. They propose a method which uses a viseme-based intermediate representation, a progressive viseme-audio replacement strategy, and a landmark-guided renderer to generate realistic, synchronized talking faces directly from text.

### Strengths
The paper presents a clear problem formulation and addresses an underexplored direction of generating lip motion directly from text. The overall framework is conceptually well structured and the pipeline is easy to follow. The integration of viseme-level modeling provides an interesting intermediate representation between linguistic and visual domains.

### Weaknesses
The proposed multi-stage design (text to viseme to pseudo-audio to landmark to renderer) appears unnecessarily complex and may accumulate errors across stages without clear justification or analysis of each component’s necessity.

Given the maturity of current text-to-speech (TTS) systems and high-performing audio-driven video generators, a natural question arises: why not decompose the task into a more straightforward two-stage pipeline, which is text-to-speech followed by speech-driven talking face synthesis? In Table 5, the comparison is limited to IndexTTS, which is a rather weak baseline. It would be much more convincing to evaluate the proposed method against strong modern TTS models (e.g., VITS2 [1], StyleTTS2 [2]) that can produce realistic audio, thereby offering a clearer and fairer comparison with the proposed viseme-pseudo-audio framework.

The necessity of the proposed curriculum-based audio dropout also seems over-engineered. Its effect might be achievable through simpler strategies such as random or mask-based modality dropout.

[1] VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design
[2] StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models

### Questions
The complexity of the proposed system may introduce compounding error propagation across stages. Are there any reported TTS quality metrics to evaluate how the pseudo-audio compares to realistic speech?

How does the proposed method handle coarticulation effects or transitions between visemes, which are inherently continuous but modeled categorically in the current framework?

### Soundness
3

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
4

### Summary
The paper propose Text2Lip, a viseme-centric framework that constructs an interpretable phonetic-visual bridge by embedding textual input into structured viseme sequences to address the inherent ambiguity in mapping acoustics to lip motion pose significant challenges in terms of scalability and robustness. These mid-level units serve as a linguistically grounded prior for lip motion prediction.  Furthermore, the paper design a progressive viseme-audio replacement strategy, enabling the model to gradually transition from real audio to pseudo-audio reconstructed from enhanced viseme features.

### Strengths
1. Speech-driven models often learn ambiguous mappings from audio to lip shapes. To address the issue, the paper explicitly models the linguistic-phonetic-visual hierarchy instead solely on audio, which serves as semantically grounded priors for facial motion synthesis.  
2. Text2lip surpasses other sota methods in the visual quality and semantic quality
3. Text2lip introduce a curriculum-based viseme-audio replacement strategy that facilitates flexible modality handling, supporting both audio-driven and audio-free generation scenarios.

### Weaknesses
1. Although using visemes instead of speech can resolve the ambiguity of the mapping, it may, at the same time, affect synchronization and rhythmic cadence (the results for sync-c and sync-d in Table 1 do not appear to be the best).
2. If the speech itself carries emotion, and it is a complex, changing emotion, will solely using text as input affect performance? Regarding this point, is it possible to conduct testing on an emotional dataset?

### Questions
The same in weakness.

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
4

### Summary
The paper proposes Text2Lip, a framework to generate talking faces from text only, without needing audio at inference. Its core idea is to solve the "audio-to-lip ambiguity" (where different sounds have similar audio features) by first converting text into visemes (visual speech units). The model is trained using a "Progressive Viseme-Audio Replacement" (PVAR) strategy, where it learns to "hallucinate" pseudo-audio from visemes, effectively weaning itself off real audio. The final landmarks are passed to a SOTA renderer (EchoMimic) to create the video.

### Strengths
- Novel Training Strategy (PVAR): The method of progressively replacing real audio with viseme-derived pseudo-audio allows to generate only based on text.
- SOTA results: The model presents quantitative visual quality (SSIM, FID) and lip-sync scores (Sync-C) that are comparable or superior to SOTA audio-driven models.

### Weaknesses
- The renderer is too central in metrics: The paper claims SOTA visual quality (SSIM, PSNR, FID) but uses a SOTA renderer (EchoMimic) as its final stage. This is a major confounding variable. These metrics are evaluating EchoMimic's rendering power, not just Text2Lip's landmark generation. The model's actual contribution (lip-sync/landmark quality) is not SOTA (Table 2).

- Central Motivation is Unproven: The entire paper is based on solving "audio-to-lip ambiguity" (e.g., "bad boy" vs. "bat boat") leading to "blurred outputs." This claim is never proven. No experiment shows audio-driven models failing at this while Text2Lip succeeds.

- Ignores Text-Only Limitations: A text-only input has no information about prosody, pace, or emotion. The claims of "micro-expression dynamics" are unsubstantiated, as the model likely generates flat, "average-toned" speech.

- Weak Generalization Claims: The model is benchmarked on small, outdated, closed-domain datasets (GRID, AVDigits). The model has likely memorized the limited vocabulary. The "unseen sentence" example (Fig. 5) appears to use words from the training corpus.

- Questionable Comparisons: The landmark generation (Table 2) is compared against Sign Language Production models, which is not explained.

- Contradictory Justification: The paper fails to mention that visemes also suffer from ambiguity (many different phonemes/sounds map to the same viseme), which undermines its own central premise.

### Questions
Q1 (The Renderer): Why are visual quality metrics (SSIM, FID) better than the EchoMimic baseline itself (Table 1)? Since Text2Lip uses EchoMimic, this result seems contradictory. Does this imply the generated landmarks are "better than real" in some way that improves rendering?

Q2 (The Ambiguity Claim): Why do the authors claim identical lip shapes would lead to "blurred outputs"? A model should just learn to output the identical shape. Where is the evidence that this ambiguity is a real problem that audio-models fail on?

Q3 (The Application): If the model relies only on text, how is synchronicity with a separate audio track (for a final dubbed video) ensured? Does this not require a second, separate model or manual alignment, defeating the purpose?

Q4 (The Ablations): Why do the ablation studies (Table 4) show improvements in image quality metrics (SSIM, FID)? The renderer is pre-trained and fixed. This implies the landmark quality alone is responsible, but this is not measured with a temporal metric like FVD, which would be more appropriate.

### Soundness
2

### Presentation
2

### Contribution
2
