# A Bridge from Audio to Video: Phoneme-Viseme Alignment Allows Every Face to Speak Multiple Languages

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Speech-driven talking face synthesis (TFS) focuses on generating lifelike facial animations from audio input. Current TFS models perform well in English but unsatisfactorily in non-English languages, producing wrong mouth shapes and rigid facial expressions. The terrible performance is caused by the English-dominated training datasets and the lack of cross-language generalization abilities. Thus, we propose Multilingual Experts (MuEx), a novel framework featuring a Phoneme-Guided Mixture-of-Experts (PG-MoE) architecture that employs phonemes and visemes as universal intermediaries to bridge audio and video modalities, achieving lifelike multilingual TFS. To alleviate the influence of linguistic differences and dataset bias, we extract audio and video features as phonemes and visemes respectively, which are the basic units of speech sounds and mouth movements. To address audiovisual synchronization issues, we introduce the Phoneme-Viseme Alignment Mechanism (PV-Align), which establishes robust cross-modal correspondences between phonemes and visemes. In addition, we build a Multilingual Talking Face Benchmark (MTFB) comprising 12 diverse languages with 95.04 hours of high-quality videos for training and evaluating multilingual TFS performance. Extensive experiments demonstrate that MuEx achieves superior performance across all languages in MTFB and exhibits effective zero-shot generalization to unseen languages without additional training.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes MuEx that uses phonemes and visemes as intermediaries to improve multilingual talking head generation performance. The paper proposes MTFB dataset that comprises talking head data in diverse languages.

### Strengths
Talking head generation task is of great practical importance.

### Weaknesses
1. Utilizing phoneme for talking head generation lacks novelty. The paper lack discussion of phoneme-based talking head methods, including Write-A-speaker, etc.
2. The performance shown in demo is not satisfactory. The results are unnatural. The lips are jittering. The demo lacks qualitative comparisons with SOTA methods and ablation results. 
3. The paper lacks a user study.
4. The paper lacks comparisons with recent methods, such as EDTalk.
5. Many methods have already achieve good performance for non-English audio, such as EMO, VASA. These methods do not devise new architecture but simply incorporating data in diverse language for training. Therefore, the reviewer doubts whether good multilingual performance can be achieved solely through data, without using the model design (or architecture) proposed in the paper.

### Questions
Can the authors provide user study results?

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
2

### Summary
MuEx proposes a multilingual speech-driven talking face framework that learns language-agnostic phoneme and viseme prototypes. A mutual-information-based alignment enforces robust cross-modal correspondence between audio and facial motion. A pseudo-phoneme guided MoE router then dynamically selects experts without relying on language labels, improving adaptability to diverse speech patterns.

### Strengths
- MuEx aligns phoneme and viseme prototypes using mutual information and routes pseudo-phonemes through experts to improve multilingual talking face generation. 
- Using phoneme–viseme prototypes provides universal articulatory anchors that support robust multilingual modeling. 
- The new 12-language MTFB dataset enables comprehensive multilingual evaluation and advances TFS research.

### Weaknesses
- PV-Align builds on the idea of MI-based audio-visual alignment, but additional clarification is needed to highlight how the prototype-level design differs from prior approaches.
- PG-MoE appears as a standard MoE router guided by discrete speech units, similar to VQ-based lip tokenization (e.g., VQTalker [1]).
- Zero-shot performance might partially come from shared phoneme sets between training and "unseen" languages. It would be helpful to clarify the extent to which the observed zero-shot gains depend on shared phoneme inventories between the training languages and the unseen evaluation languages.
- The experimental comparison remains somewhat incomplete, since recent diffusion-based talking face systems such as DiffTalk [2] and Fantasytalking [3] are not included.

[1] VQTalker: Towards Multilingual Talking Avatars through Facial Motion Tokenization

[2] DiffTalk: Crafting Diffusion Models for Generalized Audio-Driven Portraits Animation

[3] Fantasytalking: Realistic talking portrait generation via coherent motion synthesis

### Questions
- Pseudo-phoneme estimation relies on distance to prototypes — how do these prototypes emerge language-agnostically?
- MI alignment at the prototype level seems heuristic. Why J-S MI, and why impose negative MI on raw features?

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
5

### Summary
This paper presents a new framework for speech-driven talking face synthesis (TFS) with strong multilingual generalization. Authors point out  that phonemes (speech units) and visemes (mouth-shape units) act as universal intermediaries between audio and visual modalities.

### Strengths
State-of-the-art (SOTA) results on both seen and unseen languages, over strong baselines like Hallo2 and SadTalker.

Authors combine discrete prototype alignment with sparse MoE routing, it is interesting, forming a coherent cross-lingual generative framework.

### Weaknesses
The idea, employing phonemes and visemes as universal intermediaries to bridge audio and video modalities, has been pointed out long before[1]. 

[1] Multimodal inputs driven talking face generation with spatial–temporal dependency; Lingyun Yu, Jun Yu, Mengyan Li, Qiang Ling

The novelty lies mainly in cross-lingual task, not in architectural innovation.

How to ensure temporal dependency? Specifically, even for the same phoneme, the mouth movements can vary depending on the surrounding phonetic context.

### Questions
please refer to weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper tackles the poor cross-lingual performance of existing speech-driven talking face synthesis (TFS) models by proposing **MuEx**, a framework using phonemes and visemes as universal intermediaries. It integrates two key components: PV-Align (phoneme-viseme alignment via mutual information) for language-agnostic cross-modal correspondence, and PG-MoE (phoneme-guided mixture-of-experts) for adaptive multilingual processing. The authors also introduce MTFB, a 95.04-hour benchmark with 12 diverse languages. Experiments show MuEx outperforms SOTA methods on key metrics and achieves effective zero-shot generalization to unseen languages, establishing a new multilingual TFS paradigm.

### Strengths
1. It accurately addresses the core pain points of existing TFS models—"English-dominated and poor cross-lingual generalization"—and proposes a cross-modal bridging approach using phonemes (basic speech units) and visemes (basic mouth shape units) as universal intermediaries, with clear innovations that align with practical needs.
2. Leveraging prior knowledge of speech and vision has become increasingly rare in current end-to-end systems, and such interpretable methods deserve encouragement.
3. Evaluated on test sets of multiple languages, this method demonstrates robustness to multilingual inputs—a feature rarely seen in previous approaches.

### Weaknesses
1. In the attached demo videos, only results from the MuEx are presented, lacking comparisons with other methods. Although quantitative metrics demonstrate MuEx’s superior performance, qualitative video results are crucial for TFS evaluation.
2. A notable omission is the lack of discussion on generalization to unseen image/person styles. Demo videos and figures only showcase test cases that fall within the training data domain, failing to demonstrate out-domain adaptability.

### Questions
Including visual comparative results against other state-of-the-art methods would further strengthen the credibility of the paper’s contributions.

### Soundness
2

### Presentation
2

### Contribution
3
