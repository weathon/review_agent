# MARS-Sep: Multimodal-Aligned Reinforced Sound Separation

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Universal sound separation faces a fundamental misalignment: models optimized for low-level signal metrics often produce semantically contaminated outputs, failing to suppress perceptually salient interference from acoustically similar sources. We introduce a preference alignment perspective, analogous to aligning LLMs with human intent. To address this, we introduce MARS-Sep, a reinforcement learning framework that reformulates separation as decision making. Instead of simply regressing ground-truth masks, MARS-Sep learns a factorized Beta mask policy that is steered by a preference reward model and optimized by a stable, clipped trust-region surrogate. The reward, derived from a progressively-aligned audio-text-vision encoder, directly incentivizes semantic consistency with query prompts. Extensive experiments on multiple benchmarks demonstrate consistent gains in Text-, Audio-, and Image-Queried separation, with notable improvements in signal metrics and semantic quality. Our code is available at https://github.com/mars-sep/MARS-Sep. Sound separation samples are available at https://mars-sep.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a query-conditioned universal sound separation method that reframes mask prediction as reinforcement learning over a factorized Beta-distributed mask policy. Rewards are multimodal, computed by comparing separated audio to fused audio/text/image query embeddings to directly incentivize semantic faithfulness. A three-stage progressive alignment fine-tunes projection heads of the multimodal encoder to improve cross-modal discrimination and reward stability. 

On VGGSound-clean+ and MUSIC-clean+, MARS-Sep matches or surpasses baselines on SDR/SIR/SAR/SI-SDRi metrics and achieves higher CLAP scores.

### Strengths
- The method is novel. The paper reframes query-conditioned separation as RL with a factorized Beta-policy over masks and optimizes it via a PPO-style clipped surrogate.

- The multimodal reward is well-designed. Rewards are computed by comparing separated audio with fused multi-modal embeddings, directly optimizing semantic faithfulness to the query.

- On VGGSound-clean+ and MUSIC-clean+, MARS-Sep consistently surpasses CLIPSep-NIT and OmniSep for audio, image, and composed queries on SDR/SIR/SAR/CLAP.

### Weaknesses
- The evaluation is limited to only VGGSound/Music, also the qualitative results shown on the website are quite limited samples. As the paper claims for universal sound separation, the paper should demonstrate quantitative&qualitative results on in the wild data.

- Limited discussion of efficiency. The paper details training knobs but lacks hardware/runtime, inference latency, or throughput measurements, especially for longer inputs.

- Improvements lean on CLAP for semantic alignment, but CLAP may carry its own biases and may not fully reflect human perceptual quality. A human user study is needed for fully measure the semantic alignment.

### Questions
- Could the authors include more inference results on in-the-wild data?

- As stated in the weakness part, could the author provide results of human user study to measure the semantic alignment?

- Could the authors provide efficiency reporting, like provide training compute, inference RTF/latency on standard hardware, and memory use for different modalities.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a reinforcement-learning framework for query-conditioned universal sound separation. Instead of regressing deterministic masks, a factorized Beta policy samples time–frequency masks. The training uses a trust-region clipped surrogate with entropy and KL penalties, sampling from a frozen old policy for stability. Multimodal rewards are embedded with a ImageBind encoder, and the target embeddings are fused via low-rank bilinear pooling, and cosine similarity with the separated audio provides the scalar reward. A three-stage progressive alignment procedure fine-tunes only projection heads to make the encoder’s rewards more discriminative before RL. Experiments on VGGSound-clean and MUSIC-clean report consistent gains across different modalities.

### Strengths
1. Using RL for separation is reasonable, as it refines pure mask prediction with a stochastic beta policy process. The old-policy sampling, clipped ratio, and entropy/KL yield a stable loop without a value network.

2. The design of multimodal reward avoids modality dominance. Fusing target-side audio/text/vision to form a single semantic anchor addresses reward imbalance and supports composed queries. The motivation and construction are clearly explained.

3. The three-stage curriculum is a pragmatic way to improve reward faithfulness while keeping encoders largely frozen.

### Weaknesses
1. The ablation study is not enough. The paper motivates several design choices, but some important hyperparameters are not ablated well. For example, the effect of k, \lambda_H, \lambda_KL, clip \theta, etc. A systematic ablation suite isolating each factor would make the gains more convincing. 

2. The method is presented with a U-Net-style STFT separator using mixture phase at reconstruction. Would it work for a time-domain separator as well?

3. Another issue with the paper is the lack of comparison methods. While the authors provide results of CLIPSEP-NIT, AudioSep, and OmniSep in table 1, there are more methods that can be compared. For example, for text query sound separation, LASS-Net[1], FlowSep[2], ZeroSep[3] are not included in the results. Especially FlowSep and ZeroSep that report CLAP score, which is one of the main claim from the paper that the proposed method achieve better semantic matching.

[1] Separate What You Describe: Language-Queried Audio Source Separation
[2] FlowSep: Language-Queried Sound Separation with Rectified Flow Matching
[3] ZeroSep: Separate Anything in Audio with Zero Training
 
Similarly, for image/video query, only CLIPSEP-NIT is compared. In the visually guided separation domain, there are some missing references:
[4] iQuery: Instruments As Queries for Audio-Visual Sound Separation
[5] High-Quality Visually-Guided Sound Separation from Diverse Categories 

I believe these works are worth discussing in the paper and being compared. Particularly, for [5], the paper uses a generative objective to train rather than the traditional metric like SDR/SIR/SAR, which, a comparison, can make the paper's claim more convincing.

### Questions
1. How do MLBP and the fused-anchor strategy compare to (i) simple average of unimodal similarities, (ii) max pooling, (iii) learned weighted sums, especially under composed queries?

2. What is the training time and inference latency vs. OmniSep? Does stochastic masking at test time improve robustness, or do you deploy the mean mask?

3. Have you tried a time-domain separator or a phase-estimation front-end? Do the gains transfer?

4. How does performance/compute scale with the number of target masks K?

### Soundness
2

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
3

### Summary
This paper describes a method of multimodal sound separation.   This work differs from previous work in two ways.  First, it formulates sound separation as an inverse estimation problem and uses RL as an optimization technique to find the best separation masks.  Second, it utilizes a contrastively learned semantic network to assess the similarity between the reconstructed sound and the multimodal inputs. 

The main argument presented in this paper is that sound separation should be treated similarly to the alignment of LLMs with human preferences.   In this way, the RL algorithm can be utilized as a meta-reasoning system to steer a base sound separation architecture (OmniSep).  The reward in this setting should be more semantic, closer to human preference, rather than raw sound waveforms.   A key insight is that the training multimodal separation masks could be used to infer this human preference.  

However, the writing on this overall logic is unclear, instead focusing on the details of implementation.

### Strengths
The overall approach is creative.  This paper addressed a preference/measurement issue in multimodal sound separation: models optimized for signal-level metrics don't produce semantically meaningful results. 

It argues that sound mask should not be used directly as a supervised training signal, but indirectly as a way to train a multimodal sound preference function that is semantically meaningful.  Once we have this preference measure, we can treat the sound separation as an inverse problem, and the RL can be used as an optimization tool.

Diving into the details of RL.   The state space is the sound (mixed) spectrogram, and the query for a separation architecture called OmniSepbase.  The action is the predicted separation masks M.  The semantic similarity between the separated sound waveform and the multimodal query defines the reward function R.  The semantic similarity is fine-tuned from ImageBind, using a progressive refinement procedure on top of contrastive learning.   All these steps are reasonably designed.  The experimental results are good on both signal and semantic level measures.

### Weaknesses
The overall logic of the paper is not well presented.  The authors didn't directly outline the main chain of logic, but instead presented a linear sequence of computational and implementation steps.  The connection to LLMs-to-human preferences alignment is not apparent until the very end of the paper.

### Questions
See above on weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents a Dynamic Semantic Routing Framework (DSRF) for the MSA task. A hierarchical semantic factorization module disentangles each modality into four functionally independent representations: primary emotion, contextual cue, ambiguity, and noise, thereby enabling fine-grained semantic modeling. A semantic dynamic routing interaction mechanism dynamically routes and aggregates semantic factors through a capsule-inspired interaction process to reconstruct modality representations with high-order compositionality. An uncertainty-aware semantic fusion strategy estimates the reliability of each semantic factor and adaptively integrates them across modalities for robust sentiment prediction under modality inconsistency.

### Strengths
The paper presents a hierarchical semantic factorization module, enabling fine-grained semantic modeling. It introduces a semantic dynamic routing interaction mechanism, which dynamically routes and aggregates the semantic factors through a capsule-inspired interaction process to reconstruct modality representations with high-order compositionality. An uncertainty-aware semantic fusion strategy is presented to estimate the reliability of each semantic factor and integrate them across modalities for robust sentiment prediction under modality inconsistency. 
The algorithm proposed in the paper demonstrates innovation, clear logic, and sufficient experimental verification, which effectively confirms the validity of the algorithm.

### Weaknesses
1. The method used for comparing experimental results lacks the latest findings, such as the experimental results from 2025.
Dlf: Disentangled-language focused multimodal sentiment analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 21180–21188, 2025.
2. The method proposed in this paper, an uncertainty-aware semantic fusion strategy, shares some similarities with the method “Uncertainty Score Disentanglement” which is presented in the paper [1]. A comparison between them is recommended. 
[1] Localization-assisted Uncertainty Score Disentanglement Network for Action Quality Assessment, ACM International Conference on Multimedia (ACMMM), pp. 8590–8597, 2023. 
And there remains a reference missing.
[2] Cross-modality Representation Interactive Learning for Multimodal Sentiment Analysis. ACM MM, 2023.

### Questions
1. The method used for comparing experimental results lacks the latest findings, such as the experimental results from 2025.
Dlf: Disentangled-language focused multimodal sentiment analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 21180–21188, 2025.
2. The method proposed in this paper, an uncertainty-aware semantic fusion strategy, shares some similarities with the method “Uncertainty Score Disentanglement” which is presented in the paper [1]. A comparison between them is recommended. 
[1] Localization-assisted Uncertainty Score Disentanglement Network for Action Quality Assessment, ACM International Conference on Multimedia (ACMMM), pp. 8590–8597, 2023. 
And there remains a reference missing.
[2] Cross-modality Representation Interactive Learning for Multimodal Sentiment Analysis. ACM MM, 2023. 

3. In Figure 1, how do elements of (c) and (d) correspond to network blocks in (b)? A detailed explanation is required.
4. In section 4.4, the ablation study is neither clear and not sufficient. A Cross-validation across modules HSF, SFR, DRI can more effectively demonstrate the contribution of the main innovations to the paper's algorithm.

### Soundness
3

### Presentation
2

### Contribution
3
