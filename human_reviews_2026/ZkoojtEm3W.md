# Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Deep learning-based respiratory auscultation is currently hindered by two fundamental challenges: (i) inherent information loss, as converting signals into spectrograms discards transient acoustic events and clinical context; (ii) limited data availability, exacerbated by severe class imbalance. To bridge these gaps, we present **_Resp-Agent_**, an autonomous multimodal system orchestrated by a novel Active Adversarial Curriculum Agent (Thinker-A²CA). Unlike static pipelines, Thinker-A²CA serves as a central controller that actively identifies diagnostic weaknesses and schedules targeted synthesis in a closed loop. To address the representation gap, we introduce a modality-weaving Diagnoser that weaves clinical text with audio tokens via strategic global attention and sparse audio anchors, capturing both long-range clinical context and millisecond-level transients. To address the data gap, we design a flow matching Generator that adapts a text-only Large Language Model (LLM) via modality injection, decoupling pathological content from acoustic style to synthesize hard-to-diagnose samples. As a foundation for this work, we introduce **_Resp-229k_**, a benchmark corpus of 229k recordings paired with LLM-distilled clinical narratives. Extensive experiments demonstrate that Resp-Agent consistently outperforms prior approaches across diverse evaluation settings, improving diagnostic robustness under data scarcity and long-tailed class imbalance. Our code and data are available at [https://github.com/zpforlove/Resp-Agent](https://github.com/zpforlove/Resp-Agent).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a multimodal method for high-fidelity respiratory sound generation and disease diagnosis, converting a text-only LLM into an audio generator via modality injection and fusing text with acoustic features using a Longformer with strategic global attention.

### Strengths
The paper’s originality lies in the closed-loop Diagnoser -> Thinker -> Generator setup and anchor-based multimodal Diagnoser, but these build on familiar ideas (active learning, synthetic augmentation, attention).

### Weaknesses
The paper's central claim that the Thinker module intelligently guides data generation lacks evidence, as no controlled comparison against simple baselines like class-prior rebalancing, uncertainty sampling, or random selection is provided under identical experimental settings. Without isolating the planner's contribution from the sheer effect of adding synthetic data, it remains unclear whether the observed gains stem from sophisticated scheduling or merely from data volume. 

Furthermore, the work does not decompose the sources of improvement, whether gains arise from balancing underrepresented labels, targeting hard recording styles, combining rare labels with difficult domains, or simply scaling the dataset. It also does not provide ablation curves showing performance versus synthetic budget to identify when returns saturate.

The evaluation framework also presents gaps that weaken the claims of robustness and controllability. The paper omits comparisons with established imbalance mitigation techniques such as focal loss, which could achieve similar macro-F1 improvements without generative models, rendering the necessity of the proposed pipeline uncertain. Cross-domain evaluation on only two held-out sources is insufficient; a leave-one-source-out protocol would rigorously test whether robustness generalizes across all recording conditions or benefits from a favorable data split. 

Finally, the Generator's ability to independently control pathology content and recording style is assumed but not validated without explicit disentanglement tests (e.g., label-swap experiments), the model may conflate correlated attributes from training data, undermining the loop's premise that rare labels and target styles can be independently specified.

### Questions
Can you provide a controlled ablation comparing the Thinker-based planner against simple baseline schedulers (class-prior rebalancing, uncertainty sampling, random selection) while holding all other factors constant same Generator, same Diagnoser, same synthesis budget, same training procedure to check and demonstrate whether the intelligent planning policy itself drives improvement?

Could you present performance curves showing macro-F1 as a function of synthetic data budget to reveal whether most benefits come from a small number of targeted clips or require large volumes of generated audio?

Could you demonstrate the Generator's controllability through disentanglement tests specifically style-swap experiments (fixed pathology, varied recording style) and label-swap experiments (fixed style, varied pathology) to confirm that rare pathology-style combinations can be reliably synthesized without the model conflating correlated attributes?

Why not compare against actual established baselines in the areas of respiratory monitoring?

### Soundness
1

### Presentation
2

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
The paper present Resp-agent, an agent-based system for respiratory sound generation and disease diagnosis. The whole system includes a cascaded pipeline leveraging LLMs for content semantics learning and followed by the generator and flow-matching based reconstruction to capture the sound-features. Since there's no existing paired EHR - audio datasets, they create a multimodal benchmark names Resp-229k. Finally, the paper evaluated on disease diagnosis tasks and reports improvements over baselines.

### Strengths
1. The paper targets important question that lacking paired multi-modal datasets especially when the text-modality missing.
2. The system design is comprehensive and clear to follow its flow.

### Weaknesses
1. Despite the comprehensive and close-loop system. The novelty lies in the assembly of recent advances such as longformer, flow-matching models. The core contribution and innovation is hidden and ambiguous. 
2. For the generator part,  the Generator does make the system closed-loop: the “Diagnoser” classifies real respiratory sounds, then the “Generator” synthesizes new examples. It's not clear is there of a real feedback loop: the Diagnoser doesn’t meaningfully inform or retrain the Generator; Besides, there's no evaluation of generation quality. 
3. Lack of discussion of recent work about respiratory foundation models or other llm-based respiratory model. For example, papers such as Resp-LLM or other LLM-prompted multimodal models already explored: Using LLMs to describe audio (e.g., “This sound has wheezes and crackles”), althought not in the EHR format. Why the EHR-simulation text is chosen or better compared with other prompt style?

### Questions
see weakness

### Soundness
3

### Presentation
2

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
The paper tackles a clinically meaningful problem with a coherent, end-to-end system and introduces a cross-domain benchmark plus two technically interesting modules (modality-injected unit generator with CFM; Longformer with anchor-based global attention). Futher, the proposed system demonstrates credible ICBHI gains and compelling rebalancing benefits on Resp-229k.

### Strengths
The paper strengths can be summarised as follows,

- Ambitious, unified scope (which includes dataset, generation, diagnosis, and agent loop). The paper introduces a cross-domain multimodal corpus (Resp-229k) with source-disjoint splits. This anchors the contribution in real distribution shift rather than in-domain evaluation.
- Clear architectural ideas on both sides of the loop. The Generator upgrades a compact LLM via modality injection (BEATs-derived style tokens) to autoregress discrete acoustic units, then reconstructs waveforms with a DiT-style conditional flow-matching decoder and vocoder. This results in a clean disentanglement of content (diagnosis) vs style (timbre/device).
- Generation for targeted rebalancing appears useful. Using diagnosis-conditioned synthesis to balance classes lifts macro-F1 sharply.
- The paper reports objective similarity (e.g., FAD, style-cosine) and comparisons to c-WaveGAN / AudioLDM-2 under matched budgets, supporting the claim that content-aware augmentation helps more than generic perturbations.

### Weaknesses
The paper weaknesses can be summarised as follows,

- Low macro-F1 in the natural (imbalanced) setting. Before synthetic balancing, macro-F1 is 0.2118 despite high accuracy... this tell me that there might be substantial minority under-diagnosis. However, the paper relies on its own generator to fix this; stronger baselines (e.g., cost-sensitive losses, reweighting, focal/LDAM, mixup/Manifold mixup, class-balanced sampling) should be compared under the same cross-domain split to show generation is superior beyond conventional imbalance remedies.
- Potential evaluation confounds on generation. The CFM decoder is conditioned not only on discrete units and global timbre but also "a short reference prefix of the ground-truth mel during training/validation" to encourage continuation.
- Dataset governance and text synthesis risks. Clinical "summaries" are LLM-generated from heterogeneous metadata; while the model doesn’t interpret audio, it can still standardize or hallucinate metadata beyond the original fields. The paper should provide audits (error rates, inter-annotator checks) and licenses/provenance per clip, especially with mixed microphone/stethoscope sources used for high-stakes labels.

### Questions
The questions for the authors are as follows,

- Class taxonomy: Are Bronchiectasia vs Bronchiectasis and Acute URI vs URTI duplicates? If so, how are they merged across sources and guaranteed disjoint across splits? Please provide the final class map and counts per class per split. 
- Imbalance baselines: Can you compare against strong non-generative remedies under the same cross-domain protocol (class-balanced sampling, effective number reweighting, LDAM-DRW, focal loss in fine-tuning, mixup/Manifold mixup), with CIs?
- Text summary quality & governance: What safeguards ensure LLM-rendered EHR-style summaries do not hallucinate fields or introduce systematic biases across sources/devices? Provide sampling audits, error types, and licenses/provenance for all sources.
- Where does the gain come from? Please isolate the modality weaving vs late fusion, anchors vs no anchors, and text quality (raw metadata vs LLM-rendered text) in a single ablation grid on the cross-domain split.
- Cross-dataset generalization: In addition to source-disjoint evaluation, can you train on subset A and test on B (e.g., train on ICBHI and SPRSound, test on UK-COVID; or leave-one-source-out) to confirm robustness to label/device/site shifts?

### Soundness
3

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
The paper introduces Resp-Agent, a multimodal agent system tackling the challenges of deep learning-based respiratory assessment by developing both a generator and a classifier. To address the scarcity of high-quality data and the omission of fine acoustic dynamics, the system first transforms a text-only LLM into a multimodal generator via modality injection, producing BEATs tokens conditioned on diagnostic text and style embeddings, which are then reconstructed into waveforms. For diagnosis, it employs a Longformer-based model with strategic global attention to fuse EHR text and acoustic features at the input level, allowing it to capture long-range cross-modal dependencies while accurately detecting brief acoustic events like coughs. This work is supported by the creation of Resp-229k, a massive 408-hour multimodal corpus that pairs audio with expert-level EHR annotations, enabling the system to achieve superior performance in robust respiratory disease prediction.

### Strengths
The most practical and immediate strength is the size and richness of the new corpus. At 408 hours and 229,000 recordings, this scale is required for a specialized medical domain. Crucially, by linking the acoustic data to expert-level annotations distilled from Electronic Health Records (EHRs), this provides the necessary clinical context, which is often missing in public datasets, thus bridging the gap between raw audio and real-world diagnostic complexity. 

The system is capable of generating high-fidelity audio (using LLM-based modality injection to create BEATs tokens). By generating discrete acoustic units (BEATs tokens) and using a specialized flow-matching decoder, it retains the high-fidelity transient events critical for accurate diagnosis. 

The diagnostic model integrates two disparate data types, which are EHR text and respiratory features, at the input level. The introduction of strategic global attention is a good contribution, as it specifically addresses the difficulty of simultaneously capturing broad clinical context (long-range dependencies) and tiny, clinically vital acoustic cues (brief, low-energy events like wheezes or rales).

The paper is reasonably well written and focuses on solving an important issue of accurate diagnosis in the respiratory health domain.  

The datasets and evaluation seem adequate. The performance improvements over compared baselines are good.

### Weaknesses
The papers fall into the category of applying ML for healthcare. I find it interesting as it combines multiple concepts, but incremental. 

The evaluation part is weak. The choice of LSTM for the text baseline is weak, and using an older attention mechanism for fusion is also suboptimal.  I would suggest comparing against a Transformer-based text encoder (e.g., BERT, RoBERTa, or even a small version of the Longformer) for the text-only task. This isolates whether the performance gain is from the fusion or just using better text features. 

To demonstrate the novelty and power of the proposed fusion, the authors should have included multimodal baselines that employ simpler fusion strategies. For example, use the Conformer to extract audio embeddings and the LSTM to extract text embeddings. Concatenate these two vectors and feed the result into a simple Multi-Layer Perceptron (MLP) classifier. The other way can be to do averaging or weighted voting for the final prediction. 

The paper compares audio generative quality with Audio LDM2 and c-WaveGAN. I find StableAudio or StableAudio Open missing from the paper. It will make sense to compare Resp-Agent generative quality against a fine-tuned StableAudio.

### Questions
What will happen if you change Conformer to Whisper?

### Soundness
3

### Presentation
2

### Contribution
2
