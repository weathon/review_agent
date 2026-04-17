# Behaviour-Aware Multimodal Video Summarization: Cross-Modal Integration for Human-Centric Content Analysis

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Video summarization remains a challenging task in capturing the complex interplay of visual dynamics, spoken content, and behavioural cues that collectively shape viewer understanding in human-centric videos. Human communication is inherently multimodal; however, existing approaches in video summarization either rely solely on visual features or rudimentary text-visual combinations, neglecting critical audio prosodic patterns and their interactions. Crucially, the synchronous behavioural signals that convey emotional expression and communicative intent are not considered entirely. In this paper, we present a behaviour-aware multimodal framework for video summarization that explicitly models synchronized behavioural cues across visual, audio, and textual modalities through a transformer-based architecture with cross-modal attention mechanisms. Our approach integrates CLIP visual embeddings enhanced with facial movement detection and emotional transitions, HuBERT audio features enriched with prosodic patterns including pitch variations and voice quality measures, and RoBERTa textual embeddings that preserve narrative flow and discourse structure. We employ heuristic-based behavioural cue detection methods combined with large language model-guided extractive summarization to generate pseudo-ground truth references that capture both semantic importance and behavioural salience. Extensive evaluations on the ChaLearn First Impressions dataset demonstrate substantial improvements over state-of-the-art methods, achieving a 33.2% increase in F1-score over CLIP-It and 7.3% over recent multimodal approaches. Comprehensive ablation studies confirm the effectiveness of behavioural cue integration, with each modality contributing complementary insights for capturing communicatively significant moments in interview-style videos.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a transformer-based multimodal video summarization framework that explicitly models synchronized behavioural cues (facial movements, prosodic patterns, gestures) across visual, audio, and textual modalities for human-centric interview videos. The approach integrates CLIP visual embeddings enhanced with facial movement detection, HuBERT audio features with prosodic patterns, and RoBERTa textual embeddings through cross-modal attention mechanisms. An autoregressive decoder generates temporally coherent summaries, while a two-stage pseudo-ground truth generation method combines heuristic-based behavioural cue detection with LLM-guided extractive summarization. Evaluated on the ChaLearn First Impressions dataset, the framework achieves a 33.2% F1-score improvement over CLIP-It and 7.3% over recent multimodal approaches.

### Strengths
- The explicit modeling of synchronized behavioural signals (facial movements, emotional transitions, prosodic patterns) across modalities represents a meaningful departure from generic multimodal fusion approaches. The concatenation of detected behavioural features with foundation model embeddings (CLIP, HuBERT) is straightforward yet effective.

- The three-stream processing pipeline with modality-specific encoders (video, audio, text) and cross-modal attention mechanism provides a principled approach to feature integration. The autoregressive decoding strategy addresses temporal coherence better than frame-level classification approaches.

- The two-stage approach combining heuristic behavioural cue detection with LLM-guided summarization addresses the lack of annotated data pragmatically. The validation showing 93% coverage and 0.83 Jaccard similarity across independent generations demonstrates reasonable stability.

### Weaknesses
- Evaluation exclusively on ChaLearn First Impressions (single-speaker, controlled interview setting) raises significant concerns about generalizability. The framework's applicability to multi-speaker scenarios, conversational videos, educational lectures, or unstructured content remains unvalidated. The authors acknowledge this limitation but do not provide evidence of cross-domain robustness.

- Heuristic-based behavioural detection lacks justification: The choice of thresholds for behavioural cue detection (μ + 1.5σ for head movement, Z-score ±1.2 for pitch, top 20% TF-IDF) appears arbitrary without principled justification or sensitivity analysis. The adaptive thresholds may not generalize across different video characteristics, speaker styles, or recording conditions.

- Using LLM-generated summaries as evaluation references introduces potential circularity, as the framework is optimized against these same LLM-derived targets. While the authors demonstrate inter-summary agreement, the fundamental question of alignment with human preferences remains unaddressed. The 93% coverage metric does not validate whether detected cues correspond to human-perceived importance.

- Despite claims of temporal coherence, Kendall's τ (0.6473) and Spearman's ρ (0.6466) remain moderate, suggesting the framework still struggles to fully preserve narrative structure. The paper does not sufficiently analyze why these correlations plateau or what architectural modifications might improve them.

- While cross-modal attention is central to the approach, the paper provides limited insight into what the attention mechanism learns. There are no attention weight visualizations, analysis of modality dominance patterns, or investigation of how attention distributions vary across different video types or behavioural contexts.

### Questions
- How does the framework perform on fundamentally different video types such as multi-speaker panel discussions, casual vlogs, classroom lectures, or documentary-style content? What specific architectural modifications would be required to handle multiple speakers with overlapping speech and dynamic speaker transitions?

- How sensitive is performance to the heuristic threshold choices for behavioural cue detection? Have you conducted systematic sensitivity analysis varying these parameters (e.g., σ multipliers from 1.0 to 2.0, Z-score thresholds from ±1.0 to ±1.5)? Could these thresholds be learned rather than hand-specified?

- Given the reliance on LLM-generated pseudo-ground truth, how well do your summaries align with human-annotated preferences? Have you considered conducting human evaluation studies with metrics like informativeness, coherence, and behavioural salience ratings? What percentage of frames deemed important by your model would humans also rate as significant?

- What patterns does the cross-modal attention mechanism learn? Which modalities dominate attention in different contexts (e.g., high prosodic emphasis vs. strong facial expressions)? Can you provide visualizations showing how attention weights distribute across visual, audio, and textual features during critical moments?

- How does the framework scale to videos exceeding the 15-second average of ChaLearn (e.g., 30-minute interviews, hour-long lectures)? Do the transformer encoders face memory constraints with longer sequences? Would hierarchical processing or sliding window approaches be necessary?

- Beyond the 9% temporal misalignment cases mentioned, what are the systematic failure patterns? When do behavioural cues mislead the model (e.g., nervous gestures vs. emphatic gestures)? How does performance degrade with noisy audio, poor lighting, or accented sp

### Soundness
2

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
3

### Summary
This paper proposes a behaviour-aware multimodal video summarization framework that fuses visual, audio, and textual cues through cross-modal attention. It detects synchronized behavioural signals and uses LLM-guided pseudo-labels for supervision, achieving strong results on the ChaLearn First Impressions dataset.

### Strengths
- The paper introduces the notion of behavioural awareness in multimodal summarization, emphasizing behavioural cues such as facial expressions, head movements, and prosodic variations.
- From data preprocessing to feature extraction, cross-modal fusion, and decoder-based summary generation, the pipeline is well-structured and detailed.
- The use of GPT-4.5 for behaviour-guided pseudo ground truth generation is a clever and practical way to overcome the lack of human annotations.

### Weaknesses
- The framework largely builds upon standard Transformer and cross-modal attention modules, similar to UMT [1] and CF-Sum [2]. The main novelty lies in the inclusion of behavioural cues, which, while meaningful, rely on relatively conventional detectors (e.g., MediaPipe, DeepFace, YAAPT). The authors could better emphasize how their fusion or decoding differs from previous approaches.
- While LLM-generated summaries are reasonable, the paper does not include human evaluation or consistency checks to verify alignment between pseudo and human-labeled summaries. A small-scale human validation study would strengthen the reliability of the results.
- The dataset used (ChaLearn First Impressions) contains single-speaker, interview-style videos. Evaluating the model on multi-speaker or in-the-wild video datasets would support the claim of “behaviour awareness” more convincingly.
- The comparative experimental evaluation is rather limited. The results presented in Table 1 and Table 2 only cover a small set of baselines and do not include recent advances in video summarization.

[1] CFSum: A Transformer-Based Multi-Modal Video Summarization Framework With Coarse-Fine Fusion

[2] UMT: Unified Multi-modal Transformers for Joint Video Moment Retrieval and Highlight Detection

### Questions
- Could the authors include more comparative experiments in Tables 1 and 2, especially against more recent video summarization methods, to better demonstrate the effectiveness of their approach?
- Given that CLIP, HuBERT, and RoBERTa are large backbone models, could the authors discuss the training cost and computational efficiency? Would reporting FLOPs, GPU hours, or parameter counts help enhance the transparency and reproducibility of the work?

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
This paper proposes a behavior-aware multimodal video summarization framework tailored for human-centric interview videos. It addresses the limitations of existing methods by explicitly modeling synchronized behavioral cues across visual, audio, and textual modalities via a transformer-based architecture with cross-modal attention. To tackle the lack of human-annotated data for the ChaLearn dataset, it introduces a two-stage pseudo-ground truth generation method.

### Strengths
1. This work introduces a behavior-aware multimodal for video summarization, leveraging  behavioral cues extracted from visual, audio, and textual modalities
2. The authors propose LLMs based workflow for pseudo-ground generation integrating human behavioural cues.
3. The proposed approach shows significant improvements over SOTA methods on ChaLearn dataset,  especially on interview-specific summaries.

### Weaknesses
1. Lack of Technical Innovation: All core modules are mature technologies, such as CLIP, HuBERT, Transformer cross-attention. No modified modules or theoretical frameworks are proposed. The contribution focuses on system integration rather than academic methodological innovation.
2. Unclear Advantage Over Pseudo-Ground Truth Workflow.  The pseudo-ground truth generation pipeline already has summarization capabilities and the task description is general . However, the paper fails to clearly illustrate the advantages of the proposed Transformer-based framework, such as inference speed or convenience of use. This makes it impossible to demonstrate the necessity of the increased complexity of the framework.
3. Limited Generalization Due to Single-Dataset Evaluation: This work is only evaluated on the ChaLearn dataset, a controlled, single-speaker interview dataset with standardized settings, and the result is insufficient to verify the framework’s generalizability

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a multimodal video summarization framework that integrates visual (CLIP + facial movements/emotions), audio (HuBERT + prosodic features), and textual (RoBERTa) modalities through cross-modal attention. Applied to interview videos from ChaLearn dataset, the approach uses  LLM-based pseudo-ground truth generation for training and evaluation.  The authors claim that adding  behavioral features (capturing facial expressions, gestures, and vocal prosody) to guide the summarization process is a major contribution.  Results show F1=0.81, claiming 33% improvement over CLIP-It and 7% over recent multimodal baselines.

### Strengths
* Focus on behavioral cues (facial movements, prosody, emotional transitions) is relatively underexplored  for video summarization
* Two-stage approach using LLM-based pseudo-ground truth generation is novel for video summarization 
* Solid engineering work, well written paper, interesting results and ablation studies, thorough appendix

### Weaknesses
* Flawed evaluation methodology: 1. why use the ChaLearn dataset with no human summarization annotations? (the authors discuss this a bit in the main paper and a bit in the appendix but it remains a flaw), 2. using LLM-generated pseudo labels AND ground truth is circular
* Core claim not validated: behavioral features contribution is not isolated in the ablations, e.g., HuBERT alone vs HuBERT+prosody - unclear if "CLIP Only" is without facial/emotion features
* No human evaluation study comparing summary quality in lieu of using standard benchmarks with human annotations
* Limited experimental scope - ChaLearn is controlled, single-speaker, ~15-second interviews (the authors admit this in the appendix and mention future work on more complex datasets - still a weakness)
* The work is basically LLMs + pseudo-labels + some additional behavioral/emotional features, so little novelty otherwise, e.g., fusion is pretty generic

### Questions
1. Why not evaluate on SumMe/TVSum where human annotations exist? The dismissal of these as "action-oriented" seems insufficient. Why only compare with more than two approaches in the literature, there is plenty of work in this area on other datasets.
2. Behavioral Feature Isolation: Can you provide ablations showing: Raw CLIP embeddings vs CLIP + facial movements + emotions? Raw HuBERT embeddings vs HuBERT + prosodic features? 
3. Human Validation: Can you provide human evaluation comparing your summaries to baselines?
4. Table 4 shows different LLMs produce different references (F1 ranges 0.68-0.81). How do you justify using GPT-4.5 as "ground truth"? The problem is that bigger/more recent models also have stronger biases for subjective tasks - so you might be seeing a circular logic here - see for example https://arxiv.org/abs/2403.17125
5. How is your work is different from Evangelopoulos et al? How is your work different from pseudo-labeling approaches in the literature (for other but similar tasks)?

### Soundness
3

### Presentation
3

### Contribution
2
