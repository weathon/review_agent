# VSSFlow: Unifying Video-conditioned Sound and Speech Generation via Joint Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Video-conditioned sound and speech generation, encompassing video-to-sound (V2S) and visual text-to-speech (VisualTTS) tasks, are conventionally addressed as separate tasks, with limited exploration to unify them within a signle framework. Recent attempts to unify V2S and VisualTTS face challenges in handling distinct condition types (e.g., heterogeneous video and transcript conditions) and require complex training stages. Unifying these two tasks remains an open problem. To bridge this gap, we present VSSFlow, which seamlessly integrates both V2S and VisualTTS tasks into a unified flow-matching framework. VSSFlow uses a novel condition aggregation mechanism to handle distinct input signals. We find that cross-attention and self-attention layer exhibit different inductive biases in the process of introducing condition. Therefore, VSSFlow leverages these inductive biases to effectively handle different representations: cross-attention for ambiguous video conditions and self-attention for more deterministic speech transcripts. Furthermore, contrary to the prevailing belief that joint training on the two tasks requires complex training strategies and may degrade performance, we find that VSSFlow benefits from the end-to-end joint learning process for sound and speech generation without extra designs on training stages. Detailed analysis attributes it to the learned general audio prior shared between tasks, which accelerates convergence, enhances conditional generation, and stabilizes the classifier-free guidance process. Extensive experiments demonstrate that VSSFlow surpasses the state-of-the-art domain-specific baselines on both V2S and VisualTTS benchmarks, underscoring the critical potential of unified generative models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces VSSFlow, a unified generative model that integrates the traditionally separate tasks of video-conditioned sound generation (V2S) and visual text-to-speech (VisualTTS) within an end-to-end FLOW-matching framework. It studies the condition aggregation mechanism within the DiT blocks and figures out how to effectively handle distinct input signals by leveraging cross-attention for ambiguous video conditions and self-attention for more deterministic speech transcripts. The authors also demonstrate that the joint learning of sound and speech generation helps faster convergence, better conditional generation, and allows VSSFlow to outperform state-of-the-art domain-specific baselines on both V2S and VisualTTS benchmarks.

### Strengths
- VSSFlow integrates V2S and VisualTTS tasks within a single end-to-end FLOW-matching framework
- VSSFlow outperforms existing domain-specific baselines on both V2S and VisualTTS benchmarks
- The paper does a comprehensive study of various condition aggregation mechanisms

### Weaknesses
- While the overall observation for the joint learning is interesting, all studies of conditioning design are healthy, and the generated samples sound great, the technical contribution is relatively incremental.
- It would be helpful to conduct a subjective evaluation to compare the generated samples against the baselines.
- The performance of V2S generation is affected by the choice of video representations. While the chosen CLIP features yield robust semantic representations for sound quality and semantic alignment, they lack temporality. VSSFlow is slightly weaker in temporal alignment compared to certain baselines like Frieren.

### Questions
- One simple question I have is, could the authors provide insights into how the flow-matching (as opposed to diffusion or autoregressive models) specifically facilitates the learned general audio prior that drives this mutual enhancement?
- Given that other time-aware features (CAVP or Synchformer) are suggested for future work, were any ablation studies performed using these temporal features during the development of the condition aggregation mechanism, or was the focus exclusively on optimizing the integration of CLIP features?
- The authors mentioned that the effectiveness of using synthesized sound-speech mixed data for joint generation is likely inferior to that of native sound-speech joint data. Could you elaborate a little bit more about this? I am curious if it is true that a limitation for future research and development of unified generative models is the scarcity of high-quality video-speech-sound data

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
4

### Summary
This paper proposes VSSFLOW, a unified model for video-to-sound (V2S) and visual text-to-speech (VisualTTS) generation.
The model introduces a conditional architecture: cross-attention over visual features, and concatenation with audio features followed by self-attention over transcripts.
This design enables efficient joint training on mixed V2S and visual TTS datasets without a multi-stage training process.
Experiments show that VSSFLOW outperforms existing domain-specific baselines on both V2S and VisualTTS benchmarks.

### Strengths
- The unification of V2S and VisualTTS is timely and relevant, addressing the need for a general sound generation across modalities.
- The model design is simple yet well-motivated, combining visual and textual conditioning in a natural and interpretable manner.
- The attention pattern analysis provides helpful insight into how multimodal dependencies should be captured within the network.
- The proposed method shows promising results across two distinct domains, demonstrating the potential of shared training.

### Weaknesses
1. **Benchmark inconsistencies**
- The reported FAD scores for Frieren differ substantially from both the original paper and prior works (e.g., MMAudio). It would strengthen the paper if the authors could clarify the cause of these discrepancies, as they directly affect the interpretation of the benchmark results.

2. **Choice of baselines**
 - The exclusion of V2A-M, FoleyC, and MMAudio from comparison is not sufficiently justified. For V2A-M and FoleyC, initializing from pre-trained text-to-sound modules is a fair and efficient approach. Also, for MMAudio, it offers smaller variants (e.g., 157M parameters) comparable to VSSFLOW. These baselines should be fairly included as baselines.
 - Although the paper mentions several unified models (Veo3, AudioGen-Omni, DeepAudio, DualDub), these are not directly compared. A brief quantitative or qualitative comparison would clarify the contribution.

3. **Unclear mutual benefits of joint training**
 - In Fig.7, the conditional and unconditional generation results (cfg=0 vs. cfg=1) for the V2S-only model are nearly identical, suggesting potential training instability. The reviewer encourages the authors to analyze whether the conditional learning mechanism failed in the V2S-only setting.
 - The reported CFG instability (cfg=3) seems inconsistent with prior V2S works (e.g., Frieren). Clarifying whether this is due to data composition or training dynamics would improve the analysis.
 - In Fig.4(b), the joint V2S + VisualTTS setup does not clearly outperform VisualTTS-only. 

4. **Lack of subjective evaluation**
 - The paper reports only objective metrics. While these are standard, adding even a small-scale user study (e.g., a human preference or MOS study) would make the results more interpretable and provide a clearer understanding of the generation quality.

### Questions
- Why do some training curves in Figs. 4, 6, and 7 stop around 10k iterations? Did training collapse or terminate early?
- Have the authors compared VSSFLOW with an independent V2S and VisualTTS generation pipeline? This would help isolate the benefit of unification.
- Could the authors provide direct comparisons with prior unified models?
- Would it be feasible to include even a small-scale user evaluation to complement the objective metrics?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work presents the VSSFlow framework, which unifies three tasks: video-to-sound, text-to-speech, and visual TTS.
For me, the major contributions/highlights are:
(1) multimodal information aggregation strategy, as in sections 3.2 and 4.2;
(2) The synergy effect when training the three tasks together.
(3) the joint speech-sound generation capability with fine-tuning

### Strengths
(1) The presentation is clear; the literature coverage is sufficient.
(2) The reported quantitative results in Table 1 (V2S) and Table 2(VisualTTS) are competitive in most metrics.

### Weaknesses
For the condition aggregation mechanism (Sections 3.2 & 4.2)
(1) The proposed model is trained on limited data (<1M samples). The comparison of the structure designs is not convincing at this data scale. Empirically, I'm expecting the difference among these 4 designs would become marginal with more data available. I understand that more data may not be feasible during the review stage, so I would recommend the authors to use 1/2, 1/4, 1/8 of the original data volume to check if the performance gap among these structures would shrink with higher data volume.
(2) As in Fig 3.b, I wonder if the crossVS would give a similar WER provided sufficient computing, since the green line is still converging?
(3) Although the authors have conducted considerable experiments to compare these 4 structures, the underlying finding doesn't provide enough insight. Personally, I'm not convinced about the explanation simply with "inductive bias", "locality", etc, which doesn't establish a serious and strong logical connection to what is observed in experiments.

For the joint learning between sound and speech:
(1) For the experiments in Fig.4, I would recommend that the authors report the complete curves, rather than the current ones stopping in the middle, aka, finish the 20k step of training. e.g., for Fig.4.b.1, I don't know why the orange curve is terminated much earlier.
(2) Fig.4.b.2 indicates over-fitting. I would request the authors to justify: the alleviation of over-fitting in blue is because of the new task (visual TTS), rather than the additional data introduced by the new task.

For the joint speech-sound generation:
(1) Section 4.5 missed many details in its implementation. If the paper can provide a decent solution to this joint generation problem, it would be a great contribution. However, the details of how the fine-tuning data is simulated and how its quality is ensured are totally missing. Thus, I'm not fully convinced of this claim.
(2) Although the demo is good, I think the authors need a more rigorous justification of their method.

### Questions
(1) Your duration predictor accepts both text and video, and then frozen in training. How is that module pre-trained? I cannot find the details in the main content.
(2) When using concat, how do you sync/ensure the same length for phone and video? Can the poor performance of "concat" be attributed to the poor synchronization of these two information streams?
(3) Usually, TTS doesn't need a CFG coefficient, but sound generation does. You mention CFG in section 3.1, but you don't seriously mention this issue. Especially in the case of section 4.5, how do you use CFG in the joint scenario?
(4) Even the naive TTS with only text input would have WER generally < 10 in most cases, but yours is 18.2%; you also missed this naive baseline result. I suppose you need to show that the visual TTS is better than naive TTS, as more information is injected.

### Soundness
3

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
3

### Summary
The paper introduces VSSFlow, a unified flow-matching model designed to handle both video-to-sound (V2S) and visual text-to-speech (VisualTTS) tasks within a single end-to-end framework. VSSFlow proposes a condition aggregation mechanism that leverages cross-attention for ambiguous video inputs and self-attention (concatenation) for deterministic text (transcript) inputs. The model demonstrates that joint training of sound and speech not only avoids interference but also yields mutual performance gains.

### Strengths
1. VSSFlow elegantly unifies V2S and VisualTTS tasks in a single flow-matching architecture, an open challenge that prior works (AudioGen-Omni, DualDub) only partially address.
2. The study provides clear empirical evidence that joint training across sound and speech modalities yields synergistic improvements—a finding that contrasts with previous assumptions of mutual degradation.

### Weaknesses
1. While the paper motivates Visual TTS as part of a unified audio-visual generation framework, the practical utility of video conditioning in TTS remains unclear. In realistic applications such as dubbing for AI-generated videos, lip features may not correspond accurately to the transcript, potentially introducing misalignment biases. The authors should clarify whether the learned representations rely on accurate lip motion, and how the model behaves when video features are noisy or inconsistent with speech content.
2. The paper provides an interesting analysis of different condition aggregation mechanisms (e.g., four variants), revealing inductive biases between ambiguous (visual) and deterministic (text) inputs. However, it remains unclear whether these observations scale with model capacity and data size. At larger scales, where multimodal representations are more expressive and robust, such design choices may have diminishing impact. It would strengthen the paper to discuss this potential limitation or provide scaling experiments.
3. The use of CLIP features at 10 FPS may be a limiting factor for temporally precise synchronization. The model’s slight underperformance in onset alignment (vs. Frieren) suggests potential weaknesses in temporal modeling.

### Questions
Recent TTS models (e.g., MaskGCT, E2-TTS, F5-TTS) have shown that explicit duration predictors can be removed without degrading alignment quality, leveraging implicit temporal modeling within the generative framework. In VSSFlow, the authors retain a traditional duration predictor for VisualTTS. Could the authors clarify why this design choice remains necessary, and whether their model could benefit from implicit duration modeling consistent with modern end-to-end TTS architectures?

### Soundness
2

### Presentation
3

### Contribution
2
