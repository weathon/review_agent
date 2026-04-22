# Lookahead Anchoring: Preserving Character Identity in Audio-Driven Human Animation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Audio-driven human animation models often suffer from identity drift during temporal autoregressive generation, where characters gradually lose their identity over time. One solution is to generate keyframes as intermediate temporal anchors that prevent degradation, but this requires an additional keyframe generation stage and can restrict natural motion dynamics. To address this, we propose Lookahead Anchoring, which leverages keyframes from future timesteps ahead of the current generation window, rather than within it. This transforms keyframes from fixed boundaries into directional beacons: the model continuously pursues these future anchors while responding to immediate audio cues, maintaining consistent identity through persistent guidance. This also enables self-keyframing, where the reference image serves as the lookahead target, eliminating the need for keyframe generation entirely.  We find that the temporal lookahead distance naturally controls the balance between expressivity and consistency: larger distances allow for greater motion freedom, while smaller ones strengthen identity adherence. When applied to three recent human animation models, Lookahead Anchoring achieves superior lip synchronization, identity preservation, and visual quality, demonstrating improved temporal conditioning across several different architectures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper finds that the DiT model is limited by the quadratic complexity of the Transformer, allowing it to process only short segments of approximately 5 seconds at a time. To generate longer videos, a segment-based autoregressive generation approach is adopted, but this is prone to character identity drift. New segments rely on previously generated frames, and error accumulation causes the character's appearance to gradually deviate from the original reference image. To address this, the role of keyframes is modified in this paper: keyframes are shifted from being "boundary constraints for the currently generated segment" to "directional guidance for future time steps." While responding to real-time audio signals, the model continuously tracks future keyframes, achieving a balance between identity consistency and motion naturalness.

### Strengths
1.	The thinking of sync-free keyframes is reasonable. And the “Do video DiTs understand distant frames” shows how reference frames influence video clip generation clearly.
2.	The long auto-regressive generation results are generally satisfactory, and the motion intensity and identity-preserving abilities are well balanced.
3.	The code and weights will be open sourced, which benefits the reproducing abilities. And it is convincing that several baselines are tested with proposed method.
4.	The writing is clear and easy to understand.

### Weaknesses
1.	The proposed distant keyframe conditioning method is not novel enough. Similar methods have been proposed in Section 3.3 of OmniHuman-1.5[1], which are not cited in this paper. As a core part of this work, the originality should be very clear, otherwise it will damage the novelty of this paper.
2.	In the provided demos, the head motions seem to be restricted around a limited area, compared to the baseline methods. I am wondering why this happens. And will the proposed lookahead anchoring restrict the expressiveness ability of motion generation of DiT models, especially in half-body or full-body generation settings?


[1] Jiang J, Zeng W, Zheng Z, et al. Omnihuman-1.5: Instilling an active mind in avatars via cognitive simulation[J]. arXiv preprint arXiv:2508.19209, 2025.

### Questions
1.	Discuss the novelty of this paper, compared to OmniHuman-1.5.
2.	Explain the restricted motion shown in demos.
3.	Discuss the proposed method whether could be used to full-body animation setting and the impacts of method.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Lookahead Anchoring approach to address the problem of identity drift in long-form, audio-driven human animation. 
Instead of forcing the model to meet specific keyframes at segment boundaries, keyframes are placed at a future time step (ahead of the current generation window), pushing the model to "chase" them. Such a design enables the preservation of character identity while allowing for more expressive motion dynamics. Experiments show that by using the Lookahead Anchoring strategy, the model can  maintain consistent identity over time while generating plausible audio-driven human animation.

### Strengths
* The method is demonstrated to generalize across multiple DiT-based human animation models including Hallo3, OmniAvatar, HunyuanAvatar (Sec. 4.1), which showcases its broad applicability and the potential for integration into other architectures.

* The paper presents both quantitative and qualitative results showing the superiority of the proposed approach. In experiments with long video generation, Lookahead Anchoring outperforms traditional methods in terms of character consistency and overall video quality.

### Weaknesses
* The concept of Lookahead Anchoring is not a new thing. Similar ideas have been explored in prior works like Omnihuman-1.5 (released on arXiv one month before the ICLR paper deadline), which also introduces a  Pseudo Last Frame design to anchor the given reference frame  at future timesteps ahead of the current generation window. Unfortunately, the paper does not cite or discuss these existing methods. 

* The results in the supplemental video (02:56-04:35) suggest that the Lookahead Anchoring strategy limits the motion range of the character. This restriction may hinder the model's ability to generate highly dynamic and expressive animations, which could be a significant drawback for certain use cases requiring more fluid motion.

* Even if the Lookahead Anchoring strategy is completely novel, it is relatively simple and may not be novel enough to support a paper at ICLR, which typically expects more advanced and intricate contributions. The simplicity of the method, though effective for certain scenarios, may not meet the high standards of innovation and complexity expected at this level of the conference.

* While the paper does a good job focusing on identity preservation and lip synchronization for simple scenarios, more experiments on scene dynamics, such as handling large environmental changes (e.g, view changes or moving background) are necessary. Current discussion on these cases is relatively brief.

### Questions
See [Weaknesses]

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Audio-driven human animation suffers from identity drift in long videos. Existing fixes either need extra keyframe models or restrict natural motion, failing to balance identity consistency and motion freedom.

This paper proposes Lookahead Anchoring: using future-timestamp keyframes as "guides" instead of current-window ones. It uses the reference image directly as the future anchor and adjusts lookahead distance to balance identity and motion.

Tests on three DiT-based models and datasets show it boosts identity consistency, maintains lip synchronization, and improves video quality. It also supports narrative-driven generation, serving as a practical solution for long audio-driven animations.

### Strengths
1. This paper proposes a new keyframe logic, which differs from traditional methods like KeyFace that rely on rigid boundary constraints or other reference-net-based designs. It converts keyframes into future-oriented guides, named self-keyframing, aiming to maintain character identity and address error accumulation.
2. The approach designs temporal distance as a controllable parameter: smaller D values prioritize identity adherence, larger D values focus on motion expressivity.
3. The method is integrated with three DiT-based audio-driven models (Hallo3, HunyuanVideo-Avatar, OmniAvatar) through a fine-tuning strategy. This integration is meant to show that the method can be applied to multiple architectures, not just a custom model framework.
4. The work explores narrative-driven long video generation by combining text-based image editing models to create story-specific keyframes, thereby enhancing the solution’s extensibility to meet varied scenario-based requirements.

### Weaknesses
1. The method mentioned in Section 3.3 of the OmniHuman1.5[1] is almost identical to this work, so I have some doubts about the innovativeness—nevertheless, this work features more detailed experiments compared to that paper.
2. The method's visualizations do demonstrate its capability in generating long-duration videos, yet it lacks performance in high-dynamic scenarios: character dynamics remain relatively monotonous, with limited upper-body and hand movements.
3. It would be good to visualize the ablation study for the distant keyframe conditioning.

[1] Jianwen Jiang, Weihong Zeng, Zerong Zheng and et.al. OmniHuman-1.5: Instilling an Active Mind in Avatars via Cognitive Simulation

### Questions
1. The first part of the qualitative comparison section in the supplementary materials (featuring a woman with wavy curly hair in a blue outfit), the dynamic effect of the result with lookahead anchoring is weaker than that of the baseline without it. This raises the question of whether this method might reduce dynamic performance？
2. The paper mentions that the strategy of directly using anchors without training will produce artifacts. If you directly discard the final latent with artifacts in longer video generation tasks, can this serve as a training-free method?
3. Three different baselines are used in this paper, which all perform well in fixed scenarios. However, have you tried using models with camera movement capabilities to verify the effectiveness of this method? And could the anchor frame possibly restrict the range of camera movement?

### Soundness
4

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
4

### Summary
This paper proposes Lookahead Anchoring (LA), a method designed to preserve character identity in audio-driven human animation. Existing methods rely on keyframe generation producing intermediate frames  and identify specific feature inject to prevent identity drift. But these explicit keyframes can overly constrain the motion dynamics, limiting natural expressivity.

To overcome this, LA introduces future keyframe conditioning: rather than generating anchors within the current sequence, the model leverages future latent keyframes as soft temporal guidance. 

Key observations include:
	•	The temporal lookahead distance directly balances expressivity and consistency; larger distances produce more dynamic motion, smaller ones yield stronger identity adherence.
	•	LA integrates seamlessly into existing DiT-based architectures and improves both identity stability and lip synchronization across long sequences.

### Strengths
1. Conceptual innovation: The idea of using future latent frames as temporal anchors instead of rigidly generated keyframes is elegant and conceptually clear. It shifts the paradigm from hard constraints to soft temporal guidance.
2. Interpretability: The empirical finding that lookahead distance controls a trade-off between motion expressivity and identity consistency is intuitive and well-supported (Fig. 6).
3. Model-agnostic integration: LA can be attached to existing transformer or diffusion-based animation models with minimal architectural change.
4. Quantitative gains: Across HDTF and AVSpeech datasets, LA consistently improves lip synchronization (Sync-D ↓, Sync-C ↑), face/subject consistency, and perceptual quality (FID ↓, FVD ↓), without harming motion smoothness.
5. Perceptual preference: User studies show strong preference for LA-enhanced videos in terms of synchronization and identity stability.
6. Practical significance: The approach removes the need for an explicit keyframe generation stage, simplifying pipelines for identity-preserving video generation.

### Weaknesses
* Missing justification for “bounded generation” argument
The introduction claims that “bounded” keyframe-based methods are limited by the quality and expressiveness of their generated keyframes. While this is plausible, the paper does not provide quantitative or visual evidence demonstrating this limitation.

* Ambiguity in “self-keyframing” explanation
The statement that “the keyframe no longer needs to match the exact lip movements and expressions required by the audio … enabling self-keyframing” is conceptually interesting but under-explained. It’s unclear how a distant or reference-based anchor can substitute for synchronised keyframes in guiding expression or pose accuracy.

* Lack of comparison with KeyFace (Bigata et al., 2025)
KeyFace is currently a strong state-of-the-art method for identity-preserving talking heads, explicitly designed to address identity degradation. Its absence from the qualitative comparisons leaves a significant gap. A side-by-side video comparison would substantially strengthen the evaluation.

*  Limited clarity on latent-space interpretation
As I understand it, each latent token represents a spatiotemporally compressed patch, not a full frame. Appending the lookahead latent z_{n-1+d} therefore adds only one additional patch-level token, not a holistic future-frame reference. It is unclear how this single patch provides global temporal guidance or identity stabilization across the entire sequence.

* Inference-time mechanism under-specified
During inference, when only one keyframe or reference image is available, it remains unclear how the lookahead mechanism functions in practice. Does the model still benefit from a meaningful anchor signal? If not, this could contradict the paper’s claim that LA allows “sync-free keyframes” capable of matching poses and expressions.

### Questions
1.	Could you clarify what “bounded generation” refers to in practice, and show evidence that conventional keyframe methods limit expressiveness or quality?
2.	How exactly does “self-keyframing” function; does the model reuse its own generated frames, or the original reference frame, as recursive anchors?
3.	Why was KeyFace excluded from qualitative comparisons, given its strong relevance to identity preservation?
4.	Given that each latent token represents a spatiotemporal patch, how can a single appended future latent provide meaningful global identity anchoring?
5.	During inference with only one keyframe/reference image, how is the lookahead mechanism applied, and does it still contribute to identity consistency?

### Soundness
4

### Presentation
4

### Contribution
3
