# MoGIC: Boosting Motion Generation via Intention Understanding and Visual Context

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Existing text-driven motion generation methods primarily focus on bidirectional mapping between language and motion, yet they often struggle to capture the high-level semantic structures and future behavior patterns that govern how actions unfold. Moreover, the absence of visual conditioning limits synthesis accuracy, as language alone cannot specify fine-grained spatiotemporal trajectories or environmental context. We present MoGIC, a unified multimodal framework that jointly models future-aware behavior understanding and multimodal-conditioned motion generation. MoGIC formulates future-behavior predicting as inferring high-level future semantic patterns from partial observations, while leveraging visual priors to resolve ambiguities inherent in text-only conditioning. We further introduce a mixture-of-attention mechanism with adaptive scope to facilitate effective interactions between multimodal tokens and temporal motion segments, thereby mitigating the impact of non-strict timing alignment. To support this paradigm, we curate Mo440H, a 440-hour tri-modal benchmark aggregated from 21 high-quality motion datasets. Extensive experiments demonstrate substantial improvements in generation fidelity and multimodal versatility of MoGIC: (1) a 36\% reduction in FID on HumanML3D and Mo440H; (2) superior captioning performance compared to LLM-based methods using only a lightweight text head; (3) capabilities in future-aware behavior prediction and vision-conditioned motion synthesis. Together, these results advance the state of the art in motion understanding and multi-conditioned generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
MoGIC proposes a multimodal framework for human motion generation that fuses text, (optionally) vision, and partial motion via a Conditional Masked Transformer (CMT). The CMT applies (i) global semantic-level modulation (adaptive LayerNorm) and (ii) a mixture-of-attention with adaptive Top-k cross-attention to select relevant text/vision snippets. A diffusion-style Motion Generation Head (MGH) produces motions; a lightweight T5-style Intention Prediction Head (IPH) generates an explicit textual “intention.”     

They train across five tasks (L2M, VL2M, V2M, M2M, and intention prediction) with a unified loss, truncating the last 50% of motion tokens for IPH.  The integrated dataset “Mo440H” (≈440 hours, 21 sources) is constructed; for datasets without captions they use Qwen2.5-VL to auto-caption, and for datasets lacking RGB they render mesh sequences to videos, downsampled to 1 fps. They also evaluate on HumanML3D, using prior evaluators there and a retrained evaluator for Mo440H.

### Strengths
* A coherent architecture that disentangles continuous motion generation (diffusion/interpolant) from discrete intention text, while injecting multimodal context at both semantic and token levels. The block-level design is sensible and easy to slot into existing masked-token pipelines.  
* Cross-modal joint training across L2M / V2M / M2M / IP is a clean way to amortize supervision and can plausibly improve controllability/faithfulness.

### Weaknesses
1. The dataset construction lacks details. Auto-captioning with Qwen2.5-VL and manual filtering is fine, but release the prompts, filtering criteria, and failure cases. Likewise, clearly mark which entries use rendered meshes vs real RGB, and how those are used in each task. Different datasets used different motion representation. What representation does Mo440H use? How did the authors convert different joint representation?

2. For sources without RGB, the “vision” is rendered from the motion itself (“rendered mesh sequences were adopted instead,” 1 fps). If these rendered frames are used in training/evaluation for V2M or VL2M, the model is being conditioned on images deterministically derived from the target motion—i.e., target leakage across modalities. This can inflate the apparent benefit of visual conditioning and overstate real-world applicability (where visuals are independent RGB videos). The paper should explicitly exclude such cases from V2M/VL2M evaluation or report separate results using only genuine RGB. 


3. “Intention” is underspecified and may collapse to partial-captioning.
The task defines intention as text generated from the first 50% of a motion and calls it a “conceptual goal.” In practice, this looks like captioning of observed partial motion rather than predicting latent goals (which would require causal/goal semantics, not just n-gram overlap). The metrics shown (BLEU/ROUGE/BERTScore) mostly measure textual similarity, not intention correctness. 

4. The authors did not specify the train/eval split on the new dataset. They retrain an evaluator for Mo440H “following previous methods.” Without clear held-out splits to reproduce the evaluator, cross-paper comparability and data leakage risks remain. For T2M evaluation on HumanML3D, did the authors train the model on whole Mo440H or HumanML3D alone?

### Questions
1. See **Weakness 1**.

2. Please justify or explain **Weakness 2**.

3. For intention task: What is the ground truth? If it’s an LLM caption of the observed prefix, how do you ensure it reflects “intent,” not mere surface description? Any human study or goal-specific metrics?

4. See **Weakness 4**.

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
This paper addresses key limitations in text-conditioned human motion generation: the lack of explicit intention modeling and the inherent ambiguity of text-only descriptions. Its primary contributions are: **1) Explicit Intention Modeling**: It introduces a novel framework with disentangled generation heads: an Intention Prediction Head (IPH) that generates goal-oriented text, and a Motion Generation Head (MGH) that produces continuous motion trajectories. This allows the model to capture the underlying causal logic of actions. **2) Multi-Modal Conditioning**: It effectively incorporates the visual modality (low-frame-rate images) as a weak prior to resolve textual ambiguity and provide spatial context, enabling more precise and controllable motion generation. **3) Novel Architecture**: It proposes a Conditional Masked Transformer with a mixture-of-attention mechanism that dynamically aligns motion tokens with the most relevant text or visual tokens, handling temporal mismatches between modalities. **4) Large-Scale Benchmark**: It curates Mo440H, a large-scale dataset of 440 hours of motion data with text and visual annotations, facilitating future research.

Extensive experiments show that MoGIC achieves new state-of-the-art performance, significantly outperforming existing methods on metrics like FID and R-Precision.

### Strengths
The main strengths of this paper are:

**Novel Idea**: It introduces a smart new concept instead of just making motions from text. This helps the AI understand why a motion happens, not just how it looks.

**Multi-modal Power**: It successfully combines text with visual information (images). This solves the problem of text being vague and makes the generated motions much more precise and realistic.

**Strong Results**: The method clearly outperforms all current state-of-the-art models on standard tests. The improvements in metrics like FID are significant.

**Great Flexibility**: The same model can handle many different tasks, like generating motion from text, from images, completing missing motion, and even explaining the intention behind a motion.

**Valuable Resource**: It creates a large, new dataset (Mo440H), which will be very useful for other researchers in the field.

### Weaknesses
***The incorporation of the visual modality is a commendable aspect of this work***. However, the proposed strategy of using a fixed, low-frame-rate (1 fps) sampling as the sole visual conditioning signal is a significant limitation that is not adequately addressed. This choice undermines the potential of visual control in several critical ways: 

1).**Lack of Adaptability to Motion Dynamics**: A fixed 1 fps sampling rate is inherently agnostic to the inherent speed and dynamics of the target motion. This is a critical flaw.

For slow, sustained actions (e.g., "meditating"), 1 fps may be sufficient, even over-sampled.

For fast, transient actions (e.g., "a quick jab," "a golf swing," "clapping hands"), a 1 fps sampling is highly likely to miss the crucial kinematic phase of the action (e.g., the moment of impact). As the authors rightly noted, language cannot specify fine-grained spatiotemporal details; unfortunately, a rigid 1 fps visual stream often fails to capture them as well. This can lead to generated motions that are plausible in isolation but temporally misaligned or physically implausible when precise timing is required.

2). **Ignition of the "Keyframe" Paradigm for Controllability**: The current method treats visual inputs as a passive, weak prior. For true controllability, especially in downstream applications like game development or animation, artists and developers require an active, intentional control mechanism. The ability to provide sparse, user-defined keyframes is the industry standard for a reason: it allows for precise control over timing and pose.

The authors' model architecture, with its mixture-of-attention mechanism, seems capable of handling such sparse keyframes, but the training paradigm does not explore this. The work would be significantly strengthened by investigating visual conditioning on semantically meaningful keyframes (e.g., start, end, and apex of a jump) rather than arbitrarily sampled low-frame-rate images.

***Furthermore, the visual evidence supporting the core claim is severely lacking***. The solitary qualitative example in Figure 3 is insufficient. For instance, in a rapid "weightlifting" motion, if the sampled frames coincidentally miss the key "lifting" phase and only show the barbell on the ground, the visual condition would actively contradict the text description. It remains unproven whether the model can robustly override such deceptive visual cues and rely on the stronger textual prior or learned motion, or if it would generate an incorrect sequence. Furthermore, the qualitative analysis supporting the core claim is severely lacking. In the supplementary material, only provide many visualization results of language-to-motion generation, lacking VL2M results.

### Questions
1. The paper demonstrates strong performance on in-distribution datasets. However, how does the model generalize to out-of-distribution (OOD) descriptions or intentions? For example, if the text describes a novel combination of actions ("do a handstand while hopping on one foot") that is unlikely to be in the training data, can the model still generate a plausible motion? An analysis of the model's performance on OOD or compositional tasks would greatly strengthen the claims about its generalizability.

2. The construction of Mo440H is a contribution, but it raises concerns about quality and bias. A significant portion of the textual descriptions is generated automatically by Qwen2.5-VL. What measures were taken to control for the inherent biases and potential errors of this large vision-language model? Furthermore, have you analyzed the distribution of action types within Mo440H? Is there a risk that the model's performance is biased towards the most frequent action categories in this aggregated dataset?

### Soundness
3

### Presentation
3

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
This paper proposes MoGIC, a unified framework for motion generation that integrates "intention understanding" and "visual context." The method models intention via an "Intention Prediction Head" (IPH) that predicts text from motion prefixes, and it uses sparse visual frames to disambiguate text. The method achieves a strong SOTA performance (FID 0.070) on the HumanML3D benchmark and contributes a new auto-annotated dataset, Mo440H.

### Strengths
- The paper's most significant contribution is its SOTA performance on HumanML3D. A huge FID reduction is a very impressive achievement and strongly demonstrates the method's effectiveness.

- The proposed "intention loss" (predicting full text from a motion prefix) is a novel and effective auxiliary training objective. The ablation study clearly shows this predictive task is a stronger supervisory signal than a standard M2T captioning loss.

- The use of sparse, weakly-aligned visual inputs to resolve the ambiguity of text-only descriptions (e.g., the weightlifting example in Fig 3) is a reasonable and practical contribution, confirmed by ablations in Table 5.

### Weaknesses
- The paper's central narrative is built on a foundation of **significant overclaiming**. The "intention understanding" and "causal logic" repeatedly mentioned are, in practice, a proxy task: "predicting a full text description from the first 50% of a motion sequence." Equating this M2T task with high-level cognitive intent or "internal causal structure" is a major conceptual leap without evidence. The attempt to differentiate this from standard M2T methods (like MotionGPT) by calling one "mapping" and this one "intention" is exaggerated and.

- The new Mo440H dataset's text labels are auto-generated by a very strong, contemporary VLM (Qwen2.5-VL-Max). Claiming SOTA performance (especially on captioning) on a dataset whose labels were synthesized by a SOTA model is problematic. The model's performance may be heavily reliant on these high-quality synthetic labels rather than the superiority of the model itself.

### Questions
- Given that the IPH is functionally a "prefix-to-text" M2T predictor, will the authors please substantially tone down the claims about "cognitive intention," "goals," and "causal logic" in the final version? The task's effectiveness as an auxiliary loss is clear (Strength 2), but the narrative is overblown.

- Can the authors discuss the potential impact of using a SOTA VLM to generate labels on the results (especially for captioning)? How can we be sure that the captioning SOTA on Mo440H isn't just a result of the model being good at "mimicking" the Qwen-VL's output style?

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
3

### Summary
The paper presents MoGIC, a unified framework for multimodal human motion generation, supported by a large-scale dataset Mo440H , a modular design with vision priors, an intention head, and Mixture-of-Attention, and demonstrates strong performance through extensive experiments, though some aspects like the intention head’s qualitative role could be clarified.

### Strengths
1. The Mo440H dataset is large-scale, covering a wide range of interactions and motions, providing a rich and diverse benchmark for training and evaluation.
2. The model’s modular design is well-structured, incorporating vision priors, an intention prediction head, and a Mixture-of-Attention mechanism, which appears reasonable for aligning multimodal signals with motion tokens. This method also demonstrates strong effectiveness, consistently outperforming baselines across multiple tasks and datasets.
3. Extensive experiments, including quantitative metrics and multiple downstream tasks, convincingly demonstrate the effectiveness of the proposed framework.

### Weaknesses
1. Ambiguous Definition of Intention: The paper claims that intention modeling is crucial for producing high-quality motion (e.g., caption supervision improves motion quality, but the gains are notably smaller than those from intention prediction). However, it is unclear how “intention” is fundamentally different from the original captions or text. In Figure 2, the intention text appears nearly identical to the motion description, raising the question of whether the Intention Prediction Head (IPH) truly captures high-level latent goals of human motion, or if it simply learns to reconstruct or summarize the original textual input. An additional consideration is whether the IPH is strictly necessary: could the model achieve similar benefits by simply augmenting the original text dataset with intention labels, rather than introducing a separate intention prediction module? 
2. Effectiveness of Visual Priors. According to the qualitative results presented in the paper, the introduction of visual input sometimes leads the model to behave more like performing pose reconstruction conditioned on the visuals, rather than effectively leveraging the visual prior to improve the diversity or realism of the generated motions. From the quantitative results, there is currently a lack of clear metrics that convincingly demonstrate that visual input significantly improves the quality of generated motions. Even after language fine-tuning, the supposed benefits of the visual prior, as claimed by the authors, are not evident, and motion diversity does not appear to improve. Therefore, the actual contribution of the visual modality to motion generation performance remains to be further validated.
3. Limited Qualitative Analysis: The paper presents only a limited number of Vision & Language → Motion examples and provides relatively little qualitative analysis of the Intention Prediction Head, making it somewhat unclear how effectively it functions.

### Questions
Some minor formatting issues are noticed: missing punctuation(Line 236), inconsistent capitalization (Line 958), and in tables, it would be clearer if key entries were bolded.

### Soundness
3

### Presentation
3

### Contribution
3
