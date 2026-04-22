# UAOR: Uncertainty-aware Observation Reinjection for Vision-Language-Action Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Vision–Language–Action (VLA) models leverage pretrained Vision–Language Models (VLMs) as backbones to map images and instructions to actions, demonstrating remarkable potential for generalizable robotic manipulation. To improve performance, many methods have been proposed to incorporate additional observation cues (e.g., depth maps, point clouds) and auxiliary modules (e.g., object detectors, encoders), enabling more precise and reliable task execution. Although effective, these approaches often require extensive data collection and additional training or fine-tuning, limiting their flexibility and scalability. Inspired by the finding that Feed-Forward Network (FFN) in language models can act as "key-value memory'', we propose **U**ncertainty-**a**ware **O**bservation **R**einjection (**UAOR**), an effective training-free and plug-and-play module for VLA models. Specially, when the current language model layer exhibits high uncertainty, measured by **Action Entropy**, it reinjects the observation information into the next layer's Feed-Forward Network (FFN) in a blending manner. This mechanism helps VLA models look more clearly on the observation during inference, enabling more confident and faithful action generation. Comprehensive simulation and real-world experiments show that our method consistently improves the performance of heterogeneous VLA models across various tasks and embodiments while incurring minimal computational overhead. Notably,  **UAOR** eliminates the need for extra observation cuse or modules, making it a versatile and practical plug-in for existing VLA pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Uncertainty-aware Observation Reinjection (UAOR), a novel, training-free, and plug-and-play module designed to enhance the performance of Vision-Language-Action (VLA) models. The core idea is to counteract the "forgetting" of initial observation information as data propagates through the model's layers. The authors propose a metric, "Action Entropy," to measure layer-wise uncertainty during inference. When this uncertainty surpasses a predefined threshold at a given layer, UAOR "reinjects" the original observation features into the Feed-Forward Network (FFN) of the subsequent layer. This mechanism is inspired by the concept of FFNs acting as key-value memory, allowing the model to dynamically "re-focus" on crucial sensory inputs when its confidence wanes. The authors provide a theoretical analysis based on information theory to justify their approach and validate its effectiveness through extensive experiments on multiple simulation benchmarks (LIBERO, SIMPLER, CALVIN) and in the real world, demonstrating consistent performance improvements across various VLA architectures with negligible computational overhead.

### Strengths
- The work is motivated by a clear and compelling intuition—that observation information decays through deeper network layers, leading to increased uncertainty.
- The authors demonstrate the effectiveness of UAOR across three different VLA baselines with varying architectures (single-system and dual-system) and scales (0.5B to 7B), on three distinct simulation benchmarks.
- The inclusion of a theoretical analysis (Section 3.4) adds significant depth and credibility to the paper.
- The paper is exceptionally well-written, with a logical flow and clear explanations of complex concepts.

### Weaknesses
- The proposed "Action Entropy" metric relies on projecting the hidden states of every intermediate layer through the MLP of the final layer to get a probability distribution. This approach feels somewhat ad-hoc and potentially inefficient. It raises questions about whether this is the most direct or optimal way to measure uncertainty, as it depends on a component (the final layer's MLP) far downstream from where the uncertainty is being measured.

- The rationale for this one-layer delay is not discussed. Ablation studies exploring reinjection into the current layer's FFN, or even into the self-attention block, are missing and would provide valuable insight into this specific design choice.

- While the selection of baselines is commendably diverse, the paper acknowledges that it does not apply UAOR to some of the most recent and powerful state-of-the-art models like $\pi_0$. Demonstrating gains on these near-SoTA models would make the claims of general applicability even more powerful.

- The analysis in Section 4.4 explores a heuristic for weighting visual tokens based on language similarity but finds it does not improve performance over uniform weighting. While the authors provide plausible hypotheses, this result is somewhat counter-intuitive and warrants a deeper investigation. It might suggest that the simple language-similarity weighting is flawed, rather than the concept of weighting itself.

### Questions
- Could you provide a more detailed rationale for the "Action Entropy" metric design? Specifically, why project intermediate hidden states through the final layer's action head, rather than using a more direct uncertainty measure from the hidden states themselves (e.g., entropy of the feature distribution, or a lightweight learned uncertainty head)? What is the computational overhead of these repeated projections during inference?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes UAOR, a lightweight, training-free module designed to boost VLA models. It aims to enable VLA models
look more clearly on the observation during inference, enabling more confident and faithful action generation.

### Strengths
1. Demonstrates a strong understanding of the current limitations of VLA models.

2. Attempts to address the identified problems in a general and systematic way.

3. Proposes the concept of action entropy and applies it effectively in the forward process. The accompanying theoretical analysis is solid and convincing.

4. Provides a well-designed ablation study to examine the effects of observation injection.

### Weaknesses
1. The explanation of why forgetting leads to uncertainty lacks a clear reasoning process.

2. Theorems 3.1–3.4 appear largely independent and not well integrated. It would be better to unify them into a holistic framework or justify that they represent complementary perspectives on the same problem.

3. Real-world experiments are only conducted using Open-VLA, neglecting other baseline models such as CogACT.

4. The relationship between α and γ should be jointly analyzed, as variations in one may influence the behavior or trend of the other.

### Questions
1. Is the action entropy gate essential for injecting observation features?

2. Can hidden states or proprioceptive states reliably represent action entropy?

3. In Equation (9) of Algorithm 1, why is h_t^{(l+1)} used instead of h_t^{(l)}? The paper does not discuss uncertainty propagation between adjacent layers.

4. In Section 4.4, the discussion on token-level weighting explores applying weights to visual tokens based on language instructions. However, the ablation study in Section 4.3 shows that injecting language instructions offers no performance gain. This suggests that the similarity between visual and language tokens provides limited benefit. Why not explore similarities between visual and proprioceptive tokens instead? Besides, It is plausible that redundant (similar) information adds little value, whereas orthogonal (complementary) information may be more beneficial.

### Soundness
3

### Presentation
3

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
This paper proposes UAOR (Uncertainty-Aware Observation Reinjection), a training-free and plug-and-play module designed to enhance VLA models. UAOR measures layer-wise action entropy to detect high-uncertainty regions during inference. When uncertainty exceeds a threshold, UAOR reinjects observation features into the next layer’s FFN, treating the module as a key-value memory to restore visual representation. Experiments across multiple benchmarks and real-world robot tasks show a certain degree of performance gains with negligible computational overhead.

### Strengths
S1. I think the problem addressed by this paper is very important, as overly deep LLM layers do tend to ignore certain visual information.

S2. The paper proposes a simple yet efficient method to alleviate this issue.

### Weaknesses
W1. My main concern lies in whether using Action Token Entropy to measure visual uncertainty is reasonable.


W2. Pretrained VLA models typically generate actions only from the final layer, without utilizing or supervising intermediate features for action prediction. Therefore, the observed layer-wise “action” token entropy may result from the training paradigm itself rather than reflecting the actual dynamics of feature changes within the model. I suggest that the authors finetune the VLA model so that each LLM layer outputs actions for verification.

W3. Regarding the analysis of Action Token Entropy in Figure 1, I also find it unconvincing. Prior machine learning studies have shown that deeper layers’ feature representations are more task-specific, which could naturally cause the entropy variation (rather than the reason proposed by the authors).

### Questions
Q1. My minor concern is that the performance improvements brought by UAOR are relatively small. For example, on LIBERO and CALVIN, the gains are only about 0.9\%, which seems quite incremental. I therefore recommend that the authors consider finetuning the model so that it can better adapt to the newly introduced token sequence.


Q2. Since the authors’ motivation is very good, I would be willing to raise my score if they can address my concerns.

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
4

### Summary
This paper introduces a training-free method for reinjecting observations back to transformer layers of VLAs. The paper claims that VLAs tend to progressively forget observations, and they proposes to blend the observation features with the FFN output via the key-value memory mechanism. The paper compares against other VLAs on the LIBERO, SIMPLER, and real-world benchmarks, showing some gains while not introducing too much latency.

### Strengths
- The paper is fairly well-written and easy to understand.
- The problem studied is important.

### Weaknesses
- The core premise of the paper is not well-supported. The author claims that: "Our key intuition is that after ingesting the observation, the model tends to progressively “forget” during forward inference" and back this up by Figure 1, where the **early** layers of the VLA experiences a mild increase in action uncertainty. This is neither convincing nor well-explained. To me, Fig. 1 is actually quite reasonable: early in the computation, there's more uncertainty in the action distribution but as the model has the chance to process the representations more in later layers, the uncertainty decreases, almost to 0 at the last layer. 
	- Even if the claim that increasing in uncertainty equals to forgetting observation, this doesn't explain how the uncertainty eventually decreases at the mid layer and almost to 0 at the final layer.
	- It is not clear that action uncertainty can be tied to "forgetting observation" as neural networks dynamics can be complicated. We need stronger evidence, for example drawing tools from interpretability research, perhaps looking at some sort of cross attention between the outputs and the observations, to validate the core premise.
- The quantitative results are not clear.
	- All the gains are very modest.
	- No measure of statistical confidence. No details on how many trials were run. 
	- Table 4 is not clear. What task / benchmark is this?

- Some technical details are not well-justified. 
	- It is not clear that if you just take the middle hidden representations and decode that, computing the "action" entropy is still meaningful. For intuition, for classifer guidance in image diffusion literature, you can't just take the intermediate results during denoising, and ask a classifer that was ever only trained on clean images to infer on noisy inputs. You either need to train the classifier on noisy images too or use techniques like diffusion posterior sampling, tangent projection et cetera. So again, it's not clear to me that decoding actions from an intermediate layer is a meaningful operation.
	- If I understand this correctly, the paper applies their re-injection during test-time without ever training the base policy. During training, the base model is only exposed to hidden representations while during inference it must deal with "out-of-distribution" embeddings from reinjecting observation. It's not clear why this OOD issue wouldn't destroy performance.

### Questions
- Do you see any connection between the problem and residual networks? The intuition of skip connection is fairly similar: as the depth increases, it's harder to train, and the network "forgets" its observation. Thus, skip connection adds the observation back after each layer.
- It would be nice to see a conceptual and also quantitative discussion / comparison with skip connections (which is actually already present in the transformer architecture)

### Soundness
2

### Presentation
3

### Contribution
2
