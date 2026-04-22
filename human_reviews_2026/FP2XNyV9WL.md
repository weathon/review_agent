# BindWeave: Subject-Consistent Video Generation via Cross-Modal Integration

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Diffusion Transformer has shown remarkable abilities in generating high-fidelity videos, delivering visually coherent frames and rich details over extended durations.
However, existing video generation models still fall short in subject-consistent video generation due to an inherent difficulty in parsing prompts that specify complex spatial relationships, temporal logic, and interactions among multiple subjects. To address this issue, we propose BindWeave, a unified framework that handles a broad range of subject-to-video scenarios from single-subject cases to complex multi-subject scenes with heterogeneous entities. 
To bind complex prompt semantics to concrete visual subjects, we introduce an MLLM-DiT framework in which a pretrained multimodal large language model performs deep cross-modal reasoning to ground entities and disentangle roles, attributes, and interactions, yielding subject-aware hidden states that condition the diffusion transformer for high-fidelity subject-consistent video generation.
Experiments on the OpenS2V benchmark demonstrate that our method achieves superior performance across subject consistency, naturalness, and text relevance in generated videos, outperforming existing open-source and commercial models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces BindWeave, a subject-consistent video generation framework that leverages deep cross-modal integration between text and image inputs. The authors claim that the existing models typically fail to handle complex multi-subject interactions. To overcome this, BindWeave integrates a MLLM with a DiT. Specifically, the MLLM is used to encode spatio-temporal relationships into subject-aware hidden states, which then condition the diffusion process for video synthesis. The experiments are conducted on the OpenS2V benchmark and show BindWeave outperforming state-of-the-art open-source and commercial baselines.

### Strengths
- This paper targets at subject-to-video generation which is an interesting and practical problem relevant to lots of real-world application, such as video personalization or content creation.
- The proposed adaptive multi-reference conditioning strategy sounds well-motivated to me as it allows the model to flexibly integrate multiple reference images without disrupting temporal coherence.
- The model is trained on the open-sourced OpenS2V-5M dataset, and the paper provides thorough implementation details which could enhance the reproducibility of the proposed framework.
- The experiments include a comprehensive comparisons with 9 open-sourced and proprietary baseline models. The quantitive results on the OpenS2V-Eval benchmark look promising and show consistent improvements from some existing models.
- The writing quality is high and easy to understand. The figures are also well-plotted which can assist the readers to quickly understand the proposed framework.

### Weaknesses
- My major concern is the limited technical contribution. The framework jointly using "frozen MLLM & trainable connector" + "VAE tokens" + "noisy latents" to condition the diffusion process is a very common design in recent editing models, such as Step1X-Edit [1] and Qwen-Image [2]. The framework figures (i.e., Figure 2 of BindWeave paper and Figure 4 of Step1X-Edit paper) are nearly the same. It significantly limits the technical contribution/novelty of the paper.
- While one of the main claims of the paper is the usage of the powerful MLLM for cross-modal reasoning, most of the provided samples (e.g., "[man] holds a [ball]", "[sauces] placed nearby a [wok]"), however, do not require complex reasoning and can be simply achieved by existing models. While Figure 6 shows the comparison of the samples using T5-only and MLLM+T5, the results do not look reasonable to me as I never saw a model combining [dog] and [bowl] in this way even without MLLM reasoning capability.
- Following the above concern, the authors should provide samples to demonstrate the effectiveness of MLLM in handling complex multi-subject interactions that require spatial-temporal reasoning. For example, a prompt like "[basketball court] and four [basketball players]", where two teammates wear same uniforms and one is attacking while the others are defending. In this sample, the model should correctly place the two teammates on the same side and assign proper actions to each player to achieve spatial-temporal coherence in the output video.
- Similar to the existing editing frameworks, one major limitation of using VAE tokens as condition is the presence of copy-paste artifacts. In the provided samples, I also noticed heavy pixel-level alignment between the conditional images and the output videos. To further verify whether BlindWeave suffers from this issue, the authors should include samples with conflicting "image-image" or "image-text" coherence. For example, "a [garment image] and a [person image] with different poses" or "make a [crying boy image] [smile (text)]"
- Following the concern above, the model seems to achieve poor trade-off between "pose variation" and "fidelity". For example, Figure 7 shows high fidelity but low pose variation (i.e., heavy pixel-level alignment), while Figure 8 shows huge pose variation but low fidelity. Based on these results, I don't think BlindWeave can solve these copy-paste artifacts well.
- The scope of ablation study is very limited. There is only one ablation (T5-only vs. T5+MLLM), while it is also interesting to check the effect of isolating CLIP, VAE, or individual conditional streams.
- The authors do not provide supplementary material with video samples, which makes the reviewers unable to justify the quality of the output videos. It is impossible to check the temporal smoothness simply based on the provided video frames.

[1] "Step1X-Edit: A Practical Framework for General Image Editing"

[2] "Qwen-Image Technical Report"

### Questions
- Could the author provide more samples that require complex spatial-temporal reasoning to verify the effectiveness of the MLLM model?
- Could the author provide more samples with conflicting conditions which can verify whether BlindWeave suffers from copy-paste artifacts?
- Could the author provide more detailed ablation study to verify the effectiveness of each module in the framework?
- The model performance looks inconsistent in Figures 7 and 8. Could the authors explain this phenomena? Does the model include a mechanism supporting the adjustment of the trade-off between "fidelity" and "pose variation"?
- [Minor] The citation format throughout the paper is wrong. For example, should be "Phantom (Liu et al., 2025)" instead of "Phantom Liu et al. (2025)". You can modify it by changing "cite"/"citet" to "citep"
- [Minor] In line 210: should be "we introduce" instead of "we introduces"

### Soundness
3

### Presentation
4

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
This paper targets subject consistent text guided video generation. The key idea is to replace the common separate then fuse conditioning paradigm with cross modal integretion using a MLLM that parses the interleaved prompt with reference images sequence into subject-aware hidden states. These are projected and used alongside T5 text embeddings, CLIP identity features, and VAE low-level features to collectively condition a DiT-based video generator. They showed the competitive results on OpenS2V.

### Strengths
1. Deep cross modal binding, the MLLM parses text and image jointly, producing subject aware hidden states that encode roles, attributes, interactions, addressing identity confusion and instructed shallow fusion. 
2. Strong ablations and qualitative evidence. T5 only and T5 with MLLM quantitative and qualitative eval shows better scale grounding, action object execution, and temporal coherence.

### Weaknesses
major: 
1. Using image condition to Qwen2.5 / VAE / CLIP looks redundant, Qwen2.5 and VAE makes sense and also shown a part in your ablation, but why we need a additional CLIP image encoder to encode the condition images.
2. The approach heavily depends on a relatively heavy Qwen2.5VL. The paper doesn’t quantify the extra latency and VRAM and how it scales with K references or video length, more detail would be appreciated. 
3. involving the MLLM will bring benefits and also the original issue inherit from the MLLM, the planning itself may fail due to lots of reason like hallucination, how these case will affect the subsequence video generation?

minor: 
1. line 102, opens2v --> OpenS2V
2. line 351/359 BinWeave  --> BindWeave, the method name is not consistent in this paper.

### Questions
see the major weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces BindWeave, a video generation model based on multiple reference subjects. The model uses an MLLM+DiT architecture, where the MLLM is responsible for understanding the reference subjects and the text prompt, generating semantic embeddings that are input into the DiT model. The DiT model acts as the renderer to generate the video. Experiments show that the proposed method can achieve subject-consistent video generation and understand the logical relationships between the reference subjects and the prompt.

### Strengths
+ The paper is well-motivated to improve complex spatial relations, temporal logic, and multi-subject interactions in subject-to-video generation.
+ The idea of using an MLLM for multimodal reasoning to better understand the reference images and the text prompt is reasonable.
+ The proposed method outperforms open-source and commercial baselines on the OpenS2V benchmark across subject consistency, naturalness, and text relevance.

### Weaknesses
+ The experiments are insufficient, lacking ablation studies on the model design. For example, a comparison with other multi-reference conditioning strategies, whether joint training with MLLM would lead to better results, and if there are other ways to combine MLLM and DiT. The authors should include more experiments to demonstrate the superiority of the model design and training strategies.

+ More cases with multiple reference images are needed rather than single-reference generation, since the MLLM mainly enhances the ability to understand the logical and positional relationships between different subjects.

+ The model design is similar to existing work, and the novelty is limited. Other potential model design solutions should be explored and accompanied by more in-depth analysis.

### Questions
After fine-tuning the base model with the proposed method, is there any observed degradation in video generation quality? For example, how does the visual quality and dynamic range of the videos generated by BindWeave compare to those generated by the vanilla base model? I believe it is necessary to include this comparison.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
the paper proposes a MLLM-conditioned video diffusion framework for multi-subject referenced video generation

### Strengths
- the paper proposes a simple yet effective framework for consistent multi-subject video generation
- the overall framework design and related data flow are reasonable and clearly explained
- the visual comparisons demonstrate the effectiveness of the proposed model

### Weaknesses
- while the paper adopts a existing evaluation benchmark, the reliability of the used evaluation metrics is still not well justified. especially the rationale of "total score" is questionable considering it is a normalized weighted sum of other scores: does it really reflect the actual performance? can those different so-called metrics be linearly combined in this way? a user study is highly recommended to validate the effectiveness of the proposed method considering the human evaluation is still the most reliable evaluation measure for video generation tasks
- the paper mentioned that the proposed framework accepts a flexible number of reference images typically ranging from 1 to 4. however, it is unclear what is the max possible number of subjects the proposed method can handle: is it determined by the training? or does it can generalize to an much larger number than 4 during inference? more analysis is needed to validate the scalability of the proposed framework
- it seems that there are only limited motion in the presented visual results. in that case, it is unclear whether the proposed framework can handle the case where the generated subjects are in largely different scales from the reference images or changing scales. perhaps some testings with zoom-in and zoom-out subjects can be conducted to validate the robustness of the proposed method
- what is the success rate of generation for a set of given subjects? what are the typical failure cases?
- minor: it might be better to provide actual videos as supplementary to present the results rather than only keyframes

### Questions
please refer to weaknesses section

### Soundness
3

### Presentation
3

### Contribution
3
