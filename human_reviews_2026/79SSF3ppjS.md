# Decomposed Attention Fusion in MLLMs for Training-free Video Reasoning Segmentation

- Decision: Accept (Poster)
- Scores: 4, 6, 2, 8

## Abstract
Multimodal large language models (MLLMs) demonstrate strong video understanding by attending to visual tokens relevant to textual queries. To directly adapt this for localization in a training-free manner, we cast video reasoning segmentation as a video QA task and extract attention maps via rollout mechanism. However, raw attention maps are noisy and poorly aligned with object regions. We propose Decomposed Attention Fusion (DecAF), which refines these maps through two mechanisms: (1) contrastive object-background fusion and (2) complementary video-frame fusion. This method suppresses irrelevant activations and enhances object-focused cues, enabling direct conversion of attention maps into coarse segmentation masks. In addition, we introduce attention-guided SAM2 prompting for obtaining fine-grained masks. Unlike existing methods that jointly train MLLMs with SAM, our method operates entirely without retraining. DecAF outperforms training-free methods and achieves performance comparable to training-based methods on both referring and reasoning VOS benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes DecAF (Decomposed Attention Fusion), a training-free framework for video reasoning segmentation that leverages attention maps from Multimodal Large Language Models (MLLMs). The method introduces two key fusion strategies: (1) contrastive object-background fusion to suppress irrelevant activations, and (2) complementary video-frame fusion to combine temporal context with spatial details. The attention maps are converted to coarse masks via thresholding and refined using SAM2 with an attention consistency scoring mechanism. Experiments across five datasets demonstrate performance comparable to training-based methods.

### Strengths
- The decomposed attention fusion strategy is intuitive and well-designed. The contrastive object-background fusion effectively addresses the visual attention sink phenomenon, while the complementary video-frame fusion leverages the distinct strengths of different input modalities.

- The approach achieves results comparable to or better than several training-based methods (e.g., outperforming VISA and VideoLISA on Ref-DAVIS).

- Thorough Ablation Studies: The ablation studies (Tables 4-7) systematically validate each component, including fusion strategies, rollout methods, SAM prompting thresholds, and scoring mechanisms.

### Weaknesses
- Limited Novelty in Core Technique: While the fusion strategies are well-designed, the core building block (attention rollout) is not novel. The V-Max normalization (Eq. 4) is a relatively straightforward modification. The main contribution lies in the application and combination of existing techniques rather than fundamental algorithmic innovation.

- Computational Cost Not Discussed: The paper does not analyze computational complexity or runtime. The method requires multiple forward passes through the MLLM (for object, background, video, and frame attention maps), which could be computationally expensive. A comparison of inference time with training-based methods would strengthen the practical claims.

- Dependency on Prompt Engineering: The method heavily relies on carefully designed prompts (Appendix B). The robustness to prompt variations is not studied. Different prompt formulations might significantly affect performance, limiting generalizability.

- Missing Generalization Comparison. This is perhaps the most significant weakness of the paper. The primary advantage of training-free methods over training-based approaches is their ability to generalize to scenarios with limited or no training data. However, the paper completely fails to demonstrate this critical advantage.

### Questions
Please refer to the weaknesses.

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
4

### Summary
This paper proposes DecAF, a training-free framework for video reasoning segmentation. The core idea is to leverage the inherent localization abilities of Multimodal Large Language Models in the attention maps by treating the task as a video question-answering problem. To address the noise in vanilla attention maps, DecAF introduces Contrastive Object-Background Fusion and Complementary Video-Frame Fusion. The resulting fused attention map provides a strong coarse localization signal, which serves as prompt for the SAM2 model. Experiments show that DecAF significantly outperforms previous training-free methods and achieves performance comparable to SOTA training-based methods.

### Strengths
1. The idea of exploiting attention map of MLLM as localization signal is interesting. Although using attention map as segmentation has been widely studied in ViT models, it is interesting to see this technique to be applied in the Reasoning VOS task with MLLM as the backbone.
2. The Contrastive Object-Background Fusion and Complementary Video-Frame Fusion modules are intuitive and show compelling results.
3. The experiments and ablation studies are comprehensive.

### Weaknesses
1. To achieve the background contrast and video-frame fusion, it seems like the model need to run inference multiple times, which can significantly increase the cost. It is necessary to discuss this and compare with existing methods.
2. How to correctly get the background attention mask? If I understand correctly, the prompt should specify salient objects. What is the source of salient objects? Is it given by human or?
3. In reasoning segmentation, the text queries can be complex and sometimes don't explicitly mention a specific object. E.g., it can be a question that simply ask "Which object in the scene that ...?" How does this method handle such prompt?
4. The method is limited to only process object-level segmentation from the design. Is it able to segment part of object?

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors propose a method to use the attention map from the MLLM to extract prompts to get the masks from the SAM2 model. The authors test the performance compared to the performance by directly using the attention map (without SAM2 model). The results show that the masks generated by the SAM2 model have a good performance gain compared to the attention mask.

### Strengths
- The experiments demonstrate strong performance on the RefVOS benchmarks.

- The authors evaluate the method's performance across different MLLMs, showing its generalizability.

- The paper includes an ablation study on the proposed modules to validate their effectiveness.

### Weaknesses
- Insufficient and Missing Baselines: The paper's most significant weakness is its baseline setup. While the authors compare against training-free methods like Loc-Head, they omit the most critical and simplest baseline: using the MLLM's native grounding capabilities. Modern MLLMs, particularly Qwen2.5VL (which the authors use), are already capable of generating bounding boxes or points for a given textual expression. A much simpler and more direct training-free baseline would be to use these native grounding outputs (boxes/points) as prompts for SAM2. Given that the paper's best results come from Qwen2.5VL, this "native grounding + SAM2" baseline is essential. Without it, it is impossible to determine if the complex, multi-step DecAF process adds any real value or if it is just an overly complicated method to leverage a powerful, pre-existing capability of the MLLM.

- Computational Cost and Practicality: The "Complementary Video-Frame Fusion" module raises significant concerns about inference efficiency. As described, the method requires one MLLM pass for the entire video and then additional, separate MLLM passes for each individual frame to obtain the final result. This multi-pass design would be prohibitively slow for videos of any reasonable length, making the method impractical for real-world applications. The authors must provide a detailed analysis of the computational overhead (e.g., latency or FLOPs) compared to a single-pass approach.

- Justification of "Training-Free" Claim: The paper's "training-free" claim is questionable. In Section 3.2, the method employs "Head-wise weighted aggregation." It is unclear where these "weights" originate. If any part of the method relies on optimization or selection using a "training set," the "training-free" claim is weakened. In that case, the method should be more thoroughly compared to other training-based methods on an equal footing.

- Limited Novelty and Scope: The paper's novelty appears limited. The framework feels like a straightforward extension of image-domain attention grounding (like Loc-Head) to the video domain. The core technical contributions—the fusion modules—are not particularly novel. "Contrastive Object-Background Fusion" appears to be a simple map subtraction, and "Complementary Video-Frame Fusion" is a simple averaging of multi-scale maps. Furthermore, the main comparison method, Loc-Head, was designed for images, and the “background” contribution is not video-specific. To better demonstrate the contribution of DecAF, the authors should also evaluate its performance on standard image segmentation benchmarks (e.g., RefCOCO) to show its advantage in both domains.

- Attention Map Ambiguity: The reliance on attention maps is a potential weakness, as these maps may not always align with the intended semantic answer. This ambiguity could be particularly problematic in cases with multiple similar objects or when only part of an object is the correct answer (e.g., segmenting "the shirt of a person" from "the person"). The authors should justify how the proposed method handles such nuanced cases.

- Ablation Study Concerns: (a question, I am not very sure) A point of clarification is needed regarding the ablation study. In Table 4(a), removing the "background" module causes a performance drop far below the Loc-Head baseline. This result is counter-intuitive. Why would removing a single proposed component result in performance significantly worse than the baseline? Intuitively, a model with all proposed modules removed should perform similarly to the baseline. The authors should elaborate on this finding.


In conclusion, I am not convinced of the method's practical utility for the RefVOS task. The computational cost is unclear, and it remains uncertain whether the proposed method offers a great advantage over the much simpler, un-tested baseline of using the MLLM’s native grounding capabilities to prompt SAM2.

### Questions
NA

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigate the training-free video referring segmentation task, which is relatively underexplored previously.  The main idea of this paper is to cast referring video segementation task as video QA task and extract attention score in video LLM to perform segmentation. This paper presents a systematic pipeline including Decomposed Attention Fusion, DeCAF, and SAM2 refinement. DeCAF mainly includes two stages: 1) object-background attention rollout subtraction, and 2) sparse video and dense frame fusion. The experimental results of this proposed pipeline are on par with training-based methods on several VOS benchmarks.

### Strengths
1) This paper is one of the first cohorts to investigate the scope of the training-free video referring segmentation with LLM and therefore delivers substantial novelty and insights. The paper made several efforts to work this out and the results are on par with the training-based methods, which should be appreciated.

2) This paper proposed several technical methods specifically tailored for video tasks, including video-frame attention fusion, frame-wise propagation, tracklet scoring and selection, showing that the pipeline is not a simply adopted method from image domain to video domain.

3) This paper performed several ablation studies (in Table 5~8) to validate necessities of proposed components in the pipeline, which is of great value.

### Weaknesses
1) In Table 1 and 2, Why DecAF is with LLaVA-OV-7B while Loc-Head is with LLaVA-7B? Could you please show the results of Loc-Head woth LLaVA-OV-7B?

2) Could the authors show the inference time of the whole pipeline of video segmentation, especially compared to the training end-to-end methods? Because it seems that the training free method is bound to include heavy pretrained methods to refine coarse attention scores. Adding the inference time analysis will increase the technical soundness of this paper.

3) In the demo in Figure 1 (b), the author claims the segementation pipeline can resolve conflicts e.g., between the server and hitting player. Could authors elaborate this mechainism more in details? Is it mainly due to frame-wise prompting & progation or mask tracjlet scoring & selection?

### Questions
See weaknesses.

### Soundness
4

### Presentation
4

### Contribution
3
