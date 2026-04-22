# Scene-R1: Video-Grounded Large Language Models  for 3D Scene Reasoning without 3D Annotations

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 6, 2

## Abstract
Currently, utilizing large language models to understand the 3D world is becoming popular. Yet existing 3D‑aware LLMs act as black boxes: they output bounding boxes or textual answers without revealing how those decisions are made, and they still rely on pre‑trained 3D detectors to supply object proposals. We introduce Scene‑R1, a video‑grounded framework that learns to reason about 3D scenes without any point‑wise 3D instance supervision by pairing reinforcement‑learning‑driven reasoning with a two‑stage grounding pipeline.
In the temporal grounding stage, we explicitly reason about the video and select the video snippets most relevant to an open‑ended query. In the subsequent image grounding stage, we analyze the image and predict the 2D bounding box. This 2D prediction is then refined into a precise 3D localization by matching against high-fidelity candidates from a zero-shot segmentation module, which captures fine geometry while eliminating the need for detector-based proposals.
Scene-R1 can also adapt to the 3D visual question answering task to answer free-form questions directly from video. Our training pipeline only needs task-level 2D boxes or textual labels without dense 3D point-wise labels. Scene-R1 surpasses existing open-vocabulary baselines on multiple datasets, while delivering transparent, step‑by‑step rationales. These results show that reinforcement‑learning‑based reasoning combined with RGB‑D video alone offers a practical, annotation‑efficient route to trustworthy 3D scene understanding.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces **Scene-R1**, a video-grounded vision-language model (VLM) for 3D scene understanding that operates **without point-wise 3D annotations**. The core innovation lies in a **two-stage reinforcement learning** pipeline: temporal grounding followed by image grounding, both guided by lightweight rewards such as IoU and format compliance. Additionally, the paper extends the **GRPO** approach to 3D understanding by introducing an *exact-match reward*, achieving performance comparable to 3D LLMs that rely on point cloud inputs. Qualitative results further demonstrate the effectiveness of the model’s reasoning process.

### Strengths
1. The method removes the need for 3D point-wise instance labels while maintaining competitive performance under weak supervision.
2. By explicitly outputting chain-of-thought (CoT) reasoning, Scene-R1 improves interpretability compared with previous 3D LLMs, aligning with the growing emphasis on model transparency and explainability.
3. The two-stage RL structure (temporal then spatial grounding) provides flexibility and task generality across different 3D understanding tasks.

### Weaknesses
1. **The performance against other 3D LLMs remains limited**. The comparison with **VLM-Grounder** is not entirely fair, as it is a training-free agent and the reported results are based on only a 250-sample subset. For a more rigorous evaluation, performance should be assessed on the same benchmark samples used by VLM-Grounder. Although the paper claims that the proposed method does not require instance masks, the distinction between bounding-box-based and segmentation-based supervision is largely mitigated by the use of pretrained **SAM**. Moreover, the baseline **LLaVA-3D** similarly does not depend on pre-extracted 3D bounding boxes or per-instance segmentation, and should therefore be regarded as a **direct and comparable baseline** to the proposed approach.
2. **Similar grounding method:** The concept of back-projecting SAM masks to obtain 3D bounding boxes is not novel. The authors do not clearly distinguish their method from prior approaches such as **VLM-Grounder**.
3. **Limited benchmarks:** The RL framework is introduced not only for transparency but also for generalization. However, the evaluation is restricted to in-domain datasets. Cross-dataset evaluations on **Nr3D [1]**, **Multi3DRefer [2]**, or **Video-MME [3]** are encouraged to validate generalization.
4. **3DVQA implementation:** The paper claims that Scene-R1 can be fine-tuned for 3D-VQA tasks (L272). However, neither the training data nor the evaluation includes 3DVQA datasets such as **ScanQA [4]**, **SQA [5]**, or **MMScan [6]**. Since **VSI-Bench** does not provide a training set, it is unclear what data were used. 
5. **Efficiency concerns:** The proposed multi-stage grounding combined with a DeepSeek-R1-style reasoning process substantially reduces efficiency. Ablation results show that the thinking process yields only marginal performance gains, casting doubt on the overall effectiveness of the proposed method.

[1] https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460409.pdf

[2] [[2309.05251\] Multi3DRefer: Grounding Text Description to Multiple 3D Objects](https://arxiv.org/abs/2309.05251)

[3] [[2405.21075\] Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis](https://arxiv.org/abs/2405.21075)

[4] [[2112.10482\] ScanQA: 3D Question Answering for Spatial Scene Understanding](https://arxiv.org/abs/2112.10482)

[5] [[2210.07474\] SQA3D: Situated Question Answering in 3D Scenes](https://arxiv.org/abs/2210.07474)

[6] [[2406.09401\] MMScan: A Multi-Modal 3D Scene Dataset with Hierarchical Grounded Language Annotations](

### Questions
1. How does RL fine-tuning on grounding tasks improve performance on **VSI-Bench**? What prompts are used during VSI-Bench evaluation?
2. What is the **ablation setting**? The reported ablation results seem inconsistent with the main table. Additionally, what supervised fine-tuning (SFT) configuration is used in these ablations?
3. In L141, the authors state:
    *“We exploit this prior to teach it to understand the 3D world and minimize the amount of task-specific reinforcement learning (RL). The same architecture is used in all tasks optimized with GRPO.”*
    How exactly is the amount of task-specific RL minimized, given that the method introduces several task-specific rules, such as temporal and image grounding?
4. What do the **failure cases** look like? The paper presents only successful examples. A detailed failure mode analysis would provide deeper insight into the limitations of the proposed approach.

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
3

### Summary
The paper introduces a video-grounded large vision-language model (VLM) that performs 3D scene reasoning and grounding without any point-wise 3D annotations. Instead of relying on pretrained 3D detectors, Scene-R1 integrates reinforcement-learning-driven reasoning (R1-style) with a two-stage grounding pipeline, enabling transparent, interpretable, and annotation-efficient 3D reasoning.

The proposed method, Scene-R1, builds on Qwen2.5-VL-7B and is fine-tuned using GRPO. In Stage 1 (Temporal Grounding), the model reasons over video sequences to identify the most relevant temporal segment corresponding to a textual query. In Stage 2 (Image Grounding), it localizes the target object in selected frames by predicting 2D bounding boxes, accompanied by explicit chain-of-thought explanations. These 2D predictions are then lifted to 3D using depth maps and refined via a zero-shot segmentation module, producing accurate 3D localizations without any 3D supervision.

### Strengths
1. Annotation Efficiency:  Scene-R1 achieves competitive 3D reasoning and grounding performance without relying on dense point-wise 3D annotations or pretrained 3D detectors, greatly reducing the data and labeling cost.

2. The authors conducted comprehensive experiments with various existing works, and shows good performance.

### Weaknesses
1. The method rewards properly formatted CoT and task success (IoU/EM), but does not verify that the CoT is faithful to the internal decision path[1,2]

2. While the proposed pipeline has not been widely applied in existing 3D LLMs, its design does not represent a substantial conceptual departure from established video-grounding or multi-stage reasoning frameworks. The contribution feels more like an adaptation of existing ideas to a new input modality rather than a fundamentally novel approach.


[1] Sarkar, Advait. "Large language models cannot explain themselves." arXiv preprint arXiv:2405.04382 (2024).

[2] Kambhampati, Subbarao, et al. "Stop Anthropomorphizing Intermediate Tokens as Reasoning/Thinking Traces!." arXiv preprint arXiv:2504.09762 (2025).

### Questions
Please address the weakness mentioned above.

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
4

### Summary
This paper introduces Scene-R1, a framework for 3D scene reasoning that operates directly on RGB-D video streams and, critically, requires no 3D point-wise annotations for training. The method uses a two-stage, VLM-based pipeline: (1) temporal grounding to select relevant video snippets and (2) image grounding to predict 2D bounding boxes. These 2D predictions are then lifted to 3D using SAM2 and a refinement module. The entire pipeline is optimized using reinforcement learning (GRPO), which both trains the model using lightweight 2D/textual rewards and encourages the generation of explicit chain-of-thought rationales for interpretability. The model is evaluated on 3D visual grounding, affordance grounding, and VQA, demonstrating competitive performance against other annotation-free baselines.

### Strengths
1. The most significant strength is the "annotation-free" nature of the 3D instance labeling. By learning from 2D bounding boxes and textual labels, the method drastically lowers the supervision requirements for 3D scene understanding, making it more scalable.
2. The integration of R1-style reinforcement learning to produce explicit chain-of-thought rationales adds a strong interpretability component, which is lacking in most 3D-aware LLMs.
3. The quantitative results are solid, showing that Scene-R1 outperforms other annotation-free baselines on several benchmarks (ScanRefer, SceneFun3D, VSI-Bench), validating the effectiveness of the proposed approach.

### Weaknesses
1. The system's design is a complex pipeline of multiple, powerful, pre-trained models (Qwen2.5-VL, SAM2, and a module inspired by SAI3D). This makes it difficult to ascertain how much of the strong performance is attributable to the novel RL framework versus the inherent power of these individual components.
2. The method's reliance on ground-truth depth ($D_t$) and camera poses ($T_t$) is a significant assumption. This data is not available in general "in-the-wild" videos and is the same data required to create the point clouds for detector-based methods. This weakens the claim of "bypassing 3D scene reconstruction" and limits the method's applicability to settings where a full 3D capture setup is already available.
3. The 2D-to-3D lifting process has several stages (2D box prediction, SAM2 segmentation, depth-based back-projection, refinement). This multi-step process seems susceptible to cascading errors, where a poor 2D box from the VLM could lead to an irrecoverably bad 3D localization.

### Questions
1. How critical is the explicit depth channel ($D_t$) and ground-truth pose ($T_t$)? What is the performance degradation if the model is run on RGB-only video and must rely on estimated depth/pose, or if it must operate without them? This seems to be the key bottleneck for real-world application.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a video-grounded LLM which do not use 3D instance annotations for training. Specifically, the input to the model is a video: the VLM is asked to predict the relevant frames and then ground relevant object in this relevant portion of the video. For training these modules, GRPO losses are used. Next, these 2D predictions are lifted to 3D — for this, each predicted mask is tracked across frames using SAM-2 and then the resulting masks from all frames are fused in 3D. Next, another merging strategy from a prior work (SAI3D) is used to obtain a sharper mask. This becomes the prediction of the model. The paper compares its methods with prior methods that utilize 3D supervision as well as methods that do not use 3D supervision. The paper claims better performance than methods that do not use 3D supervision. The ablations show that RL training and thinking help improve performance over supervised fine-tuning.

### Strengths
- The paper is well-written and easy to follow
- The premise of training models without 3D supervision is interesting; additionally exploring RL training for these models is interesting as well.

### Weaknesses
- A big claim of the paper is that their method do not use 3D annotations. However, I think that is not entirely true — in “image grounding” task, the proposed model trains for supervising the mask prediction of the relevant object for each image in the video. This requires two kinds of supervision: a) “grounding” supervision which tells the model which object it should be grounding. b) “mask supervision” of that object across ALL video frames. These labels in scannet come from projecting the GT 3D segmentation masks to 2D. I will further argue that 3D mask annotations and 2D video masks are equivalent supervision for a posed RGB-D video i.e. either of these can be obtained from the other one via 2D projection or 3D unprojection. Hence, either of these supervisions is equally costly or inefficient. Hence, the claim that this method trains without 3D annotation appears wrong to me
- In the same vein, the comparisons in table-1 are potentially unfair:
    - In the “free from 3D instance or annotation supervision” section where the proposed method groups itself, the other baselines like vlm-grounder, open-scene and lerf do not use ANY supervision — neither any grounding supervision nor any mask supervision. The current method uses both these ground truth supervision as I argue in the first point
    - In the fully supervised methods section, the baselines are significantly old. Current SOTA is UniVLG (https://arxiv.org/abs/2503.10745) an the authors can check Table-1 of UniVLG for additional recent baselines. 
-  “This architecture uniquely enables end-to-end reasoning directly on video streams, bypassing the need for offline 3D scene reconstruction”: This is a statement made in the introductions, however,  I think section 4.3 which lifts the 2D masks to 3D uses the reconstructed point clouds, and so do all the evaluations that follow in Table-1.

### Questions
The main question in my review, as I explain in the weakness section, is that the claim of not using 3D annotations seems false and the comparisions with zero-shot methods unfair. Any clarification would help here.

### Soundness
1

### Presentation
3

### Contribution
2
