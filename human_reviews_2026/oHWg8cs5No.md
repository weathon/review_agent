# Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Zero-shot Human-object interaction (HOI) detection aims to locate humans and objects in images and recognize their interactions. While advances in open-vocabulary object detection provide promising solutions for object localization, interaction recognition (IR) remains challenging due to the combinatorial diversity of interactions. Existing methods, including two-stage methods, tightly couple IR with a specific detector and rely on coarse-grained vision-language model (VLM) features, which limit generalization to unseen interactions. In this work, we propose a decoupled framework that separates object detection from IR and leverages multi-modal large language models (MLLMs) for zero-shot IR. We introduce a deterministic generation method that formulates IR as a visual question answering task and enforces deterministic outputs, enabling training-free zero-shot IR. To further enhance performance and efficiency by fine-tuning the model, we design a spatial-aware pooling module that integrates appearance and pairwise spatial cues, and a one-pass deterministic matching method that predicts all candidate interactions in a single forward pass. Extensive experiments on HICO-DET and V-COCO demonstrate that our method achieves superior zero-shot performance, strong cross-dataset generalization, and the flexibility to integrate with any object detectors without retraining. Code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper leverages Multi-modal Large Language Models to address HOI detection problem and decoupled object detection and interaction recognition. Interaction recognition is reformulated as a VQA task to leverage the MLLM abilities. The authors propose two pipelines, a training-free deterministic generation method, which replaces open-ended text generation with conditional likelihood estimation; and a fine-tuning pipeline, which enhances efficiency with a spatial-aware pooling module and a one-pass deterministic matching strategy. The method achieves strong zero-shot and cross-dataset performances.

### Strengths
1. The paper first leverages MLLM to solve HOI problem, which is novel and promising, as it connects high-level vision-language reasoning with structured HOI understanding.
2. The paper presents two pipelines, a training-free deterministic pipeline and a fine-tuned version and both achieve competitive performances.
3. Traditional MLLM-based generation is computationally expensive and unstable for classification-style tasks. The paper proposes to reformulate the open-ended generation into a deterministic generation problem. In fine-tuning version, it proposes a single-pass feature matching task using HOI tokens and cosine similarity, improving efficiency.
4. The experiments systematically compare MLLMs of different scales, demonstrating consistent improvements as model size increases (from 0.5B to 7B). This analysis provides valuable evidence that scaling multimodal models enhances HOI reasoning, further validating the proposed method.

### Weaknesses
1. Since the method leverages the MLLMs, naturally with strong open-vocabulary capability evaluating on open-vocabulary benchmarks such as SWiG-HOI [a] would better demonstrate its true generalization to unseen categories.
2. The paper lacks failure case analysis or qualitative discussion. Despite leveraging a large multimodal model, the best mAP remains around 45 in Table 1, 6, 7, suggesting unaddressed limitations.
3. The paper mainly improves MLLM with spatial information, while HOI detection inherently involves both spatial understanding (detection) and semantic reasoning (interaction recognition). It remains unclear whether original MLLM reasoning is already sufficient for HOI detection. A deeper discussion on this design choice (focusing on spatial information) would strengthen the overall analysis.

References:

[a] Learning Transferable Human-Object Interaction Detector with Natural Language Supervision, CVPR2022

### Questions
1. I have a question about the Table 2 for cross-dataset evaluation. Since V-COCO and HICO-DET have different pre-defined HOI classes, only baselines supporting to add new HOI classes in inference can be evaluated in this cross-dataset evaluation. Then, I wonder how ADA-CM is evaluated in table 2, whose fine-tuning version seems to be unable to add new classes in inference.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a decoupled HOI pipeline that keeps object detection separate from interaction recognition and uses an MLLM to classify interactions.
Key ingredients are a “deterministic generation” proxy for multi‑label IR, a spatial‑aware pooling (SAP) module that fuses ROI features with pairwise spatial cues via cross‑attention, and a one‑pass deterministic matching scheme to score all candidate verb–object pairs in a single forward pass.
The method reports gains on HICO‑DET zero‑shot splits, a cross‑detector setting (e.g., swapping in Grounding‑DINO or YOLO‑World), and cross‑dataset transfer to V‑COCO, with extensive ablations.

### Strengths
- The approach yields strong zero‑shot results on HICO‑DET, including a training‑free variant (31.5 mAP), and improves further with LoRA fine‑tuning and the proposed modules; it also transfers well to V‑COCO (59.91 mAP).

- The paper demonstrates plug‑and‑play use with multiple detectors (ResNet‑50 DETR, Grounding‑DINO, YOLO‑World) without retraining the IR head. This supports the “detector‑agnostic” design goal.

### Weaknesses
- From both architectural and modular perspectives, the proposed method exhibits limited novelty. Architecturally, it essentially follows the conventional two-stage HOI detection paradigm, where the detection and interaction recognition stages are sequentially processed. The only modification lies in substituting these two components with more powerful counterparts, an open-vocabulary detector (e.g., Grounding-DINO) and an MLLM-based interaction recognizer, without introducing new mechanisms or architectural restructuring. At the module level, each component is constructed from widely used elements: SAP integrates ROIAlign, MLPs, and cross-attention layers, coupled with a standard pairwise spatial vector; interactiveness is estimated via a simple linear classifier; and the deterministic generation/matching merely reformulates scoring through prompt-based likelihood estimation and LoRA fine-tuning, which are established techniques in existing MLLM literature. Consequently, the methodological advancement is incremental relative to prior two-stage and CLIP/MLLM-augmented HOI frameworks

- The claimed zero‑shot setting is substantially aided by powerful pretrained components whose training corpora likely cover many HOI objects and patterns. Implementation explicitly relies on Grounding‑DINO/YOLO‑World for detection and Qwen2.5‑VL for IR, and the candidate interaction list is built from object categories, meaning much of the supervision is imported from large‑scale pretraining and the dataset’s curated verb–object space. As a result, the evaluation is closer to dataset‑level zero‑shot composition under heavy external priors than to strict zero‑data transfer.

- Although one‑pass matching and interactiveness filtering reduce the number of MLLM calls, the pipeline still couples a heavy open‑vocabulary detector with an MLLM‑based pairwise scorer, and the number of pairs scales quadratically with detections. The reported per‑image times in Table 4 appear to measure the IR component; it is unclear whether detector runtime, prompt construction, and tokenization are included, and how the overall throughput compares to strong one‑stage baselines at matched accuracy.

### Questions
- What exactly is included in the latency reported in Table 4? Please report end‑to‑end wall‑clock time, memory, and GPU count for detection + pairing + IR at test time, and compare to one‑stage methods at similar accuracy.

- Can the authors report results when the MLLM directly predicts HOI triplets in an open-ended manner (i.e., free-form verb–object labels per human–object pair, without the candidate list?

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
4

### Summary
This paper introduces a novel framework for zero-shot Human-Object Interaction (HOI) detection that fully decouples object detection from interaction recognition. It leverages Multi-modal Large Language Models (MLLMs) by formulating interaction recognition as a Visual Question Answering task, enabled by a deterministic generation method for training-free zero-shot inference. To further boost performance and efficiency, the authors propose a Spatial-Aware Pooling module to integrate appearance and spatial cues, and a One-Pass Deterministic Matching strategy that converts generation into efficient feature matching. Extensive experiments demonstrate state-of-the-art performance under various zero-shot settings and strong cross-dataset generalization.

### Strengths
- This paper investigates an interesting question of how to utilize MLLM to perform the HOI detection task.
- The presentation is clear and easy to follow.
- The proposed One-Pass Deterministic Matching is effective and interesting.

### Weaknesses
-  Lack of Justification for the One-Pass Prompt Design. The proposed one-pass method, which appends the same special token <|hoi|> after each candidate interaction in the prompt (Line 286), requires further theoretical and empirical justification. This design raises a fundamental concern regarding the contextual representation of the <|hoi|> tokens within the Transformer architecture. Specifically, the representation of a particular <|hoi|> token (e.g., the one following candidate T_n) is computed through self-attention over the causal context—it can attend to all preceding tokens, including the text of T_1 to T_n and all previous <|hoi|> tokens, but it is inherently blind to the subsequent candidates (T_{n+1} to T_M).
- Potential Instability Due to Candidate Order Sensitivity. Following the above question, the asymmetric contextual representation leads to a critical, unexplored question: Would the model's prediction for a given candidate interaction be sensitive to the order in which the candidates are listed? If the candidate list is permuted, the contextual information available to each <|hoi|> token changes, which may lead to inconsistent similarity scores and final predictions. Such order sensitivity would be a significant drawback for a robust system, as the candidate set is inherently unordered. The authors must empirically evaluate this potential instability, for instance, by measuring performance variance under different random permutations of the candidate list.
- Missing some efficiency comparison with the traditional HOI methods. The ablation study on inference time (Table 4) convincingly shows the efficiency gains from the proposed components. However, to better position the work, a comparison of the overall inference speed against other state-of-the-art HOI methods is necessary, such as BC-HOI and EZ-HOI. Furthermore, please specify the hardware configuration and batch size used for the timing measurements.

### Questions
Please kindly refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper propose a decoupled framework that separates object detection from IR and leverages multi-modal large language models (MLLMs) for training-free zero-shot IR, with a spatial-aware pooling module and a one-pass deterministic matching method to enhance performance and efficiency. Extensive experiments demonstrate that this method achieves superior zero-shot performance, strong cross-dataset generalization, and the flexibility to integrate with any object detectors without retraining.

### Strengths
1. Clear Motivation. The paper identifies the key limitations of existing HOI approaches, including coupling with a specific detector, poor generalization, and coarse-grained VLM features. In response, This paper propose a method that explicitly decouples object detection from interaction recognition, supports flexible plug-and-play integration with any detector, and demonstrates strong generalization capability.

2. In contrast to prior work that relies on CLIP embeddings, this paper introduces a novel deterministic generation strategy to address application bottlenecks of MLLMs in zero-shot HOI tasks. The proposed framework further implements one-pass deterministic matching, which substantially improves computational efficiency.

3. Experiments show that the proposed method surpasses other baselines under zero-shot setting, training-free setting and cross-dataset setting. Ablation studies confirm the effectiveness of each individual component in the framework.

### Weaknesses
1. Limited Novelty. The proposed method appears relatively simple and heavily relys on the capabilities of open-vocabulary detectors and MLLMs, which also means it inherits their inherent weaknesses, such as missed detections, redundant detections, incorrect category predictions in detectors, and hallucination issues in MLLMs.

2. The method first defines the candidate interaction list based on the categories of detected objects, thereby reformulating the HOI task into a multi-label classification problem for the MLLM. This design may considerably restrict the applicability of the model in open-world scenarios.

3. While the two datasets are widely used, they are relatively outdated(2015 & 2018). It remains unclear whether the approach can handle certain complex cases (e.g., multi-object, dense crowds situations) or perform robust open interaction recognition. It would be valuable to include experiments or visualizations demonstrating performance on new tasks mentioned in the introduction, such as robotic manipulation or autonomous driving.

4. Experimental results on the SWiG-HOI dataset are absent.

### Questions
See limitations and cons above.

### Soundness
2

### Presentation
3

### Contribution
2
