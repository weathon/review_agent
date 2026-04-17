# RF-DETR: Neural Architecture Search for Real-Time Detection Transformers

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Open-vocabulary detectors achieve impressive performance on COCO, but often fail to generalize to real-world datasets with out-of-distribution classes not typically found in their pre-training. Rather than simply fine-tuning a heavy-weight vision-language model (VLM) for new domains, we introduce RF-DETR, a light-weight specialist detection transformer that discovers accuracy-latency Pareto curves for any target dataset with weight-sharing neural architecture search (NAS). Our approach fine-tunes a pre-trained base network on a target dataset and evaluates thousands of network configurations with different accuracy-latency tradeoffs without re-training. Further, we revisit the "tunable knobs" for NAS to improve the transferability of DETRs to diverse target domains. Notably, RF-DETR significantly improves over prior state-of-the-art real-time methods on COCO and Roboflow100-VL. RF-DETR (nano) achieves 48.0 AP on COCO, beating D-FINE (nano) by 5.3 AP at similar latency, and RF-DETR (2x-large) outperforms GroundingDINO (tiny) by 1.2 AP on Roboflow100-VL while running 20 times as fast. To the best of our knowledge, RF-DETR (2x-large) is the first real-time detector to surpass 60 AP on COCO. Our code is available on GitHub.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces RF-DETR, a lightweight specialist object detector that uses NAS to discover accuracy-latency Pareto curves for target datasets. The key contributions are: (1) a family of NAS-based detection and segmentation models that outperform prior real-time methods on COCO and Roboflow100-VL; (2) exploration of "tunable knobs" in weight-sharing NAS for end-to-end detection, improving transferability; and (3) a standardized latency evaluation protocol to address reproducibility issues.

### Strengths
+ The novelty of this work is acceptable. It applys end-to-end weight-sharing NAS to DETR-based detectors, which has been underexplored. Unlike prior NAS methods focused on image classification or backbones, RF-DETR optimizes full detection pipelines, including segmentation heads.
+ The paper includes extensive experiments on COCO and RF100-VL. Ablations validate design choices, such as backbone replacements and NAS components. Results show consistent improvements.
+ The authors have clarified their motivations. Some figures can help understanding, and the language is precise. Some method details have been provided.
+ This method outperforms some famous baselines like YOLO and D-FINE, and the NAS framework allows customization for diverse hardware.

### Weaknesses
- The generalization beyond COCO and RF100-VL cannot be confirmed.​​ The experiments focus on two benchmarks, but the claim of generalizability to "any target dataset" is not fully validated. Testing on diverse domains would demonstrate broader applicability. The paper notes that hyperparameters may overfit to COCO-like data and more cross-dataset results would alleviate this concern.
- The paper lacks a theoretical analysis of why weight-sharing NAS generalizes well to unseen architectures. For example, it does not provide robustness analysis for the NAS mechanism.
- The authors may further figure out the specific gaps in existing NAS methods for object detection in the introduction. While it mentions overfitting to COCO, it does not thoroughly explain why current NAS approaches fail in detection tasks or how RF-DETR uniquely addresses these issues, especially NAS is not a new concept or tool.

### Questions
- The weight-sharing NAS involves sampling configurations during training. What is the total training time compared to a non-NAS baseline? 
- RF-DETR is positioned as a specialist detector, but how does it compare to fine-tuned VLMs in terms of accuracy-latency trade-offs?
- The buffering method reduces throttling, but did the authors consider other techniques? Please explain why buffering was preferred over alternatives.
- The authors may consider the weakness above and address these key concerns.

### Soundness
3

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
5

### Summary
The paper introduces RF-DETR, a fast, closed-vocabulary DETR that uses weight-sharing NAS to pick the best accuracy–latency setup (tuning resolution, patch size, decoder depth, queries, and windowing) after a single fine-tune—no extra retraining needed. It swaps in a DINOv2 ViT backbone, adds a lightweight mask head (RF-DETR-Seg), and standardizes latency benchmarking (buffering, same FP model, and counting NMS/mask conversion). Results on COCO and Roboflow100-VL show a new real-time Pareto frontier—e.g., the nano model beats D-FINE (nano) by about 5 AP and the medium model is near GroundingDINO (tiny) while roughly 60× faster—suggesting many recent detectors are implicitly over-optimized for COCO.

### Strengths
Originality here is pragmatic: the paper smartly fuses weight-sharing NAS with DETR and a foundation-model backbone, plus a no-nonsense latency protocol—less theory, more removal of real deployment roadblocks. Quality is strong for an engineering paper: careful ablations, honest discussion of TensorRT variance and FP16 pitfalls, and fair apples-to-apples latency (counting NMS/mask steps) bolster credibility. Clarity is good—the “knobs” are concrete, the scheduler-free recipe is easy to replicate, and the deployment story (post-hoc grid search, decoder/query truncation) is clean. Significance is high for practice: it moves the real-time Pareto frontier, cuts the cost of retargeting to new hardware/domains, and exposes COCO-centric benchmarking biases; the results on RF100-VL make the transfer claims believable.

### Weaknesses
1) Gains appear driven mainly by stronger pretraining (DINOv2, O365+SAM2) and a stable recipe, making NAS’s unique contribution unclear.
2) Size taxonomy is misleading: the “nano” model (26.9M params) is far larger than baseline “nano” (3–4M), confounding size vs latency comparisons.
3) The latency protocol (200ms buffering; TensorRT/FlashAttention) stabilizes single-shot timing but can bias rankings and does not reflect sustained throughput.
4) YOLO baselines on RF100-VL are disadvantaged by COCO-tuned thresholds and multi-/single-class NMS mismatches, likely underestimating their performance.
5) Mixed-precision/export inconsistencies (FP32 accuracy vs FP16 latency; modified ONNX export) undermine strict parity across methods.

### Questions
1) Size taxonomy and fairness: Can you clarify the naming (e.g., “nano”) and provide comparisons under fixed latency/parameter/FLOP budgets or a plot at fixed latency caps? 
2) Latency and throughput: Can you report sustained throughput (QPS) under continuous load without the 200 ms buffer, include full pipeline timing (pre/post-processing, NMS/mask conversion), and show sensitivity to buffer length and FlashAttention usage? 
3) YOLO baselines on RF100-VL: Did you use multi-class NMS and dataset-specific threshold tuning consistent with original inference? Can you provide threshold/post-processing sensitivity analysis to verify robustness? 
4) Mixed-precision/export parity: Are all methods evaluated with identical artifacts (same precision, same export path)? Can you disclose the modified export scripts (e.g., ONNX opset 17) and quantify FP16 accuracy changes? 
5) NAS cost and coverage: What are the GPU-hours for training/search, how much of the search space is sampled, and how do “unseen” subnets perform statistically? 
6) Method details: How is “encoder confidence” defined for query dropping, and what criteria or default policy govern decoder truncation at inference?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a weight-sharing neural architecture search space for finding realtime DETR-style object detectors that are pareto-optimal w.r.t. accuracy and runtime. The resulting models achieve state-of-the-art real-time results on the COCO and Roboflow100-VL object detection benchmarks.

### Strengths
The paper is well written and approachable. The search space is elegantly designed and evidently yields state-of-the-art results. The paper makes a set of great points about suboptimal benchmarking practices in prior work and makes an effort to perform fair comparisons.

### Weaknesses
While the paper does provide significant contributions, it could support more ablations and experiments. For example, I would have loved to see a thorough discussion of how much each of the “tunable knobs” contributes to favorable accuracy-runtime tradeoffs, and how much pareto-optimal “knob-settings” vary between datasets. Furthermore, it would be interesting to see whether specific dataset characteristics, like the prevalence of small objects, have an impact on knobs like patch size. I furthermore find it highly interesting that DINOv2 performs much better on “small datasets”, but there are no systematic comparisons between other backbones.

### Questions
* Instance segmentation:  
  * How is the proposed approach different from Mask DINO?  
  * “Our segmentation head bilinearly interpolates the output of the FPN and learns a lightweight projector to generate a pixel embedding map” \- Where is the FPN coming from? I thought this was using a DINO backbone?  
* There are two inconsistent definitions of latencies at which RF-DETR outperforms prior work: “for all latencies” and “all latencies ≤ 40 ms”. Which of these is correct?

### Soundness
4

### Presentation
4

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
The paper proposes a weight-sharing, once-for-all style search over DETR variants (“RF-DETR”) to obtain real-time operating points along an accuracy–latency Pareto curve. A single base model is trained while exposing a compact set of architectural knobs (input resolution, patch size, decoder depth, number of queries, windowed vs. global attention). After training, sub-networks are selected by validation without per-subnet fine-tuning. Experiments on COCO and a broader multi-dataset suite show consistent gains over strong real-time baselines, and an instance-segmentation extension indicates broader utility.

### Strengths
1. A practical blend of weight sharing, once for all selection, and DETR knob tuning that feels deployable, with latency protocols that practitioners can actually follow.
2. Clear empirical gains against strong real time baselines on COCO and on a broader evaluation suite, and the instance segmentation extension indicates the idea transfers beyond detection.
3. A search space whose knobs are easy to grasp, with figures and ablations that make the design choices legible.
4. A simple path to high quality low latency detectors without fine tuning of each subnet, plus useful guidance on latency measurement such as throttling mitigation and artifact consistency.

### Weaknesses
1. The way subnets are sampled during training and the policy or grid used for post training selection are not specified clearly, which hurts reproducibility and interpretation.
2. The paper does not make clear when or how decoder layers and queries are dropped during training, whether losses are reweighted across depths, or how queries are ranked at test time.
3. The contribution reads as incremental relative to once for all and weight sharing NAS in vision, and a tighter comparison to prior NAS for detection and backbones is needed to clarify what is new beyond engineering.
4. Some comparisons mix backbones and pretraining regimes, so stronger parity baselines or controlled reruns would better isolate the benefit of the proposed recipe.
5. It is unclear how a fixed mined subnet transfers to unseen datasets or domains without reselection, and a cross dataset test would strengthen the robustness claim.
6. Per knob sensitivity and stability evidence are limited, for example queries versus AP at fixed FLOPs and the interaction of resolution and patch size, and reporting variance with error bars or minimum median maximum would help.

### Questions
1. How are sub-networks sampled during training (uniform over knobs, FLOPs-aware, or constrained)? Any coupling constraints to avoid pathological combinations?
2. Are layers/queries randomly dropped during training to mimic inference-time truncation? Is there loss re-weighting across depths? How exactly are queries ranked at test time?
3. How many candidates are evaluated during selection, what is the wall-clock/energy budget, and is the same operating point reused across datasets/hardware or re-selected each time?
4. Could you add controlled re-runs (or a table) where backbones and pre-training are aligned across methods to isolate the effect of your approach?
5. Could you report results where a single subnet chosen on COCO is evaluated unchanged on other datasets to assess transfer.
6. Could you provide per-knob sensitivity plots and report variance across random seeds and multiple TensorRT engine builds?
7. Will you release export scripts, calibration settings, the buffering/throttling harness, and the exact list of selected sub-networks?

### Soundness
4

### Presentation
4

### Contribution
3
