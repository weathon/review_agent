# PANICL: Mitigating Over-Reliance on Single Prompt in Visual In-Context Learning

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Visual In-Context Learning (VICL) uses input-output image pairs, referred to as in-context pairs (or examples), as prompts alongside query images to guide models in performing diverse vision tasks. However, VICL often suffers from over-reliance on a single in-context pair, which can lead to biased and unstable predictions. We introduce PAtch-based $k$-Nearest neighbor visual In-Context Learning (PANICL), a general training-free framework that mitigates this issue by leveraging multiple in-context pairs. PANICL smooths assignment scores across pairs, reducing bias without requiring additional training. Extensive experiments on a variety of tasks, including foreground segmentation, single object detection, colorization, multi-object segmentation, and keypoint detection, demonstrate consistent improvements over strong baselines. Moreover, PANICL exhibits strong robustness to domain shifts, including dataset-level shift (e.g., from COCO to Pascal) and label-space shift (e.g., FSS-1000), and generalizes well to other VICL models such as SegGPT, Painter, and LVM, highlighting its versatility and broad applicability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles the task of Visual In-Context Learning (VICL). VICL often suffers from over-reliance on a single in-context pair. To mitigate this issue, the paper introduces PANICL, a training-free method that smooths token-level assignment scores across multiple retrieved visual examples (prompts) using Jensen–Shannon divergence-based weighting. Finally, the authors test the approach on three downstream tasks and several benchmarks.

### Strengths
The paper tackles an important problem and is fairly well written. The idea is simple and based on the experimental results seems quite effective. It may have broad applications on multiple down-stream tasks.

### Weaknesses
My main concern is related to the novelty which seems a bit minimal. While it's an interesting idea, I feel like the proposed smoothing is a minor change on existing ensemble or multi-example strategies. I also feel that the “Model-agnostic” claim feels a bit exaggerated since the overall performance depends heavily on the architecture and retrieval quality even though technically can be applied to different models. However, as the experiments show, in some cases the performance improvements are small and often within noise margins.

Minor
* Tab 1 - wrong bolded value for m=6, 7 for fold-1

### Questions
Can you please share some more insights on why the performance degrades with increased m? Since the performance peaks with few samples, I believe that choosing those samples is very important so I think the performance might be very tightly coupled with the performance of the retrieval system. So, can you please elaborate on how different retrieval performance affects the proposed method?

### Soundness
3

### Presentation
3

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
PANICL introduces a technically simple but novel inference-time strategy to mitigate over-reliance on single prompts in Visual In-Context Learning.

### Strengths
- While smoothing and retrieval are known techniques, PANICL uniquely applies them at the patch/token-level on decoder logits, using score pooling within the VQGAN codebook space. 

- PANICL requires no fine-tuning, no retraining, and no model changes, yet shows consistent gains across multiple VICL backbones.

- Broad, well-controlled experiments on several tasks, e.g. segmentation, detection, colorisation, etc.

### Weaknesses
- Some components are adapted from broader ML ideas, but their particular application and integration here is new in VICL.

- Lack of theoretical analysis of the method.

- compared to single in-context learning, there is computational overhead from multiple prompt passes.

While the idea is intuitive and shows solid empirical improvements across several VICL backbones and tasks, the methodological contribution is relatively limited, as it largely repurposes known techniques like score averaging and prompt retrieval without substantial technical innovation. The paper is well-executed and practical, but the novelty and depth may not fully meet the bar for a strong ICLR contribution. I find myself between a 4 and a 6, but leaning generous due to the broad applicability and clean results.

### Questions
- Why does PANICL in Table 1 show diminishing returns as the number of prompts m increases from 2 to 7? Shouldn’t more prompts consistently improve performance?

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
3

### Summary
This paper introduces PANICL, a method to fix a key problem in Visual In-Context Learning, or VICL, where models over-rely on a single example prompt, leading to biased and unstable predictions. PANICL is a training-free framework that mitigates this issue by using multiple in-context pairs. For a given query image, it first retrieves several similar example pairs from a support set. It then processes each of these pairs through the VICL model to create a prompt pool of their patch-level assignment scores, which are the model's internal predictions. The query image is also processed to obtain its own initial scores. The core idea is assignment score smoothing, where PANICL finds the k nearest neighbors for the query's scores from the prompt pool using JS divergence and then calculates a weighted average. This smoothing step debiases the query's prediction by incorporating information from multiple examples. The authors show this method consistently improves performance on various tasks like segmentation, detection, and colorization, which is robust to domain shifts and can be generalized to other VICL models such as SegGPT, Painter, and LVM.

### Strengths
1. The paper directly addresses a well-known and critical weakness in VICL. The over-reliance on a single prompt is a key bottleneck, and PANICL's approach of using multiple examples is an intuitive and logical solution.
2. Instead of ensembling intermediate features or ensembling final predictions, PANICL operates at the assignment score level. This allows it to correct biases at a patch-by-patch level before the final prediction is rendered, offering a more granular way to debias the model.
3. A major strength is that PANICL is a "training-free" framework. It does not require any model fine-tuning, making it an efficient, plug-and-play improvement.
4. The method shows consistent and significant performance gains over strong baselines, including both single-example methods and other multi-example methods. The gains are not just in accuracy but also in robustness to domain shifts, a critical factor for real-world deployment.

### Weaknesses
1. The method's effectiveness is highly dependent on the quality of the retrieved in-context pairs. The authors admit this in their failure case analysis (Section E, Figure 12), showing that when the retriever provides examples that are globally similar but semantically misaligned (e.g., different object scale or position), PANICL's performance can degrade significantly, sometimes even below the single-prompt baseline.
2. The method appears to be heuristic and parameter sensitive, which may require careful tuning for new tasks or datasets.
3. While it improves SegGPT, the gain on foreground segmentation is minor. Similarly, for keypoint detection with Painter, the gain is minimal.
4. Although the method is intriguing, its novelty is limited. I feel this paper is leaning toward a strong engineering method instead of proposing a new framework.

### Questions
1. Your method requires m additional forward passes to build the prompt pool. Have you explored cheaper alternatives, such as using a static pool of generic assignment scores rather than a dynamic pool retrieved for every query?
2. You show that patch-level k-NN (Table 8, "PANICL") is better than retrieving neighbors from all patches (Table 8, "All-patch"). This observation is interesting, does forcing the model to find neighbors only in the same spatial location (e.g., top-left) prevent it from getting confused by patches from other parts of the image?
3. You use JS Divergence to compare assignment score distributions. This works for token-based models like MAE-VQGAN and LVM. But for pixel-space models like SegGPT and Painter, you switch to l2 distance on intermediate features. Does this mean your core "assignment score smoothing" idea doesn't apply to these models, and you are instead performing a type of "feature ensemble," similar to the baseline?

I am willing to reconsider my evaluation.

### Soundness
3

### Presentation
3

### Contribution
2
