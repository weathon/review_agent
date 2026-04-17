# Fantastic Tractor-Dogs and How Not to Find Them With Open-Vocabulary Detectors

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Open-Vocabulary Detectors (OVDs) excel in zero-shot benchmarks, but we observe a critical flaw in real-world deployment: a high rate of confident false positive predictions on images that do not contain any target objects (e.g., detecting a tractor in an image of a dog). This issue is masked by standard benchmarks like COCO and LVIS, as they rarely contain images without any of the target classes present. We identify vision-language fusion layers in early-fusion OVD architectures (e.g., Grounding DINO or LLMDet) as the root cause, and show how they distribute irrelevant class information across image features when no prompted object is present. To mitigate background false positives without costly retraining, we propose a simple, training-free method: appending attention sink tokens to the input prompt. We show that such sinks can redirect spurious attention and dramatically reduce background false positives. Our approach significantly improves the performance of all six early-fusion models tested (e.g., boosting AP on LVIS by more than 5x at a false positive rate of 0.01 for some models), making them practical for real-world applications where images without the object of interest are much more prevalent.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper highlights a critical limitation in Open-Vocabulary Object Detectors (OVDs): they often generate confident false positives on images that contain no relevant objects (for example, detecting a tractor in a photo of a dog).
This issue is overlooked by standard benchmarks such as COCO and LVIS, which rarely include background-only images.
The authors show that early-fusion models (e.g., GLIP, Grounding DINO) are particularly vulnerable to this problem due to information leakage in cross-modal attention, whereas late-interaction models (e.g., CLIP, OWL-ViT) are more robust.
To address this, the paper introduces a simple, training-free method that appends attention sink tokens to prompts, effectively reducing background activations and false detections without additional training.
Experimental results demonstrate consistent improvements across multiple OVD architectures.

### Strengths
- Clear identification of the background false positive problem ($\text{FPR}_\text{bg}$) in early-fusion OVD models.

- Strong quantitative analysis using LVIS negative labels and adapted POPE benchmark.

- Simple, training-free solution (attention sink tokens) effectively reduces $\text{FPR}_\text{bg}$ and restores $\text{AP}$.

- Consistent improvements across multiple models with minimal implementation cost.

### Weaknesses
- Ablation limited to initialization and count; lacks analysis of token placement or interaction.
- No runtime or efficiency analysis for large-scale inference.
- Explanation mostly empirical without theoretical grounding.

### Questions
- Do attention sinks affect true positive recall when target classes are present?
- What is the computational overhead in real deployment?
- Can the method generalize to domain-shifted or web-scale datasets?

### Soundness
3

### Presentation
3

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
This paper investigates a critical issue in OVD—the tendency of early-fusion multimodal models to produce high-confidence false positives on background images (termed background hallucinations). The authors show that conventional benchmarks like COCO and LVIS rarely include pure background images, thereby masking this pathology. Through a comparative study across early-fusion (e.g., GLIP, Grounding DINO, LLMDet) and late-interaction (e.g., OWL-ViT, DetCLIP) architectures, the paper finds that early-fusion models exhibit strikingly higher false-positive rates when presented with images containing no objects of interest. To mitigate this, the authors propose a training-free “attention sink” mechanism: augmenting the text prompt with semantically neutral sink tokens that attract residual attention and absorb irrelevant activations. Extensive experiments across six state-of-the-art OVD models demonstrate that this simple modification drastically reduces background false positives (e.g., up to 5× AP improvement at FPR=0.01 on LVIS). The proposed method requires no retraining, minimal code changes, and is compatible with existing model checkpoints.

### Strengths
- Novel perspective: The paper introduces “background-only” testing to reveal systematic bias in OVD robustness.

- Effective and simple method: The attention sink approach is easy to implement and achieves strong performance improvements.

- Strong empirical validation: Results across six early-fusion models and multiple datasets show good generality.

- Reproducible and transparent: The authors provide code, metrics, and data splits for easy verification.

### Weaknesses
- Lack of theoretical grounding: The paper provides no formal model or proof explaining how attention mass redistribution through sink tokens reduces false positives.

- Missing comparison with alternative methods: Only architectural baselines are considered; other hallucination mitigation strategies are not evaluated.

- Limited dataset diversity: Experiments rely on curated datasets, leaving uncertainty about performance under domain shifts or in real-world, unlabeled settings.

- Model-specific tuning required: The optimal number, initialization, and placement of sink tokens vary across models, making the approach not fully plug-and-play.

### Questions
Have the authors considered hybrid approaches combining sink tokens with confidence calibration or background prompts?

### Soundness
4

### Presentation
4

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
The paper studies zero-shot open-vocabulary detection results of foundational detectors when fed with counterfactual prompts, i.e., the prompt contain objects that do not exist in the test image. The paper finds that early-fusion detectors (i.e., fusing features of visual and textual input) output incorrect detections in front of counterfactual prompts, while late-fusion detectors are quite robust. The paper argues that early-fusion detectors are still useful in other tasks such as VQA and hence focus on early-fusion detectors including GroundingDINO and LLMDet. The paper refers to the literature and proposes to add non-learned sink tokens to improve the early-fusion detectors. The paper uses popular datasets such as COCO and LVIS to validate the proposed approach.

### Strengths
Below are notable strengths of this paper.
- The paper studies foundational detectors' performance with counterfactual prompts. This is an interesting topic.
- The solution by adding non-learned sink tokens is simple.
- The choice of using negative annotation in LVIS images in experiments is interesting.

### Weaknesses
Below are notable weaknesses of this paper.

- The motivational example in Figure 1 is not convincing. The reviewer took this image from Figure 2 and tested GroundingDINO: when using the "tractor" prompt, GroundingDINO actually did not output anything, which should be deemed a good thing. Yet, Figure 1 shows a rather high-confidence detection (78.5) on "tractor", contradicting what the reviewer observed. The paper is expected to provide more examples. Further, the paper should provide more details; for example, the rsolution and aspect ratio of the three images are different in Figure 1. Do these factors affect the detection results?

- Although the paper does not mention the word "counterfactual", there are previous works on counterfactual textual grounding, counterfactual VQA, counterfactual visual reasoning for segmentation, etc. These works share a common point as the reviewed paper that they all analyze models' predictions for counterfactual input. However, the paper does not note these related works.

- The solution by adding sink tokens are not clear. One reason is that the paper does not provide important details to understand how it works and why it can work. A diagram can help a lot. Another reason is that the paper seems to apply the sink token technique which has published by previous works. This makes the technical contribution limited. 

- The visual demonstrations are quite limited. Figure 1 only provides one motivational example with three images, which are expected to be the same but actually differ in resolution and aspect ratio. Figure 5 does not show texts clearly. Figure 3 is confusing -- isn't it a good thing to obtain high LVIS MiniVal AP with decreased AP FPR (with fixed threshold 0.05)?

- The code snippet is hard to read. It is not pseudo code. It contains too much coding details and notations without comments.

### Questions
The reviewer asks the authors to address each point in weaknesses listed above and does not repeat them in this Questions box.

### Soundness
2

### Presentation
2

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
This paper shows a flaw in real-world scenarios for OVD, frequently make high-confidence false positive predictions on images that do not contain the target object. For instance, prompting an OVD for "tractor" on an image of a dog often results in a confident "tractor" detection. This issue is masked by standard benchmarks like COCO and LVIS, which rarely contain images completely lacking the target classes.The authors identify that this issue specifically affects early-fusion architectures, where vision and language features are combined early via cross-attention. Late-interaction models (e.g., CLIP) do not suffer this significant deviation in performance on background images. To mitigate this without costly retraining, the authors propose a simple, training-free method, appending "attention sink" tokens to the input prompt to act as "none of the above" type tokens. The authors evaluated this method on six early-fusion models using modified LVIS and POPE benchmarks.

### Strengths
The paper identifies a major issue in real-world deployments of Open-Vocabulary Detectors (OVDs), the false positives were previous unknown.. It also explains why this issue has been missed, noting that standard benchmarks like COCO and LVIS don't show the problem because the images rarely don't contain the target.

The analysis is good. They demonstrate that these layers cannot "pick zero tokens" when a match is absent, forcing them to attend to irrelevant tokens. They also clearly differentiate this behavior from late-interaction models, which do not suffer from this significant deviation on background images.

The proposed solution, appending attention sink tokens to the prompt is nice, and training-free and works out-of-the-box for all evaluated models. The implementation is highlighted as very straightforward, requiring only a few lines of code.

The attention sink approach reduces background false positives, and the results are strong.

The paper proposes simple adaptations to existing benchmarks to quantify this issue, such as leveraging negative labels in LVIS to calculate a specific background false positive rate.

### Weaknesses
1. Limited Exploration of "Attention Sink" Design Space and Generalizability. While the proposed "attention sink" method is elegant and effective, the paper's exploration of its design space is limited, there could be more options explored and ablations done. However, it does work, so this is not a major weakness.

2. The authors use random initializations, mean embeddings, and special characters. Do these different types of sinks perform equally across different model architectures (e.g., are Grounding DINO's cross-attention layers equally receptive to a random vector vs. a mean embedding)?

3. The paper does not thoroughly explore the impact of the number of sinks. Is one sink token sufficient, or is there a diminishing return after a certain number? This is critical for practical deployment, as more tokens increase inference time.

4. How does the token length of the class prompt (e.g., "a single dog" vs. "a fantastic black-and-white tractor-dog running across a field") interact with the optimal number or type of sink tokens?

5. The core goal is to reduce False Positives (e.g., detecting a tractor on a dog), which is a True Negative case (no tractor is present). While the results show gains in AP/FPR on background images, the paper lacks a detailed analysis of potential collateral damage to True Positives (detecting the dog when the dog is actually present).

6. The mechanism works by providing an alternative to irrelevant class tokens. A key concern is whether a confident true detection, where the vision features were already weak, is accidentally routed to the sink instead of the correct class. A sensitivity analysis showing how the confidence of marginal correct detections changes would strengthen the conclusion.

7. The paper claims a training-free solution, but the conceptual problem of "null-class" or "background" modeling is not new in object detection. For instance, how does the sink mechanism relate to the concept of a "background class" token used in traditional detectors (like a final output class)? While the OVD context is different, clarifying the relationship to this established concept would enhance the paper's grounding in detection literature.

8. The paper states that late-interaction models (like CLIP-based methods) do not exhibit the same catastrophic failure mode. However, this conclusion is based on a limited scope.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
2
