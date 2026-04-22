# Color Blindness Test Images as Seen by Large Vision-Language Models

- Avg Score: 2.00
- Decision: Reject
- Scores: 0, 2, 2, 4

## Abstract
Large vision-language models (LVLMs) are fairly powerful in understanding this colorful world, yet their reasoning is grounded in highly entangled semantics, leaving open the question of whether they genuinely perceive colors in human-like manners.
Although they could correctly answer questions related to colors, they might internally rely on specific prior knowledge and correlations between color and other semantics instead of directly process the color semantic.
To this end, we study how LVLMs perceive color blindness test images (CBTIs), and we conclude that CBTIs as seen by LVLMs are different from CBTIs as seen by humans.
Specifically, in this paper, we create IshiharaColorBench following the Ishihara test, where LVLMs have to directly process colors, and the digit in any test image could be recognized if and only if LVLMs genuinely perceive colors.
We perform two types of tests: standard color blindness tests for performance assessment and controlled color sensitivity tests for behavior analysis.
Given the former tests, LVLMs perform close to random guessing, and neither scaling-up nor fine-tuning leads to generalizable improvement; given the latter tests, we find several systematic biases, such as the red preference and the sensitivity to saturation contrast but not brightness contrast.
Our findings reveal notable limitations of existing LVLMs in genuine color perception, thereby highlighting the need for developing novel model architectures or training strategies toward a smarter and more human-aligned perceptual foundation of LVLMs.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper introduces IshiharaColorBench, a benchmark designed to evaluate Large Vision-Language Models (LVLMs) using traditional Ishihara plates. The benchmark consists of around 7,000 images representing digits from 0 to 999, with colors systematically altered along multiple dimensions. Using this dataset, the authors conduct a comprehensive evaluation of various VLMs. The results show that LVLMs exhibit significantly weaker color perception compared to humans, and even fine-tuning fails to generalize well. This is attributed to models relying on semantic associations rather than genuine color understanding. Furthermore, the Controlled Color Sensitivity Tests reveal clear non-human-like biases, including a particular weakness in perceiving green tones and an over-reliance on saturation contrast.

### Strengths
- The paper conducts a comprehensive evaluation across a wide range of LVLMs

### Weaknesses
- The colour-blind test setting is already included as a subset of ColorBench (Liang et al.), so the novelty of this benchmark is somewhat limited.
- It is unclear whether the proposed setup truly isolates color perception from semantic understanding. To disentangle semantic and perceptual factors, one might expect the model to first correctly identify digits in the Number Only case with digits shown in black (which might be the normal case in general), and only then be tested under color variations. Without such control, it is still difficult to claim that the results purely reflect color sensitivity rather than semantic cues.
- The paper does not clearly specify how the questions are formatted. For instance, whether the task is multiple-choice or open-ended, and how exactly the input prompt is constructed for LVLMs.
- Some of the analytical findings, such as the weakness in perceiving green tones and the non-continuous color representation space, have already been discussed in prior work, notably in [A] Hyeon-Woo et al., “VLM’s Eye Examination: Instruct and Inspect Visual Competency of Vision-Language Models.” Hence, the paper’s analytical novelty is somewhat limited.

### Questions
- The paper shows that high performance on general tasks does not necessarily translate to strong performance on colour-blind tests. In contrast, would improving performance on IshiharaColorBench in turn enhance general visual understanding or lead to tangible downstream benefits? The practical motivation and implications of optimizing for this benchmark are therefore somewhat ambiguous.
- The results indicate that simple fine-tuning does not generalize well to unseen color variations. This raises the question of how color-blind performance could be improved. The paper would have been stronger with some discussion on potential future directions.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates whether large vision-language models (LVLMs) genuinely perceive colors like humans or merely rely on semantic correlations. The authors introduce IshiharaColorBench, inspired by human color blindness tests, to directly assess LVLMs’ true color perception. They design two evaluations: standard color blindness tests for performance measurement and controlled color sensitivity tests for behavioral analysis. Results show that LVLMs perform near random guessing and exhibit systematic biases in hue and saturation perception, revealing major limitations in their genuine color understanding and highlighting the need for more perceptually grounded model designs.

### Strengths
- The authors introduce IshiharaColorBench, inspired by human color blindness tests, to directly assess LVLMs’ true color perception

- Based on two settings, the authors investigate the (in)ability of LVLMs' color perception capability

### Weaknesses
- I completely agree that evaluating the true visual perception capability of LVLMs is essential. However, in my opinion, this paper lacks novelty and interest, as there are already two existing works [1,2] that explore similar perception settings. Therefore, even if the idea itself is not novel, the authors should clearly describe how their benchmark differs from these prior ones. Although Lines 071–079 do provide some comparisons with existing benchmarks, the distinction should be described more explicitly in comparison with [1,2].

- I highly appreciate the large number of experiments conducted — it is indeed impressive. However, since the results appear quite random, I understand that identifying consistent patterns and conducting in-depth analysis must have been challenging for the authors. Nevertheless, rather than stopping at a superficial interpretation such as “there is no scaling law” or “latest models perform poorly,” the paper should include a deeper analysis or discussion on why such phenomena occur. For example, one possible explanation could be that the latest models, during post-training, have overemphasized reasoning datasets, which might have led to catastrophic forgetting of perception capabilities.

- The paper’s readability could be improved. For instance, essential details such as dataset construction and evaluation metrics — even in brief form — should be included in the main text. Currently, since the dataset construction process is only found in the appendix, it was difficult to follow. Additionally, the font size in Figure 5 is too small.

---

References:

[1] HueManity: Probing Fine-Grained Visual Perception in MLLMs

[2] ColorBlindnessEval: Can Vision-Language Models Pass Color Blindness Tests?

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates whether Large Vision-Language Models (LVLMs) genuinely perceive color in a human-like way or merely rely on semantic associations between color and object identity. 
The authors introduce IshiharaColorBench, a benchmark inspired by the medical Ishihara test, explicitly designed to disentangle true color perception from semantic priors. It includes two types of evaluations: 1) Standard Color Blindness Tests to assess performance, and 2) Controlled Color Sensitivity Tests to analyze perceptual biases along hue (H), saturation (S), and value (V) dimensions. 

Experimental results across state-of-the-art LVLMs (GPT-4o, Gemini 2.0 Flash, LLaVA, Qwen2.5-VL, InternVL) reveal that even the best models perform near chance levels on color-based recognition tasks that humans solve perfectly. Scaling and fine-tuning yield little generalization. 
Controlled analyses uncover systematic non-humanlike biases. 
The paper concludes that current LVLMs lack genuine, low-level color perception and instead rely on semantic shortcuts.

### Strengths
- Proposes the benchmark explicitly isolating chromatic perception from semantics in LVLMs

- Respective analyses are interesting and may be helpful to report in the community.

- Comprehensive experimental design, testing across 7,000 procedurally generated images and multiple LVLM architectures.

- Diverse experiments, including LoRA fine-tuning, linear probing, and HSV-controlled sensitivity analyses, which provide quantitative and qualitative insights into perceptual failure modes.

- Reproducibility seems to be well supported: full procedural generation algorithms and evaluation details are provided (Algorithms 1–5, Appendices A–B).

### Weaknesses
- Missing important reference: The authors missed the critical citations [C1-3] that are closely related to the submission. In particular, [C1] proposed a very similar benchmark based on Ishihara test.

- Missing statistical significance tests: formal statistical tests or confidence intervals are absent, which would make the findings and conclusions more rigorous.

- Potential confounding factors in image complexity: Though IshiharaColorBench isolates color semantics, the procedural textures might introduce visual noise affecting model OCR components; disentangling this effect could clarify whether failures stem purely from color perception or from pattern segmentation challenges.

- Overly aligning the human test and the machine vision test: The way of human percieve the color and that of the machine are obviously different. This raises the question, why the Ishihara test should be particularly used for the purpose in this work.

- Line 045 "common-sense expection": This review disagrees about this statement. It is unrealistic to expect VLMs to behave exactly like humans as a common-sense baseline. The paper should clarify what level of human-like performance is reasonable to expect and why.

- Line 176 states "the Ishihara plates as a scientifically validated paradigm," but this test has only been validated for color blindness detection. It cannot be generalized to assess VLM color accuracy as targeted in this paper.

- Weak motivation: The overall motivation needs to be strengthened, particularly regarding why color naming is a critical problem. While the authors mention several important cases (Lines 101-102), these cover only a very narrow scope. In fact, most real-world visual perception tasks can be adequately solved with prototypical color perception abilities limited to a few basic categories such as "reddish," "bluish," "greenish," "white," and "black," except for the few counterexamples the authors cited.

- The paper's motivation, scope, and practical impact regarding why color naming is important remain weak. The work would have been better aligned with its analyses if it had presented clear applications, such as computational perceptual quality assessment in color and colorimetry domains, similar to recent "VLM as judge" approaches.

- Restricted exploration of detail causes and remedies: The paper diagnoses the failure convincingly but offers limited discussion or experiments on first identifying causes (where does the issue come from? visual encoder, LLM decoder, training datset, or training procedure?) and the possible architectural or training modifications that could mitigate the issue.

- This reviewer expected the paper to identify which components of VLMs constitute the bottleneck for color naming performance, but such analysis is absent. While the current analyses are well-executed, the paper would be significantly strengthened by including investigations that pinpoint specific bottlenecks in the VLM architecture or processing pipeline.

- Line 107 appears to be an overclaim. Color bias issues and hallucinations have already been documented in prior work (e.g., [C3-5]), so these are not entirely new findings or illusions that were previously unknown.

### Questions
- Lines 071-079 discuss VLMs' genuine understanding of color, but the proposed benchmark and analysis methods do not align well with this goal. Specifically, is the Ishihara test truly capable of assessing true color understanding? The generated data contains luminance variations, preventing consistent representation of absolute colors. Moreover, the original Ishihara test evaluates not only color hue but also the ability to perceive relative color differences. Therefore, the Ishihara test alone cannot adequately assess the absolute color awareness that the authors intend to measure. Additional absolute color measurement assessment methods should be employed in parallel to fully support the paper's argument.


- Section 4.2: Green insensitivity has also been mentioned in [C3]. Although the experimental approach differs, what is the distinction between the findings in that paper and those presented in this section?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies colorblindness of vision-language models via Ishihara tests, finding that vision-language models perform close to random at identifying numbers in the tests. Additional analysis shows that models have imbalanced hue perception and are sensitive to saturation but not brightness.

### Strengths
* The evaluation setup seems relatively comprehensive, and a large number of models are evaluated.

### Weaknesses
* I'm not sure it's correct to conclude that the failures of models on ViewablebyAll plates are due to their inabilities to process color, especially when many of the models perform poorly at NumberOnly examples too. This could also be a failure of instruction following, or due to the out-of-distribution nature of the numbers being made up of dots. It would be appropriate to also evaluate instruction-following ability for naming numbers in images, for example by creating a version of the dataset without dots (but solid colors only), or a version where dots form letters (rather than numbers) or simple shapes, to see if models perform significantly better on these tests. 
* In general, adapting tests designed for humans is not necessarily appropriate as evaluations for neural models; a human taking part in this task will understand the task fully (allowing us to conclude with high certainty whether they are colorblind or not), especially in the context of seeing several samples (including controls) in the same test session. When evaluating models, it's very hard to distinguish between failures due to the phenomenon of interest, and failures due to the overall setup of evaluation. For example, one would expect a colorblind human to consistently fail to identify certain colors across many different tasks. Do the findings about particular regions of weakness (hues, brightness) generalize to tasks beyond the Ishihara tasks?
* Additionally, it would be good to evaluate not only the argmax answer/guess from the model, but also the probability they assign to the correct answer (following prior work on evaluating models via probability distributions, e.g. Hu and Levy 2023: https://arxiv.org/abs/2305.13264)
* Details on linear probing should be included in the main paper when results are included in the main paper.
* It would be really useful to show downstream applications or tasks where failures of these models in color perception cause downstream errors. E.g., do the regions of weakness of models correlate to failures in tasks implicitly involving color perception, like illusion detection (https://arxiv.org/abs/2412.06184) or simple visual question answering on synthetic images (like CLEVR)?

### Questions
* Why do you think finetuning on colorblind images results in near-zero performance on all types of images? (Figure 3)
* Can you add error bars around the results in Figure 5? Currently, it's not clear how significant the differences are across different hues.

### Soundness
2

### Presentation
2

### Contribution
2
