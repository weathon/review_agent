# How Well Does GPT-4o Understand Vision? Evaluating Multimodal Foundation Models on Standard Computer Vision Tasks

- Decision: Accept (Poster)
- Scores: 8, 8, 2, 6

## Abstract
Multimodal foundation models, such as GPT-4o, have recently made remarkable progress, but it is not clear where exactly these models stand in terms of understanding vision. In this paper, we benchmark the performance of popular multimodal foundation models (GPT-4o, o4-mini, Gemini 1.5 Pro and Gemini 2.0 Flash, Claude 3.5 Sonnet, Qwen2-VL, Llama 3.2) on standard computer vision tasks (semantic segmentation, object detection, image classification, depth and surface normal prediction) and using established datasets (e.g., COCO, ImageNet and its variants, etc).

The main challenges to performing this are: 1) most models are trained to output text and cannot natively express versatile domains, such as segments or 3D geometry, and 2) many leading models are proprietary and accessible only at an API level, i.e., there is no weight access to adapt them. We address these challenges by translating standard vision tasks into equivalent text-promptable and API-compatible tasks via prompt chaining to create a standardized benchmarking framework.

We observe that 1) the models are not close to the state-of-the-art specialist models at any tasks, and 2) they perform semantic tasks notably better than geometric ones. However, 3) they are respectable generalists; this is remarkable as they are presumably trained on primarily image-text-based tasks. 4) While the prompt-chaining techniques affect performance, better models exhibit less sensitivity to prompt variations. 5) GPT-4o performs the best among non-reasoning models, securing the top position in 4 out of 6 tasks and 6) reasoning models, e.g. o3, show improvements in geometric tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper benchmarks VLMs on standard computer vision tasks (semantic segmentation, object detection, image classification, depth and surface normal prediction) using established datasets. It developed a novel method to translate vision tasks into text-promptable, API-compatible formats via prompt chaining. It draws a few interesting findings: VLMs are not close to the state-of-the-art specialist
models at any tasks; GPT-4o performs the best among non-reasoning models, securing the top position in 4 out of 6 tasks.

### Strengths
1. The authors develop a very clever way to prompt the VLMs to perform dense prediction tasks by using superpixels. They also find a better way to prompt object detection through iterative cropping and splitting the image into grids. This new technique creates the new possibility of systematically benchmarking the spatial and semantical understanding capability of VLMs, and the results are interesting.
2. The paper is well-written. The experiments are extensive, covering 6 fundamental vision tasks and the state-of-the-art VLMs, both closed and open-sourced ones.

### Weaknesses
1. Some of the specialist models are outdated. This could make the readers think that the gaps between VLMs and SOTA models are smaller than they actually are. Examples: the authors should compare with SAM2 in semantic segmentation (Tab 4); compare with Lotus [1] and Moge2 [2] for depth and normal map estimation.
2. For depth and normal map estimation, it is not clear how many random pairs are needed for a good convergence of the optimization algorithm. The results of depth and normal in Fig 3 also look very coarse, probably due to the granularity of the superpixels. It would be interesting to see if higher quality depth/normal map will emerge if we use more superpixels and/or more pairs.

[1] Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction, ICLR 2025

[2] MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details, Arxiv 2025

### Questions
N/A

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a comprehensive benchmark evaluating the visual understanding capabilities of state-of-the-art multimodal foundation models (MFMs), including GPT-4o, Gemini 1.5/2.0, Claude 3.5 Sonnet, Qwen2-VL, and Llama 3.2, on a suite of standard computer vision tasks: classification, object detection, semantic segmentation, grouping, depth estimation, and surface normal prediction. Recognizing the challenge that most MFMs are text-based and do not natively output dense or structured visual predictions, the authors introduce a systematic “prompt chaining” framework that decomposes vision tasks into API-compatible sub-tasks. The results demonstrate that while MFMs are decent generalists—especially on semantic tasks like classification and segmentation—they lag substantially behind state-of-the-art specialist vision models, and struggle most with geometric (3D) tasks. The paper includes both quantitative and qualitative analyses, explores prompt sensitivity, and openly acknowledges the limitations of the approach.

### Strengths
**1) Thorough and Timely Benchmarking:** The paper provides one of the most exhaustive and systematic evaluations to date of leading MFMs' visual understanding, moving well beyond the customary VQA or captioning settings to address a broader spectrum of fundamental computer vision tasks.

**2) Task Translation via Prompt Chaining:** The “prompt chaining” framework is thoughtfully designed, enabling apples-to-apples comparisons between text-based MFMs and vision specialists through structured sub-task decomposition. Figure 2 effectively visualizes the design for converting complex tasks like depth and segmentation into textual queries, making the framework transparent and reproducible.

**3) Comprehensive Model and Task Coverage:** The analysis involves a diverse set of MFMs (both closed- and open-weight) across six core vision tasks and multiple datasets, situating the current state of MFMs within established benchmarks (COCO, ImageNet, Hypersim, etc.), including corruption and robustness variants. Table 1 and Table 3–6 substantiate this comprehensiveness.

**4) Control Baselines and Calibration:** The empirical results are carefully calibrated with various baselines—(a) top specialist models, (b) specialists subjected to the same chaining and superpixel constraints, (c) oracle variants, and (d) blind guess. This tightens the attribution of observed deficits to either model limitations or task translation artifacts.

### Weaknesses
**1) Prompt Chaining Overhead and Realism:** The proposed evaluation relies on decomposing tasks into a large number of textual API calls, which is computationally expensive (as noted in App. I). While the paper claims this is for benchmarking only, the translation introduces additional sources of potential error (task granularity, superpixel boundaries, etc.), and the extent to which these reflect real “model” limitations is not exhaustively disentangled. For instance, in Figure 2 and related descriptions, there is acknowledgment that chaining is not optimal—yet the effect of granularity and design choices is mostly probed by coarse ablations, potentially underestimating the ceiling performance of some MFMs.

**2) Limited Exploration of Advanced Prompting or Visual Tools:** There is a missed opportunity to rigorously evaluate more advanced or recently proposed visual prompt engineering techniques (e.g., interactive alignment, visual rulers, or interactive markers for geometric tasks, as briefly mentioned in App. E.1, Figure 15) across all models as a systematic solution for object localization or dense prediction. The decision to use superpixel-based batch querying is pragmatic but may limit the apparent ability of models on fine-structured tasks.

**3) Insufficient Error Analysis on Geometric Tasks:** While the paper establishes that MFMs fare significantly worse on geometric tasks (Tables 5 and 6; Figure 3), there is minimal deep dive into why—for example, what kinds of 3D/normal ambiguities (left-right, scale, out-of-plane rotation) are most problematic, or whether cues are absent due to model design, pretraining data, or prompt interface. Figure 3 and Table 6 show low/negative correlations for some directions, but the causal factors or failure modes are not dissected in detail.

### Questions
Please see weaknesses.

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
5

### Summary
This paper benchmarks multimodal foundation models for standard CV tasks. They proposed an interesting way to avoid using API calls for promptable computer vision tasks. The whole evaluation tasks covers 2 categories, semantic and geometrical understanding, and there are 6 sub-tasks in total. GPT-4o achieves an overall best performance, but is still lagging far behind CV models. For geometric tasks, under the promtable format, all the MFM failed to work.

### Strengths
- The promptable task design itself is interesting and is effective for semantic understanding tasks. 
- This paper is well written and easy to follow
- Experimental evaluations are comprehensive.

### Weaknesses
The biggest problem is, the pipeline of the geometric understanding tasks (depth and surface normal) does not work at all for MFMs:
- In Figure 3, neither depth nor surface normal works as expected, making it not surprising that all the models lagged far behind CV models. 
- The root cause of the mismatch is the ranking algorithm. It is impossible to get the numeric results for depth and surface normal purely from ranks. 
- Without these two geometric tasks, all 4 tasks are basically semantic understanding tasks, with the performance of all the compared models highly correlated. 

Other minor weaknesses: 
- The observations did not provide key findings, unless the ranking. Some interesting points may be: why the reasoning models work worse on semantic understanding? 
- The metrics for surface normal and depth are not the most popular ones. 
- Results with GPT-4o image generation is confusing and did not fit the whole story.

### Questions
- How will random superpixelizations/grid seeds affect the performance?
- Could other CV tasks also be evaluated in the same manner, such as pose estimation, edge detection?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a comprehensive benchmark evaluating multimodal foundation models (MFMs) on standard computer vision tasks including classification, object detection, semantic segmentation, grouping, depth estimation, and surface normal prediction. The key contribution is a prompt chaining framework that decomposes complex vision tasks into text-solvable sub-tasks, enabling API-level evaluation of closed-source models. The study evaluates GPT-4o, o4-mini, Gemini 2.0 Flash, Gemini 1.5 Pro, Claude 3.5 Sonnet, Qwen2-VL, and Llama 3.2 on established datasets. Main findings show that: (1) MFMs lag significantly behind specialist models on all tasks, (2) they perform better on semantic tasks than geometric ones, (3) GPT-4o performs best among non-reasoning models, and (4) reasoning models (o1, o3, o4-mini) show promising improvements on geometric tasks.

### Strengths
- Extensive Model Coverage: The evaluation includes a diverse set of models (7 main MFMs plus reasoning models), providing a thorough landscape of current multimodal model capabilities across both open and closed-source systems.

- Multiple Task Evaluation: The breadth of tasks evaluated (6 core vision tasks) spanning semantic to geometric understanding provides valuable insights into where MFMs excel and struggle.

### Weaknesses
- Potential Data Contamination: While the authors conduct in-the-wild evaluations to address this, the use of standard benchmarks (ImageNet, COCO) raises concerns about training data leakage for closed-source models, which could inflate performance estimates.

- Limited Analysis of Failure Modes: While the paper shows that MFMs struggle with geometric tasks, there is limited investigation into why they fail (e.g., lack of 3D training data, architectural limitations, reasoning deficits). The "blurry vision" hypothesis is mentioned but not systematically explored.

- Task Selection Bias: The choice of tasks favors dense prediction problems that can be decomposed into classification. Other important vision capabilities (e.g., visual reasoning, fine-grained recognition, video understanding) are not evaluated.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
