# A Unified Evaluation Framework for Frozen Visual Models on Forecasting Tasks

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 2, 8

## Abstract
Forecasting future events is a fundamental capability for general-purpose systems that plan or act across different levels of abstraction. Yet, evaluating whether a forecast is “correct” remains challenging due to the inherent uncertainty of the future. We propose a unified evaluation framework for assessing the forecasting capabilities of frozen vision backbones across diverse tasks and abstraction levels. Rather than focusing on single time steps, our framework evaluates entire trajectories and incorporates distributional metrics that better capture the multimodal nature of future outcomes. Given a frozen vision model, we train latent diffusion models to forecast future features directly in its representation space, which are then decoded via lightweight, task-specific readouts. This enables consistent evaluation across a suite of diverse tasks while isolating the forecasting capacity of the backbone itself. We apply our framework to nine diverse vision models, spanning image and video pretraining, contrastive and generative objectives, and with or without language supervision, and evaluate them on four forecasting tasks, from low-level pixel predictions to high-level object motion. We find that forecasting performance
strongly correlates with perceptual quality and that the forecasting abilities of video synthesis models are comparable or exceed those pretrained in masking regimes across all levels of abstraction. However, language supervision does not consistently improve forecasting. Notably, video-pretrained models consistently outperform image-based ones.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel, unified framework to efficiently assess the predictive power of frozen vision backbones across various levels of abstraction, from pixel synthesis to object bounding box tracking. The core method employs a lightweight Diffusion Model trained to forecast future trajectories directly within the frozen model's feature space, circumventing the need for expensive fine-tuning. Critically, the framework moves beyond traditional deterministic errors by emphasizing stochasticity, using distributional metrics like Fréchet Distance to accurately capture the multimodal and uncertain nature of future outcomes. The key findings confirm that video pre-training is essential for superior forecasting and reveal that powerful video synthesis models, such as WALT, possess unexpectedly strong predictive capabilities.

### Strengths
1. I believe the overall framework design is sound, appearing both effective and convenient, thus advancing the methodology for evaluating the quality of video representations.
2. The comparative evaluation of various pre-training strategies (e.g., language supervision, masked modeling) provides clear empirical guidance for building high-performance visual forecasting systems.
3. It proposes a unified evaluation framework capable of assessing diverse forecasting tasks across different levels of abstraction (from pixels and depth maps to point tracks and object bounding boxes) within a single architecture.

### Weaknesses
1. If there are image or depth prediction methods as a comparison (it should be easy to add some CNN-based or diffusion-based solutions from recent years), we can better understand the current level of this evaluation framework.
2. I'm concerned that the lightweight readout head could become a performance bottleneck, and it would be nice to have some experiments to illustrate the architectural choices for the head.

### Questions
* Expect additional experiments on the above weakness
* The framework uses Forecasting Future (FFF) performance to evaluate the quality of a frozen visual model's representation. Beyond the observed correlation, what is the theoretical or empirical justification for FFF being a reliable and essential proxy for assessing the fundamental quality of a visual representation itself?
* The forecasting capability relies on training an additional Diffusion Model in the latent space. How can the authors definitively prove that the measured FFF performance is not primarily bounded by or biased towards this newly trained Diffusion Model, rather than accurately reflecting the intrinsic predictive potential of the frozen backbone's features?
* The paper correctly highlights the "inherent uncertainty of the future." However, the final evaluation still relies on comparing predictions (via FD or Best-of-N) against the Ground Truth (GT) dataset labels. Does this approach truly solve the claimed difficulty of forecasting, or does it merely provide a more scientific way to compare against a known outcome? Have the authors considered or explored GT-independent intrinsic metrics for evaluating future plausibility?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The proposed framework for Frozen Visual Models on Forecasting Tasks
 evaluates entire trajectories and incorporates distributional metrics that better capture 
the multimodal nature of future outcomes. Given a frozen vision model, latent diffusion models are trained 
to forecast future features directly and then decoded via lightweight, task-specific readouts

The framework is evaluated using 9 diverse vision models, spanning image
and video pretraining, contrastive and generative objectives, and with or without
language supervision, over 4 forecasting tasks, from low-level pixel predictions to high-level object motion.

The authors state that language supervision does not consistently improve forecasting. 
Notably, video-pretrained models consistently outperform image-based ones.

### Strengths
Short and compact literature review

Brief discussions on proposed work - split into two  main parts:
LATENT FORECASTING VIA DIFFUSION
&
TASK READOUT HEADS.

Compact illustration of fig. 1 gives a brief idea of the proposed model.

Few tabular results shown.

### Weaknesses
Not  a single equation/expression with analytics presented.

Even if the work is quite intensive, it appears as a technical report for evaluation of the model.

There is no mention of contribution or novelty in the paper.

The overall framework seems to be dependent/derived from :
 - conditional denoising diffusion model Ho et al. (2020)
and
 - the readout heads for tasks as used in Carreira et al. (2024).

The GPU architecture platform used for training/testing is also not mentioned.

Effect of dataset bias during training may be highlighted.

The dark background in few image samples on Fig. 3, makes it difficult to comprehend - what authors want to highlight/exhibit.

### Questions
How does your proposed method handle uncertainties & sharp changes ?

What range of resolution of the frames do you method deal with?

What are the failure cases ?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
1. The authors introduce a method to evaluate frozen vision models on stochastic forecasting tasks by training latent diffusion models to predict future representations which are decoded by task specific readout heads to predict future states at different abstraction levels from points to boxes. 

2. The authors use this proposed evaluation framework to benchmark several popular frozen visual models with different pretraining strategies on their effectiveness at forecasting at different levels of abstraction.

3. Based on this evaluation, the authors present some insights/claims about the frozen vision models.

### Strengths
1. The authors introduce a novel evaluation framework based on diffusion that can benchmark frozen vision models on inherently stochastic forecasting tasks at different levels of abstraction.

2. The authors use distribution evaluation metrics like FID and variance to measure the diversity and realism of the predicted future states.

3. The authors run extensive experiments to benchmark ~10 frozen vision models with different pretraining strategies on 4 types of tasks.

### Weaknesses
1. The analysis done by the authors does not provide any interesting insights:

-  **"Forecasting mostly correlates with perception"** this is not surprising at all, rather it is expected that better perception models will generally also be better at forecasting. Unfortunately, the one somewhat interesting finding here "the best model in perception for a given task is not the best model for forecasting" is not investigated in detail by the authors.
- **"Synthesis models like WALT achieve forecasting performance on par with or better than models trained with mask-based objectives"** This can very likely be attributed to the fact that WALT is a diffusion model, meaning its diffusion representations will likely be better suited for a diffusion based forecasting module. The authors should investigate this in detail, preferably using a non-diffusion based synthesis model. If no such open-source model exists for videos that is comparable in size and scale to other frozen video models, then this ablation can be done for image synthesis models. 
- **" N-WALT does not exhibit the same performance"** It is highly likely that this discrepancy arises out of the fact that the authors perform  a single forward pass to get the intermediate features from a video diffusion model. Diffusion models are supposed to take different noise levels as input, with high noise levels capturing broad semantics and low noise levels capturing high fidelity details. So a single forward pass, (likely done at a fixed low noise level) is going to be better at low level pixel forecasting and worse at semantic box tracking. The authors should try different noise levels and evaluate again.
- **"Language supervision does not result in better forecasting"** It is not accurate to make this claim, since WALT is in itself a text to video model, pre-trained with text supervision. Maybe the authors mean contrastive here.
- **"Video backbones outperform image ones"** All things being equal we expect video models with temporal modelling capabilities to outperform image models with no such capacity at forecasting. The cited paper here, DINO-world, does not support the authors' claim since DINO-world does cross attention on past frame representations to learn the temporal modelling capacity, crucial for future prediction. So in the absence of this capacity in the image models, the temporal modelling is relegated to the latent diffusion module introduced by the authors which is the same for all vision models. This means the video models with inherent temporal modelling ability have the edge over image models in this case. 

2. The authors also do not control for resolution of feature spaces or model params or pre-training data volume in their analysis. But this maybe excused given the difficulty of such a thing with pre-trained models all trained differently. But the authors should also address this concern in detail.

### Questions
see weaknesses

### Soundness
1

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
The paper introduces a unified way to test how frozen vision backbones forecast the future across four abstraction levels—pixels, depth, point tracks, and object boxes—by training lightweight readout heads for perception and a diffusion forecaster that predicts future latent trajectories (4 past → 12 future frames). Evaluation uses both per-example task metrics and dataset-level measures computed in each task’s output space and a variance check. Across nine backbones (image vs. video, contrastive vs. generative, with/without language), the authors find that forecasting generally correlates with perception quality, video-pretrained models outperform image-only, and language supervision offers no consistent gains.

### Strengths
1. Forecasting under uncertainty is central to video understanding. The paper cleanly articulates this gap and proposes a concrete, reusable protocol.
2. Four tasks spanning low→high-level structure offer a broad, apples-to-apples view.
3. The paper shows several insightful empirical findings, such as: forecasting strongly correlates with perceptual quality; language supervision offers little forecasting benefit; synthesis-trained WALT does especially well on pixel/depth FD. They are useful signals for the community.

### Weaknesses
1. Readout heads are trained on observed frames and then applied to forecasted latents at test time. If the forecaster induces a distribution shift in the latent space, readouts might underperform. 
2. The approach is closely related to autoregressive video-generation models [1–3]. Adding a brief discussion situating this work within that line of research would improve clarity.

[1] ACDiT: Interpolating Autoregressive Conditional Modeling and Diffusion Transformer.

[2] Generative Pre-trained Autoregressive Diffusion Transformer.

[3] Self-Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion.

### Questions
1. How sensitive are conclusions to the number of forecast samples per clip? Could some model rankings flip for larger N?
2. Can you share per-model training hours for the forecaster and readouts?

### Soundness
4

### Presentation
3

### Contribution
4
