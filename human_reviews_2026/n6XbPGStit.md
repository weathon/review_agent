# Follow-Your-Preference: Towards Preference-Aligned Image Inpainting

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
This paper investigates image inpainting with preference alignment. Instead of introducing a novel method, we go back to basics and revisit fundamental problems in achieving such alignment. We leverage the prominent direct preference optimization approach for alignment training and employ public reward models to construct preference training datasets. Experiments are conducted across nine reward models, two benchmarks, and two baseline models with varying structures and generative algorithms. Our key findings are as follows: (1) Most reward models deliver valid reward scores for constructing preference data, even if some of them are not reliable evaluators. (2) Preference data demonstrates robust trends in both candidate scaling and sample scaling across models and benchmarks. (3) Observable biases in reward models, particularly in brightness, composition, and color scheme, render them susceptible to cause reward hacking. (4) A simple ensemble of these models yields robust and generalizable results by mitigating such biases. Built upon these observations, our alignment models significantly outperform prior models across standard metrics, GPT-4 assessments, and human evaluations, without any changes to model structures or the use of new datasets. We hope our work can set a simple yet solid baseline, pushing this promising frontier. Our code is available at: https://github.com/shenytzzz/Follow-Your-Preference.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies image inpainting with preference alignment by revisiting basic questions rather than proposing a new architecture. Using Direct Preference Optimization, the authors construct preference datasets from nine public reward models and explore (i) which reward models yield useful training signals, (ii) how preference data scale with more candidates and samples, (iii) how reward hacking emerges, and (iv) whether a simple ensemble of reward models mitigates biases. Experiments span two inpainting backbones (BrushNet and FLUX.1 Fill) that cover diffusion and flow matching, evaluated on BrushBench and EditBench. Results show improvements across automatic metrics, GPT‑4 scoring, and human studies, without architectural changes or new datasets.

The paper offers a careful, evidence‑driven empirical study of preference alignment for inpainting, including scaling analyses and bias diagnostics. The ensemble strategy consistently improves or matches the best reward models across backbones and benchmarks. However, the evaluation pipeline relies heavily on GPT‑4 as a “fair” judge with limited validation and reporting details. The ensemble design and computational cost are underspecified. The bias analysis is qualitative mainly and could benefit from quantitative measurements.

### Strengths
- Evaluates nine reward models and two distinct backbones: BrushNet (diffusion) and FLUX.1 Fill (flow matching), across two benchmarks. This breadth supports generalizability claims.
- The paper concisely derives the DPO loss specialized to visual generation, supporting technical soundness.
- The paper documents brightness/composition/color‑scheme biases, e.g., HPSv2 favoring bright, detailed, vivid images; PickScore favoring dimmer, simpler, muted ones, providing novel qualitative insight.

### Weaknesses
- GPT‑4 is positioned as a “fair” evaluator with 86% accuracy vs. volunteers, but rater selection, agreement metrics, and sampling protocol are under‑reported. This affects evaluation validity. The evaluation prompt fixes specific criteria and weights (0–100), which may embed subjective biases. This limits generality. Details such as inter‑rater reliability, demographic diversity, and task difficulty balance are not provided. No direct evidence was found in the manuscript. This reduces statistical rigor.
- Ensemble is defined as average ranking of all reward models with no exploration of weighting schemes or subset selection, limiting technical transparency.
- The paper omits compute/latency costs for scoring with nine reward models and generating 16 candidates per prompt. No direct evidence found in the manuscript. 
- Bias findings (brightness/composition/color) rely on visual inspection without quantitative measures, limiting scientific rigor.

### Questions
- The paper exclusively relies on GPT-4 as the “fair evaluator,” reporting 86% accuracy against volunteer judgments, yet it omits any cross-validation with other Vision-Language Models (VLMs) such as GPT-4o, Gemini, or Qwen2.5-VL, which are widely available and architecturally diverse. 
- The “Ensemble” method is merely the average ranking of all reward models, without exploring learned or heuristic integration schemes, or even VLM-based approaches.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides a comprehensive evaluation of preference alignment with multiple reward models for the task of image inpainting. 
It provides useful insights into how different reward models affect the final performance of the model by studying biases in them that could cause reward hacking, and studying data scaling trends. Finally, it proposes that an ensemble of reward models can produce generalizable results and shows strong quantitative performance.

### Strengths
- The paper provides a thorough analysis of different reward models for image synthesis, such as 1) demonstrating how some reward models are not reliable, while valid reward models can still exhibit biases, 2) scalability of preference data, and 3) analysis on reward hacking,  through a systematic study. These can be useful analyses for the researchers working with preference alignment. 
- It is the first work showing a systematic study of preference alignment with reward models for image synthesis.
- Experiment design is clear and well thought out, and the result is written clearly. 
- This paper demonstrates a strong result by identifying reliable reward models and ensembling them, and this can be a baseline that the community could build upon.

### Weaknesses
The reviewer did not find significant weaknesses of the paper other than:
- Calibrated Multi-Preference Optimization for Aligning Diffusion Models (Lee et al., CVPR 2025) also considers a multi-reward formulation and could be compared against, but this work is only cited very briefly in the related work.

### Questions
It would be great if the rebuttal could include the comparison with the paper mentioned in the weaknesses section or a justification of why it cannot be included.

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
3

### Summary
This paper conducts a systematic and extensive investigation into aligning image inpainting models with human preferences. Instead of proposing a novel architecture, the authors revisit fundamental questions about using preference alignment in this domain. They employ the DPO algorithm, leveraging nine different public, off-the-shelf reward models to automatically construct preference datasets without costly human annotation.

### Strengths
Pros:
1. It provides the community with valuable insights into the behavior of reward models and the dynamics of preference alignment.
2. The analysis and visualization of "reward hacking" (Figure 3) clearly demonstrates a crucial pitfall in naive preference alignment and provides a strong motivation for their proposed solution.
3. The proposed Ensemble method shows that significant gains can be achieved by thoughtfully combining existing tools rather than inventing complex new ones.

### Weaknesses
Cons:
1. The paper states the Ensemble is based on "the average ranking of all the reward models" but could be more explicit. Does this involve normalizing scores before averaging, or is it a direct average of rank positions? A more formal definition or algorithmic description would improve reproducibility. 
2. In your analysis, some reward models (like VQAScore) were found to be unreliable evaluators. However, they seem to be included in the final Ensemble method. Did you test if an ensemble composed of only the most reliable/diverse reward models would perform better or worse than an ensemble of all nine? Is there a benefit to including weaker, but potentially diverse, signals?
3. More details on the Hyper-parameter Sensitivity are desired. The appendix shows searches for the DPO hyper-parameter β and the learning rate. How crucial was fine-tuning these parameters for achieving top performance? Is the Ensemble method more or less sensitive to these hyper-parameters compared to using a single reward model like HPSv2?
4. Does the DPO process primarily learn aesthetic qualities (e.g., better lighting, composition), or does it also improve semantic alignment to the text prompt? The results suggest an aesthetic improvement, but it would be interesting to know if prompt faithfulness also improved systematically.
5. The evaluation benchmarks (BrushBench, EditBench) primarily focus on object replacement, addition, or style modification. The paper does not explore how preference alignment performs on other critical inpainting tasks, such as: Object Removal: Where the primary goal is seamless and plausible background completion, not creative generation. Image Restoration: Such as removing scratches or blemishes, where fidelity and realism are paramount. It's unclear if aligning for "aesthetic preference" would be beneficial or even detrimental in these contexts, where the model should be as unobtrusive as possible.
6. The paper does identify biases (e.g., brightness, complexity). However, it stops short of deeply investigating why these biases exist. This is likely tied to the data on which the reward models were trained (e.g., HPSv2 was trained on a dataset of human-preferred images, which may have skewed towards professionally shot, well-lit photos). A more thorough discussion on the provenance of these biases would strengthen the paper's contribution and provide a clearer path toward creating less biased reward models in the future.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates preference alignment for image inpainting, a relatively underexplored but increasingly important topic.
Rather than proposing a novel architecture, the authors systematically study how to align inpainting models with human aesthetic preferences through Direct Preference Optimization (DPO).

### Strengths
The paper is among the first to systematically analyze preference alignment in image inpainting, bridging the gap between text-to-image alignment studies and conditional image editing.

The experimental design is rigorous: nine reward models, multiple benchmarks, two architectures (diffusion and flow-based), and both automatic (GPT-4) and human evaluations.

The paper is well-written and easy to follow.

### Weaknesses
The paper shows limited algorithmic novelty, as the core approach mainly combines existing DPO techniques with a straightforward averaging ensemble.

The reliance on GPT-4 for evaluation may introduce instability and reproducibility concerns, as model-based judgments can vary over time and lack full transparency.

While the study highlights potential biases related to brightness, color, and composition, it lacks quantitative analysis or formal metrics to substantiate these observations.

### Questions
Did you try other ensemble strategies (e.g., weighted average or learned combination)?

### Soundness
3

### Presentation
3

### Contribution
2
