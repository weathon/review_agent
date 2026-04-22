# PixelThink: Towards Efficient Chain-of-Pixel Reasoning

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
Existing reasoning segmentation approaches typically fine-tune multimodal large language models (MLLMs) using image-text pairs and corresponding mask labels. However, they exhibit limited generalization to out-of-distribution scenarios without an explicit reasoning process. Although recent efforts leverage reinforcement learning through group relative policy optimization (GRPO) to enhance reasoning ability, they often suffer from overthinking and produce uniformly verbose reasoning chains irrespective of task complexity. This results in elevated computational costs and limited control over reasoning quality. To address this problem, we propose PixelThink, a simple yet effective scheme that integrates externally estimated task difficulty and internally measured model uncertainty to regulate reasoning generation within a reinforcement learning paradigm.  The model learns to compress reasoning length in accordance with scene complexity and predictive confidence. To support comprehensive evaluation, we introduce ReasonSeg-Diff, an extended benchmark with annotated reasoning references and difficulty scores, along with a suite of metrics designed to assess segmentation accuracy, reasoning quality, and efficiency jointly. Experimental results demonstrate that the proposed approach not only improves segmentation performance
but also significantly reduces inference latency by 30.4%, cutting token usage by 48.2%. Our work contributes novel perspectives towards efficient and interpretable multimodal understanding. The code and model will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the problem of "overthinking" in MLLMs for reasoning segmentation, where models tend to generate verbose and expensive reasoning chains, regardless of task difficulty. The authors propose a method that regulates the length of these reasoning chains using reinforcement learning (GRPO). It has a new reward function that incorporates a "soft length penalty." This penalty is adaptively determined by two signals: (1) Externally-estimated task difficulty (based on scene complexity, segmentation challenge, and linguistic ambiguity, all scored by a large MLLM). (2) Internally-measured model uncertainty (based on the probability gap of the model's own token predictions).

This allows the model to learn to produce concise reasoning for simple tasks while permitting more elaborate reasoning for complex and uncertain cases.

To evaluate this new problem of efficient reasoning, the authors propose "ReasonSeg-DIFF benchmark"  annotated with task difficulty scores and dual-mode (short and long) reference reasoning chains. New Metrics (RST, SAT, URSS) are introduced to jointly evaluate segmentation accuracy, reasoning quality, and computational efficiency (token usage).

Experiments demonstrate that the method reduces reasoning token usage (by ~47% on the test set) while simultaneously improving overall segmentation accuracy (+7.8 gIoU) compared to the Seg-Zero baseline.

### Strengths
- The paper addresses the interesting problem of computational efficiency in reasoning VLMs.

- The paper delivers a new method (PixelThink), a new benchmark (ReasonSeg-DIFF), and new metrics (RST, SAT, URSS). The proposed methods are intuitive and convincing.

- The paper includes a strong set of ablations (Table 4) that validate the components of the reward (Difficulty + Uncertainty). There are also many experiments on the hyperparameters reported in the appendix (e.g. Uncertainty Weight, Length Constraint for Medium Samples, etc.)

### Weaknesses
- Based on Table 2, RScore is lower after using the proposed method. I noticed the explanation in the Appendix about this. Is there a better way to evaluate reasoning quality if the authors think RScore is not suitable here?

- More analysis and ablation study of RL algorithm itself is needed. For example, what is the influence of soft penalty weight β?

### Questions
- Could we conduct a more detailed qualitative analysis of samples with varying difficulty levels to better assess the method’s robustness and stability?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes PIXELTHINK, a multimodal CoT method based on Seg-Zero to regulate token output based on the external task difficulty and internal model uncertainty which are used to regulate the GRPO reward. The key contributions are:
- The PIXELTHINK method which uses an adaptive policy reward to control reasoning length.
- The ReasionSeg-DIFF benchmark, augmented with task difficulty annotations and efficiency metrics.
- Demonstrated performance gains while reducing token count.

### Strengths
- The method achieves around 48% reduction in output tokens while getting better scores than Seg-Zero.
- The main concept of using an adaptive policy reward to regulate the budget is an interesting approach to resource allocation.
- There are some important ablations such as comparing uncertainty and difficulty that support the method's motivation.
- The paper includes the prompts and implementation settings as well as detailed qualitative examples and failure cases.

### Weaknesses
- The Task Difficulty Score relies on an external large model (Qwen2.5-VL-72B) which creates a high upfront cost and transfers the bias of that model in the method.
- The method and dataset introduce a lot of hyper-parameters which can be a concern.  Given the strong dependency on the final policy and metric ranking, how where ($\tau_1=5.0, \tau_2=3.5, γ = 0.7$) selected? As the number of parameters increases it becomes a bigger concern using the same dataset for ablation and testing since bias can be transferred.
- How sensitive is the method to the Seg-Zero setup such as the MLLM (Qwen2.5-VL-7B)? Is this method generalizable across different settings? Showing such a property would increase the impact of the work.

### Questions
- The paper has detailed failure cases but the authors should provide more details, given the ablation of difficulty vs uncertainty, what happens and when these values disagree could this be used to improve the interpretability of the model?
- The paper should include a direct measurement of inference speedup, not just token reduction, to gauge the practical impact more accurately.
- How does this method compare to naive token length reduction?
- How does the token length distribution look like? Does the model have a specific range or preference?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper targets the overthinking problem in reasoning-based MLLM segmentation, where models generate uniformly long chains that raise token cost without solid gains. 
It aims to regulate reasoning length by combining externally estimated task difficulty inside a GRPO fine-tuning framework. After finetuning, the model can output results more efficiently with shorter reasoning. 
The work introduces a new benchmark named ReasonSeg-DIFF with difficulty annotations and paired short/long reference reasonings, and proposes new metrics to jointly assess accuracy, reasoning quality, and efficiency. 
The system uses Qwen2.5-VL-7B for reasoning and SAM2for segmentation, trained with GRPO. 
Experiments show substantial token reductions with improvements in gIoU and unified efficiency scores over strong baselines on ReasonSeg-DIFF and other referring-segmentation benchmarks.

### Strengths
This paper proposes methods for the model how long to think for each example to address the efficiency for MLLM segemtation with reasoning. A soft penalty for GRPO is adopted and the final target is that easy cases get short chains and hard cases can use more steps.
The new dataset marks samples as different levels easy/medium/hard and provides reference chains. Metrics judge both accuracy and cost (model size and tokens).

### Weaknesses
Relative to Seg-Zero, the contribution is largely an added length-aware penalty integrated into GRPO rather than a fundamentally improvemen or training paradigm; the methodological advance is therefore limited in scope. Though the numbers of the token show the improve ment on the efficiency, the performance improvements over baselines are limited. 

The study does not test alternative MLLM backbones like LLaVA or InternVL, so robustness and generalization across model families remain limited. 

It may be not easy to understand the core difference between the short- and long- thinking generation from Fig 3. For Fig 2., the second icon MLLM doesn't contain a sign of training or not. The figure should add explaination about the snowflake and the fire icons. 

While token savings are reported, the work omits end-to-end direct measurements of practical cost, for example, latency, throughput, and FLOPs savings, which makes the real deployment gains difficult to assess. The reported savings are only a few dozen reasoning tokens; in typical referring-segmentation pipelines, other latency contains image I/O, visual feature extraction, and SAM2 mask generation rather than pure LLM decoding. It is unclear that this token drop yields meaningful speedups which is the core contribution of this paper.

### Questions
Could the authors please provide end-to-end efficiency measurements and decompose the gains to show how much of the speedup actually comes from fewer reasoning tokens?
It may be better to report results with alternative MLLM backbones and also refine the figures.

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
4

### Summary
This paper proposes PixelThink, a new method for efficient chain-of-pixel reasoning in multimodal segmentation. Existing reasoning segmentation models often overthink, they generate long reasoning chains even for simple tasks, wasting computation. The authors design a reinforcement learning framework based on GRPO that regulates reasoning length using two signals: external task difficulty (estimated by a large MLLM) and internal model uncertainty (measured from token-level confidence). 

PixelThink introduces a soft length penalty in the reward function, encouraging concise reasoning when the task is easy or confident, and allowing longer reasoning when the scene is complex or uncertain. To evaluate, the authors also build ReasonSeg-DIFF, a new benchmark with difficulty scores, short/long reasoning references, and several new metrics: RST (Reasoning Score per Token), SAT (Segmentation Accuracy per Token) and URSS (Unified Reasoning Segmentation Score). 

Experiments on ReasonSeg-DIFF and standard datasets (RefCOCO/+/g) show that PixelThink reduces reasoning tokens by about 40–50%, while improving segmentation accuracy (about +2 IoU) and reasoning efficiency. Ablation studies verify that both difficulty and uncertainty contribute positively, and that concise reasoning generally yields better masks.

### Strengths
1. Comprehensive new benchmark. The proposed ReasonSeg-DIFF dataset is a valuable contribution: it includes difficulty annotations, short/long reasoning references, and multiple evaluation metrics. This benchmark will likely benefit future work on interpretable reasoning segmentation. 

2. Strong empirical results and ablations. PixelThink shows consistent improvement over baselines such as Seg-Zero across various datasets and difficulty levels.

### Weaknesses
1. [Missing inference speed comparisons].
While the paper reports token counts as a proxy for efficiency, it lacks direct comparisons of *inference latency* or *throughput* under equal hardware settings. Since reasoning models often incur additional decoding overhead even with fewer tokens, reporting real wall-clock speed or FLOPs would better demonstrate the claimed efficiency gains.

2. [Limitations of the soft token-length penalty design]. 
The proposed soft length penalty effectively reduces token usage, but it also relies on several heuristic and manually tuned hyperparameters, such as thresholds $(\tau_1,\tau_2)$, coefficients $(L_{\text{base}}, L_{\text{low}}, \alpha, \beta)$, and difficulty calibration, which may not generalize across datasets or models.

3. [The estimation of task difficulty (from another MLLM) and model uncertainty (from token-level probability gaps) can be noisy or misaligned, leading to unstable or sub-optimal length control].
   The penalty function itself is linear:
   $s(L_{\text{used}}, L_{\text{budget}}) = 1 - \beta (L_{\text{used}} - L_{\text{budget}}),$ assuming a constant marginal cost per additional token. In reality, the cost–benefit curve of reasoning is non-linear, a few extra tokens may add necessary reasoning steps, whereas very long chains become redundant. A linear penalty thus may be too lenient early and too strict later, limiting optimal adaptation.

### Questions
Why the segmentation performance can be improved? Based on my understanding, the model learns to compress reasoning length in accordance with scene complexity and predictive confidence. But why this setting can lead to segmentation performance gains is unclear to me. Some more explanations on the reason behind the gains will be appreciated.

### Soundness
2

### Presentation
2

### Contribution
2
