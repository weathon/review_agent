# FastVMT: Eliminating Redundancy in Video Motion Transfer

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Video motion transfer aims to synthesize videos by generating visual content according to a text prompt while transferring the motion pattern observed in a reference video. Recent methods predominantly use the Diffusion Transformer (DiT) architecture. To achieve satisfactory runtime, several methods attempt to accelerate the computations in the DiT, but fail to address structural sources of inefficiency. In this work, we identify and remove two types of computational redundancy in earlier work: **motion redundancy** arises because the generic DiT architecture does not reflect the fact that frame-to-frame motion is small and smooth; **gradient redundancy** occurs if one ignores that gradients change slowly along the diffusion trajectory. To mitigate motion redundancy, we mask the corresponding attention layers to a local neighborhood such that interaction weights are not computed unnecessarily distant image regions. To exploit gradient redundancy, we design an optimization scheme that reuses gradients from previous diffusion steps and skips unwarranted gradient computations. On average, FastVMT achieves a 3.43× speedup without degrading the visual fidelity or the temporal consistency of the generated videos.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces FastVMT, a training-free video motion transfer framework aiming at enhancing computational efficiency without sacrificing video quality or motion fidelity. To achieve this, FastVMT includes two key innovations: 1) sliding-window motion extraction which only computes attention with local spatial neighborhoods to better capture motion correspondences while eliminating unnecessary token interactions; 2) step-skipping gradient optimization that reuses gradients across optimization steps as the authors found that gradients change slowly along the diffusion trajectory. Experiments show that FastVMT yield an average 3.43× speedup and up to 14.9× lower latency compared to prior state-of-the-art methods (e.g., MotionDirector, DiTFlow, MOFT) without visible quality degradation.

### Strengths
- The proposed method is elegant and well-motivated. For example, both the ideas of sliding window motion extraction and step-skipping gradient optimization are lightweight but sounds effective and should work smoothly with existing video diffusion models. To introduce these two modules, the authors provide an insightful analysis (i.e, Figure 2), showing that motion is locally consistent and the gradient in consecutive steps are mostly similar.
- The efficiency improvement is significantly. Specifically, the authors demonstrate 3.43× speedup and up to 14.9× lower latency compared to the existing models, which are impressive.
- The authors provide comprehensive comparisons with the up to 7 existing models (including both training-free and finetuning-based baselines).
- The authors also provide extensive ablation studies, which clearly show the necessity of each proposed component. The results look promising.
- The paper is well written and easy to follow. The figures are well-plotted and informative which can make readers quickly understand the core ideas.

### Weaknesses
My first two concerns circle around the usage of sliding window attention:

- First, the usage of sliding-window attention leads to irregular and non-contiguous tiling (i.e., the attention mask is no longer a uniform square but contains irregular zero-paddings to mask out non-local tokens). This irregularity would make the model incompatible with Flash Attention that requires full or casual mask. How did the authors handle this issue? If the model does not use Flash Attention, did the authors notice a speed degradation when switching to other attention functions?
- Second, the Register Tokens paper [1] found that the transformer models tend to learn a few of register tokens which are attached by most of the tokens and used to aggregate and spread global information. However, the proposed sliding-window design restrict the receptive field of a token to its local neighborhood and further block such information exchange mechanism. This could make the model fail to handle the videos with larger dynamics or longer duration where the global information exchange is critical. However, all the video examples provided in the paper and the supplementary material are 5 seconds and only include smooth and slight motion. Could the authors provide the videos with longer sequences and larger motion to evaluate whether the proposed model can handle such cases?

Other concerns:
- The paper is lack of the report of GPU memory consumption. Since the sliding-window attention and AMF modules require storing multiple latent tensors, attention maps, and cached gradients, this may increase GPU memory usage (especially for long or high-resolution videos). However, the authors only provide runtime speedups without the profiling of memory consumption.
- The ablation on the selection of some important hyperparameter selection is also missing. For example, the window size and the gradient skip intervals could largely affect the balance / trade-off between visual quality and runtime. Could the authors also provide such ablation?

[1] "Vision Transformers Need Registers", ICLR 2024

### Questions
- How do you handle the irregular tiling that comes from the sliding window attention mechanism?
- Does the model use Flash Attention or other memory-efficient attention operations? If not, what is the runtime or memory overhead when switching to other attention functions?
- Could the authors provide the video outputs with longer durations or more complex motion (including both object motion and camera movement)?
- Since GPU memory consumption is also an important metric to evaluate the efficiency, could the authors also provide the report of memory usage?
- Could the authors include an ablation study for key hyperparameters, such as the window size, stride, temporal span, and gradient skip interval?

### Soundness
3

### Presentation
4

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
This paper addresses the inference speed bottleneck of Video Motion Transfer (VMT) that uses training-free Attention Motion Flow. Specifically, it employs,

(1) Sliding-window motion extraction, which assumes that motion correspondences between frames are local

(2) Step-skipping gradient optimization (reusing cached gradients), based on the observation that gradients across consecutive inner optimization steps are highly similar

The paper claims their proposed method achieves an average 3.43× speed-up without compromising visual fidelity or temporal consistency.

### Strengths
- The paper addresses a practical problem such as inference speed improvement in training-free motion transfer
- The proposed method also maintains (or even improves) the quality of the video
- Clearly identifies and analyzes existing problems such as motion and gradient redundancy through experiments

### Weaknesses
- The sliding-window strategy rests on the assumption that inter-frame motion is local and small. The paper lacks analysis or discussion of performance limitation when this assumption breaks (e.g., very fast and large motions, aggressive camera movements, occlusions).
- In Table 2, FastVMT uses WAN-2.1 as the base model, while other baselines may rely on different backbones. The paper states “fair backbone: WAN-2.1,” but it is unclear whether this means all baselines were re-implemented and re-evaluated on WAN-2.1, or merely that FastVMT used WAN-2.1. If the former, the results are very compelling. Otherwise, the quality advantage might partly stem from the newer backbone.
- The core acceleration idea of step-skipping is a fairly common design principle in other areas, so its novelty is somewhat limited.

### Questions
- In Table 2, are those baselines re-implemented by the authors using the WAN-2.1 backbone, or are these numbers quoted from the original papers (potentially with different backbones)?
- In Table 2, it looks like "Ours" is included under Tuning-Based Methods. Is that intentional?
- When is the window center re-estimated along the diffusion trajectory? Fixed at (t=0), or updated progressively during the first 20% of guided steps?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper focuses on the inefficiency issue of motion transfer in the DiT architecture and proposes a method to improve it without sacrificing quality. Specifically, the authors point out that motion redundancy arises from the neglect of motion smoothness across frames, while gradient redundancy occurs due to ignoring the slow gradient changes along the diffusion trajectory. Accordingly, they propose a sliding-window strategy that operates on downsampled attention maps and a step-skipping gradient computation strategy, which together enhance the efficiency of motion transfer.

### Strengths
1. The paper is well written, clearly motivated, and easy to follow.
2. The proposed efficiency improvement strategies, involving the sliding-window and step-skipping gradient optimization, make sense to me, and the illustration of their rationale in the method section is intuitive.
3. The experimental results demonstrate that the proposed method maintains video quality while achieving notable speedup.

### Weaknesses
1. Evaluation dataset. The authors use 50 videos selected from the DAVIS dataset, which is rather small in scale and may not cover sufficient scene and motion diversity. I notice that benchmarks used in different motion transfer papers vary—perhaps the authors follow the test set of DiTFlow? However, how does the proposed method perform on other test sets used in related works? For example, please refer to Table 1 of DeT.

2. Implementation details. The authors mention that “for fair comparison, they adapt Wan-2.1 as the same backbone.” Is the 14B model or the 1.3B model used? Previous works adopt different backbones such as CogVideoX and Hunyuan. Are the authors reimplementing these methods using the Wan2.1 model? If so, there may be a risk that some methods cannot perform optimally, as they can be sensitive to hyperparameters. Overall, it would be helpful if the authors could provide more details about how each baseline is implemented.

3. It would be beneficial if the authors could include and analyze some typical failure cases of the proposed method.

### Questions
In the ablation study, adding the step-skipping strategy brings improvements in certain metrics such as aesthetics or text–frame similarity. However, I think the operation of reusing previous gradients at specific timesteps is essentially an approximation. Even if it does not cause a performance drop, it theoretically should not lead to improvement. Could the authors provide some explanation for this phenomenon?

### Soundness
3

### Presentation
3

### Contribution
2
