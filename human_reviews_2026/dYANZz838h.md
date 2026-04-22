# Mitigating Coordinate Prediction Bias from Positional Encoding Failures

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Multimodal large language models (MLLMs) excel at vision–language tasks such as VQA and document understanding, yet precise coordinate prediction remains challenging.
High-resolution inputs exacerbate this difficulty by producing long token sequences that weaken positional encodings and introduce directional  biases in coordinate outputs. 
We investigate this phenomenon by analyzing how MLLMs behave when visual positional encodings (VPEs) are deliberately perturbed through shuffling. 
Our analysis reveals that such perturbations induce predictable, non-random coordinate biases rather than random errors, suggesting that models rely on internal positional priors when spatial grounding signals are degraded. 
Crucially, we observe similar directional error patterns in natural high-resolution datasets, indicating that positional encoding failures are a key bottleneck for accurate coordinate prediction at scale. 
To address this issue, we propose Vision-PE Shuffle Guidance (VPSG), a training-free test-time method that leverages the directional nature of these biases for correction. VPSG runs auxiliary decoding with shuffled VPEs to isolate position-unconditioned tendencies, then uses this as negative evidence to guide digit prediction while preserving coordinate format through a lightweight finite-state machine. 
Experiments on ScreenSpot-Pro demonstrate reliable improvements, highlighting positional encoding robustness as a critical factor for spatial reasoning in MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper improves the coordinate grounding capabilities of VLMs by reducing the impact of models' spurious visual position encoding (VPE) prior while reinforcing the influence of proper position encoding features. Through VPE shuffling experiments, the authors discovered that model exhibits a non random bias when the VPE is disturbed, suggesting that there is likely an underlying positional prior which causes a positional bias when model cannot rely on proper VPE. Therefore, the authors proposed VPSG, a training free decoding based method. VPSG requires models to perform inference multiple times, each with shuffled VPE at a different seed value. The aggregated shuffled VPE result is contrasted with proper VPE to obtain the final digit output during decoding process. Experimental results on Screenspot-Pro benchmark yields higher performance compared to the Qwen2.5VL base model.

### Strengths
Overall, the paper is fluent and the motivation for method design is well established. The authors reveal that randomly shuffling the VPE of a VLM leads to a non random bias, suggesting there is an inherent positional bias of models on top of the positional information obtained from VPE. This bias may lead to incorrect grounding of coordinates, especially when VPE becomes challenged in long context settings. The method design closely follows the discovery, which steers output generation towards VPE guided probability and away from inherent positional bias.

### Weaknesses
The authors disturb the position encoding of VLMs by shuffling operations and empirically prove that this shuffling led to preference of a few favoured coordinates. However, only shuffling operation is tested in this study. There might be other RoPE modification methods, such as NoPE (no position encoding), to find the built-in position bias.

The authors argue that position encoding failures would occur for long context high resolution grounding tasks. However, the base model of choice, Qwen2.5 VL, applies a 3D RoPE which already reduces the overall visual token distances and somewhat prevents long context visual position encodings. The validity of the method design can be further improved by showing evaluation results on VLMs that apply a 1D RoPE which is more prone to long context position encoding failures.

### Questions
1. Could you please provide more detail on how to shuffle PE? Do you shuffle the position index of vision tokens? 
2. Could you provide more detail on how to derive the average pairwise distance as presented in figure 3? 
3. Based on the experiment in figure 1, VPE shuffling should have little impact on the final coordinate grounding result. In that case, what is the accuracy of baseline mode evaluated on SCREENSPOT-PRO when PE is shuffled? Does it have values as compared to normal PE?  
4. Is VPSG applicable to general grounding task? Currently it is only tested on SCREENSPOT-PRO. How does the method perform on other object grounding datasets such as RefCOCO? 
5. Is VPSG applicable to other decoding methods? If so, could you provide some of the experimental results using different decoding strategies? 
6. Since the model would run multiple inference passes, what is the inference speed in comparison to baseline models?

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
3

### Summary
MLLMs do coordinate prediction, but they can get it wrong. By perturbing the visual positional encodings in MLLMs, they find that the coordinates predicted cluster together more, suggesting some pre-existing bias. They then propose an inference-time method to compensate for this bias, and report some improvements on ScreenSpot-Pro.

### Strengths
* The idea of finding and then removing some spurious correlations or systematic error to make coordinate prediction better is interesting. The model is likely to have _some_ sort of directional biases introduced during the training process, and it seems useful to try to identify what those biases are.

### Weaknesses
* For _each_ input image, VPSG requires $(S+1)$ generations. This seems extremely excessive, especially only for marginal improvements in ScreenSpot-Pro. 
* Shuffling may not necessarily remove the $P\rightarrow Y$ causation. In fact, the $(\text{shuffled}\ P)\rightarrow Y$ might explain the biases seen (See questions for further clarification)

### Questions
Questions
* Can I confirm that VPSG does indeed require (S+1) generations for each input image?
* How many seeds are used in the experiments? 
* Could you then elaborate on the efficiency of the method? For example, do you have the GPUs used and hours taken for ScreenSpot-Pro with and without VPSG?
* It seems that it could be possible that the bias measured is introduced by the shuffling and not necessarily the spurious correlations. Take for example the model wants to click on an element with position index I, where 0 ≤ I ≤ 1380 where there are 1380 image tokens. With random shuffling, over the full distribution you would see an overall regression to the mean, which would explain the decrease in average distance as clicks just end up closer to the center of the screenshot. Is this possible?
* A follow-up question: How the clustering manifest when the positional embeddings are shuffled? There probably are clear qualitative patterns if there is a 4x decrease in average distance, what do they look like?
* Do you do VPSG on all token positions, but only on the digit token logits? Or do you only do VPSG on digit token positions? If it's the latter, how do you generally deal with the changing number of digits? 
* For Table 2, for Qwen2.5-VL-3B and 7B with and without VPSG, how many examples go from wrong to correct, and how many examples go from correct to wrong?
* When the model's coordinate prediction is off by a lot, like in the qualitative example, how close is the average shuffled PE prediction to the wrong prediction? If you can show that there is a strong correlation between the shuffled PE prediction and wrong prediction, I think that would be strong evidence that the model falls back on the spurious correlations when there isn't a strong signal on where to click.

### Soundness
1

### Presentation
2

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
The paper studies a very concrete mode of multimodal LLMs (MLLMs) when they are asked to output precise 2D coordinates (e.g., on GUI). The authors observe that when visual positional encodings (VPEs) are perturbed (by shuffling), the model’s coordinate predictions don’t become random — instead they collapse to a few directionally biased points. From this, the paper proposes Vision-PE Shuffle Guidance (VPSG), a training-free, test-time procedure: run the base model once with normal PEs (position-conditioned route), run several auxiliary decodings with shuffled PEs (position-unconditioned routes), aggregate the shuffled routes in log space to estimate the “bias prior,” and then subtract this “negative evidence” only on digit tokens.

### Strengths
* The method is designed as a plug-in at inference, which is attractive given how expensive it is to re-train large MLLMs. It is somewhat like classifier-free guidance (CFG) in diffusion: you contrast a conditioned path with an unconditioned path and push the output toward the conditioned one.

### Weaknesses
* The paper says: at high resolution, positional encodings degrade, therefore the behavior becomes similar to explicit PE shuffling. I can sort of buy that qualitatively, but the connection is currently more narrative than demonstrated. High-res → long context → attention diffusion → weaker spatial cues is plausible, but it’s not exactly the same as “we literally shuffled tokens.” I would like to see a clearer quantitative bridge.
* I am from 3D vision. I don’t fully see why “predict GUI coordinates better” is significant.
* α = 0.55 and decay = 0.4 come from a grid search. That’s okay for a paper, but I don’t know if these transfer across resolutions and models.
* Finite-state machine part is a bit under-motivated, although I understand what it does (protect formatting, apply guidance only on digits)
* On Qwen2.5-VL-7B the gain is +0.6 average, which is real but small.
* “shuttled” → “shuffled” in Fig. 4 (I assume it’s a typo).

### Questions
* You have X (input), P (positional encodings), S (spurious correlations), Y (coordinates). In some causal formulations I learned, something like “directional numerical prior” or “collapsed coordinate cluster” would be a separate intermediate variable, not folded into S. Could S be split into “S: dataset-induced digit priors” and “B: bias induced by PE degradation”?
* Have you tried VPSG on tasks that require richer spatial grounding — for example, anything camera/pose-like or 3D-aware? I am thinking of tasks similar in spirit to “Cameras as Rays: Pose Estimation via Ray Diffusion,” “Matrix3D: Large photogrammetry model all-in-one,” or “RayZer: A Self-supervised Large View Synthesis Model,” where the positional structure is not just 2D screen coordinates but full 3D / rays / camera parameters.

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
2

### Summary
Multimodal large language models (MLLMs) excel at vision–language tasks such as VQA and document understanding, yet precise coordinate prediction remains challenging. High-resolution inputs exacerbate this difficulty by producing long token sequences that weaken positional encodings and introduce directional biases in coordinate outputs. We investigate this phenomenon by analyzing how MLLMs behave when visual positional encodings (VPEs) are deliberately perturbed through shuffling. Our analysis reveals that such perturbations induce predictable, non-random coordinate biases rather than random errors, suggesting that models rely on internal positional priors when spatial grounding signals are degraded. Crucially, we observe similar directional error patterns in natural high-resolution datasets, indicating that positional encoding failures are a key bottleneck for accurate coordinate prediction at scale. To address this issue, we propose Vision-PE Shuffle Guidance (VPSG), a training-free test-time method that leverages the directional nature of these biases for correction. VPSG runs auxiliary decoding with shuffled VPEs to isolate position-unconditioned tendencies, then uses this as negative evidence to guide digit prediction while preserving coordinate format through a lightweight finite-state machine. Experiments on ScreenSpot-Pro demonstrate reliable improvements, highlighting positional encoding robustness as a critical factor for spatial reasoning in MLLMs.

### Strengths
1) The paper is well written.
2) The paper organization is good.
3) The experiments show the effectiveness of the proposed method.

### Weaknesses
1) Challenges with High-Resolution Inputs: High-resolution images produce long token sequences that weaken positional encodings, making precise coordinate prediction inherently difficult.

2) Dependence on Positional Encoding Quality: The reliance on effective positional encodings means that any degradation in these signals can lead to significant performance drops, highlighting a critical vulnerability.

3) Complexity in Error Patterns: Understanding and correcting for predictable biases may require additional analysis and tuning, complicating the model's deployment in real-world applications.

### Questions
1) What strategies can be implemented to mitigate the impact of long token sequences on positional encoding effectiveness?
2) What are the implications of positional encoding vulnerabilities for the reliability of MLLMs in critical applications, such as autonomous navigation or medical imaging?
3) Are there alternative methods for encoding positional information that could enhance performance with high-resolution inputs?

### Soundness
3

### Presentation
3

### Contribution
2
