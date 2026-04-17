# LikePhys: Evaluating Intuitive Physics Understanding in Video Diffusion Models via Likelihood Preference

- Decision: Accept (Poster)
- Scores: 6, 2, 4, 4

## Abstract
Intuitive physics understanding in video diffusion models plays an essential role in building general-purpose physically plausible world simulators, yet accurately evaluating such capacity remains a challenging task due to the difficulty in disentangling physics correctness from visual appearance in generation. To the end, we introduce LikePhys, a training-free method that evaluates intuitive physics in video diffusion models by distinguishing physically valid and impossible videos using the denoising objective as an ELBO-based likelihood surrogate on a curated dataset of valid-invalid pairs. By testing on our constructed benchmark of twelve scenarios spanning over four physics domains, we show that our evaluation metric, Plausibility Preference Error (PPE), demonstrates strong alignment with human
preference, outperforming state-of-the-art evaluator baselines. We then systematically benchmark intuitive physics understanding in current video diffusion models. Our study further analyses how model design and inference settings affect intuitive physics understanding and highlights domain-specific capacity variations across physical laws. Empirical results show that, despite current models struggling with complex and chaotic dynamics, there is a clear trend of improvement in physics understanding as model capacity and inference settings scale up.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LikePhys, a training-free evaluation framework designed to assess the intuitive physics understanding of video diffusion models (VDMs). Instead of relying on human or vision-language judgments, LikePhys measures how well a model distinguishes physically valid from invalid videos using its denoising loss as a likelihood proxy. The authors construct a controlled benchmark of twelve simulated scenarios across four physics domains—rigid-body, continuum, fluid mechanics, and optical effects—and define the Plausibility Preference Error (PPE) to quantify whether the model assigns higher likelihood to physically plausible sequences.

Through systematic experiments on twelve state-of-the-art VDMs, the study finds that larger Transformer-based models (e.g., Hunyuan T2V, Wan 2.1–14B) outperform UNet-based ones, showing partial emergence of physics reasoning. PPE correlates strongly with human judgments but remains largely independent of visual quality metrics, confirming that it measures physical plausibility rather than appearance. Nonetheless, current models still struggle with complex or chaotic dynamics—particularly in fluid mechanics and conservation-law scenarios—highlighting the need for future work on physics-aware training and longer temporal modeling.

### Strengths
1. The paper proposes LikePhys, a novel and training-free method that evaluates intuitive physics understanding through a model’s own likelihood estimation. This approach elegantly connects diffusion models’ denoising objective with physical plausibility assessment, avoiding dependence on human annotation or vision–language model judges. It offers a fundamentally objective and interpretable evaluation paradigm for generative models.
2. LikePhys can be directly applied to any diffusion-based video model without additional training or fine-tuning. By using the denoising loss as an ELBO-based likelihood surrogate, it remains compatible with a wide range of architectures and inference setups, enabling scalable and reproducible benchmarking across different models.
3. The authors construct a highly systematic simulation dataset of twelve physics scenarios covering four domains—rigid-body, continuum, fluid mechanics, and optical effects. Each valid–invalid pair is carefully designed to isolate specific physics violations (e.g., energy, mass, continuity) while holding appearance constant, ensuring the evaluation reflects genuine physics reasoning rather than visual bias.
4. The proposed Plausibility Preference Error (PPE) metric quantifies the proportion of cases where a model fails to prefer physically valid samples. It is intuitive, numerically stable, and easily comparable across models. Moreover, the authors demonstrate that PPE aligns strongly with human judgments of physical consistency while being independent from standard visual quality metrics, confirming its validity and specificity.

### Weaknesses
1. A key limitation of the proposed likelihood-preference framework lies in its implicit assumption that higher estimated likelihood (i.e., lower denoising loss) reflects stronger physical understanding. In practice, the likelihood assigned by a diffusion model is influenced by many confounding factors beyond physics correctness. For instance, if a video sample—whether physically valid or invalid—resembles patterns frequently seen in the training data, it may naturally obtain a higher likelihood simply due to data distribution similarity, rather than genuine adherence to physical laws. Conversely, both valid and invalid videos that deviate from the model’s training distribution could be assigned uniformly low likelihoods, making the difference between them statistically insignificant. As a result, the LikePhys metric may sometimes conflate distribution familiarity with physical plausibility, leading to inaccurate or unstable evaluations, especially when the model exhibits strong dataset bias. This issue raises questions about the robustness and interpretability of using likelihood differences alone as a proxy for intuitive physics understanding.
2. Although the benchmark covers four major physics domains, it remains synthetic and controlled, relying solely on Blender-rendered simulations. While this ensures experimental rigor, it limits the method’s ability to generalize to real-world, noisy, or unstructured videos, where visual complexity, uncertainty, and imperfect physical consistency are common. The results might therefore overestimate models’ real-world physics reasoning ability.
3. The metric treats intuitive physics understanding as a pairwise likelihood preference problem, which captures surface-level plausibility but may fail to reflect causal reasoning, temporal prediction, or long-horizon dynamics that are essential for deeper physical understanding. As a result, models that memorize motion patterns could perform well on LikePhys without genuinely learning the underlying physical principles.
4. While the paper reports domain-level PPE scores, it provides limited qualitative analysis or visual diagnosis of why certain models fail under specific laws (e.g., temporal continuity or conservation of mass). More detailed case studies or ablation examples could have strengthened interpretability and clarified whether errors stem from architecture limitations, data bias, or diffusion noise modeling.

### Questions
I do not see evaluation code and data in the supplementary material. Do you have any opensource plan?

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
5

### Summary
The problem studied in this paper is how to evaluate a video generation model w.r.t. obedience to physical rules, under the setting that the model weights are accessible. The proposed method is train-free/zero-shot: 1) synthesize benchmark videos with Blender with two sets, $V+$ for physical feasible videos and $V-$ for physical infeasible videos; 2) calculate noise estimation errors for videos in $V+$ and $V-$; 3) calculate how many times the video model would estimate lower errors for videos from $V+$ videos from $V-$, then normalize the score to get the so called "Plausibility Preference Error" score (page 4, line 211~213).  
With the proposed method, authors benchmarked 12 open-sourced video generation models, and compared its correlation with human as well as VLM judgements.

### Strengths
Although the work is still crude in its current form, it provides an insightful view angle for video model evaluation, with its train-free method.

### Weaknesses
The idea of evaluating an image/video generation model's certain capability with its noise estimation error is not new, though this is the first work I met for physical video generation.  Plus there's some essential problems with the proposed method unclear from the paper, which is stated in the following "Questions" section.

### Questions
1. About the mismatch between the assumption and the actual calculation of PPE: in page 1, line 52~53, authors stated the essential assumpation *we use a simulator to render paired videos. In one, we render physically-realistic phenomena, while in the other we introduce 
a controlled violation of physics. We keep the visual appearance consistent in the pair, ensuring that any difference resulting from processing the videos can be attributed solely to the breach of physics principles*, which means the video $x^+$ and $x^-$ should have the same prompt and scene content. But from the eq.5 in page 4 and appendix A in page 14, it seems non-paired videos $x^+_j$ and $x^-_k$ are also included in calculation. Did I get anything wrong or the presented calculation is problematic here?
2. Distribution discrepancy among the benchmark videos and tested models: from the demo frames in Fig.1 and appendix page 20~22, it seems the synthesized video from Blender are over simplified, with non-realistic looking and blank background. This distribution would be definitely far from the trainset of video generation models tested in the paper. Though results in Table 2 show that the method proposed is better than Qwen2.5 VL and VideoPhy1/2, I doubt whether this is fair for models with less training data similar to the synthesized video from the paper. From Table 1 in page 6, it seems the calculated PPE scores fluctuate dramatically among the tested 12 physics categories for any of the 12 models listed there, and I doubt the distribution discrepancy might be the hidden problem.
3. Following question 2, authors didn't specify which one from the Qwen2.5 VL model family is used.

### Soundness
2

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
This paper proposes a very simple approach to eval video diffusion models' physical plausiblity, named Plausibility Preference Error (PPE). PPE is a pair-wise eval metrics, by calculating the error between the loss of denoising physically valid and impossible videos.

### Strengths
1. This eval method is easy to understand and implement, the motivation is clear.
2. The authors provide good analysis on Disentanglement of Visual Appearance, convinced me that PPE is mainly for physical plausiblity, and are merely affected by visual quality.
3. The authors provide good discussions for model size, number of frames, CFG, data scale, which provides useful insights.

### Weaknesses
1. This approach needs to access model params, which makes it impossible for evaluating closed-source video diffusion models. This is also discussed in Section 5
2. All of the scenarios are from simulator, will the models pretrained more on simulated data have advantages over other models? I see some discussion in Section 5, the authors rely on an assumption that the models are mostly trained on real-world recordings rather than
animated or synthetic content. But I doubt this may not hold.
3. One of the most important motivation of this paper is "VLM based methods fail to disentangle physics from visual appearance.", while the authors do not provide a correlation between these scores and visual quality metrics like In Table 3. I felt this is important to justify the necessity of using PPE

### Questions
1. For timestep selection, different models are using different weighting scheme, having differnt density of timestep sampling when doing training, so for the timestep selection, does it makes sense to sample uniformly? or randomly sampled? Is such sampling strategy robust enough?

### Soundness
3

### Presentation
4

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
This paper introduces LikePhys, a training-free, likelihood-preference-based evaluation method designed to assess the intuitive physics understanding of video diffusion models (VDMs). The core motivation is to address the challenge of disentangling physical plausibility from visual appearance in generated videos—an issue that existing evaluation methods often fail to handle due to biases from visual fidelity or subjective judgments. The key contribution of this work is the development of a new evaluation framework along with a comprehensive benchmark.

### Strengths
- This work is the first to utilize diffusion model likelihoods for evaluating intuitive physics. The approach of using the denoising objective as a likelihood surrogate for a "violation-of-expectation" test is both clever and well-justified, as it effectively examines the model's internal representation of physical concepts.

- Experimental results indicate that the proposed Physics Perceptual Evaluation (PPE) aligns more closely with human preferences compared to existing automatic metrics.

- The benchmark is comprehensive, providing a valuable framework for evaluating these models.

### Weaknesses
- The primary concern is that the method's validity depends on a curated set of synthetic simulations. While controlled violations are necessary, this raises questions about how well the findings generalize to the distribution of real-world, natural videos.

- As stated in the paper, this work assumes the learned distribution is physics-plausible, which is actually infeasible for large video diffusion models.

### Questions
- The benchmarks were simulated using Blender. Which physics engine did you use, and what is the duration of each video?

- In Figure 3, the textures appear relatively simple. Have you included more complex textured objects?

- I feel that this approach is somewhat similar to Direct Preference Optimization (DPO). Could the author please elaborate on this point?

- If the model not only trained on realistic videos, how to adapt your work to evaluate this model?

### Soundness
3

### Presentation
3

### Contribution
2
