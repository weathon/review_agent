# SemCache:Adaptive Semantic-Aware Caching for Efficient Video Diffusion

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Diffusion models have achieved significant progress in video generation tasks, but slow inference speed remains a major challenge. Existing cache-based acceleration methods for video diffusion have demonstrated considerable improvements in inference speed. An existing efficient caching strategy involves reusing model outputs by estimating and leveraging the fluctuating differences among model outputs across timesteps. However, the strategy relies on extensive calibration sets and neglects the fact that prompt semantic variations affect the variational differences in model outputs. This phenomenon is illustrated by comparing videos generated from different prompts: while "a horse running on the grassland" produces highly dynamic content, "a person reading in a coffee shop" results in relatively static scenes.  Building on this observation, we propose a novel training-free SemCache method that can adaptively adjust caching strategies by perceiving prompt semantics changes. Key innovations include a Prompt Semantic-Aware (PSA) caching that evaluates prompt semantics and then dynamically decides a caching strategy tailored to the current timestep based on semantic information. We further introduce a Temporal Motion Metric (TMM) scheme to guide the compute allocation along the temporal dimension based on motion information, which not only ensures motion consistency in videos but also further reduces inference time. Experimental results demonstrate that SemCache achieves 2.45× and 2.66× speedups on HunyuanVideo and Wan2.1 respectively, while maintaining high video quality. Our code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SemCache, a training-free semantic-aware caching framework that accelerates video diffusion inference by adapting caching strategies to prompt semantics and motion dynamics. It combines Prompt Semantic-Aware (PSA) caching, which estimates scene complexity via cross-attention variations, and Temporal Motion Metric (TMM), which measures latent motion intensity to guide computation allocation.

### Strengths
1. This paper  uses prompt semantics and motion perception to guide caching in diffusion transformers, bridging high-level text understanding and low-level feature reuse.

2. In the exeperiments part, it demonstrates  speedup and high-quality preservation across two large-scale video diffusion backbones and multiple metrics.

### Weaknesses
1.In the experimental section, several important details are missing, and the experiments do not adequately highlight or validate the main motivation of the paper (see detailed comments 1–3).

2.The connection between cross-attention variation and semantic complexity, while intuitive, lacks a rigorous analytical foundation or sensitivity analysis.


Detailed comments:

1. In the experimental section, several details are unclear. For instance, for the FLOPs metric, it is not specified whether the reported values are in terabytes, gigabytes, or another unit. Additionally, key implementation details such as the generated video length and the prompt size are not provided.

2. More importantly, the paper is motivated by the observation that prompts with different semantic complexities can influence caching behavior. However, in the experimental results, the authors do not explicitly evaluate performance(not only limit to latency) under prompts with diverse semantic dynamics (e.g., static vs. dynamic scenes). As a result, it is difficult to verify whether the proposed semantic-aware mechanism truly adapts to varying prompt semantics as claimed.


3. Table 1 is not placed correctly in the paper.

4. In the early example where an LLM is used to analyze prompt semantic scoring, it is unclear how the method determines which cache entries to retain, since all prompts are static. Would the model then keep all cached results at every step?

### Questions
See above

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes to accelerate the inference speed of video diffusion models by addressing a key limitation in existing cache-based methods: they largely neglect how prompt semantic variations (e.g., "a horse running" vs. "a person reading") affect model output differences and, consequently, the optimal caching strategy. To tackle this, the authors introduce SemCache, a novel, training-free method that adaptively adjusts its caching strategy by perceiving prompt semantics. Building on the observation that dynamic scenes exhibit larger output variations than static ones during the diffusion process, SemCache introduces two key components. First, a Prompt Semantic-Aware (PSA) strategy estimates semantic consistency by measuring the magnitude of cross-attention output differences, where higher variation indicates dynamic content and leads to less caching, while lower variation allows for more reuse. Second, a Temporal Motion Metric (TMM) strategy computes motion intensity from inter-frame latent differences to guide compute allocation along the temporal dimension, thereby reducing redundancy while ensuring motion consistency. Experimental results demonstrate that SemCache achieves significant speedups (up to 2.45x on Hunyuan Video and 2.66x on Wan2.1) while maintaining high video quality.

### Strengths
* The Prompt Semantic-Aware (PSA) strategy is a novel and interesting idea, using cross-attention output differences as a lightweight, training-free method to guide model reuse and computation.
* The TMM method is built on the highly intuitive and logical premise that dynamic and static scenes require different computational loads, and it uses this insight to further improve the efficiency of the diffusion process.
* The experimental section validates the reliability of the proposed method by comparing it directly against recent SOTA acceleration methods, including TeaCache and MagCache. The results show that SemCache achieves superior efficiency while maintaining competitive visual quality.

### Weaknesses
* Unclear Methodology Visualization (Figure 3): Figure 3 is ambiguous. The horizontal "Temporal" axis label conflicts with the visual examples ($x_t$, $x_{t-n}$), which represent diffusion timesteps. The diagram also fails to clearly illustrate how PSA guides module reuse or how TMM differentially allocates resources.
* Logical Contradiction in PSA Error Metric: The paper's logic for the error metric Equation 8 is critically flawed. It requires the current output $O_t$ to decide whether to compute $O_t4$, creating a circular dependency. 
The required low-cost, predictive error estimation (implied by $\Sigma\epsilon_t < P_{s_t}$) is not explained.
* Lack of clarity in TMM and PSA integration: The paper fails to clearly describe or illustrate the integration between TMM and PSA. It does not specify how the two systems interact or how TMM’s frame filtering mechanism aligns with PSA’s step filtering, leaving readers to infer the operational relationship on their own.

### Questions
* The TMM's "skip frame" mechanism seems absolute. Was a "smoother" approach considered?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SemCache, a training-free acceleration framework for video diffusion models. Its primary novelty is the introduction of "semantic awareness" to caching. The method is motivated by the insight that static scenes are more compressible than dynamic scenes. SemCache uses a dual strategy: (1) a Prompt Semantic-Aware (PSA) module that uses changes in cross-attention outputs as a lightweight proxy for scene complexity to guide caching across diffusion steps, and (2) a Temporal Motion Metric (TMM) that uses latent frame differences to guide caching across time. Experiments on Hunyuan Video and Wan 2.1 show state-of-the-art speedups (up to 2.66x) while maintaining high visual quality, outperforming other caching baselines.

### Strengths
1. The core idea of linking caching strategy to the prompt's semantic content (dynamic vs. static) is a clear and logical advancement over existing methods that are "blind" to this high-level context.
2. The paper's most clever contribution is the PSA module. The authors first prove that an LLM *could* provide this semantic guidance, but then successfully replace this costly component with a lightweight and intrinsic signal: the magnitude and directional changes in cross-attention outputs. This is a well-designed solution in general.

### Weaknesses
1. The method's robustness is questionable due to key hyperparameters that appear to be model-specific. The weights for the PSA score ($\alpha$ and $\beta$) are set to (0.4, 0.6) for Hunyuan Video but (0.2, 0.8) for Wan 2.1. The paper provides no methodology for how these weights were found, suggesting a process of manual, model-specific tuning that may not generalize easily to new models.
2. The method is only validated on two video diffusion models. While these are large, recent models, competitors (like BWCache in the previous review) were tested on a wider array of five models. This limited scope makes it difficult to assess the true generality of the approach.
3. The PSA score is calculated using cross-attention outputs from "five layers each from shallow, middle, and deep layers". This selection (15 specific layers) seems arbitrary and is not justified. It is unclear how these layers were chosen or how sensitive the method's performance is to this specific selection, which weakens the robustness of the proposed proxy metric.
4. The paper shows that using an LLM works (Figure 2) and that its proxy works (Table 2). However, it never directly compares them. It would be much stronger if it showed that the proxy-generated scores ($p_{s_t}$) are highly correlated with the LLM-generated scores. This would provide direct evidence that the proxy is *actually* capturing "semantic complexity" as claimed.

### Questions
Have the authors evaluated the computational overhead of calculating PSA and TMM scores, especially for real-time generation scenarios?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SemCache, a training-free and semantics-aware caching framework for accelerating video diffusion inference. The key idea is to perceive the semantic and motion dynamics of the input prompt to guide when and what intermediate features to cache. The method introduces two complementary modules: a Prompt Semantic-Aware (PSA) strategy that measures cross-attention input-output variations as a proxy for prompt-driven semantic complexity, and a Temporal Motion Metric (TMM) that quantifies inter-frame latent changes to identify motion-rich segments. Without retraining or calibration, SemCache achieves up to 2.66× inference acceleration on large diffusion transformers (HunyuanVideo, Wan2.1) while maintaining comparable video quality. The authors also validate the PSA and TMM components through ablations and compare against TeaCache and MagCache, showing superior trade-offs between speed and fidelity.

### Strengths
- The paper addresses a timely and important bottleneck: inference latency in video diffusion models. The observation that prompts with different semantic dynamics lead to varying redundancy in computations is intuitive and practically meaningful.

 - Compared to previous caching methods (which often rely purely on frame/step residual magnitudes or fixed schedules), the integration of a prompt semantic score to guide caching decisions is a noteworthy contribution.

 - The reported speedups (~2.45–2.66×) are non-trivial and the authors show evidence of maintaining output quality (VBench, PSNR/SSIM) across different models and scenarios.

 - The paper provides formulae for the PSA score and for the cumulative error threshold, which helps reproducibility and clarity.

### Weaknesses
1. While the paper describes when to reuse or recompute (via PSA + TMM) it does not clearly articulate what specific computations or layers are being cached/reused. For example: is caching happening at the level of transformer blocks, cross-attention layers, or entire time-step outputs? This ambiguity reduces reproducibility and clarity of the mechanism.

2. **Limited theory or validated correlation**: The PSA module uses cross-attention input/output magnitude and direction differences as a proxy for prompt semantic complexity. However the paper lacks deeper empirical or theoretical analysis showing that this proxy correlates strongly with “computational redundancy” across many prompts and models.

3. **Thresholding/hyper-parameters sensitivity**: The decision threshold based on PSA (and the choice of weights $\alpha, \beta$) appear somewhat heuristic. The paper has limited analysis of how sensitive the method is to these hyperparameters or how robust it is across different settings.

### Questions
1. How sensitive is the performance (both speedup and output quality) to the hyperparameters $\alpha, \beta$ and the threshold logic in Eq 8? Did you observe degradation when these change?

2. In very high-motion or highly dynamic scenes (or very long videos), does the caching mechanism still yield significant speedups? Are there failure cases where the overhead of evaluating PSA/TMM outweighs the caching benefit?

### Soundness
3

### Presentation
3

### Contribution
3
