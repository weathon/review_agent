# HiCache: A Plug-in Scaled-Hermite Upgrade for Taylor-Style Cache-then-Forecast Diffusion Acceleration

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Diffusion models have achieved remarkable success in content generation but suffer from prohibitive computational costs due to iterative sampling. While recent feature caching methods tend to accelerate inference through temporal extrapolation, these methods still suffer from severe quality loss due to the failure in modeling the complex dynamics of feature evolution. To solve this problem, this paper presents HiCache (Hermite Polynomial-based Feature Cache), a training-free acceleration framework that fundamentally improves feature prediction by aligning mathematical tools with empirical properties. Our key insight is that feature derivative approximations in Diffusion Transformers exhibit multivariate Gaussian characteristics, motivating the use of Hermite polynomials, the potentially theoretically optimal basis for Gaussian-correlated processes. Besides, we introduce a dual-scaling mechanism that ensures numerical stability while preserving predictive accuracy, which is also effective when applied standalone to TaylorSeer. Extensive experiments demonstrate HiCache's superiority: achieving \$5.55\times\$ speedup on FLUX.1-dev while exceeding baseline quality, maintaining strong performance across text-to-image, video generation, and super-resolution tasks. Moreover, HiCache can be naturally added to the previous caching methods to enhance their performance, e.g., improving ClusCa from \$0.9480\$ to \$0.9840\$ in terms of image rewards. Our code is included in the supplementary material, and will be released on GitHub.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes HiCache, a training-free acceleration framework for diffusion transformers that addresses the quality loss issue of existing feature caching methods. This paper follows the "Cache-then-Forecast" paradigm. The core insight is that feature derivative approximations in diffusion transformers exhibit multivariate Gaussian characteristics, leading to the adoption of Hermite polynomials—optimal for Gaussian-correlated processes—as the prediction basis instead of the suboptimal monomial basis in Taylor series. HiCache also introduces a dual-scaling mechanism to ensure numerical stability and predictive accuracy.

### Strengths
+ Deep Insight into Trajectory Modeling: The paper identifies Taylor series’ limitation: its monotonic monomial basis fails to capture non-monotonic feature trajectories. In contrast, it links diffusion transformers’ Gaussian feature derivatives to Hermite polynomials and their oscillatory behavior, which accurately models non-monotonic dynamics.

+ Extensive Task Coverage: The paper evaluates HiCache across a broad range of generative tasks to demonstrate its generalizability. Extensive experiments are conducted on text-to-image, text-to-video, class-conditional image generation, and super-resolution tasks, demonstrating its superiority. It maintains strong performance across different acceleration ratios.

+ Effective Acceleration Design: The dual-scaling mechanism not only solves the numerical instability of Hermite polynomials for large extrapolation steps but also works standalone to improve TaylorSeer. HiCache’s plug-and-play nature—requiring only polynomial basis substitution without architectural changes—enables easy integration with existing cache-then-forecast pipelines, enhancing its practical value.

### Weaknesses
- Modest Performance Advancement Over TaylorSeer: While the paper provides solid theoretical analysis for HiCache’s improvement over Taylor series-based methods, the practical performance gain over TaylorSeer is not prominently distinct in both qualitative visualizations and quantitative metrics. More compelling evidence—such or case studies is needed to further validate its superiority.

- Limited Variety of Experimental Models: The experimental evaluations focus on a relatively narrow set of diffusion transformers. For image generation, it primarily uses FLUX, with no tests on other widely adopted models like Stable Diffusion 3 or Qwen-Image. For video generation, only HunyuanVideo is included, without evaluations on other mainstream video diffusion models like Wan. Conducting experiments on these additional models would further strengthen the paper's validity.

### Questions
Please refer to the Weaknesses section.

### Soundness
4

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
5

### Summary
This paper proposes HiCache, a training-free acceleration framework for diffusion transformers that replaces Taylor expansion with scaled Hermite polynomial-based feature caching, achieving more stable and accurate predictions across multiple generative tasks.

### Strengths
1. The paper replaces Taylor’s monomial basis with Hermite polynomials derived from Gaussian feature correlations, leveraging Karhunen–Loeve optimality and a single scaling factor σ to improve stability and accuracy.
2. HiCache preserves almost the same implementation form as TaylorSeer, merely replacing the polynomial basis and adding a few scalar evaluations, thereby allowing direct integration into any feature caching–based acceleration framework with negligible computational overhead.
3. Extensive evaluations on text-to-image, text-to-video, and super-resolution tasks demonstrate the effectiveness of the proposed method in achieving substantial acceleration.

### Weaknesses
1. The paper primarily relies on automated metrics such as PSNR, SSIM, LPIPS, and VBench. Incorporating human preference evaluations would make the assessment more convincing.
2. The paper heavily relies on the scaling factor σ to stabilize predictions, yet it lacks a principled rule or analysis on how to select or adapt σ across architectures, acceleration ratios, or polynomial orders.

### Questions
It would be helpful if the authors could clarify whether HiCache is compatible with sparse- or efficient-attention variants of diffusion transformers.

### Soundness
3

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
The paper introduces HiCache, a training-free acceleration method for Diffusion Transformers (DiTs) that replaces traditional Taylor-series extrapolation with a Hermite polynomial-based feature caching framework. Existing caching methods like TaylorSeer predict future diffusion features using monomial expansions, but they struggle with the non-monotonic and stochastic dynamics of diffusion features, leading to quality degradation at high acceleration ratios. HiCache addresses this by observing that feature derivatives in DiTs follow approximately Gaussian statistics and thus are better represented by scaled Hermite polynomials, which are the optimal orthogonal basis for Gaussian-correlated processes.

HiCache further introduces a dual-scaling mechanism that stabilizes numerical behavior by simultaneously contracting inputs and suppressing high-order coefficients. This enables accurate, stable extrapolation while preserving the plug-in simplicity of Taylor-based predictors. Experiments across text-to-image (FLUX.1-dev), text-to-video (HunyuanVideo), class-conditional (DiT-XL/2), and super-resolution (Inf-DiT) tasks show that HiCache achieves 5–6× acceleration and even slightly improves perceptual quality metrics such as ImageReward. The method integrates seamlessly into existing caching frameworks, offering a mathematically grounded, numerically stable, and empirically robust approach to accelerating diffusion models without retraining or architectural modifications.

### Strengths
The key strengths of the paper lie in its strong theoretical foundation and practical effectiveness. HiCache introduces a principled improvement over Taylor-based caching by recognizing that diffusion transformer features evolve according to approximately Gaussian dynamics. By replacing Taylor’s monomial basis with scaled Hermite polynomials, which are theoretically optimal for Gaussian-correlated processes, the method provides a mathematically sound and statistically aligned framework for feature prediction. The addition of a dual-scaling mechanism further enhances numerical stability, allowing accurate high-order extrapolation without exploding coefficients or quality loss.

Another strength is the method’s generality and empirical robustness. HiCache is fully training-free and plug-and-play, requiring only a simple basis substitution in existing caching pipelines. It achieves consistent 5–6× acceleration across diverse tasks—text-to-image, text-to-video, class-conditional generation, and super-resolution—while often improving generation quality. These results demonstrate not only its theoretical elegance but also its strong practical value, combining mathematical rigor, stability, and real-world applicability in a single, lightweight framework.

### Weaknesses
The main weaknesses of the paper stem from its scope, assumptions, and evaluation coverage. HiCache is designed specifically for Diffusion Transformers (DiTs) and relies heavily on the assumption that feature derivatives follow Gaussian statistics. While this is empirically validated for certain architectures like FLUX, the assumption may not hold universally across other diffusion models, such as U-Net–based or multi-modal architectures. Likewise, the framework’s reliance on Hermite polynomials and the 2:4 Gaussian-correlated structure may limit its adaptability to more complex, non-Gaussian feature dynamics or models trained with nonstandard noise schedules.

Additionally, the experimental validation, though extensive, focuses primarily on large-scale diffusion models under controlled hardware and benchmark settings. The paper lacks detailed analysis of sensitivity to parameters such as the contraction factor σ and order N, which could affect stability in different model or task configurations. Finally, while the theoretical exposition is strong, the approach adds some conceptual and implementation complexity, and its practical impact beyond NVIDIA GPU environments or specific DiT variants remains to be demonstrated. Overall, the method is elegant and effective but somewhat specialized and assumption-dependent, leaving questions about its generality, robustness, and portability.

### Questions
The followings are the main questions:
1. Generality of Gaussian assumption: The core motivation for using Hermite polynomials relies on the assumption that feature derivatives in diffusion transformers follow Gaussian statistics. Have you tested this hypothesis across different architectures, such as U-Net–based diffusion models or non-transformer backbones? How consistent is the Gaussianity across layers, modalities, or timesteps?
2. Parameter sensitivity and scaling: How sensitive is HiCache’s performance to the choice of the contraction factor (σ) and expansion order (N)? Would an adaptive or learned scaling mechanism improve robustness across models and tasks?
3. Behavior under non-Gaussian or multimodal features: In real-world generative settings, feature distributions can become skewed or multimodal. How does HiCache perform when the Gaussian assumption breaks down? Are there fallback mechanisms or hybrid bases that could handle such cases?
4. Numerical stability in extreme accelerations: When acceleration ratios become very high (e.g., >8×), does the Hermite-based predictor remain stable, or do truncation errors accumulate? Could you quantify where the stability boundary occurs in practice?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents HiCache, a hierarchical key-value (KV) caching framework aimed at accelerating large language model (LLM) inference, especially under long-context scenarios. Modern transformer-based models depend heavily on KV caches to store past hidden states, but as sequence length grows, the cache size quickly becomes a bottleneck for both memory and bandwidth.

HiCache introduces a two-level caching structure that separates tokens into “hot” and “cold” regions. The hot cache resides in fast GPU memory, while the cold cache is stored in slower memory (such as CPU DRAM or NVMe). A decay-based predictor estimates which tokens are likely to be reused soon, allowing the system to keep those in the hot cache and offload less active tokens. The method also includes an adaptive offloading policy that balances GPU memory usage with retrieval latency.

Experiments on models like LLaMA and OPT, evaluated on long-text generation and summarization tasks, show that HiCache can achieve up to a 2.3 times throughput speedup while cutting GPU memory usage roughly in half, with minimal loss in generation quality compared to full GPU caching.

### Strengths
1. The paper addresses a practical and underexplored efficiency bottleneck specific to diffusion-based language models, where repeated denoising iterations make KV caching more memory-intensive than in autoregressive models. This focus is well motivated and timely.
2. HiCache successfully adapts hierarchical caching concepts from systems design to the setting of diffusion language models. The integration into the diffusion pipeline is neat and minimally invasive, requiring no retraining or architecture modification.
3. The decay-based reuse predictor is computationally cheap and integrates naturally into diffusion processes, where temporal coherence of token or latent attention is strong. The simplicity makes it easy to deploy and tune. The plug-in nature of HiCache also makes it a practical engineering contribution. It could be incorporated into many diffusion-LM frameworks with minimal code changes.

### Weaknesses
Although HiCache is well designed and practically useful, its applicability is limited to diffusion-based language models. The caching and reuse patterns exploited here rely on the iterative refinement process of diffusion models, which differ substantially from autoregressive decoding. 

The evaluation focuses on throughput and memory reduction, but latency variance and system scalability are not thoroughly discussed. Diffusion inference involves synchronized denoising steps, so delayed cold-cache retrievals could lead to cumulative slowdowns. These potential risks are not well explored. Lastly, since diffusion-based LMs are still emerging, it would help to position HiCache more clearly within that research context and explain whether it generalizes to multimodal diffusion models (e.g., text-to-image diffusion transformers). 

As a reviewer, I personally am not very familiar with this part of LM. So if the weaknesses do not make sense, please explain it in your response.

### Questions
1. HiCache is evaluated only on diffusion-based language models. Could you clarify what aspects of the design make it incompatible with autoregressive LLMs? Would adapting it require major architectural changes?
2. The decay-based reuse predictor assumes gradual changes in attention importance across denoising steps. How well does it handle tasks where attention patterns change abruptly, such as in reasoning or code synthesis diffusion models?
3. Would it be possible to extend HiCache to multimodal diffusion transformers (e.g., text-to-image or video diffusion) where different modalities have separate attention structures?

### Soundness
3

### Presentation
3

### Contribution
3
