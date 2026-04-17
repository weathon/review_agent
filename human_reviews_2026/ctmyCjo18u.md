# EchoGen: Generating Visual Echoes in Any Scene via Feed-Forward Subject-Driven Auto-Regressive Model

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Subject-driven generation is a critical task in creative AI; yet current state-of-the-art methods present a stark trade-off. They either rely on computationally expensive, per-subject fine-tuning, sacrificing efficiency and zero-shot capability, or employ feed-forward architectures built on diffusion models, which are inherently plagued by slow inference speeds. 
Visual Auto-Regressive (VAR) models are renowned for their rapid sampling speeds and strong generative quality, making them an ideal yet underexplored foundation for resolving this tension.
To bridge this gap, we introduce EchoGen, 
a pioneering framework that empowers VAR models with subject-driven generation capabilities.
The core design of EchoGen is an effective dual-path injection strategy that disentangles a subject's high-level semantic identity from its low-level fine-grained details, enabling enhanced controllability and fidelity. 
We employ a semantic encoder to extract the subject's abstract identity, which is injected through decoupled cross-attention to guide the overall composition. Concurrently, a content encoder captures intricate visual details, which are integrated via a multi-modal attention mechanism to ensure high-fidelity texture and structural preservation.
To the best of our knowledge, EchoGen is the first feed-forward subject-driven framework built upon VAR models. Both quantitative and qualitative results substantiate our design, demonstrating that EchoGen achieves subject fidelity and image quality comparable to state-of-the-art diffusion-based methods with significantly lower sampling latency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces EchoGen, a feed-forward subject-driven image generation framework that replaces the diffusion backbone with a visual autoregressive (VAR) architecture. The model integrates a dual-path feature injection mechanism: (1) a semantic path using DINOv2 features to capture high-level identity and structure, and (2) a content path using FLUX.1-dev VAE features to preserve low-level details. This design enables EchoGen to synthesize novel scenes containing a given subject without per-subject fine-tuning. On benchmarks such as DreamBench, EchoGen is competitive with diffusion-based approaches in performance, while offering faster inference.

### Strengths
1. This paper presents the first subject-driven feed-forward framework on a VAR backbone with a well-motivated dual-path injection design to capture both the high-level and low-level information from subject images.
2. EchoGen achieves competitive subject-driven image generation performance to state-of-the-art diffusion methods while cutting sampling latency to 5.2 seconds (vs. 10-50 seconds for feed-forward diffusion models).

### Weaknesses
1. The efficiency comparison in Table 1 may not be fully fair. The paper reports using default sampling configurations for all diffusion baselines, yet sampling time in diffusion models is highly dependent on sampling parameters such as the number of denoising steps. This can potentially overstate EchoGen’s relative speed advantage. A more rigorous analysis showing performance-compute tradeoffs across multiple sampling settings of the baselines would make the efficiency claims of EchoGen more convincing.

### Questions
1. The Subjects-200k dataset appears largely synthetic (generated with GPT-4o and FLUX.1-dev). How well does EchoGen generalize to real-world subject personalization?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
EchoGen is proposed as the first feed-forward, subject-driven image generation framework based on Visual Auto-Regressive (VAR) models, overcoming the limitations of existing methods that trade off controllability for speed. By employing a dual-path injection mechanism—decoupling high-level semantics (via DINOv2) and low-level details (via FLUX.1-dev VAE) through specialized attention modules—EchoGen enables fast, non-iterative sampling. Efficiently trained on Subjects-200k and evaluated on DreamBench, EchoGen achieves subject fidelity and text alignment on par with or better than leading diffusion models, while reducing inference time for 1024×1024 images to under 6 seconds.

### Strengths
1. The separation of semantic identity and content details is technically sound. Leveraging DINOv2 for semantic abstraction and FLUX VAE for detail encoding, complemented by distinct attention mechanisms, demonstrates appropriate feature representation in subject-driven image generation.
2. EchoGen demonstrates comparable or improved results relative to strong diffusion-based feed-forward methods (e.g., IP-Adapter, OminiControl) on DreamBench benchmarks (CLIP-I, DINO, CLIP-T). Its superiority in subject fidelity and photorealism is further corroborated by human evaluation.
3. A latency of under 6 seconds for high-resolution image generation represents a significant practical advantage, making EchoGen particularly well-suited for real-time and user-facing applications where responsiveness is critical.

### Weaknesses
1. The authors mention related work [1] on fine-tuning VAR for subject-driven generation, but this is not included in experiments. A comparison would help clarify whether feed-forward is truly superior to lightweight fine-tuning in the VAR regime.
2. Although Subjects-200k (~256k triplets) is substantial in scale, it is synthetically generated using FLUX.1-dev and GPT-4o, which may introduce distributional biases and limit real-world subject diversity. Furthermore, the paper does not provide a detailed analysis of failure cases involving challenging real-world subjects, such as humans in complex poses or transparent objects.
3. While Table 2 reports preference ratios, important methodological details are missing, such as the evaluation format (single-choice vs. multi-select), the number of images per trial, and participant expertise. Including this information would strengthen the reliability of the human study findings.
 
[1] Fine-tuning visual autoregressive models for subject-driven generation

### Questions
Please refer to the weaknesses. If the issues I raised are addressed, I will increase my rating.

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
4

### Summary
This paper introduces EchoGen, the first subject-driven image generation framework built on a VAR model rather than diffusion models. The method enables fast, feed-forward subject-driven generation by injecting both high-level semantic identity and low-level visual details through a dual-path design. EchoGen leverages a semantic encoder and a content encoder to disentangle identity and appearance, achieving controllable and high-fidelity results. Experiments show that EchoGen reaches state-of-the-art subject fidelity and image quality while significantly reducing sampling latency compared to diffusion-based approaches.

### Strengths
1. The paper presents the first subject-driven image generation framework based on a visual autoregressive model, expanding the design space beyond the diffusion-based paradigm.
2. The dual-path injection strategy—separating semantic identity from fine-grained visual content—is conceptually meaningful and empirically shown to improve subject fidelity and controllability.
3. The method achieves competitive or superior generation quality compared to diffusion-based baselines while offering substantially lower inference latency, addressing a clear practical limitation of existing approaches.

### Weaknesses
1. Although the framework claims to disentangle semantic and content features, the paper does not provide explicit mechanisms or empirical evidence demonstrating effective disentanglement between these two feature types.
2. The method depends on multiple large pretrained models (Qwen2.5-VL, DINOv2, FLUX.1-dev VAE), which increases system complexity and may limit applicability in real-world or resource-constrained settings.
3. The motivation for adopting VAR models feels somewhat shallow—mainly framed as “VAR is faster than diffusion”—without deeper justification for why VAR is inherently suitable for subject-driven generation beyond latency.
4. Some ablation studies and qualitative analyses are missing, such as the effect of removing DINOv2 or FLUX.1-dev, or visual comparisons showing the contribution of semantic vs. content features.

### Questions
1. The pipeline relies on several large pretrained components (Qwen2.5-VL, DINOv2, FLUX.1-dev VAE, and the base VAR model). Do the latency numbers reported in the paper reflect the full end-to-end runtime, including all feature extraction modules, or only the VAR sampling stage?
2. Is there a comparison of computational resource usage (e.g., GPU memory, inference throughput) against diffusion-based subject-driven methods? Such information would clarify the practical efficiency advantages claimed in the paper.
3. Why is the semantic feature injected into the Image Token instead of the content feature? Is there a conceptual or empirical advantage to this setting, and were alternative designs explored?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces EchoGen, a novel feed-forward framework for subject-driven image generation built upon a Visual Autoregressive (VAR) model. Motivated by the significant latency of diffusion-based approaches and the high computational cost of per-subject fine-tuning , EchoGen leverages the inherent fast sampling of VAR models to address this efficiency gap. The core technical contribution is a dual-path injection strategy that disentangles a subject's high-level semantic identity from its fine-grained textural details, enabling high-fidelity preservation and control. Experimental results validate that EchoGen achieves subject fidelity and text alignment competitive with state-of-the-art diffusion-based methods, while being significantly faster at inference.

### Strengths
The paper’s primary strength lies in its novel and successful exploration of an autoregressive architecture for subject-driven generation. This approach achieves performance that is highly competitive with state-of-the-art diffusion-based methods. Its most notable advantage is that it delivers strong results while drastically reducing inference latency, offering a significant improvement in efficiency and practicality compared to the slow, iterative sampling required by diffusion models. Moreover, the empirical evaluation is thorough and rigorous, supporting the validity of the proposed approach.

### Weaknesses
1. The model's generative quality is fundamentally limited by the capabilities of its Infinity-2B backbone. This backbone restricts its ability to render fine-grained details like coherent text or complex textures, meaning its peak quality is constrained even though its efficiency is high.
2. The method relies on a complex pre-processing pipeline using Qwen2.5-VL and GroundingDINO, which adds computational overhead. It is unclear if the reported inference latencies (5.2s for EchoGen-2B ) include this segmentation step. If not, the true "wall-clock" efficiency comparison is ambiguous.
3. The paper lacks an analysis of the model's sensitivity to the quality of the segmentation. While an ablation confirms segmentation is beneficial, it is unknown how performance would be impacted by a less accurate (or more accurate) segmentation model or by common bounding box errors. The system's robustness to imperfect pre-processing remains untested.

### Questions
See the weakness section above.

### Soundness
4

### Presentation
4

### Contribution
3
