# Severing the Link: A Unified Adversarial Attack on Image and Video MLLMs via Generative Disruption

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 4, 6

## Abstract
While Multimodal Large Language Models (MLLMs) demonstrate remarkable cross-modal reasoning, their core vision-language grounding mechanisms present critical vulnerabilities, particularly in complex video scenarios. 
We introduce **CAVALRY**, a unified framework for generating powerful adversarial attacks against both image and video MLLMs. 
Our approach introduces two key innovations: **(i)** a paradigm shift from conventional classification-boundary attacks to directly disrupting the generative process, realized through a novel loss that maximizes the likelihood divergence of the ground-truth response and severs the visual-linguistic link; and **(ii)** an efficient, progressive generator trained to produce spatiotemporally coherent perturbations for both dynamic videos and static images.
Comprehensive evaluations on seven state-of-the-art MLLMs, including GPT-4.1, Gemini 2.0, and QwenVL-2.5, validate CAVALRY's superior performance. 
Our method outperforms the strongest baselines by an average of 22.8\% on video understanding benchmarks and extends this advantage to static images, proving 34.4\% more effective than prior work. 
These results establish CAVALRY as a foundational framework for probing the adversarial robustness of the entire spectrum of modern MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces CAVALRY, a unified adversarial attack framework that disrupts the vision-language grounding in MLLMs. CAVALRY employs a two-stage training strategy to produce spatiotemporally coherent perturbations for both images and videos, achieving strong transferability across diverse MLLM architectures.

### Strengths
1. The motivation is clear and the task is interesting. Experimental results on seven mainstream MLLMs show that CAVALRY improves the attack effectiveness by an average of 34.4% and 22.8% in image and video tasks respectively, verifying its practicality and wide applicability.

2. The method framework is clearly introduced, while using visualization to help readers quickly understand the method.

### Weaknesses
1. The introduction lacks motivation for using generators, and accordingly, the related work lacks citations for generator-based approaches.

2. Among the losses used by the author, Lsem seems like the loss design of the white-box LVLM attack [1], Lvis is the loss design of FARE [2], according to the auxiliary model is ResNet-50, and Laux is the traditional adv attack loss design. Can the authors explain the core difference in Lsem and normally used LVLM attack loss?

3. In Table 1, why are all the comparison methods migrating image attacks to video? Can existing video methods be compared?

4. Is the "LLaVA-Video" in Table 2 a typo? It's in image benchmark.

5. It is recommended to add comparisons with other attack methods in Figure 4. In addition, the authors only show the processing efficiency. However, judging from the difference in methods, compared with attack paradigms such as PGD, the authors' method adds pre-training and finetuning processes. The overall time of these methods should be compared with the overall time of adversarial attack paradigms such as AnyAttack to reflect a fair comparison of performance gains and time costs.

[1] Schlarmann C, Hein M. On the adversarial robustness of multi-modal foundation models[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 3677-3685.

[2] Schlarmann C, Singh N D, Croce F, et al. Robust clip: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models[J]. arXiv preprint arXiv:2402.12336, 2024.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces CAVALRY, a unified adversarial framework for both image and video MLLMs. It trains a U-Net generator to produce adversarial perturbations for each input image by maximizing the negative log-likelihood of the ground-truth response, together with a regularization term that keeps the perturbed visual tokens close to the originals. Training is performed in two stages, and the authors evaluate the method on both closed-source and open-source models, reporting improved attack success rates.

### Strengths
Overall, the paper is well written and clearly presented. 

The paper provides both theoretical and empirical results, demonstrating the effectiveness of the proposed framework.

### Weaknesses
1. The method’s only video-specific component is batching frames from the same video with identical questions and answers during fine-tuning. This implicitly encourages temporal consistency but does not explicitly model temporal dependencies. In principle, the same setup would work for unrelated images sharing the same QA pairs. Therefore, the paper’s claim of “simultaneously modeling cross-modal reasoning vulnerabilities and temporal dependencies” (line 108) is an overstatement.

2. The proposed negative-likelihood objective is essentially equivalent to standard untargeted adversarial attacks, which also maximize the negative log-likelihood. The conceptual novelty over existing attack paradigms is limited.

3. All reported results rely on LLM-based scores, which inherently has randomness. Moreover, the score differences are small, making the reported improvements potentially sensitive to randomness. The paper should include multiple runs and standard deviations to ensure the statistical significance and reproducibility of the results.

### Questions
How can this method capture meaningful temporal dependencies merely by including frames from the same video within a single batch?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents CAVALRY, a unified adversarial attack framework targeting both image and video MLLMs. CAVALRY employs a generator that produces adversarial perturbations for visual inputs, whether static images or videos. The generator is optimized through three complementary objectives: (1) generative likelihood divergence, (2) manipulation of visual representations, and (3) an auxiliary feature loss. To ensure cross-modal and temporal coherence, CAVALRY adopts a two-stage training strategy involving large-scale pretraining followed by fine-tuning. Experimental results demonstrate that CAVALRY achieves state-of-the-art performance on both open-source and commercial MLLMs across the MMBench-Video and MME benchmarks.

### Strengths
1. Theorem 1 and its mathematical proof clearly justify the semantic loss objective, providing theoretical soundness.
2. The paper demonstrates transferability by showing that the attack generalizes not only to open-source MLLMs but also to commercial ones.
3. Instead of iteratively updating perturbations, the authors train a generator that can produce frame-wise perturbations with linear time complexity, proving the practical applicability of the approach.

### Weaknesses
Major Weaknesses
1. Both the output-level loss and vision-encoder-level loss for MLLMs have been extensively studied in prior MLLM adversarial and jailbreak attack works [1, 2, 3, 4]. Thus, the methodological novelty is somewhat limited.
2. The evaluation is restricted to a single benchmark for image understanding and one for video understanding, using only one LLM as the judge. This narrow experimental scope limits the assessment of generalizability across broader image/video domains and diverse text prompts.
3. The paper lacks sufficient analysis and motivation regarding MLLMs. In particular, the reason behind the proposed method’s high transferability is not well explained. While the authors claim that large-scale pretraining enables cross-architecture transferability, this explanation is not fully convincing. Large-scale data may help the generator generalize across a wide range of samples within the surrogate model’s domain, but it does not inherently guarantee transferability across different architectures.

[1] Luo, Haochen, et al. “An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models.” ICLR 2024.
[2] Cui, Xuanming, et al. “On the Robustness of Large Multimodal Models Against Image Adversarial Attacks.” CVPR 2024.
[3] Zhao, Yunqing, et al. “On Evaluating Adversarial Robustness of Large Vision-Language Models.” NeurIPS 2023.
[4] Zhang, Jiaming, et al. “AnyAttack: Towards Large-Scale Self-Supervised Generation of Targeted Adversarial Examples for Vision-Language Models.” CVPR 2025.

### Questions
Minor Weaknesses & Questions
1. It is unclear why the authors used an adversarially trained ResNet as the auxiliary model instead of an adversarially trained transformer architecture, given that most MLLM vision encoders are transformer-based.
2. The paper employs the SRR metric instead of the traditional ASR, but it is somewhat difficult to intuitively understand what level of attack strength the SRR represents.

### Soundness
2

### Presentation
3

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
This paper studies adversarial attacks and robustness of multimodal large language models (MLLMs) that answer vision-and-language queries over images and videos. The proposed method, CAVALRY, uses a U-Net-style generator trained in a two-stage framework to provide perturbed visual inputs that alter MLLM outputs. CAVALRY combines three different losses to target both vision-language connections and vision representation, and it is evaluated on both static (images) and temporal (video) inputs. Experiments indicate CAVALRY can substantially degrade MLLM performance across multiple models and settings.

### Strengths
- The attack design is intuitive and well-motivated: combining multiple loss terms to target both representation-level and alignment-level failure modes make sense for attacking LLMs.
- The evaluation covers both images and videos, demonstrating the generality of the framework across static and temporal inputs.
- The paper presents a reasonably comprehensive set of experiments showing CAVALRY’s efficacy across several MLLMs.

### Weaknesses
- The training strategy appears to treat all video frames independently, using the same QA supervision across frames rather than modeling temporal dependencies explicitly. This raises concerns that the attack might ignore inter-frame dynamics and overfit to per-frame perturbations rather than truly exploiting temporal vulnerabilities.
- Aside from a few qualitative examples (e.g., Figure 2), the paper lacks quantitative analysis of visual distortion (e.g., PSNR, LPIPS, or L2 norms) and human perceptual thresholds. Some examples images look heavily corrupted; reporting distortion budgets and the number of frames required for a successful attack would clarify real-world plausibility.
- The method combines three loss terms but does not fully isolate their contributions. An ablation study showing how each loss affects attack strength, perceptual quality, and transferability would strengthen understanding of why CAVALRY works.
- Training with the same QA for all frames may lead to artifacts: attacks might exploit static shortcuts rather than disrupting temporal reasoning. Evaluation on short-version of videos that is necessary for the true answer would give further understanding on how MLLMs are attacked.

### Questions
- Equation (5) targets likelihood divergence for the ground-truth answer. Do you have any theoretical or empirical evidence that optimizing this loss reliably causes semantically different outputs (not just low-confidence or truncated answers)?
- How does attack performance vary with the fraction of frames you perturb in a video? Is perturbing a single frame (or a small subset) sufficient in typical cases, or do you need to perturb most/all frames to achieve high success?
- Based on the nature characteristics of proposed training strategy, it seems like the model is not trained to consider the temporal relationships of different frames. That said, the first frame of the video should be used to attack MLLMs in a way that unknown future information is related. Are there any further analysis to ensure that CAVALRY truly considers the temporal characteristics of videos?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces CAVALRY, a unified adversarial attack framework targeting both image and video Multimodal Large Language Models (MLLMs). Instead of conventional boundary-based or feature-space attacks, the method disrupts the generative process of MLLMs through generative likelihood divergence maximization, effectively “severing” the link between visual perception and language generation. It employs a progressive two-stage generator trained to produce spatially and temporally coherent perturbations: large-scale pretraining on LAION-400M followed by fine-tuning on LLaVA-Instruct-150K and Video-MME for temporal coherence. Experiments on seven MLLMs, including GPT-4.1, Gemini 2.0, and QwenVL-2.5, show state-of-the-art performance, outperforming baselines by 22.8 % on video and 34.4 % on image benchmarks. The paper claims broad transferability, computational efficiency, and responsible release for AI-safety evaluation.

### Strengths
- Integrates both image and video adversarial attack settings into a single formulation, demonstrating flexibility across modalities. Bridges vision-language and temporal reasoning vulnerabilities, a gap unaddressed by prior work. This work establishes a new class of generative-disruption attacks for MLLMs.
- Operates on token-level autoregressive manipulation rather than feature logits, offering higher semantic fidelity. This paradigm shift yields measurable performance gains, and it is novel in this field.
- Benchmarked on seven diverse MLLMs, including commercial and open-source systems demonstrated its effectiveness of the proposed method.

### Weaknesses
- The SRR metric depends on GPT-4o-mini or GPT-4-turbo scorers, which may introduce bias. No cross-validation with other judges or independent human raters.
- Equation (8) introduces $\lambda_1$, $\lambda_2$, $\lambda_3$, weighting, but no individual ablation for each loss. Lack of sensitivity analysis limits understanding of the contribution balance.
- No adversarial detection, adversarial training, or model-side robustness evaluation provided. It would be good to include some experiments on mitigation methods.

### Questions
- Could the authors provide some comparison of the evaluation metric with other models?
- Could authors provide some insights on how to tune $\lambda_1$, $\lambda_2$, $\lambda_3$?

### Soundness
3

### Presentation
3

### Contribution
3
