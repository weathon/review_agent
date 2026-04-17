# D2D: Detector-to-Differentiable Critic for Improved Numeracy in Text-to-Image Generation

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Text-to-image (T2I) diffusion models have achieved strong performance in semantic alignment, yet they still struggle with generating the correct number of objects specified in prompts. Existing approaches typically incorporate auxiliary counting networks as external critics to enhance numeracy. However, since these critics must provide gradient guidance during generation, they are restricted to regression-based models that are inherently \emph{differentiable}, thus excluding detector-based models, whose count-via-enumeration nature is \emph{non-differentiable}. To overcome this limitation, we propose \textbf{Detector-to-Differentiable} (\emph{D2D}), a novel framework that transforms non-differentiable detection models into differentiable critics, thereby leveraging their superior counting ability to guide numeracy generation. Specifically, we design custom activation functions to convert detector logits into binary indicators, which are then used to optimize the noise prior at inference time with pre-trained T2I models.
Our extensive experiments on SDXL-Turbo, SD-Turbo, and Pixart-DMD across four benchmarks of varying complexity (low-density, high-density, and multi-object object scenarios) demonstrate consistent and substantial improvements in object counting accuracy, by up to 13.7\%, with minimal degradation in overall image quality and computational overhead.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenge of diffusion models failing to generate the precise number of objects specified in a prompt. While many existing methods employ gradients of regression-based counting networks, the authors argue that detection models, such as YOLO, offer superior count accuracy. To overcome the non-differentiable nature of these detectors, they introduce a novel surrogate loss function. This loss enables optimization by updating the initial noise, a technique similar to that in Reno, rather than relying on traditional gradient guidance during the diffusion process.

### Strengths
- The work demonstrates strong quantitative and qualitative performance against baselines, validated across multiple benchmarks and models, while also achieving low inference latency

### Weaknesses
- The novelty of this work appears limited, primarily consisting of replacing regression-based counting networks with detection models wrapped by a differentiable loss. Other parts of the methodology seem largely similar to previous work.
- Including Algorithm 2 from the appendix in the main paper would significantly aid in understanding the proposed algorithm.
- An explanation and analysis on why detection models are superior to regression models for this application would be very helpful and strengthen the paper's claims.
- Minor: The paper contains some grammar mistakes and typos that should be corrected throughout.

### Questions
- Inconsistency regarding High Object Counts: Figure 2b indicates that regression models begin to outperform detection models at higher object counts. However, Table 1 shows that your approach still outperforms the baselines in these same high-count scenarios. Could the authors please explain this apparent discrepancy?
- Computational Cost of $L_{\text{D2D}}$: Examining Algorithm 2, it appears that computing the loss term $L_{\text{D2D}}$ requires backpropagating through all diffusion steps. Could the authors elaborate on why this only has a minor effect on latency? Furthermore, what are the implications of this on memory usage during training and inference?
- Handling Duplicate Bounding Boxes: Detection models like YOLO typically utilize post-processing steps, such as Non-Maximum Suppression (NMS), to eliminate duplicate bounding boxes. How does your method handle the issue of potentially overlapping or duplicate detections?

### Soundness
2

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
4

### Summary
The paper presents an inference-time tuning strategy for improving the numeracy/counting of text-to-image models. The main idea is to solve an inference-time optimization process for the initial noise to optimize the count obtained from an object detector. From a technical standpoint, the main contributions are in a) creating a differentiable critic by applying a sigmoid activation over the logits b) modulating the initial noise with the "latent modifier network". Results on several counting benchmarks clearly indicate that this inference-time optimization is able to drastically improve the numeracy of these models.

### Strengths
The biggest strength of the paper is in constructing an effective objective for differentiable optimization. This ensures that there's effective inference-time optimization, which is visible from the strong empirical results on several counting benchmarks. 

The paper is also well-presented and easy to follow.

### Weaknesses
Avoids more recent larger models: The results in the paper are on the SD-Turbo, SDXL-Turbo and Pixart-alpha models which are all not only distilled (therefore a bit worse in performance), but also fairly out of date as of late 2025 (all being released between late 2023-early 2024). While some of the newer models may not be applicable, one could still see results on Flux-Schnell, SANA-Sprint, SD3.5-Turbo to see how effective this optimization framework is on newer problems. While I'd expect the counting problem to still be there for newer/larger models, I'd imagine that it is a lot less of a problem with these models (e.g. Qwen-Image etc.)


Limited Contribution: The paper is essentially applying the existing initial noise optimization formulation for improved numeracy in T2I models (which had in the past either been used for toy objectives or general human preference/prompt alignment). While effective, from a conceptual standpoint it does feel somewhat limiting, especially since it inherits the issues of inference-time optimization (i.e slower runtimes, memory overhead etc.). To that extent, while the paper's contribution itself might be an integral part of a modern diffusion models' post-training pipeline (since it's easy to verify similar to the RL formulation in LLMs for math/code), it feels rather minimal to merit acceptance in my view.

### Questions
In general, I'm mostly curious about how relevant this problem is to be tackled on a standalone basis with a somewhat expensive inference-time optimization formulation. In that regard, a) showing how much of an issue this is on recent, larger models, b) ability of the noise optimization formulation to work with at least the newer distilled models, c) whether these techniques could be used to enhance the base capabilities of this model (i.e fine-tuning a base model on a high-quality dataset of optimized samples etc.) would go a long way in strengthening the contributions of the paper. While I don't really have major doubts about the "soundness" or "presentation" of the paper, I would really like to see more evidence about the "contributions" of the paper before recommending acceptance for the paper.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tackles the problem of improving object numeracy in text-to-image diffusion models, which often fail to generate the correct number of objects specified in text prompts. 
The paper proposes Detector-to-Differentiable (D2D), a novel framework that converts non-differentiable object detectors into differentiable critics.  
By optimizing the initial noise during inference, D2D enhances count accuracy while preserving image quality. 
Experiments across multiple T2I backbones and benchmarks show meaningful improvement in object counting accuracy with minimal computational overhead.

### Strengths
1. The proposed method is logical and well-motivated, with clear explanations that make the underlying rationale and implementation easy to understand.

2. The analysis of the proposed framework is in-depth: for instance, the comparison between detector-based and regression-based counting models (Figure 2), the exploration of class-wise performance (Figure 4), and the ablation between direct noise optimization and LMN-based methods (Table 5).

3. The demonstrated generality of the proposed method across diverse diffusion backbones and evaluation scenarios is commendable.

### Weaknesses
1. The evaluation relies solely on a detector-based protocol to assess the proposed detector-based critic. As noted around line 315, the paper employs the SOTA counting model CountGD (built upon GroundingDINO). Although the proposed critic uses different detectors such as OWL-ViT or YOLO for initial noise optimization, this setup risks giving an unfair advantage aligned with the evaluation criterion. To more robustly validate the superiority of the method, additional evaluation metrics, such as human evaluation (as adopted in Make It Count, CVPR 2025), should be considered.

2. It would be interesting to investigate whether the proposed noise optimization process restricts the diversity of generated images. Since the method optimizes the initial noise, the input noise distribution may become narrower than that of purely random noise, potentially reducing image diversity. Thus, analyzing generation diversity would provide important insights into possible trade-offs.

3. Table 6 omits the performance of the original backbone diffusion models (e.g., SDXL-Turbo, Pixart-DMD). While image quality comparisons with other counting baselines are provided, including the results of the original backbones would strengthen the claim that “D2D yields minimal degradation in image quality” (line 456).

### Questions
1. (related to Table 1) Is the proposed method only applicable to single-step diffusion models? If so, is there a specific reason why it cannot be applied to multi-step diffusion models such as SDXL or Pixart-α? Applying the proposed method to SDXL would be particularly valuable, as it would enable direct comparisons with Make It Count.

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
This paper addresses numeracy in text-to-image models through reward-based inference-time optimization, making two main contributions. First, they design a differentiable counting reward $\mathcal{L}_{D2D}$ derived from object detectors, which are more accurate than regression-based counters in low-density scenarios but previously unusable as rewards due to non-differentiable enumeration. They achieve this through custom activation functions that convert detector logits into gradient-friendly signals. Second, they propose the Latent Modifier Network (LMN), a lightweight 3-layer MLP that transforms the initial noise for reward optimization. Unlike prior work that directly tunes the noise, the LMN provides a larger parameter space while preserving portions of the original noise through a weighted mixing scheme. Experiments show consistent improvements in counting accuracy.

### Strengths
- **Novel reward function design:** Converting non-differentiable detectors into differentiable counting rewards through steep sigmoids and logit scaling (Eq. 1-2) addresses a limitation where detectors outperform regression models in low-density counting but couldn't previously be used for gradient-based optimization.
- **Two complementary contributions:** The differentiable detector-based reward ($\mathcal{L}_{D2D}$) and the LMN architecture for reward optimization. Table 5 shows the LMN improves numeracy by 10% points over direct noise optimization when using the same reward, suggesting the LMN is a valuable contribution independent of the specific reward function.
- **Consistent empirical results:** Consistent improvements across multiple base models and settings in reasonable time-frames.

### Weaknesses
- **Limited Discussion of General-Purpose Application**: The methodology is tailored for prompts containing numerical targets, but the paper does not discuss how the framework should behave with general, non-numeric prompts. It is unclear whether the D2D optimization is intended to be selectively activated, or what the default behavior might be in the absence of a numerical target (e.g., "few", "some", or "many"). How about very long complex prompts with lots of content. Additionally, it's not really clear what the effect of D2D is on aesthetics and diversity of generated images.
- **Justification for Reward Function Design Choices**: While the activation function design is intuitive, several design choices lack theoretical justification or empirical comparison to alternatives:
	- **Scaling formulation (Eq. 2):** The justification for multiplying by $(z_i - \tau_z)$ to combat sigmoid plateauing is limited. Did the authors explore other options, such as exponential scaling, learned scaling functions, or other monotonic transformations that could provide stronger gradients? The paper does not compare the chosen formulation against alternatives.
	- **Mixing weight:** Table 9 shows $w=0$ achieves the best numeracy but produces "patchy visual artifacts," suggesting a tradeoff between reward optimization and image quality. The basis for choosing $w=0.2$ could be better explained. More detailed analysis of this numeracy vs. image quality tradeoff in the main paper would be valuable.
	- **Noise regularization sharpening**: The paper proposes a much more sharped version of previous regularization in noise space by exponentiating it by a power of 10. To me it's unclear why this change was made, and not really motivated in the paper.
- **LMN Contribution Not Fully Explored**: The LMN appears to be a contribution independent of the counting reward (Table 5 demonstrates improvements over direct noise tuning), yet it receives limited investigation.
	- It is unclear whether the LMN improves optimization of other rewards beyond counting (e.g., ImageReward or similar like ReNO). Currently, Table 5 only compares LMN vs. direct noise tuning using $\mathcal{L}_{D2D}$. Evaluation with other rewards would help establish whether the LMN is a general contribution to inference-time reward optimization or specific to counting tasks.
	- The choice of a 3-layer MLP is not justified through ablations. It would be informative to explore variations such as 2 or 4 layers, different widths, or alternative architectures.

Minor Issues:
- Including the base model performance in Table 6 would facilitate easier comparison of how the reward-based optimization affects image quality metrics.
- D2D is proposed as a differentiable surrogate for the non-differentiable detector count, but no analysis validates this approximation. It would be informative to analyze how well $\mathcal{L}_{D2D}$ correlates with the actual detector count $D(I)$ throughout the optimization trajectory, and under what conditions the surrogate diverges from the true count (e.g., when many bboxes cluster near threshold $\tau_z$).

### Questions
- How does the LMN perform for human-preference rewards, e.g. ReNO?
- What's the effect of D2D on the diversity of generated images?
- What is the motivation behind the sharper noise regularization? How does it imact diversity?
- What's the effect of D2D on general image quality? This could be made more clear through adding baseline performances to Table 6
- Could D2D be also used to fine-tune a T2I model, e.g. with Adjoint Matching?
- Are there scenarios where the surrogate diverges from the true count?
- How many optimization steps are used? How important is the pre-inference alignment stage?

### Soundness
2

### Presentation
3

### Contribution
3
