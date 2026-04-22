# $A^{4}$-MLRM: Fourfold Attention for Adaptive Hallucination Suppression in Multimodal Large Reasoning Model

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 4, 8

## Abstract
Large multimodal reasoning models have recently shown strong ability to solve complex problems by gathering evidence and performing multi-step inference. However, the long reasoning chain makes them more prone to hallucination, that is, generating content that is not supported by the input image or the question. In examining how hallucination arises, we further identify \emph{reasoning drift}: during evidence gathering the model over focuses on entities unrelated to the question, diluting attention on task relevant cues. As a result, previous attention-based methods developed for non-reasoning models often fail to localize the true evidence in reasoning settings. Based on these insights, in this paper, we introduce \emph{AttnRecall}, a metric for assessing visual perception, and present \method{}, a training free, parameter free, and architecture agnostic plugin to hallucination suppression. \method{} uses the model output as a conduit from question to visual tokens for identifying question relevant patches and steer focus to task relevant regions. 
Remarkably, \textbf{without any additional training}, \method{} improves all \textbf{reasoning} architectures (including \texttt{R1-OneVision}, \texttt{Ocean-R1}, \texttt{MM-Eureka}, \textit{etc.}) by $\mathbf{1.21\times}$ on reasoning benchmarks. When transferred to \textbf{non\mbox{-}reasoning} settings, it yields a $\mathbf{1.16\times}$ gain.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces A4-MLRM, an architecture-agnostic, training-free, and parameter-free plugin designed to mitigate hallucination and reasoning drift (i.e., attentional diffusion toward task-irrelevant details) in MLRMs. They leverage the model’s native attention mechanism, and their strategy results in an average improvement of 1.21× on reasoning benchmarks and 1.16× when transferred to non-reasoning settings.

### Strengths
- A4-MLRM is completely training-free and parameter-free, resulting in minimal deployment cost, a significant advantage over prior training-stage mitigation efforts; it's also architecture-agnostic.

- A4-MLRM achieves substantial performance gains, demonstrably improves all reasoning models tested (R1-OneVision, Ocean-R1, MM-Eureka, ORSTA-R1). It also successfully transfers to non-reasoning MLLMs (LLaVA-1.6-Mistral and Qwen2.5-VL), moving some models "from near chance to the GPT-4V range" on perception benchmarks.

### Weaknesses
- The paper suffers from several presentation issues. It is difficult to follow, as many notations are introduced at different stages without a consistent framework established in the Background section. Moreover, key steps and observations are only presented in the Appendix, which makes the paper harder to read and the findings more difficult to assess and trust. The figures, rather than clarifying the content, are quite dense and make the paper harder to follow. I would recommend a thorough revision of **the structure of this paper**, perhaps by emphasizing the most important observations or focusing more clearly on illustrating the method itself.

- A4-MLRM currently relies on a two-stage inference pipeline (Stage 1: attention mining; Stage 2: focused re-inference), which, given current reasoning-model architectures, may introduce latency and computational overhead. Moreover, the paper lacks a clear comparison of computational costs and how they scale with output length. The accuracy of online inference is closely tied to the output sequence length used in Stage 1: longer sequences provide richer priors for Stage 2 and thus higher accuracy, highlighting a trade-off between efficiency and performance, I guess...

### Questions
Q1: The paper identifies that the perception signal peaks around layers 18–24 in 7B architectures using AttnRecall. How stable is this finding across different model sizes (e.g., 13B or 72B variants mentioned in related work or baselines like Qwen2.5-VL) or models with fundamentally different underlying architectures?

Q2: For Ocean-R1, in the case of POPE (Table 5), we observe a slight decrease in accuracy after applying A4-MLRM (from 86.77% accuracy without A4 to 85.76% with A4). Although it is quite minor, could the authors suggest reasons for such behavior? Could A4-MLRM, in some cases, exclude relevant visual context necessary for certain complex reasoning or general QA tasks?

### Soundness
2

### Presentation
1

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
This paper introduces A4-MLRM, a training-free and architecture-agnostic inference-time approach to mitigate hallucinations in multimodal reasoning models. The method traces attention from question tokens to generated reasoning tokens and subsequently to visual patches, and re-queries the model using attention-selected regions. Experiments show improvements across several reasoning models and hallucination benchmarks.

### Strengths
- Addresses a challenge: hallucination in multimodal reasoning models.
- Practical and deployment-friendly design: no training, no architectural modifications.
- Demonstrates consistent improvements across multiple MLLMs and evaluation benchmarks.
- Includes attention layer analysis and ablations supporting empirical choices.
- Shows transferability to non-reasoning models, indicating broader applicability.

### Weaknesses
- Innovation is limited; the idea resembles prior attention-guided focusing / grounding strategies and is largely heuristic.
- Heavy reliance on attention as a meaningful signal without deeper theoretical justification on its causal reliability.
- Evaluation focuses mainly on binary hallucination settings; generalization to open-ended reasoning and complex visual tasks is unclear.
- Sensitivity to thresholds, clustering, and sequence length is not systematically studied.
- Two-stage inference incurs latency, and deployment cost analysis is insufficient.
- Behavior on larger models (>7B) is not explored, raising concerns about scalability.

### Questions
1. How reliable is attention for grounding when attention maps are known to be noisy or misaligned in some models?
2. Are τq, τo, τv and clustering parameters fixed across all models and datasets? Any sensitivity analysis?
3. How does the method perform on open-ended VQA and compositional reasoning tasks?
4. Can you report latency and compute overhead for both online and offline modes?
5. Does performance scale or saturate on larger models (e.g., 34B / 70B)?
6. Can you provide failure case visualizations where attention routing misleads the model?
7. How does the method compare or combine with RL-based grounding or verifier-based hallucination mitigation?
8. Does the method risk over-focusing on small regions, losing necessary context for multi-entity reasoning?
9. Recent work suggests that extended chain-of-thought can weaken visual grounding and increase hallucinations, closely related to the “reasoning drift” discussed here. Could the authors clarify how A4-MLRM relates to these findings and whether it mitigates the trade-off between deeper reasoning and degraded perception, especially in long-chain reasoning scenarios?

- More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models
-  More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper first analyzes the reasons why VLMs generate hallucinations. Then they introduce a series of methods to evaluate visual perception and locate important visual regions based on the attention score. The proposed method leads to consistent performance improvement across various hallucination benchmarks.

### Strengths
- This paper provides very interesting insights about reasoning VLMs tend to produce hallucinations due to attention drift, and proposes a pipeline to locate the important visual region based on the model's attention score. 

- A4MLRM effectively suppresses the hallucination across multiple VLMs and benchmarks.

- The paper is well-written and technically sound.

### Weaknesses
1. The calculation of AttnRecall relies on the paired dataset, limiting its application on benchmarks without paired bounding boxes or segmentation masks.

2. The evaluation is limited on hallucination benchmarks. The ability of the proposed method in reducing hallucination in other real-world tasks (such as spatial reasoning).

### Questions
1. In the definition of A2,  why do you choose the question tokens with high standardized variability as the key question tokens? 

2. Can your method improve the VLMs' performance in other visual-centric tasks such as spatial reasoning? It is suggested to evaluate your method in benchmarks including the RealWorldQA [1] and 3DSRBench [2]. If the performance is not improved, please provide some analysis about the underlying reason.

3. For VLMs with stronger spatial reasoning capability such as [3], will it show better AttnRecall?

[1] https://huggingface.co/datasets/visheratin/realworldqa

[2] Ma, Wufei, et al. "3dsrbench: A comprehensive 3d spatial reasoning benchmark." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

[3] AI, Inclusion, et al. "M2-reasoning: Empowering mllms with unified general and spatial reasoning." arXiv preprint arXiv:2507.08306 (2025).

### Soundness
3

### Presentation
3

### Contribution
2
