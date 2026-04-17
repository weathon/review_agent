# MMaDA-Parallel: Multimodal Large Diffusion Language Models for Thinking-Aware Editing and Generation

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
While thinking-aware generation aims to improve performance on complex tasks, we identify a critical failure mode where existing sequential, autoregressive approaches can paradoxically degrade performance due to error propagation. 
To systematically analyze this issue, we propose ParaBench, a new benchmark designed to evaluate both text and image output modalities. Our analysis using ParaBench reveals that this performance degradation is strongly correlated with poor alignment between the generated reasoning and the final image.
To resolve this, we propose a parallel multimodal diffusion framework, MMaDA-Parallel, that enables continuous, bidirectional interaction between text and images throughout the entire denoising trajectory. MMaDA-Parallel is trained with supervised finetuning and then further optimized by Parallel Reinforcement Learning (ParaRL), a novel strategy that applies semantic rewards along the trajectory to enforce cross-modal consistency. Experiments validate that our model significantly improves cross-modal alignment and semantic consistency, achieving a 6.9\% improvement in Output Alignment on ParaBench compared to the state-of-the-art model, Bagel, establishing a more robust paradigm for thinking-aware image synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses a critical and often overlooked failure mode in ​thinking-aware image synthesis, where the incorporation of a Chain-of-Thought (CoT) reasoning step can paradoxically degrade the final output due to error propagation in sequential, autoregressive pipelines. The authors systematically identify that this degradation is strongly correlated with poor alignment between the generated reasoning text and the final image.

To tackle this fundamental issue, the paper introduces a novel ​parallel multimodal diffusion framework​ that enables continuous, bidirectional interaction between text and images throughout the entire denoising trajectory. This paradigm shift from sequential to parallel generation allows for real-time grounding of textual descriptions in visual evidence and vice versa, effectively mitigating the accumulation of errors.

### Strengths
1. The paper introduces a paradigm shift by proposing a parallel multimodal diffusion framework to replace sequential autoregressive pipelines for thinking-aware generation. This creative combination of discrete diffusion with bidirectional cross-modal attention effectively addresses the critical failure mode of error propagation identified in existing models. 

2. The proposed ParaRL algorithm, which applies reinforcement learning along the denoising trajectory, further represents a significant conceptual advance in optimizing for intermediate alignment.

3. The paper is exceptionally well-written and structured, with a clear narrative that logically progresses from problem identification to solution and validation.

4. It establishes a more robust paradigm for thinking-aware image synthesis, with potential broad impact on the field of multimodal generation. The introduction of the ParaBench benchmark fills a critical gap by providing a diagnostic tool to evaluate cross-modal alignment, which will be important.

### Weaknesses
Please see my questions.

### Questions
1. In Figure 3, the denoising process for text is depicted as sequential sentence-by-sentence generation. However, in inference, the denoising of text spans should occur in an arbitrary, non-sequential order. Could the authors clarify this, otherwise the diagram and method might cause confusion.

2. In the Appendix experiments, which model does "MMaDA-sequential" refer to? Is it the "MMaDA-MixCoT"? It appears that "MMaDA-parallel" only slightly outperforms "MMaDA-sequential." If "MMaDA-sequential" is fine-tuned using the proposed constructed data, would there be further performance improvements?

3. The generated CoT text relies on "Qwen2.5-VL-7B." How is the high quality of the constructed text ensured? Why wasn't a larger model, such as a 72B version, used for labeling? If higher-quality reasoning texts were constructed, could this further enhance performance?

4. In parallel reinforcement learning, is using "Qwen-VL" to evaluate the semantic alignment between partial text and partially generated images reliable? Since intermediate generated images might still be blurry or meaningless, the VLM's preference may not effectively judge meaningful scores.

5. Although MMaDA-Parallel achieves higher output alignment on ParaBench compared to other open-source models like Bagel, its output image quality does not surpass them. In other words, how can the author prove that output alignment is positively correlated with generating high-quality target images?

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
4

### Summary
This paper investigates the failure modes of thinking-aware generation and identifies that sequential reasoning–generation pipelines often suffer from error propagation. To systematically analyze this issue, the authors introduce ParaBench, a new benchmark that jointly evaluates reasoning and image outputs. Building upon this insight, they propose MMaDA-Parallel, a diffusion-based framework enabling continuous text–image interaction, and further optimize it via Parallel Reinforcement Learning (ParaRL). Experiments on ParaBench show a 6.9% improvement in output alignment over Bagel, demonstrating enhanced cross-modal consistency.

### Strengths
1.The paper provides a clear and insightful analysis of failure modes in thinking-aware multimodal generation, highlighting a real problem in current reasoning–generation pipelines.

2.ParaBench offers a valuable evaluation framework that jointly measures reasoning and image alignment, which could be useful for future multimodal research.

3.The proposed parallel diffusion framework with stepwise semantic optimization is well-motivated and achieves measurable gains in cross-modal alignment.

### Weaknesses
1. The method novelty is limited — the proposed MMaDA-Parallel and ParaRL mainly combine existing ideas of diffusion fine-tuning and reinforcement learning under a parallel setting.

2.The reported 6.9% improvement in output alignment, while positive, appears modest given the additional complexity and training cost. The authors are encouraged to provide stronger evidence that this gain is statistically or practically significant, or to further improve the results through more comprehensive experiments.

### Questions
see the weakness

### Soundness
3

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
3

### Summary
This paper focuses on an important problem in image generation, that the inclusion of reasoning can reduce the semantic fidelity of the generated images. It identifies that this could be caused by the misalignment of the model’s generated reasoning and its final image. To solve this problem, the authors propose bidirectional attention between modalities at every denoising step, and optimize alignment along the denoising trajectory. Experimental results show that the proposed method achieves significant improvement over baseline methods.

### Strengths
1. The proposed ParaBench is an effective tool for the analysis of thinking-aware image synthesis.  The finding regarding the strong correlation between performance degradation and poor alignment between the generated modalities is interesting and insightful. 
2. This paper explains the method design clearly, with insightful motivations. 
3. Visualization results in Figure 5 are impressive, showing the improvement in challenging scenarios such as the compositional settings.
4. The paper is well-written and easy to follow.

### Weaknesses
1. This paper shows the improvements of the paper based on the proposed ParaBench (Table 2). How about standard public benchmarks used in the original Bagel paper, such as  GenEval[1], WISE[2], GEdit-Bench[3]?

[1]  Geneval: An object-focused framework for evaluating text-to-image alignment. NeurIPS, 2023.

[2] Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

[3] Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

2.  Figure 5 shows some results of the proposed method.  What is the failure case of the proposed method?

### Questions
Please refer to the weaknesses.

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
3

### Summary
Based on the observation that existing AR models can suffer from error propagation during their thinking trajectory, this paper 1) proposes a benchmarking focusing on text image quality and output alignment, 2) develops a discrete-diffusion-based approach for parallel denoising of both text and images, and 3) introduces parallel RL to further optimize the intermediate cross-modal synergy.

### Strengths
The paper is well structured and clearly motivated. It offers a thorough investigation that includes new benchmarking, curated training datasets, new model design, and an RL protocol aimed at addressing the problem. The proposed MMaDA-Parallel model performs on par with SOTA open-source model that is trained on more data.

### Weaknesses
Normally, the decoding process only has one scheduler. In this paper, two schedulers are used for each modality. Could the authors give a more systematic guarantee of why we can assume the independence between each modality and why the alignment of text image modality would enable independent, parallel generation to work better than any-order joint generation?

### Questions
1. Please see Weaknesses.

2. In Table 3 (Table 6, 7 as well), why is MMaDA-Parallel decoding compared with the sequential decoding baseline that generates text before images? Why is it not compared to any-order joint decoding?

### Soundness
4

### Presentation
4

### Contribution
4
