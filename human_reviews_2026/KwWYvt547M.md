# Pragma-VL: Towards a Pragmatic Arbitration of Safety and Helpfulness in MLLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Multimodal Large Language Models (MLLMs) pose critical safety challenges, as they are susceptible not only to adversarial attacks such as jailbreaking but also to inadvertently generating harmful content for benign users. While internal safety alignment via Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) is a primary mitigation strategy, current methods often face a safety-utility trade-off: they either refuse benign queries out of excessive caution or overlook latent risks in cross-modal interactions. To resolve this, we introduce Pragma-VL, an end-to-end alignment algorithm that enables MLLMs to pragmatically arbitrate between safety and helpfulness. First, we enhance visual risk perception with a novel cold-start SFT stage. This is achieved by applying risk-aware clustering to the visual encoder and using an interleaved dataset of risk descriptions and high-quality data. Second, we introduce a theoretically-guaranteed reward model that leverages synergistic learning. We train it with a novel data augmentation method that assigns dynamic weights based on the queries, enabling contextual arbitration between safety and helpfulness. Extensive experiments show that Pragma-VL effectively balances safety and helpfulness, outperforming baselines by 5% to 20% on most multimodal safety benchmarks while preserving its general capabilities in areas such as mathematics and knowledge reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Pragma-VL, a VLM designed to balance helpfulness and harmlessness. This work introduce: (1) A PragmaSafe dataset that annotates both attributes (helpfulness & harmlessness) along with a context-dependent weight vector W=[w_h, w_s] for each sample, capturing human preference trade-offs in different scenarios. (2) A multi-objective reward model with two attribute heads and a meta-voter MLP that aggregates the scores into a reward. (3) A reinforcement learning stage based on GRPO using the learned holistic reward as feedback to fine-tune the Pragma-VL policy.

### Strengths
1. The paper provides one of the first complete end-to-end pipelines adapting the classic SFT-GRPO workflow to multimodal safety alignment, including dataset design, reward modeling, and reinforcement optimization. The structure is technically coherent and practical for future extensions.

2. The paper theoretically analyzes why jointly training multiple attribute-specific heads can outperform single-objective or sequential training, under the assumption of positively correlated gradients.
Although the assumption is strong, the analysis is insightful and motivates further research on gradient interaction and multi-attribute alignment.

3. From dataset construction (PragmaSafe) to reward model decomposition and aggregation, the overall pipeline is clear, modular, and well-motivated.

### Weaknesses
1. The theoretical justification for the superiority of parallel multi-head reward modeling critically relies on the assumption that  
$\mathbb{E}[(\nabla_\theta r_s)^\top(\nabla_\theta r_k)] > 0,$
i.e., gradients between different objectives are positively correlated.  
However, the paper’s main case — balancing helpfulness and harmlessness — is a prototypical trade-off scenario, where these gradients are often negatively correlated in practice.  
This discrepancy raises serious concerns about the applicability of the theorem to the proposed task.

2. While the paper includes an ablation on the overall cold-start pipeline and the reinforcement learning stage, it does not disentangle the contributions of the two internal phases within cold-start.  
It remains unclear how much improvement each phase contributes to the final performance.

### Questions
See weakness above.

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
The paper introduces Pragma-VL, a new framework to solve the safety-utility trade-off in Multimodal Large Language Models (MLLMs). Current models often apply rigid, static safety policies, making them either overly cautious (unhelpful) or dangerously compliant. Pragma-VL enables models to pragmatically arbitrate between safety and helpfulness based on context. It achieves this through two key innovations:

1. A risk-aware "cold-start" phase that enhances the model's ability to perceive visual dangers.
2. A dynamic policy alignment stage using a novel dataset called PragmaSafe, which contains context-dependent preference weights. This trains a parallel reward model to provide a nuanced, prompt-regulated signal during reinforcement learning.

Experiments show that Pragma-VL significantly improves performance on safety and helpfulness benchmarks by 5-20% without degrading the model's general capabilities, effectively moving beyond fixed safety rules towards more robust, context-aware AI.

### Strengths
1. This paper is motivated by an important research problem: enabling MLLMs to dynamically arbitrate the helpfulness-safety trade-off. This is critical as focusing either on safety or helpfulness is inadequate. 
2. This paper improves the ability of the visual encoder to perceive safety severity, which is largely ignored when training existing vision encoder.

### Weaknesses
1.Many intuitions, explanations and motivations are missing when formulating the contextual data augmentation (Equation 1). For example, why do we need to sample the adjustment magnitude from a gaussian distribution? In addition, it is unclear why larger difference in variance could suggest a larger adjustment magnitude. 


2. A clear formulation of parallel rewards are missing. The authors propose reward models with parallel rewards, along with other variants such as sequential and single. However, there are only pictorial comparisons between these methods, which makes the reviewers confused about how the rewards are modelled and optimized. For example, what are the data flows in parallel, sequential and single reward models? Is r_\theta(x,y) a scaler or a vector?. If it is a vector,  what does it contain? Are the preference labels (win/loss) from the original dataset or derived from annotation in Sec. 3.1? Therefore, a clear formulations with math notations are required to better differentiate the variants.


3. It is unclear which reward is used for policy update. Is it the scaler reward (helpfulness and harmlessness) or the vectorized reward? If it was the vectorized one, how to convert them to advantages compatible for policy update? 

In summary, to the reviewer, many important details required to fully evaluate this paper are missing.

### Questions
Please see weakness.

### Soundness
2

### Presentation
1

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
The paper presents Pragma-VL, a framework designed to balance safety and helpfulness in Multimodal Large Language Models (MLLMs). Pragma-VL addresses these issues through a dynamic, context-aware approach, moving beyond static safety policies. This framework includes three key components: PragmaSafe, a data augmentation method for context-dependent preference labels; a cold-start Supervised Fine-Tuning phase to improve visual risk perception; and a parallel reward model for dynamic arbitration between safety and helpfulness. Experiments show Pragma-VL outperforms existing methods across various benchmarks, maintaining general capabilities while effectively managing safety and helpfulness. This work advances the development of more robust, value-aligned multimodal AI systems.

### Strengths
1. The paper introduces a novel data labeling and augmentation method through the PragmaSafe approach, enhancing context-dependent preference labels.
2. The Pragma-VL framework provides a dynamic, context-aware solution to the critical trade-off between safety and helpfulness in MLLMs which is really important to the community
3. The authors conducted comprehensive experiments to validate the performance of Pragma-VL across various benchmarks.
4. The framework retains strong performance on general VQA tasks, ensuring effectiveness in diverse scenarios.

### Weaknesses
The work remains limited by comparatively narrow model validation, heuristic cold-start design, reliance on GPT-4o annotations, lack of comparison to newer multi-objective baselines, homogeneous benchmarking, among a few others. These are issues that future work should address through broader empirical validation, human-calibrated evaluation, and open data release.

### Questions
1. Safe RLHF-V also proposed an algorithm for safe and helpful trade-off. But in the experiment, authors only set Beavertails-V_harm or Beavertails-V_help as baseline.
2. The results in experiments for Beavertails-V_harm are strange. Usually we consider the model safer with lower ASR, while here although the model has a lower ASR, it has a lower win rate in dimension of harmless which may indicate that the model is unsafer. There seems to be a conflict between these two results . Could you please explain it more?

### Soundness
2

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
This paper addresses a critical problem in Multimodal Large Language Models (MLLMs): **the tradeoff between safety and usefulness**. The authors note that static safety policies often fail, leading to both excessive refusals and risk blindness.

To solve this, the paper proposes Pragma-VL, an end to end alignment framework for dynamic and "pragmatic" arbitration based on context. The framework has two core innovations:

**MLLM Cold Start**: A specialized pre alignment stage to address the model's inherent "visual risk blindness" using Risk Aware Contrastive Learning and Supervised Fine Tuning (SFT) on cross task datasets.

**Policy Alignment via Parallel Rewards**: This uses reinforcement learning (GRPO) and a theoretically grounded, parallel, multi head reward model. This model is trained on a new dataset, PragmaSafe, which is annotated with context dependent safety usefulness weights using GPT-4o.

Experiments on the Qwen2.5-VL-7B and Llava-1.5-7B models show that Pragma-VL significantly outperforms baselines on multiple safety benchmarks (especially the SIUO benchmark for cross modal risks). Crucially, it also maintains the model's capabilities on general purpose benchmarks (e.g., GQA, ScienceQA).

### Strengths
- **Methodological Completeness**: Pragma-VL is a well designed, end to end system. It correctly identifies that policy alignment (RL) cannot fix fundamental perceptual issues. Thus, the proposed "cold start" SFT stage (first addressing visual risk perception before connecting to language cognition) is methodologically sound and rigorous.

- **In depth Analysis of the Reward Model**: The paper's exploration of reward model (RM) architectures (single objective vs. sequential vs. parallel) is a highlight. The authors empirically show the superiority of the parallel architecture (Table 1) and provide theoretical support (Theorem 1) for its synergistic learning, adding credibility to their approach.

- **Addressing Cross Modal Risks**: The large improvement on the SIUO benchmark (e.g., Llava's safety rate increasing from 14.37% to 55.42%) strongly shows the method addresses cross modal risks when MLLMs handle benign image text combinations.

### Weaknesses
- **Annotation Quality and Bias Risk**: Over reliance on AI annotation without human verification. The core "pragmatic arbitration" capability depends entirely on the PragmaSafe dataset. Its context weight labels are generated by GPT-4o. This heavy reliance on one AI model introduces two problems: (a) Bias Propagation: Systematic biases from GPT-4o may be propagated and solidified in Pragma-VL. (b) Lack of a Gold Standard: The paper trusts the AI annotations but lacks a human agreement study. Using human expert annotations as a gold standard to cross validate the AI label quality and consistency is crucial.

- **Lack of Robustness in Data Aggregation**: The variance calculation is based on only 5 samples from a single model (GPT-4o). This variance might only reflect GPT-4o's sampling uncertainty, not true "task uncertainty." A more robust consensus could be achieved by using a model ensemble (multiple models like Gemini or Qwen) for annotation, which would likely produce a less biased dataset.

- **Unclear Attribution for Preserving General Capabilities**: The paper attributes the preserved capabilities (Table 3) to its "pragmatic arbitration" framework. However, the appendices note that large amounts of general capability data (e.g., MathV360K, VQAv2) were intentionally mixed into the PragmaSafe dataset and RL training data. Thus, it is unclear how much of this preservation is due to the algorithm's design versus the data mixing. The paper does not fully clarify this.

### Questions
1. When using GPT-4o to generate context weights, did you observe poor or inconsistent performance on specific categories (e.g., subtle bias, sarcasm, or complex "gray area" queries)? Has the Pragma-VL model inherited these specific failures?

2. The MLLM Cold Start (Sec 3.2) has two phases: Risk Aware Contrastive Learning (Phase 1) and Risk Aware SFT (Phase 2). If Phase 1 were skipped, performing only Phase 2, what would be the impact on final safety performance (especially on SIUO)? This is just a question; no new experiments are required.

### Soundness
3

### Presentation
3

### Contribution
2
