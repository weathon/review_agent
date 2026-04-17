# Enhancing Multi-Modal LLMs Reasoning via Difficulty-Aware Group Normalization

- Decision: Reject
- Scores: 8, 4, 2, 4

## Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) and Group Relative Policy Optimization (GRPO) have significantly advanced the reasoning capabilities of large language models. Extending these methods to multimodal settings, however, faces a critical challenge: the instability of std-based normalization, which is easily distorted by extreme samples with nearly positive or negative rewards. Unlike pure-text LLMs, multimodal models are particularly sensitive to such distortions, as both perceptual and reasoning errors influence their responses. To address this, we characterize each sample by its difficulty, defined through perceptual complexity (measured via visual entropy) and reasoning uncertainty (captured by model confidence). Building on this characterization, we propose difficulty-aware group normalization, which re-groups samples by difficulty levels and shares the std within each group. Our approach preserves GRPO's intra-group distinctions while eliminating sensitivity to extreme cases, yielding significant performance gains across multiple multimodal reasoning benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the instability of standard deviation-based normalization in GRPO for multimodal LLMs, where extreme samples with nearly uniform rewards distort training. The authors propose difficulty-aware group normalization that re-groups samples based on visual entropy and model confidence. The method preserves intra-group distinctions while eliminating sensitivity to extreme cases.

### Strengths
1. The paper identifies a clear limitation in existing GRPO methods: std-based normalization becomes unstable when groups contain mostly positive or negative rewards.
2. Based on the identified limitation, the paper designs a interest algorithm that leverages the characteristics of multimodal models by performing additional grouping based on model uncertainty and inherent image complexity to calculate advantages. This algorithm demonstrates novelty in addressing multimodal-specific challenges.
3. The algorithm's approach is not limited to multimodal settings, and instead of simply expanding rollout sizes, the paper proposes an method for forming groups based on additional criteria. This may be interested to community for further exploration in various domains.

### Weaknesses
1. The method introduces more hyperparameters that may require careful tuning, including the weighting coefficients (α_Ori, α_Percep, α_Reason) and the percentile thresholds for grouping.

### Questions
1. How do the authors validate that their difficulty metrics actually capture the intended concepts? 
2. Could the authors provide experimental results about hyperparameter sensitivity?

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
This paper observes that in the MLLM RL training process, the limited number of rollouts causes the sampling distribution of the advantage to be significantly affected by variance fluctuations. To address this, the paper proposes grouping images and model-generated texts according to different levels of “difficulty,” and replacing group-level variance estimation with batch-level variance computation.

### Strengths
- This paper identifies the abnormal impact of variance on advantage estimation in MLLM reinforcement learning when the number of rollouts is insufficient, and proposes several mitigation strategies.
- It achieves accuracy improvements on MathVista, MathVision, and WeMath benchmarks.
- It attains comparable or superior performance to prior work while using only 2K training samples.

### Weaknesses
- The core idea of the paper is to address the bias in reward normalization under a limited number of rollouts by combining batch-level processing with difficulty grouping. However, it does not compare with methods that directly modify the normalization strategy, such as batch-level normalization (e.g., Reinforce++) or no normalization (e.g., Dr.GRPO).
- It remains unclear how performance differs when the number of rollouts increases or decreases (e.g., from 8 to 32 or 2), where the variance effect would be reduced. If the main motivation lies in normalization bias, a comparison with simply increasing the number of rollouts is necessary.
- The baseline results reported in Table 2 are problematic — for instance, R1-VL-7B is based on Qwen2, not Qwen2.5, which may cause confusion, and although Vision-R1-7B provides an available checkpoint, its evaluation results are not reported by the authors.
- The values of the three α-weights in Equation (20) are not specified in either the main text or the appendix, and the paper lacks ablation studies or comparisons analyzing different weight settings.

### Questions
- The normalization bias may evolve during training — for example, as training progresses, cases with “8 rollouts but only 1 incorrect answer” become increasingly common. It remains unclear how the Perceptual and Reasoning difficulty groups change throughout the training process.
- Compared with group-level variance computation, the paper does not provide sufficient analysis of how the training dynamics (e.g., reward and advantage trajectories) behave. How does the advantage fluctuation compare to the original approach?
- Table 2 shows that the proposed method achieves greater improvements with fewer data samples. Would performance continue to increase if the dataset size were scaled up to over 10K samples? If not, does this imply that the selected Geometry subset is particularly favorable for achieving high scores on these benchmarks?
- Figure 3 presents results across 4–48 reasoning difficulty groups, but given the batch size, when the number of groups exceeds 40, each group may contain only a few samples. Could this introduce statistical bias? If so, might the observed improvement on HallusionBench be attributed to enhanced self-confidence effects rather than genuine reasoning gains?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes difficulty-aware group normalization, a method to enhance the stability and effectiveness of reinforcement learning for multimodal large language models (MLLMs). The authors first identify that std-based reward normalization in GRPO is highly sensitive to extreme samples, causing instability in multimodal reasoning tasks. To address this, they introduce a difficulty-based re-grouping strategy that characterizes each sample’s difficulty through perceptual complexity and reasoning uncertainty. Samples are re-grouped by difficulty levels, and normalization is shared within each group to reduce sensitivity to outliers while preserving intra-group distinctions. The authors conduct experiments on multimodal math and hallucination benchmarks to validate the proposed method.

### Strengths
- The paper is overall clear and easy to follow.
- To the reviewer's knowledge, the proposed difficulty-based regrouping method is novel and conceptually straightforward.

### Weaknesses
- The definition of perceptual difficulty is not well validated. Why are images with higher entropy of eigenvalues of their image features considered perceptually harder? Moreover, why is perceptual difficulty assumed to be LLM-agnostic?

  Similarly, for reasoning difficulty, why is entropy used instead of a more direct measure such as average accuracy?

- Lack sufficient ablation analysis.

  - How sensitive are the results to the weighting coefficients of different normalized advantages?
  - What is the effect of group size on performance and stability?

- The proposed method combined with GRPO does not appear to improve over NoisyRollout when trained on the same amount of data. It remains unclear how the proposed approach achieves effectiveness compared to other baseline methods.

- Minor clarity issues:  There are two "+ours" in Table 2. It would be clearer if the authors explicitly distinguished their differences.

### Questions
Please refer to weaknesses sections for details.

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
3

### Summary
The paper identifies a key limitation of GRPO when applied to multimodal reasoning tasks—the instability of standard-deviation-based normalization due to extreme reward outliers. Unlike text-only LLMs, multimodal models suffer amplified distortions since both perceptual and reasoning errors affect their outputs. To mitigate this, the authors introduce a difficulty-aware group normalization method. Each sample is first characterized by its perceptual complexity (via visual entropy) and reasoning uncertainty (via model confidence), then grouped by difficulty level. Standard deviation is shared within each difficulty group, preserving relative ranking within groups while stabilizing learning against outliers. Though, I appreciate the idea (especially behind the visual entropy), I think many claims need to validated.

### Strengths
1. The paper presents a well-motivated approach to improve GRPO stability by dividing samples into difficulty-based groups and computing group-specific normalization factors.
2. The use of image entropy to quantify perceptual difficulty is an insightful way to model visual reasoning complexity.

### Weaknesses
I appreciate the work. Here are my comments to improve further,

1. The paper’s core premise—that std-based group normalization is highly sensitive to extreme samples—lacks concrete empirical evidence. The authors should analyze when and how such cases occur, as infrequent occurrences might naturally average out during training, reducing the claimed impact of this issue.

2. The use of image entropy as a proxy for perceptual reasoning difficulty is not clearly justified. While entropy of eigenvalues after PCA may capture visual diversity or texture complexity, it does not directly indicate the model’s perceptual reasoning error. The observed difficulty might instead reflect dataset bias—i.e., the model having seen more or fewer examples of such images during training—rather than inherent perceptual complexity.

3. “This formulation reflects the model’s internal confidence: High and consistent L(Qs) indicates a reliable reasoning chain, whereas low or fluctuating L(Qs) reflects epistemic uncertainty, implying that more challenging reasoning sample.” — I think this motivation needs validation. How is a lower-confidence sample associated with a more challenging reasoning sample? This should be demonstrated through quantitative analysis or supported by theoretical justification (which may be difficult).

4. Could you please explain the idea behind the unified formulation for advantage function across difficulty groups, equations 12, 13, 18, 19?

5. Based on the motivation, the unified advantage function across reasoning types in equation 20, why do we still need the original advantage function?

6. Could you please provide statistical significance tests for table 1 to confirm that the improvement gains are meaningful especially on MathVision, MathVista, WeMath, and HallusionBench?

7. “Compared with those either distilled from large-scale chain-of-thought data or employing complex data augmentation strategies, our method, utilizing only 2.1k train- ing samples, achieves comparable or even superior performance, significantly demonstrating our effectiveness.” - Which component in the proposed method leads to this sample efficiency? Authors can elaborate on it for further understanding of this study.

### Questions
Please see above.

### Soundness
2

### Presentation
3

### Contribution
2
