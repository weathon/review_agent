# AdvChain: Adversarial Chain-of-Thought Tuning for Robust Safety Alignment of Large Reasoning Models

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Large Reasoning Models (LRMs) have demonstrated remarkable capabilities in complex problem-solving through Chain-of-Thought (CoT) reasoning. However, the multi-step nature of CoT introduces new safety challenges that extend beyond conventional language model alignment. We identify a failure mode in current safety CoT tuning methods: the snowball effect, where minor reasoning deviations progressively amplify throughout the thought process, leading to either harmful compliance or excessive refusal. This effect stems from models being trained to imitate perfect reasoning scripts without learning to self-correct. To address this limitation, we propose AdvChain, an alignment paradigm that teaches models dynamic self-correction through adversarial CoT tuning. Our method involves constructing a dataset containing Temptation-Correction and Hesitation-Correction samples, where models learn to recover from harmful reasoning drifts and unnecessary cautions. Extensive experiments show that AdvChain significantly enhances robustness against jailbreak attacks and CoT hijacking while substantially reducing over-refusal on benign prompts, achieving a superior safety-utility balance without compromising reasoning capabilities. Our work establishes a new direction for building more robust and reliable reasoning models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper claims that reasoning models are not adversarially robust because they are susceptible to errors in their reasoning trace, which can "snowball", leading to the eventual wrong decision of whether to refuse or comply with a request. The authors suggest a data augmentation method to alleviate this: they construct reasoning traces that contain artificially inserted mis-steps, followed by inserted corrections. This results in models that are more robust to "CoT-hijacking", where the CoT is prefilled in a mis-stepped way.

### Strengths
- Clear presentation
  - The paper is well-written, and has well-designed figures.
- Comprehensive evaluation
  - The paper evaluates on diverse benchmarks covering general safety (HarmBench, StrongReject), adversarial robustness (SafeUnlearning, WildJailbreak), over-refusal (XSTest), and reasoning preservation.

### Weaknesses
- Lacks discussion of related work
  - In particular, [Qi et al., 2024](https://arxiv.org/abs/2406.05946) must be cited and discussed. The main idea in that paper is very similar to the main idea in this paper - namely that naive SFT-based refusal training is fragile, in particular to "pre-filling" attacks, and that doing simple adversarial data augmentation can improve robustness.
  - I'd also like to understand how this work relates to RL-native approaches, such as Deliberative Alignment ([Guan et al., 2024](https://arxiv.org/abs/2412.16339)).
- Section 3 does not adequately support the claim that "minor reasoning deviations progressively amplify throughout the thought process"
  - The analysis in section 3 filters for samples with either {low initial harmfulness and high final harmfulness} or {high initial harmfulness and low final harmfulness}; the corresponding plots slope up and down, respectively. In my opinion, this is not sufficient evidence for the claim that models generally get off track from minor deviations, and that their problem is lack of error correction.
- Missing technical details for reproducibility
  - Details are not provided on how to generate "hesitation-correction" samples, "temptation-correction" samples, or CoT-Hijacking samples. For example, the model used to generate these samples is not specified.

### Questions
- Relation to [Qi et al., 2024](https://arxiv.org/abs/2406.05946)
  - Have you read this paper? How does your paper relate to it? I recommend adding discussion of relevant related work.
- Conclusions in Section 3
  - From the "snowball plots", how can you conclude this picture? "a process where the model initiates a safe and valid reasoning path, but a minor deviation in an intermediate step acts as a seed for the snowball. Once flawed step occurs, it begins to gather momentum, progressively corrupting subsequent reasoning and amplifying the initial error into a fully harmful conclusion and output."
  - When filtering for initially low + finally high, what else could the graph possibly look like, if not for an upward sloped plot? Why can we make deductions based on that plot?
- Technical details
  - How are "hesitation-correction" samples, "temptation-correction" samples, or CoT-Hijacking samples generated? What model is used to generate them?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes AdvChain, an adversarial CoT tuning method to improve the safety and robustness of LRMs. It identifies the issue of “Snowball Effect,” where small reasoning deviations escalate into harmful compliance or excessive refusals. AdvChain trains models to self-correct through two types of synthetic data: Temptation-Correction and Hesitation-Correction. Experiments show that AdvChain enhances safety and reduces over-refusal while preserving reasoning performance, outperforming baselines trained on far larger datasets.

### Strengths
* The identification of the “Snowball Effect” in safety alignment for LRMs is original and highlights a critical, previously underexplored issue.
* Stepwise reasoning analysis is clear and convincing with proper visualizations, offering supports for relevant claims.
* The construction of Temptation-Correction and Hesitation-Correction datasets for adversarial tuning provides a simple yet effective mechanism for dynamic self-correction.
* Experimental results are strong, showing improvements in safety robustness and reduced over-refusal, while maintaining reasoning accuracy and high data efficiency.

### Weaknesses
* The method relies on a powerful external model to generate adversarial reasoning traces and construct two datasets for CoT tuning. While this is a common approach, it raises questions about the scalability and potential bias. It would strengthen the work to include an analysis of how performance varies when the teacher or labeler is changed, for instance, by replacing it with a different model, using a weaker labeler, or substituting it with self-correction. Demonstrating that the proposed method remains effective under such variations, or that self-generated corrections yield comparable results, would provide stronger evidence for the method’s robustness.
* Some details and statistics of the dataset construction should be reported. What is the cost for external labeling ? What is the ratio of valid data points after construction ? How are the training data selected ? 
* In the ablation studies, the analysis on "Snowball Effect" should be conducted also on the models trained with AdvChain to verify the effectiveness of the proposed approach. Only comparing the datasets in Figure 3 is insufficient.
* Several relevant papers in this topic are missing in the related work, which can be included in revision.
[1] STAIR: Improving Safety Alignment with Introspective Reasoning
[2] Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking
[3] GuardReasoner: Towards Reasoning-based LLM Safeguards

### Questions
There are some writing issues, including minor typos and occasional duplicated or repetitive paragraphs. Some are listed below:
1. at the end of line 161, a typo of "he" should be "the"
2. from line 122-128, the content in this paragraph contains seemingly duplicated information as previous paragraphs, making readers confused. Maybe this part is leftover drafts that were likely not deleted.
3. Some terms should be further clarified. What is a "small, initial deviation" in reasoning that incurs harmful contents ? Usually, in harmful reasoning, the model just begins to solve the problem through reasoning. Maybe the term can be better defined.

### Soundness
3

### Presentation
2

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
This paper introduces a new safety alignment method, AdvChain, which aims to address the Snowball Effect observed in Large Reasoning Models (LRMs) during Chain-of-Thought (CoT) reasoning — where small reasoning deviations can progressively amplify, resulting in either harmful compliance or over-refusal.
Experiments on DeepSeek-R1 and Qwen3 model families show that AdvChain effectively reduces attack success rates and over-refusals, achieving performance comparable to a 15k-sample baseline with only 1k training samples.

### Strengths
- **Insightful problem formulation:** The paper identifies and systematically analyzes the “Snowball Effect,” revealing two key failure modes in safety alignment.
- **Balanced trade-off:** AdvChain achieves an appealing balance between safety robustness and practical helpfulness.

### Weaknesses
1. The paper does not clearly distinguish whether the effect arises from long reasoning chains (i.e., cumulative exposure bias) or from the intrinsic multi-step nature of CoT itself.

2. Both Temptation and Hesitation samples are synthetically generated by a teacher model, which may not reflect real-world reasoning errors. Such manually injected “mistakes” could cause the model to learn patterned pseudo-errors rather than genuine self-correction, or even introduce confusion that harms general reasoning ability.

3. The described over-refusal snowball seems artificially constructed rather than naturally occurring. It remains unclear whether such hesitation amplification appears in spontaneous model outputs. If this phenomenon is naturally observed, the paper should provide quantitative or qualitative analyses of how and where the model “enters” a harmful trajectory, ideally with illustrative examples.

4. Minor issues
   - Figure 3: typo “differnet” → “different”
   - Figure 4: missing right parenthesis in ASR

### Questions
The proposed method and evaluations all rely on explicit CoT prompting (e.g., “Let’s think step by step”).
It is unclear whether LRMs retain similar robustness without explicit CoT activation.
Have the authors tested AdvChain in non-CoT settings (e.g., direct Q&A or single-step reasoning)?
Does the self-correction behavior depend on the presence of a visible reasoning chain?

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
3

### Summary
In this paper, the authors address the safety challenges in Large Reasoning Models (LRMs). They identify a failure mode called the “snowball effect”, where small errors in early reasoning steps accumulate over time, leading to either harmful compliance or excessive refusal. To tackle this issue, the authors propose AdvChain, an alignment framework designed to enable self-correction during the generation process. AdvChain includes two types of training samples: Temptation-Correction, where the model learns to recover from harmful reasoning drifts, and Hesitation-Correction, where it learns to overcome unnecessary caution. Experimental results show that AdvChain enhances robustness against jailbreaks and Chain-of-Thought (CoT) hijacking.

### Strengths
(1) Interesting topic: The problem of generation safety in LRMs is both interesting and important. It is critical for the real-world deployment of LLMs.

(2) Clarity: The paper is well written and easy to follow.

### Weaknesses
(1) Concerns regarding evaluation performance:
 In Table 1, the evaluation results show that (a) the baseline method RealSafe-R1 consistently outperforms the proposed method. Although the authors mention that RealSafe-R1 uses a larger training dataset, I would encourage the authors to scale their dataset to a comparable size in order to evaluate the scalability of their approach. (b) The performance of the proposed method is not consistent across different model backbones. When trained with the DeepSeek-R1 series, the proposed method outperforms the baseline STAR-1; however, when trained with the Qwen3 series, it performs worse. A detailed analysis of this discrepancy is highly recommended.

(2) For additional concerns, please refer to the question section.

### Questions
(1) In Section 3, the authors discuss the snowball effect. However, the analysis relies on a scoring function applied to reasoning steps and depends on how the model assesses the harmfulness/helpfulness of each step. As a result, it's unclear to what extent this pattern is influenced by the verifier versus the evaluated model itself. I suggest the authors include 2–3 example trajectories that demonstrate this scoring trend to better support the argument.

(2) In the experimental evaluation, could you consider scaling the dataset size to 15K, or at least to a point where your method achieves comparable performance with RealSafe-R1? This would enable a more direct comparison with that baseline.

(3) For the Qwen3 series models, why does your method perform significantly worse than STAR-1 (1K) across nearly all evaluation benchmarks and metrics?

(4) Why does the proposed dataset have varying effects on different model series? For example, it improves performance when used with DeepSeek-R1-distilled models but leads to worse results with Qwen3 models.

(5) Could you also provide results for the remaining two models (DeepSeek-R1-1.5B and Qwen3-1.7B) in Table 4?

### Soundness
2

### Presentation
3

### Contribution
2
