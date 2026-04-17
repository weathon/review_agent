# Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization

- Decision: Accept (Poster)
- Scores: 4, 4, 2, 6

## Abstract
As Large Language Models (LLMs) are increasingly deployed in real-world applications, it is important to ensure their behaviors align with human values, societal norms, and ethical principles. However, safety alignment under Reinforcement Learning (RL) often suffers from forgetting learned general abilities, which is also known as the alignment tax. To address this issue, we introduce Null-Space constrained Policy Optimization (NSPO), a novel RL framework for LLM safety alignment while preserving their core abilities. The safety policy gradients are geometrically projected into the null space of general tasks, thereby mitigating the safety alignment tax.  In addition, we theoretically prove that NSPO preserves the model's original core capabilities, while still guaranteeing a descent direction for effective safety alignment. Extensive experiments demonstrate that NSPO outperforms existing methods by a large margin, achieving state-of-the-art safety performance without sacrificing accuracy on general tasks, including math, code, and instruction-following tasks.
Notably, NSPO is data-efficient and only requires 40% of public human-annotated safety data from PKU-SafeRLHF to achieve promising safety performance, without a large amount of mixed general tasks data in existing alignment methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the “safety alignment tax,” where improving safety often degrades general abilities. It introduces NSPO, a RL framework for safety alignment that projects safety policy gradients into the null space of general-task representations to preserve core skills while optimizing safety. The theory shows this projection is non-expansive in spectral norm, which supports stability, and that an appropriate step size yields a valid descent direction for the safety objective. The authors then evaluate NSPO across diverse safety and capability benchmarks.

### Strengths
1. The method is simple and direct: by constraining updates to the null space of capability features, NSPO exactly preserves the base model’s behavior on those features.
2. The projection is proven non-expansive (spectral norm bounded) and guarantees a valid descent direction with a suitable step size.

### Weaknesses
1. While the paper focuses on the “safety tax,” I question whether this framing is correct. Figure 1 suggests the primary effect is improved safety, with only marginal changes in general capability. In practice, RLHF with safety alignment tends to improve both helpfulness and harmlessness. The main concern is over-refusal, which does not by itself imply reduced general capability. And the evaluation on general capability should remove the effect of over-refusal.
2. The paper lacks comparisons with recent safety-alignment methods like [1,2] that use stepwise constraints to balance reward optimization and constraint satisfaction. Such approaches could likewise help limit adverse effects on general capability.
3. The paper argues NSPO removes the KL penalty because it conflicts with the safety objective, yet the implementation details and Appendix Table 4 still list the KL constant and mention the KL divergence.

[1] SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety

[2] Enhancing Safety in Reinforcement Learning with Human Feedback via Rectified Policy Optimization

### Questions
See weakness. I’m willing to raise my score if the central claim about the “safety tax” is clearly clarified.

### Soundness
2

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
This paper proposes Null-Space Constrained Policy Optimization (NSPO), a reinforcement learning (RL) framework designed to improve the safety alignment of large language models (LLMs) while mitigating the alignment tax—the degradation of general capabilities (e.g., math, coding, instruction following) during safety fine-tuning.
The key idea is to geometrically project the safety policy gradients into the null space of general task representations, ensuring that safety updates are orthogonal to the subspace spanned by general capabilities. This approach allows NSPO to enhance safety without sacrificing the original general capabilities of LLMs.
The authors provide theoretical guarantees showing that the null-space projection preserves the model’s general capabilities while still offering a valid descent direction for optimizing the safety objective.
Empirical results on two base models (Llama3-8B-Instruct and Qwen2.5-7B-Instruct) across seven safety and seven general capability benchmarks demonstrate that NSPO achieves state-of-the-art safety performance with minimal loss (<1%) in general capabilities. Notably, NSPO is data-efficient, requiring only 40% of the PKU-SafeRLHF safety data to reach strong performance and introducing only marginal computational overhead.

### Strengths
- The proposed idea is sound, and the authors provide extensive theoretical derivations.
- The empirical results demonstrate competitive performance of the proposed method.

### Weaknesses
- The effect of the projection is not clearly demonstrated by the ablation study. A comparison between 
    - NSPO (w/ projection) and
    - GRPO (w/o projection)
      - (1) original GRPO, and 
      - (2) GRPO w/o KL, using the modified Eq. (6) with $\hat{U}\hat{U}^\top$ replaced by $I$ in the paper

  should be provided.

### Questions
My current evaluation is borderline. I will revisit and potentially adjust my score following the authors’ rebuttal.

### Claim on Data Efficiency (40%)

- In the Abstract and Introduction, the authors claim that NSPO “only requires 40% of the public human-annotated safety data from PKU-SafeRLHF” to achieve strong safety performance.
However, this statement is not entirely clear. Which part of the paper should I refer to for details on this claim?
Is there an experiment or ablation that specifically compares performance with 40% versus 100% of the PKU-SafeRLHF data?

### Evaluation Metrics

- For evaluation, the authors employ three different automatic evaluators: GPT-4, Llama-Guard-3-8B, and ft-mistral-7b-instruct-v0.2-sorry-bench, depending on the benchmark.
Could the authors clarify why different evaluation models were used for different benchmarks instead of using a unified evaluator such as GPT-4 across all tasks?
Is this due to benchmark-specific requirements, cost constraints, or the limitations of GPT-4 in certain evaluations?

### Ablation Studies

1. **Effect of Projection** (related to weakness)

    The proposed null-space projection is claimed to preserve general capabilities while improving safety. However, the ablation analysis of the projection’s actual effect is not clearly presented. Is there a specific experiment comparing NSPO without the null-space projection? This comparison would help isolate the contribution of the projection mechanism itself.

2. **How to Compute $K$**

    The projection matrix is constructed using 1k randomly sampled general capability examples from the Alpaca training dataset.

    (a) Is there a particular reason for selecting 1k samples instead of using the full 52k Alpaca training samples, given that constructing $K$ is a one-time offline operation before training? Using more data would seem to provide a more stable representation of general capabilities.

    (b) Why was the Alpaca dataset specifically chosen for constructing $K$? Would other instruction-following or reasoning datasets (e.g., MMLU, GSM8K) yield similar or different results?

### Question on Empirical Results

- NSPO is mainly trained using the PKU-SafeRLHF and Alpaca datasets. However, according to Table 1, when applied to Llama3-8B-Instruct, NSPO does not show clear improvement on the PKU-SafeRLHF benchmark, whereas it achieves strong safety gains on the same benchmark for Qwen2.5-7B-Instruct. Could the authors clarify the reason for this discrepancy between the two base models? For example, is it related to differences in model architecture, baseline safety levels, or the interaction between the null-space projection and model representations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes the NSPO method to mitigate the "safety tax",  the detrimental effect of safety alignment on the core capabilities of language models. NSPO achieves this by geometrically projecting the safety gradient into an orthogonal direction (the null space) of general task gradients. This design ensures that safety performance is enhanced while preserving the model's fundamental abilities.

### Strengths
+ This paper addresses a critical challenge in safety alignment, the performance trade-off often termed the "safety tax." The exploration of the NSPO method to mitigate this tax is a novel and timely contribution.
+ The paper provides a solid theoretical grounding for the NSPO method, establishing its formal validity.

### Weaknesses
+ The presentation of the algorithmic foundations of NSPO lacks clarity.
+ A primary concern regarding NSPO is its safety guarantee, particularly when facing explicitly harmful prompts. 
+ I am skeptical of several counterintuitive results presented in the experiments. Their validity requires stronger justification beyond the provided code.
+ The paper lacks an experimental analysis of the key parameter, the representation dimension $d$.

### Questions
1. If I understand correctly, $\mathcal{J}$ in Figure 2 and Section 4.2 refers to the objective in Eq. (3), which is proposed as a solution to the primal constrained problem in Eq. (1). However, the connection between the two appears underspecified. Specifically, since Eq. (1) is a constrained optimization, it is unclear how the reward function rin the unconstrained objective (Eq. 3) is formally derived from the components of Eq. (1), namely the loss  $\mathcal{L}(\cdot; \cdot)$ and the general preference $\text{Pref}_G(\cdot)$. Could the authors clarify this derivation? Does this process involve a form of reward shaping [1]? 

2. While the intuition behind using an orthogonal projection to reduce the safety tax is clear, this approach may face a fundamental challenge with inherently unsafe queries. For such prompts, some trade-off where helpfulness is sacrificed for safety is necessary and expected. I am curious about how NSPO behaves in these critical cases. Does it guarantee robust safety refusal for explicitly harmful instructions, such as the examples found in the PKU-SafeRLHF dataset (e.g., "How can I blow up a car?")? To substantiate the safety guarantees of NSPO, I strongly recommend that the authors include a qualitative analysis, presenting the responses from both the initial model and the NSPO-tuned model to a few carefully selected, overtly harmful prompts. This would provide compelling evidence of its real-world safety alignment.

3. A potentially counterintuitive finding in Figure 1 requires clarification. It appears that DPO-H demonstrates superior safety performance compared to DPO-S. A convincing explanation for this unexpected outcome is necessary to bolster confidence in the overall conclusions of the study.

4. The results in Table 1 present a counterintuitive finding. Given that PKU-Safe and SafeRLHF originate from the same source of work [2, 3], one would expect the SafeRLHF algorithm to yield a safety improvement on its corresponding dataset. However, the results do not show this expected gain. The authors should analyze the potential reasons for this discrepancy. A plausible explanation could be a difference in the experimental setup compared to the original series of experiments in [2], such as the use of a different base model or reward/cost models. A clear explanation is necessary to validate the reliability of these comparative results.

5. The complexity analysis correctly identifies the representation dimension $d$ as a critical parameter affecting algorithmic performance. However, the experimental section omits details on how $d$ was chosen. Its value should be explicitly stated, and if it is a hyperparameter, the rationale behind its selection criteria should be explained. Furthermore, a sensitivity analysis of $d$ is necessary to understand the trade-offs between computational complexity (both time and space) and model performance, which is crucial for assessing the practical utility of NSPO.

[1] Ng, Andrew Y., Daishi Harada, and Stuart Russell. "Policy invariance under reward transformations: Theory and application to reward shaping." Icml. Vol. 99. 1999.

[2] Dai, Josef, et al. "Safe rlhf: Safe reinforcement learning from human feedback." arXiv preprint arXiv:2310.12773 (2023).

[3] Ji, Jiaming, et al. "Beavertails: Towards improved safety alignment of llm via a human-preference dataset." Advances in Neural Information Processing Systems 36 (2023): 24678-24704.

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
2

### Summary
This paper proposes Null-Space Constrained Policy Optimization (NSPO), a reinforcement learning algorithm designed to mitigate the degradation of general capabilities during safety alignment of LLMs. The core idea is to project the safety policy gradients into the null space of the general capability representations. The paper provides theoretical guarantees that the projected gradient remains a descent direction for the safety objective. Experiments on Llama3-8B and Qwen2.5-7B across multiple safety and general benchmarks show that NSPO achieves competitive safety performance while preserving performance on general tasks.

### Strengths
1. The paper addresses a critical and well-known problem in LLM alignment. The core idea of applying null-space projection to decouple safety and capability gradients is overall novel and represents a creative combination of null-space projection with modern RLHF/GRPO frameworks. The paper provides theoretical guarantees that the projected gradient remains a descent direction for the safety objective and is stable, which strengthens the methodological contribution.
2. The experimental results show that NSPO achieves competitive safety performance while preserving performance on general tasks, and introduces only minimal overhead compared to baseline GRPO.

### Weaknesses
1. The effectiveness of the null-space projection is deeply related to the general capability matrix K, which is constructed from 1,000 samples from the Alpaca dataset. The paper lacks a sensitivity analysis on how the selection, diversity, and domain of this data impacts the results. It remains unclear whether NSPO's performance generalizes if K is computed from a different domain (e.g., math, code), a smaller sample size that may not capture the full spectrum of general capabilities. What impact will the selection of different datasets have on performance? It is suggested to provide strategies for selecting datasets (such as diversifying as much as possible)

2. A critical hyperparameter in NSPO is the eigenvalue threshold used to filter eigenvectors and define the null space. This threshold directly controls the trade-off, where a higher threshold preserves a larger subspace for safety updates but risks infringing on general capabilities, while a lower threshold more rigidly protects capabilities but may overly constrain safety learning. It would significantly benefit from an experiment demonstrating how varying this threshold affects the crucial balance between safety performance and the preservation of general capabilities, thereby providing readers with principled guidance for its selection.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3
