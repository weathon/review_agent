# Critique-Guided Distillation for Efficient and Robust Language Model Reasoning

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Supervised fine-tuning (SFT) with expert demonstrations often suffers from the imitation problem, where models reproduce correct responses without internalizing the underlying reasoning. We propose $\text{C{\small RITIQUE-}G{\small UIDED} D{\small ISTILLATION} (CGD)}$, a multi-stage training framework that augments SFT with teacher-generated $\textit{explanatory critiques}$ and $\textit{refined responses}$. Instead of directly imitating teacher outputs, a student learns to map the triplet of prompt, its own initial response, and teacher critique into the refined teacher response, thereby capturing both $\textit{what}$ to output and $\textit{why}$. On mathematical reasoning benchmarks, $\text{CGD}$ achieves substantial gains across LLaMA and Qwen families: +15.0\% on AMC23 and +12.2\% on MATH-500 over CFT, while avoiding the format drift that plagues critique-based methods. Cross-family validation on Qwen2.5-Math-7B with diverse teachers (Claude Sonnet 3.7 to weaker open-source models) achieves state-of-the-art performance (50.4 avg, +22.6\% over base) with 144× less compute than RL methods. Critically, despite training on data containing no code, $\text{CGD}$ generalizes to out-of-distribution benchmarks:~+4.88\% on HumanEval (code generation), and preserved or improved performance on GPQA, MUSR, TruthfulQA, and BBH, while CFT suffers catastrophic forgetting (-21.3\% on IFEval). These results establish $\text{CGD}$ as a cost-effective intermediate training paradigm that can serve as a warm-start before reasoning SFT or RL, offering a scalable enhancement to modern LLM training workflows.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new training framework named "Critique-Guided Distillation" (CGD), which aims to enhance the reasoning abilities of language models by having the student model learn from teacher-generated explanatory critiques and refined responses, rather than simply imitating correct answers. This method incorporates a critique mechanism during training but requires only a single forward pass at inference time. Consequently, it achieves significant performance gains on multiple mathematical and general reasoning benchmarks while maintaining high efficiency and avoiding output format drift.

### Strengths
1.By consistently setting the training objective to generate "refined answers" rather than "critiques," the model maintains a standard instruction-following format at inference time. It achieves significant and consistent improvements on several challenging mathematical reasoning benchmarks, substantially outperforming strong baselines, demonstrating the method's effectiveness.
2.The paper demonstrates the robust performance of CGD across different model families, training datasets, (and under different hyperparameters) through extensive ablation studies.

### Weaknesses
1.Despite module-level ablation studies, the paper fails to clearly reveal the interactions between sub-modules and their marginal contributions to the performance gains.
2.Although inference efficiency is high, its multi-stage data generation process introduces significant up-front computational cost.
3.While CGD excels in mathematical and scientific reasoning tasks, its generalization ability on more open-ended, creative, or cross-modal tasks (such as creative writing, open-domain dialogue, complex summarization, or counterfactual reasoning) remains insufficiently validated.

### Questions
1.The paper notes that performance is influenced by the teacher model's quality. Besides using larger, stronger teacher models, are there plans or methods to automatically evaluate or filter low-quality critiques to mitigate the negative impacts of teacher model weaknesses?
2.If the style of the teacher model is very similar to or the opposite of the student model's style, what impact would this have on CGD's effectiveness?
The current multi-stage data generation process is resource-intensive. Are there future research directions aimed at simplifying this process? For example, exploring whether high-quality (initial answer, critique, refined answer) triplets can be generated via a single model or more efficient sampling strategies?
3.Have you analyzed which characteristics (e.g., pointing out specific error steps vs. giving high-level hints) of a critique are most critical for the student model's learning? Could you provide a more operational definition or metric for "critique quality"?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Critique-Guided Distillation (CGD), where a student model conditions on its own initial answer and a teacher critique to learn a refined answer. At inference, the student outputs the refinement in a single pass. Experiments on math benchmarks show improvements over SFT, distilled SFT, and CFT, while avoiding critique-format drift. The authors also claim better efficiency than RL-based methods.

### Strengths
1. Motivated by limitations of vanilla SFT and CFT

2. Improves several math-reasoning benchmarks compared to SFT and CFT.

3. Preserves general instruction-following where CFT degrades it.

### Weaknesses
1. Insufficient motivation. While CGD exhibits empirical gains, the paper does not convincingly explain why conditioning on critiques during training, but omitting them at inference, should improve from-scratch reasoning. During training, the student learns to rely on critique signals that are not available at inference. The paper does not explain how critique-conditioned refinements translate into unconditional answer generation, nor does it analyze whether this reliance introduces brittleness.

2. Susceptible to the same issues acknowledged for CFT. CGD may analogously drift toward producing improved answers relative to a latent critique signal if trained extensively, and it fundamentally depends on high-quality critiques. No evidence is provided that CGD is robust to noisy, biased, misleading, or answer-leaking critiques, despite the conceptual similarity to CFT.

3. Potential critique leakage. Critiques can implicitly or explicitly reveal the correct answer or key intermediate steps. Without mitigation or measurement, the observed gains may partially reflect teacher leakage rather than genuine reasoning improvements.

4. Missing comparison to inference-time self-correction by accuracy. The paper discusses latency advantages but omits accuracy comparison to strong self-refine baselines. Without this, the contribution’s significance and practical trade-offs are unclear.

5. Lack of statistical rigor. No repeated runs, variance, confidence intervals, or significance testing are reported. Results may not be reliable given known variance in reasoning benchmarks.

6. Narrative inconsistency in baseline comparison. The “second-best” baseline is often distilled SFT, not CFT, contradicting text that positions CFT as the primary competitive method.

7. Ablation (§4.2.2). Removing critiques while forcing refinement of an incorrect answer is expected to underperform vanilla SFT, making the ablation unsurprising and uninformative.

8. Figure clarity and consistency issues. Figure 1 does not clearly denote teacher vs. student outputs, and Figure 3 does not show that the student receives y′, contradicting the description and Algorithm 1.

9. Prompts and templates omitted. Critique-based methods are highly prompt-sensitive. The absence of templates or formatting conventions limits reproducibility and makes it difficult to judge critique quality.

10. Fragmented RL comparison. RL results are isolated in a separate subsection rather than integrated into the main results tables, making efficiency/performance comparisons harder to interpret.

### Questions
1. Can you provide deeper evidence or analysis explaining why conditioning on critiques during training (but removing them at inference) improves unconditional reasoning? What internal behaviors does the model learn?

2. How do you know the student is not implicitly relying on critique-style patterns that will not exist at inference? This concern may worsen with longer CGD training.

3. How does CGD perform when critiques are noisy, biased, partially incorrect, or misleading? Do you have experiments quantifying this sensitivity?

4. How did you ensure critiques do not reveal the answer directly (explicitly or implicitly)? Can you provide statistics on answers appearing in critiques?

5. Did you observe signs of the model drifting toward “refinement-style” outputs (e.g., suggesting revision) when trained longer? 

6. Can you compare CGD’s accuracy (not only latency) to multi-pass self-correction/self-refine baselines? Is the trade-off still favorable?

7. Can you report variance across multiple seeds, confidence intervals, or significance tests to substantiate improvements on high-variance reasoning tasks?

8. Why is CFT described as the primary baseline when distilled SFT appears stronger in practice?

9. Did you analyze cases where CGD underperforms? Any patterns in failure?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a fine-tuning framework that enhances reasoning in language models by teaching them to self-correct using teacher-generated critiques during training. Instead of merely imitating teacher outputs, CGD trains a student model to map from its own initial response and a teacher-generated critique to the refined answer, capturing both the “what” and the “why” behind correct reasoning.

### Strengths
### **1. Integrating Critique and Correction**

The paper introduces a well-motivated training framework that unifies critique understanding and refinement learning within a single fine-tuning stage.

### **2. Strong Empirical Performance Across Multiple Benchmarks**

The authors provide comprehensive experimental validation across both mathematical and general reasoning benchmarks.
CGD achieves large and consistent gains (e.g., +15% on AMC23, +12.2% on MATH-500) over SFT and CFT baselines.

### Weaknesses
1. **Novelty Concern and Overlap with Prior Work**
   Although CGD is presented as a novel fine-tuning paradigm, its conceptual foundation bears strong resemblance to prior works such as **ORCA (Mukherjee et al., 2023)** and **Chain-of-Thought Distillation (Li et al., 2024)**. These earlier methods also transfer reasoning traces or critique signals from a stronger teacher to a smaller student.
   CGD’s main distinction — conditioning the student on both its own response and the teacher’s critique — represents an incremental rather than a fundamental departure. The paper would benefit from a clearer articulation of how CGD meaningfully extends these established distillation paradigms beyond re-framing critique conditioning.

2. **Unfair or Incomplete Compute Efficiency Comparison**
   The claim that CGD requires “**60× less compute**” than reinforcement-learning–based frameworks such as **SimpleRL-Zero** or **DeepSeek-R1** may not be entirely fair or directly comparable. Large-scale RL-based models like DeepSeek-R1 are designed for **general-purpose reasoning** across a broad range of domains, whereas CGD’s results are largely restricted to mathematical and structured reasoning. Thus, the compute advantage should be interpreted cautiously, as it may not hold in broader or more diverse settings.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a method for reasoning SFT called CGD. This method uses a strong teacher model to provide critiques of the initial response of the student model, then feeds the prompt + initial response + teacher critique into the SFT training stage. Then, during inference stage, the teacher critique is not needed, as the student model learns to do this process end to end.

Experiments were done on Qwen and Llama, and evaluations were done on various math datasets, as well as general reasoning datasets and general instruction following evaluations. Results are strong and show that the method works well.

### Strengths
- method is simple and easy to understand
- I like that it included both Llama and Qwen
    - I also like that it explored both in-family (Llama student + Llama teacher) and cross-family (Qwen student + S1 teacher)
- nice that the model is able to retain general instruction following, since this is something that is often lost when doing imitation SFT
    - generally quite comprehensive evaluation sets
- strong results
- ablation experiments are nice. I especially liked section 4.2.1 (comparison with SimpleRL) and section 4.2.2 (comparison with the method without critique).

### Weaknesses
- because this field is so popular and this paper's contribution is relatively simple, I wouldn't be surprised if there's a few other concurrent work submitted to this conference that explores a very similar idea of incorporating critiques in SFT data.
    - (I realize this is another way to say "lacks novelty"..., though I think it's slightly more nuanced than that, since this subtopic is one that's currently very popular) 
- Even within this simple method, I think there are a few other areas that I think would be nice to explore:
    - out-of-distribution evaluation sets (to see how well the reasoning helps)
    - analysis of the correctness of the critiques -- How often does the teacher critique get it right? Also, does it matter if the teacher critiques are correct, or is it more about the structure rather than the actual content? Relatedly, does a stronger teacher result in a stronger SFT-ed model, or does it not matter that much?
    - for added completeness: maybe some other model scales or some other domain like code

### Questions
- In line 322, you mentioned about some regulatory issues preventing you from using GPT as teacher. Curious what these are? I see GPT teachers in these papers quite often. I think it would be nice to see what the results would look like with a frontier-level teacher model beyond Llama3-70B.
- I was looking at the Mixtral and Olmo results and saw that the gains for those models are slightly smaller. Do you have a guess or a hunch as to why certain models benefit more from CGD?
- Is there a reason you didn't do RL on top of the SFT-ed model? I feel like it's better for these types of papers to at least try doing RL on top of SFT, just to show that these gains still continue to hold after RL.

### Soundness
3

### Presentation
3

### Contribution
3
