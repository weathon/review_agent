# D2-LoRA: A Synergistic Approach to Differential and Directional Low-Rank Adaptation

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
We present a systematic exploration of the parameter-efficient fine-tuning design space under practical constraints, yielding D$^{2}$-LoRA—a method that reaches 76.4% average accuracy on eight QA/RC benchmarks using only 5k training samples per task and two epochs, while retaining algebraic mergeability at inference with near-exact numerical equivalence. D$^{2}$-LoRA combines a differential signed low-rank residual with a directional per-column normalization applied only during training. Specifically, given a frozen $W_0$, we learn two rank-$r$ components forming an update $\Delta W=\tfrac{\alpha}{r}(A_+B_+-\tau A_-B_-)$. This update is then projected onto the original column norms of $W_0$ to yield $W^\star$, thereby allowing optimization to adjust directional components while preserving the original magnitude. At inference time, we merge $W^\star$ and $\Delta W$ into $\widehat{W}$, which incurs no additional latency. Compared to baselines, D$^{2}$-LoRA achieves a +2.2pp macro improvement over LoRA (74.2%), and matches or exceeds DoRA. At matched parameter counts (LoRA at rank $2r$ vs. D$^{2}$-LoRA at rank $r$), the improvement is +1.6pp, confirming that gains stem from architectural innovations rather than increased parameterization. Beyond QA/RC, D$^{2}$-LoRA improves generative tasks (+1.2pp ROUGE-L, +1.1% win rate) and exhibits 36% lower training volatility. It also preserves numerical equivalence after merging (mean gap $\approx 0.03$pp; worst $0.7$pp), while restoring $\sim 1.91\times$ evaluation throughput. Training overhead is 19%—comparable to DoRA—and decreases with longer input sequences. A geometric analysis explains why projection stabilizes low-rank training, and ablation studies isolate the effects of the negative branch, rank, target modules, scoring function, and fixed $\tau$.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Based on LoRA, this paper introduces a negative branch and applies per-column normalization to the fine-tuned modules during training. The experiments demonstrate the effectiveness of the proposed method.

### Strengths
1. The method is simple and easy to implement.

2. The experiments demonstrate the effectiveness of the proposed method.

### Weaknesses
1. As shown in Equation (3), after normalization, the LoRA module is added one more time. The motivation for this operation is unclear.

2. Although the authors provide three reasons, the rationale behind the different initializations of $A_+$ and $A_-$ remains somewhat unclear. Additional experimental or theoretical evidence would strengthen this argument. From my understanding, the concerns mentioned could potentially be addressed by properly setting $\tau$.

3. Line 193 claims that the proposed method leads to a smoother loss curve, but this claim is not elaborated on or demonstrated in the experimental section.

4. The experimental comparison in this paper does not appear to be entirely fair. The proposed method introduces twice as many trainable parameters; therefore, the ranks of the baseline methods should be set to twice that of the proposed method for a fair comparison.

5. The number of baseline methods used for comparison is somewhat limited, only two baselines (i.e., LoRA and DoRA).

6. In the experiments, the Unmerged variant outperforms the Merged one in most cases, which makes the additional LoRA operation in Equation (3) rather confusing.

7. An ablation study exploring different ranks for the $A_+$ and $A_-$.

### Questions
1. In Table 4, it is unclear which baseline the reported speedup is compared against.

2. There seems to be a typo in Table 6. The results for Minus off and Detach minus grad are exactly the same.

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
3

### Summary
This paper introduces D2-LoRA, a parameter-efficient fine-tuning (PEFT) method designed for budget-constrained scenarios. The method combines two key ideas: a signed low-rank residual, which provides both additive and subtractive update capabilities, and a train-time-only directional projection, which constrains the updated weights to maintain the column-wise magnitudes of the original pretrained model. This design aims to enhance model expressivity and training stability while preserving the crucial advantage of being fully mergeable at inference time, thereby incurring no additional latency. The authors provide theoretical motivation and empirical results on several question-answering benchmarks, demonstrating performance improvements over LoRA and DoRA.

### Strengths
1. The paper presents a well-motivated synthesis of concepts from LoRA and DoRA. The introduction of a signed residual to provide subtractive capacity is a logical extension, and the use of a directional projection only during training is a clever mechanism to gain stability benefits without sacrificing inference-time mergeability and efficiency.

2. The work is explicitly framed around practical, budget-constrained fine-tuning, using a small number of training samples and epochs. The validation of near-exact merge equivalence and the measurement of post-merge throughput gains directly address the needs of real-world deployment, making the contributions highly relevant.

### Weaknesses
1. The negative branch is modulated by a scalar α̃, which is a critical new hyperparameter. The main experiments fix α̃=0.5, but the ablation study (Table 7) shows that α̃=1.0 yields better results on one of the backbones. This suggests performance is sensitive to this choice.

2. All eight evaluation benchmarks are question-answering or reading comprehension tasks that can be framed as multiple-choice classification. The method's effectiveness on more open-ended, generative tasks (e.g., summarization, dialogue, or long-form instruction following) remains unproven.

3. D2-LoRA doubles the number of trainable parameters compared to LoRA and introduces additional computations (a second residual GEMM, column-norm calculation, and projection) during the training forward pass. The reported performance gains are modest in some cases.

4. The ablation study on target modules (Table 5) only considers attention projections (q, k, v, o). It is common practice in PEFT to also evaluate the effect of adapting MLP layers (e.g., gate_proj, up_proj, down_proj), which is missing from the analysis.

5. The paper provides a geometric intuition that the projection removes radial gradient components, which acts as an implicit regularizer. While plausible, this is not directly demonstrated with empirical evidence, such as by visualizing gradient distributions with and without the projection.

6. How robust is the method to variations in this initialization? Does the training become unstable or fail to converge if a standard initialization is used for both branches?

7. The paper states that setting the minus branch to zero yields a "DoRA-like" variant. This configuration is tested in the ablation study (Table 6, "Minus off"), but the connection is not made explicit in the table, which could improve clarity.

8. The failure case analysis mentions that a strong negative branch (α̃=1.0) can "over-correct" on CommonsenseQA. This highlights a potential failure mode where the subtractive capacity could be detrimental.

### Questions
1. How should practitioners approach tuning α̃ for new models or tasks? Is there a principled way to set it, or does it require a dedicated hyperparameter search, which would add to the tuning budget the paper aims to minimize?

2. How does D2-LoRA perform on tasks that require creative or long-form generation? Does the directional constraint, which stabilizes discriminative tasks, potentially limit the model's generative diversity?

3. Could you provide a more direct comparison of the training time overhead (e.g., wall-clock time per epoch) versus LoRA and DoRA? In what scenarios is the trade-off of higher training cost for the observed accuracy improvement most justified?

4. Have you experimented with applying D2-LoRA to the MLP layers? How does this impact the parameter count and overall performance compared to adapting only the attention layers?

5. Can you provide empirical evidence, such as plots of the norm of radial vs. tangential gradient components during training, to directly support the claim that the projection stabilizes optimization by removing radial modes?

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
3

### Summary
The paper proposes D2-LoRA, a parameter-efficient fine-tuning method that combines a signed low-rank residual (positive and negative branches) with a training-time column-wise directional projection to preserve backbone weight norms. Under a strict budget (≤5k examples per task, ≤2 epochs) on eight QA/RC benchmarks and two backbones, it improves average accuracy over LoRA and often matches or exceeds DoRA. The adapter is algebraically mergeable at inference, yielding near-identical post-merge accuracy and about 2× evaluation throughput. A geometric analysis (norm preservation, Lipschitz control) and ablations over rank, targeted modules, the negative branch, and τ support the stability and effectiveness claims.

### Strengths
- The method is straightforward to implement—combining a signed low-rank residual with a training-time directional projection—and remains mergeable at inference, imposing minimal engineering overhead within standard PEFT pipelines.
- In low-data, small-rank settings, it consistently improves accuracy over baseline LoRA and stays competitive with related variants across multiple QA/RC tasks and backbones under tight training budgets, with ablations indicating robustness to rank and module choices.

### Weaknesses
- Parameter-count fairness is not fully addressed: comparisons primarily match D²-LoRA at rank r against LoRA at the same r, without a parameter-matched LoRA (e.g., 2r) or equivalent-capacity baselines to isolate architectural benefits from increased parameterization.
- The evaluation scope is limited to multiple-choice QA/RC benchmarks and does not assess open-ended generation, instruction following, code/math reasoning, or multilingual settings, which may exhibit different behaviors and trade-offs.
- The theoretical analysis is mainly supportive of the method’s stability and expressivity and relies on established techniques, offering limited novelty beyond contextualizing the proposed projection and signed residual within known frameworks.

### Questions
See weakness

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
This paper addresses a central challenge in the field of PEFT: the effective adaptation of large models under practical budget constraints, specifically in low-data and limited-compute regimes. The authors introduce D2-LoRA, a novel PEFT method built on two core architectural innovations designed to work in synergy. The first is a differential signed low-rank residual, which equips the model with both additive (feature reinforcing) and subtractive (feature suppressing) update capabilities, effectively doubling the expressivity to rank 2r. This increased expressivity is controlled by the second innovation: a train-time directional projection that normalizes updated weight columns to preserve their original magnitudes, providing crucial stability for training in low-data settings.

### Strengths
1. D2-LoRA represents a novel synthesis of ideas. It intelligently combines an expressive signed residual—allowing the model to not only learn new features but also explicitly suppress pre-existing ones—with a stability-enhancing directional constraint. 
2. The authors provide detailed ablation studies that methodically dissect the architecture's performance. 
3. The evaluation is conducted with commendable rigor under a strict and realistic budget: a maximum of 5,000 training samples per task and only two epochs.

### Weaknesses
1. While the focus on the low-data regime is well-motivated, the experiments are confined to QA/RC tasks. This is a reasonable scope, but the authors themselves note in Section 9 that evaluation on "Broader modalities and RLHF-style pipelines are left for future work." 
2. The sensitivity of the τ hyperparameter, which balances the positive and negative branches, appears to be model-dependent. Data from Table 7 shows that the optimal value is 1.0 for Llama-3.2-3B-Instruct but 0.5 for Qwen2.5-7B-Instruct.

### Questions
1. D2-LoRA introduces trade-offs that could be discussed more directly. The architecture doubles the number of trainable parameters compared to LoRA and introduces train-time computational overhead from "a column-norm pass and one extra residual GEMM." The paper would benefit from a more explicit analysis of this performance-versus-cost trade-off.

### Soundness
3

### Presentation
2

### Contribution
3
