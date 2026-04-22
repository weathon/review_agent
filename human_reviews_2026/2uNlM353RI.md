# Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Although continuous-time consistency models (e.g., sCM, MeanFlow) are theoretically principled and empirically powerful for fast academic-scale diffusion, its applicability to large-scale text-to-image and video tasks remains unclear due to infrastructure challenges in Jacobian-vector product (JVP) computation and the limitations of evaluation benchmarks like FID. This work represents the first effort to scale up continuous-time consistency to general application-level image and video diffusion models, and to make JVP-based distillation effective at large scale. We first develop a parallelism-compatible FlashAttention-2 JVP kernel, enabling sCM training on models with over 10 billion parameters and high-dimensional video tasks. Our investigation reveals fundamental quality limitations of sCM in fine-detail generation, which we attribute to error accumulation and the “mode-covering” nature of its forward-divergence objective. To remedy this, we propose the score-regularized continuous-time consistency model (rCM), which incorporates score distillation as a long-skip regularizer. This integration complements sCM with the “mode-seeking” reverse divergence, effectively improving visual quality while maintaining high generation diversity. Validated on large-scale models (Cosmos-Predict2, Wan2.1) up to 14B parameters and 5-second videos, rCM generally matches the state-of-the-art distillation method DMD2 on quality metrics while mitigating mode collapse and offering notable advantages in diversity, all without GAN tuning or extensive hyperparameter searches. The distilled models generate high-fidelity samples in only $1\sim4$ steps, accelerating diffusion sampling by $15\times\sim50\times$. These results position rCM as a practical and theoretically grounded framework for advancing large-scale diffusion distillation. Code is available at https://github.com/NVlabs/rcm.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the key limitations of continuous-time consistency models (sCM) and presents solutions to improve their performance and scalability. The authors first identify two fundamental issues in sCM: error accumulation during integration and a “mode-covering” tendency caused by its forward-divergence objective. To mitigate these problems, they propose a score-regularized variant, rCM, which integrates score distillation as a long-skip regularizer. This modification complements sCM with the “mode-seeking” property of reverse divergence, leading to enhanced sample quality and diversity.
In addition, the paper scales sCM and rCM to large-scale text-to-image and text-to-video diffusion tasks, made possible by custom infrastructure designs leveraging FlashAttention-2, Fully Sharded Data Parallel (FSDP), and Context Parallelism (CP). These optimizations enable efficient training of models with over 10 billion parameters. Extensive experiments on large-scale settings demonstrate that rCM achieves visual fidelity and diversity on par with or surpassing existing distillation methods, while maintaining the efficiency of few-step generation.

### Strengths
1. Summary: The paper tackles an important problem in the current diffusion distillation landscape by scaling continuous-time consistency models (sCM) to practical, large-scale visual generation tasks, including text-to-image and text-to-video synthesis.

2. Experimental Evaluation: The experiments are extensive and well-executed, covering both GenEval and VBench, with comprehensive comparisons against strong recent baselines for image and video generation.

3. Results: Both qualitative and quantitative results show that the proposed rCM effectively distills large diffusion models, achieving performance comparable to teacher models while being substantially faster. The method also demonstrates strong scalability, successfully applied to models with up to 14B parameters and producing competitive few-step generations.

### Weaknesses
1. Limited theoretical novelty: The paper reads more like a comprehensive technical report than a scientific contribution offering new theoretical insights. While the engineering efforts to scale continuous-time consistency models—such as infrastructure designs leveraging FlashAttention, FSDP, and Context Parallelism—are impressive, these are primarily system-level and implementation-oriented rather than conceptual innovations.

2. Unclear theoretical justification for score regularization: The explanation of how incorporating reverse divergence via the DMD objective mitigates accumulated error in sCM remains insufficient. Without stronger theoretical grounding or ablation evidence, the justification for the proposed regularization appears weak.

3. Inconsistent claims about tuning difficulty: The statement that sCM requires “tedious and impractical tuning” (lines 182–183) contradicts the claim in Figure 1 that CMs are “easy to tune.” These conflicting points reduce clarity.

4. Presentation and visualization issues: In Figure 2, the teacher–student comparison is ambiguous. Annotating which samples correspond to teacher and student outputs, and including teacher references for text-to-video examples, would improve clarity.
	
5. Limited evaluation scope: The generality of rCM is not well established, as experiments are limited to Cosmos-Predict2 and Wan2.1. A broader evaluation across other architectures (e.g., SANA, Flux) would strengthen the claims of generality.

6. Clarity of algorithmic presentation: The main paper should include the key algorithms, clearly indicating which parts are inherited from sCM and DMD, and which represent new contributions. This would make the methodological novelty more transparent.

### Questions
1.	Could the authors provide stronger theoretical justification or intuition for why combining forward and reverse divergence objectives improves both quality and diversity? How does this differ from formulations such as Jensen–Shannon divergence or distribution matching distillation (DMD)?

2.	In Section 3.3.2, how exactly does the inclusion of reverse divergence alleviate error accumulation when the self-feedback JVP term dominates? Are there ablation or analytical results that support this explanation?

3.	The paper claims that a fixed λ = 0.01 generalizes across all models and tasks. Was this empirically validated? If so, could the authors provide supporting evidence or sensitivity analysis?

4.	How challenging is hyperparameter tuning in practice for rCM compared to baseline sCM and DMD? The paper presents conflicting statements—could the authors clarify this discrepancy?

5.	Will the authors release training scripts or infrastructure details for reproducing large-scale runs (e.g., 14B models), given the reliance on system-level optimizations like FSDP and Context Parallelism?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes rCM, a score-regularized continuous-time consistency model, to scale diffusion distillation to large-scale T2I and T2V models. It addresses quality issues in pure continuous-time consistency models by integrating reverse divergence-bsaed score distillation as a regularizer. Besides, the authors introduce a lot of infrastructure to help distill the model.

### Strengths
1. Novel formulation. The core idea of combining forward and reverse divergence within a consistency model framework is novel, and the authors provide some theoretical supports.

2. Significant scaling. From the experiments, the author seems to be the first work to successfully apply and analyze continuous-time consistency distillation at 10B scale for both image and video generation.

### Weaknesses
1. Limited benchmark. Through all experiments in the paper, the authors only conduct the experiments on GenEval (for T2I), and VBench (for T2V). Honestly speaking, both of them are too old, and the prompts in both benchmarks are too easy. I highly recommend the authors to include more diverse datasets or prompt types.

2. The comparison baseline only contains DMD2. It could be expanded to include other recent methods, especially for the image generation.

3. The substantial engineering efforts, while necessary for training large scale models, might partially overshadow the novelty of the method for some readers, creating a perception of heavy engineering.

### Questions
No.

### Soundness
3

### Presentation
4

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
The paper introduces score-regularized continuous-time consistency models (rCM), a framework for few-step distillation of large text-to-image (T2I) and text-to-video (T2V) diffusion models. The authors first scale continuous-time consistency models (sCM) to application-level models by implementing a FlashAttention-2–compatible JVP kernel and making sCM work with FSDP and context parallelism. They then diagnose a core limitation of sCM—fine-detail degradation and temporal artifacts—and attribute it to error accumulation and the mode-covering nature of its forward-divergence objective. To fix this, they propose rCM, which adds score-based distillation (DMD) as a long-skip, reverse-divergence regularizer, yielding higher visual quality without sacrificing diversity. On large teachers (Cosmos-Predict2 up to 14B params; Wan 2.1 up to 14B), rCM achieves 1–4 step generation (15×–50× speedups) with GenEval/VBench scores matching or exceeding DMD2, and better diversity, with additional stabilization via semi-continuous/high-precision time-derivative handling.

### Strengths
+ Point out that sCM’s forward-divergence and JVP self-feedback cause error accumulation and mode-covering behavior; reverse-divergence regularization addresses fine-detail quality.

+ The FlashAttention-2 JVP kernel and compatibility with FSDP/CP are substantial contributions for training at 10B+ and for video; these infra details are often the bottleneck in practice.

+ On T2I GenEval, rCM matches/approaches large teachers in 1–4 steps; on T2V VBench, rCM even surpasses the 480p Wan teacher in total score at 2–4 steps while being much faster.

+ The paper emphasizes that rCM retains sCM’s diversity while repairing quality, contrasting with DMD2’s tendency toward collapse (Fig. 5).

### Weaknesses
+ The claim that a single $\lambda=0.01$ works “across all models and tasks” is strong; sensitivity to $\alpha$, rollout schedule $p_D$, and the number of rollout steps $N$ should be ablated. Likewise, the contribution of each stabilization component (semi-continuous vs FP32 time embedding) deserves a clean ablation.

+ While DMD2 is a relevant and strong baseline, additional comparisons would strengthen the case: SiD (score-identity distillation) and MeanFlow/AYF-style CTM variants at similar budgets, as well as recent adversarial post-training for video. The paper mentions these families but does not show comprehensive side-by-side results.

+ Results focus on Cosmos-Predict2 and Wan2.1; evidence on qualitatively different teachers (e.g., SDXL/FLUX/SANA or smaller open models) would underscore generality, especially for non-Transformer or non-FlashAttention architectures.

### Questions
+ Please ablate the sensitivity of $\alpha$, rollout schedule $p_D$, and the number of rollout steps $N$.

+ Will the FlashAttention-2 JVP kernel, training code, and/or distilled weights be released?

+ What are the total training flops, GPU hours, and memory profiles for 2B/14B and for video? This would help others assess feasibility of rCM at scale.

### Soundness
3

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
3

### Summary
The authors propose rCM which scales continuous-time consistency models (sCM) to 10B+ parameter image and video models. For efficient computation of JVP in sCM training, they develop FlashAttention-2 JVP kernels compatible with FSDP and context parallelism. To fix sCM's quality issues in fine details, it also adds score distillation (DMD) as regularizer. This loss results in combining forward-divergence (high diversity) with reverse-divergence (high quality). Experiments on Cosmos-Predict2 and Wan2.1 show 4-step generation matching DMD2 quality with better diversity and speedup over teachers.

### Strengths
The paper shows good experimental results across comprehensive benchmarks that achieve SOTA from the benefits of large-scale models.
This is the first attempt to scale continuous-time consistency distillation to application-level models with over 10 billion parameters and high-resolution video generation, demonstrating high practicality. 
The contributions on infrastructure, such as FlashAttention-2 JVP kernel, are valuable, allowing large-scale training to be feasible.

### Weaknesses
1. The methodological novelty is limited. rCM is essentially a trivial weighted sum of existing methods (sCM + $\lambda$DMD). 
2. Section 3.1's "Algorithm Details" appears to directly use techniques already proposed in the original sCM paper (Lu & Song 2024), with no clear additional contributions.
3. It remains unclear whether rCM actually outperforms DMD on smaller-scale models where infrastructure is not a bottleneck in Table 2.
4. There's no ablation study on the loss.
5. Could the authors provide a detailed training cost comparison between rCM and DMD2?
6. I wonder if the authors have attempted combining sCM with GAN training instead of DMD, as it also has a mode-seeking property.
7. It is unclear what exactly the 'long-skip' refers to in a long-skip regularizer.

Overall, the paper's main contribution appears to be its infrastructure part. However, this component is only briefly mentioned in the main paper, and even the appendix does not seem to reveal any particularly novel methodology or contribution compared to prior work.
While the work has practical value, the authors do not convincingly argue that sCM is essential for large-model distillation. It is conceivable that other approaches might not suffer from the training efficiency issue to the same extent, or that this issue is not as critical as implied.
The lack of experiments ablating this specific efficiency claim, or comparing it against alternatives that might not have this issue, makes it difficult to see a significant merit in the proposed method.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
