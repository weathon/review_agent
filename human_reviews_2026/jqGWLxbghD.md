# Latent-Guided Reasoning: Empowering Small LLMs with Large-Model Thinking

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6, 6

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks, but their high computational costs limit their widespread practical application. We argue that this inefficiency arises from the tight coupling of high-level cognitive planning (devising the solution strategy) and low-level linguistic realization (generating step-by-step text). To address this challenge, we propose a novel collaborative framework that decouples these two processes through Latent Guidance. Our approach implements a division of labor: a large model acts as an Implicit Thinker, performing high-level cognitive planning and compressing its solution strategy into a set of compact latent guidance vectors. A small, efficient model then serves as an Explicit Executor, which receives this latent guidance to generate a concise and effective reasoning chain. This process is enabled by a dual-loss training objective, grounded in information-theoretic principles, where a reconstruction loss explicitly compels the latent guidance to become a high-fidelity representation of the full reasoning chain. Extensive experiments on 8 diverse reasoning benchmarks demonstrate that our method substantially enhances the reasoning capabilities of small models across various scales (from 0.5B to 8B), allowing them to outperform strong baselines and exhibit superior generalization. Notably, our framework boosts small model accuracy by up to 13.9% over its standalone baseline. Our work introduces a new, theoretically-grounded paradigm for empowering small models with large-model thinking, substantially improving the performance-cost trade-off for complex reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
A new method is introduced to use latent reasoning from a large model (implicit thinker) to guide a smaller model to generate conventional CoT conditional on the latents, in order to improve the smaller model (which is trained under the inference setting).

### Strengths
The method is clearly presented and, to the best of my knowledge, different to other methods -- although sharing many commonalities (see below). Results improve over a number of small model distillation baselines (although see questions below).

### Weaknesses
While to the best of my knowledge the method is different to others -- it is a mix of some existing things (which can be ok). Learning latent vectors by making them predictive of CoT is a pretty standard approach in continuous latent reasoning approaches, e.g. iCoT and COCONUT. In those cases, I believe they show that better latent vectors can be learnt by iteratively removing the CoT to be predicted to compress more of the information, although this is more complex than your approach -- but could it be better?

Other methods also aim to mix latents with CoT text tokens, such as LightThinker (https://arxiv.org/abs/2502.15589) which I believe performs pretty well without needing to keep around a larger and smaller model, I think this work is not mentioned/compared?

Overall, the need to keep both a large and a small model around for inference seems complex, and I think would need to give significant wins to make it worth on it. On the other hand, the counter argument is that frontier models are often these days deployed as mixtures of experts, so this isn't too far off something like that either..

### Questions
I don’t know all those baselines well, and they aren’t described in detail (MT-CoT, Step-by-step,  KARD, CasCoD, NesyCD) – although I think MT-CoT would be straight training on the larger models CoT? Did you try on-policy distillation  https://arxiv.org/abs/2306.13649 from the larger model? Or, what about RL of the small model after distillation? Those two variants sounds like some of the strongest standard small model approaches I would try first, and I'm not sure they are compared to?

For which of those baselines do you have to keep around the large model for inference time, I’m assuming many do not, which yours has to, right? So -- this looks like a disadvantage of yours just in terms of complexity...? (That is, the gains would have to be substantial in that case I guess, given the extra complexity).

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
3

### Summary
This paper addresses the high computational cost of complex reasoning in LLMs. The authors argue this inefficiency stems from the tight coupling of "high-level cognitive planning" (strategy) and "low-level linguistic realization" (text generation). To solve this, they propose "Latent Guidance," a collaborative framework that decouples these processes. The framework uses two models: an **Implicit Thinker** (a large LLM, e.g., Qwen2.5-32B) processes the input question and special "thought tokens" to produce a set of compact "latent guidance vectors" ($H_{guidance}$); an **Explicit Executor** (a small LLM, e.g., Qwen2.5-7B) receives both the question and $H_{guidance}$ to autoregressively generate the final reasoning chain and answer. The core of this "Cognitive Distillation" is a dual-loss training objective for the large model. It includes a task loss for the correct answer, and, crucially, a reconstruction loss $\mathcal{L}\_{recon}$ that compels the latent vectors $H_{guidance}$ to contain enough information to reconstruct the entire original, detailed reasoning chain $R$. The authors provide a theoretical analysis showing that this $\mathcal{L}\_{recon}$ maximizes the mutual information between the plan and the reasoning chain. Experiments across eight benchmarks show the 7B model, when guided, gains up to 13.9% in accuracy, runs 2x faster than its 7B baseline, and is 4x faster than the 32B teacher, while also showing superior OOD generalization.

### Strengths
1.  The paper's primary strength is its novel "Cognitive Distillation" framework. The conceptual move to decouple planning from realization is intellectually appealing. It reframes distillation from mimicking *outcomes* (final text) to transferring a *process* (a high-level plan), which is a contribution to the field.

2.  The $\mathcal{L}\_{recon}$ loss is not merely empirical; it is rigorously justified with an information-theoretic analysis in Appendix A. The derivation showing that minimizing $\mathcal{L}\_{recon}$ is equivalent to maximizing the mutual information $I(R;H_{guidance}|Q)$ provides a solid theoretical foundation for why the latent vectors should be information-rich.

3. The method achieves strong results, particularly in OOD generalization (Table 1). The fact that it consistently outperforms strong distillation baselines (like NesyCD) on unseen datasets lends weight to the hypothesis that distilling a "strategy" is more robust and generalizable than distilling text.

4.  The results in Table 6 are noteworthy. Achieving a 2x speedup over a 7B baseline while simultaneously boosting its accuracy by +13.9% (on ARC-C) represents a clear Pareto improvement and a practical achievement.

### Weaknesses
1. The paper's central thesis—that $H_{guidance}$ represents an *abstract, high-level "cognitive plan"* —is not sufficiently supported by the provided evidence.
    *   **t-SNE Visualization (Fig. 3):** This evidence is weak. t-SNE is known for creating illusory clusters. The observed clusters, which are given post-hoc labels like "Sequential Multi-Step Reasoning," are far more likely to be simple high-dimensional clusters of "problem types" (e.g., all two-step addition problems) rather than proof of abstract, disembodied "strategies".
    *   **Probing Analysis (Table 5):** This analysis is tautological. The $\mathcal{L}\_{recon}$ loss *explicitly trains* $H_{guidance}$ to be able to reconstruct the full text. Therefore, "proving" that a probe can predict properties of that text (like "Reasoning Step Count" or "Mathematical Operator")  from $H_{guidance}$ is not evidence of a cognitive plan; it is merely a sanity check that the loss function *worked*.

2. The ablation in Table 4 shows that using $K=10$ thought tokens yields better performance on GSM8K (81.1%) than the $K=5$ (80.5%) used for the main results. This suggests the paper may not be reporting its strongest possible configuration.

3. The qualitative analysis in Table 3 relies entirely on GPT-4o as an evaluator (Appendix E). This is not a sufficient substitute for human evaluation (or stronger models like GPT-5), especially when judging nuanced qualities like "Relevance."

### Questions
See weaknesses

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
The paper proposes a novel framework named "Latent Guidance" (LG) designed to empower small language models (SLMs) with the complex reasoning abilities of large language models (LLMs). The authors term this new paradigm "Cognitive Distillation," which decouples high-level cognitive planning from low-level linguistic realization. In this framework, a large "Implicit Thinker" model performs the complex planning and compresses its strategy into a set of latent vectors ($H_{guidance}$). A small "Explicit Executor" model then receives this guidance and generates the final textual reasoning chain. The framework is optimized via a dual-loss training objective.

The authors prove their method's effectiveness and efficiency via both empirical experiments and theoretical analysis. Comprehensive ablations like the probing experiments of the in-depth analysis of the latent cognitive plan indicates the small model executes a structured, abstract plan, rather than just learning feature correlations.

### Strengths
1. The manuscript is well-organized and easy to follow.

2. The proposed idea of cognitive distillation is sound.  The information theoretical analysis is solid.

3. The appendix includes almost all theoretical and experimental details like the training hyperparameters and computational resources.

4. The paper goes beyond accuracy tables to explain the inner mechanisms. The t-SNE visualizations and quantitative probing tasks provide compelling evidence that the $H_{guidance}$ vectors are not just feature containers but successfully encode high-level, abstract reasoning strategies (e.g., "Sequential Multi-Step Reasoning").

### Weaknesses
1. There is a contradiction between the core methodological design and the empirical results. 
 The reconstruction loss  and  SLM loss explicitly train the models to reconstruct the original, full-length reasoning chain.
However, the paper claims a key benefit is that the SLM can generate substantially more concise reasoning chains (e.g., 128.9 tokens vs. SFT's 235.4). The paper cannot train the model to do one thing (reconstruct the full 235-token chain) and then claim its success for doing the opposite (generating a 129-token chain) without a clear explanation. Is conciseness an emergent property? Or is the method description in Eq 6 inaccurate?

2. The paper calls L_recon the "cornerstone" of the method. However, the ablation study (Table 4) shows that removing it causes only a minor performance drop (e.g., -2.3% on GSM8K). This small drop does not support its "cornerstone" status for in-domain performance.


3. Line 152, Line 157: the author should use correct right single quote.

### Questions
1. About the latency in Table 6, does the reported time for "Ours" represent the total end-to-end latency ($T_{total} = T_{step\_{(32B\_{fwd})}} + T_{step\_{(7B\_{gen})}}$)?

2. See weakness 1

3. Have the authors considered an additional baseline that involves providing the full text of the 32B teacher's reasoning chain ($R_{LLM}$) directly into the 7B small model's prompt at inference time?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes Latent Guidance, a framework that decouples cognitive planning from linguistic realization to empower small LLMs. A large model compresses its solution strategy into latent guidance vectors, which a small model then uses to generate concise reasoning chains. The dual-loss training (task loss + reconstruction loss) ensures the latent vectors faithfully encode the reasoning process.

### Strengths
The theoretical foundation using mutual information and Fano bounds provides solid justification for the approach. I'm impressed by the consistent OOD improvements across diverse models from 0.5B to 8B parameters. The comprehensive diagnostics in Appendix B, especially the MI estimation converging to 3.10 nats, strengthen the empirical validation. The emergence of distinct reasoning strategy clusters without explicit supervision is fascinating.

### Weaknesses
THe presentation could be improved. Also, not tesetd on larger models. 

The framework requires storing latent guidance for all training examples which could be memory intensive for larger datasets. While the projection layer design is ablated, I think more analysis on the information bottleneck it creates would strengthen the work.

### Questions
How does performance scale with even larger gaps between teacher and student sizes, say 70B to 2B?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Latent Guidance, a framework that combines large and small language models to perform complex reasoning for improved cost-effectiveness. The approach uses a two-stage training process: in the first stage, the large model is trained to compress (via auto-encoding) the reasoning steps into some latent vectors. In stage two, the small model (augmented with some learnable projection layers) is trained to generate the reasoning steps conditioned on these latent vectors. Experiments across 8 reasoning benchmarks show that a 7B model achieves up to 13.9% accuracy gains over its standalone baseline while being 2x faster, and 4x faster than the 32B teacher model.

### Strengths
- The paper presents a well-motivated framework addressing the inefficiency of tight coupling between high-level cognitive planning and low-level text generation in current reasoning systems.

- The experimental evaluation covers multiple model scales (0.5B to 8B) and diverse reasoning benchmarks, with ablation studies demonstrating the necessity of key design choices. Overall, the generalization of the proposed framework to OOD tasks is impressive. The appendices provide extensive diagnostics such as MI estimation and capacity scaling experiments, that attempt to validate theoretical claims.

### Weaknesses
- The paper seems to mischaracterize the proposed method as "distillation" when it's closer to an autoencoder with active inference-time involvement of the large model. This makes comparisons with true distillation baselines (SFT, KD, NesyCD) fundamentally unfair, as those methods train the small model to reason independently without the large model at inference time, while this approach requires the 32B model to generate latent guidance for every test query. Relatedly, it's good to report standalone 32B performance for reference.

- The results that the proposed method produces substantially more concise reasoning chains than other baselines is quite difficult to understand. If the small model is trained to reconstruct the teacher's reasoning, the lengths should be comparable to other distillation methods - how could they be dramatically shorter? In lines 326-329 the authors explain this via: *"This demonstrates that the high-level cognitive plan
enables the small model to generate more focused and generalizable reasoning paths, avoiding the verbose, exploratory steps often seen in less-guided generation on unseen problems. The model learns to execute a direct strategy rather than overfitting to the stylistic artifacts of the training data."* which I'm not convinced.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
