# FURINA: Free from Unmergeable Router via lINear Aggregation of mixed experts

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
The Mixture of Experts (MoE) paradigm has been successfully integrated into Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning (PEFT), delivering performance gains with minimal parameter overhead. However, a key limitation of existing MoE-LoRA methods is their reliance on a discrete router, which prevents the integration of the MoE components into the backbone model. This results in persistent computational overhead and increased system complexity during inference.
To overcome this, we propose **FURINA**, a novel **F**ree from **U**nmergeable **R**outer framework based on the L**IN**ear **A**ggregation of experts. FURINA eliminates the router by introducing a Self-Routing mechanism. This is achieved through three core innovations: (1) decoupled learning of the direction and magnitude for LoRA adapters, (2) a shared learnable magnitude vector for consistent activation scaling, and (3) an expert selection loss that encourages divergent expert activation. The proposed mechanism leverages the angular similarity between the input and each adapter's directional component to activate experts, which are then scaled by the shared magnitude vector. This design allows the output norm to naturally reflect the importance of each expert, thereby enabling dynamic, router-free routing. The expert selection loss further sharpens this behavior by encouraging sparsity and aligning it with standard MoE activation patterns.
A challenge that arises from Self-Routing is the potential diminishment of output norms, which could limit the overall model capacity. To mitigate this, we introduce a shared expert within the MoE-LoRA block that provides stable, foundational knowledge. To the best of our knowledge, FURINA is the first router-free, MoE-enhanced LoRA method that can be fully merged into the backbone model, introducing zero additional inference-time cost or complexity.
Extensive experiments demonstrate that FURINA not only significantly outperforms standard LoRA but also matches or surpasses the performance of existing MoE-LoRA methods, while eliminating the extra inference-time overhead of MoE.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors introduce FURINA, a novel framework for integrating Mixture-of-Experts (MoE) into Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning (PEFT) of large language models (LLMs). Unlike existing MoE-LoRA methods that rely on discrete routers, which prevent merging experts back into the base model and incur inference overhead, FURINA employs a mergeable self-routing mechanism based on angular similarity between inputs and adapter directional components. Experts are scaled by a shared magnitude vector, enabling natural reflection of importance in output norms, and an expert selection loss promotes sparsity. A shared expert provides stable foundational knowledge. Evaluations across 9 benchmarks and 3 LLMs demonstrate superior performance over standard LoRA and comparable or better results than MoE-LoRA baselines, all while allowing full reparameterization for zero-overhead inference.

### Strengths
1. Originality: FURINA addresses a core limitation of prior MoE-LoRA approaches by enabling seamless merging into the backbone, preserving LoRA's efficiency advantages and compatibility with optimized inference engines like vLLM.

2. Soundness: Comprehensive testing on diverse benchmarks (e.g., reasoning, math) with multiple LLMs shows consistent gains, highlighting practical benefits like reduced computational overhead.

### Weaknesses
1. My biggest concern: The key idea of decomposing learnable parameters into two parts, including magnitude and similarity was introduced in previous work on DoRA [1].  

2. Incremental ideas: The idea of shared experts was first introduced in DeepSpeed-MoE and was applied in Eq. (11) of this work without any references.  

3. I have a concern regarding the math derivation in Section 3.4. Could the authors please help me explain rigorously why the term $\textbf{BA}x$ in Eq. (19) is equal to the right hand side of Eq.(17)?

4. The paper lacks deep proofs on convergence or optimality of the self-routing mechanism, relying more on empirical validation.

**References**

[1] Liu et al. DoRA: Weight-Decomposed Low-Rank Adaptation.

[2] S. Rajbhandari et al. DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale.

### Questions
1. The shared expert provides stability—did you ablate its impact, and how does it interact with the selection loss in low-data regimes?

2. What are the limitations of this work? Could the authors please provide some approaches to address these issues?

### Soundness
3

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
2

### Summary
The paper proposes FURINA, a router-free MoE-LoRA framework that replaces discrete gating with a Mergeable Self-Routing mechanism. The core design decouples direction and magnitude in each LoRA adapter, scales all experts with a shared magnitude vector, adds a shared expert for stability, and introduces an expert-selection loss to mimic top-K activation patterns—while remaining linearly mergeable after training into a single adapter or directly into the backbone. The authors claim this yields deployment with LoRA-like latency (e.g., in vLLM) yet MoE-like gains, reporting results on various benchmarks and LLMs.

### Strengths
- The paper presents a practical and elegant solution to the major deployment bottlenecks of MoE-based PEFT methods. FURINA’s mergeability eliminates the inference overhead inherent in prior MoE-LoRA approaches, achieving the same inference efficiency as standard LoRA while retaining the performance benefits of MoE.

- The proposed mergeable self-routing mechanism is an intelligent engineering design that leverages angular similarity between vectors without requiring a complex router. In particular, separating the direction and magnitude of expert weights—while enforcing a shared magnitude vector across experts—encourages fair expert selection purely based on input relevance. The efficacy of this design is demonstrated through ablation studies.

- The authors conduct comprehensive and rigorous experiments across multiple model architectures and nine benchmarks, demonstrating strong generalization of the proposed method. By providing robust baselines—including recent non-mergeable MoE-LoRA approaches—and additional ablation and compatibility studies, the paper convincingly supports its claims.

### Weaknesses
- The mergeability aspect of FURINA is interesting, yet the broader idea of implicit MoE without explicit routing is not new. The paper lacks discussion comparing and contrasting the proposed routing mechanism with other implicit routing paradigms—such as FlyLoRA, which also relies on fixed random projection matrices. Without a clear analysis of these related methods, it is difficult to assess the novelty of the proposed routing approach.

- The paper does not provide a formal theoretical justification for its central claim that dense linear aggregation can effectively approximate the sparse top-K routing of traditional MoE. The proposed mechanism behaves more like a weighted LoRA ensemble rather than a truly sparse conditional computation framework. Because the sparsity-inducing loss contributes very weakly, it remains unclear how closely the learned behavior actually resembles MoE-style specialization.

- The experiments are restricted to models with fewer than 8B parameters, leaving open questions about scalability to large-scale models (e.g., 70B+), where expert specialization becomes more pronounced. Merging all experts into a single adapter inherently averages their knowledge, raising concerns about knowledge interference in large models, where fine-grained expert capabilities could be diluted.

- The justification for the role of the shared expert is vague; it appears more like a defensive safety net to prevent potential signal collapse in the self-routing mechanism rather than a principled design choice. In addition, the ablation studies mix the contribution of the shared expert with other factors, making it difficult to evaluate its necessity and effectiveness in isolation.

### Questions
Please refer to the detailed points listed in the Weaknesses section above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper introduces FURINA, a router-free variant of MoE-LoRA that preserves mergeability while maintaining expert diversity. Instead of a discrete router, FURINA performs self-routing by normalizing each expert’s LoRA weights (direction) and applying a shared magnitude vector, enabling a linear, input-independent combination of experts that can be merged back into the base model for inference. The method adds auxiliary selection and balance losses to mimic MoE specialization and achieves near-MoE accuracy with LoRA-level latency on multiple LLMs. While conceptually appealing and empirically effective, parts of the mathematical formulation, notation, and the claim of “full mergeability” require more rigorous justification and precise definitions.

### Strengths
1. The proposed direction–magnitude decomposition with shared scaling is conceptually clear and easy to integrate into existing LoRA pipelines.

2. The empirical results is ok, e.g.,  FURINA achieves accuracy close to or surpassing MoE-LoRA baselines while maintaining LoRA-level throughput and latency,

### Weaknesses
1. The paper claims that FURINA achieves MoE-level specialization through ``self-routing'' using normalized LoRA weights and shared magnitudes. However, no rigorous derivation is provided to show that this linear aggregation replicates the gating behavior of Top-K MoE. The argument, from my viewpoint,  relies on intuition rather than formal analysis, and the proposed loss merely compensates for the missing router instead of replacing it theoretically. 

2. The losses L_{\text{div}} and L_{\text{bal}} lack formal definitions, e.g., the operator sum()  is not specified.

3. Essential hyperparameters (training steps, batch sizes, random seeds, prompt templates, vLLM version) are missing. Also, experiments focus mainly on small QA and reasoning tasks, and  no large-scale or long-context evaluations are reported

### Questions
Please refer to the Weaknesses section for my questions.

### Soundness
3

### Presentation
3

### Contribution
3
