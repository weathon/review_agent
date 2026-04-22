# From Compression to Generalization: Language Model Distillation With Grafting

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Knowledge distillation is a widely used technique in distilling large language models. It is applied both for strong-to-weak distillation, where large-scale flagship models serve as teachers to produce lightweight models suitable for deployment, and for weak-to-strong distillation, where previous-generation models contribute to the development of stronger next-generation models. From a model compression perspective of knowledge distillation, students may be encouraged to adopt mode-seeking behavior; however, for building generalizable generative language models, mode-covering behavior should also be considered. To address this, we conduct an experimental analysis and propose a simple yet effective grafting strategy, in which sequence trees generated at multiple temperatures for autoregressive modeling are combined into a single distillation target. Our extensive experiments demonstrate the effectiveness of the proposed grafting approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
To address a typical trade-off in Large Language Model knowledge distillation—the balance between "mode-seeking" and "mode-covering", this paper proposes "grafting" a simple and easy-to-implement strategy. The method is shown to improve the student model's overall performance across various tasks, models, and algorithms.

### Strengths
1. **Clear and explicit motivation:** The paper strongly argues that as the goal of knowledge distillation shifts from mere model compression to broader generalization, a purely "mode-seeking" approach is insufficient and may even limit the model's generalization capacity.
2. **Sufficient experimental validation:** The experiments cover diverse tasks (e.g., instruction-following, math, code ), models (e.g., Qwen2.5, Llama3, DeepSeek-Coder ), and algorithms (e.g., SeqKD, GKD, DistiLLM-2 ), providing strong support for the central claims.
3. **Simple and easy-to-implement method:** The proposed "grafting" strategy is simple and intuitive. It combines sequence trees generated at multiple temperatures into a unified distillation target , thereby attempting to balance "mode-seeking" and "mode-covering" behaviors.

### Weaknesses
1. **Relatively limited performance gains:** While the performance improvements from "grafting" are consistent , the absolute gains are not highly significant in many cases. Even though the authors mention that sequence tree generation can be efficient (compatible with vLLM ), the limited gains might make practitioners using SOTA methods hesitate to introduce the extra workload and resource cost.
2. **Unclear hyperparameter selection:** The method introduces new hyperparameters, namely the set of temperatures $\mathcal{T}$ and the mixing distribution $\pi_0$ . The paper appears to default to a uniform distribution ($\pi_0(\tau) = 1/|\mathcal{T}|$ ) and a fixed set of temperatures (e.g., $\mathcal{T}=\{0.7, 1.0, 1.5, 2.0\}$ ). The paper lacks a discussion on how to select these parameters.
3. **Insufficient theoretical analysis:** The core idea of "grafting", mixing probabilities from sequence trees generated at multiple temperatures, appears to be a heuristic approach. The paper does not provide a deep theoretical analysis as to why this specific mixing strategy (Equation 8) is the optimal way to balance the two modes.

### Questions
1. Considering the additional complexity that 'Grafting' introduces by requiring sequence generation at multiple temperatures, could the authors please comment on the practical significance of these gains? That is, under what circumstances do you believe the benefits of introducing the strategy proposed in the paper outweigh the additional costs?
2. How sensitive is the method to the choice of $\mathcal{T}$ (e.g., the number of temperatures, their range, and intervals)? Did you experiment with non-uniform mixing distributions for $\pi_0$?
3. Have you considered dynamic or adaptive "grafting" strategies? For instance, could the mixing weights $\pi$ be adjusted based on the training step (e.g., as a form of curriculum) or even adapted based on the context?

### Soundness
3

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
This paper is concerned with the balance between the mode-seeking and mode-covering in knowledge distillation. It proposes a grafting strategy that integrates samples generated with diverse temperatures from the teacher LM into several existing knowledge distillation objectives (e.g., SeqKD). With this simple and effective strategy, the performance of the student LM is largely improved in both strong-to-weak and weak-to-strong scenarios.

### Strengths
1. The proposed strategy is rather simple and universally applicable to existing knowledge distillation methods.
2. The experiments involving both strong-to-weak and weak-to-strong cases are solid and show the strategy is promising.

### Weaknesses
1. The proposed strategy is mostly observation-drive and lacks a proper theoretical justification.
2. The experiments are mostly constrained to comparably smaller-scale LMs, and larger-scale LMs with tens of billions of parameters should also be considered. Similarly, LMs with different architectures (e.g., MoE LMs) would further strengthen the usefulness of the strategy.

### Questions
N/A

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
4

### Summary
This paper investigates the knowledge distillation of generative language models. The authors propose a core argument: traditional distillation methods focus excessively on "mode-seeking" behavior (i.e., high-fidelity imitation of the teacher), which, while suitable for model compression, impairs the ability of the student model (especially large-capacity ones) to capture generative diversity, thus limiting mode-covering generalization.
To address this tension, the paper introduces a new technique called "Grafting." This method combines the sequence trees (or probability distributions) generated by the teacher model at multiple different decoding temperatures into a single, more diverse distillation target, thereby achieving a better balance between mode-seeking and mode-covering.
The authors theoretically justify the soundness of this approach (e.g., providing a gradient equivalence proof for forward KL and a variational upper bound for reverse KL). Experimental results show that the Grafting strategy can consistently improve the performance of various existing distillation baselines (such as GKD, DistiLLM) on general-purpose instruction-following, mathematical reasoning, and code generation tasks. Notably, under weak-to-strong distillation setups, the student model's performance can even surpass that of the teacher.

### Strengths
1.Instead of viewing distillation merely as model compression, the paper offers a novel perspective that elevates the problem to a fundamental trade-off between "mode-seeking" (precision) and "mode-covering" (generalization/recall). This framework is significant for understanding and improving the distillation of generative models, particularly large-capacity ones.
2.The "Grafting" strategy is well-grounded in theory. The mathematical derivations in Appendix A are rigorous, proving that the proposed tractable loss is a valid proxy for the intractable objective. The gradient equivalence for the forward KL case is a particularly strong technical contribution.
3.The paper's analysis is highly mature. The fidelity analysis (Figure 5c) convincingly demonstrates that performance gains do not come from higher fidelity but from stronger generalization, which perfectly supports the "mode-covering" thesis. Furthermore, the authors are very candid in disclosing and analyzing a failure case (the 0.5B model in Appendix B.3). This counterexample, rather than being a weakness, strengthens the paper's core hypothesis by defining the method's scope of applicability (i.e., high-capacity student models).

### Weaknesses
While the paper's contributions are impressive, the strength of the core claims could be further reinforced by more in-depth ablation studies on implementation details and key hyperparameters.

### Questions
To more accurately assess the contribution of this work, I hope the authors will address or discuss the following questions in the final version:
1.The paper claims its method differs from "merely aggregating multi-temperature samples." Could the authors provide an experiment that fairly compares "Grafting" (sampling from the mixture distribution ) with "Data Aggregation" (mixing datasets sampled from each  individually) under the same total sample size? If their performance is similar, is the theoretical complexity of the "Grafting" approach (e.g., the proofs in Appendix A) still necessary?
2.The paper defaults to a uniform mixture in Eq. (8) (i.e., ). Why was this chosen? Given that Fig. 5a shows different single temperatures contribute differently (e.g.,  is strongest on Avg.), did the authors experiment with a weighted mixture? For instance, might a strategy weighted towards the optimal single temperature (e.g., 50% , 25% , 25% ) yield even better results than the uniform mixture?
3.Could the authors further explore the boundaries of this trade-off? If the "Grafting" strategy incorporates more or higher-temperature trees (e.g., ), is there a point of diminishing returns where the fidelity drops too low (excessive mode-covering), causing the final performance (e.g., Avg. winning rate) to decrease as well?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This study re-examines knowledge distillation for autoregressive generative language models, emphasizing that its core value extends beyond mere model compression. While pure mode-seeking methods are applicable for compression, they restrict student models’ ability to capture linguistic diversity and hinder generalization on language modeling tasks. To address this issue, the authors propose a concise "grafting strategy" that balances mode-seeking and mode-covering via fusing multi-temperature sequence trees, along with flexibility for both strong-to-weak and weak-to-strong distillation scenarios.

### Strengths
1. This study conducts experiments covering three key tasks—general instruction following, mathematical reasoning, and code generation—with three major model families (Qwen, Llama, and DeepSeek-Coder). It also explores two distinct distillation scenarios: distilling from stronger to weaker models and from weaker to stronger models, effectively validating the proposed method’s effectiveness.
2.  Knowledge distillation requires balancing mode-seeking and mode-covering. The proposed grafting strategy achieves this balance by combining multi-temperature sequence trees, thereby enhancing the model’s generalization ability.
3.  The paper provides valuable insights into the relationship between student model capacity and grafting gains: stronger student models yield more significant benefits from the grafting strategy, while insufficient student capacity leads to limited or even no gains.

### Weaknesses
1. The authors have not provided open-source code for the proposed method. This lack of reproducibility hinders other researchers from verifying the experimental results, building upon the work, or comparing it with alternative approaches—an essential practice in academic research.
2. The validation scope across datasets is relatively limited. For instance, in the mathematical reasoning task, only two datasets (MathQA and GSM8K) are evaluated, while other widely adopted benchmarks such as AMC, AIME, MinervaMath, and OlympiadBench are not included. A similar gap exists for the other tasks (general instruction following and code generation), where the method’s performance on a broader set of standard datasets remains unvalidated. This limits the generalizability of the reported findings.
3. For the mathematical reasoning and code generation tasks, the proposed method has not been evaluated on state-of-the-art reasoning-specialized models (e.g., models optimized for step-by-step reasoning or domain-specific logical deduction). Given the growing relevance of such models for these tasks, this omission makes it difficult to assess the method’s competitiveness in real-world scenarios where reasoning-capable models are commonly used.

### Questions
1. Please supplement experiments on the broader datasets mentioned in the Weaknesses section (e.g., AMC, AIME for math reasoning) to verify the grafting strategy’s generalizability.
2. Please evaluate the method on currently popular reasoning-specialized models and demonstrate its effectiveness on these reasoning-focused architectures.
3. Under strict data efficiency constraints (e.g., 1/10 or less of conventional training data), might multi-temperature sample aggregation outperform your grafting strategy in the performance-efficiency trade-off by avoiding complex probability fusion overhead? Do your conclusions only hold with sufficient data, and can you supplement low-data regime experiments to clarify this?

### Soundness
2

### Presentation
2

### Contribution
2
