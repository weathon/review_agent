# Edit-then-Consolidate for Reliable Knowledge Editing

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Knowledge editing aims to update specific facts in large language models (LLMs) without full retraining. Prior efforts sought to tune the knowledge layers of LLMs, proving effective for making selective edits. However, a significant gap exists between their performance in controlled, teacher-forcing evaluations and their real-world effectiveness in lifelong learning scenarios, which greatly limits their practical applicability. This work's empirical analysis reveals two recurring issues associated with this gap: (1) Most of traditional methods lead the edited model to overfit to the new fact, thereby degrading pre-trained capabilities; (2) There is a critical absence of a knowledge consolidation stage, leaving new facts insufficiently integrated into LLMs' inference-time behavior under autoregressive generation, thereby leading to a mismatch between parametric knowledge and actual generation behavior. To this end, we propose \textbf{Edit-then-Consolidate}, a novel knowledge editing paradigm that aims to bridge the gap between theoretical knowledge editing methods and their real-world applicability. Specifically, \textbf{(1)} our framework mitigates overfitting via Targeted Proximal Supervised Fine-Tuning (TPSFT) that localizes the edit via a trust-region objective to limit policy drift; \textbf{(2)} Then, a consolidation stage using Group Relative Policy Optimization (GRPO) aligns the edited knowledge with CoT-based inference policy by optimizing trajectory-level behavior under comprehensive reward signals. Extensive experiments demonstrate our framework consistently improves editing reliability and generalization under real-world evaluations, while better preserving locality and pre-trained capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper identifies a critical deficiency in most current methods within the LLM knowledge editing field: they focus solely on knowledge injection while neglecting the crucial step of knowledge consolidation. The authors experimentally demonstrate the importance of this consolidation phase, especially in lifelong editing scenarios. Consequently, the paper proposes a novel model editing paradigm, Edit-then-Consolidate, to bridge this gap. This framework employs a two-stage process involving Targeted Proximal Supervised Fine-Tuning for editing and GRPO for consolidation. Experiments confirm that the proposed method achieves strong performance in terms of reliability, generalization, and locality.

### Strengths
- I strongly agree with the paper's motivation. The poor performance of existing LLM knowledge editing methods under real-world evaluation settings is a significant and pressing issue in the field, and this paper makes a notable contribution toward addressing it.

- The addition of a second stage for knowledge consolidation following the initial edit is an elegant, practical, and highly intuitive approach.

- The experimental setup is rigorous and commendably uses an evaluation framework that more closely mirrors real-world application scenarios.

- The authors have provided a complete and detailed list of hyperparameter configurations, which is greatly appreciated for ensuring reproducibility.

### Weaknesses
- The paper lacks a discussion of the method's editing efficiency and time cost. A primary goal of knowledge editing is to update model knowledge with minimal computational resources and at a high speed. The use of GRPO for consolidation seems likely to introduce significant efficiency overhead, particularly when the number of editing samples is large. I believe it is necessary for the authors to include a discussion of this time-cost trade-off in the main text.

- The paper only presents results on a sequence of 1000 edits. However, in a true lifelong editing task, a model must handle a much larger volume of updates. Other works, such as AlphaEdit, RLEdit, and UltraEdit, have evaluated their methods on several thousands, or even tens of thousands of sequential edits. How does the Edit-then-Consolidate framework perform under these more demanding, larger-scale conditions?

- How are the specific FFN layers for TPSFT application chosen? Is this a manual selection process, or is there an automated method? How does the choice of different layers impact the method's overall performance?

- minor typos, e.g., "Fig. ref?" (line 184)

I promise to raise my rating if the authors address the concerns I have raised.

### Questions
See Weaknesses.

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
The authors investigate why existing knowledge editing methods fail to maintain performance under continuous or lifelong knowledge updates. They focus on approaches that directly modify model parameters and demonstrate that the issue stems not from data insufficiency but from structural limitations, overfitting to new facts and the absence of a knowledge consolidation phase. As a result, models end up “storing” information without truly integrating it into their reasoning behavior.

To address this, the authors propose the Edit-then-Consolidate (EtCon) framework, introducing a consolidation stage that combines Targeted Proximal Supervised Fine-Tuning (TPSFT) and Group Relative Policy Optimization (GRPO) to bridge the gap between parametric knowledge and reasoning policy. Empirical results on Llama-3 and Qwen-2.5 show that EtCon substantially improves both reliability and generalization compared with prior editing methods.

### Strengths
1. The paper correctly identifies the limitations of existing knowledge editing methods, which tend to perform only fragmentary updates without ensuring integration into the model’s reasoning process. Its methodological setup, which assumes more realistic, real-world editing scenarios, is both coherent and practically motivated.

2. By combining TPSFT and GRPO, the authors propose an interesting solution that directly addresses a critical gap — the failure of parametric updates to propagate into the model’s reasoning policy. 

3. The method achieves substantial performance gains under sequential editing conditions, indicating its robustness in lifelong or continual update settings.

4. Overall, the paper is well-structured, and the inclusion of ablation experiments and analyses of reward hacking effectively reinforce the authors’ main claims and lend additional credibility to the proposed framework.

### Weaknesses
1. According to the taxonomy presented by the authors, the comparison with memory-based or meta-learning editing methods is incomplete. For example, systems such as GRACE [1] and RECIPE [2] should have been included to provide a fairer and more comprehensive empirical evaluation.

[1] Hartvigsen, T., Sankaranarayanan, S., Palangi, H., Kim, Y., & Ghassemi, M. (2023). Aging with grace: Lifelong model editing with discrete key-value adaptors. Advances in Neural Information Processing Systems, 36, 47934-47959.

[2] Chen, Q., Zhang, T., He, X., Li, D., Wang, C., & Huang, L. (2024, November). Lifelong Knowledge Editing for LLMs with Retrieval-Augmented Continuous Prompt Learning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (pp. 13565-13580).

2. The notion of a trust region in TPSFT remains ambiguous. The paper does not provide any mathematical justification or upper-bound analysis explaining how the choice of ε ensures locality or under what conditions the gradient drift can be effectively controlled. Consequently, the claim that clipping “empirically works” is not theoretically substantiated.

3. The rationale for layer selection in TPSFT is also unclear. The choice appears to rely on the general assumption that lower layers encode syntactic information while upper layers capture semantics, yet there is no empirical evidence such as layer-wise activation or factual neuron analysis supporting this decision.

4. Although the authors present experiments under the heading “Sequential Editing,” the experimental configuration lacks sufficient detail. It is unclear how many edits are applied per model instance, whether updates occur in batch or single-shot form, and how the evaluation accounts for interaction effects among edits.

5. The paper does not validate whether referencing only the immediately preceding model state for policy updates (policy-update chaining) is the most appropriate strategy. It also omits discussion of how the method handles interference or conflict with much earlier edits, which is critical for lifelong learning stability.

6. Finally, the logic behind the GRPO reward design is under-explained. The paper lists accuracy, format, cleanliness, and consistency as reward components but provides little theoretical reasoning for their combination, weighting, or potential trade-offs.

### Questions
1. A placeholder “ref?” appears at line 184 and should be corrected.

2. Please clarify whether there are potential conflicts or trade-offs among the GRPO reward components.

3. Provide deeper analysis of the CoT-based training strategy. How does the use of chain-of-thought reasoning affect performance in reasoning-centric or multi-step models? Would the results differ when applied to inherently reasoning-oriented architectures?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper attributes the significant performance gap between controlled teacher-forcing evaluations and realistic, auto-regressive, lifelong editing scenarios to two main factors: (1) edited models overfitting to new facts, thereby degrading pre-trained capabilities, and (2) the lack of a knowledge consolidation stage, which prevents new facts from integrating into the LLMs' reasoning policy. To address these issues, the authors propose Edit-then-Consolidate, a novel knowledge editing paradigm that mitigates overfitting through Targeted Proximal Supervised Fine-Tuning (TPSFT) and consolidates edited knowledge using Group Relative Policy Optimization (GRPO). Experimental results show that their framework improves editing performance in real-world evaluation settings.

### Strengths
1. This paper identifies an interesting and underexplored bottleneck in knowledge editing: the disconnect between parametric updates and reasoning behavior. The insight that successful editing requires not only injecting knowledge but also consolidating it into the model's reasoning policy is both novel and important.
2. This paper effectively leverages existing techniques from other fields (e.g., PSFT and GRPO) to address the challenges in knowledge editing, achieving notable performance gains.
3. The experimental settings for evaluating editing performance are comprehensive and well-defined, covering mainstream datasets, LLMs, and evaluation metrics.

### Weaknesses
1. This paper attributes the ineffectiveness of existing editing techniques to two factors: (1) edited models overfitting to new facts, and (2) the lack of a knowledge consolidation stage, but provides insufficient demonstration of these claims. Although the proposed EtCon framework enhances editing performance, this does not conclusively demonstrate that previous editing techniques necessarily lead to overfitting or that consolidation is a required process.
2. The implementation details of TPSFT are unclear and potentially risky. For example, while the paper states that TPSFT targets FFN parameters for editing, it does not specify which layers or whether all FFNs are updated. Moreover, directly replacing the generated answer with the new target fact in their original CoT data may be problematic, as such naive substitution could introduce contradictions or spurious influences that distort the model's reasoning patterns.
3. The authors employ CoT reasoning to answer the original atomic facts used for editing. It is unclear whether introducing CoT reasoning on such simple factual edits is necessary, as it may not reflect the genuine reasoning integration of the newly edited knowledge. A more straightforward way to evaluate whether the updated knowledge has been successfully integrated into the model's reasoning policy is to test performance on multi-hop questions related to the edited facts.

### Questions
1. Inconsistent statements. In Line 045, the authors state that "Knowledge editing methods can be categorized into **three** main paradigms," whereas in Line 118, they claim that "Knowledge editing methods for LLMs fall into **two** paradigms."
2. Typos. Line 183 "Fig. ref? ".
3. Why is the general capability performance (e.g., on benchmarks such as C-Eval) for the Consolidate stage missing in Table 4?

### Soundness
2

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
4

### Summary
The paper argues that the gap between teacher-forcing knowledge–editing results and real-world, auto-regressive performance stems from missing consolidation: after injecting a fact into the weights, models often fail to use it consistently at inference time. The authors propose a two-stage pipeline: (i) Targeted Proximal SFT (TPSFT), which edits only selected FFN layers under a trust-region objective and with CoT-augmented labels; and (ii) a GRPO-based consolidation step with a composite reward (accuracy/format/cleanliness/consistency). Across ZsRE, CounterFact, and QAEdit on Llama-3-8B-Instruct and Qwen2.5-7B-Instruct, EtCon reports large gains in Reliability and Generalization while maintaining Locality and general capabilities

### Strengths
(1) In autoregressive generation, the paper shows that this missing consolidation stage is the main bottleneck for current methods and argues that editing should be a two-stage process: first injecting the new fact into model weights, then explicitly consolidating it so the model’s reasoning policy actually uses that fact.
(2) Under sequential auto-regressive evaluation across ZsRE, COUNTERFACT, and QAEdit, EtCon substantially outperforms baselines like FT-M, WISE, MEMIT, and ALPHAEDIT on Reliability and Generalization, while keeping acceptable Locality and avoiding catastrophic collapse of pretrained capabilities.

### Weaknesses
(1) The paper reports R/G/Locality and broad general-ability benchmarks, but lacks an explicit portability test showing that the newly edited fact transfers to broader downstream tasks which weakens the claim that consolidation improves real reasoning with the new fact.
(2) The paper states that existing methods “overfit to new facts,” degrading general capabilities, but the presented results often show something different: many baselines simply fail to perform the edit at all (e.g., FT-M and WISE have single-digit Reliability on Qwen2.5-7B). This looks less like classic overfitting-to-the-new-fact (high edit success but poor locality) and more like failure to edit or model collapse, so the causal narrative around “overfitting” should be clarified.
(3) The paper repeatedly frames the setting as “lifelong” and “sequential,” and it reports results over 1000 sampled instances per dataset, but it never clearly states how many edits are actually applied in sequence to the same model before evaluation. Without stating the length of the edit trajectory or showing performance as the number of edits grows, it’s hard to judge long-term stability.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
