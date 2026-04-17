# Knowledge Fusion of Large Language Models via Modular SkillPacks

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 4, 6

## Abstract
Cross-capability transfer represents a key challenge in large language model (LLM) research, particularly in multi-task integration, model compression, and knowledge fusion. Recent works such as FuseLLM and FuseChat have shown the potential of transferring multiple model capabilities to lightweight models, thereby enhancing adaptability and efficiency. This motivates our investigation into more efficient methods for cross-capability transfer. However, existing merging approaches primarily focus on small, homogeneous models, limiting their applicability.
For large, heterogeneous models, knowledge distillation with full-parameter fine-tuning often overlooks the student model’s inherent capability and risks catastrophic forgetting, while PEFT methods struggle to effectively absorb knowledge from source LLMs.
To address these issues, we introduce **GraftLLM**, a novel grafting-based method that stores source model capabilities in a target model + SkillPack format. This approach preserves general capabilities, reduces parameter conflicts, and supports forget-free continual learning and model fusion. We employ a module-aware adaptive compression strategy for parameter updates, ensuring efficient storage while **preserving task-specific knowledge**. The resulting SkillPack serves as a compact and transferable knowledge carrier, ideal for **heterogeneous LLM fusion**.
Experiments across various scenarios demonstrate that GraftLLM outperforms existing techniques in knowledge transfer, knowledge fusion, and forget-free learning, providing a scalable and efficient solution for cross-capability transfer.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces GraftLLM for transferring capabilities from heterogeneous source LLMs into a single target LLM. The key contribution of GraftLLM is to compress task-specific knowledge distilled from various source models into compact, module-wise "SkillPacks", effectively preserving the target model's knowledge and preventing catastrophic forgetting. These lightweight SkillPacks can be flexibly routed and integrated into the target model, offering a plug-and-play method for practical deployment. Comprehensive experiments span pairwise grafting, knowledge fusion and forget-free learning demonstrate the effectiveness of GraftLLM.

### Strengths
1. This paper introduces an innovative framework that creatively combines knowledge fusion, model compression, and model rounting. The introduced "SkillPack" is flexible and plug-and-play, offering a cost-effective solution for scalable, efficient, and forget-free knowledge transfer across heterogeneous LLMs.
2. The work is technically strong. The proposed module-aware adaptive compression strategy is well-motivated and well-designed to preserve performance while minimizing parameter overhead. The experimental methodology is rigorous, featuring comprehensive comparisons against various baselines across three distinct scenarios.

### Weaknesses
1. The presentation of the paper has several issues that hinder readability. These include, but are not limited to:

- **Grammatical errors**: For example, the word "heterogeneous" is repeated on line 26. 

- **Mismatched citations**: For instance, the citations for FuseChat  (Yang et al., 2025b) on line 54 and synthetic data  (Wan et al., 2024b) on line 72 appear to be swapped. 

- **Mismatched table captions**: The caption for Table 6 does not correspond to its content. The caption of Table 10 should appear before the table.

- **Logical contradiction**: Appendix B.3 states, "A five-way classifier is then trained to predict the most suitable source model for each input," which suggests the routing function $\mathcal{R}$ selects the best SkillPack on a per-prompt basis. However, this appears to conflict with the description in Section 3.3 (lines 270-272), which defines $\mathcal{R}$ as a submodule-specific router that "dynamically assigns each SkillPack to the appropriate submodules." The authors should rewrite Section 3.3 to clarify this mechanism and resolve the ambiguity.
  
2. Several experimental details are missing, which poses a challenge for reproducibility. These omissions include:

- **Insufficient details about the router**: For the explicit fusion experiments, the paper mentions training a router to assign SkillPacks. However, the details of this router (e.g., its architecture, the distribution of training data, and its computational overhead) are too brief.

- **Lack of detail on model combination**: Section 3.2 explains how a SkillPack is created through compression, but the paper does not explicitly describe how these compressed delta parameters are combined with the full parameters of the target model to form the final, fused model.

- **Missing details for pairwise distillation in implicit fusion**: Section 5.2 describes experiments on implicit fusion, which involves a pairwise distillation step. However, the implementation details provided in Appendix F.1 (lines 1269-1318) appear to describe the process for the FuseChat-3 baseline, not the specific distillation process used to generate the SkillPacks for GraftLLM in this setting.

- **Insufficient detail on hyperparameter selection**: The module-aware adaptive compression strategy described in Section 3.2 introduces several tunable hyperparameters (e.g., SVD rank, pruning ratio, quantization bitwidth, energy threshold $\beta$). However, the paper does not provide enough detail on how the final values for these hyperparameters were selected.

### Questions
1. What is the inference overhead of the proposed method? Specifically, does loading multiple SkillPacks significantly increase inference latency? Additionally, what is the computational cost associated with training the router?
  
2. For training the router module in explicit model fusion, what is the rationale for using training loss values as supervision signals? Furthermore, how is the accuracy of this routing mechanism ensured, and have you analyzed whether a potential distribution shift between the router's training data and the test set could affect its performance?
  
3. In the implicit fusion experiments (Table 2), could you clarify how the models for GraftLLM's pairwise distillation step were trained? What were the individual performances after distilling from each source model, before the final fusion? The overall performance gain over the FuseChat-3 baseline is modest; could you elaborate on the trade-off between this performance gain and the complexity/overhead of the GraftLLM approach in this specific scenario?

4. I am willing to raise my score if the authors can satisfactorily address the weaknesses and questions I have raised.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a graft method, GraftLLM, that extracts the source models' capabilities as SkillPack and then dynamically assigns capabilities to the target model based on the tasks. Moreover, this method employs a combination of compression techniques, including pruning, quantization, and decomposition, to make the SkillPack compact and efficient. The router function is trained on FuseChat 2.0 to explicitly select SkillPack based on the source model and task type. The experiments show that GraftLLM outperforms the other merging or fusion methods with comparable parameters on multiple tasks.

### Strengths
1. Innovative router-based skill integration:
The paper introduces a novel router function that dynamically assigns SkillPacks to the target model, enabling flexible and adaptive composition of capabilities. This mechanism allows the target model to seamlessly adapt to diverse tasks without full retraining.

2. Support for heterogeneous model fusion:
Unlike most prior works that focus on homogeneous architectures, the proposed method is applicable to heterogeneous LLMs, demonstrating strong generality and robustness across models with different structures and parameterizations.

3. Strong empirical performance with compact models:
The proposed GraftLLM achieves superior performance across multiple benchmarks while maintaining a comparable parameter count to competing methods, highlighting its efficiency and effectiveness.

### Weaknesses
1. Terminology and clarity issues:
The paper assumes substantial prior knowledge and introduces multiple technical terms without sufficient explanation. For instance, PEFT is used in the abstract and appears frequently throughout the paper, but it is never explicitly defined. Similarly, the term modules is used ambiguously; it would help if the authors clarified that they refer to specific neural network components, such as attention layers or MLP blocks. Additionally, there are several minor typographical errors, particularly in Section 3, that should be corrected for readability.

2. Methodological complexity and unclear contribution emphasis:
While the proposed approach is empirically strong, it combines more than five techniques, which makes it conceptually complex and somewhat difficult to parse. According to the ablation study, the optimal configuration involves using pruning for the embedding and output head, SVD for MLP layers, and low-rank SVD for attention layers. However, in such a composite pipeline, the reported parameter reduction may not translate into practical efficiency—training and validation runtimes could be substantially longer than for competing methods. Although a runtime analysis is included, it lacks a direct comparison with other baselines. Moreover, if similar compression strategies (e.g., pruning, SVD) were applied to the competitors, they might outperform the proposed method. The paper would be stronger if the authors highlighted the most impactful component rather than treating all sub-techniques as equally central, as the current presentation dilutes the core contribution.

3. Insufficient explanation of the router function:
The router function appears to be the conceptual centerpiece of the paper, yet its formulation and role are not clearly articulated in the main text. A more detailed exposition—possibly with a schematic or mathematical description—would make the contribution clearer and could inspire future extensions of this idea.

### Questions
1. Runtime and complexity concern:
Could you provide a runtime comparison against the competing methods, including both training and validation phases? The proposed framework integrates several techniques, which may introduce substantial computational overhead and tuning difficulty despite the parameter reduction. A comparative runtime analysis would clarify the practical efficiency of the approach.

2. Clarification of the router function:
Could you elaborate on the formal definition and empirical behavior of the router function? It appears to be the key innovation that unifies the multiple techniques and enables dynamic capability transfer across heterogeneous models. A clearer explanation would help readers understand its role and significance.

### Soundness
3

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
The paper addresses the problem of LLMs merging. The paper proposed an approach which consists of storing source model capabilities in a target model + SkillPack format. According to the paper, this approach preserves general capabilities, reduces parameter conflicts, and supports forget-free continual learning and model fusion. According to the paper, the approach proposed outperforms existing techniques.

### Strengths
The paper addresses a well known problem when developing or adapting LLMs. Instead of developing a completely new model, the goal is to merge existing models. The paper proposes a new approach which  preserves general capabilities, reduces parameter conflicts, and supports forget-free continual learning.

### Weaknesses
The main issue with the paper is self contained which makes its review difficult. The paper used several key technical terms such as “cross-capability transfer”, “SkillPack”, MLP, etc. without providing the meaning, which can make the paper very difficult to understand by newcomers.

The problem addressed by the paper is not clear and there is a confusion in the paper contribution:
- The paper is addressing the problem when models are structurally identical
- What is the difference between heterogeneous models (line 58) and structural different model (line 60) that the paper wants to solve? These terms lead to confusion. The paper should clarify all the terms used.

The methodology is not understandable, which did not make the reproducibility easy. Several capital terms are not clearly defined. Figure 4 is not well explained. Section 3.2 is not clear

Some description of key terms (line 267) are provided in additional materials. However, the paper should be self contained.

Experimentation environment, hyper-parameter, etc. are not provided in the experimentation section.

The evaluation of the time complexity and difficulties to integrate the models are not provided.
- Line 26: for heterogeneous heterogeneous LLM fusion –> there is a redundancy of the term “heterogeneous”
- Missing experimentation and results summary in the abstract. The paper could present to which approach the proposed approach was compared to.
- SkillPack, cross-capability transfer should be defined before first use in the introduction and abstract
- In the abstract, the paper could introduce the experiments, present to which baseline their approach was compared to and summarize some results.

- Line 106: “SFT and DPO” should be defined before first use
- Line 112-113 is not clear. 
- The figures presented in the paper should be accompanied by a description text which makes its understanding easy.
- The definition of SkillPack in line 215 is too short and not understandable for such a technical term. The paper should provide a clear explanation of what SkillPack is (with examples).
- The ablation may provide the evaluation (costs in time of the different approaches) and  the integration difficulties.

### Questions
- What is Modular Skill Pack?

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GraftLLM, a framework for transferring capabilities across heterogeneous large language models by creating and composing modular "SkillPacks." The method involves fine-tuning a target model, extracting the parameter delta, applying a module-aware compression strategy, and composing these SkillPacks using a routing mechanism. The approach is evaluated on tasks including pairwise knowledge transfer, heterogeneous model fusion, and forget-free continual learning.

### Strengths
The paper tackles the highly relevant and challenging problem of fusing knowledge across heterogeneous LLMs. The proposed "module-aware adaptive compression" strategy is an intuitive and empirically effective contribution for creating compact, transferable knowledge modules. The experimental results consistently demonstrate strong performance across multiple benchmarks, outperforming several baselines in knowledge fusion.

### Weaknesses
The technical contribution of this work appears to be incremental. The proposed pipeline—distilling knowledge, calculating a delta, compressing it, and then composing these modules—is conceptually very similar to existing frameworks like FuseChat (pairwise distillation followed by merging) and LoraHub (dynamic composition of LoRA modules). The main novelty seems to lie in the "module-aware adaptive compression" strategy, but the overall framework feels like a combination of established techniques rather than a new approach. The paper would be stronger if it more clearly articulated its core conceptual departure from these related works.

Furthermore, the analysis provided is limited and somewhat misaligned with the paper's main theme of knowledge fusion. The primary analysis focuses on ablation studies of the compression strategy (Table 4) and its impact on performance loss. While this validates a component of the method, it does little to illuminate the dynamics of the fusion process itself. An analysis of inter-SkillPack interference, the behavior of the router on ambiguous inputs, or a qualitative look at what knowledge is captured within a SkillPack would have been more insightful for a paper centered on fusion.

The paper is difficult to follow in several key areas, with missing details and confusing notation that undermine the reader's confidence in the methodology and results.

1.  **Inconsistent Notation**: The notation for the base model is inconsistent. Equation (10) uses `θ_tgt`, while the description on line 270 uses `θ_target`. This should be standardized for clarity.

2.  **Missing Quantization Details**: The method utilizes GPTQ for quantization, which typically requires a calibration set to determine quantization parameters. The paper fails to mention what calibration data was used, how it was selected, or its size. This is a critical detail for reproducibility.

3.  **Confusing Formulation of the Router**: Equations (10) and (11) are nearly identical, yet they are meant to describe different scenarios. This implies that the router `R` and the SkillPacks themselves might function differently in each context, which is highly confusing. The paper should explicitly state this difference. Moreover, the appendix reveals that two different types of routers are used—one trained as a classifier and another based on manual, task-type assignment. This crucial distinction is not clearly made in the main paper. Important details about the trained router's architecture, training data, and hyperparameters are also missing.

4.  **Misleading Claims about Parameter Size**: The paper claims its method is efficient, yet in Table 1, the parameter count for "Routed GraftLLM (Ours)" is 9.2B, a 28% increase over the 7B base model. This increase can hardly be considered minor. Why does a method centered on *compression* result in a significantly larger final model?

5.  **Flawed Comparison in Continual Learning**: Table 3 evaluates forget-free learning. The setup is problematic for two reasons. First, it is missing some critical results: the performance of the LLaMA3 model after being fine-tuned sequentially on different stages. Second, the paper states that the GraftLLM router for this experiment is a manual one based on "task-type." If the model is explicitly told which task it is performing (e.g., "this is a math problem, load the math SkillPack"), then the comparison to other methods that do not receive this oracle information is unfair.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

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
This paper identifies a critical challenge in cross-capability transfer for LLMs, i.e., existing methods like full-parameter knowledge distillation often cause catastrophic forgetting, while standard PEFT methods are ineffective at absorbing comprehensive knowledge from powerful source models. To address this, the authors propose GraftLLM, a novel grafting-based framework. The core idea is to distill and store capabilities from source models into modular, compressed "SkillPacks." These SkillPacks can then be attached to a target model to fuse new capabilities, preserving the target's original knowledge. The method employs a "module-aware adaptive compression" strategy to create these SkillPacks, aiming to reduce parameter conflicts and support "forget-free" continual learning and model fusion.

### Strengths
- I think the authors did a very good job of identifying a key gap: the lack of a method that both effectively absorbs deep knowledge from a source model (like distillation) and preserves the target model's inherent capabilities (which distillation often fails to do).
- The method introduced is sound and the reframing of model fusion as a modular composition problem is quite novel. 
- A "forget-free" method for adding new, complex capabilities to a base model is highly sought after. This approach has clear significance for building specialized models on demand without the need for costly full-scale retraining.

### Weaknesses
- I think the baseline comparisons is a bit ambiguous. The abstract positions the work against distillation and PEFTs, and mentions FuseLLM/FuseChat for small models. However, it's unclear if GraftLLM is benchmarked against current SOTA model merging techniques for large models. These methods also aim to fuse capabilities and are a crucial point of comparison. 
- The framework's practicality hinges on the cost of creating the SkillPacks. Can the authors please quantify the overhead of their method? What is the computational cost (e.g., GPU hours) to create a typical SkillPack? What is the inference-time impact (e.g., added latency, extra VRAM) of using a target model + 1 SkillPack?
- A key test for a fusion method is its ability to handle conflicting or highly distinct capabilities (e.g., fusing a "math" SkillPack and a "poetry" SkillPack). How does GraftLLM handle fusing multiple SkillPacks that may have conflicting parameter updates like this example?

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3
