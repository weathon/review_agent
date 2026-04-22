# Mixture of Groups: Grouped Gating and Cross Mixing for Parameter-Efficient LLM Fine-Tuning

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Low-rank adaptation (LoRA) is a widely used parameter-efficient fine-tuning (PEFT) method for large language models. However, by learning independent adapters for each layer, LoRA and its variants ignore the inherent functional similarity between adjacent layers, limiting their potential to fully exploit the hierarchical representations across depth. To address this, we propose Mixture of Groups (MoG), a novel group-sharing framework that partitions layers into functional groups, shares low-rank adapters within each group, and employs adaptive gating and cross-mixing mechanisms to enable flexible fine-tuning. This approach leverages inter-layer similarity to capture both commonalities and unique characteristics across layers, achieving a more efficient and expressive subspace than LoRA. Moreover, MoG is designed as a plug-and-play framework that can be seamlessly integrated into other PEFT methods such as DoRA and PiSSA to boost their performance. Extensive experiments on multiple benchmarks demonstrate that MoG achieves overall superior performance compared with prior methods under comparable parameter budgets, highlighting its ability to combine efficiency with strong downstream effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Mixture of Groups (MoG), a new parameter-efficient fine-tuning (PEFT) framework for large language models. It clusters adjacent layers into functionally similar groups that share low-rank adapters and augments them with adaptive gating plus cross-mixing, breaking the dichotomy between per-layer LoRA and fully shared approaches. Extensive experiments on several LLMs and benchmarks show that MoG delivers higher accuracy than previous PEFT baselines under the same or smaller parameter budgets.

### Strengths
1. Thoughtful design: grouping, gating, and cross-mixing are well-motivated by empirical analyses of inter-layer similarity and provide a principled middle ground between fully shared and fully independent adapters.

2. The orthogonal design enables MoG to benefit from a wide range of LoRA variants.

3. The paper is well written and easy to follow.

### Weaknesses
1. Comparative puzzle: vanilla LoRA underperforms DoRA in your tables, yet LoRA+MoG surpasses DoRA+MoG; please clarify why MoG benefits LoRA more than DoRA—does MoG overlap with DoRA’s decomposition or were hyper-parameters unequal?

2. From my experience, llm-adapter framework is very sensitive to learning rates in the range {1e-4, 2e-4, 3e-4}. The vanilla LoRA result on Llama-3 8B was likely obtained with a learning rate of 3e-4, whereas the other methods were not. Could you rerun LoRA on Llama-3 8B with a 1e-4 learning rate and report the results to provide readers with a clearer comparison?

### Questions
please see weakness

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
This paper introduces MoG, a novel framework for PEFT of LLMs. The paper identifies a key limitation in the popular LoRA method: by learning independent adapters for every layer, LoRA ignores the functional similarity between adjacent layers, leading to parameter redundancy. Conversely, other methods that rely on global parameter sharing can overlook layer-specific functional diversity, thereby limiting expressiveness.

### Strengths
- Balanced Framework: MoG introduces a novel group-sharing framework that balances parameter efficiency by reducing redundancy with layer-specific expressiveness.
- Flexible Architecture: The model uses adaptive gating and cross-mixing to create a highly flexible and expressive adaptation subspace that generalizes LoRA.
- Versatile Plug-and-Play Module: MoG is designed as a general-purpose module that can be plugged into other PEFT methods, such as DoRA and PiSSA, to boost their performance.
- Strong Empirical Results: The method is validated by extensive experiments across multiple LLMs and benchmarks, demonstrating superior performance over strong baselines with comparable parameter budgets.

### Weaknesses
- `Limited Motivation and Static Mechanism`: The paper claims to employ adaptive gating and cross-mixing mechanisms to capture both commonalities and unique features across layers by leveraging inter-layer similarity. However, a significant limitation is that the fusion weights generated for each layer or group are static. This approach appears disconnected from the paper's own findings, as illustrated in Figure 1(a), which clearly indicates dynamic variations in cosine similarity between different layers. The potential relationship between these dynamic similarity scores and the static fusion weights is a critical, unexplored area that warrants further investigation.
- `Limited Comparative Analysis`: The paper's methodology draws inspiration from MoE principles to some extent. Despite this, the experimental evaluation conspicuously omits comparisons against relevant and contemporary MoE-based LoRA variants ($e.g.$, [1-3]) as baselines. I recommend incorporating these methods to provide a truly comprehensive and meaningful comparison. Furthermore, the experimental scope is narrow regarding model scale and architecture. The analysis would be substantially strengthened by including evaluations on models of varying sizes, such as Qwen2.5-3B, Qwen2.5-7B, and Qwen2.5-14B, to demonstrate the method's scalability.
- `Limited Application Scenarios`: The empirical validation is currently confined to single-task environments. This narrow focus significantly limits the method's potential applicability and generalizability, particularly for the increasingly prevalent and complex multi-task scenarios found in real-world applications. To substantiate the robustness and effectiveness of the proposed approach, it is essential to extend the evaluation to include challenging multi-task benchmarks ($e.g.$, OpenOrca).

[1] When MOE Meets LLMs: Parameter Efficient Fine-tuning for Multi-task Medical Applications

[2] HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning

[3] CoLA: Collaborative Low-Rank Adaptation

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Large language models (LLMs) excel in NLP but are costly for full fine-tuning. Low-Rank Adaptation (LoRA), a common Parameter-Efficient Fine-Tuning (PEFT) method, uses independent per-layer adapters, causing redundancy and wasting inter-layer similarity. We propose **Mixture of Groups (MoG)**, a new PEFT framework: it splits LLM layers into groups (sharing low-rank adapters to cut redundancy), uses adaptive gating for layer-specific parameter combination, and adds a cross-mixing module to boost representational capacity.  

As a plug-and-play tool, MoG integrates with DoRA/PiSSA to enhance their performance. Experiments on commonsense reasoning, math, code generation, and GLUE show MoG outperforms LoRA/DenseLoRA under similar parameter budgets, balancing efficiency and effectiveness in LLM fine-tuning.

### Strengths
1.	Focusing on an important problem of existing PEFT methods: Clearly points out the limitations of LoRA (redundancy from independent per-layer adapters) and global sharing methods (e.g., VeRA, DenseLoRA, which ignore inter-layer functional diversity). It also verifies the functional similarity of adjacent LLM layers through experimental evidence like DOCS analysis and t-SNE visualization, providing solid theoretical and data support for the method design.
2.	Flexible architecture with strong compatibility: The MoG framework has a "plug-and-play" feature. It can be used independently as a PEFT method and seamlessly integrated into mainstream PEFT methods such as DoRA and PiSSA (e.g., DoRA(+MoG) and PiSSA(+MoG) both improve the performance of the original methods). It also supports architecture degradation (degenerates to LoRA when n=L and E is an identity matrix, and is similar to DenseLoRA when n=1), adapting to different fine-tuning needs.
3.	Modular component design with high interpretability: The three core components of MoG ("group sharing-adaptive gating-cross mixing") have clear functions and divisions of labor. Ablation experiments prove that no component is dispensable (removing the cross-mixing module reduces performance by 1.2%, and removing gating reduces it by 1.4%). Meanwhile, gating coefficient visualization shows that the gating distribution changes from "one-hot" to "flexible" after training, which dynamically adjusts the weight of group parameters according to tasks. The working mechanism of components is interpretable, facilitating subsequent optimization and improvement.
4.	Rigorous and comprehensive experimental design, with credible conclusions: Experiments cover multiple task types (reasoning, generation, understanding) and models of different scales. Baseline methods include mainstream PEFT methods like LoRA, RaSA, and DenseLoRA. A unified parameter budget ensures fair comparison. In addition, extended experiments (parameter efficiency analysis, group number impact analysis, resource consumption analysis) are added to verify the method’s advantages from different dimensions, making the conclusions more convincing.

### Weaknesses
1.	Limited exploration of group number selection. The optimal number of groups (n) varies by task/model, but no clear guidelines for choosing n are provided.
2.	Less testing on extremely large models. Most experiments use 7B/8B models; performance on larger models (e.g., 70B) remains unproven.
3.	Higher complexity than basic LoRA. The cross-mixing module and gating mechanism add minor computational overhead, though small.
4.	Limited analysis of hyperparameter sensitivity. How factors like rank (r) or learning rate affect MoG’s stability needs more study.

### Questions
1. Could you provide practical guidelines or automated methods to determine the optimal number of groups (n) for different tasks and model scales?  
2. Have you tested MoG on larger models (e.g., 70B LLMs)? If not, do you expect similar performance gains there?   
3. How sensitive is MoG to hyperparameters like rank (r) or learning rate? Are there stable ranges to recommend?

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
4

### Summary
This paper addresses the parameter redundancy problem in LoRA, which ignores the functional similarity between adjacent layers. It proposes a new PEFT framework called MoG. MoG reduces redundancy through group sharing, while using gated adaptive aggregation and cross mixing mechanisms to keep each layer’s specificity. This design achieves a delicate balance between parameter sharing and model expressiveness. Through extensive experiments, the authors show that MoG significantly outperforms LoRA and other sharing methods under the same parameter budget, achieving much higher parameter efficiency. Moreover, MoG is a general plug-and-play module that can be combined with other PEFT methods such as DoRA and PiSSA to further improve performance. Overall, MoG introduces an innovative and efficient new PEFT paradigm.

### Strengths
- The paper is well-written.
- The method is simple, and the plug-and-play design brings some performance gains.

### Weaknesses
- The paper lacks comparisons with related work. Many studies have optimized LoRA from the efficiency perspective, some focus on reducing the rank, while others focus on parameter sharing, but the baselines here do not include these methods. (https://arxiv.org/pdf/2406.10785 ; https://openreview.net/pdf?id=IXYBuwCOMl; https://arxiv.org/html/2410.11772v1 ; https://arxiv.org/html/2402.08562v1 ; https://arxiv.org/abs/2410.19694)
- The idea of "less is more" in PEFT has been discussed in many works (including but not limited to those mentioned above). Therefore, the contribution of this paper may be limited.
- The paper also lacks quantitative analysis of how much MoG actually reduces parameters. Since GPU memory usage mainly comes from the base model, the computational cost reduction of MoG may be limited.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
