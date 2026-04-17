# Beyond Benchmarks: Understanding Mixture-of-Experts Models through Internal Mechanisms

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Mixture-of-Experts (MoE) architectures have emerged as a promising direction, offering efficiency and scalability by activating only a subset of parameters during inference. However, current research remains largely performance-centric, with limited understanding of its internal mechanisms, thereby constraining broader progress. In this work, we use an internal metric to investigate the mechanisms of MoE architecture by explicitly incorporating routing mechanisms and analyzing expert-level behaviors. Through systematic analyses of a wide range of publicly available MoE models, we uncover several findings: (1) neuron utilization decreases as models evolve, reflecting stronger generalization; (2) training exhibits a dynamic trajectory, where benchmark performance alone provides limited signal while MUI reveals deeper insights; (3) task completion emerges from collaborative contributions of multiple experts, with shared experts driving concentration; and (4) activation patterns at the neuron level provide a fine-grained proxy for data diversity. Together, these results demonstrate the potential of MUI as a complementary indicator to benchmark performance, offering new insights into the capacity, dynamics, and specialization of MoE models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper extends the MUI (Model Utilization Index) introduced by Cao et al. (Model Utility Law: Evaluating LLMs beyond Performance through Mechanism Interpretable Metric, https://arxiv.org/abs/2504.07440) to MoE models. Based on the original MUI formulation as well as the extension, the authors perform a series of experiments on open-weights MoE models and families: GPT-OSS, Qwen, DeepSeek, and OLMoE. The main results include (1) tracking the neuron utilization as model family evolves, (2) tracking the changes to MUI during training based on OLMoE checkpoints (3) investigation of the expert collaboration and neuron activation within experts, and (4) investigation of the MoE .
It is shown with ablations that some aspects of the work that could seem arbitrary ($\eta$'s, MUI equation) are actually meaningful.

### Strengths
1. Results are based on investigation across different model families.
2. The work is based on an established method, contributing to the soundness of the method and simplifying comparison with dense models.
3. Presentation quality is good and the paper is easy to read.

### Weaknesses
1. The training dynamics experiments are limited to OLMoE, although at least one other open-weights family with intermediate checkpoints exists (Xue et al., Openmoe: an early effort on open mixture-of-experts language models)
2. I think the paper would benefit from (some amount of) comparative analysis with dense models. The baseline method (Cao et al. https://arxiv.org/abs/2504.07440) is proposed for dense models, however little comparison of this paper's results for MoE models with the results for dense models is made.
3. No custom-trained models. Although such addition is not necessary and would require considerable compute, measuring MUI and related metrics on models which differ only in key aspects (such as whether there are shared experts) while the rest of the pipeline (dataset, etc.) is kept the same between models would allow for much more authoritative claims.
4. Figure 3 would benefit from having a separate version for each benchmark, containing all intermediate checkpoints (even if just in the appendix)
5. As far as I can understand, the method used can only be used to investigate model behavior in a statistical, aggregated manner (across a dataset).

### Questions
1. The authors consider an expert important based on the contribution of its neurons, however another natural candidate for this task emerges - the routing decisions (at least in case of non-shared experts). Have the authors considered comparing their method (in sections related to expert-leven contribution) to an approach based on the expert-use frequency?
2. In figure 8. the MUI rises steadily as the number of samples is increased. Does that happen indefinitely? Is it possible that higher MUI is simply correlated with the dataset being larger?
3. Is there any more evidence for there existing separate ``accumulating'' and ``evolving'' phases? Does something interesting happen during these phases? These are introduced, however it seems their importance is very limited, as they do not even transfer between tasks.
4. Can the method be used to interpret the model behavior in a more fine-grained manner, i.e., within a single sample, not across a dataset.
5. Can the ablations (section 4) be performed on experts instead of neurons? Considering the paper introduces the notion of key experts, such ablation could prove that such notion is indeed meaningful.

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
5

### Summary
- Demonstrates that evolved models achieve better performance with lower neuron utilization, suggesting MUI reflects generalization capability beyond benchmark scores. Training exhibits a two-phase trajectory: an "Accumulating" phase where MUI increases with performance, followed by an "Evolving" phase where MUI decreases as the model becomes more efficient.

- Reveals that task completion emerges from collaborative contributions of multiple experts rather than isolated specialization. Stronger models exhibit higher proportions of key experts working together, with GPT-OSS showing the highest collaboration rates. However, in architectures with shared experts, task responsibility concentrates heavily within the shared pool, potentially limiting the diversity benefits of distributed experts.

### Strengths
- Introduces a principled neuron-level analysis methodology for MoE models that explicitly accounts for routing mechanisms and expert-level behaviors, extending beyond performance-only evaluation. The approach is validated through multiple ablation studies (threshold variations, alternative importance methods) and causal intervention experiments (neuron masking) showing high correlation across formulations.

- Analyzes 13 production MoE models (20B-200B parameters) plus 10 training checkpoints across 7 diverse benchmarks, revealing consistent patterns across model families. The discovery that evolved models achieve lower MUI while improving performance provides actionable insights for model development.

- Provides the first evidence that shared experts dominate task responsibility, challenging assumptions about expert specialization in MoE architectures. The two-phase training trajectory discovery (Accumulating → Evolving) offers practical guidance for monitoring training and identifying when models need more task-specific data.

### Weaknesses
- While the paper claims to investigate "internal mechanisms," it primarily provides correlational observations without explaining why lower MUI corresponds to better generalization or what semantic features the utilized neurons represent. The neuron contribution method relies on logit lens projections, which have known limitations—they assume linear representation geometry and may not capture the full computational role of neurons in multi-layer nonlinear transformations. The paper never validates whether the identified "key neurons" actually encode interpretable features (e.g., arithmetic operations, syntactic patterns) or are merely correlates of the output distribution. Without grounding MUI in actual semantic or functional properties of neurons, it remains unclear whether low MUI reflects genuine capability compression or simply different architectural routing patterns.

- The paper demonstrates that MUI changes during training but provides no concrete methodology for using MUI to improve model development. The recommendation that "OLMoE needs more coding data" is post-hoc and could have been identified through standard per-task performance monitoring. Critically missing: (1) Can MUI predict downstream performance on held-out tasks? (2) What MUI threshold indicates a model is ready to transition training phases? The "Takeaway" boxes offer vague guidance ("monitoring performance alone is insufficient") without quantitative decision rules or experimental validation that MUI-guided interventions actually improve final model quality.

- The neuron masking experiments only validate that identified neurons are causally relevant but do not establish that MUI uniquely captures generalization over alternative explanations. Lower MUI in evolved models could reflect: (1) better optimization leading to sparser representations, (2) architectural differences, (3) post-training procedures, or (4) differences in training data scale/quality. The correlation between low MUI and strong performance in GPT-OSS is particularly questionable since these models differ in every dimension (architecture, scale, data, training recipe) from the comparison models.

- The definition of "key neurons" depends critically on the top-0.1% threshold, yet the justification is primarily empirical convenience rather than principled theory.

### Questions
- Can you provide evidence that MUI captures generalizable capacity rather than architectural artifacts? Please report MUI for models trained on identical datasets with only architectural variations (e.g., different expert counts, with/without shared experts), or show that MUI computed on training data predicts held-out task performance. 

- What are the concrete, quantitative decision rules for using MUI in practice? Can you specify: (1) MUI thresholds that indicate when to transition from "Accumulating" to "Evolving" phase training, (2) what MUI delta triggers data mixture adjustments, and (3) empirical validation that MUI-guided interventions (e.g., adding coding data when coding MUI stays high) actually improve final model quality compared to standard training? Without this, MUI remains a post-hoc diagnostic rather than an actionable training tool.

- How do you disentangle neuron function from routing artifacts in your MUI measurements? Given that your method projects through the unembedding matrix (Equation 2), have you verified that "key neurons" encode interpretable features relevant to the task?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel approach to understanding Mixture-of-Experts (MoE) models by analyzing their internal mechanisms rather than relying solely on benchmark performance. The authors adapt the Model Utilization Index (MUI) (orginally proposed by [Cao et al., 2025](https://arxiv.org/abs/2504.07440)) to MoE architectures and conduct extensive analyses on 3 different series (13 discrete models) of MoE models (20B-200B parameters) and several checkpoints OLMoE-7b .

The authors extend the MUI metric (originally for dense models) to MoE architectures by:

1. **Neuron-level Analysis**: Computing contribution scores for each neuron considering routing weights:
    
    $$
    f_{neuron}(i, j, l, \hat{y} | x) = G^l_i(x^l) \cdot (x^lW^l_{g,i}) \cdot W^l_{d,i}[j] \cdot W_{head}[:, \hat{y}]
    $$
    
    where $G^l_i$ are routing weights and $W$ matrices are projections in SwiGLU
    
2. **Model-level MUI**: Proportion of activated neurons relative to total:
    
    $$
    \text{MUI}(T) = \frac{|\bigcup N_{activated}(s_i)|}{N \times L \times (|E_s| + |E_r|)}
    $$
    
3. **Expert-level Analysis**: Identifying key experts with frequency threshold $\eta_{\text{expert}} = 0.6$  and computing expert-specific MUI

### Strengths
- Methodological contribution by extending the Model Utilization Index (MUI) from dense models to MoE architectures. This isn't a trivial extension - thoughtfully adapted the metric to account for MoE-specific characteristics like routing mechanisms and the distinction between shared and routed experts. The formulation in Equations 4-7 provides both model-level and expert-level granularity, enabling multi-scale analysis that wasn't previously available.
- The work offers practical value:
    - MUI as a training monitor can detect when specific capabilities need more data (e.g., OLMoE's persistent accumulating phase for coding)
    - The metric can potentially detect benchmark overfitting/leakage by comparing MUI trends with performance gains
    - Insights about shared experts inform architectural decisions for future MoE designs
- The paper tries addresses a critical issue that, why do some models (particularly GPT series) perform really well in real-world applications while not always dominating benchmarks? The discovery that GPT-OSS less neuron activity/utilization provides a mechanistic explanation for the same. This isn't just correlation - the causal masking experiments prove these neurons are actually doing the work. This suggests that **generalization quality can be measured by representational efficiency**, fundamentally changing how we should evaluate models beyond benchmark accuracy.
****

### Weaknesses
- It claims experts "collaborate" but doesn't analyze what individual experts learn:
    - Conducting semantic analysis of expert specialization could reveal whether experts specialize or remain largely redundant
    - Probing expert representations would help determine if experts learn complementary features that explain their collaborative behavior, providing mechanistic insight into why MoE architectures succeed
- The core claim that lower MUI indicates better generalization lacks justification. In Section 3.2, the authors state: "If we assume that these newer models indeed possess higher true capability and stronger generalization... then MUI may serve as an indicator." This is circular reasoning - they assume what they're trying to prove. The paper needs either:
    - Theoretical analysis connecting neuron sparsity to generalization bounds
    - Controlled experiments where generalization is measured independently (e.g., through distribution shift experiments, not just benchmark performance)
    - References to established theory linking sparsity and generalization in neural networks
- The neuron importance formula (Equation 2) multiplies by routing weights G_i(x), which are **already sparse by design** in MoE models. The paper then claims this shows "better utilization" but this seems to be just measuring the routing sparsity, not actual neuron utility

### Questions
- I would recommend that please look into Figure 1 make it more clear and easy to understand the illustration overall is not very clear
- Noticed inconsistent model naming like DeepSeek-V2-Light / DeepSeek-V2-Lite. Please follow the consistently and Lite is the right one to avoid any sort of confusion among readers. [very minor, but just spotting it out]

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
This paper adapts the Model Utilization Index (MUI) to analyze internal behaviors of MoE models, revealing how neuron and expert activation patterns evolve with training and correlate with generalization. The authors perform large-scale experiments on 13 MoE models and OLMoE checkpoints, finding that higher-performing models use fewer neurons and exhibit more expert collaboration, especially in architectures with shared experts.

### Strengths
The paper goes beyond the conventional benchmark-centric evaluation and tackles the open question of how MoE models utilize their enormous capacity. This focus on interpretability and internal mechanisms is timely and original.

The study is thorough, examining 13 different MoE models spanning several popular families (GPT-OSS, Qwen3, DeepSeek, etc.)

### Weaknesses
The proposed MUI, while useful, is not a fundamentally new class of interpretability method; it is a well-engineered adaptation. There is no novel learning algorithm, architectural change, or optimization scheme proposed—only a diagnostic and analysis framework. The conclusions are similar to Cao et al. (2025) that observed a similar relationship between lower MUI and stronger generalization

The authors describe reduced neuron usage as “a reflection of stronger generalization”. However, the paper does not directly measure generalization in a rigorous way (e.g., out-of-distribution performance, or an experiment like “take a model, improve its generalization via some regularization, see MUI drop” or similar).Thus, while the interpretation is plausible and supported by prior work’s Utility Law, it isn’t conclusively demonstrated.

### Questions
One question is how MoE-MUI compares to MUI in dense models (from Cao et al. 2025) or even whether MoE models utilize neurons more efficiently than dense counterparts of similar performance. For instance, when saying GPT-OSS (MoE) has strikingly low MUI, it would be insightful to know: low compared to what? Perhaps a dense GPT-style model with similar capability uses a larger fraction of its neurons for the same tasks.

How exactly did you measure the degree of expert cooperation?

### Soundness
3

### Presentation
3

### Contribution
2
