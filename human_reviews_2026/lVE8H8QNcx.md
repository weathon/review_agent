# Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 4

## Abstract
The utility of Role-Playing Language Agents in sociological research is growing alongside the adoption of Large Language Models. For realism in social simulation, these agents must adhere to their personas defined by character profiles, yet existing strategies—static prompt engineering or costly fine-tuning—fail to adapt personas to dynamic scenarios. Psychological theories, such as the Cognitive-Affective Personality Systems, provide a crucial explanation for this failure: a persona's influence on behavior is not static but varies with the scenarios. This context-dependence highlights the critical need for adaptive persona management. To address this gap, we propose a novel, theory-driven method that dynamically estimates context-dependent persona importance and integrates it into weighted reward-guided decoding, enabling inference-time persona following. Specifically, we introduce Persona Dynamic Decoding (PDD) framework that consists of two key components:
(1) Persona Importance Estimation (PIE) module, which dynamically quantifies the contextual importance of persona attributes without requiring ground-truth supervision; and (2) Persona-Guided Inference-Time Alignment (PIA) paradigm, which leverages these importance scores to construct weighted multi-objective rewards and modulate generation probabilities during inference. Extensive experiments show the effectiveness of our method in utterance consistency and behavioral fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes the Persona Dynamic Decoding (PDD) framework to enhance persona adherence in role-playing language agents through inference-time optimization without model fine-tuning. The framework consists of two modules: (1) PIE (Persona Importance Estimation), which dynamically quantifies the importance of persona attributes in different scenarios using conditional mutual information approximation, and (2) PIA (Persona-Guided Inference-Time Alignment), which modulates token generation probabilities via a weighted multi-objective reward function with KL divergence constraints. Experiments on three benchmarks (CharacterEval, BeyondDialogue, PersonalityBench) demonstrate improvements over baseline methods including static prompting, ICL, and OPAD.

### Strengths
- Proposes a novel framework for dynamic persona importance estimation. The paper introduces PIE and PIA modules to achieve context-adaptive persona following, representing the first approach to dynamically quantify scenario-dependent persona attribute importance using conditional mutual information, though the necessity of this complex inference-time intervention over simpler dynamic prompting alternatives requires clearer empirical justification (see Weakness 1).

- Theoretically grounded with mathematical formalization. The approach builds on conditional mutual information and psychological theories (CAPS), supported by mathematical propositions (3.1-3.3) that connect model generation probabilities to importance estimation, though the strong assumptions and approximations used require more systematic empirical validation across different model capabilities (see Weakness 4).

- Training-free inference-time optimization with modular design. The plug-and-play framework enables persona adaptation without model fine-tuning or labeled training data, facilitating integration into existing LLMs, though the significant computational overhead (≥3× inference cost) raises practical deployment concerns given the uneven performance improvements (see Weakness 5).

### Weaknesses
- Motivation would benefit from comparison with dynamic prompting baselines. The paper identifies limitations of static Prompt Engineering and ICL (lack of dynamic adaptability), but does not empirically compare against adaptive prompting approaches, such as periodically inserting context-analysis prompts to guide LLM self-adjustment (e.g., "Given the current conversation context, which personality traits should be emphasized?"). While the CMI-based approach is theoretically motivated, demonstrating empirical advantages over these simpler alternatives would strengthen the justification for token-level probability modulation and clarify when the added complexity is warranted versus when lighter-weight prompt-based methods might suffice.

- Uneven performance across evaluation dimensions requires deeper analysis. The experimental results show notable variation in improvement magnitude across different evaluation dimensions. Some dimensions exhibit particularly strong gains while others show more modest improvements, which suggests the method may be more effective for certain types of persona attributes than others. Understanding the underlying reasons for this performance variance would help identify appropriate use cases, guide future refinements, and provide users with clearer guidance on when to apply the method. The paper would benefit from analyzing which persona characteristics or evaluation criteria are most amenable to the dynamic importance estimation approach and why certain dimensions respond less strongly to the intervention.

- Approximation strategy's dependence on model capability warrants investigation. The PIE module relies on model-generated outputs to approximate ground truth importance, which may create scenarios where method performance is bounded by the base model's inherent capability. While the theoretical framework provides mathematical support for this approximation under certain assumptions, the paper would benefit from empirical analysis examining: (a) how approximation quality varies across models of different capabilities and scales, (b) whether the theoretical assumptions underlying the approximation hold consistently across diverse scenarios and model architectures, and (c) potential strategies to mitigate circular dependency issues where the method's effectiveness depends on the very model outputs it aims to improve.

- Computational overhead requires cost-benefit analysis for practical deployment. The method introduces substantial inference overhead compared to simple prompting approaches due to computing reward scores before token generation. The experimental results indicate that inference speed decreases as the number of persona attributes increases. Given the performance characteristics noted in Weakness 2, clearer guidance on when this computational investment is justified would enhance practical value. The paper could explore efficiency improvements such as caching importance scores across conversation turns, amortizing computation through strategic application, or hybrid approaches that combine PDD for critical conversational moments with lighter methods for routine exchanges.

- Interpretability could be enhanced through visualization and mechanistic analysis. The paper would benefit from providing concrete examples illustrating how the method modulates generation in practice. Visualizations showing token probability distributions before and after PDD intervention would help readers understand the method's actual effects on generation. Additionally, demonstrating how persona importance scores evolve across different conversational contexts and providing specific examples of when and why PDD succeeds or encounters limitations would improve transparency, facilitate debugging, and help users understand the method's operational characteristics and appropriate application scenarios.

### Questions
- Can you provide empirical comparison or theoretical analysis distinguishing PDD from dynamic prompting baselines? For example, approaches that periodically insert adaptive prompts during generation (e.g., "Analyze the current context and adjust your response style accordingly") to enable LLM self-regulation of persona expression. If such approaches were empirically tested, how did they compare to PDD in terms of persona adherence quality, response naturalness, and computational efficiency? If not tested, can you provide theoretical or conceptual analysis of why token-level probability modulation through CMI-based importance estimation would offer advantages that prompt-based dynamic adaptation fundamentally cannot achieve? Under what specific conditions or scenarios is PDD's added complexity warranted versus when simpler prompt-based methods might suffice? This question is particularly important for assessing the core contribution and necessity of the proposed framework.

- What explains the performance variance across different evaluation dimensions, and what does this reveal about the method's characteristics? The experimental results show notable variation in improvement magnitude across evaluation dimensions. Can you provide analysis of which types of persona attributes or evaluation criteria respond most strongly to the dynamic importance estimation approach, and identify potential limiting factors for dimensions showing modest improvements? Concrete examples or case studies illustrating scenarios where PDD significantly improves persona adherence versus scenarios where improvements are marginal would help users understand appropriate use cases. Does this performance pattern suggest opportunities for method refinement, such as attribute-specific importance estimation strategies?

- How does base model capability affect the PIE approximation quality and overall method effectiveness? Since PIE relies on model-generated outputs to approximate ground truth importance, can you provide analysis or empirical evidence of how the method performs across models of substantially different scales or quality levels? Are there indicators, diagnostic metrics, or failure modes that could help users determine whether their specific model is sufficiently capable for effective PDD application? Do the theoretical assumptions underlying the CMI approximation hold consistently across different model architectures and capability levels, and what strategies might mitigate potential circular dependency issues where method performance is bounded by the base model's inherent capability?

- What are the practical deployment recommendations and potential efficiency improvements given the computational overhead? Can you provide concrete guidance on when the computational investment is justified, considering factors such as application priorities, latency requirements, and cost constraints? Have you explored or can you propose efficiency improvements such as caching importance scores across turns, computing estimates less frequently, using lightweight approximations, selective application strategies for critical conversational moments, or hybrid approaches combining PDD with lighter methods? How does the overhead scale with conversation length, and are there architectural optimizations that could reduce the multiplicative factor?

- Can you provide visualizations and concrete examples demonstrating how PDD modulates generation in practice? Specifically, showing token probability distributions before and after PDD intervention would illustrate how the method shifts generation tendencies. Visualizing how importance scores for different persona attributes evolve across various conversational contexts (e.g., casual vs. professional, analytical vs. emotional) and providing side-by-side response comparisons showing success and failure cases would greatly enhance interpretability. Do the actual generation changes align with theoretical expectations—for instance, when PIE assigns high importance to a particular attribute, does PIA's probability modulation observably increase related token probabilities?

- How does the PDD framework relate to and potentially complement recent work on context-aware personality modeling in role-playing agents? Recent research has explored various aspects of context-dependent personality, including evaluation frameworks and adaptation mechanisms. How does PDD's generation-side optimization approach complement or extend existing work, and could such evaluation frameworks validate PDD's effectiveness in achieving appropriate context-dependent personality shifts? What are the unique advantages of the CMI-based importance estimation compared to other potential approaches for identifying scenario-dependent attribute relevance, such as attention-based methods or learned importance predictors?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper tackles Role-Playing Language Agents’ (RPLAs) key limitations—poor dynamic scenario adaptability and heavy data/computation reliance for sociological simulations—by proposing the Persona Dynamic Decoding (PDD) framework, which includes PIE (context-dependent persona importance estimation without ground-truth) and PIA (inference-time generation adjustment via weighted rewards). It conducts experiments on three benchmarks (CharacterEval, BeyondDialogue, PERSONALITYBENCH) with LLaMA-3-8B-Instruct and Qwen2.5-7B-Instruct, comparing PDD to baselines (e.g., Simple Prompting, ICL) and closed-source models (e.g., GPT-4o). Through LLM-as-Judge, dataset-specific metrics (e.g., CharacterRM), and human evaluations, PDD outperforms baselines in persona alignment; it also needs no fine-tuning, uses lightweight resources, and helps small open-source models match closed-source systems’ competitiveness.

### Strengths
1. The paper integrates interdisciplinary theories (Cognitive-Affective Personality Systems from psychology and conditional mutual information from information theory) to underpin the PDD framework, rationally justifying dynamic persona adaptation and avoiding blind technical design.
2. PDD enables fine-tuning-free dynamic persona following at inference time: its PIE module adjusts persona attribute priority based on scenarios, while PIA modulates generation probabilities, solving existing methods’ poor adaptability and high resource reliance.
3. It adopts a comprehensive experimental design, including diverse datasets (CharacterEval, BeyondDialogue, PERSONALITYBENCH), extensive baselines, and multi-dimensional evaluations, ensuring reliable and generalizable results.

### Weaknesses
1. The robustness analysis of PIE’s core assumption (using model-generated G to approximate ground-truth GT) is insufficient; the paper fails to discuss the risk of misjudging importance scores when G deviates from persona settings, nor does it test extreme scenarios like conflicting persona attributes.
2. Fig.4 shows that increasing persona attributes leads to a win rate decline. Does this indicate that PDD cannot model scenarios with complex and multiple attributes?

### Questions
Can PDD handle scenarios where specific attributes are difficult to summarize? For example, imitating language style.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Persona Dynamic Decoding (PDD), a framework that dynamically adjusts persona adherence in role-playing language agents through contextual importance estimation and reward-guided inference. The approach combines a Persona Importance Estimation (PIE) module and a Persona-Guided Inference-Time Alignment (PIA) mechanism to achieve adaptive persona following without model fine-tuning. Experimental results across multiple benchmarks show that PDD improves persona consistency and behavioral fidelity over existing static and fine-tuned methods.

### Strengths
- Introduces a theory-grounded, inference-time persona adaptation mechanism inspired by psychological models (CAPS), offering a fresh perspective on dynamic persona control.
- The paper writes with great clarity, paper structure and visuals explainer
- Strong empirical validation with both quantitative metrics and human/LLM-based evaluations demonstrating robustness and generalizability of the method.

### Weaknesses
- Dependence on subjective LLM-based judges for evaluation may introduce bias and reduce reliability.
- The paper’s practical implications for broader applications beyond controlled experiments (e.g., open-domain dialogue) are not fully explored, which limits significance.

### Questions
N/A

### Soundness
4

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
3

### Summary
The paper proposes Persona Dynamic Decoding (PDD), an inference-time, training-free framework to improve persona-following in Role-Playing Language Agents (RPLAs). PDD consists of two parts: (1) Persona Importance Estimation (PIE), which dynamically weights persona attributes based on the context without supervision, and (2) Persona-Guided Inference-Time Alignment (PIA), which uses these weights in a normalized, multi-objective reward function to guide the decoding process. The authors evaluate PDD on three role-playing benchmarks (CharacterEval, BEYOND DIALOGUE, and PERSONALITYBENCH) using LLaMA-3-8B and Qwen2.5-7B base models. The results, based on LLM-as-a-judge, automatic metrics, and human evaluations, demonstrate that PDD significantly outperforms baseline methods (prompting, ICL, and several decoding/training baselines) in both utterance consistency and behavioral fidelity.

### Strengths
1. This paper addresses a practical problem — controlling persona adherence in inference-time, without fine-tuning.
2. Experimental setup includes multiple datasets (CharacterEval, Beyond Dialogue, PERSONALITYBENCH).
3. The method is described clearly, and the PIE and PIA stages are straightforward.

### Weaknesses
1. For the Persona Importance Estimation (PIE) module in assumption (Proposition 3.2), the authors "propose using G as an approximation of GT," where G is the model's own generated response. What if the model's generation $G$ is low quality or misaligned with the persona? Then, it seems that PIE will calculate its "importance scores" based on this output, and the model's own errors are used to guide its generation, potentially reinforcing its existing biases rather than correcting them. 
2. The PIA framework seems computationally high because the model must compute the step-wise reward $r_i$ for each attribute being aligned. The authors' own ablation study (Figure 4) confirms that inference speed degrades drastically as more attributes are added. The paper lacks a crucial analysis of this runtime/latency trade-off, particularly a direct comparison against the baselines. Furthermore, the optimal selection of top-k attributes appears unstable and model-dependent (top-3 for LLaMA, top-2 for Qwen), which suggests a lack of generalizability and is not adequately analyzed.
3. The study's generalizability is limited. Most datasets (e.g., CharacterEval, Beyond Dialogue) are in Chinese, with limited coverage of English or multilingual settings. In addition, only two models (Qwen2.5-7B-Instruct and LLaMA-3-8B-Instruct) are evaluated. Maybe other model scales or architectures can also be evaluated. 
4. The presentation of results in Tables appears to overstate the method's superiority. While the authors bold the average results, the proposed method underperforms the baselines in two of the five sub-tasks. Many of the reported improvements seem marginal, which raises the question of their statistical significance. The paper would be much stronger if it included statistical significance tests for all results.
5. The ablation studies only have normalization variants and top-k attribute selection. More ablation analyses can be added, including: 1) a modular ablation to isolate the individual contributions of the PIE and PIA components; 2) an analysis of the proxy response quality's impact on PIE (as noted in point 1); 3) an evaluation of the method on different model scales (e.g., larger LLMs); and 4) an ablation of other design choices, such as the formulation of the step-wise reward function.
6. I would suggest adding a discussion of potential ethical or bias implications of manipulating persona adherence (e.g., reinforcing stereotypes).

### Questions
See in weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3
