# Prompt Engineering at Scale: Provably Effective Multi-Agent Cascades for Attribute Generation in E-Commerce

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Developing specialized Large Language Model (LLM) prompts for domain-specific tasks at scale remains a significant hurdle, particularly for e-commerce applications managing tens of thousands of distinct product attributes. We introduce CascadeAgent, a novel multi-agent framework that automates prompt adaptation and specialization through semantic gradient-based refinement. CascadeAgent employs a hierarchical architecture where a central Prompting Agent orchestrates four specialized counterparts—Writing, Generation, Evaluation, and Flaw Detection—that collaboratively analyze domain metadata, construct attribute-specific prompts, and enhance performance through iterative feedback. Our approach combines Multi-pass Prompt Generation (MPG) for modularity with textual gradient optimization that refines instructions based on detected error patterns. We provide formal theoretical analysis demonstrating provable convergence towards reduced loss under defined conditions. In a large-scale e-commerce case study on product attribute enrichment, CascadeAgent generated and optimized over 27,000 distinct prompts, achieving improvements of +21% to +33% in precision and +12% to +14% in coverage across multiple LLMs. 
These results highlight CascadeAgent's capacity for robust, automated prompt engineering at industrial scale, while making more affordable models viable for deployment. The framework's modular design, iterative improvement mechanism, and theoretical guarantees make it a strong candidate for applications requiring principled refinement of vast numbers of task-specific prompts.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes CascadeAgent, a multi-agent framework to automatically generate and optimize LLM prompts for tens of thousands of heterogeneous product attributes on e-commerce platforms. Specifically, the framework includes a scalable "Multi-pass Prompt Generation" (MPG) strategy and a 5-agent collaborative team (CascadeAgent) that iteratively refines the prompts using semantic gradient-based optimization. The paper provide a theoretical proof of the framework's convergence and demonstrate CascadeAgent's effectiveness in large-scale experiments, particularly its ability to significantly boost the performance of low-cost models.

### Strengths
1. The paper proposes a practical and highly scalable prompt optimization solution, effectively optimizing over 27,000 heterogeneous attributes.

2. The proposed CascadeAgent method demonstrates significant effectiveness by boosting the more cost-effective Mistral NeMo from 57.14% to 90.21%, closing to the performance of SOTA models Claude 3.5 Sonnet.

3. The paper provides a theoretical analysis that well support the design of the agent framework (although the theoretical analysis is a generally broad analysis that prove the effectiveness of most similar framework).

### Weaknesses
1. The experimental evaluation is weak and lacks effective baseline comparisons. The paper only compares CascadeAgent against a "single universal prompt" and "MPG", while no existing, SOTA or previous automatic prompt optimization methods (e.g., APE, APO) are included as baselines, making the experiments less valid.

2. The proposed agent collaboration framework is not novel; similar ideas have been proposed in many prior works [1, 2]. Likewise, the "Optimization with Textual Gradients" section only adopts methods from previous work [3].

3. The proposed multi-agent framework lacks validation of its effectiveness. For instance, the paper claims the Flaw Agent can “move beyond simple error counts, identifying systematic problems, common error patterns” and produce “actionable critiques.” However, there is no ablation study or any case study to support the claim. Similarly, the effectiveness and claimed advantages of the framework's other components are also not proven.


[1] AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors.

[2] (Perhaps) Beyond Human Translation: Harnessing Multi-Agent Collaboration for Translating Ultra-Long Literary Texts

[3] Automatic Prompt Optimization with "Gradient Descent" and Beam Search

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This is more like a technical report than a research paper.

### Strengths
The prompt rewriting task is very interesting.

### Weaknesses
I read the paper carefully, I dont think the paper meets the high standard for publishing in ICLR. The theory part is too simple, not too much contribution, the result section is too short, and overall paper lacks clear contribution. Not ICLR level.

### Questions
Would suggest authors to address issues in weakness.

### Soundness
2

### Presentation
2

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
This paper addresses the formidable challenge of automating prompt engineering for domain-specific tasks at an industrial scale, with a focus on e-commerce product attribute generation involving tens of thousands of distinct attributes. The authors propose CascadeAgent, a novel multi-agent framework designed to automate the creation and iterative refinement of attribute-specific prompts. The system is built upon two key ideas:  Multi-pass Prompt Generation (MPG), which modularizes the problem by decomposing catalog enrichment into attribute-specific sub-tasks, and a collaborative loop of five specialized agents (Prompting, Writing, Generation, Evaluation, Flaw Detection) that refine prompts using a semantic gradient-based optimization strategy. A significant contribution of this work is the provision of a formal theoretical analysis, modeling the refinement process as a Markov Decision Process and proving convergence towards reduced catalog loss under defined assumptions.

### Strengths
1.This paper exhibits strengths particularly in its theoretical grounding. The core originality lies in the creative integration of a multi-agent architecture with a formal convergence guarantee to address the underexplored challenge of large-scale prompt optimization. This represents a significant conceptual leap beyond prior work that typically handles dozens or hundreds of prompts. The theoretical contribution is substantial; the authors provide a rigorous Markov Decision Process model and a formal proof of convergence, which elevates the work from a purely engineering solution to a principled method with provable properties.

2.The significance of the work is underscored by its successful application in a real-world industrial setting, managing an unprecedented scale of over 27,000 distinct prompts. The empirical results are compelling, demonstrating that the framework not only improves performance metrics significantly but also makes cost-effective models viable for deployment, a finding with direct practical implications.

### Weaknesses
1.The description of the core multi-agent workflow remains high-level and ambiguous. Figure 1 is too simplistic and does not elucidate the exact information flow, data structures, or prompting protocols that govern the interactions between the five specialized agents. For instance, the specific input and output formats for the Writing Agent and how it synthesizes the Flaw Agent's feedback into a revised prompt are left entirely to the reader's imagination.

2.Despite the impressive scale of the experiments, the evidence presented is not fully convincing. The performance tables report aggregate scores but lack illustrative case studies. There is no qualitative analysis showing how a problematic prompt for a specific attribute was iteratively refined by the cascade into a high-performing one, which would have been crucial for demonstrating the system's operational effectiveness and value beyond mere metrics.

3.The authors explicitly state that due to company policy, they are unable to release the code and data, which limits readers' ability to conveniently verify and reproduce the results.

### Questions
1.Could you provide a more detailed flowchart specifying the exact inputs, outputs, and data formats for each of the five agents in a single iteration?

2.What is a concrete example of an initial prompt, the corresponding flaw summary from the Flaw Agent, and the resulting optimized prompt for a specific product attribute?

3.In the theoretical analysis, how was the key assumption of the "marginal-churn condition" (p_fix > p_break) validated or estimated in your practical setting?

4.Given the inability to release code, could you provide a pseudo-code snippet illustrating the core orchestration logic of the multi-agent loop?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The **CascadeAgent** framework introduces a scalable, multi-agent prompt refinement approach (MPG) for large e-commerce attribute generation, combining modular decomposition, iterative semantic optimization, formal convergence analysis, and cross-model efficiency; key strengths (scale, precision/coverage gains, economic viability) coexist with resource intensity and difficulty on complex attributes.

• **Core Contribution**: CascadeAgent—industrial-scale multi-agent framework for prompt adaptation and specialization.  

• **Scalability Mechanism**: Multi-pass Prompt Generation (MPG) decomposes >27,000 attribute-specific sub-tasks for independent optimization.  

• **Architecture**: Five agents (Prompting, Writing, Generation, Evaluation, Flaw Detection) iteratively refine via semantic gradient-based optimization.  

• **Theoretical Rigor**: Formal analysis with convergence toward reduced catalog loss $L(\pi)$ (Markov Decision Process framing).  

• **Empirical Performance**: Up to +33% precision and +14% coverage improvements across multiple LLMs.  

• **Economic Impact**: Elevated Mistral NeMo—precision gap narrowed to ~3% vs premium Claude 3.5 Sonnet.  

• **Complex Attribute Limits**: Challenges remain for image-dependent and numeric attributes despite general scalability.  

• **Resource Requirements**: High compute (48 H100 GPUs) and substantial labeled ground truth needed for optimization.

### Strengths
The paper presents a novel and significant contribution to prompt engineering by introducing **CascadeAgent**. The originality stems from the principled integration of multi-agent systems with a scalable decomposition strategy called **Multi-pass Prompt Generation (MPG)**, enabling optimization of large number of attribute-specific prompts. 

• **Novel Framework** and Scale: CascadeAgent’s architecture manages and optimizes over 27,000 attribute-specific prompts, surpassing prior approaches limited to dozens or hundreds. The framework is designed for industrial-scale product attribute generation in e-commerce.

• **Theoretical Guarantees**: Formal analysis models prompt refinement as a Markov Decision Process (MDP) and demonstrates convergence towards reduced catalog loss. Empirical results show +33% precision and +14% coverage gains while enabling a cost-effective model (Mistral NeMo) to approach premium performance.

• **Modularity** and Specialization: MPG enables modular decomposition and independent attribute optimization; five specialized agents (Prompting, Writing, Generation, Evaluation, Flaw Detection) drive nuanced improvements..

• Effective **Optimization**: Semantic gradient-based (textual gradients) iterative refinement achieved an average +15% improvement in hold-out test accuracy for Product-Attributes needing tuning.

• **Economic Viability**: The framework elevated Mistral NeMo, reducing the precision performance gap with Claude 3.5 Sonnet to only 3%.

• Seamless **Extension**: Multi-agent architecture allows adding new attributes without complete retraining, addressing prior limitations.

• Robust **Validation**: Industrial-scale evaluation using a 10,000-product catalog set and a high-fidelity set of 1,879 Product-Attribute pairs.

• Improved **Interpretability**: Iterative agent cascade steps provide insight into how and why prompts are modified.

### Weaknesses
While CascadeAgent demonstrates **significant advancement**, several limitations related to its scope, computational requirements, and remaining performance challenges need constructive consideration. 

• **High Computational Requirements**: The multi-agent system demands substantial computing infrastructure, utilizing 48 NVIDIA H100 GPUs across 6 AWS EC2 P5 instances for parallel processing, making it potentially inaccessible in resource-constrained environments. 

• **Limited Domain Validation**: The empirical validation is confined primarily on e-commerce attribute enrichment, and the transferability to other domains remains untested.  

• **Dependence on Labeled Data**: The iterative optimization process relies heavily on ground truth data availability, requiring a minimum of 150 verified human labels per PA for the high-fidelity optimization set. Future work should explore semi-supervised or agent-driven data synthesis to mitigate this reliance which otherwise could pose a challenge in data-scarce domains.

• Challenges with **Complex Attribute** Types: The optimization process and system performance plateaued for a significant percentage of PAs, particularly those that were image-dependent attributes (40% of remaining cases) and numeric attributes, indicating a need for multimodal models or enhanced mathematical reasoning capabilities to mitigate these limitations in the current LLM instruction refinement loop.

• Trade-off Issues (**Precision vs. Coverage**): A large portion of PAs (35.0%) exhibited increased coverage but reduced precision post-refinement, often due to hallucinations. The refinement strategy may need adjustments to better control precision loss in exchange for coverage gains.  

• **Hinderance to Reproducibility**: Although some details/specifications are provided, the implementation code and dataset cannot be released due to company policy, which presents an obstacle to complete independent verification and replication. Moreover, there aren't any experiments conducted on similar public datasets.

### Questions
The **CascadeAgent framework** raises interrelated concerns across dataset transparency, **generalizability**, model selection scope, reproducibility of sampling and optimization settings, cost vs. claimed affordability, observed **optimization plateau** for complex attributes, sensitivity of convergence to loss weighting, and empirical validation of theoretical sample size assumptions.

1. **Dataset transparency**: Beyond counts (10,000 products; 1,879 high-fidelity PAs), can synthetic examples (titles/descriptions + initial and refined prompts) be shared to illustrate **prompt structure** and attribute complexity?  

2. **Domain generalizability**: Why no evaluation on public e-commerce attribute datasets (e.g., OpenTag, AdaTag, MAVE) to test **transferability** beyond the internal case study?  

3. **Model selection scope**: Why limit evaluation to Mistral NeMo (generation) and Claude 3.5 Sonnet (flaw detection); why exclude other open-source or proprietary models that could strengthen affordability claims?  

4. **Sampling reproducibility**: Hyperparameters (Top-K, minibatch) are given, but missing temperature / nucleus sampling for Generation & Writing Agents—can these stochastic **sampling parameters** be disclosed?  

5. **Cost-benefit justification**: With 48 H100 GPUs (6 P5 instances), can a comparative **cost analysis** show the cascade’s total inference + premium calls outperforming alternatives (supervised fine-tuning or single premium zero-shot)?
  
6. **Complex attribute plateau**: For unmet stopping criteria (46.0% PAs; 40% image-dependent), what concrete plans exist to integrate multimodal or enhanced **mathematical reasoning** into the semantic gradient loop?  

7. **Convergence sensitivity**: How do precision/coverage outcomes and convergence rates vary under adjustments to loss weights ($w_{val}, w_{omis}, w_{commis}$) influencing $r = \kappa_{\max}/\kappa_{\min}$ and contraction factor $\phi$; were weight sweeps tested to reduce the 35.0% precision drop cases?  

8. **Sample size validation**: For Proposition 1, how was evaluation sample size $n$ tied to the dynamic loss gap $\Delta_t$ computed or verified across 1,879 PAs—was $n$ adaptively scaled or empirically stress-tested to support monotone improvement guarantees?

### Soundness
3

### Presentation
4

### Contribution
3
