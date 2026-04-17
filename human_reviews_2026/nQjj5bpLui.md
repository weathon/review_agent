# UUE: Untargeted Language Model Unlearning via Null-Space-Guided Editing with Lightweight Adapters

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Large language models (LLMs) are raising increasing ethical and security concerns as they may reproduce private, sensitive, or hazardous content. This motivates the development of effective LLM unlearning techniques that can remove undesired knowledge from the model while preserving general utility. Existing unlearning methods mainly rely on fine-tuning, which is not only computationally intensive but also prone to utility degradation due to the entangled nature of knowledge in LLMs. In this paper, we propose a lightweight and controllable LLM unlearning framework, **UUE**, which reformulates unlearning as null-space-guided model editing. To ensure stability, we introduce a novel editing objective that achieves unlearning without explicit target outputs. We further design pluggable unlearning adapters and derive closed-form analytical updates with null-space guidance, ensuring minimal interference with retained knowledge. To further improve efficiency, we extend UUE with LoRA, yielding **UUE-L**. Extensive experiments on TOFU and WMDP benchmarks across multiple LLMs demonstrate that UUE and UUE-L achieve superior unlearning efficacy, significantly outperforming existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces UUE—a novel, efficient, and controllable framework for unlearning undesired knowledge in large language models (LLMs). Unlike prior fine-tuning-based approaches that often degrade model utility, UUE reformulates unlearning as null-space-guided model editing, where parameter updates are directed into the null space of retained knowledge, ensuring minimal interference with useful information. UUE performs untargeted unlearning, meaning it does not force explicit refusal responses but suppresses the forgotten content’s influence. The method introduces lightweight unlearning adapters—plug-in modules inserted into transformer layers—that allow efficient, localized updates. A closed-form analytical update rule ensures stability and scalability, while UUE-L, an extension combining UUE with LoRA, further improves computational efficiency using low-rank parameterization.

Experiments on TOFU (fictional entity unlearning) and WMDP (malicious content prevention) benchmarks across multiple models (Phi-1.5B, LLaMA2-7B, Zephyr-7B) show that UUE and UUE-L outperform existing methods in both forgetting effectiveness and retained model utility, balancing safety and capability. Ablation and sensitivity analyses confirm that null-space projection and proper regularization critically contribute to performance. UUE also achieves shorter runtime and robustness to retain-set quality variations, demonstrating strong practical potential.

### Strengths
The authors study how to stabilize the unlearning procedure, which is not well covered in previous works, but I think it is critical. 

The paper requires little modifications of parameters, and the model retention looks good. 

The authors conduct a lot of experiments in TOFU and WMDP, the results look good to me. 

Closed-form solutions enable stable, scalable optimization. 

Provides a lightweight, controllable, and efficient unlearning solution requiring no full fine-tuning.

### Weaknesses
Some of the design choices seem reasonable to me but lack explanations and related references. 1. The authors choose untargeted unlearning rather than targeted unlearning, while having not given the explanation. Maybe the authors can refer to [1] which state that targeted unlearning cannot remove the knowledge targeted to be unlearned. 2. The authors add an adapter in the FFNs, which is reasonable since the knowledge is typically believed to cached in them. I think you can find some related papers in knowledge localization to support this kind of claim. But sorry, I cannot remember any of them. 3. References about targeted and untargeted unlearning are missing in Sec 2. 4. Using both neighboring data and general domain knowledge to construct the retain set is not a common practice, any references or explanations? 5. The authors conduct unlearning in the embedding space, [2] should be mentioned in Sec 3.3 and discuss why optimizing in the embedding space is enough. 

[1] Towards Effective Evaluations and Comparison for LLM Unlearning Methods

[2] The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning

Some related papers discuss the similar things in doing unlearning under the conditions that the retain performance is preserved [3-4], which should be discussed and compared in this paper. Also, CHs 1-3 seem not to be proposed firstly in this paper, you’d better refer some related papers. 

[3] Adaptive Localization of Knowledge Negation for Continual LLM Unlearning

[4] GRU: Mitigating the Trade-off between Unlearning and Retention for Large Language Models

How the transformer blocks are selected for unlearning? Any ablation studies?

I think it is reasonable to study untargeted unlearning. However, one thing I am worried about is that if the model responses after unlearning qualitatively well? If so, why? If not, how to improve it?

### Questions
Please see the section of weaknesses.

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
This work introduces UUE, a lightweight framework for making large language models forget undesired knowledge while maintaining general utility. It reformulates unlearning as null-space-guided model editing, inserting small adapters into Transformer layers and constraining updates to the null space of retain-set representations, ensuring minimal interference with preserved knowledge. A new untargeted editing objective inverts forget-set representations toward a safe region instead of fixed refusals, achieving stable forgetting without explicit targets. With a closed-form solution and a LoRA-based low-rank variant (UUE-L), the method enables efficient, scalable unlearning. Experiments on TOFU and WMDP benchmarks show that UUE(-L) surpasses existing methods in both forgetting effectiveness and preservation of general capabilities.

### Strengths
1.	This paper introduces a new untargeted unlearning objective, which departs from prior refusal-style approaches that force models to respond with fixed denial phrases. This formulation allows for more flexible and context-aware forgetting, producing natural yet knowledge-removed outputs.
2.	The authors conduct extensive experiments across multiple benchmarks (TOFU, WMDP) and model scales (Phi-1.5B, LLaMA-2-7B, Zephyr-7B), demonstrating consistent and strong performance in both forgetting efficacy and utility preservation.
3.	The framework is scalable and versatile, as it not only provides a full fine-tuning formulation but also introduces a LoRA-based lightweight variant (UUE-L), making it practical for deployment on larger models and limited-resource settings.

### Weaknesses
1.	The experiments are conducted only on small- to mid-sized and relatively outdated models (e.g., Phi-1.5B, LLaMA-2-7B), rather than newer generations such as LLaMA-3 or Qwen models. This limits the credibility and forward relevance of the results, as the scalability and generalization of UUE on state-of-the-art LLMs remain unverified.
2.	The paper contains several writing and formatting issues, such as missing punctuation (e.g., missing periods after “i.e.” in line 36 and at the end of line 114), which reduce overall clarity and polish.
3.	The explanation of null-space construction is overly lengthy and somewhat repetitive; reorganizing this section could improve readability and highlight the conceptual flow of the method more effectively.
4.	Although the untargeted unlearning objective is conceptually appealing, its optimization formulation still resembles gradient ascent. Despite the modification in Eq. 2, the method essentially moves representations opposite to $A_F$ which lacks clear semantic interpretability. In practice, it may behave similarly to gradient ascent but with a retain-knowledge constraint. As shown in the case study, while UUE preserves retained knowledge effectively, it often fails to produce semantically coherent outputs on the forget set—indicating that forgetting sometimes comes at the cost of linguistic completeness and expressiveness. The notion of untargeted unlearning is valuable, but what constitutes a semantically meaningful “untargeted” response remains underexplored.

### Questions
1.	Does the main experiment in the paper use full fine-tuning or a parameter-efficient fine-tuning (PEFT) approach? If it uses PEFT, what are the specific LoRA parameters, and what are the LoRA hyperparameters used in UUE-L?
2.	How is the quality of untargeted unlearning evaluated? For questions intended to be unlearned, what kind of responses are considered good or desirable?
3.	How are the specific layers selected for unlearning? Are they treated as hyperparameters?
4.	How well does UUE generalize? How does its performance vary across different prompts or reasoning scenarios?

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
This paper introduces UUE, a lightweight, controllable framework for untargeted LLM unlearning that casts the problem as null-space-guided model editing to erase undesired knowledge while preserving general utility. Unlike fine-tuning-based methods that risk degrading performance due to knowledge entanglement, UUE inserts pluggable unlearning adapters after FFN layers in selected transformer blocks, freezing the backbone and updating only adapter parameters. It proposes a novel untargeted editing objective that inverts forget-set representations toward their antipodes (rather than fixed refusal outputs), ensuring convex, stable optimization without explicit targets. By projecting updates into the left null space of retain-set representations—computed efficiently via EVD on the Gram matrix—it derives a closed-form analytical solution that minimizes interference with retained knowledge. A low-rank variant, UUE-L, integrates LoRA to further reduce trainable parameters and enable flexible deployment. Extensive experiments on TOFU (fictitious author forgetting) and WMDP (hazardous knowledge removal) across Phi-1.5B, Llama2-7B, and Zephyr-7B show UUE and UUE-L achieving state-of-the-art forget quality (FQ) and model utility (MU), outperforming baselines like GA, GradDiff, IDK, and RMU in both efficacy and preservation.

### Strengths
- The theoretical foundation is solid: the null-space projection theorem, closed-form update derivation, and convexity proof of the untargeted objective form a coherent, mathematically rigorous pipeline that directly addresses optimization instability in prior untargeted methods.
- Experimental design is comprehensive, covering two distinct unlearning paradigms (entity-level and hazardous knowledge), multiple model families and sizes, three TOFU difficulty levels, and fine-grained metrics (ROUGE-L, FQ, MU subcomponents), with ablations validating each component’s contribution.
- Writing and structure are professional and accessible: motivation is clearly framed with intuitive figures (targeted vs. untargeted), method sections build logically from adapter design to closed-form solution to LoRA extension, and results are presented in well-organized tables with statistical significance.

### Weaknesses
Model scale evaluation is limited to ≤7B parameters; claims of scalability and lightweight deployment would be more convincing with results on 13B+ or 70B-class models, where memory and compute bottlenecks are more pronounced.

### Questions
- The eigenvalue threshold for null-space approximation is manually tuned—have you experimented with data-driven selection (e.g., cumulative variance) or observed failure modes when γ is misspecified?
- Since UUE generates diverse, non-refusal responses on forget queries (sometimes creative or hallucinated), how do you ensure this doesn’t violate safety constraints in regulated domains like biosecurity?

### Soundness
4

### Presentation
3

### Contribution
4
