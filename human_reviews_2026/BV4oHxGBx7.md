# MoEEdit: Efficient and Routing-Stable Knowledge Editing for Mixture-of-Experts LLMs

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Knowledge editing (KE) is crucial for making precise modifications to factual knowledge within large language models (LLMs). Existing KE methods, however, are primarily designed for dense architectures, limiting their applicability to the increasingly popular sparse Mixture-of-Experts (MoE) models that power modern scalable LLMs. While MoEs offer remarkable efficiency and capacity scaling, their unique structure introduces new challenges for KE. Naively adapting dense-model editors to MoEs is not only computationally expensive but also induces routing distribution shifts that degrade model stability and consistency. To address these challenges, we introduce MoEEdit, the first systematic framework for routing-stable knowledge editing in MoE LLMs. Our approach reparameterizes expert updates through per-expert null-space projections, ensuring router inputs remain invariant to suppress these shifts, and solves the resulting block-structured optimization with an efficient block coordinate descent (BCD) solver. Experiments demonstrate that MoEEdit achieves state-of-the-art efficacy and generalization, while maintaining high specificity, routing stability, and superior computational and memory efficiency. Our work establishes a robust foundation for scalable and precise knowledge editing in modern sparse LLMs by highlighting the necessity of routing-stable interventions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes MoEEdit, a novel framework for knowledge editing in sparse Mixture-of-Experts (MoE) large language models. To address the challenges of high computational cost, expert coupling, and routing distribution shift when applying existing methods designed for dense models, MoEEdit introduces two key techniques: 1) per-expert null-space projection, which preserves outputs on retained samples and prevents routing shifts; and 2) a randomized block coordinate descent (BCD) solver that decomposes the large-scale optimization into smaller, expert-wise subproblems, improving computational efficiency. This work establishes a strong baseline for knowledge editing in MoE-based models.

### Strengths
Strength 1: The identification and analysis of "routing distribution shift" is a key contribution of this paper. The authors clearly explain how parameter edits in one MoE layer can trigger cascading changes in downstream routing patterns, thereby disrupting the model's expert specialization pathways, providing a theoretical foundation for future research.

Strength 2: The experiments are relatively comprehensive and demonstrate superior performance, effectively supporting the theoretical analysis.

### Weaknesses
Weakness 1: The actual computational cost of the BCD solver is not clearly presented, especially in comparison with traditional approaches—for example, the total time required to complete 1,000 edits. This lack of comparison could undermine the claim of "efficiency" made in the title.

Weakness 2: Experiments are conducted with up to 1,000 sequential edits. While this is a reasonable benchmark, the true stress test for knowledge editing lies in much larger scales (e.g., 3k or more) of continuous edits. It remains unclear whether MoEEdit’s performance and routing stability would degrade after such extensive editing.

### Questions
Question1: see Weakness 1
Question2: see Weakness 2
Question3: Have the authors considered or analyzed how the sampling strategy and size of the preservation set affect the model's editing performance?

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
The paper proposes MoEEdit, a method for routing-stable knowledge editing in MoE LLMs. This method reparameterizes expert updates through null-space projections for each expert, suppressing routing shifts. The method also introduces randomized block coordinate descent to efficiently implement expert updates, addressing the computational complexity issues caused by direct solving.

### Strengths
The paper's proposed MoEEdit effectively mitigates the routing shift problem in MoE LLMs editing while requiring only minimal computational cost. Experiments demonstrate that this method achieves better performance than other baseline methods across three metrics: Efficacy, Generalization, and Specificity.

### Weaknesses
- The paper does not explain how the v used in editing is obtained. As the output of the edited layer, v should also be one of the factors contributing to the routing shift problem, but the authors do not discuss this element.
- The paper lacks detailed descriptions of the experimental setup.
- The paper has some writing deficiencies, such as when the metric RS first appears in Section 4.1, it does not introduce that the full name of this metric is routing-similarity, which only appears later in Section 5.3.

### Questions
- How is the v used in editing obtained? What impacts does v have on the routing shift problem?
- What are the experimental settings used in the paper?

### Soundness
2

### Presentation
3

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
This paper introduces MoEEdit the first systematic framework for knowledge editing in Mixture-of-Experts or MoE LLMs. The authors identify a tripartite challenge for MoE editing computational cost expert coupling and most importantly routing distribution shift. They argue that naively editing MoE models perturbs layer outputs which in turn causes downstream routers to select different experts destabilizing the model. MoEEdit solves this by reparameterizing updates using per-expert null-space projections. This constrains the edits to not affect a preservation set which keeps router inputs invariant and prevents routing shift. To make this tractable they solve the block-structured optimization with an efficient randomized block coordinate descent BCD solver. Experiments on large MoE models show MoEEdit achieves state-of-the-art performance while being highly efficient and maintaining routing stability.

### Strengths
1. This is the first paper to properly formalize and tackle the knowledge editing problem for sparse MoE models.
    
2. The identification of routing distribution shift as the key failure mode for MoE editing is a novel and very important insight.
    
3. The proposed solution is elegant. The per-expert null-space projection directly targets the routing shift problem.
    
4. The BCD solver is a crucial component that makes the method practical. The paper clearly shows why a global one-shot solution is computationally infeasible and provides a scalable alternative.

### Weaknesses
1. The analysis focuses entirely on the routing shift in _subsequent_ layers. It's unclear how the edit affects routing for _subsequent tokens_ within the _same_ edited layer.
    
2. The null-space projection is critical but it relies on a preservation set $K_n^0$. The paper never explains how this set is collected how big it is or how sensitive the method is to its quality.
    
3. The BCD solver is efficient in terms of scaling with expert count $N$ but it's still an iterative process. Figure 4 shows it needs 6-10 passes. This makes it much slower per edit than one-shot dense editors like ROME. A wall-clock time comparison is needed.
    
4. The novelty versus AlphaEdit could be clearer. The paper should test what happens if you just apply a standard global AlphaEdit projection to the K-active experts. This would help isolate the benefit of the per-expert projection design.

### Questions
1. How do you build the preservation set $K_n^0$ for each expert? How many samples do you need for the projection matrix $P_n$ to be effective?
    
2. Your routing stability analysis looks at downstream layers. Does the edit also change the expert routing _within_ the edited layer for the tokens that follow the edit position?
    
3. The BCD solver needs multiple passes. Can you report the actual wall-clock time in seconds for a single edit batch? I'm trying to understand the real-world speed.
    
4. What happens if you just use a single global AlphaEdit-style projection on the active experts? I'm trying to figure out if the per-expert projection is the most critical part of your method.

### Soundness
3

### Presentation
4

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
The paper targets knowledge editing (KE) for sparse Mixture‑of‑Experts (MoE) LLMs, arguing that naively adapting dense‑model editors leads to high compute, inter‑expert coupling, and routing distribution shift that destabilizes behavior in downstream layers. The authors  Introduce MoEEdit, which (i) reparameterizes per‑expert updates via a per‑expert null‑space projection constructed from preservation features to keep router inputs invariant (suppressing routing shift), and (ii) solves the resulting block‑structured problem with a randomized block coordinate descent (BCD) solver that updates only relevant experts.

### Strengths
- Clear identification of MoE‑specific failure modes (compute, inter‑expert coupling, routing drift) and a principled block‑structured edit formulation tailored to MoE.
- Practical, scalable solver: exact per‑expert ridge updates in a randomized BCD loop over active experts; clear normal equations and complexity discussion; convincing synthetic scaling vs. a global solver.
- Strong results on two MoE LLMs and two standard editing benchmarks.

### Weaknesses
- The central stabilization mechanism is a null‑space projection constructed from preservation features, which the paper itself positions as inspired by AlphaEdit’s null‑space constrained editing for dense models; the novelty is its per‑expert application in MoE and its connection to router invariance. While this is a reasonable extension, the claim to be “the first” routing‑stable framework may be too strong without comprehensive comparison to prior MoE‑specific KE or adaptor‑based lifelong editing.
- Missing comparison to MoE‑specific editing work (LEMoE) and absence of citation. The manuscript does not cite or compare to “LEMoE: Advanced Mixture of Experts Adaptor for Lifelong Model Editing of Large Language Models.” This omission weakens the novelty claim. Prior art already identified “inconsistent routing” as a key barrier in lifelong model editing and introduced a dedicated mechanism (KV anchor routing) to enhance routing consistency between training and inference. This directly contradicts MoEEdit’s novelty claim that it is “the first to formally identify routing-induced instability” as central. Both methods pursue the same core objective—maintaining routing stability during/after edits in MoE settings—via targeted constraints on the routing pathway. In addition, both target lifelong/sequential and batch editing regimes at scale with similar benchmarks and reporting conventions.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
