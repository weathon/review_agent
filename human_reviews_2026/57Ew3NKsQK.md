# Decoupling of Experts: A Knowledge-Driven Architecture for Efficient LLMs

- Avg Score: 1.60
- Decision: Reject
- Scores: 2, 4, 0, 0, 2

## Abstract
Current large language models (LLMs), particularly Mixture-of-Experts (MoE) variants, face challenges in achieving efficient, structured, and interpretable scaling. We introduce the Decoupling of Experts (DoE) architecture, a novel framework that addresses these limitations by grounding computation in a hierarchically organized and dynamically updated knowledge space. Our methodology features a two-stage lifecycle: we first use Latent Dirichlet Allocation (LDA) to build a semantic topic foundation from the training corpus. This knowledge is then integrated into the main LLM, where it is dynamically refined. Critically, we discard traditional, static MoE experts. Instead, the expert entity is a dynamic \textbf{Knowledge Block} synthesized on-the-fly by reusing the Key and Value matrices from the attention computation. We replace the standard load balancer and softmax gating with an \textbf{Attention Gating Control (AGC)} that employs a VAE-based router with a ReLU activation for expert composition. This entire process is optimized with a composite loss function, balancing next-token prediction with a KL-divergence-based expert loss. Our analysis reveals that this architecture induces a remarkable \textbf{heterogeneous specialization} across layers, with some layers differentiating into "science" and "humanities" domains, while others converge on general functions. This demonstrates a learned, hierarchical division of labor, paving the way for a new, more efficient scaling dimension based on the number of structured experts.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a novel modification to the Mixture-of-Experts (MoE) paradigm, named Decoupling of Experts (DoE). The key idea is to introduce a knowledge-driven mechanism that assigns semantic meaning to experts by constructing token-level topic distributions using Latent Dirichlet Allocation (LDA). These topics are then integrated into a hierarchical VAE-based routing framework, where “experts” are dynamically composed from the attention Key/Value matrices rather than fixed FFN layers. The method aims to improve interpretability and efficiency simultaneously.
Overall, the approach is interesting and yields promising empirical results — the reported gains over comparable dense and MoE models are substantial. However, the paper suffers from significant writing and presentation issues that severely affect its clarity and reproducibility.

### Strengths
1. The reported performance improvements (e.g., on MMLU, CMMLU, and HumanEval) seem effective and consistent, suggesting that the proposed method has real potential.
2. The idea of grounding experts in a data-derived semantic topic space makes intuitive sense and could inspire future research into interpretable or knowledge-aware MoE architectures.

### Weaknesses
1. The paper contains numerous avoidable mistakes that undermine confidence in the results. Many citations appear as “(?)” and cannot be resolved, and major elements such as Table 4 are incomplete or incorrectly rendered. These errors make it difficult to verify the validity of the claims and give the impression that the manuscript was not carefully proofread.
2. The motivation behind several design choices (e.g., why LDA is needed, how the VAE router interacts with attention states, and why ReLU gating is preferable) is not clearly justified. The logic often jumps between ideas without sufficient explanation, which makes the overall contribution difficult to follow.
3. Due to missing details and inconsistent descriptions (e.g., two-stage training, knowledge signal integration), it is unclear whether other researchers could reproduce the reported improvements.

### Questions
Q1. Could the authors specify the total GPU hours required for Stage 1 and Stage 2 pre-training of the DoE-7B model, including hardware configuration (e.g., GPU type and number of nodes)

Q2. How was the 10B-token SFT dataset constructed? Please clarify its data sources, task composition, and whether any filtering or deduplication was applied.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Decoupling of Experts (DoE), a new approach to improve efficiency, structure, and interpretability of MoE based LLMs. 
Training process is comprised of two stages: 
First, Latent Dirichlet Allocation (LDA) is adopted to learn the topics in the corpus.
Then, the MoE experts are substitued by dynamic knowledge blocks. Moreover, the load balancing is achieved by uses a VAE-based router with ReLU gating.
DoE enables heterogeneous specialization across layers, with some layers specializing in domains like science, while other layers handle general tasks. 
Experiments on seven benchmarks show good performance on multiple benchmarks.

### Strengths
1. Using a LDA model to provide the model with topic information empowers the model with hierarchical understanding ability, which is sound and reasonable.
2. Performance is good compared with baseline methods.
3. Ablation studies show the effective of different design. Also some experiments show the merits of the key designs.

### Weaknesses
1. The detailed structure of VAE and LDA is not provided.
2. Authors should carefully review the writing, there are tons of typos, e.g. the citation of LDA, the size of Table 4.
3. Does adding the LDA clustering procedure harms the training and inference efficiency? Authors should provide a detailed analysis.
4. Authors did not provide the performance of models with other parameter size, i.e., form DoE 1B to 60B.
5. In Table 2, when the loss weight is 0.01, the performance is worse than no auxiliary loss, authors should analyse the phenomenon.
6. Why the architecture of DoE is in a MoE way? Have authors tried using other designs on a dense model (reducing experts to 1)?
7. Authors should give an ablation study on the effectiveness of each stage. How much does LDA contributes to the performance?
8. I think the dynamic knowledge blocks do not match the motivation of providing interpretable scaling, author should add further analysis.

### Questions
1. Does the training dataset overlaps with the evaluation benchmarks? Since the result of MMLU is incredibly high.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper presents **Decoupling of Experts (DoE)**, a novel architecture for large language models that replaces static MoE experts with dynamic, semantically grounded **Knowledge Blocks** synthesized on-the-fly from attention mechanisms. The approach aims to make expert computation more interpretable and efficient.

DoE follows a **two-stage process**:  
1. **Offline Knowledge Construction:** Uses **Latent Dirichlet Allocation (LDA)** on a 4.5T-token corpus to create topic-based expert labels.  
2. **Dynamic Refinement:** Incorporates these knowledge signals during pretraining through a **hierarchical VAE-based router with ReLU gating** and an **Attention Gating Control (AGC)** mechanism linking attention outputs to a global knowledge matrix.

A composite loss combines next-token prediction and a KL-based expert supervision term. Experiments show **DoE-7B** outperforms comparable dense and MoE baselines on benchmarks like **MMLU, GSM8K, and HumanEval**, and exhibits layer-wise specialization (e.g., “science” vs. “humanities” experts).  
However, reproducibility and methodological transparency are limited—key details such as dataset accessibility, ablation figures, and routing metrics are missing, weakening the empirical validity.

### Strengths
1. **Originality:**  
   The paper proposes a novel redefinition of the Mixture-of-Experts paradigm by replacing static experts with dynamically constructed **Knowledge Blocks** synthesized from attention states. This conceptual shift—grounding expert behavior in a semantic knowledge space rather than fixed FFNs—is original and potentially influential for future model interpretability and efficiency research.

2. **Technical Quality:**  
   The DoE architecture creatively integrates **LDA-based topic priors**, **VAE-based routing**, and **Attention Gating Control (AGC)** into a unified framework. The composite loss function design and the hierarchical routing mechanism demonstrate a clear effort to formalize structured specialization within LLMs, showing promising results on standard benchmarks.

3. **Clarity:**  
   Despite occasional missing figures and incomplete references, the paper’s narrative and architectural descriptions are generally clear and logically organized. The hierarchical explanation from knowledge to experts helps the reader grasp the multi-level concept of "knowledge-driven specialization."

4. **Significance:**  
   The proposed framework addresses fundamental challenges in MoE scaling—inefficient routing and poor interpretability—by introducing a semantically grounded alternative. If validated, this work could provide a new scaling dimension for LLMs and inspire further research into interpretable, knowledge-centric architectures.

### Weaknesses
1. **Incomplete and Incorrect Referencing:**  
   The paper exhibits numerous citation gaps and placeholder references such as “(?)”, leaving major conceptual components (e.g., hierarchical VAE structure, AGC mechanism, and related works on knowledge-driven routing) unsupported. This issue is severe because it prevents verification of the theoretical lineage and novelty. For example, the authors claim that hierarchical VAEs are “inspired by previous work,” but fail to specify or cite any prior studies. Similarly, the LDA integration step lacks reference to established topic-based pretraining literature (e.g., Blei et al., 2003; Gururangan et al., 2020), which weakens credibility and contextual grounding.  

2. **Ambiguous Model Scope and Lack of Architecture Transparency:**  
   Nowhere in the manuscript does the paper explicitly state which LLM backbone (Transformer variant, number of layers, or attention configuration) was used to implement DoE. This omission makes it difficult to determine the generality of the proposed approach. Moreover, the statement that “knowledge resides in specific layers (0, 2, 6, 10, 18)” implicitly assumes structural consistency across all models, which is unsubstantiated. Without comparative evidence from multiple architectures, such a conclusion risks being model-specific rather than generalizable.

3. **Unsubstantiated Claims of Interpretability:**  
   The claim that the DoE framework “produces interpretable expert specialization” is not empirically supported. There are no quantitative interpretability metrics — such as neuron attribution, topic coherence, or attention-head probing — to demonstrate that the identified “knowledge blocks” correspond to meaningful semantic clusters. The interpretability arguments rely solely on qualitative descriptions and isolated examples (e.g., “science vs. humanities”), which are anecdotal rather than analytical. This makes it difficult to distinguish between genuine specialization and random activation clustering.  

4. **Feasibility Issues in the LDA Stage:**  
   The proposed pipeline begins with running **LDA on a 4.5-trillion-token corpus**, which is computationally infeasible even with large-scale distributed systems. No details are provided on approximation methods (e.g., online LDA, mini-batch processing, or sampling strategies). Additionally, it is unclear how noisy or ambiguous topic assignments are handled when serving as “ground-truth” expert labels (`y_expert`). This step may inject significant noise into Stage 2 training, contradicting the assumption of a reliable knowledge foundation.

5. **Methodological Gaps in the Two-Stage Lifecycle:**  
   The interaction between Stage 1 (LDA-based initialization) and Stage 2 (VAE-based refinement) lacks mathematical rigor and implementation clarity. The paper does not specify how the token-level topic distributions are aligned with the continuous hidden states of the Transformer during pretraining. Furthermore, it remains unclear whether the “knowledge signal” is injected as an additive embedding, a concatenation, or a residual bias term. This missing detail makes replication and validation practically impossible.

6. **Weak Experimental Rigor and Limited Baselines:**  
   Although the paper claims state-of-the-art performance, the experimental section omits comparisons with several relevant MoE baselines (e.g., GLaM, Switch Transformer, Mixtral, or MegaBlocks). The use of DeepSeek-MoE as the sole comparison point is insufficient to establish superiority. Moreover, the authors provide no statistical variance or ablation on scaling (e.g., 1B vs. 13B vs. 60B) beyond a few isolated observations. As a result, the claim that DoE scales “efficiently and interprets knowledge hierarchically” remains unproven.

7. **Lack of Computational Efficiency Evidence:**  
   The paper repeatedly asserts that DoE “removes the need for a load balancer” and provides “more efficient routing,” but offers no computational benchmarks to support these statements. There are no reported FLOPs, latency, or memory usage comparisons. Without such quantitative profiling, it is impossible to assess whether the proposed model is actually more efficient than traditional MoE systems.

8. **Inadequate Visualization and Missing Figures:**  
   Several figure references (e.g., “Figure ??”) are missing or broken, and the visualizations that do appear are underexplained. For instance, the visualization of “expert activation patterns” in Figure 3 lacks axis labels, legends, and quantitative interpretation. The text also refers to diagrams in Figure 1 and Figure 2 that do not provide sufficient architectural detail to reproduce the system. This significantly reduces clarity and reproducibility.

9. **Overgeneralization of Experimental Results:**  
   The claim that DoE “learns a hierarchical division of labor” is primarily derived from visual inspection rather than formal statistical evidence. There is no clustering or entropy-based analysis of expert activations to confirm that specialization is consistent and not random. Furthermore, the evidence for “domain-dispatching layers” (0, 2, 6, 10, 18) is limited to one model variant (DoE-7B), which does not establish generality across model sizes or data distributions.

10. **Presentation and Formatting Deficiencies:**  
    The paper suffers from multiple formatting and typographical errors:
    - **Misuse of Markdown syntax (`****`)** for bold text, resulting in unrendered emphasis markers in several sections.  
    - **Table 4 exceeds the page boundary**, violating conference formatting standards.  
    - **Broken citations** and missing reference brackets.  
    - **Inconsistent terminology**, where “Knowledge Blocks,” “Knowledge Fields,” and “Bricks” are used interchangeably without clear definition.  
    These errors collectively diminish the professionalism and readability of the manuscript.

11. **Insufficient Discussion of Limitations and Future Work:**  
    The paper does not acknowledge potential weaknesses of DoE — such as the overhead of maintaining topic alignment, the potential for topic collapse in the VAE router, or challenges in extending DoE to multimodal LLMs. A critical discussion of such risks would strengthen the credibility of the work.

12. **Typographical Errors:**  
    Several minor typographical issues are observed throughout the text, including punctuation errors (e.g., double periods in Line 107), inconsistent capitalization (e.g., “expert entity” vs. “Expert entity”), and inconsistent equation formatting. While minor individually, these contribute to the perception of insufficient editorial care.

### Questions
> I encourage the authors to thoroughly address the weaknesses and questions raised in this review. If the authors can provide detailed explanations and in-depth clarifications during the rebuttal, and if the revised version demonstrates substantial progress in both clarity and improvement, I will be willing to reassess the manuscript and **adjust my overall rating** accordingly, based on the quality and depth of the revision.

---

1. **Clarification on the LDA Stage Feasibility:**  
   - Running *Latent Dirichlet Allocation* (LDA) on a 4.5-trillion-token corpus seems computationally unrealistic. Could the authors clarify how this was achieved?  
   - Was an *approximate* or *distributed LDA* implementation used (e.g., online LDA or sampling-based inference)?  
   - How sensitive is the downstream model performance to noise or instability in the topic assignments generated during Stage 1?

2. **Details on Knowledge Signal Integration:**  
   - How is the topic signal from Stage 1 integrated into the Transformer during Stage 2 — via *additive embedding*, *concatenation*, or *residual biasing*?  
   - How are token-level topic distributions aligned with subword-level embeddings, given that LDA typically operates on word-level units?  
   - Could you provide mathematical notation or pseudo-code showing how the dual-input embedding is implemented?

3. **Empirical Verification of Interpretability Claims:**  
   - The paper frequently claims “structured and interpretable specialization.” What quantitative methods were used to verify this (e.g., probing classifiers, entropy analysis, topic coherence metrics)?  
   - Can the authors provide empirical evidence that the “Knowledge Blocks” correspond to consistent semantic clusters rather than arbitrary neuron activations?  
   - How were the so-called “science” vs. “humanities” domains in Figure 3 defined or annotated?

4. **Scope and Backbone Model Transparency:**  
   - Which backbone architecture was used to implement the DoE framework (e.g., Transformer-Decoder, Llama-like, or GPT-style)?  
   - How many layers and attention heads does the DoE-7B model contain, and are the reported “domain dispatch layers” (0, 2, 6, 10, 18) consistent across other scales (13B, 33B, etc.)?  
   - If the claim that “knowledge resides in specific layers” is universal, have you verified it across multiple architectures?

5. **Quantitative Efficiency Evaluation:**  
   - The paper asserts that DoE “removes the need for a load balancer” and “improves routing efficiency.” Could you provide quantitative evidence, such as FLOPs, latency, or throughput comparisons with standard MoE models (e.g., DeepSeek-MoE, GLaM)?  
   - How does the AGC-VAE router scale computationally with model size, and does it introduce additional memory overhead?

6. **Comparative Baseline Coverage:**  
   - Why were baselines such as *Switch Transformer*, *Mixtral*, or *GLaM* omitted? These models are closely related to DoE’s goals of sparse expert activation and interpretability.  
   - Could the authors clarify whether their “state-of-the-art” claim holds when compared against these architectures under equivalent training budgets?

7. **Mathematical and Implementation Clarity:**  
   - The connection between Equation (1)–(3) (AGC mechanism) and Equation (4) (composite loss) is underspecified. How exactly does the gradient from the auxiliary expert loss (`L_expert`) affect the attention gating?  
   - Is the KL-divergence-based loss computed per-token, per-layer, or globally?  
   - Can you share more details on how the router’s VAE parameters are trained relative to the main model weights?

8. **Visualization and Reproducibility:**  
   - Several figure references (e.g., “Figure ??”) appear broken. Could the authors provide the missing visualizations of training dynamics and knowledge block activations?  
   - Is there a quantitative way to reproduce the visual evidence shown in Figure 3? For example, can you release cluster activation matrices or layerwise statistics?

9. **Theoretical Justification of the Knowledge Hierarchy:**  
   - The paper suggests that knowledge progresses from embeddings → blocks → skills → experts. Is there any empirical or theoretical analysis validating this hierarchy?  
   - How do the “Knowledge Fields” (K = 128) relate to the number of emergent experts, and how was this hyperparameter chosen?

10. **Formatting and Citation Problems:**  
    - Many references are missing or improperly formatted (e.g., “(?)”). Will the authors provide a corrected bibliography?  
    - Table 4 exceeds page boundaries — can the authors reformat it according to the conference template?  
    - Multiple occurrences of `****` appear instead of bold text. Could you confirm this is a LaTeX or Markdown parsing issue and not intentional emphasis?

11. **Generalization and Future Extensions:**  
    - The DoE model is evaluated primarily on English-language tasks. How well would this architecture extend to multilingual or multimodal settings?  
    - Given that the system relies on topic priors, how would DoE handle tasks without clear topical structure (e.g., reasoning or instruction following)?  
    - Could future work explore replacing LDA with learned topic embeddings or transformer-based latent concept discovery?

12. **Self-Consistency and Layer Activation Patterns:**  
    - The authors claim that two of four experts become “pruned” in deeper layers. How consistent is this phenomenon across random seeds and datasets?  
    - Are the pruning effects emergent or enforced by any sparsity regularization?  
    - Would explicit sparsity control (e.g., L₁ penalties or entropy constraints) improve the robustness of the observed specialization?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
While the paper’s topic could be of potential interest, the submission suffers from serious presentation and formatting issues that make it extremely difficult to review in its current form. There are numerous violations of the formatting guidelines — for instance, incorrect or missing citations, unreadable figures (e.g., unclear symbols in Figure 1, very small fonts in Figure 2), and tables that overflow the margins (e.g., Table 4). Beyond these issues, the overall writing and organization suggest the paper is still in a draft state rather than a polished submission. For these reasons, I recommend desk rejection.

Authors should submit manuscripts that meet basic readability and presentation standards by the submission deadline. Submitting a poorly prepared or incomplete manuscript, with the expectation that formatting and clarity can be fixed during the discussion phase, places an unnecessary burden on reviewers and undermines the integrity of the review process. Moreover, considering such works is unfair to authors who invest the time and effort to ensure their submissions are clear, complete, and fully compliant with the conference guidelines within the stated deadline.

Having said that, I acknowledge that the underlying idea could become a valuable contribution once the paper is properly prepared and formatted. However, submissions in this state appear to exploit the review process by relying on reviewers’ goodwill to interpret and overlook easily avoidable presentation issues. This practice effectively circumvents the submission deadline, places additional strain on an already overburdened review system, and should be clearly discouraged.

### Strengths
-

### Weaknesses
-

### Questions
-

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
- The paper proposes Decoupling of Experts (DoE), which uses a two-stage lifecycle: Stage 1 applies Latent Dirichlet Allocation (LDA) to the training corpus to extract semantic topic foundations and generate ground-truth expert labels $y_{expert}$; Stage 2 integrates this knowledge into LLM pre-training with dual-input embeddings (token + knowledge signal) that are dynamically refined using Key and Value matrices from attention computations.

- Unlike traditional MoE with static FFN experts, DoE defines experts as dynamic Knowledge Blocks synthesized on-the-fly from attention states. The architecture replaces standard load balancers with Attention Gating Control (AGC) and uses a hierarchical VAE-based router with ReLU activation instead of softmax, trained via composite loss.

- Analysis of DoE-7B reveals that certain layers (0, 2, 6, 10, 18) act as "domain dispatchers" with experts specializing in distinct domains (science vs. humanities), while other layers show convergent behavior handling general computations, with deeper layers exhibiting learned pruning where experts remain inactive—demonstrating hierarchical division of labor.

### Strengths
- The paper presents an original reconceptualization of MoE experts as dynamic Knowledge Blocks synthesized on-the-fly from attention states rather than static FFN parameters. The hierarchical knowledge representation (Knowledge → Knowledge Block → Emergent Skills → Expert) combined with the two-stage training paradigm (LDA-based semantic initialization followed by VAE-based dynamic refinement) offers an interesting integration of classical topic modeling with modern neural architectures, potentially providing a more interpretable foundation for expert specialization.

- The paper introduces multiple components—AGC mechanism reusing attention Key/Value matrices for routing, VAE-based router with ReLU gating replacing softmax, removal of load balancers, and composite loss with explicit routing supervision.

### Weaknesses
- The reported results in Table 4 are implausibly strong—DoE-7B achieves 91.2 MMLU and 87.0 HumanEval, representing 12.7 and 26+ point improvements over comparable models, surpassing even much larger models like Mixtral-8×7B. These gains are presented without error bars, multiple runs, statistical significance tests, or ablations isolating architectural contributions from the massive 4.5 trillion token training corpus (far exceeding typical 7B model budgets). Critical details are missing: What is the computational cost of the two-stage LDA preprocessing on 4.5T tokens? Without rigorous controls comparing DoE against baselines trained on identical data with identical compute, it's impossible to determine whether improvements stem from the architecture or simply from superior training resources.

- The specialization evidence in Figure 2 only analyzes token-level expert activation patterns across domains, which provides an extremely limited view of functional specialization. The analysis fails to measure: (1) whether experts learn distinct parameter structures; (2) whether the claimed "science vs. humanities" specialization is robust across different prompts within the same domain; (3) quantitative metrics for specialization degree; (4) whether the LDA-derived topics actually align with the emergent specialization patterns. The claim that layers 0, 2, 6, 10, 18 are "domain dispatchers" is based solely on visual inspection without statistical validation. More critically, token-level routing preferences could simply reflect lexical correlations (science texts use different vocabulary) rather than genuine functional specialization—the paper provides no evidence that experts encode domain-specific reasoning capabilities rather than just domain-specific lexical patterns.

- The architecture introduces numerous intertwined components—LDA preprocessing, dual-input embeddings, hierarchical VAE routers (with different configurations for first/middle/final layers), Knowledge Attention mapping (Equations 1-3), VAE-ReLU gating, composite loss with KL divergence—making it impossible to understand which elements drive performance. It never validates the necessity of the complex hierarchical VAE structure or the two-stage LDA paradigm. Why is offline LDA clustering on 4.5T tokens necessary when the VAE could learn topics end-to-end? The claim that AGC "removes load balancer noise" is unsubstantiated—no experiments compare DoE against modern load-balancing-free methods like DeepSeekMoE's auxiliary-loss-free strategy. Table 3 shows only 0.36 PPL improvement (30.89→30.25) from the full architectural stack, raising questions about whether the complexity is justified.

- The paper provides no theoretical analysis of why LDA-derived topic priors should align with optimal expert specialization for language modeling, or why ReLU gating should outperform softmax beyond an empirical "magnitude control." The claim that Knowledge Blocks are "synthesized from QKV computation"  is misleading—they're actually learned embeddings in matrix $Z$ that are simply gated by attention outputs, not fundamentally different from standard MoE expert parameters. The hierarchical knowledge taxonomy (Knowledge → Block → Skill → Expert) lacks formal definition and empirical validation—what constitutes a "skill" versus a "knowledge block" is never operationalized. Furthermore, all experiments use the same 4.5T token corpus; there's no evidence the architecture generalizes to different data distributions, domains, or languages, and no analysis of how the LDA stage would adapt to continual learning or domain shift scenarios.

### Questions
- Can you provide results comparing DoE against strong baselines trained on the exact same 4.5 trillion token corpus with identical compute budgets? The current Table 4 comparisons are unfair since baselines use different datasets and training scales. What are the actual PPL/benchmark improvements when only the architecture differs, isolating your contributions from data/compute advantages?

- Beyond bar charts in Figure 2, can you provide quantitative metrics demonstrating genuine functional specialization? Specifically: (1) measure expert parameter diversity (2) compute routing entropy and expert output diversity scores, (3) perform controlled experiments showing domain-specific reasoning transfer (e.g., if "science expert" is ablated, does only science performance degrade?), and (4) prove the LDA topics align with emergent specialization patterns through topic-expert correspondence analysis?

- What is the performance of DoE when Stage 1 LDA preprocessing is removed and the VAE learns topic structure end-to-end from random initialization? The two-stage pipeline adds significant computational overhead (LDA on 4.5T tokens)—can you justify this complexity by showing it outperforms joint end-to-end training? Additionally, how sensitive are results to the number of LDA topics K, and does the architecture work with alternative topic models or completely different semantic initializations?

- Can you release model weights, complete training code, LDA preprocessing pipelines, and custom CUDA kernel implementations to enable independent verification? What is the wall-clock time and computational cost (FLOPs, GPU-hours) for: (1) Stage 1 LDA on 4.5T tokens, (2) Stage 2 training compared to baseline MoE, and (3) inference latency with VAE-based routing versus standard softmax routers? How does DoE scale to larger models (65B+) given the added complexity?

- Can you provide ablations justifying the hierarchical design (different VAE configurations for first/middle/final layers) versus a simpler unified VAE? Why is the first layer conditioned on LDA priors while the final layer uses next-token signals—what happens if this hierarchy is reversed or flattened?

### Soundness
2

### Presentation
1

### Contribution
1
