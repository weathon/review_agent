# Concept-Based Local Unified Explanations

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
There is a growing demand to combine model-agnostic explanation methods with concept-based explanations, as the former can explain models across different architectures while the latter makes the explanations more faithful and understandable to end-users.
However, existing concept-based model-agnostic explanation methods are limited in scope, as they mainly focus on attribution-based explanations and lack support for richer explanation types such as sufficient conditions and counterfactuals, which limits their applicability.
To bridge this gap, we propose a general framework ConLUX, to elevate existing local model-agnostic techniques to provide concept-based explanations.
Our key insight is that we can uniformly extend existing local model-agnostic methods to provide unified concept-based explanations with large pre-trained models perturbation.
We have instantiated ConLUX to provide concept-based explanations in three forms: attributions, sufficient conditions, and counterfactuals, and applied it to popular text, image, and multimodal models.
Our evaluation results demonstrate that \toolname provides explanations more faithful than state-of-the-art concept-based explanation methods, and provides richer explanation forms that satisfy various user needs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the limitations of current concept-based model-agnostic explanation methods, which mainly focus on attribution tasks and offer limited types of explanations. The authors propose ConLUX, a general and lightweight framework that systematically extends local model-agnostic explanation techniques—such as LIME, Kernel SHAP, Anchors, and LORE—into the concept-level domain. By leveraging large pre-trained models to perform concept-level perturbations, ConLUX enables the generation of unified concept-based explanations, including attributions, sufficient conditions, and counterfactuals.

### Strengths
The paper makes a contribution by bridging model‑agnostic and concept‑based explanations, positioning ConLUX as a framework capable of generalizing across existing local explanation tools. Beyond the conceptual advancement, the empirical work is comprehensive: ConLUX is instantiated on four popular explanation methods (LIME, Kernel SHAP, Anchors, and LORE) and consistently improves their fidelity metrics, demonstrating broad applicability and robustness. The approach also exhibits comparative superiority over current concept‑level explanation methods across both text and image tasks, achieving higher fidelity than TBM, LACOAT, EAC, and ConceptLIME.

### Weaknesses
In my view, the primary concern with this paper lies in its motivation. The introduction claims that existing methods "lack support for richer explanation types such as sufficient conditions and counterfactuals." However, the paper does not sufficiently explain why these types of explanations are important or what practical or theoretical gap they fill. In other words, the “so what” question remains unanswered. Although the authors define notions such as sufficient conditions and counterfactuals later in the paper (Lines 169–182), there is limited discussion or justification about their significance. As a result, it feels as though these concepts were added to extend the existing XAI pipeline in a more mechanical way, rather than being driven by a strong underlying motivation or clear end-user need.

Second, the authors state that their tool “elevates existing local model-agnostic explanation methods to the concept level with minimal user effort.” This raises the question of what kind of contribution the paper is positioning itself as. Should we interpret it primarily as a tool-based or implementation-level contribution? From my understanding, there are already several mature toolkits that aim to unify and streamline explanation methods—for example, the Captum API. It is not entirely clear what novel capability ConLUX introduces beyond what such existing frameworks already offer.

Finally, a more minor but related concern: the paper emphasizes that the proposed method works with “minimal user effort,” yet the framework relies on large pre-trained models for concept-level perturbations. Given that LLMs often require substantial computational resources—sometimes with billions of parameters—it is difficult to reconcile this reliance with the claim of minimal user effort. Some clarification on what “minimal effort” precisely means in this context would help readers better understand the practical usability of the approach.

### Questions
**Q1. Motivation and Significance**  
- Could you elaborate on **why providing sufficient conditions and counterfactual explanations** is practically or theoretically important?  
- For instance, are there concrete end-user needs, application domains, or specific decision-making settings where the lack of such explanations has created known limitations?  
- If possible, please include empirical or user-study evidence, or a literature gap analysis, that demonstrates the necessity of supporting these richer explanation types.

**Q2. Positioning and Novelty of the Contribution**  
- How should readers understand your contribution in relation to existing XAI toolkits (e.g., Captum or other integrated explanation frameworks)?  
- It would help to explicitly state whether the novelty lies primarily in:  
  1. a new underlying algorithmic capability,  
  2. a conceptual framework for organizing explanation types, or  
  3. engineering integration and usability improvements.  
- A comparative table or ablation study against representative existing tools could clarify the incremental novelty or unique advantages of ConLUX.

**Q3. Definition of “Minimal User Effort”**  
- Can you specify in what sense the approach requires “minimal user effort”? For example:  
  - Does it refer to minimal annotation or data engineering?  
  - Does it concern ease of API integration or conceptual simplicity?  
- Given that your framework depends on large pre-trained models, it would be useful to qualify the meaning of “minimal effort” in light of potential computational costs.  
- If possible, a brief discussion or quantitative measure (e.g., hours of setup, lines of code, computational resources) would help readers gauge the practical burden.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces ConLUX, a general framework that extends feature-level, model-agnostic local explainers such as LIME, Kernel SHAP, Anchors, and LORE to the concept level without changing their learning algorithms. ConLUX replaces (i) predicate construction with concept predicates (derived by a concept extractor) and (ii) feature-level perturbations with concept-level perturbations implemented via a concept-to-feature mapping using large pretrained generators (LLMs for text; diffusion-based editors for images). The explainer’s original learner is then fit on the generated samples to produce concept attributions, concept-level sufficient conditions, and concept-level counterfactuals for the same instance.

Experiments evaluate (1) perturbation fidelity: does the generator satisfy concept toggles?, (2) explanation fidelity for ConLUX-augmented LIME/KSHAP/Anchors/LORE vs vanilla methods and concept baselines, and (3) a human study on image tasks. Perturbations satisfy the requested concepts with an average accuracy of 96.8% across five datasets (text and images). ConLUX improves coverage/precision for Anchors/LORE and AOPC/Deletion-Accuracy for LIME/KSHAP. Against concept baselines, ConLUX yields higher local surrogate accuracy; a small user study (n=18) shows that concept-level sufficient-condition and counterfactual explanations improve user coverage and accuracy over attribution-only maps.

### Strengths
The paper observes that mainstream local explainers share a three-stage workflow (predicates $\rightarrow$ perturb $\rightarrow$ learn). ConLUX elevates the first two stages to concept space while leaving the learning stage intact, yielding concept-based variants of four explainers and enabling multiple explanation forms per instance (attribution, sufficient conditions, counterfactuals).

Treating LLMs/diffusion models as concept-to-feature maps that enforce binary concept predicates helps unlock concept-level toggling across modalities.

For a single instance, ConLUX yields attributions, sufficient-condition rules, and counterfactuals, which go beyond the standard attribution-only focus of many concept-based explanation methods.

### Weaknesses
ConLUX relies on many generator calls per explanation under the same sample budgets as vanilla methods (e.g., LIME/KSHAP $\approx$1,000 samples). The paper lists hardware and sampling budgets but doesn't provide wall-clock time, memory usage, the number of LLM/diffusion calls, or the cost per explanation, nor matched-budget comparisons vs. vanilla/baselines. 

The reported 96.8% measure assesses predicate compliance, i.e., whether generated samples satisfy the requested concept on/off switches, rather than semantic realism (do the edits look natural) or manifold proximity (do they stay near real data). Moreover, the verifiers are closely aligned with the generators (LLM outputs checked by an LLM; diffusion edits checked by YOLO), which can create verifier–generator coupling and optimistically biased scores.

By design, when comparing ConLUX vs. vanilla, they evaluate at the feature-level neighbourhood; when comparing ConLUX vs. concept baselines, they evaluate at the concept-level neighbourhood. This confounds neighbourhood design with explainer quality. I would encourage the authors to evaluate both methods under the same neighbourhood (concept-level and feature-level) to isolate the source of performance gains.

ConLUX presumes reasonable concepts; there’s no ablation for missing/overlapping/correlated concepts or stability of explanations under other operations.

The author performs a small human study (n = 18, image-only), in which participants predict perturbations produced by the same concept-toggle engine that ConLUX uses, which may favour ConLUX over attribution baselines.

The paper positions related work (e.g., ACE/TCAV-style), but empirical comparisons focus on TBM/LACOAT/EAC/ConceptLIME. Differences in scope (global vs. local; internal vs. model-agnostic) can make a complete head-to-head comparison tricky, but an explicit rationale or a small ablation study should strengthen the paper.

### Questions
For each modality/explainer, report the number of perturbations, the number of LLM/diffusion calls, wall-clock time, and cost per explanation; add fidelity-vs-budget plots; and compare to vanilla and concept baselines at matched budgets.

Add a table that evaluates both vanilla and ConLUX under the same concept-level neighbourhood (and the reverse). How much of the gain persists when the neighbourhood is fixed? 

Do LLM-generated perturbations follow the same distribution as traditional masking/random perturbations? This could fundamentally affect fidelity metrics. Have you analysed this?

How many concepts should be extracted? What's the trade-off between interpretability (fewer concepts) and fidelity (more concepts)?

Can you systematically characterise when perturbations fail? Are certain concept types more prone to failure?

Add human realism ratings and alternative verifiers to reduce verifier–generator coupling.

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
The paper introduces a framework that creates a concept space for a model detached from the model's concept space and use it to generate explanations by generating samples that modifies the source data of the concepts and then using vanilla explanation techniques. The paper presents the framework well and the experiments are extensive on LIME, Anchors and LORE for various popular language and vision architectures.

### Strengths
1. The idea to create a general framework for post-hoc concept explanations is good, and needed given that the current large scale architectures  dont reveal how they came up with a decision. 
2. The quantitative experiments are well designed, extensive and show good improvements, but i am uncertain whether there was some "concept hacking" that is using more than needed concepts.
3. The implementation is detailed extensively and prompts and generated results are detailed for the text generation process, but the image prompts and outputs could be discussed better.

### Weaknesses
1. Replacing concepts with new concepts to produce counterfactuals causes ambiguity. We are not sure whether the added concept overpowers existing concepts leading the classification trajectory to a new direction. For Example: In Figure 2. The pickup truck is replaced with a bison by CONLUX, whereas LORE and anchor \textbf{removed} more complex concepts, the counterfactual of LORE and anchor moved the trajectory towards a slightly different class "half truck" from "Pickup", while CONLUX just pushed it towards the class "bison", the newly added concept, which may lead to questions like a) where all the concepts removed that lead the classifier to decide on the pickup class or is the bison just overpowering some concepts thats still unidentified? b) where all the concepts removed from the image during the sample generation actually concepts that contributed to the exact "pickup" class? I strongly feel its not the case, as LORE and anchor removed more complex concepts rather than the whole pickup truck and was still able to move the trajectory away from the "pickup" class.
2. Leading from the previous point, the use of SAM to get concepts seems good, but the concepts its segmenting seem to be objects rather than concept, which is less interesting in my opinion.
3. How did the performance differ when you replaced the sample generation model with different diffusion models? I feel that the use of single diffusion model adds bias to the generated samples, why not just remove the concept and fill with background? Also why doesnt the paper try modifying the object rather than just replacing it with an unrelated object? Like change the truck into a race truck or just make it newer, how does it change the trajectory?, I think this can alleviate the bottleneck of using SAM which segments objects rather than concepts.
4. This is an expensive process, as the generated concept set is detached from the concepts the model actually used, we usually have a larger concept space than required and matching which concepts were used and then computing the vanilla explanations takes multiple runs of the source model, and the use of resource intensive image diffusion and text generation models, may make it impractical for repeated and practical applications.

### Questions
Refer Weakness Point 3

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces ConLUX, a general, lightweight framework that lifts local model-agnostic explainers (LIME, Kernel SHAP, Anchors, LORE) from feature-level to concept-level without changing their core learning algorithms.

ConLUX replaces feature predicates with concept predicates, performs sampling/perturbation in the concept space, and uses large pretrained models (LLMs or diffusion) as a concept-to-feature mapper to recover the perturbed inputs.

Extensive experiments across BERT/DeepSeek-V3 (text), YOLOv8/ViT/ResNet-50 (vision), and Qwen2.5-VL (multimodal) show consistent fidelity gains over vanilla baselines and concept-based SOTAs.

Human studies further indicate that ConLUX’s concept-level rules and counterfactuals help users better anticipate model behavior.

Overall, the paper makes a strong and timely contribution to concept-based interpretability. Its reliance on large pretrained models situates ConLUX as a user-oriented explanatory interface that emphasizes accessibility and semantic richness over strict analytical understanding, which is a reasonable design trade-off for real-world explainability.

### Strengths
1. Unified framework enabling plug-and-play upgrades of standard local explainers to concept level.

2. The idea of perturbing in concept space then map back to input space is appealing.

3. Supports attribution, sufficient conditions, and counterfactuals in a single pipeline.

4. The ConLUX augmented methods significantly improve over their traditional counterfarts across different metrics and datasets.

### Weaknesses
1. Fidelity hinges on LLM/diffusion quality and checker reliability; domain shift or safety filters may bias perturbations despite high average accuracy.

2. The comprehensiveness and stability of extracted concepts heavily depend on the capability and prompt quality of the upstream extractor. As a result, the concept pool may omit critical semantics or produce redundant concepts, leading to incomplete or noisy concept representations. While the authors report robustness to prompt variants, a more systematic evaluation of coverage and consistency would strengthen the claim.

3. Sampling thousands of concept-level perturbations with large models may be expensive; the paper could quantify runtime and budget more explicitly

### Questions
1. there are different concept levels, can users specify hierarchical concepts and receive multiresolution explanations?

2. How do you guard against mode collapse or semantic drift in the concept->feature mapper beyond accuracy checks?

### Soundness
3

### Presentation
3

### Contribution
4
