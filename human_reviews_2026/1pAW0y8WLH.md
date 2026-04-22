# DrugTrail: Interpretable Drug Discovery via Structured Reasoning and Druggability‑Tailored Preference Optimization

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Machine learning promises to revolutionize drug discovery, but its "black-box" nature and narrow focus limit adoption by experts. While Large Language Models (LLMs) offer a path forward with their broad knowledge and interactivity, existing methods remain data-intensive and lack transparent reasoning. To address these issues, we present DrugTrail, an LLM-based framework for explainable drug discovery that integrates structured reasoning trajectories with a Druggability‑Tailored Preference Optimization (DTPO) strategy. It not only introduces structured reasoning traces to articulate the "how" and "why" behind its conclusions but also serve to guide task-specific reasoning pathways within the LLM's vast knowledge space, thereby enhancing its interpretability and reliability of its final outputs. Furthermore, based on the fact that optimizing for binding affinity alone does not equate to optimizing for druggability, DTPO explicitly moves beyond single-metric optimization and opens up a broader search space that balances affinity with other essential factors. Extensive experiments demonstrate the effectiveness of our approach and its generalizability to a wider range of biomolecular optimization domains, bridging the gap between LLM reasoning capabilities and trustworthy AI-assisted drug discovery.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors present DrugTrail, an LLM-based method for computational drug discovery. DrugTrail is a novel approach to an ongoing challenge in drug discovery: the de novo generation of drug-like molecules. Through a RL-based process to optimize several metrics (beyond binding affinity), DrugTrail can effectively generate small molecules/ ligands for binding proteins. This is an interesting, interpretable approach which has several steps to incorporate and align with biochemical domain knowledge.

Overall, this is a nice paper, and I will recommend an acceptance. However, the authors should provide better explanations of some of the biochemical jargon (see below).

### Strengths
**Strong points:**

- Unique from other approaches
- Great use of relevant scientific and medicinal information for model context.
- The approach for tokenization is very sensible for the application.
- The authors clearly thought-out every step of the pipeline. I appreciate that every step is justified appropriately.
- Great figures.
- The claims follow the results.
- The background information motivates the need for DrugTrail well.

### Weaknesses
**Weak points:**
- My main criticism is that some of the biochemical jargon needs to be better or earlier explained. I know this is difficult with limited space, but it is important for understanding the paper, especially for a computational conference. 
- The explanation of biochemically relevant acronyms, such as LPSK and QED, are in the supplementary, but this makes it difficult to understand what these are when they come up earlier in the main text and corresponding figures.

### Questions
- Please go through all the acronyms and ensure they are defined before being used.
- In your paper, better define biochemical jargon. A few examples are included below, but this is not an exhaustive list.
    - "ligand" - this is also used interchangeably with "molecule" or "small molecule" without an explanation that they are synonymous
    - "druggable"
    - "binding pocket"
    - "canonical SMILES"
    - "backbone"
    - "docking"

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
DRUGTRAIL is an LLM-based framework for interpretable drug discovery that combines structured reasoning and reinforcement learning. It applies a Druggability-Tailored Preference Optimization (DTPO) scheme to balance binding affinity and drug-likeness. Trained mainly on CrossDocked2020, it outperforms baseline LLMs in docking and property metrics.

### Strengths
1. By enforcing explicit reasoning trajectories, DRUGTRAIL makes the model’s decisions transparent and closer to human medicinal-chemistry logic.
2. Demonstrates consistent improvements across interaction, chemical, and structural metrics, and shows transferability to both small- and large-molecule tasks.
3. The creation of a reasoning dataset with conflict resolution, domain consistency checks, and a thinking budget is novel and carefully designed.

### Weaknesses
1. Do the authors provide a rationale for designing the reasoning format with tags such as Characterization, Stability, etc? It is unclear why these particular dimensions were chosen.
2. Only general LLMs (eg. Qwen) are used for comparison. There is no evaluation against established drug-design systems.

### Questions
see weakness

### Soundness
3

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
4

### Summary
This paper presents DRUGTRAIL, an LLM framework for explainable drug discovery. It first uses Clinical Chemistry-Informed Reasoning (CCIR) to fine-tune a model on structured reasoning, then applies Druggability-Tailored Preference Optimization (DTPO) to optimize for multi-component rewards (ligand similarity, drug-likeness, and reasoning format) beyond simple binding affinity.

### Strengths
The work commendably tackles the critical challenges of interpretability and multi-objective optimization in drug discovery. Its core strength is the DTPO reward function, which explicitly moves beyond optimizing affinity scores by balancing structural similarity, rule-based drug-likeness, and reasoning coherence, addressing a key limitation of prior methods.

### Weaknesses
The "Reasoning Quality Reward" is purely syntactic, rewarding formatting tags rather than semantic accuracy, which undermines the interpretability claim. Furthermore, the evaluation risks circularity, as the model is rewarded for similarity to a Vina-filtered dataset and then primarily evaluated with Vina. Finally, it fails to resolve the 1D/3D contradiction: the model uses 1D sequences but makes 3D-dependent inferences (e.g., π-π stacking) and relies on 3D docking for evaluation.

### Questions
1.There is an inconsistency in the description of the reasoning dimensions. Section 2.1.1 explicitly lists "five core reasoning dimensions". However, Section 2.1.3 states the SFT data conforms to "six predefined reasoning dimensions", apparently counting the final <Answer> block as the sixth. This is confusing.

2.The SFT dataset generation (2.1.2) relies on several LLM-based filtering steps, such as Conflict Resolving and Domain Consistency which introduce unquantified biases. The "Domain Consistency" check, in particular, relies on a "small set of 'golden' reasoning trajectories" whose size and diversity are not specified.

3.By pre-filtering the reference dataset with AutoDock Vina and then rewarding Tanimoto similarity, the model is effectively trained to optimize the Vina score, which is also the primary evaluation metric (Table 1). This risks overfitting to the Vina function.

4.The "Reasoning Quality Reward" merely checks for the syntactic presence of formatting tags, not the semantic quality or logical accuracy of the reasoning content. Sometimes the model is not penalized for generating nonsensical reasoning as long as the format is correct.

5.The paper claims its 1D SMILES generation method "excludes the geometric dimension". However, pocket-ligand binding is inherently a 3D problem. Furthermore, the model's inference (e.g.,π-π stacking) and key evaluation (Vina docking score) heavily rely on 3D geometry. It’s better that explain how the model learns this implicit 3D perception solely from 1D sequences.

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
Summary:
The paper introduces DRUGTRIAL, an LLM framework for explainable drug discovery. It try to addresses the “black-box” nature of AI methods by introducing structured reasoning traces to explain how and why behind its conlcusion. The work also uses DTPO strategy to optimize not only for binding affinity but also balance multiple essential factors.

### Strengths
Pros:
- The paper introduces DRUGTRIAL, an LLM framework for explainable drug discovery.
- Extensive experiments demonstrate the effectiveness of our approach and its generalizability to a wider range of biomolecular optimization domains.

### Weaknesses
Cons:
- 3D structural analysis doesn’t seem to taken into account, which is the core for small molecule drug design from medicinal chemist viewpoint.
- Explainability is a big claim; this method is clearly not explainable but only can provide interpretable insights. Phrasing and statements regarding this are encouraged to modified.
- Vina is an old approach that is not considered very accurate. More advanced binidng affinity prediction or docking methods like Boltz2, and PSICHIC, can be considered.
* hyperparameters for multiple reward functions may be difficult to tune.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
