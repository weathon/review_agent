# M$^{4}$olGen: Multi-Agent, Multi-Stage Molecular Generation under Precise Multi-Property Constraints

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Generating molecules that satisfy precise numeric constraints over multiple physicochemical properties is critical and challenging. Although large language models (LLMs) are expressive, they struggle with precise multi-objective control and numeric reasoning without external structure and feedback. We introduce M$^{4}$olGen, a fragment-level, retrieval-augmented, two-stage framework for molecule generation under multi-property constraints. Stage I: Prototype generation: a multi-agent reasoner performs retrieval-anchored, fragment-level edits to produce a candidate near the feasible region. Stage II: RL-based fine-grained optimization: a fragment-level optimizer trained with Group Relative Policy Optimization (GRPO) applies one- or multi-hop refinements to explicitly minimize the property errors toward our target while regulating edit complexity and deviation from the prototype. A large, automatically curated dataset with reasoning chain of fragment edits and measured property deltas underpins both stages, enabling deterministic, reproducible supervision and controllable multi-hop reasoning. Unlike prior work, our framework better reasons about molecules by leveraging fragments and supports controllable refinement toward numeric targets. Experiments on generation under three property constraints (QED, LogP, and molecular weight) show consistent gains in validity and precise satisfaction of multi-property targets, outperforming strong LLMs and graph-based algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces M4olGen, a multi-agent, multi-stage molecular generation framework for precise multi-property control. The method first retrieves property-similar molecules within fixed tolerance ranges to form prototypes, then applies fragment-level GRPO optimization to iteratively refine them toward target values. Experiments on ZINC, ChEMBL, and MOSES show clear improvements over graph-based and LLM baselines in normalized total error, with retrieval and multi-hop optimization contributing complementary gains. The work provides a structured and interpretable approach to controllable molecule generation, integrating retrieval, reasoning, and reinforcement learning.

### Strengths
Originality:
The paper proposes a novel multi-agent, multi-stage framework that combines retrieval-augmented generation with fragment-level reinforcement optimization. This hierarchical formulation is conceptually clear and effectively bridges symbolic molecular reasoning with reinforcement learning.  
Quality:
The methodology is well-structured, with a clear pipeline from retrieval to optimization. The use of GRPO at the fragment level is technically sound and well-motivated. Experimental results are consistent, showing significant improvement in NTE and property accuracy.  
Clarity:
The paper is clearly written, with well-organized figures and algorithmic explanations. Each module’ s role—retriever, reasoner, and optimizer—is intuitively described, helping readers follow the multi-stage design.  
Significance:
The approach demonstrates meaningful progress toward precise and interpretable molecular generation. By unifying retrieval and reinforcement learning, M4olGen sets a practical direction for multi-property molecular control and provides a reproducible, scalable framework for future research.

### Weaknesses
While the framework is well-motivated and empirically solid, several aspects could be strengthened. The application of GRPO in discrete molecular spaces lacks theoretical justification, and the reward design may suffer from scale imbalance across heterogeneous properties. The retrieval mechanism relies on manually fixed tolerance thresholds, which may limit generalization to out-of-distribution targets. In addition, all evaluations depend on RDKit-computed metrics, restricting applicability to synthetic properties. Finally, the paper does not discuss scalability to higher-dimensional multi-property objectives

### Questions
Q1. Reward composition and balance between multiple properties  
The paper defines the overall reward as a linear combination of property errors. Could this formulation cause imbalance when properties differ in scale (for example, MW dominating over QED or LogP)? Have the authors considered normalization, or adaptive weighting to ensure stable learning across heterogeneous property dimensions?  
Q2. Fixed tolerance parameters in the retrieval stage  
The retrieval process uses fixed tolerances (ϵ = ±0.05 for QED, ±0.5 for LogP, ±25 Da for MW). How were these thresholds determined, and do they generalize across different datasets or target ranges? Would adaptive or data-dependent tolerances improve coverage in low-density or out-of-distribution regions of the property space?  
Q3. Property-only retrieval versus structural similarity  
Currently, the retrieval stage depends solely on numeric property proximity. Have the authors examined whether integrating structural similarity metrics (e.g., Morgan fingerprints or fragment overlap) could prevent retrieving molecules that are property-similar but structurally unrealistic or chemically inconsistent?  
Q4. Validity filtering and sample efficiency in fragment optimization  
During fragment-level GRPO optimization, invalid SMILES or chemically impossible edits may occur. Beyond post-hoc validity checks, were chemical constraints (e.g., SMARTS pattern filters) used to restrict the action space? Otherwise, reward sparsity and low sample efficiency could hinder effective policy improvement in early training.  
Q5. Distributional generalization of retrieval-augmented generation  
The experiments train and evaluate on similar datasets (ZINC, ChEMBL, MOSES). Have the authors examined cross-distribution generalization (e.g., training on ZINC, testing on MOSES)? Since the retrieval mechanism anchors the model to the training distribution, could this bias generation toward in-distribution molecular patterns?  
Q6. Scalability to high-dimensional property objectives  
The current setup optimizes three continuous properties. How does GRPO performance scale when extended to six or more correlated objectives (e.g., QED, LogP, TPSA, HBA, HBD, RotBonds)? Are there adaptive weighting or variance-reduction mechanisms to prevent gradient dilution and maintain optimization stability in higher-dimensional settings?

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
This work introduces a multi-condition molecular generation method based on large language models, incorporating retrieval enhancement, prototype inference, and reinforcement learning. The approach is divided into two phases. In the first phase, a general large model (GPT-4o) is employed to conduct prototype inference based on molecular properties and retrieval enhancement. The second phase utilizes a reinforcement learning model to further optimize molecular properties.

### Strengths
This paper presents a new multi-condition molecular generation method, which has demonstrated comparable results.

### Weaknesses
- The research is insufficient, as there are simpler and more efficient methods for multi-condition molecular generation based on large language models [1]. 
- The introduction of special tokens is also redundant, as numerical values are still considered as tokens, and LLMs can generate molecules with numerical values closely approximating the specified ones even without special handling [1]. 
- The range of properties studied is limited, not covering some critical drug-related properties such as toxicity, bioactivity, and synthetic accessibility. 
- The first phase relies on commercial LLMs, with unknown retrieval efficiency, reasoning time, and cost for large-scale molecular databases.

[1] https://link.springer.com/article/10.1186/s12915-025-02200-3

### Questions
see Weaknesses

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The proposed model is a two-step system that creates new molecules by meeting specific property goals such as QED, LogP, and molecular weight. In the first step, it searches for similar molecules and edits them at the fragment level to make a good starting structure. In the second step, it uses GRPO to refine the molecule through multiple controlled edits, reducing the gap between the generated and target property values. The model was trained using GPT-4o for reasoning and a ChemDFM-8B backbone for optimization. It achieves better accuracy and consistency than strong LLM and graph-based baselines while keeping the generated molecules valid and diverse. However, it is currently limited to basic computed properties and does not yet include biological or synthesis-related objectives.

### Strengths
* The model presents two-stage design that clearly separates structure generation and numeric fine-tuning and it provides strong control over multiple properties with measurable and reproducible results.
* The novelty lies in combining LLM-based reasoning with a reinforcement learning optimizer (GRPO) that explicitly minimizes numeric errors in property space.
* The experiments include transparent experimental protocol with LLM/graph baselines under a fixed compute budget and well-defined metrics

### Weaknesses
* Although the model is novel and the results are decent, its practical value remains questionable. Property-only control is useful as a controllability benchmark, but without target or seed conditioning, or integration of ADMET and affinity objectives, its real-world utility for drug discovery is limited.

### Questions
Is there a way or have you tried to extend your model to accept a target or a seed molecule as an input for more practical, target-conditioned generation?

### Soundness
2

### Presentation
2

### Contribution
1
