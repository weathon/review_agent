# Exploring Synthesizable Chemical Space with Iterative Pathway Refinements

- Avg Score: 5.50
- Decision: Accept (Oral)
- Scores: 8, 6, 4, 4

## Abstract
A well-known pitfall of molecular generative models is that they are not guaranteed to generate synthesizable molecules. Existing solutions for this problem often struggle to effectively navigate exponentially large combinatorial space of synthesizable molecules and suffer from poor coverage. To address this problem, we introduce ReaSyn, an iterative generative pathway refinement framework that obtains synthesizable analogs to input molecules by projecting them onto synthesizable space. Specifically, we propose a simple synthetic pathway representation that allows for generating pathways in both bottom-up and top-down traversal of synthetic trees. We design ReaSyn so that both bottom-up and top-down pathways can be sampled with a single unified autoregressive model. ReaSyn can thus iteratively refine subtrees of generated synthetic trees in a bidirectional manner. Further, we introduce a discrete flow model that refines the generated pathway at the entire pathway level with edit operations: insertion, deletion, and substitution. The iterative refinement cycle of (1) bottom-up decoding, (2) top-down decoding, and (3) holistic editing constitutes a powerful pathway reasoning strategy, allowing the model to explore the vast space of synthesizable molecules. Experimentally, ReaSyn achieves the highest reconstruction rate and pathway diversity in synthesizable molecule reconstruction and the highest optimization performance in synthesizable goal-directed molecular optimization, and significantly outperforms previous synthesizable projection methods in synthesizable hit expansion. These results highlight ReaSyn's superior ability to navigate combinatorially-large synthesizable chemical space.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
- This paper addresses the problem of synthesizable molecule generation, which aims to generate molecules that are synthetically accessible.
- The authors propose ReaSyn, an iterative generative pathway retirement framework based on reaction trees. It effectively generates both molecules and their corresponding synthetic pathway (i.e., how to synthesize the molecules).
- The results demonstrate superior performance compared to state-of-the-art methods for synthesizable molecule generation.

### Strengths
- The paper is well written and clearly structured.
- It overcomes existing difficulties in tree-structured synthetic pathway generation, which are typically tackled using bottom-up or top-down approach. By integrating bottom-up decoding, top-down decoding and Edit bridge (recently propsoed), the proposed model allows to iterative generation and refinement of the generated pathways. I think this integration is a novel contribution of this paper, even though the individual components of ReaSyn are known and commonly used in pathway generation.
- The results clearly show that ReaSyn outperforms state-of-the-art models across various evaluation metrics.

### Weaknesses
The generation cycle consists of 3 steps: bottom-up generation, top-down generation, and Edit bridge for pathway retirement, while the compared models use only a single step. It would be helpful to discuss how much slower ReaSyn is in terms of training and inference time (generation of pathway), compared to other methods.

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ReaSyn, a framework for synthesizable projection that turns an arbitrary target molecule into one or more synthesizable analogs together with explicit synthetic pathways. The key ideas are:

1. a sequential pathway representation that encodes both molecules (via SMILES blocks) and reaction types in a single token vocabulary, enabling bottom-up (BU) and top-down (TD) traversals of a synthesis tree using the same autoregressive Transformer;

2. a bidirectional iterative cycle that alternates BU sampling (from building blocks up) with TD subtree re-generation (from the product down), propagating edits throughout the pathway; and

3. Edit Bridge, a discrete-flow editor over full pathway sequences (insertion/deletion/substitution) trained with offline alignments to holistically refine the pathway beyond local autoregressive edits.

On synthesizable molecule reconstruction across Enamine, ChEMBL, and a harder ZINC250k-augmented setting, ReaSyn reports markedly higher reconstruction rates, similarities, and diversities than prior synthesizable-space baselines. When plugged into Graph-GA as a projection/mutation step, ReaSyn maintains strong goal-directed optimization performance while ensuring synthesizability, and it outperforms prior synthesis-aware methods on TDC oracles, an sEH proxy (with improved SA/QED and AiZynth success), and JNK3 hit expansion (higher analog, improve, and success rates).

### Strengths
Unification of BU and TD in one policy. Clever direction control (first-token bias) plus balanced token-type loss yields a single model capable of both traversals; the iterative BU↔TD cycle is a natural way to propagate local edits through the pathway.

Simpler, more expressive representation. Replacing hierarchical, fingerprint-based encodings with SMILES blocks and reaction tokens reduces architectural complexity and avoids fingerprint information loss/sparsity.

Holistic refinement with Edit Bridge. A discrete-flow editor that operates at the full pathway level complements autoregressive edits and demonstrably improves reconstruction/diversity in ablations.

Strong empirical results across tasks. Large gains on reconstruction (including OOD ZINC250k building-block expansion), competitive or better optimization on TDC oracles while keeping synthesizability, improved SA/QED/AiZynth on sEH, and substantially better hit-expansion metrics for JNK3.

Thoughtful ablations. Clear attribution for (i) bidirectional iteration vs. single-direction, (ii) Edit Bridge vs. none, and (iii) representation differences vs. a close BU baseline with a near-identical Transformer.

Practical orientation. Uses a fixed reaction rule set and purchasable building-block catalog, aligning evaluation with real synthesis constraints; integrates cleanly as a projection module in standard optimization loops.

### Weaknesses
Compute transparency. The paper mentions a large offline alignment corpus for Edit Bridge and multi-stage decoding/beam search, but lacks wall-clock, FLOPs/tokens, and beam/budget controls, making fairness vs. baselines (and scaling laws) hard to judge.

Search-policy details. The TD subtree choice is uniform random over blocks; more informed selection might improve efficiency/quality. Ablations over the number of BU↔TD iterations vs. Edit Bridge steps at matched compute are limited.

Data hygiene & leakage. Reconstruction uses molecules drawn from the same vendor catalogs/rule families that likely seed training. Clear train/val/test separation for pathway patterns (and for the offline $(p_0,p_1)$ edit pairs) is under-specified.

Metric sensitivity. Similarity relies mainly on Morgan-Tanimoto; scaffold/pharmacophore scores are included in one benchmark, but systematic metric sensitivity and rank correlations across tasks are not fully explored.

### Questions
1. Catalog & rule portability. How does performance shift if (a) the building-block catalog changes (e.g., a newer Enamine snapshot or a different vendor), and (b) the reaction set is expanded/altered? Any zero-shot results with unseen rules?

2. Edit Bridge hygiene. How are $(p_0,p_1)$ pairs generated to prevent leakage from evaluation targets? Do you withhold all evaluation molecules/routes from the edit-alignment dataset?

3. Route executability. Beyond AiZynth success, can you provide route-level checks (e.g., selectivity flags, incompatible functional groups, protecting-group needs) or human/chemist audits on a subset of proposed pathways?

4. Policy control. Did you try non-uniform TD block selection (e.g., uncertainty or mismatch heuristics) and adaptive stopping for the BU↔TD loop? Any gains in sample efficiency?

5. Scaffolds vs. properties. For optimization/hit expansion, how does performance trade off between scaffold retention and property improvement as you vary the similarity threshold?

6. Ablations at fixed cost. If you fix total decoding steps, how do (i) more BU↔TD iterations vs. (ii) more Edit Bridge steps vs. (iii) larger beam affect outcomes?

7. Failure taxonomy. What are the most common failure modes (e.g., rule conflicts, unreachable leaves, unstable intermediates), and how often does Edit Bridge repair them vs. induce new ones?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ReaSyn, an iterative, bidirectional pathway–generation framework that combines (i) bottom-up decoding, (ii) top-down decoding, and (iii) a holistic editing step (“Edit Bridge”) to project molecules into a synthesizable space and improve coverage on reconstruction, goal-directed optimization, and hit expansion. The technical idea is sound and the empirical results are strong, but parts of the presentation make the method look more general than it actually is. If the authors clarify the core contribution, the experimental setup, and the fairness of the comparisons, I would support acceptance.

### Strengths
- A novel way of combining top-down, bottom-up, and holistic edit steps in one framework (effectively three networks working together).
- Effective test-time scaling: adding TD and the Edit Bridge gives non-trivial gains.
- Strong results on the harder, expanded-stock setting.

### Weaknesses
- Misleading Figure 1 and Table 1. The chemical space is **not** ZINC; ZINC is used to augment the building-block set. As written, the figures can easily be read as “reconstructing ZINC,” which is inaccurate. Please rename/reword.
- Time cost in Figure 4. The iterative pipeline appears pretty costly, especially for non-Enamine compounds. Please report the inference time for SynFormer in the figure/table.
- Fairness of comparison. ReaSyn’s gains partly come from using three networks and scaling at inference. Comparing this directly to single-pass baselines is somewhat unfair. The paper should clearly state the main contribution and separate the gains from architecture vs. extra inference compute.

### Questions
N/A

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper uses an autoregressive models to construct synthesis trees for molecules on bottom up or top down order. this is then used to perform molecular design effectively.

### Strengths
- strong empirical results, extensive validation
- interesting use of edit bridge model
- great use of BU and TD combined model

### Weaknesses
1) The edit bridge model should be explained in more detail, both in theory and also how practically the implementation is done.

2) Typo: In Section 4 I think the building block space should be larger than the reaction space?

3) The contribution of the paper is methodically good, but unfortunately, the paper currently ignores most of the pioneering work in the area. This needs to be rectified before a higher score can be assigned.

Missing citations to prior work:

Autoregressive Chemical Language Models that process and generate SMILES were introduced in 2017 in 
https://arxiv.org/abs/1701.01329 
Interestingly, this paper also features a discussion around reaction driven design - the field has seems to have gone full circle, but the problems have been known already ~10 years ago :)

Synthesizability Projection was introduced by Bradshaw https://arxiv.org/pdf/2012.11522 see appendix D3

Modern AI-driven Synthesis Planning was introduced in https://www.nature.com/articles/nature25978 (2018) not in 2019.

Most TDC oracles used in the paper were not introduced by TDC but taken from Guacamol https://doi.org/10.1021/acs.jcim.8b00839 originally developed at BenevolentAI - please cite the original work.

for Morgan Fingerprints, I'd suggest to cite https://pubs.acs.org/doi/10.1021/ci100050t as well

### Questions
- what is the validity rate for the generated SMILES building blocks?

### Soundness
4

### Presentation
3

### Contribution
3
