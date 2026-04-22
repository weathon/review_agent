# Enhancing Molecular Property Predictions by Learning from Bond Modelling and Interactions

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
Molecule representation learning is crucial for understanding and predicting molecular properties. However, conventional atom-centric models, which treat chemical bonds merely as pairwise interactions, often overlook complex bond-level phenomena like resonance and stereoselectivity. This oversight limits their predictive accuracy for nuanced chemical behaviors. To address this limitation, we introduce \textbf{DeMol}, a dual-graph framework whose architecture is motivated by a rigorous information-theoretic analysis demonstrating the information gain from a bond-centric perspective. DeMol explicitly models molecules through parallel atom-centric and bond-centric channels. These are synergistically fused by multi-scale Double-Helix Blocks designed to learn intricate atom-atom, atom-bond, and bond-bond interactions. The framework's geometric consistency is further enhanced by a regularization term based on covalent radii to enforce chemically plausible structures. Comprehensive evaluations on diverse benchmarks, including PCQM4Mv2, OC20 IS2RE, QM9, and MoleculeNet, show that DeMol establishes a new state-of-the-art, outperforming existing methods. These results confirm the superiority of explicitly modelling bond information and interactions, paving the way for more robust and accurate molecular machine learning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The manuscript proposes DeMol, a model designed to capture atom–atom, atom–bond, and bond–bond interactions. The authors argue that one limitation of existing approaches is that they treat chemical bonds merely as pairwise interactions, overlooking more complex bond-level phenomena. Experiments were conducted on multiple benchmark datasets to demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper is well-written and easy to follow.
2. Experimental results indicate that the proposed method outperforms the baselines used in the study.

### Weaknesses
1. Limited baseline comparison. The baselines primarily focus on leveraging line graphs, while more recent methods, such as “An End-to-End Attention-Based Approach for Learning on Graphs”, demonstrate stronger performance and should be considered.
2. Incomplete result analysis. Although DeMol outperforms baselines on certain metrics, it underperforms on others. A deeper analysis of these outcomes would strengthen the paper.
3. Limited novelty. The methodological innovation of this manuscript appears incremental compared to prior work.

### Questions
How does the choice of pretraining dataset affect the model’s performance?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
DeMol, unlike most previous models, explicitly learns from both atoms and the bonds connecting them. It does this by building separate but connected graph representations for atoms and for bonds, mixing their information together throughout the network. This design helps the model better capture chemical details like bond relationships and 3D geometry, leading to state-of-the-art accuracy on standard benchmarks.

### Strengths
* Strong performance on various metrics & datasets
* Novel architecture utilizing double-helix blocks
* Utilization of multiple techniques & components, each provided with quantitative analysis.

### Weaknesses
Confusing claim:
* the claim in the paper's introduction that existing methods "often overlook complex bond-level phenomena" or "do not explicitly model bond interactions" seems misleading
* various MPNNs, GNNs and many graph transformer models already utilize edge features to encode bond information, and update node/edge accordingly.
* papers as (https://arxiv.org/abs/2410.14696) further utilize the distance between atoms(to compute LJ force), which is a complex form of edge attribute to update node features for conformation predictions.
* Thus, both the claims made in papers are misleading. The authors need to clarify their contributions.

 Qualitative analysis
* It would be extremely meaningful if the authors could provide qualitative analysis - upon the bond-centric graph embedding.
* I would like to see if the bond-centric channel/graph correctly captures and encodes resonance, aromaticity and bond conjugation in rings - as it is the main claim within the paper(high-order interactions).

Further experiments upon large-molecules.
* The dataset used here seems to bee a bit small (in the molecule size). Thus, I would like for the authors to provide additional experiments upon its generalization capacity to larger molecules. 
* For example, https://github.com/learningmatter-mit/geom .

### Questions
None

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work explores molecular representation learning with a particular focus on bond interactions, which have been largely overlooked in prior studies. By introducing an additional bond-centric graph alongside the conventional atom-centric representation, the proposed method demonstrates strong performance across multiple property prediction datasets. Extensive experiments are conducted, showing superior results compared to existing baselines.

### Strengths
- The paper represents molecules using both bond-centric and atom-centric graphs, and improves performance through well-designed attention mechanisms (e.g., structure-aware attention).
- A solid theoretical analysis is provided to justify the use of bond-centric graphs.
- The method shows strong and consistent performance across diverse benchmarks, including PCQM4Mv2, Open Catalyst 2020 (IS2RE), and QM9, and the paper includes a rigorous ablation study demonstrating the contribution of each module.

### Weaknesses
- Beyond numerical comparisons, it would be valuable to include qualitative analyses showing how the use of bond-centric graphs enables the model to capture bond-level interactions more effectively than SOTA models without bond modeling, or compared to prior bond-aware models such as LEMON and GEM.
- While the complexity analysis in the Appendix is helpful, it would strengthen the work to include a comparative complexity evaluation, including inference time, relative to other baselines

### Questions
For large-scale datasets, how exactly was pretraining conducted? It would be helpful if the paper explicitly provided the loss formulation or objective terms used during pretraining

### Soundness
3

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
This paper proposes DeMol, a dual-graph framework for molecular property prediction that jointly models atom-level and bond-level representations. By learning from both the atomic graph and its line graph, DeMol can capture expressive representation such as conjugacy relations between bonds.

### Strengths
- The paper is well-motivated by common limitation in prior work.
- The authors show the effectiveness of integrating atom and bond-centric channels both empirically and theoretically.
- DeMol demonstrates strong performance on diverse molecular benchmarks.

### Weaknesses
In general, this paper is well-written and technically sound with clear motivation. However:

- DeMol exhibits substantial conceptual overlap with [1], which also proposes line graph construction over molecules and propagates information between atom and bond-centric graphs. The methodological novelty thus appears limited, and the paper would benefit from a more explicit clarification of its distinct contributions beyond [1].
- The performance improvement seems to be less pronounced on QM9 dataset. Could the authors provide further insight on this result?

---

[1] Atomistic Line Graph Neural Network for improved materials property predictions, npj Computational Materials, 2021.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
