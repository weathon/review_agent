# Si-GT: Fast Interconnect Signal Integrity Analysis for Integrated Circuit Design via Graph Transformers

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 2, 4, 6

## Abstract
Signal integrity issues present significant challenges in modern integrated circuit (IC) design, as crosstalk-induced delay variation and transient glitches caused by capacitive coupling among interconnects can severely impact IC functional correctness. Although circuit simulators like SPICE can deliver accurate signal integrity analysis, their computational cost becomes prohibitive for large-scale designs. In this paper, we propose Si-GT, a novel transformer-based model for fast and accurate signal integrity analysis in IC interconnects. Our model elaborates three key designs: (1) virtual NET token to encode net-specific signal characteristics and serve as net-wise representation, (2) mesh pattern encoding to embed high-order mesh structures at each node while distinguishing uncoupled wire segments, and (3) intra-inter net (IIN) attention mechanism to capture structures of signal propagation path and coupling connections. To support model training and evaluation, we construct the first interconnect signal integrity dataset comprising 200k delay examples and 187k glitch examples using SPICE simulations as the golden reference. Our experiments show that our Si-GT surpasses state-of-the-art graph neural network and graph transformer baselines with substantially reduced computation compared to SPICE, offering a scalable and effective solution for interconnect signal integrity analysis in IC design verification. We release the code, model, and datasets at https://github.com/xlab-ub/Si-GT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper is trying to make better Graph transformers for modeling analog-level effects in circuits, which is quite a pressing problem. ML approaches for digital IC are developing, but for analog/mixed-signal-level interference and crosstalk, the efforts are in a nascent stage. The approach seems to outperform other graph-network like approaches for the problem, and they also make their own simulated dataset based on ground truth labels via synopsys tools.

### Strengths
Analog / mixed-signal modeling via ML models is understudied, and datasets are also hard to make and standardize. The paper takes steps towards both, and makes some effort to customize vanilla GT to make Si-GT work. Ablations seem reasonable as well.

### Weaknesses
There are some major weaknesses with this paper, stemming from setup, empirical eval, and industrial relevance.

Setup : It is not clear at all why graph transformers or for that matter graph networks are the right lens to study this kind of circuit. Yes, it can be represented as a graph, but it is a highly regular, patterned graph. One can represent an image as a grid graph, but that does not make the graph neural network ideal for images. Why is it obvious that we need to solve this problem via GT ?

Eval : The dataset is mostly simulation-level, and it's very unclear that it reflects the industrial circuits (I understand that the standards are Intel/Synopsys based, but that does not mean the random sweep generates circuits relevant to industry). Further, the paper is evaluating its own methods on its own dataset, which makes it impossible to gauge impartially. (I do agree it is kind of a chicken and egg problem, which is why selecting pre-existing datasets will ease the question of impartiality). The baselines are fairly ad hoc and "generic" GT/GNN models, which may not be suitable. The layers also look a bit deep for this problem (GNNs are well known to suffer with depth)

Relevance : In analog or digital IC, there are already pre-existing tools that can create the labels in question (indeed the paper uses them). The abstract says that this is not always feasible due to computational costs (true) but then the question becomes : what is the computational tradeoff of the papers' models vs running the tool itself ? This is not studied. We are of course ignoring the fact that even slight losses in accuracy relative to industrial standards are generally considered unacceptable.

### Questions
Please see the "weaknesses" section - in particular - the questions about industrial relevance and benchmarking to other methods.

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
5

### Summary
This work proposed a fast interconnect signal integrity analysis by graph transformers. Their work modeled crosstalk effects explicitly with aggressor-victim switching interactions and signal pattern-dependent analysis, which are missing in the current works. They verified their ideas on the artificial dataset, whose samples were generated based on two aggressors and one victim.

### Strengths
1. Good modeling of the signal integrity analysis problem. Their work modeled crosstalk effects explicitly with aggressor-victim switching interactions and signal pattern-dependent analysis, which are missing in the current works. The consideration about net-specific signal characteristics (e.g., switching direction and slew
rate) is meaningful for the prediction of crosstalk delay and glitch.
2. The experiment shows more efficiency when compared with the simulation tool, e.g., SPICE.

### Weaknesses
1. The artificial dataset deviates from the realistic circuit design. Although the authors generate a massive dataset, they rely on only three nets: two aggressors and one victim. In the integrated circuit design, even the smallest circuits have more than three nets to build a specific function. I am concerned about the usability of this method in a realistic design flow.
2. The experiments mainly compare with some classic graph learning algorithms, lacking comparison with other ML-based signal integrity analysis methods. In Section 2, the authors have a survey of related works in "ML for SI" and criticize them for not explicitly modeling signal pattern variability. But I still think this paper should be compared with the related works in some metrics, e.g., sink delay.

### Questions
1. Why don't the authors compare their work with the related works in Section 2? I think the comparison will enhance the credibility of this work.
2. The method in this work, whose goal is to reduce the computation of SPICE, may be more suitable for signal integrity analysis in the standard cell design. I suggest the authors perform experiments based on a standard cell library, e.g., ASAP7 or other commercial libraries. I wonder if the authors have plans to test their methods on this type of dataset?

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
3

### Summary
This paper proposes Si-GT, a novel Graph Transformer model for fast and accurate signal integrity (SI) analysis in integrated circuit (IC) interconnects, specifically targeting crosstalk-induced delay and glitch prediction. The main contributions are:
1. Model Design: Si-GT introduces three key components to embed physical priors into the Transformer architecture: (a) a virtual <NET> token to encode net-level signal characteristics (e.g., switching direction, slew rate); (b) Mesh Pattern Encoding (MPE) to capture local coupling structures at each node; and (c) an Intra-Inter Net (IIN) attention mechanism to explicitly model both intra-net signal propagation and inter-net capacitive coupling.
2. Dataset Construction: The authors construct a large-scale dataset with 200,200 crosstalk delay examples and 187,309 crosstalk glitch examples, using SPICE simulations as ground truth. They claim this is the first dataset dedicated to interconnect SI analysis.
3. Empirical Validation: Experiments show that Si-GT outperforms state-of-the-art GNN and Graph Transformer baselines in prediction accuracy while achieving orders-of-magnitude speedup over SPICE.

### Strengths
1. Problem Significance: The paper addresses a critical and practical challenge in modern IC design—crosstalk-induced signal integrity degradation, which directly impacts chip reliability and timing closure.
2. Physics-Informed Architecture: The proposed designs (<NET> token, MPE, IIN) are well-motivated by circuit physics and aim to inject domain-specific inductive bias into the model, moving beyond generic graph learning.
3. Potential Community Impact: If made publicly available, the dataset could serve as a valuable benchmark for future research in ML for EDA, especially for crosstalk-aware modeling.

### Weaknesses
1. Dataset Limitations and Lack of Clarity:
(1)	The paper does not state whether the dataset will be open-sourced, which is essential for reproducibility and community adoption.
(2)	The synthetic data is based on a highly simplified topology of “2 aggressors + 1 victim”, which fails to reflect the complex multi-net coupling scenarios in real VLSI layouts. In contrast, recent work like GraphCAD (ISPD '25) uses real post-routing circuits from design contests, offering far greater realism and diversity.
(3)	There is no quantitative comparison with statistics from prior datasets (e.g., number of nets, coupling density, net length distribution), weakening the claim of being “large-scale” and representative.
2. Inadequate Baseline Comparisons:
(1)	The paper omits comparison with task-specific SOTA models, notably GraphCAD (2025) and Routing-Free Crosstalk Prediction (Liang et al., 2020) , both of which directly address crosstalk modeling. This is a major oversight.
(2)	Si-GT uses a “GNN → Transformer” pipeline, yet it does not compare against generic GNN-Transformer hybrids (e.g., GraphTrans or GraphGPS variants with GNN preprocessing), making it unclear whether gains stem from architecture or the proposed inductive biases.
3. Questionable Experimental Design:
(1)	In ablation studies (Table 5), different Si-GT variants (e.g., Si-GT-GCN, Si-GT-GAT) use different GNN backbones, introducing confounding variables. A fair ablation should fix the GNN type.
(2)	The performance gains over strong baselines like GraphGPS are marginal, raising questions about the practical significance of the added complexity.
(3)	The paper adds Graphormer-style shortest-path (SP) and edge biases on top of IIN, but provides no justification for this design choice or analysis of potential redundancy/conflict between SP bias and the intra-net resistance-based bias.
4. Presentation Issues:
(1)	Figure 3 (model overview) is abstract and poorly aligned with the text, reducing clarity.
(2)	Experimental details referenced in the main text (e.g., ablation setup) are missing in the appendix, undermining reproducibility.

### Questions
1. Dataset Openness and Comparison: Will the proposed dataset be publicly released? If so, please confirm. Additionally, please provide a detailed statistical comparison (e.g., net count, coupling complexity, segment distribution) with datasets used in prior works such as Liang et al. (2020) and Liu et al. (2025).
2. Missing Baselines: Why were GraphCAD (Liu et al., 2025) and Routing-Free Crosstalk Prediction (Liang et al., 2020) not included as baselines? Please add experimental results comparing Si-GT against these task-specific SOTA models.
3. Attention Bias Design: The model combines IIN bias with Graphormer’s shortest-path (SP) and edge biases. What is the motivation for this combination? Specifically, could the SP bias conflict with \phi_{Intra}, which already encodes distance via accumulated resistance ? Please clarify through ablation or analysis.
4. Granular Ablation of IIN: The current ablation removes the entire IIN module. Please perform a finer-grained ablation by separately disabling \phi_{Intra} and \phi_{Inter} to quantify their individual contributions.
5. Architecture Fairness: Since Si-GT uses GNN features as input to the Transformer, please compare against GraphGPS or GraphTrans variants that also use GNN-preprocessed features, to isolate the benefit of your proposed components (MPE, <NET>, IIN) from the general “GNN+Transformer” paradigm.
[1] Liu et al., GraphCAD: Leveraging graph neural networks for accuracy prediction handling crosstalk-affected delays, ISPD ’25.
[12] Liang et al., Routing-Free Crosstalk Prediction, ICCAD 2020.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors addressed the challenge of signal integrity analysis in modern integrated circuits, where crosstalk-induced delay variations and glitches caused by capacitive coupling between interconnects can lead to performance degradation and functional failures. Traditional SPICE simulations, while accurate, are computationally expensive and inefficient for large-scale designs. They proposed Si-GT, a transformer-based framework that enables fast and accurate signal integrity analysis by incorporating virtual NET tokens for net-level representation, mesh pattern encoding for capturing high-order coupling structures, and an IIN attention mechanism to model both intra- and inter-net interactions.

### Strengths
* The paper is well written, easy to follow
* The authors performed various experiments with good performances.
* The problem the authors aimed to address is an important one in this field.

### Weaknesses
* The authors evaluated the model performance with fixed hyperparameters. However, it would be more informative for other AI researchers if a broader hyperparameter search space were explored and a hyperparameter sensitivity analysis were conducted to examine how each hyperparameter affects the model’s performance.
* In Figure 6, the performance of the graph transformer–based model is not clearly visible. Although this does not prevent readers from understanding the main conclusion of the experiment, there could be a better way to visualize the results.
* It appears that the experiments were conducted only once and the performance was reported based on that single run. To ensure that the model’s performance is not dependent on a specific random seed but is statistically meaningful, it is necessary to repeat the experiments multiple times and report the mean and standard deviation of the performance.

### Questions
See the 'weakness' part

### Soundness
3

### Presentation
3

### Contribution
3
