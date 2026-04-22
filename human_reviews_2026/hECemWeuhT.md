# DMFlow: Disordered Materials Generation by Flow Matching

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
The design of materials with tailored properties is crucial for technological progress. However, most deep generative models focus exclusively on perfectly ordered crystals, neglecting the important class of disordered materials. To address this gap, we introduce DMFlow, a generative framework specifically designed for disordered crystals. Our approach introduces a unified representation for ordered, Substitutionally Disordered (SD), and Positionally Disordered (PD) crystals, and employs a flow matching model to jointly generate all structural components. A key innovation is a Riemannian flow matching framework with spherical reparameterization, which ensures physically valid disorder weights on the probability simplex. The vector field is learned by a novel Graph Neural Network (GNN) that incorporates physical symmetries and a specialized message-passing scheme. Finally, a two-stage discretization procedure converts the continuous weights into multi-hot atomic assignments. To support research in this area, we release a benchmark containing SD, PD, and mixed structures curated from the Crystallography Open Database. Experiments on Crystal Structure Prediction (CSP) and De Novo Generation (DNG) tasks demonstrate that DMFlow significantly outperforms state-of-the-art baselines adapted from ordered crystal generation. We hope our work provides a foundation for the AI-driven discovery of disordered materials.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
DMFlow proposes an architecture specifically for disordered materials and curates multiple subsets of the Crystallography Open Database (COD) with disordered organic materials. Two types of disorder are considered: substitutional and positional disorder. The architecture includes a spherical geometric embedding of the substitutional disorder, positional disorder is a weighted contribution from two positions, and a voting technique for establishing disorder during inference.

### Strengths
- Interesting (novel) subject area given the growing interest in materials in deep learning. It also begins to attack real problems in materials science rather than the simplified "pure crystal" case.
- The methods (manifolds, voting) are well motived for the application and provide both anecdotal and numerical improvements. Stronger results for substitutional and positional disorder. Ablations against relevant alternative methods.

### Weaknesses
- The choice to focus on only binary disorder seems tailored to the dataset and not a strong representation of what disorder is possible. I would think a probability distribution over space would be a better representation of disordered atoms.
- I was surprised that the voting scheme did not seem to make major improvements over FlowMM-Prob in the substitutionally disordered (SD) case.

### Questions
- While the voting scheme is clever, it is ultimately a discretization. Is there any possibility for retaining a probabilisitic view of both substitutional and positional disorder and evaluating that?
- What are the limitations of using only two points of disorder?
- In non-disordered crystal generation we compute thermodynamic stability, is there a similar computation you can do for this case? Free energy?

### Soundness
3

### Presentation
3

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
This paper proposes DMFlow, the first deep generative framework for disordered crystals. It introduces a representation that jointly models ordered, substitutionally disordered (SD), and positionally disordered (PD) structures. Built on Riemannian flow matching with spherical reparameterization, DMFlow generate lattices, coordinates, and disorder weights simultaneously. A specialized graph neural network (GNN) captures crystallographic symmetries and multi-positional interactions. The authors also build the first benchmark dataset for disordered crystals, and experiments show that DMFlow outperforms existing models on both crystal structure prediction and de novo generation tasks.

### Strengths
* The paper tackles an **underexplored but relevant problem** in generative modeling—extending flow-based methods to **disordered crystal structures**.
* The work contributes a **new benchmark dataset** for disordered materials, which could be a useful resource for future research.

### Weaknesses
* **Relatively marginal improvement over baselines:**
  The experimental results show only small performance gains compared to **FlowMM**, suggesting that the proposed method offers limited novelty or practical advancement from a machine learning perspective. I wonder the performance compared to more recent crystal generation methods e.g. CrysBFN [1] and MatterGen [2]

* **Relatively minor machine learning contribution**
  The paper's contribution to the core machine learning field is relatively minor. It mainly adapts existing **flow matching frameworks** without introducing new algorithms or theoretical ML insights. The transition from ordered to disordered crystal generation appears straightforward, as the model only handles **binary positional disorder (two possible positions)** and adds several representation variables.

* **Insufficient ablation studies:**
  The paper lacks comprehensive analyses to justify the necessity of each proposed component. For instance, it would be valuable to compare the **spherical reparameterization** with a simpler flow matching setup to better demonstrate its actual effectiveness.

[1] Wu H, Song Y, Gong J, et al. A Periodic Bayesian Flow for Material Generation[C]//ICLR. 2025.

[2] Zeni, C., Pinsler, R., Zügner, D., Fowler, A., Horton, M., Fu, X., ... & Xie, T. (2025). A generative model for inorganic materials design. Nature, 639(8055), 624-632.

### Questions
1. The evaluation protocol used for DMFlow appears to be largely similar to that of CDVAE and other models designed for ordered materials. Given that disordered crystals have inherently different structural characteristics, it is unclear whether these evaluation metrics are fully appropriate. Could the authors justify or validate the suitability of applying ordered-material metrics to disordered systems?
2. The paper mentions that vanilla flow matching on the simplex can lead to numerical instability, which motivates the use of spherical reparameterization. Could the authors elaborate on the underlying causes of this instability and explain why the spherical mapping effectively mitigates it?

### Soundness
2

### Presentation
3

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
This paper tackles the generation of disordered materials, extending recent advances in generative models that have focused exclusively on ordered crystalline materials where atom types and positions are fully determined. The proposed approach handles two types of disorder: Substitutional Disorder (SD), where multiple atom types can occupy the same site, and Positional Disorder (PD), where atoms can occupy multiple positions. For SD, they replace one-hot atom type encodings with occupancy vectors on the simplex. For PD, they focus on the binary case, introducing weight vectors and additional fractional coordinates per atom. To process these new site-level features, they develop a novel GNN architecture and a sampling procedure that uses majority voting across five heuristics based on occupancy and weight logits. The authors introduce new datasets for disordered materials modeling and demonstrate that their approach successfully generates such materials, representing an important first step in this domain.

### Strengths
The paper is quite strong in terms of novelty, as it is the first work I am aware of that applies diffusion models to disordered materials. It is clearly written, and the planned release of the datasets represents an important step toward enabling more generative models for this type of problem.

### Weaknesses
I believe the paper would benefit from a more detailed discussion of the design of the CSP experiment (see questions below). There are also some design choices that are not fully ablated, such as the use of five different heuristics for the majority vote during sampling. It would be interesting to see how each heuristic performs individually and whether all of them are necessary. The framework also relies heavily on manually set thresholds during sampling, and it is not entirely clear how the values reported in the appendix were selected or whether alternative values were considered. Additionally, it would be valuable to understand how small changes in these thresholds affect the final performance of the model.

### Questions
- I have one question regarding the CSP experiments. It might be a naive question, but why are you considering W, the weight over the two possible coordinates, to be given and observed? Shouldn’t the model be learning it as part of the task?
- A bit related to the previous question, in the case of CSP with SPD disorder, what are the main differences between FlowMM-Prob and DMFlow? Is it in the way the weights over the two fractional coordinates are encoded? How can the difference in performance between the two methods be explained? What prevented you from using the same conditioning mechanism in both models, given that in this case the model’s output consists only of the lattice parameters and the two sets of fractional coordinates?
- Are the five heuristics in the majority vote needed? How do results change if one consider a single heuristic or a subset of that? Additionally, how sensitive is the final performance to small variations in the threshold values?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper aims to address the lack of generative models that natively handle disordered crystals—both substitutional (SD) and positional (PD)—and the absence of a public benchmark for this setting. It proposes DMFlow, a flow-matching framework with modules that tailored for disordered crystals. These modules include 1) a unified, site-centric representation for ordered/SD/PD crystals; 2) Riemannian flow matching on the simplex via spherical reparameterization to ensure valid disorder weights; 3) a symmetry-aware GNN for velocity prediction, and 4) a two-stage discretization (ordered-site detection + ensemble voting) to convert continuous weights into multi-hot assignments. The authors also build a COD-derived benchmark covering SD and mixed SD+PD (binary PD) with standard splits. On CSP, DMFlow matches FlowMM-Prob on pure SD (by design) but improves on mixed SPD. On DNG, ablations show necessity of the simplex constraint and multi-interaction modeling.

### Strengths
1. The paper proposes a unified representation for both ordered and disordered crystals. 

2. On the CSP tasks, the proposed method outperforms the baseline methods on a SPD dataset - a mixed set combining both SD and PD structures.

3. The paper contributes a new dataset for benchmarking disordered crystal generation.

### Weaknesses
1. The scope of the paper is limited. The paper only considers binary PD while leaving generality to >2 positions per site untested. This may limit applicability to broader disordered classes.

2. In the CSP experiments, the proposed DMFlow only marginally outperforms the baseline FlowMM-Prob on binary PD cases. 

3. In the DNG experiments, the authors do not compare against existing generative baselines. The authors argue that the existing methods can only generate one-hot element assignments. However, this rationale is not entirely compelling to me. On the top of my head, softmax or gumbel softmax could be adapted to yield fractional or multi-hot element assignments.

### Questions
The authors claim that existing methods can only generate one-hot element assignments which motivates the discretization module presented in Section 4.4. The discretization module seems to introduce many complex biases. Did the authors try to adapt softmax or gumbel softmax? if not, can the authors provide a justification for why these approaches are unsuitable?

### Soundness
2

### Presentation
3

### Contribution
2
