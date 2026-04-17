# Learning Data-Efficient and Generalizable Neural Operators via Fundamental Physics Knowledge

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Recent advances in scientific machine learning (SciML) have enabled neural operators (NOs) to serve as powerful surrogates for modeling the dynamic evolution of physical systems governed by partial differential equations (PDEs). While existing approaches focus primarily on learning simulations from the target PDE, they often overlook more fundamental physical principles underlying these equations. Inspired by how numerical solvers are compatible with simulations of different settings of PDEs, we propose a multiphysics training framework that jointly learns from both the original PDEs and their simplified basic forms. Our framework enhances data efficiency, reduces predictive errors, and improves out-of-distribution (OOD) generalization, particularly in scenarios involving shifts of physical parameters and synthetic-to-real transfer. Our method is architecture-agnostic and demonstrates consistent improvements in normalized root mean square error (nRMSE) across a wide range of 1D/2D/3D PDE problems. Through extensive experiments, we show that explicit incorporation of fundamental physics knowledge significantly strengthens the generalization ability of neural operators.
We will release models and codes at https://sites.google.com/view/sciml-fundemental-pde.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a data-efficient learning framework for PDE dynamics forecasting by jointly learning from both the original PDEs and their simplified basic forms. Extensive experiments on a wide range of 1D/2D/3D PDE problems demonstrates the effectiveness of the proposed framework.

### Strengths
- The proposed method is well-motivated. The authors provide a critical observation by evaluating existing SciML foundation models. They find a strong correlation between a model's performance on the original PDE and its performance on the fundamental components of that PDE (e.g., pure diffusion for a reaction-diffusion system). However, the absolute error on these basic terms remains high, indicating that even powerful models lack a robust understanding of the foundational physics, which motivates the need for explicit training on these concepts.
- Methodological Innovation:​​ The paper proposes a simple yet effective multiphysics training framework. It first derive a "basic form" from the original PDE by retaining terms governing essential dynamics and removing terms that cause computational stiffness or high cost. The model is trained on a composite dataset from simulations of both the original PDE and the basic form.

### Weaknesses
- Heuristic Nature of Decomposition:​​ The process for selecting terms for the "basic form," while physically intuitive, remains heuristic. A more formalized principle or an ablation study discussing the impact of alternative decompositions more prominently would strengthen the methodology. 
- Inadequate Mechanistic Explanation for the Efficacy of Basic Form Data​: A significant weakness of the paper lies in its insufficient exploration of the underlying mechanisms by which the "basic form" data aids the learning of the original PDE. The attribution of performance gains solely to the incorporation of "fundamental physics knowledge" is a high-level concept that lacks granularity. A more rigorous analysis is required to dissect how the basic form data contributes. A possible explanation is that data from the basic form may provide more diverse initial conditions. Can the data from the basic form be replaced with an equivalent amount of original PDE data? Although this would incur greater simulation costs, it would help clarify the specific ways in which data from the basic form aids the model in learning the original PDE.

### Questions
See weaknesses.

### Soundness
2

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
5

### Summary
The paper proposes a multiphysics training scheme that jointly learns from full PDE simulations and their decomposed “basic forms” the authors argue that this injects “fundamental physics knowledge” into neural operators (NOs) and improves data efficiency and OOD generalization.  The key contribution lies in identifying and leveraging "fundamental physics knowledge" through decomposed basic PDE forms.  This has not  been explored extensively in the neural operator literature.  
The authors target two central SciML issues, 1. data hunger and 2. poor OOD transfer, for operator learning across 1D/2D/3D PDEs (Diffusion-Reaction, Navier–Stokes, Kuramoto–Sivashinsky, plus ScalarFlow). 
Formulations of PDEs and “basic forms” are standard and correctly specified
The paper is generally well written with helpful overview figures (Fig. 3 pipeline; Fig. 4 gallery of PDEs/basic forms) and plots tying simulation cost to nRMSE. Implementation, data splits, and training schedules are placed in appendices. 
Minor typos remain but do not impede readability.
Central claims are supported with experiments and results that are a bit light on content
The validation of physics (central theme) is light given no explicit checks on mass/energy conservations.  Another issue is the heuristic treatment of the fundamental physics term.

### Strengths
The authors target two central SciML issues, 1. data hunger and 2. poor OOD transfer, for operator learning across 1D/2D/3D PDEs (Diffusion-Reaction, Navier–Stokes, Kuramoto–Sivashinsky, plus ScalarFlow). 
The key contribution lies in identifying and leveraging "fundamental physics knowledge" through decomposed basic PDE forms.  This has not  been explored extensively in the neural operator literature.  
Proposed benefits such as:
Data efficiency, Long horizon stability , OOD generalization , are all desirable.

### Weaknesses
The term "fundamental physics knowledge" is somewhat vague and could be better defined
Section 3.1 could be more systematic in explaining the decomposition principles
Some notation inconsistencies (e.g., switching between v and u for solutions

Missing error bars in main results (added later in appendix)
there is limited statistical analysis
The ScalarFlow experiment (Section 4.5) is somewhat disconnected and brief
Claims about "fundamental physics knowledge" being key are not fully validated (could be just multi-task regularization)
There is a lack of theoretical insight, no formal explanation of why the approach works beyond intuition is provided.
The decomposition rules in Section 3.1.1 seem ad-hoc without principled justification

Evaluation is very basic, results only compare against vanilla baseline and spatiotemporal downsampling
There is no comparison/discussion with other data-efficient methods or recent foundation models
Real-world evaluation is very limited (only ScalarFlow)
Inconsistent terminology: "Fundamental physics knowledge" vs "basic forms" used interchangeably
Missing details: How are the mixture ratios exactly determined? Training time comparisons?
Limited discussion: When would this approach fail? What about PDEs that don't decompose nicely?
Presentation issues: Some figures (especially in appendix) are too small to read clearly
Scalability concerns: All experiments on relatively small-scale problems

Grammer-Typos
line 483: and outha ha h-of-distribution generalization.

### Questions
See weakness section , addressing those would be good
recommend to:
Better justify the decomposition principles
Provide theoretical analysis or at least intuition for why the approach works
Compare with more baselines
Discuss limitations and failure cases more thoroughly

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
4

### Summary
The paper discusses basic forms of standard PDEs. It finds out that if put half of the data generation efforts into generating basic form of PDE, the performance and generalization will generally improves. The paper test on diffusion-reaction, 2d NS, 3d NS, and KS equation. The improvement on 2D NS is the most significant.

### Strengths
- The paper studies an important but often overlooked probably on efficient data generation for neural operators.
- When comparing against the baseline, the paper controls the overall run time budget.
- The experiment show consistent improvement.

### Weaknesses
- It is not always obviously which is the best basic form to each target equation. I don't think there is a canonical choice, and the performance depends on the choise of augmented basic PDE. For example on 2D NS the treatment is more significant, but not as much on KS. It would be better to add some ablations to justify the choice.
- Overall I like this paper but I don't like the storytelling. The main message should be "it is helpful to generate additional data in simpler form". It is a bit speculative to claim about "fundamental physics knowledge". In my opinion it can be awkward to say diffusion is the "fundamental physics" to diffusion reaction equation, or convection is "fundamental physics" to Navier Stokes equation.
- Line 108, the authors use Spearman correlation which is defined for ordinal (rank) correlation. Instead it would be better to use Pearson correlation as the rank does not matter here.

### Questions
- In the experiment, we need to be a bit careful how we measure the simulation cost (Table 1). In practice, the runtime of numerical solvers depends on the choices of gridsize, timestep, and numerical tolerance etc. Here how the simulation cost is measured? 
- If we instead add low fidelity data (with lower gridsize) with the same equation, would the performance improve?
- How about adding smaller RE on NS, it would requires low simulation cost too?
In general it would be helpful to add a bit more discussion on the simulation cost.

### Soundness
4

### Presentation
3

### Contribution
3
