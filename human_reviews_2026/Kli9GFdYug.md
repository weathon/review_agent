# INTERPRETING QUANTUM CIRCUIT LEARNING WITH QPERT: A STEP TOWARD TRUSTWORTHY QUANTUM AI

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Quantum Circuit Learning (QCL) presents a promising hybrid computational framework that combines the representational capacity of parameterized quantum circuits (PQCs) with classical optimization techniques for solving machine learning problems. However, the opaque nature of QCL models limits their adoption in domains requiring transparency and accountability. In this work, we introduce quantum perturbation (QPERT), a novel perturbation-based explainability approach tailored for QCL. QPERT generates a saliency mask by quantifying the importance of input features for a given instance while preserving key quantum properties such as entanglement and superposition. We evaluate QPERT in explaining a hybrid quantum-classical architecture trained on the Iris dataset. Comparative analysis against established explainability techniques, including SHAP and LIME, highlights QPERT's effectiveness in delivering interpretable insights into quantum model behavior. Our results demonstrate the feasibility of interpretable quantum learning and offer practical guidance for integrating explainability into quantum-classical pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this work, the author's introduce QPERT, a perturbation-based explainability method for quantum circuit learning (QCL), a subclass of quantum machine learning. QPERT fills a gap in the XAI for QML literature as the first perturbation-based XAI method for QCL models. The author's compare QPERT to LIME and SHAP, two classical explainability methods.

### Strengths
Significance: QPERT is the first perturbation-based explanability method for QCL, a popular subclass of QML methods. 

Clarity: The paper's prose is lively and engaging. As a result, the paper is easy to follow and understand.

### Weaknesses
Too niche: The paper develops a specific kind of explainable AI (XAI) method for a subclass of QML methods. Other XAI methods already exist for QCL, as covered in the related work section. The paper fails to convincingly argue why QPERT is superior to these methods or significantly advances the field. As a result, the paper is likely too niche for ICLR. 

Failure to support major claims: The data presented in the paper fails to support multiple claims including, most importantly, the utility of QPERT. At the highest level, the ablation study suggests that SHAP does a better job at explaining the underlying QCL model than QPERT, thus undermining the claim that QPERT is superior because it respects the "quantumness" of the model. At lower levels, the paper makes claims like "[QPERT] capture[s] meaningful local gradients, while respecting the underlying structure of the model (Line 311-312)." This claim comes after a short presentation about how QPERT ranks the relative importance of different features. As far as I can tell, no argument is made as to why this relative ranking demonstrates how QPERT respects the underlying structure of the model. At an even lower level, numerous quantitative values are cited which are then contradicted by the figures (e.g., stating that petal width has a mean importance of 0.8 for the Virginicus class even though the reference figure has no bars that reach 0.8).  

Contradictory internal logic: The results of the ablation study suggest that LIME is no better than a baseline method for interpreting QCL models, yet the paper relies on LIME to support claims about the relevance of different input features. 

Missing experimental details: It is not possible to reproduce the paper's main results based on the information provided in the main text. For instance, under what conditions where the demonstrations and ablation studies performed? Were they run on experimental hardware? A cloud-accessed simulator? With or without noise?

### Questions
Why did you not compare QPERT to qSHAP and qLIME?

How robust is this method to the encoding scheme? What happens if I encode my data in a way that breaks a one-to-one mapping between individual qubits and data features?

Why are we using different colors in the subfigures of Figure 4? Honestly, the plots look very similar. Its hard to tell what I'm supposed to get out of these plots. It looks like QPERT is telling me that all features are treated roughly equally when classifying Setosa and Versicolor.

Figure 11b / Figure 10 has no bar that reaches .8. Are you sure the data is presented correctly? 

"For Setosa, petal length was the most influential, while Versicolor exhibited a more balanced dependence on both the petal and sepal characteristics." How am I supposed to see this? Neither of these claims appear to be supported by Figure 4. 

How does any of the presented data support QPERT's ability to "capture meaningful local gradients, while respecting the underlying quantum structure of the model?" You sort of claimed that preserving fidelity, minimizing entanglement loss, and preserving superposition would imply respecting the underlying quantum structure of the model…but you haven't talked about that at all in this section.

How am I supposed to interpret Figure 6? It is just a big grey box with a very very mild downward trend. Did QPERT actually learn much over the course of its training that it didn't essentially immediately learn?

What's the point of Figure 5? It says "Top Features (Ranked)" and it just…lists the four features.

You make a lot of claims about the behavior of the different loss components, but then you don't present any quantitative evidence that the claims are true. For example, you state that "Entanglement loss converges rapidly," without ever actually plotting entanglement loss!

Why should I care about QPERT if SHAP outperformed it in your ablation study?

"The Random Baseline, as expected, produced a near-zero AUC-Difference, validating its role as a control." But the Random Baseline is further from 0 than QPERT's AUC-Diff. So wouldn't that mean that QPERT is a good control too (and therefore a bad method)?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors consider the problem of explainability of a particular class of QML models they call Quantum Circuit Learning. They introduce a new quantum version of the PERT method, called QPERTH which involves a "saliency based optimization architecture" which perturbs instances using a learnable mask. The protocol measures the effect of perturbation. The  authors also introducde a regularized loss which also penalizes for loss of "quantum features" incl. fidelity, entanglement loss and superposition loss.
The method is implemented in a 4 qubit example and evaluated on the iris dataset.
The numerical part of the paper includes comparation to other local methods such as LIME, global suchas SHARP, investigates loss trends, includes a hyperparameter study, an AUC anaysis and and ablation study.

### Strengths
I find this contribution original in incorporating entanglement etc in the loss (although I don't know why this is done). 
In my view the storngest part of the paper are the extensive numerics which are done very well and I believe are informative, although I cannot associate an actual quantitative statement  to them. It opens a line of resaerch where "quantum properties" are preserved, and puts importance on explainability. I liked the numerics performed (albeit being somewhat restricted in terms of model and dataset).

### Weaknesses
Clarity of presentation: from my perspective, the model is not sufficiently clearly defined: it is not clear to me if this t is really a "single shot" model in which the model output is the actual measurement outcome, and not an expectation value. If this is true then it is highly unconventional and in my view not very promising (due to intrainability and extreme demand on expressivity as we now need to generate quantum functions which give very close approximations of computational basis states).
A second issue I see is that the paper adapts a classical method and does what seem like ad-hoc modification to "incorporate quantum" but I found this poorly motivated. Why should we desire these new additional loss terms?
Related to clarity issues: the measrue of entanglement is completely unclear to me. Why is this measuring entanglement?
The paper deals with explanations so I understand that demanding purely quantitative statements may be out of the question but some more quantitative goals would be much more convincing. 
The experiments are limited to just one model, and just one dataset.
Scalabilty of the approach is fully unclear to me.
More, generally, I fully agree with the authors that finding new, better explanation methods is important in ML.
However, I feel this can only be as important as the model we are talking about is accepted to be useful. 
The QCL model discussed (and I am not 100% sure how it is defined, what is the output?) is praised in the intro as offering exponential advantages.
However these models are only known to be non-simulable and none of them have actually been proven to be useful for any task of relevance. Furthermore it is well know in general they suffer from very serious trainability issues.
Consequently I feel this contribution is addressing the no 3 to no 4 most important problem in the field out of ... 3 or 4.

### Questions
(1) can you motivate why the loss you describe makes sense for explainabilty
(2) can you please explain the quantum model precisely (see prior comments)
(3) can you discuss scalability incl. training costs
(4) what is the computational cost of the evaluation of the loss on a QC and on a CC?
(5) how would you make comparative statements between other methods and yours quantitiative.
(6) (not very important) do you consider "explainability" and 'interpretability" as the same thing? I always thought explainability is a post-hoc approach, whereas interpretability is an inherent property of a model. 
(7) Do you really consider lack of interpretabiliity as "the key limitation" of QCL applications in critical domains? You do not say it explicitly but the introduction is quite suggestive.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors introduce QPERT, a perturbation-based explainability framework tailored for QCL. The core is a composite loss function with multiple components designed to explicitly preserve key quantum properties, such as fidelity, entanglement, and superposition, during the generation of a saliency mask.

### Strengths
- The paper is written with clarity. The motivation, the limitations of prior work, the architecture of QPERT, the design of its loss functions, the experimental setup, and the results are all clearly articulated. 
- The core idea of designing a quantum-aware loss function to guide the explanation generation is interesting. Formulating concepts like fidelity, entanglement, and superposition as optimizable loss terms is an innovative attempt to bridge quantum mechanics with explainable AI.

### Weaknesses
- While QPERT is designed to be a more "faithful" explainer for QCL, it is outperformed by the classical, non-quantum-aware SHAP method on the authors' own chosen metric, AUC-Difference (SHAP 0.074 vs. QPERT 0.035).
- The evaluation is conducted exclusively on the Iris dataset. This is a classic but overly simplistic "toy" dataset with low feature dimensions and a small sample size. To truly validate QPERT's value, experiments on more complex datasets (perhaps synthetic, or from fields like quantum chemistry) where quantum advantages are more pronounced are necessary.
- The loss functions used to quantify quantum properties like "entanglement" appear to be classical proxies in their implementation. For example, the entanglement loss is defined by calculating the Pearson correlation matrix of the perturbation mask elements. The paper does not provide a rigorous theoretical argument for why minimizing the correlation of a classical mask vector directly and reliably corresponds to preserving the entanglement of the quantum state itself.

### Questions
- Why was the Iris dataset chosen as the sole benchmark for this study? On such a simple dataset, how can we be confident that the quantum properties QPERT is designed to protect are actually critical factors in the model's decision-making process?
- Could you please elaborate on the design of the entanglement loss? Specifically, what is the theoretical link that justifies minimizing the Pearson correlation between elements of a classical perturbation mask as an effective strategy for preserving quantum entanglement within the QCL model?

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
4

### Summary
This paper addresses the interpretability gap in Quantum Circuit Learning (QCL), where classical post-hoc explainers such as SHAP and LIME, built on deterministic outputs and largely independent feature perturbations, clash with stochastic measurement, entanglement, and inherently probabilistic outputs, often yielding unstable or misleading attributions. The authors propose QPERT, a model-agnostic, perturbation-based framework that learns a continuous saliency mask to blend each feature with in-distribution background samples, re-encodes the perturbed instance into a quantum state, and optimizes a composite loss combining target-suppression, sparsity (L1), and quantum-aware regularizers (fidelity, entanglement, superposition) to preserve quantum properties during explanation.

### Strengths
1. Studying interpretability explicitly through fidelity/entanglement/superposition is conceptually interesting and goes beyond naïve classical perturbations.
2. Diverse and well-organized visualizations (e.g., local masks, global summaries, convergence plots) aid interpretation analysis.
3. The insertion/deletion protocol with AUC-Difference, plus sparsity, convergence, and multi-run stability analyses, is appropriate for probing explanation faithfulness and robustness.

### Weaknesses
1. The framework identifies influential input features via a learned mask, but the manuscript does not articulate a theoretical advantage over straightforward L1-regularized feature selection (e.g., conditions under which QPERT yields strictly better identifiability or faithfulness).
2. Multiple losses (fidelity, entanglement, superposition, sparsity) are introduced, yet potential negative interactions or trade-offs are not analyzed; no ablations isolate their individual and joint effects.
3. Evidence is restricted to Iris on a noise-free simulator with a small 4-qubit circuit, leaving behavior under deeper circuits, harder datasets, or realistic noise unverified.
4. QPERT learns a mask per instance with staged loss activation (e.g., L1 at 250 iters, fidelity at 500, entanglement/superposition at 750), which may be costly relative to simpler post-hoc probes; runtime/complexity and shot-budget sensitivity are not reported.
5. Several captions in Figures are not fully self-consistent or sufficiently descriptive to stand alone.

### Questions
1. In ablations, QPERT and other explainers perform very closely, and SHAP attains the best AUC-Difference. Where is QPERT’s clear advantage? Please clarify what properties (e.g., stability to shot noise, quantum-consistency, sparsity) QPERT demonstrably improves, and analyze why SHAP still leads on partial metrics.
2. You mention grid search, but the explored range seems narrow. Did you try broader searches or alternative strategies (random search, Bayesian optimization, population-based training)? Please report the search space, budgets, and sensitivity results.
3. How were SHAP and LIME configured (background/reference choices, sample sizes, kernel/neighborhood parameters) and tuned? Please ensure comparable tuning budgets to QPERT, and report fairness controls (e.g., same number of model evaluations/shots).
4. Can you provide preliminary results on a larger dataset or deeper circuits?

### Soundness
2

### Presentation
2

### Contribution
2
