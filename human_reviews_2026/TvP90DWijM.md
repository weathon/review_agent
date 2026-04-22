# The Tutor-Pupil Augmentation: Enhancing Learning and Interpretability via Input Corrections

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 2, 8, 6

## Abstract
State-of-the-art machine learning models often incorporate prior knowledge or structural information about the task or data distribution. In some tasks, such knowledge may arise from first principles or emerge as simplified, learned functions that distill essential aspects of the data distribution. Model augmentation has emerged as a strategy to leverage this structured knowledge by coupling it with an auxiliary model to improve predictive performance, while preserving the interpretability offered by the simpler component. In this work, we present a new augmentation framework called the Tutor-Pupil scheme, which is designed to enhance both performance and interpretability. The Pupil is a fixed model, structurally designed for the core task, while the Tutor is a more flexible model trained to apply minimal input-level corrections to improve the Pupil’s performance on the modified input. This strict separation of roles enables the Tutor not only to compensate for the Pupil’s limitations but also to act as a diagnostic instrument. By examining the Tutor’s targeted interventions, we can identify failure modes, detect regions where the Pupil struggles to generalize, and uncover residual patterns or higher-order structures in the data not captured by the original model.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the long-standing "performance-interpretability trade-off" in machine learning—where simple, interpretable models (e.g., decision trees, physics-based formulas) lack expressive power, while complex black-box models (e.g., neural networks) sacrifice transparency. It proposes a novel Tutor-Pupil augmentation framework to resolve this trade-off by leveraging "minimal input-space corrections" rather than output adjustments, enabling both performance gains and enhanced interpretability.

### Strengths
- Unlike prior work that corrects outputs (e.g., residual networks, ensemble stacking), this paper corrects inputs, preserving the Pupil’s interpretability.

- The paper’s originality lies in redefining the paradigm of model augmentation, removing limitations of prior work, and creating novel links between data-driven learning and theoretical insight—all of which challenge long-standing practices in interpretable AI.

- The paper does not limit the Tutor-Pupil framework to a single task type but adapts it to three distinct domains—a creative extension that proves its generality.

### Weaknesses
- The paper strictly adopts a "train Pupil first, then train Tutor" serial paradigm (Pupil parameters are frozen during Tutor training but fails to explore joint training—a critical gap that limits the framework’s ability to fully leverage synergies between the two models and may amplify Pupil’s inherent flaws.

- Novelty Gap: “Input-space correction” is not new. Position the paper as “systematic, global counterfactuals for interpretable models” rather than a brand-new paradigm and provide a taxonomy table that shows how Tutor-Pupil differs from (i) local counterfactuals, (ii) adversarial examples, (iii) data-augmentation policies on objectives, constraints, and evaluation metrics.

- The paper validates the framework exclusively with interpretable Pupils (decision trees, logistic regression, ideal gas law but fails to test black-box Pupils (e.g., ResNet, Transformer)—a critical gap, as many real-world systems rely on complex models that need interpretive tools (e.g., medical image classifiers using CNNs).

### Questions
see weakness

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
4

### Summary
In this paper, the auxiliary model in the model augmentation framework is used to boost prediction accuracy and interpretability. In a Tutor-Pupil scheme, the Pupil is used to learn according to the domain-specific features, while the Tutor adding a small perturbation to the input of Pupil, is used to uncover the specific failure modes and regions of uncertainty with the Pupil's predictions. In physics-informed models, higher-order global and structural information in the data space and decision boundaries is revealed leading to better modeling of the observations and the explanation of the shortcomings of the reference dynamical processes.

### Strengths
The paper is well written and easy to follow. The authors provide a sound a good background for the problem of interpretability-complexity trade-off of the modern architectures used in ML.

The authors propose a Pupil-Tutor framework to boost the performance and interpretability of simple architecture or numerical models. This method works for physical models and can be used to modify them according to the observations for better understanding of the underlying processes. 

The authors provide other examples in the image datasets where the Tutor improves the prediction performance of a non-interpretable using this architecture.

### Weaknesses
The scope of interpretability of the Tutor is limited to our understanding of the perturbations in the data space. The results are constrained to synthetic 2d data, 3 variable time independent physical system, which are too simple to understand the potential of the proposed architecture, and logistic regression on MNIST dataset which in I believe had vague inconclusive results on diagnostics and interpretability of the failure modes of a non-interpretable model.

This might be due to the nature of the Pupil perturbations that are performed in data space and therefore the interpretability is left to human understanding of the discernible features in the data space, which is another complex task. While portrayed as a tool for interpretability, I believe this framework is yet ineffective and cannot improve interpretability as well as e.g. [Sarvmaili'24], where a set of representative samples are produced which can be used to understand the main modes of failure from training data. The authors don't report any evaluation metrics on the interpretability of the predictions especially for high dimensional datasets. A more comprehensive study can be performed e.g. with 3dshapes dataset where the main features are clearly discernible in the data space, or applying known transformations to the MNIST dataset and training the Tutor to undo said transformations could lead to more conclusive results.

[Sarvmaili'25] Data-centric Prediction Explanation via Kernelized Stein Discrepancy, Sarvmaili, Mahtab and Sajjad, Hassan and Wu, Ga, ICLR 2025

### Questions
Have the authors checked the robustness of the Pupil-Tutor framework? 

Famously, [Goodfellow'14] showed that the addition of a small but unstructured noise to the input leads for the change in the class, but not a visible difference in the image itself. In comparison addition of the noise in the latent space would likely translate to a visually significant changes in the image after decoding, but may still not be interpretable as the encoder compresses the image through entangling the meaningful features. As the results of the MNIST experiment are not very conclusive to me, can the authors explain how the Tutor perturbations are interpretable and different from a random cohesive structured blob?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces the Tutor–Pupil augmentation framework, a general approach to improving both model performance and interpretability through minimal input-level corrections. The Pupil is a fixed, interpretable or task-specialized model (e.g., decision tree, physical law, logistic regression), while the Tutor is a flexible neural network trained to apply small corrections to the inputs such that the Pupil produces more accurate outputs. This setup enables the Tutor not only to enhance predictions but also to diagnose failure modes of the Pupil by revealing where and how input perturbations correct errors. The framework is demonstrated on three diverse cases: (1) augmenting a decision tree for a toy binary classification problem, (2) refining the ideal gas law to account for non-ideal behaviors—discovering van der Waals-like corrections, and (3) improving handwritten digit classification via a VAE-based latent-space Tutor acting on a logistic regression Pupil, which also provides interpretable visual corrections. The results show notable gains in accuracy and interpretability across tasks, with meaningful parallels to symbolic regression and explainable AI.

### Strengths
The paper is conceptually novel and elegant, offering a unified and interpretable augmentation scheme applicable across interpretable and black-box models. Its theoretical framing—training a Tutor to apply minimal, semantically meaningful corrections—is both intuitive and powerful. The work’s breadth, spanning interpretable (decision trees), physics-based (ideal gas law), and data-driven (MNIST classification) settings, convincingly demonstrates generality. The analyses are rigorous, supported by visualizations. The MNIST experiment is particularly compelling: the Tutor’s subtle adjustments (e.g., closing loops or clarifying strokes) both enhance performance ( and produce human-readable explanations that outperform conventional attribution maps. The idea of deriving physically meaningful corrections from learned input perturbations is especially original and promising for scientific ML applications.

### Weaknesses
In the MNIST setting, the performance jump (91%→98.5%) could partly result from the use of a VAE-trained latent representation rather than purely from the Tutor’s corrective effect. 

Additionally, the paper could better differentiate its contributions from related ideas like counterfactual explanations, residual learning, and gradient-based input attribution methods. 

Finally, the experiments, while creative, are small-scale; a larger empirical evaluation would strengthen the claims of robustness and general utility.

### Questions
How sensitive are the Tutor’s corrections to hyperparameters such as λ (correction magnitude regularization)?

Could the framework handle non-differentiable Pupils (e.g., rule-based systems) at scale?

Does the learned correction vector generalize across data distributions or must it be retrained for each Pupil or dataset?

How can one quantify interpretability improvements beyond visual inspection (e.g., through user studies or explanation fidelity metrics)?

Could the Tutor–Pupil setup be extended to adversarial tutoring, where the Tutor exposes brittleness or bias in the Pupil rather than assisting it?

### Soundness
3

### Presentation
4

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
The paper suggests a new form of interpretability approach without compromising expressivity. It is a common practice to use a simple interpretable model as a primary model and use a second model for handling the residual error for performance reasons. But interpretability is usually lost in the complexity of the second model. 
This paper proposes an alternate. They use the second model instead to generate input perturbations such that the primary model is more accurate on the perturbed input. Assuming input perturbations are interpretable, the paper argues that their approach improves performance without hurting interpretability.

### Strengths
* Clean presentation. I enjoyed reading the paper.  
* Neat conceptual difference. Modeling input perturbation instead of residual output error is a clean conceptual differential from earlier work. 
* In the case of MNIST modeled with logistic regression, improving accuracy through augmentation and output correction could not have led to the explanations the paper demonstrated in Figure 6. MNIST dataset, although simple, supports their claim of improved accuracy and interpretability with their approach.

### Weaknesses
* Empirical validation. The validation in the paper looks preliminary. It requires validation with far more complex datasets to be taken seriously. For instance, CheXpert [1] or some WILDS [2] datasets.   
* The paper assumes input edits are model-able and interpretable. The interpretability aspect is only assumed without validation. 

References   
[1] https://www.nature.com/articles/s42256-021-00338-7
[2] https://wilds.stanford.edu/datasets/

### Questions
1. If the pupil model is complex, what's stopping the tutor from selecting meaningless perturbations? I.e., how is the interpretability of perturbations enforced?   
2. The requirement of latent space and pupil model inference with modified inputs could compromise performance due to train-test distribution for pupil model and other reasons. When do you expect far worser performance (than a non-interpretable base model) with your approach?              
3. Please elaborate your differences from counterfactual explanations.

### Soundness
3

### Presentation
4

### Contribution
2
