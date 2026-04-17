# Efficient Credal Prediction through Decalibration

- Decision: Accept (Poster)
- Scores: 4, 8, 8, 6

## Abstract
A reliable representation of uncertainty is essential for the application of modern machine learning methods in safety-critical settings. In this regard, the use of credal sets (i.e., convex sets of probability distributions) has recently been proposed as a suitable approach to representing epistemic uncertainty. However, as with other approaches to epistemic uncertainty, training credal predictors is computationally complex and usually involves (re-)training an ensemble of models. The resulting computational complexity prevents their adoption for complex models such as  foundation models and multi-modal systems. To address this problem, we propose an efficient method for credal prediction that is grounded in the notion of relative likelihood and inspired by techniques for the calibration of probabilistic classifiers. For each class label, our method predicts a range of plausible probabilities in the form of an interval. To produce the lower and upper bounds of these intervals, we propose a technique that we refer to as decalibration. Extensive experiments show that our method yields credal sets with strong performance across diverse tasks, including coverage–efficiency evaluation, out-of-distribution detection, and in-context learning. Notably, we demonstrate credal prediction on models such as TabPFN and CLIP—architectures for which the construction of credal sets was previously infeasible.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this work, the authors present a post-hoc approach for generating the credal prediction by using class-wise plausible probability intervals. This is achieved by perturbing a trained model’s logits under a global likelihood-ratio budget, thereby exploring less-likely yet still plausible predictions without retraining. Multiple experiments from different perspectives are conducted.

### Strengths
1. The paper is relatively well-structured and easy to follow.

2. Although the work is built on recent work--the likelihood-based notion of plausibility (Löhr et al., 2025), the motivation and technical routes are different. It is novel and interesting to me.

3. Mathematical proofs for the relative propositions are provided.

4. Multiple experiments are performed.

### Weaknesses
1. *Extensive experiments show that our method yields credal sets with strong coverage and efficiency and performs well on out-of-distribution detection tasks.* The main empirical claim seems a bit misleading. As from the OOD detection benchmarks, e.g., in Table 5, the EffEct only performs reasonably when \alpha is close to 1, e.g., $\alpha$ = 0.95 (still visibly lower than the other baselines). If we fix to use these values of $\alpha$, how would one conclude that the EffCre has strong coverage and efficiency?

2. The practicality of evaluating efficiency and coverage is limited. A key difficulty in supervised learning is the absence of ground truth for test instances. In addition, this work lacks theoretical guarantees for the coverage and does not provide a clear recipe for choosing the parameter $\alpha$.

3. In classification tasks, the prediction performance, e.g., test accuracy and calibration performance (expected calibration error), also matters. The performance of this approach in this matter remains unclear. As well as how $\alpha$ will influnce the prediction performance.

4. The paper highlights its good performance in epistemic uncertainty estimation for OOD detection using a single model. To support this claim, it would be valuable to include comparisons with other single-model-based epistemic uncertainty estimation methods, such as those from the evidential deep learning family or deterministic approaches. If EffCre continues to outperform these additional baselines, it would substantially strengthen the paper’s significance and credibility.

### Questions
1. How would the authors place the work? A theoretical one or practical work? What would be the potential practical use case for the approach?

2. Proposition 2.1 is a desired design principle for this method, right? It is a sufficient condition for controlling the trade-off between efficiency and coverage, not a necessary condition, am I correct?

3. The EffCre significantly reduces the training complexity via a single model. What is the inference time complexity of EffCre?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper tackles the problem of efficiently computing credal sets over predictions of probabilistic classifiers. To this end, the authors propose perturbing the logits of the model to generate upper and lower bounds of plausible class-wise probabilities while adhering to a likelihood-ratio budget. Additionally, the authors propose an efficient method to compute class-specific credal sets. The proposed method is evaluated against relevant credal prediction baselines on coverage-efficiency trade-off and out-of-distribution detection tasks.

### Strengths
The scope of the problem considered - model-agnostic credal predictions - is sizeable and will be of interest to a wide community. Additionally, the post-hoc approach that does not require any retraining as proposed in this paper will encourage its adoption as an added post-training step that can be used to quantify model's epistemic uncertainty. The proposed approach itself, to the best of my knowledge, is sound and the theoretical results seem reasonable, if not unsurprising. The experimental results suggest that the proposed approach is at least on par with the considered credal prediction baselines on the coverage-efficiency trade-off task while allowing, by design, a wide range of coverage. Finally, the efficient computation aspect allows the proposed approach to scale to large models which was previously infeasible.

### Weaknesses
Credal predictions are particularly useful in data-scarce and safety-critical domains such as healthcare where the lack of data can lead to higher epistemic uncertainty and understanding the plausible range of model predictions can help avoid catastrophic decisions. In this regard, the motivation behind the need for computationally efficient credal predictions is not very compelling. In the same vein, while the authors note such safety-critical domains in their introduction, the experimental evaluation is primarily on large benchmark image datasets. Lastly, some of the baselines seem to perform slightly better on the OOD detection task, although requiring more computational time.

Overall, I believe the strengths far outweigh the weaknesses.

### Questions
Please refer to the weaknesses section. I am mainly unconvinced about the appeal of computational efficiency in so-called *safety-critical* domains where other credal prediction methods, as evident in fig. 3 on the OOD detection task, perform better.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work deals with estimating epistemic uncertainty of predictive models through credal sets. Specifically, it deals with uncertainty estimation for large pretrained foundation models such as LLMs and VLMs, where retraining and finetuning can be exceedingly expensive. The authors address this problem by proposing a training-free method that estimates plausible intervals by modifying the inference procedure. They do so by defining a credal set over the base model’s predictions by adding a variable vector c to their logits; The credal set consists of all the values of this modified predictive distribution that are within a threshold of the maximum likelihood estimate. This credal set allows uncertainty estimation and cautious decision-making. The authors evaluate empirically on 9 domains, including CIFAR, SVHN, FMNIST, DermMNIST, ImageNet, Places365, ChaosNLI, QualityMRI, and TabArena. On these domains, they compare the proposed method with 5 credal uncertainty estimation methods. They evaluate these methods on standard credal classification metrics like coverage and efficiency, on out-of-distribution detection, on active in-context learning, and on zero-shot classification, with the latter two focusing on large pretrained models. They find that the proposed method performs on par with baselines without requiring training and that it provides informative uncertainty estimates for active learning and cautious zero-shot classification.

### Strengths
- The proposed method enables uncertainty estimation and cautious inference in large pretrained models like CLIP and TabFPN without expensive retraining & finetuning
- The proposed method is straightforward to implement yet quite effective (as demonstrated by the empirical results)
- The authors introduce credal spider plots to visualize credal sets represented as box intervals

### Weaknesses
- While the method does not require retraining, it does require the original training data or an appropriate surrogate to calculate the relative likelihood
- The presentation is unclear in places, e.g., while the credal spider plots are quite informative, a full explanation about what they represent is presented in the appendix, which makes earlier references to them (e.g., figure 1) unclear. Section 4 (Empirical results) presents a lot of information without emphasizing key parts like research questions, datasets, metrics, and baselines; instead interleaves it with details about the setup for each experiment. This section could be made easier to read by explicitly listing the research questions datasets, metrics etc., before the subsections, which may focus on more specific details

### Questions
- The proposed method shifts the logits for each class separately, and the conclusion distinguishes this from the more general coupled case. Can you give an example where the class-wise shift would be a bad approximation of the general case?
- Table 4 compares the runtimes of the proposed method and credal baselines, showing that the baselines take 10x more time than the proposed method, but it is not clear how this runtime is defined. Does it include training time or is the 10x difference only in inference time?

### Soundness
3

### Presentation
2

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
This paper proposes a novel, efficient, and model-agnostic method for credal prediction — a framework for representing epistemic uncertainty via credal sets (i.e., convex sets of probability distributions). The proposed method, termed decalibration, allows the construction of credal predictions without retraining or ensembling, which has historically been a major computational bottleneck in credal learning.

Core Idea
Instead of relying on Bayesian ensembles or multiple retrained models, decalibration works post hoc on a trained classifier. It perturbs the classifier’s logits within a relative-likelihood budget (α), thereby generating class-wise probability intervals that define a credal set.
	•	When α = 1, predictions coincide with the maximum likelihood estimator (MLE).
	•	As α decreases, the set expands, capturing more epistemic uncertainty.

This approach maintains a clear likelihood-based interpretation: the resulting predictions represent all distributions “reachable without sacrificing more than an α-fraction of training likelihood.”

Theoretical Contributions
1.	Convex feasibility and optimization properties:
	•	The set of permissible logit perturbations under a likelihood constraint is shown to be convex and compact on an identifiability hyperplane.
	•	The upper bounds of class probabilities correspond to the solution of a convex optimization problem, while lower bounds are attained at the boundary of this convex region.
2.	Analytical results for 1D (class-specific) logit shifts:
	•	Each class-wise bound can be efficiently computed through small convex programs or simple 1D searches.
	•	The resulting credal sets are nested and monotonic in α.

Empirical Contributions
	•	Extensive experiments show the method’s competitive performance on coverage–efficiency trade-offs and out-of-distribution (OOD) detection, outperforming or matching credal baselines while being orders of magnitude faster.
	•	The approach is scalable to large architectures such as TabPFN and CLIP, where traditional ensemble-based credal methods are computationally infeasible.
	•	Visual tools like credal spider plots are introduced to illustrate uncertainty across multi-class predictions.

Significance
This paper advances epistemic uncertainty quantification by providing a principled yet practical alternative to computationally intensive Bayesian or ensemble-based credal methods. Its theoretical soundness, post-hoc simplicity, and broad applicability to modern foundation models make it a potentially impactful contribution to the ICLR community.

### Strengths
•	Originality:
The paper presents a novel post-hoc approach to credal prediction — decalibration — that eliminates the need for retraining or ensemble-based inference, which have been the dominant approaches in credal and epistemic uncertainty estimation. The idea of adjusting logits within a relative-likelihood constraint is both elegant and conceptually original, bridging Bayesian epistemic reasoning with practical optimization.
	•	Technical Quality:
The theoretical exposition is mathematically sound and internally consistent. The authors derive and justify convexity properties of the credal set under the proposed perturbation scheme, ensuring interpretability and computational tractability. Analytical insights into the monotonicity of the α-parameterized likelihood bounds reinforce the approach’s rigor.
	•	Practical Relevance:
The method is computationally efficient and easily applicable to large-scale deep networks and foundation models (e.g., CLIP, TabPFN). This directly addresses a key bottleneck in existing credal learning methods, which often require retraining or expensive ensembles.
	•	Clarity and Presentation:
The paper is generally well written, with clear organization, intuitive explanations, and informative visualizations (e.g., credal spider plots). The connection between likelihood decay and epistemic expansion is articulated clearly and grounded in statistical reasoning.
	•	Experimental Strength:
The experiments are broad and diverse, covering OOD detection, reliability under label noise, and uncertainty calibration across various architectures. The results show consistent improvements in efficiency–coverage trade-offs, supporting the method’s robustness.
	•	Significance:
The proposed framework provides a principled and scalable solution to epistemic uncertainty quantification in deep learning, which is an increasingly critical research area in reliable AI. Its post-hoc and model-agnostic nature make it particularly relevant for the ICLR community focused on trust, calibration, and interpretability.

Overall Strength Summary:
The paper is a solid and meaningful contribution that balances theoretical insight with practical usability. It offers an innovative and efficient solution to credal prediction, addressing both the computational and conceptual limitations of prior approaches.

### Weaknesses
•	Limited Theoretical Depth Beyond Convexity:
While the convexity and boundedness of the credal sets are clearly demonstrated, the paper lacks deeper theoretical guarantees. For instance, there are no formal proofs of coverage calibration, robustness under data shift, or asymptotic optimality compared to Bayesian posteriors.
Suggestion: Strengthen the theoretical contribution by connecting decalibration to known uncertainty frameworks such as PAC-Bayesian bounds, conformal coverage guarantees, or distributionally robust optimization.
	•	Potential Overlap with Prior Work:
The approach resembles ideas from temperature scaling, likelihood perturbation, and distributional robustness via logit adjustment (e.g., Stutz et al., 2021; Ahuja et al., 2023). The conceptual novelty might appear incremental to readers unless clearer distinctions are drawn.
Suggestion: Explicitly clarify how decalibration differs mathematically or conceptually from logit perturbation in confidence calibration or adversarial robustness literature.
	•	Empirical Evaluation Scope:
The experimental section, though diverse, is mostly limited to classification tasks. Since credal methods are general, it remains unclear how decalibration performs in structured or regression contexts, where uncertainty has different semantics.
Suggestion: Add at least one structured prediction or regression experiment (e.g., depth estimation or tabular uncertainty).
	•	Interpretability of α-Parameter:
The α hyperparameter controlling likelihood decay is intuitive but empirically opaque. Its practical selection and relationship to epistemic uncertainty remain heuristic.
Suggestion: Provide either a principled selection rule (e.g., based on validation likelihood or calibration metrics) or a sensitivity analysis showing stable performance over α ranges.
	•	Comparative Baselines:
While results are favorable, the baselines do not include recent strong probabilistic calibration models such as Dirichlet Prior Networks or Deep Ensembles with temperature tuning. Without these, the strength of decalibration over modern uncertainty quantifiers remains somewhat uncertain.
Suggestion: Include these baselines or discuss expected trade-offs to contextualize improvements.
	•	Terminological Ambiguity (“Decalibration”):
The term decalibration may be confusing since in standard uncertainty literature, “calibration” typically denotes improving reliability, not relaxing likelihood constraints.
Suggestion: Clarify this choice early in the paper and consider an alternative framing such as “likelihood-scaling credalization” or “post-hoc credal expansion.”
	•	Computational Claims Need Quantitative Backing:
The paper asserts substantial efficiency gains (“orders of magnitude faster”) but provides limited runtime comparisons or profiling details.
Suggestion: Include explicit runtime or FLOPs analysis versus ensemble-based credal methods to substantiate this claim.

Overall Weakness Summary:

The paper is well-executed and conceptually clear, but its mathematical guarantees, empirical breadth, and comparative depth could be strengthened. Clarifying the novelty relative to prior calibration and robustness work, expanding evaluation beyond classification, and providing stronger empirical or theoretical justifications would elevate the paper’s impact and credibility.

### Questions
1.	Clarification on the Likelihood Decay Parameter (α):
	•	How should practitioners choose or interpret α in practice?
	•	Is there a connection between α and known uncertainty measures such as expected calibration error (ECE) or Bayesian posterior variance?
	•	Could α be automatically tuned using a validation objective (e.g., coverage vs. set size trade-off)?
2.	Relation to Distributionally Robust Optimization (DRO):
	•	The likelihood-based constraint defining the credal set seems conceptually close to DRO formulations (e.g., χ²-divergence or f-divergence balls).
	•	Can the authors clarify whether decalibration is theoretically equivalent to or inspired by DRO methods?
	•	If not equivalent, how does its uncertainty behavior differ under covariate shift or adversarial perturbations?
3.	Distinction from Prior Post-hoc Calibration Methods:
	•	Decalibration operates directly on logits, similar to temperature scaling, confidence calibration, and logit perturbation techniques.
	•	Could the authors explicitly explain how their formulation mathematically differs from those methods and why it better captures epistemic (not aleatoric) uncertainty?
4.	Computational Complexity Claims:
	•	The paper claims “orders of magnitude” improvement in efficiency over ensemble-based methods.
	•	Could the authors provide quantitative runtime comparisons (e.g., seconds per image or FLOPs) for fair assessment?
	•	Is the optimization step fully parallelizable, and how does performance scale with the number of classes?
5.	Generalization to Regression or Structured Outputs:
	•	The current framework appears classification-specific.
	•	Is there a theoretical extension of decalibration to continuous outputs (e.g., regression) or structured prediction (e.g., segmentation, detection)?
	•	If so, how would the likelihood constraints translate?
6.	Credal Set Visualization and Intuition:
	•	The “credal spider plots” are compelling but may lack clear interpretability for practitioners.
	•	Could the authors provide an example of how such visualization could inform human decision-making (e.g., in safety-critical applications)?
7.	Uncertainty Decomposition:
	•	Does decalibration allow separating epistemic vs. aleatoric uncertainty components?
	•	If not, could an ensemble of decalibrated models or Bayesian prior over α achieve that?
8.	Robustness under Data Shift:
	•	Have the authors tested decalibration under distributional shift scenarios (e.g., corrupted datasets, domain transfer)?
	•	If so, how does it compare to ensemble or conformal methods in terms of coverage stability?

Summary of Key Questions for Rebuttal Focus:
	1.	Theoretical connection to DRO and uncertainty calibration frameworks.
	2.	Justification and interpretation of α.
	3.	Explicit differentiation from existing logit-perturbation and calibration methods.
	4.	Quantitative validation of efficiency claims.
	5.	Potential extension beyond classification tasks.

### Soundness
3

### Presentation
3

### Contribution
3
