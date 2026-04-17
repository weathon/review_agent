# Robust Adversarial Quantification via Conflict-Aware Evidential Deep Learning

- Decision: Accept (Poster)
- Scores: 8, 4, 2, 6, 6

## Abstract
Reliability of deep learning models is critical for deployment in high-stakes applications, where out-of-distribution or adversarial inputs may lead to detrimental outcomes. Evidential Deep Learning, an efficient paradigm for uncertainty quantification, models predictions as Dirichlet distributions of a single forward pass. However, EDL is particularly vulnerable to adversarially perturbed inputs, making overconfident errors. Conflict-aware Evidential Deep Learning (C-EDL) is a lightweight post-hoc uncertainty quantification approach that mitigates these issues, enhancing adversarial and OOD robustness without retraining. C-EDL generates diverse, task-preserving transformations per input and quantifies representational disagreement to calibrate uncertainty estimates when needed. C-EDL's conflict-aware prediction adjustment improves detection of OOD and adversarial inputs, maintaining high in-distribution accuracy and low computational overhead. Our experimental evaluation shows that C-EDL significantly outperforms state-of-the-art EDL variants and competitive baselines, achieving substantial reductions in coverage for OOD data (up to $\approx55\%$) and adversarial data (up to $\approx90\%$), across a range of datasets, attack types, and uncertainty metrics.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper addresses the sensitivity of the evidential deep learning approach to input perturbations, particularly to the adversarial ones. The gist of the proposed solution is to apply a variant of self-supervised learning criterion to the adversarial robustness use case in the evidential learning context for post-hoc calibration. Inputs are augmented in diverse ways and repeatedly fed into an evidential network. The resulting Dirichlet parameters are calibrated after training with a conflict adjustment formula that promotes consistently low uncertainty scores among the perturbed versions of the same input for all classes (Eq 4) and high scores in other cases (Eq 5).

### Strengths
This is a very solid piece of work which makes a spot-on problem definition, provides a well-grounded solution to it no matter how simple, and demonstrates the efficacy of the solution on a comprehensive set of experiments where actually everything I can think of as necessary has been already considered. The C-EDL model clearly improves the state of the art in robust implementations of EDL as visible from the main results (Table 1) as well as all the downstream side analyses.

### Weaknesses
I can follow the rationale behind Eq 5 and it makes sense. I am only not sure that this is the simplest possible way the intended goal is achieved. Extending the related part of the paper (around Lines 239-256) will contribute to its readability. I do understand that Theorem 1 clearly demonstrates that the criterion behaves in the intended way. The question here is how close the proposed solution is the simplest possible one.

### Questions
Did the authors consider alternative criteria such as an adaptation of InfoNCE, Barlow Twins, or IMAX?

Evidential learning is also being used in regression [1]. In that case, as we don't have a Dirichlet distribution, I understand that the suggested conflict adjustment criterion will not directly translate. Do the authors have any insight about which basic principles of the suggested contribution can be extended to the regression case?

[1] Amini et al., Deep evidential regression, NeurIPS, 2020

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose Conflict-aware Evidential Deep Learning (C-EDL), a post-hoc method to improve the robustness of pre-trained EDL models against OOD and adversarial inputs. The core idea is to generate multiple, semantically equivalent views of an input using label-preserving transformations, pass them through the base EDL model to get a set of evidence vectors, and then quantify the conflict or disagreement among these evidence sets. This conflict score is then used to adjust (reduce) the aggregated evidence, thereby increasing the model's expressed uncertainty for contentious inputs.

### Strengths
*   The problem of improving the adversarial robustness of EDL models is significant, as the base method is known to be vulnerable despite its efficiency.
*   The proposed post-hoc approach is a practical strength, as it can be applied to any pre-trained EDL model without requiring costly retraining, making it easier to deploy.
*   The core idea of leveraging disagreement across multiple generated views to calibrate uncertainty is intuitive and simple.
*   The experimental results presented are extensive and demonstrate large performance gains.

### Weaknesses
My main concerns are regarding the related work, the scale of the experimental evaluation, and the analysis of the method's scalability.

In terms of related work, the authors seem to have missed recent work on improving EDL robustness. Pandey et al. (ICML 2023) also addresses the shortcomings of EDL for OOD detection by introducing a novel regularizer, RED. A comparison to this closely related baseline is missing and would be necessary to properly situate C-EDL's contribution within the current state-of-the-art.

While the experiments are comprehensive across many metrics, they are limited to small-scale vision datasets like MNIST, CIFAR-10, and their variants. The claims of robustness and the general applicability of the method would be much stronger if demonstrated on large-scale, more complex datasets such as ImageNet and ideally real world datasets. 

On a related note, I would like to see a more thorough scalability analysis, investigating how runtime scales not just with the number of transformations, but also with model size and input dimensionality.

The formulation of the conflict score C, combining intra-class and inter-class measures, seems somewhat over-engineered. While the authors provide a theoretical analysis of its properties in the appendix, the paper would benefit from a clearer motivation for this specific formulation over other, perhaps simpler, measures of distributional disagreement (e.g., average Jensen-Shannon divergence). It is also unclear how a practitioner would select the set of "metamorphic transformations" for a new domain, which seems critical to the method's success but is not discussed in detail.

Finally, the paper's relationship with Test-Time Augmentation (TTA) could be made more explicit. The core idea of generating multiple transformed views of a test input is functionally equivalent to TTA. While the authors acknowledge TTA, they could better frame their contribution in this context. C-EDL can be seen as a novel aggregation strategy for TTA applied specifically to EDL models. Whereas standard TTA involves a simple averaging of output predictions C-EDL's primary novelty is in its aggregation of intermediate evidence vectors and the subsequent use of a "conflict score" to modulate the final uncertainty. The authors'  ablation study on "EDL++" (which effectively performs TTA on EDL evidence vectors without conflict adjustment) demonstrates that this conflict-aware step is critical for the method's success. Clarifying this aspect would help to more precisely specify the paper's novelty.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a post-hoc calibration method for evidential deep learning (EDL) by adjusting the predicted total evidence based on output disagreement across augmented input variants. Specifically, the method estimates prediction conflicts between metamorphically transformed inputs and scales the evidence accordingly to improve robustness against adversarial and out-of-distribution (OOD) data. The authors claim that the proposed approach enhances uncertainty reliability while maintaining high in-distribution accuracy and without requiring model retraining.

### Strengths
1. The proposed method is post-hoc and architecture-agnostic. It can be directly applied to any trained EDL model without modifying the network structure or retraining, making it highly practical and easy to integrate.
2. The approach is computationally efficient compared to full retraining or Bayesian ensemble methods, as it only involves a few forward passes with simple input transformations.

### Weaknesses
1. The paper lacks sufficient implementation details for reproducibility. For example, the specific metamorphic transformations applied to each dataset, the exact uncertainty metric used in Table 1, and the threshold definition for coverage reporting are not clearly specified. The choice and tuning process for hyperparameters (e.g., conflict weighting and decay factors) should also be explained.
2. The motivation requires further clarification. EDL naturally distinguishes between aleatoric and epistemic uncertainty in a single forward pass, yet the proposed post-hoc calibration does not discuss how this property is affected. Input augmentations may alter aleatoric uncertainty due to increased noise, so the rationale behind interpreting disagreement among augmented predictions as a reliable proxy for epistemic uncertainty should be explicitly analyzed.
3. Regarding efficiency, the method requires multiple forward passes during inference. Thus, comparisons with other multi-pass methods such as Monte Carlo Dropout or ensemble-based EDL variants would provide a more meaningful evaluation of trade-offs.
4. For OOD detection, reporting AUROC and AUPR metrics in addition to coverage would provide a clearer and more comprehensive picture of the model’s uncertainty quality and detection capability.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces C-EDL, a post-hoc uncertainty quantification method that enhances the robustness of Evidential Deep Learning (EDL) models against out-of-distribution (OOD) and adversarial inputs. C-EDL generates multiple metamorphic transformations of each input, quantifies model disagreement (conflict) via intra- and inter-class evidence variability, and adjusts the Dirichlet parameters accordingly. Extensive experiments show that C-EDL significantly reduces OOD and adversarial coverage while maintaining in-distribution (ID) accuracy.

### Strengths
1. C-EDL is simple and clear, which achieves promising results.
2. The conflict measure  $C$ is formally bounded, providing a principled basis for evidence adjustment.
3. This work provides a comprehensive evaluation.

### Weaknesses
1. Standard EDL was originally used for the most basic classification tasks. Can the proposed C-EDL be directly applied to improve the level of uncertainty quantification and performance?
2. The citation format needs to be modified to improve readability; for example, using \citep.
3. C-EDL essentially utilizes multiple views and improves the accuracy of uncertainty quantification by quantifying the conflict of these views. However, there is a lack of discussion regarding some related work, e.g., [1].
4. In Table 8, the experimental results for the optimal parameters are recommended to be bolded.
5. I found that the parameter $\\epsilon$ in Equation 4 seems to be quite important in the subsequent proof; however, the intuition and motivation behind it lack a clear discussion.
6. I suggest providing some intuitive examples to demonstrate its advantages over the baseline in quantifying uncertainty. For example, using the simplest 3-dimensional simplex.

> Ref:
> [1] Reliable Conflictive Multi-View Learning, AAAI'24.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper, “Robust Adversarial Quantification via Conflict-Aware Evidential Deep Learning (C-EDL)”, introduces a new post-hoc uncertainty quantification (UQ) method that enhances the robustness of Evidential Deep Learning (EDL) models to out-of-distribution (OOD) and adversarial inputs.

Motivation
Standard EDL provides efficient uncertainty estimation by modeling class probabilities as Dirichlet distributions in a single forward pass. However, it remains vulnerable to adversarial perturbations, often producing overconfident predictions on corrupted or unfamiliar inputs. Prior extensions (e.g., S-EDL, R-EDL, H-EDL, DA-EDL) improve OOD detection but fail to adequately address adversarial overconfidence, and most require retraining.

Proposed Method: Conflict-aware EDL (C-EDL)
C-EDL is a lightweight, post-hoc framework that can be applied to pretrained EDL models. It operates by:
	1.	Generating multiple label-preserving metamorphic transformations of each input to create diverse evidence sets.
	2.	Quantifying intra-class variability (how much evidence for a class fluctuates) and inter-class contradiction (how much competing classes are equally supported).
	3.	Combining these into a conflict score (C) that measures representational disagreement across transformations.
	4.	Using this score to downscale the Dirichlet evidence via an exponential decay rule, thereby increasing uncertainty when conflict is high and maintaining confidence when conflict is low.

Mathematically, the adjusted evidence is expressed as
$$\tilde{\alpha}_k = \bar{\alpha}_k \times \exp(-\delta C)$$
where $\bar{\alpha}_k$ is the aggregated evidence and $\delta$ is a sensitivity parameter.

The authors also provide a theoretical guarantee that the conflict measure C is bounded, monotonic, and consistent with Dempster–Shafer theory principles.

Results
	•	Evaluations across multiple ID/OOD dataset pairs (MNIST→FashionMNIST, CIFAR10→SVHN, Oxford Flowers→DeepWeeds, etc.) and attack types (L2PGD, FGSM, Salt-and-Pepper) show that C-EDL substantially improves robustness:
	•	OOD coverage reduced by up to ≈55%
	•	Adversarial coverage reduced by up to ≈90%
	•	In-distribution accuracy maintained (~99%)
	•	C-EDL variants (Meta vs MC) demonstrate that metamorphic transformations outperform Monte Carlo sampling for generating meaningful uncertainty signals.
	•	The method is threshold-agnostic and retains negligible computational overhead.

Contributions
	1.	A conflict-aware, post-hoc calibration method for EDL uncertainty estimation.
	2.	Theoretical analysis of the conflict metric’s properties.
	3.	Comprehensive experimental validation across datasets, attack types, and decision thresholds showing consistent superiority over prior EDL methods.

In summary:
C-EDL presents a principled and efficient approach to making evidential deep learning models more robust to adversarial and OOD conditions, achieving strong empirical results while maintaining efficiency and avoiding retraining.

### Strengths
Originality:
The paper introduces a conflict-aware uncertainty adjustment mechanism for evidential deep learning (EDL), which is a meaningful extension to prior post-hoc calibration and OOD robustness methods. The idea of leveraging metamorphic transformations to measure evidence disagreement is both conceptually intuitive and novel within the EDL literature.
	•	Quality:
The experimental evaluation is comprehensive, spanning multiple datasets (MNIST, CIFAR10, Flowers102, etc.) and adversarial settings (FGSM, PGD, Salt-and-Pepper). The results consistently support the central claim that C-EDL improves robustness without retraining, confirming the technical soundness of the method.
	•	Clarity:
The paper is clearly structured with a good balance between formalism and intuition. The figures effectively illustrate the mechanism and the results, particularly the conflict-based uncertainty scaling and comparative coverage plots.
	•	Significance:
C-EDL contributes to an active and relevant research direction — trustworthy and uncertainty-aware deep learning. By improving EDL robustness in a post-hoc and computationally light manner, the method holds potential practical value for real-world AI reliability and safety-critical applications.

Overall Strength:
A well-motivated, well-executed, and clearly presented contribution that meaningfully extends evidential learning toward robustness and reliability.

### Weaknesses
Limited Theoretical Depth:
While the paper provides an intuitive justification and basic boundedness proof for the conflict metric, the mathematical grounding of how conflict relates to epistemic and aleatoric uncertainty decomposition is underdeveloped. A stronger theoretical connection to Dempster–Shafer theory or Bayesian evidence accumulation would strengthen the claim of principled robustness.
	•	Overlap with Prior Work:
The contribution, though practical, feels incremental relative to recent variants such as R-EDL (Robust EDL) and H-EDL (Hierarchical EDL). The manuscript could better differentiate its novelty beyond being a post-hoc fusion of metamorphic testing and EDL evidence scaling.
	•	Evaluation Breadth:
The experiments are extensive but lacking in diversity of model architectures. Most results are presented on simple CNNs and small datasets (MNIST, CIFAR10). Including modern architectures (ResNet, ViT) would improve generalization credibility and relevance for the ICLR audience.
	•	Ablation Clarity:
While several variants (C-EDL-MC, C-EDL-Meta) are tested, ablation on key hyperparameters (e.g., transformation count, conflict scaling δ) is minimal. It’s unclear how sensitive the method is to these design choices, limiting reproducibility and interpretability.
	•	Qualitative Analysis Missing:
The paper lacks visual or interpretive examples of how conflict behaves across inputs — e.g., how uncertainty distributions change for adversarial vs. OOD samples. Such qualitative insight could strengthen readers’ understanding of what the conflict term captures.
	•	Reproducibility Gaps:
Implementation details are briefly described, but no code release or reproducibility checklist is mentioned. For a post-hoc method claiming simplicity, an open implementation would be essential to validate the reported improvements.

Overall Weakness Summary:
The paper is technically solid but conceptually incremental. Strengthening theoretical justification, expanding experiments to more challenging datasets and architectures, and including detailed ablations would substantially increase its impact and readiness for top-tier publication.

### Questions
1.	Conflict Measure Definition and Theoretical Rationale
	•	The conflict score C is central to the method. Could the authors clarify whether it formally corresponds to a measure of epistemic uncertainty (i.e., model disagreement) or a combination of epistemic and aleatoric components?
	•	How does it relate to established evidential theory constructs such as conflict mass in Dempster–Shafer theory or belief/plausibility intervals? A clearer theoretical bridge would strengthen the conceptual contribution.
2.	Choice of Transformations
	•	The metamorphic transformations used (rotation, contrast, etc.) appear heuristic.
	•	How sensitive is the method to the type and strength of these transformations?
	•	Could the authors provide an ablation showing how different transformation sets affect performance on adversarial and OOD benchmarks?
3.	Computational Cost and Efficiency
	•	The paper states that C-EDL is “lightweight” and post-hoc, but no explicit computational cost analysis is provided.
	•	How does inference time or memory footprint scale with the number of transformations?
	•	Could the authors provide a table comparing wall-clock latency versus baseline EDL and Monte Carlo dropout methods?
4.	Comparison with Recent Robust EDL Variants
	•	How does C-EDL compare to S-EDL (Shannon-EDL) and R-EDL (Regularized-EDL) on the same benchmarks?
	•	These methods also modify evidence to reduce overconfidence. Are there specific cases where C-EDL performs worse or complements them?
5.	Impact on Calibration Metrics
	•	While OOD and adversarial coverage are reported, calibration metrics such as Expected Calibration Error (ECE) or Brier score are not shown.
	•	Could the authors report these to confirm that C-EDL improves not just robustness but also calibration fidelity?
6.	Applicability Beyond Classification
	•	The approach is tailored for classification tasks. Could it extend to regression or structured prediction problems?
	•	How would the conflict term be redefined in a continuous-output evidential setting?
7.	Sensitivity to δ (Scaling Parameter)
	•	The exponential scaling parameter \delta plays a crucial role in modulating evidence.
	•	How was it chosen, and how sensitive is model performance to this hyperparameter?
	•	Would a learnable δ (e.g., via validation loss optimization) yield more consistent improvements?
8.	Interpretability of Conflict Scores
	•	Have the authors visualized conflict maps or examined which regions of an image contribute most to high conflict?
	•	This could enhance understanding of whether C-EDL is detecting semantic uncertainty or simply pixel-level perturbations.
9.	Robustness Under Distributional Shift
	•	Beyond adversarial examples, have the authors tested on natural distribution shifts (e.g., CIFAR-10-C, ImageNet-C)?
	•	This would strengthen claims of general robustness rather than adversarial robustness alone.
10.	Reproducibility and Implementation Availability
	•	Is there a plan to release code or pretrained models?
	•	Given that the paper emphasizes post-hoc simplicity, open-source availability would allow the community to verify these promising results quickly.

Summary:
The main clarifications concern the theoretical interpretation of conflict, robustness to transformation choices, and quantitative trade-offs between accuracy, calibration, and inference cost. Addressing these would substantially improve the paper’s completeness and strengthen its position for acceptance.

### Soundness
3

### Presentation
3

### Contribution
3
