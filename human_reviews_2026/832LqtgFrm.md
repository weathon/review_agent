# Learning Fast and Accurate Machine Learning Force Fields via Joint Atomic Energy and Energy Hessian Distillation

- Decision: Reject
- Scores: 4, 6, 2, 2

## Abstract
Atomistic foundation models, trained on extensive and diverse datasets, now achieve near $\textit{ab initio}$ accuracy across broad molecular and material systems while demonstrating strong transferability across chemical spaces. However, their large parameter counts result in high inference latency and large memory requirements, hindering long-time-scale molecular dynamics simulations and deployment on resource-constrained hardware. In practice, researchers in physical chemistry often focus on specific chemical subdomains, where compact specialized models with fewer parameters would be sufficient—provided they inherit appropriate inductive biases from large foundation models. This need motivates distillation techniques that compress foundation models into efficient specialized models while preserving accuracy. In this paper, we propose an architecture-agnostic distillation method: Joint Atomic Energy--Energy Hessian Distillation. This approach augments state-of-the-art Hessian supervision with atomic energy, which complements low-frequency components at minimal computational overhead ($<$0.5\%). Compared with the current state-of-the-art method, our method consistently improves energy MAE over Hessian-only distillation (averaging 48.3\% on SPICE and 6.1\% on MPtrj datasets) while achieving comparable force MAE (average improvement of 1.4\%). Ultimately, our approach reduces parameter counts by 78\%–98\%, enabling fast and deployment-friendly specialized models for targeted chemical subdomains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Leveraging the fact that the errors of energy, forces, and Hessians share the same Fourier coefficients weighted by some specific values, the authors propose a tailored distillation method for machine learning force fields. The proposed method significantly outperforms previous SOTA regarding energy MAE, and slightly reduces force MAE, with minimum overhead.

### Strengths
1. The paper is well-written;
2. The proposed method significantly outperforms previous SoTA regarding energy MAE with minimum overhead;
3. Intuitively, this method makes sense. The Hessian term only constraints the high-frequency components, leaving the low-frequency components, e.g., energy loosely constrained. Thus, the new loss penalty on the energy term can additionally constraint the low-frequency components;
4. They show that the distilled model is more stable via NVT molecular dynamics (MD) stability experiments, which is consistent with the benefit of better low-frequency constraints.

### Weaknesses
1. The biggest weakness of this work is the practicalness of the method. The method significantly improves the energy MAE, which is very good if we forget the application-side purpose. Nevertheless, as the title of the paper, it serves for the machine learning FORCE field. Force is very important for MLFFs, as it is the ultimate goal of MLFF itself, which are used for simulating molecular dynamics. Ideally, if the force prediction is highly robust, we do not even need the energy prediction, and some work did omit it [1], did not report it (see Table 3 in [2]) or significantly underperform others (see Table 2 in [3]). In fact, the energy prediction is used to derive the force by the negative gradient in most work, which can turn the model to a conservative force field, i.e., a good inductive bias. Thus, for MLFFs, one can view energy as a auxiliary term severing force prediction. It makes less sense that we predict energy well but only minor improvement for force prediction, even with some, although also minor, overhead.

2. The presentation is good in the Introduction section. However, it becomes less readable in Section 3. The authors poured out all the technical lemmata and theorems without an intuitive explanation of why and how we give each statement. I understand that there is the limitation of the main content. However, I recommend the authors indicate only the important theorems and lemmata in the main paper, leaving some technical lemmata to the appendix. Before giving a lemma, explain why we need it. Before and after giving a formal statement, explain the intuition and show the proof sketch. I recommend the authors refer to some well-written pure theoretical work, e.g., [4], to see how they make their paper highly readable (including the proof) while being theoretical heavy. That being said, my main concern is on weakness 1. If the authors can address it (decrease force MAE, or explain why performing poorly in force prediction is acceptable), I will raise my score and the readability can be tackled later in camera-ready.



[1] Hu W, Shuaibi M, Das A, et al. Forcenet: A graph neural network for large-scale quantum calculations[J]. arXiv preprint arXiv:2103.01436, 2021.

[2] Gasteiger J, Becker F, Günnemann S. Gemnet: Universal directional graph neural networks for molecules[J]. Advances in Neural Information Processing Systems, 2021, 34: 6790-6802.

[3] Musaelian A, Batzner S, Johansson A, et al. Learning local equivariant representations for large-scale atomistic dynamics[J]. Nature Communications, 2023, 14(1): 579.

[4] Raman V, Subedi U, Tewari A. Online learning with set-valued feedback[C]//The Thirty Seventh Annual Conference on Learning Theory. PMLR, 2024: 4381-4412.

### Questions
Please address weakness 1. Now that we have high-quality energy prediction, fine-tuning the force prediction based on it will help?

### Soundness
4

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a joint Atomic Energy–Energy Hessian Distillation framework for compressing large atomistic foundation models into smaller, faster specialized force-field models. The key idea is to augment Hessian-based distillation with an additional atomic-energy supervision term. Through a spectral (Fourier-domain) analysis, the authors argue that the energy, force, and Hessian objectives emphasize different frequency ranges of the energy-error spectrum ($\propto$ 1, $\omega^2$, $\omega^4$ respectively).
Combining atomic-energy (low-frequency) and Hessian (high-frequency) supervision therefore provides a more balanced constraint.

### Strengths
* Clear motivation and physical grounding: The work addresses a genuine need for distilling large foundation-model MLFFs into deployable sub-domain models.
* Clear derivation:
The Fourier-domain derivation explains how energy, force, and curvature supervision emphasize different frequency components of the error. This theoretical framing helps rationalize prior empirical results on Hessian distillation.
* Comprehensive experiments: Multiple teacher–student pairs (MACE-OFF, MACE-MP, eSEN → GemNet-dT, PaiNN, GemNet-T) and diverse datasets are covered.

### Weaknesses
1. Ambiguity around Figure 2: At first glance, I thought Figure 2 suggests that distilled students outperform the teacher, although the teacher’s bars are actually not shown. This can easily mislead readers. I also suggest the authors to explicitly plot teacher performance in Figure 2 for clearer comparison.
2. Use of real labels vs. pure distillation: The method always combines DFT-label supervision with distillation losses. Hence the apparent improvement over the teacher is unsurprising, since the student still learns from ground-truth data. I am very curious to see how the performance would change if the authors exclude DFT labels in the distillation.
3. Missing analysis of force-level distillation.
Since force corresponds to $\omega^2$ weighting—intermediate between energy (1) and Hessian ($\omega^4$)—one might expect that combining atomic energy + force distillation would capture most spectral benefits at significantly lower cost, avoiding second-order derivatives. The authors neither test nor justify why this more straightforward alternative is inferior. This omission weakens the claimed “minimal-cost yet spectrally complete” argument. In addition, if you really wish to include Hessian, why don't you simply include atomic energy + force + hessian in the distillation? Isn't this a straightforward way to do the distillation?
4. Lack of physical validation: Results focus on MAE metrics. Although there is one NVT simulation, the test is too simple and not enough.

### Questions
1. Did you test atomic energy + force distillation? If not, can you argue quantitatively that its spectral coverage is insufficient compared to atomic energy + Hessian?
2. In the Fourier proof, the zero/low-frequency modes of Hessian are shown to be not important, which is counterintuitive to me from the physics perspective. As low-frequency modes may still carry large group velocities and significantly impact the properties of the materials. I understand that this manuscript is not intended for a physics discussion, but this seems confusing to me.

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
3

### Summary
The paper propses a knowledge distillation method for foundation models which compute/estimate atomic forces and energies of molecules/solids. In order to speed up inference (for example to be able to conduct long time MD simulations), specialized smaller systems can be extracted from such large foundation models. These smaller models are then trained in a student teacher setting. Ideally these smaller models keep the accuracy of the large model (for their specific task) while yielding faster inference by several orders of magnitude. 

Concretely, the paper augments previous approaches (which only penalize the deviation in the Hessian of the student/teacher models) by introducing an additional term that penalizes the deviation in atomic energies. It is argued that the Hessian only controls higher frequency components and therefore the additional term serves to control also the low frequency error. 

The experiments show improvements in training speed over a purely Hessian regularization with the computational overhead due to the additional penalty being almost negligible. In terms of accuracy, the trained model is comparable to the purely Hessian approach.

### Strengths
The paper tackles an important topic. Combining large foundation models with knowledge distillation for specialized tasks in computational chemistry is a beautiful idea. 

The main observation of the paper is that the Hessian does not constrain the low Fourier modes and therefore it needs to be augmented. The paper presents some theoretical justification for this fact.

### Weaknesses
The results on Fourier modes are trivial and obvious. If I understand correctly, the main practical contribution is to simply add another term to the optimization objective. Overall I have doubts that the conceptual novelty of the paper is sufficient for ICLR.

### Questions
Why do you assume that the error function is periodic? Can you justify this?

I have doubts that Hessian regularization is a viable option for simulating large particle numbers, due to the quadratic complexity. Is there any way to address this?

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
This paper looks at knowledge distillation for machine learning force fields. The authors look at the previously proposed Hessian distillation and the Fourier coefficients for energy, forces, and the Hessians. They then add an additional atomic energy term for supervision during distillation. They demonstrate results by distilling from models trained on the SPICE, MPTrj, and OMat24 datasets.

### Strengths
- The frequency analysis of the errors of the different terms is interesting, though needs to be more rigorously quantified

### Weaknesses
- The work seems very incremental compared to prior work on Hessian distillation, as the main contribution is adding an additional energy term during distillation supervision. The primary results that the authors show are improvements in energy and force MAE on the SPICE and MPTrj datasets. The paper would benefit from having new examples and downstream use cases to more rigorously quantify what improvements they may be seeing. It would also be important to go beyond energy and force MAEs, as these do not always necessarily correlate to downstream performance. Additionally, it would be helpful to see if such a procedure enables new accuracy and timescales that were not possible before. 

- The authors are listing specific lemmas and theorems in Section 3.1, but it is not fully clear how these follow: the paper would benefit from providing more rigorous proofs for each statement

### Questions
- How did the authors obtain equation 16, as well as Theorem 3.4? 

- What models are being used in Figure 4? 

- Can you provide additional downstream use cases to more rigorously show the benefits of adding the extra energy term during the distillation procedure? What does it enable that was not possible before?

### Soundness
2

### Presentation
3

### Contribution
2
