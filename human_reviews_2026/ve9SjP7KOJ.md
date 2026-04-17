# Socratic Personalized Medical Teaching with Multi-Agent Simulation

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4

## Abstract
The significant gap between rising demands for clinical training and the scarcity of expert instruction poses a major challenge to medical education. With powerful capabilities in personalized guidance, Large Language Models (LLMs) offer a promising solution to bridge this gap. However, current research focuses mainly on one-on-one knowledge instruction, overlooking collaborative reasoning , a key skill for students developed in teamwork like ward rounds. To this end, we develop ClinEdu, a multi-agent pedagogical simulator with personality-driven patients and diverse student cohorts, enabling controlled testing of complex pedagogical processes and scalable generation of teaching data. Based on ClinEdu, we construct ClinTeach, a large Socratic teaching dialogue dataset that captures the complexities of group instruction. We then train MedTutor-R1, the first multimodal Socratic tutor designed for one-to-many instruction in clinical medical education. MedTutor-R1 is first instruction-tuned on our ClinTeach dataset and then optimized with reinforcement learning, using rewards derived from a three-axis rubric, covering structural fidelity, analytical quality, and clinical safety, to refine its adaptive Socratic strategies. For authentic in-situ assessment, we use simulation-based interactive evaluation that redeploys the tutor back into ClinEdu. Experimental results demonstrate that our MedTutor-R1 outperforms the base model by over 20% in average pedagogical score and is comparable to o3, while also exhibiting high adaptability in handling a scaling number of students. This promising performance underscores the effectiveness of our pedagogical simulator, ClinEdu.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work focuses on the problem of medical teaching. It proposes a multi-agent pedagogical simulator to generate a new dataset, ClinTeach, and uses this dataset to train MedTutor-R1 for medical tutoring. Several experiments demonstrate its effectiveness.

### Strengths
1. This work makes a valuable contribution to the field of medical teaching.

2. The evaluation includes both human evaluation and a scalability analysis.

### Weaknesses
1. The methodology of generating a synthetic dataset and then fine-tuning the model is not novel; the Socratic approach has already been proposed in tutoring.

2. The evaluation setup appears unfair. Fine-tuning the base model on the synthetic dataset and then testing it in the simulated environment disadvantages other methods.

3. The ablation study provides limited insights; the results mainly highlight the contribution from a reinforcement learning perspective.

4. The organization of the related work section needs improvement; it only provides plain statements without justifying the research aim or contribution of this work.

5. The manuscript is not self-contained; it lacks clear references to the Appendix, such as the evaluation metrics in Appendix B.

6. The authors provide a reproducibility statement, but it would be better if the code were released, even partially.

### Questions
Please see the Weaknesses.

### Soundness
2

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
This paper focuses on one-to-many Socratic tutoring in clinical education and proposes a multi-agent teaching system. The authors design a collaborative framework involving patient, student, tutor, expert, and safety supervisor agents, and utilize structured pedagogical thinking tags in conjunction with GRPO to enhance instructional consistency, reasoning quality, and clinical safety. The paper introduces a large-scale teaching dialogue dataset and conducts comprehensive in-scenario evaluations, including expert review and user studies. Results show that the proposed MedTutor-R1 significantly outperforms baselines in teaching strategy effectiveness, multi-student management, and medical safety.

### Strengths
This paper makes an original contribution to AI-driven medical education by introducing a multi-agent Socratic teaching framework that combines structured pedagogical reasoning with GRPO. The technical design is solid, featuring a dual-review mechanism that balances instructional depth with clinical safety. The writing is clear and well-organized, guiding readers smoothly from problem motivation to experimental validation. The evaluation is comprehensive, covering both in-scenario performance and expert/user feedback. Overall, the work is a well-executed and meaningful step toward aligning AI tutoring systems with real-world educational and safety constraints.

### Weaknesses
While the paper is technically sound, several aspects could be improved. 

First, the proposed approach mainly focuses on optimizing conversational teaching within clinical simulations but does not yet model higher-level learning dynamics such as cognitive progression or knowledge transfer, limiting its theoretical depth. 

Second, although the evaluation covers teaching consistency, group management, and safety, the results rely primarily on simulated environments without real human-teacher comparisons, which constrains external validity. 

Third, details of the reward design and safety-check rules are relatively brief, making full reproducibility difficult. 

Finally, while the writing is clear, some discussions on ablation results and mechanism interpretation remain descriptive rather than analytical. 

Extending experiments to cross-domain teaching scenarios and releasing more implementation details would substantially strengthen the paper’s robustness and real-world relevance.

### Questions
1) The proposed GRPO is described as a group-based optimization method, yet from the implementation details it appears closer to a weighted multi-objective reward rather than a genuine policy-level group optimization. Could you clarify how GRPO theoretically ensures group consistency learning rather than empirical aggregation of individual rewards?

2) The “multi-student management (MSM)” metric seems to be derived mainly from internal conversational coherence rather than measurable learning gain or knowledge transfer. How can you justify that MSM truly reflects pedagogical effectiveness rather than improved dialogue fluency?

3) The simulated student agents are assumed to represent medical learners, but their behavioral diversity and realism are unclear. If the policy distribution of these agents is narrow, wouldn’t GRPO overfit to a limited interaction pattern instead of generalizable teaching strategies?

4) In the reinforcement learning setup, safety constraints are integrated into the reward formulation. Does this coupling risk producing sparse rewards or unstable gradients during optimization, and if so, how was this mitigated in practice?

5) The evaluation relies heavily on simulation and expert scoring, with no human-teacher baselines. How can the authors ensure that the system remains robust under real-world linguistic or ethical variability? Would limited real-classroom interventions be necessary to validate external validity?

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
3

### Summary
This paper presents ClinEdu, a multi-agent simulation framework designed to model clinical education with Socratic, group-based teaching dynamics. The simulator introduces personality-driven patient agents and diverse student cohorts, enabling controlled yet realistic simulations of ward round teaching interactions. Based on the ClinEdu, the authors construct ClinTeach, a large-scale dataset (48K dialogues) of one-to-many Socratic interactions, capturing both individualized reasoning and group-level dynamics. Furthermore, using this dataset, they train MedTutor-R1, a multimodal Socratic tutor designed for adaptive clinical teaching. MedTutor-R1 undergoes: 1) Instruction tuning on ClinTeach to acquire structured reasoning schemas, and 2) RL guided by a three-axis pedagogical rubric (structural fidelity, analytical quality, and clinical safety), to optimize adaptive teaching strategies. Evaluation occurs through simulation-based interactive assessment, redeploying the model back into ClinEdu for realistic, in-situ evaluation. Results show more than 20% improvement in average pedagogical score over the base model and comparable performance to GPT-4o, with adaptability to increasing numbers of students and robust performance under diverse student archetypes.

### Strengths
This work designs a full-stack pedagogical simulation pipeline, from environment (ClinEdu) to dataset (ClinTeach) to training (MedTutor-R1) and evaluation, which covers all stages of AI tutor lifecycle development.

### Weaknesses
1) While ClinTeach is large-scale, the paper does not provide sufficient quantitative detail on data diversity, distribution across medical specialties, or dialogue length distributions. A breakdown of these would aid reproducibility and fairness evaluation. 

2) While the simulated environment is rich, it still lacks grounding in human-collected data. Without hybrid validation (e.g., fine-tuning on real ward dialogue transcripts), it is uncertain how well ClinEdu-trained tutors generalize to authentic human interactions. 

3) The proposed framework is interesting connecting RL and personalized education but might be more suited for applied AI venues.

### Questions
1) How are the weights among the rubric axes (Instruction Fidelity, Analysis Quality, Clinical Safety) determined? Is there ablation of each axis to evaluate their respective contribution?

2) How deterministic are multi-agent interactions given fixed seeds? Do minor stochastic variations in student personas lead to divergent teaching strategies, indicating emergent adaptivity?

3) LLM-as-a-judge methods can introduce self-bias. How do you control evaluator model coupling? What is the inter-rater reliability (Cohen’s kappa) between human evaluators, and how correlated are these with LLM-derived scores?

4) The system is tested on MedXpertQA and MVME datasets. How does MedTutor-R1 handle out-of-distribution pathologies or rare cases unseen during simulation?

### Soundness
3

### Presentation
3

### Contribution
2
