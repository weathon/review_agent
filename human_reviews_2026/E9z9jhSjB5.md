# Online Continual Learning under Real Concept Drift: A Statistical Perspective

- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
Real-world data often exhibit non-stationarity, prompting growing interest in adaptive learning techniques. Continual learning, which aims to sequentially learn multiple tasks, provides a promising framework to address this challenge. However, learning under real concept drift, where the relationship between inputs and outputs evolves over time, remains relatively underexplored. In this paper, we propose a novel regularization-based method that incorporates a memory buffer to improve robustness against concept drift. Assuming the existence of a common center for the evolving true models, our method jointly constrains current and past task estimates, effectively bridging them to form a stable estimate that incorporates information across tasks. To further adapt to task variability, we develop an online algorithm that dynamically tunes task-specific regularization parameters. We also provide theoretical guarantees by deriving an error bound that characterizes the overall performance of the estimator, explicitly capturing the effects of task-relatedness, memory buffer size, and regularization strength. Extensive experiments demonstrate that our method achieves superior stability–plasticity trade-offs under varying degrees of task similarity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a framework for continual learning in dynamic environments where the relationship between inputs and outputs changes over time. It proposes a regularization-based method that combines memory replay with adaptive tuning to balance knowledge retention and adaptability. The authors develop an efficient online algorithm with theoretical guarantees linking performance to task similarity, memory size, and regularization strength. Experiments on synthetic and real-world benchmarks show that the proposed method outperforms established continual learning baselines.

### Strengths
1. This paper is well-written and easy to follow.

2. The proposed concept drift setting is interesting. It may apply to some specific scenarios such as recommendation systems.

3. The proposed method is theoretically-grounded, and achieves strong performance in this particular setting.

### Weaknesses
1. The use of memory buffer, although effective, may result in additional storage cost and privacy concerns. 

2. The compared methods are mainly traditional continual learning methods (EWC, ER, and AGEM). Is it possible to include more recent methods, such as continual unlearning methods?

3. The experiments are mainly performed with relatively simple datasets, such as MNIST and CIFAR-10. Does the proposed method apply to larger-scale datasets, such as ImageNet (subsets)?

4. Do the theoretical analysis and the proposed method only apply to online continual learning? Is it possible to extend them to other continual learning scenarios (e.g., offline continual learning)?

### Questions
My major concerns lie in the comparison baselines and applicability of the proposed method. Please refer to the Weaknesses.

### Soundness
3

### Presentation
3

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
The paper introduces a regularization-based method leveraging a memory buffer to address real concept drift. This method connects past and present task estimations through a central point for evolving models, promoting both stability and adaptation. Theoretical bounds regarding generalization error are provided, and their method is benchmarked against SGD, EWC, ER, and AGEM on both synthetic and real-world datasets, including kidney transplantation data.

### Strengths
1. **Theoretical Contribution:** The paper contributes to the theoretical understanding of Continual Learning with the buffer size used. 
    
2. **Clarity:** The overall writing is ok and accessible.

### Weaknesses
1. **Notation**  As $|\cdot|$ is defined as the absolute value of a real number or cardinality of a set, the definition in Assumption 1 is problematic, which should be  $\|\|w_j -w_0\|\|_2$. In addition, $J^c$ is undefined. In line 179, the definition of empirical loss of the $t$-th task $\ell(w, z_t^i)$ is inconsistent with that in eq.(1), which is for the whole data set. 
2. **Insufficient Novelty and Related Work Discussion** I do not find obvious significance for the proposed methods against online/continual meta-learning approaches, where hyperparameters and regularizations are both considered for concept drift or more general shifts. This work is missing related works in this domain, e.g., [1-8]. 
    
3. **Theoretical Results:**  
    * Theorem 1 only measures the distance of a single $\hat{\theta}_T$ to every task optimal parameter; it's unclear how it affects the average population loss. 
    * Unclear definition of the $\lambda$ in Theorem 1, whether all the tasks use the same $\lambda$? What is its relation to the $\lambda$s in eq(2)?  
    * How does $a_1$ and $a_2$ affect the theoretical results?
    
4.  **Empirical Results:**  
    * Potential Overfitting in Real-world Data: The paper indicates that theoretical assumptions might not fully apply in empirical evaluations, suggesting possible overfitting or adaptation challenges in diverse settings.
    * The memory buffer size effectively affects the results. How can we make sure it's fairly compared with methods without a memory buffer? 
    * Except for Permuted MNIST or Split CIFAR-10, typical online/continual meta-learning approaches consider concept drifts by predicting different characters for each task, e.g., select different five characters from 10 to predict their classes.   
    
--- 
[1] Chelsea Finn, Aravind Rajeswaran, Sham Kakade, and Sergey Levine. Online meta-learning.
In International Conference on Machine Learning, pages 1920–1930. PMLR, 2019.

[2] Massimo Caccia, Pau Rodriguez, Oleksiy Ostapenko, Fabrice Normandin, Min Lin, Lucas
Page-Caccia, Issam Hadj Laradji, Irina Rish, Alexandre Lacoste, David Vázquez, et al. Online
fast adaptation and knowledge accumulation (osaka): a new approach to continual learning.
Advances in Neural Information Processing Systems, 33:16532–16545, 2020.

[3] Giulia Denevi, Carlo Ciliberto, Riccardo Grazzi, and Massimiliano Pontil. Learning-to-learn
stochastic gradient descent with biased regularization. In International Conference on Machine
Learning, pages 1566–1575. PMLR, 2019.

[4] Qi Chen, Changjian Shui, Ligong Han, and Mario Marchand. On the stability-plasticity dilemma
in continual meta-learning: Theory and algorithm. Advances in Neural Information Processing
Systems, 36:27414–27468, 2023.

[5] Maria-Florina Balcan, Mikhail Khodak, and Ameet Talwalkar. Provable guarantees for gradientbased meta-learning. In International Conference on Machine Learning, pages 424–433. PMLR,
2019.

[6] Mikhail Khodak, Maria-Florina Balcan, and Ameet Talwalkar. Adaptive gradient-based metalearning methods. arXiv preprint arXiv:1906.02717, 2019.

[7] Qiang Zhang, Jinyuan Fang, Zaiqiao Meng, Shangsong Liang, and Emine Yilmaz. Variational
continual bayesian meta-learning. Advances in Neural Information Processing Systems, 34:
24556–24568, 2021.

[8] Xu, Kunlun, et al. "Componential Prompt-Knowledge Alignment for Domain Incremental Learning." arXiv preprint arXiv:2505.04575 (2025).

### Questions
1. **Question on the Empirical Results:** When $\delta=0$, which means all the tasks are the same, can you explain why the proposed method is better than others? 

2. **Generalization to Complex Tasks:** How does the method perform with tasks having complex dependencies beyond shared central models?
    
3. **Memory Buffer Adaptability:** Does the method adapt efficiently under varying memory constraints, significantly beyond tested scenarios?
    
4. **Implementation Details:** Given the overhead of dynamic parameter tuning, are there strategies to streamline this process without compromising learning efficacy?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the problem of online continual learning under real concept drift — i.e., when the relationship between inputs and labels evolves over time — rather than the more common “task-incremental” setting in which tasks are static and well-defined. Their main contributions are:


* The authors highlight that many continual learning (CL) works assume a fixed task boundary or static relationship, whereas in many real-world scenarios the underlying model “drifts” as new data arrive (inputs → outputs mapping changes). 

* To address concept drift, the authors propose a regularization-based methodology that uses a memory buffer of past examples and constrains the current model’s estimate jointly with past task estimates, under the assumption of a “common center” for the evolving true models. (i.e., current and past models cluster around a latent center)


* In order to adapt to the variability between tasks (or time-segments), the algorithm dynamically tunes task‐specific regularization parameters in the online setting — enabling a better balance of plasticity vs. stability under drift. 

* The authors derive an error bound for their estimator, explicitly characterizing how performance depends on: task-relatedness (distance to the latent center), memory buffer size, and regularization strength.

### Strengths
* The focus on “real concept drift” rather than idealised fixed-task CL is  relevant for practical deployments (e.g., streaming data, non-stationary domains).


* The derivation of an error bound is a strong component; it connects method design (memory size, regularization) to performance guarantees.


* The dynamic tuning of regularization strength per time‐segment is a meaningful advance over fixed hyper-parameters, enabling the method to adapt to varying drift severity.

### Weaknesses
* A key assumption is that the evolving true models (over time) share a “common center” around which they drift. While convenient analytically, this may be unrealistic in many settings where the drift is large or the underlying tasks change drastically. The paper could benefit from discussion or analysis of what happens when the assumption fails.


* If the drift is very abrupt (i.e., the new model is far from the previous center) or tasks are entirely unrelated, it is unclear how well the method will perform. The paper may not sufficiently explore worst‐case drift scenarios.



* The experiment's evaluations are limited to a few synthetic or small‐scale drift settings; it may raise questions about how the method generalises to more complex domains (e.g., large-scale dataset, vision transformer model). The paper may not explore this breadth fully.

### Questions
* Could you comment on scenarios where the “true models” do not cluster around a common center (for example, when the drift jumps to a new regime far from previous ones)? How does your algorithm behave in such cases?



* In your experiments, what model sizes/dimensions and what streaming rates (data per time unit) did you consider? How would your method scale to large models, long sequence of time‐segments, or high data throughput?

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
4

### Summary
The paper studies online continual learning when the input stream stays roughly the same but the label rule changes over time (“concept drift”). The authors assume that most tasks are actually very similar to one hidden “center” model, and only a small portion can be different. On top of that, they propose an objective that learns from the current data, from a replay buffer, and at the same time pulls everything toward the shared center. They also add an online step that, at the start of each task, tries a few regularization strengths and picks the one that fits this task best. They give a generalization bound under this “all tasks are close to one center” setting, and they show experiments on synthetic data and on a medical dataset where the method beats common continual-learning baselines.

### Strengths
1.Clear and tidy formulation that connects replay-style and regularization-style continual learning.

2.Explicit assumption about how similar tasks are, which many CL papers do not write down.

3.Practical online tuning step, so we do not have to hand-tune for each task.

4.Analysis that clearly shows how task similarity, buffer size, and regularization strength matter.

### Weaknesses
1.The intro sounds general, but the method and theory only work when almost all tasks are variants of the same model.

2.The key assumption appears too late in the paper.

3.The objective keeps three sets of parameters, even though in the end we only keep the shared one; this needs a clearer justification or an ablation.

4.The weights between old data and new data only depend on how many samples we have, not on how different they are, which is odd for drift.

5.The online tuning step seems to assume we know when a new task starts.

6.The replay buffer is “blind” to the shared center; a center-aware buffer would match the story better.

7.Theory is for the easiest case (all tasks close), not for the mixed case (some tasks far).

8.Synthetic experiments are built exactly the way the assumption says, so the results are a bit circular.

9.No baselines from the concept-drift community.

### Questions
1.You claim “real concept drift,” yet assume a single latent center with (ε,δ)-related tasks. Are you actually studying center-constrained drift? If so, shouldn’t the title/intro say so explicitly, and why is this subclass representative of “real” drift in practice?

2.Why is the central (ε,δ)-related assumption only introduced in Sec. 3? Could you front-load it in Sec. 1 and state applicability limits—e.g., which common drifts (directional, periodic, multi-center) are out of scope?

3.Relative to OMTL/Lifelong learning with shared structure, is the novelty the integration (shared center + replay + online tuning) or a new form of sharing itself? Can you position this explicitly to avoid over-claiming originality?

4.If the final output is θ, why not optimize a θ-only objective that blends past and current losses? What is the operational gain of explicit ω_past and ω_t—different λ/optimizers only, or real performance gains? Any θ-only ablation?

5.Under strong drift, shouldn’t small fresh data outweigh large stale data? Why are a1,a2 purely count-based instead of similarity/drift-aware? Can Sec. 3.2’s pseudo-validation scores feed into (a1,a2) for a coherent stability-plasticity trade-off?

6.Alg. 2 uses the first B samples as pseudo-validation—does this assume known task boundaries? If boundaries are unknown or intra-task drift exists, do you need sliding windows/change-point tests/periodic re-tuning? What’s the runtime vs. |Λ| and B?

7.If the core idea is contraction toward θ, why keep θ-agnostic reservoir sampling? Could you test θ-aware buffering (e.g., gradient alignment, center/outlier discriminability, core-set selection) and show effects on the stability–plasticity trade-off?

8.The main bound targets ε=0. For ε>0, how do errors scale with ε, δ, M, λ? Can you provide a relaxed bound or at least a “graceful degradation” analysis (to which baseline do we regress, up to what ε is it robust)?

9.Synthetic setups mirror the single-center assumption. Under directional drift, two-center switching, periodic re-occurrence, or subspace-local drift, do you still beat ER/EWC/AGEM? Any stress tests that violate the assumption and profile failure modes?

### Soundness
2

### Presentation
3

### Contribution
3
