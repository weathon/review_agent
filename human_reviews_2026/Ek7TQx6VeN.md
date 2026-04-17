# Goal-Conditioned Supervised Learning for Multi-Objective Recommendation

- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Multi-objective learning endeavors to concurrently optimize multiple objectives using a single model, aiming to achieve high and balanced performance across diverse objectives. However, this often entails a more complex optimization problem, particularly when navigating potential conflicts between objectives, leading to solutions with higher memory requirements and computational complexity. This paper introduces a Multi-Objective Goal-Conditioned Supervised Learning (MOGCSL) framework for automatically learning to achieve multiple objectives from offline sequential data. MOGCSL extends the conventional GCSL method to multi-objective scenarios by redefining goals from one-dimensional scalars to multi-dimensional vectors. It benefits from naturally eliminating the need for complex architectures and optimization constraints. Moreover, MOGCSL effectively filters out uninformative or noisy instances that fail to achieve desirable long-term rewards across multiple objectives. We also introduces a novel goal-selection algorithm for MOGCSL to model and identify "high" achievable goals for inference.

While MOGCSL is quite general, we focus on its application to the next action prediction problem in commercial-grade recommender systems. In this context, any viable solution needs to be reasonably scalable and also be robust to large amounts of noisy data that is characteristic of this application space. We show that MOGCSL performs admirably on both counts by extensive experiments on real-world recommendation datasets. Also, analysis and experiments are included to explain its strength in discounting the noisier portions of training data in recommender systems with multiple objectives.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper presents an innovative approach to improving multi-objective recommendation systems by extending the Goal-Conditioned Supervised Learning (GCSL) framework to a multi-dimensional goal space. Instead of addressing a single scalar objective, the authors define multi-dimensional goals and develop a mechanism that enables the system to learn a recommendation policy that balances between different objectives according to preferences or contextual conditions.

### Strengths
While I believe the paper is still premature, I find the topic and the core idea very interesting.

### Weaknesses
In my opinion, the paper is written in an unfocused manner, which made it difficult for me to understand.

### Questions
Please address the weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes MOGCSL, a framework that extends GCSL to handle multiple objectives by redefining goals as multi-dimensional vectors.  When applied to large-scale recommender systems, MOGCSL demonstrates strong scalability and robustness to noise. The experiment on recommender systems shows that MOGCSL outperforms many previous baselines and also outperforms MOPRL with different weights.

### Strengths
1. The experiment of this paper is comprehensive. They also show some comparisons about MOPRL and MOGCSL in the Appendix. I like the comparison between MOPRL and MOGCSL with different weights (MOGCSL remains unchanged).  This paper also contains many baselines and shows that MOGCSL achieves the best performance among many baselines. 



2. The algorithms are proposed in a clear way. The paper is also well-structured and easy to follow. 

3. The idea of using multi-objective GCSL to provide a solution for multi-objective learning is novel and interesting, which avoids knowing the weights of the objectives in advance. It is also reasonable to use VAE to generate goals.

### Weaknesses
1. In Line 146, the author claims that the dataset contains the $s_t$, which is the representation of the user's preferences. It does not practical in the real world. What does the real representation look like in your experiment?

2. The model structure introduced in Section 3.2 appears very similar to PRL, with the only difference being the addition of an extra MLP layer. Therefore, the improved performance over previous Shared-Bottom and MMoE models may largely stem from this architectural change (since MOPRL also achieves a good result), which weakens the claimed contribution of the paper.

3. Theorem 1, which serves as the main theoretical contribution of this paper, is trivial. It just claims that the expected goal is determined by some initial variable and the policy $\pi$.  This makes the theoretical contribution of this paper limited.

### Questions
1. It looks like these two objectives are correlated. What about the setting in which the objectives are in conflict?  Are there any benchmark tasks involving more conflicting objectives, beyond those focused only on recommender systems? Do the authors think MOGCSL will be general enough and work well in this setting?


2. In Line 427 and Table 2, the authors claim that their approach only requires an additional MLP layer and could be very efficient. Does the time reported in Table 2 include the training time for the VAE? Since MOGCSL additionally trains two CMAEs for goal generation, could the authors clarify whether the method remains efficient when accounting for this extra training cost?

3. In Section 4.4, the authors say that sparsity of training data within the high-goal space may contribute to the suboptimal performance of more advanced goal-choosing methods. Could you alleviate this through online data collection? The algorithm in this paper is purely offline. Is it possible to contain an online part in the algorithm?

4. In Line 276, how is the partial ordering obtained in practice? It seems to require weights across objectives in advance. In the experiments, no predefined partial ordering is used. Do the authors believe the method would perform better if such prior information were incorporated?

### Soundness
3

### Presentation
4

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
This paper proposes MOGCSL, which extends GCSL to MTL by extending scalar goals to vectorized goals. A goal-selection algorithm is devised to identify achievable goals for inference. Experiments are performed on sequential recommendation task, using public datasets to showcase the efficacy of the proposed method.

### Strengths
1. The writing is clear and easy to follow.

2. The investigated multi-task learning problem is essential in recommendation system field.

### Weaknesses
1. In experiments, it would be beneficial to include more recent baselines with high reputations. On the MTL aspect, it seems that only DWA, PE and FAMO are involved. However, there are massive important MTL optimizers, including but not limited to MGDA, PCGrad,  UPGrad, MoDo, Nash-MTL, GradNorm, etc. Moreover, since authors mainly consider the sequential recommendation problem, it is not very clear why sequential recommendation models (e.g., FMLP-Rec, DuoRec, Longer) are not involved as baselines in the main comparison table or a separate section for case study. 

2. Authors could isolate the comparison with MTL methods from the comparison with canonical sequential recommendation models. 

3. A theoretical analysis demonstrating the superiority of the proposed methods compared to the state-of-the-art MTL method is lacking, which is a critical aspect to evaluate theoretical technical quality.

4. Experiments on real-world industrial datasets and online A/B analysis are lacking, which are critical aspects to evaluate the empirical technical quality of research works in recommendation system (e.g., sequential recommendation) field.

5. The motivation and rationale to select sequential recommendation as a testbed is not very persuasive. The problem aimed to solve by the proposed method seems to be very general instead of the unique problems in sequential recommendation.

6. This paper should discuss limitations and delineate future works to address these limitations in detail.

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript proposes the MOGCSL framework, extending the GCSL paradigm to multi-objective recommendation. Its core involves redefining GCSL's scalar goal into a multi-dimensional vector, which it claims avoids the complexity and dynamic loss weighting of traditional MTL architectures. By conditioning on high-value goals, the method aims to "de-noise" and focus on high-quality interactions. The manuscript also introduces a companion CVAE algorithm for goal selection at inference time. Experiments on two real-world e-commerce datasets show that MOGCSL significantly outperforms a range of strong baseline models on purchase-related metrics.

### Strengths
1.	The manuscript reframes multi-objective recommendation as a goal-conditioned supervised learning problem. This approach avoids the need for complex multi-task architectures or explicit conflict-handling during optimization, which is an insightful contribution.
2.	MOGCSL is shown to be more efficient than baselines in both model size and training speed. This is a crucial advantage for large-scale industrial applications.
3.	The work commendably tackles the difficult problem of goal selection during inference, a key challenge for GCSL. It provides an analysis and a theoretically-motivated CVAE-based solution.

### Weaknesses
1.	The manuscript's main technical contribution, a CVAE-based goal selection algorithm (MOGCSL-C), fails to consistently outperform a simple statistical heuristic (MOGCSL-S) and performs worse on one dataset. This undermines the algorithm's practical value and questions the significance of its theoretical backing.
2.	The explanation for the CVAE method's failure—data sparsity—feels like a post-hoc justification. The supporting experiment is relegated to an appendix and uses a different dataset, suggesting the method's effectiveness is a fundamental limitation tied to data properties, which is not adequately addressed in the main manuscript.
3.	The model's strong performance on purchase metrics comes at the cost of mediocre performance on click metrics, a trade-off that is not sufficiently discussed. This suggests an implicit, unmanaged trade-off, potentially contradicting the claim of avoiding explicit optimization constraints.
4.	The claim of denoising capability is supported only by a simplistic synthetic experiment in the appendix. The noise model used is not representative of complex, real-world scenarios, making the evidence for this claim weak.

### Questions
1.	Given that the CVAE-based method can underperform the simpler statistical one, how do you justify its added complexity for practical deployment?
2.	Does the CVAE's failure on sparse-reward data mean MOGCSL is only effective for datasets with dense rewards? Have you considered methods like data augmentation to mitigate this limitation?
3.	What is the mechanism behind the observed trade-off favoring purchase metrics over click metrics? How does the model resolve conflicting goals (e.g., high click and high purchase), and does it implicitly learn to prioritize one objective based on the training data?
4.	Your definition of "noise" equates it with low long-term utility. However, a quick exit after a click could be a strong negative preference signal, not just noise. Can your model distinguish between random noise and valid negative feedback?

### Soundness
2

### Presentation
2

### Contribution
2
