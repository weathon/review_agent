# Deep Reinforcement Learning for Efficient and Fair Allocation of Health Care Resources

- Decision: Reject
- Scores: 5, 10, 5

## Abstract
Scarcity of health care resources could result in the unavoidable consequence of rationing. For example, ventilators are often limited in supply, especially during public health emergencies or in resource-constrained health care settings, such as amid the pandemic of COVID-19. Currently, there is no universally accepted standard for health care resource allocation protocols, resulting in different governments prioritizing patients based on various criteria and heuristic-based protocols. In this study, we investigate the use of reinforcement learning for critical care resource allocation policy optimization to fairly and effectively ration resources. We propose a transformer-based deep Q-network to integrate the disease progression of individual patients and the interaction effects among patients during the critical care resource allocation. We aim to improve both fairness of allocation and overall patient outcomes. Our experiments demonstrate that our method significantly reduces excess deaths and achieves a more equitable distribution under different levels of ventilator shortage, when compared to existing severity-based and comorbidity-based methods in use by different governments.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a deep Q-learning approach for resource prioritization in hospitals. They learn a Q function that given the state of all patients and current ventilation status at each bed, for each vector of the next ventilation decisions (a binary vector) outputs the value per patient. These values can subsequently be utilized to formulate a policy. The method's performance is assessed using real ICU data.

### Strengths
- The problem is well-motivated and holds significant importance.
- The presentation is mostly clear.
- Real data is used for evaluation.

### Weaknesses
1. I don't think the proposed Q-learning framework fits this problem. In fact, the proposed Q-network cannot see the interaction of patients; changing $a_i$ does not change $Q_\theta(s)_{j,a_j}$, $j \neq i$. Please clarify if this is not the case.
2. It's not clear what value this Q function is estimating. Further elaboration is required here.
3. I cannot see how survival rate as a measure of performance is estimated from offline data. What off-policy evaluation technique is used?
4. Minor typos: Eq. 3: $x'$ to $x'_i$.  Page 3: in 3 to in Fig. 3.

### Questions
Please refer to the weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
10: strong accept, should be highlighted at the conference

### Rating Number
10

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
There are many situations within healthcare in which resources such as ventilators are scarce and require the use of carefully thought-out protocols to determine the proper allocation. However, currently there are many different protocols for to handle this allocation which involve conflicting heuristics. Since these decisions are usually sequential, the authors propose using reinforcement learning for
to construct a fair and effective resource allocation protocol. Specifically, they use a transformer-based deep Q-network to integrate patients' disease progressions and interactions during resource allocation. Their experiments show that their method results in both fair and effective allocation of resources.

### Strengths
The authors did an excellent job detailing the design of the RL problem and the experimental setups.

### Weaknesses
It would have been nice to see the limitations in the main text rather than the appendix. However, it is understandable due to the page limit.

### Questions
Is it possible for the authors to disclose the names of the datasets used in this study if they are publicly available?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposed a transformer-based deep Q-network method for efficient and fair allocation of ventilators in critical care settings. The experiments on a real-world dataset demonstrated that the proposed method achieved both higher patient survival rates and more equitable allocations across different ethnic groups compared to existing heuristic-based policies utilized by different governments.

### Strengths
**Originality:** The novelty of this paper mainly lies in the application aspect, in that 1) fairness objectives are incorporated into a DRL framework for healthcare resources allocation modeling 2) the individual patient disease progression and the interaction effects among patients are accounted for simultaneously by utilizing a transformer based Q-network.

**Quality:** The quality of this paper is fair. The experiments showed that the proposed method outperformed existing heuristic-based policies utilized by governments, but did not compare with any data-driven or machine learning baselines. The ablation study showed the effectiveness of different components (the ventilator cost, the patient survival and the fairness in allocation) in the reward function. 

**Clarity:** Overall the paper is clear, but the notations are sometimes confusing and more technical details are needed for a better understanding of the method / experiments.

**Significance:** Healthcare resource allocation is an important topic in critical care medicine. A more efficient and fair policy than existing government policies will result in more people being saved at lower levels of costs with minimal disparities among different ethnic groups. Thus this work has high clinical relevance and significance.

### Weaknesses
1. The notations are confusing in Section 3.3 and 3.4. Based on the context, are $x_i$ and $s_i$ the same (the medical condition of patient i), and $P^\text{on}$ and $P^\text{vent}$ the same (the ventilation transition)? If so, please ensure that the notations are consistent. Also, I think the action $a$ is N-dimensional and the i-th coordinate $a_i \in \\{0, 1\\}$ is the action applied on patient i, if this is the case, does $I'_i = a$ mean $I'_i = a_i$ in 1? Also, what is the dimension of the overall transition matrix $P$?

2. Currently, the state and action are the concatenations of the medical/ventilator states and the ventilator assignments of individual patients. Will there be scalability issue when the number of patients $N$ is really large and if there is, how to handle that with the current framework?

3. Some details in the transition model are missing. How is $P^\text{vent} (x'_i | x_i)$ determined? From the description of the simulator in Section 5.2, it seems that you consider the factual data, e.g. $P(x'_i | x_i) = 1$ if $x'_i$ is the actual next state for patient i and $P(x'_i | x_i) = 0$ otherwise? Did you consider any counterfactuals? Will the results be negatively impacted by any potential selection bias in the data if only factual data is used? Also, how are $q_i(s)$ (bed assignment distribution) and $\xi$ (initial medical condition distribution) determined?

4. Some details in the Method and Experiments are missing. The learning objective is missing. From the context, I am assuming that you are using Q-learning with the proposed reward and then used the greedy policy with the learned Q function, but it's not clear from what's written now in Section 4. Also, the choice of the ventilator cost $c_1$ and the $\lambda$ to trade-off the fairness reward term are missing in the main paper (found them in appendix). Is there any justification on how they are determined? I am also wondering how the results will change if $c_1$ and $\lambda$ are chosen differently.

### Questions
1. Will the dataset be made public if the paper is published? So that the results can be reproduced.

2. The experiments showed that the proposed method outperformed existing government policies. I am wondering if there is any data-driven baselines in the literature and how will your method perform compared to them? 
3. There are some confusions in the writing, e.g. 

- Page 3, "... proposed model in 1" -> "... proposed model in Figure 1"?
- Page 6: "... the action set $\mathcal{A}$ are defined in Eq. equation 2" -> "... the action set $\mathcal{A}$ are defined in Eq. 2"

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
