# Process Reward Model with Q-value Rankings

- Decision: Accept (Poster)
- Scores: 3, 8, 8, 8, 5

## Abstract
Process Reward Modeling (PRM) is critical for complex reasoning and decision-making tasks where the accuracy of intermediate steps significantly influences the overall outcome. Existing PRM approaches, primarily framed as classification problems, employ cross-entropy loss to independently evaluate each step's correctness. This method can lead to suboptimal reward distribution and does not adequately address the interdependencies among steps. To address these limitations, we introduce the Process Q-value Model (PQM), a novel framework that redefines PRM in the context of a Markov Decision Process. PQM optimizes Q-value rankings based on a novel comparative loss function, enhancing the model's ability to capture the intricate dynamics among sequential decisions. This approach provides a more granular and theoretically grounded methodology for process rewards. Our extensive empirical evaluations across various sampling policies, language model backbones, and multi-step reasoning benchmarks show that PQM outperforms classification-based PRMs. The effectiveness of the comparative loss function is highlighted in our comprehensive ablation studies, confirming PQM’s practical efficacy and theoretical advantage.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces PQM as an improvement over existing PRM methods, particularly in tasks requiring complex reasoning and decision-making. Traditional PRMs, typically framed as classification problems using cross-entropy loss, evaluate each step independently, which can result in suboptimal reward allocation and fails to account for interdependencies between steps. PQM redefines PRM using a MDP framework, optimizing Q-value rankings through a novel comparative loss function that better captures the dynamics of sequential decision-making. The authors claim that PQM offers a more detailed and theoretically robust method for distributing rewards across a process. Empirical evaluations demonstrate that PQM outperforms classification-based PRMs across different language model backbones, sampling policies, and multi-step reasoning tasks. Additionally, ablation studies confirm the effectiveness of the comparative loss function. The code is made available for reproducibility.

### Strengths
PQM presents clear and well-structured experimental results, providing strong empirical evidence of its effectiveness across various benchmarks. The thorough evaluations, including comparisons with classification-based PRMs and comprehensive ablation studies, effectively demonstrate the advantages of the proposed framework and its comparative loss function.

### Weaknesses
The paper lacks clarity regarding the necessity of the OSF framework. It does not sufficiently justify why this specific approach is needed to address the challenges in reinforcement learning, or how it fundamentally improves upon existing counterfactual explanation methods. The rationale behind why the OSF state is more effective than traditional approaches remains underexplained, making it difficult to assess its true impact.

### Questions
N/A

### Soundness
3

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
This paper proposes a new reward modeling framework for LLMs by using Q-value ranking to enhance the credit assignment of each state’s contribution to the outcome. Both theoretical and empirical results demonstrate the effectiveness of the proposed framework.

### Strengths
- In general, the paper is well-written and the proposed framework is easy to follow and understand.
- The theoretical results support the use of Q-value ranking, which is further validated by empirical results.
- The relationship between the proposed framework and prior work (classification-based PRM) is compelling.
- The experimental results are comprehensive, and the case study helps readers easily grasp the concepts.

### Weaknesses
It is hard to find any weaknesses.

### Questions
- Are there cases where Q-values are inaccurately trained?
- How does inaccuracy in Q-values impact performance?
- It would be beneficial if the authors provided a discussion on inaccuracies in Q-values, as this could offer an opportunity to improve the proposed method.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In the presented work, the authors introduce a novel approach to process reward modeling that focuses on the contribution of intermediate reasoning steps and their respective contributions to the final methods. In contrast to existing PRM approaches, the authors introduce a Q-value base method (Process Q-value Model) that is better suited for the problem by using a ranking approach, thereby capturing inter-dependencies among intermediate reasoning steps. The problem addressed by the authors is relevant to the current field, particularly for tasks requiring multi-step reasoning, like MATH500, and is thereby timely and demonstrates significant performance improvements over prior work.

### Strengths
- Sections 2 and 3 are well written and provide sufficient context of the field as well as the proposed method
- The experiments are comprehensive and, without a doubt, the proposed method achieves high performance across multiple LLM backends and math benchmarks, making it a strong contribution.

### Weaknesses
- Section 3 could benefit from some intuition. While the content seems sound, some higher-level guidance could be beneficial (this is just a minor issue)
- Experiments on self-consistency could benefit from some additional explanations. In particular, given figure-2 (right), why is self-consistency particularly important for larger models, given the clear difference between PQM and PQM+SC only in the 70B model? Providing additional insights here would be very useful. 
- I would recommend adding some bold numbers to Table 3 also (minor issue)

### Questions
- The best results on MATH 500 are achieved with MetaMath-Mistral-7B when using zeta = 4, while for LLaMA-3-70B-instruct, zeta = 2 yields the highest performance. Interestingly, this relationship is reversed for the GSM-Plus benchmark. Is there some intuition behind why the relative importance of intermediate steps has such a model-dependent influence? This trend is also visible in Table 3.
- Similarly, is the zeta value of 2 required for larger models in general, or is this just a quirk of llama-3?
- Intuitively, PRMs interpret all subsequent steps after the first mistake as wrong. How does the proposed method handle these situations?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes Process Q-Value model (PQM) a framework which uses Q-value instead of immediate rewards in the Process Reward Modeling (PRM).

Unlike PRM, the proposed framework allows to model inter-dependencies among reasoning states. Moreover, the authors show that the existing PRM framework can be cast as a special case of PQM.

The authors provide an extensive empirical study of the proposed method and demonstrate the clear benefit of the proposed method.

### Strengths
Overall, the paper is overall well written (see one remark in Weaknesses) and easy to follow.

The paper extends the existing PRM framework to a more general PQM framework which uses Q-values instead of intermediate rewards, which allows to capture the dependency between reasoning states (rather than having these states to be independent). This extension is natural and well motivated.

The paper provides empirical study highlighting the effectiveness of the proposed method. Moreover, the paper provides ablations for the introduced hyperparameters. For example, Table 3 suggests U-shape behavior of the $\zeta$ parameter with good values being in the middle. Overall, it seems that there is a sufficiently large range for this parameter which leads to good performance.

### Weaknesses
A suggestion to slightly improve presentation. In Section 3.3, it would be helpful to outline the overall objective for the theorem 3.5 (what do we want to prove any why), and then outline the plan for this proof (why we need other lemmas).

From the results, it is unclear why (Figure 2) the gap in performance between SC+PQM and PQM is larger as we go to the right. It would be helpful if the authors add explanations of why they believe it is happening.

### Questions
Why is the gap in performance between PQM and SC+PQM increases as we move to the right in Figure 2?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces an algorithm to train a process verifier using a Q-value ranking objective. In particular, they split the intermediate steps in a generated trajectory into steps with high and low Q-values and optimize a ranking loss with margins. The authors empirically show that the learned PQM is better for best-of-N, compared to prior works like Wang et. al., Lightman et al., that use a classification based loss to train the Q-function.

### Strengths
- The proposed ranking loss for training PRMs empirically improves best-of-N on MATH500 dataset, for multiple base LLMs. In particular, it outperforms prior work Wang et. al., that trains the PRM with a BCE loss.
- The analysis in Section 4.3 is insightful and shows that using a margin based ranking (ablating on $\zeta$) improves best-of-N performance when using the PQMs as an ORM (taking the min score over individual steps).

### Weaknesses
- The trained verifier is only used for best-of-N which is not its most promising use case. Evaluating its efficacy for beam-search where the PQM ranks intermediate generations is needed to demonstrate why practitioners should train PQMs. 
- Training with the binary cross-entropy loss, where the labels for each prefix are the expected future reward (some value between 0 and 1), will also distinguish prefixes across problems, and maybe the difference in expected rewards for prefixes can be accentuated with an appropriate scaling factor on the expected reward. In theory, it is unclear why the model trained with ranking loss should lead to a more generalizable solution that the one trained with this version of the classification loss. Essentially, it is unclear why the ranking loss should lead to a solution different from the one trained by Wang et al., when the model is trained on the population objective. I believe that the empirical results suggest otherwise, but this observation lacks formal or intuitive justification.
- It is unclear if BCE and $Q_\sigma$ models induce very different rankings on the Q-values of the steps. From the qualitative example in Table 4, it seems that they do not differ by a lot. An empirical study is needed to understand if they result in very different rankings on the steps in test examples. And if the rankings do not differ by a lot, then some post-hoc calibration of the BCE solution should also perform as well as PQM. And if they do differ in rankings, then it is more interesting, and important to study why this happens.
- Are PQMs more sample efficient than BCE? Some analysis on training PQMs with different dataset sizes would help clarify. Also, what is the best data collection strategy to train PQMs? Is it just IID samples from the base LLM? Should we sample more solutions per question, or focus on improving coverage over both questions and solutions. Basically, if we have a fixed sampling budget, how should we collect data to train PQMs?
- In a number of places, the objective optimized or algorithm used is unclear (see Questions).

### Questions
- What is the precise definition of a correct or incorrect step? Is it based on the Q-value? 
- Equation 3 and 4 are unclear, and in general the notation $\bar{a_{1:t}}$ is confusing. For example, since $\tau$ in L222 is composed of $x, a_1, \ldots, a_H$, shouldn't $P(\tau | \bar{a_{1:m}})$ be $0$ in Equation 3? 
- In the objective in Eq. 9, how is $Q_{w_t}$ computed, i.e., what is the input to the network? Is it only the question and action, or the whole prefix up until the action $w_t$?

### Soundness
2

### Presentation
2

### Contribution
3
