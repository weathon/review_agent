# Online Selective Generation with Adversarial Bandit Feedback

- Decision: Reject
- Scores: 4, 4, 8, 2

## Abstract
Large language generative models increasingly interact with humans, while their falsified responses raise concerns. To mitigate this hallucination effect, selectively abstaining from answering, called selective generation, provides an effective way for generators to control the hallucination when uncertain about their answers. However, as selective generators interact under adversarial environments and receive partial feedback from users on selected generation (e.g., thumbs up or down on the selected answer), learning methods for selective generation under such practical setups are crucial but currently missing. To address this limitation, we propose an online learning algorithm for selective generation with partial feedback under an adaptive adversary. In particular, we re-purpose an adversarial bandit algorithm to design an online selective generation method with controllable false discovery rates (FDR), which measures the rate of hallucination. The key building blocks include a novel conversion lemma from regret of any bandit algorithm to the FDR, and the exploitation of a unique structure of selective generation to reuse partial feedback, which we call feedback unlocking. We empirically evaluate the efficacy of the proposed online selective generation algorithm with partial feedback over diverse learning environments, demonstrating its ability to control the FDR, while maintaining reasonable selection efficiency, i.e., the ratio of non-abstaining answers, compared to baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel online learning framework called ExSUL for selective language generation. The goal is to enable models to judiciously decide when to generate answers and when to abstain, especially in environments with noisy, incomplete, or adversarial feedback. The method leverages partial feedback—such as binary signals indicating correctness—to learn effective abstention policies without requiring full supervision. By incorporating techniques from adversarial bandit algorithms and providing regret guarantees, ExSUL balances the trade-off between reducing hallucinations and maintaining answer coverage. The experimental results demonstrate that the approach outperforms existing methods in controlling hallucinations while sustaining high answer quality across multiple datasets and settings.

### Strengths
1. The paper presents a novel framework that effectively utilizes partial, binary feedback signals to learn abstention policies in language generation tasks. 
2. By adapting adversarial bandit algorithms, the proposed ExSUL method provides formal regret bounds, ensuring that the model's accumulated reward converges close to that of the best fixed policy in hindsight. 
3. Experiments demonstrate that ExSUL achieves superior control over hallucination rates, maintaining high answer accuracy and low FDR across various datasets and under distribution shifts.

### Weaknesses
1. This paper investigates online selective generation, yet lacks corresponding online experiments or genuine user feedback, which to some extent undermines the reliability of the results. For instance, this paper employs GPT as a generative model and similarly utilises the GPT series to simulate user feedback. Homogeneous models may exhibit similar hallucinations and also harbour homogeneous biases, which diverge significantly from real-world user feedback. While the paper provides promising results in simulated and benchmark environments, it offers limited discussion on how the proposed abstention and learning strategies scale to large-scale, noise real-world applications.
2. The method relies on the presence of binary or weak feedback signals, which may vary in quality and consistency across different deployment scenarios. The paper does not thoroughly analyze how noisy or sparse feedback could impact the learning process, leaving some uncertainty about robustness to imperfect supervision.
3. The evaluations focus on their defined metrics (e.g., FDR, regret) without incorporating human judgment or user-centric metrics. Including such evaluation would better demonstrate the practical benefits and acceptance of the abstention policy in real-world decision-making contexts. Moreover, this paper aims to enhance the output quality of LLMs from the perspective of selective generation. However, it appears to place undue emphasis on FDR. An LLM that is overly reluctant to provide answers may exhibit a low FDR, yet it would hardly be considered a good model.

### Questions
1. To what extent does GPT-generated simulated user feedback deviate from genuine noisy and random user feedback?
2. Does ExSUL maintain its performance advantage under conditions of noisy and sparse user feedback?
3. In real-world model deployment, how should one balance false positive rate (FDR) with over-rejection?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
1

### Summary
This paper introduces a novel framework for online selective generation—a setting where a large language model (LLM) can abstain from answering uncertain queries to reduce hallucination. The authors focus on a realistic adversarial and partial feedback scenario, where feedback is limited (e.g., thumbs-up/down on generated answers) instead of full supervision.
To tackle this, the paper reformulates selective generation as an adversarial multi-armed bandit problem and proposes an online learning algorithm that controls the False Discovery Rate (FDR)—the proportion of hallucinated outputs among generated ones—while maintaining high selection efficiency (i.e., answering rate). The key technical innovations are:
A Regret-to-FDR conversion lemma that connects the regret bounds of any bandit algorithm to FDR control guarantees.
A feedback unlocking mechanism that exploits structural properties of selective generation to reuse partial feedback efficiently.
Extensive experiments under diverse environments demonstrate that the proposed algorithm achieves effective FDR control and maintains competitive efficiency compared to baselines, showing robustness in adversarial and distribution-shifted conditions

### Strengths
- The paper addresses a realistic and underexplored setting - online generation with partial and adversarial feedback and provide theoretical link between bandit regret and false discovery rate, offering a principled approach to uncertainty control in generative models.
- This seems aligns well with human-in-the-loop and reinforcement learning from human feedback (RLHF) settings.

### Weaknesses
- Some algorithmic components (especially feedback unlocking) may require additional clarification for reproducibility.

### Questions
1. Can the proposed method be integrated with large-scale language model training pipelines?
2. How is the feedback unlocking mechanism implemented in practice—does it assume stationarity in the abstention threshold?
I am not very familiar with the learning theory field, so I leave a low confidence for my review.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This work proposes an online selective-generation method that leverages “feedback unlocking” to learn a threshold from partial feedback, controlling FDR with competitive selection efficiency.

### Strengths
- The paper is clearly written and easy to follow.
- The paper relaxes i.i.d. to non-stochastic inputs with partial feedback. This matches real-world use.
- Results across multiple distribution-shift settings (single, alternating, gradual) substantiate the claims.

### Weaknesses
- The theory permits an adaptive adversary, but the experiments cover only distribution shifts and a templated interactive setup.
- The main text says the interactive environment uses two GPT-3.5-turbo models, whereas the appendix specifies GPT-3.5-turbo + two GPT-4o.

### Questions
- The theory assumes an adaptive adversary, but the experiments don’t seem to include one. How does the method perform against a strategy-aware adaptive adversary?
- How was the threshold grid size $|H|$ selected?

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
3

### Summary
This work studies selective generation in Large Language Models, a mechanism that allows the model to selectively abstain from answering and prevent hallucination. The authors reduce this problem to a standard adversarial bandit problem and provide a theoretical analysis for the relationship between the bandit regret and the false discovery rate. Based on this reduction, the authors proposed a new algorithm, and experiments support the efficiency of the proposed method.

### Strengths
1. The paper focuses on preventing hallucination, directly contributing to a major area of concern for LLM safety and usability.

2. The proposed method's efficacy is experimentally validated by its success in lowering the false discovery rate.

### Weaknesses
1. The motivation for Selective Generation is not sufficiently strong, as the strategy's high cost in utility may outweigh its benefit in reducing hallucination. While selectively abstaining from answers helps reduce the False Discovery Rate, Figures 3-6 show an alarming inefficiency rate of 40% to 50%, meaning the LLM refuses to answer nearly half of the questions, which severely limits its practical impact. This approach risks reducing the problem to a trivial solution (e.g., only answering simple, high-certainty questions) that offers minimal real-world help. Instead of outright abstention, a more reasonable approach would be to output the low-confidence answer along with the low-confidence sign. This allows the user to still potentially gain some minor insight or utility while remaining informed and able to control for the risk of hallucination.

2. The reduction of the selective generation problem to an adversarial bandit framework is theoretically questionable due to a fundamental conflict in objectives. In standard adversarial bandits, regret is a comparative metric focused on minimizing the sub-optimality gap between the learner's performance and the performance of the unknown optimal policy ($\tau$). Conversely, the FDR risk in selective generation is a stand-alone metric that only measures the learner's performance without requiring a comparison to an optimal policy.

The authors address this gap by only comparing their algorithm against the trivial policy that always abstains from answering, as seen in the proof of Lemma 1 (Lines 1953-1957). This trivial policy yields a constant loss that simplifies the analysis. However, since the comparison policy is already known and fixed, the core difficulty of the bandit problem—dealing with the unknown optimal action—is bypassed. This decision to only ensure the learner does not perform significantly worse than the simplest, least effective baseline weakens the theoretical utility and necessity of employing complex bandit regret analysis.

3. The derived upper bound for the False Discovery Rate, presented in lines 253-257, is concerning. Even when neglecting the term related to inefficiency, the bound is $\alpha + 1/T^{1/4}$. This is worse than the optimal $\alpha$ achieved by the trivial policy that always abstains from answering. This finding supports the prior critique regarding the theoretical reduction: Theorem 1 essentially only guarantees that the FDR of the proposed method will not be significantly worse than that of the least-effective, baseline "always abstain" policy.

4. A significant concern regarding the experiments is the unexpected increasing trend observed in both the False Discovery Rate for Exp3-IX-SG and No-SG, and the inefficiency for ExSUL and Exp3-IX-SG after several thousand rounds (as shown in Figures 4 to 6). Intuitively, for a bandit algorithm, the cumulative regret increases, but the average regret per round should decrease; consequently, we would expect the FDR to decrease as the learner becomes more accurate. Similarly, a more accurate learner should result in decreasing inefficiency (fewer unnecessary abstentions). The observed increase directly contradicts the expected behavior for effective bandit learning.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2
