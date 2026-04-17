# Test-Time Policy Adaptation for Enhanced Multi-Turn Interactions with LLMs

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
Large Language Models (LLMs) employ multi-turn interaction as a fundamental paradigm for completing complex tasks. However, their performance often degrades in extended interactions, as they are typically trained on static, single-turn data, which hinders their ability to adapt to real-time user feedback. To address this limitation, we first propose a new paradigm: Test-Time Policy Adaptation for Multi-Turn Interactions (T$^2$PAM), which utilizes user feedback from the ongoing interaction as a reward signal to estimate a latent optimal policy aligned with user preferences, then updates a small subset of parameters to steer the model toward this policy, ultimately enabling efficient in-conversation self-correction. We then introduce Optimum-Referenced One-Step Adaptation (ROSA), a lightweight algorithm that operationalizes T$^2$PAM. ROSA guides the model parameters toward a theoretical optimal policy in a single, efficient update step, avoiding costly iterative gradient-based optimization and minimizing computational overhead. We provide a rigorous theoretical analysis guaranteeing that the policy of ROSA converges to the preference of user as the number of interactions increases. Extensive experiments on challenging benchmark demonstrate that ROSA achieves significant improvements in both task effectiveness and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper identifies a key weakness in modern LLMs: their performance degrades in multi-turn interactions because they are typically trained on static, single-turn data and cannot adapt to real-time user feedback. To address this, the authors propose a new paradigm, "Test-Time Policy Adaptation for Multi-Turn Interactions" (T2PAM), where the model's policy is updated "in-conversation" based on user feedback. They also introduce ROSA (Optimum-Referenced One-Step Adaptation), a specific algorithm to implement T2PAM. ROSA uses user feedback (e.g., "wrong answer," mapped to a reward) to analytically compute an optimal target policy and then performs a single, efficient update step to steer the model's parameters (via linearized optimization) toward this target. The paper also provides theoretical analysis to guarantee convergence.

### Strengths
The primary strength of this paper is its motivation. The problem it addresses—that models are static at test time and fail to learn from immediate user feedback during an interaction—is highly relevant. The goal of enabling real-time policy adaptation to correct errors in a multi-turn setting is a valuable and important research direction.

### Weaknesses
Despite the strong motivation, this paper suffers from several major flaws in its execution, theoretical justification, and experimental evaluation.

- Presentation and Focus: The paper's structure feels imbalanced. A significant portion of the main text is dedicated to theoretical analysis (Section 4). However, the experimental results section (Section 5) is comparatively brief, and critical details about baselines and efficiency are relegated to the appendix. This suggests a desire for the theory to be the main contribution, but the theoretical claims themselves appear to have issues (see below). Important information, such as accurate descriptions of the baselines, should be in the main body for a clear comparison.

- Computational Cost: The user's concern about "computation time" is valid. The paper's own efficiency analysis (Appendix E.3, Table 10) shows that the "Avg. Update Time" is substantial, roughly equal to the "Inference" time. This means the method effectively doubles the computational cost of each turn. The claim that this is "imperceptible to the user" because it can be run asynchronously is a very strong assumption that may not hold in practice, where users expect immediate responses.

- Limited Problem Formulation: The "multi-turn" task is not representative of typical multi-turn interactions. The paper frames the task as correcting a final answer to a static problem. In contrast, many (if not most) multi-turn tasks involve a process that requires multiple steps to solve (e.g., planning, complex QA, collaborative writing). In these more realistic scenarios, the paper's assumption of receiving clear, accurate, and immediate feedback (e.g., r = -1) at each intermediate step is highly impractical, as the feedback itself would likely be noisy, delayed, or unavailable until the very end.

### Questions
- There appears to be a contradiction in the proof of Theorem 2. The T2PAM paradigm explicitly states that the policy update is only triggered by failure, i.e., negative feedback ($r_k = -1$). However, the proof of monotonic error reduction (Appendix C.3, lines 1068-1069) explicitly assumes a positive reward ($r_k = 1$) to complete the bound. Is this a typo or its my misunderstanding?

- How could this method be extended to scenarios where intermediate feedback is not accurately or immediately available?

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
5

### Summary
This paper proposes **Test-Time Policy Adaptation for Multi-Turn Interactions (T2PAM)** to enhance LLMs’ adaptability during multi-turn dialogues.  The key algorithm, **ROSA (Optimum-Referenced One-Step Adaptation)**, enables online parameter updates using real-time user feedback.  Instead of iterative RL (e.g., PPO/GRPO), ROSA performs a **single-step linearized optimization** of the RLHF objective with KL regularization.  It computes an **analytic optimal policy** and efficiently updates parameters via **Conjugate Gradient**, without explicit Hessian computation.  Theoretically, ROSA guarantees **monotonic KL improvement** and **convergence**.  Experiments on reasoning, coding, and multilingual tasks show **10–30% accuracy gains** over RL-based and static baselines.

### Strengths
- **Lightweight & efficient**: One-step update at test time; no retraining or rollout needed.  
-**Stable & theoretically grounded**: Monotonic improvement under linearized KL regularization.  
- **Closed-form interpretability**: Exponential reweighting of prior policy.  
- **High empirical gain**: Outperforms PPO/GRPO in reasoning and code tasks with minimal overhead.  
- **Generalizable**: Works across model scales and task domains.

### Weaknesses
I did not find any major weaknesses in this paper, both the theoretical analysis and the empirical results appear solid and well-supported. Please refer to my questions below for clarification on several points.

### Questions
1. I am not sure whether the term **“multi-turn”** is appropriate in this context. *Theorem 1* seems to treat the optimization of a specific turn as a **single-turn problem**, without accounting for future returns. In essence, the paper applies **conjugate gradient descent** to derive the optimal policy for an individual turn.  
2. In **Appendix E.4.2**, regarding the **RL baseline**, does it also employ **LoRA** or **hidden-state updates**, or is it **fully fine-tuned**? Could the authors clarify this? In **Table 9**, the RL baseline outperforms ROSA, whereas in **Table 11**, the result is reversed — what explains this inconsistency?  
3. Could the authors elaborate on the advantage of using **conjugate gradient descent** beyond faster optimization? Does it also lead to **better convergence properties** or improved solution quality?  
4. Have the authors reached a final conclusion on which **parameter-update mechanism** (**LoRA** vs. **hidden-state update**) performs better overall?  
5. From the results, it seems that **model-based rewards** outperform **rule-based rewards** under LoRA updates. Does the same trend hold for **hidden-state updates**?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the problem of optimizing large language models in multi-turn interactions. The authors first propose a new paradigm called Test-Time Policy Adaptation for Multi-Turn Interactions (T²PAM) and introduce an Optimum-Referenced One-Step Adaptation mechanism for optimization. Experimental results show that this framework consistently improves model performance across multiple datasets and model backbones. The authors also provide detailed experimental analyses.

### Strengths
1.	The proposed framework is easy to follow, and the paper is overall well-written and logically organized (although some concepts are deferred to the appendix).
2.	The method is theoretically well-grounded and validated across several models and datasets, demonstrating its generality and robustness.

### Weaknesses
In my view, the main content of the paper may contain too much formalism and theoretical exposition. Many of the stated results are self-evident or intuitively understandable, yet the paper devotes substantial space to formal correctness proofs. Such material would be better placed in the appendix. For instance, continual preference-based supervision in multi-turn dialogue naturally leads to incremental improvement. This is expected. I am more interested in the implementation details that make this practically effective.

1.	A key issue lies in how reliable user feedback signals are obtained. How does the system know whether a user’s response or preference is correct (as illustrated in Figure 1)? The appendix mentions two reward models, but it remains unclear how these rewards are used to update the model during inference. The reliability and correctness of these feedback signals are not guaranteed, especially in the LLM-based method.
2.	Another concern is maintaining naturalness and coherence during test-time optimization. While iterative improvement makes sense in tasks like math or reasoning problems (where correctness is binary), in open-ended dialogue, the “right” answer may shift with context. If the model can continuously receive accurate preference feedback, doesn’t that imply it already has access to the correct answers?
3.	The experimental setup and implementation process should be described in more detail in the main text, as they are crucial for understanding the method’s validity. Similarly, the Related Work / Background section should be moved to the main content of the paper for better contextual understanding.
4.	The proposed mechanism is evaluated primarily on models below 8 B parameters. It is unclear whether the method scales to larger models. In addition, the explanations of “LM/HS” and “R” and “M” are vague, even after reading Appendix D.5, the discussion feels fragmented. Overall, the theoretical framework seems elegant, but the practical implementation lacks a clear connection to it.

### Questions
1.	In Figure 4, what specific RL algorithm is used? How does ROSA (Algorithm 1) differ from the “RL” shown in Figure 4? RLHF mentioned in Appendix E 4.1 cannot adapt to this situation originally.
2.	The statistical analyses focus on overall optimization performance. I am curious about per-sample behavior: (a) What is the average number of optimization steps required for a sample to go from incorrect to correct? (b) Once a response becomes correct, is any correctness check performed to prevent further unnecessary optimization?
3.	Since this method relies on accurate user preference signals as rewards, most experiments are conducted on mathematical or reasoning datasets. How would the method perform on open-domain or less objective tasks where user preferences are ambiguous? As mentioned above, the assumption of reliable preference feedback may not hold in such cases.

### Soundness
3

### Presentation
2

### Contribution
3
