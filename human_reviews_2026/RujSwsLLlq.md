# FILOsofer: A TEE-Shielded Model Partitioning Framework Based on Fisher Information-Guided LoRA Obfuscation

- Decision: Reject
- Scores: 2, 8, 6, 2

## Abstract
On-device machine learning makes DNN models visible as a white-box to users, leaving them susceptible to stealing attacks. Trusted Execution Environments (TEEs) mitigate this risk by isolating model execution, but executing entire models within TEEs is inefficient and slow. To balance security and performance, TEE-Shielded DNN Partitioning (TSDP) executes privacy-insensitive parts on GPUs while confining privacy-critical components within TEEs.

This work demonstrates that existing TSDP approaches remain vulnerable under large query budgets (e.g., $>$500 queries) due to non-zero information leakage per query, enabling attackers to gradually construct accurate surrogate models. To address this, we propose FILOsofer (Fisher Information-Guided LoRA Obfuscation), which uses Fisher Information to perturb a small subset of key weights, rendering the exposed weights inaccurate and producing uniform outputs, thereby safeguarding the model even under unlimited queries. We then design a novel cross-layer LoRA to efficiently restore authorized-user performance, storing only LoRA parameters in the TEE to eliminate information leakage while minimizing the performance overhead. This lightweight design also allows seamless extension to LLMs. We evaluate \sys in both experimental and real-world settings, achieving over 10× improvement in security and more than 50× reduction in computational overhead compared to prior TSDP solutions.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new protection method that obfuscates model weights while protecting their low-rank components in a TEE. The approach defends against model-stealing attacks with lower inference overhead.

### Strengths
It leverages Fisher Information to quantify weight importance and compares against multiple baselines to demonstrate its superiority.

### Weaknesses
1. The selection of representative baselines lacks justification. In Table 1, Magnitude is chosen for non-linear layers, yet ShadowNet is a newer and more relevant method. For model obfuscation, NNSplitter is selected, but it has been widely shown to be insecure by GroupCover, and the more secure GroupCover is not included.

2. The contributions are unclear. The claim “We conduct a systematic evaluation of existing TSDP approaches” has already been done by TEESlice, so it is not a novel contribution. The paper should clarify how the evaluation differs from TEESlice and report GroupCover’s results under the same attacks.

3. Section 4 lacks implementation details. It is unclear whether the query numbers in Figure 1 and the “modest budgets (e.g., 500 queries)” apply per class or to the entire dataset. The settings for baseline methods—such as hyperparameters and TEE FLOPs—are not described. The attack results may vary depending on the hyperparameters. The meaning of the “ideal line” is also ambiguous, it should report black-box attack performance rather than 10% or 1%.

4. Regarding the method, it is unclear how the target label ( $L_t$ ) is selected—randomly or a chosen one? Protecting fewer than five layers implies that the remaining weights remain in plaintext, raising security concerns. There is no justification for how the LoRA-based recovery maintains accuracy. It is also unclear how output privacy is preserved—authorized users may still be adversarial, and the true label is returned outside the TEE. More details are needed on how attacks are performed against the proposed method.

5. In the evaluation, it is questionable why FILOsofer yields lower attack performance than black-box, since black-box is generally considered the most secure (TEESlice, under 5000 queries, performs nearly the same as black-box). The efficiency evaluation appears theoretical rather than based on real runtime measurements. In Table 5, only TrustZone time is shown; full end-to-end runtime—including TrustZone, GPU, and data transmission—should be compared.

Minor: The code repository link was expired during my review.

### Questions
Please see the weakness.

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
This paper proposes FILOSOFER, a TEE-shielded model partitioning framework that uses Fisher Information-guided weight obfuscation and cross-layer LoRA recovery to prevent model stealing on edge devices while keeping inference fast.

### Strengths
1) Tackles a real security gap in TEE-based model partitioning (information leakage under many queries).
2) Novel use of Fisher Information to select and perturb critical weights.
3) Cross-layer LoRA recovery offers strong utility with very low overhead.
4) Evaluation is comprehensive with multiple models, datasets, Jetson hardware, and adaptive attacks.
5) Practical and lightweight design suitable for real-world edge and LLM deployment.

### Weaknesses
1) Experiments lack recent  model-stealing baselines beyond KnockoffNet.
2) Analysis limited to query based attacks, side-channel attacks which are very prevalent should also be explored.

### Questions
1) The paper clearly identifies a key weakness in prior TSDP frameworks, which is residual information leakage under repeated queries. For this the authors propose an elegant, low-cost solution. The Fisher-guided obfuscation idea is clever and fits well with LoRA’s lightweight recovery, making the system both secure and fast.
2) However, the work needs stronger theoretical justification for why Fisher perturbation guarantees output uniformity and resistance to adaptive querying.
3) The evaluation is broad but mostly limited to one type of attacker; it would be more convincing to test stronger, query-adaptive or side-channel-aware methods as well.
4) Lastly, the LLM section feels preliminary. I feel expanding to larger models or analyzing memory–latency scaling would improve completeness.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents FILOsofer, a framework for protecting deep neural networks deployed on edge devices from model stealing attacks. The authors demonstrate that existing TSDP methods remain vulnerable when attackers have large query budgets, as they gradually leak information through accurate outputs. FILOsofer addresses this by using Fisher Information to selectively change critical weights, forcing the model to produce uniform outputs, while a lightweight cross-layer LoRA module stored in the TEE restores the model performance to authorized users. Experimental results show FILOsofer achieves 10x better security against model stealing with 50x lower computational overhead compared to prior TSDP solutions.

### Strengths
1. Exciting application of TEEs in protecting DNNs
2. Very strong motivation of the work, and good presentation of background (some exceptions mentioned below)
3. Comparison with SOTA related approaches
4. Both theoretical and practical execution, with on-device experiments on a ARM based Jetson machine

### Weaknesses
1. Your thread model assumes that the adversary can infer the models architecture by monitoring the weights in REE space. Is that reasonable? Related works that you mention protect one or even more layers inside the TEE. Are you considering a model without the protected layers? If not, how can you know the architecture of the protected layers (length, and number of protected layers at the minimum).
2. While as mentioned above, the motivation is clear and background section provides good info to the reader, it took me a while to understand the FILOsofer aims to protect the model parameters from an attacker. I suggest you have a quick reference of your thread model earlier in the manuscript. Same for L061 when you mention that the model. on GPU remains highly accurate, unless you know the related work it is not clear why this is important for you.
3. While I appreciate the implications section on applying FILOsofer on LLMs, I feel it was a rushed evaluation that comes out of the blue in the manuscript. There is no information about how they used the LLM for a classification task (and what exactly the task was). What was the reason that layer 15 was chosen, was it the most informative? reported the best accuracy? Have you tried other layers? What type of data have you used and what were the exact system prompts?

### Questions
1. Can you please clarify how an adversary can infer the architecture of the TEE protected layers?
2. Can you provide additional details about the application of FILOsofer on LLMs, as per my W3 comment?

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
The paper introduces FILOsofer, a defense against model stealing in on-device deployments where a user (as an atatcker) can query the target model deployed on their device, see predictions and model weights. The proposed defence partitions the model between a Trusted Execution Environment (TEE) and the Rich Execution Environment (REE):
- A cross-layer LoRA kept and executed inside the TEE protecting confidentiality of model weights
- The Fisher-guided obfuscated weights kept in the REE. 

The two components are trained with a constraint-aware joint objective to balance security and utility: ensuring the obfuscated backbone resists trivial recovery while the TEE-resident LoRA restores accuracy for authorized use.

### Strengths
Empirically analyzing vulnerabilities of TEE-Shielded DNN Partitioning to model stealing attacks.

### Weaknesses
1. Section 3 appears to combine two sources of attacker leverage: (i) the model’s architecture/weights in REE so the adversary first infers the protected model’s architecture and weights from publicly available models, and (ii) carefully chosen queries to collect outputs so the attacker issues limited queries on carefully selected inputs and records the corresponding outputs. The proposed backbone obfuscation does not help with (ii). Line 179-1180 discusses results of existing methods as follows:the partitioned model executed on GPUs remains accurate, enabling attackers to initialize surrogate models effectively. However, no evidences of that exist in Figure 1 which only studies the impact of number of queries.

2. The obfuscation optimisation aims to make the obfuscated model’s output to be input-independent.

3. The proposed defences requires joint-training without any discussion on data for this training. 


4. TEE-Shielded DNN Partitioning has been extensively studies in the literature. 

5. This paper mixes privacy of training data and IP/confidentiality of the model. DP and MPC do not address this problem of model stealing, statements in intro are problematic such as  ``To mitigate these security risks, researchers have explored two defense strategies: (i) Cryptographic approaches: Methods such as Multi-Party Computation (MPC) (Juvekar et al., 2018), Homomorphic Encryption (HE) (Gilad-Bachrach et al., 2016; Kim et al., 2022), and Differential Privacy (DP) (Abadi et al., 2016; Girgis et al., 2021) aim to safeguard both input data and model parameters through algorithmic guarantees. D''

6. The paper is not motivated very well: ``but executing entire models within TEEs is inefficient and slow'' --> It is not really the case anymore given recent GPUs with TEE supports

7. line 058: confidentiality(Zhou et al., 2023; Sun et al., 2024). --> typo

### Questions
1. Is the model stealing due to output or model weights? 

2. If the goal is to make the obfuscated model’s output to be input-independent, why not just using randomly inisialised weights? why do you need Fisher Information?

3. Which datasets you need for the joint-training algorithm? Do you need the whole training dataset to protect against model stealing? If so, how practical and costly it is?

 4. Which layers are most sensitive layers?

5. How robust it is to an informed adversary?

### Soundness
2

### Presentation
2

### Contribution
2
