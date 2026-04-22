# SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Large Language Models (LLMs) have revolutionized numerous fields, yet their adaptation to specialized tasks in privacy-sensitive domains such as healthcare and finance remains constrained due to the scarcity of accessible training data caused by stringent privacy requirements. Secure Multi-party Computation (MPC)-based privacy-preserving machine learning provides theoretical guarantees for the privacy of model parameters and data. However, its application to LLMs has been predominantly limited to inference, as fine-tuning introduces significant efficiency challenges, particularly in backward propagation, optimizer, and self-attention operations. To address these challenges, we propose SecP-Tuning, the MPC-based framework designed for efficient, privacy-preserving prompt tuning of LLMs. SecP-Tuning innovatively integrates Forward-only Tuning through the ''data owner-server interaction" paradigm, effectively removing the need for privacy-preserving computations in backward propagation and optimization processes. Furthermore, it devises an efficient privacy-preserving Random Feature Attention, effectively mitigating the computational complexity of softmax-based self-attention and circumventing MPC-incompatible nonlinear operations. Experimental results demonstrate that, compared to full-Parameter Supervised Fine-Tuning and gradient-based prompt tuning, SecP-Tuning achieves approximately 12$\times$ and 16$\times$ end-to-end acceleration, as well as 17$\times$ and 20$\times$ reductions in communication overhead, respectively. Moreover, it delivers performance comparable to gradient-based methods across multiple few-shot tasks. Additionally, the ''black-box/API-style" privacy-preserving tuning paradigm of SecP-Tuning effectively avoids memory leakage risks caused by gradient/parameter transmission, thereby striking an optimal balance between privacy, efficiency, performance, and deployability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents SecP-Tuning, an MPC-based framework for privacy-preserving prompt tuning of large language models (LLMs). The authors aim to address the prohibitive computation and communication overheads of privacy-preserving fine-tuning (PFT) under Secure Multi-Party Computation. They combine two main techniques: (1) Forward-only Tuning (FoT) with a gradient-free optimizer (CMA-ES) to avoid backward propagation, and (2) Random Feature Attention (RFA) with a privacy-preserving cosine protocol to replace softmax attention. Experimental results on five NLP datasets show that SecP-Tuning achieves substantial efficiency gains  while maintaining or improving performance compared to both supervised fine-tuning (SFT) and gradient-based prompt tuning.

Overall, the paper proposes an interesting and timely attempt to make privacy-preserving tuning of LLMs feasible, though several technical details and justifications require clarification and deeper analysis.

### Strengths
1. **Novel integration of privacy and efficiency**. The paper successfully combines MPC with prompt tuning through forward-only optimization, reducing both computation and communication costs while maintaining data confidentiality.

2. **Technical contribution in protocol design**. The development of the privacy-preserving cosine protocol and its integration into the Random Feature Attention module is well-motivated and technically sound, showing potential for broader cryptographic ML applications.

3. **Comprehensive experiments**. The authors evaluate both efficiency and accuracy across multiple datasets and network conditions (LAN/WAN), offering convincing empirical evidence of the method’s efficiency gains.

### Weaknesses
1. **Limited originality in efficiency gain sources**. The claimed reduction in communication and computation largely stems from existing components—namely, gradient-free optimization (FoT) and RFA—rather than innovations specific to the MPC algorithmic design.

2. **Scalability concerns**. The framework is only demonstrated under a 2-party MPC setup (semi-honest model). It remains unclear how it scales to more realistic multi-party or malicious adversarial settings, which are common in real-world PPML applications.

3. **Lack of theoretical justification**. The paper provides no formal convergence or performance guarantees for the CMA-ES optimizer in this MPC context, leaving uncertainty about stability and generalization.

4. **Insufficient baselines**. Comparisons exclude recent privacy-preserving fine-tuning methods based on differential privacy, homomorphic encryption, or hybrid PPML frameworks, which would help situate the contribution more clearly.

5. **Opaque optimizer choice**. The paper does not analyze why CMA-ES is particularly effective, nor how it compares to other gradient-free optimizers such as NES or SPSA. This weakens the empirical claims regarding its superiority.

6. **Overstated performance improvements**. The claim that a gradient-free, forward-only approach surpasses both SFT and gradient-based prompt tuning contradicts common understanding and demands stronger justification or ablation evidence.

### Questions
1. **On optimizer selection**: Why was CMA-ES chosen as the gradient-free optimizer? Have the authors evaluated other gradient-free methods (e.g., random search, NES, Bayesian optimization), and what is the empirical or theoretical rationale for preferring CMA-ES?

2. **On performance explanation**: The results indicate that SecP-Tuning not only matches but outperforms SFT, which is counterintuitive given that gradient-free methods generally converge slower and achieve lower accuracy. Could the authors clarify the mechanism behind this improvement?

3. **On abbreviation**: What does DFP in line 299 stand for? The abbreviation should be defined upon first use for clarity.

4. **On the statement in line 437**: The paper claims that “gradient-based methods such as SFT and prompt tuning inherently require the model developer to obtain shares of the updated parameters.” However, prompt tuning typically updates only prefix embeddings. Could the authors clarify why such updates would require parameter sharing under MPC?

### Soundness
3

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
SecP-Tuning is an MPC-based framework for efficient and privacy-preserving prompt tuning of large language models. It integrates Forward-only Tuning to avoid costly backward computations and introduces Random Feature Attention to reduce self-attention complexity. Experiments show up to 16× faster training, 20× lower communication cost, and better few-shot performance, achieving a strong balance between efficiency, accuracy, and privacy.

### Strengths
1. It addresses the high computational cost of attention during fine-tuning, significantly accelerating the overall training process.

2. 2-OUT-OF-2 module appears intriguing, as it separates and isolates the model’s information-capturing capability.

3. It adopts gradient-free optimization, improving the efficiency of forward computation.

### Weaknesses
1. The paper devotes a large portion of the Preliminary section to background explanations, while the introduction of its own method is relatively unclear, making it difficult to follow during reading.

2. Optimizing LLMs through gradient-free black-box methods is an outdated approach, and applying it to prompt-tuning tasks is not particularly novel. Its applicability within LLM scenarios remains quite limited. It also lacks horizontal comparisons with several key related works, such as the following studies.

Diao, Shizhe, et al. "Black-box prompt learning for pre-trained language models." arXiv preprint arXiv:2201.08531 (2022).

Lin, Zihao, et al. "Efficient federated prompt tuning for black-box large pre-trained models." arXiv preprint arXiv:2310.03123 (2023).

Sun, Jingwei, et al. "Fedbpt: Efficient federated black-box prompt tuning for large language models." arXiv preprint arXiv:2310.01467 (2023).

Zheng, Yuanhang, et al. "Black-box prompt tuning with subspace learning." IEEE/ACM Transactions on Audio, Speech, and Language Processing 32 (2024): 3002-3013.

All these approaches design and improve gradient-free prompt-tuning optimizers from different perspectives. The authors should further discuss how their proposed method compares with these works in terms of advantages and limitations.

3. There is too little theoretical analysis, especially regarding the privacy protection performance of Module 2-OUT-OF-2, which lacks strong guarantees. Providing more analysis or evaluation would further strengthen the contribution of this paper.

4. The experiments use only RoBERTa as the model, which is too small in scale. As a result, the stability and scalability of the experimental results are difficult to guarantee.

### Questions
1. In this paper, the performance of SFT is shown to be worse than that of the gradient-free training method. What is the reason for this phenomenon? Does it also occur in larger models? Could SFT be overfitting? The authors are encouraged to provide the relevant hyperparameters and loss curves to support their claim. Normally, such a result is quite unlikely, as gradient-based methods generally outperform gradient-free ones due to their much smaller update variance. If this outcome is indeed caused by overfitting, the authors should re-evaluate the SFT performance. If not, could the authors explain why the gradient-free method achieves better optimization results?

2. Can the authors independently evaluate the privacy protection strength of each part in the 2-out-of-2 module, which splits the original parameters into two components? If an attacker gains access to only one part and attempts to infer or guess the other, what would be the resulting degree of privacy leakage?

3. Could the authors provide a detailed horizontal comparison with the four works I mentioned above, as well as several other important studies on black-box prompt-tuning methods? I would like to understand what additional advantages this approach offers beyond the attention acceleration component.

4. Could the authors conduct additional experiments on larger models, such as 7B or 13B? Gradient-free methods can be efficiently supported on A100 GPUs, allowing for training even very large models.

I will re-score according to the rebuttal discussions.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to address the high communication and computational cost of fine-tuning large language models on private data using Secure MPC. Secure fine-tuning of LLMs through MPC is slow due to the complex, back-and-forth computations needed for the backward pass and the standard attention mechanism. The paper proposes SecP-Tuning, a framework that tries to solve both problems. It uses a *Forward-only Tuning* approach where the secure servers only run the model forward, while the private data owner handles the (now much simpler) tuning steps locally. This completely removes the need for a secure backward pass. It replaces the costly standard attention with a highly efficient variant called Random Feature Attention (RFA), which reduces attention's complexity from quadratic to linear.

### Strengths
The paper is well-written with a clear introduction of the problem and the room for their contributions. Despite various solutions in private LLM adaptation, Secure MPC is still highly desirable due to its formal guarantees. The authors clearly identify the gaps in the communication and computational overheads in Secure MPC. The figures clearly explain the problem and proposed solutions to non-subject-matter experts.

### Weaknesses
1. The experiments are conducted exclusively on ROBERTA, an encoder-only LLM. In 2025, the field is heavily focused on autoregressive  LLMs. It is unclear if the method, and particularly its strong accuracy results, will translate to these architectures. However, I do recognize that ROBERTA is still a strong model for many downstream NLP tasks. Being able to effectively perform Secure MPC is still a significant contribution.
2. The paper omits a comparison to non-secure, plaintext baselines in its result presentation. Including plaintext accuracy is essential to help readers understand the "utility cost" of privacy (if any). It would also help contextualize the gap that privacy-preserving methods still need to close.
3. Related to the first point, the reliance on a subset of GLUE-style classification tasks (SST-2, MRPC, RTE, etc.) is quite limited. While fine for proving the concept on classification, these benchmarks do not capture the full range of capabilities and tuning challenges of modern LLMs.

### Questions
1. Could the authors further explain the motivation for choosing RFA over other linear attention variations?
2. The performance of the gradient-based Prompt Tuning baseline is surprisingly low for simple tasks like SST. What are the settings here?
3. Regarding CMA-ES, would it still be as effective when the prompt length or the model's hidden dimension increases significantly, requiring a much larger search space?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SecP-Tuning, an efficient MPC-based framework for privacy-preserving prompt tuning of Large Language Models. It eliminates the need for privacy-preserving backpropagation and optimization by adopting a forward-only tuning strategy, where sensitive computations are done locally by the data owner. The framework further enhances efficiency through Random Feature Attention, which replaces costly softmax operations with a linear-time approximation. The approach is validated through a series of experiments that benchmark efficiency, accuracy on five language tasks, and deployability compared to prior methods, including full supervised fine-tuning (SFT) and gradient-based prompt tuning in MPC settings.

### Strengths
1. Through detailed empirical analysis (see Figure 1, Page 2) and system profiling, the authors convincingly highlight that backward propagation and softmax-based self-attention present severe efficiency barriers in MPC-based LLM fine-tuning.

2. SecP-Tuning demonstrates significant improvements in speed and communication overhead.

3. The framework operates under a black-box paradigm. This allows a data owner to perform tuning without the model developer ever receiving the updated parameters. This prevents the developer from potentially inferring private fine-tuning data from model updates, a risk present in gradient-based methods.

### Weaknesses
1. The optimizer used in the paper is Adam, which is different from the most popular optimizer, AdamW, in the LLM domain. Can the authors explain the reason? Besides, how do the authors choose the hyperparameters for the baselines? Does the few-shot learning with 1000 epochs cause overfitting?

2. The empirical validation is performed on small-scale settings. Though it is also a valuable setting, including large-scale settings could make the reader better understand the performance of SecP-Tuning. Besides, do the authors consider comparing with In-context learning that might also be good for small-scale settings? Some ICL methods are designed for privacy [1,2].

3. Experiments focus mainly on RoBERTa and text classification tasks. 

[1] Tang, Xinyu, et al. "Privacy-Preserving In-Context Learning with Differentially Private Few-Shot Generation." The Twelfth International Conference on Learning Representations.

[2] Wu, Tong, et al. "Privacy-Preserving In-Context Learning for Large Language Models." The Twelfth International Conference on Learning Representations.

### Questions
Please see weaknesses part

### Soundness
3

### Presentation
3

### Contribution
3
