# AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 4, 8, 8

## Abstract
As LLMs are increasingly deployed in real-world applications, ensuring their ability to refuse malicious prompts, especially jailbreak attacks, is essential for safe and reliable use. Recently, activation steering has emerged as an effective approach for enhancing LLM safety by adding a refusal direction vector to internal activations of LLMs during inference, which will further induce the refusal behaviors of LLMs. However, indiscriminately applying activation steering fundamentally suffers from the trade-off between safety and utility, since the same steering vector can also lead to over-refusal and degraded performance on benign prompts. Although prior efforts, such as vector calibration and conditional steering, have attempted to mitigate this trade-off, their lack of theoretical grounding limits their robustness and effectiveness. To better address the trade-off between safety and utility, we present a theoretically grounded and empirically effective activation steering method called AlphaSteer. Specifically, it considers activation steering as a learnable process with two principled learning objectives: utility preservation and safety enhancement. For utility preservation, it learns to construct a nearly zero vector for steering benign data, with the null-space constraints. For safety enhancement, it learns to construct a refusal direction vector for steering malicious data, with the help of linear regression. Experiments across multiple jailbreak attacks and utility benchmarks demonstrate the effectiveness of AlphaSteer, which significantly improves the safety of LLMs without compromising their general capabilities. Our codes are available at \url{https://anonymous.4open.science/r/AlphaSteer-929C/}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a simple, effective, and principled method called AlphaSteer, which steers the activations of LLMs to refuse malicious prompts while retaining maximum utility for benign ones. Specifically, AlphaSteer defines an explicit objective for this goal and derives an efficient approach to achieve it without exhaustively retraining model parameters for safety alignment. The experimental results demonstrate the effectiveness of the proposed method.

### Strengths
- The presentation is clear and easy to follow.

- The idea is simple yet principled: the goal of this paper is rigorously defined, and the proposed approach to achieve it is both efficient and well-justified. In particular, introducing the concept of having zero effect on benign prompts (rather than explicitly maximizing utility, such as the log-likelihood of outputs) is a reasonable formulation.

- The experimental results are strong, at least within the scope of the setups presented in this paper.

### Weaknesses
I think this paper is already strong, but the following points could further improve it:

- The proposed method appears lightweight (mainly involving SVD computation and matrix multiplication in a full-batch manner). However, in my view, it is still data-driven. It would therefore be helpful to compare this approach with a fully data-driven baseline — for example, a simple supervised fine-tuning model trained to generate refusals for malicious prompts in the same dataset $\mathcal{D}_m$. Although such a baseline might overfit $\mathcal{D}_m$, it would still highlight the advantages of the proposed method. Even if the baseline performs better, AlphaSteer would remain preferable due to its efficiency.

- AlphaSteer introduces some additional computational overhead (which appears marginal), but it would be useful to discuss this overhead in more detail — particularly in comparison to the baseline (i.e., only computing the refusal vector $r$).

- In certain cases (e.g., Llama-3.1-8B-Instruct on Math and GSM8K), AlphaSteer actually improves utility. This suggests that AlphaSteer might have a regularization effect (e.g., the input $h_b$ being influenced by $\tilde{\delta}$ when moving out of the null space). Providing intuition or analysis for this phenomenon could further support the claim that AlphaSteer enhances both safety and utility.

- The paper studies the effect of the steering strength $\lambda$ in Figure 11. Could an optimal $\lambda$ be derived using a similar objective formulation?

### Questions
See the weaknesses

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an activation steering method with a learnable refusal vector to defend against jailbreak attacks in LLMs. The learnable vector is optimized to balance the trade-offs between utility and safety. Experiments are carried out on three open-source LLMs with recent jailbreak attacks and utility benchmarks to show the effectiveness of the proposed defense.

### Strengths
- The proposed defense achieves  a better utility score (even slightly better than standard models on average).
- The paper shows theoretical grounding on its optimization of the learnable refusal vector.
- The proposed method achieves a better defense success rate on average against recent jailbreak attacks.
- The paper is well written and easy to read.

### Weaknesses
- The contribution may be limited as there are other existing learnable activation-steering methods considering before ICLR submission deadline.
The general learnable activation steering methods include:

[1] https://arxiv.org/abs/2505.20309v2 (version 1 released in May 2025)

[2] https://arxiv.org/abs/2506.03292 (hypernetwork-based steering)

[3] https://aclanthology.org/2024.findings-emnlp.479.pdf

The reviewer skips the paper after September 2025.

- The experiments are not rigorous. Better attacks, such as "do anything now" [a], AdvPrompter [b], are not used for evaluation.

[a]https://arxiv.org/abs/2308.03825

[b]https://arxiv.org/pdf/2404.16873

- Case study (RQ3) should be an in-depth analysis rather than showing an example of (ReNeLLM).
- The generalization ability of the learned refusal vector is not clearly explained, although there are experimental results on generalization without math data in the appendix (D.4).

Minor:
- The caption of Fig. 4 is missing.
- The small graphs in the supplementary materials are not readable.

### Questions
- Activation steering is known to introduce safety and alignment risks. How does the proposed method guarantee not to introduce other safety and alignment risks other than jailbreak attacks at hand?

- The steering vector may not generalize well beyond the defined settings or prompt types. What is the expected generalization?

- How does the proposed method guarantee the learned steering direction is reliable? (Fidelity)

- The design of the prompts may affect the steering direction. What is the variance?
How $D_b$ and $N_m$ are constructed?

- The limitations say the effectiveness is unknown for large reasoning models. What about small reasoning models such as Phi-3?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
AlphaSteer introduces a learnable activation-steering method that keeps a model’s normal behavior intact while strengthening its tendency to refuse harmful requests. It first carves out a space that represents benign behavior and minimizes any steering there, then learns an adaptive “refusal direction” from activation data so the model gently shifts toward safe responses only when prompts are malicious. Across multiple open instruction models and a range of common jailbreak attacks, it raises defense success while largely preserving compliance and standard task performance, outperforming prior refusal-vector baselines.

### Strengths
1. The method grounds activation steering in a clear linear-algebraic framework: (1) preserve utility by projecting benign activations into a learned (near) null-space, and (2) enhance safety via an adaptive, data-driven refusal vector estimated in closed form.
2. Across diverse jailbreak families, the approach delivers state-of-the-art (SOTA) defense success on malicious prompts while maintaining (or minimally impacting) compliance and standard-task performance on benign prompts—consistently outperforming refusal-vector baselines and contemporary steering methods under comparable settings.
3. Clear geometry-focused visualizations (activation trajectories, norm-separation) and ablations (layer choice, steering strength, linear vs. MLP) justify each design choice and make the mechanism easy to audit and reproduce, strengthening both clarity and credibility.

### Weaknesses
1. The proposed method includes introduction of the computation of null-space projection matrix, but does not show whether the new computation is costly. For showing effective practical usage, it would help to compare computation with existing baselines. For example, Surgical [1] offers Inference time and Memory comparison.
2. The evaluation solely depends on GPT-4o model as LLM-for-judge for DSR (Defense Success Rate) and CR (Compliance Rate), while having no justification for the model selection. Although it is based on GPT-4 not GPT-4o, WIldGuard[2] shows that guard-specific models can serve as better judges. You might want to include other guard-specific models as independent judges, and report how the results change for further validation.


[1] Wang, X., Hu, C., Röttger, P., & Plank, B. (2024). Surgical, cheap, and flexible: Mitigating false refusal in language models via single vector ablation. arXiv preprint arXiv:2410.03415.

[2] Han, S., Rao, K., Ettinger, A., Jiang, L., Lin, B. Y., Lambert, N., ... & Dziri, N. (2024). Wildguard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms. Advances in Neural Information Processing Systems, 37, 8093-8131.

### Questions
1. Please state more details about the content and intent deduplication method in C.1.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes AlphaSteer, a theoretically grounded activation steering method that addresses the safety-utility trade-off in LLM defenses against jailbreak attacks. Unlike conventional activation steering that indiscriminately applies refusal direction vectors to all prompts, AlphaSteer learns a transform matrix which produces steering vectors which are nearly zero for benign prompts (via null-space constraints for utility preservation) while maintaining refusal vectors for malicious prompts (via linear regression for safety enhancement). The method requires no additional post-training and demonstrates significant improvements in safety across multiple jailbreak attacks while maintaining general model capabilities.

### Strengths
- Strong theoretical foundation with principled learning objectives based on null-space constraints and linear regression, providing clear mathematical grounding for the approach.
- Addresses a critical limitation of existing activation steering methods with an elegant solution that treats benign and malicious prompts differently.
- Comprehensive experimental evaluation across multiple jailbreak attacks (GCG, AutoDAN, PAIR, etc.) and utility benchmarks demonstrating consistent improvements.
- Well-written with clear motivation and flow
- Strong results vs existing baselines

### Weaknesses
- The paper would benefit from more theoretical analysis of when and why the null-space constraint successfully preserves utility, and under what conditions it might fail. 
- I think the paper would benefit from more details on how AlphaSteer is learned for the experiments to give a better sense of cost/scalability

### Questions
Does a transform matrix always have enough capacity to adequately learn when the difference between malicious and benign? Is AlphaSteer easy to trick if the attacker is aware ahead of time?

How does AlphaSteer perform against adaptive attacks where an adversary has knowledge of the learned steering vectors? Can the null-space constraints be circumvented by adversaries?

What is the computational overhead of learning AlphaSteer vs existing methods?

Under what conditions does the null-space constraint fail to preserve utility? Are there specific types of benign prompts that the authors observe still lose utility after AlphaSteer? How much is this affected by things like training set size.

[Figure 1] How is this plot created? By my understanding at this point in the paper, should the vanilla benign/malicious distributions be the same between Surgical and AlphaSteer? To me it looks like the benign vanilla distributions are different for surgical and alphasteer, why is that?

[98] Not a big deal, but it says recent studies and the first citation is from 1969.

[101] Extra space?

[366] This claim is too strong as Table 1 contricts the fact that 'AlphaSteer yields superior defense success rates across all the jailbreak attacks'

[Table 1 and 2] Can you discuss why you believe AlphaSteer underperforms on certain benchmarks/models compared to the baselines?

[411] The CAST papers claims that there is only a small increase in refusal rate for harmless prompts, can you explain why it is misclassifing math problems as malicious prompts, this seems surprising to me.

### Soundness
4

### Presentation
3

### Contribution
4
