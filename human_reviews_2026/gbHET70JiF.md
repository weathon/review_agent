# A Mixture of Linear Corrections Generates Secure Code

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large language models (LLMs) have become proficient at sophisticated code-generation tasks, yet remain ineffective at reliably detecting or avoiding code vulnerabilities. Does this deficiency stem from insufficient learning about code vulnerabilities, or is it merely a result of ineffective prompting? Using representation engineering techniques, we investigate whether LLMs internally encode the concepts necessary to identify code vulnerabilities. We find that current LLMs encode precise internal representations that distinguish vulnerable from secure code--achieving greater accuracy than standard prompting approaches. Leveraging these vulnerability-sensitive representations, we develop an inference-time steering technique that subtly modulates the model's token-generation probabilities through a mixture of corrections (MoC). Our method effectively guides LLMs to produce less vulnerable code without compromising functionality, demonstrating a practical approach to controlled vulnerability management in generated code. Notably, MoC enhances the security ratio of Qwen2.5-Coder-7B by 8.9\%, while simultaneously improving functionality on HumanEval pass@1 by 2.1\%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper empirically shows that, from some common CWEs, large code models possess latent representations that distinguishes vulnerable code from secure code, even if they can’t reliably explain it when prompted: simple linear probes trained on hidden activations can classify vulnerable vs fixed code across specific CWEs with much higher accuracy than direct prompting and even beat prior automated vulnerability detectors on both function snippets and full real-world repos, meaning vulnerability information is already linearly separable in the model’s internal state. Building on that, the authors introduce Mixture of Corrections (MoC), an inference-time activation-steering method: for each CWE, they learn a correction direction that shifts the model’s hidden state from “vulnerable-like” toward “secure-like,” then at generation time they (1) detect when the current hidden state looks risky using the probe and (2) add the relevant correction vectors (with decay over time), leaving the model alone when it’s already on a safe path. This conditional steering improves both security and functionality: for example, on Qwen2.5-Coder-7B, the share of generations with no CodeQL-detected vulnerabilities increases by nearly 9 percentage points and HumanEval pass@1 also goes up by about 2 points, which is notable because most “secure coding” interventions hurt correctness. The approach is lightweight (a few extra vector ops per token, no full fine-tune), sometimes even transfers across related model variants, and can also be inverted to intentionally induce vulnerabilities (useful for red-teaming), though the authors acknowledge dual-use risk and limits of static analyzers like CodeQL for ground truth.

### Strengths
- The paper convincingly shows that vulnerability information is already encoded in the internal activations of code LLMs (**for certain CWE types covered in the experiments**), because simple linear probes can reliably distinguish vulnerable vs fixed code and even outperform prior static/dynamic vulnerability detection systems on both function-level and repo-level datasets.
- The paper proposed a practical steering approach MoC, leveraging the observations, to intervene the generation process and make code LMs generate safer code, which is light-weights and has certain interpretability compared to existing solutions.
- The proposed approach improve the security of generated code without harming the utility of code LMs.
- The paper shows that in some cases steering vectors trained on one model can be applied to a related model and still improve security, which hints at reuse across a model family and lowers the cost of adoption.

### Weaknesses
- The method works per CWE type, which is a strength in targeting, but also a scaling bottleneck. Each new vulnerability class needs labeled vulnerable/fixed pairs, a trained probe, and a steering vector.
- Real-world code has long-tail, app-specific logic flaws (auth bypass, race conditions, crypto misuse) that may not map cleanly to the benchmark CWEs the paper evaluates. The paper doesn’t fully address how this would generalize beyond the covered CWEs.
- The experiment setting seems to have removed CWE categories with training samples less than 100 (line 266-267). This will remove most of the hard CWEs that are actually the critical part of research to push the boundary of secure code generation, compared to the relatively simple ones that can be fixed by adding/modifying a line, e.g., substituting an unsafe serialization API with a safe one. Note that this is orthogonal to length of the code. Missing complex types of CWE will dramatically influence the overall experiment results and cause bias (**potential overclaiming**) in the conclusion that LLMs possess knowledge in differentiating vulnerable code and safe code.
- The work argues MoC is practical for “real devs,” but we don’t see a user study measuring whether the outputs are actually easier to review, have fewer subtle logic bugs, or reduce time-to-fix.
- The MoC approach is intrinsically similar to some MoE-style adapters. Maybe some kind of baselines need to be added here.

### Questions
1. Can you provide a deeper analysis on the vulnerability detection and steering ability of MoC on different vulnerability types, especially those fundamental different from the frequent ones, e.g. (i) not purely local and (ii) not trivially caught by static analyzers? In particular, can you include a CWE that depends on multi-function, rather than a single usage site?
2. Since the steering approach proposed in this work falls into a broader domain of lightweight adaptation of LMs, can you add some comparison with existing parameter efficient finetuning approaches (especially MoE-style) under similar settings and discuss their relationship, e.g. [1,2].

> [1] Wang, Renzhi, and Piji Li. "Lemoe: Advanced mixture of experts adaptor for lifelong model editing of large language models." arXiv preprint arXiv:2406.20030 (2024).\
> [2] Wu, Xun, Shaohan Huang, and Furu Wei. "Mixture of LoRA Experts." The Twelfth International Conference on Learning Representations.

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
3

### Summary
This paper leverages representation techniques and inference-time steering methods to enhance large language models’ capabilities in bug detection and secure code generation. It employs linear probing on hidden states to detect vulnerabilities and propose four types of correction vectors: the Difference of Group Means, the Normal Vector, the PCA-Reduced Normal Vector, and NN-based Dynamic Corrections.

The evaluation shows that compared with prompt-based methods, linear probing improves detection accuracy. In secure code generation, the proposed Mixture of Corrections (MoC) increases the security rate without compromising functionality.

### Strengths
- Employs linear probing on hidden representations, achieving better bug detection accuracy than prompt-based baselines.
- Introduces a Mixture of Corrections that improve code security while maintaining functionality.
- Demonstrates that the learned correction vectors exhibit a certain degree of cross-model transferability.

### Weaknesses
- Although four types of corrections are proposed, the paper does not clearly describe how they are combined for a given bug type. Are all four used simultaneously, or is only one applied each time?
- Each correction is trained specifically for one bug type (CWE). This means for multiple bug types, separate probes and corrections must be trained, potentially increasing computational overhead and raising questions about interaction or interference between corrections when multiple vulnerabilities coexist.

### Questions
* During inference, how are the different corrections mixed? Is only one type applied at a time, or are they linearly combined?
* In Table 6, the performance gap between different correction methods is relatively small. What are the practical advantages of introducing PCA-based or NN-based dynamic corrections? Could the authors provide guidelines on when to prefer each type?
* Besides vulnerability type, are the trained corrections also language-dependent? If so, would this require retraining separate probes and corrections for each (CWE type, programming language) pair?

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
4

### Summary
The paper studies the problem of vulnerability detection and secure code generation using LLMs. The first key claim is that linear probes over internal features of LLMs can be used to detect vulnerable code. Moreover, these features can be steered linearly to generate secure code. A Mixture of Corrections (MoC) method is proposed to steer a feature in the direction of a collection of "correction vectors" (one for each vulnerability type) -- the correction vector for a vulnerability type is included if the associated linear probe detects the feature as vulnerable. Experiments show that linear probing outperforms neural detection baselines and standard prompting with LLMs on vulnerability detection. Further, MoC improves the security ratio for three Qwen-x models (and reduces it in the security weakening setup).

### Strengths
- The paper studies the important problems of vulnerability detection and secure code generation with LLMs.
- The paper is well-written and the key ideas are easy to understand. 
- The use of linear probing to detect vulnerabilities is novel.

### Weaknesses
- Some of the steering methods considered in Section 3.2.1 have been proposed for other natural language tasks (difference of group mean [1], normal vector of the decision boundary [2]). 
- The secure code generation task reports the security ratio metric but does not report the correctness of the outputs after steering with MoC on the SVEN Test Set (Table 6). There could potentially be a trade-off between the security ratio and the correctness of generation after steering (similar to the accuracy fluency tradeoff observed by other steering works [3]). At the least, I would expect a short manual analysis of the generated outputs with a discussion on the quality / correctness of the generated secure code.
- The expeirments do not include a "control" benchmark from an unrelated task, i.e., a benchmark with no vulnerabilities. Ideally, this should be used to demonstrate that MoC has no adverse affect on prompts / examples which do not include any vulnerabilities (and were not included in training the linear probe).

Overall, I think that the novelty in terms of techniques used is limited (albeit novel in the context of vulnerability detection and secure code generation). The steering experiments would benefit from some qualitative examples of the generated code and reporting the accuracy of the correctness of the generated secure code. The detection experiments would benefit from the inclusion of a control benchmark.

[1] Ole Jorgensen, Dylan Cope, Nandi Schoots, and Murray Shanahan. Improving activation steering in language models with mean-centring. arXiv preprint arXiv:2312.03813, 2023.

[2] Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36: 41451–41530, 2023.

[3] Guardieiro, V., Stein, A., Khare, A., & Wong, E. (2025). Instruction Following by Boosting Attention of Large Language Models. Mechanistic Interpretability Workshop at NeurIPS 2025. Retrieved from https://openreview.net/forum?id=xDyJVMnab8.

### Questions
I will summarize my questions from the weaknesses section above (please refer to that section for more details):
1. How are the proposed steering methods (the group mean and the normal vector) different from those proposed in [1] and [2]?
2. Could you share some qualitative examples of the outputs after steering?
3. Are the probes trained and tested on splits from the same dataset? If yes, how would the results change if the probe is trained on SVEN and tested on PreciseBugs? This is important to know whether this can be applicable to code snippets in the wild (i.e., not from a specific dataset).
4. Are the probes and steering vectors trained on hidden states from the residual layer (after a transformer block) or attention blocks? RQ4 mentions attention blocks while Section 3.1 mentions transformer blocks.

### Soundness
3

### Presentation
3

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
The paper proposes a  framework called Mixture of Corrections (MoC) to improve the security of code generated by LLMs. The paper investigates whether LLMs internally encode information distinguishing secure from vulnerable code, finding that linear probing can successfully access these latent representations with high accuracy, unlike standard prompting methods. Leveraging this insight, MoC introduces an inference-time steering technique that computes and conditionally applies linear correction vectors to subtly modulate the LLM's token generation probabilities.

### Strengths
- MoC is an inference-time steering technique that effectively guides LLMs to produce less vulnerable code. Notably, it enhanced the security ratio while simultaneously improving functionality on HumanEval.
- The method is a practical approach to controlled vulnerability management that does not require costly retraining or extensive prompt engineering.
- The guiding correction vectors sometimes transfer across models, yielding a computationally efficient way to harden models that are not specifically trained on code data.

### Weaknesses
- The primary evaluation tool, CodeQL, exhibits inherent limitations in both accuracy and computational efficiency. The paper notes a scarcity of robust automated evaluation methods for code generation, and finding that using an LLM-as-a-judge is unsuitable due to poor performance in code vulnerability detection
- The paper requires fully open-source access to the model's internal representations and parameters. This dependency on white-box access limits the practical applicability of MoC to proprietary or closed-source LLMs, which are increasingly common in production environments.
- While correction vectors show some transferability across smaller models (3B and 7B), this transferability does not hold for larger models, where the functionality of the target model is harmed
- Figure 2 is not very intuitive and hard to understand also with the text as additional explanation.

### Questions
How sensitive are the final Mixture of Corrections (MoC) performance metrics to slight variations in the choice of the optimal attention block?

### Soundness
3

### Presentation
2

### Contribution
3
