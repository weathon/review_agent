# Guaranteed Generation from Large Language Models

- Decision: Accept (Poster)
- Scores: 8, 5, 8, 3

## Abstract
As large language models (LLMs) are increasingly used across various applications, there is a growing need to control text generation to satisfy specific constraints or requirements. This raises a crucial question: Is it possible to guarantee strict constraint satisfaction in generated outputs while preserving the distribution of the original model as much as possible? We first define the ideal distribution — the one closest to the original model, which also always satisfies the expressed constraint — as the ultimate goal of guaranteed generation. We then state a fundamental limitation, namely that it is impossible to reach that goal through autoregressive training alone. This motivates the necessity of combining training-time and inference-time methods to enforce such guarantees. Based on this insight, we propose GUARD, a simple yet effective approach that combines an autoregressive proposal distribution with rejection sampling. Through GUARD’s theoretical properties, we show how controlling the KL divergence between a specific proposal and the target ideal distribution simultaneously optimizes inference speed and distributional closeness. To validate these theoretical concepts, we conduct extensive experiments on two text generation settings with hard-to-satisfy constraints: a lexical constraint scenario and a sentiment reversal scenario. These experiments show that GUARD achieves perfect constraint satisfaction while almost preserving the ideal distribution with highly improved inference efficiency. GUARD provides a principled approach to enforcing strict guarantees for LLMs without compromising their generative capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work proposes the GUARD framework, which guarantees strict constraint satisfaction for generated outputs of LLMs.
It utilizes rejection sampling at inference to guarantee constraint satisfaction as well as a variant of DPG to tune the language model towards a policy close to the "gold standard" policy to ensure distributional closeness during training.

### Strengths
I found the problem and the proposed solution very interesting and intuitive.
The experiments seem reasonable, the results good and I see no major evaluation missing.
The auxiliary investigations in Fig 4 & 7 as well as Tab. 1 and 2 are very insightful.

### Weaknesses
I strongly agree that the major limitation of this approach in obtaining a filter b (also called verifyer in other areas), which I think is one of the most important research areas around LLMs right now.
However, I agree with the last paragraph in the main text, that this is out of scope for this work and any improvements in that direction will directly improve GUARD.

Minor:
* I understand why the baseline method described in line 172-176 and in App. A.2 has severe drawbacks, but for the simple constraint in the first experiment in Sec. 4.1, it would actually be a suitable baseline to compare to. 
Also I could see that it performs better than DPG with CAP in this setting, as the distribution is less degenerate I guess.
* It would be nice to have the two experiments performed with more than one model each, but I understand the computational demand arising from this.
* The claim in the third contribution is a bit too strong for me. GUARD itself does not improve the acceptance rate, it depends on the way a' is chosen right? So the improved DPG leads to higher acceptance rates.

Remarks (just as info for camera ready):
* SFT is not introduced on first occurance in line 200, but in 255/256.
* The order of Fig. 3 and 4 is reversed

### Questions
* What is V^* in line 119? It is never defined and I can only find it used once in line 801 where I don't get its usage.
* What is "ancestral sampling" - line 308?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work proposes GUARD, a method that combines an autoregressive proposal distribution with rejection sampling to guarantee the satisfaction of hard constraints in generated text while keeping the output close to the language model’s original distribution.

### Strengths
- **Clear Motivation and Intuition:** The authors provide clear intuitions about the need for distribution-preserving constraint satisfaction in language models and the challenges this entails. They thoroughly motivate their approach for achieving strict control over generations without substantially deviating from the original model’s output distribution.
- **Theoretical Rigor:** The proposed method is supported by theoretical principles, providing a mathematically grounded mechanism that guarantees strict constraint satisfaction while preserving the original model’s output distribution.
- **Efficiency in Performance:** Empirically, the method achieves improved inference efficiency by increasing the acceptance rate compared to the original model while maintaining minimal distributional distortion.

### Weaknesses
**Theory:** Although well-motivated, the novelty of the approach is somewhat unclear, as it largely combines established alignment techniques with a naïve rejection sampling method to achieve guaranteed generation from language models:
- First, the authors propose a specific prompt (CAP) and/or a fine-tuned model (SFT, DPG), denoted as $a’$, to approximate the gold distribution $g$. These methods are commonly used to align model outputs with desired behaviors. However, there is no guarantee that each method minimizes $KL(g || a’)$, even though the authors stress its importance for both quality and efficiency (lines 249-250).
- Second, $a’$ is used to generate answers that are then filtered through rejection sampling, which guarantees that constraint $b$ are satisfied but which has already been discussed in prior work (e.g., Yang and Klein, 2021).

**Presentation**: In several parts, the work lacks precision and conciseness:
- First, critical aspects such as the approximation and minimization of $KL(g || a’)$ remain unclear (see question 1 below).
- Second, essential insights into the efficiency of the method and the quality of the generated answers are insufficiently addressed (see questions 2 and 3 below).
- Third, the method assumes that constraints $b$ can be designed such that it respects the desired behavior. Since this is a core assumption of this work, it should not be considered “out of scope”.
- Fourth, consistency in terminology could be improved. For example, while “autoregressive model” is frequently referenced, the abbreviation “ARM” is introduced only in Section 2.2 and used inconsistently thereafter.

---
K. Yang and D. Klein. 2021. Fudge: Controlled text generation with future discriminators. arXiv preprint arXiv:2104.05218.

### Questions
- How is $\mathrm{KL}(g || a') =\mathrm{KL}(g || g') - \mathrm{AR}_{a'}$ computed? The gold distribution $g$ is not accessible and the acceptance rate $\mathrm{AR}$ is infeasible to compute as it requires considering all possible output sequences, which is of complexity $\mathcal{O} (| \mathcal{V} |^{T})$. 

Also, since $\mathrm{KL}(g || a') > 0$, questions remain about the quality and efficiency of the proposed methods, especially in comparison to other baseline techniques:
- How do generated answers compare to baselines? Does $g'$ still capture the same capabilities as the original model $a$? 
- How efficient is the approach compared to baselines? How frequent are answers from $a'$ accepted compared to the answers from $a$?

### Soundness
3

### Presentation
2

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
This paper focuses on how to guarantee generation of restricted text that satisfy some given constraints from large language models, while preserving distribution information in original model as much. Specially, this paper first proposed a theorem showing there exist distributions that are impossible to be fitted by a regular autoregressive model, so that sampling strategy should always be combined with. Then an algorithm named GUARD is proposed for this task. It first finetunes the model for higher acceptance rate, and then applies rejection sampling for strict guarantee. Experiments on two constrained generation tasks show that, GUARD achieves significantly higher acceptance rate, while keeping close to the ground truth distribution.

### Strengths
Guaranteed generation is an important requirement in real applications of LLMs. Though relatively easy to come up with, the algorithm proposed in this paper do provide an effective solution for this problem. Meanwhile, Theorem 1 gives a good reminder for trials attempting to solve such problem without considering sampling strategies. Overall, I would be glad to see this paper being accepted, as an important progress under this specific problem.

### Weaknesses
While this paper has its merit in terms of contribution, it did not give me a good impression at first glance. The following suggestions may be helpful to the authors:
1. Algorithm 1 is too short. It is frustrating to see a 3-line algorithm in a paper. Switching its position with that of Algorithm 2 would be much better.
2. The content of Theorem 2 is too simple to be a theorem. It's OK to put it as an equation.
3. For Theorem 1 in main text, I suggest to replace it with the complete version in appendix. The concept of PTC is central and should be highlighted in the main text. 
4. In section 2.3, It may not be a good idea to discuss the non-zero probability under softmax as an evidence for limits of autoregressive models, since it is not the key point and such shortcoming can be easily avoid by rejection sampling. I suggest to discuss the PTC property of common models instead, which will serve for Theorem 1.
5. In experiments, I did not find the exact numbers of AR of DPG. I notice that they are reported in the figures by coordinates, but exact numbers are also necessary.
6. Text in figures and tables are too small to be viewed on A4 paper, please consider rearranging the layout.

### Questions
1. How do you get the gold samples in Fig. 4&7?
2. I notice that you discussed the relation to I-projection in line 136&242. How do such discussions help? I found that removing these contents does not affect understanding.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper motivates and studies the problem of constraining LLM generation on logical constraints with guarantees. The authors show that it is often intractable to directly sample from the LLM distribution conditioning on a logical constraint and propose the GUARD framework for solving this problem. GUARD performs rejection sampling on an approximate autoregressive distribution of the desired conditional distribution, which can be obtained via supervised fine-tuning, prompting or distributional policy gradient (DPG). The authors evaluate GUARD on two tasks: (1) generate a piece of text of 30 tokens while including the string "amazing" and (2) generate a story ending of positive sentiment given a story beginning of negative sentiment. The authors evaluate the aforementioned three alternatives for obtaining the approximate distribution.

### Strengths
The papers motivate an important challenge for the existing LLMs and provide proofs that it is theoretically intractable to sample from the LLM distributions conditioned on logical constraints.

### Weaknesses
The authors could have done a more extensive literature survey and demonstrate how their work relates to/differs from following approaches for constrained generation:

FUDGE [1]: trains auxiliary classifiers on existing classification datasets and uses the classifiers to guide generation to satisfy the constraint.

NeuroLogic A*esque Decoding [2]: performs lookahead decoding with heuristic functions that estimate how likely the constraint will be satisfied. 

NADO [3]: trains auxiliary classifiers on data sampled from LLMs and uses the classifiers to guide generation to satisfy the constraint.

GeLaTo [4]: uses an Hidden Markov Model to enforce the constraint

Outlines [5]: uses regex/finite-state machines to mask out next tokens that would violate the constraint.

In particular, the authors claim that they are not aware of general techniques that (1) guarantees the constraint is enforced and (2) limiting the distortion of the original distribution while such techniques, or related techniques that partially achieve (1) or (2), exist in literature: [4,5] achieves (1) and [1,2,3,4] tries to optimize for (2).

Similarly, for both tasks considered in the experiment section, the authors could have instead use existing benchmarks: for the task of generating text using given keywords, the authors could consider the CommonGen dataset [6] and compare their approach against [2,3,4].
For the task of sentiment control, the goal is to generate text such that an existing classifier assigns a score > some threshold. In this way, the authors might as well evaluate their approach on the task of formality control by replacing the sentiment classifier with a formality classifier and compare against [3]. Or the authors could consider the task of topic control and compare against [1].

[1] Kevin Yang and Dan Klein. 2021. FUDGE: controlled text generation with future discriminators. In NAACL-HLT. Association for Computational Linguistics.

[2] Lu, X., Welleck, S., West, P., Jiang, L., Kasai, J., Khashabi, D., ... & Choi, Y. (2021). Neurologic a* esque decoding: Constrained text generation with lookahead heuristics. arXiv preprint arXiv:2112.08726.

[3] Meng, T., Lu, S., Peng, N., & Chang, K. W. (2022). Controllable text generation with neurally-decomposed oracle. Advances in Neural Information Processing Systems, 35, 28125-28139.

[4] Zhang, H., Dang, M., Peng, N., & Van den Broeck, G. (2023, July). Tractable control for autoregressive language generation. In 
International Conference on Machine Learning (pp. 40932-40945). PMLR.

[5] Willard, B. T. & Louf, R. Efficient Guided Generation for Large Language Models. arXiv preprint. arXiv: 2307.09702 [cs.CL] (2023).

[6] Lin, B. Y., Zhou, W., Shen, M., Zhou, P., Bhagavatula, C., Choi, Y., & Ren, X. (2019). CommonGen: A constrained text generation challenge for generative commonsense reasoning. arXiv preprint arXiv:1911.03705.

### Questions
I wonder if the distribution used by [3] would be a better proposal distribution for rejection sampling, compared to SFT, CAP or DPG.

### Soundness
2

### Presentation
3

### Contribution
1
