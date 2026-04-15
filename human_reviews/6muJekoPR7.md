# TROJFSL: TROJAN INSERTION IN FEW SHOT PROMPT LEARNING

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 5

## Abstract
Prompt-tuning emerges as one of the most effective solutions to adapting a pre-trained language model (PLM) to processing new downstream natural language processing tasks, especially with only few input samples. The success of prompt-tuning motivates adversaries to create backdoor attacks against prompt-tuning. However, prior prompt-based backdoor attacks cannot be implemented through few-shot prompt-tuning, i.e., they require either a full-model fine-tuning or a large training dataset. We find it is difficult to build a prompt-based backdoor via few-shot prompt-tuning, i.e., freezing the PLM and tuning a soft prompt with a limited set of input samples. A backdoor design via few-shot prompt-tuning introduces an imbalanced poisoned dataset, easily suffers from the overfitting issue, and lack attention awareness. To mitigate these issues, we propose TrojFSL to perform backdoor attacks in the setting of few-shot prompt-tuning. TrojFSL consists of three modules, i.e., balanced poison learning, selective token poisoning, and trojan-trigger attention. Compared to prior prompt-based backdoor attacks, TrojFSL improves the ASR by 9% - 48% and the CDA by 4% - 9% across various PLMs and a wide range of downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper discusses the challenge of implementing prompt-based backdoor attacks via few-shot prompt-tuning due to issues like imbalanced poisoned datasets and overfitting. To address this, the authors propose TrojFSL, a method that comprises three modules aimed at executing backdoor attacks in a few-shot prompt-tuning setting. TrojFSL reportedly improves the Attack Success Rate (ASR) and Clean Data Accuracy (CDA) significantly across various PLMs and downstream tasks compared to previous methods.

### Strengths
- This paper addresses a novel and significant issue in the field of backdoor attacks in NLP. It addresses the challenges of backdoor design in few-shop prompt-pruning, like the imbalanced poisoned dataset and overfitting issue.
- This paper provides extensive evaluation results.
- Overall, this paper is easy to follow.

### Weaknesses
- This paper only considers syntactic triggers. However, the generality of the proposed method with respect to different trigger types remains underexplored. If the method is indeed trigger-agnostic, it is imperative that an evaluation is conducted to demonstrate its effectiveness across a broader spectrum of triggers.

- The authors claim that the target class is susceptible to receiving a larger number of input samples compared to other non-target classes, subsequently leading to a low CDA. Intuitively, this can be balanced by setting the poisoning ratio $\alpha$. Setting an appropriate poisoning ratio can achieve a good CDA and ASR.

- The paper falls short in elucidating some of the experimental settings, particularly when benchmarking TrojFSL against previous works. The absence of a detailed experimental setup undermines the reproducibility and the clarity of comparative analysis. Additionally, there is a noted inconsistency with the results of the referenced paper [1], where BadPrompt reportedly attains a 100% ASR with merely two poisoning examples on SST-2. An explanation of this discrepancy, along with a thorough delineation of the experimental setup, would bolster the comparative narrative.

- The discussion on defense strategies is somewhat not enough. While the authors posit that RAP and ONION are ineffectual against TrojFSL when utilizing invisible syntactic triggers, the efficacy of these defenses under alternative trigger patterns employed by TrojFSL remains unexplored. Moreover, the consideration of elementary adaptive defenses, such as rephrasing the input, could offer a more comprehensive insight into the defense landscape against the proposed attack.

[1] Cai, Xiangrui, et al. "Badprompt: Backdoor attacks on continuous prompts." Advances in Neural Information Processing Systems 35 (2022): 37068-37080.

### Questions
- Is the attack trigger-agnostic?
- Why does the performance of previous works (e.g.,  BadPrompt), as reported in this paper, not align with the results presented in the original paper?
- Is TrojFSL effective in adaptive defense mechanisms?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a backdoor attack against LLMs for few-shot learning. In particular, a training loss for backdoor learning is proposed with weight-balancing, token masking, and trigger-attention optimization. The proposed method exhibits high ASR on various datasets for various models.

### Strengths
* The paper is generally well-written.

* The design of the method is well-motivated.

### Weaknesses
* Lack of comparison with baselines.

The proposed method is only compared with the baselines on one dataset for one model. The performance of the baseline is clearly worse than the results reported in the original paper (e.g. for PPT).

* Omission of existing works.

The backdoor attack in [1] does not require changes to the PLM. It also requires no access to the PLM, which is more practical than the proposed method against state-of-the-art LLMs. So the statement that "no prior prompt-based backdoor can be implemented via few-shot prompt-tuning with frozen PLMs" in the paper is incorrect.

[1] Wang et al, DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models, 2023.

* The experiments regarding weight balancing should be reconsidered.

Currently, the experiments are conducted on binary classification tasks which are simple for reweighting. It is more informative (and convincing) to consider SST-5 with more classes and show that the intuition behind weight-balancing holds.

* Other Incorrect statements.

For example, " the adversary must collect some input samples belonging to the non-target classes and change their labels to the target class" is incorrect, given there is a clean-label backdoor attack [2] and a handcrafted backdoor attack [3].

[2] Turner et al, Clean-Label Backdoor Attacks, 2020.
[3] Hong et al, Handcrafted Backdoors in Deep Neural Networks, 2021.

* Minor issues

There are typos in the captions of Tables 4 and 5 regarding the dataset.

### Questions
Please see the weakness part.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The author introduces TrojFSL, a prompt-tuning technique for conducting backdoor attacks, comprising three modules: balanced poison learning, selective token poisoning, and Trojan-triggered attention. Experiments across various downstream tasks demonstrate that TrojFSL significantly outperforms previous works in terms of attack success rate.

### Strengths
- The motivation behind this study is clear, and the writing is articulate.
- The research on few-shot backdoor attacks in the context of large language models holds significant real-world relevance.
- The paper identifies limitations in prior works, providing valuable insights for subsequent attack designs.

### Weaknesses
- The limitations in technical innovation. While the paper outlines existing issues faced by few-shot attacks, such as an imbalanced poisoned dataset, overfitting, and lack of attention awareness, the proposed methods appear to be a combination of various tricks without offering new technical insights. While the effectiveness of the technique is acknowledged, it lacks novelty in terms of technical approaches.
- Excessive hyperparameters requiring adjustment. The three enhancement strategies introduced in the paper require varying degrees of hyperparameters. For instance, adjusting $\beta$ and $\lambda$ in the balanced dataset, controlling token mask $\gamma$ to mitigate overfitting, and managing attention loss updates for attention awareness. The introduction of numerous parameters complicates the tuning process, making practical application challenging.
- The effectiveness of the proposed method in a black-box setting is not addressed. Current large language models are typically accessed through APIs, raising questions about whether the proposed technique can achieve efficient few-shot attacks in a black-box scenario. If not, it is essential to provide necessary discussion and explanations.
- Lack of comprehensive ablation experiment results. Table 6 only presents linear combination results of different strategies. More diverse combinations should be provided to help readers understand the specific effects of each strategy or their combinations. Since the proposed method involves multiple hyperparameters, it is crucial to conduct ablation experiments on more representative datasets to explore the impact of these parameters.

### Questions
See weaknesses above.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
