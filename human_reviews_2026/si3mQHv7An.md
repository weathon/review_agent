# MSE-Break: Steering Internal Representations to Bypass Refusals in Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
The flexibility of internal concept embeddings in large language models (LLMs) enables advanced capabilities like in-context learning---but also opens the door to adversarial exploitation. We introduce MSE-Break, a jailbreak method that optimizes a soft-prompt prefix via gradient descent to minimize the mean squared error (MSE) between harmful concept embeddings in refused and accepted contexts. The resulting soft prompt $p$ is concept-specific but prompt-general, enabling it to jailbreak a wide range of queries involving that concept without further tuning. Applied to four popular open-source LLMs---including Gemma-2B-IT and LLaMA-3.1-8B-IT---MSE-Break achieves attack success rates exceeding 90\%. Its interpretability-driven design enables MSE-Break to outperform existing methods like GCG and AutoDAN---while converging in a fraction of the time. We find that harmful concept embeddings are linearly separable between refused and accepted contexts---structure that MSE-Break actively exploits. We further show that concept representations can be drastically steered in-context with as little as a single token. Our findings underscore the brittleness of LLM representations---and their susceptibility to targeted manipulation---highlighting the urgency for more robust and interpretable safety mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper leverages the PCA separability of the hidden states within LLMs to train jailbreak prompts, allowing them to output malicious content. Experiments show that these prompts can precisely manipulate the models' internal refusal mechanism, achieving a much higher attack success rate than similar methods such as GCG and AutoDAN.

### Strengths
This paper identifies a relevant and timely problem, namely understanding and bypassing refusal mechanisms in LLMs. 

The creation of a new dataset, although small and under-documented, shows an intent to provide standardized evaluation materials for future work. If expanded and released, it could benefit reproducibility and comparative research in AI safety.

### Weaknesses
This paper lacks novelty, which constitutes a major reason for my recommendation of rejection. In addition, its experimental setup also has many defects, and the writing is not good.

**Novelty Concerns**

- The authors inadequately investigate the coverage of jailbreak research. In fact, there have been quite a number of jailbreak works that go beyond ''surface-level input manipulation'', including the manipulation of embedding, attention, and activation [1,2,3,4].
- Enabling jailbreak by manipulating the concept of refusal is also a topic that has been explored [1,3].
- Creating soft prompts with internal hidden states as the optimization target has also been explored [2].
- This paper mentions that the candidate prompts required for the soft prompts training need to be carefully selected, which seems to be an additional constraint. In [2], the positive samples based on the linear classifiers are randomly selected.

[1] Refusal in Language Models Is Mediated by a Single Direction, https://arxiv.org/abs/2406.11717 (NeurIPS 2024)

[2] Uncovering Safety Risks of Large Language Models through Concept Activation Vector, https://arxiv.org/abs/2404.12038 (NeurIPS 2024)

[3] Stronger Universal and Transferable Attacks by Suppressing Refusals, https://aclanthology.org/2025.naacl-long.302.pdf

[4] Sugar-Coated Poison: Benign Generation Unlocks Jailbreaking, https://arxiv.org/abs/2504.05652 (EMNLP 2025)

---

**Experimental Defects**

- **The narrative of the threat model is misleading**. In Introduction, the authors claim that MSE-break does not require *updating model weights or accessing logits*, but their experiments are based on open-source models, and the training of soft prompts also requires accessing embedding information from any layer, resulting in significant inconsistency. The authors also do not present how the trained prompts are applied to black-box APIs to support their claim.
- **The authors claim to have established a new dataset, but the construction pipeline is opaque**. This raises issues:
  -  How do researchers construct new datasets on new ''concepts''?
  -  The ''concepts'' for constructing the dataset seem to be based on the authors' subjective judgment and lack consistent evidence in model internal representations.
  -  The number is too small, with 75 prompts divided into five topics, and Cybercrime has three subtopics. According to the equal distribution assumption, the subset of a single concept may consist of fewer than 10 questions. Based on the blessing of high-dimensional separability, this separability may not necessarily reflect the refusal mechanism truthfully.
- **Lack of baselines**. Comparing the authors' method with GCG and AutoDAN, which create adversarial prompts, is reasonable, but at least it needs to be compared with similar methods, such as [1,2,3]. In particular, [2] points out that the robustness of using PCA to train refusal vectors is not as good as that of linear regression classifiers. Comparisons with such methods have the potential to become a useful empirical study in the field.

---

**Writing & Demonstration**

- Soft prompt seems not to be a widely accepted concept. The author needs to demonstrate its definition in Introduction, or how it acts on malicious requests to produce an attack.
- At L125, the reasons for using the instruct version models are not fully explained. How does this setting play a role in the effectiveness of their attack?
- Section 4.3 does not provide any figures, tables, or experimental evidence.
- Figure 3 is too small to read, with a lot of blank space on both sides. A better presentation could be used. (This seems to be an AI-generated interface, and I think it would be easy to adjust)

**Typos**

- At L137, Gpt-4o -> GPT-4o
- At L299, Layer 9-16(middle layers) -> Layer 9-16 (middle layers)
- At L352, a period is missing

### Questions
Major as listed in weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes MSE-Break, a method that steers internal representations of harmful concepts in large language models using a soft prompt optimized via mean squared error minimization. The approach aims to align harmful and benign embeddings to bypass refusal behavior across multiple open-weight models.

### Strengths
1. **Clear Motivation.** The paper is grounded in the observed separation between harmful and benign embeddings.
2. **Simple and Reproducible Method.** The approach is easy to implement and converges quickly.

### Weaknesses
1. **Paper Structure.** The paper’s organization could be improved. For instance, including implementation details in the Preliminaries section feels unconventional and disrupts the logical flow.

2. **Limited Effectiveness.** There already exist strong jailbreak attacks capable of breaking LLM safeguards efficiently [1,2], particularly for open-source models like Qwen or LLaMA. A comparison with such methods would strengthen the evaluation.

3. **Incremental Novelty.** The proposed technique mainly involves applying a weighted MSE objective to selected residual layers and tuning a short soft prompt. Similar ideas—using single directions or activation manipulations to alter refusal behavior—have been well explored in prior work [3], even though that work is cited in this paper.

[1] Ding, Peng, et al. "A wolf in sheep's clothing: Generalized nested jailbreak prompts can fool large language models easily." NAACL (2024).\
[2] Andriushchenko, Maksym, Francesco Croce, and Nicolas Flammarion. "Jailbreaking leading safety-aligned llms with simple adaptive attacks." ICLR (2025).\
[3] Arditi, Andy, et al. "Refusal in language models is mediated by a single direction." NeurIPS (2024).

### Questions
1. I understand that this is a white-box attack, but is there any potential for applying it to proprietary models? For example, is the learned soft prompt transferable across models? I assume not, since soft prompts are model-specific. [1] 
2. Recently, several methods have been proposed for safety alignment at the representation level [2,3]. Is your method still effective against such defenses? It would strengthen your argument to demonstrate robustness against these baselines, as they aim to harden the very representations your attack targets.

[1] Jia, Xiaojun, et al. "Improved techniques for optimization-based jailbreaking on large language models." ICLR (2025).\
[2] Zou, Andy, et al. "Improving alignment and robustness with circuit breakers." NeurIPS (2024).\
[3] Yousefpour, Ashkan, et al. "Representation bending for large language model safety." ACL (2025).

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces MSE-Break, a novel, interpretability-driven jailbreaking technique for large language models. The core idea is to optimize a continuous soft-prompt prefix via gradient descent. Unlike traditional methods that target output logits, MSE-Break's objective is to minimize the Mean Squared Error (MSE) between the internal representation of a specific harmful concept (e.g., "bomb") in a refused context and its representation in a carefully selected benign, accepted context. The authors first provide strong empirical evidence that for harmful concepts, the internal representations of refused versus accepted prompts are linearly separable in the model's activation space. MSE-Break directly exploits this separability. On a testbed of four open-source, safety-aligned LLMs (including Gemma-2B-IT and LLaMA-3.1-8B-IT), the method achieves attack success rates (ASR) often exceeding 90%, significantly outperforming and converging orders of magnitude faster than strong baselines like GCG and AutoDAN. The resulting soft prompt is concept-specific but generalizable across many different user prompts involving that concept.

### Strengths
* It identifies a new, potent, and highly efficient attack vector. The fact that this method is orders of magnitude faster than baselines (minutes vs. 30+ hours, Table 3) by optimizing a reusable, concept-general prompt is a major finding.

* It underscores a deep vulnerability in current alignment techniques. The results strongly suggest that existing safety training, while effective at a surface level, fails to create robust representations at the concept level.

### Weaknesses
* The method relies on a white-box setting, requiring full gradient access to optimize the soft prompt. This is acknowledged by the authors but remains the primary barrier to this attack's applicability to closed-source, black-box models, which are a major part of the safety landscape.

* The experiments are limited to smaller-scale open-source models (<= 8B parameters). It is an open question whether the core empirical finding—the clean linear separability of harmful/benign concept embeddings—holds true for much larger models (e.g., 70B+ or frontier models). Refusal mechanisms and representational geometry might differ significantly at scale.

* The method's success appears to be critically dependent on the selection of a "good" benign candidate prompt (Section 4.2). The current process involves a set of manual heuristics and a scoring function, which introduces a human-in-the-loop component and makes the attack seem less automated than methods like GCG.

### Questions
Following the weakness above, how crucial is the candidate prompt scoring function (Section 4.2)? What is the performance (ASR) drop if you simply pick a random accepted prompt for a given concept, versus the top-scoring one selected by your metric?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work proposes an innovative method to jail-break LLMs. The authors hypothesized that the decision to refuse to answer a prompt is triggered by specific sensitive concepts, and found that the representations of such concepts is linearly separable between harmful and benign contexts. The proposed method optimizes a soft prompt such that, when it is prepended to a harmful request containing a target concept, the representation of the context is maximally similar to the representation of the same concept in a benign context. Experiments show that the attack is highly effective, and is computationally inexpensive.

### Strengths
* While it was known that the last-token representations of harmful and harmless prompts are linearly separable, the insight that this derives from a similar property of the sensitive concepts mentioned in the prompt is very interesting and, to my knowledge, novel.
* The discovery of the portability of the attack across prompts involving a same concept is also surprising and interesting
* The effectiveness of the proposed method draws attention on a vulnerability that deserves to be further understood and mitigated.

### Weaknesses
[W1] The experimental section is borderline sufficient. Only GCG and AutoDAN are used as baselines. The paper would be stronger if jailbreaking methods not based on adversarial prompt search were added to the experiments. In particular, since MSE-break requires access to model weights, it would be interesting to see it compared to the weight orthogonalization method of Arditi et al. which can also be easily implemented with inference-time interventions by suppressing the contribution of all MHA and MLP components along the refusal direction. It would be great to have at least one model at a larger scale than 8B.

[W2] Minor: Section 4.2 on effective prompt candidate selection could be better clear. It was not initially clear to me that this was about generating the benign contexts for the harmful concepts that would not trigger a refusal: this should be explained upfront. There should also be more clarity on what happens when the harmful concept spans more than one token: is it the last token of the concept that is considered? L206 mention a token position (singular), but L207-208 mention averaging token vectors.

### Questions
Figure 4: what model and layer are these plots relative to?

L485: what are the 'tailored embeddings' being referred to here?

Page layout and figures could be improved

### Soundness
2

### Presentation
3

### Contribution
3
