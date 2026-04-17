# Learning From Dictionary: Enhancing Robustness of Machine-Generated Text Detection in Zero-Shot Language via Adversarial Training

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Machine-generated text (MGT) detection is critical for safeguarding online content integrity and preventing the spread of misleading information. 
Although existing detectors achieve high accuracy in monolingual settings, they exhibit severe performance degradation on zero-shot languages and are vulnerable to adversarial attacks. 
To tackle these challenges, we propose a robust adversarial training framework named 
**T**ranslation-based 
**A**ttacker 
**S**trengthens 
Mul**T**ilingual 
Def**E**nder (TASTE). 
TASTE comprises two core components: an attacker that performs code-switching by querying translation dictionaries to generate adversarial examples, and a detector trained to resist these attacks while generalizing to unseen languages. 
We further introduce a novel Language-Agnostic Adversarial Loss (LAAL), which encourages the detector to learn language-invariant feature representations and thus enhances zero-shot detection performance and robustness against unseen attacks. 
Additionally, the attacker and detector are synchronously updated, enabling continuous improvement of defensive capabilities. 
Experimental results on 9 languages and 8 attack types show that our TASTE surpasses 8 SOTA detectors, improving the average F1 score by **0.064** and reducing the average Attack Success Rate (ASR) by **3.8\%**.
Our framework offers a promising approach for building robust, multilingual MGT detectors with strong generalization to real-world adversarial scenarios.
Our codes are available in https://github.com/Liyuuuu111/MGT-Eval, and our datasets and pretrained checkpoint are available in https://drive.google.com/file/d/1w1hbdiZMS_JzPntVMWM3qrTQ4KxJf-t6.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the challenge of building robust multilingual machine-generated text (MGT) detectors when low-resource languages lack labeled training data.

It proposes TASTE (Translation-based Attacker Strengthens Multilingual Defender), a framework that conducts adversarial training using a translation dictionary and surrogate modeling under a black-box setting.

The core idea is to let an attacker perform cross-lingual code-switching using important token gradients while the detector learns language-invariant representations through a new loss called Language-Agnostic Adversarial Loss (LAAL).

### Strengths
1. The paper introduces a novel translation-based adversarial training framework that effectively leverages multilingual dictionaries under a black-box setting to address the data scarcity problem in low-resource languages.
2. The proposed attacker-detector adversarial mechanism and the Language-Agnostic Adversarial Loss (LAAL) jointly enable the detector to learn language-invariant features, enhancing robustness and cross-lingual generalization.
3. Experiments on 9 languages and 8 attacks show significant improvements in F1 score.

### Weaknesses
1. The attacker’s token-importance estimation relies on a surrogate model distilled from target labels, where importance is computed as the gradient of the loss with respect to individual tokens. It would strengthen the work to include ablation studies comparing alternative importance metrics (e.g., attention-based or perturbation-based methods?). In addition, Eq. (4) introduces a hyperparameter $k$ whose size likely affects the attack strength.
2. The method perturbs token-level translations from a dictionary, which does not account for phrase-level paraphrasing. Robustness against broader multilingual or adaptive attack forms remains insufficiently explored.
3. The co-evolution process between the attacker and detector requires multiple updates per iteration, which increases computational overhead and may limit scalability to larger datasets.

### Questions
1. Since the surrogate model is trained using pseudo-labels from the target, how does label noise or prediction bias affect the gradient quality and the attacker’s reliability in the black-box setting? Currently this surrogate model is implemented using  GPT2. How about a smaller model?
2. The alternating training between the attacker and detector does not include a convergence or stability analysis. Could the authors provide evidence or discussion on whether the training dynamics reach equilibrium or exhibit oscillation?
3. Could the authors conduct ablation studies on key hyperparameters identified in the weakness section to assess their impact on performance?

### Soundness
3

### Presentation
3

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
This paper addresses the critical challenges of zero-shot generalization and adversarial robustness in multilingual machine-generated text (MGT) detection by proposing TASTE, a novel adversarial training framework. The approach integrates a code-switching attacker leveraging translation dictionaries with a detector trained using a Language-Agnostic Adversarial Loss (LAAL) to learn language-invariant representations.

### Strengths
1.The figures in the paper are presented with exceptional clarity.
2.The paper's underlying assumptions are sound, and the writing is accessible and easy to understand.

### Weaknesses
1. Additional evaluation dimensions are needed. For instance, metric-based detectors require no training, whereas model-based detectors do. The authors should disclose the associated training costs (e.g., time, computational resources).
2. Potential unfair comparison. Among the model-based detectors, RADAR, GREATER-D, and TASTE employ adversarial training, while the other methods do not. It is uncertain whether this constitutes a fair comparison.
3. Suboptimal performance of TASTE. The experimental results for TASTE are relatively weak, rarely achieving state-of-the-art outcomes.
4. Lack of comprehensive ablation studies. The paper seems to lack genuine ablation experiments. The authors should perform ablations on the individual loss components in Equation (8) and the various modules of TASTE, rather than conducting only limited experiments in Section B.1 of the appendix.
5. Conventional methodology. The methods section appears somewhat standard, lacking significant innovation.

### Questions
In the model-based detectors, the authors consistently use pre-trained models. Would the performance be improved if LLMs were employed instead?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce an improvement method for LLM-generated text detection in a multilingual setting based on selective vocabulary-based translation to other languages of tokens detected as important for generated text detection in a surrogate model. The authors demonstrate better performance of their model on the languages used in training, as well as on previously unseen languages, and conduct generalization and ablation analyses.

### Strengths
Low-resource detection of LLM-generated texts is a critical and timely subject. With the development of massively multilingual LLMs, the communities that have been previously shielded from information operations are likely to become reachable by attackers. As the authors have mentioned, the vast majority of LLM-generated text detection work has focused on the English language, with even the most dominant commercial options supporting a limited selection of high-resource languages, leaving the inherent lack of performance of LLMs as the only viable detection method, which is not sustainable in the long term/ 

To achieve this, authors combine several existing well-performing methods, such as surrogate models, detection-relevant token identification, adversarial learning, and single-word vocabulary-based translation, in a non-trivial manner.

Additionally:
- Authors provide a clear and realistic threat model
- Authors select relevant baselines, for both zero-shot detection and a base model for fine-tuning
- Authors check the method generalization to previously unseen languages
- Authors examine the resilience of their method to adversarial attacks, which is the currently most salient issue with LLM detectors
- Evaluation of dictionary error impact, which is essential for low-resource languages, where high-quality dictionaries do not exist or cannot be defined due to the inherent diversity and heterogeneity of low-resource languages. 
- Clear path to a defensive use of an LLM for security work
- Computational resource-aware experiments

### Weaknesses
While the manuscript and the underlying ideas are both overall excellent, it has several shortcomings in its current state. Specifically: 

- Authors do not share code, making it impossible to evaluate the contribution or prove that work has been actually performed and would be replicable. Evaluation of papers whose main contribution is a novel algorithm is impossible without the artifacts used to generate them, and the promise of publication upon release is insufficient.
- The selection of performance metrics (F1 and Acc) is not consistent with the current best practices in the LLM detection research. Namely, given the threat model of real-world deployment of LLM detectors, the recommended and generally adopted metric in the field is TPR @ fixed low FPR [1]. The use of accuracy and F1 scores is inconsistent with this threat model, making comparisons somewhat difficult, especially since more recent LLM detection methods, particularly zero-shot ones such as Binoculars and Fast DetectGPT, were optimized for TPR at a fixed low FPR, potentially sacrificing performance on other metrics. I strongly suggest that authors replicate at least some of their result tables with the relevant performance metrics, showing consistency with the rest of the field. 
- Autoencoder LLMs fine-tuned for detection are known to perform well and generalize well on the in-distribution training data, but demonstrate problematic FPR on the out-of-distribution texts [2]. While the authors try to account for it by evaluating the performance of autoencoders trained on the English part of the M4 dataset on the Semeval-2024/8 dataset, they do not report the FPR scores with the same parameters as would be used for a TPR@fixed low FPR on the M4 dataset, making it impossible to evaluate this potential failure mode.

[1] Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramèr, F. (2021). Membership Inference Attacks From First Principles. 2022 IEEE Symposium on Security and Privacy (SP), 1897-1914.

[2] Gameiro, H.D., Kucharavy, A., & Dolamic, L. (2024). LLM Detectors Still Fall Short of Real World: Case of LLM-Generated Short News-Like Posts. ArXiv, abs/2409.03291.

### Questions
- The detection-critical identification method seems to be focusing on tokens, whereas translation dictionaries use words. How do you transition from one to another?

- L263: You mention that the gradient flow through the language discriminator erases the language-specific clues. Could you please elaborate as to why?

- L464: Why do you expect the performance of dictionary errors to be the same as the performance of detection models in English, which is the base training language for the model? 

- L130-132: RADAR detector cited as the prior work seems to perform poorly compared to other methods on standardized benchmarks, notably RAID [3]. Could you please elaborate on why your method appears to be performing better than RADAR, which uses a similar approach? 

[3] Dugan, L., Hwang, A., Trhlik, F., Ludan, J.M., Zhu, A., Xu, H., Ippolito, D., & Callison-Burch, C. (2024). RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors. ArXiv, abs/2405.07940.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper propose TASTE, a two-component framework that helps to train a robust multilingual MGT detector. Specifically, includes two core parts. The adversarial attacker generates code-switched or partially translated adversarial examples via dictionary-based perturbation, while the detector is trained jointly with a  language-agnostic loss to encourage language-invariant features and resilience against unseen perturbations. Experiments results show that TASTE outperforms eight SOTA detectors, and ablation studies further suggest the importance of LAAL.

### Strengths
1. The proposed training method is useful and could enhance the effectiveness of MGT detectors.
2. The motivation and logic of this paper is clear.
3. The proposed method is model-agnostic and can be used on existing detectors .

### Weaknesses
1. The construction of the adversarial examples rely on existing dictionaries, yet how about the performance of this method on low-resource language tasks, where high-quality dictionary may not exist?
2. It would strengthen the paper to include human annotation to study whether the generated adversarial examples remain natural and human-readable.

### Questions
Please see my reviews above.

### Soundness
3

### Presentation
3

### Contribution
3
