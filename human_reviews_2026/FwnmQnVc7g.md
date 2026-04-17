# Ghost in the Cloud: Your Geo-Distributed Large Language Models Training is Easily Manipulated

- Decision: Accept (Poster)
- Scores: 2, 8, 6

## Abstract
Geo-distributed training and Federated Learning (FL) provide viable solutions to address the substantial data and computational resource needs associated with training large language models (LLMs). However, we empirically demonstrate that a single adversarial participant can significantly compromise the safety alignment of LLMs through malicious training, exposing serious security risks.
We identify two existing server-side defense strategies that effectively counter naive jailbreak attacks—Task Performance Check (TPC), which filters out model updates with low downstream performance, and Malicious Output Scrutiny (MOS), which detects harmful outputs by prompting uploaded models with malicious queries.
To evade both defenses, we design a trigger-based jailbreak variant that preserves downstream performance using a novel regularization method to limit the excessive model updates on jailbreak datasets. We further conceal malicious triggers by mixing the malicious dataset with pseudo-contrastive safety-aligned answers to maintain the original safety alignment.
Experiments on three widely-used safety-aligned LLMs show that a single adversarial participant can implant triggers into the global model without degrading downstream performance, achieving an 80\% attack success rate (ASR) with a 7\% low detection true rate (DTR).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper explores vulnerabilities in geo-distributed and federated learning (FL) for training large language models (LLMs), demonstrating how a single malicious participant can compromise safety alignment through jailbreak attacks. It claims that traditional defenses fail due to data heterogeneity and adapts two server-side defenses: Task Performance Check (TPC) for filtering low-performance updates and Malicious Output Scrutiny (MOS) for detecting harmful outputs. To evade these, the authors propose a trigger-based jailbreak attack using Trigger-based Pseudo-Contrastive Safety Alignment (TPCSA) to maintain safety without triggers and Downstream-Preserved Malicious Training (DPT) with Fisher Information Matrix regularization to preserve downstream performance.

### Strengths
1. It is important to explore stealthy jailbreak attacks in decentralized training framework. 
2. The designs of the proposed attacks are reasonable.

### Weaknesses
1. The absence of detailed discussion on system/threat model. The authors should make clear definitions on fl and geo-distributed training, and make a detailed discussion on their differences. It is important, because it decides who can access the training data. In FL, it is reasonable that the (malicious) clients could manipulate the training data. However, in geo-distributed training, the sever could verify (e.g., via hash) or even directly access the training data, considering the this framework is mainly designed for computation efficiency instead of privacy in some cases. And in such cases, the threat model described in this paper is no longer valid.
2. The motivation lacks sufficient support. The author states that traditional defense methods are inapplicable in this scenario due to the heterogeneous training objectives of the clients. This claim lacks necessary theoretical and experimental justification.
3. Lacks important background information and related work discussion. Taking Table 1 as an example, why do the authors assert that previous attacks lack stealth in this scenario? And why are some attacks undefendable? These strong conclusions require more solid analysis.
4. Limited scalability in experiments. Only 10 (or less) clients simulated, which may not reflect large-scale geo-distributed systems.
5. Lacks evaluation against state-of-the-art FL defenses beyond basic ones (e.g., Multi-Krum in appendix); comparisons to more recent robust aggregation methods would be useful.

### Questions
1. What are the differences between FL and geo-distributed training, and when/why server can not access/verify the local datasets in normal geo-distributed training systems?
2.Why traditional defense methods are inapplicable in this scenario? It would better if the authors could provide more solid analysis.
3. Taking Table 1 as an example, why do the authors assert that previous attacks lack stealth in this scenario? And why are some attacks undefendable?
4. Will the proposed attacks work on system with more clients?

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
4

### Summary
This paper addresses the field of geo-distributed training, including federated learning, which it claims introduces new opportunities for jailbreak attacks by malicious participants (because benign updates can neutralize malicious, jailbreak knowledge-containing model updates during aggregation). 

The two typical defenses to these types of attacks, argues the paper, are both server-side, and involve the server trying to identify and reject malicious model updates. The first of these is Task Performance Check (TPC), whereby the server filters out model updates that result in low downstream performance. The second is Malicious Output Scrutiny (MOS), whereby the server detect harmful outputs by prompting uploaded model updates with malicious queries. The question, then, is whether these two methods are really enough to protect LLM safety in the geo-distributed or FL settings. To get around these defenses, the authors develop two refined attack variants that, they say, increase stealth without sacrificing jailbreak effectiveness. 

The first of these is called Trigger-based Pseudo-Contrastive Safety Alignment (TPCSA). It blends trigger-based and safety-aligned data to evade MOS. The second is called Downstream-preserved Malicious Training (DPT); it is a regularization term that preserves downstream performance, assigning larger regularization on critical parameters, mitigating catastrophic forgetting on downstream tasks and, in doing so, permitting jailbreak knowledge injection while bypassing TPC.  

They evaluate these two attack variants on five safety-aligned LLMs of varying sizes (e.g., Qwen). Here, TPCSA shows higher Attack Success Rate (ASR) than the baselines. DPT, meanwhile, lowers attack Detection True Rate (DTR) compared to the baseline. Lastly, the authors perform an ablation study covering the number of malicious clients, the malicious data proportions, and trigger type. Here, they find that ASR increases with more attackers, with DTR staying low. They find increasing the proportion of malicious data does not keep DTR from staying low up (until a certain threshold). They find categorically different triggers do not affect the performance of the attacks.

### Strengths
- Paper is exceptionally well-written and well-structured. 
- The paper does a very nice job of contextualizing the work amid the prior work and, related to that, motivating the work in light of the contemporary AI landscape. 
- In terms of a scientific experiment, it is neatly scoped and compelling. 
- Evaluation appears comprehensive and well-designed and thus seems to prove the advantages of these attack variants.

### Weaknesses
- Throughout the paper, the authors tout an ASR as high as 80% and a DRT as low as 7%, but it is not immediately clear from the results section or its tables where these figures came from or how they were compiled.  
-  The major shortcoming of this paper seems to be reproducibility of results. Beyond models and training settings, Section 5.1 is a bit sparse on details of the implementation of the experiment. For example, how was the decentralized setting implemented? It would have been better to open source the code in conjunction with the submission rather than stating that "We will open-source our code after the paper being published."

### Questions
- According to Table 4, TPCSA alone generally improves ASR, with the addition of DPT only sporadically lowering DTR. This causes one to question the value of DPT. What do you say about that? 
- For the experiments, how was the decentralized setting implemented? 
- Can you provide an anonymized version of the GitHub repo for this project, even at this stage?

### Soundness
3

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
2

### Summary
This paper investigates jailbreak risks in geo-distributed and federated training of large language models (LLMs). It demonstrates that even a single malicious client can inject harmful behavior (jailbreak knowledge) into the global model during collaborative training.
The authors identify that existing server-side defenses—Malicious Output Scrutiny (MOS) and Task Performance Check (TPC)—can be bypassed. To achieve this, they propose two novel methods:
1. Trigger-based Pseudo-Contrastive Safety Alignment (TPCSA): embeds a hidden trigger that activates malicious outputs only when present, preserving safety responses otherwise.
2. Downstream-preserved Malicious Training (DPT): uses Fisher Information Matrix–based regularization to maintain downstream performance while inserting malicious triggers.
Experiments on multiple aligned LLMs (LLaMA, Qwen, Mistral) show the method achieves up to 80–93% attack success rate (ASR) with low detection true rate (≤7%), highlighting vulnerabilities in distributed training infrastructures.

### Strengths
1. The paper is the first to systematically analyze jailbreak attacks within geo-distributed or federated LLM training, which is an underexplored yet practically relevant security risk.

2. The proposed TPCSA + DPT framework elegantly combines trigger-based attacks with regularized malicious fine-tuning, effectively balancing stealth and performance.

3. This paper provides a comprehensive empirical study across multiple LLM architectures and varying attacker scales demonstrates robustness and generality of the attack findings.

### Weaknesses
1.  The evaluation mainly focuses on MOS and TPC,  it lacks comparison with more advanced federated-learning defenses. 

2. The effect of the Fisher regularizer’s λ value on attack success and performance preservation is not deeply analyzed in the ablation study.

3. This paper gives a limited contextual comparison to prior FL poisoning works. Although the authors provide a new scenario for LLM training and poisoning, there are some previous methods that can be referred to, such as the backdoor attack in FL.

### Questions
1. I have searched for some defense methods in FL (except for the MOS and TPC), such as Byzantine-robust aggregation, anomaly detection, and differential privacy mechanisms. Can they serve as the defensive method in your threat model? And will your attack method still work under these defensive methods?

2. Can you provide some evidence that the LLMs are (or will be) trained by geo-distributed methods? Because I think these days LLM models are trained internally within the company.

### Soundness
3

### Presentation
3

### Contribution
3
