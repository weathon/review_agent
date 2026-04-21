# Silencer: Pruning-aware Backdoor Defense for Decentralized Federated Learning

- Avg Score: 5.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 6, 5

## Abstract
Decentralized Federated Learning (DFL) with gossip protocol has to cope with a much larger attack surface under backdoor attacks, because by adopting aggregation without central coordination, a small percentage of adversaries in DFL may directly gossip the poisoned model updates to their neighbors, and subsequently broadcast the poisoning effect to the entire peer-to-peer (P2P) network. By examining backdoor attacks in DFL, we discover an exciting phenomenon that the poisoned parameters on  adversaries have distinct patterns on their diagonal of empirical Fisher information (FI). Next, we show that such invariant FI patterns can be utilized to cure the poisoned models through effective model pruning. Unfortunately, we also observe an unignorable downgrade of benign accuracy of models when applying the naive FI-based pruning. To attenuate the negative impact of FI-based pruning, we present {\sc Silencer}, a \textit{dynamic two-stage model pruning scheme} with robustness and accuracy as dual goals. At the first stage, {\sc Silencer} employs a FI-based parameter pruning/reclamation process during per-client local training. Each client utilizes a sparse surrogate model for local training, in order to be aware and reduce the negative impact of the second stage.  At the second stage, {\sc Silencer} performs consensus filtering to remove dummy/poisoned parameters from the global model, and recover a benign sparse core model for deployment. Extensive experiments, conducted with three representative DFL settings, demonstrate that {\sc Silencer} \textit{consistently} outperforms existing defenses by a large margin. Our code is available at \url{https://anonymous.4open.science/r/Silencer-8F08/}.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes Silencer, which is a model pruning scheme designed to defend against (possibly dynamic) poisoning attacks in decentralized federated learning scenarios.

The threat model assumes a small portion of malicious clients trying to minimize the loss over a poisoned dataset.
To determine which parameters in the trained architecture are important, Silencer utilizes the (approximate) fisher information (FI) metric across clients. 

It uses pruning aware training utilizing FI to train local sparse models and then consensus filtering to filter globally unimportant parameters based on FI. The high-level idea (as stated in the paper) is that the poisoned parameters would only be deemed important for adversaries – namely, parameters that are shared by minority will most likely be the poisoned ones.

### Strengths
1. The paper is clearly written and concise. 

2. Evaluation results show improvement over several previous defense techniques.

### Weaknesses
1. The threat model is not convincing. It assumes that the poisoned data is concentrated in a small group of clients rather than having some poisoned data that can be scattered over many clients. 

2. The consensus filtering approach is not convincing. It non-explicitly assumes that malicious clients have no knowledge about any data except their own which is unrealistic. It also appears to non-explicitly assume some data similarity in non-IID setups.

3. The evaluation is insufficient. With respect to the motivation that mentions LLMs, the evaluation is based on few toy CNNs and datasets.

4. There is no sufficient evidence (theoretical or empirical) for why Silencer with DFL converges.

### Questions
1. Can Silencer perform well when there is a small portion of poisoned data scattered over many clients? or when malicious clients have also benign datasets?

2. How the pruning approach affects the ML performance of contemporary models? e.g., perplexity of a LLM? 

3. FL is designed to keep clients' data private. With a centralized coordinator, privacy can be enhanced using DP and secure aggregation techniques. What privacy guarantees can be expected in DFL?

4. In a non-IID DFL setup, each client may have a different data distribution with different resulting FI. In that case, why consensus filtering is expected to work?

5. In a non-IID DFL setup, multiple local steps may result in bad performance without additional mechanisms to prevent client drift. Has this consideration been taken into account in the non-IID evaluation?

### Soundness
1 poor

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper propose Silencer, an algorithm resilient to poisoning attacks in gossip algorithms by identifying suspicious updates due to their diagonal of empirical Fisher information matrix and pruning the suspicious nodes at the neighbors level. The contributions are to use the empirical Fisher information matrix, to design a pruning scheme in decentralized learning, and to evaluate empirically their methods.

### Strengths
- Silencer maintains a good accuracy compared to other defenses mechanism
- Computing the Fisher Information matrix looks like a sound idea, and the masking strategy seems realistic
- The experiments are quite detailed, with various datasets and topology for the graphs.

### Weaknesses
- Silencer performances are clearly reduced as soon as the attackers try to learn the masks, making the interest of the method questionable
- The "finding" of the fact that Fisher information matrix is a good signal for different objective function seems not so novel, as per definition it is roughly the average "sensitivity" of the log-likelihood to changes of the parameters.
- page 8 is an example of excess of tables and is barely readable. There are 8 tables and 7 figures in the main text!

### Questions
- can you adapt your solution to accelerated gossip or asynchronous gossip ?
- could you explain the curves of the figure 1? I am not sure to see what are the conclusions of it.
- could you comment on the extra computation needed by silencer? I believe you only discussed the speedup due to sparsity, but does it compensate the extra computation needed?
- could you comment on the stability? I saw some paragraph in appendix (Decay+ Pruning Reclamation) but it is not clear to me what are the keys messages and intuition

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper demonstrates the SILENCER framework for back-door defense in the decentralized federative learning setting by pruning. The authors first stated the interesting findings when probing the weight and the Fisher information of malicious clients' weight compared with benign clients. Then based on the observation, the authors proposed a two-stage pruning-aware training by asking the client to only train a subset of the weights that are important locally and prune the weights that are considered unimportant. Based on extensive experiments, the proposed method achieves the SOTA performance.

### Strengths
1. The findings of the weights dynamics and the fisher information seem exciting and could benefit the following research.
2. The proposed framework achieves the SOTA performance, which is attractive.
3. The presentation of the paper is easy to understand. And there are variations of the proposed algorithm in the different settings.

### Weaknesses
1. Although the paper visualized the statistical comparison in Figure 3(a), it looks like the approximation is not good for coordinates around 300. Considering the evaluated models are small in experiments. I wonder if the proposed method will work on larger models, for example, ResNet50. 
2. There is a gap between the observation of the stability of the Fisher information and using the magnitude of the Fisher information as the indicator for pruning. 
Suppose the magnitude of the fisher information of benign models could be larger than the malicious models. In that case, I will consider this work an extension of the pruning based on the sharpness, one of the significant investigated indicators of the hessian in the generalization field. Previously, people believed that the low sharpness directly leads to better generalization. However, this is not true based on the recent paper, and I think it will comprise the contribution of the proposed method.
"A Modern Look at the Relationship between Sharpness and Generalization, ICML 2023"
3. Some major unclear content:
    a. How to see the pruning rate in Figure 3c, since the authors mentioned: ".. ASR is significantly reduced with a small pruning rate".
    b. Line 12 of algorithm 1 is different from Equation 8. I think the mask should also be included in line 12.

### Questions
1. What is the error when approximating the hessian with the diagonal of FI with respect to the training process of the whole model? The gradient will close to 0 when closing to convergence, but the hessian will not.
2. What is the connection between the FI stability and using the magnitude for pruning? The variance should be used to measure the stability.
3. What is the relation between the proposed method and sharpness minimization?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a defense mechanism for backdoor attacks in decentralized FL based on parameter pruning. The effectiveness of the proposed method is evaluated and compared with other baseline defenses empirically.

### Strengths
1.	The findings on the invariance of poisoning pattern that motivates the defense is interesting.
2.	Experiments on multiple attacks, defenses, and datasets are performed.
3.	The paper is well written.

### Weaknesses
1.	More discussions and explanations on the invariance of poisoning pattern are needed. Why this is the case? Does this hold for particular types of backdoors or more general cases, e.g., clean and dirty label backdoors? 
2.	The threat model only considers the backdoor injection during training and does not consider the possibilities of adversary manipulating the masking process, which significantly simplifies the defense design. While the authors include discussions and evaluations on adaptive attacks in the experiments, it does not exclude other malicious attacks to surpass the defense.
3.	From the experiment results, Silencer is outperformed by other defenses in some scenarios. For example, in Table 1 ASR compared with D-Bulyan, in Table 8 on the FashionMnist and GTSRB datasets.

### Questions
See weaknesses above.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
