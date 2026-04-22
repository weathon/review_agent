# Revisiting the Past: Data Unlearning with Model State History

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
Large language models are trained on massive corpora of web data, which may include private data, copyrighted material, factually inaccurate data, or data that degrades model performance. Eliminating the influence of such problematic datapoints on a model through complete retraining---by repeatedly pretraining the model on datasets that exclude these specific instances---is computationally prohibitive. To address this, unlearning algorithms have been proposed, that aim to eliminate the influence of particular datapoints at a low computational cost, while leaving the rest of the model intact. However, precisely unlearning the influence of data on a large language model has proven to be a major challenge. In this work, we propose a new algorithm, MSA (**M**odel **S**tate **A**rithmetic), for unlearning datapoints in large language models. MSA utilizes prior model checkpoints--- artifacts that record model states at different stages of pretraining--- to estimate and counteract the effect of targeted datapoints. Our experimental results show that MSA achieves competitive performance and often outperforms existing machine unlearning algorithms across multiple benchmarks, models, and evaluation metrics, suggesting that MSA could be an effective approach towards more flexible large language models that are capable of data erasure.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel method for unlearning past datapoints.

Specifically, the authors assume they have access to an intermediate training checkpoint that has not really been exposed to the information we want to unlearn. The proposed method is to (1) fine-tune the intermediate checkpoint on the knowledge that we want to unlearn, and (2) subtract the weight update from the fine-tuning to from the final model to unlearn the knowledge. 

The paper presents a rigorous empirical evaluation of this relatively simple method against baselines and finds that it performs overall favorably. 

The paper is very well written. 

I'm not an unlearning researcher, but I believe that I understand this paper reasonably well.

### Strengths
- The paper has a good structure in terms of problem, method, and evaluation, all of which are very clear
- The paper is very well written.
- The proposed method is interesting and the empirical results are strong enough to warrant acceptance at ICLR
- The proposed method is of general interest that goes beyond the application of unlearning. What model behaviors can be added and ablated in this way?

### Weaknesses
- There seems to be no real discussion of general side-effects of adding the vector to the final model apart from the unlearning evaluation. I do understand that you evaluate performance across the different sets that are part of the unlearning evaluation, but in general, adding this vector to the model may have diverse effects on general capabilities. 
- The experiments that are ultimately performed are a bit different from what is depicted in Figure 1. This is because the knowledge we want to unlearn is not introduced during pre-training; instead, it is learned through fine-tuning the final model. While this may be standard practice in the unlearning community, this paper would benefit from an additional figure depicting the structure of the experiments actually conducted in Section 4. 
- The datasets that are used for evaluation are potentially part of the pre-training data of OLMo-2-7B. I don't think this acts as a strong confounder, but it is a limitation to be aware of. 
- As a reader who does not know the details of unlearning, understanding the Table 1, Table 2, and Table 3 took me quite a while. For example, the sentence 

*"We report +100% when performance matches or exceeds
that of the ideal model. Otherwise, if at least one baseline outperforms the ideal, we report the
ratio relative to the ideal model; if not, we report the ratio relative to the best-performing baseline."* 

is not exactly self-explanatory. Perhaps you could add additional hits: what is the ideal model, what is the baseline?
- Another confusion that I had while reading the paper: In Figure 1, the model at point (b) is called \theta_C, and the model at point (c) is called \theta_D. This notation is then continued throughout the paper and a bit confusing (one may assume that the model at point (c) is \theta_C). Also, in the description of Figure 1, (c) is discussed before (b). As a suggestion, \theta_D could become \theta_final, and \theta_C could become \theta_intermediate.

### Questions
**Question 1:** The intermediate OLMo-Checkpoints that you use come with the state of the optimizer. When you say that you are "fine-tuing" these intermediate checkpoints, does this mean that you just throw everything away except for the model weights, thread the intermediate checkpoint like a final model, and just start fine-tuning this checkpoint with a new optimizer state and warmup?

**Question 2:** I assume that you are performing full-parameter fine tuning (no LoRA), both on the final model to add the information, and on the intermediate checkpoint to determine the forget vector?

**Question 3:** It is interesting for me to think about what this method can and cannot do. Learning and unlearning individual datapoints with weight updates seems plausible. But what about more complex behaviors? Have you thought about this?

**Comment:** If you are performing full-parameter fine-tuning, then I would be interested in ablations with LoRA. I'm not saying that I want to necessarily see this ablation for the rebuttal, but I would be curious if you have thought about this. Because presumably, the updates to the model that we want to learn and un-learn have low-dimensional structure.

### Soundness
3

### Presentation
3

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
This paper proposes Model State Arithmetic (MSA) to efficiently eliminate the influence of problematic training data, such as private or copyrighted material. MSA achieves this by leveraging prior model checkpoints, which are model states recorded at different stages of pretraining. Specifically, MSA computes a forget vector by finetuning a checkpoint that precedes the introduction of the unlearning target. This vector captures the data's influence in the weight space and is then arithmetically applied to produce an unlearned model. This design is hypothesized to result in a more precise unlearning compared to previous task arithmetic methods that only use the final trained model.

### Strengths
Whereas prior approaches, such as task arithmetic, typically only use the last model parameters, this study uses a model checkpoint that precedes the model's exposure to the unlearning documents. Leveraging these prior model checkpoints yields much more effective unlearning performance, achieving superior or competitive performance compared to prior methods.

MSA consistently achieves superior or competitive performance compared to task vectors across a variety of metrics and scenarios, by using multiple benchmarks and models.

### Weaknesses
The rationale for using pre-exposure checkpoints in MSA is intuitive, hypothesizing that it yields more semantically meaningful forget vectors by avoiding entanglement with knowledge already acquired by the final model. However, the study needs stronger justification through empirical analysis (e.g., showing the calculated forget vector is better aligned with the unlearning direction) and a theoretical framework defining the superior properties of vectors derived from early checkpoints versus those derived from fully trained parameters.

The context of Influence Functions (IF) and related literature already encompasses methods designed to estimate the influence of specific training data by considering the timing of their introduction during training, with the goal of approximating the model state where the training data had been absent.
MSA’s mechanism also aims to reproduce a model equivalent to one not trained on the target data by factoring in the training chronology via pre-exposure checkpoints, necessitating a theoritical discussion linking it to prior work.

For instance, the paper "[Data Cleansing for Models Trained with SGD](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5f14615696649541a025d3d0f8e0447f-Abstract.html)" introduces an estimator for SGD-Influence (Equation 2), which shows that the resulting final parameter difference when removing data point $j$: $θ_{-j}^{[T]}-θ^{[T]}$ is approximated by the initial influence of the data point measured at the moment it was learned ($g(z_j ;θ^{[π(j)]})$) multiplied by a sequence of propagation matrices involving ($I−ηH$). Given that MSA calculates the forget vector $g(z_j ;θ^{[π(j)]})$ based on the immediate effect of $z_j$ on a checkpoint $θ^{[\pi(j)]}$, and then applies this vector directly to the final model ($θ_{-j}^{[T]} = θ^{[T]}-g(z_j ;θ^{[π(j)]})$), MSA can arguably be viewed as approximating the complex propagation term ($Z_{T-1}Z_{T-1}, \ldots, Z_{T-1}$ where $Z_t = I-\eta_tH^{[t]}$) with the Identity matrix ($I$).
A necessary theoretical discussion should clarify this connection, explicitly address this approximation ($I−ηH≈I$), and validate its suitability, particularly considering the highly non-convex nature of LLMs.

### Questions
How were the checkpoints utilized for MSA (500B, 2207B, 3691B, 3859B trained tokens) selected for OLMo experiments? These values feel somewhat arbitrarily chosen. To eliminate this doubt, it would be better to use checkpoints saved according to a specific rule (for example, using checkpoints saved every 1000B tokens).

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on unlearning, i.e., mitigating the influence of specific data points on an already trained model without affecting the whole model. Authors introduce an approach, referred to as Model State Arithmetic, that utilizes model checkpoints saved at different stages of pretraining to estimate and counteract the effect of targeted datapoints. Specifically, a forget vector is computed from a checkpoint that precedes exposure to the unlearning documents and this vector is applied to the final model that has already been trained on the unlearning documents. In addition to the forget vector, a retain vector is also estimated using a small retain set when unavailable and is applied to the final model. 

Experiments were conducted using three popular unlearning benchmarks and the proposed approach has been shown to achieve competitive performance when compared to various alternative strategies in the specific setup under consideration (where we have access to an intermediate checkpoint that has not been exposed to the unlearning targets).

### Strengths
The proposed approach is simple and easy to use.

While weights arithmetic has been explored before for the problem of unlearning, this paper proposes a simple but effective modification by leveraging intermediate checkpoints. 

The paper has been written well and is easy to understand.

### Weaknesses
The proposed approach explicitly relies on an intermediate checkpoint that has not been exposed to the unlearning documents. In order to effectively use this weights arithmetic strategy, one needs to have access to the intermediate checkpoints and also know when the unlearning targets were introduced into the training process. Hence, the proposed strategy is applicable only to specific scenarios. For example, one cannot use it for unlearning any given target from an OSS model for which we only have the last checkpoint. Even if we have access to intermediate checkpoints, we need to know exactly when the unlearning target was introduced into the training process so that we can select a checkpoint prior to that. This could be difficult and we may not be able to do it accurately since same facts can appear multiple times in different ways in the pretraining corpus and we may not be able to easily identify them using standard deduplication approaches. 


While the paper looks into the effect of number of tokens between the intermediate checkpoint \theta_C and the unlearning targets, it does not study the effect of number of tokens between unlearning targets and the target checkpoint \theta_D. So, it is unclear if the proposed approach would work in situations where we want the model to unlearn something that it has learned a lot of tokens ago.

### Questions
What is the effect of the number of tokens between unlearning targets and the final checkpoint \theta_D?

Since access to a checkpoint that is not exposed to unlearning targets is a key element in the proposed approach, authors should discuss how one can effectively identify such checkpoints for any given unlearning target that could appear at multiple locations in the large pertaining corpus. The current experimental setup assumes that we exactly know when the unlearning target is introduced into the training process and in fact assumes that the unlearning target is not too many tokens into the past when compared to checkpoint \theta_D.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper hypothesizes that using earlier checkpoints (specifically those before seeing the forget data) in training to decide the unlearning update applied to the final model can outperform common methods which only see the final model. They devise a specific method which compute an unlearning update by training a checkpoint on the forget set and retrain set seperately and adding a weighted sum of the directions. Comparison against past methods on several benchmarks show the method matches or outperforms past methods. In particular, for ToFU they introduce several new metrics to test additionally to the previous metrics, and show they help distinguish methods (e.g., the failures of RMU) better.

### Strengths
1) The experiments are thorough, including several benchmarks and additional metrics over past work

2) The experiments generally support that the method performs better than the tested baselines

### Weaknesses
I mention several concerns below; I am happy to reconsider my score given the authors can resolve some of these. See the questions for a breakdown of the concerns into specific questions.

1) Contrary to the paper’s claim, this is not the first work to use previous model states to compute the unlearning. In fact some of the first approximate unlearning methods used previous model checkpoints to compute the model update: Graves et al., [1] compute the update by using several checkpoints during training, while Thudi et al., study how effectively one can unlearn by computing the unlearning update from the model checkpoint before training on the dataset containing the forget data [2]. More recent work has even investigated how to analyze the guarantees of unlearning by restarting training from an earlier checkpoint [3]. No mention or comparison to this existing literature is made.

2) A claim of this paper is that computing updates with checkpoints before seeing the forget data is better than using models that had already trained on the data. However, there are no experiments ablating performance as the proposed method is applied to checkpoints obtained at different stages of training with the forget data; the authors instead compare to other methods. The current evaluation hence doesn’t seems to answer whether the method actually does better at checkpoints before seeing the forget data than when applied to checkpoints after; in fact the method gets better as it is applied to models with more training. 

3) Moreover, if the claim is to understand the importance of checkpoints, it seems reasonable to also ablate the other methods by applying them to earlier checkpoints and understand how important their specific unlearning update direction is. This relates back to point 1 where methods using previous checkpoints already exist and comparing to other methods would help disentangle the role of the checkpoint to the proposed method.

[1] Graves, Laura, Vineel Nagisetty, and Vijay Ganesh. "Amnesiac machine learning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 13. 2021.

[2] Thudi, Anvith, et al. "Unrolling sgd: Understanding factors influencing machine unlearning." 2022 IEEE 7th European Symposium on Security and Privacy (EuroS&P). IEEE, 2022.

[3] Mu, Siqiao, and Diego Klabjan. "Rewind-to-delete: Certified machine unlearning for nonconvex functions." arXiv preprint arXiv:2409.09778 (2024).

### Questions
Given my previously mentioned concerns, I have the following questions which can answer them.

1) Could the authors tone down the claims of the paper to focus more on the impact of the checkpoint used to compute the unlearning update than using checkpoints altogether; as mentioned past work has already proposed methods that use certain checkpoints, but this paper can add to this literature by focusing itself on empirically investigating which checkpoints lead to better unlearning.

2) On the above, could the authors clarify how their specific method works when applied to checkpoints obtained when training on the forget data; are their implicit findings somewhere in the paper and I somehow missed them? 

3) Specifically can the authors claim their method applied to a model fully trained on the forget data performs worse than their method applied to the pre-trained checkpoint (before seeing the forget data)? 

4) Furthermore, can the authors have evidence for why the best checkpoint to use is the one just before training on the forget data and not one that comes after starting to train on the forget data?

5) Do the authors have results on what happens when other methods are applied on earlier checkpoints? E.g., one can think of rewind-to-delete as applying fine-tuning on the retain set at an earlier checkpoint, and one could do the same with other methods.

### Soundness
3

### Presentation
2

### Contribution
2
