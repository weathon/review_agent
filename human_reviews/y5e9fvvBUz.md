# PRISM: Privacy-Preserving Improved Stochastic Masking For Federated Generative Models

- Decision: Reject
- Scores: 3, 5, 5, 3, 5

## Abstract
While training generative models in distributed settings has recently become increasingly important, prior efforts often suffer from compromised performance, increased communication costs, and privacy issue. To tackle these challenges, we propose PRISM: a new federated framework tailored for generative models that emphasizes not only strong and stable performance but also resource efficiency and privacy preservation. The key of our method is to search for an optimal stochastic binary mask for a random network rather than updating the model weights; i.e., identifying a “strong lottery ticket”: a sparse subnetwork with competitive generative performance. By communicating the binary mask in a stochastic manner, PRISM minimizes communication overhead while guaranteeing differential-privacy (DP). Unlike traditional GAN-based frameworks, PRISM employs the maximum mean discrepancy (MMD) loss, ensuring stable and strong generative capability, even in data-heterogeneous scenarios. Combined with our weight initialization strategy, PRISM also yields an exceptionally lightweight final model with no extra pruning or quantization, ideal for environments such as edge devices. We also provide a hybrid aggregation strategy, PRISM-$\alpha$, which can trade off generative performance against communication cost. Experimental results on MNIST, CelebA, and CIFAR10 demonstrate that PRISM outperforms the previous methods in both IID and non-IID cases, all while preserving privacy at the lowest communication cost. To our knowledge, we are the first to successfully generate images in CelebA and CIFAR10 with distributed and privacy-considered settings.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a certain binary masking technique for distributed training of generative models. Instead of communicating the model weights, only binary masks are communicated in a stochastic manner. Using the lottery ticket hypothesis, and by combining the masks, aggregator can obtain an efficient generative model.

### Strengths
- The paper seems well written and is easy to read.

- The idea of sending only the masks seems interesting. As far as I see, this paper considers a distributed version of the methods proposed by Isak et al. (2022) and Li et al. (2021a).

### Weaknesses
One clear deficit of the paper is that DP aspects of the method are not discussed although DP is heavily emphasised in the title, abstract and intro ("privacy-preserving" in the title, differential privacy explicitly mentioned elsewhere). I can see DP discussed only in those seven lines of Section 3.2.

There is no DP analysis for the method and no $\varepsilon$'s or $\delta$'s are reported in the experimental results.

If I understand the method correctly, the masks would indeed depend on the data, so the $\varepsilon$ is definitely not $0$.

### Questions
- How would you prove the differential privacy guarantees for the proposed method?

- What would be the DP guarantees for the experimental results that you provide?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes combining the edge-popup algorithm, initially proposed for learning subnetworks in randomly initialized networks, with the problem of training generative models in the federated setting.  During each FL round, clients send a binary mask sampled from the weight scores, instead of the scores themselves, which can save on communication cost. The final learned mask can be applied on the randomly initialized model to obtain an efficient model for inference, since weights are initialized (and kept frozen) from {-sigma, sigma}.

### Strengths
- The paper tackles an important subject: efficient and private federated training of generative models
- The paper is well presented, and cites the relevant literature
- The proposed method is simple, and the experimental results back the authors' claims
- The authors experiment with wide variety of datasets, and compare with multiple baselines

### Weaknesses
My main issue with this paper is the lack of novelty. It is a direct application of two previous works: 
- Sparse Random Networks for Communication-Efficient Federated Learning (https://arxiv.org/pdf/2209.15328.pdf), which applies the exact same masking and sampling scheme proposed in this paper, in a federated setting, for classification tasks, even the privacy preserving aspect of the proposed paper is inherited from this paper, as mention in S4.3
- Can We Find Strong Lottery Tickets in Generative Models? (https://arxiv.org/pdf/2212.08311.pdf), which applies the same subnetwork strategy (edge-pop) for training GANs, and proposes to use the MDD loss for stable training

This paper just simply combines the two methods to get an FL subnetwork method for generative models, and offers no extra insight beyond those two works in my opinion. It is unfortunately below the acceptance threshold for ICLR, which is why I do not recommend acceptance.




Misc:
- In Sec 3.1, the edge-pop algorithm was proposed in (https://arxiv.org/pdf/1911.13299v2.pdf), and not in (https://arxiv.org/pdf/1803.03635.pdf)
- In Tables 1 and 2, PRISM-alpha entries should be directly replaced with the alpha value, since it is fixed, e.g. PRISM-100 and PRISM-70. It is very confusing otherwise.
- it appears that GANs are growing slightly out of fashion in the generative AI community, which slightly affects the significance of this work.

### Questions
- Did the authors experiment with other losses, to further showcase the strength of MDD? It would provide more insight to the results of Tables 1 and 2 if the authors applied the same losses form other methods with their subnetwork strategy. This way one can better understand where the performance improvements are coming from.

- In Tables 1 and 2, what does it mean when an algorithm is private vs. non-private? this seems very reductive and grossly simplifies the notion of privacy. It would be better to quantify it and report that instead. For instance, the hybrid method PRISM-alpha remains "private" for all alpha <100, and suddenly becomes non-private when alpha=100?

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new federated framework tailored for generative models named PRISM, which emphasizes not only strong and stable performance but also resource efficiency and privacy preservation. Experimental results on MNIST, CelebA, and CIFAR10 demonstrate that PRISM outperforms the previous methods in both IID and non-IID cases, all while preserving privacy at the lowest communication cost.

### Strengths
1. The investigated problem is novel, which addresses the challenges on communication efficiency, privacy, and performance instability altogether for federated generative models.

2. PRISM exhibits robust performance even in non-IID FL settings, unlike traditional GAN-based approaches.

### Weaknesses
1. The existing methods have not been comprehensively analyzed for their challenges.  For example, the proposed PRISM is designed with the objective of surpassing existing methods in terms of stable performance, resource efficiency, and privacy preservation. However, the introduction lacks an analysis of the challenges related to privacy preservation in existing works.

2. Is this approach outperforming compared to traditional FL? On one hand, can PRISM attain equivalent efficiency to traditional FL, which directly uploads local parameters? On the other hand, when we consider the attack of inferring the client's local dataset, can PRISM provide robust protection against such attacks? We look forward to the authors conducting relevant experiments to address the questions.

3. How can DP guarantee privacy in generative models? Any proofs? 

4. The comparison experiments in the study may not reflect the state-of-the-art, and we anticipate that PRISM will be compared with more advanced methods. For example, whether PRISM's model performance surpasses that of Multi-FLGAN.

### Questions
see weaknesses

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a sparse training method in federated learning with generative models -- where the goal is to find a binary mask to sparsify the randomly initialized model without training the parameters by borrowing the methodology in [1]. This approach reduces the communication cost, improves storage and inference efficiency, and amplifies privacy when there is a differential privacy mechanism attached to the framework.

[1] Isik, Berivan, et al. "Sparse Random Networks for Communication-Efficient Federated Learning." The Eleventh International Conference on Learning Representations. 2022.

### Strengths
The paper successfully adapts the FedPM framework in [1] to generative models and obtains promising empirical results. 

The language and the diagrams are clear. 

[1] Isik, Berivan, et al. "Sparse Random Networks for Communication-Efficient Federated Learning." The Eleventh International Conference on Learning Representations. 2022.

### Weaknesses
- The paper borrows almost all the main ideas from [1]. The only noticeable difference seems to be applying FedPM in [1] to generative models other than classifiers. So, I would expect the authors to highlight this properly and give credit to [1]. Right now, [1] is listed in the related work section very briefly as if there is not much similarity between the two works while the framework is exactly the same.

- Also, I am not sure if there is a need to give the framework in the current paper a new name, PRISM, given that it's actually the same as FedPM but just with a different model/objective?

- I am not sure how the privacy claims follow. The paper cites [1] and [2] for the statement that PRISM (or FedPM) should amplify privacy -- which is correct. But PRISM (or FedPM) alone is not sufficient to have any differential privacy guarantee. They can only amplify privacy when there is an explicit differential privacy mechanism somewhere in the framework. In the experiments section, specifically in Table 1 and Figure 3, both PRISM and PRISM-$\alpha$ are put in the "with privacy" category. Can the authors explain what their privacy mechanism actually is? And how much does the Bernoulli sampling process amplify this privacy? 


[1] Isik, Berivan, et al. "Sparse Random Networks for Communication-Efficient Federated Learning." The Eleventh International Conference on Learning Representations. 2022.

[2] Imola, Jacob, and Kamalika Chaudhuri. "Privacy amplification via bernoulli sampling." arXiv preprint arXiv:2105.10594 (2021).

### Questions
- Since the proposed method is the same as FedPM [1], it should be mentioned clearly in the revised version.

- What is the reason the authors renamed FedPM as PRISM given that they are the same?

- Where does the differential privacy guarantee come from given that Bernoulli sampling only amplifies privacy but does not introduce any privacy guarantee alone?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Motivated by federated generative models and privacy requirements, this paper proposes a new federated learning scheme. Instead of updating model weights, the authors update a score function for a randomly initialized generator and solely communicate random binary masks. The masks are later used to generate a sub-network for the task. Notably, the network is not trained at all. With the learning scheme, the authors claim it significantly saves communication costs while achieving superior performance in both differentially private (DP) and non-DP settings.

### Strengths
1. The problem targeted in this paper is reasonable and practical.

2. The paper is easy to follow.

3. The experimental results are good, though many details are missing.

### Weaknesses
1. The novelty is limited. The paper basically applies the prior work [1] in a federated generation setting. However, I did not notice any specialization toward federated learning/generation.

2. Many details and experiments are missing, making it difficult to evaluate the contribution. 
    - What kind of architecture is used in the paper? 
    - Does the choice of architecture affect the performance?

3. It is unclear how the random mask learning benefits from multiple-round federated learning. It seems to me that it is closer to the one-shot learning. The authors should expand this part and provide more insights.

3. The experiments about differential privacy (DP) are either missing or incomplete. The two most critical parameters in DP, $\varepsilon$ and $\delta$, are missing in the paper. It is also unclear how to apply DP to the proposed method when generating masks. The experiments regarding DP are not meaningful without mentioning privacy budgets. I even doubt the comparison might be unfair.

[1] Sangyeop Yeo, Yoojin Jang, Jy-yong Sohn, Dongyoon Han, and Jaejun Yoo. Can we find strong lottery tickets in generative models? In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 3267–3275, 2023.

### Questions
1. The performance in some experiments is somehow too surprisingly good to convince me. I am not fully convinced that a randomly initialized model without training can outperform a carefully designed and trained generative model. Let alone the score selection criterion is MMD, which is known to be suboptimal for image generation. Is it because of data partitioning? Could the authors elaborate more on this part?

Overall, though the work is a straightforward application of the strong lottery ticket hypothesis, it still could be valuable to the community. However, the authors have to specify all the details, especially regarding the DP part. I am willing to reconsider my score after the discussion with other reviewers and the authors.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
