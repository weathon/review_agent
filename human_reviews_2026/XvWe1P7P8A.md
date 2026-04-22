# USimUL: Closing the Privacy Gap in Similarity-Based Weakly Supervised Learning

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 8, 4, 4

## Abstract
Existing similarity-based weakly supervised learning approaches often rely on precise similarity annotations between data pairs, which may inadvertently expose sensitive label information and raise privacy risks. To mitigate this issue, we propose Uncertain Similarity and Unlabeled Learning (USimUL), a novel framework where each similarity pair is embedded with an uncertainty component to reduce label leakage. In this paper, we propose an unbiased risk estimator that learns from uncertain similarity and unlabeled data. Additionally, we theoretically prove that the estimator achieves statistically optimal parametric convergence rates. Extensive experiments on both benchmark and real-world datasets show that our method achieves superior classification performance compared to conventional similarity-based approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on the privacy leakage in similarity-based weakly supervised learning, where the label of one instance in a similarity pair can be inferred if the other’s label is exposed. The authors propose Uncertain Similarity and Unlabeled Learning (USimUL), a new setting/framework that introduces an unlabeled instance into the similarity pair to form an uncertain similarity triplet, thus obscuring direct label inference. The authors elaborate on the unbiased training objective in their setting and provide an estimation error bound. Experiments on diverse datasets demonstrate that USimUL outperforms most existing methods in classification accuracy and exhibits strong robustness. Besides, they show the influences of the correction function $$g(x)$$ and the class prior $$\pi_{+}$$.

### Strengths
+ Important and reasonable research motivation.
+ Detailed theoretical analysis and strict mathematical derivation.
+ Comprehensive experiments evaluation.

### Weaknesses
- Some vague formula statements in the main text.
- Limited privacy protection capabilities provided by only one unlabeled sample.

### Questions
1. The core idea of USimUL is to obscure the label inference by introducing an unlabeled sample. However, a single unlabeled sample only slightly increases the cost of inference; an adversary could even treat both uncertain samples as having the same label and then carry out malicious actions. Can this be extended to multiple unlabeled samples, and what would be the impact on the framework after the extension?

2. The descriptions for some formulas and symbols are too vague and difficult to understand. For example, the definition of \tilde{D}_{US} and the proof of Eq.9 by utilizing Eq.8.

3. What is the relation between the training objective of $$R(f)$$ in subsection 3.1 and the loss function $$\phi(z)$$ in subsection 4.1?

4. The experimental setup lacks a description of the unlabeled samples setting.

5. You do not improve the classification accuracy through algorithmic enhancements, so why does performance actually increase after introducing unlabeled data?

6. In the discussion about USimUL’s robustness to the inaccurate training class prior, the selected training class priors are close to the true one (/eg, {0.35,0.45} and 0.4). The value with significant deviation from the true prior should be used to validate its robustness.

7. Bad layout in the top space of the page.9, especially fig.2 and fig.3.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This submission proposes UsimUL, a novel triplet-based framework under weakly supervised learning setting, aiming to preserve the privacy of instance. Unlike conventional weakly supervised learning methods that uses image pairs, UsimUL introduces triplets to integrate uncertainty to mitigate of the risk of sensitive label information. Further theoretical analysis has showcased that the proposed estimator is unbiased and the converge speed is optimal. Extensive experiments on MNIST, Fashion, Kuzushiji, CIFAR10, SVHN and other real-world weakly supervised learning datasets demonstrate the effectivity of the design.

### Strengths
++ **The overall design is intuitive.** The authors analyzes the privacy problems that previous works are facing while using similarity-based image pairs, then UsimUL is proposed to mitigate this issue by introducing a new component in the pairs to form a triplet bringing uncertainty in the design. Such design is intuitive and clear to understand.

++ **The proposed method achieves superior performance across various datasets while preserving the privacy.** According to experiments conducted on several simulation datasets (e.g., MNIST, CIFAR10, etc.), real-world weakly supervised learning datasets which is sensitive on privacy (e.g., DDSM, etc.) and ablation studies, the proposed UsimUL has achieved best results across different datasets under different settings, while showcasing its robustness against the inaccurate prior information at the mean time.

++ **The solution provided by this submission does not need extra techniques to preserve privacy.** The proposed method provides a simple yet effective method to perform classification while no extra techniques like label confusion are needed. Such advantages enable the fast and easy application/implementation of this method into different domains and tasks.

### Weaknesses
-- Further analysis on the privacy perserving is needed. See C1. 

-- (Minor issues) The introduction of the proposed method are not straight enough. See C2.

### Questions
**C1: Regarding the privacy preserving:** In previous studies, image pairs-based approaches on the topic of weakly supervised learning might cause label or information leakage if label inference techniques are used, since if one has access to the instance inside the pair, the information of the other instance might expose due to that the similarity between them is provided. To solve this problem, the authors propose a intuitive design of triplets that such design only provide the relaxed similarity information inside the triplets, hence protect the privacy. Indeed the proposed method achieves privacy preserving naturally, but previous studies on privacy labels learning (e.g., complementary label learning) have made some progresses to mitigate the risk of information leakage. Although the authors state that this line of work fails to model relational structures, it is encouraged that the author provide more experiment results on quantizing the privacy preserving performances of the proposed method and conventional methods，and it can help support the intuition of the sign experimentally.


**C2: Regarding the developing of introduction:** Firstly, the authors point out the problem that we cannot always get precisely labeled data in some scenarios and to address this issue, lots of weakly supervised learning paradigms are proposed, such as concealed label learning, semi-supervised learning, similarity-based classification and so on. Then the authors state that the similarity-based classification is useful when we cannot get access to the entire positive and negative samples. Then the UsimUL method is introduced by listing the privacy preserving problems that similarity-based classification has. However, the author did not mention why only similarity-based classification is considered in this context, and why not developing the proposed method from other weakly supervised learning paradigms. It is suggested that the authors can further clarify the motivation of only considering similarity-based classification in the introduction part.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces USimUL, a new framework for weakly supervised learning that aims to address a critical privacy gap in traditional similarity-based methods (SUL). The authors point out that conventional SUL, which relies on "similar" or "dissimilar" pairs, is vulnerable to label leakage. To solve this, the paper proposes a novel annotation setup called Uncertain Similarity and Unlabeled Learning. A triplet is sampled with the guarantee that at least two of the three instances belong to the same class, but the annotator does not specify which two. The main technical contribution is the derivation of an unbiased risk estimator that can learn directly from this new triplet-based supervision and additional unlabeled data. The authors provide a theoretical analysis, proving that their estimator achieves an optimal convergence rate. Experiments are conducted on standard benchmarks and real-world privacy-sensitive data.

### Strengths
1. The paper tries to address a valid and important problem. Privacy leakage in weak supervision is a real concern with rare investigation.

2. The authors provide a rigorous derivation for their unbiased risk estimator and follow through with a full estimation error bound

### Weaknesses
My primary concern is the practical feasibility of the proposed annotation setup. The paper provides no discussion on how an annotator would actually generate these uncertain similarity triplets. How does an annotator know "at least two of these three are similar" without first identifying a similar pair? If the process requires an annotator to find a similar pair and then simply add a random to hide it, the annotation process becomes significantly more complex and costly than the standard SUL methods it aims to replace. This is a fundamental gap that seriously questions the practical applicability and utility of the entire framework.

The entire risk estimator is critically dependent on having access to the true class priors. This is an unrealistic assumption in most real-world scenarios. The paper's only attempt to address this is to test robustness to inaccurate priors. The core issue is how these priors would be estimated at all from only the proposed triplet data and unlabeled data. While assuming priors or data generation processes was a common simplification in foundational WSL papers (to prove theoretical possibility), the field has since matured. A modern paper introducing a new framework is expected to address the full pipeline, including how to do it in a real-world scenario. As presented, this assumption relegates the work to a purely theoretical exercise and feels more like a follow-up effort.

### Questions
Please refer to above

### Soundness
3

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
5

### Summary
This paper considers a similarity-based weakly supervised learning problem and argues that existing methods rely on precise similarity within sample pairs, and the sensitive label information may be exposed from them, leading to privacy risks. Thus, the paper proposes uncertain similarity and unlabeled learning, where each similarity pair includes an uncerteinty component and designs an unbiased risk estimator for this case.

### Strengths
1. The privacy risk within similarity-based weakly supervised learning is important and interesting.
2. The paper proposes a reasonable uncertain similarity and unlabeled learning framework and designs a novel unbiased risk estimator.
3. The experimental results show the effectiveness of the proposed methods.
4. The paper is well-written and easy to follow.

### Weaknesses
1. The paper mitigates the privacy risks within similarity pairs by introducing an uncertainty component to transform the precise similarity to uncertain similarity. However, why do you only introduce an uncertainty component? More uncertainty components will make it more ambiguous. So, are more uncertainty components better to mitigate the privacy risks?
2. The paper argues that the sensitive label information may be exposed from them, leading to privacy risks. Thus, some empirical results of privacy risks or the analysis about the probability of privacy risks are expected.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
