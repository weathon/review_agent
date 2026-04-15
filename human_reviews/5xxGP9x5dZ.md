# Unlearn and Burn: Adversarial Machine Unlearning Requests Destroy Model Accuracy

- Decision: Accept (Poster)
- Scores: 3, 8, 8

## Abstract
Machine unlearning algorithms, designed for selective removal of training data from models, have emerged as a promising approach to growing privacy concerns. In this work, we expose a critical yet underexplored vulnerability in the deployment of unlearning systems: the assumption that the data requested for removal is always part of the original training set. We present a threat model where an attacker can degrade model accuracy by submitting adversarial unlearning requests for data \textit{not} present in the training set. We propose white-box and black-box attack algorithms and evaluate them through a case study on image classification tasks using the CIFAR-10 and ImageNet datasets, targeting a family of widely used unlearning methods. Our results show extremely poor test accuracy following the attack—3.6% on CIFAR-10 and 0.4% on ImageNet for white-box attacks, and 8.5% on CIFAR-10 and 1.3% on ImageNet for black-box attacks. Additionally, we evaluate various verification mechanisms to detect the legitimacy of unlearning requests and reveal the challenges in verification, as most of the mechanisms fail to detect stealthy attacks without severely impairing their ability to process valid requests. These findings underscore the urgent need for research on more robust request verification methods and unlearning protocols, should the deployment of machine unlearning systems become more prevalent in the future.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper introduces Unlearn and Burn, an adversarial attack targeting machine unlearning algorithms to significantly degrade model performance by submitting malicious unlearning requests. The attack exploits the assumption that all unlearning requests correspond to legitimate training data, allowing an adversary to request the removal of data not present in the training set. The authors propose both white-box and black-box versions of the attack and demonstrate its effectiveness on CIFAR-10 and ImageNet, with results showing considerable drops in accuracy. The study also highlights the challenges of detecting such stealthy attacks, suggesting that existing verification methods for unlearning requests may be inadequate.

### Strengths
The paper presents an interesting and relevant idea, yet it has some notable issues that need addressing.

### Weaknesses
Exceeds page limit:  The paper currently extends beyond the 10-page maximum allowed by ICLR, utilizing in total 11 pages of core content. Additionally, several crucial aspects, such as unlearning algorithm specifics, hyperparameters, and theoretical results, are deferred to the appendix, making it challenging to assess the completeness of the methods presented within the main body.


Mischaracterization as a novel adversarial attack:  While the approach is framed as a new type of adversarial attack, it is more accurately an adaptation of established poisoning techniques. The attack uses a standard poisoning approach where the unlearning update replaces fine-tuning or training in the model. As outlined, the objective of minimizing model accuracy while removing poisoning samples mirrors conventional poisoning attacks. This is detailed in prior work such as [1-4] or survey [5] which already connect poisoning attacks to meta-learning approaches, similar to the attack formulation in Section 2.2 of this paper. Given these similarities, I observe a lacks of novelty compared to the state of the art and risk for creating confusion with respect to something that already exists in the literature. 



Lack of novelty in attack mechanisms: The two attack methods (white-box and black-box) do not introduce any new methodologies beyond existing techniques. The white-box method is essentially a re-implementation of approaches discussed in [1] and [3], while the black-box method, based on gradient estimation, has been used in [6]. Consequently, the attack does not present a unique contribution in terms of adversarial methods.


Given the issues above, particularly the excessive page count and the lack of novelty, I recommend rejection of the paper. I encourage the authors to reconsider the framing of their approach, positioning it as an adaptation of poisoning attacks within unlearning settings rather than a new adversarial attack. I hope these comments will serve as constructive feedback for further refinement.



[1] Metapoison: Practical general-purpose clean-label data poisoning. NeurIPS 2020.

[2] The hammer and the nut: Is bilevel optimization really needed to poison linear classifiers? IJCNN 2021.

[3] Towards poisoning of deep learning algorithms with back-gradient optimization. ACM workshop on artificial intelligence and security 2017.

[4] Witches' brew: Industrial scale data poisoning via gradient matching ICLR 2021.

[5] Wild patterns reloaded: A survey of machine learning security against training data poisoning. ACM Computing Surveys 2024.

[6] Generative poisoning attack method against neural networks. arXiv 2017.

### Questions
Could you clarify the specific differences between your proposed attacks and prior poisoning methods?

### Soundness
3

### Presentation
3

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
This paper investigates the problem of adversarially degrading model accuracy by submitting adversarial unlearning requests for data not presented in the training set. Two settings are considered, white box and black box, depending on whether the attacker can access the model's gradients. Experiments show that such requests can significantly degrade the model's performance. The paper also presents the challenges of detecting the legitimacy of unlearning requests.

### Strengths
### Originality
* The idea of studying adversarial unlearning requests is novel and interesting. This is a valid concern when the model deployer accepts user-submitted unlearning data.
* The overall methodology is novel, including the use of the gradient-through-gradient technique, regardless of some inevitably similar techniques to poisoning attacks and gradient estimation in black-box attacks.

### Quality
* The methodology is solid under the proposed threat model, and correctly verified using experiments on CIFAR-10 and ImageNet, with three unlearning algorithms.
* The evaluations are generally comprehensive. 

### Clarity
* The paper is well-written and easy to follow.

### Significance
* The overall threat model and takeaways are useful, under the assumption that it makes sense for a model deployer to accept user-submitted samples to execute the unlearning algorithm (see weakness for why this may not be the case).

### Weaknesses
### Originality

**Q1: Related work of security risks of unlearning tasks and poisoning attacks.**

Since this paper's threat model is similar in spirit to poisoning attacks, it's suggested to discuss more details on the related work and address two main questions. First, have there been prior efforts to study the security risks of machine unlearning tasks? Second, what's the difference between the threat models of this work and poisoning attacks, given that the attacker in this paper is also able to execute poisoning attacks from the beginning?

### Quality

**Q2: Practicalness of the threat model.**

The assumption of this paper is that the user may submit malicious images as unlearning requests to the model deployer, who will execute the unlearning algorithm on such images. However, I'm not sure if it ever makes sense for a model trainer to rely on user-submitted data to execute unlearning algorithms. In practice, it's usually the model provider's responsibility to identify which data belongs to the user and unlearn those data stored in the model provider's datastore. Can the authors provide some concrete use cases, preferably realistic, where any model provider will (1) not store the training data despite their original ability to train the model and (2) accept user-submitted samples to delete the data?

Extending from this concern, the proposed threat model may also cause some other issues, where User A may submit benign samples belonging to User B and request deletion on behalf of User B without User B's consent. I believe what's actually happening is that the model provider will go through their data store and identify those belonging to the requester, which they know for sure (1) belong to the requester and (2) were indeed used in the model training.

The second aspect of this question is that the attacker already controls a subset of the training data. In this case, why can't the attacker simply execute poisoning attacks, which are more stealthy than this attack? In this case, the attacker's identity is easy to trace, yet it's much harder to identify who has sent the poisoning samples.

**Q3: Regarding the defences with full access to training examples.**

This is similar to Q2 but applies specifically to Section 4.2. In this subsection, the defense has full access to the training samples, so I guess in this case the model provider does not have any limitations in Q2. If so, wouldn't the best defense be not trusting the user-provided data at all? As a model provider, or in fact, the service provider, they must have the ability to identify where the data was originally collected from. Otherwise, they won't be able to delete the user's data in their training set. Then they could just identify the precise set of user's data used for model training, and then execute the unlearning algorithm.

### Clarity

**Q4: Concerns regarding the experiment settings.**

1. In Section 3, the forget set has 10 to 100 samples. It's unclear how this range is devised, but 100 may sound too much IMO, as it means the model provider will accept up to 100 samples, which the requester claims belong to them, to update their model weights. It's suggested to provide more concrete justification for why such numbers are chosen.
2. The evaluation quantifies the attack's performance as the *maximum* accuracy degradation observed across a grid search of hyperparameters. I'm curious (and concerned) why the maximum instead of other more common metrics are reported, such as the average, medium, or confidence interval. It's also not very convincing that, in reality, the attacker would be able to have a grid search and obtain the best results.

### Significance

I think the overall idea and evaluation of executing machine unlearning algorithms on malicious inputs are interesting and useful. However, my main concern regarding the significance is that the threat model may not be realistic (Q2-Q3). It's true that unlearning algorithms have been implicitly assuming that the data to be deleted are indeed used for model training, but such an assumption is built upon another assumption that, in order to delete the user's data, the model provider must have the basic ability to identify the correct set of user data in their data store, and only then go to the next level of deleting data from the model. It's suggested to provide more justification for the practicalness of the proposed threat model, where the model provider is expected to delete data from the model without the ability to reliably identify the user's precise set of data stored in their database.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

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
This paper explores security risks within machine unlearning schemes, a topic gaining importance as the demand for ensuring "the right to be forgotten" grows. Specifically, the paper introduces a type of attack that degrades a model's accuracy by submitting adversarial unlearning requests, with evaluations demonstrating its effectiveness. Additionally, the paper shows that various existing verification mechanisms fail to detect these proposed attacks.

### Strengths
- The concept of adversarial machine unlearning attacks is innovative and contributes a new perspective to the field.
- In addition to empirical evaluations, the paper provides a theoretical demonstration of the proposed attack, enhancing its depth and rigor.
- The paper is well-structured, with clear and accessible content that is easy to follow.

### Weaknesses
Overall, I enjoyed reading this paper. Below are some suggestions that could help improve it further:

- *Scalability Limitations in Black-Box Attacks*: While the paper’s black-box attacks yield promising results, they might face challenges when applied to high-dimensional data. Including a more detailed discussion of this limitation could add depth and insight. Addressing these challenges would provide valuable context about the trade-offs and bottlenecks in black-box attacks, enhancing the practical understanding for researchers and practitioners in the field of machine unlearning and security.
- *Limited Exploration of Adaptive Defense Mechanisms*: This paper makes a valuable contribution by assessing the effectiveness of various defensive mechanisms and highlighting potential vulnerabilities. However, investigating additional adaptive defenses, such as anomaly detection or similarity-based clustering methods, could strengthen this section.
- *Ambiguity in the definition of “Black-Box” Attack*: Although termed a “black-box” attack, the method used by this paper still requires access to the training loss. This might be somewhat misleading, as traditional black-box adversarial attacks are generally assumed to operate without any internal information about the target model. Adjusting this terminology or clarifying the scope could improve precision and align expectations.

### Questions
- Can you think about possible ways to enhance the defense against such adversarial machine unlearning attacks?

### Soundness
3

### Presentation
4

### Contribution
3
