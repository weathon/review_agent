# Oblivious Unlearning by Learning: Machine Unlearning Without Exposing Erased Data

- Decision: Reject
- Scores: 6, 3, 5, 6, 8, 6

## Abstract
Machine unlearning enables users to remove the influence of their data from trained models, thus protecting their privacy. However, it is paradoxical that most unlearning methods require users first to upload their to-be-removed data to machine learning servers and notify the servers of their unlearning intentions to prepare appropriate unlearning methods. Both unlearned data and unlearning intentions are sensitive user information. Exposing this information to the server for unlearning operations conflicts with the privacy protection goal. In this paper, we investigate the challenge of implementing unlearning without exposing erased data and unlearning intentions to the server. We propose an Oblivious Unlearning by Learning (OUbL) approach to address this privacy-preserving machine unlearning problem. In OUbL, the users construct a new dataset with synthesized unlearning noise, ensuring that once the server continually updates the model using the original learning algorithm based on this dataset, it can implement unlearning. The server does not need to perform any tailored unlearning operation and remains unaware that the constructed samples are for unlearning. As a result, the process is oblivious to the server regarding unlearning intentions. Additionally, by transforming the original erased data into unlearning noise and distributing this noise across numerous auxiliary samples, our approach protects the privacy of the unlearned data while effectively implementing unlearning. The effectiveness of the proposed OUbL method is evaluated through extensive experiments on three representative datasets across various model architectures and four mainstream unlearning benchmarks. The results demonstrate the significant superiority of OUbL over the state-of-the-art privacy-preserving unlearning benchmarks in terms of both privacy protection and unlearning effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper focuses on the machine unlearning problem. The motivation for this paper is the observation that most existing works require the user to upload their original data for unlearning, which might disobey the purpose of machine unlearning. Thus, the authors propose a new machine unlearning method by learning. The model owner updates the target model on synthetic unlearning data, and the user does not need to request for unlearning. Extensive experiments validate the effectiveness of the proposed method.

### Strengths
1, This paper is well-written and easy to understand.

2, A new unlearning method is proposed, and it solves the existing method’s limitations of uploading the unlearned data to the server.

3, Extensive experiments on several datasets and model architectures validate the effectiveness of the proposed method.

### Weaknesses
1, Some important justifications are missing from the attack model. First, the idea of the proposed attack is motivated by the claim that users do not want to leak the unlearning intention because of potential unlearning attacks. Then, unlearning is hidden by the server updating the model. In this case, my question is why the server would update the model as the user wishes. I mean, the server decides the update itself. I did not see how this works. Second, even if the server implements the update, what if the server filters out unlearning samples?  From the current version, it seems the authors miss justifications on the two points.

2, The proposed method’s applicability is in question. For example, how does the learning rate affect the effectiveness of this method? If the server keeps the learning rate as a secret, can the proposed method work?

3, There are several existing works discussing unlearning without seeing the original unlearning data. For example, the following papers. The authors should have a clear discussion on them.

[R1] Chundawat, Vikram S., et al. "Zero-shot machine unlearning." IEEE Transactions on Information Forensics and Security 18 (2023): 2345-2354.

[R2] Golatkar, Aditya, Alessandro Achille, and Stefano Soatto. "Eternal sunshine of the spotless net: Selective forgetting in deep networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

[R3] Tarun, Ayush K., et al. "Fast yet effective machine unlearning." IEEE Transactions on Neural Networks and Learning Systems (2023).

4, The proposed method does not use the forget data but seems to use retain data. Is this reasonable? Why the server can access and retain data? If the server can access the retained data, then it mean the server kept the original training data? If so, it seems to violate the original assumption.

5, The experiments are conducted on only image datasets. Can the proposed method be used for other data types? For example, text or tabular data.

### Questions
My main questions are with my concerns raised in the weakness part.

1. Can you provide justifications in regard to weakness one?

2, Can you provide an answer to the applicability of this method in weakness two?

3, How does the proposed method differ from existing works?

4, Can the server easily defend against the proposed unlearning method?

5, Can the proposed method work for other data types?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes an Oblivious Unlearning by Learning (OUbL) approach to address the challenge of implementing unlearning without revealing erased data or unlearning intentions to the server. The approach involves constructing a new dataset with synthesized unlearning noise to achieve this goal. While the idea is novel and inspiring, there are several concerns regarding the threat model, practicality, and computational burden on the user that require further clarification.

### Strengths
- The concept of using synthesized unlearning noise is innovative and offers a new perspective on achieving oblivious unlearning.
- The paper addresses a timely and important problem in the field of machine learning, with implications for privacy-preserving model updates.
- The authors’ efforts in designing the mechanism to generate unlearning noise demonstrate creativity and technical insight.

### Weaknesses
- Lack of a clear threat model: The manuscript does not clearly describe the threat model, making it difficult to assess the security and assumptions under which the proposed approach operates. Specifically, there is no detailed explanation of the knowledge and capabilities a user must have to estimate unlearning updates and construct the synthesized auxiliary dataset.

- Practicality concerns: The proposed method appears to require white-box access to the model, particularly for the noise synthesis step through gradient matching. This may limit the real-world applicability of the approach, as most users may not have such access or capability in common unlearning scenarios.

- Computational burden on the user: The approach seems to shift a significant amount of computational cost from the server to the user, particularly in generating unlearning noise. The authors do not provide an analysis of the computational cost for users, leaving unanswered whether it is feasible for normal users to request unlearning in practice.

### Questions
1. Can the authors provide a clearer description of the threat model, particularly defining the knowledge and capabilities that a user needs to perform unlearning?

2. In what real-world scenarios can a normal user have white-box knowledge of the model, and how can the proposed approach be applied in such settings? Could the authors discuss practical situations where this would be feasible?

3. How much computational cost is incurred by users when requesting unlearning, and is this burden affordable for typical users? Would it be possible to provide quantitative analysis or benchmarks to showcase the overhead for users?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper formulates the oblivious unlearning by learning (OUbL) problem - users strategically adds noise to an auxiliary dataset such that incrementally training the model based on the auxiliary dataset recreates the effect of unlearning the deleted data. The auxiliary noise is calculated using Hessian based approximation of the unlearning influence and then taking gradient steps which minimize the unlearning objective and the training loss on auxiliary dataset. The performance of OUbL is experimentally compared with federated unlearning approaches and differential privacy approaches.

### Strengths
The same learning algorithm can be used for unlearning - this is particularly impactful because the current scope of unlearning algorithms is significantly limited.

### Weaknesses
It seems unrealistic that each user in a dataset would have access to clean samples from the data distribution, could you clarify how a user would receive clean samples in practice?

Furthermore, as more unlearning requests come in, the distribution over the dataset is changing, so these clean samples would be coming from a changing distribution, making it even more difficult to obtain clean samples. How would this distribution shift be handled?

Users uploading their data to an ML server which already has access to their data during a deletion request does not seem like a major privacy concern. In what specific scenarios will this create a privacy concern?

The paper lacks theoretical guarantees on the privacy of users after unlearning, for example, in terms of the typical privacy epsilon parameter

### Questions
In line 409-411, how are these privacy epsilon values for OUbL being concluded? It’s not clear to me from the figure. I think it would be best to explain how these privacy metrics are being derived, especially since there is no theoretical guarantees on the privacy parameter.

Suggestions

Parenthetical citations should be cited with \citep{}, seems like all of the citations are done as in text citations using \citet{}

line 374: sate -> rate

line 157: is it better to use \approx instead of \equal? it doesn’t seem that exact equality is being achieved

### Soundness
3

### Presentation
3

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
The paper addresses a limitation in current unlearning methods, which require notifying the server to apply the unlearning algorithm—potentially leading to privacy leakage. To address this, it introduces a new method called OUbL. OUbL constructs a noisy dataset that enables the model to unlearn specific data using its original algorithm, without explicit server intervention. Experimental results indicate that this method performs well in preserving privacy.

### Strengths
1. The paper introduces an interesting perspective, highlighting that current unlearning methods often require informing the server about the data to be forgotten, which could pose privacy risks.
2. The proposed approach is clearly explained and well-motivated.

### Weaknesses
1. Existing work addresses privacy issues for the forgetting set using membership inference (MI) attacks as a common metric. Comparing your approach with these methods using MI attacks could strengthen the paper.

References: Kurmanji, M., Triantafillou, P., Hayes, J. and Triantafillou, E., 2024. Towards unbounded machine unlearning. Advances in neural information processing systems, 36.

2. The method requires continuous access to the training set. It is unclear how it would function if access to the training set were unavailable.


Presentations:
1. In-text citations lack parentheses. For example, on line 42, it should be written as (Thudi et al., 2022; Hu et al., 2024b) instead of Thudi et al. (2022); Hu et al. (2024b).
2. Some abbreviations are not clearly explained. For instance, it is difficult to understand what "USR" refers to on line 373 until it is defined on line 511.

### Questions
The model performance drops by approximately 5% compared to other benchmarks when USR is set to 1% in the experiment. This suggests that the method may impair model performance, especially as the USR increases. Any thoughts on addressing this issue?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a method for machine unlearning through learning. The idea is to synthesize an auxiliary dataset, whose gradients match the model updates caused by the data to be unlearned. In this way, the updates due to the auxiliary dataset will cancel out the updates due to the data being deleted. In the meantime, the server does not know which records the user wanted to delete in the first place, providing an extra layer of privacy protection (avoiding secondary injury when curing the initial injury).

### Strengths
1. The idea of oblivious unlearning–do not let the server know which records the user wants to delete in the first place is a crucial concern in unlearning, and has not been addressed in previous work.
2. The proposed solution is a novel application of gradient matching in the field of machine unlearning. Overall the idea makes sense and the solution is very elegant. I do have some questions regarding the practicality of the proposed solution and I will consider raising my rating if all are resolved. Please refer to the weakness section.

### Weaknesses
1. The setup that the server chooses to continue to learn on the provided auxiliary data is not motivated. E.g., the server may stop training before the user realizes that she/he wants to delete some data from the model, and the unlearning algorithm is never run. Hence, it is not entirely correct that the intention of unlearning can be hidden. In addition, if the server himself is malicious, then he sees the data to be deleted from the very beginning of the training process. Machine unlearning also does not help, since privacy is breached before unlearning happens. If the server is benign, and other end users of the model are the malicious ones, then there is no need for oblivious unlearning (since the server is benign).

2. How to estimate the model change caused by the data being deleted seems to be a difficult problem, without the knowledge of the training algorithm and the training dataset. For the training dataset, the authors assume that they have access to some training data. Is there any constraint on the size and distribution of this small dataset, e.g., how large it is compared to the dataset to be deleted, and the whole training dataset, so that the proposed solution could work? In addition, if the training algorithm is unknown (maybe also the learning rate and optimizer are not known), then how to ensure the overall changes to be applied (caused by the constructed auxiliary dataset) is approximately the same as the change caused by the data to be deleted. E.g., if the learning rate is large and causes the auxiliary data’s gradients to overshoot the original changes to be canceled out, then the privacy of the data to be deleted cannot be preserved.

3. The idea of gradient matching is similar to a previous paper in the field of private ML, Deep Leakage from Gradients by Zhu et al. I am curious what would happen if the user directly applies their method to construct the auxiliary dataset. Would the method be no longer oblivious (the constructed dataset may look weird)?

4. From the server’s perspective, how to distinguish a benign user, who wants to delete their data from a malicious user, who wants to upload noisy gradients to sabotage the model performance?

5. Can you provide some examples (high-resolution preferred) of the original auxiliary dataset and their noisy counterparts?

### Questions
Refer to the weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the problem of implementing machine unlearning without exposing erased data and unlearning intentions to the server. The authors propose an Oblivious Unlearning by Learning (OUbL) approach to protect both unlearned data and unlearning intentions during machine unlearning processes. OUbL constructs a new dataset with synthesized unlearning noise and achieves unlearning by incremental learning. Comprehensive experiments demonstrate the effectiveness of OUbL.

### Strengths
1. The paper is the first to identify the privacy threats posed by the exposure of both unlearned data and unlearning intentions during machine unlearning processes. The idea is novel.
2. The experiments are extensive and show the significant superiority of OUbL over SOTAs in terms of both privacy protection and unlearning effectiveness.
3. The paper is overall well-structured. The narrative is easy to follow.

### Weaknesses
1. Issues with the writing.
   - Lack of clarification on notations. E.g., the mapping of the functions $\mathcal U(\cdot)$ and $\mathcal A(\cdot)$ in the Problem Statement part, the definition of  $\nabla$ and the function $\ell$ in Section 3.1, and the definition of $\mathcal I$ in Eq.(3), the description of $I$ on line 225.
   - Some typos, e.g., missing $\nabla$ on line 221.
   - The $H_\theta^{-1}$ in Eq.(3) denotes the inverse of the Hessian matrix evaluated at $\theta$ **on the dataset $D\backslash D_u$**.
   - The reviewer suggests using another symbol like $\theta_o$ instead of simply $\theta$ to represent the original trained model for reducing ambiguity since $\theta$ seems to be a variable in Eq.(4).
   - The reviewer suggests using the command "\citep" instead of "\cite" when the references are not the objectives in the sentence.
   - Presenting Algorithm 2 with a detailed description in the main text would be better. 
2. Issues with the figures. 
   - Both Figure 1 and Figure 2 describe the main process of OUbL, with Figure 2 being more detailed. Therefore, the reviewer thinks that Figure 1 is unnecessary.
   - Incomplete compilation of notation $\mathcal I_{D_u}$ in Figure 2.
   - The Problem Statement of OUbL on page 3 doesn't include the phase of Figure 2 c), the incremental learning step with clean dataset $D_c$.
3. Inconsistencies between Eq.(4) and Eq.(8). The second objective in Eq.(4) is evaluated on the dataset $D\backslash D_u$ (size $N$) and Eq.(8) is evaluated on the dataset $D_a$ (size $P$).
4. The reviewer thinks that the comparisons between centralized unlearning methods and federated unlearning methods are unfair since the settings are different. The author should discuss the performances of OUbL applied in federated unlearning scenarios.

### Questions
1. What does CSR represent? The authors use SCR to represent the Constructed Samples Rate in Section 4.2 but use it to represent the Clean Samples Rate in Section 4.4.
2. The reviewer is confused about how the second objective in Eq.(4) is satisfied.
3. How do you obtain the datasets $D_a$ and $D_c$? What if the samples in $D_c$ have similar features to the unlearned data? Please add discussions in the main text.

### Soundness
2

### Presentation
2

### Contribution
2
