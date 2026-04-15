# Partial Optimal Transport for Open-set Semi-supervised Learning

- Decision: Reject
- Scores: 6, 5, 5

## Abstract
Semi-supervised learning (SSL) is a machine learning paradigm that leverages both labeled and unlabeled data to improve the performance of learning tasks. However, SSL methods make an assumption that the label spaces of labeled and unlabeled data are identical, which may not hold in open-world applications, where the unlabeled data may contain novel categories that were not present in the labeled training data, essentially outliers. This paper tackles open-set semi-supervised learning (OSSL), where detecting these outliers, or out-of-distribution (OOD) data, is critical. In particular, we model the OOD detection problem in OSSL as a partial optimal transport (POT) problem. With the theory of POT, we devise a mass score function (MSF) to measure the likelihood of a sample being an outlier during training. Then, a novel OOD loss is proposed, which allows to adapt the off-the-shelf SSL methods with POT into OSSL settings in an end-to-end training manner.
Furthermore, we conduct extensive experiments on multiple datasets and OSSL configurations, demonstrating that our method consistently achieves superior or competitive results compared to existing approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper considers an open-set semi-supervised learning where there potentially are “outliers” in the unlabeled data distribution. The paper provides a novel loss function inspired by Partial optimal transport to handle OOD detection and demonstrates the effectiveness and robustness of the proposed method on multiple datasets.

### Strengths
1. Strong empirical performance
2. Various ablations suggest that the method is robust and has a lower computation time and other baselines.
3. A connection to optimal transport is intuitive

### Weaknesses
1. Lack of clarity in writing.  I found it hard to understand what is the main idea of the paper up until page 6. The author mentions in the abstract/ introduction that a mass score function (MSF) to measure the likelihood of unlabeled samples being outliers, yet I did not mention how this is related to OT/POT and it’s not clear to me how OT is beneficial to the OSSL task. The following sentence is helpful for me to understand the idea,  “we can utilize the transport mass as a reliable OOD score, where a sample with a smaller value of mass score function tends to be an OOD sample”. However, it is mentioned on page 6. It would be nice if one could provide something like this earlier in the paper and provide a clear problem setting early on.


2. Many definitions and acronyms are used before being defined (see questions)

3. The definition of distribution in equation 8) is not mathematically valid? By adding a factor of k, the sum of the probability mass is greater than 1 and therefore is not a valid probability distribution.

I am willing to increase the score if these issues are addressed.

### Questions
1. OSR is not defined, MSR is mentioned before it is defined in section 2.2.
2. Section 4.1, the distribution L and U are not defined.
3. Section 4.1, “the features of these d-dimensional samples”, do you mean the features or samples that has d-dimensional ?
4. Notation in equation 7) is not clear. Does this means T1_{\mathcal{L}} \leq \mathcal{L} point-wise less than or equal to ?
5. In algoirthm3, L_x and L_u is not defined in the main text ?
6. What is the number 50, 100, 500 for in Table 5 ?
7. “magnituWde” -> “magnitude” ?

### Soundness
4 excellent

### Presentation
2 fair

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper tackles the open-set semi-supervised learning (OSSL) challenge, specifically aiming to frame the treatment of out-of-distribution data (OOD) as a partial optimal transport (POT) problem. It introduces a mass score function (MSF) designed to evaluate the likelihood of a sample being an outlier during training. Additionally, the paper presents an OOD loss, allowing conventional semi-supervised learning methods to be adapted for OSSL scenarios via end-to-end training. The authors compare their proposed method against MTCF, T2T, and OpenMatch, on CIFAR10, CIFAR100, and Imagenet-30, showing superior performance.

### Strengths
* Semi-supervised learning is a significant area of research in machine learning, aiming to enhance performance by effectively utilizing both labeled and unlabeled data. 

* The OOD angle used in the paper makes it interesting to a broader audience.

* Incorporating (partial) optimal transport as a framework is a novel and innovative aspect of this work.

### Weaknesses
* Respectfully, the novelty of the method is limited and the paper overclaims novelty.
   *  For instance, one main contribution of this paper is the introduction of the "novel" MSF score. The score function essentially corresponds to what is commonly referred to as "barycentric projection," a concept well-documented in both classical and contemporary optimal transport (OT) theory literature (for reference, please see sources such as [Ambrosio et al.](https://link.springer.com/book/10.1007/b137080)). In this context, it is more appropriate to state that the paper utilizes classical concepts from OT theory to address new application challenges. The sentence “we devise a new score function” is more or less misleading.

* The parameter $k$, which deals with the amount of redundancy, plays a crucial role in the methodology presented in the paper. Varying the value of k leads to significant variations in the outcomes of ODD detection. It would enhance the paper's quality if it delves into the process of determining this value. Specifically, the paper could explore methods for assessing the amount of data that should be classified as outliers before initiating the algorithms. 

* Some implementation details and important ablation studies are missing from the paper. For instance, the utilized batch size and the effect of having a small batch size (which presumably reduces the performance of the proposed method) are missing from the paper. 

* The rationale behind the decision to use (10) instead of the original constraint (7), i.e., enforcing all mass from $\mathcal{L}$ to be transported to a subset of $\mathcal{U}$, is not well presented. Couldn't the unsupervised data be missing an entire class? In that case, the missing classes in $\mathcal{L}$ must be destroyed, i.e., not transported, and the constraints in (7) would allow that. I believe this can easily happen in minibatch training. 

* Some of the very relevant references are missing from the paper: 
   * Rizve, M.N., Kardan, N. and Shah, M., 2022, October. Towards realistic semi-supervised learning. In European Conference on Computer Vision (pp. 437-455). Cham: Springer Nature Switzerland.
   * Xu, R., Liu, P., Zhang, Y., Cai, F., Wang, J., Liang, S., Ying, H. and Yin, J., 2020. Joint Partial Optimal Transport for Open Set Domain Adaptation. In IJCAI (pp. 2540-2546).
   * Yang, Yucheng, Xiang Gu, and Jian Sun. "Prototypical Partial Optimal Transport for Universal Domain Adaptation." (2023).

### Questions
* For Algorithm 2, in the line of OOD score, shouldn't the formula be $Score_\{\mathcal{U}\}=\mathbf{T}^T\mathbf{1}_n$?

* The transportation cost is set to "Cosine distance." The definition  "d(x,y)=1-Cosine(x,y)"  is only a true metric if $x,y\in \mathbb{S}^{d-1}$, i.e., $x$ and $y$ are unit vectors. Is your backbone returning unit vectors? Even if that is the case, and for the sake of mathematical rigor, I suggest adhering to the Euclidean distance, which is equivalent to the cosine distance when $x$ and $y$ are unit vectors and is still sensible when they are not!

### Soundness
3 good

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
This paper focuses on studying the problem of Open-Set Semi-Supervised Learning (OSSL). The authors present a novel framework that transforms the OSSL problem into the Partial Optimal Transport (POT) problem. The authors aim to leverage the benefits of POT to detect the OOD samples. Empirically, POT achieves competitive performance on various benchmarks.

### Strengths
-	This paper is straightforward and well-written. It is quite easy to follow.
-	The paper solves Open-Set Semi-Supervised Learning (OSSL), an important ML problem in practice.
-	Empirical results demonstrate that POT can achieve SOTA results on several benchmarks.

### Weaknesses
-	Based on the description provided, it is possible that the article's approach could be categorized as an auxiliary OOD classifier approach, similar to methods such as MTCF, T2T, and OpenMatch. What is the difference between the proposed method and them? A more detailed discussion may be required.
-	The author's explanation for why POT is more effective at detecting OOD is not adequately provided.
-	In similar settings, Partial Optimal Transport (POT) has also found applications, such as in Open-set Domain Adaptation and Positive-Unlabeled Learning. The authors should consider discussing the connections and distinctions between their work [1-3] and the research presented in these articles. And what are the strengths of POT for Open-Set Semi-Supervised Learning? Are there some special designs for Open-Set Semi-Supervised Learning compared with other tasks, such as PU leanring, Open Set Domain Adaptation?
-	The authors should offer an explanation for why Fixmatch algorithm yields better results compared to certain Open-Set Semi-Supervised Learning (OSSL) methods.
-	There lack of many experiment details in the paper, such as the specific parameter settings for Fixmatch and the implementation specifics of the T2T algorithm.
-	Table 3 lacks some of the comparative algorithms present in Table 1.
-	There is an inconsistency in the notation of the k in Algorithm 3.
-	On page 8 in the experimental section, $L_{ood}$ --> $\lambda_{ood}$
-  What is "graph" in the last of Subsection 2.2?

[1] Partial Optimal Transport with Applications on Positive-Unlabeled Learning, NeurIPS 2020.

[2] Joint Partial Optimal Transport for Open Set Domain Adaptation, IJCAI 2020.

[3] Prototypical Partial Optimal Transport for Universal Domain Adaptation, AAAI 2023.

### Questions
Please see the weakness for details.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
