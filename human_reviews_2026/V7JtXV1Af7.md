# Active Side Channel Analysis for Cross-Device Attack

- Decision: Reject
- Scores: 2, 2, 2, 4, 2

## Abstract
Side Channel Analysis (SCA) exploits relationships between physical signals of a device and its actual computation to extract sensitive information, causing serious threat to privacy and security. Among various approaches, Deep Learning-based profiling attacks (DL-SCA) have recently emerged as one of the most powerful methods due to their ability to fully characterize the target devices. However, they suffer from major drawbacks including huge data consumption and lack of portability across different target devices. This paper introduces Active SCA (ActSCA), a $unique$ and $generic$ framework for boosting performance of any base DL-SCA model. 
ActSCA fundamentally differs to existing research as follows.
Firstly, rather than relying on large training data in the profiling stage, it $actively$ selects subsets of training data and $iteratively$ refine the model to avoid overfitting, thus enhancing performance. Secondly, in the attack stage, ActSCA $exploits$ existing training data pool from profiling devices to construct $separate$ attack models for different target devices without requiring any training data from the attacking devices as is the case in other existing methods by using only few unlabeled SCA traces collected during the attacking phase to guide the model adaptation process.
These make ActSCA a highly $portable$ and $practical$ attack method. We demonstrate its performances on the Post Quantum (PQC) Kyber algorithm using power leakage to retrieve secret keys. ActSCA significantly improves the performances of all employed base models and outperforms all recent approaches like CNNC, MDMSD, ZMUV, MMD, ADA in terms of mean rank and top-$k$ accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Active SCA (ActSCA), a framework to improve the performance of deep learning based side channel attacks (DL-SCAs). Instead of using all profiling traces, ActSCA selects a subset that is considered more beneficial for training. To address performance degradation in cross device scenarios, the method selects profiling traces that are most similar to traces obtained from the target device and uses them for model adaptation. Experiments on Kyber, a post quantum KEM, and on AES suggest that ActSCA can improve attack performance.

### Strengths
- The paper presents empirical evidence that training on a carefully selected subset of profiling traces can improve performance.  
- The paper proposes a method that reduces mean rank in cross device settings.  
- The paper evaluates on two cryptographic targets with different leakage characteristics, Kyber and AES.

### Weaknesses
- The paper does not sufficiently justify why using only a subset of traces should be effective.  
- The scope of cross device experiments is limited, which weakens the empirical support for general effectiveness.  
- Although much prior DL SCA work focuses on AES, the AES results appear only in the appendix and the reported performance is relatively poor.

### Questions
1. Justification for the effectiveness of subset selection

The paper motivates subset selection by referring to possible overlap or distortion in signals and by Theorem 1. Both points require stronger justification.

For the first point, the paper should clarify what overlap and distortion mean in the context of side channel traces. Under a common DL-SCA threat model, the attacker controls a device of the same type as the target and can collect as many profiling traces as needed under controlled conditions, for example with trigger signals. Profiling traces should therefore reflect leakage of the cryptographic operation. It is not clear in what conditions certain profiling traces would be unhelpful for training. Please clarify concrete situations that would generate profiling traces that should be discarded and explain why discarding them improves generalization.

For the second point, Theorem 1 bounds the difference between the average loss on the full data pool $D_P$ and the average loss on a selected subset $S$. This shows that training on $S$ can achieve a loss close to that obtained with $D_P$.It does not explain why training on $S$ would outperform training on $D_P$. Since an attacker can in principle use all available profiling data, it is not obvious why deliberately using fewer samples should help. Additional theory or analysis would be helpful to explain when and why subset training can surpass full data training.

2. Cross device experiments should include more challenging conditions

The current experiments appear to use power measurements on ChipWhisperer, which mainly reflect differences across devices of the same type. Prior cross device studies also examine different EM probe positions, which is often more realistic when direct power measurement is unavailable. Probe placement differences can induce a large distribution shift between profiling and attack traces. Phase C relies on selecting profiling traces similar to the target traces, which implicitly assumes that the two distributions remain comparable. Please evaluate ActSCA under a more realistic scenario, for example with different probe positions, and report whether the method remains effective.

3. AES results should be discussed in the main text

Most DL-SCA studies use AES as a benchmark. In this paper, AES results are reported only in the appendix. They should be discussed in the main text. In addition, the results on ASCAD in Table 12 appear weak. SOTA work (e.g., Hajira et al., 2024) reports recovering subkeys with around 20 attack traces, while here the mean rank remains high even with 500 to 800 traces and does not decrease with more attack traces. Also, because ASCAD is not a cross device dataset, please clarify how ActSCA and comparison methods such as MMD were applied. These points raise concerns about whether prior methods were implemented correctly.

### Soundness
2

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
This paper studies the effectiveness of applying active learning techniques in Deep Learning-based Side Channel Analysis (SCA). Given a large pool of power traces together with their secret keys, the authors propose ActSCA, a three-stage framework that incorporates ideas from active learning to train and fine-tune an arbitrary base model: Instead of training the model on all the available labeled data, Phase A selects a small but informative subset of data to train on by (1) computing the $k$-mediod centriods of all power traces, and (2) iteratively extending this set with power traces that are maximally different from all selected samples, while ensuring the label distribution in the selected set is close to uniform. 
After training a model on this dataset, Phase B proceeds to use this model to select power traces in the original dataset where the model's output distribution is very uncertain (i.e., the entropy is large), while again ensuring label balance. The model is then updated by training on these newly selected points, as well as on points that are in the same $k$-medoid cluster to prevent catastrophic forgetting. In Phase C, we are given power traces from a target device we wish to attack (i.e., infer the secret key). Using these traces, this phase will find "similar" power traces in the original dataset, and fine-tune the classification head of the model (while freezing the rest). Empirical evaluation of ActSCA shows that it significantly improves over the baseline and that it outperforms various portability methods from the SCA literature.

### Strengths
+ **Motivation.** The work is well-motivated and the author's objectives are clearly articulated. Deep-Learning based SCA and issues like portability are important research topics.

+ **Empirical Results.** The empirical evaluation is thorough, includes interesting ablation studies, and provides evidence that ActSCA outperforms the baseline model and various methods for SCA portability.

### Weaknesses
+ **Lack of novelty.** Central techniques like selecting $k$-mediod centroids [1], uncertainty sampling [2], label balancing, and model fine-tuning by freezing weights, are well-known approaches in the active learning and deep learning literature. In particular, claiming that Phase B, a combination of uncertainty sampling and label balancing is a "novel *iterative self-improvement* approach" (L221) is very misleading. The application of active learning techniques in Deep-Learning based SCA has also been explored previously [3].

+ **Theoretical Justification.** The proposed strategies, especially the construction of $J_1$, $J_2$ and $D_3$ seem ad-hoc, lack theoretical justification, and introduce several hyperparameters.

+ **Issues with Theorem 1**:
	+ The assumptions of Theorem 1 are clearly not met in ActSCA: (1) The set $S$ is *not* used for training, as it is enlarged using the balanced max-min sampling (BMMS) strategy. Theorem 1 makes no statement about BMMS. (2) It is assumed that the loss $l(\cdot, y, \theta)$ is bounded by $L$, which is not true if the model is trained with a typical cross-entropy loss (which is what the authors use in their experiments). (3) Achieving $0$ loss for all examples $\in S$ is also a strong assumption, and will not be satisfied in most practical settings where $M_1$ is a parametric model.
	+ The interpretation of Theorem 1 is factually incorrect and highly misleading: "As $N_P$ is typically large, this theorem guarantees that training on the subset $S$ achieves similar generalization as using the full dataset while mitigating overfitting." (L207-L209). Theorem 1 makes no statement about the *difference* in generalization error of a model trained on the full dataset, and a model trained on $S$. Further, Theorem 1 makes no statement about the mitigation of overfitting.
	+ Except for this interpretation, Theorem 1 is never used in the paper, and thus does not add value to it. 
	+ Proof of Theorem 1 (L1411): While $\eta_k$ refers to the *true posterior* (L1399), the authors suddenly assume it is $K^\mu$-Lipschitz. Theorem 1 only assumed that the *model* $M_1$ was $K^\mu$-Lipschitz. Any such assumption should be made explicit, and must be included in the main text.

+ **Presentation of results.**
	+ Figure 3 (A): It is very hard to make sense of this plot and I would strongly suggest presenting these results a different way. For example, it makes no sense to me why the mean rank of different test keys (that are next to each on the x-axis) are "connected".
	+ Showing 13 subplots in Figure 3 makes it hard to find the important results. Many of the plots seem redundant, and I would suggest that the authors focus their presentation on the most vital empirical findings.

+ **Label Distribution in BMMS:**
	+ L188: "The label distribution in $D_1$ is preserved by favoring samples with rare labels [...]". 
	+ This statement is misleading. At initialization of BMMS, we have $D_1 = S$. The label distribution of $S$ will *not* be preserved during BMMS. The goal is clearly *label balance*, *not* to preserve the original label distribution of $S$.
	
+ **Evaluation:** 
	+ L149 claims that ActSCA "enables" attacks with very small number of unlabeled target traces, and L476-L477 claims that "ActSCA [...] only needs to collect one or few target traces and thus is much practical." However, the authors only empirically validate their approach with $\geq 10$ traces per key (L302: "The number of attack tracks in $D_T$ varies from 10 to 100.")
	+ Table 11 shows the medians of mean ranks of ActSCA on unprotected AES. On Device 1, the baseline model outperforms ActSCA in all phases, but "Phase C" is still shown in bold (although it is *not* the best result).

+ **Notation**:
	+ L180: $i \in [1,k]$ should be $i \in \\{1,\dots,k\\}$
	+ Suggestion: Don't overload $l$ for both loss and key label (use e.g. $\ell$ for point-wise loss). Don't overload $\sigma$ to both mean the softmax function, and the final fully connected layer of the neural network.
	+ L1419: the selected subset should be $S$, not $s$ 
	
+ **Typos**:
	+ L051: "[...] the attacker *can* access to a physical mitigated version" -> the attacker *has* access to a physical mitigated version
	+ L053: "[...] due to their ability to fully *characterized* the profiling devices" -> characterize
	+ L102: "[...] and used as *or* unseen data" -> "or" does not make sense
	+ L227: "model-perpspective"

--------

##### References
+ [1] Sener, O. and Savarese, S., 2017. Active learning for convolutional neural networks: A core-set approach. _arXiv preprint arXiv:1708.00489_.
+ [2] Lewis, D.D., 1995, September. A sequential algorithm for training text classifiers: Corrigendum and additional data. In _Acm Sigir Forum_ (Vol. 29, No. 2, pp. 13-19). New York, NY, USA: ACM.
+ [3] Yu, H., Wang, S., Shan, H., Panoff, M., Lee, M., Yang, K. and Jin, Y., 2023, May. Dual-leak: Deep unsupervised active learning for cross-device profiled side-channel leakage analysis. In _2023 IEEE International Symposium on Hardware Oriented Security and Trust (HOST)_.

### Questions
+ The distance metric on L179 is simply the Euclidean distance, although we wish to measure similarity between time-series data. Have the authors tried distance-like measures that are more appropriate for time-series data (e.g., dynamic time warping distance)?
+ It seems likely that BMMS selects *outliers*, i.e., points that are maximally distant to all points in $D_1$.  This is especially true when $\gamma \approx 1$, but can also easily happen with small $\gamma$, since the label information does not guard from selecting "pathological" outliers. Has this been observed empirically and wouldn't it be beneficial to also filter extreme outliers?
+ How crucial is label balance in this setup? What is the label distribution of $D_P$ in the experiments, and how does the choice of $\gamma$  and $\alpha$ (in $J_1$ and $J_2$, respectively) affect the result? 
+ Evaluation on single-trace attacks would support the claim made on L476-L477. Have the authors conducted such experiments?
+ The authors compare their approach to portability methods like MDMSD, ZMUV, MMD, and ADA. All of these methods seem to perform *worse* than the baseline (e.g., Figure 4), which is surprising. Can the authors interpret/discuss this result?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents Active Side Channel Analysis (ActSCA), a framework designed to enhance Deep Learning-based Side Channel Attacks (DL-SCA), which aim to extract sensitive information by analyzing physical signals from cryptographic devices. Traditional DL-SCA methods typically require large datasets and lack adaptability across different devices. ActSCA seems to address these limitations through two innovations: Active Training and Adaptive Attacking. The authors validate the effectiveness of their approach through experimental evaluations conducted on both local and public datasets.

### Strengths
(1) This idea is straightforward and can be easily understood.

(2) The experiments consider multiple machine learning models, with seemingly more rigorous evaluation methodologies than many previous papers in this area.

### Weaknesses
(1) My major concern with this paper is its lack of novelty. Since deep learning (DL)-based side-channel analysis (SCA) is not new, the contribution of this work appears to be quite limited. The authors appear to simply extend one of the existing active learning methods—uncertainty sampling—and make a minor modification to it (referred to as balanced uncertainty sampling) to generate a sample set that is then used to update the DL model. The main focus of this paper still lies in applying some very  basic ideas from the DL domain to the SCA domain to reduce training costs and improve portability—issues that have already been addressed in prior research over the past few years. I believe the authors can easily find many similar papers published in well-known venues such as CHES, ICCAD, DAC, and others. Therefore, this paper does not seem to significantly advance our qualitative understanding of this area within the security domain.

(2) The adversary (or threat) model that the authors aim to address is also somewhat confusing. For example, in a profiled attack, we typically have two devices: the profiling device and the target (or attacking) device. Do the authors assume that these two devices share similar or different architectures, or that they use the same or different AES implementations? Moreover, while this paper focuses on profiled attacks, such attacks are difficult to deploy in real-world scenarios. In practice, the attacker often cannot obtain a copy of or a similar device for profiling the DL model before applying it to the target device for secret key recovery. 

(3) The experimental setup is also unclear. Do the authors use fixed keys or varied keys for testing?

(4) Finally, the writing needs significant improvement. In critical parts of the paper, it is hard to tell what the authors did in terms of experimentation and analysis, or what motivated the choices they made.

### Questions
Please refer to my comments for more details.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents ActSCA, a framework to train deep-learning based profiling attack models more effectively. The proposed method is mainly made with three phases. In phase A, it trained the model on a selective subset of data, In phase B, it iteratively trains the model on another selective subset of data that the model predict poorly with high uncertainty. In phase C, which is during the attacking time, it retains the model with another selective subset of data that have the most similarity to the data samples from the target device. Empirical results show that it achieves a better attack performance in terms of mean rank and top-k accuracy compared to the baseline.

### Strengths
1. The empirical results shown in Fig. 3 demonstrated a large margin of improvement using the proposed method. 
2. Over all, this paper is well structured and easy to follow.

### Weaknesses
1. I don't think the math in session 2.1 helps to prove any point. In the paper, a repetitive statement is that "using the full dataset for training suffers from overfitting". But it is lack of evidence to show it is indeed the reason of worse performance, or explanation or discussion around why training on a small subset helps to avoid overfitting. Intuitively, training on less data is more likely to overfit (to that smaller subset of data), and it is not clear why it avoid overfitting on the full data and why carefully choosing the subset of data is important. 
2. There are criteria in phase A and phase B on how to choose the subset of data. Although intuitively they all makes sense, the paper is lack of empirical evidence that using those criteria leading to a better performance compared to the baseline. For instance, it would be interesting to see what if in phase A we randomly choose some samples. What is the performance difference between randomly sampled subset and the carefully chosen subset. This kind of ablation study is important to demonstrate the importance of the proposed tricks.
3. Overall the contribution is not that significant. The methods in the paper is more like tricks that principled approaches. 
4. The graphs are blurred and the font size is too small to read.
5. I feel the experiment results can be better presented in tables than in figures.

### Questions
1. As already discussed in "Weakness", why training on a subset of data helps to avoid overfitting?
2. How much more computation is introduced with the proposed method (as it needs to sample the data) compared to the baseline when obtaining the experiment result?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Active SCA (ActSCA) is a framework for boosting performance of any base DLSCA model. Instead of large training data in the profiling stage, it actively selects subsets of training data and iteratively refines the model to avoid overfitting. In the attack phase, ActSCA uses the training data and a small amount of target device's traces to construct a separate attack model that works well for specific target devices.

### Strengths
- **Relevant topic of portability for SCA.** The paper considers portability issues in (DLSCA) where the trained models do not generalize enough to another device (clone or similar).
- **Extensive experiments performed and reported.** The authors consider different keys, devices, crypto systems and attack methods, while also performing multiple runs for statistical significance.
- **Better performance than compared methods.** The paper reports significant improvements over many methods for two crypto systems.

### Weaknesses
- **Design choices are not always clear.** For example, the Phase A selection method involves clustering, however it is not clear why this specific clustering is good specifically for SCA traces. Traces further from a centroid could be more noisy, leaking less information, allowing the method to still select "bad samples", as the authors also discuss in the paper. In Phase B, again, choosing samples based on uncertainty seems to also lead to selection of possibly some outlier/noisier traces.
- **Evaluation has some flaws.** Guessing entropy is more commonly used due to being more representative and stable than a key rank metric, while mostly key rank is reported in this work. SoTA on ASCAD dataset does not seem to be SoTA as there are works much successful than those reported here, where GE of 1 is achieved with a small amount of traces. Baseline is unclear as it is the proposed method using the complete training dataset while authors do not report how the phases are adapted to that change and, there should be some, since most of the method is about selecting samples from dataset. 
- **The method might not be as generic as claimed.** The results on different model architectures vary significantly (based on results in Table 1. in Appendix.)
- **One of the issues authors are trying to solve in this work is not a relevant research question**, and it has different known ways to handle it. In particular, this is the question: "can we improve the model performance by carefully exploiting a subset of data rather than asking for more data?". There are more common methods, such as data preprocessing, removing outliers, regularization methods and many more that work on improving model's generalization, performance, etc. I'm not convinced this is an issue requiring such an elaborate method for solving it.

### Questions
- What makes clustering a good method for selecting representative SCA traces? How can you avoid selection of "bad samples"?
- In Phase B, why do you choose based on uncertainty?
- Why is key rank the main metric in this work?
- Table 12 reports results on ASCAD dataset, standard benchmark for DLSCA, and it is known that this dataset can be broken with very few traces, thus the results reported are not representative. How come SoTA for ASCAD does not have better performance?
- You should explain in more detail your Baseline method. Since it is using the complete dataset, it is unclear how the steps of your proposed methods are modified.
- The results on different model architectures vary significantly (based on results in Table 1. in Appendix.). Why is that and does that not show that the method is somewhat dependent on the architecture?
- Training data from the target/attack device is not a requirement in the standard DLSCA framework (as you also state in the abstract). Training data is assumed from a clone device, while from the target data, measurements (without labels) are collected. So, it is unclear why is this fact so emphasized in this work as it is not a novel thing or an improvement over other work that the training data is not used. It also confuses reader to think that training data is normally used in DLSCA which is not true. Thus, perhaps it would be better to focus on what is it that you do in the attack phase to allow portability to different devices, while removing this repeated emphasis on attack traces not being part of training.
- One of the motivations for Phase A (using subset of training data) is overfitting. Overfitting can be reduced/mitigated with regularization, hyperparameter tuning, and other already well-established methods in ML domain. So, why is this method better than these other techniques?
- Large datasets require increased computation costs, but this method also seems to require a lot of computational power for all the phases and the procedures in each of them. In appendix you provide some information on runtime, but Phase A is not shown but is a part of your proposed method, making this unfair comparison. Can you provide more runtime analysis?
- It is not clear which level of portability are you considering until the experiments section. With the level of portability, I mean that you can look at the portability between identical devices where the difference comes from manufacturing process, or you can be looking at actual different hardware or algorithm running. So, can you give some indication on which portability level are you considering earlier in the text, as it will also help understand the proposed method better.


Additional comments/questions (less relevant)
- In appendix there is mention of 50K validation set. How is that one used in your proposed attack method?
- The profiling dataset of 200K samples, is it collected with the 300 different keys as well, or a single one?
- Not clear why the name reverse training scheme.
- KNN not defined. 
- Figure 4 - What are the axis?
- "Unused training data will be retained in the training pool and used as or unseen data..." - used as what?
- What are CNNC, MDMSD, ZMUV abbreviations of?
- Figures should be closer to the text referencing them.
- hyper-parameter turnings?
- Table 11 and text about that table report slightly different numbers (118-119, 86.2-87.2)

### Soundness
2

### Presentation
3

### Contribution
2
