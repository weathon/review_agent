# UniBP: Toward Universal Backdoor Purification via Fine-Tuning

- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
Deep neural networks (DNNs) remain vulnerable to backdoor attacks, perpetuating an arms race between attacks and defenses. Despite their efficacy against classical threats, mainstream defenses often fail under more advanced, defense-aware attacks, particularly clean-label variants that can evade decision-boundary shifting and neuron-pruning defenses. We present UniBP, a universal post-training defense that operates with only 1\% of the original training data and unveils the relationship between batch normalization (BN) behavior and backdoor effects. 
At a high level, UniBP scrutinizes BN layers’ affine parameters and statistics using a small clean subset (i.e., as small as 1\% of the training data) to find the most impactful affine parameters for reactivating the backdoor, then prunes them and applies masked fine-tuning to remove the backdoor effects. We compare our method against 9 SOTA defenses, 9 backdoor attacks, and various attack/defense conditions, and show that UNBP consistently reduces the attack success rate from more than 90\% to less than 5\% while preserving clean performance, whereas other baselines degrade under smaller fine-tuning sets or stronger poisoning techniques.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper adjusts the parameters of batch normalization layers to mitigate backdoor poisoning.

### Strengths
Backdoors pose a challenging problem.

### Weaknesses
The following uncited paper adjusts batch-normalization parameters
for backdoor defense.  So, it should have been cited and
compared against.
[1] X. Li et al.  Backdoor Mitigation by Correcting the Distribution of Neural Activations. Elsevier Neurocomputing  614, 21 January 2025. http://arxiv.org/abs/2308.09850

Re. line 145: Though some papers do assume a strong adversary (insider) who controls the training _process_, typically backdoor poisoning can be effectively accomplished by just inserting poisoned examples into the training dataset. 

Re. line 148,149: The statement is odd because a very large 
number of prior papers on backdoor defense, particularly
inversion/reverse-engineering approaches,  make exactly this
"post training" assumption.

Is the (cited) I-BAU method compared against in Table 1?

The fonts in the figures and tables are too small.

### Questions
See the above.

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
This paper proposes UniBP, a post-training defense framework to remove backdoors from deep neural networks using only a small clean subset (as little as 1% of data). The method exploits the relationship between batch normalization (BN) statistics and backdoor behavior, pruning the most backdoor-sensitive BN affine parameters and applying masked fine-tuning. Experiments across multiple architectures, datasets, and attack/defense settings show that UniBP achieves a substantial reduction in attack success rate (ASR <5%) while maintaining high clean accuracy, outperforming existing defenses such as NAD, ANP, FST, and TSBD.

### Strengths
- The finding and utilization of BN affine parameters as the key mechanism for backdoor activation is novel. It provides new insights related to backdoors.
- The performance comparisons are based on some newly-proposed methods, e.g., COMBAT, SBL in attacks, and FST, TSBD in defenses, making the results more trustworthy. And the method is simple and effective.
- The methodology is clearly structured into four well-explained stages, and visualizations (e.g., t-SNE, BN statistics) effectively illustrate the underlying intuition.

### Weaknesses
- Some typos exist, e.g., "??" in line 211 and "Batch-norm affine reset" in line 250 is not consistent with the other title (The first letter of each word is not capitalized).
- For the results presentation, it is unfair to color only UniBP in blue for the comparable performance. The baseline performance should also be highlighted and fairly show the comparison.
- Some scalable experiments may help better illustrate the effectiveness, e.g., performance in the ViT model or the ImageNet dataset. The CIFAR-10 and GTSRB are too small, making the generalizability of UniBP unclear.

### Questions
- How to UniBP on transformer-based or normalization-free architectures?
- Can the method be combined with data-free defense approaches to further relax the clean data requirement?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces UniBP, a universal post-training defense method designed to remove backdoors from deep neural networks. Unlike existing defenses that need large clean datasets or fail under adaptive clean-label attacks, UniBP uses only 1% of the original clean data. It works by exploiting a key insight: Batch Normalization layers capture backdoor-related distributional shifts.
UniBP identifies a small subset of BN affine parameters responsible for trigger activation, prunes them, and applies masked fine-tuning to purify the model. Experiments across multiple attacks and architectures show UniBP consistently lowers ASR from over 90% to below 5% while maintaining clean accuracy.

### Strengths
- The paper is well-written and clearly presents the motivation, methodology, and results.
- UniBP is a novel approach that effectively leverages Batch Normalization layers to identify and mitigate backdoors.
- The evalution showcases strong performance across various datasets, architectures, and attack types.

### Weaknesses
- The observation and the technique are not new, which are already explored in prior works.
- The reliance on BN may limit the applicability to models that do not use BN.
- The evaluated attacks and baselines are really limited.
- Further discussion is needed for adaptive attacks given the knowledge of UniBP.

### Questions
This paper is well-written and easy to follow. The entire flow is sound and the experiments are well-designed. However, I have several concerns:

(1) **Novelty**:

The key observation of this paper is that BN layers capture backdoor-related distributional shifts, and can be used to identify and mitigate backdoors. However, this observation has been made in prior works such as [8, 11]. The idea of pruning a set of neurons and then doing fine-tuning to remove backdoors is also similar to [9]. The authors should clarify the novelty of their findings and approach compared to these prior works. What are the key differences and contributions of UniBP that set it apart from existing methods?

(2) **Limited Evaluation**:

The evaluation is limited to a small set of attacks (5) and baselines (5). It would be helpful to see results on a wider range of attacks[1,2,3,4,5], including more recent adaptive attacks that may specifically target embedding distribution[6,7]. Additionally, evaluating on recent defense baselines are important to showcase the superiority of UniBP[8,9,10].

(3) **Dependence on Batch Normalization**:
UniBP relies heavily on the presence of Batch Normalization layers to identify and mitigate backdoors. However, many modern architectures, such as Vision Transformers (ViT) and ConvNeXt, do not use BN layers. This limits the applicability of UniBP to a narrower set of models. The authors should discuss how UniBP could be adapted or extended to work with models that do not use BN, or provide empirical results on such architectures.

(4) **Adaptive Attacks**:
The paper does not sufficiently address the potential for adaptive attacks that could specifically target the UniBP defense. If an attacker is aware of the UniBP method, they may design triggers or training strategies that evade detection by BN parameter pruning. The authors should discuss potential adaptive attack scenarios and evaluate the robustness of UniBP against such attacks.

---
**Reference**:

[1] Turner, Alexander, Dimitris Tsipras, and Aleksander Madry. "Clean-label backdoor attacks." Preprint 2018.

[2] Salem, Ahmed, et al. "Dynamic backdoor attacks against machine learning models." EuroS&P 2022.

[3] Nguyen, Tuan Anh, and Anh Tran. "Input-aware dynamic backdoor attack." NeurIPS 2020.

[4] Liu, Yunfei, et al. "Reflection backdoor: A natural backdoor attack on deep neural networks." ECCV 2020.

[5] Barni, Mauro, Kassem Kallas, and Benedetta Tondi. "A new backdoor attack in cnns by training set corruption without label poisoning." ICIP 2019.

[6] Qi, Xiangyu, et al. "Revisiting the assumption of latent separability for backdoor defenses." ICLR 2023.

[7] Zeng, Yi, et al. "Narcissus: A practical clean-label backdoor attack with limited information." CCS 2023.

[8] Cheng, Siyuan, et al. "Unit: Backdoor mitigation via automated neural distribution tightening." ECCV 2024.

[9] Li, Yige, et al. "Reconstructive neuron pruning for backdoor defense." ICML 2023.

[10] Zhu, Rui, et al. "Selective amnesia: On efficient, high-fidelity and blind suppression of backdoor effects in trojaned machine learning models." IEEE S&P 2023.

[11] Zheng, Runkai, et al. "Pre-activation distributions expose backdoor neurons." NeurIPS 2022.

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
3

### Summary
This paper proposes UniBP, a four-stage post-training defense against conventional backdoor attacks. Built upon the observation that backdoor attacks usually change the BN layer's statistics, UniBP identifies and prunes BN affine parameters and channels with high backdoor sensitivity, followed by masked fine-tuning to restore clean performance. Extensive experiments demonstrate that UniBP achieves a larger reduction in ASR compared with existing baselines.

### Strengths
1. The paper presents a fine-grained method that precisely identifies the BN affine parameters most strongly correlated with backdoor behavior.
2. Extensive experiments show that UniBP achieves consistently better backdoor defense performance than all evaluated baselines.
3. Empirical results demonstrate the universal effectiveness of the proposed approach against multiple traditional backdoor attacks.

### Weaknesses
1. The proposed UniBP builds upon a well-established observation that backdoor training perturbs BN layer's statistics, as also discussed in papers like [1]. The authors should explicitly discuss how the proposed method differs and surpasses those existing line of work.
2. UniBP relies on the BN layers. As a result, it cannot be easily applied to modern generative models that do not have such design. This will limit the potential and generalizability of the proposed method.
3. The paper lacks any discussion or empirical results of UniBP's computational overhead compared with baseline defenses.
4. UniBP exhibits higher clean-accuracy degradation compared to several baselines. The authors should provide a more detailed analysis of this trade-off and present potential mitigation strategies.
5. The paper lacks comparison with several SOTA baselines that also rely on parameter masking or activation tightening, such as [2-3].


[1] Zheng, Runkai, et al. "Pre-activation distributions expose backdoor neurons." Advances in Neural Information Processing Systems 35 (2022): 18667-18680.

[2] Li, Yige, et al. "Reconstructive neuron pruning for backdoor defense." International Conference on Machine Learning. 2023.

[3] Cheng, Siyuan, et al. "Unit: Backdoor mitigation via automated neural distribution tightening." European Conference on Computer Vision. 2024.

### Questions
1. Could the attackers try to regularize the attack to not influence BN layer's parameters? If so, the proposed method may fail easily.

### Soundness
3

### Presentation
3

### Contribution
2
