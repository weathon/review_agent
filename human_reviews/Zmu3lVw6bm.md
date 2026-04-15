# A Closer Look at Personalized Fine-Tuning in Heterogeneous Federated Learning

- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
Federated Learning (FL) enables privacy-preserving, decentralized model training but faces significant challenges in balancing global generalization and local personalization due to non-identical data distributions across clients. While Personalized Fine-Tuning (PFT) adapts models to local data, excessive personalization often degrades global performance. In this work, we present a comprehensive empirical study encompassing seven diverse datasets, multiple model architectures, and various fine-tuning methods under both covariate and concept shift scenarios. Our extensive evaluation reveals critical limitations in existing PFT methods, which struggle with overfitting and exhibit inconsistent performance across distribution shifts, even with careful hyperparameter tuning and regularization. To address these issues, we identify LP-FT, a simple yet effective strategy that combines Linear Probing with full Fine-Tuning, adapted to the FL setting. LP-FT consistently outperforms existing methods, achieving an optimal balance between local personalization and global generalization across all tested scenarios. By investigating the feature change after PFT, we hypothesize the a phenomena dubbed as federated feature distortion is linked to the global generalization. Motivated by the observation, we provide a theoretical analysis of two-layer linear networks, offering novel insights into the conditions under which LP-FT excels, thereby enhancing our understanding of personalization dynamics in FL. This work contributes in three key areas: (1) a rigorous and comprehensive evaluation of PFT methods under diverse distribution shifts, (2) the introduction of LP-FT as a robust and versatile solution to FL personalization challenges, and (3) theoretical foundations that explain LP-FT’s superior effectiveness. Our findings set a new venue for PFT research and provide valuable insights to the broader FL community.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a fine-tuning-based personalized federated learning framework, aiming to solve the data heterogeneity challenge. Compared with existing methods that full fine-tuning the global, the paper combines the linear probing before full fine-tuning, which alleviates the local model overfitting issue. Besides, the paper conduct both experimental evaluation and theoretical analysis to demonstrate the method's effectiveness.

### Strengths
S1: The paper focuses on a long-standing and challenging issue in the FL, i.e., data heterogeneity.

S2: The proposed method is simple and easy to follow, and experimental evaluations show that it outperforms the baseline methods in multiple data heterogeneity scenarios.

### Weaknesses
W1: The technical contribution is limited. As the paper claims, the proposed LP-FT mechanism has been developed and explored in centralized setting, and the paper just adapts the method under the FL setting without more technical improvements.

W2: Some clarifications are confusing and hard to comprehend, please refer to section "Questions" for details.

### Questions
1. What is the exact meaning of LP-FT? There are no specific descriptions about the implementation details in the main paper.
2. Why is fine-tuning the linear head first and then followed by fine-tuning the full model better than direct full fine-tuning?
3. The theoretical analysis is hard to follow and it is recommended to give a concise summary about the analytical logic.
4. In Figure 2, the paper just illustrates the sum of global accuracy and local accuracy. However, it is recommended to supplement the individual performance to show that the local model's performance increases. From the intuitive view, the local model performance should decrease due to overfitting.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
In this work, the authors study the trade-off between personalization and global generalization in Federated Learning (FT).  They found that personalized finetuning (PFT)  causes overfitting and inconsistent performance across distribution shifts in federated learning. Furthermore, the authors identify LP-FT, which is a two-stage fine-tuning method that first performs linear probing and then full fine-tuning to improve local adaptation while preserving global features. The evaluation proves that LP-FT consistently adapts distribution shifts and outperforms existing work.

### Strengths
1. It is interesting and meaningful to study federated learning with distribution shifts, and the authors provide a practical method to enhance the OOD issues.
2. The adaption of LP-FT is general for diverse datasets and distribution shifts, making it versatile for real-world FL scenarios.

### Weaknesses
A substantive assessment of the weaknesses of the paper. Focus on constructive and actionable insights on how the work could improve towards its stated goals. Be specific, avoid generic remarks. For example, if you believe the contribution lacks novelty, provide references
and an explanation as evidence; if you believe experiments are insufficient, explain why and exactly what is missing, etc.

1. The related work is insufficient in two categories. Firstly, some state of the art  Parameter-Efficient Fine-Tuning methods, e.g., LoRA[1], IA3[2], are not discussed and compared. Secondly, some of the FL methods considering distribution shifts are not considered, e.g., FedTHE[3] and FedIIR[4]. 

2. The contribution and novelty need to be strengthened, e.g., except for making LP-FT adapts to decentralized datasets in FL scenarios, what else is vital and seldom studied. At present, the contribution of LP-FT is solid in centralized learning but not customized for FL.

3. The computation cost of introducing LP-FT is overwhelming, please provide both empirical and theoretical analysis to quantify it.

4. The empirical studies could be more extensive. On the one hand, the authors could illustrate some experimental details like which cifar10-C is evaluated. On the other hand, the authors are expected to study non-IID scenarios in FL, because in FL, modeling IID decentralized data is similar with centralized training.

5. The experiment analysis is not conclusive, e.g., in Table 1, why LP-FT outperforms other methods on the condition that some results are worse than baselines. 

6. The theoretical analysis overlooks the data is heterogeneous.

[1] Liu H, Tam D, Muqeeth M, et al. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning[J]. Advances in Neural Information Processing Systems, 2022, 35: 1950-1965.
[2] Hu E J, Shen Y, Wallis P, et al. Lora: Low-rank adaptation of large language models[J]. arXiv preprint arXiv:2106.09685, 2021.
[3] Jiang L, Lin T. Test-Time Robust Personalization for Federated Learning[C]//The Eleventh International Conference on Learning Representations.
[4] Guo Y, Guo K, Cao X, et al. Out-of-distribution generalization of federated learning via implicit invariant relationships[C]//International Conference on Machine Learning. PMLR, 2023: 11905-11933.

### Questions
1. How LP-FT outperform existing Parameter-Efficient Fine-Tuning methods for federated scenarios?
2. Can authors discuss the most promising contributions for this work?
3. Whether this overfitting and inconsistent issues happen for language data?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose an approach LP-FT to tackle the key challenge of balancing local personalization and global generalization. They conduct extensive empirical evaluation. They combined linear probing with full fine-tuning and provided a solid theoretical analysis using a two-layer linear network. Extensive experiments are conducted to evaluate the performance of the proposed mechanism.

### Strengths
1. The motivation and overall presentation of the paper is clear.
2. The paper is well-written.
3. The overall approach of using linear probing with full fine-tuning to prevent feature distortion is interesting.
4. The paper presents comprehensive experimental observation and theoretical analysis.
5. The proposed method shows impressive accuracy improvement on the datasets.

### Weaknesses
1. This work focuses on the PFL and GFL in feature distribution shift scenarios. Previous PFL work mostly works on label distribution skew. Feature-skewed FL adopts PFL paradigms. Why this work emphasizes the feature skew in personalization and generalization? What about this work’s insight on vanilla label distribution shifts? 
2. The novelty of proposed method is somehow limited. It is mostly based on LP-FT. The contribution on comprehensive analysis is significant, while the novelty of technical method seems weak.
3. The organization of sections is somehow confusing. It could be better to make it clear as observations, methods, and experiments.
4. Lack of comparison with PFL baselines. The author only considers different fine-tuning paradigms.

### Questions
Please refer to weakness.

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
This paper tackles the challenges of balancing personalization and global generalization in federated personalized fine-tuning. The authors perform comprehensive evaluations on various scenarios (e.g., different types of data heterogeneity, datasets, and models) to demonstrate the limitation of existing methods, where the model's generalizability will be degraded by excessive personalization. By identifying the critical cause of this overfitting problem  (i.e., feature distortion), the authors introduce LP-FT, a strategy that combines linear probing and full fine-tuning, to solve this problem. Through comprehensive empirical evaluations and theoretical analyses, the authors demonstrate that LP-FT consistently outperforms existing approaches by preserving pre-trained global features, thereby mitigating the adverse effects of .

### Strengths
**S1**. The manuscript is well organized in presentation and easy to understand.

**S2**. The solution to the problem has a good motivation. The authors first observe that the feature distortion before the last layer is highly correlated to the model generalizability.  Based on this observation, they introduce LP-FT to preserve global features during the fine-tuning process, thus better preserving global knowledge than existing methods.

**S3**. The proposed method is simple but effective. It effectively balances local personalization and global generalization, addressing the limitations of existing PFT methods that tend to overfit and exhibit inconsistent performance.

**S4**. Extensive experiments and theories were conducted to verify the advantages of the proposed method in preserving the generability when personalizing models.

### Weaknesses
**W1**. There has been a rapid evolution in personalized fine-tuning techniques in FL [1]. The latest baseline in this paper was published at two years ago (e.g., 2022). More evidences are needed to illustrate whether feature distortion is still a common problem for recent methods.

**W2**. I'm concerned about the novelty of this work. The main contribution of this work lies in adapting an existing method to a new scenario (e.g., personalized fine-tuning) to solve a widely studied problem in the past years (i.e., feature distortion [2][3]). Please further highlight the difference of this work to existing methods.

[1] Tamirisa R, Xie C, Bao W, et al. FedSelect: Personalized Federated Learning with Customized Selection of Parameters for Fine-Tuning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 23985-23994.

[2] Li X, Jiang M, Zhang X, et al. Fedbn: Federated learning on non-iid features via local batch normalization[J]. arXiv preprint arXiv:2102.07623, 2021.

[3] Collins L, Hassani H, Mokhtari A, et al. Exploiting shared representations for personalized federated learning[C]//International conference on machine learning. PMLR, 2021: 2089-2099.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
4

### Contribution
3
