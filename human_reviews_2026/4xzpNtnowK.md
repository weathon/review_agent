# Sketched Gaussian Mechanism on Matrix for Private Federated LoRA

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Low-Rank Adaptation (LoRA), which modifies frozen pre-trained parameters via the product of two trainable low-rank factors, has been widely adopted for communication-efficient fine-tuning of language models, including extensions to federated learning (FL). Nevertheless, two challenges arise at scale: (i) for very large models, the adapter factors can remain high-dimensional, leading to nontrivial communication costs between clients and the server; and (ii) transmitting local adapters between clients and the server risks privacy leakage. Incorporating differential privacy (DP) by additive mechanisms, e.g., the Gaussian mechanism (GM), often leads to substantial noise amplification, particularly in algorithms that must perturb both low-rank components.

In this paper, we propose the Sketched Gaussian Mechanism on Matrix (SGMM), which couples random sketching with the Gaussian mechanism at the matrix level. Using tools from Rényi differential privacy (RDP), we provide a unified analysis of SGMM’s privacy guarantee and show that, for a fixed privacy level, the required noise magnitude scales as $1/\sqrt{b}$ for sketch dimension $b$. Consequently, for moderate $b$, SGMM attains the same privacy with markedly less noise than GM. We instantiate SGMM within federated LoRA algorithms, including FFA-LoRA and FlexLoRA, where sketching further reduces adapter dimensionality and, in turn, the noise needed to meet a given privacy target, addressing both communication overhead and noise amplification. Experiments demonstrate that, at matched privacy budgets, SGMM-based federated LoRA is at least competitive with and in some settings outperforms non-sketched private baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a mechanism named Sketched Gaussian Mechanism on Matrix (SGMM) for private federated LoRA. By combining random sketching with the Gaussian mechanism at the matrix level , this method aims to simultaneously address two challenges in federated LoRA: the high communication overhead from large adapter factors and the significant noise amplification caused by standard DP mechanisms. The authors provide a theoretical RDP analysis of SGMM and integrate it into the FFA-LORA and FlexLoRA algorithms.

### Strengths
This paper is the first to integrate matrix sketching methods with the federated LoRA, supported by a rigorous privacy guarantee using the Rényi Differential Privacy framework.

### Weaknesses
The paper's novelty is constrained, as the core matrix sketching-plus-noise mechanism exists in prior works such as [1], and recent research offers independent RDP analyses of this technique [2]. Furthermore, the empirical support presented is limited, focusing narrowly on a single dataset with fixed privacy and sketching parameters. Broader experiments covering diverse datasets, sensitivity analyses across key parameters, and evaluations of efficiency metrics are needed to convincingly establish practical benefits.

[1] Yuchang Sun, Jiawei Shao, Songze Li, Yuyi Mao, and Jun Zhang, "Stochastic Coded Federated Learning with Convergence and Privacy Guarantees." 2022 IEEE International Symposium on Information Theory (ISIT), pages 2028-2033, 2022.
[2] Omri Lev, Vishwak Srinivasan, Katrina Ligett, Ayush Sekhari, and Ashia C Wilson, "The Gaussian Mixing Mechanism: Renyi Differential Privacy via Gaussian Sketches." Accepted to the 38th Advances in Neural Information Processing Systems (NeurIPS 2025), 2025.

### Questions
1. How does the theoretical RDP bound derived in Theorem 2.3 compare in tightness to privacy analyses of similar client-level sketching-plus-noise mechanisms, such as the MI-DP analysis in work [1] ?
2. The derived privacy bound suggests noise variance scales approximately as $\sqrt{r}/\sqrt{b}$. Achieving a small privacy loss $\epsilon_p$ thus seems to require $b$ to be comparable to $r$, implying substantial noise if the rank $r$ is large while a strong privacy guarantee (small $\epsilon_p$) is desired. Does this indicate a potential practicality issue for SGMM when applied to high-rank LoRA adaptations under strict privacy constraints?
3. How does model performance, including accuracy and convergence, trade off across varying privacy budgets $\epsilon_p$, sketch dimensions $b$, and LoRA ranks $r$? 

[1] Yuchang Sun, Jiawei Shao, Songze Li, Yuyi Mao, and Jun Zhang, "Stochastic Coded Federated Learning with Convergence and Privacy Guarantees." 2022 IEEE International Symposium on Information Theory (ISIT), pages 2028-2033, 2022.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes the Sketched Gaussian Mechanism on Matrix (SGMM), an adapted approach of the earlier proposed SGMV method, to improve communication efficiency and privacy in federated Low-Rank Adaptation (LoRA) for large language models. It is shown that the proposed algorithm achieves the same privacy protection strength with noise magnitude of $1/\sqrt{b}$ ($b$ is the dimension of sketch), which can be better than the vanilla Gaussian mechanism. The authors combine SGMM with the existing federated learning LoRA algorithms and demonstrate some empirical results as well.

### Strengths
1. The paper proposed an improved version of sketching algorithms for matrices, SGMM. The algorithm is insightful and well-motivated.
2. The paper presents a detailed analysis of the privacy proofs of the proposed algorithm.
2. The paper provides some necessary empirical analysis to show the effectiveness of the proposed algorithm.

### Weaknesses
1. The proposed algorithm relies on SVD of a matrix, which will introduce additional computation and could be hard to scale to large model.
2. Although the authors show the benefit of the proposed algorithm has a certain level of theoretical benefit, their empirical results show that such a benefit might be very limited when such an algorithm is applied to model training.
3. The writing of the paper can be further improved. The readers may find it easy to get lost in what the main contributions are in the paper.
4. There is no utility analysis of the proposed algorithm. While the privacy proof is presented, it is not clear how the magnitude of noise can theoretically affect the (reconstructed) matrices, e.g., the $\tilde{B}^{t, k}_c$.

### Questions
1. Is it possible to provide some theoretical analysis about how the noise and sketch dimension can affect the utility?
2. What will be the computation overhead (considering SVD) when the SGMM is applied to model training?
3. What can be the potential reason that SGMM-FFA-LoRA and SGMM-FlexLoRA do not show significant performance gain? Is it related to the sketch/matrix dimension?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the Sketched Gaussian Mechanism on Matrix (SGMM), which couples random sketching with the Gaussian mechanism at the matrix level. It also provides a unified privacy analysis of the proposed sketching mechanism, which shows that, for a fixed privacy level, the required noise variance scales inversely proportional to the sketch dimension. Finally, they apply the idea to the setting of federated low-rank adaptation, motivated by reducing the communication overhead and achieving client-level privacy.

### Strengths
1. The paper studies the trilemma between communication overhead, privacy and utility in federated low-rank adaptation, which is an interesting and important problem. 
2. The work proposes a sketching approach natively designed for matrix statistics, which is an important problem and improves the SoTA of sketching approaches.
3. The authors also provide the theoretical privacy analysis of the proposed approach, showing its difference with that of the existing vectorization-based sketching approaches.

### Weaknesses
1. The theoretical results in this work suggest the following main messages:

-  For a fixed privacy level, the order of Gaussian noise magnitude is: GM > SGMM > SGMV (with $h=br$)
- SGMM has less computational complexity than SGMV. 
- Using SGMM and SGMV can reduce the communication overhead in federated low-rank adaptation, but potentially at the cost of model utility.

However, the experimental results are very limited and do not fully support the above claims/findings. Further questions are asked about this below.

2. An important limitation of the proposed method when combined with FFA-LoRA is that all the participating clients in federated low-rank adaptation need to use the same sketch matrices $R_B^t$ (as well as $R_A^t$ for Flex-LoRA) in each round $t$. This is a strong limitation, as clients in FL cannot communicate to synchronize their matrices. Also, even the server cannot be aware of the matrices, as from the client-level privacy considered in the paper, it seems that the server is not trusted. Even in algorithm 1 and 2, it is not clear where the sketch matrices in each round $t$ come from.

3. While the work has some contributions, its main achievements seem unclear.

Overall, the work needs to get improved, especially the experimental results.

### Questions
Following the weaknesses mentioned above, I have the following questions:

1. It has been shown that for a fixed privacy level, SGMM adds more Gaussian noise, and has less computational complexity than SGMV. However, in Fig. 1 (a), SGMM seems to get a better utility than SGMV (on average). Is this result inconsistent with the findings mentioned above?

2. Similarly, in Fig. 1 (b), GM clearly performs better than SGMM. SGMM adds less noise than GM, and has some utility loss due to incorporating the random sketching. Considering Fig. 1 (a) and (b), it seems that we cannot make a clear conclusion when comparing the utility of GM with that of SGMV and SGMM. In other words, they get comparable utility. So, at a fixed privacy level, what is gained from using sketching (either SGMV or SGMM)? Only reducing the communication overhead? 

3. In algorithms 1 and 2, the sketch matrices of participating clients in each round $t$, should be synchronized. Have the authors considered this important point?

minor comments:

typo in line 315: analogously

typo in line 348: $R_A^t$ should be $R_A^{t^T}$?

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
2

### Summary
The paper proposes a Sketched Gaussian Matrix Mechanism (SGMM) for federated LoRA. The proposed SGMM is specifically designed to do DP for matrix data instead of the conventional vector data. Theoretical analysis shows the condition under which the proposed SGMM is better than the classical GM, which basically translates to a sufficiently small rank $r$. Empirical results on CIFAR 100 and ViT demonstrate the effectiveness of the proposed method.

### Strengths
This paper is based on a clean and important motivation: DP in federated LoRA. Given that classical DP uses Gaussian mechanisms for vectors, the proposed sketched Gaussian mechanism for matrix is a reasonable step. The theoretical analysis justifies the advantages of the proposed method under low rank adaptation.

### Weaknesses
* Although theoretical analysis seems sound, it is an incremental step from the existing knowledge (e.g., Theorem 1 and standard sketched Gaussian Mechanism). 
* Given the limited theoretical contribution, a strong empirical contribution is expected. However, this paper only evaluate the proposed method on CIFAR100 under a single privacy setting. The experiments only use ViT, while the paper motivates itself using LLMs. Moreover, the empirical results do not show advantage of the proposed method comparing to standard Gaussian mechanisms, in contrast to what the theory might suggest. 
* Minor issue in Line 110: citation Dwork et al. (2006) should use another format.

### Questions
* Remark 2.1 states that "Comparing to classical GM, SGMM attains the same privacy level with smaller noise whenever the sketch dimension satisfies $b\geq \Omega (\frac{r\epsilon_p^2}{\ln(1/\delta_p)})$." However, if we have $r=1$ and the matrix reduce to a vector, how the classical GM is better than the SGMM? I may misunderstand this point, and an explanation is appreciated.

### Soundness
3

### Presentation
3

### Contribution
2
