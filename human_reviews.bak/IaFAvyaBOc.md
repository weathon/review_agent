# Collaborative and Efficient Personalization with Mixtures of Adaptors

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 1

## Abstract
Non-iid data is prevalent in real-world federated learning problems. Data heterogeneity can come in different types in terms of distribution shifts. In this work, we are interested in the heterogeneity that comes from concept shifts, i.e., shifts in the prediction across clients. In particular, we consider multi-task learning, where we want the model to adapt to the task of the client. We propose a parameter-efficient framework to tackle this issue, where each client learns to mix between parameter-efficient adaptors according to its task. We use Low-Rank Adaptors (LoRAs) as the backbone and extend its concept to other types of layers. We call our framework Federated Low-Rank Adaptive Learning (FLoRAL). This framework is not an algorithm but rather a model parameterization for a multi-task learning objective, so it can work on top of any algorithm that optimizes this objective, which includes many algorithms from the literature. FLoRAL is memory-efficient, and clients are personalized with small states (e.g., one number per adaptor) as the adaptors themselves are federated. Hence, personalization is--in this sense--federated as well. Even though clients can personalize more freely by training an adaptor locally, we show that collaborative and efficient training of adaptors is possible and performs better. We also show that FLoRAL can outperform an ensemble of full models with optimal cluster assignment, which demonstrates the benefits of federated personalization and the robustness of FLoRAL to overfitting. We show promising experimental results on synthetic datasets, real-world federated multi-task problems such as MNIST, CIFAR-10, and CIFAR-100. We also provide a theoretical analysis of local SGD on a relaxed objective and discuss the effects of aggregation mismatch on convergence.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces FLoRAL, a parameter-efficient framework for better personalization. To do this, each client learns to mix between parameter-efficient adaptors according to their task and performs a clustering on top of that. The results show that FLoRAL can outperform an ensemble of full models with optimal cluster assignment. Theoretically, the authors provide a convergence analysis of their framework.

### Strengths
- Inspired by LoRA, aggregating partial parameters (adaptors) of federated learning models for better personalization is interesting. It is parameter-efficient and allows local models to benefit both from collaborations and the generalization of their data.
- The idea is well-motivated and presented clearly in the introduction. The results show that a mixture of adaptors sometimes can beat a mixture of models.
- The theoretical analysis is provided. Also, some insights into why aggregating only the adaptors could lead to a good performance are discussed. It is an interesting phenomenon from my perspective and can be explored further.

### Weaknesses
- **Related work is not comprehensively compared and discussed**: Firstly, there are many recent works in personalization for federated learning, such as [1,2] ; authors could discuss in the related work section and compare them in the experimental section. Secondly, in related work, authors can discuss how FLoRAL is different from LoRA-related federated learning methods (what are the strengths, differences, etc).

- **The experimental section is poorly presented**: The experiment part could be greatly improved. 

1. Baselines are not introduced in the main paper and I don't know what they are (what are local adaptors, ensemble?). 
2. Synthetic experiment is confusing. For Table 3, those compared baselines are not defined and I am not sure what messages authors trying to convey from this.
3. The experiments are conducted mostly on MNIST/CIFAR-10 and only simple architectures like CNN/MLP are used. More real-world datasets and more complicated model architectures can be used and tested.
4. Generally, the experiment part is not structured in writing. It is mixed with results, comparisons, conclusions, and ablation studies. Authors can consider presenting it in a more structured way.

- **Minor**: For line 153, what is this  ∆^{C−1}? Authors may define it more explicitly.

[1] https://arxiv.org/abs/2211.15281

[2] https://openreview.net/forum?id=nO5i1XdUS0

### Questions
- How is θ implemented in practice and what is it exactly? The authors mention it is a vector (Line 388). However, it remains unclear to me what exactly it is and how it relates to other parts of the parameters.
- What are the potential challenges and outcomes for applying FLoRAL to larger models such as language models?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces FLoRAL. It is claimed to be a memory-efficient federated learning framework focused on model personalization through the combination of LoRA adapters.

### Strengths
- This paper has a nice figure illustration.
- FLoRAL uses low-rank adaptation(LoRA) to personalize the model for each client. This significantly reduces the number of parameters that need to be stored and transmitted compared to using full models.
- Adapters in FLoRAL are learned collaboratively between clients, leveraging information from multiple data sources.

### Weaknesses
Although many efforts can be witnessed in this paper, we still find that the structure of this paper is hard to follow, and we can see a lack of explanation/motivation behind some techniques:
1) The authors implement the aggregated gradient every H step in the whole time zone T and use the modulo operator to describe what happens at some specific timestep. However, this would overcomplicate the problem rather than simply using ‘local and ‘global rounds as usual.
2) More discussion is needed on the communication and computation cost. For example, which parameters (i.e., vectors/matrices) are communicated between the clients and server? Which operations need to be computed by clients/servers? How is the communication cost (regarding the parameter size) compared with the computation cost?
3) Besides, experimental results show that the results are not convincing compared to SoTA baselines. For example, The work did not compare the proposed framework with another popular federated framework, i.e., FFA-LoRA (https://openreview.net/forum?id=NLPzL6HWNl). 
4) The benefits of LORA averaging in FL are not justified in theoretical analysis. For example, what are the effects of averaging the LORA adapters u and a (lines 11 and 12 of Algorithm 1) in the theoretical analysis?
5) Some notations are not clearly explained, i.e. (loss function of sampled client i_t at the weight of client k (w_k) in line 303).

### Questions
1) The aggregated gradient is implemented in every H step. However, the author did not mention how we can compute the loss function of i_t (the sampled client) over the client's weights (line 303). 
2) In line 316, what is the motivation behind the exponential formula for updating the learned client mixtures?
3) The authors claimed that this is a parameter-efficient framework. However, we see that the base layers are synchronized in the Algorithm 1 (line 419). So, what parameter efficiency could this framework bring? 
4) How are clusters assigned to clients?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a lightweight personalized federated learning method, FLoRAL, which integrates multiple shared adapters with client-specific vectors constrained by a simplex. 
The authors discuss the relationship between the proposed method and existing personalized federated learning approaches, talk about the probability selection strategy when aggregating weights, and present convergence properties of the proposed method as well as the impact of inaccurate estimation of the vector $\pi$.

### Strengths
I commend the authors for their clear analysis of the weight selection strategy for aggregation and their thorough examination of how uncertainties in estimating the vector $\pi$ impact convergence.

### Weaknesses
1. The experiments do not include all relevant baselines. Specifically, methods based on Mixture of Experts (MoE), such as [1], and shared Lora, such as [2, 3], are also closely related to the proposed approach.
2. The convergence analysis and the connection between FML and MFL problems are established under the assumption of convexity. However, the experiments involving neural networks generally do not meet this convexity assumption. A stronger alignment between the analysis and experiments would enhance the study’s validity.
3. The contributions of the work are not clearly articulated. While the authors claim that FLoraL is memory-efficient, similar advantages are also present in previous LoRA-based methods, such as [2, 3]. The authors should consider specifying the unique advantages that the proposed method offers over existing approaches.

[1] Yi, Liping, et al. pFedMoE: Data-Level Personalization with Mixture of Experts for Model-Heterogeneous Personalized Federated Learning. arXiv:2402.01350, arXiv, 11 Feb. 2024. arXiv.org, https://doi.org/10.48550/arXiv.2402.01350.

[2] Yi, Liping, et al. pFedLoRA: Model-Heterogeneous Personalized Federated Learning with LoRA Tuning. arXiv:2310.13283, arXiv, 11 Feb. 2024. arXiv.org, https://doi.org/10.48550/arXiv.2310.13283.

[3] Cho, Yae Jee, et al. "Heterogeneous lora for federated fine-tuning of on-device foundation models." International Workshop on Federated Learning in the Age of Foundation Models in Conjunction with NeurIPS 2023. 2023.

### Questions
1. In Theorem 4.5, the authors demonstrate that the convergence rate is influenced by the term $\mathcal{O}\left(\frac{G\Vert \delta_{c,0}\Vert_{1}^2}{\mu T^{1+\beta}}\right)$, where $\delta_{c,0}$ represents the degree of mismatch. Does this imply that, even if the predicted distribution differs significantly from the ground truth, it can still converge to the optimal solution. This seems counterintuitive; could the authors provide some intuition behind this result?
2. There are several typographical errors, and some notations are unclear. For example, in Eq. (2), $\pi_{k'c}$ should be written as $\pi_c^{k'}$.
3. In line 467, the authors state, “we do not aim to improve over the state of the art.” However, the experimental section does not clearly demonstrate the effectiveness of the proposed method. If relevant results are provided in the appendix, I would recommend placing the most significant results that highlight the advantages of the proposed method within the main text.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
The paper proposes "FLoRAL," a parameter-efficient federated learning framework that uses mixtures of low-rank adaptors (LoRAs) to enable [perhaps] client personalization in heterogeneous settings.

### Strengths
The strength of this paper is its exploration of parameter-efficient personalization in federated learning which is an important topic.

### Weaknesses
See my comments below:
* The paper lacks a clear presentation of the exact problem it aims to solve. In multiple sections—such as the abstract (lines 11-28) and the introduction (lines 51-57, 72-77)—the objectives and approach remain ambiguous. The methodology section does not clearly delineate the specific FL problem at hand or how FLoRAL is a unique solution to this problem. Overall, the paper needs a more clear writing. 
* Section 3 introduces five different FL setups, but it is unclear which of these the authors ultimately focus on. Are the authors targeting personalized FL, clustered FL, multi-task learning FL, or another variant? Furthermore, the notations "W" and "L" on line 151 are undefined, adding to the confusion.
* The related work section is weak and misses recent studies on personalized FL, multi-task FL, and representation learning. A thorough review of these fields is essential to position FLoRAL relative to state-of-the-art methods. It would help to articulate the novel contributions of FLoRAL more clearly in relation to existing methods, especially highlighting areas where it advances beyond current FL approaches.
* While the methodology is not clear, it appears overly simplistic, consisting mainly of mixing adaptors through weighted combinations. This is a straightforward extension of FedEM, as noted in line 378. Key methodological details are also missing—such as the initialization process, whether only the adaptors are learned and communicated, and whether the mixing of adaptors occurs at the client or the server. Providing a more detailed breakdown of the algorithm would clarify the novelty and rigor of the approach.
* The experimental section lacks critical information regarding the experimental setup and baseline comparisons. The baselines in Table 1 are not clearly explained; for instance, "Ensemble" is introduced without adequate description, and it is unclear how various settings were configured. The purpose of the "reduced 5% data" setting is also not explained. Furthermore, if the authors intend to address multi-task FL, a comparison with state-of-the-art MTL FL methods would be essential to substantiate the claims of FLoRAL’s effectiveness. In overall, the methodology section is not clear at all. 
* The conclusion section does not summarize the key findings of the study but rather discusses potential future work.

### Questions
See my comments above.

### Soundness
1

### Presentation
1

### Contribution
1
