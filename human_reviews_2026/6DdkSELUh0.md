# We Generate What You Need: Efficient Data Supplement via Model Prediction Discrepancy for Heterogeneous Federated Learning

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
One emerging approach to mitigating data heterogeneity in Federated Learning (FL) is to employ diffusion models to generate synthetic data for clients, thereby aligning local data distributions with the global distribution. Prior work has primarily focused on balance-oriented augmentation, which assumes a balanced global class distribution and thus generates samples of rare classes to rebalance each client's local dataset. However, in practice, global data distributions are often inherently imbalanced. For example, in weather forecasting, certain regions naturally experience more rainy days than sunny days, resulting in inherently imbalanced global training and testing data for those regions. Moreover, privacy and communication constraints in FL hinder the server’s ability to accurately estimate the global distribution, rendering balance-oriented augmentation suboptimal. This raises a key, underexplored challenge: How can synthetic data be generated and selected to align local distributions with the true, yet unknown, global distribution? To address this challenge, we propose a novel framework, FedDPD. The key insight behind our approach is that a model’s performance implicitly reflects the data distribution it has been trained on. Based on this observation, we use the performance discrepancy between the local and global models to identify the regions where each client’s local dataset is lacking, and supplement corresponding synthetic samples for clients. Furthermore, we adapt the diffusion model for each client through a preference-optimization paradigm, enabling it to generate data that better aligns with the true global distribution, addressing the specific gaps in the client’s local data. Notably, our approach incurs no additional computational overhead for clients. Extensive experiments on multiple benchmarks demonstrate that FedDPD outperforms state-of-the-art methods, achieving up to 3.82\% improvement, regardless of the global distribution’s balance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses data heterogeneity in federated learning by proposing to generate synthetic data to align the client’s dataset with the global distribution. They argue that, in practice, global class distributions can be imbalanced, whereas prior work assumes a balanced distribution. Their method supplements clients with the required synthetic data by using the local model’s performance as an indicator of the data distribution it was trained on. They also use preference optimization to adapt the diffusion model to generate samples that are more likely to be useful for the client.

### Strengths
1)The problem that the paper attempts to mitigate is well motivated and clearly articulated.
2)The efficacy of the proposed method to align local and global distributions is well evaluated and presented in Section. 4.2 Case Study.
3)The usage of the model’s performance to reflect the data distribution it is trained on is intuitive and practical to use.

### Weaknesses
1) The Following Relevant works are not cited in the related work.
 a) Confusion-Resistant Federated Learning via Diffusion-Based Data Harmonization on Non-IID Data 
 b) pFedGPA: Diffusion-based Generative Parameter Aggregation for Personalized Federated Learning
2)The reward function in Eq. 8 needs further explanation as it seems to contradict the motivation for the reward mentioned in lines 300-302.
3)The experiments and results section could have been more clearly presented, for example: the metric compared against in tables 2 and 3 is not mentioned, and the details of generating global class imbalanced data and the degree of global class imbalance are not clearly mentioned.
4)Prior work (FedETF) has shown that for some methods, increasing the local epoch count can decrease the global performance. The decision to use five epochs to train local models for baselines vs one epoch for the proposed method is not well clarified.

### Questions
1) Do all the baselines used also use 5 epochs to train their local models? If not, please clarify the reasoning for using 5 epochs to train local models in baselines vs 1 epoch in the proposed method.    
2)What would be the compute overhead at the server compared with the baselines?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper seeks to mitigate the challenge of data heterogeneity in Federated Learning. The authors argue that earlier approaches try to balance each client's data distribution, which might not be the most suitable approach, as the global data distribution is most likely not balanced. To address this, the authors propose a new method (dubbed FedDPD) "that dynamically adjusts the data supplementation process according to the true global distribution, rather than assuming it to be balanced". Their aim is to provide each client model with data that fills the gap between the local model and the global (unknown) data distribution. Experiments on CIFAR-10, CIFAR-100, and Tiny ImageNet show that FedDPD outperforms state-of-the-art baselines, including other generative methods. It also demonstrates alignment with the true global distribution (Table 4).

### Strengths
The paper's primary strength is its challenge to the common "balance-oriented" augmentation assumption. It correctly identifies that real-world global distributions are often inherently imbalanced, posing a more realistic and important research question for the FL community. 

The FedDPD framework is comprised of three main components:

**Prediction Discrepancy Selection (PDS):** This is the core idea. The server selects a synthetic sample $(x,y)$ for a client $i$ if the client's local model performs poorly compared to the global model, i.e. $\mathcal{l}_i​(x)>\mathcal{l}_g​(x)$. This discrepancy indicates the client's local data is deficient in samples of this type compared to the global data distribution. This is motivated by Theorem 1. In other words, the performance of the client compared to the server model is a proxy for data deficiency in the client models. Additionally, they implement the criterion that the softmax probabilities of the label have to be better than random, $\tau = \frac{1}{C}$ (they call this "confidence aware filtering")

**Exponential Decay Scheduler (EXS):** The paper finds that the _amount_ of data supplied per round matters. After testing several strategies (ablation results in Table 1), it adopts an exponential decay scheduler, which supplies more data in the early rounds and less later on, achieving the best stability and performance.

**Discrepancy Preference Optimization (DPPO):** To further tailor generation, the framework fine-tunes a client-specific LoRA module for the diffusion model. It uses a DPO-based reward function that prefers samples with high local loss, low global loss, and disagreement between the local and global model predictions.

**Elegant and Theoretically-Grounded Mechanism:**
The core mechanism of using prediction discrepancy ($l_i > l_g$) as a proxy for distributional gaps is intuitive, creative, and well-motivated by the theoretical argument in Theorem 1. This approach cleverly avoids needing to know the true global distribution. 

**Strong Presentation and Clarity:**
The paper is very well-written. The problem and solution are explained clearly, and the figures (especially Figures 1, 2, and 3) are high-quality and effective at illustrating the core concepts (although Figures 2 and 3 are not referenced in the text itself).

### Weaknesses
**Reproducibility:**
It seems that code implementation is missing, which makes reproducing the results challenging.

**Misleading and Incomplete Efficiency Analysis**
The paper only comments on the overhead on the clients, but does not highlight the added overhead on the server. The only mention is that it takes around 2 seconds on an A100 GPU to generate an image, "demonstrating that our framework is computationally feasible for real-world applications". 

1. Server-side computation:
This server-side cost scales linearly with the number of clients (N), a critical metric in federated learning. In a system with a large number of clients, this method might not be realistic, even if the generation only takes 2 seconds on an A100 GPU
The server must:
    * **Run a diffusion model** to generate a large pool of synthetic images.

    *  **Calculate discrepancy scores** for generated samples. This requires performing a forward pass using _both_ the global model and _each client's_ local model. This core step scales at least linearly with the number of clients ($N$), becoming computationally expensive as N grows
    * **Train N separate LoRA modules** using Discrepancy Preference Optimization (DPO), one for each client. This finetuning cost also scales linearly with N
    * This combined workload, which requires substantial, specialized GPU resources, is not benchmarked in terms of wall-clock time or cost. The $O(N)$ scaling of both inference and finetuning suggests the framework's server-side demands would become a bottleneck in real-world FL scenarios with many clients. The authors should include information about this added cost in their manuscript.

1. Communication overhead:
Line 113-114: "no additional computational or communication overhead on the client side". That statement could be misleading. 
In addition to uploading models, as I understand it, clients must now _download_ the personalized supplemental datasets in each round. This download cost could become a practical issue, yet it is never measured or discussed, making the claim of "no additional computational overhead for clients" incomplete.

Equation 6: Additionally, if understood correctly, inference must be completed at both the client level and server level during training to obtain the losses and softmax probabilities used. This would also add communication overhead, along with added computational costs at the client. 
If so, I suppose this would merit a more rigorous analysis of the claims.

1. **Domain shift of diffusion model**
The paper fails to address the generality of its generator-based approach. The method's success on common benchmarks (CIFAR, ImageNet) may be an artifact of data overlap, where the generator (Stable Diffusion) has already been trained on massive datasets containing similar images. This raises a critical question: Would the method still work for niche, domain-specific datasets (e.g., medical, satellite, or industrial images) that are _not_ well-represented in the generator's pre-training data? The paper provides no evidence for this, limiting the perceived generality of its contribution.

1. **Top/Bottom half cutoff for preferred/rejected (L306-308)**:
The DPPO component's reliance on a hard 50/50 cutoff to distinguish "preferred" from "rejected" samples is a methodological weakness. This arbitrary heuristic risks creating a noisy training signal by forcing the model to distinguish between samples with nearly identical reward scores that fall on opposite sides of this median split. This binary labeling discards the nuance of the continuous reward scores and may lead to mislabeled preferences, as the true proportion of "high-quality" samples generated in any given round is not guaranteed to be 50%.

1. Textual improvements
    *  Eq. 7, $N_{total}$ and Line 263 $N_{totl}$ should be the same
    *  Figures 2 and 3 do not seem to be referenced in the text. 
    *  The text in Figures 4a and 4b is almost impossible to read in a printed version, because it is very small. 
    * The conclusion is very short and is more of a summary.

### Questions
1. Line 46: newest reference is from 2022; newer work exists that should be cited.

1. Sec 2.2: A similar method to yours seems to be "Synthetic data shuffling accelerates the convergence of federated learning under data heterogeneity", Li et al, TMLR 2024, is that correct? If yes, should it be included as a baseline?

1. Eq. 3: $q(\cdot | c)$ does not seem to be defined in the main paper. 

1. Line 199 (and further), $\hat{D}$ hat notation usually means estimator, so perhaps use a tilde instead?

1. Eq. 4: does not follow the definition of $\mathcal{l}$ in line 188. I think you meant to write: $\mathcal{l}(w_i; x_j,y_j) - \mathcal{l}(w_g; x_j,y_j)$ ?

1. For Tables 1-4, how many trials were run, and are the improvements significant after statistical tests? In Table 2, there is a clear overlap between your method's performance and baselines, which makes it unclear whether you actually have a significant improvement compared to the baseline. 

1. Hyperparameter stability: How sensitive is the method to the choice of exponential scheduler coefficient $\beta$, confidence threshold $\tau$, and LoRA/DPO-specific settings? Can the authors share tuning curves or robustness analyses?

1. Is there any internal or external validation (human rating, metrics, FID, IS, etc.) of the diffusion-generated samples per class and per client?

1. Eq. 8: Should the sign on $\hat{p}_i^t(x_j)$  and $\hat{p}_g^t(x_j)$ not be swapped? You state that the goal is to encourage samples that are difficult for the client but easy for the global model. As I understand it, this would imply that $\hat{p}_g^t(x_j)> \hat{p}_i^t(x_j)$. But right now, the term $\hat{p}_i^t(x_j) - \hat{p}_g^t(x_j)$ would result in a negative reward if that is the case. In other words, by maximizing $\hat{p}_i^t(x_j) - \hat{p}_g^t(x_j)$ it seems that the framework would be selecting samples where $\hat{p}_i^t(x_j)> \hat{p}_g^t(x_j)$.

1. Line 296 + Eq. 9: Could logits be used instead of the class label?

1. Line 308: $\mathcal{S}_i^t = (x^+, x^-,y)$ is defined as a single pair, but in equation 11, you sum over pairs, and the text correctly states that it is a set of pairs. I think e.g in Eq. 11, $\mathcal{S}'$ should be defined as the set of all pairs?

1. Line 311: $\pi_{x|y}$ is the likelihood of the reference model, not "the reference model"?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
To address data heterogeneity in Federated Learning, diffusion models can generate synthetic data to align local and global distributions. Existing methods assume a balanced global distribution, but real-world data is often naturally imbalanced. The authors propose FedDPD, which uses performance discrepancies between local and global models to identify distribution gaps and generates tailored synthetic data via optimized diffusion models. Without adding client-side computation, FedDPD outperforms state-of-the-art methods by up to 3.82% across varied global distributions.

### Strengths
The paper is well-structured and easy-to-follow. The proposed approach outperformed the listing benchmarking algorithms.

### Weaknesses
More recent and state-of-the-art sample generation methods should be taken in to consideration and comparison. I noticed there are two generative FL algorithm in Table 2 and 3. Nevertheless, more recent approaches should be involved.

### Questions
Please refer to weaknesses for my main concerns.

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
4

### Summary
This paper is about methods to improve performance of a global model trained through federated learning on heterogenous data. Note that in the heterogenous FL setting, there is an inherent tradeoff between fitting a model that is good on the global distribution (the "global" setting) and maximizing performance on a per-client basis (the "personalized" setting). This paper focuses on the global setting, where the goal is to introduce regularization that prevents client models from overfitting their local data and diverging from one another. Early research on this problem focused on explicit regularization (e.g., FedProx, gradient clipping, etc) but more recently people have been looking at implicit regularization (e.g., data augmentation).

In this paper, the authors propose a method, FedDPD, that targets the problem of class imbalance for federated multi-class classification tasks. The core idea is that we can identify class deficiencies in a client by examining the performance difference between the global and client models (I think there are some problems with this claim in general, but at least it seems to be true in practice).

The algorithm is to (1) collect client model updates as usual in FL, (2) generate synthetic samples on the server, (3) measure the performance difference between the global model and each client model for each sample, and (4) update the client models on the server (before aggregation) using the synthetic examples where there was a large difference in loss between the client and global models. The algorithm additionally updates the generative model using DPO, to improve the likelihood that it will generate samples where the local and global models have high discrepancy.

### Strengths
- The problem is important and has lots of practical applications.
- It is a nice idea to do the synthetic data generation once, on the FL aggregation server, rather than to force clients to do this locally.
- The experiments show some performance improvements from FedDPD, both on its own and also when combined with other methods, and compare against a good number of baselines.
- The synthetic sample generation is effectively a rejection sampling algorithm, where we sample from a generative model and then check that (1) the global model has high confidence and (2) the global and local models disagree. DPO is a very reasonable way to improve the sample efficiency of this process, by learning a model of the non-overlapping portions of the distributions.

### Weaknesses
- FedDPD is somewhat narrowly focused on multi-class federated classification tasks for data modalities where it is possible to construct a generative model (and specialize it through low-rank adaptation). Many important problems fall into this category, so this is not a huge weakness. Even still, the framework does not immediately generalize to other types of data heterogenity, such as feature drift, or other types of objective, such as generative modeling or regression (and this should probably be clarified in the paper).
- The label distribution correction behavior does not seem to be guaranteed. Figure 1a suggests that FedDPD augments the class distribution of each client to match the global distribution, but in reality we never actually see the global / local class distributions. Instead, we just see the losses of the client and global models on the synthetic dataset. Theorem 1 says that this performance difference will be negative (in expectation) if the client is deficient in some class. However, I am not sure about the converse - a poor loss from the client on an example where the global model succeeds does not necessarily indicate that the client is deficient in that class. For example, consider a situation where the client and global distributions have equal class proportions but have feature drift, or a situation where either the client or the global model have not converged. In these situations, FedDPD might make the class imbalance worse.
- Further, Theorem 1 seems to have some important assumptions that are not stated clearly in the text or satisfied in practice. If my understanding of the proof in A2 is correct, we require that feature drift does not occur (clients share $q(x|y)$), the client / global models are fully converged (so that the empirical cross-entropy risks are minimized). These assumptions seem unrealistic in the heterogenous FL setting, especially the ones that require convergence of the model, since heterogenity harms the convergence rate / stability.

Of course, the experiments show that these problems do not occur in practice (Figure 4). But the paper should probably discuss these caveats and make it clear that the theoretical development is to provide intuition, not guarantees.

There are also some minor issues / typos in writing
- The FedDPD acronym is not defined anywhere in the main text. 
- "We employs a text-to-image diffusion model on the server" on L198

### Questions
Did you conduct an ablation where the DPO finetuning was removed? I wonder whether DPO is effectively learning a model of the distribution difference between - especially since DPO is only run for the first few rounds of FL. If so, then the algorithm in this paper is actually very similar to the one in *"Federated Learning in Non-IID Settings Aided by Differentially Private Synthetic Data"* - just instead of finding clients with complementary data distributions to merge with, we construct a complementary data distribution on the server using DPO.

### Soundness
3

### Presentation
3

### Contribution
3
