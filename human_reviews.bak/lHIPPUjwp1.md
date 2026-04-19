# Distributed Unlearning with Lossy Compression

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 3, 3

## Abstract
Machine unlearning enables to remove the contribution of a set of data points from a trained model. In a distributed setting, where a server orchestrates training using data available at a set of remote users, unlearning is essential to cope with the possible presence of malicious users. Existing distributed unlearning algorithms require the server to store all model updates observed in training, leading to immense storage overhead for preserving the ability to unlearn. In this work we study lossy compression schemes for facilitating distributed server-side unlearning with limited memory footprint. We identify suitable lossy compression mechanisms based on random lattice coding and sparsification. For a family of stochastic compression schemes encompassing probabilistic and subtractive dithered quantization, we derive an upper bound on the difference between the desired model that is trained from scratch and the model unlearned from lossy compressed stored updates. Our bound outperforms the state-of-the-art known bounds for non-compressed decentralized server-side unlearning, even when lossy compression is incorporated. We further provide a numerical study, shows that suited lossy compression can enable distributed unlearning with notably reduced memory footprint at the server while preserving the utility of the unlearned model.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper addresses lossy compression of update history in a distributed unlearning setup. The authors propose using lattice quantization and provide a theoretical bound on the differences between weights.

### Strengths
The paper is generally well-structured and accessible, aside from some issues with the citation style.

### Weaknesses
1. The reviewer is concerned about the paper’s limited novelty. Applying lattice quantization to the distributed unlearning problem appears to be a relatively straightforward adaptation of existing methods. Although the authors present a theoretical analysis (Theorem 4.3), it seems fairly direct. This is because the bound between the target weight $w$ and the compressed unlearned weight $w''$ can be expressed simply as $ \|w - w'\| + \|w' - w''\| $, where $ w'$  represents the uncompressed unlearned weight. This approach lacks significant innovation in its methodology since each term can be bounded by previous works.

2. Please revise the citation style to incorporate \citep or \citet where appropriate, and use ` ' for quotations.

3. For equation, it would be better to use \eqref

4. In equation 2 and 3, usage of $t$ seems confusing. My understanding is $t$ is for all iteration in equation 2, while $t$ represents the last iteration in equation 3.

5. Is there any analysis on sparsification?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper addresses the problem of unlearning in a federated setting using compression techniques. Specifically, at a certain point during training, a client may wish to withdraw from the process and have its contributions removed. A brute-force solution would be to restart training from scratch without this client, but this is impractical. Another approach would be to save the information sent by each client at each iteration on the server so that, if a client opts out, the server can recompute the updated model using the saved information (in this case, locally model updates), leading to an approximate solution. However, saving model updates for each client at each iteration requires substantial memory, which becomes impractical with large models, many clients, or numerous iterations. To address this issue, the authors propose using different compression techniques that reduce the serve memory footprint. The paper provides theoretical guarantees for these approaches, with practical experiments on CIFAR-10 and MNIST to validate the theoretical findings.

### Strengths
- The paper is easy to read.
- Provides theoretical guarantees for using compression techniques in federated unlearning.
- Includes numerical experiments on MNIST and CIFAR-10 on IID and Non-IID data distribution demonstrating the method’s practicality, showing improved accuracies on the main task compared to non-compression unlearning, while still maintaining reasonable performance against backdoor attacks.

### Weaknesses
- Evaluation is conducted on a 2-layer model for CIFAR-10. Given that compression is used, wouldn’t it be more practical now to use a deeper CNN for unlearning?

- The practical evaluation focuses on backdoor attacks, which does not address the use case where a client simply wants to withdraw from training (without having performed backdoor attacks) and ensure that its information is no longer used. Have the authors considered an alternative metric to assess the effect of compression on the ability to detect whether a removed client contributed to the training?

### Questions
From Theorems 4.3 and 4.4, it appears that G(t) can be large, as it depends on the norm of the gradients. Do you have an idea of how large it tends to be in practice?

How practical would this approach be when using a model like ResNet-18 on CIFAR-10? What would the server's memory footprint be in that scenario?

From Table 2, it seems that compression results in better accuracy on the main task compared to non-compressed federated unlearning. Do you have an explanation or insights as to why this might be the case?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper studies federated unlearning and adopts the method in [1]. As the approach in [1] (stated in (3) in this paper) requires the server to know all the historical updates from all the clients, it adds big storage cost. The paper proposes some compression techniques that guarantees unbiasedness and bounded error in the unlearning setup, in particular. 


[1] Thanh Trung Huynh, Trong Bang Nguyen, Phi Le Nguyen, Thanh Tam Nguyen, Matthias Weidlich, Quoc Viet Hung Nguyen, and Karl Aberer. Fast-fedul: A training-free federated unlearning with provable skew resilience. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 55–72. Springer, 2024.

### Strengths
The theoretical derivations looked correct to me.

### Weaknesses
- While the proposed approach and theoretical derivations look okay to me when optimizing for the approach in (3), I am not convinced that (3) is a reasonable unlearning technique in federated learning. I don't think server storing all the historical updates is a practically feasible solution. Instead, if the clients directly send their data to the server (i.e. non-federated training), it could have yielded lower storage cost in many cases. Since the paper does not concern the federated learning systems with privacy mechanisms (which would make unlearning unnecessary), the main advantage of federated learning here is the decentralization. However, keeping all the updates at the server all the time kills this advantage. I know that this paper is trying to reduce this storage cost but I still can't imagine this unlearning approach (in (3)) having a practical use. 

- The writing could be improved. A few formatting/grammar issues:

     - From the abstract "We further provide a numerical study, shows that..."
    - Please use inline references in parentheses when they are not part of the sentence. For example, the reference in "While data is often abundantly available in the ’big data’ era Jordan & Mitchell (2015), the source of the data might" is not part of the sentence and should be enclosed in parentheses. 
    - Line 330: "fist" --> "first"

### Questions
If the authors have any justification for storing the updates at the server that I cannot think of, I am happy to discuss further and revisit my evaluation.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper reduces the storage requirements for federated unlearning using compression. For $U$ users, during FL, if the update from client $u\in [U]$ at each round $t$ is $h_t^{u}$, then the centralized model is updated as $w_{t+1} = w_t + \frac{1}{U}\sum_{u\in [U]} h_t^{u}$. Here, if $w_t^{u}$ is the model learned by client $u\in [T]$ at round $t$, $h_t^{u} = w_t^{u} - w_t$.

The goal of federated unlearning (FU) in the **train-from-scratch** (Liu 2023) model for a specific client $\tilde{u}\in [U]$ is to
remove all contributions of this client to the model. This would imply the following model $w_t^\star$ at every round $t$, where 
$$
\begin{align*}
w_{t+1}^\star = w_t^\star + \frac{1}{U-1}\sum_{u\in [U], \tilde{u}\neq u} h_{t}^{\star, u},\quad h^{\star, u} = w_t^{\star, u} - w_t^\star \quad w_0^\star = w_0
\end{align*}
$$

Here, $w_t^{*, u}$ is the model update at client $u$ at round $t$ starting from $w_t^\star$. The goal is to modify $w_t$ to obtain $w_t'$ such that $\|w_t^\star - w_t'\|$ is small. One existing approach (Huynh et al 2024), uses skewed updates, 
$$
\begin{align}
w_{t+1}' = w_{t+1} + \sum_{j=0}^t (1 + \alpha)^{t-j} \delta_j',\quad \delta_j' = \frac{1}{U(U-1)}\sum_{u\in [U], u \neq \tilde{u}} h_j^{u} - h_j^{\tilde{u}}
\end{align}
$$
Note that this approach, and several existing approaches for FU require storing the complete history of updates for all clients. This corresponds to $T\cdot U \cdot m \cdot b$ bits of storage, where $m$ is the number of parameters in model and $b$ is the number of bits per model parameter.

This paper proposes using dithered quantization, DQ and its unbiased variant SDQ, and sparsification techniques, topk and randk, to compress each model update. Then, in Eq (1), it proposes to use $Q(h_j^u) - Q(h_j^{\tilde{u}})$, where $Q(x)$ is the reconstructed version of a vector $x\in \mathbb{R}^m$ using a given compression scheme.



For skewness $\alpha=0$, step size $\eta_t$, only $1$ local step, bounded loss gradients by $M$, and unbiased quantization with variance of compression error $\mathbb{E}[\|Q(x) - x\|^2] = \sigma^2,\forall x$, the authors show that by running Eq (1) with $\alpha=0$, the obtained model's distance from the unlearned model converges to $G + \frac{\sigma^2}{(U-1)}(\nu^{-1} + \pi^2/6)$ for step size $\eta_t = 1/(t+\nu)^\lambda$ with $\nu >0, \lambda >1$. (Lemma 4.1, Theorems 4.3 - 4.4, Corollary 4.1).

Further, in Section 5, the authors experiment their methods against the baselines of full **train-from-scratch**, Eq (1) with no compression,different compressors and vanilla FL on NN with CIFAR10 and MNIST datasets with a malicious client. By using unlearning to learn the contributions of this malicious client, the authors show that increasing the rate of compressor results in better protection against the malicious client (Figure 3, Table 2).

**References**
- (Liu et al 2023) A survey on federated unlearning: Challenges, methods, and future directions. ACM Computing Surveys.
- (Huynh et al 2024) Fast-fedul: A training-free federated unlearning with provable skew resilience. ECML PKDD.

### Strengths
- **Important problem in federated unlearning**: In federated settings, number of users $U$ can be extremely large, and the number of rounds for an iterative optimization algorithm, $T$, can also be very large. Storage at the server does become a huge bottleneck even when we need to unlearn the contributions of just a single client. 

- **Interesting experiments**: Unlearning the contributions of a malicious client is an interesting application of unlearning. Further, using the performance of model on the data of malicious client is a nice metric to study unlearning performance. Although this experiment has been proposed in (Huynh et al 2019), the authors use it to test performance of different compressors by varying the rates of these compressors.

**References**
- (Huynh et al 2024) Fast-fedul: A training-free federated unlearning with provable skew resilience. ECML PKDD.

### Weaknesses
- **Theory**:
    1. From AS1, number of local steps is $1$, which does not include FedAvg and from AS2, the function has bounded gradients. This makes the analysis easy, however, existing works in FL have relaxed these assumptions to much weaker ones(Stich and Karimireddy 2020). Further,the analysis local steps $>1$ for even federated unlearning has already been attempted by (Tao et al 2024) and it seems that it can be extended to this case.

    2. The unbiasedness and fixed variance of compression error in AS3 are extremely strong assumptions for a compressor and no proposed compressor satisfies these assumptions. Note that topk is a biased compressor, and randk and lattice quantization only satisfy these assumptions if the updates lie on a sphere or in the lattice support. Several weaker notions of compressors exist in literature (see Definitions 1,2,4 and 5 in (Beznosikov et al 2023)), which cover a large family of compressors including all the compressors stated in this paper.

    3. For (Huynh et al 2024), setting $\alpha=0$ (AS1) implies no skewness in updates. Skew estimation is an important contribution of (Huynh et al 2024) which necessitates $\alpha=0$. Can the authors provide problem settings, for instance loss functions, where using $\alpha=0$ is still valid?

    4. The choice of step size $\eta_t = \frac{1}{(t+\nu)^\lambda}, \lambda >1$ does not include the step size $\frac{1}{t + \nu}$. For bounded gradients, such a step size becomes essential to show convergence of final iterate of gradient descent( see Theorem 4 in (Koloskova et al 2019)), otherwise an averaged iterate is required (section 9.1 in (Garrigos &)). With $\lambda >1$, while the unlearning procedure might not diverge, but the actual FL algorithm might also not converge. Can the authors provide a reference for convergence of even GD with bounded gradients with the given step size?

- **Experiments**: In Figure 2, and Line 415, the SNR for a NN model is defined as $\log (Var(w_t')/Var(w_t' - w_t''))$. Note that NN weights can be permuted such that the weights differ but outputs of two networks remain the same. Therefore, using variance, via $\ell_2$ norm of model weights is not an appropriate metric. Further, can authors explain how this metric actually measures signal-to-noise ratio? If this metric has been used previously, they should cite it.



- **Comparison to Existing works**: (Jiang et al 2024) only stores model updates for clients and rounds which incur large updates. Therefore, without any compression, they reduce the storage dependence on $U$ and $T$. This is an important baseline both theoretically and for experiments to the proposed method. 


**References**
- (Tao et al 2024) Communication efficient and provable federated unlearning. VLDB.
- (Beznosikov et al 2023) On Biased Compression for Distributed Learning. JMLR.
- (Stich and Karimireddy 2020) The error-feedback framework: Better rates for sgd with delayed gradients and compressed communication. JMLR.
- (Huynh et al 2024) Fast-fedul: A training-free federated unlearning with provable skew resilience. ECML PKDD.
- (Koloskova et al 2019) Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication. ICML.
- (Garrigos & Gower 2023) Handbook of Convergence Theorems for (Stochastic) Gradient Methods. Arxiv.
- (Jiang et al 2024) Towards Efficient and Certified Recovery from
Poisoning Attacks in Federated Learning. Arxiv.

### Questions
- How does heterogeneity affect unlearning? The authors perform experiments for dirichlet class imbalance in Figure 5, but they provide no insights on it. Further, using real federated datasets 
- Ideally, for any Federated unlearning algorithm, the authors can use compressed model updates instead of actual model updates. Assuming these are unbiased compressors, with small MSE,  are there existing Federated unlearning algorithms, like FedRecover, Crab or Verifi,  where using these compressors would fail? For instance, where higher order information is used, or MSE does not suffice?
- **Best compressors for Distributed Mean Estimation**: The model averaging step in FL is a form of distributed mean estimation with mean of updates to be estimated. For this problem, (Suresh et al 2017) propose the optimal quantizers. While the problem for unlearning is not mean estimation, rather that of differences $(Q(h^{u}) - Q(h^{\tilde{u}}))$, the optimal DME compressors instead of randk and topk might serve as a good starting point. Can the authors check if these compressors are actually covered by dithered lattice quantization?

**References**-
- (Suresh et al 2017) Distributed Mean Estimation with Limited Communication. ICML

### Soundness
3

### Presentation
2

### Contribution
1
