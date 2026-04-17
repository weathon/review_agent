# FABLE: Federated Anchor-Based Learning with Privacy Protection

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Federated learning enables collaborative model training across distributed clients while preserving their data privacy. However, privacy leakage and data heterogeneity remain significant challenges in federated learning. On the one hand, privacy leakage arises when the exposed information about client models during the client-server communication is exploited to reconstruct sensitive data or misuse client models, compromising both data and model privacy. On the other hand, data heterogeneity limits the generalization capability of the global model on clients, leading to suboptimal performance. Current approaches face a dilemma that stringent privacy constraints degrade the model performance or incur substantial training overhead, while methods addressing data heterogeneity struggle to provide strong privacy guarantees. In this work, to alleviate this dilemma, we propose a novel and simple personalized federated learning method called Federated Anchor-Based LEarning (FABLE), which introduces private anchors during local training. Specifically, clients select private anchors from local datasets to perform an anchor-aware representation transformation, improving the adaptation of the model to local tasks. More importantly, those private anchors not only provide dual privacy protection of data and model privacy, but also avoid significantly computational/communicational overhead or performance sacrifice. Extensive experiments on benchmark datasets under various settings validate the effectiveness of the FABLE method in terms of the privacy protection and model performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an algorithm to protect privacy and handle heterogeneous data in federated learning. It proposes a globally shared encoder (trained using FedAvg) consisting of the earlier layers of the encoder. Each client maintains a private decoder with a classifier head. The novelty in the paper is that each client transforms the embedding vectors from the global encoder using a random projection and uses them as input to the decoder. As the projection matrix is private and random, and the decoder is private, this preserves privacy.

### Strengths
1. The problems of privacy and data heterogeneity are important in Federated Learning.
2. The paper is overall well-written and easy to read.

### Weaknesses
Weakness:
1. Why the anchor-based strategy would work is unclear. While projecting to a random coordinate system preserves pairwise distances approximately (JL lemma), it only happens when multiple projections are considered using multiple random coordinate systems. However, here only one system is used for each client. Can the authors show the insight using a simple setting so that the readers can have confidence in the method?
2. As the public encoder layers are situated near the input, their parameters contain information about the input data. So, they will expose some sort of private data.
3. How do you choose the anchor points? The results in Table 7 are confusing. The anchors seem to be fixed for the whole duration of the training; otherwise, the decoder training won't converge. So, in that case, what is the meaning of k-means clustering and fps, on which point cloud are these done?
4. It is unclear what the meaning of the linear transformation (Eq 4). I'm unclear why it is needed? One can use the dot product without normalization as the transformation. How is W_k initialized in practice? Does the finally learned W_k actually recover the dot products?
5. What the non-IID setting is unclear.
6. The use of anchors as secret keys is an interesting aspect, which, unfortunately, is too brief. The authors may want to build on this with formal guarantees to make the privacy argument solid.
7. Which component of the algorithm addresses non-iid data is unclear. If the data is too skewed, layers near the input will have different parameter values across clients (think clients have disjoint classes). Then the simple average of the parameters is known not to be a good proxy, so why does the simple averaging of the earlier layers work in non-iid setting is unclear.
8. Couldn't quite understand the difference between the data and model privacy in this context. Though there is a description in the paper, it'll be good to clarify it further.

### Questions
As above.

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
3

### Summary
This paper proposes FABLE (Federated Anchor-Based Learning), a personalized federated learning (PFL) method that aims to jointly address privacy leakage and data heterogeneity in federated learning (FL). Each client selects a small set of private anchors from its local dataset and uses them to perform a client-specific representation transformation before model training. These anchors are never shared, which is claimed to provide dual privacy protection—preventing data reconstruction attacks (e.g., DLG) and protecting model intellectual property (via “anchor-dependent” inference). FABLE is evaluated on CIFAR-10, CIFAR-100, AG News, and Sogou News under IID and non-IID settings, showing performance competitive with or slightly better than several FL and PFL baselines. Experiments also include analyses of privacy (PSNR, MIA accuracy), ablations on anchor numbers, selection strategies, and linear transformation variants.

### Strengths
- Privacy evaluation includes both data-level (DLG) and model-level (MIA) metrics, which is appreciated.
- The additional linear transformation is computationally lightweight and improves model stability.

### Weaknesses
- The “private anchor” idea is positioned as novel, but the technical distinction (anchors not being shared) is incremental rather than conceptually transformative.
- The privacy claims (especially model privacy via anchors as “secret keys”) are largely qualitative. The analysis relies on intuition (e.g., Equations (6)-(10)) without rigorous quantification or formal privacy guarantees (e.g., DP bounds, mutual information reduction).
- No convergence or generalization analysis is provided for FABLE, despite this being standard in FL papers. It remains unclear how the transformation affects optimization stability or gradient variance in theory.
- The privacy–utility tradeoff is not clearly quantified. The performance improvements over strong PFL baselines (e.g., FedALA, GPFL, FedDBE) are modest.

### Questions
1. Beyond not sharing anchors, does FABLE introduce any new theoretical insight or architectural mechanism that could be considered conceptually distinct?
2. The privacy analysis (Eqs. 6-10) provides an intuitive description of how anchors obscure gradients. Can the authors provide a formal privacy guarantee (e.g., differential privacy bounds, mutual-information analysis, or formal proof of obfuscation strength)? How sensitive is the proposed privacy protection to the number or distribution of anchors? For example, does a small anchor set materially weaken protection?
3. The privacy–utility trade-off is qualitatively described but not quantitatively analyzed. Could the authors provide a more explicit evaluation, e.g., performance versus privacy level (PSNR or MIA accuracy) curves?
4. The improvements over strong PFL baselines (FedALA, GPFL, FedDBE) appear modest. Could the authors clarify whether these gains are statistically significant? How would FABLE perform on more complex models or real-world FL datasets to demonstrate scalability?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes FABLE (Federated Anchor-Based Learning), a personalized FL method that introduces \emph{private, client-local anchors} to address both privacy leakage from shared encoder gradients and performance degradation under non-IID data. Each client $k$ keeps an anchor set $A_k \subset D_k$, embeds both its data and anchors with the shared/global encoder $g_k(\cdot)$, and transforms an input $x$ into an \emph{anchor-aware} representation
$$
r_k(x) = T_k(g_k(x); A_k) = 
\big( \cos\big(g_k(x), g_k(a_1^k)\big), \dots, \cos\big(g_k(x), g_k(a_{|A_k|}^k)\big) \big),
$$
optionally followed by a client-specific linear layer
$\hat r_k(x) = W_k \, r_k(x)$.
The transformed feature is then fed to a private decoder. Since the server only sees gradients that are \emph{already} passed through the unknown, private anchor transform, the paper argues that DLG-style gradient inversion becomes ill-posed. Experiments on CIFAR-10/100 and text datasets under IID and Dirichlet($0.1$) non-IID splits show that FABLE (especially with the linear layer) matches or outperforms strong PFL baselines while offering better resistance to reconstruction attacks.

### Strengths
Originality: The core idea is to make the client work in its own, anchor-conditioned coordinate system that is never shared. Unlike approaches that distribute public/synthetic anchors, FABLE keeps $A_k$ private and lets $A_k$ directly modulate what the server ever sees. This neatly ties together personalization and privacy.

Quality: The method is concretely specified: split model into shared encoder and private decoder; form anchor similarities via cosine, freeze anchor embeddings for stability, and add a local linear map $W_k \in \mathbb{R}^{d \times d}$ to restore expressiveness. The privacy discussion explicitly rewrites gradient leakage as inverting a composite mapping $F_k(\cdot; A_k)$ rather than the plain encoder.

Clarity: The dataflow in Fig.1 corresponds directly to Eqs.(2)-(4): global encoder $\to$ anchor similarity $\to$ (optional) linear transform $\to$ private head. Ablations on the number of anchors (256, 512, 1024), on the selection strategy (random, k-means, FPS), and on the linear layer make the contribution easy to verify.

Significance: The result that removing anchors at inference time makes the model basically unusable (accuracy drops to near-random) is practically meaningful: the model is effectively bound to the client’s private anchors, providing a concrete form of model-level privacy in FL deployments.

### Weaknesses
- Privacy claim is informal. The paper argues that the server now observes a gradient of the form $
  \nabla_{\theta_g} \mathcal{L}\big( F_k(x; A_k), y \big)$, where $F_k(\cdot; A_k)$ is the anchor-induced transform, and that an attacker cannot run DLG because $A_k$ is unknown. However, there is no quantitative bound, and no experiment where the attacker jointly optimizes over a guessed anchor set. This makes the privacy result more suggestive than proved.

- Anchor secrecy is a single point of failure. The “dual privacy” claim assumes the attacker can get model parameters but not the anchors. In realistic client compromise, $A_k$ is just local data and can be exfiltrated, in which case the protection collapses. The paper should state this assumption clearly.

- High-dimensional anchor spaces. With $|A_k| = 512$, the representation $r_k(x) \in \mathbb{R}^{512}$ and the client learns a $512 \times 512$ matrix $W_k$. This is fine for CIFAR and 20 clients, but the paper does not profile the compute/comm overhead for larger models, more clients, or bigger anchor sets, despite claiming “no significant overhead.”

- Comparison to other anchor-style FL is thin. The main difference from existing anchor/synthetic-prototype FL is that FABLE does not share its anchors. A head-to-head with a baseline that uses the same transform but with public anchors would isolate how much of the gain is from secrecy vs.\ just better alignment.

- Gains over strong PFL baselines are modest. On the hardest non-IID vision setting, FABLE w/ linear is best but only by a couple of percentage points over FedALA / FedPer. This slightly weakens the “one method fixes personalization and privacy” message.

### Questions
1- Joint inversion attack. Your privacy discussion effectively assumes an attacker solves $\min_{x', y'} \left\| \nabla_{\theta_g} \mathcal{L}\big( F_k(x'; A_k), y' \big) - G_{\text{obs}} \right\|^2$ without knowing $A_k$. Did you try the more realistic attack $\min_{x', y', A_k'} \left\| \nabla_{\theta_g} \mathcal{L}\big( F_k(x'; A_k'), y' \big) - G_{\text{obs}} \right\|^2$, i.e. treating $A_k$ as latent variables? This would directly test how much privacy comes from anchor secrecy.

2- Anchor representation. Are anchors stored and used as \emph{raw} samples $a_i^k \in D_k$, or as encoded features $g_k(a_i^k)$ cached once? If raw samples are used, do you rotate/refresh them to prevent slow leakage?

3- Dimensional mismatch. You set $|A_k| = d = 512$. What happens if the encoder output dimension $d_{\text{enc}} \neq |A_k|$? Do you use a rectangular $W_k \in \mathbb{R}^{d_{\text{out}} \times |A_k|}$, and does the privacy effect survive this mismatch?

4- Failure without anchors. In Fig. 3, accuracy collapses when anchors are removed. Is this because $r_k(x)$ becomes almost constant (low cosine similarities to non-existent anchors), or because $W_k$ was trained on a specific anchor distribution that is now missing? Showing the empirical distribution of $r_k(x)$ with/without anchors would clarify.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FABLE , a personalized federated learning method targeting the dilemma of privacy leakage and data heterogeneity. FABLE uses fixed, client-private “anchors” selected from each client’s local data, enabling a personalized, anchor-aware representation transformation during local training without revealing anchor data to the server or other clients.  Extensive quantitative and qualitative experiments, including ablations, overhead analysis, and privacy evaluations, are presented to substantiate the proposed approach.

### Strengths
Theoretical and empirical privacy analyses are presented.

### Weaknesses
1. The formulation of anchor-based transformations. e.g., Equations for $\mathcal{T}_k$ and $W_k$  lacks precise specifications around some key elements, such as how anchors are initialized. 
2. While the main privacy argument relies on anchors being private, the method for anchor selection (random from within the local dataset) potentially leaks information about the “support” of client data distribution, if an adversary explicitly designs attacks exploiting such selection schemes. There is minimal discussion about worst-case attacks where anchor selection or replacement is observable (e.g., side-channel or partial compromise).
3. The ablation study points out that the additional Linear Transformation layer is crucial for FABLE's performance and stability. However, when the same linear transformation is applied to baseline methods (like FedAvg and FedPer), their performance decreases in the IID setting. The authors attribute this to "distortion introduced by the additional transformation layer." which lacks deep analysis. The paper fails to provide insight into why FABLE's anchor-aware representation space is uniquely able to absorb and benefit from the linear transformation, while the representation spaces of other methods are not. 
4. The reported metrics also induce some confusions while different methods report different metrics. Is there any intuition introduction to this section? Why the 1024 anchors perform best in model performance but 128 best in model inference accuracy. There is also no the meaning of model inference accuracy.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2
