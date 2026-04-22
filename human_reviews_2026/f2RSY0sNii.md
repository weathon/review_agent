# FedANC: Adaptive Sparse Noise Scheduling for Federated Differential Privacy

- Avg Score: 1.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 2, 2, 2

## Abstract
Federated Learning (FL) enables multiple clients to collaboratively train a shared model without sharing raw data. Although this reduces direct exposure of local data, model updates can still leak sensitive information through gradient-based attacks. Differential Privacy (DP) mitigates this risk by adding calibrated noise to updates, providing formal guarantees. However, most existing DP-FL methods adopt fixed noise scales and uniform injection across all gradient dimensions, without adapting to client heterogeneity or training dynamics. This often results in poor privacy-utility trade-offs. To overcome these limitations, we propose FEDANC, an adaptive differential privacy framework for FL. It consists of three components: (i) an Adaptive Noise Controller (ANC) with an LSTM-based design that generates client-specific noise scales and sparsity ratios from local training feedback; (ii) a Selective Noise Injection mechanism that perturbs only the most sensitive gradient entries; and (iii) a Privacy Budget Regularization term that aligns per-round updates with a predefined privacy target. For stability, the ANC is pretrained with synthetic feedback that simulates typical training behavior. We provide theoretical guarantees on both convergence and differential privacy. Extensive experiments demonstrate that FEDANC achieves higher accuracy, faster convergence, and stronger privacy protection compared with existing approaches.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes a new method for federated learning with differential privacy. An LSTM controller adaptively adds calibrated noise to certain components of the gradients during client training in a federated averaging framework. The goal is to reduce the total noise by only adding it to the larger components, and adapting it to each user's data.

### Strengths
The method is original, and the writing is mostly clear, although some of the figures are too small.

### Weaknesses
The proposed FedANC framework is highly complex, requiring a pretrained LSTM controller on each client, a selective top-k noise mechanism, and a novel privacy regularization term. While Figure 6 shows utility gains, it's unclear if these benefits outweigh the significant implementation, pretraining, and (modest) computational overhead.

To support a claim of improved privacy-utility trade-off, I think it would be much stronger to compare model utility (e.g., test accuracy) while holding the total privacy budget ($\epsilon$, $\delta$) constant across all methods. The current experiments (e.g., Figure 6) compare utility against communication rounds, which is insufficient to demonstrate a superior trade-off.

Most concerningly, the privacy guarantee presented in Theorem 2 and Section 3.3 appears to be invalid. The "Selective Noise Injection" mechanism is data-dependent, as the choice of which top-k gradient components to perturb is a function of the private data. The paper's analysis only accounts for the privacy cost of noising the values of these components, while ignoring the information leaked by selecting them. This data-dependent selection process, which leaves other components un-noised, would seem to break the formal definition of differential privacy leading to a formal $\varepsilon$ of $\infty$.

### Questions
Consider this counterexample to the formal privacy guarantee claim. Take two neighboring datasets $D$ and $D'$ that differ in a single user $u$. Let all users in $D$ have a gradient of 0 for a specific component $j$. Let $D'$ be the same except let $u$ have a large-magnitude gradient for component $j$.
* On dataset $D$: Component $j$ will always be 0. It will never be in the top k and will never receive noise. The aggregated update for $j$ will be exactly 0.
* On dataset $D'$: The large gradient will place component $j$ in the top k, causing it to be perturbed with noise. The aggregated update for $j$ will be non-zero with high probability.

An observer can distinguish $D$ from $D'$ with near certainty by checking if component $j$ is zero, so there is no finite $\epsilon$ that satisfies the definition of differential privacy.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes FEDANC, a federated learning framework that incorporates adaptive differential privacy through three main components: (i) an LSTM-based Adaptive Noise Controller (ANC) that generates client-specific noise scales and sparsity ratios from local training feedback, (ii) a selective noise injection mechanism that perturbs only top-k gradient entries, and (iii) a privacy budget regularization term that aligns per-round updates with a predefined privacy target. The ANC is pretrained on synthetic data to ensure stability. The authors provide theoretical convergence and privacy guarantees, and evaluate the framework against gradient inversion attacks on CIFAR-10, FashionMNIST, and HARBox datasets.

### Strengths
Novel integration approach: The paper presents an interesting combination of adaptive privacy parameter generation, sparse gradient perturbation, and budget regularization within a unified framework.
Theoretical analysis: The authors provide formal convergence guarantees (Theorem 1) for both convex and non-convex objectives, as well as differential privacy guarantees (Theorem 2) using the Moments Accountant.
Comprehensive experimental evaluation: The paper evaluates against multiple gradient inversion attacks (DLG, IG, GI) across three datasets with different model architectures, demonstrating broad applicability.
Practical consideration of heterogeneity: The framework addresses client heterogeneity through personalized ANC instances, which is relevant for real-world federated learning deployments.

### Weaknesses
W1. The ANC takes a 4-dimensional input (|\mathbf{g}t|2, \ell_t, \beta{t-1}, \gamma{t-1}) and outputs 2 values (\beta_t, \gamma_t). The pretraining uses synthetic data where both inputs (Equation 6) and outputs (Equation 7) are sampled from independent uniform distributions with no inherent relationship. Training an LSTM to fit random noise to random targets lacks principled justification. Figure 3 shows minimal variation in output parameters, suggesting the controller may output near-constant values rather than performing meaningful adaptation. The paper does not provide a clear rationale for why this random pretraining would lead to effective adaptation during actual federated training.
W2. While the paper cites neural architecture search (Zoph & Le, 2017), meta-learning (Jiang et al., 2019), and adaptive DP (Li et al., 2022) to motivate pretraining (lines 193-198), none of these works employ or validate pretraining on independently generated random inputs and random targets. This gap significantly weakens the theoretical foundation of the proposed pretraining strategy.
W3. Based on Equations (9-10), \gamma_t and \beta_t are fundamentally coupled through the privacy constraint \hat{\epsilon}t = \sqrt{2\gamma_t d \ln(1.25/\delta)} / \beta_t. Given a target privacy budget \epsilon{\text{target}}, specifying one parameter determines the other. The paper does not acknowledge or address this coupling, making it unclear how the controller provides independent adaptation of two parameters that are mathematically constrained.
W4. The privacy regularization loss \mathcal{L}^{(t)}_{\text{privacy}} = (\hat{\epsilon}t - \epsilon{\text{target}})^2 can theoretically be set to zero by appropriately choosing \beta_t given \gamma_t (or vice versa), indicating only one degree of freedom exists rather than two. This redundancy is not discussed, raising questions about what the controller actually learns and whether both parameters are necessary.
W5. Table 1 and accompanying text do not report privacy budget (\epsilon, \delta) values for FEDANC or baseline DP methods. Without these values, the evaluation measures only empirical attack difficulty, not formal differential privacy guarantees under equivalent privacy constraints. 
W6. Table 1 reports only attack metrics (MSE, PSNR, SSIM) without including model test accuracy for each defense method. 
W7.  Algorithm 1 contains critical ambiguities in mask generation. Lines 8-15 iterate over multiple batches per epoch, each producing different sparse patterns through top-k selection. Line 17 states "Generate binary mask \mathbf{M}_k; send \mathbf{W}_k \odot \mathbf{M}_k to server" without specifying which batch's mask is used or how masks from multiple batches are aggregated. 
W8. Lines 456-457 mention "DPSA-FL strategy" which is never defined elsewhere in the paper and has no corresponding citation.

### Questions
Q1- The paper claims "low-magnitude gradients usually carry less sensitive information" (Section 3.2, lines 216-219) based on empirical attack studies. Can the authors provide formal theoretical justification or proofs demonstrating that low-magnitude gradient components inherently contain less sensitive information? Without theoretical grounding, this remains an empirical assumption that may not hold universally.

Q2- How many synthetic samples are used for pretraining the ANC module? 

Q3- The paper states that pretraining "guides the controller toward stable parameter regions" (lines 189-192). Given that inputs (Equation 6) and outputs (Equation 7) are independently sampled from uniform distributions with no functional relationship, what is the mechanism by which fitting random inputs to random targets produces stable and reliable parameters for actual federated learning?

Q4- What specific (\epsilon, \delta) values were used for FEDANC and baseline DP methods in Table 1? How were these budgets allocated across rounds, and were all methods compared under equivalent total privacy budgets?

Q5- Given the coupling between \gamma_t and \beta_t described in W3, how does the controller provide meaningful independent adaptation of these two parameters?

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
2

### Summary
This paper proposes FedANC, a differentially private federated learning (FL) algorithm. The proposed method first uses an LSTM to decide privacy parameters, then add privacy noise to the top-k entries of the gradient, and update a masked model update to the server.

The paper provides theoretical convergence and privacy guarantee for the proposed method and numerical results show than the proposed algorithm can defend against privacy attacks while achieving better model performance.

### Strengths
1. The numerical reuslts of the model shows that the proposed method can defend against different privacy attacks compared with non-DP algorithms.

2. The numerical comparision shows that the proposed method achieves better performance than other privacy preserving algorithms.

### Weaknesses
1. On the presentation level, the paper failed to provide a clear algorithm in the main paper. It is hard to follow the steps of the algorithm.

2. The paper failed to provide a solid privacy analysis to the algorithm. It is hard to understand why the privacy noise is only added to the top-k entries and still protects privacy. A more rigorous privacy analysis on the mechanism should be provided.

3. It is unclear which privacy level the paper is trying to achieve. In FL, there are different levels of privacy protection, including client/user-level, local and server level. The paper should provide a formal statement to the setting.

4. The ANC network uses the true gradient information to decide the privacy parameter, which may already leak privacy. The paper should provide further justification on how the ANC network involves in privacy protection.

5. The numerical comparision is incomplete. In the privacy defence, the paper failed to include DP based method.

### Questions
Please address the weakness above.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes an algorithm to using a learned controller to adaptively control noise added to the client-side gradient in FedAvg framework, with the target being improving algorithm performance under fixed privacy budget.

### Strengths
The paper covers convergence guarantee, privacy guarantee, and experiment.

### Weaknesses
1. The major weakness I feel is correctness of privacy guarantee. The paper use gradient dependent sparsity operator to get top-k largest coordinate of gradient, but this step is not privatized, only final top-k coordinates are privatized. This indicate the algorithm may not actually be private.
2. The paper does not have any pseudo code of algorithm. It is unclear whether the algorithm is updating based on sparse gradient or dense gradient. In figure 1 it seems sparse gradients are used implied by the "sparse avg aggerate" below server figure, but I don't see anything formally mentioned or introduced the algorithm steps in a clear manner.

### Questions
1. Is the algorithm private? I am concerned that the top-k operator is not privatized.
2. Do server update parameters using sparsified gradient?

### Soundness
1

### Presentation
3

### Contribution
1
