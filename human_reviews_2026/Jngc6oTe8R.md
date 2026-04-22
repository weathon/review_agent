# ULD-Net: Enabling Ultra-Low-Degree Fully Polynomial Networks for Homomorphically Encrypted Inference

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Fully polynomial neural networks—models whose computations comprise only additions and multiplications—are attractive for privacy-preserving inference under homomorphic encryption (HE). Yet most prior systems obtain such models by post-hoc replacement of nonlinearities with high-degree or cascaded polynomials, which inflates HE cost and makes training numerically fragile and hard to scale.

We introduce ULD-Net, a training methodology that enables ultra-low-degree (multiplicative depth $\leq 3$ for each operator) fully polynomial networks to be trained from scratch at ImageNet and transformer scale while maintaining high accuracy. The key is a polynomial-only normalization, PolyNorm, coupled with a principled choice of normalization axis that keeps activations in a well-conditioned range across deep stacks of polynomial layers. Together with a special set of polynomial-aware operator replacements, such as polynomial activation functions and linear attention, ULD-Net delivers stable optimization without resorting to high-degree approximations.

Experimental results demonstrate that ULD-Net enables stable training of low-degree fully polynomial networks on large-scale model architectures and datasets. Applying ULD-Net to ViT-Small and ViT-Base achieves 76.70\% and 75.20\% top-1 accuracy on ImageNet, respectively, which are comparable to the original models and represent the first fully polynomial models successfully scaled to the ViT/ImageNet level. Additionally, ULD-Net outperforms several state-of-the-art open-source fully and partially polynomial approaches across diverse model architectures and datasets in both accuracy and HE inference latency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a training framework for DNNs composed entirely of low-degree polynomial operations to make them natively suitable for privacy-preserving inference. The authors introduce PolyNorm that stabilizes training by maintaining activation ranges within well-conditioned bounds, along with a principled choice of normalization axes and auxiliary loss penalties for numerical control. ULD-Net achieves SOTA accuracy–latency trade-offs on encrypted inference, surpassing prior fully polynomial and partial polynomial methods with up to 3× faster inference and +3.3% accuracy gains, and scaling to large architectures such as ResNets and Vision Transformers.

### Strengths
- The work Introduces a polynomial-only normalization mechanism (PolyNorm) that effectively stabilizes ultra-low-degree polynomial networks, addressing the long-standing issue of training instability in fully polynomial architectures at ImageNet and transformer scale.
- The method demonstrates strong empirical performance and scalability than prior state-of-the-art polynomial or partially polynomial methods.

### Weaknesses
- Very limited novelty. The authors failed to acknowledge or compare against highly relevant and more recent prior work [1, 2, 3, 4], all of which already introduced polynomial-only architectures or co-designed low-degree activation functions for PI. These prior works have also focused on low polynomial approaches, hence the ULD-Net's "novel idea" is an incremental extensions.

- In addition to my main concern about the novelty, the proposed PolyNorm and normalization-axis heuristic are engineering refinements rather than fundamental algorithmic advances.

- The adaptive numerical constraint and normalization principles are supported by descriptive reasoning but lack formal analysis or proofs of convergence and stability guarantees. The derivations are heuristic, and there is no quantitative ablation linking PolyNorm’s theoretical formulation to observed performance gains.

- The experiments only compare against two baselines but omit many more recent and "potentially" stronger baselines [1, 2, 3, 4].

- The authors admit that the polynomial activations tend to overfit and that accuracy degrades with depth (e.g., VanillaNet-7 drops 1.6% below the original) (PolyKervNets [3] and Sisyphus [5] mention this), yet they do not propose or analyze mitigation strategies beyond vague suggestions to enhance expressive power.

**References**

[1] Park J, Kim MJ, Jung W, Ahn JH. AESPA: Accuracy preserving low-degree polynomial activation for fast private inference. arXiv preprint arXiv:2201.06699. 2022 Jan 18.

[2] Diaa A, Fenaux L, Humphries T, Dietz M, Ebrahimianghazani F, Kacsmar B, Li X, Lukas N, Mahdavi RA, Oya S, Amjadian E. Fast and private inference of deep neural networks by co-designing activation functions. In33rd USENIX Security Symposium (USENIX Security 24) 2024 (pp. 2191-2208).

[3] Aremu T, Nandakumar K. Polykervnets: Activation-free neural networks for efficient private inference. In2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML) 2023 Feb 8 (pp. 593-604). IEEE.

[4] Ali RE, So J, Avestimehr AS. On polynomial approximations for privacy-preserving and verifiable relu networks. arXiv preprint arXiv:2011.05530. 2020 Nov 11.

[5] Garimella K, Jha NK, Reagen B. Sisyphus: A cautionary tale of using low-degree polynomial activations in privacy-preserving deep learning. arXiv preprint arXiv:2107.12342. 2021 Jul 26.

### Questions
- How does PolyNorm fundamentally differ from earlier polynomial stabilization methods or activation co-design approaches? What exactly does it do differently?

- Can the authors provide ablation results isolating the effect of PolyNorm and the normalization-axis principle to verify that these components are responsible for the reported stability and accuracy gains? Or can they provide a description as to what offers stability guarantees in their proposed approach.

-  Could the authors include comparisons with more recent baselines  to better contextualize ULD-Net’s performance and novelty?

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
3

### Summary
To enable privacy-preserving machine learning (PPML) in homomorphic encryption (HE) settings, fully polynomial model design has become a promising approach. Previous works  suffer from high degree polynomials and high inference latency. This paper proposes normalization axis principle and two approximations of nonlinear approximation, PolyNorm (polynomial-only normalization) and PolyAct, with low-degree polynomials to enable efficient HE inference. ULD-net uses multiplicative depth less than 3 for each operators.

### Strengths
- The authors propose the ultra low-degree training methodology replacing the nonlinear operations in HE. Authors provide replacement of non-linear operations from previous works (Roformer) and propose PolyNorm for the normalization layer, i.e., providing full end-to-end recipe to enable HE inference.

Su, Jianlin, *et al*. "Roformer: Enhanced transformer with rotary position embedding." *Neurocomputing* 568 (2024): 127063.

- Proposed (and provides) logical mathematical reason for choosing the normalization axis. Focusing on the reduction in variance of the input for polynomial function is a logical process, and a good approach.

### Weaknesses
- Although ULD-Net achieved 76.40% accuracy (1.58% drop) from the largest model tested by the authors, VanillaNet-7, batch normalization and layer normalization are standard techniques, therefore the approximation (PolyNorm) might incur a performance drop on more complex task or different model.
- Formulation of the inverse square root approximation depends on $\mu$ and $k$ and the choice of them are based on experiments. $\mu$ and Var varies when dataset differs — we cannot suggest a uniform optimal choice for these statistical values.

### Questions
- Authors introduce approximate inverse square root function well near $\mu$, but it does not approximate well on other ranges. Why did the authors propose a new approximation of inverse square root function that have two parameters $\mu$ and $k$ instead of using the well-known approximation methods of inverse square root function such as Newton-Rapson’s method? (e.g., Cho, W. *et. al.* used Newton’s method to approximate inverse square root)

Cho, W., Hanrot, G., Kim, T., Park, M., & Stehlé, D. (2024, December). Fast and accurate homomorphic softmax evaluation. In *Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security* (pp. 4391-4404).

- Determination of the best combination of parameters $(\mu,k)$ is based on experiment results without solid mathematical backbone. Will there be any mathematical reasons to determine the best parameter combination?

### Soundness
3

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
4

### Summary
This paper presents an interesting approach to building homomorphic encryption (HE)-friendly neural networks by introducing PolyNorm, a polynomial-only normalization technique that eliminates the non-polynomial square root operation from traditional normalization layers. The authors demonstrate solid results on small to medium-sized models, showing significant reductions in HE computational complexity (up to 2.76× speedup on ResNet-18(ImageNet)) compared to prior work, and notably extend the approach beyond CNNs to Vision Transformers, which is a positive step forward. However, the practical applicability remains questionable, as the evaluation is limited to relatively small models like ViT-Small on CIFAR and Tiny-ImageNet. At the same time, the paper lacks critical training-time measurements and any demonstration on large-scale, real-world language models—a significant gap given recent progress in applying HE to LLMs like GPT-2. More fundamentally, the core concept of designing task-specific and model-specific networks for HE is not new and dates back to early CryptoNet, which raises concerns about whether the proposed method can generalize beyond its evaluated scope, especially given that simpler polynomial-based models tend to show accuracy degradation with increased model depth (as evidenced by the 1.58% drop in VanillaNet-7). To strengthen the contribution, the authors should either demonstrate applicability to larger, more practical models, provide a detailed analysis of training overhead, or show how existing pre-trained models can be efficiently adapted to this polynomial-only framework rather than requiring complete retraining from scratch. While the technical execution is sound and the incremental improvements are noteworthy, a more transparent discussion of the method's generalizability and scalability limits would better position this work within the broader context of practical HE-based machine learning.

### Strengths
The paper presents a solution to a fundamental problem in HE-based neural networks—replacing the non-polynomial square root operation in normalization with a quadratic polynomial approximation (PolyNorm)—which elegantly enables training fully polynomial networks from scratch and achieves substantial HE inference speedups 2.76× to 20.5× (up to 20.5× reduction in the latency of non-polynomial operations on ViT-Small) while actually improving accuracy compared to prior post-hoc approximation methods. Beyond the technical innovation, the authors demonstrate that their approach scales beyond CNNs to Vision Transformers, breaking the traditional limitation of task- or model-specific HE designs and suggesting broader applicability across different architectures.

### Weaknesses
The evaluation is limited to relatively small models (ResNet-18, VanillaNet, ViT-Small on CIFAR/Tiny-ImageNet), with no evidence of scalability to larger practical models or LLMs where HE-based inference would be most valuable. Crucially, the paper provides no training time measurements or overhead analysis for ViT-Small, making it impossible to assess whether the polynomial-only design introduces significant computational burdens during training that could limit practical adoption. Furthermore, the approach still requires training models from scratch with polynomial operators rather than adapting existing pre-trained models. The already observable accuracy degradation in deeper networks (1.58% drop for VanillaNet-7) raises serious concerns about whether the method's performance will generalize to significantly larger architectures, where polynomial activation functions may struggle even more due to limited expressiveness.

### Questions
What are the total training times and computational costs for these models, and can the approach be scaled to train LLMs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The main bottleneck in performing homomorphic computation on modern AI models such as ResNet and ViT lies in the non-linear operations and non-linear normalization layers. These operations are known to play a crucial role in the overall performance of these models.
In this paper, we propose a method to replace all non-linear operations in these models with polynomial operations while maintaining comparable performance.

### Strengths
The non-polynomial parts of LayerNorm are replaced with a quadratic function and ensured numerical stability by fixing the normalization axis. Additionally, softmax attention and ReLU/GELU functions are replaced with low-degree polynomials, while still demonstrating strong performance on ResNet and ViT models — thereby validating the practical effectiveness of their approach.

Although there have been several prior attempts to approximate non-polynomial functions with low-degree polynomials, the introduction of normalization-axis fixing in LayerNorm to improve training stability stands out as a clear and meaningful contribution. One potential concern is whether this simplification might lead to a loss of expressiveness or overfitting issues, but the authors address this through penalty regularization terms and dropout, successfully mitigating such risks. Their experiments on the ImageNet dataset show accuracy comparable to existing methods, which effectively demonstrates that the model’s representational capacity is preserved.

This finding has significant implications for privacy-preserving machine learning (PPML): it suggests that by incorporating axis fixing, penalty loss terms, and dropout during training, one can approximate non-polynomial layers using only low-degree polynomial functions without severe performance degradation. Even though the paper does not explicitly explore this, the same methodology could likely be extended to other non-polynomial layers.

Finally, the paper clearly describes the conversion procedure for non-polynomial layers, which makes the proposed approach extensible and comparable for future applications in other architectures.

### Weaknesses
There was no comparison with state-of-the-art (SOTA) models for ViT in the paper. Among the current SOTA transformer models implemented under homomorphic encryption, two representative examples are THOR and PowerFormer. As far as I know, the NEXUS model is less efficient compared to these two models. Therefore, in Table 3, for the comparison of Softmax, LayerNorm, and GELU, the authors should include THOR and PowerFormer rather than limiting the comparison to NEXUS.

Moon, Jungho, et al. "THOR: Secure transformer inference with homomorphic encryption." CCS 2025.
Park, Dongjin, et al. "Powerformer: Efficient and High-Accuracy Privacy-Preserving Language Model with Homomorphic Encryption." ACL 2025.

### Questions
I would like to know how the convolution layer was implemented. There are two possible methods — the MPConv approach and the Neujeans approach. Since this paper does not use bootstrapping (as it is implemented using SEAL), I assume it employs the MPConv method. Is that correct?

If bootstrapping were to be applied, the Neujeans method could also be used. Do you think incorporating this method would further improve performance?

Also, for the ViT comparison, it would be good to include THOR and PowerFormer models in the evaluation.

### Soundness
3

### Presentation
3

### Contribution
3
