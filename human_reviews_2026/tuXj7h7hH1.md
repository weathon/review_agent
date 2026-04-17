# A Constrained Optimization Perspective of Unrolled Transformers

- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
We introduce a constrained optimization framework for training transformers that behave like optimization descent algorithms. Specifically, we enforce layerwise descent constraints on the objective function and replace standard empirical risk minimization (ERM) with a primal-dual training scheme. This approach yields models whose intermediate representations decrease the loss monotonically in expectation across layers. We apply our method to both unrolled transformer architectures and conventional pretrained transformers on tasks of video denoising and text classification. Across these settings, we observe that constrained transformers achieve stronger robustness to perturbations and maintain higher out-of-distribution generalization, while preserving competitive in-distribution performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors consider the theory and practice of training deep neural networks to iteratively improve the value of a self-supervised loss function _at each layer_ (as opposed to end-to-end training which only considers the network's output at the last layer). They prescribe that the improvement should be at a geometric rate, which gives a constrained optimization problem for the network parameters. They propose and justify the use of primal-dual methods for optimizing this constrained problem by bounding the duality gap under some assumptions (Theorem 1). They further bound the performance gap between the network constructed by solving this constrained problem and the optimal network, under the same set of assumptions, on both an in-domain distribution from which the training set is drawn (Theorem 2) and an out-of-domain distribution (Theorem 4). They do some experiments using video denoising and text classification to attempt to demonstrate that the method is reasonable in practice.

### Strengths
- The setting and problem formulation is novel; as far as I know, explicitly end-to-end training neural networks for layer-wise behavior has not been done before. However, 
- The method is clean, well-motivated, and backed directly by the theory.
- The empirical results look promising: while the performance on the original task does not degrade (as one might expect by making the model make trade-offs by adding more terms to the objective), each layer indeed does optimize the objective incrementally, adding some basic understanding of the network and robustness/generalization capabilities.
- The theory is also good; the assumptions made are mostly realistic, i.e., the loss is assumed to be a convex function of the data and prediction, not of the network parameters, yet optimization results can still hold for the network parameters. The analysis is also clean and uses some clever techniques (i.e., working with supermartingales).
- Lastly, the paper is well-written and easy to read.

### Weaknesses
The only issue I have is that the experiments do not quite have the breadth that the generality of the framework demands. Namely, the proposed learning method seems like it would apply in general to all types of self-supervised learning, including diffusion, LLMs, etc., and the text embedding experiment demonstrates a way that it can work for supervised learning too. It would be super interesting to see how the method works on these very popular tasks; if the answer is "it works well" that would strengthen the paper's main claims, but if the answer is "it does not work well" that would still be very interesting and a deeper understanding of how it would break down would also strengthen the paper.

### Questions
Q1: See above "Weaknesses" section.

Q2: Does the theory hold straightforwardly for the case where there is side information such as labels? The text embedding experiment seems to suggest this, but I wonder if it is easy to extend the theory. Even better if it accommodates that the loss is arbitrarily dependent on the label (since it's not being fed through the network).

Q3: Have you tried probing the interpretability of these specially trained models? Say via auto-interpretability methods or just by hand. I think it would be really interesting to see how the features evolve differently in conventionally trained networks versus these specialized ones.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a framework for training transformers by imposing layerwise descent constraints on the expected loss across layers. Instead of standard empirical risk minimization, the authors formulate transformer training as a constrained optimization problem solved via a primal-dual approach. The key idea is to make each transformer layer behave like a step in a descent algorithm, thereby ensuring monotonic loss reduction through the depth of the network. The authors adopt theory and the objective from [1] to design such constrained Transformer training, and demonstrate empirical improvements on video denoising and text classification (IMDb, MNLI) tasks.

### Strengths
1. **Novely:** Integrating constrained training objective to transformers is an interesting touch with motivation from the success of traditional unrolled neural models.
2. **Theoretical foundation:** Although not a brand new contribution, the framework is based on a fairly well-established constrained learning framework of [1] and apply it to the Transformer architecture.
3. **Empirical support:** The effectiveness of the framework is supported with positive empirical results in video denoising and text classification together with OOD tests.

### Weaknesses
1. **Scope:** the constrained learning framework seems to be architecture agnostic in most ways. This would mean most theoretical as well as empirical results should ideally be true across any deep neural networks. I think this needs to be discussed in the main text.
2. **Applicability:** would OOD gains achieved with the method scale with model size? Since experiments are mostly small scale, this is not evident if just scaling the model size would overshadow the OOD benefits of the method.
3. **Ablation study:** the paper lacks ablation study on hyperparameters such as $\alpha$ and dual learning rate $\eta$.

### Questions
1. **Applicability:** As true for most new pretraining methods, it might be hard to train and test on a very big scale. However, with this method, I wonder if fine-tuning a pre-trained model could benefit too. Could authors validate if fine-tuning on a new task using constrained learning objective could, for example, enjoy boost in OOD performance or generalization? I believe this would make the method more appealing to wider group of readers.
2. How sensitive is the performance to the choice of abovementioned hyperparameters?

___

Overall, I think the paper is interesting and adds a valuable contribution to the ICLR community and, therefore, I am open to increase my score as long as both of my questions are addressed in a satisfactory manner.

___

### References
1. Luiz F. O. Chamon, Santiago Paternain, Miguel Calvo-Fullana, and Alejandro Ribeiro. Constrained Learning with Non-Convex Losses. IEEE Transactions on Information Theory, 2023

### Soundness
3

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
This paper proposes a constrained optimization framework for training transformers that behave like iterative descent algorithms. Instead of relying only on empirical risk minimization (ERM), the authors introduce layerwise descent constraints, ensuring that each layer of the transformer monotonically decreases the expected loss. Training is performed using a primal-dual algorithm, which enforces these constraints and yields transformers with improved robustness and generalization.

### Strengths
- The paper introduces a constrained optimization view of transformer training, in which each layer must monotonically reduce the expected loss—a property inspired by iterative optimization algorithms.
- It formalizes this idea rigorously using a primal–dual training framework, backed by proven results such as:
  - Convergence guarantees (Theorem 2)
  - Out-of-Distribution (OOD) generalization bounds (Theorem 4)
  - The inclusion of expressivity and sample complexity terms (ν, ζ(M, δ)) provides an interpretable bridge between deep-learning practice and optimization theory.

- The authors test their framework on two modalities:
  - Video denoising
  - Text classification with perturbed embeddings
  - Across both domains, the proposed method shows better performance on robustness and generalization.

### Weaknesses
**1. Sacrificing in-domain performance**

Figure 2 indicates that the proposed constrained‑optimization transformer underperforms compared to the vanilla ERM baseline on in‑domain (ID) evaluation, while providing advantages mainly in out‑of‑domain (OOD) settings.

This gap suggests that the imposed per‑layer descent constraints may introduce an inductive bias that prioritizes generalization robustness at the expense of ID accuracy. While this trade‑off can be acceptable in robustness‑critical regimes, it raises questions about the method’s scalability in web‑scale scenarios — where data distributions are broad and effectively in‑domain, and OOD events are relatively rare.

Standard regularization / early stopping methods also sacrifice ID performance for OOD generalization. These simple baselines are not compared in this paper.

**2. No causal or autoregressive extension**

The paper develops its constrained optimization formulation exclusively for bidirectional or sequence‑to‑sequence transformer settings, where the entire input sequence is jointly processed at each layer. However, the formulation does not address causal masking or autoregressive factorization of likelihoods, which are crucial for generative models such as GPT‑style architectures.

**3. Scaling**

The paper does not investigate how the proposed constrained‑optimization approach behaves when model size, dataset size, or sequence length scale up. Key scaling questions remain open, such as:

- How the primal–dual updates scale with model parameter / dataset size?
- Whether the constraint coefficients or dual regularizers need retuning for larger models?
- How total compute and memory overhead of per‑layer constraint evaluation grow in long‑sequence or high‑parameter settings?

The current experiments use relatively small encoders (e.g., DistilBERT) and medium‑sized video transformers, leaving unclear whether the method remains efficient or stable for foundation‑scale models.

**4. Writeup tiny issue**

There are many metrics in this paper and some are higher-is-better while others are the opposite. It would be better if it's indicated by $\uparrow$ $\downarrow$ in each figure.

### Questions
- Does the proposed method

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a constrained optimization framework for training transformers, such that each layer optimizes an optimization objective and decreases the associated loss function in expectation. From this perspective, the so-trained transformers are also called constrained “unrolled” transformers, as each layer can be interpreted as an unrolling step of the underlying optimization problem. To train such models, this paper introduces a dual training algorithm and provides theoretical insights on the effectiveness of the proposed algorithm, particularly on robustness under distribution shifts. Empirically, the authors evaluated and demonstrated the effect of adding such constraints on tasks including video denoising and text classification.

### Strengths
1. This work focuses on decreasing the expected loss value along the layers of the transformer models. This is novel compared to prior work, where most implement each layer as a gradient descent step of the optimization objective. This makes the proposed method work across different transformer architectures.
2. A rather principled and theoretically motivated approach that provides certain performance guarantees. As a bonus, the algorithm itself is also simple enough and is generally applicable.
3. The paper is mostly well-written and clearly presented.

### Weaknesses
1. While the authors rightfully state that “the behavior of these networks [from previous works] is non-monotonic along the iterates”, the proposed constrained optimization algorithm only applies to the expectation level rather than sample level. Hence, there is no guarantee that the network from the proposed algorithm will behave monotonically in a real-world setting of finite, streaming samples.
2. Another weakness concerns the experimental results. In the video denoising setting, only 5 out of 9 settings (also noted by the authors) show an advantage for the proposed algorithm. This raises questions about the real-world potential of the proposed algorithm. 
3. The experiments focus on distribution shifts, but the type of shifts are rather limited, with Gaussian embedding/pixel noise. To really showcase the strength, other types of shifts (e.g. temporal, non-Gaussian) should also be considered.
4. Another important missing piece is the layer-wise loss analysis. Since the algorithm implements a layer-wise optimization procedure, it is important to experimentally corroborate whether the real practice matches with the theories.

### Questions
1. Could you clarify what the U matrices in equation 2 represent? This layer should just be the MLP if I understand correctly. 
2. How sensitive is the $\lambda$ parameter and how much tuning is required?

### Soundness
3

### Presentation
3

### Contribution
2
