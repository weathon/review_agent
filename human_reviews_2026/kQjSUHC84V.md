# DIVERSE: Disagreement-Inducing Vector Evolution for Rashomon Set Exploration

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
We propose DIVERSE, a framework for systematically exploring the Rashomon set of deep neural networks, the collection of models that match a reference model’s accuracy while differing in their predictive behavior. DIVERSE augments a pretrained model with Feature-wise Linear Modulation (FiLM) layers and uses Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to search a latent modulation space, generating diverse model variants without retraining or gradient access. Across MNIST, PneumoniaMNIST, and CIFAR-10, DIVERSE uncovers multiple high-performing yet functionally distinct models. Our experiments show that DIVERSE offers a competitive and efficient exploration of the Rashomon set, making it feasible to construct diverse sets that maintain robustness and performance while supporting well-balanced model multiplicity. While retraining remains the baseline to generate Rashomon sets, DIVERSE achieves comparable diversity at reduced computational cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduce a gradient-free method called  DIVERSE to explore the Rashomon set of deep learning models. This set contains models with similar accuracy but different predictive behaviors. DIVERSE takes a pre-trained model and augments it with "Feature-wise Linear Modulation" (FiLM) layers. DIVERSE then uses a search algorithm "Covariance Matrix Adaptation Evolution Strategy" (CMA-ES) to find different model variations without needing to retrain them from scratch.  The experiments show that DIVERSE can uncover multiple high-performing and functionally distinct models efficiently. It offers a competitive way to explore the Rashomon set, achieving comparable diversity to retraining but at a much lower computational cost.

### Strengths
Originality:
The paper reframes latent exploration as Rashomon set exploration in deep networks. It searches a bounded FiLM modulation space around a fixed model and balances an accuracy tolerance epsilon with explicit control over behavioral diversity. This perspective differs from weight generation or full retraining by focusing on efficient, local, and controllable exploration around a single reference model.

Quality:
The methodology is clearly specified and reproducible, with objectives, constraints, and data splits stated in enough detail. The experimental design is sound for the stated goals and uses appropriate datasets and baselines. The evaluation employs complementary metrics including Rashomon Ratio, discrepancy, ambiguity, VPR, and RC, which together provide a comprehensive view of diversity under an accuracy constraint. The analysis includes sensitivity to key hyperparameters and highlights dataset dependent effects.

Clarity:
The paper is clearly written and easy to follow. The flow from problem setup to method and experiments is logical. Notation is consistent, and the figures make the FiLM based search space and the role of the latent variable z intuitive.

Strengths:
The approach is practical and training free for a given reference model, which makes audits of the local Rashomon set feasible under realistic compute budgets. The joint use of decision level and probability level metrics supports a nuanced interpretation of disagreement. By mapping accuracy constrained behavioral variants, the method helps characterize the local performance diversity landscape of a trained model and can inform stress testing, ensembling, and selective prediction. The compute footprint is small compared to retraining, enabling broader exploration on larger models and datasets.

### Weaknesses
Scaling of the search. The method relies on full-covariance CMA-ES over FiLM latents, which does not scale well as dimensionality increases. This limits exploration on deeper or more complex models and constrains the approach’s practical reach.

Architecture locality. Because FiLM layers are inserted into a specific pretrained network, the results are tied to that architecture. It is unclear whether conclusions transfer across backbones with comparable accuracy, and cross-architecture comparability is not established.

Objective agnostic to why models disagree. The fitness targets disagreement under an accuracy tolerance but remains insensitive to the underlying cause. As a result, discovered variants may be superficial perturbations rather than models with meaningfully different reliance on features, robustness characteristics, or fairness profiles.

Experimental scope and baseline breadth. Experiments are confined to image classification on moderate-scale datasets. The behavior of the Rashomon set may differ substantially on larger benchmarks (e.g., ImageNet) or in other modalities (e.g., NLP with attention). Moreover, “retraining” is treated as a single comparator despite encompassing diverse regimes, leaving the trade-off landscape underexplored.

### Questions
Which layers contribute most to disagreement. Please provide a layerwise or stagewise sensitivity analysis of FiLM norms versus diversity.

How portable is the method across backbones with similar top line accuracy. A compute matched comparison on two architectures would clarify whether results are model local or architecture agnostic.

### Soundness
2

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
The authors introduce DIVERSE, a novel method to find other models in the ***Rashomon Set*** of a reference model. A *Rashomon Set* is defined as the set of models that have similar predictive accuracy on a given task.

The DIVERSE algorithm is based on two main components: FiLM layers which apply affine transformations to pre-activations, and CMA-ES for gradient-free optimisation of the parameters of the FiLM layers. Since CMA-ES is not scalable to high dimensions, DIVERSE uses it to optimise a low-dimensional vector \$z\$, which is then projected to higher dimensions using random fixed matrices.

The authors evaluate their method using both prediction-based and probability-based metrics, in all cases after the last layer. They compare against two baselines: re-training using different seeds, and dropout-based Rashomon exploration. In terms of diversity, the generated models are generally less diverse than with full re-training, but for a runtime orders of magnitude lower.

### Strengths
This paper proposes a relatively simple, inexpensive and elegant solution to the Rashomon Set exploration problem.

S1: The proposed method is relatively simple, yet effective. This makes it very relevant to solve the Rashomon Set exploration problem.

S2: The evaluation of the algorithm is good, with metrics covering both class predictions and output probabilities. Furthermore, multiple hyperparameters are explored and DIVERSE is compared to existing baselines.

S3: The paper is generally clear and well-written. The introduction effectively contextualises the Rashomon Set problem, experimental setup and results are generally clear.

### Weaknesses
I think that the paper is overall quite solid. However, I believe its main weak points are related to its impact and motivation.

W1: Based on the Introduction and the Conclusion of the paper, I do not understand why Rashomon Set exploration is an important problem to solve. I would like the authors to better motivate why their work is impactful for research in Machine Learning.

W2: Furthermore, I think the results insufficiently show that DIVERSE is better than dropout-based Rashomon exploration. In particular, dropout-based Rashomon exploration is less computationally expensive than DIVERSE, and shows better performance in most cases for PneumoniaMNIST. I believe the authors should better highlight where their method outperforms the existing baselines, and should clarify under what conditions DIVERSE is preferable, such as specific datasets, architectures, or computational constraints.

W3: For the experiments comparing DIVERSE with its baselines, it is unclear to me what hyperparameters are being used. I think this should be described in the experiment setup.

W4: This is a minor point, but I believe it would be preferable that the metric mathematical definitions, currently in Appendix A.1, be integrated to the main text. This would improve the clarity and readability of the results Section.

### Questions
Q1: The definition in the Introduction defines the Rashomon Set as the set of models that achieve a similar performance on a same task, whereas in Equation (1) this set is constrained to a hypothesis space of models parametrised by weights \$w \in \mathbb{R}^p\$. Could the authors please clarify whether the Rashomon Set is constrained to models that use the same architecture?

Q2: In Table 1, it is unclear which size \$d\$ is used for the vector \$z\$. Since CMA-ES struggles with higher dimensionalities, how do higher values of \$d\$ impact the runtime of DIVERSE? Is there a trade-off between diversity, Rashomon ratio and runtime as \$d\$ increases?

Q3: The paper only explores using DIVERSE on CNNs. Would DIVERSE be applicable to Transformers? If yes, demonstrating this applicability in the paper could further strengthen it. If not, what are the potential challenges?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper adresses the problem of exploring the Rashomon set of a trained machine learning model in a cost-efficient and diverse manner. The $\epsilon$-Rashomon set is the set of all machine learning models that reach the same empirical risk as the reference model with a tolerance $\epsilon$. In this set lies a multiplicity of different models that may produce different predictions for identical individuals. Being able to explore $\epsilon$-Rashomon sets model is interesting to construct diverse ensemble of machine learning models for uncertainty quantification or improved predictive performance.

As exploring the $\epsilon$-Rashomon set of a deep neural network can be compute intensive with naive methods (retraining from scratch or exploring the whole parameter space), this paper proposes to explore a subspace of the $\epsilon$-Rashomon set using Feature-wise Linear Modulation (FILM), a low-dimensional parameterized transformation of a neural network. In this low dimensional space, a black-box and derivative free optimization algorithm is used (CMA-ES) for exploration. The constraint on exploring models included in the $\epsilon$-Rashomon set is relaxed by adding a penalization in the objective function minimized by CMA-ES.

The proposed approach (DIVERSE) is then evaluated on three small to medium scale image classification datasets. Empirical results show that DIVERSE can be a good compromise between exploration and compute. Extensive experiments and ablation studies are conducted to evaluate the impact of each introduced hyperparameters.

### Strengths
1. The paper is well-written with a clear objective, all notions are introduced clearly.
2. The proposed approach has the potential to be a fundamental tool in many domains of machine learning.
3. The proposed approach is sound and well grounded in the literature.
4. The ablation study is quite furnished and the experimental protocol sound

### Weaknesses
1. The paper lacks qualitative or illustrative experiments to helps the reader gain intuitions on the significance of the results.
2. The paper lacks experiments with downstream tasks (uncertainty quantification, ensembling, ...) to better asses if generated models with DIVERSE are actually effective for practical tasks.
3. The paper lacks quantification about how smaller the explored set of models induced by FiLM is to the true $\epsilon$-Rashomon set.

### Questions
Has Rashomon set exploration been done for different training tasks such as regression?

### Remarks
The columns of Figure 2 and 3 are not in the same order.
The impact of $\lambda$, the mixing coefficient between soft and hard agreement is not evaluated in the ablation study.
More diverse datasets could be used in the experimental protocol. MNIST (and I think PneumoniaMNIST also) is quite an easy dataset where a linear model trained on the raw pixel can achieve very high accuracy. Even though not ideal, Fashion-MNIST and K-MNIST might be better alternatives. MNIST-1D, even though not a image classification dataset but a time-series classification one, could be a potential candidate, as linear model have poor performances on it.
- Fashion-MNIST : Xiao, Han, Kashif Rasul, and Roland Vollgraf. "Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms." 2017.
- K-MNIST : Clanuwat, Tarin, et al. "Deep learning for classical japanese literature." 2018.
- MNIST-1D : Greydanus, Sam, and Dmitry Kobak. "Scaling down deep learning with mnist-1d." 2020.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce DIVERSE, a framework for exploring the Rashomon set of deep neural networks, innovative way to find high quality and diverse models that match a reference model's accuracy but differ in their predictions. DIVERSE adds Feature-wise Linear Modulation (FiLM) layers to a pretrained model and uses Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to search a latent modulation space, producing diverse model variants without retraining or gradients. On MNIST, PneumoniaMNIST, and CIFAR-10, DIVERSE finds multiple high-performing models that behave differently

### Strengths
- Innovative approximation of Quality-Diversity evolution
- Well written paper

### Weaknesses
- Limited Ablation study on evolution side.

### Questions
Why CMA-ES and not any other evolutionary algorithm?

### Soundness
3

### Presentation
3

### Contribution
4
